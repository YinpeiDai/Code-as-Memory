"""InterceptSlow-VLA-v0 policy: intercept a ball rolling along +y and push it into the target.

Perception (front RGB-D, wrist fallback): target = mean 3D point of the target's red ring pixels
(colour ~ (204,71,76)); ball = saturated red pixels (g,b < r/4) back-projected with depth, centre
corrected by 2/3 radius along the viewing ray.  Control: pursuit-push in the ball frame (see policy.md).
"""

BASE_XY = (-0.615, 0.0)       # Panda base in world (offline FK of demo joints)
R_BALL = 0.022                # ball radius (pixel extent * depth / f in demos)
RC = 0.024                    # measured TCP-to-ball-centre distance in contact, hand x face (R003-R005)
RC_SIDE = 0.031               # same for the hand y (finger side) face (R006)
STANDOFF = 0.05               # behind-ball distance before descending (v9: 0.065 -> 0.05 to engage reach-limited balls earlier)
Z_PUSH = 0.028
Z_FLY = 0.085
REACH = 0.77                  # max horizontal TCP distance from base used for planning (offline IK: vertical reach ~0.79 at z=0.03)
PATH_DX = 0.2                 # demos: target x - ball path x = 0.199 +- 0.005 (50 episodes)
STEP_GAIN = 0.04              # metres of TCP motion per step per unit command (demo regression)
S_DES = 0.030                 # nominal ball push speed, m/step (R004-R023: ball follows at ~0.031)
S_MAX = 0.040                 # push speed cap when time is short (TCP max ~0.045 m/step)
ENTER = 0.1                   # success radius hypothesis (demo end distances 0.105 -> <0.106)
COAST_MISS = 0.03             # predicted miss distance accepted for coasting
CLOSE_V = 0.012               # max closing speed onto the ball beyond feed-forward, m/step
YAW_TOL = 0.15                # max yaw error (rad) before descending behind the ball
FACE_STEP = math.pi / 2       # hand +-x and +-y faces all usable (R006: side face sent ball along its normal)
LAT_OK = 0.005                # lateral ball offset on the face tolerated at full push speed
CHUNK = 3                     # control steps per decision (60 native steps, 40 decisions)


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-3:  # near pi
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        axis = V[:, int(np.argmax(w))]
        if np.dot(axis, v) < 0:
            axis = -axis
        return axis * ang
    return v / (2.0 * s) * ang


def _backproject(K, E, us, vs, d):
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    return (pc - t) @ R


def _cam_center(E):
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return -R.T @ t


def _ball_in_cam(frame, cam):
    rgb = np.asarray(getattr(frame, cam + "_rgb")).astype(np.int32)
    dep = np.asarray(getattr(frame, cam + "_depth")).astype(np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    m = (r > 50) & (g * 4 < r) & (b * 4 < r) & (dep > 0)
    n = int(m.sum())
    if n < 3:
        return None, n
    vs, us = np.nonzero(m)
    P = _backproject(K, E, us.astype(np.float64), vs.astype(np.float64), dep[vs, us])
    P = P[P[:, 2] < 0.08]          # ball lies on the table
    if len(P) < 3:
        return None, 0
    c = np.median(P, 0)
    P = P[np.linalg.norm(P - c, axis=1) < 0.05]   # drop stray pixels far from the blob
    if len(P) < 3:
        return None, 0
    mean = P.mean(0)
    ray = mean - _cam_center(E)
    ray = ray / (np.linalg.norm(ray) + 1e-9)
    centre = mean + (2.0 / 3.0) * R_BALL * ray
    return centre, int(len(P))


def _ball(frame):
    """Ball centre [x, y, z] in world and its source, or None."""
    c, n = _ball_in_cam(frame, "front")
    if c is not None and n >= 12:
        return [float(c[0]), float(c[1]), float(c[2])], "front", n
    c2, n2 = _ball_in_cam(frame, "wrist")
    if c2 is not None and n2 >= 25:
        return [float(c2[0]), float(c2[1]), float(c2[2])], "wrist", n2
    if c is not None and n >= 5:
        return [float(c[0]), float(c[1]), float(c[2])], "front_weak", n
    return None, "none", max(n, n2)


def _target(frame):
    rgb = np.asarray(frame.front_rgb).astype(np.int32)
    dep = np.asarray(frame.front_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    m = (np.abs(rgb[..., 0] - 204) < 14) & (np.abs(rgb[..., 1] - 71) < 14) & (np.abs(rgb[..., 2] - 76) < 14) & (dep > 0)
    n = int(m.sum())
    if n < 100:
        return None, n
    vs, us = np.nonzero(m)
    P = _backproject(K, E, us.astype(np.float64), vs.astype(np.float64), dep[vs, us])
    P = P[np.abs(P[:, 2]) < 0.03]
    if len(P) < 100:
        return None, n
    c = np.median(P, 0)
    P = P[np.linalg.norm(P[:, :2] - c[:2], axis=1) < 0.13]
    return [float(P[:, 0].mean()), float(P[:, 1].mean())], n


def _reach_r(x, y):
    return math.hypot(x - BASE_XY[0], y - BASE_XY[1])


def _wait_y(x):
    dx = x - BASE_XY[0]
    return -math.sqrt(max(REACH * REACH - dx * dx, 0.04))


def _ball_estimate(track, now_step):
    """Linear fit of the last observations -> (pos_now [x,y], vel [vx,vy] per step, age)."""
    pts = [p for p in track if now_step - p[2] <= 8]
    if not pts:
        return None, None, None
    pts = pts[-4:]
    last = pts[-1]
    if len(pts) >= 2 and pts[-1][2] > pts[0][2]:
        s = np.array([p[2] for p in pts], dtype=np.float64)
        X = np.array([[p[0], p[1]] for p in pts], dtype=np.float64)
        sm = s.mean()
        den = float(((s - sm) ** 2).sum())
        vel = ((s - sm)[:, None] * (X - X.mean(0))).sum(0) / den
        vel = np.clip(vel, -0.08, 0.08)
    else:
        vel = np.array([0.0, 0.012])
    age = now_step - last[2]
    pos = np.array([last[0], last[1]]) + vel * age
    return pos, vel, age


def _scale(v, lim=1.0):
    m = float(np.max(np.abs(v))) if len(v) else 0.0
    return v * (lim / m) if m > lim else v


def _coasting(bpos, bvel, g, steps_left):
    """True when the ball's current straight-line motion already enters the target zone in time."""
    sp = float(np.linalg.norm(bvel))
    if sp < 0.006:
        return False, None
    rel = g - bpos
    t_c = float(np.dot(rel, bvel)) / (sp * sp)
    if t_c <= 0:
        return False, None
    miss = float(np.linalg.norm(bpos + bvel * t_c - g))
    # time until within ENTER radius
    if miss >= ENTER:
        return False, round(miss, 3)
    t_in = t_c - math.sqrt(ENTER * ENTER - miss * miss) / sp
    return (miss < COAST_MISS and t_in < steps_left - 2), round(miss, 3)


def _yaw_near(angle, cur):
    """Yaw that points a usable finger face normal along `angle`, nearest to cur.

    FACE_STEP = pi: only the hand +-x faces (normal = yaw direction); pi/2: also the hand +-y faces.
    """
    best = angle
    for k in range(-4, 5):
        cand = angle + k * FACE_STEP
        if abs(cand - cur) < abs(best - cur):
            best = cand
    return best


def _face_normal(cur_yaw, d):
    """Unit face normal (one of the usable hand axes at cur_yaw) best aligned with direction d."""
    best, bd, bk = None, -2.0, 0
    k = 0
    while k * FACE_STEP < 2 * math.pi - 1e-6:
        m = np.array([math.cos(cur_yaw + k * FACE_STEP), math.sin(cur_yaw + k * FACE_STEP)])
        if float(np.dot(m, d)) > bd:
            best, bd, bk = m, float(np.dot(m, d)), k
        k += 1
    return best


def _face_rc(cur_yaw, m):
    """Contact distance for the face whose normal is m: hand +-x face 0.024 m, hand +-y face 0.031 m (R001-R006)."""
    hx = np.array([math.cos(cur_yaw), math.sin(cur_yaw)])
    return RC if abs(float(np.dot(hx, m))) > 0.7 else RC_SIDE


def _control(tcp, prev_tcp, target, track, now_step, state):
    """One control command from measured TCP, target and ball track.

    state: dict with 'stage'.  Returns (action7, info, state).
    Stages: approach (fly at Z_FLY to the behind point or reach-limited waiting point),
    descend (lower behind the ball), push (push along the measured finger-face normal m = yaw
    direction; R001-R003 showed the ball leaves along m), coast (ball line already enters the target).
    The yaw is driven toward the ball->target direction from step 0 (rotation is rate limited ~0.045 rad/step).
    """
    tcp = np.asarray(tcp, dtype=np.float64)
    p = tcp[:3]
    cur_yaw = float(tcp[5])
    vel_tcp = p - np.asarray(prev_tcp[:3], dtype=np.float64) if prev_tcp is not None else np.zeros(3)
    g = np.array(target, dtype=np.float64)
    bpos, bvel, age = _ball_estimate(track, now_step)
    stage = state.get("stage") or "approach"
    xp = g[0] - PATH_DX
    info = {}
    ff = np.zeros(2)
    V = None
    if bpos is None:
        # ball not seen yet: hover over its predicted path at the reach-limited waiting point
        P = np.array([xp, _wait_y(xp)])
        z_goal = Z_FLY
        d = g - P
        d = d / (np.linalg.norm(d) + 1e-9)
        info["mode"] = "wait_unseen"
    else:
        b1 = bpos + bvel                      # one-step-ahead ball prediction
        if stage == "approach" and _reach_r(b1[0] - STANDOFF * 0.4, b1[1] - STANDOFF * 0.9) > REACH:
            ref = np.array([b1[0], _wait_y(b1[0])])   # expected contact region
        else:
            ref = b1
        d = g - ref
        dist_g = float(np.linalg.norm(g - b1))
        d = d / (float(np.linalg.norm(d)) + 1e-9)
        yaw_goal = _yaw_near(math.atan2(d[1], d[0]), cur_yaw)
        yaw_err = yaw_goal - cur_yaw
        m = _face_normal(cur_yaw, d)
        t = np.array([-m[1], m[0]])
        rel = bpos - p[:2]                    # ball relative to TCP
        al_m = float(np.dot(rel, m))
        lt_m = float(np.dot(rel, t))
        dist = float(np.linalg.norm(rel))
        dg = g - b1
        dg = dg / (float(np.linalg.norm(dg)) + 1e-9)
        B = b1 - STANDOFF * m          # behind the ball along the face normal actually in use
        coast, miss = _coasting(bpos, bvel, g, 60 - now_step)
        info.update(al=round(al_m, 3), lt=round(lt_m, 3), dist=round(dist, 3), miss=miss,
                    yaw_err=round(yaw_err, 2), bv=[round(float(bvel[0]), 4), round(float(bvel[1]), 4)])
        if stage == "coast":
            stage = "push"     # v6: never coast (R007: released ball curves back toward +y)
        if stage == "approach":
            if _reach_r(B[0], B[1]) <= REACH:
                P = B
                z_goal = Z_FLY
                ff = bvel
                along = float(np.dot(p[:2] - bpos, dg))
                lat = float(dg[0] * (p[1] - bpos[1]) - dg[1] * (p[0] - bpos[0]))
                behind = along < -0.04 and abs(lat) < 0.025
                yaw_ok = abs(yaw_err) < YAW_TOL or now_step >= 40
                if behind and float(np.linalg.norm(p[:2] - B)) < 0.03 and yaw_ok:
                    stage = "descend"
                info["mode"] = "to_behind"
            else:
                P = np.array([bpos[0], _wait_y(bpos[0])])
                z_goal = Z_FLY
                info["mode"] = "wait_seen"
        if stage == "descend":
            P = B
            ff = bvel
            z_goal = Z_PUSH
            along = float(np.dot(p[:2] - bpos, dg))
            if along > -0.02 and dist < 0.05:
                stage = "approach"        # ball slipped beside/under: go back up and realign
                z_goal = Z_FLY
            elif p[2] < Z_PUSH + 0.015:
                stage = "push"
            info["mode"] = "descend"
        if stage == "push":
            z_goal = Z_PUSH
            s_des = min(S_MAX, max(S_DES, (dist_g - 0.08) / max(60 - now_step - 10, 3)))
            if al_m < 0.012 or abs(lt_m) > 0.03:
                # ball beside / behind the face: go around to the behind point at push height
                side = t * (-1.0 if lt_m > 0 else 1.0)
                P = bpos - STANDOFF * m + (side * 0.04 if al_m < 0.012 else 0.0)
                ff = bvel
                info["mode"] = "regrip"
            else:
                gap = al_m - _face_rc(cur_yaw, m)
                vb_t = float(np.dot(bvel, t))
                # R008: |lt| > ~0.012 on the narrow side face moves contact to the finger edge -> slow the
                # push while re-centring the ball on the face
                s_eff = s_des * max(0.3, min(1.0, 1.0 - (abs(lt_m) - LAT_OK) / 0.01))
                V = (s_eff + max(-0.006, min(CLOSE_V, 0.5 * gap))) * m + (0.8 * lt_m + vb_t) * t
                info["s_eff"] = round(s_eff, 3)
                P = p[:2] + V
                info["mode"] = "push"
                info["s"] = round(s_des, 3)
        if stage == "coast":
            P = p[:2] - 0.5 * vel_tcp[:2]
            z_goal = Z_PUSH
            info["mode"] = "coast"
    rr = _reach_r(P[0], P[1])
    if rr > REACH:
        P = np.array([BASE_XY[0], BASE_XY[1]]) + (np.asarray(P) - np.array([BASE_XY[0], BASE_XY[1]])) * (REACH / rr)
    err = np.array([P[0] - p[0], P[1] - p[1], z_goal - p[2]])
    if V is not None:
        axy = V / STEP_GAIN          # velocity command (steady state ~STEP_GAIN m/step per unit)
    else:
        axy = (0.8 * err[:2] + np.asarray(ff) - 0.3 * vel_tcp[:2]) / STEP_GAIN
    if p[2] < 0.06 and err[2] > 0.01:
        axy = axy * 0.3     # rising: do not sweep low through the ball
    az = (0.7 * err[2] - 0.3 * vel_tcp[2]) / STEP_GAIN
    axy = _scale(axy)
    az = max(-1.0, min(1.0, az))
    # orientation: gripper down, hand x axis (closed-finger face normal) along ball->target direction
    yaw_d = _yaw_near(math.atan2(d[1], d[0]), cur_yaw)
    R_d = _rot([math.pi, 0.0, yaw_d])
    R_m = _rot(tcp[3:6])
    w = _rotvec(R_d @ R_m.T)
    r = -0.6 * w / 0.1
    nr = float(np.linalg.norm(r))
    if nr > 1.0:
        r = r / nr
    state = dict(state)
    state["stage"] = stage
    info["P"] = [round(float(P[0]), 3), round(float(P[1]), 3), round(float(z_goal), 3)]
    action = [float(axy[0]), float(axy[1]), float(az), float(r[0]), float(r[1]), float(r[2]), -1.0]
    return action, info, state


def _update_track(track, frame):
    c, src, n = _ball(frame)
    if c is not None:
        track = list(track) + [[round(c[0], 4), round(c[1], 4), int(frame.step)]]
    return track[-12:], src, n


def perceive_decide(obs, goal, memory, last):
    """Track target and ball from every new frame, keep stage, and run the next control chunk."""
    mem = dict(memory or {})
    target = mem.get("target_xy")
    track = list(mem.get("ball_track") or [])
    seen_steps = set(p[2] for p in track)
    for fr in obs.frames:
        if target is None:
            tg, n = _target(fr)
            if tg is not None:
                target = tg
        if int(fr.step) not in seen_steps:
            track, src, n = _update_track(track, fr)
            seen_steps.add(int(fr.step))
    stage = mem.get("stage") or "approach"
    if last and isinstance(last, str) and '"stage": "' in last:
        stage = last.split('"stage": "')[1].split('"')[0]
    n_dec = int(mem.get("decisions") or 0) + 1
    cur = obs.current
    notes = mem.get("notes") or ""
    if target is None:
        # no target visible: observe one step (should not happen; target is visible at reset)
        new = {"target_xy": None, "ball_track": track, "stage": stage, "decisions": n_dec,
               "first_ball_step": mem.get("first_ball_step"), "notes": (notes + " no_target@%d" % cur.step)[-400:]}
        return {"memory": new, "act": {"tool": "act_control", "args": {"target": [0.0, 0.0], "track": track,
                                                                       "stage": "hold", "n_steps": 1}}}
    first = mem.get("first_ball_step")
    if first is None and track:
        first = int(track[0][2])
    # remaining decisions: budget the chunk so 60 steps fit in 40 decisions
    n_steps = CHUNK
    prev_tcp = [round(float(v), 5) for v in obs.frames[-2].tcp] if len(obs.frames) >= 2 else None
    new = {"target_xy": [round(target[0], 4), round(target[1], 4)], "ball_track": track, "stage": stage,
           "decisions": n_dec, "first_ball_step": first, "notes": notes[-400:]}
    return {"memory": new, "act": {"tool": "act_control", "args": {"target": new["target_xy"], "track": track,
                                                                   "stage": stage, "n_steps": n_steps,
                                                                   "prev_tcp": prev_tcp}}}


def act_control(robot, target, track, stage, n_steps=3, prev_tcp=None):
    """Run n_steps closed-loop intercept/push steps; returns measured ball, TCP, stage and log."""
    track = [list(p) for p in (track or [])]
    state = {"stage": stage}
    log = []
    prev = prev_tcp
    for k in range(int(n_steps)):
        fr = robot.now()
        if not track or track[-1][2] != int(fr.step):
            track, src, n = _update_track(track, fr)
        else:
            src, n = "cached", 0
        if stage == "hold":
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
            continue
        action, info, state = _control(fr.tcp, prev, target, track, int(fr.step), state)
        prev = fr.tcp
        robot.step(*action)
        log.append({"s": int(fr.step), "tcp": [round(float(v), 3) for v in fr.tcp[:3]],
                    "ball": track[-1][:2] if track and track[-1][2] == int(fr.step) else None,
                    "src": src, "a": [round(v, 2) for v in action[:3]], **info})
    fr = robot.now()
    track, src, n = _update_track(track, fr) if (not track or track[-1][2] != int(fr.step)) else (track, "cached", 0)
    return {"stage": state.get("stage"), "step": int(fr.step), "tcp": [round(float(v), 3) for v in fr.tcp[:3]],
            "ball": track[-1] if track else None, "log": log}
