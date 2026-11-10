"""InterceptMedium policy: perceive ball/target with front RGB-D, escort the rolling
ball from its -x side with the closed fingers and push it in +x so that its straight
path crosses the target disc (success hypothesis: ball centre within ~0.1 m of target
centre in xy). See /workspace/policy.md for evidence and uncertainties."""

BALL_R = 0.02          # ball radius (m), from sphere fits on demo depth
GOAL_R = 0.10          # inferred target-disc radius / success radius (m)
Z_PUSH = 0.020         # TCP height while escorting / pushing
Z_SAFE = 0.07          # TCP height for repositioning over the ball
PITCH = 0.30           # hand tilt about world y (improves reach near the base)
FACE_OFF = 0.033       # ball_x - tcp_x when the closed fingers' +x face touches the ball
FACE_OFF_Y = 0.038     # |ball_y - tcp_y| when a finger y-face touches the ball
CLEAR = 0.022          # extra x clearance while waiting / aligning
GAIN0 = 0.39           # measured per-step fraction of the commanded offset (demo fit)
VMAX = 0.036           # planning speed per axis (m/step)
VX_MAX = 0.034         # max lateral push speed (m/step)
AIM_BIAS = 0.012       # aim slightly to the -x side of centre (only +x corrections exist)
LINE_PRIOR_DX = -0.20  # ball line x - target x at release (all 50 demos: -0.200 +- 0.003)
STEP_LIMIT = 60        # native episode step limit


# ----------------------------------------------------------------- geometry
def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        # near pi: use the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        axis = d.copy()
        for j in range(3):
            if j != i:
                axis[j] = (R[i, j] + R[j, i]) / (4.0 * d[i])
        return axis / np.linalg.norm(axis) * ang
    return v / (2.0 * s) * ang


def _backproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth[v, u].astype(float) / 1000.0
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return (pc - t) @ R


# --------------------------------------------------------------- perception
def _sphere_fit(P, r=BALL_R, it=8):
    c = P.mean(0).copy()
    c[2] = max(c[2], r)
    for _ in range(it):
        dv = P - c
        n = np.linalg.norm(dv, axis=1) + 1e-9
        res = n - r
        J = -dv / n[:, None]
        dc = np.linalg.lstsq(J, -res, rcond=None)[0]
        c = c + dc
    return c


def _ball_from_cam(frame, cam, near=None):
    rgb = getattr(frame, cam + "_rgb", None)
    dep = getattr(frame, cam + "_depth", None)
    K = getattr(frame, cam + "_K", None)
    E = getattr(frame, cam + "_extrinsic", None)
    if rgb is None or dep is None or K is None or E is None:
        return None
    im = rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 90) & (g < 40) & (b < 40) & (r - g > 80)
    if int(m.sum()) < 6:
        return None
    P = _backproject(dep, K, E, m)
    if len(P) == 0:
        return None
    P = P[(P[:, 2] < 0.06) & (P[:, 2] > -0.01)]
    if len(P) < 6:
        return None
    if near is not None:
        dn = np.linalg.norm(P[:, :2] - np.asarray(near)[None, :], axis=1)
        if np.any(dn < 0.12):
            P = P[dn < 0.12]
    med = np.median(P, 0)
    P = P[np.linalg.norm(P - med, axis=1) < 0.05]
    if len(P) < 6:
        return None
    c = _sphere_fit(P)
    if not np.all(np.isfinite(c)) or abs(c[2] - BALL_R) > 0.03:
        return None
    return [float(c[0]), float(c[1]), float(c[2]), int(len(P))]


def _detect_ball(frame, near=None):
    """Ball centre [x, y, z, npts] in world metres, front camera first, wrist fallback."""
    b = _ball_from_cam(frame, "front", near)
    if b is not None and b[3] >= 8:
        return b
    w = _ball_from_cam(frame, "wrist", near)
    if w is not None and w[3] >= 15:
        return w
    return b


def _detect_target(frame):
    """Target disc centre from the red ring (front camera). Returns [x, y, ext_x, ext_y, n] or None."""
    rgb = getattr(frame, "front_rgb", None)
    dep = getattr(frame, "front_depth", None)
    if rgb is None or dep is None:
        return None
    im = rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (np.abs(r - 204) < 25) & (np.abs(g - 71) < 25) & (np.abs(b - 76) < 25)
    n = (red[1:-1, 1:-1].astype(int) + red[:-2, 1:-1] + red[2:, 1:-1]
         + red[1:-1, :-2] + red[1:-1, 2:])
    m = np.zeros_like(red)
    m[1:-1, 1:-1] = n >= 4
    P = _backproject(dep, frame.front_K, frame.front_extrinsic, m)
    if len(P) < 30:
        return None
    P = P[P[:, 2] < 0.006]
    if len(P) < 30:
        return None
    c = np.median(P, 0)
    Q = P
    for _ in range(4):
        Q = P[np.linalg.norm(P[:, :2] - c[None, :2], axis=1) < 0.13]
        if len(Q) < 30:
            return None
        c = (Q.max(0) + Q.min(0)) / 2.0
    ext = Q.max(0) - Q.min(0)
    return [float(c[0]), float(c[1]), float(ext[0]), float(ext[1]), int(len(Q))]


def _ball_estimate(track, step):
    """Constant-velocity fit to the most recent consistent detections.
    track: list of [step, x, y]. Returns dict(pos at `step`, vel per step, n) or None."""
    if not track:
        return None
    pts = np.array(track[-8:], dtype=float)
    best = None
    for n in (6, 5, 4, 3, 2):
        if len(pts) < n:
            continue
        S = pts[-n:]
        t = S[:, 0]
        if t[-1] - t[0] < 1:
            continue
        A = np.stack([t - t[-1], np.ones(n)], 1)
        cx = np.linalg.lstsq(A, S[:, 1], rcond=None)[0]
        cy = np.linalg.lstsq(A, S[:, 2], rcond=None)[0]
        res = np.sqrt((A @ cx - S[:, 1]) ** 2 + (A @ cy - S[:, 2]) ** 2)
        if n == 2 or res.max() < 0.003:
            best = (cx, cy, n, float(t[-1]))
            break
    if best is None:
        last = pts[-1]
        return {"pos": [float(last[1]), float(last[2])], "vel": [0.0, 0.0], "n": 1,
                "age": float(step - last[0])}
    cx, cy, n, t_last = best
    dt = step - t_last
    return {"pos": [float(cx[1] + cx[0] * dt), float(cy[1] + cy[0] * dt)],
            "vel": [float(cx[0]), float(cy[0])], "n": int(n), "age": float(dt)}


def _closest(b, v, T):
    """Closest future approach of a ball at b moving with v to point T; signed x side."""
    b = np.asarray(b, float)
    v = np.asarray(v, float)
    T = np.asarray(T, float)
    vv = float(v @ v)
    ts = 0.0 if vv < 1e-10 else max(0.0, float((T - b) @ v) / vv)
    p = b + v * ts
    return float(np.linalg.norm(p - T)), float(p[0] - T[0]), ts


def _vx_needed(b, v, T):
    """Lateral speed (m/step) that sends the ball through the aim point at the target row."""
    dy = T[1] - b[1]
    vy = max(v[1], 0.004)
    if dy < 0.03:
        return VX_MAX
    return (T[0] - AIM_BIAS - b[0]) * vy / dy


# ------------------------------------------------------------------ control
def _rot_cmd(rpy, rpy_des, k=0.8):
    """Base-frame rotation command (CPU IK left-multiplies Rx(-.1r0)Ry(-.1r1)Rz(-.1r2))."""
    Rc = _rpy2R(*rpy)
    Rd = _rpy2R(*rpy_des)
    w = _rotvec(Rd @ Rc.T)
    r = -w / (0.1 * GAIN0) * k
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r


def _trans_cmd(p, p_des, gains):
    a = (np.asarray(p_des, float) - np.asarray(p, float)) / (0.1 * np.asarray(gains, float))
    return np.clip(a, -1.0, 1.0)


def _limit(p, goal, vmax):
    d = np.asarray(goal, float) - np.asarray(p, float)
    vm = np.asarray(vmax, float)
    return np.asarray(p, float) + np.clip(d, -vm, vm)


def _plan_meet(est, tcp, T, step):
    """Choose a waiting y on the ball line the hand can reach before the ball."""
    b, v = est["pos"], est["vel"]
    vy = max(v[1], 0.008)
    pref = T[1] - 0.30
    ys = list(np.arange(pref, T[1] - 0.02, 0.01))
    for y in ys:
        if y < b[1] + 0.04:
            continue
        tb = (y - b[1]) / vy
        if step + tb > STEP_LIMIT - 8:
            break
        xs = b[0] + v[0] * tb - FACE_OFF - CLEAR
        th = max(abs(xs - tcp[0]) / VMAX, abs(y - tcp[1]) / VMAX,
                 abs(Z_PUSH - tcp[2]) / VMAX) + 2.0
        if th <= tb:
            return float(y), float(xs), float(tb)
    return None


def _safe_step(robot, a, r, grip, info):
    cmd = [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])]
    if info.get("dry_ok", True):
        try:
            chk = robot.dry_run(*cmd, mode="delta")
            if isinstance(chk, dict) and chk.get("within_joint_limits") is False:
                info["limit_hits"] = info.get("limit_hits", 0) + 1
                # retry with a smaller translation; keep the rotation command
                cmd2 = [0.5 * cmd[0], 0.5 * cmd[1], 0.5 * cmd[2], cmd[3], cmd[4], cmd[5]]
                chk2 = robot.dry_run(*cmd2, mode="delta")
                if isinstance(chk2, dict) and chk2.get("within_joint_limits") is not False:
                    cmd = cmd2
        except Exception as exc:  # dry-run unavailable: run without it
            info["dry_ok"] = False
            info["dry_err"] = str(exc)[:120]
    robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)


def _intercept_loop(robot, T, max_steps, log):
    """Closed-loop escort-and-push controller. Runs until the episode ends or max_steps."""
    T = np.asarray(T, float)
    track = []
    gains = [GAIN0, GAIN0, GAIN0]
    stage = "approach"
    base_vx = None
    u = 0.0
    contact_steps = 0
    release_steps = 0
    pushes = 0
    pushing = False
    plan = None
    prev_tcp = None
    prev_cmd = None
    last_known = None
    info = log
    ori = (math.pi, PITCH, 0.0)
    for _ in range(max_steps):
        f = robot.now()
        step = int(f.step)
        tcp = np.asarray(f.tcp, float)
        # online gain adaptation per translation axis
        if prev_tcp is not None and prev_cmd is not None:
            for k in range(3):
                c = prev_cmd[k]
                if abs(c) > 0.006:
                    gm = (tcp[k] - prev_tcp[k]) / c
                    if 0.2 < gm < 0.7:
                        gains[k] = 0.7 * gains[k] + 0.3 * gm
        near = track[-1][1:3] if track else None
        bd = _detect_ball(f, near)
        if bd is not None:
            if track and track[-1][0] == step:
                track[-1] = [step, bd[0], bd[1]]
            else:
                track.append([step, bd[0], bd[1]])
        est = _ball_estimate(track, step)
        good = est is not None and est["n"] >= 3 and est["age"] <= 4
        if good:
            last_known = [np.array(est["pos"]), np.array(est["vel"]), step]
        elif last_known is not None and est is not None and est["n"] >= 2 and est["age"] <= 2:
            # transient (e.g. during a push): trust the short fit
            last_known = [np.array(est["pos"]), np.array(est["vel"]), step]
        known = last_known is not None and step - last_known[2] <= 8
        p = tcp[:3]
        goal = None
        if known:
            b = last_known[0] + last_known[1] * (step - last_known[2])
            v = last_known[1]
            dist_T = float(np.linalg.norm(b - T))
            dmin, side, tstar = _closest(b, v, T)
        # ------------------------------------------------------ stage logic
        if not known:
            xs = T[0] + LINE_PRIOR_DX - FACE_OFF - CLEAR
            ys = min(max(T[1] - 0.30, -0.50), 0.05)
            goal = _limit(p, [xs, ys, Z_PUSH], [VMAX, VMAX, VMAX])
        else:
            speed = float(np.linalg.norm(v))
            slow = speed < 0.004
            ahead_ok = b[1] < T[1] + 0.12
            if stage == "approach":
                plan = _plan_meet({"pos": [float(b[0]), float(b[1])],
                                   "vel": [float(v[0]), float(v[1])]}, tcp, T, step)
                if plan is None:
                    y_wait = b[1] + v[1]
                    xs = b[0] + v[0] - FACE_OFF - CLEAR
                else:
                    y_wait, xs, _tb = plan
                yref = max(y_wait, b[1] + v[1])
                goal = _limit(p, [xs, yref, Z_PUSH], [VMAX, 0.039, VMAX])
                if (b[1] + v[1] >= p[1] - 0.03 and abs(p[2] - Z_PUSH) < 0.012) or slow:
                    stage = "escort"
                    base_vx = float(v[0])
                    u = float(v[0])
                    contact_steps = 0
                    pushing = False
                    info["escort_start"] = step
                if not ahead_ok:
                    stage = "done"
            if stage == "escort":
                need = _vx_needed(b, v, T)
                if need < v[0] + 0.002 and contact_steps == 0:
                    # nothing to add with the +x face (ball already heading at/past the aim)
                    stage = "shadow"
                    info["no_push_needed"] = step
            if stage == "escort" and p[0] > b[0] - FACE_OFF + 0.015:
                # hand is level with or beyond the ball in x (should not happen): back off
                # to the stand-off, lifting if it is close to the ball
                xs = b[0] + v[0] - FACE_OFF - CLEAR
                zt = Z_SAFE if abs(p[0] - b[0]) < 0.05 and abs(p[1] - b[1]) < 0.05 else Z_PUSH
                goal = _limit(p, [xs, b[1] + v[1], zt], [VMAX, 0.039, VMAX])
                info["backoff"] = info.get("backoff", 0) + 1
            elif stage == "escort":
                on_course = dmin < 0.035 or (side < 0 and dmin < 0.06 and need <= v[0] + 0.002)
                if on_course and contact_steps == 0 and abs(side) < 0.06 and dmin < GOAL_R - 0.03:
                    stage = "shadow"
                    info["skip_push"] = step
                else:
                    gap = b[0] - FACE_OFF - p[0]
                    if not pushing:
                        pushing = (base_vx is not None and v[0] > base_vx + 0.0025
                                   and gap < 0.012)
                    in_contact = pushing
                    vx_des = min(max(need, 0.0), VX_MAX)
                    if not in_contact:
                        x_stand = b[0] + v[0] - FACE_OFF - CLEAR
                        if p[0] < x_stand - 0.006:
                            # far behind the line: close in quickly to the stand-off
                            xg = p[0] + min(x_stand - p[0], VMAX)
                        else:
                            u = float(v[0]) + 0.006
                            xg = p[0] + u
                        # never place the face far beyond the modelled contact
                        xg = min(xg, b[0] + v[0] - FACE_OFF + 0.012)
                        if abs(p[1] - b[1]) > 0.015:
                            # not level with the ball yet: stay at the stand-off
                            xg = min(xg, x_stand)
                    else:
                        contact_steps += 1
                        u = min(vx_des, u + 0.008) if u < vx_des else vx_des
                        xg = p[0] + max(u, 0.0)
                        if abs(p[1] - b[1]) > 0.02:
                            xg = min(xg, b[0] + v[0] - FACE_OFF)
                    yg = b[1] + v[1]
                    goal = [xg, yg, Z_PUSH]
                    if in_contact and v[0] >= vx_des - 0.0015 and contact_steps >= 2:
                        stage = "shadow"
                        pushes += 1
                        release_steps = 0
                        info["release_%d" % pushes] = [step, round(float(v[0]), 4), round(float(v[1]), 4)]
                if not ahead_ok:
                    stage = "done"
            if stage == "shadow":
                release_steps += 1
                # stay behind (-x) and level with the ball so that a second push is quick
                xg = min(p[0] + max(v[0], 0.0) - 0.006, b[0] + v[0] - FACE_OFF - CLEAR)
                yg = b[1] + v[1]
                goal = _limit(p, [xg, yg, Z_PUSH], [VMAX, 0.039, VMAX])
                need = _vx_needed(b, v, T)
                miss_short = side < -0.035 and dmin > 0.04
                if release_steps >= 3 and ahead_ok and miss_short and not slow and pushes < 4:
                    stage = "escort"
                    base_vx = float(v[0])
                    u = float(v[0])
                    contact_steps = 0
                    pushing = False
                    info["repush_%d" % pushes] = step
                if not ahead_ok:
                    stage = "done"
            if stage in ("escort", "shadow", "done") and v[1] < 0.006 and dmin > GOAL_R - 0.02 \
                    and dist_T > GOAL_R and (release_steps >= 2 or slow):
                stage = "nudge"
                info["nudge_start"] = step
            if stage == "nudge" and v[1] >= 0.008 and b[1] < T[1] - 0.05:
                # the ball rolls toward the target row again: escort it as usual
                stage = "escort"
                base_vx = float(v[0])
                u = float(v[0])
                contact_steps = 0
                pushing = False
                info["nudge_to_escort"] = step
                goal = _limit(p, [b[0] + v[0] - FACE_OFF - CLEAR, b[1] + v[1], Z_PUSH],
                              [VMAX, 0.039, VMAX])
            if stage == "nudge":
                goal = _nudge_goal(p, b, v, T, info)
            if stage == "done":
                goal = _limit(p, [p[0] - 0.01, p[1], Z_SAFE], [VMAX, VMAX, VMAX])
        a = _trans_cmd(p, goal, gains)
        r = _rot_cmd(tcp[3:], ori)
        prev_tcp = tcp.copy()
        prev_cmd = (np.asarray(a, float) * 0.1).tolist()
        info["stage"] = stage
        info["last_step"] = step
        if known:
            info["ball"] = [round(float(b[0]), 3), round(float(b[1]), 3)]
            info["ball_v"] = [round(float(v[0]), 4), round(float(v[1]), 4)]
            info["dmin"] = round(float(dmin), 3)
        _safe_step(robot, a, r, -1.0, info)
    info["gains"] = [round(g, 3) for g in gains]
    info["track_len"] = len(track)
    return info


def _seg_dist(a, b, c):
    """Distance from point c to segment a-b (2D)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    ab = b - a
    L = float(ab @ ab)
    s = 0.0 if L < 1e-12 else min(1.0, max(0.0, float((c - a) @ ab) / L))
    return float(np.linalg.norm(a + ab * s - c))


def _nudge_goal(p, b, v, T, info):
    """Fallback for a slow ball (little +y speed left): push it toward the target with the
    closed fingers' +x face or a y face, lifting over the ball to reach the push side."""
    bn = np.asarray(b, float) + np.asarray(v, float)
    d = np.asarray(T, float) - bn
    if float(np.linalg.norm(d)) < GOAL_R - 0.015:
        return _limit(p, [p[0], p[1], p[2]], [VMAX, VMAX, VMAX])
    if d[0] > 0.03 and (abs(d[1]) < 0.06 or d[0] >= abs(d[1])):
        stand = [bn[0] - FACE_OFF - 0.02, bn[1]]
        on_side = p[0] < bn[0] - FACE_OFF + 0.006 and abs(p[1] - bn[1]) < 0.012
        push = [p[0] + 0.012, bn[1]]
        info["nudge_axis"] = "+x"
    else:
        sgn = 1.0 if d[1] > 0 else -1.0
        stand = [bn[0], bn[1] - sgn * (FACE_OFF_Y + 0.02)]
        on_side = sgn * (bn[1] - p[1]) > FACE_OFF_Y - 0.006 and abs(p[0] - bn[0]) < 0.012
        push = [bn[0], p[1] + sgn * 0.012]
        info["nudge_axis"] = "+y" if sgn > 0 else "-y"
    near_stand = float(np.linalg.norm(np.asarray(p[:2]) - np.asarray(stand))) < 0.015
    if on_side and (near_stand or info.get("nudge_push")) and p[2] < Z_PUSH + 0.012:
        info["nudge_push"] = True
        info["nudges"] = info.get("nudges", 0) + 1
        return _limit(p, [push[0], push[1], Z_PUSH], [0.015, 0.015, VMAX])
    info["nudge_push"] = False
    if near_stand:
        return _limit(p, [stand[0], stand[1], Z_PUSH], [VMAX, VMAX, VMAX])
    blocked = _seg_dist(p[:2], stand, bn) < 0.05
    if blocked and p[2] < Z_SAFE - 0.01:
        return _limit(p, [p[0], p[1], Z_SAFE], [0.01, 0.01, VMAX])
    zt = Z_SAFE if blocked else Z_PUSH
    return _limit(p, [stand[0], stand[1], zt], [VMAX, VMAX, VMAX])


# ------------------------------------------------------------- public tools
def perceive_decide(obs, goal, memory, last):
    """Detect the target disc once, then run the closed-loop intercept controller."""
    mem = {
        "target_xy": memory.get("target_xy"),
        "stage": memory.get("stage") or "start",
        "decisions": int(memory.get("decisions") or 0) + 1,
        "last_outcome": memory.get("last_outcome") or "",
    }
    if last:
        mem["last_outcome"] = str(last)[:600]
    if mem["target_xy"] is None:
        tgt = None
        for fr in reversed(list(obs.frames)):
            tgt = _detect_target(fr)
            if tgt is not None:
                break
        if tgt is None:
            mem["stage"] = "no_target"
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
        mem["target_xy"] = [round(tgt[0], 4), round(tgt[1], 4)]
    step = int(obs.current.step)
    remaining = max(1, STEP_LIMIT + 5 - step)
    mem["stage"] = "intercept"
    T = mem["target_xy"]
    return {"memory": mem,
            "act": {"tool": "act_intercept",
                    "args": {"target_x": float(T[0]), "target_y": float(T[1]),
                             "max_steps": int(remaining)}}}


def act_intercept(robot, target_x, target_y, max_steps=65):
    """Closed loop until the episode ends: track the ball (front RGB-D), wait beside its
    line, escort it level in y and push it in +x until its path crosses the target.
    Returns the stage reached, ball estimate, release events and controller gains."""
    log = {"target": [round(float(target_x), 3), round(float(target_y), 3)]}
    return _intercept_loop(robot, [target_x, target_y], int(max_steps), log)


def act_wait(robot, steps=1):
    """Hold position with the gripper closing for a few native steps (used if no target yet)."""
    n = 0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
    return {"waited": n, "step": int(robot.now().step)}
