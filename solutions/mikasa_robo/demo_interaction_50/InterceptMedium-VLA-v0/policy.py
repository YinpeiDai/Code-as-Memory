"""InterceptMedium policy: track the rolling ball with the front RGB-D camera,
meet it with the flat side face of the (nearly closed) fingers so that the
inelastic contact turns the ball toward the target, then escort it there."""

R_BALL = 0.02        # ball radius (demo sphere fit: 0.0197-0.0199 m)
FACE_H = 0.0105      # +hand-x finger face offset from TCP (wrist depth: +0.0094..+0.0115)
Z_LOW = 0.013        # TCP height while intercepting (fingertips ~0.008 below TCP)
Z_SAFE = 0.075       # TCP height that clears the ball top (0.04) with the fingers
GRIP_CMD = -0.4      # absolute finger target: nearly closed, gap < ball diameter
K_SS = 0.045         # free-motion TCP metres per step per unit command (demo fit)
KP = 0.55            # position feedback fraction per step
KR = 0.5             # rotation feedback fraction per step
D_PRE = 0.10         # preferred capture distance before the target line (y)
T_MIN = 17           # earliest capture step (descent + rotation take ~10-12 steps)
T_MAX = 34           # latest capture step (leave >=20 steps for the escort)
N_APP = 4            # minimum steps of the final approach along the face normal
N_PAR = 3            # steps of parallel (gap-holding) motion before closing in
N_SWING = 3          # steps to swing from the side waiting point in behind the ball
PUSH_SPEED = 0.021   # desired ball speed along the push direction (m/step)
PUSH_MAX = 0.032
T_DONE = 46.0        # planned latest disc entry (R041 entered at 51 with a 52-step plan)
YAW_READY = 0.30     # yaw error allowed at commit; rotation (~0.04 rad/step) continues during sync
CLOSE_MIN = 0.009    # minimum closing speed face->ball at contact (m/step)
CLEAR_X = 0.075      # waiting TCP must be this far (-x) from the ball line
PRESS = 0.002        # face pre-load into the ball during escort
ESC_MIN = 0.015      # escort speed bounds (m/step)
ESC_MAX = 0.026
BASE_XY = (-0.615, 0.0)
MIN_BASE_DIST = 0.30
MAX_BASE_DIST = 0.75   # table-height reach (R030: TCP stalled at 0.75 m trying to reach 0.86 m)
CLAMP_BASE_DIST = 0.28


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _rotvec(Rm):
    c = (np.trace(Rm) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    if ang > math.pi - 1e-3:
        # near pi: axis from the largest diagonal entry
        i = int(np.argmax(np.diag(Rm)))
        ax = np.zeros(3)
        ax[i] = math.sqrt(max(0.0, (Rm[i, i] + 1.0) / 2.0))
        for j in range(3):
            if j != i:
                ax[j] = (Rm[i, j] + Rm[j, i]) / (4.0 * ax[i])
        return ax * ang
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return w * (ang / (2.0 * math.sin(ang)))


def _front_world(frame, mask):
    """World XYZ of the masked front-camera pixels (uses depth, K, extrinsic)."""
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    depth = np.asarray(frame.front_depth)
    vs, us = np.nonzero(mask & (depth > 0))
    if len(us) == 0:
        return np.zeros((0, 3)), us, vs
    z = depth[vs, us].astype(float) / 1000.0
    pc = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], 1)
    return (pc - E[:, 3]) @ E[:, :3], us, vs


def _cam_center(frame):
    E = np.asarray(frame.front_extrinsic, dtype=float)
    return -E[:, :3].T @ E[:, 3]


def _detect_target(frame):
    """Centre of the red/white target rings lying flat on the table."""
    img = np.asarray(frame.front_rgb).astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    m = (r > 140) & (g < 90) & (g >= 50) & (b < 95) & (r - g > 90)
    w, us, vs = _front_world(frame, m)
    if len(w) == 0:
        return None
    w = w[(w[:, 2] < 0.012) & (w[:, 2] > -0.01)]
    if len(w) < 30:
        return None
    med = np.median(w, 0)
    k = np.sqrt(((w[:, :2] - med[:2]) ** 2).sum(1)) < 0.13
    if k.sum() < 30:
        return None
    c = w[k].mean(0)
    return [float(c[0]), float(c[1]), int(k.sum())]


def _detect_ball(frame, near=None):
    """Ball centre from saturated-red pixels above the table plane, or None."""
    img = np.asarray(frame.front_rgb).astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    m = (r > 80) & (g < 55) & (b < 55) & (r > 2 * g + 40)
    w, us, vs = _front_world(frame, m)
    if len(w) < 5:
        return None
    k = (w[:, 2] > 0.010) & (w[:, 2] < 0.06) & (np.abs(w[:, 0]) < 0.8) & (np.abs(w[:, 1]) < 1.2)
    w, us = w[k], us[k]
    if len(w) < 5:
        return None
    ref = np.median(w, 0) if near is None else np.array([near[0], near[1], 0.03])
    dist = np.sqrt(((w[:, :2] - ref[:2]) ** 2).sum(1))
    k = dist < (0.035 if near is None else 0.06)
    if k.sum() < 5:
        k = dist < 0.06
        if near is not None and k.sum() < 5:
            return _detect_ball(frame, None)
        if k.sum() < 5:
            return None
    w, us = w[k], us[k]
    med = np.median(w, 0)
    k = np.sqrt(((w - med) ** 2).sum(1)) < 0.03
    w, us = w[k], us[k]
    if len(w) < 5:
        return None
    mean = w.mean(0)
    ray = mean - _cam_center(frame)
    ray = ray / (np.linalg.norm(ray) + 1e-9)
    c = mean + 0.013 * ray
    edge = bool(us.min() <= 1 or us.max() >= 254)
    return [float(c[0]), float(c[1]), float(c[2]), int(len(w)), edge]


def _fit_line(hist):
    """Least-squares constant-velocity fit over (step, x, y) samples."""
    if len(hist) < 2:
        return None
    h = np.array(hist[-12:], dtype=float)
    t = h[:, 0]
    if t.max() - t.min() < 1:
        return None
    A = np.stack([np.ones_like(t), t], 1)
    cx = np.linalg.lstsq(A, h[:, 1], rcond=None)[0]
    cy = np.linalg.lstsq(A, h[:, 2], rcond=None)[0]
    return [float(cx[0]), float(cy[0]), float(cx[1]), float(cy[1])]


def _pred(line, t):
    return np.array([line[0] + line[2] * t, line[1] + line[3] * t])


def _plan_at(line, T, t_c):
    """Geometry of a capture at step t_c (see _plan)."""
    v = np.array([line[2], line[3]])
    C = _pred(line, t_c)
    d = T - C
    dn = float(np.linalg.norm(d))
    n = d / max(dn, 1e-6)
    vn = float(v @ n)
    # push fast enough that the ball reaches the disc (~0.1 m) by step T_DONE
    need = (dn - 0.09) / max(T_DONE - t_c, 4.0)
    s_n = min(max(PUSH_SPEED, need, vn + CLOSE_MIN), PUSH_MAX)
    c = max(s_n - vn, 0.004)
    gap = max(c * N_APP, 0.03)
    n_close = gap / c
    t_s = t_c - n_close - N_PAR           # start of the gap-holding parallel motion
    Gc = C - (R_BALL + FACE_H) * n
    Gs = _pred(line, t_s) - (R_BALL + FACE_H + gap) * n
    # wait beside the ball lane (>= CLEAR_X in -x) at the y of Gs; swing in behind the
    # ball during N_SWING steps once it has passed (R030: a gap-based clearance made the
    # waiting point unreachable for steep push directions)
    x_lane = line[0] + line[2] * ((Gs[1] - line[1]) / max(line[3], 1e-3))
    Pw = np.array([min(float(Gs[0]), x_lane - CLEAR_X), float(Gs[1])])
    n_swing = N_SWING if Pw[0] < Gs[0] - 0.005 else 0
    n_tot = n_close + N_PAR + n_swing
    # TCP path samples: waiting point, closing path, contact point, first push steps
    pts = [Pw, Gs]
    for k in range(1, 6):
        tt = t_s + ((t_c - t_s) * k) / 5.0
        gk = min(gap, max(0.0, c * (t_c - tt)))
        pts.append(_pred(line, tt) - (R_BALL + FACE_H + gk) * n)
    for k in range(1, 6):
        pts.append(Gc + n * s_n * k)
    dists = [math.hypot(P[0] - BASE_XY[0], P[1] - BASE_XY[1]) for P in pts]
    dmin = min(dists)
    # reach margin: negative when some path point is too close to or too far from the base
    margin = min(dmin - MIN_BASE_DIST, MAX_BASE_DIST - max(dists))
    return {"t_c": t_c, "C": C, "d": n, "n": n, "s": s_n, "c": c, "V": float(np.linalg.norm(v)), "v": v,
            "gap": gap, "n_tot": n_tot, "Gc": Gc, "Pw": Pw, "base_dmin": dmin, "reach_margin": margin,
            "yaw": math.atan2(n[1], n[0]), "dist_CT": dn}


def _plan(line, target, t_now, t_floor=0.0):
    """Capture step t_c, capture point C on the ball line, push direction n = unit(T - C).

    The face (+hand-x side of the closed fingers) is turned to n. The gripper matches the
    ball's motion across n and closes in along n at speed c, so the ball meets the face
    head-on; afterwards the face pushes the ball along n (R001/R002: after contact the
    ball's along-face velocity dies within ~4 steps, its normal velocity follows the face).
    Candidates whose TCP path comes closer than MIN_BASE_DIST to the robot base (J4 limit)
    are replaced by the nearest acceptable capture step."""
    T = np.array(target[:2])
    vy = max(1e-3, line[3])
    t_geo = (T[1] - D_PRE - line[1]) / vy
    lo = max(t_now + 2.0, float(t_floor))
    t_nom = max(float(T_MIN), min(float(T_MAX), t_geo), lo)
    best = _plan_at(line, T, t_nom)
    if best["reach_margin"] >= 0.0:
        return best
    for k in range(1, 30):
        for tc in (t_nom - k, t_nom + k):
            if tc < max(lo, float(T_MIN) - 4.0) or tc > 45.0:
                continue
            p = _plan_at(line, T, tc)
            if p["reach_margin"] >= 0.0:
                return p
            if p["reach_margin"] > best["reach_margin"]:
                best = p
    return best


def _clip(a, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(a)))


def _command(tcp, g_des, ff, z_des, yaw_des):
    """Normalized step command toward g_des (xy), z_des and the down-pointing yaw."""
    pos = np.asarray(tcp[:3], dtype=float)
    gx, gy = float(g_des[0]), float(g_des[1])
    rx, ry = gx - BASE_XY[0], gy - BASE_XY[1]
    rr = math.hypot(rx, ry)
    if rr < CLAMP_BASE_DIST:
        # keep the TCP away from the base (J4 elbow limit at table height)
        gx = BASE_XY[0] + rx / max(rr, 1e-6) * CLAMP_BASE_DIST
        gy = BASE_XY[1] + ry / max(rr, 1e-6) * CLAMP_BASE_DIST
    e = np.array([gx - pos[0], gy - pos[1], z_des - pos[2]])
    a = (KP * e + np.array([ff[0], ff[1], 0.0])) / K_SS
    a = np.array([_clip(a[0]), _clip(a[1]), _clip(a[2])])
    R_cur = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    R_des = _rpy_to_R(math.pi, 0.0, yaw_des)
    w = _rotvec(R_des @ R_cur.T)
    r = -KR * w / 0.1
    nr = float(np.linalg.norm(r))
    if nr > 1.0:
        r = r / nr
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])]


def _r3(x):
    return [round(float(v), 4) for v in x]


def perceive_decide(obs, goal, memory, last):
    """Detect the target once, then run the closed-loop intercept routine."""
    stage = memory.get("stage") or "start"
    target = memory.get("target")
    notes = memory.get("notes") or ""
    if target is None:
        for fr in reversed(obs.frames):
            t = _detect_target(fr)
            if t is not None:
                target = [t[0], t[1]]
                break
    ball = _detect_ball(obs.current)
    mem = {"stage": stage, "target": target, "ball_first": memory.get("ball_first"),
           "notes": notes, "last_result": (last or "")[:600]}
    if mem["ball_first"] is None and ball is not None:
        mem["ball_first"] = [ball[0], ball[1]]
    if target is None:
        mem["stage"] = "search_target"
        return {"memory": mem, "act": {"tool": "act_observe", "args": {"steps": 1}}}
    if stage in ("start", "search_target"):
        mem["stage"] = "intercept"
        return {"memory": mem, "act": {"tool": "act_intercept",
                                       "args": {"target_x": target[0], "target_y": target[1]}}}
    mem["stage"] = "hold"
    return {"memory": mem, "act": {"tool": "act_observe", "args": {"steps": 5}}}


def act_observe(robot, steps=1):
    """Hold the arm still (zero motion, fingers nearly closed) for a few steps."""
    n = 0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP_CMD)
        n += 1
    f = robot.now()
    return {"steps": n, "tcp": _r3(f.tcp), "fingers": _r3(f.fingers)}


def act_intercept(robot, target_x, target_y, max_steps=70):
    """Closed-loop intercept: fit the ball line, wait beside/behind the capture point,
    move with the ball while closing in along the push direction, then push the ball
    with the finger face toward the target centre."""
    target = [float(target_x), float(target_y)]
    T = np.array(target)
    hist = []           # (step, x, y) ball samples before contact
    plan = None
    committed = False
    phase = "goto"
    contact_step = None
    commit_step = None
    t_floor = 0.0
    log = []
    last_ball = None
    last_ball_t = 0
    esc = []
    yaw_cmd = None
    min_dist = 9.0
    n_postpone = 0
    for k in range(int(max_steps)):
        f = robot.now()
        t = int(f.step)
        tcp = np.asarray(f.tcp, dtype=float)
        b = _detect_ball(f, last_ball)
        if b is not None and not b[4]:
            last_ball = [b[0], b[1]]
            last_ball_t = t
            if phase in ("goto", "sync") and t >= 3:
                hist.append((t, b[0], b[1]))
            min_dist = min(min_dist, math.hypot(b[0] - T[0], b[1] - T[1]))
        line = _fit_line(hist)
        if line is None:
            if b is not None:
                line = [b[0], b[1] - 0.022 * t, 0.0005, 0.022]
            else:
                line = [target[0] - 0.185, -0.72, 0.0005, 0.022]
        if not committed:
            plan = _plan(line, target, t, t_floor)
            if t >= plan["t_c"] - plan["n_tot"] - 0.5 and len(hist) >= 4:
                ready = (math.hypot(tcp[0] - plan["Pw"][0], tcp[1] - plan["Pw"][1]) < 0.035
                         and tcp[2] < 0.035
                         and abs(math.atan2(math.sin(tcp[5] - plan["yaw"]), math.cos(tcp[5] - plan["yaw"]))) < YAW_READY)
                if ready or t >= 40 or n_postpone >= 2:
                    committed = True
                    phase = "sync"
                    commit_step = t
                else:
                    n_postpone += 1
                    t_floor = t + plan["n_tot"] + 2
                    plan = _plan(line, target, t, t_floor)
            yaw_cmd = plan["yaw"]
        n = plan["n"]
        tv = np.array([-n[1], n[0]])
        t_c = plan["t_c"]
        z_des = Z_LOW
        g = tcp[:2]
        ff = np.zeros(2)
        if phase == "goto":
            g = plan["Pw"]
            if b is not None and math.hypot(b[0] - tcp[0], b[1] - tcp[1]) < 0.06 and tcp[2] > 0.03:
                z_des = Z_SAFE
        elif phase == "sync":
            if b is not None and not b[4]:
                rel = np.array([b[0] - tcp[0], b[1] - tcp[1]])
                if float(rel @ n) < R_BALL + FACE_H + 0.006 and abs(float(rel @ tv)) < 0.03:
                    phase = "escort"
                    contact_step = t
            if t >= t_c + 1 and phase == "sync":
                phase = "escort"
                contact_step = t
            if phase == "sync":
                gap1 = min(plan["gap"], max(0.0, plan["c"] * (t_c - (t + 1))))
                gap0 = min(plan["gap"], max(0.0, plan["c"] * (t_c - t)))
                p1 = _pred(line, t + 1) - (R_BALL + FACE_H + gap1) * n
                p0 = _pred(line, t) - (R_BALL + FACE_H + gap0) * n
                g = p0
                ff = p1 - p0
        if phase == "escort":
            if b is not None and not b[4]:
                esc.append((t, b[0], b[1]))
            if last_ball is not None:
                bp = np.array(last_ball)
            else:
                bp = _pred(line, t)
            to_t = T - bp
            dist = float(np.linalg.norm(to_t))
            if dist < 0.025 or float(to_t @ plan["d"]) < -0.02:
                phase = "hold"
            else:
                u = to_t / max(dist, 1e-6)
                # re-aim the face toward the target centre (rotation is rate limited by KR)
                yaw_cmd = math.atan2(u[1], u[0])
                n_cur = np.array([math.cos(float(tcp[5])), math.sin(float(tcp[5]))])
                t_cur = np.array([-n_cur[1], n_cur[0]])
                vt = 0.0
                if len(esc) >= 2 and esc[-1][0] > esc[-2][0]:
                    vb = (np.array(esc[-1][1:]) - np.array(esc[-2][1:])) / float(esc[-1][0] - esc[-2][0])
                    vt = float(vb @ t_cur)
                g = bp - (R_BALL + FACE_H - PRESS) * n_cur
                ff = plan["s"] * n_cur + vt * t_cur
        if phase == "hold":
            g = tcp[:2]
            ff = np.zeros(2)
        cmd = _command(tcp, g, ff, z_des, yaw_cmd)
        if len(log) < 80:
            log.append([t, phase[0], _r3(tcp[:3]), None if b is None else _r3(b[:2]), _r3(cmd[:3])])
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], GRIP_CMD)
    return {"phase": phase, "contact_step": contact_step, "commit_step": commit_step, "t_c": plan["t_c"],
            "postponed": n_postpone,
            "C": _r3(plan["C"]), "n": _r3(plan["n"]), "s": round(plan["s"], 4),
            "yaw": round(plan["yaw"], 3), "line": _r3(line), "min_dist": round(min_dist, 4),
            "log_tail": log[-8:]}
