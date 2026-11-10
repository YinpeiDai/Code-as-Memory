"""ShellGameTouch policy: remember the ball seen during the cue, locate the three
inverted mugs once they appear, choose the mug nearest the remembered ball and
touch it with the open gripper (fingers straddling the mug wall), then close."""

BALL_MIN_PX = 5
CUP_MIN_PX = 500


def _backproject(K, E, us, vs, depth_mm):
    z = depth_mm.astype(float) / 1000.0
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return (pc - t) @ R


def _masks(rgb):
    f = rgb.astype(int)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    ball = (r > 180) & (g < 50) & (b < 50)
    cup = (r > 40) & (r <= 180) & (r > 2.5 * g) & (r > 2.5 * b)
    return ball, cup


def _ball_in(rgb, depth, K, E):
    ball, _ = _masks(rgb)
    vs, us = np.nonzero(ball & (depth > 0))
    if len(vs) < BALL_MIN_PX:
        return None
    P = _backproject(K, E, us, vs, depth[vs, us])
    return [float(v) for v in P.mean(0)] + [int(len(vs))]


def _detect_ball(frame):
    """World position of the visible red ball (surface mean, front camera; wrist camera
    as fallback) or None."""
    b = _ball_in(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic)
    if b is None:
        b = _ball_in(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    return b


def _kmeans3(P):
    c = P[:, :2].mean(0)
    u, s, vt = np.linalg.svd(P[:, :2] - c, full_matrices=False)
    proj = (P[:, :2] - c) @ vt[0]
    order = np.argsort(proj)
    cent = np.array([P[order[int(q * (len(P) - 1))], :2] for q in (0.1, 0.5, 0.9)])
    lab = np.zeros(len(P), dtype=int)
    for _ in range(25):
        dd = ((P[:, None, :2] - cent[None, :, :]) ** 2).sum(-1)
        lab = np.argmin(dd, 1)
        new = []
        for k in range(3):
            sel = P[lab == k]
            new.append(sel[:, :2].mean(0) if len(sel) else cent[k])
        cent = np.array(new)
    return lab


def _detect_cups(frame):
    """List of [x, y, top_z, n_top_px] for three cups (top-disc centres), or None."""
    _, cup = _masks(frame.front_rgb)
    d = frame.front_depth
    vs, us = np.nonzero(cup & (d > 0))
    if len(vs) < CUP_MIN_PX:
        return None
    P = _backproject(frame.front_K, frame.front_extrinsic, us, vs, d[vs, us])
    P = P[(P[:, 2] > -0.01) & (P[:, 2] < 0.2)]
    if len(P) < CUP_MIN_PX:
        return None
    lab = _kmeans3(P)
    out = []
    for k in range(3):
        Q = P[lab == k]
        if len(Q) < 50:
            return None
        top = float(np.percentile(Q[:, 2], 98))
        T = Q[Q[:, 2] > top - 0.006]
        out.append([float(T[:, 0].mean()), float(T[:, 1].mean()), top, int(len(T))])
    out.sort(key=lambda c: c[1])
    return out


def _choose(ball, cups):
    d = [math.hypot(c[0] - ball[0], c[1] - ball[1]) for c in cups]
    i = int(np.argmin(d))
    srt = sorted(d)
    return i, srt[0], srt[1]


def perceive_decide(obs, goal, memory, last):
    """Track cue (ball) -> cups -> target, then run the touch routine."""
    mem = {
        "stage": memory.get("stage") or "cue",
        "ball": memory.get("ball"),
        "ball_step": memory.get("ball_step"),
        "cups": memory.get("cups"),
        "target": memory.get("target"),
        "target_index": memory.get("target_index"),
        "target_confidence": memory.get("target_confidence"),
        "touch_tries": memory.get("touch_tries") or 0,
        "log": memory.get("log") or "",
    }
    frames = list(obs.frames)
    if isinstance(last, str) and last:
        mem["log"] = (mem["log"] + " | last:" + last[:300])[-1500:]
    # 1. cue: remember the ball as soon as it is visible
    if mem["ball"] is None:
        for f in frames:
            b = _detect_ball(f)
            if b is not None:
                mem["ball"] = b[:3]
                mem["ball_step"] = int(f.step)
                break
    # 2. cups (static once they appear)
    if mem["cups"] is None:
        cups = _detect_cups(obs.current)
        if cups is not None:
            mem["cups"] = [c[:3] for c in cups]
    if mem["cups"] is not None and mem["target"] is None:
        if mem["ball"] is not None:
            i, d1, d2 = _choose(mem["ball"], mem["cups"])
            mem["target_confidence"] = float(min(1.0, max(0.0, 1.0 - d1 / max(d2, 1e-6))))
        else:
            i = 1
            mem["target_confidence"] = 0.0
        mem["target_index"] = int(i)
        mem["target"] = mem["cups"][i]
        mem["stage"] = "touch"
    if mem["cups"] is None:
        mem["stage"] = "cue"
        return {"memory": mem, "act": {"tool": "act_wait_for_cups", "args": {"max_steps": 12}}}
    tx, ty, tz = mem["target"]
    tries = int(mem["touch_tries"])
    mem["touch_tries"] = tries + 1
    if tries == 0:
        return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": tx, "y": ty, "top": tz}}}
    # fallback: alternate side offset
    side = 0.036 if tries == 1 else (-0.048 if tries % 2 == 0 else 0.060)
    mem["stage"] = "retry"
    return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": tx, "y": ty, "top": tz, "side_dx": side}}}


def act_wait_for_cups(robot, max_steps=12):
    """Send zero motion (gripper open) until the cups are visible in the front camera."""
    n = 0
    ball = None
    while n < max_steps:
        f = robot.now()
        if ball is None:
            ball = _detect_ball(f)
        if _detect_cups(f) is not None:
            return {"cups_visible": True, "steps": n, "step": int(f.step)}
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
        n += 1
    return {"cups_visible": False, "steps": n, "ball": ball}


def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    return w / (2.0 * s) * th


# canonical gripper orientation: roll=pi, pitch=0, yaw=0 (fingers along world y, pointing down)


def _rot_cmd(tcp, prev_tcp, kp=1.0, kd=0.5):
    """Normalized rotation command (drx, dry, drz) that turns the measured orientation toward
    roll=pi, pitch=0, yaw=0. Applied offset is -0.1*r about base axes (measured in demos:
    positive diagonal response ~0.3 per step), so r = -(kp*w_err - kd*w_vel)/0.1."""
    Rd = _rpy2R(math.pi, 0.0, 0.0)
    Rc = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rp = _rpy2R(float(prev_tcp[3]), float(prev_tcp[4]), float(prev_tcp[5]))
    w_err = _logR(Rd @ Rc.T)
    w_vel = _logR(Rc @ Rp.T)
    r = -(kp * w_err - kd * w_vel) / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return [float(v) for v in r], float(np.linalg.norm(w_err))


def _cmd(e, v, kp=2.0, kd=1.0):
    """Damped P command (normalized units) from position error e and last-step velocity v (m)."""
    return np.clip((kp * np.asarray(e) - kd * np.asarray(v)) / 0.1, -1.0, 1.0)


def act_touch(robot, x, y, top, side_dx=0.048, hover=0.04, depth=0.025, close_steps=5, hold_steps=4, max_steps=26):
    """Move to the side of the target mug axis (x offset side_dx so the open fingers
    straddle a short chord of the mug wall) at hover height, descend below the rim once
    xy is aligned, close the gripper on the wall and hold. If the fingertips stall on the
    mug top, back up and move the offset outward. Returns measured progress."""
    off = float(side_dx)
    sgn = 1.0 if off >= 0 else -1.0
    out = {"events": []}
    phase = "travel"
    prev_full = np.asarray(robot.now().tcp, dtype=float)
    n = 0
    stall_count = 0
    desc_n = 0
    low = top - depth
    max_rot_err = 0.0
    while n < max_steps:
        full = np.asarray(robot.now().tcp, dtype=float)
        tcp = full[:3]
        v = tcp - prev_full[:3]
        gxy = np.array([x + off, y])
        exy = gxy - tcp[:2]
        if phase == "travel" and np.linalg.norm(exy) < 0.012:
            phase = "descend"
            out["events"].append(["descend", int(robot.now().step), [round(float(c), 4) for c in tcp]])
        if phase == "lift" and tcp[2] > top + 0.008:
            off += sgn * 0.012
            phase = "descend"
            desc_n = 0
            out["events"].append(["shifted", int(robot.now().step), round(off, 4)])
            exy = np.array([x + off, y]) - tcp[:2]
        if phase == "travel":
            gz = top + hover
        elif phase == "lift":
            gz = top + 0.015
        else:
            gz = low
        e = np.array([exy[0], exy[1], gz - tcp[2]])
        if phase == "travel" and tcp[2] < top + 0.015 and np.linalg.norm(exy) > 0.02:
            e[0] = 0.0
            e[1] = 0.0
        if phase == "lift":
            e[0] = 0.0
            e[1] = 0.0
        if phase == "descend":
            desc_n += 1
            if tcp[2] < low + 0.006:
                break
            # stalled above the rim: fingertip resting on the mug top -> lift, shift outward
            if desc_n > 2 and abs(v[2]) < 0.002 and tcp[2] > top - 0.006 and e[2] < -0.01:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= 2:
                stall_count = 0
                phase = "lift"
                out["events"].append(["stall", int(robot.now().step), [round(float(c), 4) for c in full]])
                e = np.array([0.0, 0.0, top + 0.015 - tcp[2]])
        c = _cmd(e, v)
        rc, rerr = _rot_cmd(full, prev_full)
        max_rot_err = max(max_rot_err, rerr)
        prev_full = full
        robot.step(float(c[0]), float(c[1]), float(c[2]), rc[0], rc[1], rc[2], 1.0)
        n += 1
    out["max_rot_err"] = round(max_rot_err, 4)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["reach"] = {"steps": n, "phase": phase, "tcp": [round(float(c), 4) for c in tcp], "off": round(off, 4),
                    "step": int(robot.now().step)}
    fing = []
    regrips = 0
    for attempt in range(2):
        goal = np.array([x + off, y, low])
        prev_full = np.asarray(robot.now().tcp, dtype=float)
        closed_empty = False
        for i in range(close_steps + hold_steps):
            full = np.asarray(robot.now().tcp, dtype=float)
            tcp = full[:3]
            c = np.clip(_cmd(goal - tcp, tcp - prev_full[:3]), -0.5, 0.5)
            rc, rerr = _rot_cmd(full, prev_full)
            prev_full = full
            robot.step(float(c[0]), float(c[1]), float(c[2]), rc[0], rc[1], rc[2], -1.0)
            fw = float(robot.now().fingers[0])
            fing.append(round(fw, 4))
            if fw < 0.004:
                closed_empty = True
                break
        if not closed_empty:
            break
        # fingers closed on nothing: the wall chord was missed, move the offset inward
        regrips += 1
        off -= sgn * 0.012
        out["events"].append(["closed_empty", int(robot.now().step), round(off, 4)])
        for i in range(5):
            full = np.asarray(robot.now().tcp, dtype=float)
            tcp = full[:3]
            # reopen in place first, then shift inward once the fingers are apart
            tgt = goal if float(robot.now().fingers[0]) < 0.03 else np.array([x + off, y, low])
            c = np.clip(_cmd(tgt - tcp, tcp - prev_full[:3]), -0.5, 0.5)
            rc, rerr = _rot_cmd(full, prev_full)
            prev_full = full
            robot.step(float(c[0]), float(c[1]), float(c[2]), rc[0], rc[1], rc[2], 1.0)
    tcp = robot.now().tcp
    out["close"] = {"fingers": fing, "regrips": regrips, "tcp": [round(float(v), 4) for v in tcp],
                    "step": int(robot.now().step)}
    # lift clear of the rim before any retry
    for i in range(2):
        t = np.asarray(robot.now().tcp, dtype=float)
        if t[2] > top + 0.03:
            break
        robot.step(0, 0, 1.0, 0, 0, 0, 1.0)
    return out
