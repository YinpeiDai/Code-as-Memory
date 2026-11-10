"""ShellGameShuffleTouch policy: watch ball -> track 3 inverted cups through the
shuffle in the front camera (3-D, top-down world xy) -> touch the tracked cup top.

Perception lives in perceive_decide (it processes every frame in obs.frames);
acts only move/wait and report measured outcomes."""

CUP_MIN_PX = 500        # cup-colour pixels needed to say "cups are on the table"
BALL_MIN_PX = 10        # bright-red pixels needed to say "ball is visible"
MOVE_PX = 5             # pixels changed (>12 levels) between frames that count as motion
STILL_NEEDED = 3        # consecutive still frames after motion => shuffle finished
TOP_Z = 0.095           # world z above which cup points belong to the (inverted) cup top
TRAVEL_Z = 0.20         # transit height above cups (cup tops are at ~0.106 m)
TOUCH_Z = 0.090         # commanded z for the final descent (below cup top -> contact)
HOVER_Z = 0.120         # rest height over the cup top (contact happens at TCP z ~0.116)


def _masks(rgb):
    im = rgb.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    ball = (r > 150) & (g < 60) & (b < 60)
    cup = (r > 2 * g) & (r > 2 * b) & (r < 150)
    return ball, cup


def _world_pts(depth, K, E, mask):
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs].astype(np.float64) / 1000.0
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    xc = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    yc = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([xc, yc, z], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t) @ R


def _kmeans(p, c, iters):
    c = np.array(c, dtype=np.float64)
    lab = np.zeros(len(p), dtype=int)
    for _ in range(iters):
        d = ((p[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(len(c)):
            if (lab == k).any():
                c[k] = p[lab == k].mean(0)
    return c, lab


def _farthest_init(p):
    m = p.mean(0)
    c1 = p[np.argmax(((p - m) ** 2).sum(1))]
    c2 = p[np.argmax(((p - c1) ** 2).sum(1))]
    d = np.minimum(((p - c1) ** 2).sum(1), ((p - c2) ** 2).sum(1))
    c3 = p[np.argmax(d)]
    return np.array([c1, c2, c3])


def _cup_points(frame, cupmask):
    p = _world_pts(frame.front_depth, frame.front_K, frame.front_extrinsic, cupmask)
    return p[p[:, 2] > 0.05]


def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(a)
    if s < 1e-6:
        return np.zeros(3)
    return v / (2 * s) * a


def _rot_cmd(tcp, rpy_des, gain=0.3, dead=0.03, lim=0.3):
    """Rotation command that turns the measured orientation toward rpy_des.
    Native CPU: new R = Rx(-0.1r0)Ry(-0.1r1)Rz(-0.1r2) R_cur ~ exp(-0.1 r) R_cur."""
    Rc = _rpy_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_R(rpy_des[0], rpy_des[1], rpy_des[2])
    w = _rotvec(Rd @ Rc.T)
    n = float(np.linalg.norm(w))
    if n < dead:
        return [0.0, 0.0, 0.0], n
    r = -gain * w / 0.1
    r = np.clip(r, -lim, lim)
    return [float(v) for v in r], n


def _defaults(m):
    d = {
        "stage": "watch", "ball_xy": None, "ball_px": 0, "ball_step": None,
        "cups": None, "ball_cup": None, "ball_match_dist": None, "cups_step": None,
        "motion_frames": 0, "still_frames": 0, "last_frame_step": -1,
        "target": None, "shuffle_end_step": None, "home_rpy": None,
        "attempts": 0, "log": "",
    }
    for k, v in d.items():
        if m.get(k) is None:
            m[k] = v
    return m


def _process_frames(obs, m):
    prev_rgb = None
    for f in obs.frames:
        if m["home_rpy"] is None:
            m["home_rpy"] = [float(v) for v in f.tcp[3:6]]
        if f.step <= m["last_frame_step"]:
            prev_rgb = f.front_rgb
            continue
        ballm, cupm = _masks(f.front_rgb)
        ncup = int(cupm.sum())
        moved = None
        if prev_rgb is not None:
            ch = int((np.abs(f.front_rgb.astype(np.int16) - prev_rgb.astype(np.int16)).max(-1) > 12).sum())
            moved = ch > MOVE_PX
        prev_rgb = f.front_rgb
        m["last_frame_step"] = int(f.step)
        if m["cups"] is None:
            nb = int(ballm.sum())
            if nb > BALL_MIN_PX and ncup < CUP_MIN_PX:
                bp = _world_pts(f.front_depth, f.front_K, f.front_extrinsic, ballm)
                if len(bp) > 0:
                    m["ball_xy"] = [float(v) for v in np.median(bp, 0)[:2]]
                    m["ball_px"] = nb
                    m["ball_step"] = int(f.step)
            if ncup >= CUP_MIN_PX:
                p = _cup_points(f, cupm)[:, :2]
                if len(p) < 30:
                    continue
                c, _ = _kmeans(p, _farthest_init(p), 12)
                m["cups"] = [[float(a), float(b)] for a, b in c]
                m["cups_step"] = int(f.step)
                if m["ball_xy"] is not None:
                    d = np.sqrt(((c - np.array(m["ball_xy"])) ** 2).sum(1))
                    m["ball_cup"] = int(np.argmin(d))
                    m["ball_match_dist"] = float(d.min())
                m["log"] += "cups@%d ball_cup=%s d=%.3f; " % (f.step, m["ball_cup"], m["ball_match_dist"] or -1)
            continue
        p = _cup_points(f, cupm)[:, :2]
        if len(p) >= 30:
            c, _ = _kmeans(p, np.array(m["cups"]), 5)
            m["cups"] = [[float(a), float(b)] for a, b in c]
        if moved is None:
            continue
        if moved:
            m["motion_frames"] += 1
            m["still_frames"] = 0
        else:
            m["still_frames"] += 1
    return m


def _target_from_frame(frame, m):
    """Centre of the tracked cup's top disc (points with z > TOP_Z) in world xy."""
    _, cupm = _masks(frame.front_rgb)
    p = _cup_points(frame, cupm)
    c = np.array(m["cups"])
    lab = ((p[:, None, :2] - c[None]) ** 2).sum(-1).argmin(1)
    q = p[lab == m["ball_cup"]]
    top = q[q[:, 2] > TOP_Z]
    if len(top) >= 20:
        xy = top[:, :2].mean(0)
        ztop = float(np.percentile(top[:, 2], 90))
    else:
        xy = c[m["ball_cup"]]
        ztop = 0.106
    return [float(xy[0]), float(xy[1]), ztop]


def perceive_decide(obs, goal, memory, last):
    """Track ball and cups, then pick the next act. Stages: watch -> approach -> hover -> descend -> lift/retry."""
    m = _defaults(dict(memory))
    m = _process_frames(obs, m)
    rpy = m["home_rpy"]
    stage = m["stage"]
    if stage == "watch":
        ended = (m["cups"] is not None and m["motion_frames"] > 0 and m["still_frames"] >= STILL_NEEDED)
        if not ended:
            act = {"tool": "act_wait_shuffle", "args": {
                "cups_seen": m["cups"] is not None, "motion_seen": m["motion_frames"] > 0,
                "max_steps": 590}}
            return {"memory": m, "act": act}
        if m["ball_cup"] is None:
            m["ball_cup"] = 1
            m["log"] += "no ball seen; guessing middle cup; "
        m["shuffle_end_step"] = int(obs.current.step)
        m["target"] = _target_from_frame(obs.current, m)
        m["log"] += "shuffle end@%d target=%s; " % (obs.current.step, [round(v, 3) for v in m["target"]])
        stage = "approach"
    if stage == "approach":
        tx, ty, _ = m["target"]
        m["stage"] = "hover"
        return {"memory": m, "act": {"tool": "act_move_to", "args": {
            "x": tx, "y": ty, "z": TRAVEL_Z, "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "tol": 0.006, "max_steps": 60}}}
    if stage == "hover":
        # R009: native success fired while the TCP decelerated ~1.7 cm above the cup top (no contact), so first
        # come to rest just above the cup; contact descent (v4) is the fallback.
        tx, ty, _ = m["target"]
        m["stage"] = "descend"
        return {"memory": m, "act": {"tool": "act_move_to", "args": {
            "x": tx, "y": ty, "z": HOVER_Z, "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "tol": 0.004, "max_steps": 40, "gain": 1.5, "vmax": 0.5, "settle_steps": 10}}}
    if stage == "descend":
        tx, ty, _ = m["target"]
        m["attempts"] += 1
        m["stage"] = "lift"
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": tx, "y": ty, "z": TOUCH_Z, "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "max_steps": 45, "hold_steps": 15, "vz": 0.6,
            "hold_push": 0.0 if m["attempts"] <= 1 else -0.05}}}
    if stage == "lift":
        # touched but no success recorded: lift clear and try again (release/lift hypothesis)
        tx, ty, _ = m["target"]
        m["stage"] = "descend" if m["attempts"] < 4 else "done"
        return {"memory": m, "act": {"tool": "act_move_to", "args": {
            "x": tx, "y": ty, "z": 0.16, "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2],
            "tol": 0.008, "max_steps": 30}}}
    return {"memory": m, "act": {"tool": "act_hold", "args": {"steps": 20}}}


def act_wait_shuffle(robot, cups_seen=False, motion_seen=False, max_steps=590):
    """Send zero motion (gripper open) every step until cups have appeared, moved, and then stayed
    still for STILL_NEEDED frames (shuffle finished), or max_steps. Returns measured counts."""
    prev = robot.now().front_rgb.astype(np.int16)
    still = 0
    n = 0
    first_cup = None
    first_motion = None
    while n < max_steps:
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        cur = f.front_rgb.astype(np.int16)
        ch = int((np.abs(cur - prev).max(-1) > 12).sum())
        prev = cur
        if not cups_seen:
            _, cupm = _masks(f.front_rgb)
            if int(cupm.sum()) >= CUP_MIN_PX:
                cups_seen = True
                first_cup = int(f.step)
            continue
        if ch > MOVE_PX:
            if not motion_seen:
                first_motion = int(f.step)
            motion_seen = True
            still = 0
        else:
            still += 1
            if motion_seen and still >= STILL_NEEDED:
                break
    return {"steps": n, "end_step": int(robot.now().step), "cups_seen": cups_seen,
            "motion_seen": motion_seen, "still": still, "first_cup_step": first_cup,
            "first_motion_step": first_motion}


def act_move_to(robot, x, y, z, roll, pitch, yaw, tol=0.006, max_steps=60, gain=2.0, vmax=1.0, settle_steps=0):
    """Closed-loop move of the TCP to world (x,y,z) holding orientation (roll,pitch,yaw); gripper open.
    Stops when position error < tol and per-step motion < 2 mm. Returns measured errors."""
    goal = np.array([x, y, z], dtype=np.float64)
    f0 = robot.now()
    dry = None
    try:
        dr = robot.dry_run(float(x), float(y), float(z), float(roll), float(pitch), float(yaw), mode="absolute")
        dry = {"ok": dr.get("within_joint_limits"), "min_margin": dr.get("min_joint_margin_rad")}
    except Exception as e:
        dry = {"error": str(e)[:100]}
    prev = np.array(f0.tcp[:3], dtype=np.float64)
    n = 0
    maxrot = 0.0
    while n < max_steps:
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        err = goal - tcp[:3]
        speed = float(np.linalg.norm(tcp[:3] - prev))
        prev = tcp[:3].copy()
        if np.linalg.norm(err) < tol and (n > 0 and speed < 0.002):
            break
        cmd = np.clip(err * gain / 0.1, -vmax, vmax)
        rc, rerr = _rot_cmd(tcp, [roll, pitch, yaw])
        maxrot = max(maxrot, rerr)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), rc[0], rc[1], rc[2], 1.0)
        n += 1
    for _ in range(int(settle_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    tcp = np.array(robot.now().tcp, dtype=np.float64)
    return {"steps": n, "end_step": int(robot.now().step), "tcp": [round(float(v), 4) for v in tcp[:3]],
            "err": [round(float(v), 4) for v in (goal - tcp[:3])], "max_rot_err": round(maxrot, 4),
            "dry_run": dry}


def act_touch(robot, x, y, z, roll, pitch, yaw, max_steps=40, hold_steps=15, vz=0.6, hold_push=0.0):
    """Descend vertically over (x,y) toward z with bounded speed (vz*0.1 m target offset per step) holding xy and
    orientation. When the TCP stops descending (contact with the cup top), stop commanding motion
    (dz=hold_push, everything else 0) so the arm comes to rest, for up to hold_steps.
    (R001-R003: native success fired only once joint motion had died down; pushing on made the hand tilt.)
    Returns measured z/contact summary. The episode ends automatically on native success."""
    n = 0
    zs = []
    held = 0
    contact_z = None
    contact_step = None
    while n < max_steps:
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        zs.append(float(tcp[2]))
        if contact_z is None and n >= 2 and zs[-1] < 0.14 and (zs[-2] - zs[-1]) < 0.002:
            contact_z = zs[-1]
            contact_step = int(robot.now().step)
        if contact_z is not None:
            if held >= hold_steps:
                break
            held += 1
            robot.step(0.0, 0.0, float(hold_push), 0.0, 0.0, 0.0, 1.0)
            n += 1
            continue
        exy = (np.array([x, y]) - tcp[:2]) * 2.0 / 0.1
        dz = max(-vz, min(vz, (z - tcp[2]) * 2.0 / 0.1))
        rc, _ = _rot_cmd(tcp, [roll, pitch, yaw])
        robot.step(float(np.clip(exy[0], -0.3, 0.3)), float(np.clip(exy[1], -0.3, 0.3)), dz,
                   rc[0], rc[1], rc[2], 1.0)
        n += 1
    tcp = np.array(robot.now().tcp, dtype=np.float64)
    return {"steps": n, "end_step": int(robot.now().step), "min_z": round(min(zs), 4) if zs else None,
            "held": held, "contact_z": contact_z, "contact_step": contact_step,
            "tcp": [round(float(v), 4) for v in tcp[:6]], "z_tail": [round(v, 4) for v in zs[-6:]]}


def act_hold(robot, steps=20):
    """Hold still (zero motion, gripper open) for `steps` native steps."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": int(steps), "end_step": int(robot.now().step)}
