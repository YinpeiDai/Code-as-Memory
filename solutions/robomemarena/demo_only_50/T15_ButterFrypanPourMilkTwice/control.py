"""Closed-loop end-effector control tools (OSC_POSE delta, world frame)."""

ORI = {
    # gripper pointing straight down, fingers along world y (butter pick/place)
    "down": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    # milk grasp: approach axis tilted ~49.5 deg toward -x (mean of 50 demos)
    "grasp": [[0.650, -0.003, -0.760], [-0.002, -1.0, 0.002], [-0.760, 0.0, -0.650]],
    # expert pour pose (mean of 50 demos) = grasp rotated 45.0 deg about world -y
    "pour_expert": [[0.997, -0.002, -0.077], [-0.002, -1.0, 0.0], [-0.077, 0.0, -0.997]],
}
# Our pour target tilts the carton a little further than the expert's 45 deg so
# that stopping within the rotation tolerance still exceeds the expert's tilt.
POUR_TILT_DEG = 49.0


def _ori_matrix(name):
    if name == "pour":
        th = -math.radians(POUR_TILT_DEG)
        Ry = np.array([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0],
                       [-math.sin(th), 0.0, math.cos(th)]])
        return Ry @ _ori_matrix("grasp")
    M = np.array(ORI[name], dtype=float)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]
        R = U @ Vt
    return R


def _pose(frame):
    tcp = np.asarray(frame.tcp, dtype=float)
    return tcp[:3].copy(), _rodrigues(tcp[3:6])


def _finger_gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def act_move(robot, x, y, z, ori="down", grip=-1.0, tol=0.004, ori_tol_deg=2.0,
             max_steps=300, kp=12.0, kr=2.5, vmax=1.0, ki=1.0):
    """Drive the TCP to world (x, y, z) [m] with named orientation ori
    ('down', 'grasp', 'pour' or 'keep' = no rotation control) while sending
    gripper command grip (+1 close, -1 open). Proportional control on measured
    TCP each step (plus a small clamped integral term against payload sag);
    stops when position error < tol and orientation error < ori_tol_deg for 2
    consecutive steps, when it has stalled within 3*tol for 15 steps, or after
    max_steps. Returns steps,
    final position error (m), orientation error (deg) and finger gap (m)."""
    target = np.array([x, y, z], dtype=float)
    Rt = None if ori == "keep" else _ori_matrix(ori)
    good = 0
    steps = 0
    perr = 1.0
    oerr = 0.0
    integ = np.zeros(3)
    hist = []
    stalled = False
    while steps < int(max_steps):
        p, Rc = _pose(robot.now())
        e = target - p
        perr = float(np.linalg.norm(e))
        hist.append(p)
        if perr < 0.03:
            integ = np.clip(integ + e, -0.3 / max(ki, 1e-6), 0.3 / max(ki, 1e-6))
        a = kp * e + ki * integ
        n = float(np.linalg.norm(a))
        if n > vmax:
            a = a * (vmax / n)
        w = np.zeros(3)
        if Rt is not None:
            w = _log_rot(Rt @ Rc.T)
            oerr = math.degrees(float(np.linalg.norm(w)))
            w = np.clip(kr * w, -1.0, 1.0)
        if perr < tol and oerr < ori_tol_deg:
            good += 1
            if good >= 2:
                break
        else:
            good = 0
        if len(hist) > 15 and perr < 3 * tol and oerr < 3 * ori_tol_deg:
            if float(np.linalg.norm(hist[-1] - hist[-16])) < 0.0005:
                stalled = True
                break
        robot.step(float(a[0]), float(a[1]), float(a[2]),
                   float(w[0]), float(w[1]), float(w[2]), float(grip))
        steps += 1
    f = robot.now()
    p, Rc = _pose(f)
    return {"steps": steps, "stalled": stalled, "pos_err": float(np.linalg.norm(target - p)),
            "ori_err_deg": oerr, "tcp": [float(v) for v in p],
            "gap": _finger_gap(f)}


def act_rotate(robot, ori, grip=1.0, max_steps=120, kr=2.5, ori_tol_deg=2.0, vmax=0.8, settle=0):
    """Rotate the gripper to named orientation ori with zero translation
    command (the TCP is allowed to swing as in the expert pours), gripper
    command grip. Stops at ori_tol_deg or max_steps. Returns steps, final
    orientation error (deg), tcp and finger gap. settle = extra zero-motion
    steps after reaching the orientation."""
    Rt = _ori_matrix(ori)
    steps = 0
    oerr = 180.0
    while steps < int(max_steps):
        p, Rc = _pose(robot.now())
        w = _log_rot(Rt @ Rc.T)
        oerr = math.degrees(float(np.linalg.norm(w)))
        if oerr < ori_tol_deg:
            break
        w = np.clip(kr * w, -vmax, vmax)
        robot.step(0.0, 0.0, 0.0, float(w[0]), float(w[1]), float(w[2]), float(grip))
        steps += 1
    for _ in range(int(settle)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
    f = robot.now()
    p, Rc = _pose(f)
    return {"steps": steps, "ori_err_deg": oerr, "tcp": [float(v) for v in p],
            "gap": _finger_gap(f)}


def act_hold(robot, steps=25, grip=-1.0):
    """Send zero motion for `steps` native steps with gripper command grip
    (+1 close, -1 open); used to close/open the fingers or to hold a pour.
    Returns the finger gap (m) and tcp afterwards."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
    f = robot.now()
    p, _ = _pose(f)
    return {"steps": int(steps), "gap": _finger_gap(f), "tcp": [float(v) for v in p]}
