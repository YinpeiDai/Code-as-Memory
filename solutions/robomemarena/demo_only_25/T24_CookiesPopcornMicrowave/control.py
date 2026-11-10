"""Low-level control: rotation maths and a waypoint follower reproducing the expert's P-controller."""

R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]   # fingers pointing down (top grasp)
R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]   # fingers pointing +y (into the microwave)
POS_GAIN = 6.0     # expert: a_pos = clip(6 * (target - tcp), -1, 1)
ROT_GAIN = 2.0     # expert: a_rot = clip(2 * rotvec(R_target R_cur^T), -0.5, 0.5)
ROT_CLIP = 0.5


def _rotmat(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _rotvec(Rm):
    Rm = np.asarray(Rm, dtype=float)
    c = (np.trace(Rm) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(th)
    if s > 1e-4:
        return w / (2.0 * s) * th
    # angle close to pi: axis from the symmetric part
    B = (Rm + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(B)))
    ax = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
    if np.dot(ax, w) < 0:
        ax = -ax
    return ax * th


def _rot_error(tcp, name):
    Rt = np.array(R_SIDE if name == "side" else R_DOWN)
    Rc = _rotmat(np.asarray(tcp, dtype=float)[3:6])
    return _rotvec(Rt @ Rc.T)


def _command(tcp, target, rot, grip, gain=POS_GAIN):
    tcp = np.asarray(tcp, dtype=float)
    a = [0.0] * 7
    if target is not None:
        d = (np.asarray(target, dtype=float) - tcp[:3]) * gain
        for i in range(3):
            a[i] = float(min(1.0, max(-1.0, d[i])))
    if rot is not None:
        e = _rot_error(tcp, rot) * ROT_GAIN
        for i in range(3):
            a[3 + i] = float(min(ROT_CLIP, max(-ROT_CLIP, e[i])))
    a[6] = 1.0 if grip > 0 else -1.0
    return a


def act_run(robot, segments):
    """Execute timed waypoint segments with the expert-style P-controller.

    segments: list of {"target": [x,y,z] or null (hold still), "rot": "down"|"side"|null,
    "grip": +1 close / -1 open, "steps": int, "name": str}. Position action = clip(6*(target-tcp)),
    rotation action = clip(2*world rotation error, +-0.5). Returns per-segment end tcp, finger gap and
    position error, plus the final state.
    """
    report = []
    for seg in segments:
        target = seg.get("target")
        rot = seg.get("rot")
        grip = seg.get("grip", -1)
        n = int(seg.get("steps", 1))
        for _ in range(n):
            f = robot.now()
            a = _command(f.tcp, target, rot, grip)
            robot.step(*a)
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        fg = np.asarray(f.fingers, dtype=float)
        err = None
        if target is not None:
            err = float(np.linalg.norm(np.asarray(target, dtype=float) - tcp[:3]))
        rerr = None
        if rot is not None:
            rerr = float(np.linalg.norm(_rot_error(tcp, rot)))
        report.append({"name": seg.get("name", ""), "tcp": [round(float(v), 4) for v in tcp[:3]],
                       "gap": round(float(fg[0] - fg[1]), 4), "pos_err": err, "rot_err": rerr})
    f = robot.now()
    fg = np.asarray(f.fingers, dtype=float)
    return {"segments": report, "tcp": [float(v) for v in np.asarray(f.tcp, dtype=float)],
            "gap": float(fg[0] - fg[1]), "step": int(f.step)}


def act_hold(robot, steps=50, grip=-1):
    """Hold the arm still for `steps` native steps (gripper +1 close / -1 open); used while the checker settles."""
    g = 1.0 if grip > 0 else -1.0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
    f = robot.now()
    fg = np.asarray(f.fingers, dtype=float)
    return {"tcp": [float(v) for v in np.asarray(f.tcp, dtype=float)], "gap": float(fg[0] - fg[1]),
            "step": int(f.step)}
