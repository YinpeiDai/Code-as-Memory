"""Motion helpers: closed-loop Cartesian servoing on measured TCP with orientation hold."""


def _rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    return Rz @ Ry @ Rx


def _log_R(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / math.sqrt(max(A[k, k], 1e-9))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_cmd(tcp, ref_rpy, gain=0.3, dead=0.015, cap=0.5):
    """Rotation arguments (drx, dry, drz) that turn the measured orientation toward ref_rpy.

    Native semantics: the applied world-frame offset is -0.1*r, so r = -gain*w/0.1 with
    w = log(R_ref R_cur^T)."""
    if ref_rpy is None:
        return [0.0, 0.0, 0.0]
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rr = _rpy_to_R(float(ref_rpy[0]), float(ref_rpy[1]), float(ref_rpy[2]))
    w = _log_R(Rr @ Rc.T)
    if float(np.linalg.norm(w)) < dead:
        return [0.0, 0.0, 0.0]
    r = -gain * w / 0.1
    return [float(min(cap, max(-cap, v))) for v in r]


def _servo(robot, target, ref_rpy, grip, kp=1.0, vmax=1.0):
    """One native step toward target xyz (closed loop on measured TCP)."""
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    err = np.asarray(target, dtype=np.float64) - tcp[:3]
    cmd = kp * err / 0.1
    n = float(np.linalg.norm(cmd))
    if n > vmax:
        cmd = cmd * (vmax / n)
    r = _rot_cmd(tcp, ref_rpy)
    robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), r[0], r[1], r[2], float(grip))
    return float(np.linalg.norm(err))


def _move_to(robot, target, ref_rpy, grip, tol=0.006, max_steps=40, kp=1.0, vmax=1.0,
             settle_v=0.004, stop_fn=None):
    """Servo to target until within tol and moving slower than settle_v per step."""
    prev = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    n = 0
    err = float(np.linalg.norm(np.asarray(target) - prev))
    while n < max_steps:
        cur = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
        err = float(np.linalg.norm(np.asarray(target) - cur))
        speed = float(np.linalg.norm(cur - prev)) if n > 0 else 1.0
        if err < tol and speed < settle_v:
            break
        if stop_fn is not None and stop_fn(robot.now()):
            break
        prev = cur
        _servo(robot, target, ref_rpy, grip, kp, vmax)
        n += 1
    cur = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    return {"steps": n, "err": float(np.linalg.norm(np.asarray(target) - cur)),
            "tcp": [round(float(v), 4) for v in cur]}


def _hold(robot, ref_rpy, grip, steps, target=None):
    """Hold position (servo to target or the current TCP) for a number of steps."""
    if target is None:
        target = [float(v) for v in robot.now().tcp[:3]]
    for _ in range(int(steps)):
        _servo(robot, target, ref_rpy, grip, 1.0, 0.5)
    return [float(v) for v in robot.now().fingers]


def _check_pose(robot, xyz, ref_rpy):
    """One live absolute dry-run of a target pose; returns a compact summary (no motion)."""
    if ref_rpy is None:
        ref_rpy = [float(v) for v in robot.now().tcp[3:6]]
    try:
        c = robot.dry_run(float(xyz[0]), float(xyz[1]), float(xyz[2]),
                          float(ref_rpy[0]), float(ref_rpy[1]), float(ref_rpy[2]), mode="absolute")
    except Exception as e:
        return {"error": str(e)[:120]}
    m = c.get("min_joint_margin_rad")
    return {"ok": bool(c.get("within_joint_limits")), "min_margin": None if m is None else round(float(m), 3)}
