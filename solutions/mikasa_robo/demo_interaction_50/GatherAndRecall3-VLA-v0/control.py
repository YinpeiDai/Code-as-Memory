"""Motion helpers (closed loop on measured TCP) for GatherAndRecall3."""

POS_GAIN = 14.0      # action units per metre of error (0.37 m/m per-step response measured in demos)
ROT_GAIN = 8.0       # action units per radian of orientation error
CARRY_Z = 0.12       # transport height of the TCP
GRASP_Z = 0.022      # TCP height for closing on a 4 cm cube (demo grasps 0.018-0.044)
RELEASE_Z = 0.032    # TCP height for opening over the disc (demo 0.03-0.046)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if ang < 1e-6:
        return 0.5 * v
    if s < 1e-4:
        # near pi: use diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return axis * ang
    return v * (ang / (2.0 * s))


def _rot_cmd(tcp, yaw_target):
    """Normalized rotation command that drives the TCP to roll=pi, pitch=0, yaw=yaw_target."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(math.pi, 0.0, yaw_target)
    w = _rotvec(Rt @ Rc.T)
    r = -ROT_GAIN * w
    n = float(np.linalg.norm(r))
    if n > 0.6:
        r = r * (0.6 / n)
    return r, float(np.linalg.norm(w))


def _goto(robot, target, grip, tol=0.005, max_steps=40, vmax=1.0, settle=0.0, yaw=0.0):
    """Drive the TCP to `target` (xyz) holding gripper command `grip`.

    Stops when within `tol` (and, if settle>0, when the last per-step TCP motion < settle).
    Returns dict with steps used, final error and whether it converged."""
    target = np.asarray(target, dtype=np.float64)
    prev = None
    steps = 0
    err_n = 1.0
    for i in range(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=np.float64)
        p = tcp[:3]
        err = target - p
        err_n = float(np.linalg.norm(err))
        vel = 1.0 if prev is None else float(np.linalg.norm(p - prev))
        if err_n < tol and (settle <= 0.0 or vel < settle):
            return {"ok": True, "steps": steps, "err": round(err_n, 4)}
        a = POS_GAIN * err
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        r, _ = _rot_cmd(tcp, yaw)
        prev = p
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
    return {"ok": False, "steps": steps, "err": round(err_n, 4)}


def _gripper(robot, grip, max_steps=10, min_steps=3):
    """Hold position while the fingers move; stop once they stop changing."""
    f0 = robot.now()
    hold = np.asarray(f0.tcp, dtype=np.float64)[:3]
    last = float(np.sum(f0.fingers))
    steps = 0
    for i in range(max_steps):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        a = POS_GAIN * (hold - tcp[:3])
        a = np.clip(a, -0.5, 0.5)
        r, _ = _rot_cmd(tcp, 0.0)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
        cur = float(np.sum(robot.now().fingers))
        if steps >= min_steps and abs(cur - last) < 0.0008:
            break
        last = cur
    return {"steps": steps, "fingers": round(float(np.sum(robot.now().fingers)), 4)}
