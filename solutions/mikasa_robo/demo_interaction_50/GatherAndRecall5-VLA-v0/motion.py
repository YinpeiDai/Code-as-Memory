"""Closed-loop motion helpers (pd_ee_delta_pose, CPU IK) for GatherAndRecall5."""

TRANSIT_Z = 0.10      # carry / travel height of the TCP (m)
GRASP_Z = 0.028       # TCP height for grasping a 4 cm cube resting on the table
RELEASE_DZ = 0.036    # TCP height above the disc top when releasing
GRASP_OFFSET = [-0.010, 0.0045]   # TCP - cube-centre offset used by every expert grasp (50 demos)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:
        # near pi: use diagonal
        w = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(w))
        axis = R[:, k] + np.eye(3)[k]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        return axis * ang
    return v / (2.0 * s) * ang


def _rot_cmd(tcp, yaw, gain=0.6):
    """Normalized rotation command that turns the measured TCP toward Rz(yaw) Rx(pi) (gripper down)."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(math.pi, 0.0, yaw)
    w = _rotvec(Rt @ Rc.T)
    cmd = -gain * w / 0.1
    n = np.linalg.norm(cmd)
    if n > 1.0:
        cmd = cmd / n
    return cmd, float(np.linalg.norm(w))


def _move(robot, target, yaw, grip, tol=0.006, vtol=0.004, max_steps=60, gain=1.0, damp=0.5,
          min_steps=0):
    """Drive the TCP to target xyz (world) with gripper-down orientation; returns a measurement dict."""
    target = np.asarray(target, dtype=np.float64)
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    n = 0
    err_n = 1.0
    vel = np.zeros(3)
    for n in range(1, max_steps + 1):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        pos = tcp[:3]
        err = target - pos
        err_n = float(np.linalg.norm(err))
        if n > 1:
            vel = pos - prev
        if n > min_steps and err_n < tol and float(np.linalg.norm(vel)) < vtol:
            n -= 1
            break
        cmd = gain * (err - damp * vel) / 0.1
        m = float(np.max(np.abs(cmd)))
        if m > 1.0:
            cmd = cmd / m
        rc, rerr = _rot_cmd(tcp, yaw)
        prev = pos
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), grip)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return {"steps": n, "err": round(float(np.linalg.norm(target - tcp[:3])), 4),
            "pos": [round(float(v), 4) for v in tcp[:3]]}


def _gripper(robot, grip, yaw, n_min=3, n_max=10):
    """Hold the pose while the fingers move; stop when they settle."""
    hold = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    last = float(np.mean(robot.now().fingers))
    k = 0
    for k in range(1, n_max + 1):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        err = hold - tcp[:3]
        cmd = np.clip(err / 0.1, -1, 1)
        rc, _ = _rot_cmd(tcp, yaw)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), grip)
        f = float(np.mean(robot.now().fingers))
        if k >= n_min and abs(f - last) < 0.0007:
            break
        last = f
    return {"steps": k, "fingers": round(float(np.mean(robot.now().fingers)), 4)}
