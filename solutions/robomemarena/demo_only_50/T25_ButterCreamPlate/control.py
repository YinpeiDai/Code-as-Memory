"""Closed-loop end-effector motion helpers (world-frame OSC_POSE deltas)."""

GRASP_ROTVEC = [2.902, -1.202, 0.003]   # expert grasp orientation (gripper yawed -45 deg), all 100 demo grasps
PLACE_ROTVEC = [3.1416, 0.0, 0.0]       # expert release orientation (gripper straight down)
POS_GAIN = 15.0    # action units per metre of error (1.0 -> ~1.3 cm/step measured in demos)
VEL_DAMP = 20.0   # action units per (metre/step) of measured velocity (damping)
ROT_GAIN = 2.0     # action units per radian of error (1.0 -> ~0.1 rad/step measured in demos)


def _rotmat(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _errors(frame, target, rot_target):
    p = np.asarray(frame.tcp[:3], float)
    e = np.asarray(target, float) - p
    Rc = _rotmat(frame.tcp[3:6])
    Rt = _rotmat(rot_target)
    er = _rotvec(Rt @ Rc.T)
    return p, e, er


def _move(robot, target, rot_target, grip, pos_tol=0.004, rot_tol=0.04, max_steps=200,
          vmax=1.0, stall_steps=10, stall_err=0.03):
    """Servo the TCP to target xyz / orientation with a PD law on measured position.
    Stops when within tolerance, after max_steps, or when blocked (speed < 0.4 mm/step for
    stall_steps steps while error < stall_err, e.g. fingertips on the table)."""
    slow = 0
    steps = 0
    stalled = False
    f = robot.now()
    p, e, er = _errors(f, target, rot_target)
    prev = p
    while steps < max_steps:
        en = float(np.linalg.norm(e))
        rn = float(np.linalg.norm(er))
        if en < pos_tol and rn < rot_tol:
            break
        v = p - prev
        if steps > 3 and float(np.linalg.norm(v)) < 0.0004:
            slow += 1
        else:
            slow = 0
        if slow >= stall_steps and en < stall_err and rn < rot_tol * 2:
            stalled = True
            break
        u = np.clip(POS_GAIN * e - VEL_DAMP * v, -vmax, vmax)
        ur = np.clip(ROT_GAIN * er, -1.0, 1.0)
        prev = p
        f = robot.step(float(u[0]), float(u[1]), float(u[2]), float(ur[0]), float(ur[1]), float(ur[2]), grip)
        steps += 1
        p, e, er = _errors(f, target, rot_target)
    return {"steps": steps, "pos_err": float(np.linalg.norm(e)), "rot_err": float(np.linalg.norm(er)),
            "stalled": stalled, "tcp": [float(v) for v in p]}


def _hold(robot, target, rot_target, grip, n):
    """Hold the pose for n steps while commanding the gripper."""
    f = robot.now()
    for _ in range(n):
        p, e, er = _errors(f, target, rot_target)
        u = np.clip(POS_GAIN * e, -0.3, 0.3)
        ur = np.clip(ROT_GAIN * er, -0.5, 0.5)
        f = robot.step(float(u[0]), float(u[1]), float(u[2]), float(ur[0]), float(ur[1]), float(ur[2]), grip)
    return f
