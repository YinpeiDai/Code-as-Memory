"""Closed-loop OSC_POSE control helpers (world-frame delta steps)."""

HIGH_Z = 0.70


def _rotvec_to_R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _R_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(th)) * th


def _R_des(yaw_deg):
    """Top-down gripper rotated by yaw about world z (yaw 0 = the reset orientation)."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _rot_err(frame, yaw_deg):
    R = _rotvec_to_R(np.asarray(frame.tcp)[3:6])
    return _R_to_rotvec(_R_des(yaw_deg) @ R.T)


def _goto(robot, target, yaw_deg, grip, tol=0.006, rtol=0.05, max_steps=250, kp=1.2, vmax=1.0):
    """Drive the TCP to `target` xyz with orientation `yaw_deg`; returns (steps, pos_err, rot_err)."""
    tgt = np.asarray(target, dtype=float)
    n = 0
    f = robot.now()
    hist = []
    while n < max_steps:
        p = np.asarray(f.tcp)[:3]
        e = tgt - p
        er = _rot_err(f, yaw_deg)
        en = float(np.linalg.norm(e))
        if en < tol and float(np.linalg.norm(er)) < rtol:
            break
        hist.append(p.copy())
        # stalled close to the target (controller steady-state offset or contact): stop
        if len(hist) > 12 and en < 0.015 and float(np.linalg.norm(hist[-1] - hist[-11])) < 0.001:
            break
        a = np.clip(e / 0.05 * kp, -vmax, vmax)
        r = np.clip(er * 2.5, -0.5, 0.5)
        f = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    p = np.asarray(f.tcp)[:3]
    return n, float(np.linalg.norm(tgt - p)), float(np.linalg.norm(_rot_err(f, yaw_deg)))


def _hold(robot, steps, yaw_deg, grip):
    """Hold the current position (servo to where we are) for `steps` steps."""
    f = robot.now()
    tgt = np.asarray(f.tcp)[:3].copy()
    for _ in range(int(steps)):
        e = tgt - np.asarray(f.tcp)[:3]
        a = np.clip(e / 0.05 * 1.2, -1, 1)
        r = np.clip(_rot_err(f, yaw_deg) * 2.5, -0.5, 0.5)
        f = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return f


def _gap(frame):
    fg = np.asarray(frame.fingers)
    return float(fg[0] - fg[1])
