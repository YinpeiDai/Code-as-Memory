"""Closed-loop end-effector control on measured tcp pose (world frame OSC deltas)."""

POS_GAIN = 10.0      # normalized action per metre of position error (1.0 = 5 cm target offset)
ROT_GAIN = 2.0       # normalized action per radian of orientation error (1.0 = 0.5 rad)
ROT_CLIP = 0.8


def _rotvec_to_mat(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _mat_to_rotvec(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the largest diagonal of (R + I) / 2
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        ax = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _target_R(tilt_deg, yaw_deg):
    """Gripper pointing down (rotvec [pi,0,0]) yawed about world z, then tilted about world +y."""
    t = math.radians(float(tilt_deg))
    y = math.radians(float(yaw_deg))
    Ry = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    Rd = np.diag([1.0, -1.0, -1.0])
    return Ry @ Rz @ Rd


def _pose_errors(tcp, target_pos, Rt):
    tcp = np.asarray(tcp, dtype=float)
    ep = np.asarray(target_pos, dtype=float) - tcp[:3]
    Rc = _rotvec_to_mat(tcp[3:6])
    er = _mat_to_rotvec(Rt @ Rc.T)
    return ep, er


def _control(tcp, target_pos, Rt, max_speed):
    ep, er = _pose_errors(tcp, target_pos, Rt)
    a = ep * POS_GAIN
    n = float(np.max(np.abs(a)))
    if n > max_speed:
        a = a * (max_speed / n)
    r = er * ROT_GAIN
    nr = float(np.max(np.abs(r)))
    if nr > ROT_CLIP:
        r = r * (ROT_CLIP / nr)
    return a, r, ep, er


def _tilt_of(tcp):
    """Gripper rotation relative to straight-down, as world rotvec in degrees."""
    Rc = _rotvec_to_mat(np.asarray(tcp, dtype=float)[3:6])
    return np.degrees(_mat_to_rotvec(Rc @ np.diag([1.0, -1.0, -1.0]).T))


def act_move(robot, x, y, z, tilt=0.0, yaw=0.0, grip=-1.0, tol=0.004, rot_tol_deg=2.0,
             max_steps=200, min_steps=0, max_speed=1.0, stall_ok=False):
    """Servo the measured tcp to world (x, y, z) with gripper orientation tilt/yaw (degrees about
    world +y / z from straight-down) while holding gripper command grip (+1 close, -1 open).
    Stops when within tol (m) and rot_tol_deg for 3 steps and at least min_steps elapsed, when
    stalled against contact (if stall_ok), or after max_steps. Returns the measured end state."""
    Rt = _target_R(tilt, yaw)
    tgt = [float(x), float(y), float(z)]
    g = 1.0 if grip > 0 else -1.0
    hist = []
    ok = 0
    reason = "max_steps"
    steps = 0
    ep = np.zeros(3)
    er = np.zeros(3)
    for i in range(int(max_steps)):
        cur = robot.now()
        tcp = np.asarray(cur.tcp, dtype=float)
        a, r, ep, er = _control(tcp, tgt, Rt, float(max_speed))
        perr = float(np.linalg.norm(ep))
        rerr = math.degrees(float(np.linalg.norm(er)))
        ok = ok + 1 if (perr < tol and rerr < rot_tol_deg) else 0
        hist.append(tcp[:3].copy())
        if i >= min_steps and ok >= 3:
            reason = "reached"
            break
        if stall_ok and i >= max(min_steps, 15) and len(hist) > 10:
            moved = float(np.linalg.norm(hist[-1] - hist[-11]))
            if moved < 0.0015 and perr < 0.02 and rerr < rot_tol_deg * 2:
                reason = "stalled"
                break
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
        steps += 1
    cur = robot.now()
    tcp = np.asarray(cur.tcp, dtype=float)
    ep, er = _pose_errors(tcp, tgt, Rt)
    return {"reason": reason, "steps": steps, "tcp": [round(float(v), 4) for v in tcp[:3]],
            "pos_err": round(float(np.linalg.norm(ep)), 4),
            "rot_err_deg": round(math.degrees(float(np.linalg.norm(er))), 2),
            "fingers": [round(float(v), 4) for v in np.asarray(cur.fingers, dtype=float)]}


def act_grip(robot, close=True, steps=25, tilt=0.0, yaw=0.0):
    """Close (close=True) or open the gripper for `steps` native steps while holding the current
    position and the given orientation. Returns finger positions."""
    Rt = _target_R(tilt, yaw)
    tgt = np.asarray(robot.now().tcp, dtype=float)[:3].copy()
    g = 1.0 if close else -1.0
    for _ in range(int(steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a, r, ep, er = _control(tcp, tgt, Rt, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
    cur = robot.now()
    return {"fingers": [round(float(v), 4) for v in np.asarray(cur.fingers, dtype=float)],
            "tcp": [round(float(v), 4) for v in np.asarray(cur.tcp, dtype=float)[:3]]}
