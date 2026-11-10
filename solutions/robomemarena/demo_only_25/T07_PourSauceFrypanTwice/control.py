"""Closed-loop Cartesian servo for the OSC_POSE delta controller (world frame).

Plant model fitted on the 25 demos (scratch/plant_fit): per step
dp[t+1] = 0.915 dp[t] + 0.0053 a[t] - 0.0040 a[t-1]  (~1.5 cm/step at |a|=1),
w[t+1]  = 0.751 w[t]  + 0.0337 a[t] - 0.0116 a[t-1]  (~0.09 rad/step at |a|=1).
A PD law on measured TCP pose gives small overshoot on that model.
"""

KP_POS = 6.0
KD_POS = 25.0
KP_ROT = 1.6
KD_ROT = 2.0


def _rv2m(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _m2rv(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def _target_rot(tilt_deg):
    """Gripper pointing down (expert start orientation), tilted about world y so the
    gripper z-axis leans toward +x by tilt_deg (expert pour: 45 deg)."""
    a = -math.radians(float(tilt_deg))
    Ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    return Ry @ np.diag([1.0, -1.0, -1.0])


def _tcp(frame):
    t = np.asarray(frame.tcp, dtype=float)
    return t[:3].copy(), _rv2m(t[3:6])


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _servo(robot, target, Rt, grip, max_steps, min_steps, pos_tol, rot_tol_deg, vmax, rmax):
    target = np.asarray(target, dtype=float)
    prev_p, prev_R = _tcp(robot.now())
    steps = 0
    conv_at = -1
    perr = 1.0
    rerr = 180.0
    vel = np.zeros(3)
    wvel = np.zeros(3)
    while steps < max_steps:
        p, R = _tcp(robot.now())
        if steps > 0:
            vel = p - prev_p
            wvel = _m2rv(R @ prev_R.T)
        prev_p, prev_R = p, R
        e = target - p
        er = _m2rv(Rt @ R.T)
        perr = float(np.linalg.norm(e))
        rerr = math.degrees(float(np.linalg.norm(er)))
        if perr < pos_tol and rerr < rot_tol_deg and conv_at < 0:
            conv_at = steps
        if conv_at >= 0 and steps >= min_steps:
            break
        a = KP_POS * e - KD_POS * vel
        n = float(np.linalg.norm(a))
        if n > vmax:
            a = a * (vmax / n)
        r = KP_ROT * er - KD_ROT * wvel
        nr = float(np.linalg.norm(r))
        if nr > rmax:
            r = r * (rmax / nr)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
        steps += 1
    p, R = _tcp(robot.now())
    return {
        "steps": steps,
        "converged": conv_at >= 0,
        "pos_err": round(float(np.linalg.norm(target - p)), 4),
        "rot_err_deg": round(math.degrees(float(np.linalg.norm(_m2rv(Rt @ R.T)))), 2),
        "tcp": [round(float(v), 4) for v in p],
        "finger_gap": round(_gap(robot.now()), 4),
    }


def act_move(robot, x, y, z, tilt_deg=0.0, gripper=-1.0, max_steps=300, min_steps=0,
             pos_tol=0.004, rot_tol_deg=2.5, vmax=1.0, rmax=0.6):
    """Servo the TCP to world (x, y, z) with the gripper pointing down, tilted by tilt_deg
    toward +x about world y (45 = pour). Gripper sign held every step (+1 close, -1 open).
    Runs until within tolerance and at least min_steps (holding the pose), or max_steps.
    Returns steps, converged, pos_err, rot_err_deg, tcp and finger_gap."""
    max_steps = int(max(1, min(int(max_steps), 2400)))
    min_steps = int(max(0, min(int(min_steps), max_steps)))
    return _servo(robot, [x, y, z], _target_rot(tilt_deg), gripper, max_steps, min_steps,
                  float(pos_tol), float(rot_tol_deg), float(vmax), float(rmax))


def act_gripper(robot, close=True, steps=30):
    """Hold the current TCP pose (pointing-down orientation kept as is) while closing
    (close=True) or opening the gripper for `steps` steps. Returns finger_gap and tcp."""
    p, R = _tcp(robot.now())
    grip = 1.0 if close else -1.0
    steps = int(max(1, min(int(steps), 400)))
    out = _servo(robot, p, R, grip, steps, steps, 0.0, 0.0, 0.5, 0.3)
    out["closed"] = bool(close)
    return out


def act_idle(robot, steps=100, gripper=-1.0):
    """Hold the current pose for `steps` steps (used after the task sequence is finished)."""
    p, R = _tcp(robot.now())
    steps = int(max(1, min(int(steps), 2400)))
    return _servo(robot, p, R, gripper, steps, steps, 0.0, 0.0, 0.5, 0.3)
