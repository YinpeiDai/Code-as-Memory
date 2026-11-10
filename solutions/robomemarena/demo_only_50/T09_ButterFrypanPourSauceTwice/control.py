"""Low-level closed-loop control helpers (world-frame OSC_POSE delta steps)."""

POS_GAIN = 12.0      # action per metre of error (1 unit ~ 1.2 cm/step)
ROT_GAIN = 2.0       # action per radian of error (1 unit ~ 0.075 rad/step)
ROT_CAP = 0.8        # expert never exceeded 0.8 on rotation


def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _R2aa(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _R_upright():
    """Gripper pointing straight down, fingers along world y (expert grasp orientation)."""
    return _aa2R([math.pi, 0.0, 0.0])


def _R_tilted(deg):
    """Upright orientation rotated by -deg about world y (the expert pour: 45 deg)."""
    a = -math.radians(deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return Ry @ _R_upright()


def _tilt_deg(frame):
    R = _aa2R(np.asarray(frame.tcp)[3:6])
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 2])))))


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _step_towards(robot, target, Rt, grip, vmax, hold_pos=True):
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    a = np.zeros(3)
    if hold_pos and target is not None:
        a = POS_GAIN * (np.asarray(target, dtype=float) - tcp[:3])
        n = float(np.max(np.abs(a)))
        if n > vmax:
            a = a * (vmax / n)
    r = np.zeros(3)
    if Rt is not None:
        e = _R2aa(Rt @ _aa2R(tcp[3:6]).T)
        r = ROT_GAIN * e
        n = float(np.max(np.abs(r)))
        if n > ROT_CAP:
            r = r * (ROT_CAP / n)
    return robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))


def _errors(frame, target, Rt):
    tcp = np.asarray(frame.tcp, dtype=float)
    pe = 0.0 if target is None else float(np.linalg.norm(np.asarray(target, dtype=float) - tcp[:3]))
    re = 0.0 if Rt is None else float(np.linalg.norm(_R2aa(Rt @ _aa2R(tcp[3:6]).T)))
    return pe, re


def _servo(robot, target, Rt, grip, vmax=1.0, tol=0.004, rtol=0.03, max_steps=200, settle=2, hold_pos=True):
    """Drive to a pose; returns (steps used, final position error, final rotation error)."""
    n = 0
    good = 0
    pe, re = _errors(robot.now(), target if hold_pos else None, Rt)
    while n < max_steps:
        fr = _step_towards(robot, target, Rt, grip, vmax, hold_pos)
        n += 1
        pe, re = _errors(fr, target if hold_pos else None, Rt)
        if pe < tol and re < rtol:
            good += 1
            if good >= settle:
                break
        else:
            good = 0
    return n, pe, re


def _hold(robot, target, Rt, grip, steps, vmax=0.5):
    for _ in range(int(steps)):
        _step_towards(robot, target, Rt, grip, vmax, target is not None)
    return int(steps)


def _summary(robot):
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {'tcp': [round(float(v), 4) for v in tcp[:3]], 'width': round(_width(fr), 4),
            'tilt_deg': round(_tilt_deg(fr), 1), 'step': int(fr.step)}
