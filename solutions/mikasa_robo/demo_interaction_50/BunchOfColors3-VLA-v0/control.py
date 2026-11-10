"""Motion helpers: closed-loop Cartesian servo on measured TCP with orientation hold."""

GAIN = 0.8        # fraction of the remaining error requested per step (in 0.1 m command units)
DAMP = 0.6        # velocity damping (per-step TCP displacement, command units)
ROT_GAIN = 0.4    # fraction of orientation error requested per step
AMAX = 1.0        # max translation command norm


def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the diagonal
        w = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return w * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_cmd(tcp, rpy):
    """Rotation command (drx, dry, drz) that moves the measured orientation toward rpy.
    Native: angular offset = -0.1*r about base axes, left-multiplied."""
    Rc = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy2R(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    e = _logR(Rt @ Rc.T)
    r = -ROT_GAIN * e / 0.1
    n = float(np.linalg.norm(r))
    if n > 0.5:
        r = r * 0.5 / n
    return r, float(np.linalg.norm(e))


def _servo(robot, target, rpy, grip, tol=0.006, vtol=0.004, max_steps=60, amax=AMAX, min_steps=0):
    """Drive TCP to target xyz holding orientation rpy. Returns a summary dict."""
    target = np.asarray(target, dtype=np.float64)
    f = robot.now()
    prev = np.asarray(f.tcp[:3], dtype=np.float64)
    n = 0
    err = target - prev
    rot_err = 0.0
    vmax = 0.0
    while n < max_steps:
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=np.float64)
        v = p - prev
        prev = p
        err = target - p
        speed = float(np.linalg.norm(v))
        if n >= min_steps and float(np.linalg.norm(err)) < tol and speed < vtol:
            break
        a = GAIN * err / 0.1 - DAMP * v / 0.1
        na = float(np.linalg.norm(a))
        if na > amax:
            a = a * amax / na
        r, rot_err = _rot_cmd(f.tcp, rpy)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
        n += 1
        vmax = max(vmax, speed)
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=np.float64)
    return {"steps": n, "final": [round(float(x), 4) for x in p],
            "err": round(float(np.linalg.norm(target - p)), 4), "rot_err": round(rot_err, 4),
            "vmax": round(vmax, 4)}


def _hold(robot, rpy, grip, steps):
    """Hold position (zero translation, orientation hold) for a number of steps; returns z trace."""
    zs = []
    for _ in range(int(steps)):
        f = robot.now()
        r, _ = _rot_cmd(f.tcp, rpy)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), float(grip))
        zs.append(round(float(robot.now().tcp[2]), 4))
    return zs
