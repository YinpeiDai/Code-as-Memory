"""Motion helpers: closed-loop Cartesian servo on measured TCP with orientation hold."""


def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if ang < 1e-6:
        return 0.5 * v
    if s < 1e-6:
        return np.zeros(3)
    return v * (ang / (2.0 * s))


def _servo_cmd(tcp, goal, R_des, vmax, kp=1.0, krot=0.4):
    """One step command toward `goal` (world xyz). vmax caps the commanded offset (m)."""
    p = np.asarray(tcp[:3], dtype=float)
    err = np.asarray(goal, dtype=float) - p
    off = kp * err
    n = float(np.linalg.norm(off))
    if n > vmax:
        off = off * (vmax / n)
    a = np.clip(off / 0.1, -1.0, 1.0)
    Rm = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _rotvec(R_des @ Rm.T)          # world-frame rotation still needed
    r = np.clip(-krot * w / 0.1, -0.3, 0.3)   # target = R(-0.1 r) * R_meas
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])], err, float(np.linalg.norm(w))
