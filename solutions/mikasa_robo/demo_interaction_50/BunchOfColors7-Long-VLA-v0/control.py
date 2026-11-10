"""Motion helpers: closed-loop Cartesian P-control with orientation hold."""

_K_POS = 1.0        # normalized command = K * error / 0.1 m (offline model: fast, ~no overshoot)
_K_ROT = 0.5        # fraction of the orientation error corrected per step
_ROT_CLIP = 0.3


def _rot_from_rpy(r, p, y):
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
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return w * (ang / (2.0 * s))


def _rot_cmd(tcp, rpy_ref):
    """Rotation arguments that turn the measured orientation toward rpy_ref (world-frame error)."""
    R = _rot_from_rpy(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    R0 = _rot_from_rpy(rpy_ref[0], rpy_ref[1], rpy_ref[2])
    d = _rotvec(R0 @ R.T)            # desired world-frame rotation increment
    r = -_K_ROT * d / 0.1            # native: signed angular offset is -0.1 * r
    return np.clip(r, -_ROT_CLIP, _ROT_CLIP), float(np.linalg.norm(d))


def _pos_cmd(tcp, target, vmax=1.0, k=None):
    k = _K_POS if k is None else k
    err = np.asarray(target, dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64)
    u = k * err / 0.1
    n = float(np.linalg.norm(u))
    if n > vmax:
        u = u * (vmax / n)
    return u, err


def _step_toward(robot, target, rpy_ref, grip, vmax=1.0):
    tcp = robot.now().tcp
    u, err = _pos_cmd(tcp, target, vmax)
    r, rerr = _rot_cmd(tcp, rpy_ref)
    robot.step(float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return robot.now().tcp
