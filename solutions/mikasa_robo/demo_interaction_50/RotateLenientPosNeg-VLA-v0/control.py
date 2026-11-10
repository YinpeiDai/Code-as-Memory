"""Motion helpers for pd_ee_delta_pose (CPU IK): translation 0.1*d m per unit,
rotation offset -0.1*r rad left-multiplied in the base (world) frame."""

Z_PUSH = 0.030      # TCP height while pushing (finger tips ~1 cm lower)
Z_SAFE = 0.105      # travel height above the peg (top at 0.05)
R_PUSHER = 0.015    # half-size of the closed finger block (assumed)
LEVER = 0.080       # contact distance from the peg centre along its axis
GAP = 0.020         # clearance between pusher and peg face before pushing


def _rpy_mat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(M):
    c = (np.trace(M) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        # near pi: use the diagonal
        ax = np.sqrt(np.maximum((np.diag(M) + 1.0) / 2.0, 0.0))
        return ax * ang
    return ang / (2.0 * s) * v


def _rot_cmd(tcp, yaw_keep, gain=0.6):
    """Normalised rotation command holding a downward gripper at yaw_keep."""
    Rc = _rpy_mat(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_mat(math.pi, 0.0, yaw_keep)
    w = _rotvec(Rd @ Rc.T)          # world-frame rotation from current to desired
    r = -gain * w / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _trans_cmd(tcp, target, gain=1.0, vmax=1.0):
    e = np.asarray(target, dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64)
    u = gain * e / 0.1
    n = float(np.linalg.norm(u))
    if n > vmax:
        u = u * (vmax / n)
    return u, float(np.linalg.norm(e))


def _qvel(robot):
    """Max |joint change| per second over the last native step (approximate)."""
    fr = robot.frames()
    if len(fr) >= 2:
        return float(np.max(np.abs(np.asarray(fr[-1].joints) - np.asarray(fr[-2].joints)))) * 20.0
    return None
