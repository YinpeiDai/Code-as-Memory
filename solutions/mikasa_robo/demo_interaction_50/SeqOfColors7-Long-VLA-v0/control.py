"""Motion helpers: rotation maths, closed-loop pose stepping."""

STEP_M = 0.1      # metres per unit translation command
STEP_RAD = 0.1    # radians per unit rotation command


def _rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    """Rotation vector (axis*angle) of a rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        # near pi: use the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * ang
    return w / (2.0 * s) * ang


def _rot_cmd(tcp, rpy_goal, gain=0.5, dead=0.01):
    """Normalised rotation command that turns the measured orientation towards rpy_goal.
    Demo regression shows the world rotation responds with the opposite sign of the command."""
    if rpy_goal is None:
        return [0.0, 0.0, 0.0], 0.0
    R = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    G = _rpy_to_R(float(rpy_goal[0]), float(rpy_goal[1]), float(rpy_goal[2]))
    w = _rotvec(G @ R.T)
    err = float(np.linalg.norm(w))
    if err < dead:
        return [0.0, 0.0, 0.0], err
    r = -gain * w / STEP_RAD
    n = float(np.linalg.norm(r))
    if n > 0.6:
        r = r * (0.6 / n)
    return [float(v) for v in r], err


def _trans_cmd(pos, target, vel, gain=1.0, kd=0.5, vmax=1.0):
    """P-D translation command (normalised), direction-preserving saturation."""
    e = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    a = gain * e / STEP_M - kd * np.asarray(vel, dtype=np.float64) / STEP_M
    n = float(np.max(np.abs(a)))
    if n > vmax:
        a = a * (vmax / n)
    return [float(v) for v in a]


def _state(robot, prev):
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=np.float64)
    pos = tcp[:3]
    vel = pos - prev if prev is not None else np.zeros(3)
    return f, tcp, pos, vel


def _step_to(robot, target, rpy_goal, grip, prev, gain=1.0, kd=0.5, vmax=1.0):
    """One control step towards an absolute xyz target. Returns new prev position."""
    f, tcp, pos, vel = _state(robot, prev)
    t = _trans_cmd(pos, target, vel, gain, kd, vmax)
    r, _ = _rot_cmd(tcp, rpy_goal)
    robot.step(t[0], t[1], t[2], r[0], r[1], r[2], grip)
    return pos
