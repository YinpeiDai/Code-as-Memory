"""Closed-loop TCP servo helpers (pd_ee_delta_pose, 20 Hz). Helpers only."""

# Demo regression (10 episodes): per-step TCP displacement ~= 0.025*a[t] + 0.012*a[t-1] m,
# i.e. ~0.037 m/step at |a| = 1; so a = error/0.1 ("aim at the goal") converges without
# oscillation in the linear model.
POS_GAIN = 1.0
ROT_GAIN = 0.3
ROT_MAX = 0.3
TARGET_RPY = (math.pi, 0.0, 0.0)


def _rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, float(c)))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        # near pi: take the axis from the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return d * ang
    return v / (2.0 * s) * ang


def _rot_error(tcp):
    """World-frame rotation vector taking the measured orientation to TARGET_RPY."""
    Rm = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(TARGET_RPY[0], TARGET_RPY[1], TARGET_RPY[2])
    return _rotvec(Rt @ Rm.T)


def _command(tcp, target, vmax=1.0, gain=POS_GAIN):
    """Six normalised arm values steering the TCP towards `target` (xyz) while holding
    the gripper-down orientation. Native: new orientation = Rot(-0.1*r) * measured."""
    tcp = np.asarray(tcp, dtype=float)
    e = np.asarray(target, dtype=float) - tcp[:3]
    a = e * gain / 0.1
    n = float(np.max(np.abs(a)))
    if n > vmax:
        a = a * (vmax / n)
    w = _rot_error(tcp)
    r = -ROT_GAIN * w / 0.1
    rn = float(np.linalg.norm(r))
    if rn > ROT_MAX:
        r = r * (ROT_MAX / rn)
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])]


def _tcp(robot):
    return np.asarray(robot.now().tcp, dtype=float)


def _servo(robot, target, grip, stop, max_steps, vmax=1.0, log=None):
    """Step towards `target` until stop(tcp, prev_tcp, k) is True or max_steps.
    Returns (steps_taken, reached)."""
    prev = _tcp(robot)
    for k in range(max_steps):
        cur = _tcp(robot)
        if stop(cur, prev, k):
            return k, True
        cmd = _command(cur, target, vmax)
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
        prev = cur
        if log is not None:
            log.append(cmd)
    cur = _tcp(robot)
    return max_steps, bool(stop(cur, prev, max_steps))


def _hold(robot, target, grip, n):
    for k in range(n):
        cmd = _command(_tcp(robot), target, 0.5)
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)


def _r(v, d=4):
    return [round(float(x), d) for x in v]
