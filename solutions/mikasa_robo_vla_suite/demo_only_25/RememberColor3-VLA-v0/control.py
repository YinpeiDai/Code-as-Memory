"""Motion helpers: model-based per-axis approach controller and orientation hold.

Plant model fitted on all 25 demos (scratch/dyn.py, scratch/rotdyn.py):
  per-step TCP displacement d[t+1] ~= 0.357*(0.1*a[t]) + 0.359*d[t] - 0.096*(0.1*a[t-1])
  (|a|=1 sustained gives ~0.04 m/step);
  world rotation-vector change ~= 0.40*(-0.1*r[t]).
"""

_KA = 0.0357      # m of displacement per unit command (this step)
_KV = 0.359       # carried-over fraction of last displacement
_KP = 0.0096      # lagged command term (m per unit command of the previous step)
_VMAX = 0.042     # largest desired per-step displacement (m)
_GAIN = 0.7       # fraction of remaining error to cover per step
_ROT_GAIN = 6.0   # rotation command per rad of orientation error


def _rpy_matrix(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _rotvec(M):
    c = max(-1.0, min(1.0, (float(np.trace(M)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        # near pi: fall back to the dominant diagonal axis
        d = np.diag(M)
        k = int(np.argmax(d))
        axis = np.zeros(3)
        axis[k] = 1.0
        return axis * th
    return th / (2.0 * s) * v


def _orientation_error(tcp_rpy, yaw_goal):
    """World rotation vector taking the measured TCP orientation to the top-down goal (roll=pi, pitch=0)."""
    Rm = _rpy_matrix(tcp_rpy)
    Rd = _rpy_matrix([math.pi, 0.0, yaw_goal])
    return _rotvec(Rd @ Rm.T)


def _axis_command(err, d_prev, a_prev):
    """Command for one axis so the predicted displacement covers _GAIN of the error (bounded)."""
    d_des = max(-_VMAX, min(_VMAX, _GAIN * err))
    a = (d_des - _KV * d_prev + _KP * a_prev) / _KA
    return max(-1.0, min(1.0, a))


def _command(tcp, d_prev, a_prev, goal, yaw_goal):
    """Six arm arguments for robot.step toward goal xyz with a top-down orientation hold."""
    a = [0.0] * 6
    for i in range(3):
        a[i] = _axis_command(float(goal[i]) - float(tcp[i]), float(d_prev[i]), float(a_prev[i]))
    w = _orientation_error(tcp[3:6], yaw_goal)
    r = -_ROT_GAIN * w
    n = float(np.linalg.norm(r))
    if n > 0.5:
        r = r * (0.5 / n)
    a[3], a[4], a[5] = float(r[0]), float(r[1]), float(r[2])
    return a


def _snap_yaw(yaw, cube_yaw=0.0):
    """Nearest gripper yaw equivalent (mod 90 deg) to cube_yaw, close to the current yaw."""
    q = math.pi / 2.0
    k = round((yaw - cube_yaw) / q)
    return cube_yaw + k * q
