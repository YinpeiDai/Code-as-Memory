"""Motion helpers: bounded proportional pd_ee_delta_pose steps with orientation hold."""

GAIN_XYZ = 1.2
GAIN_ROT = 0.5


def _rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if ang < 1e-6:
        return 0.5 * v
    if s < 1e-6:
        return np.zeros(3)
    return v * (ang / (2.0 * s))


def _rot_cmd(tcp, home_rpy):
    """Rotation command (drx, dry, drz) that turns the measured orientation toward home_rpy."""
    if not home_rpy:
        return [0.0, 0.0, 0.0]
    Rm = _rot_rpy(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rh = _rot_rpy(float(home_rpy[0]), float(home_rpy[1]), float(home_rpy[2]))
    w = _rotvec(Rh @ Rm.T)
    r = -GAIN_ROT * w / 0.1
    n = float(np.abs(r).max())
    if n > 1.0:
        r = r / n
    return [float(r[0]), float(r[1]), float(r[2])]


def _step_toward(robot, x, y, z, g, home_rpy, min_z_for_xy=None, dz_min=-1.0):
    tcp = np.asarray(robot.now().tcp, dtype=float)
    e = np.array([x, y, z]) - tcp[:3]
    if min_z_for_xy is not None and tcp[2] < min_z_for_xy:
        e[0] = 0.0
        e[1] = 0.0
    a = GAIN_XYZ * e / 0.1
    n = float(np.abs(a).max())
    if n > 1.0:
        a = a / n
    a[2] = max(a[2], dz_min)
    r = _rot_cmd(tcp, home_rpy)
    robot.step(float(a[0]), float(a[1]), float(a[2]), r[0], r[1], r[2], g)
    return e


def _goto(robot, x, y, z, g, home_rpy, xy_tol=0.01, z_tol=0.01, max_steps=40, min_z_for_xy=None,
          settle=False, stall=False, dz_min=-1.0):
    """Closed-loop move; stops when within tolerance (and slow if settle), or z stalls (stall)."""
    steps = 0
    prev = np.asarray(robot.now().tcp, dtype=float)[:3]
    status = "max_steps"
    for _ in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        e = np.array([x, y, z]) - tcp[:3]
        v = float(np.linalg.norm(tcp[:3] - prev))
        if math.hypot(e[0], e[1]) < xy_tol and abs(e[2]) < z_tol and (not settle or v < 0.003):
            status = "reached"
            break
        if stall and steps >= 3 and abs(tcp[2] - prev[2]) < 0.0015 and e[2] < 0 and tcp[2] < z + 0.03:
            status = "stalled"
            break
        prev = tcp[:3]
        _step_toward(robot, x, y, z, g, home_rpy, min_z_for_xy=min_z_for_xy, dz_min=dz_min)
        steps += 1
    return {"steps": steps, "status": status}
