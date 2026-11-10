"""Motion helpers: closed-loop pd_ee_delta_pose servoing on the measured TCP."""

ROBOT_BASE_XY = (-0.615, 0.0)  # Panda base in world, from FK(joints) vs measured tcp in demos
STEP_GAIN = 0.04  # measured steady-state TCP metres per step per unit action (demo regression)


def _rpy2R(rpy):
    r = float(rpy[0])
    p = float(rpy[1])
    y = float(rpy[2])
    cr = math.cos(r)
    sr = math.sin(r)
    cp = math.cos(p)
    sp = math.sin(p)
    cy = math.cos(y)
    sy = math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _R2rpy(R):
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    r = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(R[1, 0], R[0, 0])
    return [r, p, y]


def _logR(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-6:
        # near pi: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
    return w * th


def _axis_angle(axis, ang):
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _down(yaw=0.0):
    return _rpy2R([math.pi, 0.0, yaw])


def _cmd(robot, target, R_des, grip, kp=1.2, vmax=1.0, ff=None, kr=0.8):
    """One native step toward target xyz / orientation R_des. Returns the position error before the step."""
    f = robot.now()
    tcp = np.array(f.tcp, float)
    e = np.asarray(target, float) - tcp[:3]
    a = kp * e / 0.1
    if ff is not None:
        a = a + np.asarray(ff, float)
    m = float(np.max(np.abs(a)))
    if m > vmax:
        a = a * (vmax / m)
    w = _logR(R_des @ _rpy2R(tcp[3:]).T)
    r = -kr * w / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return e


def _move_to(robot, target, R_des, grip, tol=0.005, max_steps=150, vmax=1.0, settle=2, kp=1.2):
    """Servo until the measured TCP is within tol for `settle` consecutive steps.
    Stall guard: with CPU iterative IK a too-large offset can leave the arm frozen (R003: 140 steps at a
    constant command, joints unchanged). If the TCP moves < 0.6 mm over 3 steps while far from the
    target, the command magnitude is halved (down to 0.1); it recovers gradually once motion resumes."""
    ok = 0
    steps = 0
    err = 1.0
    scale = 1.0
    stalls = 0
    hist = []
    for k in range(max_steps):
        _cmd(robot, target, R_des, grip, kp=kp, vmax=vmax * scale)
        steps += 1
        p = np.array(robot.now().tcp[:3], float)
        hist.append(p)
        err = float(np.linalg.norm(np.asarray(target, float) - p))
        if err < tol:
            ok += 1
            if ok >= settle:
                break
        else:
            ok = 0
        if len(hist) >= 4 and err > max(3 * tol, 0.015):
            moved = float(np.linalg.norm(hist[-1] - hist[-4]))
            if moved < 0.0006:
                stalls += 1
                scale = max(0.1, scale * 0.5)
                hist = [p]
            elif scale < 1.0 and moved > 0.003:
                scale = min(1.0, scale * 1.3)
    return {'steps': steps, 'err': round(err, 4), 'stalls': stalls}


def _hold(robot, target, R_des, grip, n, kp=1.2):
    for k in range(n):
        _cmd(robot, target, R_des, grip, kp=kp, vmax=0.5)


def _seg_dist(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def _densify(pts, corner, ds):
    out = []
    flags = []
    for i in range(len(pts) - 1):
        a = np.array(pts[i], float)
        b = np.array(pts[i + 1], float)
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / ds)))
        for j in range(n):
            out.append(a + (b - a) * (j / n))
            flags.append(bool(corner[i]) if j == 0 else False)
    out.append(np.array(pts[-1], float))
    flags.append(True)
    return out, flags


def _press_orientation(robot, x, y, z):
    """Gripper-down orientation, tilted toward the outward radial direction when the button is far."""
    dx = x - ROBOT_BASE_XY[0]
    dy = y - ROBOT_BASE_XY[1]
    d = math.hypot(dx, dy)
    cx = dx / max(d, 1e-9)
    cy = dy / max(d, 1e-9)
    base = _down(0.0)
    cands = [0.0, 0.2, 0.35, 0.5, 0.6]
    info = []
    for ang in cands:
        R = _axis_angle([cy, -cx, 0.0], ang) @ base
        rpy = _R2rpy(R)
        reach = None
        try:
            res = robot.dry_run(x, y, z, rpy[0], rpy[1], rpy[2])
            if isinstance(res, dict):
                reach = res.get('reachable', res.get('within_joint_limits'))
        except Exception as exc:
            info.append('dry_run error ' + str(exc)[:80])
            reach = None
        info.append([round(ang, 2), reach])
        if reach is None or reach:
            return R, ang, info
    R = _axis_angle([cy, -cx, 0.0], cands[-1]) @ base
    return R, cands[-1], info
