"""Motion helpers and act_* tools (absolute world TCP targets, gripper -1 close / +1 open)."""

ROLL = math.pi
PITCH = 0.0
SAFE_Z = 0.15          # travel height over buttons
HIGH_Z = 0.20          # approach height over containers
GRASP_Z = 0.064        # TCP height when closing on a container (expert: 0.062-0.066)
PLACE_Z = 0.070        # TCP height when releasing a container back on the table
PRESS_Z = 0.008        # commanded press depth; contact stops the TCP near 0.024
MAX_LIN = 0.010        # metres per control step
MAX_ANG = 0.06         # radians per control step


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _pose(robot):
    t = np.asarray(robot.now().tcp, dtype=np.float64)
    return t


def _move(robot, x, y, z, yaw, grip, tol=0.004, settle=40, max_lin=MAX_LIN):
    """Interpolated straight-line motion to (x,y,z,yaw); returns final tcp."""
    p = _pose(robot)
    start = p[:3].copy()
    yaw0 = float(p[5])
    dyaw = _wrap(yaw - yaw0)
    dist = float(np.linalg.norm(np.array([x, y, z]) - start))
    n = max(1, int(math.ceil(max(dist / max_lin, abs(dyaw) / MAX_ANG))))
    for k in range(1, n + 1):
        s = k / n
        q = start + s * (np.array([x, y, z]) - start)
        robot.step(q[0], q[1], q[2], ROLL, PITCH, _wrap(yaw0 + s * dyaw), grip)
    for _ in range(settle):
        p = _pose(robot)
        if np.linalg.norm(p[:3] - np.array([x, y, z])) < tol:
            break
        robot.step(x, y, z, ROLL, PITCH, yaw, grip)
    return _pose(robot)


def _hold(robot, n, grip, x=None, y=None, z=None, yaw=None):
    p = _pose(robot)
    x = p[0] if x is None else x
    y = p[1] if y is None else y
    z = p[2] if z is None else z
    yaw = p[5] if yaw is None else yaw
    for _ in range(n):
        robot.step(x, y, z, ROLL, PITCH, yaw, grip)


def act_press_button(robot, x, y):
    """Press the button whose cap centre is at world (x, y): travel at 15 cm with the gripper open,
    close the gripper, push down until contact stalls the TCP, then lift back to 15 cm (gripper stays closed)."""
    p = _pose(robot)
    if p[2] < SAFE_Z - 0.01:
        _move(robot, p[0], p[1], SAFE_Z, 0.0, 1.0)
    _move(robot, x, y, SAFE_Z, 0.0, 1.0)
    _hold(robot, 12, -1.0, x, y, SAFE_Z, 0.0)
    zmin = 1.0
    z = SAFE_Z
    stall = 0
    prev = _pose(robot)[2]
    for _ in range(80):
        z = max(PRESS_Z, z - 0.008)
        robot.step(x, y, z, ROLL, PITCH, 0.0, -1.0)
        cz = float(_pose(robot)[2])
        zmin = min(zmin, cz)
        if cz < 0.045 and abs(prev - cz) < 0.0008:
            stall += 1
        else:
            stall = 0
        prev = cz
        if stall >= 4:
            break
    _hold(robot, 4, -1.0, x, y, z, 0.0)
    _move(robot, x, y, SAFE_Z, 0.0, -1.0, max_lin=0.012)
    return {"pressed_min_z": zmin, "tcp": [float(v) for v in _pose(robot)]}


def act_hold(robot, steps=20):
    """Hold the current pose for `steps` control steps (gripper keeps its last command) while the scene settles."""
    f = robot.now()
    grip = 1.0 if float(np.mean(f.fingers)) > 0.03 else -1.0
    _hold(robot, int(steps), grip)
    return {"held": int(steps)}


def act_pick(robot, x, y, yaw, lift_z=HIGH_Z):
    """Grasp the container centred at world (x, y) with gripper yaw `yaw`: open, approach at 20 cm,
    descend to the grasp height, close, lift to `lift_z`. Returns finger width and whether something is held;
    on an empty grasp the gripper reopens and rises."""
    p = _pose(robot)
    if p[2] < SAFE_Z - 0.01:
        _move(robot, p[0], p[1], SAFE_Z, float(p[5]), 1.0)
    _move(robot, x, y, HIGH_Z, yaw, 1.0, max_lin=0.012)
    _hold(robot, 6, 1.0, x, y, HIGH_Z, yaw)
    _move(robot, x, y, GRASP_Z, yaw, 1.0, tol=0.003, max_lin=0.008)
    _hold(robot, 15, -1.0, x, y, GRASP_Z, yaw)
    width = float(np.sum(robot.now().fingers))
    held = width > 0.01
    if not held:
        _hold(robot, 10, 1.0, x, y, GRASP_Z, yaw)
        _move(robot, x, y, HIGH_Z, yaw, 1.0)
        return {"held": False, "finger_width": width}
    _move(robot, x, y, lift_z, yaw, -1.0, max_lin=0.008)
    _hold(robot, 10, -1.0, x, y, lift_z, yaw)
    width = float(np.sum(robot.now().fingers))
    return {"held": width > 0.01, "finger_width": width, "tcp": [float(v) for v in _pose(robot)]}


def act_place(robot, x, y, yaw):
    """Lower the held container back onto the table at (x, y), open the gripper and rise to 15 cm."""
    _move(robot, x, y, PLACE_Z, yaw, -1.0, tol=0.004, max_lin=0.008)
    _hold(robot, 12, 1.0, x, y, PLACE_Z, yaw)
    _move(robot, x, y, SAFE_Z, yaw, 1.0)
    return {"placed": True, "tcp": [float(v) for v in _pose(robot)]}
