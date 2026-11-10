"""Motion helpers and act tools for PickHighlight."""

MAX_LIN = 0.009      # metres per control step along the commanded path
MAX_YAW = 0.06       # radians per control step
HOVER_Z = 0.15
GRASP_Z = 0.02       # TCP height when grasping a 4 cm cube resting on the table
PLACE_Z = 0.022
PRESS_Z = 0.005      # commanded press depth (button stops TCP near 0.024)
# measured world offset: TCP command = perceived centre + offset (from 25 demos)
CUBE_OFFSET = (-0.007, -0.001)
BUTTON_OFFSET = (-0.0155, -0.002)
FINGER_HELD = 0.007


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _go(robot, x, y, z, yaw, grip, settle=40, tol=0.003):
    """Interpolate the commanded TCP to (x,y,z,yaw) then wait for convergence."""
    f = robot.now()
    p = np.array(f.tcp[:3], dtype=float)
    cyaw = float(f.tcp[5])
    goal = np.array([x, y, z], dtype=float)
    dyaw = _wrap(yaw - cyaw)
    n = int(max(np.linalg.norm(goal - p) / MAX_LIN, abs(dyaw) / MAX_YAW, 1))
    for i in range(1, n + 1):
        a = i / n
        q = p + (goal - p) * a
        robot.step(q[0], q[1], q[2], math.pi, 0.0, cyaw + dyaw * a, grip)
    for _ in range(settle):
        f = robot.now()
        err = np.linalg.norm(np.array(f.tcp[:3]) - goal)
        yerr = abs(_wrap(float(f.tcp[5]) - yaw))
        if err < tol and yerr < 0.03:
            break
        robot.step(x, y, z, math.pi, 0.0, yaw, grip)
    return robot.now()


def _hold(robot, n, grip):
    f = robot.now()
    t = f.tcp
    for _ in range(n):
        robot.step(t[0], t[1], t[2], math.pi, 0.0, t[5], grip)
    return robot.now()


def _grip(robot, grip, n=12):
    """Command gripper (-1 close, +1 open) holding position."""
    f = robot.now()
    t = f.tcp
    for _ in range(n):
        robot.step(t[0], t[1], t[2], math.pi, 0.0, t[5], grip)
    return robot.now()


def act_hold(robot, steps=30):
    """Keep the robot still with an open gripper for `steps` control steps (observation window)."""
    f = robot.now()
    t = f.tcp
    for _ in range(int(steps)):
        robot.step(t[0], t[1], t[2], math.pi, 0.0, 0.0, 1.0)
    return {"held_steps": int(steps)}


def act_press_button(robot, x, y):
    """Press the button whose perceived disc centre is world (x, y) with a closed gripper, then retreat."""
    bx, by = x + BUTTON_OFFSET[0], y + BUTTON_OFFSET[1]
    f = robot.now()
    _go(robot, f.tcp[0], f.tcp[1], max(f.tcp[2], HOVER_Z), 0.0, 1.0, settle=5)
    _go(robot, bx, by, HOVER_Z, 0.0, 1.0)
    _grip(robot, -1.0, 8)
    _go(robot, bx, by, 0.04, 0.0, -1.0, settle=3)
    # slow final press
    zs = []
    for k in range(14):
        zc = max(PRESS_Z, 0.04 - 0.004 * (k + 1))
        f = robot.step(bx, by, zc, math.pi, 0.0, 0.0, -1.0)
        zs.append(float(f.tcp[2]))
    _hold(robot, 4, -1.0)
    zmin = min(zs)
    _go(robot, bx, by, HOVER_Z, 0.0, -1.0)
    _grip(robot, 1.0, 10)
    return {"pressed": True, "min_tcp_z": zmin}


def _refine(robot, x, y, color, radius=0.06):
    f = robot.now()
    cubes = _detect_cubes(f.wrist_rgb, f.wrist_depth, f.wrist_K, f.wrist_extrinsic)
    best = None
    for c in cubes:
        if color and c['color'] != color:
            continue
        d = math.hypot(c['xyz'][0] - x, c['xyz'][1] - y)
        if d < radius and (best is None or d < best[0]):
            best = (d, c)
    return None if best is None else best[1]


def _pick_yaw(cube_yaw, current):
    best = None
    for k in (-2, -1, 0, 1, 2):
        cand = cube_yaw + k * math.pi / 2
        if abs(cand) > 2.6:
            continue
        if best is None or abs(_wrap(cand - current)) < abs(_wrap(best - current)):
            best = cand
    return best if best is not None else cube_yaw


def act_pick(robot, x, y, color, place_back=True):
    """Pick the `color` cube near perceived world (x, y): hover, re-perceive, align yaw, grasp, lift.
    If place_back, lower it to the table, release and retreat. Returns grasp outcome and refined position."""
    f = robot.now()
    _go(robot, f.tcp[0], f.tcp[1], max(f.tcp[2], HOVER_Z), float(f.tcp[5]), 1.0, settle=5)
    cx, cy, cyaw = x, y, 0.0
    found = False
    for _ in range(2):
        _go(robot, cx + CUBE_OFFSET[0], cy + CUBE_OFFSET[1], HOVER_Z, 0.0, 1.0)
        _hold(robot, 3, 1.0)
        c = _refine(robot, cx, cy, color)
        if c is None:
            break
        found = True
        moved = math.hypot(c['xyz'][0] - cx, c['xyz'][1] - cy)
        cx, cy, cyaw = c['xyz'][0], c['xyz'][1], c['yaw']
        if moved < 0.01:
            break
    yaw = _pick_yaw(cyaw, 0.0)
    gx, gy = cx + CUBE_OFFSET[0], cy + CUBE_OFFSET[1]
    _go(robot, gx, gy, HOVER_Z, yaw, 1.0)
    # second look after rotation, small correction only
    c = _refine(robot, cx, cy, color, radius=0.03)
    if c is not None:
        cx, cy = c['xyz'][0], c['xyz'][1]
        gx, gy = cx + CUBE_OFFSET[0], cy + CUBE_OFFSET[1]
        if abs(_wrap(c['yaw'] - cyaw)) < 0.3:
            yaw = _pick_yaw(c['yaw'], yaw)
    _go(robot, gx, gy, 0.06, yaw, 1.0)
    _go(robot, gx, gy, GRASP_Z, yaw, 1.0)
    _grip(robot, -1.0, 12)
    f = _go(robot, gx, gy, HOVER_Z, yaw, -1.0)
    held = float(min(f.fingers)) > FINGER_HELD
    if held and place_back:
        _go(robot, gx, gy, PLACE_Z + 0.03, yaw, -1.0)
        _go(robot, gx, gy, PLACE_Z, yaw, -1.0, settle=10)
        _grip(robot, 1.0, 10)
        _go(robot, gx, gy, HOVER_Z, yaw, 1.0)
    elif not held:
        _grip(robot, 1.0, 8)
    return {"held": held, "found": found, "x": cx, "y": cy, "fingers": [float(v) for v in f.fingers]}
