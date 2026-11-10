"""Motion helpers and the pick-and-place act tool (no imports)."""

CUBE_X_BIAS = 0.0083     # front-camera top-face centroid sits +8.3 mm in x of the grasp TCP (25 episodes, std 0.6 mm)
HOVER_Z = 0.15
GRASP_Z = 0.018
PLACE_Z = 0.02
FAST = 0.012             # metres per control step in free space
SLOW = 0.006             # metres per control step near the table
YAW_STEP = 0.06
CLEAR_POSE = (-0.06, 0.0, 0.35)   # out of the front camera's view of the table


def _pose(robot):
    p = np.asarray(robot.now().tcp, float)
    return p


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _move(robot, x, y, z, yaw, grip, speed=FAST, settle=25, tol=0.004):
    """Interpolate the TCP to (x, y, z, yaw) with a gripper command, then settle on feedback."""
    p = _pose(robot)
    start = np.asarray(p[:3], float)
    goal = np.array([x, y, z], float)
    y0 = float(p[5])
    dyaw = _wrap(yaw - y0)
    dist = float(np.linalg.norm(goal - start))
    n = int(max(1, math.ceil(max(dist / speed, abs(dyaw) / YAW_STEP))))
    for k in range(1, n + 1):
        a = k / float(n)
        q = start + (goal - start) * a
        robot.step(float(q[0]), float(q[1]), float(q[2]), math.pi, 0.0, float(y0 + dyaw * a), grip)
    for _ in range(settle):
        p = _pose(robot)
        if np.linalg.norm(p[:3] - goal) < tol:
            break
        robot.step(float(x), float(y), float(z), math.pi, 0.0, float(y0 + dyaw), grip)
    return _pose(robot)


def _hold(robot, n, grip):
    p = _pose(robot)
    for _ in range(n):
        robot.step(float(p[0]), float(p[1]), float(p[2]), math.pi, 0.0, float(p[5]), grip)
    return _pose(robot)


def _fingers(robot):
    f = getattr(robot.now(), 'fingers', None)
    if f is None:
        return None
    return float(np.mean(np.asarray(f, float)))


def _locate_cube(frame, color):
    """Cube [x, y, yaw] (grasp frame) from the front camera, or None."""
    top = _cube_top_points(frame, color)
    c = _detect_cube(frame, color)
    if c is None:
        return None
    yaw = 0.0
    if top is not None and len(top) >= 6:
        yaw = _square_yaw(top[:, :2])
    return [c[0] - CUBE_X_BIAS, c[1], yaw, c[2]]


def act_pick_place(robot, color, target_x, target_y, cube_x=None, cube_y=None, cube_yaw=0.0):
    """Pick the cube of `color` and place it on the target at (target_x, target_y) world metres.

    The cube is re-located in the front camera at the start (falling back to
    cube_x/cube_y/cube_yaw). Moves above the cube, descends, closes, lifts, carries
    to the target, lowers, opens and retreats. Returns grasp/placement facts."""
    f = robot.now()
    loc = _locate_cube(f, color)
    if loc is None and cube_x is not None:
        loc = [float(cube_x), float(cube_y), float(cube_yaw), 0.04]
    if loc is None:
        return {'ok': False, 'reason': 'cube not found'}
    cx, cy, cyaw = float(loc[0]), float(loc[1]), float(loc[2])
    # choose the gripper yaw equivalent (mod 90 deg) closest to zero
    cyaw = (cyaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    _move(robot, cx, cy, HOVER_Z, cyaw, 1.0, FAST)
    _move(robot, cx, cy, 0.06, cyaw, 1.0, FAST, settle=5)
    _move(robot, cx, cy, GRASP_Z, cyaw, 1.0, SLOW, settle=20, tol=0.003)
    _hold(robot, 18, -1.0)
    w = _fingers(robot)
    _move(robot, cx, cy, HOVER_Z, cyaw, -1.0, FAST, settle=5)
    w2 = _fingers(robot)
    held = w2 is None or w2 > 0.008
    if not held:
        _hold(robot, 6, 1.0)
        _move(robot, CLEAR_POSE[0], CLEAR_POSE[1], CLEAR_POSE[2], 0.0, 1.0, FAST, settle=5)
        return {'ok': False, 'reason': 'grasp missed', 'cube': [cx, cy, cyaw], 'finger': w2}
    _move(robot, float(target_x), float(target_y), HOVER_Z, cyaw, -1.0, FAST)
    _move(robot, float(target_x), float(target_y), 0.06, cyaw, -1.0, FAST, settle=5)
    _move(robot, float(target_x), float(target_y), PLACE_Z, cyaw, -1.0, SLOW, settle=20, tol=0.003)
    _hold(robot, 12, 1.0)
    _move(robot, float(target_x), float(target_y), HOVER_Z, cyaw, 1.0, FAST, settle=5)
    _move(robot, CLEAR_POSE[0], CLEAR_POSE[1], CLEAR_POSE[2], 0.0, 1.0, FAST, settle=5)
    _hold(robot, 5, 1.0)
    after = _locate_cube(robot.now(), color)
    return {'ok': True, 'cube': [cx, cy, cyaw], 'finger_closed': w, 'finger_lifted': w2,
            'cube_after': after}


def act_wait(robot, steps=10):
    """Hold the current pose with the gripper open for `steps` control steps."""
    _hold(robot, int(max(1, min(200, steps))), 1.0)
    return {'waited': int(steps)}
