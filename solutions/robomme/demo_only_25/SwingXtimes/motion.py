"""Motion helpers and act tools for SwingXtimes (absolute world TCP targets, gradual steps)."""

MAX_LIN = 0.008      # metres per control step
MAX_YAW = 0.06       # radians per control step
ROLL = math.pi


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _tcp(robot):
    p = np.asarray(robot.now().tcp, dtype=float)
    return p


def _fingers(robot):
    f = np.asarray(robot.now().fingers, dtype=float)
    return float(np.sum(f))


def _goto(robot, x, y, z, yaw, grip, tol=0.004, max_steps=250, settle=40, speed=MAX_LIN, integ=0.3):
    """Move the TCP gradually along a straight line to (x,y,z,yaw); integral correction while settling."""
    goal = np.array([x, y, z], dtype=float)
    p = _tcp(robot)
    cmd = p[:3].copy()
    cyaw = float(p[5])
    steps = 0
    corr = np.zeros(3)
    while steps < max_steps:
        d = goal - cmd
        n = float(np.linalg.norm(d))
        if n > speed:
            cmd = cmd + d / n * speed
        else:
            cmd = goal.copy()
        dy = _wrap(yaw - cyaw)
        cyaw = cyaw + max(-MAX_YAW, min(MAX_YAW, dy))
        tgt = cmd + corr
        robot.step(float(tgt[0]), float(tgt[1]), float(tgt[2]), ROLL, 0.0, float(cyaw), grip)
        steps += 1
        if n <= speed and abs(_wrap(yaw - cyaw)) < 1e-6:
            p = _tcp(robot)
            err = goal - p[:3]
            if float(np.linalg.norm(err)) < tol:
                break
            settle -= 1
            if settle <= 0:
                break
            corr = np.clip(corr + integ * err, -0.02, 0.02)
    p = _tcp(robot)
    return {'steps': steps, 'tcp': [round(float(v), 4) for v in p[:3]],
            'err': round(float(np.linalg.norm(goal - p[:3])), 4)}


def _hold(robot, grip, n):
    p = _tcp(robot)
    for _ in range(n):
        robot.step(float(p[0]), float(p[1]), float(p[2]), ROLL, 0.0, float(p[5]), grip)


def act_move(robot, x, y, z, yaw=0.0, gripper=1.0, lift_first=False, safe_z=0.15):
    """Move the TCP gradually to world (x,y,z) with top-down orientation and yaw; gripper +1 open, -1 closed.
    lift_first raises to safe_z before travelling horizontally."""
    out = {}
    if lift_first:
        p = _tcp(robot)
        if p[2] < safe_z - 0.01:
            out['lift'] = _goto(robot, p[0], p[1], safe_z, p[5], gripper)
    out['move'] = _goto(robot, x, y, z, yaw, gripper)
    out['fingers'] = round(_fingers(robot), 4)
    return out


def act_grasp(robot, x, y, yaw, grasp_z=0.02, hover_z=0.15):
    """Descend open onto a cube centred at (x,y) with the given yaw, close the gripper and lift to hover_z.
    Reports finger width sum; an empty grasp closes to about 0."""
    out = {}
    out['align'] = _goto(robot, x, y, hover_z, yaw, 1.0, max_steps=120)
    out['descend'] = _goto(robot, x, y, grasp_z, yaw, 1.0, tol=0.003, speed=0.006)
    _hold(robot, -1.0, 12)
    out['closed_fingers'] = round(_fingers(robot), 4)
    out['lift'] = _goto(robot, x, y, hover_z, yaw, -1.0, speed=0.006)
    f = _fingers(robot)
    out['fingers'] = round(f, 4)
    out['holding'] = bool(f > 0.012)
    return out


def act_carry(robot, x, y, z=0.10, yaw=0.0):
    """With the gripper closed on the cube, travel to (x,y,z) and settle there briefly (a target visit)."""
    out = _goto(robot, x, y, z, yaw, -1.0, tol=0.004)
    _hold(robot, -1.0, 3)
    f = _fingers(robot)
    out['fingers'] = round(f, 4)
    out['holding'] = bool(f > 0.012)
    return out


def act_place(robot, x, y, yaw=0.0, place_z=0.05, hover_z=0.15):
    """Lower the held cube over (x,y) to place_z, open the gripper and lift back to hover_z."""
    out = {}
    out['over'] = _goto(robot, x, y, 0.10, yaw, -1.0)
    out['lower'] = _goto(robot, x, y, place_z, yaw, -1.0, speed=0.005)
    _hold(robot, 1.0, 12)
    out['lift'] = _goto(robot, x, y, hover_z, yaw, 1.0, speed=0.006)
    out['fingers'] = round(_fingers(robot), 4)
    return out


def act_press(robot, x, y, press_z=0.007, hover_z=0.15):
    """Close the gripper above the button at (x,y), push straight down to press_z, then rise to hover_z."""
    out = {}
    p = _tcp(robot)
    if p[2] < hover_z - 0.01:
        out['lift'] = _goto(robot, p[0], p[1], hover_z, p[5], 1.0)
    out['over'] = _goto(robot, x, y, hover_z, 0.0, 1.0)
    _hold(robot, -1.0, 10)
    out['down'] = _goto(robot, x, y, press_z, 0.0, -1.0, tol=0.004, speed=0.006, settle=15, integ=0.0)
    _hold(robot, -1.0, 5)
    out['up'] = _goto(robot, x, y, hover_z, 0.0, -1.0, speed=0.006)
    return out


def act_wait(robot, steps=5):
    """Hold the current pose (gripper state kept by sign of finger width) for a few steps."""
    p = _tcp(robot)
    g = 1.0 if _fingers(robot) > 0.07 else -1.0
    for _ in range(int(max(1, min(steps, 50)))):
        robot.step(float(p[0]), float(p[1]), float(p[2]), ROLL, 0.0, float(p[5]), g)
    return {'tcp': [round(float(v), 4) for v in _tcp(robot)[:3]]}
