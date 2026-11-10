"""Act tools for place_block_mat. Every motion is one native robot.step with
absolute world end-effector targets for both arms; the idle arm repeats its
current pose. MotionError (an Exception) is caught per step and reported."""


def _poses(robot):
    now = robot.now()
    return ([float(v) for v in now.left_ee_pose], float(now.left_gripper),
            [float(v) for v in now.right_ee_pose], float(now.right_gripper))


def _move(robot, arm, pose, grip):
    """One native step moving `arm` to pose (7) with gripper; returns error text or ''."""
    L, lg, R, rg = _poses(robot)
    if arm == "left":
        L, lg = list(pose), float(grip)
    else:
        R, rg = list(pose), float(grip)
    try:
        robot.step(*L, lg, *R, rg)
    except Exception as e:
        return str(e)[:160]
    return ""


def _held_pixels(robot, arm):
    now = robot.now()
    rgb = getattr(now, arm + "_wrist_rgb", None)
    dep = getattr(now, arm + "_wrist_depth", None)
    if rgb is None or dep is None:
        return -1
    m = _red_mask(rgb) & (dep > 0) & (dep < 100)
    return int(m.sum())


def _move_retry(robot, arm, pose, grip, errs, tag):
    """_move; on a planning error, retry once via a waypoint 5 cm higher at the
    current xy. Returns True on success."""
    e = _move(robot, arm, pose, grip)
    if not e:
        return True
    errs.append(tag + ":" + e)
    now = robot.now()
    cur = [float(v) for v in getattr(now, arm + "_ee_pose")]
    _move(robot, arm, [cur[0], cur[1], cur[2] + 0.05] + cur[3:], grip)
    e = _move(robot, arm, pose, grip)
    if not e:
        return True
    errs.append(tag + "2:" + e)
    return False


def _abort_home(robot, arm, home, errs):
    """Open, lift straight up and return `arm` home after a failed grasp."""
    now = robot.now()
    cur = [float(v) for v in getattr(now, arm + "_ee_pose")]
    _move(robot, arm, cur, 1.0)
    _move_retry(robot, arm, [cur[0], cur[1], max(cur[2] + 0.06, 0.99)] + cur[3:], 1.0, errs, "abort_lift")
    _move_retry(robot, arm, home, 1.0, errs, "abort_home")


def act_pick(robot, arm, x, y, yaw_deg, z_grasp, z_approach, z_carry, home_left, home_right):
    """Top-down grasp of a block at world (x, y) with gripper yaw yaw_deg using
    `arm` ('left'/'right'): approach above, descend to z_grasp, close, lift to
    z_carry. Returns holding (wrist-camera check), red_px and any errors. On a
    failed grasp the arm opens, lifts and returns home (holding false)."""
    home = home_left if arm == "left" else home_right
    q = _q_yaw(yaw_deg)
    errs = []
    if not _move_retry(robot, arm, [x, y, z_approach] + q, 1.0, errs, "approach"):
        _abort_home(robot, arm, home, errs)
        return {"holding": False, "red_px": -1, "errors": errs}
    if not _move_retry(robot, arm, [x, y, z_grasp] + q, 1.0, errs, "descend"):
        _abort_home(robot, arm, home, errs)
        return {"holding": False, "red_px": -1, "errors": errs}
    e = _move(robot, arm, [x, y, z_grasp] + q, 0.0)
    if e:
        errs.append("close:" + e)
    _move_retry(robot, arm, [x, y, z_carry] + q, 0.0, errs, "lift")
    px = _held_pixels(robot, arm)
    if px <= HOLD_PIXELS:
        _abort_home(robot, arm, home, errs)
    return {"holding": bool(px > HOLD_PIXELS), "red_px": px, "errors": errs}


def act_place(robot, arm, x, y, z_place, z_carry, home_left, home_right):
    """Carry the held block with `arm` above world (x, y) at z_carry (gripper
    yaw 0), lower to z_place, open, lift straight up, then return the arm home.
    If the carry cannot be planned the block is kept (placed false, holding)."""
    q = _q_yaw(0.0)
    errs = []
    opened = False
    px = _held_pixels(robot, arm)
    if px <= HOLD_PIXELS:
        errs.append("block not in gripper before carry")
    elif not _move_retry(robot, arm, [x, y, z_carry] + q, 0.0, errs, "carry"):
        return {"placed": False, "holding": True, "errors": errs}
    else:
        if not _move_retry(robot, arm, [x, y, z_place] + q, 0.0, errs, "lower"):
            _move(robot, arm, [x, y, z_place + 0.03] + q, 0.0)
        e = _move(robot, arm, [x, y, z_place] + q, 1.0)
        if e:
            errs.append("open:" + e)
        opened = True
    now = robot.now()
    cur = [float(v) for v in getattr(now, arm + "_ee_pose")]
    _move_retry(robot, arm, [cur[0], cur[1], max(cur[2], z_place) + 0.07] + cur[3:], 1.0, errs, "retreat")
    home = home_left if arm == "left" else home_right
    _move_retry(robot, arm, home, 1.0, errs, "home")
    return {"placed": opened, "holding": False, "errors": errs}


def act_home(robot, home_left, home_right):
    """Open both grippers and return both arms to their reset (home) poses."""
    errs = []
    L, lg, R, rg = _poses(robot)
    for arm, home, pose in (("left", home_left, L), ("right", home_right, R)):
        if math.hypot(pose[0] - home[0], pose[1] - home[1]) > 0.02 or abs(pose[2] - home[2]) > 0.02:
            # lift straight up first so the open fingers clear any block
            e = _move(robot, arm, [pose[0], pose[1], max(pose[2], 0.99)] + pose[3:], 1.0)
            if e:
                errs.append(arm + " lift:" + e)
        e = _move(robot, arm, home, 1.0)
        if e:
            errs.append(arm + " home:" + e)
    return {"home": True, "errors": errs}


def act_hold(robot):
    """Hold both arms at their current poses for one native step."""
    L, lg, R, rg = _poses(robot)
    robot.step(*L, lg, *R, rg)
    return {"held": True}
