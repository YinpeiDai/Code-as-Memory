"""Action tools for SeqOfColors3: wait for the response layout, touch a cube, press the button."""

TRAVEL_Z = 0.13      # TCP height for moves between targets (cube tops are at 0.041 m)
CLEAR_Z = 0.10       # lift at least this high before moving sideways
TOUCH_Z = 0.046      # expert touch minima: 0.044-0.062 m; 0.07 failed (R011)
PRESS_TARGET_Z = 0.03  # pushed target; the cap stalls the TCP near 0.053 m in demos
LIFT_Z = 0.18        # expert success frames occur while lifting through 0.16-0.17 m


def act_wait_for_layout(robot, max_steps=790):
    """Hold still (zero motion, gripper open) during the cue and delay phases until the
    response layout of coloured cubes is visible in the front camera, or max_steps.
    Returns steps waited, whether the layout was seen and the native step reached."""
    n = 0
    seen = _layout_visible(_scene_colors(robot.now()))
    while not seen and n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        seen = _layout_visible(_scene_colors(robot.now()))
    return {"waited": n, "layout_seen": bool(seen), "step": int(robot.now().step)}


def _reach_note(robot, x, y, z):
    try:
        chk = robot.dry_run(x, y, z, TARGET_RPY[0], TARGET_RPY[1], TARGET_RPY[2])
    except Exception as exc:  # dry-run is advisory only
        return {"dry_run_error": str(exc)[:80]}
    return {"reachable": bool(chk.get("reachable")), "path_clear": bool(chk.get("path_clear")),
            "margin": chk.get("joint_margin")}


def _lift_clear(robot, grip, z_clear):
    cur = _tcp(robot)
    if cur[2] >= z_clear:
        return 0
    tgt = [cur[0], cur[1], z_clear + 0.03]
    n, _ = _servo(robot, tgt, grip, lambda c, p, k: c[2] >= z_clear, 25)
    return n


def act_touch(robot, x, y, color="", hold_steps=3, touch_z=TOUCH_Z):
    """Touch the cube whose (offset-corrected) top centre is at world (x, y): lift clear,
    travel at TRAVEL_Z with the gripper open, descend to touch_z, hold hold_steps, lift to
    CLEAR_Z. Returns measured bottom pose, xy error, step counts and the reach check."""
    out = {"color": color, "start_step": int(robot.now().step)}
    out["reach"] = _reach_note(robot, x, y, touch_z)
    n_lift = _lift_clear(robot, 1.0, CLEAR_Z)
    def at_xy(c, p, k):
        return math.hypot(c[0] - x, c[1] - y) < 0.012
    n_trav, ok_trav = _servo(robot, [x, y, TRAVEL_Z], 1.0, at_xy, 60)
    zmin = [1.0]
    def down(c, p, k):
        zmin[0] = min(zmin[0], float(c[2]))
        return c[2] < touch_z + 0.004 or (k > 3 and abs(c[2] - p[2]) < 0.001 and c[2] < touch_z + 0.02)
    n_down, ok_down = _servo(robot, [x, y, touch_z], 1.0, down, 30)
    _hold(robot, [x, y, touch_z], 1.0, hold_steps)
    bottom = _tcp(robot)
    zmin[0] = min(zmin[0], float(bottom[2]))
    n_up = _lift_clear(robot, 1.0, CLEAR_Z)
    out.update({"lift": n_lift, "travel": n_trav, "travel_ok": bool(ok_trav), "down": n_down,
                "descended": bool(ok_down), "zmin": round(zmin[0], 4),
                "bottom_xy_err": round(math.hypot(bottom[0] - x, bottom[1] - y), 4),
                "bottom_tcp": _r(bottom), "up": n_up, "end_step": int(robot.now().step)})
    return out


def act_press(robot, x, y, hold_steps=4, wait_steps=20):
    """Press the button cap at world (x, y) with the gripper closed: lift clear, travel at
    TRAVEL_Z, descend towards PRESS_TARGET_Z until the cap stalls the TCP, hold hold_steps,
    lift to LIFT_Z and hold up to wait_steps (the episode ends on native success).
    Returns measured stall height, xy error and step counts."""
    out = {"start_step": int(robot.now().step)}
    out["reach"] = _reach_note(robot, x, y, 0.06)
    n_lift = _lift_clear(robot, -1.0, CLEAR_Z)
    def at_xy(c, p, k):
        return math.hypot(c[0] - x, c[1] - y) < 0.01 and k >= 4
    n_trav, ok_trav = _servo(robot, [x, y, TRAVEL_Z], -1.0, at_xy, 60)
    fing = float(np.max(robot.now().fingers))
    def stall(c, p, k):
        return k > 3 and c[2] < 0.066 and abs(c[2] - p[2]) < 0.0015
    n_down, ok_down = _servo(robot, [x, y, PRESS_TARGET_Z], -1.0, stall, 30, vmax=0.6)
    stall_tcp = _tcp(robot)
    _hold(robot, [x, y, PRESS_TARGET_Z], -1.0, hold_steps)
    held_tcp = _tcp(robot)
    n_up, _ = _servo(robot, [x, y, LIFT_Z], -1.0, lambda c, p, k: c[2] > LIFT_Z - 0.01, 40)
    _hold(robot, [x, y, LIFT_Z], -1.0, wait_steps)
    out.update({"fingers_before_descent": round(fing, 4), "travel": n_trav, "travel_ok": bool(ok_trav),
                "down": n_down, "stalled": bool(ok_down), "stall_tcp": _r(stall_tcp),
                "held_tcp": _r(held_tcp), "xy_err": round(math.hypot(stall_tcp[0] - x, stall_tcp[1] - y), 4),
                "up": n_up, "end_step": int(robot.now().step), "success_not_signalled": True})
    return out


def act_idle(robot, steps=20):
    """Hold the current pose with the gripper closed for `steps` steps (fallback)."""
    cur = _tcp(robot)
    _hold(robot, [cur[0], cur[1], max(cur[2], CLEAR_Z)], -1.0, steps)
    return {"idled": steps, "step": int(robot.now().step)}
