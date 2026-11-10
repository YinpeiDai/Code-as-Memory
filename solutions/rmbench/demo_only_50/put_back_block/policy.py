"""put_back_block policy: pick block from its mat -> centre of the mat cross -> press button with the
left arm -> pick block from centre -> put it back on the remembered original mat.

All positions come from the head RGB-D camera (colour masks back-projected with depth and calibration).
Stage transitions are confirmed by re-perceiving the block after every act."""

# Right gripper pointing straight down (local +x -> world -z), as used by the expert for every grasp.
_Q_RIGHT_DOWN = [0.5, -0.5, 0.5, 0.5]
# Left gripper pointing down, as used by the expert to press the button.
_Q_LEFT_DOWN = [0.7071, 0.0, 0.7071, 0.0]
# Calibration measured on all 50 demos: expert ee xy minus perceived object xy.
_RIGHT_XY_OFF = [0.004, 0.002]
_LEFT_XY_OFF = [-0.002, 0.002]
# Heights relative to the perceived block top (z ~0.782 in demos).
_GRASP_DZ = 0.12      # expert grasp ee z 0.902
_APPROACH_DZ = 0.20   # expert pre-grasp ~0.98
_DROP_DZ = 0.165      # expert release over centre, ee z 0.947
_RETURN_DZ = 0.13     # expert release over original mat, ee z 0.913
_CARRY_Z = 1.0        # expert carry height
# Button heights relative to perceived button top (z ~0.780).
_PRESS_HOVER_DZ = 0.184   # expert ee z 0.964
_PRESS_DOWN_DZ = 0.146    # expert lowest ee z 0.926


def _pose(frame, arm):
    p = np.asarray(getattr(frame, arm + "_ee_pose"), dtype=float)
    return [float(v) for v in p]


def _grip(frame, arm):
    return float(np.asarray(getattr(frame, arm + "_gripper")))


def _step(robot, left, lg, right, rg):
    """left/right: 7-pose lists. Returns None or an error message (MotionError etc.)."""
    try:
        robot.step(*[float(v) for v in left], float(lg), *[float(v) for v in right], float(rg))
    except Exception as e:  # MotionError: arm could not plan; EpisodeEnd is a BaseException and propagates
        return str(e)[:200]
    return None


def _right_seq(robot, waypoints):
    """waypoints: list of ([x,y,z], gripper). Left arm holds its current pose."""
    errs = []
    for xyz, g in waypoints:
        now = robot.now()
        left = _pose(now, "left")
        err = _step(robot, left, _grip(now, "left"), list(xyz) + _Q_RIGHT_DOWN, g)
        if err:
            errs.append(err)
    end = _pose(robot.now(), "right")
    return {"ok": not errs, "errors": errs[:3], "right_end": [round(v, 4) for v in end[:3]],
            "right_gripper": _grip(robot.now(), "right")}


def act_pick(robot, x, y, ztop):
    """Right arm: approach above block at perceived (x, y) with top height ztop, descend, close, lift to carry height."""
    gx, gy = x + _RIGHT_XY_OFF[0], y + _RIGHT_XY_OFF[1]
    return _right_seq(robot, [
        ([gx, gy, ztop + _APPROACH_DZ], 1.0),
        ([gx, gy, ztop + _GRASP_DZ], 1.0),
        ([gx, gy, ztop + _GRASP_DZ], 0.0),
        ([gx, gy, ztop + _GRASP_DZ], 0.0),
        ([gx, gy, _CARRY_Z], 0.0),
    ])


def act_place_centre(robot, x, y, ztop):
    """Right arm carrying the block: move over the mat-cross centre (x, y), lower and release (expert drop height)."""
    gx, gy = x + _RIGHT_XY_OFF[0], y + _RIGHT_XY_OFF[1]
    return _right_seq(robot, [
        ([gx, gy, _CARRY_Z + 0.02], 0.0),
        ([gx, gy, ztop + _DROP_DZ], 0.0),
        ([gx, gy, ztop + _DROP_DZ], 1.0),
    ])


def act_place_back(robot, x, y, ztop):
    """Right arm carrying the block: move over the original mat (x, y), lower to just above it, release, rise."""
    gx, gy = x + _RIGHT_XY_OFF[0], y + _RIGHT_XY_OFF[1]
    return _right_seq(robot, [
        ([gx, gy, _CARRY_Z], 0.0),
        ([gx, gy, ztop + _RETURN_DZ], 0.0),
        ([gx, gy, ztop + _RETURN_DZ], 1.0),
        ([gx, gy, ztop + 0.25], 1.0),
    ])


def act_release(robot):
    """Right arm: open the gripper where it is, then rise 0.08 m (recovery when the block is not seen on the table)."""
    p = _pose(robot.now(), "right")
    return _right_seq(robot, [(p[:3], 1.0), ([p[0], p[1], p[2] + 0.08], 1.0)])


def act_press(robot, x, y, ztop, home_xyz, home_q):
    """Left arm: hover above the button at (x, y, top ztop) gripper-down, close fingers, press down, rise,
    return to home pose. Right arm holds its pose."""
    bx, by = x + _LEFT_XY_OFF[0], y + _LEFT_XY_OFF[1]
    seq = [
        ([bx, by, ztop + _PRESS_HOVER_DZ], _Q_LEFT_DOWN, 1.0),
        ([bx, by, ztop + _PRESS_HOVER_DZ], _Q_LEFT_DOWN, 0.0),
        ([bx, by, ztop + _PRESS_DOWN_DZ], _Q_LEFT_DOWN, 0.0),
        ([bx, by, ztop + _PRESS_HOVER_DZ], _Q_LEFT_DOWN, 0.0),
        (list(home_xyz), list(home_q), 0.0),
    ]
    errs = []
    lowest = 9.0
    for xyz, q, g in seq:
        now = robot.now()
        err = _step(robot, list(xyz) + list(q), g, _pose(now, "right"), _grip(now, "right"))
        if err:
            errs.append(err)
        for f in robot.frames()[-40:]:
            lowest = min(lowest, float(np.asarray(f.left_ee_pose)[2]))
    return {"ok": not errs, "errors": errs[:3], "lowest_left_z": round(lowest, 4),
            "press_target_z": round(ztop + _PRESS_DOWN_DZ, 4)}


def act_hold(robot):
    """Hold both arms at their current poses for one native step."""
    now = robot.now()
    err = _step(robot, _pose(now, "left"), _grip(now, "left"), _pose(now, "right"), _grip(now, "right"))
    return {"ok": err is None, "error": err}


def _mem(memory):
    keys = ["stage", "origin", "centre", "button", "pick_at", "attempts", "left_home", "note"]
    m = {k: memory.get(k) for k in keys}
    if m["attempts"] is None:
        m["attempts"] = 0
    return m


def perceive_decide(obs, goal, memory, last):
    """Stage machine (memory.stage): None -> pick1 -> placed -> pressed -> pick2 -> returned -> done.
    Every decision re-perceives the block in the head camera to confirm the previous act."""
    m = _mem(memory)
    f = obs.current
    s = _scene(f)
    stage = m["stage"]
    blk = s["block"]
    if stage is not None and m["centre"] is not None:
        blk = _block_excluding(s, m["button"], m["centre"])

    def pick(xyz, nxt):
        m["pick_at"] = [xyz[0], xyz[1], xyz[2]]
        m["stage"] = nxt
        return {"memory": m, "act": {"tool": "act_pick", "args": {"x": xyz[0], "y": xyz[1], "ztop": xyz[2]}}}

    def hold(note):
        m["note"] = note
        return {"memory": m, "act": {"tool": "act_hold", "args": {}}}

    if stage is None:
        lp = _pose(f, "left")
        m["left_home"] = [lp[:3], lp[3:]]
        if blk is None or len(s["mats"]) < 2:
            m["note"] = "scene not found at reset"
            return hold(m["note"])
        m["origin"] = [blk[0], blk[1], blk[2]]
        m["centre"] = _mat_centre(s["mats"], blk)
        if s["button"] is not None:
            m["button"] = [s["button"][0], s["button"][1], s["button"][2]]
        m["note"] = "origin mat remembered"
        return pick(m["origin"], "pick1")

    if stage in ("pick1", "pick2"):
        pa = m["pick_at"]
        n_low = _low_red_near(f, pa)
        if n_low >= 20:  # still resting where we tried to grasp -> retry with fresh estimate
            m["attempts"] += 1
            m["note"] = "grasp failed, retry"
            near = blk if (blk is not None and math.hypot(blk[0] - pa[0], blk[1] - pa[1]) < 0.06) else pa
            return pick(near, stage)
        if blk is not None and math.hypot(blk[0] - pa[0], blk[1] - pa[1]) > 0.06 and \
                math.hypot(blk[0] - m["centre"][0], blk[1] - m["centre"][1]) < 0.35:
            # block knocked elsewhere on the table: grasp it there
            m["attempts"] += 1
            m["note"] = "block moved, regrasp"
            return pick(blk, stage)
        o = m["origin"]
        if stage == "pick1":
            m["stage"] = "placed"
            c = m["centre"]
            return {"memory": m, "act": {"tool": "act_place_centre", "args": {"x": c[0], "y": c[1], "ztop": o[2]}}}
        m["stage"] = "returned"
        return {"memory": m, "act": {"tool": "act_place_back", "args": {"x": o[0], "y": o[1], "ztop": o[2]}}}

    if stage == "placed":
        c = m["centre"]
        if blk is None:
            m["note"] = "block not seen after placing; release"
            return {"memory": m, "act": {"tool": "act_release", "args": {}}}
        if math.hypot(blk[0] - c[0], blk[1] - c[1]) > 0.03:
            m["attempts"] += 1
            m["note"] = "block off centre; re-place"
            return pick(blk, "pick1")
        if m["button"] is None:
            return hold("button not perceived")
        b = m["button"]
        lh = m["left_home"]
        m["stage"] = "pressed"
        return {"memory": m, "act": {"tool": "act_press", "args": {
            "x": b[0], "y": b[1], "ztop": b[2], "home_xyz": lh[0], "home_q": lh[1]}}}

    if stage == "pressed":
        if last is not None and '"ok": false' in str(last) and m["attempts"] < 6:
            m["attempts"] += 1
            b = m["button"]
            lh = m["left_home"]
            m["note"] = "press motion error; repeat"
            return {"memory": m, "act": {"tool": "act_press", "args": {
                "x": b[0], "y": b[1], "ztop": b[2], "home_xyz": lh[0], "home_q": lh[1]}}}
        if blk is None:
            return hold("block not seen before regrasp")
        return pick(blk, "pick2")

    if stage == "returned":
        o = m["origin"]
        if blk is None:
            m["note"] = "block not seen after return; release"
            return {"memory": m, "act": {"tool": "act_release", "args": {}}}
        if math.hypot(blk[0] - o[0], blk[1] - o[1]) > 0.025 and m["attempts"] < 12:
            m["attempts"] += 1
            m["note"] = "block not on origin; regrasp"
            return pick(blk, "pick2")
        m["stage"] = "done"
        return hold("block back on origin mat")

    return hold("done")
