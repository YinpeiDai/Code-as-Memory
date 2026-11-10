"""swap_T policy: perceive both T-blocks once from the static head camera, grasp
both top-down at the same block-frame point, park both arms at home, then place
each block at the other block's initial pose (position + yaw), and finally
verify with the head camera and re-pick / re-place any block that is off."""

MAX_CORR = 4          # correction pick-and-place rounds after the swap
MAX_REGRASP = 3       # retries of a failed initial grasp
CORR_POS = 0.008      # a block counts as "off" beyond this position error (m)
CORR_YAW = 0.07       # ... or this yaw error (rad, ~4 deg)


def _det(frame, color):
    d = _detect_block(frame, color, "head")
    if d is None or d["cover"] < 0.6:
        return None
    return d


def _pose3(d):
    return [round(float(d["x"]), 5), round(float(d["y"]), 5), round(float(d["yaw"]), 5)]


def _err(d, target):
    dp = math.hypot(d["x"] - target[0], d["y"] - target[1])
    dy = abs(_wrap(d["yaw"] - target[2]))
    return dp, dy


def perceive_decide(obs, goal, memory, last):
    """Swap-T state machine. Memory: phase, initial poses of both blocks, which arm
    holds which block, grasp flips and counters. Returns the next act."""
    m = dict(memory)
    for k, v in (("phase", None), ("red_init", None), ("blue_init", None),
                 ("left_holds", ""), ("right_holds", ""), ("left_flip", 0),
                 ("right_flip", 0), ("n_corr", 0), ("n_retry", 0), ("notes", "")):
        if m.get(k) is None:
            m[k] = v
    if m["phase"] is None:
        m["phase"] = "start"
    f = obs.current
    phase = m["phase"]

    def out(tool, args):
        return {"memory": m, "act": {"tool": tool, "args": args}}

    if phase == "start":
        r = _det(f, "red")
        b = _det(f, "blue")
        if r is None or b is None:
            m["n_retry"] = int(m["n_retry"]) + 1
            m["notes"] = "start: block not detected (red=%s blue=%s)" % (r is not None, b is not None)
            return out("act_observe", {})
        m["red_init"] = _pose3(r)
        m["blue_init"] = _pose3(b)
        m["left_holds"], m["right_holds"] = "red", "blue"
        m["phase"] = "grasped"
        m["notes"] = "red %s blue %s" % (m["red_init"], m["blue_init"])
        return out("act_grasp", {"left": m["red_init"], "right": m["blue_init"]})

    if phase == "grasped":
        # remember the gripper-yaw flips actually used by the last act_grasp
        if last and last.startswith("act_grasp"):
            for arm in ("left", "right"):
                key = '"' + arm + '_flip": '
                i = last.find(key)
                if i >= 0:
                    m[arm + "_flip"] = 1 if last[i + len(key):i + len(key) + 1] == "1" else 0
        m["phase"] = "homed"
        return out("act_home", {"left": True, "right": True})

    if phase == "homed":
        # both arms parked; a block still seen on the table was not picked up
        r = _det(f, "red")
        b = _det(f, "blue")
        left_arg = _pose3(r) if r is not None else None
        right_arg = _pose3(b) if b is not None else None
        if (left_arg or right_arg) and int(m["n_retry"]) < MAX_REGRASP:
            m["n_retry"] = int(m["n_retry"]) + 1
            m["notes"] = "regrasp left=%s right=%s" % (left_arg, right_arg)
            m["phase"] = "grasped"
            return out("act_grasp", {"left": left_arg, "right": right_arg})
        m["phase"] = "left_placed"
        tgt = m["blue_init"] if m["left_holds"] == "red" else m["red_init"]
        return out("act_place", {"arm": "left", "target": tgt, "flip": int(m["left_flip"])})

    if phase == "left_placed":
        m["phase"] = "verify"
        tgt = m["red_init"] if m["right_holds"] == "blue" else m["blue_init"]
        return out("act_place", {"arm": "right", "target": tgt, "flip": int(m["right_flip"])})

    # verify / correct
    r = _det(f, "red")
    b = _det(f, "blue")
    cands = []
    for d, tgt, nm in ((r, m["blue_init"], "red"), (b, m["red_init"], "blue")):
        if d is None or tgt is None:
            continue
        dp, dy = _err(d, tgt)
        cands.append((dp / CORR_POS + dy / CORR_YAW, dp, dy, nm, d, tgt))
    cands.sort(key=lambda c: -c[0])
    m["notes"] = "; ".join("%s off %.1fmm %.1fdeg" % (c[3], c[1] * 1000, math.degrees(c[2])) for c in cands)
    if cands and int(m["n_corr"]) < MAX_CORR:
        c = cands[0]
        fine = c[0] <= 1.0
        if not fine or int(m["n_corr"]) < 1:
            m["n_corr"] = int(m["n_corr"]) + 1
            d = c[4]
            arm = "left" if d["x"] < 0 else "right"
            m["phase"] = "verify"
            return out("act_repick", {"arm": arm, "source": _pose3(d), "target": c[5]})
    return out("act_observe", {})


def act_observe(robot):
    """Hold both arms for one native step (used while waiting / after finishing)."""
    L, lg, R, rg = _cur(robot)
    _send(robot, L, lg, R, rg)
    return {"observed": True}


def act_grasp(robot, left=None, right=None):
    """Top-down grasp with either/both arms. left/right = [x, y, block_yaw] of the
    block origin (middle of the T along its stem axis) or None. Ends lifted
    LIFT_DZ above the grasp height with the gripper closed."""
    log = []
    tg = {}
    flips = {"left_flip": 0, "right_flip": 0}
    for arm, p in (("left", left), ("right", right)):
        if p is not None:
            fl = _default_flip(p[2])
            flips[arm + "_flip"] = fl
            tg[arm] = [p[0], p[1], _gripper_yaw(p[2], fl)]
    if not tg:
        return {"error": "nothing to grasp"}

    def pose(arm, z):
        q = tg[arm]
        return _pose(q[0], q[1], z, q[2])

    zpre = EE_GRASP_Z + PRE_DZ
    L = pose("left", zpre) if "left" in tg else None
    R = pose("right", zpre) if "right" in tg else None
    e = _move(robot, L, 1.0 if L else None, R, 1.0 if R else None, log=log)
    # an arm that missed the pre-grasp (plan failure): retry it alone, then with
    # the 180-deg flipped gripper yaw (the parallel grasp is symmetric)
    for arm in ("left", "right"):
        if arm not in tg or e[arm + "_err"] <= 0.01:
            continue
        for attempt in range(2):
            if attempt == 1:
                flips[arm + "_flip"] = 1 - flips[arm + "_flip"]
                tg[arm][2] = _wrap(tg[arm][2] + math.pi)
                log.append(arm + " pre-grasp missed, flipping gripper yaw")
            if arm == "left":
                e2 = _move(robot, pose("left", zpre), 1.0, None, None, log=log)
            else:
                e2 = _move(robot, None, None, pose("right", zpre), 1.0, log=log)
            e[arm + "_err"] = e2[arm + "_err"]
            if e2[arm + "_err"] <= 0.01:
                break
    L = pose("left", EE_GRASP_Z) if "left" in tg else None
    R = pose("right", EE_GRASP_Z) if "right" in tg else None
    e_d = _move(robot, L, 1.0 if L else None, R, 1.0 if R else None, log=log)
    _move(robot, L, 0.0 if L else None, R, 0.0 if R else None, correct=False, log=log)
    _move(robot, L, 0.0 if L else None, R, 0.0 if R else None, correct=False, log=log)
    L = pose("left", EE_GRASP_Z + LIFT_DZ) if "left" in tg else None
    R = pose("right", EE_GRASP_Z + LIFT_DZ) if "right" in tg else None
    _move(robot, L, 0.0 if L else None, R, 0.0 if R else None, correct=False, log=log)
    res = {"descend_err": e_d, "log": log[-6:]}
    for arm in tg:
        res[arm + "_flip"] = flips[arm + "_flip"]
    return res


def act_home(robot, left=True, right=True):
    """Move the selected arms to their home poses, keeping gripper openings."""
    log = []
    e = _move(robot, HOME_LEFT if left else None, None, HOME_RIGHT if right else None, None,
              correct=False, log=log)
    return {"home_err": e, "log": log[-4:]}


def _place_seq(robot, arm, target, flip, log):
    """Carry the held block over target=[x,y,block_yaw] via a high waypoint (so the
    carried block clears the other, already placed block), lower, release ~2 cm
    above the table, retreat straight up and return home."""
    yaw = _gripper_yaw(target[2], flip)
    top = _pose(target[0], target[1], EE_GRASP_Z + ROUTE_DZ, yaw)
    hi = _pose(target[0], target[1], EE_GRASP_Z + PRE_DZ + 0.01, yaw)
    lo = _pose(target[0], target[1], EE_GRASP_Z + PLACE_DZ, yaw)
    home = HOME_LEFT if arm == "left" else HOME_RIGHT

    def mv(p, g, correct=True):
        if arm == "left":
            return _move(robot, p, g, None, None, correct=correct, log=log)["left_err"]
        return _move(robot, None, None, p, g, correct=correct, log=log)["right_err"]

    e0 = mv(top, 0.0, correct=False)
    e1 = mv(hi, 0.0)
    e2 = mv(lo, 0.0)
    mv(lo, 1.0, correct=False)
    mv(top, 1.0, correct=False)
    mv(home, 1.0, correct=False)
    return {"route_err": e0, "pre_err": e1, "place_err": e2}


def act_place(robot, arm, target, flip=0):
    """Place the block held by `arm` ('left'/'right') so that its origin lands at
    target=[x, y, block_yaw]; flip=1 if it was grasped with the gripper yaw
    rotated by pi. Releases ~2 cm above the table and returns the arm home."""
    log = []
    r = _place_seq(robot, arm, target, flip, log)
    r["log"] = log[-6:]
    return r


def act_repick(robot, arm, source, target):
    """Correction: grasp the block at source=[x,y,yaw] with `arm`, lift, and place
    it at target=[x,y,yaw]; the arm ends at home with the gripper open."""
    log = []
    fl = _default_flip(source[2])
    yaw = _gripper_yaw(source[2], fl)
    hi = _pose(source[0], source[1], EE_GRASP_Z + PRE_DZ, yaw)
    lo = _pose(source[0], source[1], EE_GRASP_Z, yaw)
    up = _pose(source[0], source[1], EE_GRASP_Z + ROUTE_DZ, yaw)
    if arm == "left":
        _move(robot, hi, 1.0, None, None, log=log)
        e = _move(robot, lo, 1.0, None, None, log=log)
        _move(robot, lo, 0.0, None, None, correct=False, log=log)
        _move(robot, lo, 0.0, None, None, correct=False, log=log)
        _move(robot, up, 0.0, None, None, correct=False, log=log)
    else:
        _move(robot, None, None, hi, 1.0, log=log)
        e = _move(robot, None, None, lo, 1.0, log=log)
        _move(robot, None, None, lo, 0.0, correct=False, log=log)
        _move(robot, None, None, lo, 0.0, correct=False, log=log)
        _move(robot, None, None, up, 0.0, correct=False, log=log)
    # the target grasp yaw keeps the same block-in-hand relation (flip fl)
    r = _place_seq(robot, arm, target, fl, log)
    r["grasp_err"] = e[arm + "_err"]
    r["log"] = log[-6:]
    return r
