"""cover_blocks policy: remember block colours before covering, cover left->right, uncover in the instructed colour order."""

Q_DOWN = [0.5, -0.5, 0.5, 0.5]          # gripper pointing down (expert grasp orientation, both arms)
GRASP_ABOVE_TOP = 0.129                 # EE z above lid handle top when grasping (expert: 0.975 vs 0.845)
PLACE_EXTRA = 0.008                     # EE z offset when releasing a held lid (expert: 0.982-0.983)
APPROACH = 0.045                        # pre-grasp / pre-place height above grasp/place pose
RETREAT = 0.10                          # lift after release
NOMINAL_HOME = {"left": [-0.298, -0.314, 0.942, 0.7, 0.0, 0.0, 0.714],
                "right": [0.306, -0.313, 0.941, 0.704, 0.0, 0.0, 0.711]}
NOMINAL_LID_TOP = 0.846
MAX_TRIES = 3


def _parse_goal(goal):
    """Cover direction ('ltr' or 'rtl') and uncover colour order from the instruction."""
    g = (goal or "").lower()
    direction = "rtl" if "right to left" in g else "ltr"
    tail = g
    for key in ("uncover", "sequence", "order"):
        k = g.find(key)
        if k >= 0:
            tail = g[k:]
            break
    found = []
    for c in ("red", "green", "blue"):
        k = tail.find(c)
        if k >= 0:
            found.append((k, c))
    found.sort()
    order = [c for _, c in found]
    for c in ("red", "green", "blue"):
        if c not in order:
            order.append(c)
    return direction, order


def _arm_for_x(x):
    return "right" if x > 0.1 else "left"


def _nearest(points, xy, max_d):
    best, bd = None, max_d
    for p in points:
        d = math.hypot(p[0] - xy[0], p[1] - xy[1])
        if d < bd:
            best, bd = p, d
    return best


def _blank_memory():
    return {"phase": "init", "colors": [], "block_pos": [], "lid_home": [], "uncover_order": [],
            "index": 0, "tries": 0, "pending": "", "home_poses": [], "log": ""}


def perceive_decide(obs, goal, memory, last):
    """Perceive the head RGB-D view, update episode memory and choose the next lid move.

    Phases: init (record block colours/positions and lid positions while blocks are visible) ->
    cover (lid k onto the k-th block from the viewer's left) -> uncover (lids lifted off in the
    instructed colour order and returned to their original spots) -> done (hold).
    """
    mem = _blank_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    f = obs.current
    hold = {"tool": "act_hold", "args": {}}
    notes = []

    if mem["phase"] == "init":
        direction, order = _parse_goal(goal)
        blocks = _detect_blocks(f)
        vis = [c for c in BLOCK_COLORS if blocks[c]["pos"] is not None]
        if len(vis) < 3:
            notes.append("init: only %s visible" % vis)
        vis.sort(key=lambda c: blocks[c]["u"], reverse=(direction == "rtl"))
        lids = _detect_lids(f)
        homes = []
        for c in vis:
            bp = blocks[c]["pos"]
            # lid that belongs to this block: the free lid nearest in x (lids stand in a row behind)
            cand = [l for l in lids if math.hypot(l[0] - bp[0], l[1] - bp[1]) > 0.06 and l not in homes]
            best = None
            for l in cand:
                if best is None or abs(l[0] - bp[0]) < abs(best[0] - bp[0]):
                    best = l
            if best is None:
                best = [bp[0], bp[1] + 0.13, NOMINAL_LID_TOP]
                notes.append("lid for %s not seen; nominal" % c)
            homes.append([float(v) for v in best])
        mem["colors"] = vis
        mem["block_pos"] = [[float(v) for v in blocks[c]["pos"]] for c in vis]
        mem["lid_home"] = homes
        mem["uncover_order"] = [c for c in order if c in vis]
        mem["home_poses"] = [[float(v) for v in f.left_ee_pose], [float(v) for v in f.right_ee_pose]]
        mem["phase"] = "cover"
        mem["index"] = 0
        mem["tries"] = 0
        notes.append("colors L->R %s, uncover %s" % (vis, mem["uncover_order"]))

    # verify the previous move with the current view
    if mem["pending"]:
        kind, slot = mem["pending"].split(":")
        slot = int(slot)
        c = mem["colors"][slot]
        bp = mem["block_pos"][slot]
        n = _color_count(f, c, bp[:2], 0.06)
        ok = (n < 25) if kind == "cover" else (n > 120)
        notes.append("check %s %s n=%d ok=%s" % (kind, c, n, ok))
        if ok:
            mem["index"] = int(mem["index"]) + 1
            mem["tries"] = 0
        else:
            mem["tries"] = int(mem["tries"]) + 1
            if mem["tries"] >= MAX_TRIES:
                notes.append("giving up on %s %s" % (kind, c))
                mem["index"] = int(mem["index"]) + 1
                mem["tries"] = 0
        mem["pending"] = ""

    if mem["phase"] == "cover" and int(mem["index"]) >= len(mem["colors"]):
        mem["phase"] = "uncover"
        mem["index"] = 0
        mem["tries"] = 0
    if mem["phase"] == "uncover" and int(mem["index"]) >= len(mem["uncover_order"]):
        mem["phase"] = "done"

    act = hold
    if mem["phase"] in ("cover", "uncover"):
        lids = _detect_lids(f)
        if mem["phase"] == "cover":
            slot = int(mem["index"])
            bp = mem["block_pos"][slot]
            home = mem["lid_home"][slot]
            free = [l for l in lids if all(math.hypot(l[0] - b[0], l[1] - b[1]) > 0.05 for b in mem["block_pos"])]
            src = _nearest(free, home, 0.12) or home
            dst = [bp[0], bp[1], float(src[2])]
        else:
            c = mem["uncover_order"][int(mem["index"])]
            slot = mem["colors"].index(c)
            bp = mem["block_pos"][slot]
            src = _nearest(lids, bp, 0.06) or [bp[0], bp[1], NOMINAL_LID_TOP]
            home = mem["lid_home"][slot]
            dst = [home[0], home[1], float(src[2])]
        arm = _arm_for_x(0.5 * (src[0] + dst[0]))
        hp = mem["home_poses"][0 if arm == "left" else 1] if mem["home_poses"] else NOMINAL_HOME[arm]
        act = {"tool": "act_move_lid", "args": {
            "arm": arm,
            "pick": [float(src[0]), float(src[1]), float(src[2]) + GRASP_ABOVE_TOP],
            "place": [float(dst[0]), float(dst[1]), float(dst[2]) + GRASP_ABOVE_TOP + PLACE_EXTRA],
            "home": [float(v) for v in hp]}}
        mem["pending"] = "%s:%d" % (mem["phase"], slot)
        notes.append("%s slot %d (%s) with %s" % (mem["phase"], slot, mem["colors"][slot], arm))

    mem["log"] = (str(mem["log"]) + " | " + "; ".join(notes))[-1500:]
    return {"memory": mem, "act": act}


def _step(robot, arm, pose, grip):
    """One native step moving `arm` to pose (7) with gripper `grip`; the other arm holds."""
    now = robot.now()
    lp = [float(v) for v in now.left_ee_pose]
    rp = [float(v) for v in now.right_ee_pose]
    lg = float(now.left_gripper)
    rg = float(now.right_gripper)
    if arm == "left":
        lp, lg = list(pose), grip
    else:
        rp, rg = list(pose), grip
    robot.step(*lp, lg, *rp, rg)


def act_move_lid(robot, arm, pick, place, home):
    """Pick a lid with `arm` ('left'/'right') at EE position `pick` [x,y,z] (gripper pointing down),
    carry it at a safe height, lower it to EE position `place`, release, retreat up and return to
    the 7-value `home` pose. Returns the steps completed and any motion error."""
    px, py, pz = [float(v) for v in pick]
    qx, qy, qz = [float(v) for v in place]
    carry = max(pz, qz) + 0.05
    plan = [
        ("pre_pick", [px, py, pz + APPROACH] + Q_DOWN, 1.0),
        ("pick", [px, py, pz] + Q_DOWN, 1.0),
        ("close", [px, py, pz] + Q_DOWN, 0.0),
        ("close2", [px, py, pz] + Q_DOWN, 0.0),
        ("lift", [px, py, carry] + Q_DOWN, 0.0),
        ("carry", [qx, qy, carry] + Q_DOWN, 0.0),
        ("lower", [qx, qy, qz] + Q_DOWN, 0.0),
        ("open", [qx, qy, qz] + Q_DOWN, 1.0),
        ("open2", [qx, qy, qz] + Q_DOWN, 1.0),
        ("retreat", [qx, qy, qz + RETREAT] + Q_DOWN, 1.0),
        ("home", [float(v) for v in home], 1.0),
    ]
    done = []
    for name, pose, grip in plan:
        try:
            _step(robot, arm, pose, grip)
        except Exception as e:  # MotionError: arm stayed still; EpisodeEnd is not an Exception
            if name in ("close", "close2", "open", "open2"):
                # gripper-only step: retry holding the measured pose instead of the nominal target
                now = robot.now()
                cur = [float(v) for v in (now.left_ee_pose if arm == "left" else now.right_ee_pose)]
                try:
                    _step(robot, arm, cur, grip)
                except Exception as e2:
                    done.append(name + "!")
                    continue
                done.append(name + "~")
                continue
            return {"done": done, "error": "%s: %s" % (name, str(e)[:200])}
        done.append(name)
    now = robot.now()
    return {"done": done, "left_gripper": float(now.left_gripper), "right_gripper": float(now.right_gripper)}


def act_hold(robot):
    """Hold both arms at their current poses for one native step."""
    now = robot.now()
    robot.step(*[float(v) for v in now.left_ee_pose], float(now.left_gripper),
               *[float(v) for v in now.right_ee_pose], float(now.right_gripper))
    return {"held": True}
