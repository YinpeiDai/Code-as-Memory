"""place_block_mat policy: move each red block (left to right) from its blue
mat to the green mat and back to its own blue mat.

Decision loop (arms are at home whenever a 'pick' decision perceives):
  pick decision : head RGB-D -> blocks; locate the current block; if it is at
                  its stage target advance the stage, else act_pick.
  place decision: if the previous act_pick reported holding -> act_place at
                  the stage target, else act_home and retry.
"""

Q_DOWN = [0.5, -0.5, 0.5, 0.5]   # expert top-down grasp orientation (w,x,y,z), yaw 0
GRASP_DZ = 0.162                 # ee z above table at grasp (expert: 0.903 with table 0.741)
PLACE_DZ = 0.172                 # ee z above table at release (expert: 0.913)
APPROACH_DZ = 0.241              # pre-grasp ee z above table (expert: ~0.982)
CARRY_DZ = 0.259                 # carry / lift ee z above table (expert: ~1.000)
AT_TOL = 0.03                    # block within this xy distance of a target counts as there
SAME_TOL = 0.035                 # detection belongs to a resting block's origin
YAW_DEADBAND = 7.0               # degrees; head yaw estimate noise measured <= 6.4 deg
MAX_FAILS = 3
MAX_LOST = 3
HOLD_PIXELS = 3000               # red wrist pixels nearer than 100 mm when holding


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _q_yaw(yaw_deg):
    """Top-down gripper orientation rotated by yaw about world z."""
    h = math.radians(yaw_deg) / 2.0
    return _qmul([math.cos(h), 0.0, 0.0, math.sin(h)], Q_DOWN)


def _arm_for(x):
    return "left" if x < 0 else "right"


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _empty_memory():
    return {"origins": None, "idx": 0, "stage": "to_green", "phase": "pick",
            "arm": "", "green": None, "home_left": None, "home_right": None,
            "fails": 0, "lost": 0, "note": ""}


def _locate_current(blocks, origins, idx, expect):
    """Detection that is not some other block resting on its own origin,
    nearest to the expected location."""
    cands = []
    for b in blocks:
        p = [b["x"], b["y"]]
        other = False
        for j in range(len(origins)):
            if j != idx and _dist(p, origins[j]) < SAME_TOL:
                other = True
        if not other:
            cands.append(b)
    if not cands:
        return None
    cands.sort(key=lambda b: _dist([b["x"], b["y"]], expect))
    return cands[0]


def perceive_decide(obs, goal, memory, last):
    """Perceive with the head RGB-D camera and choose the next pick/place act."""
    mem = _empty_memory()
    for k in mem:
        if memory.get(k) is not None:
            mem[k] = memory[k]
    f = obs.current
    if mem["home_left"] is None:
        mem["home_left"] = [float(v) for v in f.left_ee_pose]
        mem["home_right"] = [float(v) for v in f.right_ee_pose]
    homes = {"home_left": mem["home_left"], "home_right": mem["home_right"]}
    tz = TABLE_Z

    held = isinstance(last, str) and ('"holding": true' in last)
    if mem["phase"] == "place" or held:
        mem["phase"] = "pick"
        arm = mem["arm"]
        if held and int(mem["fails"]) <= MAX_FAILS:
            idx = int(mem["idx"])
            if mem["stage"] == "to_green":
                tgt = mem["green"]
            else:
                tgt = mem["origins"][idx]
            if last.startswith("act_place"):
                mem["fails"] = int(mem["fails"]) + 1   # carry could not be planned
            else:
                mem["fails"] = 0
            mem["note"] = "place block %d %s at %.3f,%.3f" % (idx, mem["stage"], tgt[0], tgt[1])
            return {"memory": mem, "act": {"tool": "act_place", "args": {
                "arm": arm, "x": float(tgt[0]), "y": float(tgt[1]),
                "z_place": tz + PLACE_DZ, "z_carry": tz + CARRY_DZ, **homes}}}
        mem["fails"] = 0
        if not (isinstance(last, str) and last.startswith("act_pick") and '"holding": false' in last):
            mem["note"] = "unclear act outcome or carry failed repeatedly; returning home"
            return {"memory": mem, "act": {"tool": "act_home", "args": homes}}
        # failed grasp: act_pick already returned the arm home -> re-perceive now

    blocks = _detect_blocks(f)
    if mem["origins"] is None:
        green0 = _detect_green(f)
        start = [b for b in blocks if b["y"] > green0[1] + 0.06]
        if not start:
            mem["note"] = "no blocks detected on the blue mats yet; observing"
            return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
        mem["origins"] = [[round(b["x"], 4), round(b["y"], 4)] for b in start]
        mem["green"] = green0
        mem["idx"] = 0
        mem["stage"] = "to_green"
    origins = mem["origins"]
    green = mem["green"]

    for _ in range(4):
        idx = int(mem["idx"])
        if idx >= len(origins):
            mem["stage"] = "done"
            mem["note"] = "all blocks cycled"
            return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
        target = green if mem["stage"] == "to_green" else origins[idx]
        expect = origins[idx] if mem["stage"] == "to_green" else green
        cur = _locate_current(blocks, origins, idx, expect)
        if cur is not None and _dist([cur["x"], cur["y"]], target) < AT_TOL:
            # stage complete
            mem["fails"] = 0
            mem["lost"] = 0
            if mem["stage"] == "to_green":
                mem["stage"] = "to_origin"
            else:
                mem["stage"] = "to_green"
                mem["idx"] = idx + 1
            continue
        if cur is None:
            mem["lost"] = int(mem["lost"]) + 1
            if int(mem["lost"]) > MAX_LOST:
                # the block cannot be seen any more: continue with the next one
                mem["idx"] = idx + 1
                mem["stage"] = "to_green"
                mem["lost"] = 0
                mem["fails"] = 0
                continue
            mem["note"] = "block %d not found; re-observing from home" % idx
            return {"memory": mem, "act": {"tool": "act_home", "args": homes}}
        mem["lost"] = 0
        arm = _arm_for(origins[idx][0])
        yaw = cur["yaw"]
        if abs(yaw) < YAW_DEADBAND:
            yaw = 0.0
        ztop = min(max(cur["z_top"], tz + 0.035), tz + 0.05)
        mem["arm"] = arm
        mem["phase"] = "place"
        mem["note"] = "pick block %d (%s) at %.3f,%.3f yaw %.1f" % (idx, mem["stage"], cur["x"], cur["y"], yaw)
        return {"memory": mem, "act": {"tool": "act_pick", "args": {
            "arm": arm, "x": float(cur["x"]), "y": float(cur["y"]), "yaw_deg": float(yaw),
            "z_grasp": ztop + (GRASP_DZ - 0.042), "z_approach": tz + APPROACH_DZ,
            "z_carry": tz + CARRY_DZ, **homes}}}
    mem["note"] = "stage loop exhausted"
    return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
