"""GatherAndRecall: put every cube on the disc, remember the lamp flash colour, press the matching button.

Stages (memory.stage): perceive -> pick -> place -> (repeat for each cube) -> wait -> press -> done.
Every decision scans all frames produced since the previous action for the lamp colour, so a
flash that happens in the middle of a motion is never missed."""

GRASP_BELOW_TOP = 0.018   # TCP z below cube top face for the grasp (cube 4 cm, top ~0.041)
PLACE_ABOVE_DISC = 0.032  # TCP z above disc surface at release (cube centre ~TCP)
CARRY_Z = 0.20            # transport height
PRESS_DEPTH = 0.030       # how far below the cap top the press target is
STEP_LIMIT = 200          # native step limit of the task


def _f(v, nd=4):
    return [round(float(x), nd) for x in v]


def _scan_lamp(frames, bbox, mem):
    """Update flash memory from a list of frames."""
    for fr in frames:
        state, n = _lamp_state(fr, bbox)
        st = int(fr.step)
        if state != "off":
            if mem["flash_color"] is None:
                mem["flash_color"] = state
                mem["flash_start"] = st
                mem["flash_votes"] = []
            if len(mem["flash_votes"] or []) < 40:
                mem["flash_votes"] = (mem["flash_votes"] or []) + [state]
            mem["flash_last_on"] = st
            mem["flash_ended"] = False
        else:
            if mem["flash_color"] is not None and mem["flash_last_on"] is not None and st > mem["flash_last_on"]:
                mem["flash_ended"] = True
    votes = mem["flash_votes"] or []
    if votes:
        counts = {c: votes.count(c) for c in set(votes)}
        mem["flash_color"] = max(counts, key=lambda c: counts[c])
    return mem


def perceive_decide(obs, goal, memory, last):
    """Plan the next GatherAndRecall stage from the latest frames and episode memory."""
    mem = dict(memory)
    cur = obs.current
    step = int(cur.step)
    mem["last_outcome"] = (str(last)[:600] if last is not None else None)
    if mem["stage"] is None:
        sc = _scene(cur)
        mem["ref_rpy"] = _f(cur.tcp[3:6])
        mem["cubes"] = [_f(c["pos"]) for c in sc["cubes"]]
        mem["disc"] = _f(sc["disc"]["pos"]) if sc["disc"] else None
        mem["button_pos"] = [_f(b["pos"]) for b in sc["buttons"]]
        mem["button_colors"] = [b["color"] for b in sc["buttons"]]
        mem["lamp_bbox"] = sc["lamp"]["bbox"] if sc["lamp"] else None
        mem["flash_color"] = None
        mem["flash_start"] = None
        mem["flash_last_on"] = None
        mem["flash_ended"] = False
        mem["flash_votes"] = []
        mem["cube_index"] = 0
        mem["pick_attempts"] = 0
        mem["press_attempts"] = 0
        mem["notes"] = "scene: %d cubes, disc=%s, buttons=%s, lamp=%s" % (
            len(mem["cubes"]), mem["disc"], mem["button_colors"], mem["lamp_bbox"])
        ok = bool(mem["cubes"]) and mem["disc"] is not None and len(mem["button_pos"]) >= 1 and mem["lamp_bbox"] is not None
        mem["stage"] = "pick" if ok else "perceive"
        mem["init_tries"] = 1
    elif mem["stage"] == "perceive":
        # retry scene detection on the newest frame
        sc = _scene(cur)
        if sc["cubes"] and not mem["cubes"]:
            mem["cubes"] = [_f(c["pos"]) for c in sc["cubes"]]
        if sc["disc"] and mem["disc"] is None:
            mem["disc"] = _f(sc["disc"]["pos"])
        if sc["buttons"] and not mem["button_pos"]:
            mem["button_pos"] = [_f(b["pos"]) for b in sc["buttons"]]
            mem["button_colors"] = [b["color"] for b in sc["buttons"]]
        if sc["lamp"] and mem["lamp_bbox"] is None:
            mem["lamp_bbox"] = sc["lamp"]["bbox"]
        mem["init_tries"] = (mem["init_tries"] or 0) + 1
        ok = bool(mem["cubes"]) and mem["disc"] is not None and len(mem["button_pos"]) >= 1
        if ok:
            mem["stage"] = "pick"
    if mem["lamp_bbox"] is not None:
        mem = _scan_lamp(obs.frames, mem["lamp_bbox"], mem)

    stage = mem["stage"]
    ref = mem["ref_rpy"]
    fingers = [float(v) for v in cur.fingers]
    gap = fingers[0] + fingers[1]

    # ---- transitions based on the measured result of the previous act
    if stage == "pick_check":
        # previous act was act_pick: holding a cube if the fingers stopped on it
        if 0.02 < gap < 0.07:
            stage = "place"
        else:
            mem["pick_attempts"] = (mem["pick_attempts"] or 0) + 1
            sc = _scene(cur, mem["button_pos"])
            if sc["cubes"]:
                d = mem["disc"]
                off = [c for c in sc["cubes"] if math.hypot(c["pos"][0] - d[0], c["pos"][1] - d[1]) > 0.08]
                if off:
                    mem["cubes"][mem["cube_index"]] = _f(off[0]["pos"])
            stage = "pick" if mem["pick_attempts"] < 3 else "wait"
    elif stage == "place_check":
        mem["cube_index"] = (mem["cube_index"] or 0) + 1
        stage = "pick" if mem["cube_index"] < len(mem["cubes"]) else "wait"
        mem["pick_attempts"] = 0
    if stage == "wait":
        if mem["flash_color"] is not None and mem["flash_ended"]:
            stage = "press"
        elif step >= STEP_LIMIT - 30:
            stage = "press"
    mem["stage"] = stage

    if stage == "perceive":
        act = {"tool": "act_observe", "args": {"steps": 1}}
    elif stage == "pick":
        c = mem["cubes"][mem["cube_index"]]
        mem["stage"] = "pick_check"
        act = {"tool": "act_pick", "args": {"x": c[0], "y": c[1], "z_top": c[2], "ref_rpy": ref}}
    elif stage == "place":
        d = mem["disc"]
        mem["stage"] = "place_check"
        act = {"tool": "act_place", "args": {"x": d[0], "y": d[1], "z_surface": d[2], "ref_rpy": ref}}
    elif stage == "wait":
        budget = max(1, min(60, STEP_LIMIT - 30 - step))
        act = {"tool": "act_wait_flash", "args": {"bbox": mem["lamp_bbox"], "max_steps": budget,
                                                   "seen": mem["flash_color"] is not None, "ref_rpy": ref}}
    elif stage == "press":
        colors = mem["button_colors"]
        target = None
        if mem["flash_color"] in colors:
            target = mem["button_pos"][colors.index(mem["flash_color"])]
            mem["notes"] = "pressing %s (flash at step %s)" % (mem["flash_color"], mem["flash_start"])
        else:
            target = mem["button_pos"][len(colors) // 2]
            mem["notes"] = "no flash seen; guessing middle button"
        mem["press_attempts"] = (mem["press_attempts"] or 0) + 1
        depth = min(0.045, PRESS_DEPTH + 0.0075 * (mem["press_attempts"] - 1))
        act = {"tool": "act_press", "args": {"x": target[0], "y": target[1], "z_top": target[2],
                                              "depth": depth, "ref_rpy": ref}}
    else:
        act = {"tool": "act_observe", "args": {"steps": 5}}
    return {"memory": mem, "act": act}


def act_observe(robot, steps=1):
    """Hold still for `steps` native steps (gripper unchanged=open); returns the TCP."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"tcp": _f(robot.now().tcp)}


def act_pick(robot, x, y, z_top, ref_rpy=None):
    """Grasp the cube whose top-face centre is (x, y, z_top): hover, refine with the wrist
    camera, descend, close, and lift to 0.15 m. Returns finger gap and measured poses."""
    out = {}
    hover = [x, y, z_top + 0.07]
    r1 = _move_to(robot, hover, ref_rpy, 1.0, tol=0.012, max_steps=35, settle_v=0.012)
    out["hover"] = r1
    ref = _wrist_cube(robot.now(), [x, y], z_top)
    if (ref is not None and math.hypot(ref["pos"][0] - x, ref["pos"][1] - y) < 0.03
            and max(ref["ext"]) < 0.05):
        out["wrist_refine"] = [_f(ref["pos"]), ref["n"]]
        x, y = ref["pos"][0], ref["pos"][1]
    zg = z_top - GRASP_BELOW_TOP
    out["dry_grasp"] = _check_pose(robot, [x, y, zg], ref_rpy)
    pre = [x, y, zg + 0.03]
    out["pre"] = _move_to(robot, pre, ref_rpy, 1.0, tol=0.006, max_steps=20, vmax=0.8, settle_v=0.01)
    out["grasp"] = _move_to(robot, [x, y, zg], ref_rpy, 1.0, tol=0.004, max_steps=20, vmax=0.5, settle_v=0.003)
    fing = _hold(robot, ref_rpy, -1.0, 6, [x, y, zg])
    out["fingers_closed"] = _f(fing)
    out["lift"] = _move_to(robot, [x, y, 0.15], ref_rpy, -1.0, tol=0.02, max_steps=20, settle_v=0.05)
    f = [float(v) for v in robot.now().fingers]
    out["gap_after_lift"] = round(f[0] + f[1], 4)
    out["holding"] = 0.02 < f[0] + f[1] < 0.07
    return out


def act_place(robot, x, y, z_surface, ref_rpy=None):
    """Carry the held cube above the disc centre (x, y), lower it, release, and lift away."""
    out = {}
    out["carry"] = _move_to(robot, [x, y, CARRY_Z], ref_rpy, -1.0, tol=0.012, max_steps=40, vmax=1.0, settle_v=0.01)
    zp = z_surface + PLACE_ABOVE_DISC
    out["dry_place"] = _check_pose(robot, [x, y, zp], ref_rpy)
    out["lower"] = _move_to(robot, [x, y, zp], ref_rpy, -1.0, tol=0.005, max_steps=25, vmax=0.7, settle_v=0.004)
    fing = _hold(robot, ref_rpy, 1.0, 5, [x, y, zp])
    out["fingers_open"] = _f(fing)
    out["retreat"] = _move_to(robot, [x, y, 0.15], ref_rpy, 1.0, tol=0.02, max_steps=15, settle_v=0.05)
    return out


def act_wait_flash(robot, bbox, max_steps=40, seen=False, ref_rpy=None):
    """Hold the arm still while watching the lamp; stop 2 steps after a flash has ended
    (or when max_steps is used). Returns observed lamp colours with their steps."""
    target = [float(v) for v in robot.now().tcp[:3]]
    on = []
    off_after = 0
    was_seen = bool(seen)
    n = 0
    while n < int(max_steps):
        _servo(robot, target, ref_rpy, 1.0, 1.0, 0.5)
        n += 1
        fr = robot.now()
        state, cnt = _lamp_state(fr, bbox)
        if state != "off":
            on.append([int(fr.step), state])
            was_seen = True
            off_after = 0
        elif was_seen:
            off_after += 1
            if off_after >= 2:
                break
    return {"steps": n, "on": on[:20], "seen": was_seen}


def act_press(robot, x, y, z_top, depth=0.03, ref_rpy=None):
    """Close the gripper, move above the button cap (x, y, z_top), push down to z_top-depth,
    hold briefly and lift. The episode ends on native success."""
    out = {}
    out["above"] = _move_to(robot, [x, y, z_top + 0.07], ref_rpy, -1.0, tol=0.008, max_steps=35, settle_v=0.01)
    out["align"] = _move_to(robot, [x, y, z_top + 0.025], ref_rpy, -1.0, tol=0.005, max_steps=15, vmax=0.6, settle_v=0.006)
    out["dry_push"] = _check_pose(robot, [x, y, z_top - depth], ref_rpy)
    out["push"] = _move_to(robot, [x, y, z_top - depth], ref_rpy, -1.0, tol=0.004, max_steps=15, vmax=0.4, settle_v=0.002)
    _hold(robot, ref_rpy, -1.0, 4, [x, y, z_top - depth])
    out["lift"] = _move_to(robot, [x, y, z_top + 0.07], ref_rpy, -1.0, tol=0.02, max_steps=12, settle_v=0.05)
    return out
