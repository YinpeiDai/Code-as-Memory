"""T18 ChocolateButterCabinet: move chocolate then butter from cabinet1's top onto cabinet2's top.

Stage machine (memory.stage):
  None/'start' -> agentview RGB-D: find both cabinet tops; cabinet1 = the one with objects on it.
  'survey'     -> wrist RGB-D above cabinet1: find objects; pick chocolate first, else butter.
  'picked'     -> if fingers hold something, carry it to its slot on cabinet2 and release.
  'placed'     -> go back above cabinet1 and survey again (or verify when cabinet1 is empty).
  'verify'     -> nothing left on cabinet1: hold still for the checker; re-survey later.
"""

HOVER_DZ = 0.26          # survey hover: TCP this far above the cabinet top (~0.92 m)
HOVER_BACK = 0.05        # TCP x offset (-x) so the wrist camera (+0.05 m in x) centres on the cabinet
TRAVEL_DZ = 0.24         # carrying height above cabinet top
GRASP_DZ = 0.006         # grasp target height above top (fingertips stall on the top at ~+0.010)
RELEASE_DZ = 0.008       # place target height above top (stall on contact)
SLOT_DY = {"chocolate": 0.065, "butter": -0.065}   # slot offsets on cabinet2 (demo evidence: +0.066 / -0.07)
SLOT_DX = 0.005
HOLD_MIN_WIDTH = 0.010   # finger gap (m) above which something is held (choc ~0.054, butter ~0.038)
HOLD_MAX_WIDTH = 0.072   # fully open is ~0.080: then nothing was closed on
CAB_SIZE = (0.25, 0.20)


def _mem(memory, key, default):
    v = memory.get(key) if isinstance(memory, dict) else None
    return default if v is None else v


def _box_of(c):
    return [[float(c["x0"]), float(c["y0"])], [float(c["x1"]), float(c["y1"])]]


def _cab_from_box(box):
    (x0, y0), (x1, y1) = box
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2}


def _nominal_box(cx, cy):
    return [[cx - CAB_SIZE[0] / 2, cy - CAB_SIZE[1] / 2], [cx + CAB_SIZE[0] / 2, cy + CAB_SIZE[1] / 2]]


def _classify(objs):
    """Label detections: butter has a strongly orange top (demo 0.43-0.65 vs chocolate <=0.06);
    chocolate is the thicker box (top 3.0-5.0 cm vs butter 2.0 cm)."""
    for o in objs:
        if o["orange"] > 0.25:
            o["kind"] = "butter"
        elif o["orange"] < 0.15:
            o["kind"] = "chocolate"
        else:
            o["kind"] = "chocolate" if o["h"] > 0.026 else "butter"
    return objs


def _survey_args(cab, top):
    return {"waypoints": [[cab["cx"] - HOVER_BACK, cab["cy"], top + HOVER_DZ]], "gripper": -1.0, "settle": 6}


def perceive_decide(obs, goal, memory, last):
    """Decide the next act from the latest frame and the episode memory."""
    memory = dict(memory) if isinstance(memory, dict) else {}
    stage = _mem(memory, "stage", "start")
    top = float(_mem(memory, "top_z", 0.663))
    cab1_box = _mem(memory, "cab1_box", None)
    cab2_box = _mem(memory, "cab2_box", None)
    placed = list(_mem(memory, "placed", []))
    tries = int(_mem(memory, "tries", 0))
    target = _mem(memory, "target", None)
    kind = _mem(memory, "target_kind", "")
    note = ""
    f = obs.current
    tcp = np.asarray(f.tcp, dtype=float)
    act = None

    if stage == "start" or cab1_box is None or cab2_box is None:
        W, m = _cloud(f.agentview_depth, f.agentview_K, f.agentview_extrinsic)
        top, cabs = _detect_cabinets(W, m, tcp)
        if len(cabs) >= 2:
            c1 = max(cabs, key=lambda c: c["nobj"])
            if c1["nobj"] < 20:
                c1 = cabs[0]            # fallback: objects start on the lower-y cabinet in every demo
            c2 = [c for c in cabs if c is not c1][0]
            cab1_box, cab2_box = _box_of(c1), _box_of(c2)
            note = "cabinets from agentview"
        elif len(cabs) == 1:
            c = cabs[0]
            cab1_box = _box_of(c)
            cab2_box = _nominal_box(c["cx"], c["cy"] + 0.45)
            note = "one cabinet seen; cabinet2 assumed +0.45 m in y"
        else:
            cab1_box = _nominal_box(0.0, -0.18)
            cab2_box = _nominal_box(0.0, 0.28)
            note = "no cabinet seen; nominal layout"
        cab1 = _cab_from_box(cab1_box)
        args = _survey_args(cab1, top)
        args["waypoints"] = [[float(tcp[0]), float(tcp[1]), top + HOVER_DZ]] + args["waypoints"]
        act = {"tool": "act_move", "args": args}
        stage = "survey"

    elif stage == "survey":
        cab1 = _cab_from_box(cab1_box)
        W, m = _cloud(f.wrist_depth, f.wrist_K, f.wrist_extrinsic)
        t2, cabs = _detect_cabinets(W, m, tcp, top=top)
        if cabs:
            best = min(cabs, key=lambda c: abs(c["cy"] - cab1["cy"]) + abs(c["cx"] - cab1["cx"]))
            if abs(best["cy"] - cab1["cy"]) < 0.08 and best["full"]:
                cab1_box = _box_of(best)
                cab1 = _cab_from_box(cab1_box)
        objs = _classify(_detect_objects(W, m, f.wrist_rgb, top, cab1, zcut=float(tcp[2]) - 0.03))
        choc = [o for o in objs if o["kind"] == "chocolate"]
        butt = [o for o in objs if o["kind"] == "butter"]
        pick = choc[0] if choc else (butt[0] if butt else None)
        if pick is not None:
            target = [pick["xc"], pick["yc"]]
            kind = pick["kind"]
            note = "survey: %d objects; picking %s at (%.3f, %.3f) h=%.3f" % (len(objs), kind, target[0], target[1], pick["h"])
            act = {"tool": "act_pick", "args": {"x": target[0], "y": target[1], "top_z": top,
                                                "approach_z": top + 0.14, "lift_z": top + TRAVEL_DZ}}
            stage = "picked"
        else:
            if len(placed) >= 2 or tries >= 6:
                note = "cabinet1 empty; waiting for the checker"
                act = {"tool": "act_wait", "args": {"steps": 40}}
                stage = "verify"
            else:
                tries += 1
                note = "survey found nothing; retry %d" % tries
                args = _survey_args(cab1, top)
                args["waypoints"][0][2] = top + HOVER_DZ + 0.04 * (tries % 2)
                act = {"tool": "act_move", "args": args}
                stage = "survey"

    elif stage == "picked":
        fg = np.asarray(f.fingers, dtype=float)
        width = float(fg[0] - fg[1])     # measured after the lift (last frame of act_pick)
        cab2 = _cab_from_box(cab2_box)
        if HOLD_MIN_WIDTH < width < HOLD_MAX_WIDTH:
            dy = SLOT_DY.get(kind, 0.0)
            px, py = cab2["cx"] + SLOT_DX, cab2["cy"] + dy
            note = "holding %s (width %.3f); placing at (%.3f, %.3f)" % (kind, width, px, py)
            act = {"tool": "act_place", "args": {"x": px, "y": py, "top_z": top,
                                                 "travel_z": top + TRAVEL_DZ, "retreat_z": top + 0.2}}
            stage = "placed"
        else:
            tries += 1
            note = "grasp of %s failed (width %.3f); re-survey" % (kind, width)
            act = {"tool": "act_move", "args": _survey_args(_cab_from_box(cab1_box), top)}
            act["args"]["gripper"] = -1.0
            stage = "survey"

    elif stage == "placed":
        if kind and kind not in placed:
            placed.append(kind)
        note = "placed %s; placed=%s" % (kind, ",".join(placed))
        act = {"tool": "act_move", "args": _survey_args(_cab_from_box(cab1_box), top)}
        stage = "survey"

    else:  # verify
        tries += 1
        if tries % 3 == 0:
            note = "still running; re-survey cabinet1"
            act = {"tool": "act_move", "args": _survey_args(_cab_from_box(cab1_box), top)}
            stage = "survey"
        else:
            note = "waiting"
            act = {"tool": "act_wait", "args": {"steps": 40}}
            stage = "verify"

    new_mem = {"stage": stage, "top_z": float(top), "cab1_box": cab1_box, "cab2_box": cab2_box,
               "target": target, "target_kind": kind, "placed": placed, "tries": int(tries), "note": note}
    return {"memory": _plain(new_mem), "act": _plain(act)}


def _plain(x):
    """Plain JSON values (python floats/ints/lists) for memory and act arguments."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def act_move(robot, waypoints, gripper=-1.0, settle=0):
    """Move the TCP through world-xyz waypoints (top-down orientation), gripper sign held; then settle steps."""
    total = 0
    errs = []
    for wp in waypoints:
        n, err, _ = _servo(robot, [float(v) for v in wp], gripper, tol=0.004, max_steps=250)
        total += n
        errs.append(round(err, 4))
    if settle:
        _hold(robot, gripper, settle)
    out = _state(robot)
    out.update({"steps": total, "errors": errs})
    return out


def act_pick(robot, x, y, top_z, approach_z, lift_z):
    """Open, go above (x, y), descend until the fingers stall near the cabinet top, close, lift.

    Returns the finger gap `width` after lifting (object held when > ~0.01 m)."""
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    hi = max(float(tcp[2]), float(approach_z))
    _servo(robot, [float(tcp[0]), float(tcp[1]), hi], -1.0, tol=0.006, max_steps=150)
    _servo(robot, [x, y, hi], -1.0, tol=0.004, max_steps=250)
    _servo(robot, [x, y, approach_z], -1.0, tol=0.003, max_steps=200)
    _hold(robot, -1.0, 3)
    _, _, stalled = _servo(robot, [x, y, top_z + GRASP_DZ], -1.0, tol=0.002, max_steps=200, vmax=0.6, stall_z=True)
    z_grasp = float(robot.now().tcp[2])
    _hold(robot, 1.0, 18)
    w_closed = _width(robot)
    _servo(robot, [x, y, lift_z], 1.0, tol=0.005, max_steps=200, vmax=0.8)
    _hold(robot, 1.0, 4)
    out = _state(robot)
    out.update({"z_grasp": round(z_grasp, 4), "stalled": bool(stalled), "width_closed": round(w_closed, 4)})
    return out


def act_place(robot, x, y, top_z, travel_z, retreat_z):
    """Carry the held object at travel height to (x, y), lower it until contact, open, retreat upward."""
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    if tcp[2] < travel_z - 0.01:
        _servo(robot, [float(tcp[0]), float(tcp[1]), travel_z], 1.0, tol=0.006, max_steps=150)
    _servo(robot, [x, y, travel_z], 1.0, tol=0.004, max_steps=300)
    _hold(robot, 1.0, 3)
    _servo(robot, [x, y, top_z + 0.06], 1.0, tol=0.004, max_steps=200)
    _, _, stalled = _servo(robot, [x, y, top_z + RELEASE_DZ], 1.0, tol=0.002, max_steps=200, vmax=0.4, stall_z=True)
    z_rel = float(robot.now().tcp[2])
    w_before = _width(robot)
    _hold(robot, -1.0, 15)
    t2 = np.asarray(robot.now().tcp, dtype=float)
    _servo(robot, [float(t2[0]), float(t2[1]), retreat_z], -1.0, tol=0.006, max_steps=200)
    out = _state(robot)
    out.update({"z_release": round(z_rel, 4), "stalled": bool(stalled), "width_before_open": round(w_before, 4)})
    return out


def act_wait(robot, steps=40):
    """Hold the current pose with the gripper open for `steps` steps."""
    _hold(robot, -1.0, int(steps))
    return _state(robot)
