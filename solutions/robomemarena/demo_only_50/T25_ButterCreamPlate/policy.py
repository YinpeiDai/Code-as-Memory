"""Stage machine for 'First pick and place X from plate1 to plate2, then pick and place Y ...'.

Memory holds the stage, the parsed object order, the current object index, pick attempts, the last
known xy of each object, the plate2 centre and the grasp point used. Every decision re-perceives."""

KNOWN_OBJECTS = ("butter", "cream")
FIELDS = ("stage", "order", "index", "attempts", "butter_xy", "cream_xy", "plate2_xy", "grasp_xy", "rechecks",
          "notes")
MAX_RECHECKS = 2
# Calibrated on 50 demos: expert grasp xy minus detected blob median xy.
AGENT_OFFSET = {"butter": [-0.0175, 0.0], "cream": [-0.003, -0.001]}   # agentview blob -> grasp
WRIST_OFFSET = {"butter": [-0.011, 0.003], "cream": [0.002, 0.0]}      # wrist blob (hover z~0.65) -> grasp
DEFAULT_XY = {"butter": [0.21, -0.23], "cream": [0.21, -0.15]}       # demo means, last-resort fallback
MAX_ATTEMPTS = 3
MIN_AGENT_PIX = 15
MIN_WRIST_PIX = 120


def _parse_order(goal):
    g = (goal or "").lower()
    found = []
    for name in KNOWN_OBJECTS:
        i = g.find(name)
        if i >= 0:
            found.append((i, name))
    found.sort()
    order = [n for _, n in found]
    return order if order else list(KNOWN_OBJECTS)


def _agent(frame):
    return _detect(frame.agentview_rgb, frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)


def _wrist(frame):
    return _detect(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)


def _choose_plate2(det, objs):
    best = None
    for p in det["plates"]:
        if p["n"] < 40:
            continue
        far = True
        for o in objs:
            if o is not None and math.hypot(p["xy"][0] - o[0], p["xy"][1] - o[1]) < 0.09:
                far = False
        if far and (best is None or p["n"] > best["n"]):
            best = p
    return None if best is None else best["xy"]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _update_obj(m, det, name):
    seen = det.get(name)
    if seen is not None and det.get(name + "_n", 0) >= MIN_AGENT_PIX:
        m[name + "_xy"] = seen
    if m[name + "_xy"] is None:
        m[name + "_xy"] = DEFAULT_XY.get(name, [0.21, -0.19])


def _hover_target(m, name):
    p = m[name + "_xy"]
    off = AGENT_OFFSET.get(name, [0.0, 0.0])
    return {"x": p[0] + off[0], "y": p[1] + off[1]}


def perceive_decide(obs, goal, memory, last):
    """Re-detect objects/plates (agentview RGB-D; wrist RGB-D after hovering), update memory, pick next act."""
    m = {k: memory.get(k) for k in FIELDS}
    cur = obs.current
    det = _agent(cur)

    def out(tool, args):
        return {"memory": {k: m[k] for k in FIELDS}, "act": {"tool": tool, "args": args}}

    if m["stage"] is None:
        m["order"] = _parse_order(goal)
        m["index"] = 0
        m["attempts"] = 0
        for n in m["order"]:
            m[n + "_xy"] = None
            _update_obj(m, det, n)
        m["plate2_xy"] = _choose_plate2(det, [m[n + "_xy"] for n in m["order"]])
        m["notes"] = "init n_b=%d n_c=%d plates=%d" % (det["butter_n"], det["cream_n"], len(det["plates"]))
        m["stage"] = "hover"
    if m["plate2_xy"] is None:
        objs = [m[n + "_xy"] for n in m["order"]] if m["index"] == 0 else []
        m["plate2_xy"] = _choose_plate2(det, objs)
        if m["plate2_xy"] is None:
            m["plate2_xy"] = [0.17, 0.12]   # demo mean of plate2, last resort
            m["notes"] = "plate2 not detected, using fallback"

    if m["stage"] == "done" or m["index"] >= len(m["order"]):
        m["stage"] = "done"
        # the episode should already have ended on success; if not, look for an object left off plate2
        if int(m["rechecks"] or 0) < MAX_RECHECKS:
            for i, n in enumerate(m["order"]):
                seen = det.get(n)
                if seen is not None and det.get(n + "_n", 0) >= MIN_AGENT_PIX and _dist(seen, m["plate2_xy"]) > 0.08:
                    m["rechecks"] = int(m["rechecks"] or 0) + 1
                    m["index"] = i
                    m[n + "_xy"] = seen
                    m["attempts"] = 1
                    m["stage"] = "grasp"
                    m["notes"] = "recheck: %s off plate2 at [%.3f, %.3f]" % (n, seen[0], seen[1])
                    return out("act_hover", _hover_target(m, n))
        return out("act_wait", {"n": 60})
    name = m["order"][m["index"]]
    key = name + "_xy"
    fing = cur.fingers
    width = float(abs(fing[0]) + abs(fing[1]))

    if m["stage"] == "hover":
        m["attempts"] = int(m["attempts"] or 0) + 1
        m["stage"] = "grasp"
        return out("act_hover", _hover_target(m, name))

    if m["stage"] == "grasp":
        w = _wrist(cur)
        coarse = _hover_target(m, name)
        gx, gy = coarse["x"], coarse["y"]
        src = "agent"
        if w.get(name) is not None and w.get(name + "_n", 0) >= MIN_WRIST_PIX:
            off = WRIST_OFFSET.get(name, [0.0, 0.0])
            cand = [w[name][0] + off[0], w[name][1] + off[1]]
            if _dist(cand, [gx, gy]) < 0.06:
                gx, gy = cand
                src = "wrist"
        m["grasp_xy"] = [gx, gy]
        m["notes"] = "grasp %s via %s" % (name, src)
        m["stage"] = "verify_grasp"
        return out("act_grasp", {"x": gx, "y": gy})

    if m["stage"] == "verify_grasp":
        if width > 0.003:
            m["stage"] = "verify_place"
            return out("act_place", {"x": m["plate2_xy"][0], "y": m["plate2_xy"][1]})
        m["notes"] = "empty grasp of %s (width %.4f)" % (name, width)
        if m["attempts"] < MAX_ATTEMPTS:
            _update_obj(m, det, name)
            m["attempts"] += 1
            m["stage"] = "grasp"
            return out("act_hover", _hover_target(m, name))
        m["index"] += 1
        m["attempts"] = 0
        m["stage"] = "hover"
        return out("act_wait", {"n": 3})

    if m["stage"] == "verify_place":
        seen = det.get(name)
        p2 = m["plate2_xy"]
        missed = seen is not None and det.get(name + "_n", 0) >= MIN_AGENT_PIX and _dist(seen, p2) > 0.08
        if missed and m["attempts"] < MAX_ATTEMPTS:
            m[key] = seen
            m["notes"] = "%s missed plate2 at [%.3f, %.3f]" % (name, seen[0], seen[1])
            m["attempts"] += 1
            m["stage"] = "grasp"
            return out("act_hover", _hover_target(m, name))
        m["notes"] = "placed %s" % name
        m["index"] += 1
        m["attempts"] = 0
        if m["index"] >= len(m["order"]):
            m["stage"] = "done"
            return out("act_wait", {"n": 60})
        nxt = m["order"][m["index"]]
        _update_obj(m, det, nxt)
        m["attempts"] = 1
        m["stage"] = "grasp"
        return out("act_hover", _hover_target(m, nxt))

    m["stage"] = "done"
    return out("act_wait", {"n": 60})
