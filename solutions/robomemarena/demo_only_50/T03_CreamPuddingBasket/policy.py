"""Decision program for T03_CreamPuddingBasket.

Goal (all 50 demonstrations): "Pick and place cream into the basket, then pick and place chocolate into
the same basket." Order: cream first, then chocolate, both into the basket the cream went to. The
expert always used the basket on the robot side (smaller world x) of the two baskets."""

SURVEY_POSE = [-0.058, 0.0, 0.671]     # the episode start pose; wrist view covers both boxes and baskets
HOVER_Z = 0.60                         # pre-grasp height for wrist refinement (expert passed ~0.60)
GRASP_DZ = 0.004                       # grasp TCP height above the table (expert: 0.433-0.435, table 0.430)
LIFT_Z = 0.78
CARRY_Z = 0.80
RELEASE_Z = 0.655                      # expert released at 0.654-0.655 over the basket (rim ~0.565)
NOMINAL = {"cream": [-0.019, -0.017], "chocolate": [-0.019, -0.237], "basket": [-0.016, 0.199]}
BASKET_HALF = 0.075                    # basket footprint half-size (rim 0.15 x 0.16 m)
HOLD_MIN, HOLD_MAX = 0.012, 0.07       # finger width that means a box is held
MAX_TRIES = 3


def _goal_order(goal):
    """Object order from the instruction: cream then chocolate unless the text says otherwise."""
    g = (goal or "").lower()
    ic, ik = g.find("cream"), g.find("chocolate")
    if ic >= 0 and ik >= 0 and ik < ic:
        return ["chocolate", "cream"]
    return ["cream", "chocolate"]


def _in_basket(xy, basket):
    return xy is not None and basket is not None and abs(xy[0] - basket[0]) < BASKET_HALF and abs(xy[1] - basket[1]) < BASKET_HALF


def _key(name):
    return "cream_xy" if name == "cream" else "choc_xy"


def _mem(memory):
    m = dict(memory or {})
    defaults = {"stage": "start", "target": "", "cream_xy": None, "choc_xy": None, "basket_xy": None,
                "table_z": 0.43, "placed": [], "tries": 0, "log": ""}
    for k, v in defaults.items():
        if m.get(k) is None:
            m[k] = v
    m["placed"] = list(m["placed"])
    return m


def _next_target(m, goal):
    for name in _goal_order(goal):
        if name not in m["placed"]:
            return name
    return ""


def _go_hover(m, name, note):
    xy = m[_key(name)]
    m["target"] = name
    m["stage"] = "hover"
    m["log"] = note
    return {"tool": "act_move", "args": {"x": xy[0], "y": xy[1], "z": HOVER_Z, "gripper": -1.0}}


def _go_survey(m, note):
    m["stage"] = "survey"
    m["log"] = note
    return {"tool": "act_move", "args": {"x": SURVEY_POSE[0], "y": SURVEY_POSE[1], "z": SURVEY_POSE[2],
                                         "gripper": -1.0}}


def _survey(m, frame, goal, first):
    """Locate boxes (wrist RGB-D) and the target basket (agentview depth) and update memory."""
    det = _detect_objects(frame)
    m["table_z"] = float(det["table_z"])
    if first or m["basket_xy"] is None:
        baskets = _detect_baskets(frame)
        m["basket_xy"] = baskets[0]["xy"] if baskets else list(NOMINAL["basket"])
    notes = []
    for name in ("cream", "chocolate"):
        blobs = det[name + "_all"]
        on_table = [b for b in blobs if not _in_basket(b, m["basket_xy"])]
        in_basket = [b for b in blobs if _in_basket(b, m["basket_xy"])]
        if on_table:
            m[_key(name)] = list(on_table[0])
            if name in m["placed"]:
                m["placed"] = [p for p in m["placed"] if p != name]
            notes.append(name + " on table")
        elif in_basket and not first:
            if name not in m["placed"]:
                m["placed"] = m["placed"] + [name]
            notes.append(name + " in basket")
        elif m[_key(name)] is None:
            m[_key(name)] = list(NOMINAL[name])
            notes.append(name + " not seen; nominal")
    return "; ".join(notes)


def perceive_decide(obs, goal, memory, last):
    """Stage machine: survey -> (hover+refine -> grasp -> check -> place) for cream then chocolate -> verify."""
    m = _mem(memory)
    frame = obs.current
    stage = m["stage"]
    act = None

    if stage == "start":
        note = _survey(m, frame, goal, True)
        m["tries"] = 0
        act = _go_hover(m, _next_target(m, goal), "start survey: " + note)

    elif stage == "survey":
        note = _survey(m, frame, goal, False)
        name = _next_target(m, goal)
        m["tries"] = 0
        if name:
            act = _go_hover(m, name, "survey: " + note)
        else:
            m["stage"] = "verify_wait"
            m["log"] = "survey: all placed; " + note
            act = {"tool": "act_wait", "args": {"steps": 40}}

    elif stage == "hover":
        name = m["target"]
        prior = m[_key(name)]
        det = _detect_objects(frame, {name: prior}, 0.04)
        blob = det[name]
        if blob is None:
            det = _detect_objects(frame, {name: prior}, 0.08)
            blob = det[name]
        if blob is not None and not _in_basket(blob["xy"], m["basket_xy"]):
            m[_key(name)] = list(blob["xy"])
            note = "refined"
        else:
            note = "not refined"
        xy = m[_key(name)]
        m["stage"] = "check_grasp"
        m["log"] = "grasp %s at %.3f %.3f (%s)" % (name, xy[0], xy[1], note)
        act = {"tool": "act_grasp", "args": {"x": xy[0], "y": xy[1], "z_grasp": m["table_z"] + GRASP_DZ,
                                             "z_lift": LIFT_Z}}

    elif stage == "check_grasp":
        w = _width(frame)
        name = m["target"]
        if HOLD_MIN < w < HOLD_MAX:
            b = m["basket_xy"]
            m["stage"] = "check_place"
            m["log"] = "holding %s (w=%.3f); place" % (name, w)
            act = {"tool": "act_place", "args": {"x": b[0], "y": b[1], "z_carry": CARRY_Z, "z_release": RELEASE_Z}}
        else:
            m["tries"] = int(m["tries"]) + 1
            if m["tries"] >= MAX_TRIES:
                act = _go_survey(m, "grasp of %s failed %d times (w=%.3f); survey" % (name, m["tries"], w))
            else:
                act = _go_hover(m, name, "grasp of %s missed (w=%.3f); retry" % (name, w))

    elif stage == "check_place":
        w = _width(frame)
        name = m["target"]
        if w < HOLD_MIN:
            # fingers closed on nothing: the box was lost on the way; find it again
            m["tries"] = int(m["tries"]) + 1
            act = _go_survey(m, "%s dropped in transit (w=%.3f); survey" % (name, w))
        else:
            if name not in m["placed"]:
                m["placed"] = m["placed"] + [name]
            m["tries"] = 0
            nxt = _next_target(m, goal)
            if nxt:
                act = _go_hover(m, nxt, "%s released over basket; next %s" % (name, nxt))
            else:
                act = _go_survey(m, "%s released; all placed; verify" % name)

    else:  # verify_wait: everything believed placed but the episode continues
        m["stage"] = "survey"
        m["log"] = "re-survey after wait"
        act = {"tool": "act_move", "args": {"x": SURVEY_POSE[0], "y": SURVEY_POSE[1], "z": SURVEY_POSE[2],
                                            "gripper": -1.0}}

    return {"memory": m, "act": act}
