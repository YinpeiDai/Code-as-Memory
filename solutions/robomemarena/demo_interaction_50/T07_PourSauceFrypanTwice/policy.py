"""Task program for T07_PourSauceFrypanTwice.

Sequence (inferred from 50 expert demos, all of which end -- with scored
success -- right after the second tilt over the frypan):
  locate -> pick bottle -> pour #1 (tilt 45 deg over pan, back upright)
         -> pour #2 -> [fallback if the episode is still running: place the
            bottle into the bowl drainer] -> hold.
Progress is only advanced on measured evidence: wrist-depth/finger in-hand
test after picking, and the TCP tilt profile in the frames of each pour.
"""

MAX_GRASP_TRIES = 3
MAX_LOOK_TRIES = 2


def _mem_defaults(memory):
    m = dict(memory or {})
    for k, v in (("phase", None), ("bottle_xy", None), ("pour_xy", None), ("drainer_xy", None),
                 ("grasp_tries", None), ("pours_done", None), ("look_tries", None), ("log", None)):
        if k not in m:
            m[k] = v
    if m["grasp_tries"] is None:
        m["grasp_tries"] = 0
    if m["pours_done"] is None:
        m["pours_done"] = 0
    if m["look_tries"] is None:
        m["look_tries"] = 0
    if m["log"] is None:
        m["log"] = ""
    if m["phase"] is None:
        m["phase"] = "locate"
    return m


def _note(m, s):
    log = (m["log"] + " | " + s) if m["log"] else s
    m["log"] = log[-1500:]


def _max_tilt(obs):
    best = 0.0
    for f in obs.frames:
        best = max(best, _tilt_deg(f.tcp))
    return best


def _out(m, tool, args):
    return {"memory": m, "act": {"tool": tool, "args": args}}


def perceive_decide(obs, goal, memory, last):
    """Decide the next act from the latest frames and the episode memory."""
    m = _mem_defaults(memory)
    cur = obs.current
    phase = m["phase"]

    if phase == "locate":
        s = _scene(cur)
        _note(m, "locate: " + s["notes"])
        if s["pour"] is not None:
            m["pour_xy"] = [round(v, 4) for v in s["pour"]]
        if s["drainer"] is not None:
            m["drainer_xy"] = [round(v, 4) for v in s["drainer"][:2]]
        if s["bottle"] is None:
            if m["look_tries"] < MAX_LOOK_TRIES:
                m["look_tries"] += 1
                return _out(m, "act_raise", {"z": 0.80, "grip": -1})
            _note(m, "bottle never seen; using demo-mean grasp point (assumption)")
            s["bottle"] = [0.0, 0.0, 0.5]
        m["bottle_xy"] = [round(s["bottle"][0], 4), round(s["bottle"][1], 4)]
        if m["pour_xy"] is None:
            m["pour_xy"] = list(DEFAULT_POUR)
            _note(m, "pan not seen; demo-mean pour point (assumption)")
        if m["drainer_xy"] is None:
            m["drainer_xy"] = list(DEFAULT_DRAINER)
        m["phase"] = "pick"
        m["grasp_tries"] = 1
        return _out(m, "act_pick", {"x": m["bottle_xy"][0], "y": m["bottle_xy"][1]})

    if phase == "pick":
        h = _in_hand(cur)
        if h["held"]:
            _note(m, "picked (wrist %.0fmm, w=%.3f)" % (h["wrist_med_mm"], h["finger_width"]))
            m["phase"] = "pour"
            return _out(m, "act_pour", {"x": m["pour_xy"][0], "y": m["pour_xy"][1]})
        _note(m, "grasp miss %d (wrist %.0fmm, w=%.3f)" % (m["grasp_tries"], h["wrist_med_mm"], h["finger_width"]))
        if m["grasp_tries"] >= MAX_GRASP_TRIES:
            m["phase"] = "failed"
            return _out(m, "act_raise", {"z": 0.80, "grip": -1})
        # re-observe the bottle (it may have been nudged) and retry
        s = _scene(cur)
        if s["bottle"] is not None:
            m["bottle_xy"] = [round(s["bottle"][0], 4), round(s["bottle"][1], 4)]
            _note(m, "re-located bottle " + s["notes"])
        m["grasp_tries"] += 1
        return _out(m, "act_pick", {"x": m["bottle_xy"][0], "y": m["bottle_xy"][1]})

    if phase == "pour":
        # evaluate the pour that just ran
        tilt = _max_tilt(obs)
        h = _in_hand(cur)
        if tilt > 38.0 and h["held"]:
            m["pours_done"] += 1
            _note(m, "pour %d done (max tilt %.1f)" % (m["pours_done"], tilt))
        elif not h["held"]:
            _note(m, "bottle lost during pour (tilt %.1f); re-pick" % tilt)
            s = _scene(cur)
            if s["bottle"] is not None:
                m["bottle_xy"] = [round(s["bottle"][0], 4), round(s["bottle"][1], 4)]
            if m["grasp_tries"] >= MAX_GRASP_TRIES + 1:
                m["phase"] = "failed"
                return _out(m, "act_raise", {"z": 0.80, "grip": -1})
            m["phase"] = "pick"
            m["grasp_tries"] += 1
            return _out(m, "act_pick", {"x": m["bottle_xy"][0], "y": m["bottle_xy"][1]})
        else:
            _note(m, "pour tilt too small (%.1f); repeat" % tilt)
        if m["pours_done"] < 2:
            return _out(m, "act_pour", {"x": m["pour_xy"][0], "y": m["pour_xy"][1]})
        m["phase"] = "place"
        _note(m, "two pours done, episode still running -> place in drainer")
        return _out(m, "act_place", {"x": m["drainer_xy"][0], "y": m["drainer_xy"][1]})

    if phase == "place":
        h = _in_hand(cur)
        _note(m, "after place: wrist %.0fmm w=%.3f" % (h["wrist_med_mm"], h["finger_width"]))
        m["phase"] = "done"
        return _out(m, "act_observe", {"steps": 40})

    # done / failed: hold still and let the checker settle
    return _out(m, "act_observe", {"steps": 40})
