"""RememberShapeAndColor5x3-Long: read the cue (one coloured shape), wait through the
empty delay, then find the candidate with the same colour and shape among 15 objects
(5 shapes x 3 colours on an arc) and move the TCP down onto it."""

MEM_DEFAULTS = {
    "stage": "start",
    "cue_color": "",
    "cue_shape": "",
    "cue_margin": 0.0,
    "cue_step": -1,
    "target": [],
    "target_info": "",
    "attempts": 0,
    "notes": "",
    "excluded": [],
}

N_CAND_MIN = 8  # coloured components in the front view that mean "candidates are shown"


def _mem(memory):
    m = {}
    for k, v in MEM_DEFAULTS.items():
        m[k] = memory.get(k, v) if isinstance(memory, dict) else v
        if m[k] is None:
            m[k] = v
    return m


def _note(m, text):
    s = (m["notes"] + "; " + text) if m["notes"] else text
    m["notes"] = s[-1500:]


def _scan_for_cue(frames):
    """Look back through frames for one showing exactly one coloured object."""
    n = len(frames)
    idx = list(range(n - 1, -1, -1))
    if n > 40:
        idx = idx[::max(1, n // 40)]
    for i in idx:
        f = frames[i]
        c = _count_objects(f, "front")
        if c == 1:
            cue = _read_cue(f)
            if cue is not None:
                return cue, f
    return None, None


def perceive_decide(obs, goal, memory, last):
    """Policy step: record the cue as soon as visible, wait (zero motion) until the
    candidate objects appear, choose the same-colour same-shape object and touch it."""
    m = _mem(memory)
    cur = obs.current
    step = int(getattr(cur, "step", 0))
    g = (goal or "").lower()
    if m["stage"] == "start":
        _note(m, "goal_ok=%s" % ("touch" in g and "shape" in g and "color" in g))
    n_front = _count_objects(cur, "front")

    # 1. cue
    if not m["cue_color"]:
        cue = None
        cf = None
        if n_front <= 2:
            cue = _read_cue(cur)
            cf = cur
        if cue is None:
            cue, cf = _scan_for_cue(obs.frames)
        if cue is not None:
            m["cue_color"] = cue["color"]
            m["cue_shape"] = cue["shape"]
            m["cue_margin"] = round(float(cue["margin"]), 3)
            m["cue_step"] = int(getattr(cf, "step", step))
            _note(m, "cue %s %s margin %.2f cam %s @%d" % (cue["color"], cue["shape"], cue["margin"], cue["cam"], m["cue_step"]))

    if not m["cue_color"]:
        if n_front >= N_CAND_MIN:
            # candidates already shown but no cue seen: nothing reliable to match
            _note(m, "no cue before candidates @%d" % step)
            m["stage"] = "no_cue"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"until": "end", "max_steps": 5}}}
        m["stage"] = "wait_cue"
        return {"memory": m, "act": {"tool": "act_wait", "args": {"until": "cue", "max_steps": 300}}}

    # 2. wait for candidates
    if n_front < N_CAND_MIN:
        m["stage"] = "wait_candidates"
        return {"memory": m, "act": {"tool": "act_wait", "args": {"until": "candidates", "max_steps": 590}}}

    # 3. select target; after a touch that did not end the episode, retry on the same
    # target a little lower, the last retry with closed fingers (contact at the TCP).
    if m["stage"] == "touch" and isinstance(last, str) and '"verify_failed": true' in last:
        # close view showed another shape: exclude this object and re-select from the first
        # frame of the touch call (arm still near its start pose, wide views)
        ex = list(m["excluded"]) + [[m["target"][0], m["target"][1]]]
        m["excluded"] = ex
        _note(m, "verify failed at %s: %s @%d" % (m["target"], last[last.find('"seen_shape"'):][:40], step))
        fr0 = obs.frames[0] if len(obs.frames) else cur
        tgt, info = _find_target(fr0, m["cue_color"], m["cue_shape"], exclude=ex)
        if tgt is not None:
            m["target"] = [round(tgt["x"], 4), round(tgt["y"], 4), round(tgt["ztop"], 4)]
            m["target_info"] = str(info)[:800]
            _note(m, "reselected %s" % m["target"])
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": m["target"][0], "y": m["target"][1], "ztop": m["target"][2], "color": m["cue_color"],
            "shape": m["cue_shape"], "final_dz": 0.015, "gripper": -1.0, "verify": len(ex) < 2}}}
    if m["stage"] == "touch":
        m["attempts"] = int(m["attempts"]) + 1
        _note(m, "touch attempt %d no success: %s" % (m["attempts"], str(last)[:300]))
        if int(m["attempts"]) >= 4:
            m["stage"] = "give_up"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"until": "end", "max_steps": 600}}}
        t = m["target"]
        plan = [(0.015, -1.0), (0.005, -1.0), (0.0, -1.0), (0.012, 1.0)][int(m["attempts"])]
        _note(m, "retry %d final_dz %.3f gripper %.0f @%d" % (m["attempts"], plan[0], plan[1], step))
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": t[0], "y": t[1], "ztop": t[2], "color": m["cue_color"], "shape": m["cue_shape"],
            "final_dz": plan[0], "gripper": plan[1], "refine": False}}}
    if m["stage"] in ("start", "wait_cue", "wait_candidates", "retry"):
        tgt, info = _find_target(cur, m["cue_color"], m["cue_shape"])
        if tgt is None:
            _note(m, "no target found @%d %s" % (step, str(info)[:200]))
            m["stage"] = "retry"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"until": "steps", "max_steps": 2}}}
        m["target"] = [round(tgt["x"], 4), round(tgt["y"], 4), round(tgt["ztop"], 4)]
        m["target_info"] = str(info)[:800]
        m["stage"] = "touch"
        _note(m, "target %s @%d" % (m["target"], step))
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": tgt["x"], "y": tgt["y"], "ztop": tgt["ztop"], "color": m["cue_color"],
            "shape": m["cue_shape"], "final_dz": 0.015, "gripper": -1.0}}}

    return {"memory": m, "act": {"tool": "act_wait", "args": {"until": "steps", "max_steps": 5}}}
