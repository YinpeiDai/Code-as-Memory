"""SeqOfColors3 policy: read the cue colours, wait for the layout, touch every cued cube
(in cue order), then press the centre button."""

RETOUCH_Z = 0.043   # lower second-round touch height (fallback only)

MEMORY_DEFAULTS = {
    "stage": "cue",          # cue -> touch -> press -> done
    "cue_colors": [],
    "cue_first_steps": [],
    "layout_step": -1,
    "target_colors": [],
    "targets": [],
    "button": [],
    "touch_index": 0,
    "touch_attempts": 0,
    "press_attempts": 0,
    "touch_round": 0,
    "last_step": -1,
    "notes": "",
}


def _init(memory):
    m = {}
    for k, v in MEMORY_DEFAULTS.items():
        cur = memory.get(k) if isinstance(memory, dict) else None
        m[k] = (list(v) if isinstance(v, list) else v) if cur is None else cur
    return m


def _read_cue(frames, m):
    """Scan new frames for the cue cube colour sequence; stop at the first layout frame."""
    for f in frames:
        counts = _scene_colors(f)
        if _layout_visible(counts):
            m["layout_step"] = int(f.step)
            return True
        c = _cue_color(counts)
        if c is not None and (not m["cue_colors"] or m["cue_colors"][-1] != c):
            if c not in m["cue_colors"]:
                m["cue_colors"].append(c)
                m["cue_first_steps"].append(int(f.step))
    return False


def _locate(frame, m):
    cubes = _top_face_cubes(frame, "front")
    m["target_colors"] = []
    m["targets"] = []
    missing = []
    for c in m["cue_colors"]:
        if c in cubes:
            m["target_colors"].append(c)
            m["targets"].append([round(cubes[c][0] + CUBE_XY_OFFSET[0], 4),
                                 round(cubes[c][1] + CUBE_XY_OFFSET[1], 4)])
        else:
            missing.append(c)
    b = _button_cap(frame, "front")
    if b is not None:
        m["button"] = [round(b[0] + BUTTON_XY_OFFSET[0], 4), round(b[1] + BUTTON_XY_OFFSET[1], 4)]
    else:
        m["button"] = [-0.11, 0.0]
        missing.append("button")
    m["notes"] = (m["notes"] + " layout@%d cubes=%d missing=%s;" % (m["layout_step"], len(cubes), missing))[-600:]


def perceive_decide(obs, goal, memory, last):
    """Track the cue colours and stage progress in memory; choose the next act tool."""
    m = _init(memory)
    frames = list(obs.frames)
    m["last_step"] = int(obs.current.step)
    if m["stage"] == "cue":
        if _read_cue(frames, m):
            _locate(obs.current, m)
            m["stage"] = "touch"
        else:
            return {"memory": m, "act": {"tool": "act_wait_for_layout", "args": {"max_steps": 790}}}
    if m["stage"] == "touch" and last is not None and str(last).startswith("act_touch"):
        ok = '"descended": true' in str(last)
        if ok or m["touch_attempts"] >= 2:
            m["touch_index"] = m["touch_index"] + 1
            m["touch_attempts"] = 0
    if m["stage"] == "touch":
        i = m["touch_index"]
        if i < len(m["targets"]):
            m["touch_attempts"] = m["touch_attempts"] + 1
            x, y = m["targets"][i]
            args = {"x": x, "y": y, "color": m["target_colors"][i]}
            if m["touch_round"] > 0:
                args["touch_z"] = RETOUCH_Z
                args["hold_steps"] = 5
            return {"memory": m, "act": {"tool": "act_touch", "args": args}}
        m["stage"] = "press"
    if m["stage"] == "press" and last is not None and str(last).startswith("act_press"):
        # The episode is still running after a full press + lift: success did not register.
        # R011 showed unregistered touches leave the episode running, so re-touch every
        # target lower (once) if enough native steps remain, then press again.
        remaining = 800 - m["last_step"]
        if m["touch_round"] == 0 and len(m["targets"]) > 0 and remaining > 40 * len(m["targets"]) + 45:
            m["touch_round"] = 1
            m["touch_index"] = 0
            m["touch_attempts"] = 1
            m["stage"] = "touch"
            m["notes"] = (m["notes"] + " press%d no success -> retouch@%d;" % (m["press_attempts"], m["last_step"]))[-600:]
            x, y = m["targets"][0]
            return {"memory": m, "act": {"tool": "act_touch", "args": {
                "x": x, "y": y, "color": m["target_colors"][0], "touch_z": RETOUCH_Z, "hold_steps": 5}}}
    if m["stage"] == "press":
        if m["press_attempts"] < 3:
            m["press_attempts"] = m["press_attempts"] + 1
            bx, by = m["button"]
            # button = cap centre + expert offset; both the cap centre (R005) and the expert
            # offset (R001-R004) produced success, so press midway first, then try each.
            if m["press_attempts"] == 1:
                bx = bx - 0.5 * BUTTON_XY_OFFSET[0]
            elif m["press_attempts"] == 2:
                bx = bx - BUTTON_XY_OFFSET[0]
            return {"memory": m, "act": {"tool": "act_press", "args": {"x": bx, "y": by}}}
        m["stage"] = "done"
    return {"memory": m, "act": {"tool": "act_idle", "args": {"steps": 20}}}
