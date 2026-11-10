"""ChainOfColors policy: watch the colour cue, map it to the cube layout, touch cubes in cue order,
then press the centre button.  Perception helpers live in perception.py; motion helpers in motion.py."""

# Calibrated from 175 expert touches over 25 demos: expert TCP x is 7.6 mm below our top-face
# centroid estimate (std 1.7 mm); y offset ~0.
CUBE_X_OFFSET = -0.0076
BUTTON_X_OFFSET = 0.005
Z_TOUCH = 0.05  # R009: touches at TCP z 0.062-0.065 failed the task; 0.05 succeeded 8/8
Z_TRAVEL = 0.17
Z_LIFT = 0.17
PRESS_LIFT = 0.20
# R004 exploration showed a press followed by a 0.12 m lift + 17-step wait never succeeded, while
# lifting to 0.20 succeeded ~5-6 steps after release.  Schedule kept empty (all presses lift to 0.20).
PRESS_LIFT_SCHEDULE = []
DEFAULTS = {
    "stage": "watch", "cue": [], "cue_steps": [], "gap": False, "layout": [], "layout_step": -1,
    "targets": [], "target_colors": [], "button": [], "next_index": 0, "press_count": 0,
    "home_rpy": [], "log": "", "touch_attempts": 0,
}


def _mem(memory):
    m = {}
    for k, v in DEFAULTS.items():
        val = memory.get(k) if isinstance(memory, dict) else None
        if val is None:
            val = v if not isinstance(v, list) else list(v)
        m[k] = val
    return m


def _proto_rgb(name):
    for key, nm in _COLOR_NAMES.items():
        if nm == name:
            lv = {0: 4.0, 1: 169.0, 2: 231.0}
            return np.array([lv[key[0]], lv[key[1]], lv[key[2]]])
    return None


def _plan(m):
    """Assign each cue colour to a layout cube; returns (targets, colors, notes)."""
    layout = [b for b in m["layout"] if b.get("xyz")]
    used = []
    targets, colors, notes = [], [], []
    for c in m["cue"]:
        cands = [i for i, b in enumerate(layout) if b["color"] == c and i not in used]
        if not cands:
            proto = _proto_rgb(c)
            free = [i for i in range(len(layout)) if i not in used
                    and layout[i]["color"] not in m["cue"]]
            if proto is None or not free or "rgb" not in layout[free[0]]:
                notes.append("missing:" + c)
                continue
            cands = [min(free, key=lambda i: float(((np.array(layout[i]["rgb"]) - proto) ** 2).sum()))]
            notes.append("fallback:%s->%s" % (c, layout[cands[0]]["color"]))
        i = max(cands, key=lambda q: layout[q].get("n", 0))
        used.append(i)
        b = layout[i]
        targets.append([b["xyz"][0] + CUBE_X_OFFSET, b["xyz"][1], b["xyz"][2]])
        colors.append(c)
    return targets, colors, notes


def perceive_decide(obs, goal, memory, last):
    """Track cue colours and layout from all new frames, then choose the next stage action."""
    m = _mem(memory)
    frames = list(obs.frames)
    if not m["home_rpy"]:
        m["home_rpy"] = [float(v) for v in np.asarray(frames[0].tcp)[3:6]]
    last_s = last if isinstance(last, str) else ""
    if m["stage"] == "watch":
        st = {"cue": list(m["cue"]), "cue_steps": list(m["cue_steps"]), "gap": bool(m["gap"]),
              "layout": None}
        _scan_frames(frames, st)
        m["cue"], m["cue_steps"], m["gap"] = st["cue"], st["cue_steps"], st["gap"]
        if st.get("layout"):
            m["layout"] = st["layout"]
            m["layout_step"] = st["layout_step"]
            m["button"] = st["button"] or []
            targets, colors, notes = _plan(m)
            m["targets"], m["target_colors"] = targets, colors
            m["stage"] = "touch"
            print("perceive", "cue", m["cue"], "steps", m["cue_steps"], "layout_step", m["layout_step"],
                  "targets", [[round(v, 3) for v in t] for t in targets], "button", m["button"], notes)
            m["log"] = ("cue=%s layout_step=%d n_layout=%d notes=%s" %
                        (",".join(m["cue"]), m["layout_step"], len(m["layout"]), notes))
        else:
            return {"memory": m, "act": {"tool": "act_watch", "args": {"max_steps": 80}}}
    if m["stage"] == "touch" and last_s.startswith("act_touch"):
        if '"reached": true' in last_s:
            m["next_index"] = int(m["next_index"]) + 1
            m["touch_attempts"] = 0
        else:
            m["touch_attempts"] = int(m["touch_attempts"]) + 1
            if m["touch_attempts"] >= 2:
                m["next_index"] = int(m["next_index"]) + 1
                m["touch_attempts"] = 0
    if m["stage"] == "touch" and int(m["next_index"]) >= len(m["targets"]):
        m["stage"] = "press"
    if m["stage"] == "touch":
        t = m["targets"][int(m["next_index"])]
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": float(t[0]), "y": float(t[1]), "z_touch": Z_TOUCH, "z_travel": Z_TRAVEL,
            "z_lift": Z_LIFT, "dwell": 2, "home_rpy": m["home_rpy"]}}}
    if m["stage"] == "press":
        if last_s.startswith("act_press"):
            m["press_count"] = int(m["press_count"]) + 1
        b = m["button"] if m["button"] else [-0.125, 0.0, 0.058]
        hold = 12 + 6 * int(m["press_count"])
        return {"memory": m, "act": {"tool": "act_press", "args": {
            "x": float(b[0]) + BUTTON_X_OFFSET, "y": float(b[1]), "z_top": float(b[2]),
            "z_travel": Z_TRAVEL,
            "z_lift": PRESS_LIFT_SCHEDULE[int(m["press_count"])] if int(m["press_count"]) < len(PRESS_LIFT_SCHEDULE) else PRESS_LIFT,
            "hold_top": hold,
            "home_rpy": m["home_rpy"]}}}
    return {"memory": m, "act": {"tool": "act_watch", "args": {"max_steps": 5}}}


def act_watch(robot, max_steps=80):
    """Hold still (zero motion, gripper open) until the cube layout appears or max_steps elapse."""
    n = 0
    seen = False
    for _ in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        if len(_blobs(robot.now())) >= 7:
            seen = True
            break
    print("act_watch", n, seen, int(robot.now().step))
    return {"steps": n, "layout_seen": seen, "step": int(robot.now().step)}


def act_touch(robot, x, y, z_touch=0.05, z_travel=0.17, z_lift=0.17, dwell=2, home_rpy=None):
    """Travel at z_travel to (x, y), descend to z_touch with open gripper, dwell, lift to z_lift."""
    rec = {"target": [round(x, 4), round(y, 4)], "start_step": int(robot.now().step)}
    g = 1.0
    r1 = _goto(robot, x, y, z_travel, g, home_rpy, xy_tol=0.012, z_tol=0.04, max_steps=60,
               min_z_for_xy=0.09, settle=False)
    r2 = _goto(robot, x, y, z_touch, g, home_rpy, xy_tol=0.006, z_tol=0.006, max_steps=25,
               settle=False, stall=True, dz_min=-0.6)
    p = np.asarray(robot.now().tcp)[:3]
    rec["bottom"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
    zmin = float(p[2])
    for _ in range(int(dwell)):
        _step_toward(robot, x, y, z_touch, g, home_rpy)
        zmin = min(zmin, float(np.asarray(robot.now().tcp)[2]))
    r3 = _goto(robot, x, y, z_lift, g, home_rpy, xy_tol=0.02, z_tol=0.012, max_steps=25,
               settle=False)
    p = np.asarray(robot.now().tcp)[:3]
    rec["zmin"] = round(zmin, 4)
    rec["xy_err_bottom"] = round(float(np.hypot(rec["bottom"][0] - x, rec["bottom"][1] - y)), 4)
    rec["reached"] = bool(zmin < z_touch + 0.012 and rec["xy_err_bottom"] < 0.015)
    rec["phase_steps"] = [r1["steps"], r2["steps"], int(dwell), r3["steps"]]
    rec["end_step"] = int(robot.now().step)
    rec["end_z"] = round(float(p[2]), 4)
    print("act_touch", rec)
    return rec


def act_press(robot, x, y, z_top=0.058, z_travel=0.17, z_lift=0.20, hold_top=12, home_rpy=None,
              push_steps=4):
    """Close gripper, travel above the button, push down until stalled, hold, lift and wait still."""
    rec = {"target": [round(x, 4), round(y, 4)], "start_step": int(robot.now().step)}
    g = -1.0
    r1 = _goto(robot, x, y, z_travel, g, home_rpy, xy_tol=0.008, z_tol=0.04, max_steps=60,
               min_z_for_xy=0.09, settle=False)
    r2 = _goto(robot, x, y, z_top - 0.025, g, home_rpy, xy_tol=0.006, z_tol=0.004, max_steps=25,
               settle=False, stall=True, dz_min=-0.7)
    p = np.asarray(robot.now().tcp)[:3]
    rec["contact"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
    zmin = float(p[2])
    for _ in range(int(push_steps)):
        _step_toward(robot, x, y, z_top - 0.025, g, home_rpy)
        zmin = min(zmin, float(np.asarray(robot.now().tcp)[2]))
    rec["zmin"] = round(zmin, 4)
    r3 = _goto(robot, x, y, z_lift, g, home_rpy, xy_tol=0.02, z_tol=0.006, max_steps=30,
               settle=True)
    for _ in range(int(hold_top)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
    p = np.asarray(robot.now().tcp)[:3]
    rec["phase_steps"] = [r1["steps"], r2["steps"], int(push_steps), r3["steps"], int(hold_top)]
    rec["end"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
    rec["end_step"] = int(robot.now().step)
    print("act_press", rec)
    return rec
