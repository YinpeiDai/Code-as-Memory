"""SeqOfColors3: memorise the cued colour sequence, touch those cubes in cue order, then press the button."""

DEFAULTS = {
    "stage": "cue", "cue_colors": [], "cue_log": "", "cube_colors": [], "cube_xy": [],
    "button_xy": [], "touched": 0, "presses": 0, "rounds": 0, "rot_ref": [], "pending": "", "log": "",
}


def _mem(memory):
    m = {}
    for k, v in DEFAULTS.items():
        val = memory.get(k) if isinstance(memory, dict) else None
        m[k] = (list(v) if isinstance(v, list) else v) if val is None else val
    return m


def _record_array(m, det):
    cubes = _array_cubes(det)
    names = sorted(cubes.keys())
    m["cube_colors"] = names
    m["cube_xy"] = [[round(cubes[k][0], 4), round(cubes[k][1], 4)] for k in names]
    if det["button"] is not None:
        m["button_xy"] = [round(det["button"]["xy"][0], 4), round(det["button"]["xy"][1], 4)]


def perceive_decide(obs, goal, memory, last):
    """Track the cue sequence, the array/button layout and stage progress; choose the next act."""
    m = _mem(memory)
    cur = obs.current
    if not m["rot_ref"]:
        t = cur.tcp
        m["rot_ref"] = [float(t[3]), float(t[4]), float(t[5])]
    lt = last if isinstance(last, str) else ""
    # commit progress of the previous act only when it reports success of its motion stage
    if m["pending"] and lt:
        ok = '"ok": true' in lt
        if m["pending"].startswith("touch") and ok:
            m["touched"] = int(m["touched"]) + 1
        elif m["pending"].startswith("press"):
            m["presses"] = int(m["presses"]) + 1
        m["log"] = (m["log"] + " | " + lt[:160])[-1500:]
    m["pending"] = ""

    if m["stage"] == "cue":
        array_det = None
        for fr in obs.frames:
            det = _detect(fr)
            if _is_array(det):
                array_det = det
                continue
            if array_det is not None:
                continue
            c = _cue_color(det)
            m["cue_log"] = (m["cue_log"] + (c[0] if c else "."))[-200:]
            if c is not None and (not m["cue_colors"] or m["cue_colors"][-1] != c):
                m["cue_colors"] = m["cue_colors"] + [c]
        if array_det is not None:
            _record_array(m, array_det)
            m["stage"] = "touch"
            m["cue_log"] = m["cue_log"] + "|A@" + str(cur.step)
        else:
            return {"memory": m, "act": {"tool": "act_wait_for_array", "args": {"max_steps": 60}}}

    # re-detect layout if a needed colour or the button is missing
    need = [c for c in m["cue_colors"] if c not in m["cube_colors"]]
    if need or not m["button_xy"]:
        det = _detect(cur)
        cubes = _array_cubes(det)
        names = list(m["cube_colors"])
        xy = [list(p) for p in m["cube_xy"]]
        for k, v in cubes.items():
            if k not in names:
                names.append(k)
                xy.append([round(v[0], 4), round(v[1], 4)])
        m["cube_colors"], m["cube_xy"] = names, xy
        if not m["button_xy"] and det["button"] is not None:
            m["button_xy"] = [round(det["button"]["xy"][0], 4), round(det["button"]["xy"][1], 4)]

    if m["stage"] == "touch":
        idx = int(m["touched"])
        if idx < len(m["cue_colors"]):
            color = m["cue_colors"][idx]
            if color in m["cube_colors"]:
                p = m["cube_xy"][m["cube_colors"].index(color)]
                z = TOUCH_Z if int(m["rounds"]) == 0 else TOUCH_Z - 0.004
                m["pending"] = "touch:" + color
                return {"memory": m, "act": {"tool": "act_touch", "args": {
                    "x": float(p[0]) + TOUCH_X_OFFSET, "y": float(p[1]), "z_touch": z,
                    "rot_ref": m["rot_ref"], "label": color}}}
            # colour not found: skip it (logged) rather than touch a wrong cube
            m["log"] = (m["log"] + " | missing " + color)[-1500:]
            m["touched"] = idx + 1
            return {"memory": m, "act": {"tool": "act_wait_for_array", "args": {"max_steps": 1}}}
        m["stage"] = "press"

    if m["stage"] == "press":
        if int(m["presses"]) >= 2 and int(m["rounds"]) == 0:
            # two presses without native success: redo the whole touch sequence once, then press again
            m["rounds"] = 1
            m["touched"] = 0
            m["presses"] = 0
            m["stage"] = "touch"
            return _retry(m)
        b = m["button_xy"] if m["button_xy"] else [-0.126, 0.003]
        m["pending"] = "press"
        return {"memory": m, "act": {"tool": "act_press", "args": {
            "x": float(b[0]) + PRESS_X_OFFSET, "y": float(b[1]), "rot_ref": m["rot_ref"]}}}

    return {"memory": m, "act": {"tool": "act_wait_for_array", "args": {"max_steps": 5}}}


def _retry(m):
    color = m["cue_colors"][0] if m["cue_colors"] else None
    if color is not None and color in m["cube_colors"]:
        p = m["cube_xy"][m["cube_colors"].index(color)]
        m["pending"] = "touch:" + color
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": float(p[0]) + TOUCH_X_OFFSET, "y": float(p[1]), "z_touch": TOUCH_Z - 0.004,
            "rot_ref": m["rot_ref"], "label": color}}}
    m["stage"] = "press"
    b = m["button_xy"] if m["button_xy"] else [-0.126, 0.003]
    m["pending"] = "press"
    return {"memory": m, "act": {"tool": "act_press", "args": {
        "x": float(b[0]) + PRESS_X_OFFSET, "y": float(b[1]), "rot_ref": m["rot_ref"]}}}


def act_wait_for_array(robot, max_steps=60):
    """Hold still (zero motion, gripper open) until the response cube array is visible or max_steps pass."""
    n = 0
    seen = False
    while n < max_steps:
        if _is_array(_detect(robot.now())):
            seen = True
            break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    return {"ok": True, "steps": n, "array_seen": seen, "step": int(robot.now().step)}


def act_touch(robot, x, y, z_touch, rot_ref, label=""):
    """Touch the cube top at world (x, y): travel high, descend to z_touch with open gripper, hold, rise."""
    out = {"label": label, "start_step": int(robot.now().step)}
    out["reach"] = _reach_check(robot, x, y, z_touch, rot_ref)
    z0 = float(robot.now().tcp[2])
    if z0 < TRAVEL_Z - 0.03:
        out["up0"] = _move(robot, [float(robot.now().tcp[0]), float(robot.now().tcp[1]), TRAVEL_Z],
                           rot_ref, 1.0, tol=0.02, vtol=0.05, max_steps=20, ztol=0.02)
    out["over"] = _move(robot, [x, y, TRAVEL_Z], rot_ref, 1.0, tol=0.004, vtol=0.004, max_steps=70, ztol=0.015)
    out["down"] = _move(robot, [x, y, z_touch], rot_ref, 1.0, tol=0.004, vtol=0.003, max_steps=40, ztol=0.0025)
    tcp = robot.now().tcp
    out["bottom"] = [round(float(tcp[0]), 4), round(float(tcp[1]), 4), round(float(tcp[2]), 4)]
    out["hold"] = _move(robot, [x, y, z_touch], rot_ref, 1.0, tol=0.004, vtol=0.003, max_steps=3, min_steps=3)
    out["up"] = _move(robot, [x, y, TRAVEL_Z], rot_ref, 1.0, tol=0.02, vtol=0.05, max_steps=25, ztol=0.02)
    out["ok"] = bool(out["down"]["reached"] or abs(out["bottom"][2] - z_touch) < 0.01)
    out["end_step"] = int(robot.now().step)
    return out


def act_press(robot, x, y, rot_ref):
    """Close the gripper, go above the button cap at (x, y), push down until it stops, then lift high and hold."""
    out = {"start_step": int(robot.now().step)}
    z0 = float(robot.now().tcp[2])
    if z0 < TRAVEL_Z - 0.03:
        out["up0"] = _move(robot, [float(robot.now().tcp[0]), float(robot.now().tcp[1]), TRAVEL_Z],
                           rot_ref, -1.0, tol=0.02, vtol=0.05, max_steps=20, ztol=0.02)
    out["over"] = _move(robot, [x, y, TRAVEL_Z], rot_ref, -1.0, tol=0.004, vtol=0.004, max_steps=70, ztol=0.015)
    out["approach"] = _move(robot, [x, y, 0.075], rot_ref, -1.0, tol=0.004, vtol=0.004, max_steps=30)
    # push: target well below the cap; measured z stops at the mechanical bottom (demo ~0.053)
    prev = np.asarray(robot.now().tcp, dtype=np.float64)
    zs = []
    stall = 0
    n = 0
    while n < 25:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        cmd, err, v = _command(tcp, prev, [x, y, 0.03], rot_ref, vmax=0.5)
        prev = tcp
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], -1.0)
        n += 1
        z = float(robot.now().tcp[2])
        zs.append(round(z, 4))
        if len(zs) >= 2 and abs(zs[-1] - zs[-2]) < 0.0015 and z < 0.066:
            stall += 1
        else:
            stall = 0
        if stall >= 3:
            break
    out["push_steps"] = n
    out["push_z"] = zs[-8:]
    out["press_bottom"] = min(zs) if zs else None
    out["lift"] = _move(robot, [x, y, LIFT_Z], rot_ref, -1.0, tol=0.006, vtol=0.003, max_steps=40, ztol=0.006)
    out["hold"] = _move(robot, [x, y, LIFT_Z], rot_ref, -1.0, tol=0.006, vtol=0.003, max_steps=10, min_steps=10)
    out["ok"] = True
    out["end_step"] = int(robot.now().step)
    return out
