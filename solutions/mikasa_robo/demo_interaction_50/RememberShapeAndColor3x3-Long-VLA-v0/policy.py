"""RememberShapeAndColor3x3-Long policy: remember the cue (colour + shape) shown at reset,
wait (arm still) until the nine candidate objects appear, then move above the matching
object and descend slowly so the TCP comes close to it; hold, then fall back lower."""

FIELDS = ("stage", "cue_color", "cue_shape", "cue_conf", "cue_step", "target",
          "target_shape", "objects", "attempt", "log")
FINAL_Z = {"cube": 0.045, "sphere": 0.045, "T": 0.031}


def _mem(memory):
    m = {k: memory.get(k) for k in FIELDS}
    if m["attempt"] is None:
        m["attempt"] = 0
    if m["log"] is None:
        m["log"] = ""
    return m


def _find_cue(frames):
    for f in frames:
        c = _color_counts(f.wrist_rgb)
        tot = sum(c.values())
        if 30 <= tot <= 600:
            cue = _parse_cue(f)
            if cue is not None:
                return cue, int(f.step)
    return None, None


def _choose_target(objs, color, shape):
    same = [o for o in objs if o["color"] == color]
    exact = [o for o in same if o["shape"] == shape]
    if exact:
        if len(exact) > 1:
            ab = [o for o in exact if o["abs_shape"] == shape]
            if ab:
                exact = ab
        return exact[0], "exact" if len(same) == 3 else "exact_unconstrained"
    ab = [o for o in same if o["abs_shape"] == shape]
    if ab:
        return ab[0], "abs_only"
    if same:
        return same[0], "color_only"
    return None, "none"


def perceive_decide(obs, goal, memory, last):
    """Stage machine: cue -> wait for objects -> touch matching object -> fallbacks."""
    m = _mem(memory or {})
    stage = m["stage"] or "start"
    if last:
        m["log"] = (m["log"] + " | " + str(last)[:600])[-3000:]
    if m["cue_color"] is None:
        cue, step = _find_cue(obs.frames)
        if cue is not None:
            m["cue_color"], m["cue_shape"], m["cue_conf"], m["cue_step"] = cue["color"], cue["shape"], cue["conf"], step
            m["log"] = (m["log"] + " | cue " + str(cue))[-3000:]
    cur = obs.current
    if stage in ("start", "waiting"):
        if not _scene_ready(cur.wrist_rgb):
            m["stage"] = "waiting"
            return {"memory": m, "act": {"tool": "act_wait_for_objects", "args": {"max_steps": 590}}}
        objs = _assign_shapes(_detect(cur, "wrist"))
        m["objects"] = "; ".join("%s %s(%s) %.3f,%.3f zt%.3f sr%.1f n%d" % (
            o["color"], o["shape"], o["abs_shape"], o["cx"], o["cy"], o["zt"], o["sres"], o["n"]) for o in objs)
        color = m["cue_color"] or "red"
        shape = m["cue_shape"] or "cube"
        tgt, how = _choose_target(objs, color, shape)
        if tgt is None:
            m["stage"] = "waiting"
            return {"memory": m, "act": {"tool": "act_wait_for_objects", "args": {"max_steps": 5}}}
        m["target"] = [round(tgt["cx"], 4), round(tgt["cy"], 4), round(FINAL_Z[shape], 4)]
        m["target_shape"] = shape
        m["stage"] = "touch"
        m["attempt"] = 1
        m["log"] = (m["log"] + " | target %s %s via %s at step %d" % (color, shape, how, int(cur.step)))[-3000:]
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": m["target"][0], "y": m["target"][1], "shape": shape, "color": color,
            "final_z": FINAL_Z[shape], "hold_steps": 10, "grip": 1.0, "refine": True}}}
    # previous touch returned without success: progressively lower, then close gripper
    m["attempt"] = int(m["attempt"]) + 1
    shape = m["target_shape"] or "cube"
    color = m["cue_color"] or "red"
    x, y = m["target"][0], m["target"][1]
    k = m["attempt"]
    if k == 2:
        fz, grip = FINAL_Z[shape] - 0.008, 1.0
    elif k == 3:
        fz, grip = FINAL_Z[shape] - 0.014, 1.0
    else:
        fz, grip = FINAL_Z[shape] - 0.008, -1.0
    m["stage"] = "fallback%d" % k
    return {"memory": m, "act": {"tool": "act_touch", "args": {
        "x": x, "y": y, "shape": shape, "color": color, "final_z": round(fz, 4),
        "hold_steps": 10, "grip": grip, "refine": True}}}


def act_wait_for_objects(robot, max_steps=590):
    """Hold the arm still (zero command, gripper open) until all nine candidate objects are
    visible in the wrist camera for two consecutive frames, or max_steps elapse. Reports steps."""
    n = 0
    ready = 0
    first_empty = None
    while n < max_steps:
        f = robot.now()
        if _scene_ready(f.wrist_rgb):
            ready += 1
            if ready >= 2:
                break
        else:
            ready = 0
            if first_empty is None and sum(_color_counts(f.wrist_rgb).values()) == 0:
                first_empty = int(f.step)
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    f = robot.now()
    return {"steps": n, "ready": ready >= 2, "cue_gone_step": first_empty, "now_step": int(f.step),
            "counts": _color_counts(f.wrist_rgb)}


def _refine(frame, x, y, shape, color):
    objs = [o for o in _assign_shapes(_detect(frame, "wrist", expect=0)) if o["color"] == color]
    best = None
    for o in objs:
        cx, cy = _center(o, o["abs_shape"])
        d = math.hypot(cx - x, cy - y)
        if o["abs_shape"] == shape and d < 0.03 and (best is None or d < best[0]):
            best = (d, cx, cy)
    return best


def act_touch(robot, x, y, shape, color, final_z=0.045, hover_z=0.12, hold_steps=10,
              grip=1.0, refine=True):
    """Move above (x, y) at hover_z (orientation held gripper-down), optionally refine the target
    centre from the close wrist view, descend slowly to final_z, then hold. Returns measured path."""
    f0 = robot.now()
    tcp0 = np.asarray(f0.tcp, dtype=float)
    R_des = _rpy2R(math.pi, 0.0, float(tcp0[5]))
    out = {"start_step": int(f0.step), "start_tcp": [round(float(v), 4) for v in tcp0]}
    try:
        chk = robot.dry_run(x, y, hover_z, math.pi, 0.0, float(tcp0[5]), mode="absolute")
        out["dry_hover"] = {"ok": chk.get("within_joint_limits"), "min_margin": chk.get("min_joint_margin_rad")}
    except Exception as e:
        out["dry_hover"] = str(e)[:80]
    goal = [x, y, hover_z]
    steps = 0
    # phase 1: transit to hover
    for i in range(45):
        tcp = robot.now().tcp
        cmd, err, rerr = _servo_cmd(tcp, goal, R_des, vmax=0.1)
        if math.hypot(err[0], err[1]) < 0.008 and abs(err[2]) < 0.015:
            break
        robot.step(*cmd, grip)
        steps += 1
    out["transit_steps"] = steps
    out["hover_tcp"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    if refine:
        best = _refine(robot.now(), x, y, shape, color)
        if best is not None:
            out["refine"] = [round(best[1] - x, 4), round(best[2] - y, 4)]
            x, y = best[1], best[2]
        else:
            out["refine"] = None
    # phase 2: slow descent
    goal = [x, y, final_z]
    zs = []
    for i in range(40):
        tcp = robot.now().tcp
        cmd, err, rerr = _servo_cmd(tcp, goal, R_des, vmax=0.02)
        zs.append(round(float(tcp[2]), 4))
        if abs(err[2]) < 0.003 and math.hypot(err[0], err[1]) < 0.004:
            break
        robot.step(*cmd, grip)
        steps += 1
    out["descent_z"] = zs
    # phase 3: hold
    j0 = np.asarray(robot.now().joints, dtype=float)
    for i in range(hold_steps):
        tcp = robot.now().tcp
        cmd, err, rerr = _servo_cmd(tcp, goal, R_des, vmax=0.01)
        robot.step(*cmd, grip)
        steps += 1
    f = robot.now()
    out["hold_joint_change"] = round(float(np.abs(np.asarray(f.joints, dtype=float) - j0).max()), 4)
    out["final_tcp"] = [round(float(v), 4) for v in f.tcp]
    out["goal"] = [round(float(v), 4) for v in goal]
    out["steps"] = steps
    out["end_step"] = int(f.step)
    return out
