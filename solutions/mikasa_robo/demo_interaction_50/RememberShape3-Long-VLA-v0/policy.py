"""RememberShape3-Long policy: remember the cue shape, wait for the three choices,
then move the open gripper's TCP straight down onto the matching object and hold."""

_FIELDS = ["stage", "cue_shape", "cue_info", "cue_step", "choices", "target",
           "target_shape", "touch_tries", "notes"]
NATIVE_LIMIT = 600
_JLIM = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
         (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]


def _joint_margin(frame):
    q = [float(v) for v in frame.joints]
    return min(min(q[i] - _JLIM[i][0], _JLIM[i][1] - q[i]) for i in range(7))
Z_HOVER = 0.13
Z_TOUCH = 0.06


def _mem(memory):
    m = dict(memory or {})
    for k in _FIELDS:
        if k not in m:
            m[k] = None
    if m["stage"] is None:
        m["stage"] = "observe_cue"
    if m["touch_tries"] is None:
        m["touch_tries"] = 0
    return m


def _fmt(f):
    return "%s(x=%.3f,y=%.3f,rg=%.1f,sq=%.2f,margin=%.1f,n=%d,%s)" % (
        f["shape"], f["x"], f["y"], f["rg"], f["sq"], f["margin"], f["n"], f.get("src", ""))


def _step_of(frame, fallback=0):
    s = getattr(frame, "step", None)
    return int(s) if s is not None else int(fallback)


def perceive_decide(obs, goal, memory, last):
    """Stage machine: record the single cue object's shape, wait for the 3-object choice
    scene, select the object with the same shape, then call act_touch on it."""
    m = _mem(memory)
    frames = list(obs.frames)
    cur = obs.current
    step_now = _step_of(cur)

    # 1) Cue: any frame (since the last action) that shows exactly one object.
    if m["cue_shape"] is None:
        for fr in reversed(frames[-6:]):
            objs = _detect_scene(fr)
            if len(objs) == 1:
                f = objs[0]
                m["cue_shape"] = f["shape"]
                m["cue_info"] = _fmt(f)
                m["cue_step"] = _step_of(fr, step_now)
                break

    objs_now = _detect_scene(cur)
    prev_objs = _detect_scene(frames[-2]) if len(frames) >= 2 else objs_now
    choices_visible = len(objs_now) >= 3 and len(prev_objs) >= 3

    if m["target"] is None and choices_visible:
        labels = _assign_choices(objs_now)
        m["choices"] = "; ".join("%s@(%.3f,%.3f)[%s,m=%.0f]" % (labels[i], o["x"], o["y"], o["src"], o["margin"])
                                 for i, o in enumerate(objs_now))
        idx = [i for i, l in enumerate(labels) if l == m["cue_shape"]]
        if len(idx) == 1:
            k = idx[0]
            note = "matched cue shape"
        elif m["cue_shape"] is not None:
            # fall back to the object whose own cost for the cue shape is lowest
            si = SHAPES.index(m["cue_shape"])
            k = min(range(len(objs_now)), key=lambda i: objs_now[i]["costs"][si])
            note = "cue shape not uniquely assigned; nearest cost"
        else:
            k = len(objs_now) // 2
            note = "cue never observed; guessing middle object"
        o = objs_now[k]
        m["target"] = [o["x"], o["y"], o["ztop"]]
        m["target_shape"] = labels[k]
        m["stage"] = "touch"
        m["notes"] = "%s at step %d" % (note, step_now)

    if m["target"] is not None:
        tries = int(m["touch_tries"])
        remaining = NATIVE_LIMIT - step_now
        # escalation schedule if a hold did not end the episode
        plan = [(Z_TOUCH, 0.0, 0.0, False), (0.05, 0.0, 0.0, False),
                (0.065, 0.0, 0.0, True), (0.055, 0.004, 0.0, False),
                (0.055, -0.004, 0.0, False), (0.055, 0.0, 0.004, False),
                (0.055, 0.0, -0.004, False), (0.045, 0.0, 0.0, True)]
        zt, ox, oy, lift = plan[tries % len(plan)]
        m["touch_tries"] = tries + 1
        m["stage"] = "touch_%d" % (tries + 1)
        act = {"tool": "act_touch", "args": {
            "x": float(m["target"][0]), "y": float(m["target"][1]),
            "z_touch": zt, "hold_steps": 15 if remaining > 60 else 40,
            "refine": True, "ox": ox, "oy": oy, "lift_first": lift,
            "max_steps": max(10, min(150, remaining + 5))}}
        return {"memory": {k: m[k] for k in _FIELDS}, "act": act}

    # Waiting: cue visible (1 object) or delay (0 objects) or partial choice view.
    m["stage"] = "wait_cue_gone" if len(objs_now) == 1 else "wait_choices"
    act = {"tool": "act_wait_change", "args": {"n_from": len(objs_now), "max_steps": 700}}
    return {"memory": {k: m[k] for k in _FIELDS}, "act": act}


def act_wait_change(robot, n_from, max_steps=700):
    """Hold still (zero motion, gripper open) until the number of visible blue objects in
    the wrist camera differs from n_from on two consecutive frames (or 3+ objects appear).
    Returns the step and counts observed."""
    changed = 0
    n = n_from
    steps = 0
    for _ in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        steps += 1
        n = len(_detect_scene(robot.now()))
        if n != n_from:
            changed += 1
        else:
            changed = 0
        if changed >= 2:
            break
    return {"steps": steps, "n_from": n_from, "n_now": n,
            "frame_step": _step_of(robot.now(), -1)}


def _ctrl(err, vel, kp=1.5, kd=0.5):
    """PD in metres -> normalised step units (0.1 m per unit), clipped to [-1, 1]."""
    out = []
    for i in range(3):
        a = (kp * err[i] - kd * vel[i]) / 0.1
        out.append(max(-1.0, min(1.0, a)))
    return out


def act_touch(robot, x, y, z_touch=0.06, hold_steps=15, refine=True, max_steps=150,
              ox=0.0, oy=0.0, lift_first=False):
    """Move the open-gripper TCP above (x, y), descend vertically to z_touch only when the
    horizontal error is small, and hold still there for hold_steps.  If refine, the target
    xy is re-measured from the wrist RGB-D on the way (nearest blue object within 25 mm).
    ox/oy offset the aim point from the measured object centre; lift_first raises the TCP
    to 0.10 m before re-descending (a re-press).
    Returns measured TCP, errors and phase timings (success ends the episode)."""
    tx, ty = float(x) + ox, float(y) + oy
    lifted = not lift_first
    meas = []
    f = robot.now()
    p_prev = np.asarray(f.tcp[:3], dtype=np.float64)
    vel = np.zeros(3)
    held = 0
    descending = False
    steps = 0
    t_low = None
    max_hold_speed = 0.0
    log = []
    for _ in range(int(max_steps)):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=np.float64)
        if steps > 0:
            vel = p - p_prev
        p_prev = p
        if refine and p[2] < 0.2 and steps % 2 == 0:
            objs = _detect_objects(f)
            best = None
            for o in objs:
                d = math.hypot(o["x"] - float(x), o["y"] - float(y))
                if d < 0.025 and not o["border"] and o["n"] >= 150:
                    if best is None or d < best[0]:
                        best = (d, o)
            if best is not None:
                meas.append((best[1]["x"], best[1]["y"]))
                mx = sum(a for a, b in meas[-6:]) / len(meas[-6:])
                my = sum(b for a, b in meas[-6:]) / len(meas[-6:])
                tx, ty = mx + ox, my + oy
        exy = math.hypot(tx - p[0], ty - p[1])
        if not lifted:
            if p[2] > 0.095:
                lifted = True
            else:
                a = _ctrl([0.0, 0.0, 0.10 - p[2]], vel)
                robot.step(a[0], a[1], a[2], 0.0, 0.0, 0.0, 1.0)
                steps += 1
                continue
        if exy < 0.006 or (descending and exy < 0.012):
            descending = True
            zg = z_touch
        elif exy < 0.02:
            zg = max(0.09, z_touch)
        else:
            zg = Z_HOVER
        if exy >= 0.012:
            descending = False
        if p[2] < 0.085 and exy > 0.012:
            zg = max(zg, 0.10)  # retreat upward if we drift off the target while low
        err = [tx - p[0], ty - p[1], zg - p[2]]
        a = _ctrl(err, vel)
        at_goal = zg == z_touch and abs(err[2]) < 0.004 and exy < 0.004
        if at_goal:
            if t_low is None:
                t_low = steps
            held += 1
            sp = float(np.sqrt((vel ** 2).sum()))
            max_hold_speed = max(max_hold_speed, sp) if held > 3 else max_hold_speed
            if held > hold_steps:
                break
        if steps % 5 == 0:
            log.append([steps, round(p[0], 4), round(p[1], 4), round(p[2], 4), round(exy, 4)])
        robot.step(a[0], a[1], a[2], 0.0, 0.0, 0.0, 1.0)
        steps += 1
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=np.float64)
    return {"steps": steps, "target_xy": [tx, ty], "n_refine": len(meas),
            "final_tcp": [float(v) for v in f.tcp], "xy_err": math.hypot(tx - p[0], ty - p[1]),
            "z": float(p[2]), "held": held, "t_low": t_low,
            "max_hold_speed": max_hold_speed, "frame_step": _step_of(f, -1),
            "joint_margin": _joint_margin(f), "fingers": [float(v) for v in f.fingers],
            "log": log[-12:]}
