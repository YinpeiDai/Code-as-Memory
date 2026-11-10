"""BunchOfColors7-Long policy: remember cue colours, wait for the response cubes,
touch every cue-coloured cube (shortest order), then press the centre button."""

_Z_HOVER = 0.10       # approach height of the TCP (cube tops are at ~0.04 m)
_Z_TOUCH = 0.052      # touch depth target (expert minima 0.045-0.063 m)
_Z_LIFT_EXIT = 0.085  # lift height reached before the next lateral move
_Z_PRESS = 0.035      # press command depth (expert TCP stalls on the button at ~0.053 m)
_TOUCH_OFFSET = (-0.0075, 0.0025)  # expert touch point minus our top-face centroid (350 touches)
_CUE_MIN_PIX = 150    # colour pixels needed to call the cue/response "visible"
_GONE_PIX = 20        # colour pixels below which the table is "empty"
_Z_TOUCH_RETRY = 0.047  # deeper touch depth used for a second pass if the press gave no success
_MEM_KEYS = ("stage", "cue_colors", "cue_step", "rpy_ref", "response_cubes", "response_step",
             "plan_colors", "plan_positions", "next_index", "touched", "attempts", "button", "pass_count",
             "presses", "log")


def _plan_order(start_xy, targets, end_xy):
    """Exact shortest path start -> all targets -> end (Held-Karp, targets <= 9)."""
    n = len(targets)
    if n == 0:
        return []
    P = np.array([[t[0], t[1]] for t in targets], dtype=np.float64)
    s = np.asarray(start_xy, dtype=np.float64)
    e = np.asarray(end_xy, dtype=np.float64)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    ds = np.sqrt(((P - s) ** 2).sum(-1))
    de = np.sqrt(((P - e) ** 2).sum(-1))
    INF = 1e9
    N = 1 << n
    cost = np.full((N, n), INF)
    par = np.full((N, n), -1, dtype=int)
    for i in range(n):
        cost[1 << i, i] = ds[i]
    for m in range(1, N):
        for i in range(n):
            if not (m >> i) & 1:
                continue
            c = cost[m, i]
            if c >= INF:
                continue
            for j in range(n):
                if (m >> j) & 1:
                    continue
                v = c + D[i, j]
                m2 = m | (1 << j)
                if v < cost[m2, j]:
                    cost[m2, j] = v
                    par[m2, j] = i
    full = N - 1
    i = int(np.argmin(cost[full] + de))
    order = []
    m = full
    while i >= 0:
        order.append(i)
        pi = int(par[m, i])
        m = m & ~(1 << i)
        i = pi
    return list(reversed(order))


def _scan_cue(frames):
    """Colours of the first visible cube set (the cue) in a frame sequence; None if not seen."""
    best, bestn = None, -1
    for f in frames:
        n = _color_count(f)
        if n >= _CUE_MIN_PIX and n > bestn:
            best, bestn = f, n
        if best is not None and n < _GONE_PIX:
            break  # cue segment ended
    if best is None:
        return None, None
    cols = sorted(set(c["color"] for c in _frame_cubes(best)))
    return (cols or None), int(best.step)


def _make_plan(mem, frame):
    cubes = _frame_cubes(frame)
    mem["response_cubes"] = [[c["color"], round(c["x"], 4), round(c["y"], 4)] for c in cubes]
    mem["response_step"] = int(frame.step)
    b = _button_estimate(frame)
    mem["button"] = [b[0], b[1]] if b is not None else [-0.11, 0.0]
    cue = mem["cue_colors"] or []
    by_color = {}
    for c in cubes:
        if c["color"] in cue:
            # if a colour is detected twice keep the larger blob
            if c["color"] not in by_color or c["n"] > by_color[c["color"]]["n"]:
                by_color[c["color"]] = c
    cols = sorted(by_color)
    tg = [[by_color[k]["x"] + _TOUCH_OFFSET[0], by_color[k]["y"] + _TOUCH_OFFSET[1]] for k in cols]
    tcp = frame.tcp
    order = _plan_order([float(tcp[0]), float(tcp[1])], tg, mem["button"])
    mem["plan_colors"] = [cols[i] for i in order]
    mem["plan_positions"] = [[round(tg[i][0], 4), round(tg[i][1], 4)] for i in order]
    missing = [c for c in cue if c not in by_color]
    mem["log"] = (mem["log"] or "") + "plan@%d %s missing=%s; " % (int(frame.step), ",".join(mem["plan_colors"]), missing)


def _next_act(mem):
    idx = int(mem["next_index"] or 0)
    plan = mem["plan_positions"] or []
    if idx < len(plan):
        p = plan[idx]
        mem["stage"] = "touch"
        zt = _Z_TOUCH if int(mem["pass_count"] or 0) == 0 else _Z_TOUCH_RETRY
        return {"tool": "act_touch", "args": {"x": float(p[0]), "y": float(p[1]), "color": mem["plan_colors"][idx],
                                              "rpy_ref": list(mem["rpy_ref"]), "z_touch": zt}}
    b = mem["button"] or [-0.11, 0.0]
    mem["stage"] = "press"
    return {"tool": "act_press_button", "args": {"x": float(b[0]), "y": float(b[1]), "rpy_ref": list(mem["rpy_ref"])}}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: record cue colours -> wait for response cubes -> touch each cue colour -> press button."""
    mem = {k: memory.get(k) for k in _MEM_KEYS}
    cur = obs.current
    if mem["stage"] is None:
        mem["stage"] = "wait"
        mem["rpy_ref"] = [float(cur.tcp[3]), float(cur.tcp[4]), float(cur.tcp[5])]
        mem["touched"] = []
        mem["attempts"] = 0
        mem["pass_count"] = 0
        mem["presses"] = 0
        mem["next_index"] = 0
        mem["log"] = ""
        cols, st = _scan_cue(obs.frames)
        if cols:
            mem["cue_colors"], mem["cue_step"] = cols, st
            mem["log"] += "cue@%d %s; " % (st, ",".join(cols))
        return {"memory": mem, "act": {"tool": "act_wait_for_response", "args": {"cue_known": bool(cols)}}}
    if mem["stage"] == "wait":
        if not mem["cue_colors"]:
            cols, st = _scan_cue(obs.frames)
            if cols:
                mem["cue_colors"], mem["cue_step"] = cols, st
                mem["log"] += "cue(late)@%d %s; " % (st, ",".join(cols))
        appeared = last is not None and '"appeared": true' in last
        if appeared and _color_count(cur) >= _CUE_MIN_PIX:
            _make_plan(mem, cur)
            return {"memory": mem, "act": _next_act(mem)}
        return {"memory": mem, "act": {"tool": "act_wait_for_response", "args": {"cue_known": bool(mem["cue_colors"])}}}
    if mem["stage"] == "touch":
        ok = last is not None and '"reached": true' in last
        idx = int(mem["next_index"] or 0)
        if ok or int(mem["attempts"] or 0) >= 1:
            mem["touched"] = list(mem["touched"] or []) + [mem["plan_colors"][idx] + ("" if ok else "?")]
            mem["next_index"] = idx + 1
            mem["attempts"] = 0
        else:
            mem["attempts"] = int(mem["attempts"] or 0) + 1
        return {"memory": mem, "act": _next_act(mem)}
    if mem["stage"] == "press":
        # The press act returned, so the episode did not end with success: some touch was probably not registered.
        mem["presses"] = int(mem["presses"] or 0) + 1
        mem["log"] += "press%d_no_success %s; " % (mem["presses"], str(last)[:120])
        n_plan = len(mem["plan_positions"] or [])
        steps_left = 700 - int(cur.step)
        if int(mem["pass_count"] or 0) < 1 and steps_left > 15 * n_plan + 35:
            mem["pass_count"] = int(mem["pass_count"] or 0) + 1
            mem["next_index"] = 0
            mem["attempts"] = 0
            mem["log"] += "retry_pass@%d; " % int(cur.step)
            return {"memory": mem, "act": _next_act(mem)}
        b = mem["button"] or [-0.11, 0.0]
        return {"memory": mem, "act": {"tool": "act_press_button", "args": {"x": float(b[0]), "y": float(b[1]),
                                                                          "rpy_ref": list(mem["rpy_ref"])}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


def act_wait_for_response(robot, cue_known=True, max_steps=690):
    """Hold still (zero motion, gripper open) until the response cubes appear after the cue vanished.

    Returns appeared, steps used, the step of the cue end and of the appearance, and pixel counts."""
    seen = bool(cue_known)
    gone_run = 0
    gone = False
    vis_run = 0
    cue_end = None
    n = 0
    steps = 0
    for _ in range(max_steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        steps += 1
        f = robot.now()
        n = _color_count(f)
        if not seen:
            if n >= _CUE_MIN_PIX:
                seen = True
            continue
        if not gone:
            gone_run = gone_run + 1 if n < _GONE_PIX else 0
            if gone_run >= 3:
                gone = True
                cue_end = int(f.step)
            continue
        vis_run = vis_run + 1 if n >= _CUE_MIN_PIX else 0
        if vis_run >= 2:
            return {"appeared": True, "steps": steps, "cue_end_step": cue_end, "appear_step": int(f.step), "pixels": n}
    return {"appeared": False, "steps": steps, "cue_end_step": cue_end, "seen": seen, "gone": gone, "pixels": n}


def act_touch(robot, x, y, color="", rpy_ref=None, z_touch=0.052, max_steps=70):
    """Move above (x, y), descend to the touch depth with the gripper open, dwell, lift to clear height.

    Returns reached (touch depth reached near the target), minimum z, xy error at the bottom, steps used."""
    ref = rpy_ref if rpy_ref is not None else [math.pi, 0.0, 0.0]
    steps = 0
    phase = "approach"
    zmin = 9.0
    xy_at_min = 9.0
    dwell = 0
    reached = False
    for _ in range(max_steps):
        tcp = robot.now().tcp
        exy = math.hypot(x - float(tcp[0]), y - float(tcp[1]))
        z = float(tcp[2])
        if phase == "approach":
            if exy > 0.03 and z < _Z_LIFT_EXIT - 0.01:
                tgt = [float(tcp[0]), float(tcp[1]), _Z_HOVER]      # climb first, no lateral sweep low
            elif exy > 0.02:
                tgt = [x, y, _Z_HOVER]
            else:
                tgt = [x, y, z_touch]
            if z < z_touch + 0.007 and exy < 0.012:
                phase = "dwell"
        if phase == "dwell":
            tgt = [x, y, z_touch]
            dwell += 1
            reached = True
            if dwell > 2:
                phase = "lift"
        if phase == "lift":
            tgt = [x, y, _Z_HOVER + 0.01]
            if z > _Z_LIFT_EXIT:
                break
        if z < zmin:
            zmin = z
            xy_at_min = exy
        _step_toward(robot, tgt, ref, 1.0)
        steps += 1
    tcp = robot.now().tcp
    return {"reached": reached, "color": color, "z_min": zmin, "xy_err_at_min": xy_at_min, "steps": steps,
            "phase": phase, "tcp": [float(v) for v in tcp[:3]], "step": int(robot.now().step)}


def act_press_button(robot, x, y, rpy_ref=None, max_steps=90):
    """Close the gripper, move above the button, press down until the TCP stalls, hold, then lift and wait.

    Returns stall height, steps and phase (the episode normally ends with success during the lift)."""
    ref = rpy_ref if rpy_ref is not None else [math.pi, 0.0, 0.0]
    steps = 0
    phase = "approach"
    hold = 0
    stall = 0
    zprev = None
    zstall = None
    waited = 0
    for _ in range(max_steps):
        tcp = robot.now().tcp
        exy = math.hypot(x - float(tcp[0]), y - float(tcp[1]))
        z = float(tcp[2])
        if phase == "approach":
            if exy > 0.03 and z < _Z_LIFT_EXIT - 0.01:
                tgt = [float(tcp[0]), float(tcp[1]), _Z_HOVER]
            elif exy > 0.015:
                tgt = [x, y, _Z_HOVER]
            else:
                tgt = [x, y, _Z_PRESS]
                if zprev is not None and z < 0.064 and abs(z - zprev) < 0.0015:
                    stall += 1
                if stall >= 2 or z < 0.050:
                    phase = "hold"
                    zstall = z
        if phase == "hold":
            tgt = [x, y, _Z_PRESS + 0.008]
            hold += 1
            if hold > 5:
                phase = "lift"
        if phase == "lift":
            tgt = [x, y, 0.18]
        zprev = z
        _step_toward(robot, tgt, ref, -1.0)
        steps += 1
        if phase == "lift" and float(robot.now().tcp[2]) > 0.17:
            phase = "lifted_wait"
        if phase == "lifted_wait":
            waited += 1
            if waited > 15:
                break
    tcp = robot.now().tcp
    return {"phase": phase, "z_stall": zstall, "steps": steps, "tcp": [float(v) for v in tcp[:3]],
            "step": int(robot.now().step)}


def act_hold(robot, steps=20):
    """Hold position with zero motion command for a number of steps."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": int(steps), "step": int(robot.now().step)}
