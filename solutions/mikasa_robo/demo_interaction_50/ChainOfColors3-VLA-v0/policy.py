"""ChainOfColors3: remember the cue colour order, touch those cubes in order, press the button."""

DEFAULTS = {
    "stage": "cue",
    "cue": [],
    "cue_runs": [],
    "layout_colors": [],
    "layout_xy": [],
    "button_xy": None,
    "rpy0": None,
    "home": None,
    "touched": 0,
    "pending": "",
    "press_attempts": 0,
    "retouch_done": False,
    "frames_seen": 0,
    "notes": "",
}

Z_SAFE = 0.15
Z_TOUCH = 0.048
Z_TOUCH_RETRY = 0.044


def _init_memory(memory):
    mem = {}
    for k in DEFAULTS:
        v = memory.get(k) if memory else None
        if v is None:
            d = DEFAULTS[k]
            v = list(d) if isinstance(d, list) else d
        mem[k] = v
    return mem


def _final_cue(runs):
    """runs: ['color:count', ...] -> colour order. Drops 1-frame blips when real runs are longer."""
    items = []
    for r in runs:
        parts = str(r).split(":")
        items.append((parts[0], int(parts[1])))
    if len(items) > 3:
        mx = max(c for _, c in items)
        if mx >= 2:
            items = [(n, c) for n, c in items if c >= 2 or c * 2 >= mx]
    out = []
    for n, c in items:
        if not out or out[-1] != n:
            out.append(n)
    return out


def _update_from_frames(mem, frames):
    for f in frames:
        mem["frames_seen"] = int(mem["frames_seen"]) + 1
        if mem["rpy0"] is None:
            t = np.asarray(f.tcp, dtype=np.float64)
            mem["rpy0"] = [float(t[3]), float(t[4]), float(t[5])]
            mem["home"] = [float(t[0]), float(t[1]), float(t[2])]
        if mem["stage"] != "cue":
            continue
        an = _analyze_front(f)
        if _is_layout(an):
            cols = sorted(an["cubes"].keys())
            mem["layout_colors"] = cols
            mem["layout_xy"] = [
                [an["cubes"][c]["xy"][0] + CUBE_XY_OFFSET[0], an["cubes"][c]["xy"][1] + CUBE_XY_OFFSET[1]]
                for c in cols
            ]
            if an["button"] is not None:
                mem["button_xy"] = [an["button"]["xy"][0] + BUTTON_XY_OFFSET[0],
                                    an["button"]["xy"][1] + BUTTON_XY_OFFSET[1]]
            mem["cue"] = _final_cue(mem["cue_runs"])
            mem["stage"] = "touch"
            mem["notes"] = (mem["notes"] + " layout@" + str(f.step) + " n=" + str(len(cols))).strip()
            continue
        c = _cue_color(an)
        if c is not None:
            runs = list(mem["cue_runs"])
            if runs and str(runs[-1]).split(":")[0] == c:
                n = int(str(runs[-1]).split(":")[1]) + 1
                runs[-1] = c + ":" + str(n)
            else:
                runs.append(c + ":1")
            mem["cue_runs"] = runs
    return mem


def perceive_decide(obs, goal, memory, last):
    """Track the cue, locate cubes/button from the front RGB-D, and sequence touch -> press."""
    mem = _init_memory(memory)
    mem = _update_from_frames(mem, obs.frames)
    last_s = str(last) if last is not None else ""
    print("[decide] step", obs.current.step, "stage", mem["stage"], "cue", mem["cue_runs"],
          "touched", mem["touched"], "press", mem["press_attempts"], "| last:", last_s[:600])
    rpy0 = mem["rpy0"]

    # Confirm the previous touch from its measured outcome.
    if mem["pending"]:
        if last_s.startswith("act_touch") and '"reached": true' in last_s:
            mem["touched"] = int(mem["touched"]) + 1
        mem["pending"] = ""
    if last_s.startswith("act_press"):
        mem["press_attempts"] = int(mem["press_attempts"]) + 1

    if mem["stage"] == "cue":
        return {"memory": mem, "act": {"tool": "act_wait_layout", "args": {"max_steps": 60}}}

    cue = list(mem["cue"])
    if mem["stage"] == "touch" and int(mem["touched"]) >= len(cue):
        mem["stage"] = "press"

    if mem["stage"] == "touch":
        color = cue[int(mem["touched"])]
        cols = list(mem["layout_colors"])
        if color not in cols:
            # Unknown target position: skip it (logged) rather than touching a wrong cube.
            mem["notes"] = (mem["notes"] + " missing:" + color).strip()
            mem["touched"] = int(mem["touched"]) + 1
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 1}}}
        xy = mem["layout_xy"][cols.index(color)]
        mem["pending"] = color
        zt = Z_TOUCH_RETRY if mem["retouch_done"] else Z_TOUCH
        return {"memory": mem, "act": {"tool": "act_touch", "args": {
            "x": float(xy[0]), "y": float(xy[1]), "z_touch": zt, "rpy": rpy0}}}

    # press stage
    attempts = int(mem["press_attempts"])
    if attempts >= 2 and not mem["retouch_done"]:
        mem["retouch_done"] = True
        mem["touched"] = 0
        mem["stage"] = "touch"
        mem["notes"] = (mem["notes"] + " retouch").strip()
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 1}}}
    bxy = mem["button_xy"]
    if bxy is None:
        bxy = [-0.105, 0.0]
        mem["notes"] = (mem["notes"] + " button_default").strip()
    # Retries shift the press point slightly along x (both +0.005 and +0.022 offsets succeeded).
    dx = [0.0, -0.006, 0.006][attempts % 3]
    return {"memory": mem, "act": {"tool": "act_press", "args": {
        "x": float(bxy[0]) + dx, "y": float(bxy[1]), "rpy": rpy0, "attempt": attempts}}}


def act_wait_layout(robot, max_steps=60):
    """Hold still (zero command, gripper open) during the cue until the 9-cube layout is visible."""
    n = 0
    seen = False
    for n in range(1, int(max_steps) + 1):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        if _is_layout(_analyze_front(robot.now())):
            seen = True
            break
    return {"steps": n, "layout_seen": seen, "step": int(robot.now().step)}


def act_hold(robot, steps=1):
    """Zero-motion steps with the gripper open."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": int(steps)}


def _reach_check(robot, x, y, z, rpy):
    try:
        r = robot.dry_run(float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2]))
        return bool(r.get("reachable", False))
    except Exception as e:
        return "err:" + str(e)[:60]


def act_touch(robot, x, y, z_touch=0.048, rpy=None, z_safe=0.15, hold=2):
    """Lift to z_safe, move above cube (x,y), descend to z_touch, hold, lift back. Gripper open."""
    t0 = _tcp(robot)
    if rpy is None:
        rpy = [float(t0[3]), float(t0[4]), float(t0[5])]
    reach = _reach_check(robot, x, y, z_touch, rpy)
    out = {"reach": reach, "start": [round(float(c), 3) for c in t0[:3]]}
    st = _new_state()
    horiz = math.hypot(float(t0[0]) - x, float(t0[1]) - y)
    if horiz > 0.03 and float(t0[2]) < z_safe - 0.02:
        out["up"] = _move_to(robot, [t0[0], t0[1], z_safe], rpy, 1.0, st, tol=0.02, vtol=0.02, max_steps=25)["steps"]
    r = _move_to(robot, [x, y, z_safe], rpy, 1.0, st, tol=0.01, vtol=0.006, max_steps=50)
    out["over"] = [r["steps"], round(r["err"], 4)]
    r = _move_to(robot, [x, y, z_touch], rpy, 1.0, st, tol=0.004, vtol=0.002, max_steps=40)
    out["down"] = [r["steps"], round(r["err"], 4)]
    out["bottom"] = _hold(robot, [x, y, z_touch], rpy, 1.0, int(hold), st)
    b = np.array(out["bottom"])
    out["bottom_xy_err"] = round(float(math.hypot(b[0] - x, b[1] - y)), 4)
    out["reached"] = bool(out["bottom_xy_err"] < 0.01 and abs(b[2] - z_touch) < 0.012)
    r = _move_to(robot, [x, y, z_safe], rpy, 1.0, st, tol=0.02, vtol=0.02, max_steps=25)
    out["lift"] = r["steps"]
    out["step"] = int(robot.now().step)
    return out


def act_press(robot, x, y, rpy=None, attempt=0, z_safe=0.15, z_lift=0.22, push_cmd=-0.6):
    """Close gripper, move above button (x,y), push down until stalled, then lift to z_lift and hold."""
    t0 = _tcp(robot)
    if rpy is None:
        rpy = [float(t0[3]), float(t0[4]), float(t0[5])]
    out = {"attempt": int(attempt), "reach": _reach_check(robot, x, y, 0.055, rpy)}
    st = _new_state()
    horiz = math.hypot(float(t0[0]) - x, float(t0[1]) - y)
    if horiz > 0.03 and float(t0[2]) < z_safe - 0.02:
        _move_to(robot, [t0[0], t0[1], z_safe], rpy, -1.0, st, tol=0.02, vtol=0.02, max_steps=25)
    r = _move_to(robot, [x, y, z_safe], rpy, -1.0, st, tol=0.01, vtol=0.006, max_steps=50)
    out["over"] = [r["steps"], round(r["err"], 4)]
    r = _move_to(robot, [x, y, 0.08], rpy, -1.0, st, tol=0.005, vtol=0.003, max_steps=30)
    out["pre"] = [r["steps"], round(r["err"], 4)]
    fing = [round(float(v), 4) for v in robot.now().fingers]
    out["fingers"] = fing
    zs = []
    stall = 0
    for i in range(16):
        prev = _cmd(robot, [x, y, 0.0], rpy, -1.0, st, zcmd=push_cmd)
        z = float(_tcp(robot)[2])
        zs.append(round(z, 4))
        if z < 0.057 and abs(z - float(prev[2])) < 0.0015:
            stall += 1
        if stall >= 3:
            break
    out["push_z"] = zs
    out["z_min"] = min(zs) if zs else None
    r = _move_to(robot, [x, y, z_lift], rpy, -1.0, st, tol=0.01, vtol=0.004, max_steps=30)
    out["lift"] = [r["steps"], r["pos"]]
    out["after"] = _hold(robot, [x, y, z_lift], rpy, 1.0, 8, st)
    out["step"] = int(robot.now().step)
    return out
