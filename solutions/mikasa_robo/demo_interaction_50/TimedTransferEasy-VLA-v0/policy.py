"""TimedTransferEasy policy: watch lamp, pick blue cube from green disc, hover over red disc,
descend so that the cube reaches the red disc just after count 100 from the lamp-green frame."""

# ---- parameters (from demonstrations; see task_analysis.md) ----
HOVER_Z = 0.19          # expert hover height above red disc (0.191-0.195 in all 50 demos)
PRE_GRASP_Z = 0.10      # approach height above the cube
GRASP_Z = 0.032         # expert grasp TCP z 0.025-0.043 (mean 0.034)
PLACE_Z = 0.042         # final TCP z while holding cube on the red disc
DESC_COUNT = 95         # descent start; live crossing of the placement height happens at DESC_COUNT+5 = count 100,
                        # centre of the measured success window [95, 105] (R003-R008)
XY_BIAS = -0.008        # expert TCP x is ~0.007-0.010 below front-cam estimates of cube / disc
GIVE_UP_COUNT = 135     # if no success by this count while holding low, release and retreat
TARGET_RPY = (math.pi, 0.0, 0.0)


def _masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    blue = (b > r + 50) & (np.abs(r - g) < 20) & (b > 100)
    red = (r > g + 70) & (np.abs(g - b) < 20) & (r > 120)
    lamp = (g > 220) & (r < 80) & (b < 80)
    return blue, red, lamp


def _deproject(depth, K, E, mask):
    ys, xs = np.nonzero(mask & (depth > 0))
    if len(ys) == 0:
        return None
    z = depth[ys, xs].astype(float) / 1000.0
    xc = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    yc = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    P = np.stack([xc, yc, z], 1)
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return (P - t) @ R


def _lamp_green(frame):
    """Lamp is pure green (~(5,255,5)). Front camera first; wrist camera as backup if the front view is
    occluded (both give identical first-green frames on all 50 demos)."""
    _, _, lamp = _masks(frame.front_rgb)
    if int(lamp.sum()) > 30:
        return True
    _, _, lamp_w = _masks(frame.wrist_rgb)
    return int(lamp_w.sum()) > 30


def _locate(frame):
    """Front-camera estimates of cube top-face centre and red-disc centre (world xy)."""
    blue, red, _ = _masks(frame.front_rgb)
    d = frame.front_depth
    K = frame.front_K
    E = frame.front_extrinsic
    out = {"cube": None, "red": None, "cube_px": int(blue.sum()), "red_px": int(red.sum())}
    Wb = _deproject(d, K, E, blue)
    if Wb is not None:
        top = Wb[Wb[:, 2] > 0.035]
        if len(top) >= 5:
            out["cube"] = [float(top[:, 0].mean()), float(top[:, 1].mean()), float(np.median(top[:, 2]))]
    Wr = _deproject(d, K, E, red)
    if Wr is not None:
        Wr = Wr[Wr[:, 2] < 0.02]
        if len(Wr) >= 30:
            lo = np.percentile(Wr[:, :2], 2, axis=0)
            hi = np.percentile(Wr[:, :2], 98, axis=0)
            c = (lo + hi) / 2.0
            out["red"] = [float(c[0]), float(c[1])]
            out["red_extent"] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    return out


def _rot(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logm(M):
    c = (np.trace(M) - 1.0) / 2.0
    th = math.acos(max(-1.0, min(1.0, c)))
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    return th / (2.0 * s) * v


def _rot_err(tcp):
    """World-frame rotation vector w with R_target = exp(w) R_current."""
    Rt = _rot(*TARGET_RPY)
    Rc = _rot(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    return _logm(Rt @ Rc.T)


def _clock_update(state, frames):
    """state: dict with green_step (or None). Scan frames for the first green-lamp frame."""
    if state["green_step"] is not None:
        return
    for f in frames:
        if _lamp_green(f):
            state["green_step"] = int(f.step)
            return


def _count(state, frame):
    if state["green_step"] is None:
        return None
    return int(frame.step) - int(state["green_step"])


def _servo_cmd(tcp, target, vmax, kp=10.0, rot_on=True):
    err = np.asarray(target, dtype=float) - np.asarray(tcp[:3], dtype=float)
    a = np.clip(err * kp, -vmax, vmax)
    r = np.zeros(3)
    if rot_on:
        w = _rot_err(tcp)
        r = np.clip(-w / 0.1 * 0.4, -0.3, 0.3)
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])], float(np.linalg.norm(err))


def _step(robot, cmd, grip, state):
    n_before = len(robot.frames())
    robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
    fr = robot.frames()[n_before:]
    _clock_update(state, fr)
    return robot.now()


def _go(robot, target, grip, state, tol=0.004, max_steps=40, vmax=1.0, settle_speed=0.002, rot_on=True):
    """Closed-loop move of TCP to target xyz. Returns (steps, final error, max rot err)."""
    prev = np.asarray(robot.now().tcp[:3], dtype=float)
    n = 0
    err = 1.0
    max_rot = 0.0
    for n in range(1, max_steps + 1):
        tcp = robot.now().tcp
        we = float(np.linalg.norm(_rot_err(tcp)))
        max_rot = max(max_rot, we)
        use_rot = rot_on and state.get("rot_ok", True)
        if we > 0.25:
            state["rot_ok"] = False
            use_rot = False
        cmd, err = _servo_cmd(tcp, target, vmax, rot_on=use_rot)
        fr = _step(robot, cmd, grip, state)
        cur = np.asarray(fr.tcp[:3], dtype=float)
        speed = float(np.linalg.norm(cur - prev))
        prev = cur
        err = float(np.linalg.norm(np.asarray(target) - cur))
        if err < tol and speed < settle_speed:
            break
    return n, err, max_rot


def _new_state(memory):
    gs = memory.get("green_step")
    return {"green_step": (None if gs is None or gs < 0 else int(gs)), "rot_ok": True}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: locate -> pick -> transport/hover -> timed place -> done.
    The lamp-green step is taken from the first frame whose front image shows the green lamp."""
    mem = dict(memory) if memory else {}
    stage = mem.get("stage") or "start"
    frame = obs.current
    gs = mem.get("green_step")
    if gs is None or gs < 0:
        gs = -1
        for f in obs.frames:
            if _lamp_green(f):
                gs = int(f.step)
                break
    notes = mem.get("notes") or ""
    cube = mem.get("cube_xyz") or []
    red = mem.get("red_xy") or []
    attempts = int(mem.get("attempts") or 0) + 1
    nxt = _next_stage_from_last(last)
    if nxt:
        stage = nxt
    if stage == "start" or (stage == "pick" and nxt == "pick"):
        loc = _locate(frame)
        if loc["cube"]:
            cube = loc["cube"]
        if loc["red"] and not red:
            red = loc["red"]
        notes = (notes + " | " if notes else "") + "locate@%d cube_px %d red_px %d cube %s red %s" % (
            int(frame.step), loc["cube_px"], loc["red_px"],
            str([round(v, 4) for v in cube]), str([round(v, 4) for v in red]))
        notes = notes[-600:]
        stage = "pick"
    if stage in ("transport", "place") and not red:
        loc = _locate(frame)
        red = loc["red"] or []
    act = {"tool": "act_observe", "args": {"steps": 2}}
    if stage == "pick" and cube:
        act = {"tool": "act_pick", "args": {"x": cube[0] + XY_BIAS, "y": cube[1], "green_step": gs}}
    elif stage == "transport" and red:
        act = {"tool": "act_transport", "args": {"x": red[0] + XY_BIAS, "y": red[1], "green_step": gs}}
    elif stage == "place" and red:
        act = {"tool": "act_timed_place", "args": {"x": red[0] + XY_BIAS, "y": red[1], "green_step": gs}}
    elif stage == "done":
        act = {"tool": "act_observe", "args": {"steps": 20}}
    mem_out = {"stage": stage, "green_step": gs, "cube_xyz": cube, "red_xy": red,
               "notes": notes, "attempts": attempts}
    return {"memory": mem_out, "act": act}


def _next_stage_from_last(last):
    if last is None:
        return None
    s = last if isinstance(last, str) else str(last)
    i = s.find('"next_stage": "')
    if i < 0:
        i = s.find("'next_stage': '")
    if i < 0:
        return None
    j = i + len('"next_stage": "')
    k = j
    while k < len(s) and s[k] not in "\"'":
        k += 1
    return s[j:k]


def act_observe(robot, steps=1):
    """Hold still (gripper unchanged-open) for `steps` native steps; reports lamp state."""
    state = {"green_step": None, "rot_ok": True}
    grip = 1.0 if float(np.mean(robot.now().fingers)) > 0.03 else -1.0
    for _ in range(int(steps)):
        _step(robot, [0, 0, 0, 0, 0, 0], grip, state)
    return {"step": int(robot.now().step), "green_step_seen": state["green_step"]}


def act_pick(robot, x, y, green_step=-1):
    """Approach above (x,y), descend, close gripper, lift to hover height. Reports grasp evidence."""
    state = {"green_step": (None if green_step is None or green_step < 0 else int(green_step)), "rot_ok": True}
    _clock_update(state, [robot.now()])
    log = {"next_stage": "pick"}
    pre = [float(x), float(y), PRE_GRASP_Z]
    chk = robot.dry_run(pre[0], pre[1], pre[2], TARGET_RPY[0], TARGET_RPY[1], TARGET_RPY[2], mode="absolute")
    log["dry_pre_ok"] = bool(chk.get("within_joint_limits", False))
    log["dry_pre_margin"] = chk.get("min_joint_margin_rad")
    n1, e1, r1 = _go(robot, pre, 1.0, state, tol=0.006, max_steps=35)
    n2, e2, r2 = _go(robot, [float(x), float(y), GRASP_Z], 1.0, state, tol=0.004, max_steps=30, vmax=0.6)
    for _ in range(6):
        _step(robot, [0, 0, 0, 0, 0, 0], -1.0, state)
    f_closed = float(np.mean(robot.now().fingers))
    n3, e3, r3 = _go(robot, [float(x), float(y), HOVER_Z], -1.0, state, tol=0.01, max_steps=30, vmax=0.8)
    f_lift = float(np.mean(robot.now().fingers))
    held = f_lift > 0.008
    log.update({"steps": [n1, n2, n3], "errs": [round(e1, 4), round(e2, 4), round(e3, 4)],
                "max_rot_err": round(max(r1, r2, r3), 3), "fingers_closed": round(f_closed, 4),
                "fingers_after_lift": round(f_lift, 4), "held": held,
                "tcp": [round(float(v), 4) for v in robot.now().tcp],
                "step": int(robot.now().step), "green_step": state["green_step"],
                "rot_ok": state["rot_ok"]})
    log["next_stage"] = "transport" if held else "pick"
    return log


def act_transport(robot, x, y, green_step=-1):
    """Carry the grasped cube at hover height to above (x,y) and settle there."""
    state = {"green_step": (None if green_step is None or green_step < 0 else int(green_step)), "rot_ok": True}
    _clock_update(state, [robot.now()])
    tgt = [float(x), float(y), HOVER_Z]
    chk = robot.dry_run(tgt[0], tgt[1], tgt[2], TARGET_RPY[0], TARGET_RPY[1], TARGET_RPY[2], mode="absolute")
    n, e, r = _go(robot, tgt, -1.0, state, tol=0.004, max_steps=45, vmax=0.8)
    f = float(np.mean(robot.now().fingers))
    held = f > 0.008
    return {"next_stage": "place" if held else "pick", "steps": n, "err": round(e, 4), "max_rot_err": round(r, 3), "fingers": round(f, 4),
            "held": held, "dry_ok": bool(chk.get("within_joint_limits", False)),
            "tcp": [round(float(v), 4) for v in robot.now().tcp], "step": int(robot.now().step),
            "green_step": state["green_step"]}


def act_timed_place(robot, x, y, green_step=-1):
    """Hover over (x,y) until count DESC_COUNT after lamp-green, then lower the cube onto the disc and hold.
    Episode is expected to end with native success during/after descent."""
    state = {"green_step": (None if green_step is None or green_step < 0 else int(green_step)), "rot_ok": True}
    _clock_update(state, [robot.now()])
    hover = [float(x), float(y), HOVER_Z]
    low = [float(x), float(y), PLACE_Z]
    trace = []
    n = 0
    while n < 400:
        fr = robot.now()
        c = _count(state, fr)
        if c is None:
            c = int(fr.step)  # lamp never seen: fall back to raw step
        if c >= GIVE_UP_COUNT:
            break
        if c < DESC_COUNT:
            cmd, _ = _servo_cmd(fr.tcp, hover, 0.5, rot_on=False)
        else:
            cmd, _ = _servo_cmd(fr.tcp, low, 0.7, rot_on=False)
        if c >= DESC_COUNT - 1:
            trace.append([c, round(float(fr.tcp[2]), 4)])
        _step(robot, cmd, -1.0, state)
        n += 1
    # no success: release and retreat (diagnostic; expert never needed this)
    for _ in range(5):
        _step(robot, [0, 0, 0, 0, 0, 0], 1.0, state)
    _go(robot, hover, 1.0, state, tol=0.01, max_steps=20, vmax=0.5, rot_on=False)
    return {"next_stage": "done", "result": "no success by give-up count", "trace": trace[-40:], "step": int(robot.now().step),
            "green_step": state["green_step"]}
