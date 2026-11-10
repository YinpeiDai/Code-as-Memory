"""ChainOfColors7 policy: read 7-cube colour cue, wait for the ring of 9 cubes, touch the cued
colours in order with the open gripper, then press the centre button with the closed gripper."""

Z_TOUCH = 0.049      # TCP height for a cube touch (cube top ~0.042; expert minima 0.044-0.062)
Z_TRAVEL = 0.11      # transit height between cubes
Z_LIFT_FREE = 0.085  # below this height xy motion is throttled (avoid dragging fingers over cubes)
Z_PRESS = 0.040      # commanded press depth (button cap top 0.060, stalls near TCP 0.053)
Z_PRESS_LIFT = 0.16  # lift after press (expert lifts to ~0.16 before success)
CUBE_X_BIAS = -0.003 # front-cam centroid x correction: demos suggest -0.008, wrist-cam R044/R045 suggests ~0
BTN_X_OFFSET = 0.010 # press slightly toward +x of cap centre (expert pressed +0.017..0.02)
KP = 1.0
KD = 0.3
TOUCH_DWELL = 2      # extra servo steps held at the bottom of each touch
RESEQUENCE_AFTER = 2 # failed presses before the whole touch chain is repeated once


def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logmap(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return v * (th / (2.0 * s))


def _rot_cmd(tcp, rpy_target, gain=0.8, lim=0.4):
    """Rotation command components so the measured orientation moves toward rpy_target.
    Native semantics: angular offset = -0.1 * r (base frame, left-multiplied)."""
    Rc = _rpy_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_R(rpy_target[0], rpy_target[1], rpy_target[2])
    w = _logmap(Rt @ Rc.T)
    r = -gain * w / 0.1
    return np.clip(r, -lim, lim), float(np.linalg.norm(w))


def _servo(robot, target, grip, rpy, prev_pos, xy_throttle=True, vmax=1.0):
    """One closed-loop step toward an xyz target. Returns (frame, pos)."""
    fr = robot.now()
    tcp = np.array(fr.tcp, dtype=np.float64)
    pos = tcp[:3]
    vel = pos - prev_pos if prev_pos is not None else np.zeros(3)
    e = np.array(target, dtype=np.float64) - pos
    cmd = (KP * e - KD * vel) / 0.1
    if xy_throttle and pos[2] < Z_LIFT_FREE and np.linalg.norm(e[:2]) > 0.02:
        # lift vertically off a cube first (only damp residual xy motion)
        cmd[0] = -KD * vel[0] / 0.1
        cmd[1] = -KD * vel[1] / 0.1
    cmd = np.clip(cmd, -vmax, vmax)
    r, _ = _rot_cmd(tcp, rpy)
    robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return robot.now(), pos


def _fields():
    return ["stage", "cue_runs", "cue_seq", "last_step", "rpy0", "targets", "target_labels",
            "button", "touched", "presses", "ring_step", "notes", "touch_retries", "resequenced"]


def _cue_runs_update(runs, frames, last_step):
    """Append per-frame cue labels (run-length 'label*count') for frames newer than last_step."""
    runs = list(runs or [])
    for fr in frames:
        if last_step is not None and fr.step <= last_step:
            continue
        if _ring_visible(fr.wrist_rgb)[0]:
            continue
        lab = _cue_label(fr.wrist_rgb)[0]
        lab = lab if lab is not None else "none"
        if runs:
            name, n = runs[-1].split("*")
            if name == lab:
                runs[-1] = name + "*" + str(int(n) + 1)
                continue
        runs.append(lab + "*1")
        last_step = fr.step
    return runs


def _cue_sequence(runs):
    seq = []
    for r in runs:
        name, n = r.split("*")
        if name == "none" or int(n) < 3:
            continue
        if seq and seq[-1] == name:
            continue
        seq.append(name)
    return seq


def perceive_decide(obs, goal, memory, last):
    """Track the cue, localise the ring, then issue one touch per cued cube and finally the button press."""
    mem = {k: memory.get(k) for k in _fields()}
    cur = obs.current
    if mem["stage"] is None:
        mem.update(stage="cue", cue_runs=[], cue_seq=[], last_step=-1,
                   rpy0=[float(cur.tcp[3]), float(cur.tcp[4]), float(cur.tcp[5])],
                   targets=[], target_labels=[], button=None, touched=0, presses=0,
                   ring_step=None, notes="", touch_retries=0, resequenced=False)
    if mem["stage"] == "cue":
        mem["cue_runs"] = _cue_runs_update(mem["cue_runs"], obs.frames, mem["last_step"])
        mem["last_step"] = int(max(fr.step for fr in obs.frames))
        mem["cue_seq"] = _cue_sequence(mem["cue_runs"])
        ring_on = _ring_visible(cur.wrist_rgb)[0]
        if not ring_on:
            return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 150}}}
        ring = _locate_ring(cur)
        btn = _locate_button(cur)
        seq = mem["cue_seq"]
        tg, tl, miss = [], [], []
        for lab in seq:
            if lab in ring:
                c = ring[lab]
                tg.append([c[0] + CUBE_X_BIAS, c[1], c[2]])
                tl.append(lab)
            else:
                miss.append(lab)
        mem["targets"] = tg
        mem["target_labels"] = tl
        if btn is not None:
            mem["button"] = [btn[0] + BTN_X_OFFSET, btn[1], btn[2]]
        else:
            mem["button"] = [-0.12, 0.0, 0.06]
        mem["ring_step"] = int(cur.step)
        mem["notes"] = ("cue=%s ring=%s missing=%s btn=%s" % (",".join(seq), ",".join(sorted(ring.keys())),
                        ",".join(miss), "found" if btn is not None else "default"))[:600]
        mem["stage"] = "touch"
    if mem["stage"] == "touch":
        if (last is not None and str(last).startswith("act_touch") and '"reached": false' in str(last)
                and int(mem["touch_retries"] or 0) < 3 and int(mem["touched"]) > 0):
            mem["touched"] = int(mem["touched"]) - 1
            mem["touch_retries"] = int(mem["touch_retries"] or 0) + 1
        i = int(mem["touched"])
        if i < len(mem["targets"]):
            t = mem["targets"][i]
            mem["touched"] = i + 1
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": float(t[0]), "y": float(t[1]), "label": mem["target_labels"][i], "index": i,
                "rpy": [float(v) for v in mem["rpy0"]]}}}
        mem["stage"] = "press"
    if int(mem["presses"]) >= RESEQUENCE_AFTER and not mem["resequenced"]:
        # presses are not accepted: assume a touch was not registered and repeat the whole chain once
        mem["resequenced"] = True
        mem["touched"] = 1
        mem["stage"] = "touch"
        t = mem["targets"][0]
        return {"memory": mem, "act": {"tool": "act_touch", "args": {
            "x": float(t[0]), "y": float(t[1]), "label": mem["target_labels"][0], "index": 0,
            "rpy": [float(v) for v in mem["rpy0"]]}}}
    b = mem["button"]
    mem["presses"] = int(mem["presses"]) + 1
    return {"memory": mem, "act": {"tool": "act_press", "args": {
        "x": float(b[0]), "y": float(b[1]), "attempt": int(mem["presses"]),
        "rpy": [float(v) for v in mem["rpy0"]]}}}


def act_watch(robot, max_steps=150):
    """Hold still (gripper open) while the cue plays, until the cube ring appears in the wrist camera
    or max_steps elapse. Returns steps used and whether the ring was seen."""
    n = 0
    seen = False
    for _ in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        if _ring_visible(robot.now().wrist_rgb)[0]:
            seen = True
            break
    fr = robot.now()
    return {"steps": n, "ring_seen": seen, "step": int(fr.step), "cue_now": _cue_label(fr.wrist_rgb)[0]}


def act_touch(robot, x, y, label="", index=0, rpy=None, max_steps=90):
    """Lift to transit height, move above cube (x, y), descend to Z_TOUCH with the gripper open.
    Returns measured touch height, xy error and steps."""
    fr0 = robot.now()
    if rpy is None:
        rpy = [float(fr0.tcp[3]), float(fr0.tcp[4]), float(fr0.tcp[5])]
    rpy = [float(v) for v in rpy]
    dr = robot.dry_run(float(x), float(y), Z_TOUCH, rpy[0], rpy[1], rpy[2], mode="absolute")
    prev = None
    n = 0
    zmin = 9.0
    exy = 9.0
    reached = False
    descending = False
    flips = 0
    for _ in range(int(max_steps)):
        pos = np.array(robot.now().tcp[:3], dtype=np.float64)
        dxy = math.hypot(x - pos[0], y - pos[1])
        if not descending and dxy < 0.015:
            descending = True
        elif descending and dxy > 0.04:
            descending = False
            flips += 1
        if descending:
            tgt = [x, y, Z_TOUCH - 0.004]
        else:
            tgt = [x, y, Z_TRAVEL]
        fr, prev = _servo(robot, tgt, 1.0, rpy, prev, xy_throttle=not descending)
        n += 1
        p2 = np.array(fr.tcp[:3], dtype=np.float64)
        d2 = math.hypot(x - p2[0], y - p2[1])
        if p2[2] < zmin:
            zmin = float(p2[2])
            exy = float(d2)
        if p2[2] <= Z_TOUCH + 0.003 and d2 < 0.02:
            reached = True
            break
    if reached:
        for _ in range(TOUCH_DWELL):
            fr, prev = _servo(robot, [x, y, Z_TOUCH - 0.004], 1.0, rpy, prev, xy_throttle=False)
            n += 1
            zmin = min(zmin, float(fr.tcp[2]))
    fr = robot.now()
    return {"label": label, "index": index, "steps": n, "reached": reached, "zmin": zmin, "xy_err": exy,
            "flips": flips,
            "end_step": int(fr.step), "dry_ok": bool(dr.get("within_joint_limits", True)),
            "dry_margin": dr.get("min_joint_margin_rad")}


def act_press(robot, x, y, attempt=1, rpy=None, max_steps=120):
    """Close the gripper, move above the button, press down until the cap stalls, hold, lift to
    Z_PRESS_LIFT, then command the gripper open (expert pattern). Returns measured press data."""
    fr0 = robot.now()
    if rpy is None:
        rpy = [float(fr0.tcp[3]), float(fr0.tcp[4]), float(fr0.tcp[5])]
    rpy = [float(v) for v in rpy]
    prev = None
    n = 0
    phase = "lift" if float(fr0.tcp[2]) < Z_LIFT_FREE else "approach"
    zmin = 9.0
    hold = 0
    stall = 0
    zprev = None
    for _ in range(int(max_steps)):
        pos = np.array(robot.now().tcp[:3], dtype=np.float64)
        dxy = math.hypot(x - pos[0], y - pos[1])
        grip = -1.0
        if phase == "lift":
            # rise straight up with the gripper still open so the fingers do not catch the last cube
            tgt = [float(pos[0]), float(pos[1]), Z_TRAVEL]
            grip = 1.0
            if pos[2] >= Z_LIFT_FREE:
                phase = "approach"
        if phase == "approach":
            tgt = [x, y, Z_TRAVEL]
            if dxy < 0.01 and min(robot.now().fingers) < 0.01:
                phase = "down"
        if phase == "down":
            tgt = [x, y, Z_PRESS]
            if pos[2] < 0.057:
                if zprev is not None and abs(pos[2] - zprev) < 0.0015:
                    stall += 1
                hold += 1
                if hold >= 4:
                    phase = "up"
        if phase == "up":
            tgt = [x, y, Z_PRESS_LIFT + 0.01]
            if pos[2] >= Z_PRESS_LIFT - 0.005:
                break
        zprev = pos[2]
        fr, prev = _servo(robot, tgt, grip, rpy, prev, xy_throttle=False)
        n += 1
        zmin = min(zmin, float(fr.tcp[2]))
    opened = 0
    for _ in range(3):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        opened += 1
    fr = robot.now()
    return {"attempt": attempt, "steps": n, "phase": phase, "zmin": zmin, "stall_steps": stall,
            "open_steps": opened, "end_step": int(fr.step), "z_end": float(fr.tcp[2])}
