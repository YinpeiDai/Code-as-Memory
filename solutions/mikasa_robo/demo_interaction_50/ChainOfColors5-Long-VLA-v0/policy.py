"""ChainOfColors5-Long policy: watch the 5-cube colour cue, wait for the 9-cube ring,
touch the cued cubes in order (hover-dip above each cube top), then press the centre
button with a closed gripper, lift and hold still."""

HUE_NAMES = ["red", "green", "blue", "yellow", "orange", "magenta", "cyan"]
HUE_VECS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 169.0 / 231.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
DARK_NAMES = {"red": "darkred", "cyan": "teal"}
CUE_ROWS = (132, 176)
CUE_COLS = (104, 152)
# mean offset (expert TCP at touch) - (front top-face centroid) over 250 demo touches
CUBE_XY_CORR = (-0.006, 0.002)
BUTTON_XY_CORR = (0.0218, 0.0049)  # expert press - pink cap centroid, 50 demos
Z_TOUCH = 0.052
Z_TRAVEL = 0.115
Z_CLEAR = 0.058
Z_APPROACH = 0.08
DROP_RADIUS = 0.02  # final descent starts within 2 cm; a touch still requires < 1.2 cm and z < z_touch + 4 mm
Z_PRESS = 0.046
Z_LIFT = 0.17
PRESS_STALL = 2  # stalled steps on the button before lifting; R010: success needs the release (~7 steps after lift)
KP = 1.3
KD = 1.0


def _pure_mask(im):
    """Pixels of saturated pure cube colours (top-face or shaded)."""
    a = im.astype(np.int32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return (mn < 40) & (mx - mn > 90) & (mx > 90)


def _top_face_labels(im, tol=6.0, hue_tol=0.12):
    """Per-pixel cube-colour label index (0..8) for flat top-face pixels, else -1.
    Top faces render with max channel 231 (bright palette) or 169 (dark palette)."""
    a = im.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    bright = np.abs(mx - 231.0) <= tol
    dark = np.abs(mx - 169.0) <= tol
    m = (mn < 40.0) & (bright | dark)
    nrm = a / np.maximum(mx, 1.0)[..., None]
    H = np.array(HUE_VECS, dtype=np.float32)
    d = np.abs(nrm[..., None, :] - H[None, None, :, :]).sum(axis=3)
    hi = d.argmin(axis=2)
    m = m & (d.min(axis=2) < hue_tol)
    lab = np.where(m, hi + np.where(dark, 7, 0), -1)
    return lab


def _label_name(k):
    if k < 7:
        return HUE_NAMES[k]
    base = HUE_NAMES[k - 7]
    return DARK_NAMES.get(base, base + "_dark")


def _backproject(K, E, dep, ys, xs):
    z = dep[ys, xs].astype(np.float64) / 1000.0
    xc = (xs - K[0, 2]) * z / K[0, 0]
    yc = (ys - K[1, 2]) * z / K[1, 1]
    Pc = np.stack([xc, yc, z], axis=1)
    R = np.array(E[:, :3], dtype=np.float64)
    t = np.array(E[:, 3], dtype=np.float64)
    return (Pc - t) @ R


def _ring_present(frame):
    m = _pure_mask(frame.front_rgb)
    total = int(m.sum())
    inside = int(m[CUE_ROWS[0]:CUE_ROWS[1], CUE_COLS[0]:CUE_COLS[1]].sum())
    return (total - inside) > 150


def _cue_color(frame):
    crop = frame.front_rgb[CUE_ROWS[0]:CUE_ROWS[1], CUE_COLS[0]:CUE_COLS[1]]
    lab = _top_face_labels(crop)
    vals = lab[lab >= 0]
    if len(vals) < 15:
        return None
    counts = np.bincount(vals, minlength=14)
    k = int(counts.argmax())
    if counts[k] < 0.8 * len(vals):
        return None
    return _label_name(k)


def _detect_cubes(frame, tol=6.0, hue_tol=0.12, min_px=4):
    """World xy of every visible cube top face, keyed by colour name (largest cluster)."""
    lab = _top_face_labels(frame.front_rgb, tol, hue_tol)
    dep = frame.front_depth
    ok = (lab >= 0) & (dep > 0)
    ys, xs = np.nonzero(ok)
    out = {}
    if len(ys) == 0:
        return out
    P = _backproject(frame.front_K, frame.front_extrinsic, dep, ys, xs)
    labs = lab[ys, xs]
    for k in range(14):
        sel = labs == k
        if int(sel.sum()) < min_px:
            continue
        pts = P[sel]
        best = None
        rem = pts
        while len(rem) > 0:
            seed = rem[0]
            dd = np.hypot(rem[:, 0] - seed[0], rem[:, 1] - seed[1])
            c = np.median(rem[dd < 0.04], axis=0)
            dd = np.hypot(rem[:, 0] - c[0], rem[:, 1] - c[1])
            g = rem[dd < 0.04]
            if len(g) == 0:
                break
            c = np.median(g, axis=0)
            if best is None or len(g) > best[1]:
                best = (c, len(g))
            rem = rem[dd >= 0.04]
        if best is not None and best[1] >= min_px:
            c = best[0]
            out[_label_name(k)] = [float(c[0]) + CUBE_XY_CORR[0], float(c[1]) + CUBE_XY_CORR[1],
                                   float(c[2]), int(best[1])]
    return out


def _detect_button(frame):
    """Button xy from the pink cap top-face pixels (211,136,136) back-projected with front depth,
    plus the demo-calibrated offset to the expert press point."""
    a = frame.front_rgb.astype(np.int32)
    cap = (np.abs(a[..., 0] - 211) <= 3) & (np.abs(a[..., 1] - 136) <= 3) & (np.abs(a[..., 2] - 136) <= 3)
    dep = frame.front_depth
    ys, xs = np.nonzero(cap & (dep > 0))
    if len(ys) < 15:
        return None
    P = _backproject(frame.front_K, frame.front_extrinsic, dep, ys, xs)
    c = np.median(P, axis=0)
    dd = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    P = P[dd < 0.04]
    if len(P) < 15:
        return None
    c = P.mean(axis=0)
    return [float(c[0]) + BUTTON_XY_CORR[0], float(c[1]) + BUTTON_XY_CORR[1]]


def _parse_targets_from_goal(goal):
    g = (goal or "").lower()
    return {"touch": "touch" in g, "button": "button" in g}


def perceive_decide(obs, goal, memory, last):
    """Track cue colours and ring appearance from all frames, then choose the next stage action."""
    mem = dict(memory)
    if mem.get("stage") is None:
        mem = {"stage": "cue", "cue_colors": [], "cue_steps": [], "cue_run": "", "cue_run_len": 0,
               "ring_step": -1, "targets": [], "target_positions": [], "button": [],
               "next_index": 0, "touch_log": [], "press_count": 0, "notes": ""}
    stage = mem["stage"]
    if stage == "cue" and not mem["button"]:
        b0 = _detect_button(obs.frames[0])
        if b0 is not None:
            mem["button"] = [round(b0[0], 4), round(b0[1], 4)]
    if stage == "cue":
        colors = list(mem["cue_colors"])
        steps = list(mem["cue_steps"])
        run = mem["cue_run"]
        run_len = int(mem["cue_run_len"])
        ring_frame = None
        for fr in obs.frames:
            if _ring_present(fr):
                ring_frame = fr
                break
            c = _cue_color(fr)
            c = c if c is not None else ""
            if c == run:
                run_len += 1
            else:
                run = c
                run_len = 1
            if run != "" and run_len == 2 and (len(colors) == 0 or colors[-1] != run):
                colors.append(run)
                steps.append(int(fr.step) - 1)
        mem["cue_colors"] = colors
        mem["cue_steps"] = steps
        mem["cue_run"] = run
        mem["cue_run_len"] = run_len
        if ring_frame is not None:
            cubes = _detect_cubes(ring_frame)
            if any(c not in cubes for c in colors[:5]):
                loose = _detect_cubes(ring_frame, 10.0, 0.2, 2)
                for c in colors[:5]:
                    if c not in cubes and c in loose:
                        cubes[c] = loose[c]
            btn = mem["button"] if mem["button"] else _detect_button(ring_frame)
            if btn is None:
                btn = [-0.107, 0.0]
            targets = colors[:5] if len(colors) >= 5 else colors
            pos = []
            missing = []
            for c in targets:
                if c in cubes:
                    pos.append([round(cubes[c][0], 4), round(cubes[c][1], 4)])
                else:
                    missing.append(c)
                    pos.append(None)
            mem["ring_step"] = int(ring_frame.step)
            mem["targets"] = targets
            mem["target_positions"] = [p for p in pos if p is not None]
            mem["targets"] = [c for c, p in zip(targets, pos) if p is not None]
            mem["button"] = [round(btn[0], 4), round(btn[1], 4)]
            mem["notes"] = "ring@%d cubes=%s missing=%s" % (int(ring_frame.step), ",".join(sorted(cubes.keys())), ",".join(missing))
            mem["stage"] = "touch"
            stage = "touch"
        else:
            return {"memory": mem, "act": {"tool": "act_wait_for_ring", "args": {"max_steps": 150}}}
    if stage == "touch":
        i = int(mem["next_index"])
        if i < len(mem["targets"]):
            x, y = mem["target_positions"][i]
            mem["next_index"] = i + 1
            mem["touch_log"] = list(mem["touch_log"]) + [mem["targets"][i]]
            return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": x, "y": y, "z_touch": Z_TOUCH}}}
        mem["stage"] = "button"
        stage = "button"
    if stage == "button":
        bx, by = mem["button"]
        mem["press_count"] = int(mem["press_count"]) + 1
        if int(mem["press_count"]) >= 3:
            mem["stage"] = "hold"
        z = Z_PRESS if int(mem["press_count"]) == 1 else Z_PRESS - 0.006
        return {"memory": mem, "act": {"tool": "act_press_button", "args": {"x": bx, "y": by, "z_press": z, "hold_after": 25, "stall_steps": PRESS_STALL}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 200}}}


def _tcp(robot):
    return np.array(robot.now().tcp, dtype=np.float64)


def _servo(robot, target, prev, grip, kp=KP, kd=KD, xy_on=True):
    """One damped P step toward target xyz; returns (new tcp, velocity)."""
    p = _tcp(robot)
    v = p[:3] - prev[:3] if prev is not None else np.zeros(3)
    e = np.array(target, dtype=np.float64) - p[:3]
    u = (kp * e - kd * v) / 0.1
    if not xy_on:
        u[0] = 0.0
        u[1] = 0.0
    u = np.clip(u, -1.0, 1.0)
    robot.step(float(u[0]), float(u[1]), float(u[2]), 0.0, 0.0, 0.0, grip)
    return p


def act_wait_for_ring(robot, max_steps=1000):
    """Send zero actions (gripper open) until the cube ring appears in the front camera; returns step count."""
    n = 0
    seen = False
    while n < max_steps:
        if _ring_present(robot.now()):
            seen = True
            break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    return {"ring_seen": seen, "steps": n, "step": int(robot.now().step)}


def act_touch(robot, x, y, z_touch=0.052, z_travel=0.115, dwell=1, max_steps=90):
    """Hop to the cube at world (x, y) with the gripper open: no sideways motion below Z_CLEAR,
    travel at z_travel, pre-descend to Z_APPROACH within 6 cm, drop to z_touch within 1.2 cm
    (aiming 6 mm low until within 4 mm), dwell a few steps; returns measurements."""
    n = 0
    prev = None
    reached = False
    reached_step = None
    minz = 9.0
    xy_err_at_min = None
    dwell_left = dwell
    start = _tcp(robot)
    reach = None
    try:
        chk = robot.dry_run(float(x), float(y), float(z_touch), float(start[3]), float(start[4]), float(start[5]))
        reach = {"reachable": chk.get("reachable"), "margin": chk.get("joint_margin")}
    except Exception as exc:
        reach = {"error": str(exc)[:80]}
    while n < max_steps:
        p = _tcp(robot)
        dxy = math.hypot(x - p[0], y - p[1])
        if dxy < 0.03 and p[2] < minz:
            minz = float(p[2])
            xy_err_at_min = dxy
        if (not reached) and dxy < 0.012 and p[2] < z_touch + 0.004:
            reached = True
            reached_step = int(robot.now().step)
        if reached:
            if dwell_left <= 0:
                break
            dwell_left -= 1
            tz = z_touch
        elif dxy < DROP_RADIUS:
            tz = z_touch - 0.006
        elif dxy < 0.06 and p[2] > Z_APPROACH - 0.01:
            tz = Z_APPROACH
        else:
            tz = max(z_travel, Z_APPROACH) if dxy >= 0.06 else max(p[2], Z_APPROACH)
        xy_on = (p[2] > Z_CLEAR) or dxy < 0.025
        prev = _servo(robot, [x, y, tz], prev, 1.0, xy_on=xy_on)
        n += 1
    p = _tcp(robot)
    return {"reached": reached, "reached_step": reached_step, "steps": n, "min_z": round(minz, 4),
            "xy_err_at_min": None if xy_err_at_min is None else round(float(xy_err_at_min), 4),
            "end_tcp": [round(float(v), 4) for v in p], "start_tcp": [round(float(v), 4) for v in start],
            "reach": reach, "step": int(robot.now().step)}


def act_press_button(robot, x, y, z_press=0.046, hold_after=25, stall_steps=6, max_steps=140):
    """Close the gripper, move above the button at (x, y), press down until the TCP stalls,
    hold, lift to Z_LIFT and hold still; reports pressed depth and stall timing."""
    n = 0
    prev = None
    phase = "approach"
    minz = 9.0
    stall = 0
    press_steps = 0
    lift_steps = 0
    hold = 0
    events = []
    while n < max_steps:
        p = _tcp(robot)
        dxy = math.hypot(x - p[0], y - p[1])
        minz = min(minz, float(p[2]))
        if phase == "approach":
            if dxy <= 0.012:
                tz = z_press
            elif dxy < 0.06 and p[2] > Z_APPROACH - 0.01:
                tz = Z_APPROACH + 0.005
            else:
                tz = Z_TRAVEL
            xy_on = p[2] > Z_CLEAR or dxy < 0.02
            if dxy < 0.012:
                phase = "press"
                events.append(["press_start", int(robot.now().step), round(float(p[2]), 4)])
            prev = _servo(robot, [x, y, tz], prev, -1.0, xy_on=xy_on)
        elif phase == "press":
            vz = 0.0 if prev is None else float(p[2] - prev[2])
            if p[2] < 0.062 and abs(vz) < 0.0015:
                stall += 1
            press_steps += 1
            if stall >= stall_steps or press_steps > stall_steps + 24:
                phase = "lift"
                events.append(["lift_start", int(robot.now().step), round(float(p[2]), 4)])
            prev = _servo(robot, [x, y, z_press], prev, -1.0)
        elif phase == "lift":
            lift_steps += 1
            if p[2] > Z_LIFT - 0.008 or lift_steps > 25:
                phase = "hold"
                events.append(["hold_start", int(robot.now().step), round(float(p[2]), 4)])
            prev = _servo(robot, [x, y, Z_LIFT], prev, -1.0)
        else:
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
            hold += 1
            if hold >= hold_after:
                break
        n += 1
    p = _tcp(robot)
    return {"phase": phase, "steps": n, "min_z": round(minz, 4), "events": events,
            "end_tcp": [round(float(v), 4) for v in p], "step": int(robot.now().step)}


def act_hold(robot, steps=200):
    """Hold still with zero commands (gripper closed) for a number of steps."""
    n = 0
    while n < steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
    return {"steps": n, "step": int(robot.now().step)}
