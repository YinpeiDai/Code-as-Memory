"""FindImposterShapeAndColor3x2: remember the cue objects, then touch the test object whose
(color, shape) combination was absent from the cue.

Perception uses the fixed front camera (RGB + depth + calibration): saturated red/green/blue
blobs, back-projected to world XYZ; shape from the depth-normalised projected area
(sphere ~12-14 cm^2, cube ~20-30 cm^2 in all 50 demos)."""

COLORS = ("red", "green", "blue")
AREA_SPLIT = 17.5      # cm^2, sphere < split < cube (demo range: sphere<=14.1, cube>=20.6)
Z_TOUCH = 0.015        # TCP height for the touch (object centre z = 0.02, expert ended 0.01-0.04)
GAIN = 2.0             # position gain: command = (GAIN * err - DAMP * v) / 0.1 (clipped)
DAMP = 1.0             # velocity damping on measured per-step TCP motion (R15/R16 overshoot 1-2 cm)


def _color_masks(img):
    im = img.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (r > 50) & (g * 10 < r * 4) & (b * 10 < r * 4)
    grn = (g > 50) & (r * 10 < g * 6) & (b * 10 < g * 6)
    blu = (b > 50) & (r * 10 < b * 6) & (g * 10 < b * 6)
    return {"red": red, "green": grn, "blue": blu}


def _components(mask, minpix=12):
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while pts:
        p = pts.pop()
        stack = [p]
        comp = [p]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.remove(q)
                        stack.append(q)
                        comp.append(q)
        if len(comp) >= minpix:
            comps.append(np.array(comp))
    return comps


def _detect(frame):
    """List of blobs {color, shape, area, x, y, zmax, n} from the front camera."""
    img = frame.front_rgb
    depth = frame.front_depth
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    R = E[:, :3]
    t = E[:, 3]
    out = []
    for c, m in _color_masks(img).items():
        for comp in _components(m):
            ys = comp[:, 0]
            xs = comp[:, 1]
            z = depth[ys, xs].astype(float) / 1000.0
            ok = z > 0
            if ok.sum() < 8:
                continue
            ys, xs, z = ys[ok], xs[ok], z[ok]
            px = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
            py = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
            pc = np.stack([px, py, z], 1)
            w = (pc - t) @ R
            zmax = float(np.percentile(w[:, 2], 95))
            top = w[w[:, 2] > zmax - 0.005]
            area = float(len(comp) * np.mean(z) ** 2 / (K[0, 0] * K[1, 1]) * 1e4)
            shape = "cube" if area > AREA_SPLIT else "sphere"
            out.append({"color": c, "shape": shape, "area": round(area, 2),
                        "x": float(top[:, 0].mean()), "y": float(top[:, 1].mean()),
                        "zmax": zmax, "n": int(len(comp))})
    return out


def _combos(blobs):
    return sorted(b["color"] + "_" + b["shape"] for b in blobs)


def _choose_target(cue, test):
    """cue/test: lists of blobs. Returns (blob, reason)."""
    cue_set = set(_combos(cue))
    new = [b for b in test if b["color"] + "_" + b["shape"] not in cue_set]
    counts = {c: sum(1 for b in cue if b["color"] == c) for c in COLORS}
    single = [c for c in COLORS if counts[c] == min(counts.values())]
    if len(new) == 1:
        return new[0], "unique new combo"
    if len(new) > 1:
        pref = [b for b in new if b["color"] in single]
        if len(pref) >= 1:
            return pref[0], "several new combos; chose cue-singleton color"
        return new[0], "several new combos; first"
    # no new combo: area-based fallback within the cue-singleton colour
    cands = [b for b in test if b["color"] in single]
    if cands:
        ref = [b for b in cue if b["color"] in single]
        if ref:
            ra = ref[0]["area"]
            cands = sorted(cands, key=lambda b: -abs(b["area"] - ra))
        return cands[0], "no new combo; most different area within singleton color"
    return (test[0] if test else None), "fallback first blob"


def perceive_decide(obs, goal, memory, last):
    """Track cue -> delay -> test phases from front-camera blobs, then touch the imposter."""
    mem = dict(memory)
    phase = mem.get("phase") or "start"
    cue = mem.get("cue_blobs") or []
    cue = [dict(zip(("color", "shape", "area", "x", "y"), v)) for v in cue]
    frames_seen = int(mem.get("frames_seen") or 0)
    log = mem.get("log") or ""
    target = mem.get("target")
    for f in obs.frames:
        frames_seen += 1
        blobs = _detect(f)
        n = len(blobs)
        if phase in ("start", "cue"):
            if n >= 3:
                if phase == "start":
                    phase = "cue"
                    cue = blobs
                    mem["cue_start_step"] = int(f.step)
                elif n >= 6 and n > len(cue) and set(_combos(blobs)) != set(_combos(cue)):
                    phase = "test"
                elif n > len(cue):
                    cue = blobs
            elif n == 0 and phase == "cue":
                phase = "delay"
                mem["delay_start_step"] = int(f.step)
        elif phase == "delay":
            if n >= 4:
                phase = "test"
        if phase == "test" and target is None:
            tgt, why = _choose_target(cue, blobs)
            mem["test_start_step"] = int(f.step)
            mem["test_combos"] = _combos(blobs)
            if tgt is not None:
                target = [tgt["x"], tgt["y"], 0.02]
                mem["imposter"] = tgt["color"] + "_" + tgt["shape"]
                log += "target %s at (%.3f,%.3f) step %d: %s; " % (mem["imposter"], tgt["x"], tgt["y"], f.step, why)
    mem["phase"] = phase
    mem["cue_blobs"] = [[b["color"], b["shape"], b["area"], round(b["x"], 4), round(b["y"], 4)] for b in cue]
    mem["cue_combos"] = _combos(cue)
    mem["frames_seen"] = frames_seen
    mem["target"] = target
    tries = int(mem.get("touch_tries") or 0)
    if target is not None and phase in ("test", "touch"):
        mem["phase"] = "touch"
        mem["touch_tries"] = tries + 1
        if last:
            log += "last: " + str(last)[:300] + "; "
        mem["log"] = log[-2500:]
        variant = ("open_then_squeeze", "lower_squeeze", "press_top")[min(tries, 2)]
        out = {k: mem.get(k) for k in _FIELDS}
        return {"memory": out, "act": {"tool": "act_touch",
                                        "args": {"x": target[0], "y": target[1], "variant": variant}}}
    mem["log"] = log[-2500:]
    out = {k: mem.get(k) for k in _FIELDS}
    return {"memory": out, "act": {"tool": "act_wait", "args": {}}}


_FIELDS = ("phase", "cue_blobs", "cue_combos", "cue_start_step", "delay_start_step",
           "test_start_step", "test_combos", "imposter", "target", "frames_seen",
           "touch_tries", "log")


def act_wait(robot):
    """One native step with no arm motion and an open-gripper command (cue/delay phases)."""
    before = robot.now()
    f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"step": int(f.step), "fingers": [round(float(v), 4) for v in f.fingers],
            "fingers_before": [round(float(v), 4) for v in before.fingers],
            "tcp_z": round(float(f.tcp[2]), 4)}


def _cmd(err, gain, vel):
    return [float(np.clip((gain * e - DAMP * v) / 0.1, -1.0, 1.0)) for e, v in zip(err, vel)]


def act_touch(robot, x, y, variant="open_then_squeeze"):
    """Move the TCP over (x, y) and down to touch height with the gripper open, hold, then close
    the gripper on the object. variant: open_then_squeeze | lower_squeeze | press_top.
    Returns the measured TCP log and stage reached (the episode ends on native success)."""
    if variant == "lower_squeeze":
        z_goal, grip_app = 0.008, 1.0
    elif variant == "press_top":
        z_goal, grip_app = 0.03, -1.0
    else:
        z_goal, grip_app = Z_TOUCH, 1.0
    goal = np.array([x, y, z_goal])
    log = []
    stage = "approach"
    arrived_at = None
    p_prev = None
    for i in range(24):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        vel = np.zeros(3) if p_prev is None else p - p_prev
        p_prev = p
        dxy = float(np.linalg.norm(goal[:2] - p[:2]))
        zf = z_goal + 0.5 * max(0.0, dxy - 0.015)
        g = goal.copy()
        g[2] = max(z_goal, zf)
        err = g - p
        if stage == "approach" and dxy < 0.008 and abs(p[2] - z_goal) < 0.008:
            stage = "hold"
            arrived_at = i
        grip = grip_app
        gain = GAIN
        if stage == "hold":
            gain = 1.0
            if i - arrived_at >= 2:
                stage = "squeeze"
        if stage == "squeeze":
            gain = 1.0
            grip = -1.0
            if i - arrived_at >= 8:
                break
        a = _cmd(err, gain, vel)
        log.append([int(f.step), stage] + [round(float(v), 4) for v in p] + [round(float(f.fingers[0]), 4)])
        robot.step(a[0], a[1], a[2], 0.0, 0.0, 0.0, grip)
    f = robot.now()
    return {"variant": variant, "stage": stage, "arrived_iter": arrived_at,
            "final_tcp": [round(float(v), 4) for v in f.tcp], "fingers": [round(float(v), 4) for v in f.fingers],
            "log": log[-10:]}
