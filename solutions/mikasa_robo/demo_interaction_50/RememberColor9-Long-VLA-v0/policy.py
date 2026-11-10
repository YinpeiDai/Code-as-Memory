"""RememberColor9-Long: remember the cue cube's colour, wait for the 9 candidate cubes, touch the matching one.

Perception uses only RGB-D + calibration of the current frames.  Colour key = hue class
(red, orange, yellow, green, cyan, blue, magenta) plus a brightness bit for red and cyan,
because the 9 colours include red/dark-red and cyan/dark-cyan (teal).
"""

HUES = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]
HUE_PROTO = [[1.0, 0.0, 0.0], [1.0, 0.735, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
TWO_SHADE = ["red", "cyan"]
BRIGHT_THR = 200.0        # top-face max channel: bright ~231, dark ~169 (demos)
CUBE_TOP_Z = 0.04         # cube top height in world (demo depth: 0.040-0.041)
HOVER_Z = 0.13
TOUCH_ZS = [0.05, 0.042, 0.035, 0.03]
MAX_NATIVE = 600


def _cube_mask(rgb):
    rgb = rgb.astype(np.float32)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    return (mn < 25) & (mx > 60), mx


def _hue_classes(rgb, mask, mx):
    nrm = rgb.astype(np.float32) / np.maximum(mx, 1.0)[..., None]
    P = np.array(HUE_PROTO, dtype=np.float32)
    d = ((nrm[..., None, :] - P) ** 2).sum(axis=-1)
    cls = d.argmin(axis=-1)
    cls[~mask] = -1
    return cls


def _deproject(frame, cam, ys, xs):
    K = np.array(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.array(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    D = np.array(getattr(frame, cam + "_depth"), dtype=np.float64) / 1000.0
    z = D[ys, xs]
    u = xs + 0.5
    v = ys + 0.5
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    R = E[:, :3]
    t = E[:, 3]
    W = (pc - t) @ R
    return W, z


def _key(hue, bright):
    if hue in TWO_SHADE:
        return hue + ("_bright" if bright else "_dark")
    return hue


def _scene_pixels(frame, cam="front"):
    """Cube-coloured pixels with world coordinates: returns (hue_idx, brightness, W) arrays."""
    rgb = np.array(getattr(frame, cam + "_rgb"))
    mask, mx = _cube_mask(rgb)
    cls = _hue_classes(rgb, mask, mx)
    ys, xs = np.nonzero(cls >= 0)
    if len(ys) == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros((0, 3))
    W, z = _deproject(frame, cam, ys, xs)
    ok = (z > 0.01) & (W[:, 2] > -0.01) & (W[:, 2] < 0.08) & (np.abs(W[:, 0]) < 0.6) & (np.abs(W[:, 1]) < 0.6)
    return cls[ys, xs][ok], mx[ys, xs][ok], W[ok]


def _hue_count(frame):
    """Number of hue classes with >= 20 cube-like pixels in the front image (cheap appearance test)."""
    rgb = np.array(frame.front_rgb)
    mask, mx = _cube_mask(rgb)
    if mask.sum() < 40:
        return 0
    cls = _hue_classes(rgb, mask, mx)
    n = 0
    for i in range(len(HUES)):
        if (cls == i).sum() >= 20:
            n += 1
    return n


def _read_cue(frame):
    """Colour key of the single cue cube (front camera, wrist as fallback)."""
    best = None
    for cam in ("front", "wrist"):
        rgb = np.array(getattr(frame, cam + "_rgb"))
        mask, mx = _cube_mask(rgb)
        if mask.sum() < 15:
            continue
        cls = _hue_classes(rgb, mask, mx)
        counts = [int((cls == i).sum()) for i in range(len(HUES))]
        hi = int(np.argmax(counts))
        vals = mx[cls == hi]
        med = float(np.median(vals))
        best = {"hue": HUES[hi], "bright": med >= BRIGHT_THR, "median_max": med,
                "pixels": counts[hi], "cam": cam, "purity": counts[hi] / max(1, sum(counts))}
        break
    return best


def _candidates(frame):
    """Per colour key: top-face centre (world xy) of the cube, from front RGB-D."""
    cls, mx, W = _scene_pixels(frame, "front")
    out = {}
    top = W[:, 2] > CUBE_TOP_Z - 0.006 if len(W) else np.zeros(0, bool)
    for i, h in enumerate(HUES):
        sel = (cls == i) & top
        if sel.sum() < 4:
            continue
        b = mx[sel] >= BRIGHT_THR
        groups = [(h, np.ones(int(sel.sum()), bool))] if h not in TWO_SHADE else [
            (h + "_bright", b), (h + "_dark", ~b)]
        Ws = W[sel]
        for k, g in groups:
            if g.sum() < 4:
                continue
            c = _robust_center(Ws[g][:, :2])
            if c is not None:
                out[k] = c
    return out


def _robust_center(xy, radius=0.022):
    if len(xy) == 0:
        return None
    c = np.median(xy, axis=0)
    for _ in range(3):
        d = np.sqrt(((xy - c) ** 2).sum(axis=1))
        s = xy[d < radius]
        if len(s) < 3:
            return None
        c = s.mean(axis=0)
    return [float(c[0]), float(c[1]), int(len(s))]


def _wrist_refine(frame, key, guess_xy, radius=0.018):
    """Refine the target's top-face centre with the wrist camera; selects hue (+shade) pixels near guess."""
    cls, mx, W = _scene_pixels(frame, "wrist")
    if len(W) == 0:
        return None
    hue = key.split("_")[0]
    hi = HUES.index(hue)
    sel = (cls == hi) & (W[:, 2] > CUBE_TOP_Z - 0.006)
    if hue in TWO_SHADE:
        want_bright = key.endswith("_bright")
        sel = sel & ((mx >= BRIGHT_THR) == want_bright)
    xy = W[sel][:, :2]
    if len(xy) < 20:
        return None
    g = np.array(guess_xy[:2])
    c = g
    for _ in range(4):
        d = np.sqrt(((xy - c) ** 2).sum(axis=1))
        s = xy[d < radius]
        if len(s) < 20:
            return None
        c = s.mean(axis=0)
    return [float(c[0]), float(c[1]), int(len(s))]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: read cue -> wait for cubes -> locate matching cube -> hover/refine -> descend to touch."""
    f = obs.current
    m = dict(memory) if memory else {}
    stage = m.get("stage") or "start"
    log = m.get("log") or ""
    dec = int(m.get("decisions") or 0) + 1
    m["decisions"] = dec
    for k, v in (("cue_key", ""), ("cue_info", ""), ("target_xy", []), ("touch_index", 0), ("cubes_step", -1),
                 ("candidates", "")):
        if k not in m or m[k] is None:
            m[k] = v
    act = {"tool": "act_wait", "args": {"steps": 1}}

    if stage == "start":
        cue = _read_cue(f)
        if cue is None:
            # cue not visible yet: observe one more step
            act = {"tool": "act_wait", "args": {"steps": 1}}
            log += "no cue at step %s; " % f.step
        else:
            m["cue_key"] = _key(cue["hue"], cue["bright"])
            m["cue_info"] = "step %s cam %s hue %s median %.0f px %d purity %.2f" % (
                f.step, cue["cam"], cue["hue"], cue["median_max"], cue["pixels"], cue["purity"])
            log += "cue=%s; " % m["cue_key"]
            stage = "waiting"
            act = {"tool": "act_wait_for_cubes", "args": {"max_steps": MAX_NATIVE}}
    elif stage == "waiting":
        if last and isinstance(last, str) and '"appeared": true' in last:
            pass
        n = _hue_count(f)
        if n >= 5:
            m["cubes_step"] = int(f.step)
            stage = "locate"
        else:
            act = {"tool": "act_wait_for_cubes", "args": {"max_steps": MAX_NATIVE}}
    if stage == "locate":
        cands = _candidates(f)
        m["candidates"] = ";".join("%s:%.3f,%.3f,%d" % (k, v[0], v[1], v[2]) for k, v in sorted(cands.items()))
        key = m["cue_key"]
        pick = key if key in cands else None
        if pick is None and m["decisions"] >= 6:
            base = key.split("_")[0]
            alts = [k for k in cands if k.split("_")[0] == base]
            if alts:
                pick = alts[0]
                log += "fallback %s->%s; " % (key, pick)
        if pick is not None:
            c = cands[pick]
            m["target_xy"] = [c[0], c[1]]
            log += "target %s at %.3f,%.3f (n=%d); " % (key, c[0], c[1], c[2])
            stage = "hover"
            act = {"tool": "act_move_to", "args": {"x": c[0], "y": c[1], "z": HOVER_Z, "tol": 0.004,
                                                   "max_steps": 60, "settle": 2, "refine_key": key}}
        else:
            log += "target %s not found among %s; " % (key, list(cands.keys()))
            act = {"tool": "act_wait", "args": {"steps": 1}}
    elif stage == "hover":
        # last act refined target in wrist cam; read from last result text
        txy = _parse_refined(last)
        if txy is not None:
            m["target_xy"] = txy
            log += "wrist-refined %.4f,%.4f; " % (txy[0], txy[1])
        stage = "touch"
        m["touch_index"] = 0
        t = m["target_xy"]
        act = {"tool": "act_move_to", "args": {"x": t[0], "y": t[1], "z": TOUCH_ZS[0], "tol": 0.003,
                                               "max_steps": 40, "settle": 8, "refine_key": m["cue_key"]}}
    elif stage == "touch":
        txy = _parse_refined(last)
        if txy is not None:
            m["target_xy"] = txy
        i = int(m["touch_index"]) + 1
        if i >= len(TOUCH_ZS):
            i = len(TOUCH_ZS) - 1
        m["touch_index"] = i
        t = m["target_xy"]
        log += "touch level %d z=%.3f; " % (i, TOUCH_ZS[i])
        act = {"tool": "act_move_to", "args": {"x": t[0], "y": t[1], "z": TOUCH_ZS[i], "tol": 0.003,
                                               "max_steps": 40, "settle": 8, "refine_key": m["cue_key"]}}
    m["stage"] = stage
    m["log"] = log[-1500:]
    out = {k: m[k] for k in ("stage", "cue_key", "cue_info", "target_xy", "touch_index", "cubes_step",
                             "candidates", "decisions", "log")}
    return {"memory": out, "act": act}


def _parse_refined(last):
    if not last or not isinstance(last, str):
        return None
    tag = '"refined_xy": ['
    i = last.find(tag)
    if i < 0:
        return None
    j = last.find("]", i)
    parts = last[i + len(tag):j].split(",")
    if len(parts) < 2:
        return None
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError:
        return None


def act_wait(robot, steps=1):
    """Hold still (zero arm command, gripper open) for `steps` native steps."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"waited": int(steps), "step": int(robot.now().step)}


def act_wait_for_cubes(robot, max_steps=600):
    """Hold still until >=5 hue classes of cubes are visible in the front camera, plus one confirmation frame."""
    n = 0
    seen = 0
    for k in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n = _hue_count(robot.now())
        if n >= 5:
            seen += 1
            if seen >= 2:
                return {"appeared": True, "hues": n, "step": int(robot.now().step), "waited": k + 1}
        else:
            seen = 0
    return {"appeared": False, "hues": n, "step": int(robot.now().step)}


def act_move_to(robot, x, y, z, tol=0.004, max_steps=60, settle=2, refine_key="", gain=1.0, vmax=1.0):
    """Closed-loop TCP move to world (x,y,z) with zero rotation command and open gripper.

    Each step commands the remaining error (offset = error, clipped to 0.1*vmax m).  Arrival = error < tol and
    per-step motion < 1 mm for `settle` consecutive steps.  If refine_key is given, after arrival the wrist camera
    re-estimates the target top-face centre (returned as refined_xy; not used for this move)."""
    goal = np.array([x, y, z], dtype=np.float64)
    prev = np.array(robot.now().tcp[:3], dtype=np.float64)
    calm = 0
    trace = []
    n = 0
    t0 = robot.now().tcp
    margin = None
    try:
        chk = robot.dry_run(float(x), float(y), float(z), float(t0[3]), float(t0[4]), float(t0[5]), mode="absolute")
        margin = chk.get("min_joint_margin_rad")
    except Exception as ex:
        margin = "dry_run failed: %s" % str(ex)[:80]
    for n in range(1, int(max_steps) + 1):
        p = np.array(robot.now().tcp[:3], dtype=np.float64)
        err = goal - p
        cmd = np.clip(err / 0.1 * gain, -vmax, vmax)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, 1.0)
        q = np.array(robot.now().tcp[:3], dtype=np.float64)
        sp = float(np.sqrt(((q - p) ** 2).sum()))
        e = float(np.sqrt(((goal - q) ** 2).sum()))
        if n % 3 == 0 or n < 4:
            trace.append([n, round(float(q[0]), 4), round(float(q[1]), 4), round(float(q[2]), 4), round(sp, 4)])
        if e < tol and sp < 0.001:
            calm += 1
            if calm >= int(settle):
                break
        else:
            calm = 0
    f = robot.now()
    tcp = [round(float(v), 4) for v in f.tcp]
    res = {}
    if refine_key:
        r = _wrist_refine(robot.now(), refine_key, [x, y])
        if r is not None:
            res["refined_xy"] = [round(r[0], 4), round(r[1], 4)]
            res["refined_n"] = r[2]
    res.update({"arrived": calm >= int(settle), "steps": n, "tcp": tcp, "step": int(f.step),
           "err": [round(float(v), 4) for v in (goal - np.array(f.tcp[:3]))], "dry_run_margin": margin, "trace": trace[-12:]})
    return res
