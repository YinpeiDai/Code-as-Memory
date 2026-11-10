"""RememberColor5-Long policy: remember the cue cube's colour, wait for the five
candidate cubes, then bring the open gripper's TCP straight down onto the cube
whose hue matches the cue.  Perception: HSV segmentation of saturated pixels in
the fixed front camera (and the wrist camera for refinement), deprojected with
depth + calibration into world coordinates."""

SAT_MIN = 0.8          # cube pixels are ~fully saturated; wooden table <= 0.72
VAL_MIN = 0.25
HUE_TOL = 20.0         # degrees
CUBE_TOP_Z = 0.04      # measured in demos (top face at 0.040-0.042 m)
CUBE_CENTER_Z = 0.02
COLOR_NAMES = ((0.0, "red"), (30.0, "orange"), (60.0, "yellow"), (120.0, "green"),
               (180.0, "cyan"), (240.0, "blue"), (300.0, "magenta"))
GAIN = 8.0             # normalized command per metre of error (cmd = GAIN*err)
CMD_NORM_MAX = 1.0
Z_HOVER = 0.11
Z_TOUCH = 0.05


def _hsv(rgb):
    f = rgb.astype(np.float32) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    d = mx - mn + 1e-6
    r = f[..., 0]
    g = f[..., 1]
    b = f[..., 2]
    h = np.where(mx == r, np.mod((g - b) / d, 6.0),
                 np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    s = (mx - mn) / (mx + 1e-6)
    return h, s, mx


def _hue_dist(h, h0):
    return np.abs(np.mod(h - h0 + 180.0, 360.0) - 180.0)


def _color_name(h0):
    best = None
    for hc, name in COLOR_NAMES:
        dd = float(_hue_dist(np.array([h0]), hc)[0])
        if best is None or dd < best[0]:
            best = (dd, name)
    return best[1]


def _circ_mean(hs):
    a = np.deg2rad(hs)
    return float(np.mod(np.rad2deg(math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))), 360.0))


def _clusters(rgb, min_px=12):
    """Hue clusters of saturated pixels: list of dicts {hue, n, u, v}."""
    h, s, v = _hsv(rgb)
    m = (s > SAT_MIN) & (v > VAL_MIN)
    out = []
    if int(m.sum()) < min_px:
        return out
    hs = h[m]
    vv, uu = np.nonzero(m)
    used = np.zeros(hs.shape[0], dtype=bool)
    hist = np.zeros(36)
    bins = (np.mod(hs + 5.0, 360.0) // 10).astype(int) % 36
    for i in range(36):
        hist[i] = float((bins == i).sum())
    order = np.argsort(-hist)
    for i in order:
        if hist[i] < min_px:
            break
        center = i * 10.0
        sel = (~used) & (_hue_dist(hs, center) < HUE_TOL)
        if int(sel.sum()) < min_px:
            continue
        used = used | sel
        out.append({"hue": _circ_mean(hs[sel]), "n": int(sel.sum()),
                    "u": float(np.median(uu[sel])), "v": float(np.median(vv[sel]))})
    return out


def _deproject(depth, K, E, vv, uu):
    z = depth[vv, uu].astype(np.float64) / 1000.0
    ok = z > 0
    z = z[ok]
    u = uu[ok].astype(np.float64)
    v = vv[ok].astype(np.float64)
    x = (u + 0.5 - float(K[0, 2])) * z / float(K[0, 0])
    y = (v + 0.5 - float(K[1, 2])) * z / float(K[1, 1])
    pc = np.stack([x, y, z], axis=1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R


def _locate(rgb, depth, K, E, hue0, min_px=10, radius=None, need_clean=False):
    """World xy of the top-face centre of the cube with hue near hue0, or None."""
    h, s, v = _hsv(rgb)
    m = (s > SAT_MIN) & (v > VAL_MIN) & (_hue_dist(h, hue0) < HUE_TOL)
    if int(m.sum()) < min_px:
        return None
    vv, uu = np.nonzero(m)
    if radius is not None:
        mu = np.median(uu)
        mv = np.median(vv)
        keep = (np.abs(uu - mu) < radius) & (np.abs(vv - mv) < radius)
        vv = vv[keep]
        uu = uu[keep]
    touches = bool(vv.min() <= 0 or uu.min() <= 0 or vv.max() >= rgb.shape[0] - 1 or uu.max() >= rgb.shape[1] - 1)
    if need_clean and touches:
        return None
    P = _deproject(depth, K, E, vv, uu)
    if P.shape[0] < min_px:
        return None
    P = P[(P[:, 2] > -0.01) & (P[:, 2] < 0.07)]
    if P.shape[0] < min_px:
        return None
    zt = float(P[:, 2].max())
    top = P[P[:, 2] > zt - 0.005]
    ext = top.max(axis=0) - top.min(axis=0)
    c = (top.max(axis=0) + top.min(axis=0)) / 2.0
    return {"x": float(c[0]), "y": float(c[1]), "top_z": zt, "n": int(P.shape[0]),
            "ext_x": float(ext[0]), "ext_y": float(ext[1]), "touches_border": touches}


def _front_locate(frame, hue0):
    return _locate(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic,
                   hue0, min_px=10, radius=14)


def _wrist_locate(frame, hue0):
    r = _locate(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic,
                hue0, min_px=150, radius=None, need_clean=True)
    if r is None:
        return None
    if not (0.025 < r["ext_x"] < 0.06 and 0.025 < r["ext_y"] < 0.06):
        return None
    if abs(r["top_z"] - CUBE_TOP_Z) > 0.01:
        return None
    return r


def _cue_from_frame(frame):
    cl = _clusters(frame.front_rgb)
    if len(cl) == 1 and cl[0]["n"] >= 20:
        return cl[0]
    return None


def _cubes_present(frame):
    cl = _clusters(frame.front_rgb)
    return len(cl) >= 3, cl


def _mget(memory, key, default):
    if not isinstance(memory, dict):
        return default
    v = memory.get(key)
    return default if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Track the cue colour, wait for the candidate cubes, then touch the matching cube."""
    mem = {
        "stage": _mget(memory, "stage", "observe"),
        "cue_hue": float(_mget(memory, "cue_hue", -1.0)),
        "cue_color": _mget(memory, "cue_color", ""),
        "cue_step": int(_mget(memory, "cue_step", -1)),
        "cue_pixels": int(_mget(memory, "cue_pixels", 0)),
        "cubes_step": int(_mget(memory, "cubes_step", -1)),
        "target": list(_mget(memory, "target", [])),
        "touch_attempts": int(_mget(memory, "touch_attempts", 0)),
        "notes": _mget(memory, "notes", ""),
    }
    frames = list(obs.frames)
    # 1) cue: a single saturated cube visible before the candidates appear
    if mem["cue_hue"] < 0:
        for fr in frames:
            c = _cue_from_frame(fr)
            if c is not None:
                mem["cue_hue"] = round(c["hue"], 2)
                mem["cue_color"] = _color_name(c["hue"])
                mem["cue_step"] = int(fr.step)
                mem["cue_pixels"] = int(c["n"])
                break
    if isinstance(last, str) and len(last) > 0:
        mem["notes"] = (mem["notes"] + " | " + last[:300])[-1500:]
    cur = obs.current
    present, cl = _cubes_present(cur)
    if not present:
        mem["stage"] = "wait"
        return {"memory": mem, "act": {"tool": "act_wait_for_cubes",
                                       "args": {"cue_hue": float(mem["cue_hue"]), "max_steps": 200}}}
    if mem["cubes_step"] < 0:
        mem["cubes_step"] = int(cur.step)
    hue0 = float(mem["cue_hue"])
    if hue0 < 0:
        # cue never seen: fall back to the most saturated/largest cluster (low confidence)
        hue0 = float(max(cl, key=lambda c: c["n"])["hue"])
        mem["notes"] = (mem["notes"] + " | cue missing, fallback hue %.0f" % hue0)[-1500:]
    loc = _front_locate(cur, hue0)
    if loc is not None and mem["touch_attempts"] == 0:
        mem["target"] = [round(loc["x"], 4), round(loc["y"], 4), CUBE_CENTER_Z]
    tgt = mem["target"]
    n = int(mem["touch_attempts"])
    mem["touch_attempts"] = n + 1
    mem["stage"] = "touch%d" % (n + 1)
    z_touch = [Z_TOUCH, 0.045, 0.042, 0.055, 0.048][n % 5]
    tx = float(tgt[0]) if len(tgt) >= 2 else 99.0
    ty = float(tgt[1]) if len(tgt) >= 2 else 99.0
    return {"memory": mem, "act": {"tool": "act_touch",
                                   "args": {"cue_hue": hue0, "tx": tx, "ty": ty,
                                            "z_touch": z_touch, "lift_first": n > 0,
                                            "hold_steps": 25, "max_steps": 150}}}


def act_wait_for_cubes(robot, cue_hue=-1.0, max_steps=200):
    """Hold still (zero motion, gripper open) until >=3 coloured cubes are visible
    in the front camera or max_steps elapse.  Reports the cue if seen meanwhile."""
    cue = None
    steps = 0
    fr = robot.now()
    present, cl = _cubes_present(fr)
    while not present and steps < max_steps:
        fr = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        steps += 1
        if cue_hue < 0 and cue is None:
            c = _cue_from_frame(fr)
            if c is not None:
                cue = {"hue": round(c["hue"], 1), "step": int(fr.step), "n": c["n"]}
        present, cl = _cubes_present(fr)
    return {"cubes_present": bool(present), "steps_waited": steps, "frame_step": int(fr.step),
            "clusters": [[round(c["hue"]), c["n"]] for c in cl], "cue_seen": cue}


def act_touch(robot, cue_hue, tx, ty, z_touch=0.05, lift_first=False, hold_steps=25, max_steps=150):
    """Move the open gripper above the cube of hue cue_hue (front estimate tx,ty,
    refined by the wrist camera), descend until TCP z == z_touch, then hold."""
    fr = robot.now()
    tgt = [float(tx), float(ty)]
    src = "arg"
    if tgt[0] > 50:
        loc = _front_locate(fr, cue_hue)
        if loc is None:
            return {"error": "target not visible"}
        tgt = [loc["x"], loc["y"]]
        src = "front"
    wr_hist = []
    phase = "lift" if lift_first else "approach"
    held = 0
    steps = 0
    log = []
    min_d = 9.0
    wrist_used = 0
    while steps < max_steps:
        p = np.asarray(fr.tcp[:3], dtype=np.float64)
        w = _wrist_locate(fr, cue_hue)
        if w is not None:
            wr_hist.append([w["x"], w["y"]])
            wr_hist = wr_hist[-3:]
            if len(wr_hist) >= 2:
                est = np.median(np.array(wr_hist), axis=0)
                if math.hypot(est[0] - tgt[0], est[1] - tgt[1]) < 0.03:
                    tgt = [float(est[0]), float(est[1])]
                    src = "wrist"
                    wrist_used += 1
        dxy = math.hypot(tgt[0] - p[0], tgt[1] - p[1])
        d3 = math.sqrt(dxy * dxy + (p[2] - CUBE_CENTER_Z) ** 2)
        min_d = min(min_d, d3)
        if phase == "lift":
            gz = Z_HOVER
            if p[2] > Z_HOVER - 0.01:
                phase = "approach"
        if phase == "approach":
            gz = Z_HOVER
            if dxy < 0.012:
                gz = z_touch
                phase = "descend"
        if phase in ("descend", "hold"):
            gz = z_touch
            if dxy > 0.02:
                phase = "approach"
                gz = Z_HOVER
        err = np.array([tgt[0] - p[0], tgt[1] - p[1], gz - p[2]])
        cmd = GAIN * err
        nrm = float(np.linalg.norm(cmd))
        if nrm > CMD_NORM_MAX:
            cmd = cmd * (CMD_NORM_MAX / nrm)
        if phase in ("descend", "hold") and abs(p[2] - z_touch) < 0.005 and dxy < 0.006:
            phase = "hold"
            held += 1
        if steps % 3 == 0:
            log.append([int(fr.step), phase[0], round(p[0], 3), round(p[1], 3), round(p[2], 3), round(dxy, 4)])
        if held > hold_steps:
            break
        fr = robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, 1.0)
        steps += 1
    p = np.asarray(fr.tcp[:3], dtype=np.float64)
    return {"done": "hold_complete" if held > hold_steps else "max_steps", "steps": steps,
            "held": held, "target": [round(tgt[0], 4), round(tgt[1], 4)], "target_source": src,
            "wrist_updates": wrist_used, "final_tcp": [round(float(v), 4) for v in p],
            "min_dist_to_center": round(min_d, 4), "log": log[-14:]}
