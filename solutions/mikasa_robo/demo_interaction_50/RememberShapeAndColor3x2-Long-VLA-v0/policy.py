"""Policy for RememberShapeAndColor3x2-Long-VLA-v0.

Stages (memory.stage):
  cue    -> read the single cue object (colour + shape) from the front camera,
            then wait (zero commands) until the 6-object choice scene appears.
  wait   -> waiting returned; read the choice scene, pick the object with the
            remembered colour and shape, store its 3D centre.
  reach  -> move the TCP above the target, descend to ~0.035 m from its centre
            (hypothesis from demos: success = TCP within ~0.05 m of the
            object centre) and hold.
  refine -> if no success after the reach, re-perceive and descend lower.
"""

COLORS = ("red", "green", "blue")
# metric (depth-normalised) blob area in cm^2 seen by the front camera:
# demos: spheres 11.8-13.6, cubes 21.4-29.2 (cue and choice objects alike)
SHAPE_AREA_SPLIT_CM2 = 17.5
HOVER_ABOVE_TOP = 0.06      # hover height above object top (m)
TOUCH_ABOVE_TOP = 0.010     # final TCP height above object top (m); R001/R002: success only once d<~0.043
HALF_HEIGHT = 0.02          # assumed object half height (4 cm objects)


def _color_masks(img):
    x = img.astype(np.int32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    red = (R > 90) & (G * 100 < 35 * R) & (B * 100 < 35 * R)
    green = (G > 60) & (G > R + 40) & (G > B + 40)
    blue = (B > 60) & (B > R + 40) & (B > G + 30)
    return {"red": red, "green": green, "blue": blue}


def _colored_count(img):
    m = _color_masks(img)
    return int(m["red"].sum() + m["green"].sum() + m["blue"].sum())


def _components(mask, min_px=8):
    ys, xs = np.nonzero(mask)
    pix = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while pix:
        seed = pix.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pix:
                        pix.remove(q)
                        stack.append(q)
                        comp.append(q)
        if len(comp) >= min_px:
            comps.append(comp)
    return comps


def _backproject(frame, cam, ys, xs):
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    D = np.asarray(getattr(frame, cam + "_depth"))
    z = D[ys, xs].astype(np.float64) / 1000.0
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    pc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
    Rm = E[:, :3]
    t = E[:, 3]
    pw = (pc - t[None, :]) @ Rm
    return pw, z, K


def _blobs(frame, cam="front"):
    """All coloured blobs: colour, pixel area, metric area, top z, centre xyz."""
    img = getattr(frame, cam + "_rgb")
    out = []
    for col, m in _color_masks(img).items():
        for comp in _components(m):
            ys = np.array([p[0] for p in comp])
            xs = np.array([p[1] for p in comp])
            pw, z, K = _backproject(frame, cam, ys, xs)
            if len(z) < 5:
                continue
            marea = float(np.sum((z / K[0, 0]) * (z / K[1, 1]))) * 1e4
            ztop = float(np.percentile(pw[:, 2], 95))
            top = pw[pw[:, 2] > ztop - 0.006]
            cx, cy = float(top[:, 0].mean()), float(top[:, 1].mean())
            shape = "cube" if marea > SHAPE_AREA_SPLIT_CM2 else "sphere"
            margin = abs(marea - SHAPE_AREA_SPLIT_CM2) / SHAPE_AREA_SPLIT_CM2
            out.append({"color": col, "shape": shape, "px": len(comp),
                        "area_cm2": round(marea, 2), "shape_margin": round(margin, 2),
                        "top_z": round(ztop, 4),
                        "center": [round(cx, 4), round(cy, 4), round(ztop - HALF_HEIGHT, 4)],
                        "u": float(xs.mean()), "v": float(ys.mean())})
    return out


def _find_cue(frames):
    """First frame showing exactly one coloured blob (the cue)."""
    for f in frames:
        n = _colored_count(f.front_rgb)
        if 15 <= n <= 250:
            bl = _blobs(f)
            if len(bl) == 1:
                b = bl[0]
                return {"color": b["color"], "shape": b["shape"], "area": b["area_cm2"],
                        "step": int(f.step), "conf": min(1.0, b["shape_margin"] * 3)}
    return None


def _scene_frame(frames):
    """Latest frame that shows the multi-object choice scene."""
    for f in reversed(list(frames)):
        if _colored_count(f.front_rgb) > 250:
            bl = _blobs(f)
            if len(bl) >= 4:
                return f, bl
    return None, None


def _choose(blobs, color, shape):
    same = [b for b in blobs if b["color"] == color]
    if not same:
        return None, "no blob of cue colour"
    exact = [b for b in same if b["shape"] == shape]
    if len(exact) == 1:
        return exact[0], "unique colour+shape match"
    if len(same) >= 2:
        # pick by area ordering: cube = larger metric area
        same = sorted(same, key=lambda b: b["area_cm2"])
        return (same[-1] if shape == "cube" else same[0]), "area-ordered within colour"
    return same[0], "only one blob of cue colour"


def _mem(memory):
    d = {"stage": "cue", "cue_color": "", "cue_shape": "", "cue_area_cm2": 0.0,
         "cue_step": -1, "cue_confidence": 0.0, "scene_step": -1, "objects": "",
         "target": [], "target_reason": "", "reach_attempts": 0, "log": ""}
    for k in d:
        if k in memory and memory[k] is not None:
            d[k] = memory[k]
    return d


def perceive_decide(obs, goal, memory, last):
    """Track cue -> wait -> choose target -> reach stages using declared memory."""
    m = _mem(memory or {})
    frames = list(obs.frames)
    cur = obs.current
    if not m["cue_color"]:
        cue = _find_cue(frames)
        if cue is not None:
            m["cue_color"], m["cue_shape"] = cue["color"], cue["shape"]
            m["cue_area_cm2"], m["cue_step"] = cue["area"], cue["step"]
            m["cue_confidence"] = cue["conf"]
            m["log"] += "cue %s %s area %.1f at step %d; " % (cue["color"], cue["shape"], cue["area"], cue["step"])
    if m["stage"] in ("cue", "wait"):
        f, bl = _scene_frame(frames)
        if f is None:
            m["stage"] = "wait"
            return {"memory": m, "act": {"tool": "act_wait_for_scene", "args": {"max_steps": 590}}}
        m["scene_step"] = int(f.step)
        m["objects"] = "; ".join("%s %s %.1fcm2 (%.3f,%.3f)" % (b["color"], b["shape"], b["area_cm2"], b["center"][0], b["center"][1]) for b in bl)
        if not m["cue_color"]:
            # cue never seen: cannot do better than a guess; take the first blob
            b, why = bl[0], "cue missing: guess"
        else:
            b, why = _choose(bl, m["cue_color"], m["cue_shape"])
            if b is None:
                b, why = bl[0], why + ": guess"
        m["target"] = [b["center"][0], b["center"][1], b["center"][2]]
        m["target_reason"] = why
        m["stage"] = "reach"
        top = b["center"][2] + HALF_HEIGHT
        m["log"] += "scene at %d, target %s (%s); " % (m["scene_step"], m["target"], why)
        m["reach_attempts"] = 1
        return {"memory": m, "act": {"tool": "act_reach_touch", "args": {
            "x": m["target"][0], "y": m["target"][1],
            "hover_z": top + HOVER_ABOVE_TOP, "touch_z": top + TOUCH_ABOVE_TOP, "hold_steps": 15}}}
    # reach returned without episode end: retry lower (bounded)
    tx, ty, tz = m["target"]
    top = tz + HALF_HEIGHT
    m["reach_attempts"] = int(m["reach_attempts"]) + 1
    k = m["reach_attempts"]
    m["stage"] = "refine"
    m["log"] += "reach %d returned (%s); " % (k - 1, str(last)[:120] if last else "")
    touch = max(top + TOUCH_ABOVE_TOP - 0.005 * (k - 1), top + 0.003)
    return {"memory": m, "act": {"tool": "act_reach_touch", "args": {
        "x": tx, "y": ty, "hover_z": top + HOVER_ABOVE_TOP, "touch_z": touch, "hold_steps": 15}}}


def act_wait_for_scene(robot, max_steps=590):
    """Send zero motion (gripper open) until the front camera shows the multi-object scene."""
    n = 0
    cue_seen = 0
    while n < max_steps:
        f = robot.now()
        c = _colored_count(f.front_rgb)
        if c > 250 and len(_blobs(f)) >= 4:
            return {"scene_visible": True, "steps_waited": n, "step": int(f.step), "cue_frames": cue_seen}
        if 15 <= c <= 250:
            cue_seen += 1
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    return {"scene_visible": False, "steps_waited": n, "step": int(robot.now().step), "cue_frames": cue_seen}


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(Rm):
    c = max(-1.0, min(1.0, (np.trace(Rm) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return w / (2 * s) * ang


def _cmd(tcp, prev_tcp, target, R_goal, gain=1.0, kd=0.3, rot_gain=0.5):
    err = np.array(target) - tcp[:3]
    vel = tcp[:3] - prev_tcp[:3]
    a = np.clip(gain * err / 0.1 - kd * vel / 0.1, -1.0, 1.0)
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _rotvec(R_goal @ Rc.T)          # world-frame rotation needed
    r = np.clip(-rot_gain * w / 0.1, -0.5, 0.5)   # native: offset = -0.1*r
    return a, r, err


def act_reach_touch(robot, x, y, hover_z, touch_z, hold_steps=15, max_steps=90):
    """Move TCP above (x,y) at hover_z, descend to touch_z and hold; returns measured TCP log."""
    f0 = robot.now()
    tcp0 = np.array(f0.tcp, dtype=np.float64)
    R_goal = _rpy_to_R(math.pi, 0.0, float(tcp0[5]))   # gripper pointing down, keep yaw
    prev = tcp0.copy()
    phase = "hover"
    n = 0
    held = 0
    hist = []
    min_z = 9.0
    while n < max_steps:
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        if phase == "hover":
            exy = math.hypot(x - tcp[0], y - tcp[1])
            tgt = [x, y, hover_z]
            if exy < 0.01 and abs(tcp[2] - hover_z) < 0.03:
                phase = "descend"
        if phase == "descend":
            tgt = [x, y, touch_z]
            if abs(tcp[2] - touch_z) < 0.004:
                phase = "hold"
        if phase == "hold":
            tgt = [x, y, touch_z]
            held += 1
            if held > hold_steps:
                break
        a, r, err = _cmd(tcp, prev, tgt, R_goal)
        prev = tcp
        min_z = min(min_z, float(tcp[2]))
        if n % 3 == 0:
            hist.append([n, phase[0], round(float(tcp[0]), 3), round(float(tcp[1]), 3), round(float(tcp[2]), 3)])
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), 1.0)
        n += 1
    tcp = np.array(robot.now().tcp, dtype=np.float64)
    return {"phase": phase, "steps": n, "final_tcp": [round(float(v), 4) for v in tcp],
            "xy_err": round(math.hypot(x - tcp[0], y - tcp[1]), 4), "z": round(float(tcp[2]), 4),
            "min_z": round(min_z, 4), "hist": hist[-12:]}
