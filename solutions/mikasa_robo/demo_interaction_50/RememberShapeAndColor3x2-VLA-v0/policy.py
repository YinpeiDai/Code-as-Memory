"""RememberShapeAndColor3x2: remember the cue object's colour+shape, then touch the matching object.

Perception (wrist camera, front camera as fallback): saturated red/green/blue blobs on the orange table,
back-projected to world XYZ with depth + intrinsics + extrinsics. Shape = XY extent of the pixels within 3 mm
of the blob's highest point: cube top face (4 cm square) gives >= 0.04 m, sphere cap gives <= 0.028 m.
"""

COLORS = ("red", "green", "blue")
SHAPE_EXTENT_SPLIT = 0.034  # m; demos: spheres 0.016-0.028, cubes 0.040-0.060
MIN_AREA = {"wrist": 8, "front": 3}
HOVER_ABOVE_TOP = 0.045      # m above object top while travelling in XY
PRESS_BELOW_TOP = 0.012      # m below object top for the final descent target (ensures contact)
XY_DESCEND_TOL = 0.015       # m XY error below which the final descent starts


def _color_masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return {
        "red": (r > 90) & (g * 100 < 35 * r) & (b * 100 < 35 * r),
        "green": (g > 80) & (r * 10 < 6 * g) & (b * 10 < 6 * g),
        "blue": (b > 80) & (r * 10 < 5 * b) & (g * 10 < 6 * b),
    }


def _components(mask, minarea):
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    todo = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while todo:
        seed = todo.pop()
        stack = [seed]
        pix = [seed]
        while stack:
            y, x = stack.pop()
            for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if q in todo:
                    todo.discard(q)
                    stack.append(q)
                    pix.append(q)
        if len(pix) >= minarea:
            comps.append(np.array(pix, dtype=np.int64))
    return comps


def _backproject(pix, depth, K, E):
    z = depth[pix[:, 0], pix[:, 1]].astype(np.float64) / 1000.0
    ok = z > 0
    if not ok.any():
        return np.zeros((0, 3))
    u = pix[ok, 1] + 0.5
    v = pix[ok, 0] + 0.5
    z = z[ok]
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return (cam - E[:, 3]) @ E[:, :3]


def _detect(frame, cam):
    """Coloured objects seen by one camera: colour, pixel area, world top-centre, top extent, shape."""
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    out = []
    for col, m in _color_masks(rgb).items():
        for pix in _components(m, MIN_AREA[cam]):
            Wp = _backproject(pix, depth, K, E)
            if len(Wp) < 3:
                continue
            zt = float(Wp[:, 2].max())
            top = Wp[Wp[:, 2] > zt - 0.003]
            c = top[:, :2].mean(0)
            ext = float(2.0 * np.sqrt(((top[:, :2] - c) ** 2).sum(1)).max())
            out.append({
                "color": col, "area": int(len(pix)),
                "x": float(c[0]), "y": float(c[1]), "top": zt, "extent": ext,
                "shape": "cube" if ext > SHAPE_EXTENT_SPLIT else "sphere",
                "uv": [float(pix[:, 1].mean()), float(pix[:, 0].mean())],
            })
    return out


def _full_set(dets):
    """True when the 3x2 choice set (each colour as one cube and one sphere candidate) is visible."""
    if len(dets) < 6:
        return False
    for col in COLORS:
        if len([d for d in dets if d["color"] == col]) < 2:
            return False
    return True


def _pick(dets, color, shape):
    """Pick the candidate of the remembered colour whose shape matches; relative extent breaks ties."""
    cand = [d for d in dets if d["color"] == color]
    if not cand:
        return None, 0.0
    exact = [d for d in cand if d["shape"] == shape]
    if len(exact) == 1 and len(cand) == 2:
        return exact[0], 1.0
    # ambiguous: cube = largest top extent, sphere = smallest
    cand = sorted(cand, key=lambda d: d["extent"])
    best = cand[-1] if shape == "cube" else cand[0]
    return best, 0.5


def _empty_memory():
    return {
        "phase": "observe_cue",
        "cue_color": "",
        "cue_shape": "",
        "cue_step": -1,
        "cue_confidence": 0.0,
        "target": [],
        "target_info": "",
        "choice_step": -1,
        "frames_seen": 0,
        "objects_seen_last": 0,
        "touch_attempts": 0,
        "log": "",
    }


def perceive_decide(obs, goal, memory, last):
    """Record the cue (single object) as soon as visible, wait through the delay, then when the six
    candidates appear choose the one matching remembered colour and shape and touch it."""
    mem = _empty_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    frames = obs.frames
    mem["frames_seen"] = int(mem["frames_seen"]) + len(frames)
    cur = obs.current

    # 1. cue: the first frame showing exactly one coloured object (wrist cam; front cam must agree if possible)
    if not mem["cue_color"] and mem["phase"] == "observe_cue":
        for fr in frames:
            dets = _detect(fr, "wrist")
            if len(dets) == 1:
                d = dets[0]
                conf = 1.0
                fd = _detect(fr, "front")
                if len(fd) == 1:
                    if fd[0]["color"] != d["color"] or fd[0]["shape"] != d["shape"]:
                        conf = 0.5
                mem["cue_color"] = d["color"]
                mem["cue_shape"] = d["shape"]
                mem["cue_step"] = int(fr.step)
                mem["cue_confidence"] = conf
                mem["log"] = "cue %s %s ext=%.3f at step %d" % (d["color"], d["shape"], d["extent"], fr.step)
                mem["phase"] = "delay"
                break

    # 2. choice set
    if mem["phase"] in ("observe_cue", "delay"):
        dets = _detect(cur, "wrist")
        src = "wrist"
        if not _full_set(dets):
            fd = _detect(cur, "front")
            if _full_set(fd):
                dets, src = fd, "front"
        mem["objects_seen_last"] = len(dets)
        ready = _full_set(dets)
        if not ready and mem["cue_color"] and len(dets) >= 4 and \
                len([d for d in dets if d["color"] == mem["cue_color"]]) == 2:
            ready = True  # choice phase with one object hidden; the remembered colour pair is visible
            src = src + "-partial"
        if ready:
            if not mem["cue_color"]:
                # cue was never seen: no information; fall back to the first detection (low confidence)
                mem["cue_confidence"] = 0.0
                mem["cue_color"] = dets[0]["color"]
                mem["cue_shape"] = dets[0]["shape"]
            tgt, conf = _pick(dets, mem["cue_color"], mem["cue_shape"])
            mem["target"] = [tgt["x"], tgt["y"], tgt["top"]]
            mem["target_info"] = "%s %s via %s ext=%.3f conf=%.1f" % (
                tgt["color"], tgt["shape"], src, tgt["extent"], conf)
            mem["choice_step"] = int(cur.step)
            mem["phase"] = "touch"

    if mem["phase"] == "touch":
        mem["touch_attempts"] = int(mem["touch_attempts"]) + 1
        t = mem["target"]
        # later attempts: re-locate target with the static front camera (it may have moved)
        if int(mem["touch_attempts"]) > 1:
            fd = _detect(cur, "front")
            cand = [d for d in fd if d["color"] == mem["cue_color"] and d["shape"] == mem["cue_shape"]]
            if cand:
                c = min(cand, key=lambda d: (d["x"] - t[0]) ** 2 + (d["y"] - t[1]) ** 2)
                t = [c["x"], c["y"], c["top"]]
                mem["target"] = t
        return {"memory": mem, "act": {"tool": "act_touch", "args": {
            "x": float(t[0]), "y": float(t[1]), "top": float(t[2]), "max_steps": 30,
            "color": mem["cue_color"], "shape": mem["cue_shape"]}}}

    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}


def act_wait(robot, steps=1):
    """Hold the arm still (zero motion, gripper open) for `steps` native steps; returns the TCP."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    p = robot.now().tcp
    return {"waited": int(steps), "tcp": [float(v) for v in p[:3]]}


def _p_cmd(err, gain):
    """Proportional normalized command (offset = gain*err, clipped). Pure P: a velocity-compensating
    controller oscillated in R006 because the per-axis response differs with pose."""
    return max(-1.0, min(1.0, gain * err / 0.1))


def _relocate(frame, color, shape, x, y):
    """Current top-centre of the remembered object near (x, y) from the static front camera (wrist fallback)."""
    for cam in ("front", "wrist"):
        cand = [d for d in _detect(frame, cam) if d["color"] == color
                and (d["x"] - x) ** 2 + (d["y"] - y) ** 2 < 0.06 ** 2]
        same = [d for d in cand if d["shape"] == shape]
        cand = same or cand
        if cand:
            d = min(cand, key=lambda d: (d["x"] - x) ** 2 + (d["y"] - y) ** 2)
            return d["x"], d["y"], cam
    return None


def act_touch(robot, x, y, top, max_steps=30, gripper=1.0, low_z=0.03, hover=0.02, static_dq=0.01,
              settle_wait=2, offsets=(0.0, 0.0, 0.012, -0.012), color="", shape=""):
    """Reach the object whose top centre is (x, y, top) with the open gripper and come to rest there:
    travel at hover height (top+`hover`); once the object is between the fingers (|ex|<0.02, |ey|<0.016;
    fingers spread along world y) descend to TCP z=`low_z` (object mid-height); when within 1.5 cm (xy) / 1.2 cm (z) send zero
    commands so the arm stops. Evidence (R002-R005): success fires when the TCP is close to the target AND
    the arm is static (max |dq| <~0.01 rad/step). If static `settle_wait` steps without the episode ending,
    re-locate the object (`color`/`shape`, front camera) and retry with the next y offset. Returns per-step trace (x, y, z, stage, max|dq|)."""
    hist = []
    fr = robot.now()
    p0 = fr.tcp
    chk = robot.dry_run(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, mode="delta")
    margin0 = chk.get("min_joint_margin_rad")
    sgn = 1.0 if y - float(p0[1]) >= 0.0 else -1.0
    oi = 0
    relocs = []
    stage = "travel"
    static_n = 0
    q_prev = np.array(fr.joints, dtype=np.float64)
    z_prev = float(p0[2])
    for i in range(int(max_steps)):
        fr = robot.now()
        p = np.array(fr.tcp[:3], dtype=np.float64)
        q = np.array(fr.joints, dtype=np.float64)
        dq = float(np.abs(q - q_prev).max()) if i > 0 else 0.0
        vz = float(p[2] - z_prev) if i > 0 else 0.0
        q_prev = q
        z_prev = float(p[2])
        gy = y + sgn * float(offsets[oi])
        ex, ey = x - p[0], gy - p[1]
        if stage == "travel" and abs(ex) < 0.02 and abs(ey) < 0.016:
            stage = "descend"
        gz = top + hover if stage == "travel" else low_z
        ez = gz - p[2]
        near = ex * ex + ey * ey < 0.015 ** 2 and abs(ez) < 0.012
        if stage == "descend" and near:
            stage = "hold"
            static_n = 0
        elif stage == "hold" and not (ex * ex + ey * ey < 0.02 ** 2 and abs(ez) < 0.015):
            stage = "descend"
        if stage == "hold":
            if dq < static_dq:
                static_n += 1
            if static_n > int(settle_wait) and oi < len(offsets) - 1:
                # resting near the target without success: re-locate it (it may have been bumped)
                oi += 1
                stage = "descend"
                if color:
                    loc = _relocate(fr, color, shape, x, y)
                    if loc is not None:
                        relocs.append([round(loc[0], 3), round(loc[1], 3), loc[2]])
                        x, y = loc[0], loc[1]
        hist.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3),
                     stage[0] + str(oi), round(dq, 3)])
        if stage == "travel":
            cmd = (_p_cmd(ex, 1.6), _p_cmd(ey, 1.6), _p_cmd(ez, 1.6))
        elif stage == "descend":
            # z: PD (mild damping on the measured per-step z velocity) brakes the fast descent
            cmd = (_p_cmd(ex, 1.3), _p_cmd(ey, 1.3), max(-1.0, min(1.0, (2.5 * ez - 1.2 * vz) / 0.1)))
        else:
            cmd = (0.0, 0.0, 0.0)
        robot.step(cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, float(gripper))
    p = robot.now().tcp
    return {"reached_stage": stage, "offset_index": oi, "relocations": relocs, "final_tcp": [float(v) for v in p[:3]],
            "start_tcp": [float(v) for v in p0[:3]], "dry_run_min_margin": margin0,
            "trace": hist[-14:]}
