"""FindImposterShape5 policy: remember the first shape set (wrist RGB-D), wait through the blank,
classify the second set, pick the shape absent from the first set, and touch it."""

STEP_LIMIT = 25          # native episode step limit (task description)
MIN_BLUE_PIX = 60        # a frame "shows objects" when at least this many blue pixels are visible
FEAT_SCALE = {"zmax": 0.006, "top2": 0.15, "top_circ": 0.08, "rad_ratio": 0.12, "hole_px": 8.0}


def _safe_detect(frame):
    try:
        return _detect(frame)
    except Exception:  # perception fault on one frame: treat as no detection, keep the episode going
        return []


def _blue_count(frame):
    return int(_blue_mask(frame.wrist_rgb).sum())


def _feat_dist(a, b):
    """Shape-feature distance; height enters only as tall/short (set-2 short shapes appear at 0.016 m vs 0.028 m in set 1)."""
    d = 0.0
    for k, sc in FEAT_SCALE.items():
        if k == "zmax":
            d += (3.0 * (float(a[k] > 0.034) - float(b[k] > 0.034))) ** 2
        else:
            d += ((a[k] - b[k]) / sc) ** 2
    return float(d ** 0.5)


def _choose_imposter(set1, set2):
    """Return (index, confidence, reason) of the set-2 object whose shape is absent from set 1."""
    names1 = [o["shape"] for o in set1]
    cands = [i for i, o in enumerate(set2) if o["shape"] not in names1]
    # novelty of each set-2 object: feature distance to its nearest set-1 object
    nov = [min([_feat_dist(o, p) for p in set1] or [0.0]) for o in set2]
    if len(cands) == 1:
        return cands[0], 1.0, "unique_absent_class"
    pool = cands if cands else list(range(len(set2)))
    best = max(pool, key=lambda i: nov[i])
    return best, 0.5, "fallback_feature_novelty(%d candidates)" % len(cands)


def _summ(objs):
    return [[round(o["xy"][0], 4), round(o["xy"][1], 4), round(o["zmax"], 4)] for o in objs]


def perceive_decide(obs, goal, memory, last):
    """Track cue phases from wrist images and choose the next action."""
    m = {
        "phase": _mget(memory, "phase", "observe_set1"),
        "set1_shapes": _mget(memory, "set1_shapes", []),
        "set1_positions": _mget(memory, "set1_positions", []),
        "set1_step": _mget(memory, "set1_step", -1),
        "blank_step": _mget(memory, "blank_step", -1),
        "set2_step": _mget(memory, "set2_step", -1),
        "set2_shapes": _mget(memory, "set2_shapes", []),
        "set2_positions": _mget(memory, "set2_positions", []),
        "target_shape": _mget(memory, "target_shape", ""),
        "target_pos": _mget(memory, "target_pos", []),
        "target_conf": _mget(memory, "target_conf", 0.0),
        "touch_attempts": _mget(memory, "touch_attempts", 0),
        "decisions": _mget(memory, "decisions", 0) + 1,
        "notes": _mget(memory, "notes", ""),
        "set1_feats": _mget(memory, "set1_feats", ""),
    }
    frames = list(obs.frames)
    step_now = int(getattr(frames[-1], "step", 0)) if frames else 0
    set1_objs = None
    for fr in frames:
        if m["phase"] in ("touch", "close", "done"):
            break
        n = _blue_count(fr)
        st = int(getattr(fr, "step", 0))
        if m["phase"] == "observe_set1":
            if n >= MIN_BLUE_PIX:
                objs = _safe_detect(fr)
                if objs:
                    m["set1_shapes"] = [o["shape"] for o in objs]
                    m["set1_positions"] = _summ(objs)
                    m["set1_feats"] = repr([[round(o[k], 4) for k in FEAT_SCALE] for o in objs])
                    m["set1_step"] = st
                    m["phase"] = "wait_blank"
                    set1_objs = objs
        elif m["phase"] == "wait_blank":
            if n < MIN_BLUE_PIX:
                m["phase"] = "wait_set2"
                m["blank_step"] = st
        elif m["phase"] == "wait_set2":
            if n >= MIN_BLUE_PIX:
                objs = _safe_detect(fr)
                if len(objs) >= 2:
                    s1 = _set1_from_memory(m)
                    idx, conf, why = _choose_imposter(s1, objs)
                    t = objs[idx]
                    m["set2_shapes"] = [o["shape"] for o in objs]
                    m["set2_positions"] = _summ(objs)
                    m["set2_step"] = st
                    m["target_shape"] = t["shape"]
                    m["target_pos"] = [round(t["xy"][0], 4), round(t["xy"][1], 4), round(t["zmax"], 4)]
                    m["target_conf"] = conf
                    m["notes"] = "target via " + why
                    m["phase"] = "touch"
    if m["phase"] in ("observe_set1", "wait_blank", "wait_set2"):
        return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 1}}}
    remaining = STEP_LIMIT - step_now
    if last is not None and m["phase"] in ("touch", "close"):
        m["notes"] = (m["notes"] + " | last: " + str(last))[-1500:]
    x, y, ztop = m["target_pos"]
    if m["phase"] == "touch":
        m["touch_attempts"] += 1
        m["phase"] = "close"
        return {"memory": m, "act": {"tool": "act_touch", "args": {"x": x, "y": y, "ztop": ztop, "max_steps": max(1, remaining)}}}
    if m["phase"] == "close":
        m["phase"] = "done"
        return {"memory": m, "act": {"tool": "act_close_hold", "args": {"x": x, "y": y, "ztop": ztop, "steps": max(1, remaining)}}}
    return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 1}}}


def _mget(memory, key, default):
    v = memory.get(key) if isinstance(memory, dict) else None
    return default if v is None else v


def _set1_from_memory(m):
    out = []
    feats = []
    s = m.get("set1_feats", "")
    # parse "[[a, b, ...], ...]" without eval
    for chunk in s.replace("[[", "").replace("]]", "").split("], ["):
        vals = [v for v in chunk.split(",") if v.strip()]
        if len(vals) == len(FEAT_SCALE):
            feats.append([float(v) for v in vals])
    for shape, fv in zip(m["set1_shapes"], feats):
        o = {"shape": shape}
        for k, v in zip(FEAT_SCALE, fv):
            o[k] = v
        out.append(o)
    return out


RESP_A = 0.27   # fitted (R002, CPU): next TCP step = RESP_A * 0.1 * cmd + RESP_B * previous step
RESP_B = 0.30
POS_GAIN = 1.2   # translation command = POS_GAIN * error / 0.1 (R002 used 1.5: 1 cm overshoot, success)
ROT_GAIN = 0.6   # rotation command = -ROT_GAIN * base-frame rotation-vector error / 0.1


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(Rm):
    c = max(-1.0, min(1.0, (float(np.trace(Rm)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])


def _down_rpy(yaw):
    """Gripper pointing straight down (roll pi, pitch 0) with the given yaw."""
    return [math.pi, 0.0, float(yaw)]


def _rot_cmd(tcp, rpy_des, gain=ROT_GAIN, rmax=0.5):
    """Rotation command holding the desired orientation. Offset applied is -0.1*r about base axes,
    so r = -gain * w_err / 0.1 where R_des = exp(w_err) R_meas (demo fit: one step realises ~0.35 of it)."""
    w = _rotvec(_rot(rpy_des) @ _rot(tcp[3:6]).T)
    r = -gain * w / 0.1
    n = float(np.abs(r).max())
    if n > rmax:
        r = r * (rmax / n)
    return r, float(np.linalg.norm(w))


def _cmd_toward(tcp, target, gain=POS_GAIN, vmax=1.0):
    """Translation command toward target, max-component normalised to vmax (keeps the straight direction)."""
    d = (np.asarray(target, float) - np.asarray(tcp[:3], float)) * gain / 0.1
    big = np.abs(d).max()
    if big > vmax:
        d = d * (vmax / big)
    return d


def act_wait(robot, steps=1):
    """Hold still (zero arm command, gripper open request) for `steps` native steps during the cue phases."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"step": int(f.step), "blue_px": _blue_count(f)}


def act_touch(robot, x, y, ztop, max_steps=15):
    """Move the open gripper over (x, y), then lower the TCP between the fingers to the object's mid-height.

    Holds the gripper pointing straight down (initial yaw). The height target ramps smoothly from
    ztop + 0.045 (xy error >= 30 mm) to z = max(0.015, ztop/2) (xy error <= 8 mm); fingers must be open
    (>= 0.034 m) before going below ztop + 0.03. Stops after 3 settled steps (<5 mm, <3 mm/step)."""
    zf = max(0.015, 0.5 * float(ztop))
    zsafe = float(ztop) + 0.045
    log = []
    steps = 0
    settled = 0
    f0 = robot.now()
    rpy_des = _down_rpy(np.asarray(f0.tcp, float)[5])
    prev = None
    min_err = 9.0
    for i in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, float)
        fing = float(np.asarray(f.fingers, float).min())
        exy = math.hypot(x - tcp[0], y - tcp[1])
        ramp = min(1.0, max(0.0, (exy - 0.008) / 0.022))
        zt = zf + ramp * (max(zf, zsafe) - zf)
        if fing < 0.034:
            zt = max(zt, float(ztop) + 0.03)
        cmd = _cmd_toward(tcp, [x, y, zt])
        rc, rerr = _rot_cmd(tcp, rpy_des)
        if i == 0:
            chk = robot.dry_run(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), mode="delta")
            log.append(["dry", round(float(chk.get("min_joint_margin_rad", -1)), 3)])
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), 1.0)
        steps += 1
        g = robot.now()
        t2 = np.asarray(g.tcp, float)
        err = float(np.linalg.norm(t2[:3] - np.array([x, y, zf])))
        min_err = min(min_err, err)
        mv = float(np.linalg.norm(t2[:3] - tcp[:3]))
        log.append([int(g.step), round(float(t2[0]), 3), round(float(t2[1]), 3), round(float(t2[2]), 3), round(err, 3), round(rerr, 3)])
        if err < 0.005 and mv < 0.003:
            settled += 1
            if settled >= 3:
                break
    g = robot.now()
    t2 = np.asarray(g.tcp, float)
    return {"steps": steps, "final_tcp": [round(float(v), 4) for v in t2], "target": [x, y, zf], "min_err": round(min_err, 4),
            "fingers": [round(float(v), 4) for v in g.fingers], "log": log[-12:]}


def act_close_hold(robot, x, y, ztop, steps=5):
    """Hold the TCP at the object's mid-height (gripper down) and close the gripper; returns measured state."""
    zf = max(0.015, 0.5 * float(ztop))
    log = []
    rpy_des = _down_rpy(np.asarray(robot.now().tcp, float)[5])
    for _ in range(int(steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, float)
        cmd = _cmd_toward(tcp, [x, y, zf])
        rc, rerr = _rot_cmd(tcp, rpy_des)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), -1.0)
        g = robot.now()
        log.append([int(g.step), [round(float(v), 3) for v in g.tcp[:3]], [round(float(v), 4) for v in g.fingers]])
    return {"log": log[-8:]}
