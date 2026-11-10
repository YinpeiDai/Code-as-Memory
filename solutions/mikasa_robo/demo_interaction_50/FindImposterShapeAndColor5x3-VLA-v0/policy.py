"""FindImposterShapeAndColor5x3 policy: remember the cue objects, wait through the
blank phase, find the object whose (colour, shape) was absent from the cue and
touch its top with closed fingers by a vertical descent."""

VISIBLE_PIX = 40          # coloured front pixels needed to call the objects visible
HOVER_Z = 0.10            # transit / hover height of the TCP (m)
PRESS_DEPTH = 0.012       # commanded TCP target below the object top (m)
GAIN = 1.0                # P gain: commanded offset = GAIN * error
XY_FINE_GAIN = 1.6        # xy gain when within 5 cm of the hover point


def _obj_list(objs):
    out = []
    for o in objs:
        out.append([o["color"], o["shape"], round(float(o["x"]), 4), round(float(o["y"]), 4),
                    bool(o["reliable"]), round(float(o.get("zmax", 0.03)), 4)])
    return out


def _choose(cue, test):
    """Score every (colour, shape) hypothesis X = 'X is the imposter'.
    cue/test: lists [color, shape, x, y, reliable, zmax]. Lower score is better."""
    seen = set((c[0], c[1]) for c in cue if c[4])
    cue_unk = {}
    for c in cue:
        if not c[4]:
            cue_unk[c[0]] = cue_unk.get(c[0], 0) + 1
    scores = []
    for col in COLORS:
        for sh in SHAPES:
            p = 0.0
            if (col, sh) in seen:
                p += 10.0
            for c2 in COLORS:
                for s2 in SHAPES:
                    if (c2, s2) == (col, sh) or (c2, s2) in seen:
                        continue
                    p += 0.5 if cue_unk.get(c2, 0) else 3.0
            hit = [t for t in test if t[4] and t[0] == col and t[1] == sh]
            if not hit:
                p += 1.0 if any((not t[4]) and t[0] == col for t in test) else 5.0
            scores.append([p, col, sh])
    scores.sort(key=lambda s: s[0])
    best = scores[0]
    cands = [t for t in test if t[0] == best[1] and t[1] == best[2] and t[4]]
    if not cands:
        cands = [t for t in test if t[0] == best[1] and not t[4]]
    return scores, cands


def _target_from(objs, cand):
    """Touch point (x, y, ztop) for the chosen candidate using its full detection."""
    best = None
    bd = 1e9
    for o in objs:
        if o["color"] != cand[0] or o["shape"] != cand[1]:
            continue
        d = math.hypot(o["x"] - cand[2], o["y"] - cand[3])
        if d < bd:
            bd, best = d, o
    if best is None:
        return [cand[2], cand[3], 0.04]
    if best.get("split"):
        dx, dy = _template_touch_offset(best["shape"])
        return [best["x"] + dx, best["y"] + dy, best["zmax"]]
    x, y, zt = _touch_point(best["P3"])
    return [x, y, zt]


def perceive_decide(obs, goal, memory, last):
    """Track cue -> blank -> test phases from the front camera, record the cue objects,
    pick the imposter at test onset and command the touch."""
    m = dict(memory)
    defaults = {"phase": "cue", "steps_seen": 0, "cue_objs": [], "cue_frame": -1,
                "blank_frames": 0, "test_objs": [], "target": [], "target_label": "",
                "scores": "", "attempts": 0, "log": ""}
    for k, v in defaults.items():
        if m.get(k) is None:
            m[k] = v
    frames = list(obs.frames)
    m["steps_seen"] = int(m["steps_seen"]) + len(frames)
    if last is not None:
        m["log"] = (str(m["log"]) + " | " + str(last)[:600])[-2500:]
    test_frame = None
    for fr in frames:
        n = _count_visible(fr)
        if m["phase"] == "cue":
            if n >= VISIBLE_PIX and not m["cue_objs"]:
                objs, nw, nf = _scene(fr)
                m["cue_objs"] = _obj_list(objs)
                m["cue_frame"] = int(fr.step)
            elif n < VISIBLE_PIX and m["cue_objs"]:
                m["phase"] = "blank"
                m["blank_frames"] = 1
            elif n < VISIBLE_PIX:
                m["blank_frames"] = int(m["blank_frames"]) + 1
        elif m["phase"] == "blank":
            if n >= VISIBLE_PIX:
                m["phase"] = "test"
                test_frame = fr
            else:
                m["blank_frames"] = int(m["blank_frames"]) + 1
    if test_frame is not None:
        objs, nw, nf = _scene(test_frame)
        m["test_objs"] = _obj_list(objs)
        scores, cands = _choose(m["cue_objs"], m["test_objs"])
        m["scores"] = str([[round(s[0], 2), s[1], s[2]] for s in scores[:4]])
        if cands:
            cand = cands[0]
            m["target"] = [round(v, 4) for v in _target_from(objs, cand)]
            m["target_label"] = cand[0] + " " + cand[1]
        m["phase"] = "reach"
    if m["phase"] == "reach" and m["target"]:
        t = m["target"]
        m["attempts"] = int(m["attempts"]) + 1
        col, sh = (m["target_label"].split(" ") + ["", ""])[:2]
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": t[0], "y": t[1], "ztop": t[2], "color": col, "shape": sh,
            "attempt": int(m["attempts"])}}}
    return {"memory": m, "act": {"tool": "act_wait", "args": {}}}


def act_wait(robot):
    """One zero-motion native step (cue / blank phases), fingers commanded closed."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    f = robot.now()
    return {"waited": 1, "step": int(f.step)}


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    a = math.acos(c)
    if a < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(a))
    return v * a


def _desired_rpy(x, y, yaw0, extra_pitch=0.0):
    """Gripper pointing down, tilted like the expert so far targets stay reachable:
    pitch ~ -0.9 x, roll offset ~ -0.36 y (fits of the 50 demo end poses)."""
    pitch = max(-0.30, min(0.22, -0.9 * x)) + extra_pitch
    roll = math.pi + max(-0.12, min(0.12, -0.36 * y))
    return roll, pitch, yaw0


def _rot_cmd(tcp, rpy_des):
    """Rotation arguments of robot.step that move the measured orientation toward rpy_des."""
    Rm = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_to_R(rpy_des[0], rpy_des[1], rpy_des[2])
    w = _rotvec(Rd @ Rm.T)
    r = -w / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _checked_step(robot, cmd, grip, stats):
    """Send one step after a live delta dry-run; shrink translation if the native IK fails."""
    for scale in (1.0, 0.5, 0.25):
        c = [float(cmd[0]) * scale, float(cmd[1]) * scale, float(cmd[2]) * scale,
             float(cmd[3]), float(cmd[4]), float(cmd[5])]
        chk = robot.dry_run(c[0], c[1], c[2], c[3], c[4], c[5], mode="delta")
        if chk.get("ik_computed") and chk.get("within_joint_limits"):
            if scale < 1.0:
                stats["scaled"] = stats.get("scaled", 0) + 1
            robot.step(*(list(chk["command"]) + [grip]))
            return True
        stats["ik_fail"] = stats.get("ik_fail", 0) + 1
    robot.step(0.0, 0.0, 0.0, float(cmd[3]), float(cmd[4]), float(cmd[5]), grip)
    return False


def _goto(robot, goal, rpy_des, tol, max_steps, grip, log, stats, ztol=None):
    """Closed-loop translation toward goal (world xyz) plus orientation tracking.
    Arrival: xy error < tol, |z error| < ztol (default tol) and rotation error < 0.06 rad."""
    if ztol is None:
        ztol = tol
    for _ in range(max_steps):
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        err = np.array(goal, dtype=np.float64) - tcp[:3]
        r, werr = _rot_cmd(tcp, rpy_des)
        if math.hypot(err[0], err[1]) < tol and abs(err[2]) < ztol and werr < 0.06:
            return True
        t = np.clip(GAIN * err / 0.1, -1.0, 1.0)
        if math.hypot(err[0], err[1]) < 0.05:
            # near the goal the arm follows only 20-40% of a command per step: boost xy
            t[0] = max(-1.0, min(1.0, XY_FINE_GAIN * err[0] / 0.1))
            t[1] = max(-1.0, min(1.0, XY_FINE_GAIN * err[1] / 0.1))
        _checked_step(robot, [t[0], t[1], t[2], r[0], r[1], r[2]], grip, stats)
        t2 = robot.now().tcp
        log.append([round(float(v), 3) for v in list(t2[:3])])
    return False


def _refine(robot, x, y, color, shape):
    """Re-detect the target in the wrist view while hovering; returns (x, y, ztop) or None."""
    fr = robot.now()
    det = _detect(fr, "wrist")
    best = None
    bd = 0.035
    for o in det:
        if o["color"] != color:
            continue
        d = math.hypot(o["x"] - x, o["y"] - y)
        if o["shape"] == "merged":
            parts, rem = _decompose(o["P3"])
            for p in parts:
                if p["shape"] == shape:
                    dx, dy = _template_touch_offset(shape)
                    dd = math.hypot(p["x"] + dx - x, p["y"] + dy - y)
                    if dd < bd:
                        bd = dd
                        best = (p["x"] + dx, p["y"] + dy, 0.04 if shape in TALL_SHAPES else 0.028, "split")
            continue
        if o["shape"] != shape or o["partial"]:
            continue
        tx, ty, zt = _touch_point(o["P3"])
        dd = math.hypot(tx - x, ty - y)
        if dd < bd:
            bd = dd
            best = (tx, ty, zt, "single")
    return best


def act_touch(robot, x, y, ztop, color, shape, attempt=1):
    """Fly to a hover point above (x, y) while tilting like the expert, refine the target in
    the wrist camera, then descend with closed fingers until the TCP stalls on the object top.
    Every step is preceded by a live delta dry-run. Returns measured TCP path and diagnostics."""
    grip = -1.0
    log = []
    stats = {}
    out = {"attempt": attempt, "target_in": [round(x, 4), round(y, 4), round(ztop, 4)]}
    tcp0 = robot.now().tcp
    yaw0 = float(tcp0[5])
    extra = 0.0
    rpy = _desired_rpy(x, y, yaw0, extra)
    hover = max(HOVER_Z, ztop + 0.06)
    if attempt > 1:
        hover = max(ztop + 0.04, min(hover, float(tcp0[2])))
    ok = _goto(robot, [x, y, hover], rpy, 0.012, 14, grip, log, stats, ztol=0.025)
    out["hover_reached"] = bool(ok)
    ref = _refine(robot, x, y, color, shape)
    if ref is not None:
        out["refined"] = [round(ref[0], 4), round(ref[1], 4), round(ref[2], 4), ref[3]]
        shift = math.hypot(ref[0] - x, ref[1] - y)
        out["refine_shift"] = round(shift, 4)
        x, y, ztop = ref[0], ref[1], ref[2]
        if shift > 0.015:
            _goto(robot, [x, y, hover], rpy, 0.008, 3, grip, log, stats, ztol=0.03)
    goal = [x, y, ztop - PRESS_DEPTH]
    zs = []
    stuck = 0
    for k in range(14):
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        err = np.array(goal) - tcp[:3]
        r, werr = _rot_cmd(tcp, rpy)
        t = np.clip(GAIN * err / 0.1, -1.0, 1.0)
        _checked_step(robot, [t[0], t[1], t[2], r[0], r[1], r[2]], grip, stats)
        t2 = robot.now().tcp
        log.append([round(float(v), 3) for v in list(t2[:3])])
        zs.append(float(t2[2]))
        if len(zs) >= 2 and abs(zs[-1] - zs[-2]) < 0.001:
            if zs[-1] > ztop + 0.025:
                # stalled high above the object: native IK not progressing; tilt further
                stuck += 1
                extra -= 0.08 if x > 0 else -0.08
                rpy = _desired_rpy(x, y, yaw0, extra)
                stats["retilt"] = stats.get("retilt", 0) + 1
            elif len(zs) >= 3 and abs(zs[-1] - zs[-3]) < 0.002:
                break
    out["final_tcp"] = [round(float(v), 4) for v in list(robot.now().tcp)]
    out["fingers"] = [round(float(v), 4) for v in list(robot.now().fingers)]
    out["stats"] = stats
    out["path"] = log[-12:]
    return out
