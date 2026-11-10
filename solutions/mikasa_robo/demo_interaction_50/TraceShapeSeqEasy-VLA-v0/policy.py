"""TraceShapeSeqEasy policy: watch red-cube circles during the cue, grasp the green
cube, trace the same circles in order (start point, CCW/CW as observed), drop the
cube, press the button.  Perception: front RGB-D colour segmentation projected to
world with calibration; wrist RGB-D to refine the grasp point."""

CUBE_TOP_RGB = (216, 110, 110)
GREEN_TOP_RGB = (110, 216, 110)
BUTTON_RGB = (208, 136, 136)
SHAPE_FRAMES = 64
TRACE_Z = 0.06
GRASP_Z = 0.028
HOVER_Z = 0.15
LIFT_Z = 0.18
BUTTON_SAFE_Z = 0.21
LEAD_DEG = 10.0
TRACE_AMAX = 0.18


# ---------------------------------------------------------------- geometry
def _euler_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    s = math.sin(ang)
    if s < 1e-4:
        # near pi: use diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * ang
    return v * (ang / (2.0 * s))


def _roty(b):
    c, s = math.cos(b), math.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _pix_world(depth, K, E, ys, xs):
    z = depth[ys, xs].astype(np.float64) / 1000.0
    x = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t) @ R


def _color_points(frame, cam, rgb, tol, zlo, zhi):
    img = np.asarray(getattr(frame, cam + "_rgb")).astype(np.int16)
    depth = np.asarray(getattr(frame, cam + "_depth"))
    m = (np.abs(img - np.array(rgb, dtype=np.int16)).max(-1) < tol) & (depth > 0)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros((0, 3))
    P = _pix_world(depth, np.asarray(getattr(frame, cam + "_K")), np.asarray(getattr(frame, cam + "_extrinsic")), ys, xs)
    return P[(P[:, 2] > zlo) & (P[:, 2] < zhi)]


def _lamp(frame):
    img = np.asarray(frame.front_rgb).astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    ng = int(((g > 180) & (r < 80) & (b < 80)).sum())
    nr = int(((r > 180) & (g < 60) & (b < 60)).sum())
    if ng > 40 and ng > nr:
        return "green"
    if nr > 40:
        return "red"
    return "off"


def _red_cube_xy(frame):
    P = _color_points(frame, "front", CUBE_TOP_RGB, 22, 0.03, 0.05)
    if len(P) < 8:
        return None
    return P[:, :2].mean(0)


def _green_cube_front(frame):
    P = _color_points(frame, "front", GREEN_TOP_RGB, 30, 0.025, 0.055)
    if len(P) < 6:
        return None
    return P[:, :2].mean(0)


def _green_cube_wrist(frame):
    img = np.asarray(frame.wrist_rgb).astype(np.int16)
    depth = np.asarray(frame.wrist_depth)
    m = (img[..., 1] - img[..., 0] > 60) & (img[..., 1] - img[..., 2] > 60) & (depth > 0)
    ys, xs = np.nonzero(m)
    if len(ys) < 30:
        return None
    P = _pix_world(depth, np.asarray(frame.wrist_K), np.asarray(frame.wrist_extrinsic), ys, xs)
    P = P[(P[:, 2] > 0.0) & (P[:, 2] < 0.08)]
    if len(P) < 30:
        return None
    top = P[P[:, 2] > P[:, 2].max() - 0.004]
    if len(top) < 15:
        return None
    return [float(top[:, 0].mean()), float(top[:, 1].mean()), float(P[:, 2].max())]


def _button_xy(frame):
    P = _color_points(frame, "front", BUTTON_RGB, 30, 0.045, 0.09)
    if len(P) < 20:
        return None
    top = P[P[:, 2] > P[:, 2].max() - 0.003]
    return top[:, :2].mean(0)


def _fit_circle(P):
    x, y = P[:, 0], P[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones(len(x))], 1)
    b = x ** 2 + y ** 2
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = s[0], s[1]
    r = math.sqrt(max(s[2] + cx ** 2 + cy ** 2, 1e-9))
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    return cx, cy, r, float(res.max()), float(ang[0]), float(ang[-1] - ang[0])


def _analyse_cue(frames):
    """Return dict with circles fitted from red-cube track between lamp red and lamp green."""
    lamps = [_lamp(f) for f in frames]
    red_idx = [i for i, l in enumerate(lamps) if l == "red"]
    green_idx = [i for i, l in enumerate(lamps) if l == "green"]
    out = {"n_frames": len(frames), "red_on": None, "green_on": None, "circles": []}
    if not red_idx:
        return out
    lr = red_idx[0]
    lg = green_idx[0] if green_idx else len(frames)
    out["red_on"], out["green_on"] = lr, lg
    pts = {}
    for i in range(lr, lg):
        xy = _red_cube_xy(frames[i])
        if xy is not None:
            pts[i] = xy
    n = int(round((lg - lr) / float(SHAPE_FRAMES)))
    n = max(1, n)
    chunk = (lg - lr) / float(n)
    circles = []
    for k in range(n):
        a = lr + int(round(k * chunk))
        b = lr + int(round((k + 1) * chunk))
        idx = [i for i in range(a, b) if i in pts]
        if len(idx) < 12:
            continue
        P = np.array([pts[i] for i in idx])
        cx, cy, r, res, a0, sweep = _fit_circle(P)
        circles.append({"cx": float(cx), "cy": float(cy), "r": float(r), "a0": a0,
                        "dir": 1 if sweep >= 0 else -1, "sweep_deg": float(math.degrees(sweep)),
                        "res": res, "start": [float(P[0, 0]), float(P[0, 1])], "n": len(idx), "f0": a})
    # Easy variant: shared centre; use the median centre when a fit is poor
    if circles:
        cxm = float(np.median([c["cx"] for c in circles]))
        cym = float(np.median([c["cy"] for c in circles]))
        for c in circles:
            if c["res"] > 0.012 or math.hypot(c["cx"] - cxm, c["cy"] - cym) > 0.02:
                c["cx"], c["cy"] = cxm, cym
                c["r"] = float(math.hypot(c["start"][0] - cxm, c["start"][1] - cym))
                c["a0"] = float(math.atan2(c["start"][1] - cym, c["start"][0] - cxm))
                c["refit"] = True
    out["circles"] = circles
    return out


# ---------------------------------------------------------------- control
def _cmd(tcp, target, Rdes, a_max, kp=1.0, kr=0.5):
    p = np.asarray(tcp[:3], dtype=np.float64)
    e = np.asarray(target, dtype=np.float64) - p
    a = kp * e / 0.1
    m = float(np.linalg.norm(a))
    if m > a_max:
        a = a * (a_max / m)
    R = _euler_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _rotvec(Rdes @ R.T)
    rv = -kr * w / 0.1
    nr = float(np.linalg.norm(rv))
    if nr > 0.6:
        rv = rv * (0.6 / nr)
    return [float(a[0]), float(a[1]), float(a[2]), float(rv[0]), float(rv[1]), float(rv[2])]


def _move(robot, target, Rdes, grip, a_max=0.6, tol=0.004, max_steps=150, settle=0.0015, stall_steps=8):
    """Closed-loop move; stops when settled at target, or when the measured TCP has not
    moved (>0.6 mm) for stall_steps steps although still far away (IK/joint-limit stall)."""
    prev = None
    n = 0
    stalled = False
    hist = []
    f = robot.now()
    for n in range(max_steps):
        tcp = np.asarray(f.tcp, dtype=np.float64)
        err = float(np.linalg.norm(np.asarray(target) - tcp[:3]))
        moved = 1.0 if prev is None else float(np.linalg.norm(tcp[:3] - prev))
        if err < tol and moved < settle:
            break
        hist.append(tcp[:3].copy())
        if len(hist) > stall_steps and err > 3 * tol and float(np.linalg.norm(hist[-1] - hist[-1 - stall_steps])) < 0.0006:
            stalled = True
            break
        prev = tcp[:3].copy()
        a = _cmd(tcp, target, Rdes, a_max)
        f = robot.step(a[0], a[1], a[2], a[3], a[4], a[5], grip)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return {"steps": n, "err": float(np.linalg.norm(np.asarray(target) - tcp[:3])), "tcp": tcp[:3].tolist(),
            "stalled": stalled}


def _hold(robot, Rdes, grip, steps, target=None):
    f = robot.now()
    tgt = np.asarray(f.tcp[:3], dtype=np.float64) if target is None else np.asarray(target)
    for _ in range(steps):
        a = _cmd(f.tcp, tgt, Rdes, 0.3)
        f = robot.step(a[0], a[1], a[2], a[3], a[4], a[5], grip)
    return f


def _Rdes(rpy, tilt=0.0):
    R = _euler_R(rpy[0], rpy[1], rpy[2])
    if tilt:
        R = _roty(tilt) @ R
    return R


# ---------------------------------------------------------------- policy
def perceive_decide(obs, goal, memory, last):
    """Stage machine: watch -> grasp -> trace circles (one act per circle) -> press."""
    mem = dict(memory)
    for k in ("stage", "circles", "cube_xy", "button_xy", "home_rpy", "circle_idx", "notes", "cue_info", "grasp_tries"):
        mem.setdefault(k, None)
    cur = obs.current
    stage = mem["stage"] or "start"
    notes = list(mem["notes"] or [])
    fingers = [float(v) for v in cur.fingers]

    if stage == "start":
        mem["home_rpy"] = [float(v) for v in cur.tcp[3:6]]
        mem["stage"] = "watching"
        mem["circle_idx"] = 0
        mem["grasp_tries"] = 0
        return {"memory": mem, "act": {"tool": "act_watch_cue", "args": {"max_steps": 700}}}

    if stage == "watching":
        info = _analyse_cue(list(obs.frames))
        circles = info["circles"]
        cube = _green_cube_front(cur)
        btn = _button_xy(cur)
        mem["circles"] = circles
        mem["cube_xy"] = None if cube is None else [float(cube[0]), float(cube[1])]
        mem["button_xy"] = None if btn is None else [float(btn[0]), float(btn[1])]
        mem["cue_info"] = {"red_on": info["red_on"], "green_on": info["green_on"], "n_frames": info["n_frames"],
                           "n_circles": len(circles)}
        notes.append("cue: %d circles, lamp red@%s green@%s" % (len(circles), info["red_on"], info["green_on"]))
        mem["notes"] = notes
        if _lamp(cur) != "green":
            mem["stage"] = "watching"
            return {"memory": mem, "act": {"tool": "act_watch_cue", "args": {"max_steps": 400}}}
        stage = "grasp"

    if stage == "grasp_check":
        holding = fingers[0] + fingers[1] > 0.02
        if holding:
            stage = "trace"
        else:
            notes.append("grasp failed fingers=%.4f" % (fingers[0] + fingers[1]))
            cube = _green_cube_front(cur)
            if cube is not None:
                mem["cube_xy"] = [float(cube[0]), float(cube[1])]
            stage = "grasp"

    if stage == "trace_check":
        holding = fingers[0] + fingers[1] > 0.02
        ok = last is not None and '"completed": true' in last
        if holding and ok:
            mem["circle_idx"] = int(mem["circle_idx"]) + 1
        elif not holding:
            notes.append("cube lost during circle %d" % int(mem["circle_idx"]))
            cube = _green_cube_front(cur)
            if cube is not None:
                mem["cube_xy"] = [float(cube[0]), float(cube[1])]
            mem["notes"] = notes
            stage = "grasp"
        stage = "trace" if stage != "grasp" else "grasp"

    if stage == "grasp":
        tries = int(mem["grasp_tries"] or 0)
        mem["grasp_tries"] = tries + 1
        mem["notes"] = notes
        cube = mem["cube_xy"]
        if cube is None:
            cube = [-0.27, 0.0]
        if tries >= 4:
            stage = "trace"
        else:
            mem["stage"] = "grasp_check"
            return {"memory": mem, "act": {"tool": "act_grasp", "args": {"x": cube[0], "y": cube[1], "home_rpy": mem["home_rpy"]}}}

    if stage == "trace":
        circles = mem["circles"] or []
        i = int(mem["circle_idx"] or 0)
        mem["notes"] = notes
        if i < len(circles):
            c = circles[i]
            mem["stage"] = "trace_check"
            return {"memory": mem, "act": {"tool": "act_trace_circle", "args": {
                "cx": c["cx"], "cy": c["cy"], "r": c["r"], "a0": c["a0"], "direction": c["dir"],
                "home_rpy": mem["home_rpy"]}}}
        stage = "press"

    if stage in ("press", "pressing"):
        btn = mem["button_xy"]
        if btn is None:
            b2 = _button_xy(cur)
            btn = [0.15, 0.15] if b2 is None else [float(b2[0]), float(b2[1])]
        mem["stage"] = "pressing"
        mem["notes"] = notes
        return {"memory": mem, "act": {"tool": "act_release_and_press", "args": {
            "bx": btn[0], "by": btn[1], "home_rpy": mem["home_rpy"], "release": stage == "press"}}}

    mem["stage"] = "pressing"
    return {"memory": mem, "act": {"tool": "act_release_and_press", "args": {
        "bx": 0.15, "by": 0.15, "home_rpy": mem["home_rpy"], "release": False}}}


def act_watch_cue(robot, max_steps=700):
    """Hold still (gripper open) while the red cube traces the cue; stop as soon as the
    lamp is green. Returns lamp onsets and step counts (frames are analysed by perceive)."""
    f = robot.now()
    red_on = None
    n = 0
    for n in range(max_steps):
        l = _lamp(f)
        if l == "red" and red_on is None:
            red_on = n
        if l == "green":
            return {"green": True, "steps": n, "red_on_step": red_on}
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"green": _lamp(robot.now()) == "green", "steps": n + 1, "red_on_step": red_on}


def act_grasp(robot, x, y, home_rpy):
    """Hover over the green cube, refine its xy with the wrist camera, descend, close
    the gripper and lift to the tracing height. Returns measured finger width."""
    Rd = _Rdes(home_rpy)
    out = {}
    out["hover"] = _move(robot, [x, y, HOVER_Z], Rd, 1.0, a_max=0.8, tol=0.008, max_steps=80, settle=0.004)
    refined = None
    for _ in range(2):
        w = _green_cube_wrist(robot.now())
        if w is not None and math.hypot(w[0] - x, w[1] - y) < 0.05:
            refined = w
            x, y = w[0], w[1]
            _move(robot, [x, y, 0.10], Rd, 1.0, a_max=0.6, tol=0.004, max_steps=40, settle=0.002)
    out["wrist_cube"] = refined
    out["pre"] = _move(robot, [x, y, 0.07], Rd, 1.0, a_max=0.5, tol=0.004, max_steps=40, settle=0.002)
    out["down"] = _move(robot, [x, y, GRASP_Z], Rd, 1.0, a_max=0.3, tol=0.003, max_steps=60, settle=0.0015)
    f = _hold(robot, Rd, -1.0, 12, target=[x, y, GRASP_Z])
    out["fingers_closed"] = [float(v) for v in f.fingers]
    out["lift"] = _move(robot, [x, y, TRACE_Z], Rd, -1.0, a_max=0.3, tol=0.004, max_steps=40, settle=0.002)
    f = robot.now()
    out["fingers"] = [float(v) for v in f.fingers]
    out["holding"] = bool(f.fingers[0] + f.fingers[1] > 0.02)
    return out


def act_trace_circle(robot, cx, cy, r, a0, direction, home_rpy):
    """Carry the held cube to the circle start (cx+r cos a0, cy+r sin a0) at tracing height
    and pursue the circle in the observed direction for one full turn. Reports measured
    radius statistics, swept angle and steps."""
    Rd = _Rdes(home_rpy)
    sx, sy = cx + r * math.cos(a0), cy + r * math.sin(a0)
    tr = _move(robot, [sx, sy, TRACE_Z], Rd, -1.0, a_max=0.5, tol=0.004, max_steps=80, settle=0.002)
    f = robot.now()
    th_prev = math.atan2(float(f.tcp[1]) - cy, float(f.tcp[0]) - cx)
    sweep = 0.0
    target_sweep = 2.0 * math.pi + math.radians(6)
    lead = math.radians(LEAD_DEG)
    radii = []
    n = 0
    lost = False
    for n in range(260):
        tcp = np.asarray(f.tcp, dtype=np.float64)
        th = math.atan2(tcp[1] - cy, tcp[0] - cx)
        d = (th - th_prev + math.pi) % (2 * math.pi) - math.pi
        sweep += d * direction
        th_prev = th
        radii.append(math.hypot(tcp[0] - cx, tcp[1] - cy))
        if sweep >= target_sweep:
            break
        tt = th + direction * lead
        tgt = [cx + r * math.cos(tt), cy + r * math.sin(tt), TRACE_Z]
        e = np.asarray(tgt) - tcp[:3]
        a = _cmd(tcp, tgt, Rd, 1.0)
        # constant-speed pursuit in xy; z handled separately
        exy = e[:2]
        m = float(np.linalg.norm(exy))
        axy = exy / max(m, 1e-9) * min(TRACE_AMAX, m / 0.1)
        az = max(-0.5, min(0.5, e[2] / 0.1))
        f = robot.step(float(axy[0]), float(axy[1]), float(az), a[3], a[4], a[5], -1.0)
        if f.fingers[0] + f.fingers[1] < 0.01:
            lost = True
            break
    rr = np.array(radii[5:]) if len(radii) > 8 else np.array(radii + [0.0])
    return {"completed": bool(sweep >= target_sweep and not lost), "sweep_deg": math.degrees(sweep),
            "steps": n, "transit": tr, "r_cmd": r, "r_mean": float(rr.mean()), "r_min": float(rr.min()),
            "r_max": float(rr.max()), "lost": lost}


def _reach_ok(robot, pos, R, log):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    try:
        d = robot.dry_run(float(pos[0]), float(pos[1]), float(pos[2]), roll, pitch, yaw)
    except Exception as ex:
        log.append(["err", str(ex)[:60]])
        return False
    jm = d.get("joint_margin")
    if isinstance(jm, (list, tuple)):
        jm = min(jm) if len(jm) else None
    ok = bool(d.get("reachable")) and (jm is None or jm > 0.05)
    log.append([round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3), ok,
                None if jm is None else round(float(jm), 3)])
    return ok


def _plan_press(robot, bx, by, home_rpy, tilts, log):
    """First tilt (about base y) whose press pose (z .065) and approach pose (z .11) above
    the button are kinematically reachable, plus a reachable staging point at z .17."""
    for tilt in tilts:
        R = _Rdes(home_rpy, tilt)
        tz = R[:, 2]
        px = bx - 0.008 - 0.02 * tz[0]
        py = by - 0.02 * tz[1]
        log.append(["tilt", tilt])
        if not _reach_ok(robot, [px, py, 0.065], R, log):
            continue
        if not _reach_ok(robot, [px, py, 0.11], R, log):
            continue
        stage = None
        for back in (0.0, 0.05, 0.10):
            if _reach_ok(robot, [px - back, py, 0.17], R, log):
                stage = [px - back, py, 0.17]
                break
        if stage is None:
            stage = [px - 0.12, py, 0.15]
        return tilt, px, py, stage
    tilt = -0.3
    R = _Rdes(home_rpy, tilt)
    px, py = bx - 0.008 - 0.02 * R[0, 2], by - 0.02 * R[1, 2]
    return tilt, px, py, [px - 0.12, py, 0.15]


def act_release_and_press(robot, bx, by, home_rpy, release=True):
    """Lift and open the gripper to drop the cube, close it, plan a reachable press
    orientation/staging point with dry_run, move above the button and press down
    (z .065/.05/.04 with re-lifts); replans with a larger tilt if the arm stalls."""
    Rd = _Rdes(home_rpy)
    out = {}
    if release:
        p = np.asarray(robot.now().tcp[:3], dtype=np.float64)
        out["lift"] = _move(robot, [p[0], p[1], LIFT_Z], Rd, -1.0, a_max=0.6, tol=0.01, max_steps=40, settle=0.005)
        _hold(robot, Rd, 1.0, 10)
        _hold(robot, Rd, -1.0, 10)
    tilts = [0.0, -0.15, -0.3, -0.45]
    attempts = []
    for attempt in range(3):
        log = []
        tilt, px, py, stage = _plan_press(robot, bx, by, home_rpy, tilts, log)
        rec = {"tilt": tilt, "press_xy": [px, py], "stage": stage, "checks": log[-12:]}
        Rp = _Rdes(home_rpy, tilt)
        rec["stage_mv"] = _move(robot, stage, Rp, -1.0, a_max=0.7, tol=0.01, max_steps=120, settle=0.004)
        rec["approach"] = _move(robot, [px, py, 0.11], Rp, -1.0, a_max=0.5, tol=0.006, max_steps=80, settle=0.003)
        stalls = int(rec["stage_mv"]["stalled"]) + int(rec["approach"]["stalled"])
        for zt in (0.065, 0.05, 0.04):
            m = _move(robot, [px, py, zt], Rp, -1.0, a_max=0.4, tol=0.004, max_steps=50, settle=0.0015)
            rec["press_%d" % int(zt * 1000)] = m
            stalls += int(m["stalled"])
            _hold(robot, Rp, -1.0, 5, target=[px, py, zt])
            _move(robot, [px, py, 0.12], Rp, -1.0, a_max=0.5, tol=0.01, max_steps=40, settle=0.004)
        rec["stalls"] = stalls
        attempts.append(rec)
        # still no success: use the next larger tilt
        tilts = [t for t in tilts if t < tilt] or [-0.45]
    out["attempts"] = attempts
    return out
