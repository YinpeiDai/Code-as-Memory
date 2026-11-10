"""TimedTransfer policy: detect lamp onset, pick the cube, carry it above the target disc,
hold, and lower it onto the target disc so that it arrives at a calibrated count after the
lamp turned green. All perception from the fixed front camera (depth + extrinsics) and the
wrist camera; timing from the observed frame step indices."""

# ---------------- constants (from demo analysis, see task_analysis.md) ----------------
LAMP_MIN_PIXELS = 20          # lime-green lamp pixels in the front image when lit (~180-240 seen)
ARRIVE_OFFSET = 4             # descent scheduled so the cube enters the target zone at count ~N (=500).
                              # Rollouts R1-R7: entry at counts 478/483/500/518 succeeded, 473 failed (early),
                              # hovering until count 526 failed (window closed) -> window ~[475,525].
DESCENT_STEPS = 8             # descent ramp length (steps) before arrival
HOVER_ABOVE_PLACE = 0.06      # TCP hover height above the placing height while waiting (m)
CARRY_Z = 0.15                # TCP height while carrying (m)
PREGRASP_Z = 0.13             # TCP height above the cube before descending (m)
GRASP_BELOW_TOP = 0.017       # TCP below the cube top face at grasp (m)
PLACE_CLEAR = 0.004           # TCP above the grasp height when the cube rests on the disc (m)
KP = 1.0                      # translation gain (fraction of error commanded as offset)
KR = 0.6                      # rotation gain
VMAX = 1.0                    # max normalised translation command (norm)

COLORS = ("red", "green", "blue", "yellow")


# ---------------- small math helpers ----------------
def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        # near pi: use diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return d * th
    return v / (2.0 * s) * th


# ---------------- perception ----------------
def _color_mask(im, name):
    im = im.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if name == "red":
        return (r > 170) & (g < 130) & (b < 130) & (np.abs(g - b) < 30) & (r - g > 70)
    if name == "blue":
        return (b > 120) & (b - r > 40) & (b - g > 40)
    if name == "green":
        return (g > 150) & (g - r > 60) & (g - b > 60) & (r > 50) & (b > 50)
    if name == "yellow":
        return (r > 170) & (g > 170) & (b < 90)
    if name == "lime":
        return (g > 150) & (r < 120) & (b < 80) & (g - r > 80)
    return np.zeros(r.shape, dtype=bool)


def _backproject(mask, depth, K, E):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs].astype(np.float64) / 1000.0
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    X = (xs - K[0, 2]) / K[0, 0] * z
    Y = (ys - K[1, 2]) / K[1, 1] * z
    Pc = np.stack([X, Y, z], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (Pc - t) @ R


def _lamp_pixels(frame):
    return int(_color_mask(frame.front_rgb, "lime").sum())


def _scene(frame, cube_color, target_color):
    """Front-camera estimate of cube top-face centre and target disc centre (world)."""
    K, E, im, d = frame.front_K, frame.front_extrinsic, frame.front_rgb, frame.front_depth
    out = {"ok": True}
    P = _backproject(_color_mask(im, cube_color), d, K, E)
    if len(P) >= 10:
        ztop = float(P[:, 2].max())
        top = P[P[:, 2] > ztop - 0.006]
        c = np.median(top, 0)
        out["cube"] = [float(c[0]), float(c[1]), ztop]
        out["cube_pixels"] = int(len(P))
    else:
        out["ok"] = False
        out["cube"] = None
    Q = _backproject(_color_mask(im, target_color), d, K, E)
    if len(Q) >= 30:
        c = np.median(Q, 0)
        out["target"] = [float(c[0]), float(c[1]), float(np.percentile(Q[:, 2], 90))]
        out["target_pixels"] = int(len(Q))
    else:
        out["ok"] = False
        out["target"] = None
    return out


def _wrist_cube(frame, cube_color):
    P = _backproject(_color_mask(frame.wrist_rgb, cube_color), frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    if len(P) < 50:
        return None
    ztop = float(P[:, 2].max())
    top = P[P[:, 2] > ztop - 0.006]
    if len(top) < 30:
        return None
    c = np.median(top, 0)
    return [float(c[0]), float(c[1]), ztop, int(len(top))]


def _parse_goal(goal):
    g = (goal or "").lower()
    words = g.replace(",", " ").replace(".", " ").split()
    count = 500
    for i, w in enumerate(words):
        if w == "step" and i + 1 < len(words) and words[i + 1].isdigit():
            count = int(words[i + 1])
    cube_color, target_color = "blue", "red"
    for i, w in enumerate(words):
        if w == "cube" and i > 0 and words[i - 1] in COLORS:
            cube_color = words[i - 1]
    # "... to the <color> disc"
    for i, w in enumerate(words):
        if w == "to" and i + 3 < len(words) and words[i + 1] == "the" and words[i + 2] in COLORS and words[i + 3] == "disc":
            target_color = words[i + 2]
    return count, cube_color, target_color


# ---------------- control helpers ----------------
def _cmd(tcp, target, rpy_target, kp=KP, vmax=VMAX):
    err = np.array(target, dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64)
    a = kp * err / 0.1
    n = float(np.linalg.norm(a))
    if n > vmax:
        a = a * (vmax / n)
    Rc = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy2R(rpy_target[0], rpy_target[1], rpy_target[2])
    w = _logR(Rt @ Rc.T)
    r = -KR * w / 0.1
    rn = float(np.linalg.norm(r))
    if rn > 1.0:
        r = r / rn
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])], float(np.linalg.norm(err))


def _move(robot, target, grip, rpy, tol=0.004, max_steps=80, settle=2, kp=KP, vmax=VMAX):
    """Closed-loop move to an absolute TCP target. Returns (steps, final error, settled)."""
    n_ok = 0
    steps = 0
    e = 1.0
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    for _ in range(max_steps):
        f = robot.now()
        c, e = _cmd(f.tcp, target, rpy, kp, vmax)
        robot.step(c[0], c[1], c[2], c[3], c[4], c[5], grip)
        steps += 1
        cur = np.asarray(robot.now().tcp[:3], dtype=np.float64)
        v = float(np.linalg.norm(cur - prev))
        prev = cur
        e = float(np.linalg.norm(np.array(target) - cur))
        if e < tol and v < 0.002:
            n_ok += 1
            if n_ok >= settle:
                return steps, e, True
        else:
            n_ok = 0
    return steps, e, False


def _rpy_target(cube_yaw=0.0):
    return (math.pi, 0.0, cube_yaw)


# ---------------- policy ----------------
def perceive_decide(obs, goal, memory, last):
    """Stage machine: pick -> carry -> wait (lamp count) -> place -> hold. Scans every frame for the lamp onset."""
    mem = dict(memory or {})
    for k in ("stage", "onset_step", "cube", "target", "grasp_z", "n_target", "log", "place_tries", "cube_color", "target_color"):
        mem.setdefault(k, None)
    if mem["stage"] is None:
        n, cc, tc = _parse_goal(goal)
        mem["n_target"] = n
        mem["cube_color"] = cc
        mem["target_color"] = tc
        mem["stage"] = "pick"
        mem["place_tries"] = 0
        mem["log"] = ""
        mem["onset_step"] = -1
    # lamp onset: first frame showing the lit lamp
    if mem["onset_step"] is None or mem["onset_step"] < 0:
        for f in obs.frames:
            if _lamp_pixels(f) >= LAMP_MIN_PIXELS:
                mem["onset_step"] = int(f.step)
                break
    cur = obs.current
    stage = mem["stage"]
    if mem["cube"] is None or mem["target"] is None:
        sc = _scene(cur, mem["cube_color"], mem["target_color"])
        if mem["cube"] is None and sc["cube"] is not None:
            mem["cube"] = sc["cube"]
        if mem["target"] is None and sc["target"] is not None:
            mem["target"] = sc["target"]
    if stage == "pick":
        if mem["cube"] is None:
            act = {"tool": "act_wait", "args": {"steps": 2, "grip": 1.0}}
        else:
            act = {"tool": "act_pick", "args": {"cx": mem["cube"][0], "cy": mem["cube"][1], "ztop": mem["cube"][2],
                                                "cube_color": mem["cube_color"]}}
            mem["stage"] = "check_grasp"
    elif stage == "check_grasp":
        res = last or ""
        if '"grasped": true' in res:
            gz = None
            i = res.find('"grasp_z": ')
            if i >= 0:
                gz = float(res[i + 11:].split(",")[0].split("}")[0])
            mem["grasp_z"] = gz
            mem["stage"] = "carry"
            stage = "carry"
        else:
            mem["log"] = (mem["log"] or "") + "|grasp failed, retry"
            mem["cube"] = None
            sc = _scene(cur, mem["cube_color"], mem["target_color"])
            mem["cube"] = sc["cube"]
            mem["stage"] = "pick"
            act = {"tool": "act_retreat", "args": {"z": CARRY_Z}}
    if mem["stage"] == "carry":
        gz = mem["grasp_z"] if mem["grasp_z"] is not None else 0.032
        place_z = gz + PLACE_CLEAR
        act = {"tool": "act_carry", "args": {"x": mem["target"][0], "y": mem["target"][1], "z": place_z + HOVER_ABOVE_PLACE}}
        mem["stage"] = "wait_lamp" if mem["onset_step"] < 0 else "place"
    elif stage in ("wait_lamp", "place", "hold"):
        gz = mem["grasp_z"] if mem["grasp_z"] is not None else 0.032
        place_z = gz + PLACE_CLEAR
        if mem["onset_step"] < 0:
            act = {"tool": "act_hold", "args": {"x": mem["target"][0], "y": mem["target"][1], "z": place_z + HOVER_ABOVE_PLACE,
                                                "until_step": int(cur.step) + 200, "stop_on_lamp": True}}
            mem["stage"] = "wait_lamp"
        elif stage in ("wait_lamp", "place") and float(np.sum(cur.fingers)) < 0.01:
            # cube lost (fingers closed on nothing): re-perceive and pick again
            mem["log"] = (mem["log"] or "") + "|cube lost before place at step %d" % int(cur.step)
            sc = _scene(cur, mem["cube_color"], mem["target_color"])
            mem["cube"] = sc["cube"]
            mem["stage"] = "pick"
            act = {"tool": "act_retreat", "args": {"z": CARRY_Z}}
        elif stage in ("wait_lamp", "place"):
            arrive = mem["onset_step"] + mem["n_target"] + ARRIVE_OFFSET
            act = {"tool": "act_place", "args": {"x": mem["target"][0], "y": mem["target"][1], "z_place": place_z,
                                                 "z_hover": place_z + HOVER_ABOVE_PLACE, "arrive_step": int(arrive),
                                                 "hold_until_step": int(arrive) + 60}}
            mem["stage"] = "hold"
            mem["place_tries"] = (mem["place_tries"] or 0) + 1
        else:
            # placed and held but no success yet: keep holding on the disc
            act = {"tool": "act_hold", "args": {"x": mem["target"][0], "y": mem["target"][1], "z": place_z,
                                                "until_step": int(cur.step) + 100, "stop_on_lamp": False}}
    return {"memory": mem, "act": act}


def act_wait(robot, steps=1, grip=1.0):
    """Send zero-motion commands for `steps` steps with the given gripper target."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
    return {"step": int(robot.now().step)}


def act_pick(robot, cx, cy, ztop, cube_color="blue"):
    """Open gripper, go above the cube, refine its position with the wrist camera, descend,
    close and lift to carry height. Returns measured grasp state."""
    rpy = _rpy_target(0.0)
    out = {"start_step": int(robot.now().step)}
    s1, e1, ok1 = _move(robot, [cx, cy, PREGRASP_Z], 1.0, rpy, tol=0.006, max_steps=60)
    w = _wrist_cube(robot.now(), cube_color)
    out["front_cube"] = [cx, cy, ztop]
    out["wrist_cube"] = w
    if w is not None and abs(w[0] - cx) < 0.04 and abs(w[1] - cy) < 0.04:
        cx, cy = w[0], w[1]
        ztop = 0.5 * (ztop + w[2])
    gz = max(ztop - GRASP_BELOW_TOP, 0.02)
    s2, e2, ok2 = _move(robot, [cx, cy, gz + 0.05], 1.0, rpy, tol=0.004, max_steps=40)
    s3, e3, ok3 = _move(robot, [cx, cy, gz], 1.0, rpy, tol=0.003, max_steps=40, kp=0.8, vmax=0.6)
    for _ in range(8):
        f = robot.now()
        c, _e = _cmd(f.tcp, [cx, cy, gz], rpy)
        robot.step(c[0], c[1], c[2], c[3], c[4], c[5], -1.0)
    f = robot.now()
    out["grasp_tcp"] = [float(v) for v in f.tcp]
    out["grasp_z"] = float(f.tcp[2])
    s4, e4, ok4 = _move(robot, [cx, cy, CARRY_Z], -1.0, rpy, tol=0.01, max_steps=40, settle=1)
    f = robot.now()
    fing = [float(v) for v in f.fingers]
    out["fingers"] = fing
    out["grasped"] = bool(sum(fing) > 0.01)
    out["steps"] = [s1, s2, s3, s4]
    out["errs"] = [round(e1, 4), round(e2, 4), round(e3, 4), round(e4, 4)]
    out["end_step"] = int(f.step)
    out["tcp"] = [float(v) for v in f.tcp]
    return out


def act_retreat(robot, z=0.15):
    """Open the gripper and move straight up to height z."""
    f = robot.now()
    s, e, ok = _move(robot, [float(f.tcp[0]), float(f.tcp[1]), z], 1.0, _rpy_target(0.0), tol=0.01, max_steps=40)
    return {"steps": s, "err": e, "step": int(robot.now().step)}


def act_carry(robot, x, y, z):
    """Carry the grasped cube (gripper closed) to the hover pose (x, y, z)."""
    rpy = _rpy_target(0.0)
    f = robot.now()
    p = [float(v) for v in f.tcp[:3]]
    # first make sure we are high enough, then translate
    if p[2] < z - 0.01:
        _move(robot, [p[0], p[1], z], -1.0, rpy, tol=0.01, max_steps=30, settle=1)
    s, e, ok = _move(robot, [x, y, z], -1.0, rpy, tol=0.003, max_steps=120)
    f = robot.now()
    return {"steps": s, "err": round(e, 4), "settled": ok, "step": int(f.step), "tcp": [float(v) for v in f.tcp],
            "fingers": [float(v) for v in f.fingers]}


def act_hold(robot, x, y, z, until_step, stop_on_lamp=False):
    """Hold the TCP at (x, y, z) with the gripper closed until frame step `until_step`
    (or until the lamp is first seen lit when stop_on_lamp)."""
    rpy = _rpy_target(0.0)
    onset = -1
    maxerr = 0.0
    while int(robot.now().step) < int(until_step):
        f = robot.now()
        c, e = _cmd(f.tcp, [x, y, z], rpy)
        maxerr = max(maxerr, e)
        robot.step(c[0], c[1], c[2], c[3], c[4], c[5], -1.0)
        if stop_on_lamp and _lamp_pixels(robot.now()) >= LAMP_MIN_PIXELS:
            onset = int(robot.now().step)
            break
    f = robot.now()
    return {"step": int(f.step), "lamp_onset": onset, "max_err": round(maxerr, 4), "tcp": [float(v) for v in f.tcp]}


def act_place(robot, x, y, z_place, z_hover, arrive_step, hold_until_step):
    """Hold above the target at z_hover until the descent start, then ramp the target down
    so the cube reaches z_place at frame step arrive_step; then hold there until hold_until_step.
    Returns the measured descent trace (step, tcp z)."""
    rpy = _rpy_target(0.0)
    start = int(arrive_step) - DESCENT_STEPS
    trace = []
    maxerr_hold = 0.0
    while int(robot.now().step) < start:
        f = robot.now()
        c, e = _cmd(f.tcp, [x, y, z_hover], rpy)
        maxerr_hold = max(maxerr_hold, e)
        robot.step(c[0], c[1], c[2], c[3], c[4], c[5], -1.0)
    while int(robot.now().step) < int(hold_until_step):
        f = robot.now()
        k = int(f.step) - start + 1
        frac = min(1.0, k / float(max(1, DESCENT_STEPS - 2)))
        zr = z_hover + (z_place - z_hover) * frac
        c, e = _cmd(f.tcp, [x, y, zr], rpy, kp=1.5, vmax=1.0)
        robot.step(c[0], c[1], c[2], c[3], c[4], c[5], -1.0)
        g = robot.now()
        if len(trace) < 40:
            trace.append([int(g.step), round(float(g.tcp[0]), 4), round(float(g.tcp[1]), 4), round(float(g.tcp[2]), 4)])
    f = robot.now()
    return {"step": int(f.step), "hold_err_max": round(maxerr_hold, 4), "trace": trace,
            "fingers": [float(v) for v in f.fingers]}
