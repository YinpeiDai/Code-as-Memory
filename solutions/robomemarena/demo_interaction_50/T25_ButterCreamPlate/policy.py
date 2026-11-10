"""T25_ButterCreamPlate policy: pick objects in goal order from the source plate and place them on the target plate.

Perception: agentview RGB-D back-projected to world points (coarse object/plate localisation), wrist RGB-D at a
hover pose (fine object localisation). Control: closed-loop OSC delta servo on measured TCP pose.
"""

TABLE_Z = 0.430          # observed table height in demos (depth back-projection)
HOVER_Z = 0.70           # transport / hover height (demo expert used ~0.69-0.71)
RELEASE_Z = 0.525        # demo expert released at tcp z ~0.524 above plate2
GRASP_YAW = -0.785398    # demo expert grasped every object at yaw -45 deg
HOLD_WIDTH = 0.008       # finger width (m) above which a closed gripper is holding something
OBJECTS = ("butter", "cream")
STRETCH_Q4 = -0.35       # q4 above this = elbow flipped past max reach (R1/R13 stalled at the -0.07 limit)
AMIN = 0.12              # minimum normalised speed while outside tolerance (avoids slow crawl)
BASE_XY = (-0.51, 0.0)   # robot base xy (from forward kinematics vs measured TCP in R1)
SHOULDER = (-0.51, 0.0, 0.75)        # joint-2 origin in world (FK)
WRIST_OFF = (-0.084, 0.014, 0.211)   # wrist centre minus TCP for a downward gripper (FK fit, demos, std 3 mm)
D_SAFE = 0.700           # shoulder-wrist distance kept for free-space moves (elbow q4 ~ -0.95)
D_MAX = 0.719            # kinematic maximum (q4 ~ -0.5); beyond it the elbow flips toward its straight limit
LOOK_BACK = 0.05         # look from this far toward the base: wrist camera sits ~0.055 m ahead of the TCP
LOOK_ZMIN = 0.56


def _R_aa(aa):
    aa = np.asarray(aa, dtype=float)
    ang = float(np.linalg.norm(aa))
    if ang < 1e-9:
        return np.eye(3)
    ax = aa / ang
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _aa_R(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    if ang > math.pi - 1e-3:
        # near pi: axis from the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        ax = R[:, i] + np.eye(3)[:, i]
        ax = ax / max(np.linalg.norm(ax), 1e-9)
        return ax * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return w * ang


def _R_down(yaw):
    """Gripper pointing straight down, rotated by yaw about world z (yaw 0 = the reset orientation)."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


def _yaw_of(aa):
    R = _R_aa(aa)
    return math.atan2(R[1, 0], R[0, 0])


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _cloud(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    Pc = np.stack([x, y, z], -1)
    P = (Pc - E[:, 3]) @ E[:, :3]
    return P, depth > 0


def _masks(rgb, P, valid):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    tab = valid & (P[..., 0] > -0.15) & (P[..., 0] < 0.45) & (np.abs(P[..., 1]) < 0.45) \
        & (P[..., 2] > 0.40) & (P[..., 2] < 0.56)
    obj = tab & (P[..., 2] > TABLE_Z + 0.007)
    butter = obj & (r - b > 50) & (g - b > 10) & (r > 80)
    cream = obj & (b - r > 15) & (b > 60)
    white = tab & (P[..., 2] < TABLE_Z + 0.015) & (r > 170) & (g > 170) & (b > 165) & (np.abs(r - b) < 25)
    pink = (r - g > 25) & (r - b > 25) & (np.abs(g - b) < 20)
    return {"butter": butter, "cream": cream, "white": white, "pink": pink, "tab": tab}


def _plates(W):
    """Split white plate points (N x 3) into up to two clusters; returns list of (center_xy, count)."""
    if len(W) < 20:
        return []
    xy = W[:, :2]
    c = [xy[np.argmin(xy[:, 1])].copy(), xy[np.argmax(xy[:, 1])].copy()]
    for _ in range(8):
        d0 = np.hypot(xy[:, 0] - c[0][0], xy[:, 1] - c[0][1])
        d1 = np.hypot(xy[:, 0] - c[1][0], xy[:, 1] - c[1][1])
        lab = d1 < d0
        for k in (0, 1):
            sel = xy[lab == bool(k)]
            if len(sel):
                c[k] = np.median(sel, 0)
    out = []
    if np.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]) < 0.06:
        out.append((np.median(xy, 0), len(xy)))
        return out
    d0 = np.hypot(xy[:, 0] - c[0][0], xy[:, 1] - c[0][1])
    d1 = np.hypot(xy[:, 0] - c[1][0], xy[:, 1] - c[1][1])
    lab = d1 < d0
    for k in (0, 1):
        sel = xy[lab == bool(k)]
        sel = sel[np.hypot(sel[:, 0] - c[k][0], sel[:, 1] - c[k][1]) < 0.06]
        if len(sel) >= 15:
            out.append((sel.mean(0), len(sel)))
    return out


def _scene(frame):
    """Agentview detections: object centroids (xy, pixel count) and plate clusters."""
    P, valid = _cloud(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    M = _masks(frame.agentview_rgb, P, valid)
    det = {}
    for name in OBJECTS:
        m = M[name]
        n = int(m.sum())
        if n >= 6:
            Q = P[m][:, :2]
            c = np.median(Q, 0)
            Q = Q[np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]) < 0.05]
            det[name] = [float(Q[:, 0].mean()), float(Q[:, 1].mean()), n]
        else:
            det[name] = None
    det["plates"] = [[float(c[0]), float(c[1]), int(n)] for c, n in _plates(P[M["white"]])]
    return det


def _wrist_locate(frame, name, est, other, rad=0.07):
    """Locate an object in the wrist RGB-D: colour seed, then bounding centre of raised non-plate points."""
    P, valid = _cloud(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    M = _masks(frame.wrist_rgb, P, valid)
    m = M[name] & (np.hypot(P[..., 0] - est[0], P[..., 1] - est[1]) < rad)
    if int(m.sum()) < 12:
        return None
    c = np.median(P[m][:, :2], 0)
    near = valid & (np.hypot(P[..., 0] - c[0], P[..., 1] - c[1]) < 0.12) & (P[..., 2] > 0.38) & (P[..., 2] < 0.6)
    if int(near.sum()) < 30:
        return None
    tz = float(np.percentile(P[near][:, 2], 20))
    high = near & (P[..., 2] > tz + 0.007) & ~M["pink"] & ~M["white"]
    for o in OBJECTS:
        if o != name:
            high = high & ~M[o]
    H = P[high]
    if other is not None:
        # drop raised points closer to the other object's colour centre
        mo = None
        for o in OBJECTS:
            if o != name:
                mo = M[o] & (np.hypot(P[..., 0] - other[0], P[..., 1] - other[1]) < rad)
        if mo is not None and int(mo.sum()) >= 12:
            oc = np.median(P[mo][:, :2], 0)
            keep = np.hypot(H[:, 0] - c[0], H[:, 1] - c[1]) <= np.hypot(H[:, 0] - oc[0], H[:, 1] - oc[1])
            H = H[keep]
    cc = c.copy()
    top = tz + 0.02
    sel = H[:0]
    for _ in range(3):
        sel = H[np.hypot(H[:, 0] - cc[0], H[:, 1] - cc[1]) < 0.035]
        if len(sel) < 12:
            break
        cc = (np.percentile(sel[:, :2], 5, 0) + np.percentile(sel[:, :2], 95, 0)) / 2
        top = float(np.percentile(sel[:, 2], 90))
    short = None
    dims = None
    S = H[np.hypot(H[:, 0] - cc[0], H[:, 1] - cc[1]) < 0.045][:, :2] if len(H) else H
    if len(S) >= 30:
        Q = S - cc
        w, v = np.linalg.eigh(np.cov(Q.T))
        pl = Q @ v[:, 1]
        ps = Q @ v[:, 0]
        L = float(np.percentile(pl, 97) - np.percentile(pl, 3))
        Sw = float(np.percentile(ps, 97) - np.percentile(ps, 3))
        dims = [L, Sw]
        short = float(math.atan2(v[1, 0], v[0, 0]))
    return {"xy": [float(cc[0]), float(cc[1])], "color_xy": [float(c[0]), float(c[1])],
            "n": int(m.sum()), "table_z": tz, "top": top, "short_dir": short, "dims": dims}


def _wrap_half(a):
    """Wrap an angle to (-pi/2, pi/2] (parallel-jaw grasps are symmetric under pi)."""
    while a > math.pi / 2:
        a -= math.pi
    while a <= -math.pi / 2:
        a += math.pi
    return a


def _grasp_yaw(loc, other, mode):
    """Gripper yaw for a grasp. Fingers close along world (-sin yaw, cos yaw).
    mode 'auto': close across the box's short top-face axis (from wrist PCA) when the top face is clearly
    elongated and the other object is not in the finger sweep; otherwise the demonstrated -45 deg."""
    if mode != "auto":
        return float(mode)
    if loc is None or loc.get("short_dir") is None or loc.get("dims") is None:
        return GRASP_YAW
    L, Sw = loc["dims"]
    if L < 1.3 * Sw or Sw > 0.06:
        return GRASP_YAW
    s = loc["short_dir"]
    yaw = _wrap_half(math.atan2(-math.cos(s), math.sin(s)))
    if other is not None:
        c = np.array([-math.sin(yaw), math.cos(yaw)])
        d = np.array([other[0] - loc["xy"][0], other[1] - loc["xy"][1]])
        along = abs(float(d @ c))
        perp = abs(float(d[0] * c[1] - d[1] * c[0]))
        if along < 0.075 and perp < 0.045:
            return GRASP_YAW
    return yaw


def _reach(p):
    """Shoulder-to-wrist distance needed for a downward TCP at p (elbow straightens as it nears D_MAX)."""
    return math.sqrt((p[0] + WRIST_OFF[0] - SHOULDER[0]) ** 2 + (p[1] + WRIST_OFF[1] - SHOULDER[1]) ** 2
                     + (p[2] + WRIST_OFF[2] - SHOULDER[2]) ** 2)


def _safe_point(x, y, zmax=None, zmin=LOOK_ZMIN, dmax=D_SAFE):
    """Highest point at/below zmax (>= zmin) over (x, y) within reach dmax; pulls xy toward the base if needed."""
    if zmax is None:
        zmax = HOVER_Z
    z = zmax
    while z > zmin and _reach([x, y, z]) > dmax:
        z -= 0.01
    z = max(z, zmin)
    k = 0
    while _reach([x, y, z]) > dmax and k < 40:
        d = np.array([BASE_XY[0] - x, BASE_XY[1] - y])
        d = d / max(np.linalg.norm(d), 1e-6)
        x, y = x + 0.005 * d[0], y + 0.005 * d[1]
        k += 1
    return [float(x), float(y), float(z)]


def _servo(robot, pos, yaw, grip, tol=0.005, max_steps=150, gain=5.0, vmax=1.0, rtol=0.05, recover=1):
    """Closed-loop move of the measured TCP to pos (xyz) with the gripper pointing down at yaw.
    Gentle exponential approach (demo-like gain). If the elbow (joint 4) is found past the max-reach point
    (q4 > STRETCH_Q4, the flipped branch that stalled R1/R13), run a bounded unflip manoeuvre first."""
    Rt = _R_down(yaw)
    n = 0
    rec = 0
    fr = robot.now()
    while n < max_steps:
        fr = robot.now()
        p = np.asarray(fr.tcp[:3], dtype=float)
        e = np.asarray(pos, dtype=float) - p
        er = _aa_R(Rt @ _R_aa(fr.tcp[3:]).T)
        if np.linalg.norm(e) < tol and np.linalg.norm(er) < rtol:
            break
        if float(fr.joints[3]) > STRETCH_Q4 and rec < recover:
            rec += 1
            n += _unflip(robot, yaw, grip)
            continue
        a = e * gain
        na = float(np.linalg.norm(a))
        if 1e-9 < na < AMIN and np.linalg.norm(e) >= tol:
            a = a * (AMIN / na)
        a = np.clip(a, -vmax, vmax)
        ar = np.clip(er * 1.0, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), grip)
        n += 1
    fr = robot.now()
    return {"steps": n, "recoveries": rec, "q4": float(fr.joints[3]),
            "err": float(np.linalg.norm(np.asarray(pos) - np.asarray(fr.tcp[:3])))}


def _raw_move(robot, tgt, yaw, grip, steps, gain=4.0, vmax=0.5):
    Rt = _R_down(yaw)
    k = 0
    while k < steps:
        fr = robot.now()
        e = np.asarray(tgt, dtype=float) - np.asarray(fr.tcp[:3], dtype=float)
        if np.linalg.norm(e) < 0.006:
            break
        er = _aa_R(Rt @ _R_aa(fr.tcp[3:]).T)
        a = np.clip(e * gain, -vmax, vmax)
        ar = np.clip(er, -0.3, 0.3)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), grip)
        k += 1
    return k


def _unflip(robot, yaw, grip):
    """Exploratory recovery for a flipped (over-straight) elbow: push the wrist radially out to the max-reach
    point at shoulder height (elbow returns to q4 ~ -0.5), then pull slowly in to a well-bent pose."""
    fr = robot.now()
    p = np.asarray(fr.tcp[:3], dtype=float)
    z = max(SHOULDER[2] - WRIST_OFF[2], 0.56)
    d = np.array([p[0] - BASE_XY[0], p[1] - BASE_XY[1]])
    d = d / max(np.linalg.norm(d), 1e-6)
    k = 0
    # outward: target reach slightly above the maximum
    r_now = math.hypot(p[0] + WRIST_OFF[0] - SHOULDER[0], p[1] + WRIST_OFF[1] - SHOULDER[1])
    out = [p[0] + (0.735 - r_now) * d[0], p[1] + (0.735 - r_now) * d[1], z]
    k += _raw_move(robot, out, yaw, grip, 35)
    inn = _safe_point(p[0] - 0.15 * d[0], p[1] - 0.15 * d[1], zmax=0.62, dmax=0.62)
    k += _raw_move(robot, inn, yaw, grip, 60, vmax=0.4)
    return k


def _hold(robot, grip, steps):
    for _ in range(steps):
        robot.step(0, 0, 0, 0, 0, 0, grip)


def perceive_decide(obs, goal, memory, last):
    """Plan the next action from agentview detections, finger width, declared memory and the last outcome."""
    mem = dict(memory)
    fr = obs.current
    # ---- goal parsing: object order by first mention in the instruction
    if not mem.get("order"):
        g = (goal or "").lower()
        found = sorted([(g.find(o), o) for o in OBJECTS if g.find(o) >= 0])
        mem["order"] = [o for _, o in found] if found else list(OBJECTS)
    placed = list(mem.get("placed") or [])
    attempts = int(mem.get("attempts") or 0)
    fails = int(mem.get("fails") or 0)
    waits = int(mem.get("waits") or 0)
    det = _scene(fr)
    for o in OBJECTS:
        if det.get(o) is not None:
            mem[o + "_pos"] = [det[o][0], det[o][1]]
        elif mem.get(o + "_pos") is None:
            mem[o + "_pos"] = None
    # ---- target plate: the plate away from where the objects started (source plate = the one next to them)
    if mem.get("plate2") is None:
        objs = [mem[o + "_pos"] for o in OBJECTS if mem.get(o + "_pos") is not None]
        plates = det["plates"]
        if objs and plates:
            oc = np.mean(np.array(objs), 0)
            best = max(plates, key=lambda p: math.hypot(p[0] - oc[0], p[1] - oc[1]))
            if math.hypot(best[0] - oc[0], best[1] - oc[1]) > 0.12:
                mem["plate2"] = [best[0], best[1]]
            near = min(plates, key=lambda p: math.hypot(p[0] - oc[0], p[1] - oc[1]))
            if math.hypot(near[0] - oc[0], near[1] - oc[1]) < 0.12:
                mem["plate1"] = [near[0], near[1]]
    p2 = mem.get("plate2")
    width = _width(fr)
    stage = mem.get("stage")
    cur = mem.get("current")
    holding = mem.get("holding")
    note = mem.get("note")
    # ---- interpret the outcome of the previous action from measurements
    if stage == "pick":
        if HOLD_WIDTH < width < 0.075:
            holding = cur
            note = "holding %s, width %.4f" % (cur, width)
        else:
            holding = None
            fails += 1
            note = "grasp of %s failed, width %.4f" % (cur, width)
    elif stage == "place":
        holding = None
        d = det.get(cur) if cur is not None else None
        if p2 is not None and d is not None and math.hypot(d[0] - p2[0], d[1] - p2[1]) < 0.07:
            if cur not in placed:
                placed.append(cur)
            fails = 0
            note = "%s seen on plate2" % cur
        elif d is None:
            if cur not in placed:
                placed.append(cur)
            fails = 0
            note = "%s not visible after release over plate2; assumed placed (uncertain)" % cur
        else:
            note = "%s seen off plate2 at %.3f,%.3f; retry" % (cur, d[0], d[1])
    elif holding is not None and width < HOLD_WIDTH:
        note = "lost %s, width %.4f" % (holding, width)
        holding = None
    # a placed object seen far from plate2 again is not placed
    if p2 is not None:
        for o in list(placed):
            d = det.get(o)
            if d is not None and d[2] >= 15 and math.hypot(d[0] - p2[0], d[1] - p2[1]) > 0.12:
                placed.remove(o)
                note = "%s left plate2; redo" % o
    todo = [o for o in mem["order"] if o not in placed]
    if p2 is None:
        stage = "search"
        attempts += 1
        act = {"tool": "act_home", "args": {}}
    elif holding is not None:
        stage = "place"
        cur = holding
        act = {"tool": "act_place", "args": {"x": p2[0], "y": p2[1]}}
    elif not todo:
        stage = "monitor"
        waits += 1
        act = {"tool": "act_observe", "args": {"steps": 40}}
    else:
        o = todo[0]
        cur = o
        est = mem.get(o + "_pos")
        if est is None:
            src = mem.get("plate1") or [0.2, -0.19]
            est = [src[0], src[1]]
        other = None
        for q in OBJECTS:
            if q != o and q not in placed and mem.get(q + "_pos") is not None:
                other = mem[q + "_pos"]
        attempts += 1
        yaws = ["auto", GRASP_YAW, GRASP_YAW + math.pi / 2]
        stage = "pick"
        act = {"tool": "act_pick", "args": {"name": o, "x": est[0], "y": est[1],
                                            "other": other, "yaw": yaws[fails % 3]}}
    mem.update({"placed": placed, "attempts": attempts, "fails": fails, "waits": waits, "width": width,
                "stage": stage, "current": cur, "holding": holding, "note": note})
    return {"memory": _mem_out(mem), "act": act}


MEM_FIELDS = ("order", "placed", "holding", "current", "stage", "plate1", "plate2", "butter_pos", "cream_pos",
              "attempts", "fails", "waits", "width", "note")


def _mem_out(mem):
    return {k: mem.get(k) for k in MEM_FIELDS}


def act_home(robot):
    """Move to a well-bent pose near the reset position so the agentview sees the table; returns the pose."""
    fr = robot.now()
    p = np.asarray(fr.tcp[:3], dtype=float)
    up = _safe_point(p[0], p[1], zmax=max(p[2], 0.60))
    _servo(robot, up, _yaw_of(fr.tcp[3:]), -1, max_steps=60, tol=0.02)
    r = _servo(robot, [-0.05, 0.0, HOVER_Z], 0.0, -1, max_steps=150, tol=0.02)
    _hold(robot, -1, 5)
    return {"tcp": [float(v) for v in robot.now().tcp[:3]], "servo": r}


def act_observe(robot, steps=10):
    """Hold still (gripper state kept) for a number of steps and report the finger width."""
    _hold(robot, 0, int(steps))
    return {"width": _width(robot.now()), "q4": float(robot.now().joints[3])}


def act_pick(robot, name, x, y, other=None, yaw="auto"):
    """Look at (x, y) with the wrist camera from a reachable pose, refine the object position, choose the
    grasp yaw ('auto' = across the short top-face axis, else rad), descend, close, lift to a reachable carry pose.
    Returns whether the fingers stayed apart (holding), widths, refined position, yaw and elbow state."""
    out = {"name": name}
    fr = robot.now()
    p = np.asarray(fr.tcp[:3], dtype=float)
    look_yaw = GRASP_YAW if yaw == "auto" else float(yaw)
    if p[2] < 0.55:
        _servo(robot, _safe_point(p[0], p[1], zmax=0.60), _yaw_of(fr.tcp[3:]), -1, max_steps=80, tol=0.02)
    look = _safe_point(x - LOOK_BACK, y, zmax=HOVER_Z)
    _servo(robot, look, look_yaw, -1, max_steps=220, tol=0.008)
    _hold(robot, -1, 3)
    est = [x, y]
    loc = _wrist_locate(robot.now(), name, est, other)
    if loc is None:
        loc = _wrist_locate(robot.now(), name, est, other, rad=0.15)
    out["wrist"] = loc
    out["look"] = look
    gyaw = _grasp_yaw(loc, other, yaw)
    if loc is not None:
        est = loc["xy"]
        # second, closer look already at the grasp yaw
        look2 = _safe_point(est[0] - LOOK_BACK, est[1], zmax=0.60)
        _servo(robot, look2, gyaw, -1, max_steps=100, tol=0.006)
        _hold(robot, -1, 3)
        loc2 = _wrist_locate(robot.now(), name, est, other, rad=0.05)
        if loc2 is not None and math.hypot(loc2["xy"][0] - est[0], loc2["xy"][1] - est[1]) < 0.03:
            est = [(est[0] + loc2["xy"][0]) / 2, (est[1] + loc2["xy"][1]) / 2]
            loc = loc2
        out["wrist2"] = loc2
        top = loc["top"]
        tz = loc["table_z"]
    else:
        top = TABLE_Z + 0.03
        tz = TABLE_Z
    zg = min(max(top - 0.018, tz + 0.007), tz + 0.02)
    out["xy"] = est
    out["yaw"] = gyaw
    out["z_grasp"] = zg
    out["reach_grasp"] = _reach([est[0], est[1], zg])
    _servo(robot, [est[0], est[1], zg + 0.05], gyaw, -1, max_steps=130, tol=0.006)
    _servo(robot, [est[0], est[1], zg], gyaw, -1, max_steps=80, tol=0.004, vmax=0.5)
    _hold(robot, -1, 2)
    out["tcp_grasp"] = [float(v) for v in robot.now().tcp[:3]]
    out["q4_grasp"] = float(robot.now().joints[3])
    _hold(robot, 1, 18)
    out["width_closed"] = _width(robot.now())
    _servo(robot, [est[0], est[1], zg + 0.07], gyaw, 1, max_steps=40, tol=0.01)
    _servo(robot, _safe_point(est[0], est[1]), gyaw, 1, max_steps=120, tol=0.015)
    _hold(robot, 1, 3)
    w = _width(robot.now())
    out["width_lifted"] = w
    out["holding"] = bool(w > HOLD_WIDTH)
    out["q4"] = float(robot.now().joints[3])
    return out


def act_place(robot, x, y):
    """Carry the held object at a reachable height to above (x, y), lower to the release height, open, lift."""
    fr = robot.now()
    yaw = _yaw_of(fr.tcp[3:])
    p = np.asarray(fr.tcp[:3], dtype=float)
    if p[2] < 0.55:
        _servo(robot, _safe_point(p[0], p[1], zmax=0.62), yaw, 1, max_steps=80, tol=0.02)
    _servo(robot, _safe_point(x, y), yaw, 1, max_steps=220, tol=0.01)
    _servo(robot, [x, y, RELEASE_Z], yaw, 1, max_steps=120, tol=0.008)
    w_before = _width(robot.now())
    _hold(robot, -1, 15)
    _servo(robot, _safe_point(x, y), yaw, -1, max_steps=120, tol=0.015)
    return {"released_at": [float(v) for v in robot.now().tcp[:3]], "width_before_release": w_before,
            "held_until_release": bool(w_before > HOLD_WIDTH), "q4": float(robot.now().joints[3])}
