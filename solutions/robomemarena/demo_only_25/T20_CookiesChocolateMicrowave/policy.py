"""T20 Cookies -> microwave, then chocolate -> same place (the microwave).

Perception (first frame, agentview RGB-D + calibration) locates the microwave
door handle and the two boxes; a waypoint P-controller (pos gain 6 clip 1,
rot gain 2 clip 0.5, fixed segment lengths) executes the task in phases.
"""

POS_GAIN = 6.0
ROT_GAIN = 2.0
ROT_CLIP = 0.5
# gripper pointing horizontally into +y (into the microwave) / pointing down
R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
# microwave-relative offsets (handle frame H = (x98(dark) - 0.124, y2(dark) - 0.0232))
H_FROM_X98 = -0.124
H_FROM_Y2 = -0.0232
NOMINAL_H = [0.0462, 0.4144]
NOMINAL_COOKIE = [0.1713, -0.0945]
NOMINAL_CHOC = [0.1734, -0.1935]
COOKIE_GRASP_OFF = [-0.0033, -0.0001]
CHOC_GRASP_OFF = [-0.0066, -0.0016]
COOKIE_Z = 0.413
CHOC_Z = 0.417
GRASP_MIN_GAP = 0.008


def _rv2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _mat2rv(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * w
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        ax = np.zeros(3)
        ax[i] = d[i]
        for j in range(3):
            if j != i:
                ax[j] = (R[i, j] + R[j, i]) / (4.0 * d[i])
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        return ax * th
    return w * (th / (2.0 * math.sin(th)))


def _world_points(frame):
    """Agentview depth -> world XYZ per pixel (H x W x 3) and valid mask."""
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    D = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    R = E[:, :3]
    t = E[:, 3]
    h, w = D.shape
    v, u = np.mgrid[0:h, 0:w]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * D, (v - K[1, 2]) / K[1, 1] * D, D], -1)
    W = (P - t) @ R
    return W, D > 0


def _detect_microwave(W, ok, rgb):
    dark = rgb.astype(int).max(-1) < 45
    m = ok & dark & (W[..., 2] > 0.45) & (W[..., 2] < 0.9) & (W[..., 1] > 0.15) & (W[..., 1] < 1.0) & (W[..., 0] > -0.4) & (W[..., 0] < 0.4)
    n = int(m.sum())
    if n < 300:
        return None, n
    P = W[m]
    x98 = float(np.percentile(P[:, 0], 98))
    y2 = float(np.percentile(P[:, 1], 2))
    return [x98 + H_FROM_X98, y2 + H_FROM_Y2], n


def _object_points(W, ok, rgb):
    m = ok & (W[..., 2] > 0.438) & (W[..., 2] < 0.5) & (W[..., 0] > 0.0) & (W[..., 0] < 0.35) & (W[..., 1] > -0.35) & (W[..., 1] < 0.1)
    return W[m], rgb[m].astype(float)


def _detect_objects(W, ok, rgb):
    """Two boxes lying on the table: returns (cookie_xy, choc_xy, info)."""
    P, C = _object_points(W, ok, rgb)
    info = {"n": int(len(P))}
    if len(P) < 20:
        return None, None, info
    order = np.argsort(P[:, 1])
    ys = P[order, 1]
    gaps = np.diff(ys)
    if len(gaps) == 0:
        return None, None, info
    gi = int(np.argmax(gaps))
    info["gap"] = float(gaps[gi])
    if gaps[gi] < 0.02:
        return None, None, info
    a = order[: gi + 1]
    b = order[gi + 1:]
    if len(a) < 10 or len(b) < 10:
        return None, None, info
    ba = float(C[a].sum(-1).mean())
    bb = float(C[b].sum(-1).mean())
    ca = P[a].mean(0)
    cb = P[b].mean(0)
    info["bright"] = [ba, bb]
    if ba > bb:
        ck, ch = ca, cb
    else:
        ck, ch = cb, ca
    return [float(ck[0]), float(ck[1])], [float(ch[0]), float(ch[1])], info


def _redetect_near(W, ok, rgb, prev, radius=0.06):
    P, C = _object_points(W, ok, rgb)
    if len(P) == 0 or prev is None:
        return None
    d = np.sqrt((P[:, 0] - prev[0]) ** 2 + (P[:, 1] - prev[1]) ** 2)
    sel = d < radius
    if int(sel.sum()) < 10:
        return None
    c = P[sel].mean(0)
    return [float(c[0]), float(c[1])]


def _clip(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def _seg_action(frame, seg, hold):
    """Action for one step of segment `seg` from the measured frame."""
    g = float(seg.get("grip", -1.0))
    if seg.get("zero"):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g]
    tcp = np.asarray(frame.tcp, dtype=float)
    p = tcp[:3]
    tgt = seg.get("pos")
    if tgt is None:
        tgt = hold
    tgt = np.asarray(tgt, dtype=float)
    dp = _clip(POS_GAIN * (tgt - p), -1.0, 1.0)
    ori = seg.get("ori")
    if ori is None:
        dr = np.zeros(3)
    else:
        RT = np.asarray(R_SIDE if ori == "side" else R_DOWN, dtype=float)
        Rc = _rv2mat(tcp[3:6])
        err = _mat2rv(RT @ Rc.T)
        dr = _clip(ROT_GAIN * err, -ROT_CLIP, ROT_CLIP)
    return [float(dp[0]), float(dp[1]), float(dp[2]), float(dr[0]), float(dr[1]), float(dr[2]), g]


def _seg(pos, ori, grip, n, zero=False):
    return {"pos": pos, "ori": ori, "grip": grip, "n": n, "zero": zero}


def _plan(stage, H, ck, ch):
    """Segments for each stage. H: handle xy; ck/ch: cookie/choc grasp xy."""
    hx, hy = H[0], H[1]
    push = [hx - 0.049, hy + 0.044, 0.602]
    if stage == "door":
        return [
            _seg([hx + 0.001, hy - 0.136, 0.582], "side", -1, 30),
            _seg([hx + 0.001, hy + 0.014, 0.582], "side", -1, 30),
            _seg(None, None, 1, 20, True),
            _seg([hx - 0.499, hy - 0.186, 0.582], "side", 1, 40),
            _seg(None, None, -1, 10, True),
        ]
    if stage == "grasp_cookie":
        return [
            _seg([ck[0], ck[1], COOKIE_Z + 0.27], "down", -1, 40),
            _seg([ck[0], ck[1], COOKIE_Z], "down", -1, 30),
            _seg(None, None, 1, 20, True),
        ]
    if stage == "place_cookie":
        return [
            _seg([ck[0], ck[1], COOKIE_Z + 0.15], "down", 1, 40),
            _seg([ck[0] - 0.151, ck[1] + 0.2, COOKIE_Z + 0.145], "down", 1, 40),
            _seg(None, "side", 1, 30),
            _seg(push, "side", 1, 75),
            _seg([hx + 0.002, hy + 0.094, 0.572], "side", 1, 25),
            _seg(None, None, -1, 20, True),
            _seg([hx + 0.001, hy - 0.106, 0.572], "side", -1, 30),
            _seg([hx + 0.001, hy - 0.106, 0.722], "side", -1, 30),
        ]
    if stage == "grasp_choc":
        return [
            _seg([ch[0], ch[1], CHOC_Z + 0.27], "down", -1, 40),
            _seg([ch[0], ch[1], CHOC_Z], "down", -1, 30),
            _seg(None, None, 1, 20, True),
        ]
    if stage == "place_choc":
        return [
            _seg([ch[0], ch[1], CHOC_Z + 0.15], "down", 1, 40),
            _seg([ch[0] - 0.201, ch[1] + 0.2, CHOC_Z + 0.145], "down", 1, 40),
            _seg([ch[0] - 0.201, ch[1] + 0.2, CHOC_Z + 0.145], "side", 1, 40),
            _seg(push, "side", 1, 75),
        ]
    if stage == "release_choc":
        return [
            _seg(None, None, -1, 20, True),
            _seg([hx + 0.001, hy - 0.106, 0.575], "side", -1, 30),
            _seg([hx + 0.001, hy - 0.106, 0.722], "side", -1, 30),
        ]
    if stage == "retreat_cookie" or stage == "retreat_choc":
        z = COOKIE_Z if stage == "retreat_cookie" else CHOC_Z
        obj = ck if stage == "retreat_cookie" else ch
        return [
            _seg(None, None, -1, 15, True),
            _seg([obj[0], obj[1], z + 0.27], "down", -1, 40),
            _seg([-0.06, 0.0, 0.67], "down", -1, 40),
            _seg(None, None, -1, 5, True),
        ]
    # idle: hold position, gripper open
    return [_seg(None, None, -1, 40, True)]


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def perceive_decide(obs, goal, memory, last):
    """Detect the microwave handle and boxes on the first call, then step a phase machine."""
    fr = obs.current
    stage = memory.get("stage")
    H = memory.get("handle")
    ck = memory.get("cookie")
    ch = memory.get("choc")
    retries = memory.get("retries")
    log = memory.get("log")
    if retries is None:
        retries = 0
    if log is None:
        log = ""
    if stage is None or H is None or ck is None or ch is None:
        W, ok = _world_points(fr)
        rgb = np.asarray(fr.agentview_rgb)
        H, nmw = _detect_microwave(W, ok, rgb)
        if H is None:
            H = list(NOMINAL_H)
            log += "microwave:nominal;"
        cd, kd, info = _detect_objects(W, ok, rgb)
        if cd is None:
            cd = list(NOMINAL_COOKIE)
            kd = list(NOMINAL_CHOC)
            log += "objects:nominal;"
        ck = [cd[0] + COOKIE_GRASP_OFF[0], cd[1] + COOKIE_GRASP_OFF[1]]
        ch = [kd[0] + CHOC_GRASP_OFF[0], kd[1] + CHOC_GRASP_OFF[1]]
        stage = "door"
        log += "init;"
    gap = _gap(fr)
    nxt = stage
    if stage == "door":
        run, nxt = "door", "grasp_cookie"
    elif stage == "grasp_cookie":
        run, nxt = "grasp_cookie", "check_cookie"
    elif stage == "check_cookie":
        if gap > GRASP_MIN_GAP or retries >= 2:
            run, nxt = "place_cookie", "grasp_choc"
        else:
            run, nxt = "retreat_cookie", "redetect_cookie"
            retries += 1
            log += "cookie_miss;"
    elif stage == "redetect_cookie":
        W, ok = _world_points(fr)
        d = _redetect_near(W, ok, np.asarray(fr.agentview_rgb), [ck[0] - COOKIE_GRASP_OFF[0], ck[1] - COOKIE_GRASP_OFF[1]])
        if d is not None:
            ck = [d[0] + COOKIE_GRASP_OFF[0], d[1] + COOKIE_GRASP_OFF[1]]
        run, nxt = "grasp_cookie", "check_cookie"
    elif stage == "grasp_choc":
        run, nxt = "grasp_choc", "check_choc"
    elif stage == "check_choc":
        if gap > GRASP_MIN_GAP or retries >= 4:
            run, nxt = "place_choc", "release_choc"
        else:
            run, nxt = "retreat_choc", "redetect_choc"
            retries += 1
            log += "choc_miss;"
    elif stage == "redetect_choc":
        W, ok = _world_points(fr)
        d = _redetect_near(W, ok, np.asarray(fr.agentview_rgb), [ch[0] - CHOC_GRASP_OFF[0], ch[1] - CHOC_GRASP_OFF[1]])
        if d is not None:
            ch = [d[0] + CHOC_GRASP_OFF[0], d[1] + CHOC_GRASP_OFF[1]]
        run, nxt = "grasp_choc", "check_choc"
    elif stage == "release_choc":
        run, nxt = "release_choc", "idle"
    else:
        run, nxt = "idle", "idle"
    segs = _plan(run, H, ck, ch)
    if len(log) > 400:
        log = log[-400:]
    mem = {"stage": nxt, "handle": [float(H[0]), float(H[1])],
           "cookie": [float(ck[0]), float(ck[1])], "choc": [float(ch[0]), float(ch[1])],
           "retries": int(retries), "log": log + run + ";"}
    return {"memory": mem, "act": {"tool": "act_segments", "args": {"segments": segs, "label": run}}}


def act_segments(robot, segments, label=""):
    """Run waypoint segments with a P-controller (pos gain 6, rot gain 2).

    Each segment: {"pos": [x,y,z] or None (hold pose at segment start), "ori": "side"/"down"/None,
    "grip": +1 close / -1 open, "n": steps, "zero": true for pure gripper waiting}.
    Returns the final tcp, finger gap and step count."""
    steps = 0
    for seg in segments:
        start = robot.now()
        hold = [float(x) for x in np.asarray(start.tcp, dtype=float)[:3]]
        n = int(seg.get("n", 0))
        for _ in range(n):
            a = _seg_action(robot.now(), seg, hold)
            robot.step(*a)
            steps += 1
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"label": label, "steps": steps, "tcp": [round(float(v), 4) for v in tcp[:3]],
            "finger_gap": round(_gap(fr), 4)}
