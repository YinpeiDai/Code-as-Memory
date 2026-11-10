"""TraceShapeEasy policy: watch the red cube trace a path during the cue, then
pick the green cube and drag it (lifted) along the same world-frame path, in the
same order and direction, starting at the red cube's start point."""

CARRY_Z = 0.058        # TCP height while carrying (expert demos: 0.059-0.060)
HOVER_Z = 0.10         # TCP height above the green cube before descending
GRASP_DZ = 0.0075      # TCP below the cube top face when closing (demos: 0.0074)
STEP_GAIN = 0.039      # metres of steady TCP motion per step per unit action (fit on demos)
CUBE_HALF = 0.02
ROT_GAIN = 0.5         # fraction of the orientation error commanded per step (world rotvec)
DOWN_RPY = (math.pi, 0.0, 0.0)   # hold the gripper pointing straight down, yaw 0


# ---------------------------------------------------------------- perception
def _masks(rgb):
    w = rgb.astype(np.int32)
    r, g, b = w[..., 0], w[..., 1], w[..., 2]
    red = (r * 10 > 16 * g) & (np.abs(g - b) < 10) & (g > 40)
    grn = (g * 10 > 16 * r) & (np.abs(r - b) < 10) & (r > 40)
    lamp_r = (r > 200) & (g < 40) & (b < 40)
    lamp_g = (g > 200) & (r < 40) & (b < 40)
    return red, grn, lamp_r, lamp_g


def _world_points(depth, K, E, m):
    ys, xs = np.nonzero(m & (depth > 0))
    if len(ys) == 0:
        return None
    z = depth[ys, xs].astype(np.float64) / 1000.0
    x = (xs + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (ys + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t) @ R


def _top_center(pw, min_pts=8):
    """Centroid of the cube's top face (points within 4 mm of the highest)."""
    if pw is None or len(pw) < min_pts:
        return None
    top = pw[pw[:, 2] > pw[:, 2].max() - 0.004]
    if len(top) < 4:
        return None
    return top.mean(0)


def _frame_objects(frame):
    """Red/green cube top-face centres (world) fused over both cameras, lamp state."""
    out = {"red": None, "green": None, "lamp": "-"}
    reds, grns = [], []
    nlr, nlg = 0, 0
    for cam in ("front", "wrist"):
        rgb = getattr(frame, cam + "_rgb")
        dep = getattr(frame, cam + "_depth")
        K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
        E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
        red, grn, lr, lg = _masks(rgb)
        nlr += int(lr.sum())
        nlg += int(lg.sum())
        c = _top_center(_world_points(dep, K, E, red))
        if c is not None:
            reds.append(c)
        c = _top_center(_world_points(dep, K, E, grn))
        if c is not None:
            grns.append(c)
    if reds:
        out["red"] = np.mean(reds, 0)
    if grns:
        out["green"] = np.mean(grns, 0)
    if nlg > 50 and nlg > nlr:
        out["lamp"] = "G"
    elif nlr > 50:
        out["lamp"] = "R"
    return out


def _extract_path(frames):
    """Ordered red-cube xy positions over the cue frames, consecutive duplicates removed."""
    pts = []
    for f in frames:
        o = _frame_objects(f)
        if o["red"] is not None and abs(o["red"][2] - 2 * CUBE_HALF) < 0.015:
            pts.append([float(o["red"][0]), float(o["red"][1])])
    path = []
    for p in pts:
        if not path or math.hypot(p[0] - path[-1][0], p[1] - path[-1][1]) > 0.003:
            path.append(p)
    # remove isolated outliers (jumps far larger than neighbours)
    if len(path) >= 5:
        P = np.array(path)
        keep = [True] * len(P)
        for i in range(1, len(P) - 1):
            a = np.linalg.norm(P[i] - P[i - 1])
            b = np.linalg.norm(P[i + 1] - P[i])
            c = np.linalg.norm(P[i + 1] - P[i - 1])
            if a > 0.03 and b > 0.03 and c < 0.03:
                keep[i] = False
        path = [path[i] for i in range(len(P)) if keep[i]]
    return path


def _smooth(path):
    P = np.array(path, dtype=np.float64)
    if len(P) < 5:
        return P
    Q = P.copy()
    Q[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
    return Q


# ---------------------------------------------------------------- control
def _tcp(robot):
    return np.asarray(robot.now().tcp, dtype=np.float64)


def _rpy_mat(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(M):
    c = min(max((float(np.trace(M)) - 1.0) / 2.0, -1.0), 1.0)
    ang = math.acos(c)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return v / 2.0
    return v * (ang / (2.0 * s))


def _rot_hold(tcp):
    """Rotation command (drx, dry, drz) nudging the TCP orientation toward DOWN_RPY.
    Convention (checked on demos): world rotvec change per step ~ -0.03 * r."""
    e = _rotvec(_rpy_mat(DOWN_RPY) @ _rpy_mat(tcp[3:6]).T)
    n = float(np.linalg.norm(e))
    if n < 0.01:
        return np.zeros(3)
    r = -ROT_GAIN * e / 0.1
    m = float(np.linalg.norm(r))
    if m > 0.5:
        r = r * (0.5 / m)
    return r


def _cmd(robot, a3, grip, amax=1.0):
    a = np.asarray(a3, dtype=np.float64)
    m = float(np.abs(a).max())
    if m > amax:
        a = a * (amax / m)
    r = _rot_hold(np.asarray(robot.now().tcp, dtype=np.float64))
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return a


def _move_to(robot, target, grip, tol=0.004, max_steps=40, gain=1.0, amax=1.0,
             settle_v=0.002, offset=None):
    """P-control the point (TCP + offset) to target; stop when close and slow."""
    off = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prev = _tcp(robot)[:3]
    n = 0
    err = 1.0
    for n in range(1, max_steps + 1):
        p = _tcp(robot)[:3]
        e = target - (p + off)
        err = float(np.linalg.norm(e))
        v = float(np.linalg.norm(p - prev))
        if err < tol and v < settle_v:
            n -= 1
            break
        prev = p
        _cmd(robot, gain * e / 0.1, grip, amax)
    p = _tcp(robot)[:3]
    return {"steps": n, "err": round(float(np.linalg.norm(target - (p + off))), 4),
            "tcp": [round(float(x), 4) for x in p]}


def _interp(P, s):
    n = len(P)
    if s <= 0:
        return P[0]
    if s >= n - 1:
        return P[n - 1]
    i = int(math.floor(s))
    f = s - i
    return P[i] * (1 - f) + P[i + 1] * f


# ---------------------------------------------------------------- policy
def perceive_decide(obs, goal, memory, last):
    """Stage machine: watch -> pick -> trace (-> retrace)."""
    mem = {k: memory.get(k) for k in
           ("stage", "path", "green", "grip_offset", "picks", "traces", "log")}
    stage = mem["stage"] or "watch"
    log = mem["log"] or ""
    picks = int(mem["picks"] or 0)
    traces = int(mem["traces"] or 0)
    last = last or ""

    if stage == "watch" and not last.startswith("act_watch_cue"):
        cur = _frame_objects(obs.current)
        if cur["green"] is not None:
            mem["green"] = [round(float(x), 4) for x in cur["green"]]
        mem.update(stage="watch", picks=picks, traces=traces, log=log + "start;")
        return {"memory": mem, "act": {"tool": "act_watch_cue", "args": {"max_steps": 140}}}

    if stage == "watch":
        path = _extract_path(obs.frames)
        if mem["path"] and len(mem["path"]) > len(path):
            path = mem["path"]
        mem["path"] = [[round(p[0], 4), round(p[1], 4)] for p in path]
        cur = _frame_objects(obs.current)
        if cur["green"] is not None:
            mem["green"] = [round(float(x), 4) for x in cur["green"]]
        log += "path_n=%d;" % len(path)
        stage = "pick"

    if stage == "pick" and last.startswith("act_pick") and '"grasped": true' in last:
        stage = "trace"
    elif stage == "pick" and last.startswith("act_pick"):
        cur = _frame_objects(obs.current)
        if cur["green"] is not None:
            mem["green"] = [round(float(x), 4) for x in cur["green"]]
        log += "regrasp;"

    if stage == "pick":
        if picks >= 3 or not mem["green"]:
            mem.update(stage="pick", picks=picks, traces=traces, log=log + "giveup;")
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}
        g = mem["green"]
        mem.update(stage="pick", picks=picks + 1, traces=traces, log=log)
        return {"memory": mem, "act": {"tool": "act_pick",
                                       "args": {"x": g[0], "y": g[1], "top_z": g[2]}}}

    if stage == "trace":
        if last.startswith("act_pick"):
            # parse the grip offset written by act_pick: '"offset": [dx, dy]'
            off = [0.0, 0.0]
            k = last.find('"offset": [')
            if k >= 0:
                seg = last[k + 11: last.find("]", k)]
                vals = [s.strip() for s in seg.split(",")]
                if len(vals) == 2:
                    off = [float(vals[0]), float(vals[1])]
            mem["grip_offset"] = off
        if '"dropped": true' in last:
            mem.update(stage="pick", picks=picks, traces=traces, log=log + "dropped;")
            cur = _frame_objects(obs.current)
            if cur["green"] is not None:
                mem["green"] = [round(float(x), 4) for x in cur["green"]]
            g = mem["green"]
            mem["picks"] = picks + 1
            return {"memory": mem, "act": {"tool": "act_pick",
                                           "args": {"x": g[0], "y": g[1], "top_z": g[2]}}}
        if not mem["path"] or len(mem["path"]) < 3:
            mem.update(stage="trace", picks=picks, traces=traces, log=log + "nopath;")
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}
        z = CARRY_Z if traces == 0 else CARRY_Z - 0.012
        mem.update(stage="trace", picks=picks, traces=traces + 1, log=log + "trace%d;" % traces)
        off = mem["grip_offset"] or [0.0, 0.0]
        return {"memory": mem, "act": {"tool": "act_trace",
                                       "args": {"path": mem["path"], "ox": off[0], "oy": off[1],
                                                "z": z, "speed": 1.0}}}

    mem.update(stage=stage, picks=picks, traces=traces, log=log)
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


def act_watch_cue(robot, max_steps=140):
    """Hold still with the gripper open while the red cube traces the shape; stop
    one step after the lamp is first seen green (after having been red)."""
    seen_red = False
    n = 0
    lamp_hist = ""
    green_at = -1
    for n in range(1, max_steps + 1):
        o = _frame_objects(robot.now())
        lamp_hist += o["lamp"]
        if o["lamp"] == "R":
            seen_red = True
        if o["lamp"] == "G" and seen_red:
            if green_at < 0:
                green_at = n
            if o["red"] is None:
                break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"steps": n, "seen_red": seen_red, "green_at": green_at,
            "step_now": int(robot.now().step), "lamp": lamp_hist[-12:]}


def act_pick(robot, x, y, top_z):
    """Hover above the green cube, refine its position, descend, close, lift to carry height."""
    r = {"x": x, "y": y}
    a = _move_to(robot, [x, y, HOVER_Z], 1.0, tol=0.006, max_steps=40)
    r["hover"] = a
    # refine the cube estimate from the current view
    o = _frame_objects(robot.now())
    if o["green"] is not None and math.hypot(o["green"][0] - x, o["green"][1] - y) < 0.04:
        x, y = float(o["green"][0]), float(o["green"][1])
        if abs(o["green"][2] - 2 * CUBE_HALF) < 0.01:
            top_z = float(o["green"][2])
    r["refined"] = [round(x, 4), round(y, 4), round(top_z, 4)]
    gz = max(top_z - GRASP_DZ, 0.022)
    a = _move_to(robot, [x, y, gz + 0.03], 1.0, tol=0.004, max_steps=20, amax=0.8)
    a = _move_to(robot, [x, y, gz], 1.0, tol=0.003, max_steps=25, amax=0.5)
    r["descend"] = a
    for _ in range(7):
        p = _tcp(robot)
        e = np.array([x, y, gz]) - p[:3]
        _cmd(robot, 0.8 * e / 0.1, -1.0, 0.3)
    fing = [float(v) for v in robot.now().fingers]
    r["fingers_closed"] = [round(v, 4) for v in fing]
    p = _tcp(robot)
    off = [round(float(min(max(x - p[0], -0.012), 0.012)), 4),
           round(float(min(max(y - p[1], -0.012), 0.012)), 4)]
    a = _move_to(robot, [p[0], p[1], CARRY_Z], -1.0, tol=0.004, max_steps=20, amax=0.6)
    r["lift"] = a
    fing = [float(v) for v in robot.now().fingers]
    o = _frame_objects(robot.now())
    lifted = o["green"] is not None and o["green"][2] > 2 * CUBE_HALF + 0.012
    r["fingers"] = [round(v, 4) for v in fing]
    r["green_after"] = None if o["green"] is None else [round(float(v), 4) for v in o["green"]]
    r["grasped"] = bool(0.008 < sum(fing) / 2 < 0.03 and (lifted or o["green"] is None))
    r["offset"] = off
    return r


def act_trace(robot, path, ox=0.0, oy=0.0, z=CARRY_Z, speed=1.0):
    """Carry the held cube (centre = TCP + (ox, oy)) to path[0], then follow the path
    in order with a moving set-point (feed-forward + P feedback), close the loop
    back to path[0] and hold."""
    P = _smooth(path)
    P = np.vstack([P, P[:1]]) if np.linalg.norm(P[0] - P[-1]) < 0.06 else P
    off = np.array([ox, oy, 0.0])
    r = {"n": len(P)}
    start = np.array([P[0][0], P[0][1], z])
    a = _move_to(robot, start, -1.0, tol=0.004, max_steps=40, amax=0.6, offset=off)
    r["to_start"] = a
    s = 0.0
    errs = []
    n = 0
    end = len(P) - 1
    while s < end + 4 and n < 160:
        p = _tcp(robot)
        cube = p[:3] + off
        pd = _interp(P, s)
        lag = float(np.linalg.norm(pd - cube[:2]))
        errs.append(lag)
        sp = speed if lag < 0.012 else (0.5 * speed if lag < 0.02 else 0.0)
        s_next = s + sp
        pn = _interp(P, s_next)
        pnn = _interp(P, s_next + sp)
        exy = pn - cube[:2]
        a = np.zeros(3)
        a[:2] = 1.0 * exy / 0.1 + 0.6 * (pnn - pn) / STEP_GAIN
        a[2] = 1.0 * (z - cube[2]) / 0.1
        _cmd(robot, a, -1.0, 0.8)
        s = s_next
        n += 1
        if n % 10 == 0 and robot.now().fingers[0] < 0.004:
            r["dropped"] = True
            break
    r["steps"] = n
    r["max_lag"] = round(max(errs), 4) if errs else None
    r["mean_lag"] = round(float(np.mean(errs)), 4) if errs else None
    h = _move_to(robot, np.array([P[-1][0], P[-1][1], z]), -1.0, tol=0.003, max_steps=15,
                 amax=0.4, offset=off)
    r["hold"] = h
    fing = [float(v) for v in robot.now().fingers]
    r["fingers"] = [round(v, 4) for v in fing]
    if fing[0] < 0.004:
        r["dropped"] = True
    return r


def act_hold(robot, steps=20):
    """Hold position (gripper unchanged-closed) for a number of steps."""
    g = -1.0 if robot.now().fingers[0] < 0.03 else 1.0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
    return {"steps": int(steps)}
