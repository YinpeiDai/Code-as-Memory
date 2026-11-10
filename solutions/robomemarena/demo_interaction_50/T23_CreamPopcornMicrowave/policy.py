"""T23 CreamPopcornMicrowave: open the microwave door, put the cream inside, then put the
popcorn where the cream was placed (inside the microwave).

Perception: agentview depth + calibration -> world point cloud. The microwave handle anchor H
comes from the microwave body extent; cream/popcorn grasp points come from coloured clusters on
the table. All motion targets are expressed relative to these perceived anchors.
Control: Cartesian P-D servo on the measured TCP with OSC_POSE delta steps (world frame).
"""

TABLE_Z = 0.43
GRASP_Z = 0.438
# fixed side orientation (gripper approach axis = world +y, fingers close along world x)
R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
# top-down orientation (approach axis = world -z, fingers close along world y)
R_TOP = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
PRIOR_H = [0.097, 0.411, 0.583]
PRIOR_CREAM = [0.165, -0.094, GRASP_Z]
PRIOR_POP = [0.164, -0.192, GRASP_Z]
FIELDS = ["stage", "handle", "cream", "popcorn", "door_open", "cream_placed",
          "retries", "handle_tries", "decisions", "notes"]


# ----------------------------------------------------------------------------- geometry
def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    """Shortest rotation vector (angle in [0, pi]) of a rotation matrix, via a quaternion."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    w = min(1.0, w)
    ang = 2.0 * math.acos(w)
    sn = math.sqrt(max(1e-16, 1.0 - w * w))
    if ang < 1e-7:
        return np.zeros(3)
    return np.array([x, y, z]) / sn * ang


def _orient(name):
    if name == "side":
        return np.array(R_SIDE)
    if name == "top":
        return np.array(R_TOP)
    return None


def _cloud(frame):
    D = np.asarray(frame.agentview_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.agentview_K)
    E = np.asarray(frame.agentview_extrinsic)
    H, W = D.shape
    u = np.arange(W)[None, :].repeat(H, 0)
    v = np.arange(H)[:, None].repeat(W, 1)
    x = (u - K[0, 2]) / K[0, 0] * D
    y = (v - K[1, 2]) / K[1, 1] * D
    C = np.stack([x, y, D], -1)
    P = (C - E[:, 3]) @ E[:, :3]
    return P, D > 0


def _width(frame):
    f = np.asarray(frame.fingers)
    return float(f[0] - f[1])


def _add(a, b):
    return [float(a[0] + b[0]), float(a[1] + b[1]), float(a[2] + b[2])]


# ----------------------------------------------------------------------------- perception
def _microwave(frame):
    """Handle anchor H from microwave body extent: x = front (+x) edge - 0.075, y = -y face."""
    P, ok = _cloud(frame)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    m = ok & (z > 0.45) & (z < 0.665) & (y > 0.2) & (y < 0.8) & (x > -0.4) & (x < 0.5)
    q = P[m]
    if len(q) < 300:
        return None, int(len(q))
    xmax = float(np.percentile(q[:, 0], 99.5))
    ymin = float(np.percentile(q[:, 1], 2))
    return [xmax - 0.075, ymin + 0.001, 0.583], int(len(q))


def _components(mask):
    """4-connected components of a small boolean mask; list of (rows, cols) arrays."""
    idx = np.argwhere(mask)
    seen = set()
    pts = set((int(r), int(c)) for r, c in idx)
    comps = []
    for r, c in idx:
        key = (int(r), int(c))
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        comp = []
        while stack:
            a, b = stack.pop()
            comp.append((a, b))
            for nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1), (a + 2, b), (a, b + 2), (a - 2, b), (a, b - 2)):
                if nb in pts and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _objects(frame, wide=False):
    """Cream / popcorn grasp points from table-top clusters (top-down grasp, TCP z = GRASP_Z).
    wide=True also searches the table in front of the microwave (dropped objects), objects low only."""
    P, ok = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(np.float64)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    if wide:
        m = ok & (z > TABLE_Z + 0.007) & (z < 0.52) & (x > -0.12) & (x < 0.42) & (y > -0.42) & (y < 0.37)
    else:
        m = ok & (z > TABLE_Z + 0.007) & (z < 0.56) & (x > 0.0) & (x < 0.42) & (y > -0.42) & (y < 0.2)
    if int(m.sum()) > 3000:
        return []
    out = []
    for comp in _components(m):
        if len(comp) < 15 or len(comp) > 600:
            continue
        rr = np.array([c[0] for c in comp])
        cc = np.array([c[1] for c in comp])
        q = P[rr, cc]
        col = rgb[rr, cc].mean(0)
        if q[:, 0].max() - q[:, 0].min() > 0.15 or q[:, 1].max() - q[:, 1].min() > 0.10:
            continue
        out.append({"x": float(q[:, 0].mean()), "ymid": float((q[:, 1].min() + q[:, 1].max()) / 2),
                    "ztop": float(q[:, 2].max()), "n": int(len(comp)),
                    "blue": float(col[2] - col[0])})
    return out


def _assign(objs, prev_cream=None, prev_pop=None):
    """Label clusters by colour (cream is blue/white, popcorn is red/yellow); grasp points."""
    cream, pop = None, None
    if not objs:
        return None, None
    objs = sorted(objs, key=lambda o: -o["blue"])
    cands_c = [o for o in objs if o["blue"] > 8.0]
    cands_p = [o for o in objs if o["blue"] <= 8.0]
    if len(objs) >= 2 and (not cands_c or not cands_p):
        cands_c, cands_p = [objs[0]], [objs[-1]]

    def near(cands, prev):
        if not cands:
            return None
        if prev is None:
            return max(cands, key=lambda o: o["n"])
        return min(cands, key=lambda o: (o["x"] - prev[0]) ** 2 + (o["ymid"] - prev[1]) ** 2)

    c = near(cands_c, prev_cream)
    p = near(cands_p, prev_pop)
    if c is not None:
        cream = [c["x"] - 0.0083, c["ymid"] + 0.0013, max(GRASP_Z, c["ztop"] - 0.025)]
    if p is not None:
        pop = [p["x"] - 0.0057, p["ymid"] + 0.0032, max(GRASP_Z, p["ztop"] - 0.025)]
    return cream, pop


def _door_state(frame, H):
    """Door panel points in front of the -y face. Hinge ~ (H.x-0.29, H.y+0.03); the median polar angle
    of panel points about the hinge estimates the opening angle (closed: no panel points, open: ~90 deg)."""
    P, ok = _cloud(frame)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    m = ok & (y > H[1] - 0.30) & (y < H[1] - 0.04) & (x > H[0] - 0.45) & (x < H[0] + 0.06) & (z > 0.50) & (z < 0.63)
    q = P[m]
    cav = ok & (y > H[1] + 0.06) & (y < H[1] + 0.25) & (x > H[0] - 0.3) & (x < H[0] + 0.05) & (z > 0.45) & (z < 0.62)
    n = int(len(q))
    hx, hy = H[0] - 0.29, H[1] + 0.03
    ang = None
    if n >= 20:
        ang = float(np.degrees(np.median(np.arctan2(hy - q[:, 1], q[:, 0] - hx))))
    is_open = bool(n >= 60 and ang is not None and ang > 72.0)
    return {"panel_n": n, "angle_deg": ang, "cavity_n": int(cav.sum()), "open": is_open}


def perceive_scene(obs):
    """Measure handle anchor, cream/popcorn grasp points and door state from the latest frame."""
    f = obs.current
    H, n = _microwave(f)
    objs = _objects(f)
    c, p = _assign(objs)
    door = _door_state(f, H if H is not None else PRIOR_H)
    return {"handle": H, "mw_points": n, "cream": c, "popcorn": p, "clusters": objs, "door": door,
            "tcp": [float(v) for v in f.tcp], "width": _width(f)}


# ----------------------------------------------------------------------------- control
def _servo(robot, wp, grip):
    tgt = np.array(wp["p"], dtype=float)
    f = robot.now()
    Rt = _orient(wp.get("o"))
    if Rt is None:
        Rt = _aa2R(f.tcp[3:])
    tol = float(wp.get("tol", 0.004))
    rtol = float(wp.get("rtol", 0.06))
    vmax = float(wp.get("vmax", 1.0))
    rmax = float(wp.get("rmax", 0.5))
    n = int(wp.get("n", 150))
    prev_p, prev_R = None, None
    hist = []
    rhist = []
    status = "timeout"
    steps = 0
    for i in range(n):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        R = _aa2R(f.tcp[3:])
        e = tgt - p
        w = _R2aa(Rt @ R.T)
        if np.linalg.norm(e) < tol and np.linalg.norm(w) < rtol:
            status = "reached"
            break
        v = p - prev_p if prev_p is not None else np.zeros(3)
        wv = _R2aa(R @ prev_R.T) if prev_R is not None else np.zeros(3)
        a = 15.0 * e - 40.0 * v
        am = float(np.max(np.abs(a)))
        if am > vmax:
            a = a * (vmax / am)
        r = 1.9 * w - 5.0 * wv
        rm = float(np.max(np.abs(r)))
        if rm > rmax:
            r = r * (rmax / rm)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
        prev_p, prev_R = p, R
        hist.append(p)
        rhist.append(R)
        if i >= 14 and np.linalg.norm(hist[-1] - hist[-10]) < 0.0015 and \
                np.linalg.norm(_R2aa(rhist[-1] @ rhist[-10].T)) < 0.01:
            status = "stalled"
            break
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=float)
    err = float(np.linalg.norm(tgt - p))
    rerr = float(np.linalg.norm(_R2aa(Rt @ _aa2R(f.tcp[3:]).T)))
    return {"status": status, "steps": steps, "err": round(err, 4), "rerr": round(rerr, 3),
            "tcp": [round(float(t), 4) for t in p], "width": round(_width(f), 4)}


def act_move(robot, waypoints, grip=-1.0, pre=0):
    """Servo the TCP through waypoints [{p:[x,y,z], o:'top'|'side'|None, tol, rtol, vmax, n, stall}].
    grip: -1 open / +1 close held during the whole call; pre: hold-still steps first (gripper actuates).
    Stops early when a waypoint without stall=True stalls. Returns per-waypoint measured results."""
    out = []
    for k in range(int(pre)):
        robot.step(0, 0, 0, 0, 0, 0, grip)
    for wp in waypoints:
        res = _servo(robot, wp, grip)
        res["name"] = wp.get("name", "")
        out.append(res)
        if res["status"] == "stalled" and not wp.get("stall", False):
            break
    f = robot.now()
    return {"segments": out, "final_tcp": [round(float(t), 4) for t in f.tcp[:3]], "width": round(_width(f), 4),
            "completed": len(out) == len(waypoints)}


def act_gripper(robot, close=True, steps=20):
    """Hold the arm still and close (+1) or open (-1) the gripper for `steps` steps; returns finger widths."""
    g = 1.0 if close else -1.0
    ws = []
    for k in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, g)
        ws.append(_width(robot.now()))
    return {"width_start": round(ws[0], 4), "width_end": round(ws[-1], 4),
            "settled": bool(len(ws) > 3 and abs(ws[-1] - ws[-4]) < 0.001)}


def act_wait(robot, steps=10, grip=0.0):
    """Hold still for `steps` steps (observe / monitoring period)."""
    for k in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, grip)
    f = robot.now()
    return {"waited": int(steps), "width": round(_width(f), 4)}


# ----------------------------------------------------------------------------- decisions
def _wp(p, o=None, name="", **kw):
    d = {"p": [float(v) for v in p], "o": o, "name": name}
    d.update(kw)
    return d


def _ret(mem, stage, tool, args, note=None):
    mem = dict(mem)
    mem["stage"] = stage
    if note is not None:
        mem["notes"] = (str(mem.get("notes") or "") + " | " + note)[-1500:]
    out = {k: mem.get(k) for k in FIELDS}
    return {"memory": out, "act": {"tool": tool, "args": args}}


def _approach_handle(H, k):
    off = [[0.0, 0.0], [-0.012, 0.0], [0.012, 0.0], [0.0, 0.01]][k % 4]
    G = _add(H, [off[0], off[1], 0.0])
    return [
        _wp(_add(G, [-0.06, -0.20, 0.0]), "side", "pre_far", tol=0.03, rtol=0.4, n=120),
        _wp(_add(G, [-0.054, -0.062, 0.0]), "side", "pre_near", tol=0.005, rtol=0.04, n=100),
        _wp(G, "side", "handle", tol=0.003, rtol=0.04, vmax=0.5, n=60, stall=True),
    ]


def _pull_path(H):
    rel = [(-0.063, -0.071), (-0.11, -0.125), (-0.155, -0.162), (-0.2, -0.188),
           (-0.246, -0.205), (-0.30, -0.208), (-0.351, -0.20), (-0.387, -0.194)]
    wps = []
    for i, (dx, dy) in enumerate(rel):
        last = i == len(rel) - 1
        wps.append(_wp(_add(H, [dx, dy, 0.0]), "side", "pull%d" % i, tol=0.008 if last else 0.02,
                       rtol=0.5, n=40, stall=True))
    return wps


def _pick_wps(obj):
    return [
        _wp(_add(obj, [0.0, 0.0, 0.11]), "top", "above", tol=0.004, rtol=0.04, n=150),
        _wp(obj, "top", "descend", tol=0.003, rtol=0.04, vmax=0.6, n=80, stall=True),
    ]


def _place_wps(H, which):
    S = _add(H, [-0.127, -0.413, -0.026])
    wps = [
        _wp(S, "top", "rot_spot", tol=0.01, rtol=0.1, n=100),
        _wp(S, "side", "rotate", tol=0.012, rtol=0.04, n=100, stall=True),
    ]
    if which == "cream":
        wps += [
            _wp(_add(H, [-0.10, -0.12, 0.017]), "side", "front", tol=0.008, rtol=0.05, n=100),
            _wp(_add(H, [-0.10, 0.038, 0.017]), "side", "inside", tol=0.006, rtol=0.05, vmax=0.6, n=80),
            _wp(_add(H, [-0.078, 0.059, -0.004]), "side", "release_pt", tol=0.004, rtol=0.05, vmax=0.4, n=60, stall=True),
        ]
    else:
        wps += [
            _wp(_add(H, [-0.11, -0.16, 0.017]), "side", "front", tol=0.008, rtol=0.05, n=100),
            _wp(_add(H, [-0.11, -0.06, 0.017]), "side", "entry", tol=0.006, rtol=0.05, vmax=0.5, n=80),
            _wp(_add(H, [-0.11, -0.005, 0.017]), "side", "inside", tol=0.004, rtol=0.05, vmax=0.3, n=80, stall=True),
        ]
    return wps


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive -> open door -> pick cream -> place in microwave -> pick popcorn ->
    place where the cream was placed (microwave). Every stage checks measured fingers/TCP/depth."""
    mem = {k: memory.get(k) if isinstance(memory, dict) else None for k in FIELDS}
    mem["decisions"] = int(mem.get("decisions") or 0) + 1
    mem["retries"] = int(mem.get("retries") or 0)
    mem["handle_tries"] = int(mem.get("handle_tries") or 0)
    f = obs.current
    w = _width(f)
    tcp = np.asarray(f.tcp[:3], dtype=float)
    stage = mem.get("stage") or "start"
    H = mem.get("handle")

    # ---------------- initial perception
    if stage == "start":
        Hm, n = _microwave(f)
        c, p = _assign(_objects(f))
        if (Hm is None or c is None or p is None) and mem["retries"] < 2:
            mem["retries"] += 1
            return _ret(mem, "start", "act_wait", {"steps": 3}, "perception incomplete n=%d" % n)
        mem["handle"] = Hm if Hm is not None else PRIOR_H
        mem["cream"] = c if c is not None else PRIOR_CREAM
        mem["popcorn"] = p if p is not None else PRIOR_POP
        mem["door_open"] = False
        mem["cream_placed"] = False
        mem["retries"] = 0
        note = "H=%s C=%s P=%s" % (np.round(mem["handle"], 3).tolist(), np.round(mem["cream"], 3).tolist(),
                                   np.round(mem["popcorn"], 3).tolist())
        return _ret(mem, "approach_handle", "act_move",
                    {"waypoints": _approach_handle(mem["handle"], 0), "grip": -1.0}, note)

    if H is None:
        H = PRIOR_H
        mem["handle"] = H

    # ---------------- door
    if stage == "approach_handle":
        return _ret(mem, "grasp_handle", "act_gripper", {"close": True, "steps": 20})

    if stage == "grasp_handle":
        if 0.008 < w < 0.05:
            return _ret(mem, "pull", "act_move", {"waypoints": _pull_path(_add(tcp, [0.023, 0.007, 0.0])),
                                                  "grip": 1.0}, "handle grasped w=%.3f" % w)
        mem["handle_tries"] += 1
        if mem["handle_tries"] > 4:
            mem["notes"] = (mem.get("notes") or "") + " | handle tries exhausted"
        G = H
        back = [_wp(_add(G, [-0.054, -0.09, 0.0]), "side", "backoff", tol=0.01, rtol=0.1, n=60, stall=True)]
        return _ret(mem, "regrasp_backoff", "act_move", {"waypoints": back, "grip": -1.0, "pre": 8},
                    "handle missed w=%.3f" % w)

    if stage == "regrasp_backoff":
        # door presumably still closed; re-perceive the anchor if the door is closed
        d = _door_state(f, H)
        Hm, n = _microwave(f)
        if Hm is not None and d["panel_n"] < 30 and abs(Hm[0] - H[0]) < 0.03 and abs(Hm[1] - H[1]) < 0.03:
            pass
        return _ret(mem, "approach_handle", "act_move",
                    {"waypoints": _approach_handle(H, mem["handle_tries"])[1:], "grip": -1.0},
                    "re-approach k=%d" % mem["handle_tries"])

    if stage == "pull":
        ret = [
            _wp(_add(tcp, [0.0, -0.03, 0.03]), "side", "lift_off", tol=0.01, rtol=0.2, n=30, stall=True),
            _wp(_add(H, [-0.15, -0.42, 0.09]), "top", "clear", tol=0.01, rtol=0.08, n=150),
        ]
        return _ret(mem, "door_check", "act_move", {"waypoints": ret, "grip": -1.0, "pre": 10},
                    "pull done w=%.3f" % w)

    if stage == "door_check":
        d = _door_state(f, H)
        c, p = _assign(_objects(f), mem.get("cream"), mem.get("popcorn"))
        if c is not None and np.hypot(c[0] - mem["cream"][0], c[1] - mem["cream"][1]) < 0.06:
            mem["cream"] = c
        if p is not None and np.hypot(p[0] - mem["popcorn"][0], p[1] - mem["popcorn"][1]) < 0.06:
            mem["popcorn"] = p
        note = "door n=%d ang=%s cav=%d" % (d["panel_n"], None if d["angle_deg"] is None else round(d["angle_deg"], 1), d["cavity_n"])
        if d["open"] or mem["retries"] >= 2:
            mem["door_open"] = bool(d["open"])
            mem["retries"] = 0
            return _ret(mem, "cream_descend", "act_move", {"waypoints": _pick_wps(mem["cream"]), "grip": -1.0}, note)
        mem["retries"] += 1
        if d["panel_n"] < 30 or d["angle_deg"] is None or d["angle_deg"] < 12.0:
            # door still closed: grasp the handle again
            mem["handle_tries"] += 1
            return _ret(mem, "approach_handle", "act_move",
                        {"waypoints": _approach_handle(H, mem["handle_tries"]), "grip": -1.0}, note + " closed->regrasp")
        # partially open: push the door open from its inner side with a sweep around the hinge
        hinge = [H[0] - 0.29, H[1] + 0.03]
        th = math.radians(d["angle_deg"])
        wps = []
        z0 = 0.60
        r = 0.20
        a0 = max(0.15, th - 0.35)
        sx, sy = hinge[0] + r * math.cos(a0), hinge[1] - r * math.sin(a0)
        wps.append(_wp([sx, sy, 0.70], "top", "push_above", tol=0.01, rtol=0.08, n=150))
        wps.append(_wp([sx, sy, z0], "top", "push_down", tol=0.008, rtol=0.08, vmax=0.5, n=60, stall=True))
        for k in range(1, 6):
            a = a0 + (1.75 - a0) * k / 5.0
            wps.append(_wp([hinge[0] + r * math.cos(a), hinge[1] - r * math.sin(a), z0], "top", "push%d" % k,
                           tol=0.015, rtol=0.2, vmax=0.6, n=50, stall=True))
        wps.append(_wp([hinge[0] + 0.15, hinge[1] - 0.35, 0.70], "top", "push_up", tol=0.015, rtol=0.1, n=80))
        wps.append(_wp(_add(H, [-0.15, -0.42, 0.09]), "top", "clear", tol=0.01, rtol=0.08, n=150))
        return _ret(mem, "door_check", "act_move", {"waypoints": wps, "grip": 1.0}, note + " partial th=%.2f push" % th)

    # ---------------- cream
    if stage in ("cream_descend", "pop_descend"):
        nxt = "cream_close" if stage == "cream_descend" else "pop_close"
        return _ret(mem, nxt, "act_gripper", {"close": True, "steps": 20})

    if stage in ("cream_close", "pop_close"):
        key = "cream" if stage == "cream_close" else "popcorn"
        obj = mem[key]
        if 0.008 < w < 0.065:
            lift = [_wp(_add(obj, [0.0, 0.0, 0.12]), "top", "lift", tol=0.006, rtol=0.05, vmax=0.7, n=80)]
            nxt = "cream_lift" if key == "cream" else "pop_lift"
            return _ret(mem, nxt, "act_move", {"waypoints": lift, "grip": 1.0}, "%s grasp w=%.3f" % (key, w))
        mem["retries"] += 1
        up = [_wp(_add(obj, [0.0, 0.0, 0.12]), "top", "lift_empty", tol=0.01, rtol=0.1, n=60)]
        return _ret(mem, "cream_redetect" if key == "cream" else "pop_redetect", "act_move",
                    {"waypoints": up, "grip": -1.0, "pre": 8}, "%s grasp missed w=%.3f" % (key, w))

    if stage in ("cream_redetect", "pop_redetect"):
        key = "cream" if stage == "cream_redetect" else "popcorn"
        c, p = _assign(_objects(f, True), mem.get("cream"), mem.get("popcorn"))
        new = c if key == "cream" else p
        if new is not None:
            mem[key] = new
        nxt = "cream_descend" if key == "cream" else "pop_descend"
        # vary the grasp point slightly on repeated misses instead of repeating the same motion
        off = [[0.0, 0.0], [0.0, 0.0], [0.008, 0.0], [-0.008, 0.0], [0.0, 0.006], [0.0, -0.006]][mem["retries"] % 6]
        tgt = _add(mem[key], [off[0], off[1], 0.0])
        return _ret(mem, nxt, "act_move", {"waypoints": _pick_wps(tgt), "grip": -1.0},
                    "%s redetect %s" % (key, None if new is None else np.round(new, 3).tolist()))

    if stage in ("cream_lift", "pop_lift"):
        key = "cream" if stage == "cream_lift" else "popcorn"
        if 0.008 < w < 0.065:
            nxt = "cream_insert" if key == "cream" else "pop_insert"
            return _ret(mem, nxt, "act_move", {"waypoints": _place_wps(H, key), "grip": 1.0},
                        "%s lifted w=%.3f" % (key, w))
        mem["retries"] += 1
        up = [_wp(_add(H, [-0.05, -0.45, 0.12]), "top", "clear", tol=0.02, rtol=0.1, n=80)]
        return _ret(mem, "cream_redetect" if key == "cream" else "pop_redetect", "act_move",
                    {"waypoints": up, "grip": -1.0, "pre": 5}, "%s lost after lift" % key)

    if stage == "cream_insert" and not (0.008 < w < 0.065):
        mem["retries"] += 1
        up = [_wp(_add(tcp, [0.0, -0.12, 0.05]), "side", "back_out", tol=0.02, rtol=0.2, n=80, stall=True),
              _wp(_add(H, [-0.05, -0.45, 0.12]), "top", "clear", tol=0.02, rtol=0.1, n=120)]
        return _ret(mem, "cream_redetect", "act_move", {"waypoints": up, "grip": -1.0},
                    "cream lost during transport w=%.3f" % w)

    if stage == "cream_insert":
        out = [
            _wp(_add(H, [-0.05, -0.09, -0.011]), "side", "retreat", tol=0.01, rtol=0.08, vmax=0.6, n=80, stall=True),
            _wp(_add(H, [-0.05, -0.10, 0.128]), "side", "up", tol=0.01, rtol=0.08, n=80, stall=True),
        ]
        mem["cream_placed"] = True
        return _ret(mem, "cream_release", "act_move", {"waypoints": out, "grip": -1.0, "pre": 15},
                    "cream released at tcp=%s w=%.3f" % (np.round(tcp, 3).tolist(), w))

    if stage == "cream_release":
        objs = _objects(f, True)
        blue = [o for o in objs if o["blue"] > 12.0 and o["n"] >= 20]
        c, p = _assign(objs, mem.get("cream"), mem.get("popcorn"))
        if blue and mem["retries"] < 4:
            c, _unused = _assign(blue, mem.get("cream"), None)
            # the cream is still (or again) on the table: placement did not happen -> pick it again
            mem["cream_placed"] = False
            mem["cream"] = c
            mem["retries"] += 1
            wps = [_wp(_add(H, [0.03, -0.30, 0.12]), "top", "via", tol=0.03, rtol=0.4, n=100)] + _pick_wps(c)
            return _ret(mem, "cream_descend", "act_move", {"waypoints": wps, "grip": -1.0},
                        "cream still on table %s" % np.round(c, 3).tolist())
        if p is not None and np.hypot(p[0] - mem["popcorn"][0], p[1] - mem["popcorn"][1]) < 0.06:
            mem["popcorn"] = p
        mem["retries"] = 0
        wps = [_wp(_add(H, [0.03, -0.30, 0.12]), "top", "via", tol=0.03, rtol=0.4, n=100)] + _pick_wps(mem["popcorn"])
        return _ret(mem, "pop_descend", "act_move", {"waypoints": wps, "grip": -1.0},
                    "popcorn at %s" % np.round(mem["popcorn"], 3).tolist())

    if stage == "pop_insert" and not (0.008 < w < 0.065):
        mem["retries"] += 1
        up = [_wp(_add(tcp, [0.0, -0.12, 0.05]), "side", "back_out", tol=0.02, rtol=0.2, n=80, stall=True),
              _wp(_add(H, [-0.05, -0.45, 0.12]), "top", "clear", tol=0.02, rtol=0.1, n=120)]
        return _ret(mem, "pop_redetect", "act_move", {"waypoints": up, "grip": -1.0},
                    "popcorn lost during transport w=%.3f" % w)

    if stage == "pop_insert":
        out = [_wp(_add(H, [-0.11, -0.13, 0.017]), "side", "retreat", tol=0.01, rtol=0.08, vmax=0.5, n=80, stall=True)]
        return _ret(mem, "pop_release", "act_move", {"waypoints": out, "grip": -1.0, "pre": 15},
                    "popcorn released at tcp=%s w=%.3f" % (np.round(tcp, 3).tolist(), w))

    if stage == "pop_release":
        return _ret(mem, "pop_monitor", "act_wait", {"steps": 40}, "monitoring")

    if stage == "pop_monitor":
        # no success yet: popcorn still on the table -> pick again; otherwise push it deeper once
        c, p = _assign(_objects(f, True), None, mem.get("popcorn"))
        if p is not None and mem["retries"] < 3:
            mem["popcorn"] = p
            mem["retries"] += 1
            wps = [_wp(_add(H, [-0.11, -0.25, 0.12]), "side", "back", tol=0.03, rtol=0.4, n=80, stall=True),
                   _wp(_add(H, [0.03, -0.30, 0.12]), "top", "via", tol=0.03, rtol=0.4, n=100)] + _pick_wps(p)
            return _ret(mem, "pop_descend", "act_move", {"waypoints": wps, "grip": -1.0},
                        "popcorn still on table %s" % np.round(p, 3).tolist())
        wps = [_wp(_add(H, [-0.11, -0.15, 0.005]), "side", "push_front", tol=0.008, rtol=0.05, n=80, stall=True),
               _wp(_add(H, [-0.11, 0.03, 0.005]), "side", "push_in", tol=0.005, rtol=0.05, vmax=0.3, n=80, stall=True),
               _wp(_add(H, [-0.11, -0.15, 0.017]), "side", "push_back", tol=0.01, rtol=0.05, n=80, stall=True)]
        return _ret(mem, "done", "act_move", {"waypoints": wps, "grip": 1.0}, "push popcorn deeper")

    return _ret(mem, "done", "act_wait", {"steps": 40})
