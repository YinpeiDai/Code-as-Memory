"""T03_CreamPuddingBasket: pick cream, place in the near basket, then pick chocolate pudding, place in same basket.

Perception: wrist RGB-D (home view) for the two low boxes, agentview RGB-D for the near basket.
Control: Cartesian P-control on measured TCP with world-frame orientation hold (gripper down).
"""

# ---------------- constants (from assigned demonstrations, see task_analysis.md) ----------------
_PRIOR = {
    "cream": [-0.020, -0.019],
    "chocolate": [-0.019, -0.238],
}
_PRIOR_BASKET = [-0.017, 0.200, 0.566]
_PRIOR_TABLE_Z = 0.430
_PRIOR_HOME = [-0.058, 0.0, 0.671]
_GRASP_DZ = 0.002        # TCP target above table for grasp (expert measured TCP = table + 0.003)
_HOVER_DZ = 0.12         # pre-grasp hover above grasp height
_RELEASE_DRIM = 0.09     # release height above basket rim (expert 0.655 vs rim 0.566)
_CARRY_DRIM = 0.235      # carry height above basket rim (expert ~0.78-0.83)
_KP = 15.0               # position gain: command = KP * error (m), vector-scaled to <= 1
_KR = 2.0                # rotation gain on world-frame axis-angle error (rad)
_MAX_PICK_TRIES = 3


# ---------------- geometry helpers ----------------
def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = np.asarray(depth).astype(np.float64) / 1000.0
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (P - t) @ R, np.asarray(depth) > 0


def _table_z(Wp, valid):
    z = Wp[..., 2][valid]
    z = z[(z > 0.2) & (z < 0.8)]
    if len(z) < 100:
        return _PRIOR_TABLE_Z
    h, edges = np.histogram(z, bins=np.arange(0.2, 0.8, 0.002))
    i = int(np.argmax(h))
    sel = z[(z >= edges[i] - 0.002) & (z < edges[i + 1] + 0.002)]
    if len(sel) == 0:
        return _PRIOR_TABLE_Z
    return float(np.median(sel))


def _components(mask, min_px):
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=np.int64)
    rows, cols = np.nonzero(mask)
    comps = []
    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if lab[r0, c0] >= 0:
            continue
        idx = len(comps)
        stack = [(r0, c0)]
        lab[r0, c0] = idx
        pix = []
        while stack:
            r, c = stack.pop()
            pix.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and lab[rr, cc] < 0:
                    lab[rr, cc] = idx
                    stack.append((rr, cc))
        comps.append(pix)
    return [p for p in comps if len(p) >= min_px]


def _blob_stats(Wp, rgb, pix):
    r = np.array([q[0] for q in pix])
    c = np.array([q[1] for q in pix])
    pts = Wp[r, c]
    col = np.asarray(rgb)[r, c].astype(np.float64)
    blue = float(np.mean(col[:, 2] - col[:, 0] > 8))
    red = float(np.mean(col[:, 0] - col[:, 2] > 25))
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    xy = pts[:, :2] - pts[:, :2].mean(0)
    yaw = 0.0
    elong = 1.0
    if len(xy) > 5:
        C = np.cov(xy.T)
        w, V = np.linalg.eigh(C)
        ax = V[:, 1]
        ang = math.atan2(float(ax[1]), float(ax[0]))
        # fold long-axis angle into [-pi/2, pi/2]
        while ang > math.pi / 2:
            ang -= math.pi
        while ang < -math.pi / 2:
            ang += math.pi
        yaw = ang
        elong = math.sqrt(max(float(w[1]), 1e-12) / max(float(w[0]), 1e-12))
    return {"n": len(pix),
            "center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
            "top": float(np.percentile(pts[:, 2], 90)),
            "blue": blue, "red": red,
            "ext": [float(hi[0] - lo[0]), float(hi[1] - lo[1])],
            "yaw": yaw, "elong": elong}


def _detect_objects_wrist(frame):
    """Cream (blue) and chocolate (red-brown) low boxes on the table from the wrist RGB-D frame."""
    Wp, valid = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    tz = _table_z(Wp, valid)
    z = Wp[..., 2]
    elev = valid & (z > tz + 0.004) & (z < tz + 0.05)
    comps = _components(elev, 40)
    rgb = frame.wrist_rgb
    best = {"cream": None, "chocolate": None}
    for pix in comps:
        if len(pix) > 4000:
            continue
        s = _blob_stats(Wp, rgb, pix)
        if s["ext"][0] > 0.12 or s["ext"][1] > 0.12:
            continue
        if s["blue"] > 0.5:
            b = best["cream"]
            if b is None or s["blue"] * s["n"] > b["blue"] * b["n"]:
                best["cream"] = s
        elif s["red"] > 0.25:
            b = best["chocolate"]
            if b is None or s["red"] * s["n"] > b["red"] * b["n"]:
                best["chocolate"] = s
    return tz, best


def _detect_basket_agent(frame):
    """Near (smallest world-x, i.e. robot-side) basket from agentview rim pixels: [x, y, rim_z, ext_x, ext_y] or None."""
    Wp, valid = _backproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    tz = _table_z(Wp, valid)
    X = Wp[..., 0]
    Y = Wp[..., 1]
    Z = Wp[..., 2]
    tall = valid & (Z > tz + 0.07) & (Z < tz + 0.2) & (np.abs(X) < 0.45) & (np.abs(Y) < 0.45)
    if int(tall.sum()) < 50:
        return None
    xs = X[tall]
    ys = Y[tall]
    zs = Z[tall]
    edges = np.arange(float(xs.min()) - 0.01, float(xs.max()) + 0.02, 0.01)
    h, _ = np.histogram(xs, bins=edges)
    occ = h > 2
    if not bool(occ.any()):
        return None
    start = int(np.argmax(occ))
    end = len(occ)
    gap = 0
    for i in range(start, len(occ)):
        if occ[i]:
            gap = 0
        else:
            gap += 1
            if gap >= 2:
                end = i - 1
                break
    x_hi = edges[end]
    sel = xs < x_hi
    if int(sel.sum()) < 30:
        return None
    lx, hx = np.percentile(xs[sel], [2, 98])
    ly, hy = np.percentile(ys[sel], [2, 98])
    return [float((lx + hx) / 2), float((ly + hy) / 2), float(np.percentile(zs[sel], 95)),
            float(hx - lx), float(hy - ly)]


# ---------------- rotation helpers ----------------
def _aa2R(v):
    v = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _R2aa_small(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = float(np.linalg.norm(w))
    c = (float(np.trace(R)) - 1.0) / 2.0
    th = math.atan2(s, c)
    if s < 1e-9:
        return np.zeros(3)
    return w / s * th


def _target_R(yaw):
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


def _rot_cmd(tcp, yaw):
    R = _aa2R(np.asarray(tcp)[3:6])
    err = _R2aa_small(_target_R(yaw) @ R.T)
    return np.clip(_KR * err, -1.0, 1.0)


# ---------------- motion helpers ----------------
def _fgap(frame):
    f = np.asarray(frame.fingers, dtype=np.float64)
    return float(f[0] - f[1])


def _move(robot, target, grip, yaw, tol, max_steps, stall=False):
    """P-control the TCP to `target` (world xyz). Returns (reached, steps, final_error)."""
    target = np.asarray(target, dtype=np.float64)
    slow = 0
    prev = None
    k = 0
    err_n = 1.0
    for k in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=np.float64)
        p = tcp[:3]
        err = target - p
        err_n = float(np.linalg.norm(err))
        if err_n < tol:
            return True, k, err_n
        if stall and prev is not None and err_n < 0.012:
            if float(np.linalg.norm(p - prev)) < 0.0003:
                slow += 1
            else:
                slow = 0
            if slow >= 5:
                return False, k, err_n
        prev = p
        a = _KP * err
        m = float(np.max(np.abs(a)))
        if m > 1.0:
            a = a / m
        r = _rot_cmd(tcp, yaw)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return False, k, err_n


def _hold(robot, grip, yaw, n):
    for _ in range(int(n)):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        r = _rot_cmd(tcp, yaw)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), grip)


def _r3(v):
    return [round(float(x), 4) for x in v]


# ---------------- act tools ----------------
def act_pick(robot, x, y, z, yaw=0.0):
    """Open gripper, travel above (x, y), descend vertically to TCP height z, close, lift to carry height.
    z is the grasp TCP height (table_z + 0.002). Returns finger gap (m) after lift and holding flag."""
    yaw = float(yaw)
    hover = float(z) + _HOVER_DZ
    carry = _PRIOR_BASKET[2] + _CARRY_DRIM
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    log = []
    if p[2] < hover - 0.01:
        ok, n, e = _move(robot, [p[0], p[1], hover], -1, yaw, 0.006, 80)
        log.append(["rise", ok, n])
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    if math.hypot(x - p[0], y - p[1]) > 0.03:
        ok, n, e = _move(robot, [x, y, max(float(p[2]), hover)], -1, yaw, 0.008, 200)
        log.append(["travel", ok, n])
    ok, n, e = _move(robot, [x, y, hover], -1, yaw, 0.004, 120)
    log.append(["hover", ok, n])
    ok, n, e = _move(robot, [x, y, float(z)], -1, yaw, 0.0025, 150, stall=True)
    log.append(["descend", ok, n, round(e, 4)])
    _hold(robot, -1, yaw, 3)
    _hold(robot, 1, yaw, 22)
    gap_closed = _fgap(robot.now())
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    ok, n, e = _move(robot, [p[0], p[1], carry], 1, yaw, 0.01, 120)
    log.append(["lift", ok, n])
    _hold(robot, 1, yaw, 3)
    fr = robot.now()
    gap = _fgap(fr)
    holding = bool(0.010 < gap < 0.075)
    return {"holding": holding, "gap_closed": round(gap_closed, 4), "gap": round(gap, 4),
            "tcp": _r3(np.asarray(fr.tcp)[:3]), "log": log}


def act_place(robot, x, y, rim_z, yaw=0.0):
    """Carry the held object at a safe height to above (x, y), descend to rim_z + 0.09, open, then lift."""
    yaw = float(yaw)
    carry = float(rim_z) + _CARRY_DRIM
    release = float(rim_z) + _RELEASE_DRIM
    log = []
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    if p[2] < carry - 0.02:
        ok, n, e = _move(robot, [p[0], p[1], carry], 1, yaw, 0.01, 100)
        log.append(["lift", ok, n])
    ok, n, e = _move(robot, [x, y, carry], 1, yaw, 0.008, 200)
    log.append(["travel", ok, n])
    ok, n, e = _move(robot, [x, y, release], 1, yaw, 0.005, 120)
    log.append(["lower", ok, n])
    gap_before = _fgap(robot.now())
    _hold(robot, -1, yaw, 20)
    ok, n, e = _move(robot, [x, y, carry], -1, yaw, 0.01, 100)
    log.append(["retreat", ok, n])
    fr = robot.now()
    return {"released": True, "gap_before_open": round(gap_before, 4), "gap": round(_fgap(fr), 4),
            "tcp": _r3(np.asarray(fr.tcp)[:3]), "log": log}


def act_survey(robot, x, y, z):
    """Open the gripper and return to the observation (home) pose (x, y, z), lifting first if low; settle."""
    log = []
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    if p[2] < float(z) - 0.02:
        ok, n, e = _move(robot, [p[0], p[1], float(z)], -1, 0.0, 0.01, 100)
        log.append(["rise", ok, n])
    ok, n, e = _move(robot, [x, y, z], -1, 0.0, 0.005, 250)
    log.append(["home", ok, n])
    _hold(robot, -1, 0.0, 8)
    fr = robot.now()
    return {"tcp": _r3(np.asarray(fr.tcp)[:3]), "gap": round(_fgap(fr), 4), "log": log}


def act_wait(robot, n=40):
    """Hold position with the gripper open for n steps (lets the scorer's monitoring period elapse)."""
    _hold(robot, -1, 0.0, int(max(1, min(int(n), 400))))
    return {"waited": int(n)}


# ---------------- decision ----------------
def _parse_order(goal):
    g = str(goal or "").lower()
    names = {"cream": ["cream cheese", "cream"], "chocolate": ["chocolate pudding", "chocolate", "pudding"]}
    found = []
    for name, keys in names.items():
        pos = [g.find(k) for k in keys if g.find(k) >= 0]
        if pos:
            found.append((min(pos), name))
    found.sort()
    order = [n for _, n in found]
    for n in ["cream", "chocolate"]:
        if n not in order:
            order.append(n)
    return order


def _in_basket(xy, basket):
    if basket is None or xy is None:
        return False
    return abs(xy[0] - basket[0]) < 0.075 and abs(xy[1] - basket[1]) < 0.08


def _yaw_for(stat):
    if stat is None:
        return 0.0
    yaw = float(stat["yaw"])
    if stat["elong"] < 1.3 or abs(yaw) < math.radians(8):
        return 0.0
    return max(-1.0, min(1.0, yaw))


_FIELDS = ["stage", "order", "current", "placed", "cream_pos", "chocolate_pos", "basket_pos",
           "home", "table_z", "cream_yaw", "chocolate_yaw", "attempts", "decisions", "waits", "log"]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: start -> pick -> place -> (next object) ... ; survey/re-detect on failures."""
    m = {k: memory.get(k) for k in _FIELDS}
    f = obs.current
    m["decisions"] = int(m["decisions"] or 0) + 1
    if m["order"] is None:
        m["order"] = _parse_order(goal)
    if m["placed"] is None:
        m["placed"] = []
    if m["attempts"] is None:
        m["attempts"] = 0
    if m["waits"] is None:
        m["waits"] = 0
    last_s = str(last or "")
    failed = ("the tool failed" in last_s) or ("-> error" in last_s)
    stage = m["stage"] or "start"
    notes = []

    def _redetect():
        tz, det = _detect_objects_wrist(f)
        if abs(tz - _PRIOR_TABLE_Z) < 0.05:
            m["table_z"] = float(tz)
        elif m["table_z"] is None:
            m["table_z"] = _PRIOR_TABLE_Z
        seen = {}
        for name in ["cream", "chocolate"]:
            s = det[name]
            if s is not None and not _in_basket(s["center"], m["basket_pos"]):
                m[name + "_pos"] = [s["center"][0], s["center"][1], float(m["table_z"])]
                m[name + "_yaw"] = _yaw_for(s)
                seen[name] = True
            else:
                seen[name] = False
        return seen

    def _pick(name):
        pos = m[name + "_pos"]
        if pos is None:
            pr = _PRIOR[name]
            pos = [pr[0], pr[1], float(m["table_z"] or _PRIOR_TABLE_Z)]
            m[name + "_pos"] = pos
        z = float(m["table_z"] or _PRIOR_TABLE_Z) + _GRASP_DZ
        m["current"] = name
        m["stage"] = "picking"
        return {"tool": "act_pick", "args": {"x": float(pos[0]), "y": float(pos[1]), "z": z,
                                             "yaw": float(m[name + "_yaw"] or 0.0)}}

    def _survey():
        h = m["home"] or _PRIOR_HOME
        return {"tool": "act_survey", "args": {"x": float(h[0]), "y": float(h[1]), "z": float(h[2])}}

    def _next_unplaced():
        for n in m["order"]:
            if n not in m["placed"]:
                return n
        return None

    act = None
    if stage == "start":
        m["home"] = [float(v) for v in np.asarray(f.tcp)[:3]]
        if abs(m["home"][2] - _PRIOR_HOME[2]) > 0.08:
            m["home"] = list(_PRIOR_HOME)
        bk = _detect_basket_agent(f)
        if bk is not None and bk[3] < 0.25 and bk[4] < 0.25:
            m["basket_pos"] = [bk[0], bk[1], bk[2]]
        else:
            m["basket_pos"] = list(_PRIOR_BASKET)
            notes.append("basket prior")
        m["table_z"] = None
        seen = _redetect()
        for n in ["cream", "chocolate"]:
            if not seen[n]:
                notes.append(n + " prior")
        act = _pick(_next_unplaced())
        m["attempts"] = 1
    elif stage == "picking":
        gap = _fgap(f)
        holding = (0.010 < gap < 0.075) and not failed
        if holding:
            b = m["basket_pos"] or _PRIOR_BASKET
            m["stage"] = "placing"
            act = {"tool": "act_place", "args": {"x": float(b[0]), "y": float(b[1]), "rim_z": float(b[2]),
                                                 "yaw": float(m[m["current"] + "_yaw"] or 0.0)}}
            notes.append("holding gap %.3f" % gap)
        else:
            m["stage"] = "surveying"
            act = _survey()
            notes.append("grasp miss gap %.3f" % gap)
    elif stage == "placing":
        if m["current"] is not None and m["current"] not in m["placed"]:
            m["placed"] = list(m["placed"]) + [m["current"]]
        nxt = _next_unplaced()
        if nxt is not None:
            m["attempts"] = 1
            act = _pick(nxt)
        else:
            m["stage"] = "verifying"
            act = _survey()
    elif stage in ("surveying", "verifying"):
        seen = _redetect()
        if stage == "verifying":
            # objects visible on the table outside the basket are not placed
            for n in ["cream", "chocolate"]:
                if seen[n] and n in m["placed"]:
                    m["placed"] = [p for p in m["placed"] if p != n]
                    m["attempts"] = 0
                    notes.append(n + " still on table")
        cur = m["current"]
        if stage == "surveying" and cur in seen and not seen[cur]:
            # object not visible outside the basket: treat as placed (fell in / out of view)
            notes.append(cur + " not seen")
            if cur not in m["placed"]:
                m["placed"] = list(m["placed"]) + [cur]
            m["attempts"] = 0
        nxt = _next_unplaced()
        if nxt is not None:
            if nxt == cur and stage == "surveying":
                m["attempts"] = int(m["attempts"]) + 1
            else:
                m["attempts"] = 1
            if m["attempts"] > _MAX_PICK_TRIES + 1:
                # give up on this object, move on
                m["placed"] = list(m["placed"]) + [nxt]
                notes.append("gave up " + nxt)
                nxt = _next_unplaced()
                m["attempts"] = 1
        if nxt is not None:
            act = _pick(nxt)
        else:
            m["stage"] = "waiting"
            m["waits"] = int(m["waits"]) + 1
            act = {"tool": "act_wait", "args": {"n": 40}}
    else:  # waiting
        m["waits"] = int(m["waits"]) + 1
        if m["waits"] % 3 == 0:
            m["stage"] = "verifying"
            act = _survey()
        else:
            m["stage"] = "waiting"
            act = {"tool": "act_wait", "args": {"n": 40}}
    m["log"] = "d%d %s->%s %s" % (m["decisions"], stage, m["stage"], "; ".join(notes))
    return {"memory": m, "act": act}
