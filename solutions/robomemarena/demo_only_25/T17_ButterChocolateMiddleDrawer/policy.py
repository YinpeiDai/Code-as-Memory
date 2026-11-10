"""T17 Butter/Chocolate -> middle drawer policy (Code-as-Memory).

Pipeline (all constants measured from the 25 assigned demonstrations, see policy.md):
  1. perceive cabinet (agentview RGB-D back-projection) -> middle-drawer handle (hx, hy, hz)
  2. perceive butter / chocolate on the table near their demo locations
  3. open the drawer: horizontal grasp of the handle, pull -y 0.24 m, release, retreat
  4. verify the drawer front moved (depth) ; retry once if not
  5. pick object 1 (top-down), verify fingers not empty, place into open drawer
  6. pick object 2, place into drawer at a second slot
  7. if the episode has not ended: look again at the table, re-pick leftovers, then wait.
"""

TABLE_Z_NOM = 0.4309
HANDLE_DZ = {"middle": 0.176, "top": 0.245, "bottom": 0.105}   # handle z - table z (middle measured; others hypotheses)
HANDLE_DX = -0.127      # handle x - cabinet front-face x (98th pct of dark cabinet points)
HANDLE_DY = 0.0111      # handle y - 1st pct of cabinet y (side face carrying the handles)
OBJ_NOM = {"butter": [0.1092, -0.2026], "chocolate": [-0.2283, -0.3232]}
OBJ_GRASP_DZ = {"butter": 0.0023, "chocolate": 0.0052}  # grasp tcp z - table z
OBJ_HMIN = {"butter": 0.006, "chocolate": 0.012}
OBJ_TOP_OFF = {"butter": [-0.0032, 0.0026], "chocolate": [-0.0093, 0.0046]}  # grasp - top-surface centroid
DROP_REL = [[-0.0027, -0.0958, 0.1195], [-0.0447, -0.0435, 0.133]]  # drop tcp - handle (1st, 2nd object)
PULL_DY = -0.24
RETREAT_DY = -0.35
TRANSIT_Z = 0.78
CARRY_Z = 0.93
PARK = [-0.06, 0.0, 0.85]


def _rv2m(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _m2rv(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        k = d.copy()
        for j in range(3):
            if j != i:
                k[j] = (R[i, j] + R[j, i]) / (4.0 * d[i])
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _target_rot(name):
    if name == "handle":
        return _rv2m([1.5 * math.pi, 0.0, 0.0])
    return _rv2m([math.pi, 0.0, 0.0])


AGENT_K = [[231.76450199, 0.0, 127.5], [0.0, 309.01933598, 127.5], [0.0, 0.0, 1.0]]


def _cloud(frame):
    Kf = getattr(frame, "agentview_K", None)
    K = np.asarray(AGENT_K if Kf is None else Kf, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    D = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    h, w = D.shape
    v, u = np.mgrid[0:h, 0:w]
    pix = np.stack([u.ravel(), v.ravel(), np.ones(h * w)], 0).astype(float)
    cam = (np.linalg.inv(K) @ pix) * D.reshape(1, -1)
    W = (E[:, :3].T @ (cam - E[:, 3:4])).T
    return W.reshape(h, w, 3), D > 0


def _table_z(W, valid):
    m = valid & (W[..., 1] > -0.6) & (W[..., 1] < 0.2) & (W[..., 0] > -0.45) & (W[..., 0] < 0.45)
    z = W[..., 2][m]
    z = z[np.abs(z - TABLE_Z_NOM) < 0.03]
    if len(z) < 200:
        return TABLE_Z_NOM, False
    return float(np.median(z)), True


def _detect_cabinet(frame):
    W, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.max(-1) < 90
    m = valid & dark & (W[..., 1] > 0.2) & (W[..., 1] < 0.9) & (W[..., 2] > 0.47) & (W[..., 2] < 0.85) \
        & (W[..., 0] > -0.3) & (W[..., 0] < 0.6)
    P = W[m]
    if len(P) < 300:
        return None
    xmax = float(np.percentile(P[:, 0], 98))
    P2 = P[P[:, 0] > xmax - 0.25]
    if len(P2) < 200:
        return None
    ymin = float(np.percentile(P2[:, 1], 1))
    return [xmax + HANDLE_DX, ymin + HANDLE_DY, len(P2)]


def _detect_object(frame, name, center, radius=0.06):
    """Top-surface centroid of the object blob near `center`; returns grasp xy or None."""
    W, valid = _cloud(frame)
    zt, _ = _table_z(W, valid)
    hmin = OBJ_HMIN.get(name, 0.008)
    d = np.hypot(W[..., 0] - center[0], W[..., 1] - center[1])
    m = valid & (d < radius) & (W[..., 2] > zt + hmin) & (W[..., 2] < zt + 0.08)
    P = W[m]
    if len(P) < 15:
        return None
    top = P[P[:, 2] > P[:, 2].max() - 0.004]
    if len(top) < 5:
        top = P
    off = OBJ_TOP_OFF.get(name, [0.0, 0.0])
    gx = float(top[:, 0].mean()) + off[0]
    gy = float(top[:, 1].mean()) + off[1]
    return [gx, gy, zt, float(P[:, 2].max() - zt), int(len(P))]


def _drawer_front_rel(frame, handle):
    """5th percentile y of the middle-drawer front band minus the closed handle y."""
    W, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    zc = handle[2]
    m = valid & (rgb.max(-1) < 90) & (W[..., 1] > 0.0) & (W[..., 1] < 0.9) \
        & (W[..., 2] > zc + 0.008) & (W[..., 2] < zc + 0.038) & (np.abs(W[..., 0] - handle[0]) < 0.1)
    P = W[m]
    if len(P) < 30:
        return None
    return float(np.percentile(P[:, 1], 5)) - handle[1]


def _parse_goal(goal):
    g = (goal or "").lower()
    drawer = "middle"
    for name in ("top", "bottom", "middle"):
        if name + " drawer" in g:
            drawer = name
            break
    names = []
    for name in OBJ_NOM:
        i = g.find(name)
        if i >= 0:
            names.append((i, name))
    names = [n for _, n in sorted(names)]
    if len(names) < 2:
        names = ["butter", "chocolate"]
    return drawer, names[:2]


def _wp(p, rot="down", g=-1, v=1.0, tol=0.005, hold=0, rtol=0.04, maxs=200):
    return {"p": [float(p[0]), float(p[1]), float(p[2])], "rot": rot, "g": g, "v": v,
            "tol": tol, "hold": hold, "rtol": rtol, "maxs": maxs}


def _pick_wps(o):
    """Top-down pick of a table object at grasp point o=[x,y,z]: transit, pre-grasp, descend, close, lift."""
    return [_wp([o[0], o[1], TRANSIT_Z], "down", -1, 1.0, 0.012, 0, 0.04, 200),
            _wp([o[0], o[1], o[2] + 0.07], "down", -1, 1.0, 0.006, 0, 0.03, 120),
            _wp([o[0], o[1], o[2]], "down", -1, 0.3, 0.003, 0, 0.03, 120),
            _wp([o[0], o[1], o[2]], "down", 1, 0.3, 1.0, 25, 1.0, 30),
            _wp([o[0], o[1], CARRY_Z], "down", 1, 1.0, 0.02, 0, 1.0, 120)]


def _fingers_open(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def perceive_decide(obs, goal, memory, last):
    """Stage machine for: open the named drawer, put object 1 then object 2 inside."""
    mem = dict(memory)
    for k in ("stage", "drawer", "order", "handle", "grab", "obj1", "obj2", "retries", "log", "done"):
        if k not in mem:
            mem[k] = None
    if mem["stage"] is None:
        mem["stage"] = "init"
    if mem["retries"] is None:
        mem["retries"] = 0
    if mem["done"] is None:
        mem["done"] = []
    if mem["log"] is None:
        mem["log"] = ""
    fr = obs.current
    tcp = np.asarray(fr.tcp, dtype=float)
    stage = mem["stage"]
    act = {"tool": "act_hold", "args": {"steps": 40}}

    if stage == "init":
        drawer, order = _parse_goal(goal)
        mem["drawer"] = drawer
        mem["order"] = order
        zt, okt, cab = TABLE_Z_NOM, False, None
        try:
            W, valid = _cloud(fr)
            zt, okt = _table_z(W, valid)
            cab = _detect_cabinet(fr)
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            mem["log"] = "perception error %s" % str(exc)[:80]
        if cab is None or abs(cab[0] - 0.137) > 0.12 or abs(cab[1] - 0.452) > 0.12:
            cab = [0.137, 0.452, 0]   # demo mean fallback
        hz = zt + HANDLE_DZ.get(drawer, HANDLE_DZ["middle"])
        mem["handle"] = [cab[0], cab[1], hz]
        objs = []
        for name in order:
            nom = OBJ_NOM[name]
            try:
                det = _detect_object(fr, name, nom)
            except (ValueError, TypeError, IndexError, AttributeError):
                det = None
            if det is None or math.hypot(det[0] - nom[0], det[1] - nom[1]) > 0.06:
                det = [nom[0], nom[1], zt]
            objs.append([det[0], det[1], zt + OBJ_GRASP_DZ.get(name, 0.003)])
        mem["obj1"], mem["obj2"] = objs[0], objs[1]
        mem["log"] = "cab=%s table=%.4f ok=%s" % (str([round(c, 4) for c in cab]), zt, okt)
        mem["stage"] = "open_drawer"
        stage = "open_drawer"

    h = mem["handle"]
    if stage == "open_drawer":
        gb = mem["grab"] if mem["grab"] is not None else h
        wps = [_wp([gb[0], gb[1] - 0.08, gb[2]], "handle", -1, 1.0, 0.01, 0, 0.05, 160),
               _wp([gb[0], gb[1], gb[2]], "handle", -1, 0.4, 0.003, 0, 0.03, 120),
               _wp([gb[0], gb[1], gb[2]], "handle", 1, 0.4, 1.0, 28, 1.0, 40),
               _wp([h[0], h[1] + PULL_DY, h[2]], "handle", 1, 0.8, 0.006, 0, 1.0, 150),
               _wp([h[0], h[1] + PULL_DY, h[2]], "handle", -1, 0.4, 1.0, 20, 1.0, 30),
               _wp([h[0], h[1] + RETREAT_DY, h[2]], "handle", -1, 1.0, 0.01, 0, 1.0, 80),
               _wp([h[0], h[1] + RETREAT_DY, TRANSIT_Z], "handle", -1, 1.0, 0.015, 0, 1.0, 80)]
        act = {"tool": "act_path", "args": {"waypoints": wps}}
        mem["stage"] = "check_drawer"
    elif stage == "check_drawer":
        try:
            rel = _drawer_front_rel(fr, h)
        except (ValueError, TypeError, IndexError, AttributeError):
            rel = None
        mem["log"] = "drawer_front_rel=%s" % (None if rel is None else round(rel, 4))
        if rel is not None and rel > -0.10 and mem["retries"] < 2:
            # drawer not (fully) open: grab the handle where the measured front now is, pull again
            mem["retries"] = mem["retries"] + 1
            mem["grab"] = [h[0], h[1] + (rel - 0.018), h[2]]
            mem["stage"] = "open_drawer"
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([h[0], mem["grab"][1] - 0.15, h[2]], "handle", -1, 1.0, 0.01, 0, 0.05, 120)]}}
        else:
            mem["stage"] = "pick1"
            stage = "pick1"
    if stage in ("pick1", "pick2"):
        o = mem["obj1"] if stage == "pick1" else mem["obj2"]
        wps = _pick_wps(o)
        act = {"tool": "act_path", "args": {"waypoints": wps}}
        mem["stage"] = "place1" if stage == "pick1" else "place2"
    elif stage in ("place1", "place2"):
        idx = 0 if stage == "place1" else 1
        width = _fingers_open(fr)
        if width < 0.006:
            # empty grasp: go to the viewing pose and re-detect the object
            mem["retries"] = mem["retries"] + 1
            mem["log"] = "empty grasp (%.4f) on %s" % (width, stage)
            mem["stage"] = "redetect1" if idx == 0 else "redetect2"
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp(PARK, "down", -1, 1.0, 0.02, 10, 0.1, 150)]}}
        else:
            # each object has its own drop slot in the drawer (object 1 deeper, object 2 beside it)
            d = DROP_REL[idx]
            p = [h[0] + d[0], h[1] + d[1], h[2] + d[2]]
            wps = [_wp([p[0], p[1], CARRY_Z], "down", 1, 1.0, 0.012, 0, 0.05, 200),
                   _wp(p, "down", 1, 0.6, 0.006, 0, 0.05, 100),
                   _wp(p, "down", -1, 0.3, 1.0, 22, 1.0, 30),
                   _wp([p[0], p[1], 0.90], "down", -1, 1.0, 0.02, 0, 1.0, 60)]
            act = {"tool": "act_path", "args": {"waypoints": wps}}
            done = [n for n in mem["done"] if n != mem["order"][idx]] + [mem["order"][idx]]
            mem["done"] = done
            remaining = [n for n in mem["order"] if n not in done]
            if not remaining:
                mem["stage"] = "verify"
            else:
                mem["stage"] = "pick1" if remaining[0] == mem["order"][0] else "pick2"
    elif stage in ("redetect1", "redetect2"):
        idx = 0 if stage == "redetect1" else 1
        name = mem["order"][idx]
        o = mem["obj1"] if idx == 0 else mem["obj2"]
        try:
            det = _detect_object(fr, name, [o[0], o[1]], 0.08)
        except (ValueError, TypeError, IndexError, AttributeError):
            det = None
        if det is not None:
            o = [det[0], det[1], o[2]]
        if idx == 0:
            mem["obj1"] = o
        else:
            mem["obj2"] = o
        if mem["retries"] > 4:
            mem["stage"] = "wait"
            act = {"tool": "act_hold", "args": {"steps": 60}}
        else:
            mem["stage"] = "pick1" if idx == 0 else "pick2"
            wps = _pick_wps(o)
            act = {"tool": "act_path", "args": {"waypoints": wps}}
            mem["stage"] = "place1" if idx == 0 else "place2"
    elif stage == "verify":
        # the scorer did not end the episode right after the release: wait, then look from the park pose
        act = {"tool": "act_path", "args": {"waypoints": [
            _wp([tcp[0], tcp[1], 0.92], "down", -1, 1.0, 0.02, 30, 0.1, 60),
            _wp(PARK, "down", -1, 1.0, 0.02, 10, 0.1, 150)]}}
        mem["stage"] = "verify2"
    elif stage == "verify2":
        left = None
        for idx, name in enumerate(mem["order"]):
            o = mem["obj1"] if idx == 0 else mem["obj2"]
            try:
                det = _detect_object(fr, name, [o[0], o[1]], 0.05)
            except (ValueError, TypeError, IndexError, AttributeError):
                det = None
            if det is not None and det[3] < 0.05:
                left = idx
                if idx == 0:
                    mem["obj1"] = [det[0], det[1], o[2]]
                else:
                    mem["obj2"] = [det[0], det[1], o[2]]
                break
        mem["log"] = "verify left=%s" % str(left)
        if left is not None and mem["retries"] < 5:
            mem["retries"] = mem["retries"] + 1
            name = mem["order"][left]
            mem["done"] = [n for n in mem["done"] if n != name]
            o = mem["obj1"] if left == 0 else mem["obj2"]
            wps = _pick_wps(o)
            act = {"tool": "act_path", "args": {"waypoints": wps}}
            mem["stage"] = "place1" if left == 0 else "place2"
        else:
            mem["stage"] = "wait"
            act = {"tool": "act_hold", "args": {"steps": 60}}
    elif stage == "wait":
        act = {"tool": "act_hold", "args": {"steps": 60}}
    out = {k: mem[k] for k in ("stage", "drawer", "order", "handle", "grab", "obj1", "obj2", "retries", "log", "done")}
    return {"memory": out, "act": act}


def act_path(robot, waypoints):
    """Follow waypoints with closed-loop OSC delta control.

    Each waypoint: {"p": [x,y,z] world m, "rot": "down"|"handle", "g": +1 close/-1 open,
    "v": max normalized speed, "tol": position tolerance m, "rtol": rotation tolerance rad,
    "hold": steps to hold after reaching (or immediately if tol>=1), "maxs": step cap}.
    Returns per-waypoint steps/convergence, final tcp and finger positions."""
    report = []
    total = 0
    for wp in waypoints:
        tgt = np.asarray(wp["p"], dtype=float)
        Rt = _target_rot(wp.get("rot", "down"))
        g = 1.0 if float(wp.get("g", -1)) > 0 else -1.0
        vmax = float(wp.get("v", 1.0))
        tol = float(wp.get("tol", 0.005))
        rtol = float(wp.get("rtol", 0.04))
        hold = int(wp.get("hold", 0))
        maxs = int(wp.get("maxs", 200))
        n = 0
        conv = False
        held = 0
        while n < maxs:
            fr = robot.now()
            tcp = np.asarray(fr.tcp, dtype=float)
            e = tgt - tcp[:3]
            er = _m2rv(Rt @ _rv2m(tcp[3:6]).T)
            dist = float(np.linalg.norm(e))
            rerr = float(np.linalg.norm(er))
            if dist < tol and rerr < rtol:
                conv = True
            if conv and held >= hold:
                break
            a = e / 0.05
            m = float(np.max(np.abs(a)))
            if m > vmax:
                a = a * (vmax / m)
            r = er / 0.5
            mr = float(np.max(np.abs(r)))
            if mr > 1.0:
                r = r / mr
            robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
            n += 1
            if conv:
                held += 1
        total += n
        fr = robot.now()
        report.append([n, conv])
    fr = robot.now()
    return {"segments": report, "steps": total, "tcp": [float(x) for x in fr.tcp[:3]],
            "fingers": [float(x) for x in fr.fingers]}


def act_hold(robot, steps=40):
    """Hold the current pose with the gripper open for `steps` native steps (lets the scorer settle)."""
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    tgt = tcp[:3].copy()
    for _ in range(int(steps)):
        cur = np.asarray(robot.now().tcp, dtype=float)
        a = np.clip((tgt - cur[:3]) / 0.05, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, -1.0)
    return {"held": int(steps), "tcp": [float(x) for x in robot.now().tcp[:3]]}
