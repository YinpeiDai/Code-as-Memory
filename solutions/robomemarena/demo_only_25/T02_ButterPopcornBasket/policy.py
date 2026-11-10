"""Butter -> basket, then popcorn -> same basket (RoboMemArena T02).

Perception: agentview RGB-D deprojected to world points; table height from the
dominant plane; baskets = tall (>2.5 cm) clusters; small flat objects = low
clusters on the table, told apart by colour (butter box is red/orange, popcorn
box is blue/white). Control: closed-loop OSC delta servo on the measured TCP.
"""

TABLE_XY = (-0.35, 0.35, -0.50, 0.50)   # x_min, x_max, y_min, y_max world (m)
CELL = 0.01                             # grid cell for clustering (m)
HOVER_DZ = 0.10                         # hover height above object top
GRASP_BELOW_TOP = 0.011                 # tcp below object top when grasping (demo: 0.444 -> 0.433)
RELEASE_ABOVE_RIM = 0.087               # tcp above basket rim when releasing (demo: 0.568 -> 0.655)
TRANSIT_Z = 0.78                        # carry height (demo 0.76-0.83)
HELD_MIN = 0.004                        # finger joint (m) above which an object is held
DOWN_AA = (math.pi, 0.0, 0.0)           # gripper pointing down, yaw 0 (demo orientation)


# ----------------------------------------------------------------- geometry
def _aa2R(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s > 1e-4:
        return w / (2.0 * s) * th
    # angle near pi: axis from the symmetric part
    B = (R + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(B)))
    axis = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
    return axis / max(np.linalg.norm(axis), 1e-12) * th


def _rot_err(tcp_aa, target_aa):
    """World-frame axis-angle w with exp(w) * R_cur = R_target (demo sign convention)."""
    return _R2aa(_aa2R(target_aa) @ _aa2R(tcp_aa).T)


def _deproject(frame):
    depth = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, depth > 0


def _components(occ):
    """4-connected components of a boolean grid; list of lists of (i, j)."""
    occ = np.asarray(occ, dtype=bool)
    seen = np.zeros(occ.shape, dtype=bool)
    comps = []
    H, W = occ.shape
    cells = np.argwhere(occ)
    for c in cells:
        i0, j0 = int(c[0]), int(c[1])
        if seen[i0, j0]:
            continue
        stack = [(i0, j0)]
        seen[i0, j0] = True
        comp = []
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < H and 0 <= b < W and occ[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        comps.append(comp)
    return comps


def _cluster(P, min_pts):
    """Cluster world points on an xy grid; returns list of index arrays into P."""
    if len(P) == 0:
        return []
    x0, x1, y0, y1 = TABLE_XY
    gi = np.floor((P[:, 0] - x0) / CELL).astype(int)
    gj = np.floor((P[:, 1] - y0) / CELL).astype(int)
    H = int(round((x1 - x0) / CELL)) + 1
    W = int(round((y1 - y0) / CELL)) + 1
    gi = np.clip(gi, 0, H - 1)
    gj = np.clip(gj, 0, W - 1)
    occ = np.zeros((H, W), dtype=bool)
    occ[gi, gj] = True
    lab = -np.ones((H, W), dtype=int)
    out = []
    for n, comp in enumerate(_components(occ)):
        for (i, j) in comp:
            lab[i, j] = n
    ids = lab[gi, gj]
    for n in range(int(ids.max()) + 1 if len(ids) else 0):
        idx = np.flatnonzero(ids == n)
        if len(idx) >= min_pts:
            out.append(idx)
    return out


def _scene(frame):
    """Detect table height, baskets and small objects from one agentview RGB-D frame."""
    pw, ok = _deproject(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    x0, x1, y0, y1 = TABLE_XY
    inxy = ok & (pw[..., 0] > x0) & (pw[..., 0] < x1) & (pw[..., 1] > y0) & (pw[..., 1] < y1)
    z = pw[..., 2]
    cand = inxy & (z > 0.2) & (z < 0.6)
    table_z = None
    if cand.sum() > 200:
        hist, edges = np.histogram(z[cand], bins=200, range=(0.2, 0.6))
        k = int(np.argmax(hist))
        near = cand & (np.abs(z - 0.5 * (edges[k] + edges[k + 1])) < 0.004)
        table_z = float(np.median(z[near]))
    if table_z is None:
        return {"table_z": None, "baskets": [], "objects": []}
    # baskets: tall structures (walls/rim) below the robot hand region
    tall = inxy & (z > table_z + 0.025) & (z < table_z + 0.20)
    P = pw[tall]
    baskets = []
    for idx in _cluster(P, 150):
        Q = P[idx]
        lo = np.percentile(Q, 2, axis=0)
        hi = np.percentile(Q, 98, axis=0)
        ext = hi - lo
        if ext[0] < 0.06 or ext[1] < 0.06 or ext[0] > 0.35 or ext[1] > 0.35:
            continue
        baskets.append({
            "center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
            "rim_z": float(np.percentile(Q[:, 2], 99)),
            "extent": [float(ext[0]), float(ext[1])],
            "n": int(len(idx)),
        })
    # small low objects lying on the table (outside basket footprints)
    low = inxy & (z > table_z + 0.005) & (z < table_z + 0.06)
    for b in baskets:
        cx, cy = b["center"]
        ex, ey = b["extent"]
        low = low & ~((np.abs(pw[..., 0] - cx) < ex / 2 + 0.02) & (np.abs(pw[..., 1] - cy) < ey / 2 + 0.02))
    P = pw[low]
    C = rgb[low]
    objects = []
    for idx in _cluster(P, 12):
        Q = P[idx]
        ext = Q.max(0) - Q.min(0)
        if ext[0] > 0.12 or ext[1] > 0.12:
            continue
        ztop = float(Q[:, 2].max())
        top = Q[Q[:, 2] > ztop - 0.004]
        col = C[idx].mean(0)
        objects.append({
            "xy": [float(top[:, 0].mean()), float(top[:, 1].mean())],
            "top_z": ztop,
            "rgb": [float(col[0]), float(col[1]), float(col[2])],
            "redness": float(col[0] - col[2]),
            "n": int(len(idx)),
        })
    return {"table_z": table_z, "baskets": baskets, "objects": objects}


def _assign(objects, names):
    """Map instruction object names to detections by colour: butter = reddest, popcorn = least red."""
    out = {}
    if not objects:
        return out
    order = sorted(range(len(objects)), key=lambda i: objects[i]["redness"])
    if "butter" in names:
        out["butter"] = objects[order[-1]]
    if "popcorn" in names:
        cand = [i for i in order if not ("butter" in out and objects[i] is out["butter"])]
        if cand:
            out["popcorn"] = objects[cand[0]]
    return out


def _parse_goal(goal):
    g = str(goal or "").lower()
    found = []
    for name in ("butter", "popcorn"):
        k = g.find(name)
        if k >= 0:
            found.append((k, name))
    found.sort()
    names = [n for _, n in found]
    return names if names else ["butter", "popcorn"]


# ------------------------------------------------------------------ control
def _tcp(robot):
    f = robot.now()
    t = np.asarray(f.tcp, dtype=float)
    return t[:3], t[3:6]


def _finger_gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1])) / 2.0


def _servo(robot, target, grip, tol=0.004, umax=1.0, max_steps=250, gain=10.0):
    """Closed-loop move of the TCP to `target` (world xyz), gripper sign held at `grip`.
    Orientation is servoed to DOWN_AA. Stops after 3 consecutive in-tolerance slow steps,
    or when the hand stalls (blocked) for 8 steps once within 3 cm."""
    target = np.asarray(target, dtype=float)
    ok_count = 0
    stall = 0
    prev = None
    steps = 0
    pos, aa = _tcp(robot)
    for steps in range(1, max_steps + 1):
        pos, aa = _tcp(robot)
        err = target - pos
        dist = float(np.linalg.norm(err))
        speed = 0.0 if prev is None else float(np.linalg.norm(pos - prev))
        prev = pos
        if dist < tol and speed < 0.0015:
            ok_count += 1
            if ok_count >= 3:
                return {"reached": True, "steps": steps, "err": dist, "pos": pos}
        else:
            ok_count = 0
        if dist < 0.03 and speed < 0.0003:
            stall += 1
            if stall >= 8:
                return {"reached": False, "stalled": True, "steps": steps, "err": dist, "pos": pos}
        else:
            stall = 0
        u = np.clip(gain * err, -umax, umax)
        w = np.clip(3.0 * _rot_err(aa, DOWN_AA), -0.5, 0.5)
        robot.step(float(u[0]), float(u[1]), float(u[2]),
                   float(w[0]), float(w[1]), float(w[2]), float(grip))
    pos, aa = _tcp(robot)
    return {"reached": False, "steps": steps, "err": float(np.linalg.norm(target - pos)), "pos": pos}


def _hold_gripper(robot, target, grip, min_steps, max_steps):
    """Hold position at `target` while the fingers move; stop once they are stable."""
    target = np.asarray(target, dtype=float)
    gaps = []
    for k in range(max_steps):
        pos, aa = _tcp(robot)
        u = np.clip(10.0 * (target - pos), -0.3, 0.3)
        w = np.clip(3.0 * _rot_err(aa, DOWN_AA), -0.5, 0.5)
        f = robot.step(float(u[0]), float(u[1]), float(u[2]),
                       float(w[0]), float(w[1]), float(w[2]), float(grip))
        gaps.append(_finger_gap(f))
        if k + 1 >= min_steps and len(gaps) >= 4 and max(gaps[-4:]) - min(gaps[-4:]) < 0.0003:
            break
    return gaps[-1] if gaps else _finger_gap(robot.now())


def act_pick(robot, x, y, top_z, table_z):
    """Pick a small box lying on the table at world (x, y) whose top surface is at top_z.
    Opens the gripper, hovers 10 cm above, descends slowly to just below the top, closes,
    and lifts to the carry height. Returns finger gap (m per finger), held flag and TCP."""
    pos, aa = _tcp(robot)
    hover_z = float(top_z) + HOVER_DZ
    grasp_z = max(float(top_z) - GRASP_BELOW_TOP, float(table_z) + 0.002)
    log = {}
    # rise first if low and far away, then travel, then hover
    if pos[2] < hover_z - 0.01 and math.hypot(pos[0] - x, pos[1] - y) > 0.03:
        log["rise"] = _servo(robot, [pos[0], pos[1], hover_z], -1, tol=0.01)["reached"]
        pos, aa = _tcp(robot)
    travel_z = max(float(pos[2]), hover_z)
    log["travel"] = _servo(robot, [x, y, travel_z], -1, tol=0.006)["reached"]
    log["hover"] = _servo(robot, [x, y, hover_z], -1, tol=0.004)["reached"]
    r = _servo(robot, [x, y, grasp_z], -1, tol=0.002, umax=0.5, max_steps=200)
    log["descend"] = r["reached"]
    pos, aa = _tcp(robot)
    gap = _hold_gripper(robot, [x, y, float(pos[2])], 1, 15, 45)
    log["lift"] = _servo(robot, [x, y, TRANSIT_Z], 1, tol=0.01, max_steps=200)["reached"]
    gap = _finger_gap(robot.now())
    pos, aa = _tcp(robot)
    return {"held": bool(gap > HELD_MIN), "gap": gap, "tcp": pos, "log": log}


def act_place(robot, x, y, release_z):
    """Carry the held object at the carry height to above world (x, y), descend to
    release_z, open the gripper, wait for the drop, and lift back to the carry height."""
    pos, aa = _tcp(robot)
    log = {}
    if pos[2] < TRANSIT_Z - 0.02:
        log["rise"] = _servo(robot, [pos[0], pos[1], TRANSIT_Z], 1, tol=0.01)["reached"]
    log["travel"] = _servo(robot, [x, y, TRANSIT_Z], 1, tol=0.006, max_steps=300)["reached"]
    log["descend"] = _servo(robot, [x, y, release_z], 1, tol=0.004, umax=0.7)["reached"]
    pos, aa = _tcp(robot)
    gap_before = _finger_gap(robot.now())
    _hold_gripper(robot, [x, y, float(pos[2])], -1, 20, 40)
    # monitoring period: stay still with the gripper open
    for k in range(15):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    log["lift"] = _servo(robot, [x, y, TRANSIT_Z], -1, tol=0.01, max_steps=200)["reached"]
    pos, aa = _tcp(robot)
    return {"gap_before_release": gap_before, "tcp": pos, "log": log}


def act_wait(robot, steps=60):
    """Hold still with the gripper open for `steps` steps (lets the success check settle)."""
    n = max(1, min(int(steps), 400))
    for k in range(n):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    return {"waited": n}


# ------------------------------------------------------------------ decision
def _detect(frame, names):
    """Scene facts plus a colour-based name assignment of the small table objects."""
    s = _scene(frame)
    named = {}
    for o in s["objects"]:
        if o["n"] < 12:
            continue
        nm = "butter" if o["redness"] > 30.0 else "popcorn"
        if nm in names and (nm not in named or o["n"] > named[nm]["n"]):
            named[nm] = o
    return s, named


def _choose_basket(baskets):
    """Target basket: the one nearest the robot (smallest world x); all 25 demos used it."""
    if not baskets:
        return None
    b = sorted(baskets, key=lambda q: q["center"][0])[0]
    return [b["center"][0], b["center"][1], b["rim_z"]]


def perceive_decide(obs, goal, memory, last):
    """State machine: init (perceive scene, remember basket and object positions) ->
    pick(object i) -> place(object i into the remembered basket) -> verify -> next object.
    Retries a pick when the fingers close empty or the object is still seen on the table."""
    m = dict(memory or {})
    frame = obs.current
    names = m.get("order")
    if not names:
        names = _parse_goal(goal)
    stage = m.get("stage") or "init"
    idx = m.get("idx")
    idx = 0 if idx is None else int(idx)
    tries = m.get("tries")
    tries = 0 if tries is None else int(tries)
    placed = list(m.get("placed") or [])
    positions = m.get("positions")
    positions = list(positions) if positions else [None] * len(names)
    while len(positions) < len(names):
        positions.append(None)
    basket = m.get("basket")
    table_z = m.get("table_z")
    note = ""

    s, named = _detect(frame, names)
    if table_z is None and s["table_z"] is not None:
        table_z = s["table_z"]
    tz = 0.431 if table_z is None else float(table_z)   # fallback: demo table height
    if basket is None and (len(s["baskets"]) >= 2 or stage in ("picking", "placing")):
        basket = _choose_basket(s["baskets"])
    for i, nm in enumerate(names):
        if nm in named and nm not in placed and positions[i] is None:
            o = named[nm]
            positions[i] = [o["xy"][0], o["xy"][1], o["top_z"]]

    gap = _finger_gap(frame)
    act = None
    if stage == "init":
        stage = "pick"
    elif stage == "picking":
        held = gap > HELD_MIN and float(np.asarray(frame.tcp)[2]) > tz + 0.10
        if held:
            stage = "place"
        else:
            tries += 1
            stage = "pick"
            note = "empty grasp"
    elif stage == "placing":
        nm = names[idx] if idx < len(names) else None
        if nm is not None and nm in named and tries < 4:
            # still on the table: pick it again from where it is now
            o = named[nm]
            positions[idx] = [o["xy"][0], o["xy"][1], o["top_z"]]
            tries += 1
            stage = "pick"
            note = nm + " still on table"
        else:
            if nm is not None and nm not in placed:
                placed.append(nm)
            idx += 1
            tries = 0
            stage = "pick" if idx < len(names) else "done"

    if stage == "pick" and idx < len(names):
        nm = names[idx]
        if nm in named:  # refresh with the current view when visible
            o = named[nm]
            positions[idx] = [o["xy"][0], o["xy"][1], o["top_z"]]
        p = positions[idx]
        if tries >= 6:
            stage = "done"
            note = "giving up on " + nm
        elif p is None:
            # not seen yet (occluded or missed): look again after a short wait
            tries += 1
            stage = "pick"
            note = "cannot find " + nm
            act = {"tool": "act_wait", "args": {"steps": 10}}
        else:
            act = {"tool": "act_pick", "args": {"x": float(p[0]), "y": float(p[1]),
                                                 "top_z": float(p[2]), "table_z": tz}}
            stage = "picking"
    elif stage == "place":
        b = basket if basket is not None else [-0.02, 0.20, 0.567]
        act = {"tool": "act_place", "args": {"x": float(b[0]), "y": float(b[1]),
                                              "release_z": float(b[2]) + RELEASE_ABOVE_RIM}}
        stage = "placing"
    if act is None:
        stage = "done"
        act = {"tool": "act_wait", "args": {"steps": 100}}

    mem = {
        "stage": stage,
        "order": list(names),
        "idx": int(idx),
        "tries": int(tries),
        "placed": placed,
        "positions": positions,
        "basket": basket,
        "table_z": None if table_z is None else float(table_z),
        "note": note,
    }
    return {"memory": mem, "act": act}
