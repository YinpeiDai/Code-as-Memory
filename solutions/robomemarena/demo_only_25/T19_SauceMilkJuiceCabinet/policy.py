"""T19 SauceMilkJuiceCabinet: move tomato sauce, milk and orange juice from the cabinet holding them
(cabinet1) onto the top of the other cabinet (cabinet2), in the order named by the goal.

Perception: agentview RGB-D back-projected to world coordinates with the supplied calibration; cabinet
tops found as the dominant horizontal plane; objects segmented as coloured points above the source top.
Control: top-down grasps with the initial gripper orientation held, proportional OSC delta control."""

OBJECTS = ("sauce", "milk", "juice")
DEFAULT_ORDER = ["sauce", "milk", "juice"]
# expert release offsets (x, y) relative to destination-top centre, per object
PLACE_OFFSETS = {"sauce": (-0.026, 0.053), "milk": (0.062, 0.035), "juice": (-0.030, -0.030)}
# perception bias correction (expert grasp minus detected median), per object
DETECT_BIAS = {"sauce": (-0.006, 0.0), "milk": (0.0, -0.003), "juice": (0.0, 0.003)}
HOVER_Z = 0.97
MIN_PIX = 8
MAX_ATTEMPTS = 3
GRASP_WIDTH = 0.015
OPEN_WIDTH = 0.074  # wider than any held object (max seen 0.062); fully open is ~0.08


# ---------------------------------------------------------------- geometry helpers
def _cloud(frame):
    """World xyz (256x256x3) and validity mask from agentview depth."""
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    D = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * D, (v - K[1, 2]) / K[1, 1] * D, D], -1)
    P = (pc - E[:, 3]) @ E[:, :3]
    return P, D > 0


def _hsv(rgb):
    x = np.asarray(rgb, dtype=float) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1)
    mn = x.min(-1)
    d = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    s = np.where(mx > 0, d / (mx + 1e-9), 0.0)
    return h, s, mx


def _class_masks(rgb):
    h, s, v = _hsv(rgb)
    juice = (s > 0.4) & (h >= 22) & (h < 70) & (v > 0.3)
    milk = (s > 0.35) & ((h < 20) | (h > 340)) & (v > 0.33)
    sauce = ((s > 0.3) & (h >= 70) & (h < 170) & (v > 0.12)) | \
            ((s > 0.08) & (s < 0.6) & (h >= 180) & (h < 260) & (v > 0.15))
    return {"sauce": sauce, "milk": milk, "juice": juice}


def _rotvec_to_mat(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _mat_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _fwidth(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0] - f[1]))


# ---------------------------------------------------------------- perception
def _find_cabinets(frame):
    """Return dict with top z and the two cabinet top boxes [xmin,xmax,ymin,ymax] (sorted by y), or None."""
    P, valid = _cloud(frame)
    valid = valid & (np.abs(P[..., 0]) < 0.4) & (np.abs(P[..., 1]) < 0.7)
    z = P[..., 2]
    zs = z[valid & (z > 0.5) & (z < 0.95)]
    if zs.size < 200:
        return None
    hist, edges = np.histogram(zs, bins=np.arange(0.5, 0.95, 0.005))
    zt = float(edges[int(np.argmax(hist))] + 0.0025)
    top = valid & (np.abs(z - zt) < 0.006)
    ys = P[..., 1][top]
    xs = P[..., 0][top]
    if ys.size < 100:
        return None
    srt = np.sort(ys)
    gaps = np.diff(srt)
    k = int(np.argmax(gaps))
    if gaps[k] < 0.03:
        return None
    split = 0.5 * (srt[k] + srt[k + 1])
    boxes = []
    for side in (ys < split, ys >= split):
        if side.sum() < 30:
            return None
        xx, yy = xs[side], ys[side]
        boxes.append([float(np.percentile(xx, 2)), float(np.percentile(xx, 98)),
                      float(np.percentile(yy, 2)), float(np.percentile(yy, 98))])
    # refine the top height with the median of the plane points
    zt = float(np.median(z[top]))
    return {"top": zt, "boxes": boxes}


def _above_box_mask(P, valid, box, top, margin=0.02):
    return valid & (P[..., 0] > box[0] - margin) & (P[..., 0] < box[1] + margin) & \
        (P[..., 1] > box[2] - margin) & (P[..., 1] < box[3] + margin) & \
        (P[..., 2] > top + 0.009) & (P[..., 2] < top + 0.2)


def _detect_objects(frame, box, top):
    """Detect sauce/milk/juice above a cabinet top box. Returns {name: {xy, ztop, n}}."""
    P, valid = _cloud(frame)
    m = _above_box_mask(P, valid, box, top)
    cls = _class_masks(frame.agentview_rgb)
    out = {}
    for name in OBJECTS:
        mm = m & cls[name]
        n = int(mm.sum())
        if n < MIN_PIX:
            out[name] = {"n": n}
            continue
        pts = P[mm]
        med = np.median(pts[:, :2], 0)
        # keep the main blob (reject stray pixels farther than 5 cm from the median)
        d = np.linalg.norm(pts[:, :2] - med, axis=1)
        keep = pts[d < 0.05]
        if keep.shape[0] >= MIN_PIX:
            pts = keep
            med = np.median(pts[:, :2], 0)
        b = DETECT_BIAS[name]
        out[name] = {"n": int(pts.shape[0]),
                     "xy": [float(med[0] + b[0]), float(med[1] + b[1])],
                     "ztop": float(np.percentile(pts[:, 2], 95))}
    return out


def _parse_order(goal):
    g = str(goal or "").lower()
    keys = {"sauce": ["tomato sauce", "sauce", "ketchup"], "milk": ["milk"], "juice": ["orange juice", "juice"]}
    found = []
    for name, words in keys.items():
        idx = [g.find(w) for w in words if g.find(w) >= 0]
        if idx:
            found.append((min(idx), name))
    found.sort()
    order = [n for _, n in found]
    for n in DEFAULT_ORDER:
        if n not in order:
            order.append(n)
    return order


def _source_index(frame, cab):
    """Which cabinet box holds the objects (more coloured pixels above it)."""
    P, valid = _cloud(frame)
    cls = _class_masks(frame.agentview_rgb)
    anyc = cls["sauce"] | cls["milk"] | cls["juice"]
    counts = [int((_above_box_mask(P, valid, b, cab["top"]) & anyc).sum()) for b in cab["boxes"]]
    if counts[0] == counts[1]:
        return 0, counts
    return (0 if counts[0] > counts[1] else 1), counts


# ---------------------------------------------------------------- decision
def _mem_defaults(memory):
    m = dict(memory or {})
    fields = {"stage": None, "order": None, "obj_index": None, "attempts": None, "top_z": None,
              "src_box": None, "dst_center": None, "home_rot": None, "target": None, "placed": None,
              "note": None}
    for k in fields:
        if k not in m:
            m[k] = None
    if m["stage"] is None:
        m["stage"] = "init"
    if m["obj_index"] is None:
        m["obj_index"] = 0
    if m["attempts"] is None:
        m["attempts"] = 0
    if m["placed"] is None:
        m["placed"] = []
    if m["note"] is None:
        m["note"] = ""
    return m


def _last_dropped(last):
    if isinstance(last, dict):
        return bool(last.get("dropped", False))
    if isinstance(last, str):
        t = last.replace(" ", "")
        return '"dropped":true' in t
    return False


def _hold(m, note):
    m["note"] = note
    rot = m["home_rot"] if m["home_rot"] is not None else [math.pi, 0.0, 0.0]
    return {"memory": m, "act": {"tool": "act_hold", "args": {"steps": 100, "rot": rot}}}


def perceive_decide(obs, goal, memory, last):
    """Plan the next step of the sauce -> milk -> juice cabinet transfer from the latest frame and memory."""
    m = _mem_defaults(memory)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    if m["order"] is None:
        m["order"] = _parse_order(goal)
    if m["home_rot"] is None:
        m["home_rot"] = [float(v) for v in tcp[3:6]]
    rot = m["home_rot"]

    if m["stage"] == "init":
        # lift straight up so the arm does not pollute the cabinet / object view
        m["stage"] = "survey"
        m["note"] = "lift to survey pose"
        return {"memory": m, "act": {"tool": "act_move", "args": {
            "x": float(tcp[0]), "y": float(tcp[1]), "z": HOVER_Z, "rot": rot, "gripper": -1.0,
            "max_steps": 120}}}

    if m["stage"] == "survey":
        cab = _find_cabinets(cur)
        if cab is None:
            m["attempts"] = int(m["attempts"]) + 1
            if m["attempts"] > 3:
                return _hold(m, "cabinets not found")
            m["note"] = "cabinets not found, retry"
            return {"memory": m, "act": {"tool": "act_move", "args": {
                "x": -0.1, "y": 0.0, "z": HOVER_Z + 0.03, "rot": rot, "gripper": -1.0, "max_steps": 100}}}
        si, counts = _source_index(cur, cab)
        src = cab["boxes"][si]
        dst = cab["boxes"][1 - si]
        m["top_z"] = cab["top"]
        m["src_box"] = [[src[0], src[2]], [src[1], src[3]]]
        m["dst_center"] = [0.5 * (dst[0] + dst[1]), 0.5 * (dst[2] + dst[3]), cab["top"]]
        m["attempts"] = 0
        m["stage"] = "pick"
        m["note"] = "cabinets top=%.3f src_counts=%s" % (cab["top"], counts)
        # fall through to pick planning on this same frame

    if m["stage"] == "check_grasp":
        w = _fwidth(cur)
        name = m["order"][int(m["obj_index"])]
        if GRASP_WIDTH < w < OPEN_WIDTH:
            dc = m["dst_center"]
            off = PLACE_OFFSETS.get(name, (0.0, 0.0))
            top = float(m["top_z"])
            m["stage"] = "check_place"
            m["note"] = "holding %s width=%.3f, placing" % (name, w)
            return {"memory": m, "act": {"tool": "act_place", "args": {
                "x": float(dc[0] + off[0]), "y": float(dc[1] + off[1]), "z_release": top + 0.10,
                "z_hover": HOVER_Z, "rot": rot, "max_steps": 600}}}
        m["attempts"] = int(m["attempts"]) + 1
        m["stage"] = "pick"
        m["note"] = "grasp of %s failed width=%.3f" % (name, w)

    if m["stage"] == "check_place" and _last_dropped(last):
        # the object slipped out before release: look for it again on the source cabinet
        m["attempts"] = int(m["attempts"]) + 1
        m["stage"] = "pick"
        m["note"] = "object dropped in transit; re-detecting"

    if m["stage"] == "check_place":
        name = m["order"][int(m["obj_index"])]
        placed = list(m["placed"])
        placed.append(name)
        m["placed"] = placed
        m["obj_index"] = int(m["obj_index"]) + 1
        m["attempts"] = 0
        m["stage"] = "pick"

    if m["stage"] == "pick":
        while int(m["obj_index"]) < len(m["order"]):
            name = m["order"][int(m["obj_index"])]
            if int(m["attempts"]) >= MAX_ATTEMPTS:
                m["obj_index"] = int(m["obj_index"]) + 1
                m["attempts"] = 0
                continue
            sb = m["src_box"]
            box = [sb[0][0], sb[1][0], sb[0][1], sb[1][1]]
            top = float(m["top_z"])
            det = _detect_objects(cur, box, top)
            d = det.get(name, {})
            if "xy" not in d:
                # not visible: one retry from a clear viewpoint, then skip this object
                m["attempts"] = int(m["attempts"]) + 1
                if int(m["attempts"]) >= MAX_ATTEMPTS:
                    m["obj_index"] = int(m["obj_index"]) + 1
                    m["attempts"] = 0
                    continue
                m["note"] = "%s not detected (n=%d); moving to clear view" % (name, d.get("n", 0))
                dc = m["dst_center"]
                return {"memory": m, "act": {"tool": "act_move", "args": {
                    "x": float(dc[0]) - 0.05, "y": float(dc[1]), "z": HOVER_Z + 0.03, "rot": rot,
                    "gripper": -1.0, "max_steps": 150}}}
            zg = max(top + 0.042, float(d["ztop"]) - 0.035)
            zg = min(zg, top + 0.12)
            m["target"] = [d["xy"][0], d["xy"][1], zg]
            m["stage"] = "check_grasp"
            m["note"] = "pick %s at (%.3f,%.3f,%.3f) n=%d" % (name, d["xy"][0], d["xy"][1], zg, d["n"])
            return {"memory": m, "act": {"tool": "act_pick", "args": {
                "x": d["xy"][0], "y": d["xy"][1], "z_grasp": zg, "z_hover": HOVER_Z, "rot": rot,
                "max_steps": 600}}}
        m["stage"] = "done"

    if m["stage"] == "done":
        return _hold(m, "all objects handled; holding")

    return _hold(m, "unknown stage")


# ---------------------------------------------------------------- control
def _step_towards(robot, target, rot, gripper, vmax):
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    err = np.asarray(target, dtype=float) - tcp[:3]
    a = np.clip(err * 20.0, -vmax, vmax)
    R = _rotvec_to_mat(tcp[3:6])
    R0 = _rotvec_to_mat(rot)
    rv = _mat_to_rotvec(R0 @ R.T)
    r = np.clip(rv * 1.5, -0.3, 0.3)
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(gripper))
    return err


def _move(robot, target, rot, gripper, vmax=1.0, tol=0.004, max_steps=200, stall=False):
    """Servo to target; returns (steps, reached)."""
    zs = []
    for i in range(int(max_steps)):
        f = robot.now()
        p = np.asarray(f.tcp, dtype=float)[:3]
        e = np.asarray(target, dtype=float) - p
        if float(np.linalg.norm(e)) < tol:
            return i, True
        zs.append(float(p[2]))
        if stall and len(zs) > 8 and abs(zs[-1] - zs[-8]) < 0.0007 and float(np.linalg.norm(e[:2])) < 0.006 \
                and e[2] < 0 and abs(e[2]) < 0.03:
            return i, True
        _step_towards(robot, target, rot, gripper, vmax)
    return int(max_steps), False


def _travel(robot, x, y, z_hover, rot, gripper, budget):
    """Rise to hover height, then move horizontally at hover."""
    used = 0
    p = np.asarray(robot.now().tcp, dtype=float)[:3]
    if p[2] < z_hover - 0.03:
        n, _ = _move(robot, [p[0], p[1], z_hover], rot, gripper, tol=0.015, max_steps=min(120, budget))
        used += n
    n, ok = _move(robot, [x, y, z_hover], rot, gripper, tol=0.006, max_steps=max(1, min(200, budget - used)))
    return used + n, ok


def act_move(robot, x, y, z, rot, gripper=-1.0, max_steps=150):
    """Servo the gripper to world (x, y, z) holding orientation `rot` (axis-angle); gripper sign held."""
    n, ok = _move(robot, [x, y, z], rot, gripper, tol=0.005, max_steps=max_steps)
    f = robot.now()
    return {"steps": n, "reached": ok, "tcp": [float(v) for v in f.tcp[:3]], "width": _fwidth(f)}


def act_pick(robot, x, y, z_grasp, z_hover, rot, max_steps=600):
    """Top-down grasp at (x, y, z_grasp): hover above, descend (open), close, lift back to z_hover."""
    used, ok1 = _travel(robot, x, y, z_hover, rot, -1.0, max_steps)
    # settle the opening of the fingers before descending
    for _ in range(4):
        _step_towards(robot, [x, y, z_hover], rot, -1.0, 1.0)
        used += 1
    n, ok2 = _move(robot, [x, y, z_grasp + 0.06], rot, -1.0, vmax=1.0, tol=0.008, max_steps=120)
    used += n
    n, ok3 = _move(robot, [x, y, z_grasp], rot, -1.0, vmax=0.4, tol=0.003, max_steps=120, stall=True)
    used += n
    zreach = float(robot.now().tcp[2])
    for _ in range(20):
        _step_towards(robot, [x, y, zreach], rot, 1.0, 0.3)
        used += 1
    n, ok4 = _move(robot, [x, y, z_hover], rot, 1.0, vmax=1.0, tol=0.01, max_steps=150)
    used += n
    f = robot.now()
    return {"steps": used, "reached_hover": ok1, "reached_grasp": ok3, "z_at_close": zreach,
            "width": _fwidth(f), "tcp": [float(v) for v in f.tcp[:3]]}


def act_place(robot, x, y, z_release, z_hover, rot, max_steps=600):
    """Carry the held object at z_hover to above (x, y), descend to z_release, open, lift back."""
    used, ok1 = _travel(robot, x, y, z_hover, rot, 1.0, max_steps)
    if _fwidth(robot.now()) < GRASP_WIDTH:
        f = robot.now()
        return {"steps": used, "dropped": True, "width_before_open": _fwidth(f),
                "tcp": [float(v) for v in f.tcp[:3]]}
    n, ok2 = _move(robot, [x, y, z_release], rot, 1.0, vmax=0.6, tol=0.004, max_steps=150, stall=True)
    used += n
    w_before = _fwidth(robot.now())
    zr = float(robot.now().tcp[2])
    for _ in range(20):
        _step_towards(robot, [x, y, zr], rot, -1.0, 0.3)
        used += 1
    n, ok3 = _move(robot, [x, y, z_hover], rot, -1.0, vmax=1.0, tol=0.01, max_steps=120)
    used += n
    f = robot.now()
    return {"steps": used, "dropped": bool(w_before < GRASP_WIDTH), "reached_above": ok1,
            "reached_release": ok2, "width_before_open": w_before,
            "tcp": [float(v) for v in f.tcp[:3]]}


def act_hold(robot, steps=40, rot=None):
    """Hold the current pose with the gripper open for `steps` steps (lets the checker settle)."""
    p = np.asarray(robot.now().tcp, dtype=float)
    r = rot if rot is not None else [float(v) for v in p[3:6]]
    for _ in range(int(steps)):
        _step_towards(robot, p[:3], r, -1.0, 0.3)
    return {"held": int(steps)}
