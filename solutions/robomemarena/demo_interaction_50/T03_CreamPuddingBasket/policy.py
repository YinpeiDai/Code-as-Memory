"""Task program: cream into the basket, then chocolate pudding into the SAME basket."""

GRASP_ABOVE_TABLE = 0.004   # demos grasp at table + 3..5 mm (TCP z)
RELEASE_ABOVE_RIM = 0.09    # demos release at rim (0.563) + 0.092
Z_TRAVEL = 0.78
HELD_GAP = 0.008            # finger gap (sum) above this after lift => something held
OPEN_GAP = 0.06             # finger gap above this after a pick => gripper never closed (blocked)
MAX_PICK = 4                # pick attempts per object before stronger variation
MAX_LOOK = 3                # re-observation attempts when an object is not seen
DEFAULT_HOME = [-0.058, 0.0, 0.671]
BASKET_HALF_RIM = 0.085     # half size of a basket rim footprint (agentview ext ~0.155 x 0.167)
FINGER_REACH = 0.075        # needed y clearance from a basket rim edge to the TCP (R012: 0.069 still blocked)
CLEAR_MARGIN = 0.005
MAX_GRASP_SHIFT = 0.015     # box half width 0.0195 vs open finger 0.04: off-centre grasp still encloses it
BOX_HALF_LEN = 0.038
DIAG_DROP_Y = None          # diagnostic only; None in production

FIELDS = ("stage", "order", "idx", "basket", "baskets", "home", "target", "attempts",
          "blocked", "pushes", "looks", "placed", "table_z", "log")
MAX_PUSH = 2


def _parse_order(goal):
    """Object order from the instruction; defaults to cream then chocolate."""
    g = (goal or "").lower()
    names = []
    for key, kind in (("cream", "cream"), ("chocolate", "chocolate"), ("pudding", "chocolate")):
        p = g.find(key)
        if p >= 0:
            names.append((p, kind))
    names.sort()
    order = []
    for _, k in names:
        if k not in order:
            order.append(k)
    for k in ("cream", "chocolate"):
        if k not in order:
            order.append(k)
    return order


def _init_memory(memory, goal, f):
    m = {k: memory.get(k) if isinstance(memory, dict) else None for k in FIELDS}
    if m["stage"] is None:
        m["stage"] = "observe"
    if m["order"] is None:
        m["order"] = _parse_order(goal)
    if m["idx"] is None:
        m["idx"] = 0
    if m["home"] is None:
        tcp = [float(v) for v in f.tcp[:3]]
        # the first frame is the observation pose; fall back to the demo home if too low
        m["home"] = tcp if tcp[2] > 0.6 else list(DEFAULT_HOME)
    for k in ("attempts", "blocked", "pushes", "looks"):
        if m[k] is None:
            m[k] = 0
    if m["placed"] is None:
        m["placed"] = []
    if m["baskets"] is None:
        m["baskets"] = []
    if m["log"] is None:
        m["log"] = ""
    return m


def _choose_basket(baskets, obj_xy):
    """The basket in the objects' row: nearest basket centre to the first object."""
    if not baskets:
        return None
    if obj_xy is None:
        return min(baskets, key=lambda b: b["center"][0])
    return min(baskets, key=lambda b: (b["center"][0] - obj_xy[0]) ** 2 + (b["center"][1] - obj_xy[1]) ** 2)


def _ret(m, tool, args):
    return {"memory": {k: m[k] for k in FIELDS}, "act": {"tool": tool, "args": args}}


def _look_pose(m):
    h = m["home"]
    # widen the view a little on repeated looks
    dz = 0.05 * min(m["looks"], 2)
    return {"x": float(h[0]), "y": float(h[1]), "z": float(h[2] + dz), "gripper": -1.0}


def _y_limits(gx, baskets, reach):
    """Allowed TCP y interval so open fingers stay clear of every basket rim overlapping in x.

    baskets: [x, y, rim_z, side] with side +1 when the basket lies on the +y side of the object."""
    lo, hi = -1.0, 1.0
    for b in baskets:
        if abs(gx - b[0]) < BASKET_HALF_RIM + 0.02:
            if b[3] > 0:
                hi = min(hi, b[1] - BASKET_HALF_RIM - reach - CLEAR_MARGIN)
            else:
                lo = max(lo, b[1] + BASKET_HALF_RIM + reach + CLEAR_MARGIN)
    return lo, hi


def _plan_grasp(obj, baskets, blocked):
    """Grasp point (shifted up to MAX_GRASP_SHIFT away from nearby baskets) or None if a push is needed."""
    gx, gy = obj["center"]
    reach = FINGER_REACH + 0.008 * blocked
    tagged = [[b[0], b[1], b[2], 1 if b[1] > gy else -1] for b in baskets]
    lo, hi = _y_limits(gx, tagged, reach)
    ty = min(max(gy, lo), hi)
    if lo > hi or abs(ty - gy) > MAX_GRASP_SHIFT:
        return None, gy - ty
    return [gx, ty], ty - gy


def _plan_push(obj, baskets, tz):
    """Slide along x to the nearest x (with 2 cm margin) where a centred grasp is clear of all baskets."""
    gx, gy = obj["center"]

    def clear(nx):
        for xx in (nx - 0.02, nx, nx + 0.02):
            p, _ = _plan_grasp({"center": [xx, gy]}, baskets, 0)
            if p is None or abs(p[1] - gy) > 0.002:
                return False
        return True

    best = None
    for k in range(1, 40):
        for sgn in (-1.0, 1.0):
            nx = gx + sgn * 0.01 * k
            if -0.25 <= nx <= 0.25 and clear(nx):
                best = nx
                break
        if best is not None:
            break
    if best is None:
        return None
    sgn = 1.0 if best > gx else -1.0
    x_start = gx - sgn * (BOX_HALF_LEN + 0.03)
    x_end = x_start + (best - gx) + sgn * 0.02
    return {"x_start": float(x_start), "y": float(gy), "x_end": float(x_end), "z_push": float(tz + 0.006)}


def perceive_decide(obs, goal, memory, last):
    """Detect, pick, place and verify each object in order; the basket is memorised at the start."""
    f = obs.current
    m = _init_memory(memory or {}, goal, f)
    order = m["order"]

    if m["stage"] == "done" or m["idx"] >= len(order):
        m["stage"] = "done"
        m["log"] = "all objects verified placed; waiting for the scorer"
        return _ret(m, "act_wait", {"steps": 40, "gripper": -1.0})

    kind = order[m["idx"]]
    gap = _finger_gap(f)

    if m["stage"] == "picked":
        if gap > OPEN_GAP:
            # descent stalled on an obstacle; gripper never closed
            m["blocked"] = m["blocked"] + 1
            m["attempts"] = m["attempts"] + 1
            m["stage"] = "observe"
            m["log"] = "pick of %s blocked above target (blocked=%d)" % (kind, m["blocked"])
            return _ret(m, "act_goto", _look_pose(m))
        if gap > HELD_GAP and m["basket"] is not None:
            b = m["basket"]
            m["stage"] = "released"
            m["log"] = "holding %s gap=%.4f" % (kind, gap)
            ry = b[1]
            if DIAG_DROP_Y is not None and kind == order[0] and m["attempts"] == 0:
                ry = DIAG_DROP_Y
            return _ret(m, "act_place", {"x": float(b[0]), "y": float(ry),
                                          "z_release": float(b[2] + RELEASE_ABOVE_RIM), "z_travel": Z_TRAVEL})
        m["attempts"] = m["attempts"] + 1
        m["stage"] = "observe"
        m["log"] = "grasp of %s missed gap=%.4f attempt=%d" % (kind, gap, m["attempts"])
        return _ret(m, "act_goto", _look_pose(m))

    if m["stage"] == "released":
        # object released above the basket; go to the observation pose to verify the table is clear
        m["stage"] = "verify"
        m["log"] = "released %s gap=%.4f" % (kind, gap)
        return _ret(m, "act_goto", _look_pose(m))

    # stages observe / verify / pushed: need a view from the observation pose
    tcp = np.array(f.tcp, dtype=float)[:3]
    h = np.array(m["home"], dtype=float)
    if float(np.linalg.norm(tcp[:2] - h[:2])) > 0.03 or tcp[2] < h[2] - 0.03:
        m["log"] = "moving to observation pose"
        return _ret(m, "act_goto", _look_pose(m))

    scene = _scene(f, m["basket"])
    objs = scene["objects"]
    tz = scene["table_z_wrist"] if scene["table_z_wrist"] is not None else m["table_z"]
    if tz is None:
        tz = 0.43
    m["table_z"] = float(tz)
    obj = _best(objs, kind)

    if m["basket"] is None:
        first = _best(objs, order[0])
        b = _choose_basket(scene["baskets"], first["center"] if first else None)
        if b is not None:
            m["basket"] = [float(b["center"][0]), float(b["center"][1]), float(b["rim_z"])]
            m["baskets"] = [[float(c["center"][0]), float(c["center"][1]), float(c["rim_z"])]
                            for c in scene["baskets"]]

    if m["stage"] == "verify":
        if obj is None:
            m["placed"] = list(m["placed"]) + [kind]
            m["idx"] = m["idx"] + 1
            m["attempts"] = 0
            m["blocked"] = 0
            m["pushes"] = 0
            m["looks"] = 0
            m["log"] = "%s no longer on table -> placed" % kind
            if m["idx"] >= len(order):
                m["stage"] = "done"
                return _ret(m, "act_wait", {"steps": 40, "gripper": -1.0})
            kind = order[m["idx"]]
            obj = _best(objs, kind)
        else:
            m["attempts"] = m["attempts"] + 1
            m["log"] = "%s still on table after place -> retry" % kind
        m["stage"] = "observe"

    if obj is None or m["basket"] is None:
        m["looks"] = m["looks"] + 1
        m["log"] = "missing %s (basket=%s) look=%d" % (kind, m["basket"] is not None, m["looks"])
        if m["looks"] > MAX_LOOK:
            m["looks"] = 0
        return _ret(m, "act_goto", _look_pose(m))

    m["target"] = [float(obj["center"][0]), float(obj["center"][1]), float(obj["top"])]
    m["looks"] = 0
    baskets = m["baskets"] or [m["basket"]]
    g, shift = _plan_grasp(obj, baskets, m["blocked"])
    if g is None or m["blocked"] >= 2:
        push = _plan_push(obj, baskets, tz)
        if push is not None and m["pushes"] < MAX_PUSH:
            m["stage"] = "pushed"
            m["pushes"] = m["pushes"] + 1
            m["blocked"] = 0
            m["log"] = "%s too close to a basket (shift %.3f); pushing along x" % (kind, shift)
            return _ret(m, "act_push", push)
        g = [obj["center"][0], obj["center"][1] + max(-MAX_GRASP_SHIFT, min(MAX_GRASP_SHIFT, -shift))]
    # retries nudge the grasp point along the box's long axis (x) to vary the contact
    nudges = [0.0, 0.006, -0.006, 0.0]
    dx = nudges[m["attempts"] % len(nudges)]
    m["stage"] = "picked"
    if not m["log"].startswith("%s " % kind):
        m["log"] = "%s pick at shift %.3f attempt %d" % (kind, g[1] - obj["center"][1], m["attempts"])
    return _ret(m, "act_pick", {"x": float(g[0] + dx), "y": float(g[1]), "z_grasp": float(tz + GRASP_ABOVE_TABLE),
                                "z_lift": Z_TRAVEL})
