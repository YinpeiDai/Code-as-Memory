"""Perception helpers: RGB-D deprojection, basket and small-item detection, pose math."""

ITEM_COLOURS = {
    "butter": "red",
    "popcorn": "blue",
}
TABLE_XY = (-0.45, 0.45, -0.45, 0.45)


def _get(frame, name):
    return getattr(frame, name, None)


def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _mat2aa(r):
    c = (float(np.trace(r)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-3:
        # near pi: axis from the dominant column of (R + I) / 2 = k k^T
        b = (r + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(b)))
        k = b[:, i] / math.sqrt(max(float(b[i, i]), 1e-12))
        w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        if float(np.dot(k, w)) < 0:
            k = -k
        return k / float(np.linalg.norm(k)) * th
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * s)
    return w * th


def _down_rot(yaw):
    """Top-down hand orientation rotated by yaw about world z; yaw 0 = fingers open along world y."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ np.diag([1.0, -1.0, -1.0])


def _yaw_of(tcp):
    r = _aa2mat(np.asarray(tcp)[3:6])
    return math.atan2(r[1, 0], r[0, 0])


def _rot_error(tcp, yaw=0.0):
    """World-frame axis-angle that turns the current hand orientation into the top-down orientation at yaw."""
    return _mat2aa(_down_rot(yaw) @ _aa2mat(np.asarray(tcp)[3:6]).T)


def _wrap_half(a):
    """Wrap an angle to (-pi/2, pi/2] (the two-finger hand is symmetric under 180 deg)."""
    while a > math.pi / 2:
        a -= math.pi
    while a <= -math.pi / 2:
        a += math.pi
    return a


def _cloud(frame, cam):
    d = _get(frame, cam + "_depth")
    k = _get(frame, cam + "_K")
    e = _get(frame, cam + "_extrinsic")
    rgb = _get(frame, cam + "_rgb")
    if d is None or k is None or e is None or rgb is None:
        return None, None, None
    d = np.asarray(d).astype(float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - k[0, 2]) / k[0, 0] * d
    y = (v - k[1, 2]) / k[1, 1] * d
    c = np.stack([x, y, d], -1).reshape(-1, 3)
    e = np.asarray(e, dtype=float)
    p = (c - e[:, 3]) @ e[:, :3]
    valid = d.reshape(-1) > 0.01
    return p, np.asarray(rgb).reshape(-1, 3).astype(float), valid


def _components(ij):
    """8-connected components of a list of integer grid cells; returns lists of cell indices."""
    cells = {}
    for n, c in enumerate(ij):
        cells[(int(c[0]), int(c[1]))] = n
    seen = set()
    comps = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        comp = []
        while stack:
            a = stack.pop()
            comp.append(cells[a])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    b = (a[0] + di, a[1] + dj)
                    if b in cells and b not in seen:
                        seen.add(b)
                        stack.append(b)
        comps.append(comp)
    return comps


def _table_z(p, valid):
    x0, x1, y0, y1 = TABLE_XY
    m = valid & (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
    m = m & (p[:, 2] > 0.2) & (p[:, 2] < 0.7)
    if m.sum() < 200:
        return None
    z = p[m, 2]
    hist, edges = np.histogram(z, bins=np.arange(0.2, 0.7, 0.002))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


def _detect_baskets(frame, table_z=None):
    """Basket candidates from the agentview cloud: tall (6-20 cm) clusters on the table."""
    p, rgb, valid = _cloud(frame, "agentview")
    if p is None:
        return {"table_z": table_z, "baskets": []}
    tz = _table_z(p, valid)
    if tz is None:
        tz = table_z if table_z is not None else 0.43
    x0, x1, y0, y1 = TABLE_XY
    m = valid & (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
    m = m & (p[:, 2] > tz + 0.06) & (p[:, 2] < tz + 0.2)
    tcp = _get(frame, "tcp")
    if tcp is not None:
        tcp = np.asarray(tcp)
        near = (np.hypot(p[:, 0] - tcp[0], p[:, 1] - tcp[1]) < 0.10) & (p[:, 2] > tcp[2] - 0.15)
        m = m & ~near
    pts = p[m]
    out = []
    if len(pts) < 20:
        return {"table_z": tz, "baskets": out}
    cell = 0.01
    ij = np.floor(pts[:, :2] / cell).astype(int)
    uniq, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    keep = cnt >= 1
    comps = _components(uniq[keep])
    idx_keep = np.flatnonzero(keep)
    for comp in comps:
        cells = idx_keep[np.array(comp)]
        sel = np.isin(inv, cells)
        q = pts[sel]
        if len(q) < 30:
            continue
        xmin, xmax = float(q[:, 0].min()), float(q[:, 0].max())
        ymin, ymax = float(q[:, 1].min()), float(q[:, 1].max())
        if xmax - xmin < 0.06 or ymax - ymin < 0.06 or xmax - xmin > 0.35 or ymax - ymin > 0.35:
            continue
        out.append({
            "center": [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, float(np.percentile(q[:, 2], 95))],
            "box": [xmin, xmax, ymin, ymax],
            "n": int(len(q)),
        })
    return {"table_z": tz, "baskets": out}


def _in_boxes(xy, boxes, margin):
    m = np.zeros(len(xy), dtype=bool)
    for b in boxes:
        m |= (xy[:, 0] > b[0] - margin) & (xy[:, 0] < b[1] + margin) & (xy[:, 1] > b[2] - margin) & (xy[:, 1] < b[3] + margin)
    return m


def _detect_items(frame, table_z, basket_boxes, cam="wrist"):
    """Small flat objects lying on the table (0.4-4 cm tall) outside basket footprints."""
    p, rgb, valid = _cloud(frame, cam)
    if p is None or table_z is None:
        return []
    m = valid & (p[:, 2] > table_z + 0.004) & (p[:, 2] < table_z + 0.04)
    x0, x1, y0, y1 = TABLE_XY
    m = m & (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
    if basket_boxes:
        m = m & ~_in_boxes(p[:, :2], basket_boxes, 0.025)
    pts = p[m]
    cols = rgb[m]
    if len(pts) < 5:
        return []
    cell = 0.005 if cam == "wrist" else 0.01
    ij = np.floor(pts[:, :2] / cell).astype(int)
    uniq, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    comps = _components(uniq)
    out = []
    min_pts = 40 if cam == "wrist" else 3
    for comp in comps:
        sel = np.isin(inv, np.array(comp))
        q = pts[sel]
        if len(q) < min_pts:
            continue
        ex = float(q[:, 0].max() - q[:, 0].min())
        ey = float(q[:, 1].max() - q[:, 1].min())
        if max(ex, ey) < 0.012 or max(ex, ey) > 0.12:
            continue
        c = cols[sel]
        xy = q[:, :2] - np.mean(q[:, :2], 0)
        cov = xy.T @ xy / max(1, len(xy))
        w, vec = np.linalg.eigh(cov)
        major = vec[:, 1]
        theta = _wrap_half(math.atan2(float(major[1]), float(major[0])))
        pl = xy @ major
        pw = xy @ vec[:, 0]
        rb = float(np.mean(c[:, 0] - c[:, 2]))
        br = float(np.mean(c[:, 2] - c[:, 0]))
        out.append({
            "xy": [float(np.mean(q[:, 0])), float(np.mean(q[:, 1]))],
            "top": float(np.percentile(q[:, 2], 90)),
            "size": [ex, ey],
            "axis": theta,
            "length": float(pl.max() - pl.min()),
            "width": float(pw.max() - pw.min()),
            "n": int(len(q)),
            "rgb": [float(v) for v in np.mean(c, 0)],
            "red": rb,
            "blue_frac": float(np.mean((c[:, 2] - c[:, 0]) > 15)),
            "red_frac": float(np.mean((c[:, 0] - c[:, 2]) > 40)),
        })
    return out


def _match_item(items, name, near=None, radius=0.06):
    """Pick the detection that matches an item name by colour; optional proximity prior."""
    colour = ITEM_COLOURS.get(name, None)
    best, score = None, -1e9
    for it in items:
        if colour == "red":
            ok = it["red"] > 35 or it["red_frac"] > 0.35
            s = it["red"]
        elif colour == "blue":
            ok = it["red"] < 30 and it["blue_frac"] > 0.1
            s = it["blue_frac"] * 100 - abs(it["red"])
        else:
            ok, s = True, 0.0
        if not ok:
            continue
        if near is not None:
            d = math.hypot(it["xy"][0] - near[0], it["xy"][1] - near[1])
            if d > radius:
                continue
            s = s - 500 * d
        if s > score:
            best, score = it, s
    return best


def _parse_order(goal):
    g = (goal or "").lower()
    found = []
    for name in ITEM_COLOURS:
        i = g.find(name)
        if i >= 0:
            found.append((i, name))
    found.sort()
    order = [n for _, n in found]
    return order if order else ["butter", "popcorn"]


def _choose_basket(baskets, ref_xy):
    if not baskets:
        return None
    best, bd = None, 1e9
    for b in baskets:
        c = b["center"]
        d = math.hypot(c[0] - ref_xy[0], c[1] - ref_xy[1])
        if d < bd:
            best, bd = b, d
    return best


def _basket_contents(frame, box, table_z):
    """Coloured pixel counts inside the basket footprint seen by the wrist camera (evidence only)."""
    p, rgb, valid = _cloud(frame, "wrist")
    if p is None or box is None:
        return {"red": 0, "blue": 0}
    m = valid & _in_boxes(p[:, :2], [box], -0.01) & (p[:, 2] > table_z) & (p[:, 2] < table_z + 0.12)
    c = rgb[m]
    red = int(np.sum((c[:, 0] - c[:, 2] > 50) & (c[:, 0] > 90)))
    blue = int(np.sum((c[:, 2] - c[:, 0] > 30) & (c[:, 2] > 80)))
    return {"red": red, "blue": blue}


HAND_HALF_ALONG = 0.11   # half extent of hand+open fingers along the finger axis (m)
HAND_HALF_ACROSS = 0.035


def _hand_overlap(x, y, yaw, boxes, margin=0.01):
    """Overlap area (m^2, approx.) between the hand footprint at (x, y, yaw) and basket footprints."""
    ax = np.array([-math.sin(yaw), math.cos(yaw)])   # finger axis direction in world xy
    bx = np.array([math.cos(yaw), math.sin(yaw)])
    total = 0.0
    for b in boxes or []:
        gx = np.linspace(b[0] - margin, b[1] + margin, 24)
        gy = np.linspace(b[2] - margin, b[3] + margin, 24)
        px, py = np.meshgrid(gx, gy)
        d = np.stack([px.reshape(-1) - x, py.reshape(-1) - y], 1)
        inside = (np.abs(d @ ax) < HAND_HALF_ALONG) & (np.abs(d @ bx) < HAND_HALF_ACROSS)
        cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
        total += float(np.sum(inside)) * cell
    return total


def _grasp_yaw(det, x, y, boxes, avoid=None):
    """Grasp yaw: close across the item's short axis unless the hand would overlap a basket; then try the
    perpendicular yaw if the item is short enough for the open fingers."""
    base = 0.0
    length = None
    if det is not None:
        base = _wrap_half(float(det.get("axis", 0.0)))
        length = det.get("length")
        if det.get("width") is not None and length is not None and length - det["width"] < 0.008:
            base = 0.0  # near-square blob: axis meaningless
    cands = [base]
    if length is None or length < 0.078:  # must fit the 8 cm open fingers (butter measures ~7.5 cm; R017 held it)
        cands.append(_wrap_half(base + math.pi / 2))
    best, bo = None, 1e9
    for k, yv in enumerate(cands):
        o = _hand_overlap(x, y, yv, boxes)
        if avoid is not None and abs(_wrap_half(yv - avoid)) < 0.3:
            o += 1.0
        o += 1e-6 * k
        if o < bo:
            best, bo = yv, o
    return best, bo
