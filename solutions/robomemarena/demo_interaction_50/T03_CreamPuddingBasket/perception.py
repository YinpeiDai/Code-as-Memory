"""Perception helpers: RGB-D deprojection, table/basket/object detection."""

TABLE_BAND = 0.005      # metres above table that count as an object
OBJ_MAX_H = 0.06        # objects are low boxes (cream ~1.2 cm, pudding ~2.1 cm)
BASKET_HALF = 0.09      # half footprint of a basket used to exclude its pixels


def _deproject(depth, K, E):
    """World points (H,W,3) and valid mask from an optical-axis depth image in mm."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    return pw, z > 0


def _table_z(pw, ok):
    """Mode of world z over valid pixels (the table top is the dominant plane)."""
    z = pw[..., 2][ok]
    z = z[(z > 0.2) & (z < 1.2)]
    if len(z) < 100:
        return None
    hist, edges = np.histogram(z, bins=np.arange(0.2, 1.2, 0.002))
    i = int(np.argmax(hist))
    sel = z[(z >= edges[i] - 0.002) & (z <= edges[i + 1] + 0.002)]
    return float(np.median(sel))


def _components(mask, min_px):
    """4-connected components of a boolean image; list of (N,2) pixel arrays."""
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=int)
    out = []
    pts = np.argwhere(mask)
    for p in pts:
        v0, u0 = int(p[0]), int(p[1])
        if lab[v0, u0] >= 0:
            continue
        k = len(out)
        lab[v0, u0] = k
        stack = [(v0, u0)]
        mem = []
        while stack:
            a, b = stack.pop()
            mem.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = a + da, b + db
                if 0 <= x < H and 0 <= y < W and mask[x, y] and lab[x, y] < 0:
                    lab[x, y] = k
                    stack.append((x, y))
        out.append(np.array(mem))
    return [c for c in out if len(c) >= min_px]


def _detect_baskets(frame):
    """Baskets from the agentview: clusters of the highest rim points above the table.

    Returns (table_z, list of dicts {center:[x,y], rim_z, ext:[ex,ey], n})."""
    pw, ok = _deproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    ok = ok & (np.abs(pw[..., 0]) < 0.45) & (np.abs(pw[..., 1]) < 0.55)
    tz = _table_z(pw, ok)
    if tz is None:
        return None, []
    m = ok & (pw[..., 2] > tz + 0.03) & (pw[..., 2] < tz + 0.19)
    P = pw[m]
    if len(P) < 30:
        return tz, []
    top = float(np.percentile(P[:, 2], 99.5))
    R = P[P[:, 2] > top - 0.012]
    # cluster rim points on a 1 cm grid
    cells = {}
    for p in R:
        key = (int(math.floor(p[0] / 0.01)), int(math.floor(p[1] / 0.01)))
        cells.setdefault(key, []).append(p)
    seen = set()
    out = []
    for key in list(cells.keys()):
        if key in seen:
            continue
        seen.add(key)
        stack = [key]
        group = []
        while stack:
            c = stack.pop()
            group.extend(cells[c])
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    n = (c[0] + dx, c[1] + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        G = np.array(group)
        lo = np.percentile(G[:, :2], 2, axis=0)
        hi = np.percentile(G[:, :2], 98, axis=0)
        ext = hi - lo
        if len(G) >= 40 and ext[0] > 0.06 and ext[1] > 0.06:
            out.append({"center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
                        "rim_z": float(np.median(G[:, 2])), "ext": [float(ext[0]), float(ext[1])],
                        "n": int(len(G))})
    return tz, out


def _detect_objects(frame, baskets):
    """Low boxes on the table in the wrist view, classified by colour.

    Returns (table_z, list of dicts {kind, center:[x,y], top, ext, n, rgb})."""
    pw, ok = _deproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    tz = _table_z(pw, ok)
    if tz is None:
        return None, []
    z = pw[..., 2]
    m = ok & (z > tz + TABLE_BAND) & (z < tz + OBJ_MAX_H)
    for b in baskets or []:
        cx, cy = b["center"]
        m = m & ~((np.abs(pw[..., 0] - cx) < BASKET_HALF) & (np.abs(pw[..., 1] - cy) < BASKET_HALF))
    rgb = frame.wrist_rgb.astype(float)
    out = []
    for c in _components(m, 40):
        P = pw[c[:, 0], c[:, 1]]
        lo = np.percentile(P[:, :2], 3, axis=0)
        hi = np.percentile(P[:, :2], 97, axis=0)
        ext = hi - lo
        if ext[0] > 0.14 or ext[1] > 0.14 or max(ext[0], ext[1]) < 0.02:
            continue
        col = rgb[c[:, 0], c[:, 1]].mean(0)
        if col[2] - col[0] > 10 and col[2] - col[1] > 8:
            kind = "cream"
        elif col[0] - col[2] > 12 and col[0] - col[1] > 8:
            kind = "chocolate"
        else:
            kind = "unknown"
        # long-axis yaw from a PCA of the footprint (logged, not acted on)
        Q = P[:, :2] - P[:, :2].mean(0)
        w, V = np.linalg.eigh(Q.T @ Q)
        ax = V[:, 1]
        yaw = float(math.degrees(math.atan2(ax[1], ax[0])))
        out.append({"kind": kind, "center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
                    "top": float(np.percentile(P[:, 2], 90)), "ext": [float(ext[0]), float(ext[1])],
                    "n": int(len(c)), "rgb": [float(col[0]), float(col[1]), float(col[2])], "yaw": yaw})
    return tz, out


def _best(objs, kind):
    c = [o for o in objs if o["kind"] == kind]
    if not c:
        return None
    return max(c, key=lambda o: o["n"])


def _scene(f, known_basket=None):
    tz_a, baskets = _detect_baskets(f)
    excl = list(baskets)
    if known_basket is not None:
        excl.append({"center": [known_basket[0], known_basket[1]]})
    tz_w, objs = _detect_objects(f, excl)
    return {"table_z_agent": tz_a, "table_z_wrist": tz_w, "baskets": baskets, "objects": objs}


def perceive_scene(obs):
    """Detect baskets (agentview) and cream/chocolate boxes (wrist view) in the latest frame."""
    f = obs.current
    tz_a, baskets = _detect_baskets(f)
    tz_w, objs = _detect_objects(f, baskets)
    return {"table_z_agent": tz_a, "table_z_wrist": tz_w, "baskets": baskets, "objects": objs}
