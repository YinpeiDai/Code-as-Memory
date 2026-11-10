"""Perception helpers: agentview RGB-D -> world points -> microwave / butter / chocolate anchors."""

TABLE_Z_NOMINAL = 0.431


def _frame_get(frame, name, default=None):
    return getattr(frame, name, default)


def _world_points(frame):
    """Return (P[N,3] world xyz, RGB[N,3] float, uv[N,2]) for valid agentview depth pixels."""
    d = np.asarray(_frame_get(frame, "agentview_depth"), dtype=float) / 1000.0
    K = np.asarray(_frame_get(frame, "agentview_K"), dtype=float)
    E = np.asarray(_frame_get(frame, "agentview_extrinsic"), dtype=float)
    rgb = np.asarray(_frame_get(frame, "agentview_rgb"), dtype=float)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    m = d > 0.05
    z = d[m]
    x = (u[m] - K[0, 2]) / K[0, 0] * z
    y = (v[m] - K[1, 2]) / K[1, 1] * z
    C = np.stack([x, y, z], -1)
    P = (C - E[:, 3]) @ E[:, :3]
    return P, rgb[m], np.stack([u[m], v[m]], -1)


def _table_height(P):
    sel = (P[:, 2] > 0.30) & (P[:, 2] < 0.60) & (np.abs(P[:, 0]) < 0.45) & (np.abs(P[:, 1]) < 0.7)
    z = P[sel, 2]
    if len(z) < 50:
        return TABLE_Z_NOMINAL
    hist, edges = np.histogram(z, bins=150, range=(0.30, 0.60))
    k = int(np.argmax(hist))
    zc = (edges[k] + edges[k + 1]) / 2
    zz = z[np.abs(z - zc) < 0.004]
    return float(np.median(zz)) if len(zz) else float(zc)


def _components(cells_i, cells_j, shape):
    """4/8-connected components on an occupancy grid. Returns label per occupied cell (same order)."""
    occ = -np.ones(shape, dtype=int)
    n = len(cells_i)
    for k in range(n):
        occ[cells_i[k], cells_j[k]] = k
    label = -np.ones(n, dtype=int)
    cur = 0
    for k in range(n):
        if label[k] >= 0:
            continue
        stack = [k]
        label[k] = cur
        while stack:
            q = stack.pop()
            i, j = cells_i[q], cells_j[q]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < shape[0] and 0 <= b < shape[1]:
                        r = occ[a, b]
                        if r >= 0 and label[r] < 0:
                            label[r] = cur
                            stack.append(r)
        cur += 1
    return label, cur


def _clusters(P, res, x0, y0, nx, ny, min_pts):
    """Cluster points in an XY grid; returns list of index arrays (into P), largest first."""
    ci = ((P[:, 0] - x0) / res).astype(int)
    cj = ((P[:, 1] - y0) / res).astype(int)
    ok = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []
    keys = ci[idx] * ny + cj[idx]
    uk, inv = np.unique(keys, return_inverse=True)
    lab, n = _components(uk // ny, uk % ny, (nx, ny))
    pl = lab[inv]
    out = []
    for c in range(n):
        members = idx[pl == c]
        if len(members) >= min_pts:
            out.append(members)
    out.sort(key=lambda m: -len(m))
    return out


def _perceive_scene(frame):
    """Locate microwave (handle anchor) and the two small boxes on the table from one agentview RGB-D frame."""
    P, RGB, _ = _world_points(frame)
    tz = _table_height(P)
    res = {"table_z": tz, "ok": True, "notes": []}
    # ---------- microwave: largest raised cluster with a flat top 0.15-0.30 m above table
    sel = (P[:, 2] > tz + 0.03) & (P[:, 2] < tz + 0.32) & (np.abs(P[:, 0]) < 0.5) & (P[:, 1] > -0.75) & (P[:, 1] < 0.85)
    Q = P[sel]
    cl = _clusters(Q, 0.01, -0.5, -0.75, 100, 160, 200)
    best = None
    for m in cl:
        pts = Q[m]
        top = pts[:, 2].max()
        tp = pts[pts[:, 2] > top - 0.01]
        ext = np.percentile(tp[:, :2], 98, 0) - np.percentile(tp[:, :2], 2, 0)
        if top - tz > 0.12 and ext[0] > 0.15 and ext[1] > 0.12 and len(tp) > 150:
            if best is None or len(tp) > best[1]:
                best = (pts, len(tp), tp)
    if best is None:
        res["ok"] = False
        res["notes"].append("microwave not found")
        res["handle"] = [0.149, 0.428, 0.582]
    else:
        pts, _, tp = best
        x99 = float(np.percentile(tp[:, 0], 99))
        yc = float((np.percentile(tp[:, 1], 1) + np.percentile(tp[:, 1], 99)) / 2)
        res["handle"] = [x99 - 0.0218, yc - 0.1193, 0.582 + (tz - TABLE_Z_NOMINAL)]
        res["mw_top"] = float(tp[:, 2].max())
        res["mw_box"] = [float(np.percentile(tp[:, 0], 1)), x99, float(np.percentile(tp[:, 1], 1)), float(np.percentile(tp[:, 1], 99))]
    # ---------- small objects: 4-50 mm above table, outside microwave footprint
    sel = (P[:, 2] > tz + 0.0045) & (P[:, 2] < tz + 0.06) & (P[:, 0] > -0.3) & (P[:, 0] < 0.45) & (P[:, 1] > -0.6) & (P[:, 1] < 0.6)
    if "mw_box" in res:
        bx = res["mw_box"]
        inside = (P[:, 0] > bx[0] - 0.06) & (P[:, 0] < bx[1] + 0.06) & (P[:, 1] > bx[2] - 0.06) & (P[:, 1] < bx[3] + 0.04)
        sel = sel & ~inside
    Q = P[sel]
    C = RGB[sel]
    cl = _clusters(Q, 0.01, -0.3, -0.6, 75, 120, 15)
    objs = []
    for m in cl[:6]:
        pts = Q[m]
        col = C[m].mean(0)
        top = pts[:, 2].max()
        tp = pts[pts[:, 2] > top - 0.004]
        cen = tp.mean(0)
        objs.append({"xy": [float(cen[0]), float(cen[1])], "top": float(top - tz), "n": int(len(m)),
                     "rgb": [float(col[0]), float(col[1]), float(col[2])],
                     "red": float(col[0] - col[2])})
    res["objects"] = objs
    # butter: reddest/orange low box; chocolate: darker, taller box
    cand = [o for o in objs if o["top"] < 0.045 and o["n"] < 600]
    if len(cand) >= 2:
        cand2 = sorted(cand, key=lambda o: -o["n"])[:2]
        # score: redness high -> butter ; height high -> chocolate
        s0 = cand2[0]["red"] - 2000.0 * cand2[0]["top"]
        s1 = cand2[1]["red"] - 2000.0 * cand2[1]["top"]
        b, c = (cand2[0], cand2[1]) if s0 > s1 else (cand2[1], cand2[0])
        res["butter"] = [b["xy"][0] - 0.0040, b["xy"][1] - 0.0008]
        res["choc"] = [c["xy"][0] - 0.0029, c["xy"][1] - 0.0015]
    else:
        res["ok"] = False
        res["notes"].append("objects not found (%d)" % len(cand))
        res["butter"] = [0.1672, -0.0954]
        res["choc"] = [0.1672, -0.1953]
    return res


def _door_front_count(frame, mw_box):
    """Points on the closed-door plane just in front of the microwave top's front edge.
    Demos: closed 219-259, open 62-71 (fully open), 101-132 while being dragged."""
    if mw_box is None:
        return -1
    P, _, _ = _world_points(frame)
    x0, x1, y1 = mw_box[0], mw_box[1], mw_box[2]
    s = (P[:, 0] > x0 + 0.03) & (P[:, 0] < x1 - 0.03) & (P[:, 1] > y1 - 0.06) & (P[:, 1] < y1 + 0.01) \
        & (P[:, 2] > 0.47) & (P[:, 2] < 0.62)
    return int(s.sum())
