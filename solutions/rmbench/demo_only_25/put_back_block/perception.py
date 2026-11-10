"""Perception helpers for put_back_block (head RGB-D camera -> world points)."""

TABLE_Z = 0.741


def _world_points(frame, cam, mask):
    """Back-project masked pixels with valid depth to world xyz (N,3)."""
    depth = np.asarray(getattr(frame, cam + "_depth"))
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    m = mask & (depth > 0)
    v, u = np.nonzero(m)
    if len(v) == 0:
        return np.zeros((0, 3))
    z = depth[v, u].astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _red_mask(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (r > 130) & (r - g > 70) & (r - b > 70)


def _blue_mask(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (b > 150) & (b - r > 90) & (g > 100)


def _clusters(P, cell=0.01, min_n=15):
    """Connected components of points on an xy grid (8-neighbour). Returns list of index arrays."""
    if len(P) == 0:
        return []
    ij = np.floor(P[:, :2] / cell).astype(int)
    keys = {}
    for n in range(len(ij)):
        k = (int(ij[n, 0]), int(ij[n, 1]))
        if k in keys:
            keys[k].append(n)
        else:
            keys[k] = [n]
    seen = set()
    out = []
    for k0 in keys:
        if k0 in seen:
            continue
        stack = [k0]
        seen.add(k0)
        members = []
        while stack:
            k = stack.pop()
            members.extend(keys[k])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nk = (k[0] + di, k[1] + dj)
                    if nk in keys and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        if len(members) >= min_n:
            out.append(np.array(members))
    return out


def _describe_red(Q):
    zt = float(np.percentile(Q[:, 2], 95))
    T = Q[Q[:, 2] > zt - 0.006]
    lo = T[:, :2].min(0)
    hi = T[:, :2].max(0)
    mid = (lo + hi) / 2.0
    return {"xy": [float(mid[0]), float(mid[1])], "top": zt, "n": int(len(Q)),
            "ntop": int(len(T)), "ext": [float(hi[0] - lo[0]), float(hi[1] - lo[1])]}


def _scene(frame, cam="head"):
    """Red objects (block/button candidates) and blue mats seen by one camera."""
    rgb = np.asarray(getattr(frame, cam + "_rgb"))
    reds = []
    P = _world_points(frame, cam, _red_mask(rgb))
    if len(P):
        keep = (P[:, 2] > TABLE_Z - 0.01) & (P[:, 2] < 1.3) & (np.abs(P[:, 0]) < 0.6) & (P[:, 1] > -0.5) & (P[:, 1] < 0.4)
        P = P[keep]
        for idx in _clusters(P):
            reds.append(_describe_red(P[idx]))
    mats = []
    B = _world_points(frame, cam, _blue_mask(rgb))
    if len(B):
        keep = (np.abs(B[:, 2] - TABLE_Z) < 0.012) & (np.abs(B[:, 0]) < 0.6) & (B[:, 1] > -0.5) & (B[:, 1] < 0.4)
        B = B[keep]
        for idx in _clusters(B, cell=0.01, min_n=40):
            Q = B[idx]
            c = Q[:, :2].mean(0)
            lo = Q[:, :2].min(0)
            hi = Q[:, :2].max(0)
            mats.append({"xy": [float(c[0]), float(c[1])], "n": int(len(Q)),
                         "lo": [float(lo[0]), float(lo[1])], "hi": [float(hi[0]), float(hi[1])]})
    return {"reds": reds, "mats": mats}


def _dist2(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _in_box(xy, m, pad=0.01):
    return (m["lo"][0] - pad <= xy[0] <= m["hi"][0] + pad) and (m["lo"][1] - pad <= xy[1] <= m["hi"][1] + pad)


def _initial_layout(frame):
    """Locate block, button, original mat and the centre of the mat arrangement.

    Block = red object lying on (inside the footprint of) a blue mat; button = the other
    red object. Centre = midpoint of the farthest-apart pair of free mats (opposite arms
    of the cross), falling back to the mean of the mat centroids."""
    sc = _scene(frame)
    reds = [r for r in sc["reds"] if r["ntop"] >= 8]
    mats = sc["mats"]
    info = {"ok": False, "n_reds": len(reds), "n_mats": len(mats)}
    if not reds:
        return info
    def score(r):
        on = [m for m in mats if _in_box(r["xy"], m)]
        dmin = min([_dist2(r["xy"], m["xy"]) for m in mats]) if mats else 1.0
        return (0 if on else 1, dmin)
    ranked = sorted(reds, key=score)
    block = ranked[0]
    others = [r for r in reds if r is not block]
    button = None
    if others:
        button = max(others, key=lambda r: min([_dist2(r["xy"], m["xy"]) for m in mats]) if mats else r["n"])
    free = [m for m in mats if not _in_box(block["xy"], m, pad=0.0) and _dist2(m["xy"], block["xy"]) > 0.06]
    center = None
    if len(free) >= 2:
        best = None
        for i in range(len(free)):
            for j in range(i + 1, len(free)):
                d = _dist2(free[i]["xy"], free[j]["xy"])
                if best is None or d > best[0]:
                    best = (d, i, j)
        a, b = free[best[1]]["xy"], free[best[2]]["xy"]
        if best[0] > 0.15:
            center = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
    if center is None and mats:
        pts = [m["xy"] for m in free] + [block["xy"]]
        center = [float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))]
    info.update({"ok": center is not None, "block": block, "button": button, "center": center,
                 "free_mats": [m["xy"] for m in free]})
    return info
