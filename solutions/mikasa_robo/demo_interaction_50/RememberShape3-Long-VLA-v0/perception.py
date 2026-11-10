"""Perception helpers for RememberShape3: blue-object segmentation, 3D back-projection
and shape classification (sphere / cylinder / cube) from the wrist RGB-D camera."""

SHAPES = ["sphere", "cylinder", "cube"]
# Prototype top-surface features measured on the 50 assigned demos (wrist camera,
# initial robot pose): rg = radius of gyration (mm) of the top-surface points
# (z > zmax-6mm), sq = (rg / r98)^2.  See task_analysis.md.
_PROTO = {
    "sphere": (10.2, 0.47, 0.45, 0.05),
    "cylinder": (14.35, 0.517, 0.30, 0.012),
    "cube": (16.55, 0.423, 0.45, 0.017),
}


def _blue_mask(rgb):
    im = rgb.astype(np.int32)
    r = im[..., 0]
    g = im[..., 1]
    b = im[..., 2]
    return (b > r + 40) & (b > g + 40)


def _components(mask, min_pixels=15):
    """4-connected components of a boolean mask; list of (rows, cols) arrays."""
    vv, uu = np.nonzero(mask)
    if len(vv) == 0:
        return []
    W = mask.shape[1]
    keys = (vv * W + uu).tolist()
    remaining = set(keys)
    out = []
    for k in keys:
        if k not in remaining:
            continue
        remaining.discard(k)
        stack = [k]
        comp = []
        while stack:
            p = stack.pop()
            comp.append(p)
            r = p // W
            c = p - r * W
            for q in (p - W, p + W, p - 1 if c > 0 else -1, p + 1 if c < W - 1 else -1):
                if q >= 0 and q in remaining:
                    remaining.discard(q)
                    stack.append(q)
        if len(comp) >= min_pixels:
            a = np.array(comp)
            out.append((a // W, a % W))
    return out


def _backproject(depth, K, E, vv, uu):
    z = depth[vv, uu].astype(np.float64) / 1000.0
    ok = z > 0
    vv = vv[ok]
    uu = uu[ok]
    z = z[ok]
    x = (uu + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vv + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R


def _blob_features(P):
    zmax = float(np.percentile(P[:, 2], 95))
    top = P[P[:, 2] > zmax - 0.006]
    if len(top) < 5:
        top = P
    xy = top[:, :2] - top[:, :2].mean(0)
    r = np.sqrt((xy ** 2).sum(1))
    r98 = float(np.percentile(r, 98)) + 1e-9
    rg = float(np.sqrt((r ** 2).mean()))
    top4 = P[P[:, 2] > zmax - 0.004]
    if len(top4) < 3:
        top4 = top
    c = top4[:, :2].mean(0)
    return {
        "x": float(c[0]),
        "y": float(c[1]),
        "ztop": zmax,
        "rg": rg * 1000.0,
        "sq": (rg / r98) ** 2,
        "tf": float(len(top)) / len(P),
        "n": int(len(P)),
    }


def _shape_cost(f, shape):
    rg0, sq0, tf_unused, sqs = _PROTO[shape]
    return ((f["rg"] - rg0) / 0.6) ** 2 + ((f["sq"] - sq0) / sqs) ** 2


def _classify(f):
    costs = [_shape_cost(f, s) for s in SHAPES]
    order = sorted(range(3), key=lambda i: costs[i])
    best = SHAPES[order[0]]
    margin = costs[order[1]] - costs[order[0]]
    return best, margin, costs


def _detect_objects(frame, cam="wrist", min_pixels=15):
    """Blue objects visible in one camera of a frame, with world features."""
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    out = []
    for vv, uu in _components(_blue_mask(rgb), min_pixels):
        P = _backproject(depth, K, E, vv, uu)
        if len(P) < 10:
            continue
        f = _blob_features(P)
        f["u"] = float(uu.mean())
        f["v"] = float(vv.mean())
        f["border"] = bool(uu.min() <= 1 or vv.min() <= 1 or uu.max() >= rgb.shape[1] - 2 or vv.max() >= rgb.shape[0] - 2)
        # ignore blobs that are not resting on the table (e.g. reflections)
        if f["ztop"] < 0.01 or f["ztop"] > 0.08:
            continue
        s, m, c = _classify(f)
        f["shape"] = s
        f["margin"] = m
        f["costs"] = c
        out.append(f)
    out.sort(key=lambda f: f["y"])
    return out


def _blue_count(frame, cam="wrist"):
    return int(_blue_mask(getattr(frame, cam + "_rgb")).sum())


def _assign_choices(objs):
    """Jointly assign distinct shapes to (up to 3) choice objects by minimum total cost."""
    n = len(objs)
    if n == 0:
        return []
    perms = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if len({a, b, c}) == 3:
                    perms.append((a, b, c))
    best = None
    for p in perms:
        cost = sum(objs[i]["costs"][p[i]] for i in range(min(n, 3)))
        if best is None or cost < best[0]:
            best = (cost, p)
    labels = [SHAPES[best[1][i]] for i in range(min(n, 3))]
    for i in range(3, n):
        labels.append(objs[i]["shape"])
    return labels


def _detect_scene(frame, front_weight=0.5):
    """Merge wrist and front detections of the blue objects (matched within 30 mm in xy).
    Wrist features dominate when the wrist blob is complete; front-camera features are added
    (weight front_weight) and are the only evidence for objects missing/cut off in the wrist view."""
    w = _detect_objects(frame, "wrist")
    fr = _detect_objects(frame, "front")
    used = set()
    out = []
    for fo in fr:
        best = None
        for i, wo in enumerate(w):
            if i in used:
                continue
            d = math.hypot(wo["x"] - fo["x"], wo["y"] - fo["y"])
            if d < 0.04 and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            used.add(best[1])
            wo = w[best[1]]
            if wo["border"]:
                costs = [front_weight * c for c in fo["costs"]]
                src = "front(wrist-border)"
                x, y = fo["x"], fo["y"]
            else:
                costs = [wo["costs"][k] + front_weight * fo["costs"][k] for k in range(3)]
                src = "wrist+front"
                x, y = wo["x"], wo["y"]
            ztop = wo["ztop"]
        else:
            costs = [front_weight * c for c in fo["costs"]]
            src = "front"
            x, y, ztop = fo["x"], fo["y"], fo["ztop"]
        out.append({"x": x, "y": y, "ztop": ztop, "costs": costs, "src": src,
                    "rg": fo["rg"], "sq": fo["sq"], "n": fo["n"]})
    for i, wo in enumerate(w):
        if i not in used:
            out.append({"x": wo["x"], "y": wo["y"], "ztop": wo["ztop"], "costs": list(wo["costs"]),
                        "src": "wrist", "rg": wo["rg"], "sq": wo["sq"], "n": wo["n"]})
    for o in out:
        order = sorted(range(3), key=lambda k: o["costs"][k])
        o["shape"] = SHAPES[order[0]]
        o["margin"] = o["costs"][order[1]] - o["costs"][order[0]]
    out.sort(key=lambda o: o["y"])
    return out
