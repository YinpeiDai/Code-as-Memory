"""Shape classification, cue reading, candidate fusion (front + wrist) and target choice."""

# Prototypes measured from the 50 assigned demos (top-surface height above table, m;
# top-surface horizontal area, m^2).  cube 4.1 cm tall, sphere 4.0 cm, flat shapes 1.6 cm.
SHAPE_PROTO = {
    "cube": (0.041, 0.0017),
    "sphere": (0.040, 0.0005),
    "torus": (0.016, 0.0022),
    "cross": (0.016, 0.0017),
    "tee": (0.016, 0.0010),
}
# Tee geometry in its own frame, origin at the stem tip (measured from the top-down cue):
# stem x in [-0.007, 0.007], y in [0, 0.026]; bar x in [-0.02, 0.02], y in [0.026, 0.040].
TEE_STEM = (-0.007, 0.007, 0.0, 0.026)
TEE_BAR = (-0.020, 0.020, 0.026, 0.040)


def _shape_cost(o, s):
    """Dissimilarity of measured object features to a shape prototype (lower = better).
    Terms: top height, top-surface area, enclosed hole (torus only), footprint size (sphere)."""
    Z, A = SHAPE_PROTO[s]
    dz = (o["zmax"] - Z) / (0.003 if Z > 0.03 else 0.005)
    da = (o["top_area"] - A) / 0.00015
    c = min(dz * dz, 25.0) + min(da * da, 25.0)
    holey = o["hole"] > 0.012
    w = 2.0 if o.get("split") else 6.0
    if s == "torus":
        c += -w if holey else 1.0
    elif holey:
        c += w
    if s == "sphere" and o.get("ev1", 0.0) > 0.0105:
        e = (o["ev1"] - 0.0105) / 0.001
        c += min(e * e, 16.0)
    return c


def _classify_single(o):
    costs = {s: _shape_cost(o, s) for s in SHAPE_PROTO}
    best = min(costs, key=lambda s: costs[s])
    others = sorted(costs.values())
    return best, float(others[1] - others[0])


def _tee_template_points(step=0.002):
    pts = []
    for (x0, x1, y0, y1) in (TEE_STEM, TEE_BAR):
        x = x0 + step / 2
        while x < x1:
            y = y0 + step / 2
            while y < y1:
                pts.append((x, y))
                y += step
            x += step
    return np.array(pts)


def _tee_origin(top_xy):
    """Fit the tee template (rotation + translation) to top-surface points; return origin xy
    (stem tip), stem direction angle and fit score."""
    P = np.asarray(top_xy, dtype=np.float64)
    if len(P) < 5:
        c = P.mean(0) if len(P) else np.zeros(2)
        return float(c[0]), float(c[1]), 0.0, 0.0
    tpl = _tee_template_points()
    tc = tpl.mean(0)
    tpl0 = tpl - tc
    pc = P.mean(0)
    best = None
    for k in range(72):
        th = 2 * math.pi * k / 72
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        T = tpl0 @ R.T
        for ox in (-0.004, -0.002, 0.0, 0.002, 0.004):
            for oy in (-0.004, -0.002, 0.0, 0.002, 0.004):
                Tq = T + pc + np.array([ox, oy])
                d = ((P[:, None, :] - Tq[None, :, :]) ** 2).sum(-1)
                a = np.sqrt(d.min(1))  # observed -> template
                b = np.sqrt(d.min(0))  # template -> observed
                sc = float(np.mean(np.minimum(a, 0.01)) + np.mean(np.minimum(b, 0.01)))
                if best is None or sc < best[0]:
                    best = (sc, th, ox, oy)
    sc, th, ox, oy = best
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    org = (np.zeros(2) - tc) @ R.T + pc + np.array([ox, oy])
    return float(org[0]), float(org[1]), float(th), sc


def _view_objects(frame, cam):
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    return _objects_in_view(rgb, depth, K, E)


def _read_cue(frame):
    """Cue = the single coloured object shown before candidates appear.
    Returns dict(color, shape, margin, x, y, n_objects, cam) or None."""
    for cam in ("wrist", "front"):
        objs = [o for o in _view_objects(frame, cam) if o["pixels"] >= 15]
        if len(objs) == 1:
            o = objs[0]
            s, m = _classify_single(o)
            return {"color": o["color"], "shape": s, "margin": m, "x": o["x"], "y": o["y"],
                    "n_objects": 1, "cam": cam, "zmax": o["zmax"], "top_area": o["top_area"]}
    return None


def _count_objects(frame, cam="front"):
    """Number of coloured connected components (cheap; no depth features)."""
    masks = _color_masks(getattr(frame, cam + "_rgb"))
    n = 0
    for c in COLOR_NAMES:
        n += len(_components(masks[c], 6))
    return n


def _fuse(front, wrist, radius=0.025):
    """Fuse per-camera detections by colour and world distance. Each fused object keeps
    the per-camera feature dicts; positions prefer the wrist (higher resolution)."""
    fused = [{"color": f["color"], "front": f, "wrist": None} for f in front]
    pairs = []
    for j, w in enumerate(wrist):
        for i, f in enumerate(front):
            if f["color"] != w["color"]:
                continue
            d = math.hypot(f["x"] - w["x"], f["y"] - w["y"])
            if d < radius:
                pairs.append((d, i, j))
    pairs.sort()
    used_f, used_w = set(), set()
    for d, i, j in pairs:
        if i in used_f or j in used_w:
            continue
        used_f.add(i)
        used_w.add(j)
        fused[i]["wrist"] = wrist[j]
    for j, w in enumerate(wrist):
        if j not in used_w and not w["border"] and w["pixels"] >= 40:
            fused.append({"color": w["color"], "front": None, "wrist": w})
    for o in fused:
        f, w = o["front"], o["wrist"]
        wv = w is not None and not w["border"]
        src = w if wv else f
        if src is None:
            src = w
        o["x"], o["y"] = src["x"], src["y"]
        o["zmax"] = src["zmax"]
        costs = {}
        for s in SHAPE_PROTO:
            cs, ws = [], []
            if f is not None:
                cs.append(_shape_cost(f, s))
                ws.append(0.4 if not f["split"] else 0.25)
            if w is not None:
                cs.append(_shape_cost(w, s))
                ws.append(0.15 if w["border"] else (0.6 if not w["split"] else 0.4))
            costs[s] = sum(c * q for c, q in zip(cs, ws)) / sum(ws)
        o["costs"] = costs
        o["src"] = "wrist" if src is w else "front"
    return fused


def _assign_color(objs):
    """Assign the 5 shapes one-to-one to same-colour objects (min total cost).
    Returns list of labels (extra objects get None)."""
    names = list(SHAPE_PROTO)
    n = len(objs)
    if n == 0:
        return []
    C = [[o["costs"][s] for s in names] for o in objs]
    # brute force: permutations of objects for the 5 shape slots (None allowed when n < 5)
    idx = list(range(n))
    slots = len(names)
    best_total, best_map = None, None
    stack = [(0, (), 0.0)]
    while stack:
        k, chosen, total = stack.pop()
        if best_total is not None and total >= best_total:
            continue
        if k == slots:
            best_total, best_map = total, chosen
            continue
        missing_allowed = (slots - k) > (n - len([c for c in chosen if c is not None]))
        for i in idx:
            if i not in chosen:
                stack.append((k + 1, chosen + (i,), total + C[i][k]))
        if missing_allowed:
            stack.append((k + 1, chosen + (None,), total + 12.0))
    labels = [None] * n
    for k, i in enumerate(best_map):
        if i is not None:
            labels[i] = names[k]
    return labels


def _find_target(frame, color, shape, exclude=None):
    """Detect all candidates in the frame, assign shapes per colour and return the target
    plus a summary. Target touch point: the object origin (tee: stem tip from template fit)."""
    front = _view_objects(frame, "front")
    wrist = _view_objects(frame, "wrist")
    fused = _fuse(front, wrist)
    same = [o for o in fused if o["color"] == color]
    for o in same:
        for ex in (exclude or []):
            if math.hypot(o["x"] - ex[0], o["y"] - ex[1]) < 0.025:
                o["costs"][shape] = o["costs"][shape] + 40.0
                o["excluded"] = True
    labels = _assign_color(same)
    cand = [o for o, l in zip(same, labels) if l == shape]
    info = {"n_front": len(front), "n_wrist": len(wrist), "n_fused": len(fused), "n_color": len(same),
            "labels": [[l, round(o["x"], 3), round(o["y"], 3)] for o, l in zip(same, labels)]}
    if not cand:
        if not same:
            return None, info
        cand = [min(same, key=lambda o: o["costs"][shape])]
        info["fallback"] = True
    t = cand[0]
    # margin: how much worse the second-best object would be for this shape
    alt = sorted(o["costs"][shape] for o in same)
    info["cost"] = round(t["costs"][shape], 3)
    info["alt_costs"] = [round(a, 2) for a in alt[:3]]
    tx, ty = t["x"], t["y"]
    if shape == "tee":
        tx, ty = _tee_point_from_frame(frame, color, t["x"], t["y"])
    ztop = SHAPE_PROTO[shape][0]
    return {"x": tx, "y": ty, "cx": t["x"], "cy": t["y"], "ztop": ztop, "src": t["src"]}, info


def _top_points_near(frame, cam, color, x, y, radius=0.03):
    """Top-surface world points of the given colour within radius of (x, y)."""
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    m = _color_masks(rgb)[color]
    rows, cols = np.nonzero(m)
    if len(rows) == 0:
        return np.zeros((0, 3))
    pts, ok = _backproject(rows, cols, depth, K, E)
    sel = (pts[:, 2] > 0.003) & (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < radius)
    pts = pts[sel]
    if len(pts) == 0:
        return pts
    zmax = np.percentile(pts[:, 2], 95)
    return pts[pts[:, 2] > zmax - 0.004]


def _tee_point_from_frame(frame, color, x, y):
    """Stem-tip estimate for a tee near (x, y), using the best camera view."""
    best = None
    for cam in ("wrist", "front"):
        P = _top_points_near(frame, cam, color, x, y, 0.032)
        if len(P) < 8:
            continue
        ox, oy, th, sc = _tee_origin(P[:, :2])
        # prefer more points (closer view) unless the fit is poor
        key = sc - 0.0005 * min(len(P), 200) / 100.0
        if best is None or key < best[0]:
            best = (key, ox, oy)
    if best is None:
        return x, y
    return best[1], best[2]
