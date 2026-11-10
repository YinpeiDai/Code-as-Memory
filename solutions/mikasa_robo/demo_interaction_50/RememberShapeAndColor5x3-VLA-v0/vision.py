"""Perception helpers: colour segmentation, RGB-D back-projection to world, per-colour
clustering into objects and shape classification (sphere, cube, cross, torus, T)."""

CELL = 0.003          # top-down grid resolution (m)
ZMIN = 0.003          # points above the table plane (table is z=0)
ZMAX = 0.09
SHAPES = ["sphere", "cube", "cross", "torus", "T"]
COLORS = ["red", "green", "blue"]
# prototypes measured on the cue objects of all 50 demonstrations (see task_analysis.md)
PROTO = {
    "sphere": {"h": 0.040, "area": 12.7, "top": 0.66, "hole": 1.0, "rmax": 0.020},
    "cube":   {"h": 0.041, "area": 17.2, "top": 0.97, "hole": 1.0, "rmax": 0.025},
    "cross":  {"h": 0.016, "area": 17.2, "top": 0.95, "hole": 1.0, "rmax": 0.030},
    "torus":  {"h": 0.016, "area": 22.2, "top": 0.97, "hole": 0.12, "rmax": 0.0275},
    "T":      {"h": 0.016, "area": 10.3, "top": 0.97, "hole": 1.0, "rmax": 0.0235},
}
SCALE = {"h": 0.004, "area": 2.5, "top": 0.12, "hole": 0.25, "rmax": 0.002}


def _color_masks(rgb):
    I = rgb.astype(np.int32)
    r, g, b = I[..., 0], I[..., 1], I[..., 2]
    red = (r > 60) & (g * 100 < 35 * r) & (b * 100 < 35 * r)
    green = (g > 60) & (r * 10 < 6 * g) & (b * 10 < 6 * g)
    blue = (b > 60) & (r * 10 < 6 * b) & (g * 100 < 75 * b)
    return {"red": red, "green": green, "blue": blue}


def _backproject(depth, K, E, mask):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u].astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R


def _frame_points(frame, cams=("wrist", "front")):
    """World points of red/green/blue pixels from both RGB-D cameras."""
    out = {c: [] for c in COLORS}
    for cam in cams:
        rgb = getattr(frame, cam + "_rgb")
        depth = getattr(frame, cam + "_depth")
        K = getattr(frame, cam + "_K")
        E = getattr(frame, cam + "_extrinsic")
        M = _color_masks(rgb)
        for c in COLORS:
            p = _backproject(depth, K, E, M[c])
            if len(p):
                p = p[(p[:, 2] > ZMIN) & (p[:, 2] < ZMAX) & (np.abs(p[:, 0]) < 0.6) & (np.abs(p[:, 1]) < 0.6)]
            out[c].append(p)
    return {c: (np.concatenate(v, 0) if len(v) else np.zeros((0, 3))) for c, v in out.items()}


def _cells(p):
    """Unique top-down cells (centres) and the max height in each."""
    if len(p) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    ij = np.floor(p[:, :2] / CELL).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    z = np.zeros(len(keys))
    np.maximum.at(z, inv, p[:, 2])
    return (keys + 0.5) * CELL, z


def _kmeans(xy, k, iters=40):
    best = None
    n = len(xy)
    starts = [int(np.argmax(((xy - xy.mean(0)) ** 2).sum(1)))]
    # deterministic extra starts spread through the data
    for s in range(1, 6):
        starts.append(int((s * 7919) % n))
    for s0 in starts:
        c = [xy[s0]]
        for _ in range(k - 1):
            d = np.min(np.stack([((xy - ci) ** 2).sum(1) for ci in c]), 0)
            c.append(xy[int(np.argmax(d))])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((xy[:, None, :] - c[None]) ** 2).sum(2), 1)
            newc = np.array([xy[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
            if np.allclose(newc, c):
                break
            c = newc
        lab = np.argmin(((xy[:, None, :] - c[None]) ** 2).sum(2), 1)
        inertia = float(((xy - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best[1], best[2]


def _features(xy, z, cxy):
    h = float(np.percentile(z, 95))
    r = np.sqrt(((xy - cxy) ** 2).sum(1))
    hole = float((r < 0.008).sum() / (math.pi * 0.008 ** 2 / CELL ** 2))
    top = float((z > 0.8 * h).mean())
    return {"h": h, "area": float(len(xy) * CELL * CELL * 1e4), "hole": min(hole, 1.2), "top": top,
            "rmax": float(np.percentile(r, 95))}


def _shape_cost(f, s):
    P = PROTO[s]
    return float(sum(((f[k] - P[k]) / SCALE[k]) ** 2 for k in SCALE))


def _classify(f):
    costs = {s: _shape_cost(f, s) for s in SHAPES}
    best = min(costs, key=costs.get)
    srt = sorted(costs.values())
    return best, costs, srt[1] - srt[0]


def _permutations(items):
    if len(items) <= 1:
        return [list(items)]
    out = []
    for i in range(len(items)):
        for rest in _permutations(items[:i] + items[i + 1:]):
            out.append([items[i]] + rest)
    return out


def _assign_shapes(objs):
    """One-to-one shape assignment for the five objects of one colour (minimum total cost)."""
    n = len(objs)
    if n == 0:
        return []
    C = [[_shape_cost(o, s) for s in SHAPES] for o in objs]
    best = None
    for perm in _permutations(list(range(len(SHAPES)))):
        cost = sum(C[i][perm[i]] for i in range(n))
        if best is None or cost < best[0]:
            best = (cost, perm[:n])
    return [SHAPES[j] for j in best[1]]


def _cue_object(frame):
    """Detect the single cue object: colour with most points; returns dict or None."""
    P = _frame_points(frame)
    counts = {c: len(P[c]) for c in COLORS}
    c = max(counts, key=counts.get)
    if counts[c] < 30:
        return None
    xy, z = _cells(P[c])
    if len(xy) < 8:
        return None
    cxy = xy.mean(0)
    keep = np.sqrt(((xy - cxy) ** 2).sum(1)) < 0.05
    xy, z = xy[keep], z[keep]
    cxy = xy.mean(0)
    f = _features(xy, z, cxy)
    shape, costs, margin = _classify(f)
    others = sum(counts[k] for k in COLORS if k != c)
    f.update({"color": c, "x": float(cxy[0]), "y": float(cxy[1]), "shape": shape, "margin": margin,
              "npts": counts[c], "other_pts": others, "spread": float(np.sqrt(((xy - cxy) ** 2).sum(1)).max())})
    return f


def _scene_objects(frame, per_color=5):
    """Segment all candidates: per colour k-means into `per_color` clusters, features and shapes."""
    P = _frame_points(frame)
    objs = []
    for c in COLORS:
        xy, z = _cells(P[c])
        if len(xy) < per_color * 4:
            continue
        lab, cen = _kmeans(xy, per_color)
        group = []
        for i in range(per_color):
            m = lab == i
            if m.sum() < 3:
                continue
            f = _features(xy[m], z[m], cen[i])
            f.update({"color": c, "x": float(cen[i][0]), "y": float(cen[i][1]), "ncells": int(m.sum())})
            group.append(f)
        shapes = _assign_shapes(group)
        for f, s in zip(group, shapes):
            f["shape"] = s
            f["solo_shape"] = _classify(f)[0]
            objs.append(f)
    return objs


def _scene_counts(frame):
    """Number of coloured pixels per camera image (cheap phase detector)."""
    tot = 0
    for cam in ("wrist", "front"):
        M = _color_masks(getattr(frame, cam + "_rgb"))
        tot += int(sum(int(m.sum()) for m in M.values()))
    return tot


def _t_template():
    pts = []
    for u in np.arange(0.0, 0.025, 0.0015):
        for v in np.arange(-0.006, 0.0061, 0.0015):
            pts.append((u, v))
    for u in np.arange(0.025, 0.040, 0.0015):
        for v in np.arange(-0.018, 0.0181, 0.0015):
            pts.append((u, v))
    P = np.array(pts)
    return P - P.mean(0), float(-P.mean(0)[0])   # centred template, stem-tip u in centred frame


def _fit_t(xy, cxy):
    """Fit a T footprint to cluster cells; returns (theta, score, tip_xy, junction_xy).
    theta is the direction (world xy) from the stem tip towards the bar."""
    tmpl, tip_u = _t_template()
    rel = xy - cxy
    best = None
    for k in range(72):
        th = k * 2 * math.pi / 72
        c, s = math.cos(th), math.sin(th)
        Rt = np.array([[c, -s], [s, c]])
        W = tmpl @ Rt.T
        for dx in (-0.003, 0.0, 0.003):
            for dy in (-0.003, 0.0, 0.003):
                Wt = W + np.array([dx, dy])
                d1 = np.sqrt(((rel[:, None, :] - Wt[None]) ** 2).sum(2))
                cover_obs = float((d1.min(1) < 0.0035).mean())
                cover_t = float((d1.min(0) < 0.0035).mean())
                score = cover_obs + cover_t
                if best is None or score > best[1]:
                    best = (th, score, dx, dy)
    th, score, dx, dy = best
    axis = np.array([math.cos(th), math.sin(th)])
    off = np.array([dx, dy])
    tip = cxy + off + axis * tip_u
    junction = tip + axis * 0.018
    return th, score, tip, junction


T_THETA = math.pi / 2   # all observed T's (7 demos + rollouts) have the stem tip toward -y


def _fit_t_fixed(xy, cxy, theta=T_THETA):
    """Translation-only T template fit at the fixed orientation; returns (theta, score, tip, junction)."""
    tmpl, tip_u = _t_template()
    c, s = math.cos(theta), math.sin(theta)
    W = tmpl @ np.array([[c, -s], [s, c]]).T
    rel = xy - cxy
    best = None
    for dx in np.arange(-0.012, 0.0121, 0.0015):
        for dy in np.arange(-0.012, 0.0121, 0.0015):
            Wt = W + np.array([dx, dy])
            d1 = np.sqrt(((rel[:, None, :] - Wt[None]) ** 2).sum(2))
            score = float((d1.min(1) < 0.0035).mean()) + float((d1.min(0) < 0.0035).mean())
            if best is None or score > best[0]:
                best = (score, dx, dy)
    score, dx, dy = best
    axis = np.array([c, s])
    tip = cxy + np.array([dx, dy]) + axis * tip_u
    return theta, score, tip, tip + axis * 0.018
