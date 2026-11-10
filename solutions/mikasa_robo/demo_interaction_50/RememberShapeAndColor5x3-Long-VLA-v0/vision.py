"""Vision helpers: colour segmentation, connected components, depth back-projection,
merged-blob splitting and rotation-invariant shape features."""

COLOR_NAMES = ("red", "green", "blue")
SHAPE_NAMES = ("cube", "sphere", "torus", "cross", "tee")


def _color_masks(rgb):
    """Boolean masks for saturated red / green / blue pixels (table is orange-brown)."""
    w = rgb.astype(np.int32)
    r, g, b = w[..., 0], w[..., 1], w[..., 2]
    red = (r > 40) & (g * 10 < r * 3) & (b * 10 < r * 3)
    green = (g > 40) & (r * 10 < g * 6) & (b * 10 < g * 6)
    blue = (b > 40) & (r * 10 < b * 6) & (g * 10 < b * 7)
    return {"red": red, "green": green, "blue": blue}


def _components(mask, min_pixels=4):
    """8-connected components of a boolean mask; list of (rows, cols) int arrays."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    h, w = mask.shape
    lab = -np.ones(mask.shape, dtype=np.int32)
    comps = []
    for i in range(len(ys)):
        y0, x0 = int(ys[i]), int(xs[i])
        if lab[y0, x0] >= 0:
            continue
        cid = len(comps)
        stack = [(y0, x0)]
        lab[y0, x0] = cid
        py, px = [], []
        while stack:
            y, x = stack.pop()
            py.append(y)
            px.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] < 0:
                        lab[yy, xx] = cid
                        stack.append((yy, xx))
        comps.append((np.array(py, dtype=np.int64), np.array(px, dtype=np.int64)))
    return [c for c in comps if len(c[0]) >= min_pixels]


def _backproject(rows, cols, depth, K, E):
    """Pixels + depth (mm) -> world points (N,3) and valid-depth mask."""
    d = depth[rows, cols].astype(np.float64) / 1000.0
    ok = d > 0
    rows, cols, d = rows[ok], cols[ok], d[ok]
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    xc = (cols + 0.5 - cx) / fx * d
    yc = (rows + 0.5 - cy) / fy * d
    pc = np.stack([xc, yc, d], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R, ok


def _project(points, K, E):
    """World points (N,3) -> pixel u, v and camera depth."""
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    pc = np.asarray(points, dtype=np.float64) @ R.T + t
    u = pc[:, 0] / pc[:, 2] * float(K[0][0]) + float(K[0][2])
    v = pc[:, 1] / pc[:, 2] * float(K[1][1]) + float(K[1][2])
    return u, v, pc[:, 2]


def _pixel_area(rows, cols, depth, K, E):
    """Area each pixel covers on a horizontal plane (m^2)."""
    d = depth[rows, cols].astype(np.float64) / 1000.0
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    rx = (cols + 0.5 - cx) / fx
    ry = (rows + 0.5 - cy) / fy
    rn = np.sqrt(rx * rx + ry * ry + 1.0)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    nz = R[:, 2]
    cosi = np.abs(rx * nz[0] + ry * nz[1] + nz[2]) / rn
    cosi = np.maximum(cosi, 0.2)
    return d * d / (fx * fy) / rn / cosi


def _hole_fraction(rows, cols):
    """Enclosed background pixels / object pixels (torus hole)."""
    r0, c0 = rows.min(), cols.min()
    h, w = int(rows.max() - r0 + 3), int(cols.max() - c0 + 3)
    m = np.zeros((h, w), dtype=bool)
    m[rows - r0 + 1, cols - c0 + 1] = True
    outside = np.zeros((h, w), dtype=bool)
    stack = [(0, 0)]
    outside[0, 0] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < h and 0 <= xx < w and not m[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                stack.append((yy, xx))
    holes = (~m) & (~outside)
    return float(holes.sum()) / float(max(1, m.sum()))


def _hull_area(xy):
    """Convex hull area of 2D points (monotone chain)."""
    pts = sorted(set((round(float(p[0]), 4), round(float(p[1]), 4)) for p in xy))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    a = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _features(rows, cols, pts, parea, touch_border):
    """Metric features of one object from its pixels and world points (all valid, above table)."""
    z = pts[:, 2]
    zmax = float(np.percentile(z, 95))
    sel = z > zmax - 0.004
    top = pts[sel]
    top_area = float(parea[sel].sum())
    cen = top[:, :2].mean(0) if len(top) >= 3 else pts[:, :2].mean(0)
    allc = pts[:, :2].mean(0)
    xy = pts[:, :2] - allc
    if len(xy) > 2:
        ev = np.sort(np.linalg.eigvalsh(np.cov(xy.T)))[::-1]
    else:
        ev = np.array([0.0, 0.0])
    txy = top[:, :2] - cen
    trad = np.sqrt((txy ** 2).sum(1)) if len(txy) else np.zeros(1)
    hull = _hull_area(top[:, :2]) if len(top) >= 3 else 0.0
    # tee origin is the stem tip: farthest top point from the top centroid
    far = top[int(np.argmax(trad))][:2] if len(top) else cen
    return {
        "x": float(cen[0]), "y": float(cen[1]),
        "zmax": zmax, "top_area": top_area, "area": float(parea.sum()),
        "top_frac": float(sel.sum()) / float(len(z)),
        "ev1": float(np.sqrt(max(ev[0], 0.0))), "ev2": float(np.sqrt(max(ev[1], 0.0))),
        "rmax": float(np.percentile(trad, 95)) if len(txy) else 0.0,
        "hull": hull, "solidity": top_area / hull if hull > 1e-6 else 1.0,
        "hole": _hole_fraction(rows, cols),
        "tip_x": float(far[0]), "tip_y": float(far[1]),
        "tip_d": float(trad.max()) if len(txy) else 0.0,
        "pixels": int(len(rows)), "border": bool(touch_border),
    }


def _kmeans2(xy, iters=12):
    """Split 2D points into two clusters, initialised at extremes of the principal axis."""
    c = xy.mean(0)
    w, v = np.linalg.eigh(np.cov((xy - c).T))
    ax = v[:, int(np.argmax(w))]
    p = (xy - c) @ ax
    a, b = xy[int(np.argmin(p))], xy[int(np.argmax(p))]
    lab = np.zeros(len(xy), dtype=bool)
    for _ in range(iters):
        da = ((xy - a) ** 2).sum(1)
        db = ((xy - b) ** 2).sum(1)
        lab = db < da
        if lab.all() or (~lab).all():
            break
        a, b = xy[~lab].mean(0), xy[lab].mean(0)
    return lab


SINGLE_EV1 = 0.0175  # largest principal std (m) of a single object footprint seen in demos


def _objects_in_view(rgb, depth, K, E, min_pixels=6, max_split=4):
    """Segment coloured objects in one camera view, splitting merged same-colour blobs.
    Returns a list of feature dicts with colour."""
    out = []
    masks = _color_masks(rgb)
    H, W = rgb.shape[0], rgb.shape[1]
    for cname in COLOR_NAMES:
        for rows, cols in _components(masks[cname], min_pixels):
            pts, ok = _backproject(rows, cols, depth, K, E)
            rows, cols = rows[ok], cols[ok]
            keep = pts[:, 2] > 0.003
            if keep.sum() < min_pixels:
                continue
            rows, cols, pts = rows[keep], cols[keep], pts[keep]
            parea = _pixel_area(rows, cols, depth, K, E)
            parts = [(rows, cols, pts, parea)]
            done = []
            guard = 0
            while parts and guard < 3 * max_split:
                guard += 1
                r_, c_, p_, a_ = parts.pop()
                xy = p_[:, :2]
                ev = np.linalg.eigvalsh(np.cov((xy - xy.mean(0)).T)) if len(xy) > 3 else np.zeros(2)
                if len(xy) >= 2 * min_pixels and math.sqrt(max(ev.max(), 0.0)) > SINGLE_EV1 and len(done) + len(parts) < max_split:
                    lab = _kmeans2(xy)
                    if lab.sum() >= min_pixels // 2 and (~lab).sum() >= min_pixels // 2:
                        parts.append((r_[lab], c_[lab], p_[lab], a_[lab]))
                        parts.append((r_[~lab], c_[~lab], p_[~lab], a_[~lab]))
                        continue
                done.append((r_, c_, p_, a_))
            for r_, c_, p_, a_ in done:
                if len(r_) < min_pixels:
                    continue
                border = bool(r_.min() <= 1 or c_.min() <= 1 or r_.max() >= H - 2 or c_.max() >= W - 2)
                f = _features(r_, c_, p_, a_, border)
                f["color"] = cname
                f["u"] = float(c_.mean())
                f["v"] = float(r_.mean())
                f["split"] = len(done) > 1
                out.append(f)
    return out
