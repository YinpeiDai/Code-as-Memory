"""Perception helpers for FindImposterShape5: blue-object segmentation and 3D shape classification
from the wrist RGB-D camera (pure numpy, no imports)."""

SHAPES = ("sphere", "cube", "cylinder", "torus", "cross")


def _blue_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (b > 90) & (b > r + 50) & (b > g + 50)


def _components(mask, min_pix=25):
    """4-connected components of a boolean mask; returns list of (ys, xs) arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    ys0, xs0 = np.nonzero(mask)
    comps = []
    cur = 0
    for k in range(len(ys0)):
        y, x = int(ys0[k]), int(xs0[k])
        if lab[y, x]:
            continue
        cur += 1
        lab[y, x] = cur
        stack = [(y, x)]
        py, px = [], []
        while stack:
            cy, cx = stack.pop()
            py.append(cy)
            px.append(cx)
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
        if len(py) >= min_pix:
            comps.append((np.array(py), np.array(px)))
    return comps


def _fill_holes(ys, xs):
    """Filled local mask of a component: returns (filled_mask, y0, x0, raw_mask)."""
    y0, x0 = int(ys.min()) - 1, int(xs.min()) - 1
    h, w = int(ys.max()) - y0 + 2, int(xs.max()) - x0 + 2
    m = np.zeros((h, w), bool)
    m[ys - y0, xs - x0] = True
    outside = np.zeros((h, w), bool)
    stack = [(0, 0)]
    outside[0, 0] = True
    while stack:
        cy, cx = stack.pop()
        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
            if 0 <= ny < h and 0 <= nx < w and not m[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return ~outside, y0, x0, m


def _pix_to_world(ys, xs, depth, K, E):
    d = depth[ys, xs].astype(np.float64) / 1000.0
    ok = d > 0
    xc = (xs + 0.5 - K[0, 2]) / K[0, 0] * d
    yc = (ys + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([xc, yc, d], 1)
    R = np.asarray(E, np.float64)[:, :3]
    t = np.asarray(E, np.float64)[:, 3]
    return (pc - t) @ R, ok


def _circularity(ys, xs):
    """Pixel area / area of circle through the farthest pixel (1 circle, 0.64 square, ~0.6 cross)."""
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max() + 0.5
    return float(len(ys) / (math.pi * r * r))


def _extent_ratio(ys, xs):
    """max/min projected width over orientations (1 circle, ~1.41 square)."""
    ws = []
    for a in range(0, 180, 5):
        th = a * math.pi / 180.0
        p = xs * math.cos(th) + ys * math.sin(th)
        ws.append(p.max() - p.min() + 1.0)
    return float(max(ws) / min(ws))


def _classify(f):
    """Rule classifier fitted on all 450 demo objects (50 episodes x 9); margins in task_analysis.md."""
    if f["zmax"] < 0.034:  # short shapes: torus (0.028 m) or cross (0.028 m)
        torus_votes = int(f["rad_ratio"] > 0.72) + int(f["hole_px"] >= 3) + int(f["top_circ"] > 0.63)
        return "torus" if torus_votes >= 2 else "cross"
    if f["top2"] < 0.40:  # tall with a domed top
        return "sphere"
    cube_votes = int(f["top_circ"] < 0.74) + int(f["top_ext"] > 1.23)
    if cube_votes == 1:
        return "cube" if (0.74 - f["top_circ"]) / 0.07 + (f["top_ext"] - 1.23) / 0.07 > 0 else "cylinder"
    return "cube" if cube_votes == 2 else "cylinder"


def _detect(frame):
    """Detect and classify blue objects in a wrist frame; list of dicts with world xy/zmax and shape."""
    rgb = frame.wrist_rgb
    depth = frame.wrist_depth
    K = np.asarray(frame.wrist_K, np.float64)
    E = frame.wrist_extrinsic
    out = []
    for ys, xs in _components(_blue_mask(rgb)):
        P, ok = _pix_to_world(ys, xs, depth, K, E)
        if ok.sum() < 15:
            continue
        P = P[ok]
        yk, xk = ys[ok], xs[ok]
        z = P[:, 2]
        zmax = float(np.percentile(z, 98))
        top4 = z > zmax - 0.004
        top2 = float((z > zmax - 0.002).mean())
        filled, y0, x0, raw = _fill_holes(ys, xs)
        fy, fx = np.nonzero(filled)
        f = {
            "xy": [float(P[top4, 0].mean()), float(P[top4, 1].mean())],
            "zmax": zmax,
            "top2": top2,
            "top_circ": _circularity(yk[top4], xk[top4]),
            "top_ext": _extent_ratio(yk[top4], xk[top4]),
            "fill_circ": _circularity(fy, fx),
            "rad_ratio": _radial_ratio(fy, fx),
            "hole_px": int(filled.sum() - raw.sum()),
            "px": int(len(ys)),
            "uv": [float(xs.mean()), float(ys.mean())],
        }
        f["shape"] = _classify(f)
        out.append(f)
    return out


def _radial_ratio(ys, xs, nbins=16):
    """min/max over angular bins of the farthest pixel radius (≈1 round, ≈0.5 cross)."""
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    r = np.sqrt(dy * dy + dx * dx)
    ang = np.arctan2(dy, dx)
    b = ((ang + math.pi) / (2 * math.pi) * nbins).astype(int) % nbins
    rm = np.zeros(nbins)
    for i in range(nbins):
        sel = b == i
        rm[i] = r[sel].max() if sel.any() else 0.0
    return float(rm.min() / max(rm.max(), 1e-6))
