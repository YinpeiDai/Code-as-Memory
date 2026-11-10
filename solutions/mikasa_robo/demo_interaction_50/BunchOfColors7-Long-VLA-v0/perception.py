"""Cube perception helpers (front camera RGB-D -> world cube list)."""

# Top-face palette observed in the demonstrations (front camera, top faces).
_PALETTE = {
    "red": (231, 4, 4),
    "darkred": (169, 4, 4),
    "green": (4, 231, 4),
    "blue": (4, 4, 231),
    "yellow": (231, 231, 4),
    "magenta": (231, 4, 231),
    "cyan": (4, 231, 231),
    "teal": (4, 169, 169),
    "orange": (231, 169, 4),
}
_CELL = 0.005  # world xy grid cell for clustering (m)


def _world_points(depth, K, E):
    d = depth.astype(np.float64) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, d], -1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t) @ R


def _color_mask(rgb):
    im = rgb.astype(np.int16)
    mx = im.max(-1)
    mn = im.min(-1)
    return (mn < 60) & (mx > 90) & ((mx - mn) > 80)


def _label_grid(occ):
    """Connected components (8-neighbour) of a boolean grid by label propagation."""
    H, W = occ.shape
    big = H * W + 10
    lab = np.where(occ, np.arange(H * W).reshape(H, W), big)
    for _ in range(400):
        p = np.pad(lab, 1, constant_values=big)
        nb = lab.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nb = np.minimum(nb, p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W])
        nb = np.where(occ, nb, big)
        if np.array_equal(nb, lab):
            break
        lab = nb
    return lab, big


def _classify(col):
    best, bd = None, 1e9
    for name, ref in _PALETTE.items():
        dd = float(np.sum((np.asarray(col, dtype=np.float64) - np.asarray(ref)) ** 2))
        if dd < bd:
            best, bd = name, dd
    return best, math.sqrt(bd)


def _detect_cubes(rgb, depth, K, E, min_pixels=12):
    """Return list of dicts: color, x, y, z_top, n, dist (palette distance)."""
    pw = _world_points(depth, K, E)
    m = _color_mask(rgb) & (depth > 0)
    z = pw[..., 2]
    m = m & (z > 0.005) & (z < 0.07) & (pw[..., 0] > -0.6) & (pw[..., 0] < 0.6) & (np.abs(pw[..., 1]) < 0.6)
    if m.sum() < min_pixels:
        return []
    P = pw[m]
    C = rgb[m].astype(np.float64)
    gx = np.floor((P[:, 0] + 0.6) / _CELL).astype(int)
    gy = np.floor((P[:, 1] + 0.6) / _CELL).astype(int)
    x0, y0 = gx.min(), gy.min()
    gx -= x0
    gy -= y0
    occ = np.zeros((gx.max() + 1, gy.max() + 1), bool)
    occ[gx, gy] = True
    lab, big = _label_grid(occ)
    pl = lab[gx, gy]
    out = []
    for L in np.unique(pl):
        sel = pl == L
        n = int(sel.sum())
        if n < min_pixels:
            continue
        Ps = P[sel]
        Cs = C[sel]
        col = np.percentile(Cs, 90, axis=0)
        name, dist = _classify(col)
        top = Ps[:, 2] > (np.percentile(Ps[:, 2], 95) - 0.004)
        cx = float(np.mean(Ps[top, 0]))
        cy = float(np.mean(Ps[top, 1]))
        out.append({"color": name, "x": cx, "y": cy, "z_top": float(np.percentile(Ps[:, 2], 95)),
                    "n": n, "dist": round(dist, 1)})
    return out


def _frame_cubes(frame):
    return _detect_cubes(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic)


def _color_count(frame):
    """Number of saturated cube-coloured pixels in the lower (table) part of the front image."""
    im = frame.front_rgb[90:256].astype(np.int16)
    return int(((im.min(-1) < 25) & (im.max(-1) > 100)).sum())


def _button_estimate(frame):
    """Estimate the press point of the pink button cap from the front camera (world xy).

    The visible cap-top pixels are biased toward the camera; the demos show the expert pressing
    ~(+0.018, +0.004) m from this raw estimate, which is applied as an offset.
    Returns (x, y, n_pixels) or None.
    """
    im = frame.front_rgb.astype(np.int16)
    pw = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    pink = (im[..., 0] > 150) & ((im[..., 0] - im[..., 1]) > 40) & (np.abs(im[..., 1] - im[..., 2]) < 30) & (im[..., 1] > 60)
    top = pink & (pw[..., 2] > 0.05) & (pw[..., 2] < 0.08) & (np.abs(pw[..., 1]) < 0.3) & (pw[..., 0] > -0.4) & (pw[..., 0] < 0.2)
    n = int(top.sum())
    if n < 15:
        return None
    P = pw[top]
    return float(np.mean(P[:, 0]) + 0.0177), float(np.mean(P[:, 1]) + 0.0037), n
