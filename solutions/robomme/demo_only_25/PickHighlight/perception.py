"""Perception helpers for PickHighlight (wrist RGB-D based)."""

CUBE_MIN_PIX = 12
RING_R_IN = 0.026
RING_R_OUT = 0.040


def _backproject(depth, K, E):
    """Return HxWx3 world points (nan where depth unknown)."""
    H, W = depth.shape
    z = depth.astype(np.float64) / 1000.0
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    x = (u - K[0][2]) / K[0][0] * z
    y = (v - K[1][2]) / K[1][1] * z
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    pw = (pc - t) @ R
    pw[z <= 0] = np.nan
    return pw


def _label(mask, max_iter=400):
    """4-connected component labels (0 = background) by min-propagation."""
    H, W = mask.shape
    big = H * W + 10
    lab = np.where(mask, np.arange(1, H * W + 1).reshape(H, W), big).astype(np.int64)
    for _ in range(max_iter):
        old = lab
        m = lab.copy()
        m[1:, :] = np.minimum(m[1:, :], lab[:-1, :])
        m[:-1, :] = np.minimum(m[:-1, :], lab[1:, :])
        m[:, 1:] = np.minimum(m[:, 1:], lab[:, :-1])
        m[:, :-1] = np.minimum(m[:, :-1], lab[:, 1:])
        lab = np.where(mask, m, big)
        if np.array_equal(lab, old):
            break
    lab[~mask] = 0
    return lab


def _color_masks(rgb):
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    out = {}
    out['red'] = (r > 120) & (r - g > 70) & (r - b > 70)
    out['green'] = (g > 120) & (g - r > 70) & (g - b > 70)
    out['blue'] = (b > 120) & (b - r > 70) & (b - g > 50)
    return out


def _white_mask(rgb):
    im = rgb.astype(np.int32)
    return (im.min(-1) > 200) & ((im.max(-1) - im.min(-1)) < 40)


def _yaw_of(points_xy):
    """Edge angle of a square footprint, in [-pi/4, pi/4), via min bounding box."""
    p = points_xy - points_xy.mean(0)
    best, best_a = None, 0.0
    for k in range(90):
        a = k * math.pi / 180.0
        c, s = math.cos(a), math.sin(a)
        x = p[:, 0] * c + p[:, 1] * s
        y = -p[:, 0] * s + p[:, 1] * c
        area = (np.percentile(x, 97) - np.percentile(x, 3)) * (np.percentile(y, 97) - np.percentile(y, 3))
        if best is None or area < best:
            best, best_a = area, a
    if best_a >= math.pi / 4:
        best_a -= math.pi / 2
    return best_a


def _detect_cubes(rgb, depth, K, E):
    """List of cube dicts: color, xyz (top centre), yaw, ring (white fraction around)."""
    pw = _backproject(depth, K, E)
    white = _white_mask(rgb)
    valid = np.isfinite(pw[..., 2])
    table = valid & (pw[..., 2] < 0.012)
    wxy = pw[white & table][:, :2]
    cubes = []
    for color, m in _color_masks(rgb).items():
        m = m & valid & (pw[..., 2] > 0.01) & (pw[..., 2] < 0.08)
        if m.sum() < CUBE_MIN_PIX:
            continue
        lab = _label(m)
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        for i, n in zip(ids, counts):
            if n < CUBE_MIN_PIX:
                continue
            pts = pw[lab == i]
            ztop = np.percentile(pts[:, 2], 90)
            top = pts[pts[:, 2] > ztop - 0.008]
            if len(top) < 5:
                top = pts
            c = np.median(top[:, :2], 0)
            yaw = _yaw_of(top[:, :2]) if len(top) >= 8 else 0.0
            ring = 0
            cover = 0
            if len(wxy):
                d = np.hypot(wxy[:, 0] - c[0], wxy[:, 1] - c[1])
                sel = (d > RING_R_IN) & (d < RING_R_OUT)
                ring = int(sel.sum())
                if ring:
                    ang = np.arctan2(wxy[sel, 1] - c[1], wxy[sel, 0] - c[0])
                    bins = np.floor((ang + math.pi) / (2 * math.pi) * 12).astype(int) % 12
                    cover = int(len(set(bins.tolist())))
            cubes.append({'color': color, 'xyz': [float(c[0]), float(c[1]), float(ztop)],
                          'yaw': float(yaw), 'npix': int(n), 'ring': ring, 'cover': cover})
    return cubes


def _detect_button(rgb, depth, K, E):
    """Button: grey disc inside a white square base. Returns world xyz of grey disc centre or None."""
    pw = _backproject(depth, K, E)
    im = rgb.astype(np.int32)
    grey = (im.max(-1) - im.min(-1) < 20) & (im.min(-1) > 110) & (im.max(-1) < 200)
    valid = np.isfinite(pw[..., 2])
    m = grey & valid & (pw[..., 2] > 0.0) & (pw[..., 2] < 0.08) & (np.hypot(pw[..., 0], pw[..., 1] ) < 0.6)
    m = m & (pw[..., 0] < 0.25)
    if m.sum() < 5:
        return None
    lab = _label(m)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    best = None
    for i, n in zip(ids, counts):
        pts = pw[lab == i]
        sx = np.ptp(pts[:, 0]); sy = np.ptp(pts[:, 1])
        if sx > 0.08 or sy > 0.08:
            continue
        if best is None or n > best[0]:
            best = (n, pts)
    if best is None:
        return None
    pts = best[1]
    c = np.median(pts, 0)
    return [float(c[0]), float(c[1]), float(np.percentile(pts[:, 2], 90))]
