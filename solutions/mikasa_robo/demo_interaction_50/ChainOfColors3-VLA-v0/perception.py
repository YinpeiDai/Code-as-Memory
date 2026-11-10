"""Perception helpers: front RGB-D -> world points -> palette cube tops and button cap."""

PALETTE = {
    "red": (220, 4, 4),
    "darkred": (161, 4, 4),
    "orange": (220, 161, 4),
    "yellow": (220, 220, 4),
    "green": (4, 220, 4),
    "cyan": (4, 220, 220),
    "teal": (4, 161, 161),
    "blue": (4, 4, 220),
    "magenta": (220, 4, 220),
}
CAP_RGB = (211, 136, 136)
# Empirical offsets (demo evidence, 150 touches / 50 presses): expert TCP xy minus
# measured top-face centroid. See task_analysis.md.
CUBE_XY_OFFSET = (-0.0059, 0.0029)
BUTTON_XY_OFFSET = (0.012, 0.0028)


def _world_points(K, E, depth):
    z = np.asarray(depth, dtype=np.float64) / 1000.0
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    c = np.stack([x, y, z], -1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    w = (c - t) @ R
    w[z <= 0] = np.nan
    return w


def _frame_world(frame):
    return _world_points(frame.front_K, frame.front_extrinsic, frame.front_depth)


def _analyze_front(frame):
    """Return dict color -> {xy, n, spread, rgb} for cube top faces, plus button cap."""
    w = _frame_world(frame)
    rgb = np.asarray(frame.front_rgb).astype(np.float64)
    sat = rgb.max(-1) - rgb.min(-1)
    zz = np.nan_to_num(w[..., 2], nan=-1.0)
    xx = np.nan_to_num(w[..., 0], nan=9.0)
    yy = np.nan_to_num(w[..., 1], nan=9.0)
    inside = (np.abs(xx) < 0.7) & (np.abs(yy) < 0.7)
    top = inside & (zz > 0.0385) & (zz < 0.05) & (sat > 100)
    names = list(PALETTE.keys())
    pal = np.array([PALETTE[k] for k in names], dtype=np.float64)
    cubes = {}
    idx = np.flatnonzero(top.ravel())
    if len(idx) > 0:
        cols = rgb.reshape(-1, 3)[idx]
        d = np.sqrt(((cols[:, None, :] - pal[None, :, :]) ** 2).sum(-1))
        best = d.argmin(1)
        bd = d.min(1)
        pts = w.reshape(-1, 3)[idx]
        for k, name in enumerate(names):
            sel = (best == k) & (bd < 60)
            n = int(sel.sum())
            if n < 4:
                continue
            p = pts[sel]
            med = np.median(p[:, :2], axis=0)
            near = np.sqrt(((p[:, :2] - med) ** 2).sum(-1)) < 0.03
            if near.sum() < 4:
                continue
            p2 = p[near]
            c = p2[:, :2].mean(0)
            spread = float(np.sqrt(((p[:, :2] - med) ** 2).sum(-1)).max())
            cubes[name] = {"xy": [float(c[0]), float(c[1])], "n": n, "spread": spread}
    capc = np.array(CAP_RGB, dtype=np.float64)
    capd = np.sqrt(((rgb - capc) ** 2).sum(-1))
    capm = inside & (zz > 0.052) & (zz < 0.075) & (capd < 45)
    button = None
    if capm.sum() >= 8:
        p = w[capm]
        button = {"xy": [float(p[:, 0].mean()), float(p[:, 1].mean())], "n": int(capm.sum()),
                  "ztop": float(p[:, 2].max())}
    return {"cubes": cubes, "button": button}


def _cue_color(an):
    """A single cube shown at the centre during the cue; returns its color or None."""
    cubes = an["cubes"]
    if len(cubes) == 0 or len(cubes) > 2:
        return None
    best = None
    for name in cubes:
        c = cubes[name]
        if abs(c["xy"][0]) < 0.08 and abs(c["xy"][1]) < 0.08 and c["n"] >= 12:
            if best is None or c["n"] > cubes[best]["n"]:
                best = name
    return best


def _is_layout(an):
    return len(an["cubes"]) >= 7
