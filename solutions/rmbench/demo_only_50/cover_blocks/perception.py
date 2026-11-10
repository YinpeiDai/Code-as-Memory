"""Perception helpers for cover_blocks (head RGB-D camera)."""

BLOCK_COLORS = ("red", "green", "blue")
LID_TOP_BAND = (0.835, 0.875)   # world z band of the lid handle top (observed 0.845)
TABLE_REGION = (-0.42, 0.42, -0.32, 0.12)  # x_min, x_max, y_min, y_max (world, m)


def _color_masks(im):
    im = im.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "red": (r > 150) & (g < 90) & (b < 90),
        "green": (g > 120) & (r < 90) & (b < 90),
        "blue": (b > 150) & (r < 90) & (g < 90),
        "brown": (r > 90) & (r > g) & (g > b) & (r - b > 30) & (g > 60),
    }


def _backproject(K, E, depth, mask):
    """World points (N,3) and pixel coords (N,2 as u,v) for masked pixels with valid depth."""
    vv, uu = np.nonzero(mask & (depth > 0))
    z = depth[vv, uu].astype(np.float64) / 1000.0
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    xc = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (xc - t) @ R
    return pw, np.stack([uu, vv], 1)


def _in_region(P):
    x0, x1, y0, y1 = TABLE_REGION
    return (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)


def _detect_blocks(frame):
    """Return {color: {"pos": [x,y,ztop], "u": mean image column, "n": pixel count}} for visible blocks."""
    im = frame.head_rgb
    d = frame.head_depth
    m = _color_masks(im)
    out = {}
    for c in BLOCK_COLORS:
        P, UV = _backproject(frame.head_K, frame.head_extrinsic, d, m[c])
        keep = _in_region(P) & (P[:, 2] > 0.70) & (P[:, 2] < 0.95)
        P = P[keep]
        UV = UV[keep]
        if len(P) < 30:
            out[c] = {"pos": None, "u": None, "n": int(len(P))}
            continue
        zt = float(np.max(P[:, 2]))
        top = P[P[:, 2] > zt - 0.004]
        if len(top) < 5:
            top = P
        cx = 0.5 * (float(np.min(top[:, 0])) + float(np.max(top[:, 0])))
        cy = 0.5 * (float(np.min(top[:, 1])) + float(np.max(top[:, 1])))
        out[c] = {"pos": [cx, cy, zt], "u": float(np.mean(UV[:, 0])), "n": int(len(P))}
    return out


def _color_count(frame, color, xy=None, radius=0.06):
    """Number of pixels of `color` (optionally only those within `radius` of world xy)."""
    m = _color_masks(frame.head_rgb)[color]
    P, _ = _backproject(frame.head_K, frame.head_extrinsic, frame.head_depth, m)
    keep = _in_region(P) & (P[:, 2] > 0.70) & (P[:, 2] < 0.95)
    P = P[keep]
    if xy is not None and len(P):
        dd = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1])
        P = P[dd < radius]
    return int(len(P))


def _detect_lids(frame):
    """Lid handle tops: list of [x, y, ztop] (bounding-box centre of the top face points)."""
    m = _color_masks(frame.head_rgb)["brown"]
    P, _ = _backproject(frame.head_K, frame.head_extrinsic, frame.head_depth, m)
    keep = _in_region(P) & (P[:, 2] > LID_TOP_BAND[0]) & (P[:, 2] < LID_TOP_BAND[1])
    P = P[keep]
    if len(P) < 20:
        return []
    # greedy clustering in xy (lids are >= 0.1 m apart; handle top is ~0.04 m wide)
    order = np.argsort(P[:, 0])
    P = P[order]
    clusters = []
    for p in P:
        placed = False
        for cl in clusters:
            if abs(p[0] - cl["cx"]) < 0.05 and abs(p[1] - cl["cy"]) < 0.05:
                cl["pts"].append(p)
                n = len(cl["pts"])
                cl["cx"] += (p[0] - cl["cx"]) / n
                cl["cy"] += (p[1] - cl["cy"]) / n
                placed = True
                break
        if not placed:
            clusters.append({"cx": float(p[0]), "cy": float(p[1]), "pts": [p]})
    lids = []
    for cl in clusters:
        Q = np.array(cl["pts"])
        if len(Q) < 25:
            continue
        zt = float(np.max(Q[:, 2]))
        top = Q[Q[:, 2] > zt - 0.006]
        cx = 0.5 * (float(np.min(top[:, 0])) + float(np.max(top[:, 0])))
        cy = 0.5 * (float(np.min(top[:, 1])) + float(np.max(top[:, 1])))
        lids.append([cx, cy, zt])
    lids.sort(key=lambda q: q[0])
    return lids
