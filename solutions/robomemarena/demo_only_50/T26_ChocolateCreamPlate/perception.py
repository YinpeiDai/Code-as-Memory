"""Perception helpers for T26 (chocolate pudding + cream cheese, plate1 -> plate2).

Agentview RGB-D is back-projected into world coordinates with the supplied
intrinsics/extrinsics; objects are segmented by colour and height above the
table (z ~ 0.430 m in every assigned demonstration).
"""

TABLE_Z = 0.430


def _cloud(frame, cam="agentview"):
    depth = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    valid = (depth.reshape(-1) > 0.05)
    return pw.reshape(h, w, 3), valid.reshape(h, w)


def _masks(frame):
    P, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    tab = valid & (x > -0.15) & (x < 0.45) & (np.abs(y) < 0.45) & (z > 0.36) & (z < 0.62)
    hi = z > TABLE_Z + 0.007
    blue = tab & hi & (b > r + 15) & (b > g + 5)
    s = r + g + b
    # plate rims are pinkish (sum 270-330) and at most ~1.4 cm above the table
    brown = tab & hi & (r > b + 20) & (r < 140) & ((s < 265) | ((z > TABLE_Z + 0.016) & (s < 330)))
    white = tab & (z < TABLE_Z + 0.02) & (r > 185) & (g > 185) & (b > 185)
    return P, blue, brown, white


def _blob(points, min_n=12):
    """Robust single-object estimate: median, keep points within 4 cm, top surface."""
    if len(points) < min_n:
        return None
    c = np.median(points[:, :2], 0)
    for _ in range(2):
        d = np.sqrt(((points[:, :2] - c) ** 2).sum(1))
        keep = points[d < 0.04]
        if len(keep) < min_n:
            return None
        c = np.median(keep[:, :2], 0)
    top = float(np.percentile(keep[:, 2], 95))
    q = keep[keep[:, 2] > top - 0.012]
    return {"x": float(np.median(q[:, 0])), "y": float(np.median(q[:, 1])),
            "top": top, "n": int(len(keep))}


def _detect(frame):
    """Detect chocolate (brown), cream cheese (blue) and white plate regions."""
    P, blue, brown, white = _masks(frame)
    choc = _blob(P[brown])
    cream = _blob(P[blue])
    wp = P[white]
    return {"choc": choc, "cream": cream, "white": wp}


def _plate2(white_pts, avoid):
    """Plate2 centre: white plate points far (>0.12 m in xy) from every avoided xy."""
    if len(white_pts) < 20:
        return None
    keep = np.ones(len(white_pts), bool)
    for a in avoid:
        d = np.sqrt(((white_pts[:, :2] - np.array(a[:2])) ** 2).sum(1))
        keep &= d > 0.12
    p = white_pts[keep]
    if len(p) < 20:
        return None
    c = np.median(p[:, :2], 0)
    d = np.sqrt(((p[:, :2] - c) ** 2).sum(1))
    p = p[d < 0.08]
    if len(p) < 20:
        return None
    return [float(np.median(p[:, 0])), float(np.median(p[:, 1]))]
