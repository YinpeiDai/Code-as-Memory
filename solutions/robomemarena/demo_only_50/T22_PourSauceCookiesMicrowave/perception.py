"""Scene perception for T22 from the agentview RGB-D frame (no motion)."""

# Nominal object positions (world metres) observed across the assigned demonstrations; used only
# as search priors and as a fallback when a cluster cannot be found.
_NOM_SAUCE = (-0.02, -0.195)
_NOM_COOKIE = (0.172, -0.194)
_NOM_TABLE_Z = 0.43


def _cloud(frame):
    """Back-project the agentview depth into world XYZ; returns (N,3) points and (N,3) rgb."""
    depth = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], -1).reshape(-1, 3)
    ok = cam[:, 2] > 0.05
    R = E[:, :3]
    t = E[:, 3]
    world = (cam - t) @ R
    rgb = np.asarray(frame.agentview_rgb).reshape(-1, 3)
    return world[ok], rgb[ok]


def _table_height(p):
    """Mode of point heights over the table workspace (the table top is the dominant plane)."""
    m = (p[:, 0] > -0.3) & (p[:, 0] < 0.45) & (p[:, 1] > -0.5) & (p[:, 1] < 0.3) & (p[:, 2] > 0.2) & (p[:, 2] < 0.7)
    zs = p[m, 2]
    if len(zs) < 200:
        return _NOM_TABLE_Z, False
    hist, edges = np.histogram(zs, bins=np.arange(0.2, 0.7, 0.002))
    k = int(np.argmax(hist))
    z0 = 0.5 * (edges[k] + edges[k + 1])
    near = zs[np.abs(zs - z0) < 0.004]
    return float(np.median(near)), True


def _cluster(p, center, radius, zlo, zhi, iters=3, shrink=None):
    """Iteratively re-centred cylinder cluster of points with heights in (zlo, zhi)."""
    c = np.array(center, dtype=float)
    sel = None
    for i in range(iters):
        r = radius if (i == 0 or shrink is None) else shrink
        d = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
        m = (d < r) & (p[:, 2] > zlo) & (p[:, 2] < zhi)
        if m.sum() < 15:
            return None
        sel = p[m]
        c = sel[:, :2].mean(0)
    return sel


def _perceive_scene(frame):
    """Locate table height, tomato-sauce can, cookie box and microwave from one agentview RGB-D frame."""
    p, rgb = _cloud(frame)
    tz, tz_ok = _table_height(p)
    out = {"table_z": tz, "table_ok": tz_ok}
    # sauce can: a 7-8 cm tall object near the left-front of the table
    s = _cluster(p, _NOM_SAUCE, 0.12, tz + 0.008, tz + 0.16, iters=4, shrink=0.05)
    if s is not None:
        out["sauce"] = [float(s[:, 0].mean()), float(s[:, 1].mean()), float(s[:, 2].max())]
        out["sauce_n"] = int(len(s))
    else:
        out["sauce"] = None
    # cookie box: a flat (~1.5 cm) box; exclude the sauce neighbourhood
    q = p
    if out["sauce"] is not None:
        q = p[np.hypot(p[:, 0] - out["sauce"][0], p[:, 1] - out["sauce"][1]) > 0.07]
    c = _cluster(q, _NOM_COOKIE, 0.12, tz + 0.004, tz + 0.04, iters=4, shrink=0.07)
    if c is not None:
        out["cookie"] = [float(c[:, 0].mean()), float(c[:, 1].mean()), float(c[:, 2].max())]
        out["cookie_n"] = int(len(c))
    else:
        out["cookie"] = None
    # microwave: the large box on the right side of the table (y > 0.15)
    m = (p[:, 1] > 0.15) & (p[:, 1] < 0.8) & (p[:, 0] > -0.45) & (p[:, 0] < 0.4) & (p[:, 2] > tz + 0.015) & (p[:, 2] < tz + 0.4)
    mw = p[m]
    if len(mw) > 300:
        lo = np.percentile(mw, 0.3, axis=0)
        hi = np.percentile(mw, 99.7, axis=0)
        out["mw_lo"] = [float(v) for v in lo]
        out["mw_hi"] = [float(v) for v in hi]
        out["mw_n"] = int(len(mw))
    else:
        out["mw_lo"] = None
        out["mw_hi"] = None
    return out
