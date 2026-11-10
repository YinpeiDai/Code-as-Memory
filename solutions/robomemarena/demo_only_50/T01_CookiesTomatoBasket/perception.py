"""Perception helpers: agentview RGB-D back-projection and table-top object segmentation.

All positions are world metres. Objects are identified by geometry measured from
depth (height above the table and footprint), not by colour templates:
  cookies  - flat box, about 1.5 cm above the table
  tomato   - tomato-sauce can, about 7.6 cm tall, narrow
  basket   - large (> 10 cm) and tall (> 10 cm) containers, two of them
"""

# Nominal values measured from all 50 assigned demonstrations (frame 0); used
# only as fallbacks when depth segmentation does not find an object.
NOMINAL_TABLE_Z = 0.430
NOMINAL_COOKIES = [-0.140, 0.084, 0.444]
NOMINAL_TOMATO = [-0.069, -0.062, 0.503]
NOMINAL_BASKET = [-0.119, 0.551, 0.567]
CELL = 0.01


def _world_points(frame, cam):
    """Back-project a camera depth image to world XYZ, shape (H, W, 3); invalid depth -> nan."""
    depth = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    c = np.stack([x, y, depth], -1)
    w = (c - E[:, 3]) @ E[:, :3]
    w[depth <= 0] = np.nan
    return w


def _table_height(P):
    """Mode of world z among points over the central table area."""
    sel = (np.abs(P[:, 0]) < 0.4) & (np.abs(P[:, 1]) < 0.7) & (P[:, 2] > 0.2) & (P[:, 2] < 0.7)
    z = P[sel, 2]
    if len(z) < 500:
        return NOMINAL_TABLE_Z
    hist, edges = np.histogram(z, bins=np.arange(0.2, 0.7, 0.002))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2.0)


def _components(P, min_count=1, cell=CELL):
    """Connected components of occupied XY grid cells (8-connected); returns index arrays into P."""
    if len(P) == 0:
        return []
    ix = np.floor((P[:, 0] + 0.6) / cell).astype(int)
    iy = np.floor((P[:, 1] + 0.9) / cell).astype(int)
    nx, ny = int(ix.max()) + 2, int(iy.max()) + 2
    flat = ix * ny + iy
    counts = np.bincount(flat, minlength=nx * ny)
    occ = counts >= min_count
    label = -np.ones(nx * ny, dtype=int)
    cells = np.flatnonzero(occ)
    n = 0
    for c in cells:
        if label[c] >= 0:
            continue
        label[c] = n
        stack = [int(c)]
        while stack:
            k = stack.pop()
            kx, ky = k // ny, k % ny
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    jx, jy = kx + dx, ky + dy
                    if 0 <= jx < nx and 0 <= jy < ny:
                        j = jx * ny + jy
                        if occ[j] and label[j] < 0:
                            label[j] = n
                            stack.append(j)
        n += 1
    pl = label[flat]
    return [np.flatnonzero(pl == i) for i in range(n)]


def _split_x(Q):
    """Split a merged two-basket cluster along x with 2-means; returns two index arrays."""
    x = Q[:, 0]
    c = [float(np.percentile(x, 15)), float(np.percentile(x, 85))]
    a = x < (c[0] + c[1]) / 2
    for _ in range(20):
        if a.all() or (~a).all():
            break
        c = [float(x[a].mean()), float(x[~a].mean())]
        a = x < (c[0] + c[1]) / 2
    return [np.flatnonzero(a), np.flatnonzero(~a)]


def _describe(Q, table_z):
    zt = float(np.percentile(Q[:, 2], 98))
    top = Q[Q[:, 2] > zt - 0.006]
    if len(top) < 3:
        top = Q
    x0, x1 = np.percentile(Q[:, 0], [2, 98])
    y0, y1 = np.percentile(Q[:, 1], [2, 98])
    return {
        "n": int(len(Q)),
        "height": zt - table_z,
        "top_z": zt,
        "top_xy": [float(top[:, 0].mean()), float(top[:, 1].mean())],
        "extent": [float(x1 - x0), float(y1 - y0)],
        "box_mid": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
    }


def _basket_info(Q, table_z):
    d = _describe(Q, table_z)
    rim = Q[Q[:, 2] > d["top_z"] - 0.012]
    if len(rim) >= 10:
        x0, x1 = np.percentile(rim[:, 0], [1, 99])
        y0, y1 = np.percentile(rim[:, 1], [1, 99])
        d["center"] = [float((x0 + x1) / 2), float((y0 + y1) / 2)]
    else:
        d["center"] = d["box_mid"]
    return d


def _scene(frame, exclude_xyz=None):
    """Segment the table top seen by the agentview camera.

    Returns {table_z, cookies, tomato, baskets, found}. cookies / tomato are
    [x, y, top_z]; baskets is a list of [x, y, rim_z] sorted by x. exclude_xyz
    (normally the measured gripper position) removes points within 8 cm
    horizontally that are higher than 6 cm below it, so the hand and a held
    object are not segmented as table objects.
    """
    W = _world_points(frame, "agentview")
    P = W.reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    table_z = _table_height(P)
    sel = (P[:, 2] > table_z + 0.007) & (P[:, 2] < table_z + 0.20)
    sel &= (np.abs(P[:, 0]) < 0.42) & (np.abs(P[:, 1]) < 0.75)
    if exclude_xyz is not None:
        g = exclude_xyz
        near = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1]) < 0.08
        sel &= ~(near & (P[:, 2] > g[2] - 0.06))
    Q = P[sel]
    # baskets: clusters of tall points (rims and walls higher than 9 cm)
    tall = Q[Q[:, 2] > table_z + 0.09]
    baskets = []
    for idx in _components(tall, min_count=2, cell=0.03):
        S = tall[idx]
        if len(S) < 30:
            continue
        x0, x1 = np.percentile(S[:, 0], [1, 99])
        y0, y1 = np.percentile(S[:, 1], [1, 99])
        if max(x1 - x0, y1 - y0) < 0.08:
            continue
        zt = float(np.percentile(S[:, 2], 98))
        baskets.append([float((x0 + x1) / 2), float((y0 + y1) / 2), zt,
                        float(x0), float(x1), float(y0), float(y1)])
    keep = np.ones(len(Q), dtype=bool)
    for b in baskets:
        inside = (Q[:, 0] > b[3] - 0.015) & (Q[:, 0] < b[4] + 0.015)
        inside &= (Q[:, 1] > b[5] - 0.015) & (Q[:, 1] < b[6] + 0.015)
        keep &= ~inside
    R = Q[keep]
    cookies, tomato = [], []
    for idx in _components(R):
        if len(idx) < 12:
            continue
        S = R[idx]
        zt = float(np.percentile(S[:, 2], 98))
        h = zt - table_z
        x0, x1 = np.percentile(S[:, 0], [2, 98])
        y0, y1 = np.percentile(S[:, 1], [2, 98])
        big = max(x1 - x0, y1 - y0)
        band = 0.004 if h < 0.035 else 0.008
        top = S[S[:, 2] > S[:, 2].max() - band]
        if len(top) < 4:
            top = S[S[:, 2] > zt - 0.006]
        d = [float(top[:, 0].mean()), float(top[:, 1].mean()), zt, len(S)]
        if 0.045 < h < 0.10 and big < 0.11:
            tomato.append(d)
        elif 0.006 < h < 0.035 and big < 0.15:
            cookies.append(d)
    ck = max(cookies, key=lambda d: d[3]) if cookies else None
    tm = max(tomato, key=lambda d: d[3]) if tomato else None
    return {
        "table_z": table_z,
        "cookies": ck[:3] if ck else None,
        "tomato": tm[:3] if tm else None,
        "baskets": sorted([b[:3] for b in baskets]),
        "found": {"cookies": ck is not None, "tomato": tm is not None, "basket": len(baskets) > 0},
        "n_cookie_like": len(cookies),
        "n_tomato_like": len(tomato),
    }
