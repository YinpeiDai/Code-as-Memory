"""Perception helpers: agentview RGB-D back-projection, table-plane estimate, object detection."""


def _backproject(depth, K, E):
    """World XYZ (HxWx3) of every pixel; depth int16 mm, K 3x3, E 3x4 world->camera."""
    d = np.asarray(depth, dtype=float) / 1000.0
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    x = (u - K[0][2]) / K[0][0] * d
    y = (v - K[1][2]) / K[1][1] * d
    Pc = np.stack([x, y, d], -1).reshape(-1, 3)
    R = np.asarray(E, dtype=float)[:, :3]
    t = np.asarray(E, dtype=float)[:, 3]
    Pw = (Pc - t) @ R
    return Pw.reshape(H, W, 3), d


def _table_height(Pw, valid):
    """Mode of world z over plausible table-top points (1 mm bins)."""
    z = Pw[..., 2]
    sel = valid & (np.abs(Pw[..., 0]) < 0.5) & (np.abs(Pw[..., 1]) < 0.9) & (z > 0.2) & (z < 0.8)
    zs = z[sel]
    if zs.size < 100:
        return None
    hist, edges = np.histogram(zs, bins=np.arange(0.2, 0.8, 0.002))
    i = int(np.argmax(hist))
    near = zs[(zs > edges[i] - 0.004) & (zs < edges[i + 1] + 0.004)]
    return float(np.median(near))


def _components(mask, min_size, Pw=None, link=0.02):
    """8-connected components of a boolean mask; with Pw, neighbours link only when their
    world points are closer than `link` metres. Returns list of (rows, cols) arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=int)
    vs, us = np.nonzero(mask)
    out = []
    n = 0
    for k in range(len(vs)):
        v0 = int(vs[k]); u0 = int(us[k])
        if lab[v0, u0]:
            continue
        n += 1
        lab[v0, u0] = n
        stack = [(v0, u0)]
        rows = []; cols = []
        while stack:
            v, u = stack.pop()
            rows.append(v); cols.append(u)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    a = v + dv; b = u + du
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and not lab[a, b]:
                        if Pw is not None:
                            dd = Pw[a, b] - Pw[v, u]
                            if dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2] > link * link:
                                continue
                        lab[a, b] = n
                        stack.append((a, b))
        if len(rows) >= min_size:
            out.append((np.array(rows), np.array(cols)))
    return out


def _detect_scene(frame):
    """Detect cookies box, tomato-sauce can and baskets in the agentview RGB-D frame.

    Returns dict: table_z, cookies, tomato (each None or dict with centroid xyz, top z, size, rgb),
    baskets (list sorted by world x ascending = farther from the camera first).
    """
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    depth = np.asarray(frame.agentview_depth)
    rgb = np.asarray(frame.agentview_rgb).astype(float)
    Pw, d = _backproject(depth, K, E)
    valid = depth > 0
    tz = _table_height(Pw, valid)
    res = {"table_z": tz, "cookies": None, "tomato": None, "baskets": [], "blobs": []}
    if tz is None:
        return res
    z = Pw[..., 2]
    tcp = np.asarray(frame.tcp, dtype=float)[:3]
    horiz = np.sqrt((Pw[..., 0] - tcp[0]) ** 2 + (Pw[..., 1] - tcp[1]) ** 2)
    near_arm = (horiz < 0.09) & (z > tcp[2] - 0.06)
    m = (valid & (z > tz + 0.006) & (z < tz + 0.22) & (np.abs(Pw[..., 0]) < 0.45)
         & (Pw[..., 1] > -0.7) & (Pw[..., 1] < 0.85) & ~near_arm)
    blobs = []
    for rows, cols in _components(m, 12, Pw, 0.02):
        P = Pw[rows, cols]
        c = rgb[rows, cols].mean(0)
        b = {
            "n": int(len(rows)),
            "xyz": [float(v) for v in P.mean(0)],
            "top": float(P[:, 2].max() - tz),
            "height_q": float(np.percentile(P[:, 2], 90) - tz),
            "xr": [float(P[:, 0].min()), float(P[:, 0].max())],
            "yr": [float(P[:, 1].min()), float(P[:, 1].max())],
            "rgb": [float(v) for v in c],
            "uv": [float(cols.mean()), float(rows.mean())],
        }
        blobs.append(b)
    res["blobs"] = blobs
    baskets = []
    smalls = []
    for b in blobs:
        ext = max(b["xr"][1] - b["xr"][0], b["yr"][1] - b["yr"][0])
        if b["height_q"] > 0.09 and ext > 0.1 and b["n"] > 150:
            baskets.append(b)
        elif ext < 0.14 and b["height_q"] < 0.12:
            smalls.append(b)
    baskets.sort(key=lambda b: b["xyz"][0])
    for b in baskets:
        b["center"] = [0.5 * (b["xr"][0] + b["xr"][1]), 0.5 * (b["yr"][0] + b["yr"][1]), b["xyz"][2]]
    res["baskets"] = baskets
    # drop fragments (shadowed basket walls) lying inside an expanded basket footprint
    kept = []
    for b in smalls:
        x, y = b["xyz"][0], b["xyz"][1]
        inside = False
        for k in baskets:
            if k["xr"][0] - 0.03 < x < k["xr"][1] + 0.03 and k["yr"][0] - 0.03 < y < k["yr"][1] + 0.03:
                inside = True
        if not inside:
            kept.append(b)
    smalls = kept
    # tomato sauce: taller dark can; cookies: low brown/red box
    best_t = None; best_c = None
    for b in smalls:
        r, g, bl = b["rgb"]
        bright = (r + g + bl) / 3.0
        h = b["height_q"]
        st = (1.0 if 0.04 < h < 0.11 else 0.0) + (1.0 if bright < 75 else 0.0) + min(b["n"], 150) / 300.0
        sc = (1.0 if h < 0.04 else 0.0) + (1.0 if (r - bl) > 25 else 0.0) + min(b["n"], 150) / 300.0
        if st >= 2.0 and (best_t is None or st > best_t[0]):
            best_t = (st, b)
        if sc >= 2.0 and (best_c is None or sc > best_c[0]):
            best_c = (sc, b)
    if best_t is not None:
        res["tomato"] = best_t[1]
    if best_c is not None and (best_t is None or best_c[1] is not best_t[1]):
        res["cookies"] = best_c[1]
    return res
