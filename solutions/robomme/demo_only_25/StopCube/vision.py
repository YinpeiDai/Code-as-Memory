"""Perception helpers for StopCube: height map from front depth, object components, classification."""

TABLE_X = (-0.42, 0.40)
TABLE_Y = (-0.48, 0.48)
OBJ_ZMIN = 0.006
OBJ_ZMAX = 0.07


def _world_map(depth, K, E):
    """Back-project an aligned depth image (int16 mm) into world XYZ (HxWx3) and a valid mask."""
    d = np.asarray(depth).astype(float) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    P = ((pc - t) @ R).reshape(H, W, 3)
    return P, d > 0


def _object_mask(P, valid):
    """Pixels on the table area raised above the tabletop."""
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    return (valid & (Z > OBJ_ZMIN) & (Z < OBJ_ZMAX) & (X > TABLE_X[0]) & (X < TABLE_X[1])
            & (Y > TABLE_Y[0]) & (Y < TABLE_Y[1]))


def _components(mask, min_pixels=12):
    """4-connected components of a boolean mask; returns list of (rows, cols) index arrays."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rs, cs = np.nonzero(mask)
    comps = []
    for r0, c0 in zip(rs.tolist(), cs.tolist()):
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        pr = []
        pcl = []
        while stack:
            r, c = stack.pop()
            pr.append(r)
            pcl.append(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if len(pr) >= min_pixels:
            comps.append((np.array(pr), np.array(pcl)))
    return comps


def _color_classes(rgb):
    im = np.asarray(rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    white = (mn > 195)
    gray = (mx - mn < 18) & (mx > 110) & (mn <= 195)
    purple = (b > g + 40) & (r > g + 20) & (b > 150)
    return white, gray, purple


def _scene(rgb, depth, K, E):
    """Find target disc, button and cube candidates in one front frame.

    Returns dict with 'target', 'button' (world xyz or None), 'others' list of
    (xyz centroid of top pixels, n pixels), plus raw maps for reuse.
    """
    P, valid = _world_map(depth, K, E)
    obj = _object_mask(P, valid)
    white, gray, purple = _color_classes(rgb)
    comps = _components(obj)
    target = None
    button = None
    others = []
    best_t = 0.0
    best_b = 0.0
    feats = []
    for rr, cc in comps:
        n = len(rr)
        z = P[rr, cc, 2]
        fw = float(white[rr, cc].mean())
        fg = float(gray[rr, cc].mean())
        fp = float(purple[rr, cc].mean())
        zmax = float(np.percentile(z, 95))
        xyz = P[rr, cc].mean(0)
        feats.append((rr, cc, n, fw, fg, fp, zmax, xyz))
    for i, f in enumerate(feats):
        rr, cc, n, fw, fg, fp, zmax, xyz = f
        st = fp * 2.0 + fw if (fp > 0.25 and fw > 0.05) else 0.0
        sb = fw + fg if (fw > 0.2 and fg > 0.08 and fp < 0.05) else 0.0
        if st > best_t:
            best_t = st
            target = i
        if sb > best_b:
            best_b = sb
            button = i
    res = {"target": None, "button": None, "others": [], "P": P, "obj": obj}
    if target is not None:
        rr, cc = feats[target][0], feats[target][1]
        sel = purple[rr, cc] | white[rr, cc]
        res["target"] = P[rr[sel], cc[sel]].mean(0)
        res["target_px"] = (rr, cc)
    if button is not None:
        rr, cc = feats[button][0], feats[button][1]
        z = P[rr, cc, 2]
        top = z > z.max() - 0.006
        res["button"] = P[rr[top], cc[top]].mean(0)
        res["button_top"] = float(np.percentile(z, 97))
        res["button_px"] = (rr, cc)
    for i, f in enumerate(feats):
        if i != target and i != button:
            res["others"].append((f[7], f[2], f[6]))
    return res


def _dilate(mask, k=2):
    m = mask.copy()
    for _ in range(k):
        n = m.copy()
        n[1:, :] |= m[:-1, :]
        n[:-1, :] |= m[1:, :]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def _cube_xy(rgb, depth, K, E, target_xy, button_xy, target_r=0.035):
    """Cube centroid (world xy) of raised pixels that are not the static target disc or button.

    Pixels over the target disc count only if they rise clearly above the disc top.
    Returns (xy or None, n_pixels).
    """
    P, valid = _world_map(depth, K, E)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    obj = _object_mask(P, valid) & (Z < 0.045)
    dt = np.hypot(X - target_xy[0], Y - target_xy[1])
    db = np.hypot(X - button_xy[0], Y - button_xy[1])
    m = obj & (db > 0.05) & ((dt > target_r) | (Z > 0.024))
    comps = _components(m, min_pixels=8)
    if not comps:
        return None, 0
    rr, cc = max(comps, key=lambda q: len(q[0]))
    z = Z[rr, cc]
    top = z > z.max() - 0.008
    xy = P[rr[top], cc[top], :2].mean(0)
    return [float(xy[0]), float(xy[1])], int(len(rr))
