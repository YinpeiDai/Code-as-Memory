"""Perception helpers for press_button: find the two number tiles in the head camera, rectify each onto the
table plane (top-down view), segment the dark digit and classify it against _DIGIT_TEMPLATES."""

_TABLE_Z = 0.745  # tile surface height measured from head depth in all 50 demos (0.741-0.747)


def _components(mask, min_size=1):
    """4-connected components of a boolean mask; returns a list of (N,2) int arrays of (row, col)."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    out = []
    ys, xs = np.nonzero(mask)
    n = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        stack = [(y0, x0)]
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            if y + 1 < H and mask[y + 1, x] and not lab[y + 1, x]:
                lab[y + 1, x] = n
                stack.append((y + 1, x))
            if y > 0 and mask[y - 1, x] and not lab[y - 1, x]:
                lab[y - 1, x] = n
                stack.append((y - 1, x))
            if x + 1 < W and mask[y, x + 1] and not lab[y, x + 1]:
                lab[y, x + 1] = n
                stack.append((y, x + 1))
            if x > 0 and mask[y, x - 1] and not lab[y, x - 1]:
                lab[y, x - 1] = n
                stack.append((y, x - 1))
        if len(pts) >= min_size:
            out.append(np.array(pts, dtype=np.int64))
    return out


def _ray_plane(u, v, K, E, zp):
    R = E[:, :3]
    t = E[:, 3]
    C = -R.T @ t
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ R
    s = (zp - C[2]) / d[:, 2]
    return C[None, :] + s[:, None] * d


def _deproject(u, v, depth_mm, K, E):
    z = depth_mm / 1000.0
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return (pc - E[:, 3]) @ E[:, :3]


def _project(P, K, E):
    pc = P @ E[:, :3].T + E[:, 3]
    return K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]


def _bilinear(img, u, v):
    H, W = img.shape
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    a = u - u0
    b = v - v0
    return (img[v0, u0] * (1 - a) * (1 - b) + img[v0, u0 + 1] * a * (1 - b)
            + img[v0 + 1, u0] * (1 - a) * b + img[v0 + 1, u0 + 1] * a * b)


def _resample(patch, GH, GW):
    H, W = patch.shape
    ys = np.clip((np.arange(GH) + 0.5) * H / GH - 0.5, 0, H - 1)
    xs = np.clip((np.arange(GW) + 0.5) * W / GW - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    return (patch[y0][:, x0] * (1 - wy) * (1 - wx) + patch[y1][:, x0] * wy * (1 - wx)
            + patch[y0][:, x1] * (1 - wy) * wx + patch[y1][:, x1] * wy * wx)


def _find_tiles(rgb):
    """Tan tile components (sorted by image column, i.e. world -x first for the head camera)."""
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    red = (r > 140) & (g < 100) & (b < 100)
    tan = (r - b > 25) & (r > 120) & ~red
    cs = [c for c in _components(tan) if len(c) > 300]
    cs = sorted(cs, key=lambda c: -len(c))[:2]
    return sorted(cs, key=lambda c: float(c[:, 1].mean()))


def _rectify_tile(rgb, depth, comp, K, E, res=0.002):
    v = comp[:, 0].astype(float)
    u = comp[:, 1].astype(float)
    zp = _TABLE_Z
    if depth is not None:
        dd = depth[comp[:, 0], comp[:, 1]].astype(float)
        ok = dd > 0
        if ok.sum() > 50:
            zw = float(np.median(_deproject(u[ok], v[ok], dd[ok], K, E)[:, 2]))
            if 0.65 < zw < 0.85:
                zp = zw
    P = _ray_plane(u, v, K, E, zp)[:, :2]
    cen = P.mean(0)
    Q = P - cen
    best = None
    for deg in range(-80, 81):
        ang = math.radians(deg * 0.5)
        ca, sa = math.cos(ang), math.sin(ang)
        x = Q[:, 0] * ca + Q[:, 1] * sa
        y = -Q[:, 0] * sa + Q[:, 1] * ca
        x0, x1 = np.percentile(x, 1), np.percentile(x, 99)
        y0, y1 = np.percentile(y, 1), np.percentile(y, 99)
        area = (x1 - x0) * (y1 - y0)
        if best is None or area < best[0]:
            best = (area, ang, x0, x1, y0, y1)
    area, ang, x0, x1, y0, y1 = best
    ca, sa = math.cos(ang), math.sin(ang)
    xs = np.arange(x0, x1, res)
    ys = np.arange(y1, y0, -res)
    X, Y = np.meshgrid(xs, ys)
    wx = cen[0] + X * ca - Y * sa
    wy = cen[1] + X * sa + Y * ca
    Pw = np.stack([wx.ravel(), wy.ravel(), np.full(wx.size, zp)], -1)
    uu, vv = _project(Pw, K, E)
    gray = rgb.astype(float).mean(-1)
    img = _bilinear(gray, uu, vv).reshape(X.shape)
    return img, [float(cen[0]), float(cen[1]), float(zp)], math.degrees(ang)


def _digit_feature(img):
    H, W = img.shape
    m = 0.15
    sub = img[int(H * m):int(H * (1 - m)), int(W * m):int(W * (1 - m))]
    bg = float(np.median(sub))
    thr = 0.5 * (bg + 60.0)
    dark = sub < thr
    cc = _components(dark)
    if not cc:
        return None
    big = max(cc, key=len)
    if len(big) < 20:
        return None
    keep = [x for x in cc if len(x) >= 0.2 * len(big)]
    pts = np.concatenate(keep)
    a0, a1 = int(pts[:, 0].min()), int(pts[:, 0].max()) + 1
    b0, b1 = int(pts[:, 1].min()), int(pts[:, 1].max()) + 1
    patch = np.clip((bg - sub[a0:a1, b0:b1]) / max(bg - 60.0, 1.0), 0, 1)
    h, w = patch.shape
    W2 = max(w, int(round(0.75 * h)))
    P = np.zeros((h, W2))
    off = (W2 - w) // 2
    P[:, off:off + w] = patch
    return _resample(P, 14, 10).ravel()


def _classify(f):
    def nz(x):
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x
    fn = nz(f)
    scores = {}
    for k, t in _DIGIT_TEMPLATES.items():
        scores[int(k)] = float(nz(t) @ fn)
    order = sorted(scores, key=lambda k: -scores[k])
    return order[0], scores[order[0]], scores[order[0]] - scores[order[1]]


def _read_numbers(frame, camera="head"):
    """Returns dict with left/right digits (world -x tile first), scores, margins, tile centres; or error."""
    rgb = getattr(frame, camera + "_rgb", None)
    if rgb is None:
        return {"ok": False, "error": "no " + camera + " rgb"}
    depth = getattr(frame, camera + "_depth", None)
    K = np.asarray(getattr(frame, camera + "_K"), dtype=float)
    E = np.asarray(getattr(frame, camera + "_extrinsic"), dtype=float)
    tiles = _find_tiles(rgb)
    if len(tiles) != 2:
        return {"ok": False, "error": "found %d tiles" % len(tiles)}
    res = []
    for comp in tiles:
        img, cen, ang = _rectify_tile(rgb, depth, comp, K, E)
        f = _digit_feature(img)
        if f is None:
            return {"ok": False, "error": "no digit strokes in tile"}
        d, s, mg = _classify(f)
        res.append((cen, d, s, mg, ang))
    res = sorted(res, key=lambda r: r[0][0])  # world x: left tile first
    return {"ok": True, "left": res[0][1], "right": res[1][1],
            "scores": [round(res[0][2], 3), round(res[1][2], 3)],
            "margins": [round(res[0][3], 3), round(res[1][3], 3)],
            "tile_centres": [res[0][0], res[1][0]], "tile_yaw_deg": [res[0][4], res[1][4]]}


def _find_buttons(frame, camera="head"):
    """Red button tops in world coordinates (sorted by x) from head RGB-D; [] when unavailable."""
    rgb = getattr(frame, camera + "_rgb", None)
    depth = getattr(frame, camera + "_depth", None)
    if rgb is None or depth is None:
        return []
    K = np.asarray(getattr(frame, camera + "_K"), dtype=float)
    E = np.asarray(getattr(frame, camera + "_extrinsic"), dtype=float)
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    red = (r > 140) & (g < 100) & (b < 100)
    out = []
    for c in _components(red):
        if len(c) < 30:
            continue
        dd = depth[c[:, 0], c[:, 1]].astype(float)
        ok = dd > 0
        if ok.sum() < 10:
            continue
        w = _deproject(c[ok, 1].astype(float), c[ok, 0].astype(float), dd[ok], K, E)
        out.append([float(np.median(w[:, 0])), float(np.median(w[:, 1])), float(np.max(w[:, 2]))])
    return sorted(out, key=lambda p: p[0])
