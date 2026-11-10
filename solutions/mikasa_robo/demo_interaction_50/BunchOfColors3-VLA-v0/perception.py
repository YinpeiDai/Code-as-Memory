"""Perception helpers for BunchOfColors3 (front RGB-D -> coloured cubes and button)."""

FAMILIES = ("red", "orange", "yellow", "green", "cyan", "blue", "magenta")
FAMILY_HUE = (0.0, 44.0, 60.0, 120.0, 180.0, 240.0, 300.0)
# brightness split (top-face max channel) for the two families that hold a bright and a dark colour
DARK_SPLIT = {"red": 172.0, "cyan": 172.0}


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    pw = (pc - t) @ R
    return pw.reshape(H, W, 3), depth > 0


def _hue(rgb):
    r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0))
    return h * 60.0, mx, (mx - mn) / np.maximum(mx, 1e-6)


def _components(mask):
    """4-connected components of a boolean image; returns list of (rows, cols) arrays."""
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    comps = []
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0] >= 0:
            continue
        idx = len(comps)
        lab[y0, x0] = idx
        st = [(y0, x0)]
        pr = []; pc = []
        while st:
            y, x = st.pop()
            pr.append(y); pc.append(x)
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = idx
                    st.append((yy, xx))
        comps.append((np.array(pr), np.array(pc)))
    return comps


def _split_world(p, maxext=0.06):
    """Split a point set (N,3) into clusters if its world-xy extent is larger than one cube."""
    out = [np.arange(len(p))]
    res = []
    while out:
        ids = out.pop()
        q = p[ids, :2]
        ext = q.max(0) - q.min(0) if len(q) else np.zeros(2)
        if len(ids) < 8 or max(ext) <= maxext:
            res.append(ids)
            continue
        ax = int(np.argmax(ext))
        c = np.array([q[:, ax].min(), q[:, ax].max()])
        for _ in range(10):
            lab = np.abs(q[:, ax][:, None] - c[None, :]).argmin(1)
            for k in range(2):
                if (lab == k).any():
                    c[k] = q[lab == k, ax].mean()
        a = ids[lab == 0]; b = ids[lab == 1]
        if len(a) == 0 or len(b) == 0:
            res.append(ids)
            continue
        out.append(a); out.append(b)
    return res


def _detect_cubes(rgb, depth, K, E, min_px=12):
    """Detect saturated cubes standing on the table. Returns list of dicts with
    family, colour name, xy (top-face centre estimate), brightness, pixel count."""
    pw, ok = _backproject(depth, K, E)
    rgbf = rgb.astype(np.float64)
    hue, mx, sat = _hue(rgbf)
    z = pw[..., 2]
    base = ok & (z > 0.004) & (z < 0.055) & (sat > 0.8) & (mx > 50) \
        & (np.abs(pw[..., 0]) < 0.7) & (np.abs(pw[..., 1]) < 0.7)
    fam = np.full(hue.shape, -1)
    hd = np.abs(((hue[..., None] - np.array(FAMILY_HUE)[None, None, :]) + 180.0) % 360.0 - 180.0)
    best = hd.argmin(-1)
    fam = np.where(base & (hd.min(-1) < 25.0), best, -1)
    cubes = []
    for f in range(len(FAMILIES)):
        for rr, cc in _components(fam == f):
            if len(rr) < min_px:
                continue
            p = pw[rr, cc]
            for ids in _split_world(p):
                if len(ids) < min_px:
                    continue
                q = p[ids]
                top = q[:, 2] > 0.03
                if top.sum() < 4:
                    top = q[:, 2] > np.percentile(q[:, 2], 60)
                bright = float(np.median(mx[rr[ids], cc[ids]][top]))
                xy = q[top, :2].mean(0)
                cubes.append({"family": FAMILIES[f], "xy": [float(xy[0]), float(xy[1])],
                              "bright": bright, "n": int(len(ids)),
                              "ztop": float(np.percentile(q[:, 2], 95))})
    return cubes


def _name_colours(cubes):
    """Assign colour names; red/cyan families split into bright vs dark.
    If two cubes of such a family are present, the brighter is the bright colour."""
    out = []
    for c in cubes:
        c = dict(c)
        c["colour"] = c["family"]
        out.append(c)
    for famname, thr in DARK_SPLIT.items():
        members = [c for c in out if c["family"] == famname]
        if len(members) == 2:
            members.sort(key=lambda c: -c["bright"])
            members[0]["colour"] = famname
            members[1]["colour"] = "dark_" + famname
            members[0]["split"] = "rank"; members[1]["split"] = "rank"
        else:
            for c in members:
                c["colour"] = famname if c["bright"] >= thr else "dark_" + famname
                c["split"] = "thr"
    return out


def _button(rgb, depth, K, E):
    """Pink button cap: hue near red, moderate saturation, height above 0.045 m."""
    pw, ok = _backproject(depth, K, E)
    rgbf = rgb.astype(np.float64)
    hue, mx, sat = _hue(rgbf)
    z = pw[..., 2]
    m = ok & (z > 0.045) & (z < 0.09) & (sat > 0.2) & (sat < 0.6) & (mx > 120) \
        & ((hue < 20) | (hue > 340)) & (np.abs(pw[..., 0]) < 0.5) & (np.abs(pw[..., 1]) < 0.5)
    if m.sum() < 8:
        return None
    p = pw[m]
    return [float(np.median(p[:, 0])), float(np.median(p[:, 1])), float(np.percentile(p[:, 2], 90)), int(m.sum())]


def _scene(frame):
    rgb = frame.front_rgb; depth = frame.front_depth
    K = frame.front_K; E = frame.front_extrinsic
    cubes = _name_colours(_detect_cubes(rgb, depth, K, E))
    return cubes
