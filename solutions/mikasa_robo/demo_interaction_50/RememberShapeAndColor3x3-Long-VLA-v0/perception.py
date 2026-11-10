"""Perception helpers for RememberShapeAndColor3x3: colour segmentation, 3-D back-projection,
shape classification (cube / sphere / T) and scene parsing from the wrist RGB-D camera."""

COLORS = ("red", "green", "blue")
T_TOP_MAX = 0.028      # T-shapes are flat (top ~0.016 m); cubes/spheres top ~0.040 m
SPHERE_RES_MAX = 2.0   # mm RMS residual of a r=0.02 m sphere fit; spheres ~0.3, cubes/T > 4.5
T_ORIGIN_DY = -0.027   # T reference point lies 2.7 cm in world -y from the T's top-face centroid


def _masks(rgb):
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "red": (r > 100) & (g * 10 < 3 * r) & (b * 10 < 3 * r),
        "green": (g > r + 40) & (g > b + 40),
        "blue": (b > r + 40) & (b > g + 40),
    }


def _color_counts(rgb):
    m = _masks(rgb)
    return {c: int(m[c].sum()) for c in COLORS}


def _label(mask):
    """4-connected components; returns list of (ys, xs) index arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    comps = []
    ys0, xs0 = np.nonzero(mask)
    n = 0
    for y0, x0 in zip(ys0.tolist(), xs0.tolist()):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        st = [(y0, x0)]
        py, px = [], []
        while st:
            y, x = st.pop()
            py.append(y)
            px.append(x)
            if y + 1 < H and mask[y + 1, x] and not lab[y + 1, x]:
                lab[y + 1, x] = n
                st.append((y + 1, x))
            if y > 0 and mask[y - 1, x] and not lab[y - 1, x]:
                lab[y - 1, x] = n
                st.append((y - 1, x))
            if x + 1 < W and mask[y, x + 1] and not lab[y, x + 1]:
                lab[y, x + 1] = n
                st.append((y, x + 1))
            if x > 0 and mask[y, x - 1] and not lab[y, x - 1]:
                lab[y, x - 1] = n
                st.append((y, x - 1))
        comps.append((np.array(py), np.array(px)))
    return comps


def _backproject(xs, ys, d, K, E):
    z = d / 1000.0
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


def _sphere_fit(P):
    cx, cy = float(np.median(P[:, 0])), float(np.median(P[:, 1]))
    for _ in range(25):
        v = P - np.array([cx, cy, 0.02])
        r = np.linalg.norm(v, axis=1)
        g = (r - 0.02) / np.maximum(r, 1e-6)
        cx += float(np.mean(g * v[:, 0]))
        cy += float(np.mean(g * v[:, 1]))
    res = np.linalg.norm(P - np.array([cx, cy, 0.02]), axis=1) - 0.02
    return cx, cy, float(np.sqrt(np.mean(res ** 2)) * 1000.0)


def _features(P, npix, color):
    z = P[:, 2]
    zt = float(np.percentile(z, 95))
    top = P[z > zt - 0.005]
    tx, ty = float(np.median(top[:, 0])), float(np.median(top[:, 1]))
    cx, cy, sres = _sphere_fit(P)
    return {"color": color, "n": int(npix), "zt": zt, "sres": sres,
            "tx": tx, "ty": ty, "sx": cx, "sy": cy,
            "mx": float(np.median(P[:, 0])), "my": float(np.median(P[:, 1]))}


def _split2(P):
    """2-means split of a merged blob in world xy."""
    xy = P[:, :2]
    a = xy[np.argmax(np.linalg.norm(xy - xy.mean(0), axis=1))]
    b = xy[np.argmax(np.linalg.norm(xy - a, axis=1))]
    lab = np.zeros(len(xy), dtype=bool)
    for _ in range(15):
        lab = np.linalg.norm(xy - a, axis=1) > np.linalg.norm(xy - b, axis=1)
        if lab.all() or (~lab).all():
            break
        a = xy[~lab].mean(0)
        b = xy[lab].mean(0)
    return P[~lab], P[lab], float(np.linalg.norm(a - b))


def _abs_shape(f):
    if f["zt"] < T_TOP_MAX:
        return "T"
    return "sphere" if f["sres"] < SPHERE_RES_MAX else "cube"


def _center(f, shape):
    if shape == "sphere":
        return [f["sx"], f["sy"]]
    if shape == "T":
        return [f["tx"], f["ty"] + T_ORIGIN_DY]
    return [f["tx"], f["ty"]]


def _detect(frame, cam="wrist", minpix=15, expect=3):
    """Detect coloured objects; per colour, split merged blobs until `expect` are found."""
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    masks = _masks(rgb)
    out = []
    for color in COLORS:
        groups = []
        for ys, xs in _label(masks[color]):
            if len(ys) < minpix:
                continue
            d = depth[ys, xs].astype(float)
            ok = d > 0
            if ok.sum() < minpix:
                continue
            P = _backproject(xs[ok].astype(float), ys[ok].astype(float), d[ok], K, E)
            groups.append(P)
        guard = 0
        while expect and 0 < len(groups) < expect and guard < 3:
            guard += 1
            groups.sort(key=lambda q: -len(q))
            A, B, sep = _split2(groups[0])
            if sep < 0.025 or len(A) < minpix or len(B) < minpix:
                break
            groups = [A, B] + groups[1:]
        for P in groups:
            out.append(_features(P, len(P), color))
    return out


def _assign_shapes(objs):
    """Within each colour, use the 3-shape constraint (one T, one cube, one sphere)."""
    for color in COLORS:
        g = [o for o in objs if o["color"] == color]
        if len(g) == 3:
            gT = min(g, key=lambda o: o["zt"])
            rest = [o for o in g if o is not gT]
            gs = min(rest, key=lambda o: o["sres"])
            for o in g:
                o["shape"] = "T" if o is gT else ("sphere" if o is gs else "cube")
                o["constrained"] = True
        else:
            for o in g:
                o["shape"] = _abs_shape(o)
                o["constrained"] = False
        for o in g:
            o["abs_shape"] = _abs_shape(o)
            o["cx"], o["cy"] = _center(o, o["shape"])
    return objs


def _parse_cue(frame):
    """Cue = exactly one small coloured blob. Returns dict or None."""
    counts = _color_counts(frame.wrist_rgb)
    tot = sum(counts.values())
    if tot < 30 or tot > 600:
        return None
    objs = _detect(frame, "wrist", minpix=15, expect=0)
    if len(objs) != 1:
        return None
    o = objs[0]
    shape = _abs_shape(o)
    margin = min(abs(o["zt"] - T_TOP_MAX) / 0.012,
                 abs(o["sres"] - SPHERE_RES_MAX) / 1.7 if shape != "T" else 1.0)
    return {"color": o["color"], "shape": shape, "zt": round(o["zt"], 4),
            "sres": round(o["sres"], 2), "n": o["n"], "conf": round(min(1.0, margin), 2)}


def _scene_ready(rgb):
    c = _color_counts(rgb)
    return min(c.values()) >= 150 and sum(c.values()) >= 1000
