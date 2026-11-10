"""Perception helpers: colour segmentation, 3-D back-projection and object classification."""

_COLORS = ("red", "green", "blue")
_CELL = 0.005          # metres, XY grid cell for clustering
_SPLIT_EXTENT = 0.058  # a cluster wider than this holds more than one object
_T_ZMAX = 0.0345       # T-shaped objects are ~0.03 m tall, cubes/spheres ~0.04 m
_CUBE_Z50 = 0.0385     # cubes have a flat top: median surface height ~0.040-0.042


def _color_masks(im):
    im = np.asarray(im).astype(np.float32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "red": (r > 50) & (g < 0.35 * r) & (b < 0.35 * r),
        "green": (g > 50) & (g > 1.3 * r) & (g > 1.3 * b),
        "blue": (b > 50) & (b > 1.3 * r) & (b > 1.2 * g),
    }


def _backproject(vs, us, depth, K, E):
    z = depth[vs, us].astype(np.float64) / 1000.0
    ok = z > 0
    vs, us, z = vs[ok], us[ok], z[ok]
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _occluder_map(depth, own, r=3, gap=25):
    """True where some pixel within r, outside the `own` mask, is closer to the camera by more than gap mm."""
    D0 = depth.astype(np.int32)
    D0 = np.where(D0 <= 0, 100000, D0)
    D = np.where(own, 100000, D0)
    H, W = D.shape
    P = np.pad(D, r, mode="edge")
    mn = D.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            mn = np.minimum(mn, P[r + dy:r + dy + H, r + dx:r + dx + W])
    return mn < D0 - gap


def _camera_points(frame, cam):
    """World points of saturated red/green/blue pixels: dict colour -> (N,5) x,y,z,intensity,occluded."""
    im = getattr(frame, cam + "_rgb")
    depth = np.asarray(getattr(frame, cam + "_depth"))
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    out = {}
    for col, m in _color_masks(im).items():
        occ = _occluder_map(depth, m)
        vs, us = np.nonzero(m)
        if len(vs) == 0:
            out[col] = np.zeros((0, 5))
            continue
        z = depth[vs, us]
        ok = z > 0
        vs, us = vs[ok], us[ok]
        P = _backproject(vs, us, depth, K, E)
        inten = np.asarray(im)[vs, us].max(1).astype(np.float64)
        P = np.concatenate([P, inten[:, None], occ[vs, us].astype(np.float64)[:, None]], 1)
        keep = (P[:, 2] > -0.01) & (P[:, 2] < 0.07) & (np.abs(P[:, 0]) < 0.6) & (np.abs(P[:, 1]) < 0.6)
        out[col] = P[keep]
    return out


def _grid_components(P):
    """Connected components of occupied XY cells; returns list of index arrays."""
    if len(P) == 0:
        return []
    cells = np.floor(P[:, :2] / _CELL).astype(np.int64)
    keys = {}
    for i, (a, b) in enumerate(cells.tolist()):
        keys.setdefault((a, b), []).append(i)
    seen = set()
    comps = []
    for k0 in keys:
        if k0 in seen:
            continue
        stack = [k0]
        seen.add(k0)
        idx = []
        while stack:
            k = stack.pop()
            idx.extend(keys[k])
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nk = (k[0] + da, k[1] + db)
                    if nk in keys and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        comps.append(np.array(idx))
    return comps


def _kmeans2(X):
    """Split XY points into two groups, initialised at the two extreme points."""
    c = X.mean(0)
    d = ((X - c) ** 2).sum(1)
    a = X[int(np.argmax(d))]
    b = X[int(np.argmax(((X - a) ** 2).sum(1)))]
    lab = np.zeros(len(X), dtype=bool)
    for _ in range(20):
        lab = ((X - b) ** 2).sum(1) < ((X - a) ** 2).sum(1)
        if lab.all() or (~lab).all():
            break
        a2, b2 = X[~lab].mean(0), X[lab].mean(0)
        if np.allclose(a, a2) and np.allclose(b, b2):
            break
        a, b = a2, b2
    return np.nonzero(~lab)[0], np.nonzero(lab)[0]


def _split(P, idx, depth=0):
    X = P[idx, :2]
    ext = X.max(0) - X.min(0)
    if depth < 3 and len(idx) >= 20 and ext.max() > _SPLIT_EXTENT:
        i1, i2 = _kmeans2(X)
        if len(i1) > 0 and len(i2) > 0:
            return _split(P, idx[i1], depth + 1) + _split(P, idx[i2], depth + 1)
    return [idx]


def _features(Q):
    """Per-camera shape features of one object's points (columns x,y,z,intensity)."""
    n = len(Q)
    if n < 8:
        return {"n": int(n)}
    z = Q[:, 2]
    zmax = float(np.percentile(z, 95))
    z50 = float(np.median(z))
    top = float((z > zmax - 0.003).mean())
    it = Q[:, 3]
    hist = np.bincount(np.clip(it, 0, 255).astype(np.int64), minlength=256)
    sm = np.convolve(hist, np.ones(5), "same")
    uni = float(sm.max() / n)
    occ = float(Q[:, 4].mean())
    return {"n": int(n), "zmax": zmax, "z50": z50, "top": top, "uni": uni, "occ": occ}


_SHAPES = ("T", "cube", "sphere")


def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _view_loglik(f):
    """Log-probabilities over (T, cube, sphere) from one camera view, and the view weight."""
    if f.get("n", 0) < 12:
        return [0.0, 0.0, 0.0], 0.0
    a = (f["zmax"] - _T_ZMAX) / 0.001            # >0: tall (cube/sphere), <0: T
    s = (f["uni"] - 0.30) / 0.05 + (f["top"] - 0.41) / 0.03   # >0: cube, <0: sphere
    pn = _sig(a)
    pc = _sig(s)
    p = [1.0 - pn, pn * pc, pn * (1.0 - pc)]
    lp = [math.log(max(1e-3, v)) for v in p]
    w = 1.0
    if f["occ"] > 0.3:
        w = 0.0
    elif f["occ"] > 0.1:
        w = 0.3
    if f["n"] < 25:
        w *= 0.5
    return lp, w


def _object_loglik(o):
    tot = [0.0, 0.0, 0.0]
    wsum = 0.0
    for f in o["feats"]:
        lp, w = _view_loglik(f)
        wsum += w
        for i in range(3):
            tot[i] += w * lp[i]
    return tot, wsum


def _perms(items, k):
    """All ordered selections of k distinct items."""
    if k == 0:
        return [[]]
    out = []
    for i, s in enumerate(items):
        rest = items[:i] + items[i + 1:]
        for p in _perms(rest, k - 1):
            out.append([s] + p)
    return out


def _ranked_assignments(group):
    """Distinct-shape assignments of one colour group, best first: list of (score, shapes)."""
    if len(group) > 3:
        return []
    out = []
    for p in _perms(list(_SHAPES), len(group)):
        sc = sum(o["ll"][_SHAPES.index(s)] for o, s in zip(group, p))
        out.append((sc, p))
    out.sort(key=lambda t: -t[0])
    return out


def _choose_target(cue, test):
    """Pick the test object whose (colour, shape) was absent from the cue.

    Structured rule (observed in all 50 demos): the cue shows 8 objects, the test 9;
    in the test every colour has one T, one cube and one sphere; in the cue one colour
    has only two objects. The missing combination is that colour plus the shape absent
    from its two cue objects. Fallback: independent classification and set difference."""
    info = {}
    cols = {c: [o for o in cue if o["color"] == c] for c in _COLORS}
    tcols = {c: [o for o in test if o["color"] == c] for c in _COLORS}
    counts = sorted(len(v) for v in cols.values())
    info["cue_counts"] = {c: len(v) for c, v in cols.items()}
    info["test_counts"] = {c: len(v) for c, v in tcols.items()}
    if counts == [2, 3, 3]:
        cstar = [c for c in _COLORS if len(cols[c]) == 2][0]
        ra = _ranked_assignments(cols[cstar])
        sc_missing = {}
        for sc, p in ra:
            m = [s for s in _SHAPES if s not in p][0]
            sc_missing[m] = max(sc_missing.get(m, -1e18), sc)
        order = sorted(sc_missing, key=lambda m: -sc_missing[m])
        sstar = order[0]
        info["cue_margin"] = round(sc_missing[order[0]] - sc_missing[order[1]], 2) if len(order) > 1 else 99.0
        grp = tcols[cstar]
        if len(grp) >= 1:
            if len(grp) <= 3:
                tr = _ranked_assignments(grp)
                p = tr[0][1]
                info["test_margin"] = round(tr[0][0] - tr[1][0], 2) if len(tr) > 1 else 99.0
                if sstar in p:
                    tgt = grp[p.index(sstar)]
                else:
                    tgt = max(grp, key=lambda o: o["ll"][_SHAPES.index(sstar)])
            else:
                tgt = max(grp, key=lambda o: o["ll"][_SHAPES.index(sstar)])
            info["rule"] = "structured"
            info["color"] = cstar
            info["shape"] = sstar
            return tgt, info
    # fallback: independent labels
    seen = set((o["color"], o["shape"]) for o in cue)
    cands = [o for o in test if (o["color"], o["shape"]) not in seen]
    info["rule"] = "fallback"
    if cands:
        tgt = max(cands, key=lambda o: o["w"])
    elif test:
        # least cue-like: lowest likelihood of any seen combination
        def cue_like(o):
            return max([o["ll"][_SHAPES.index(s)] for (c, s) in seen if c == o["color"] and s in _SHAPES] or [-99.0])
        tgt = min(test, key=cue_like)
    else:
        return None, info
    info["color"] = tgt["color"]
    info["shape"] = tgt["shape"]
    return tgt, info


def _detect(frames_cams):
    """frames_cams: list of (frame, cam). Fuse points; return list of object dicts."""
    pts = {c: [] for c in _COLORS}
    for ci, (frame, cam) in enumerate(frames_cams):
        cp = _camera_points(frame, cam)
        for c in _COLORS:
            if len(cp[c]):
                Q = cp[c]
                pts[c].append(np.concatenate([Q, np.full((len(Q), 1), float(ci))], 1))
    objs = []
    for c in _COLORS:
        if not pts[c]:
            continue
        P = np.concatenate(pts[c], 0)
        for comp in _grid_components(P):
            if len(comp) < 15:
                continue
            for idx in _split(P, comp):
                if len(idx) < 15:
                    continue
                Q = P[idx]
                xy = np.median(Q[:, :2], 0)
                feats = []
                for ci in range(len(frames_cams)):
                    feats.append(_features(Q[Q[:, 5] == ci]))
                o = {"color": c, "x": float(xy[0]), "y": float(xy[1]),
                     "n": int(len(idx)), "feats": feats}
                ll, wsum = _object_loglik(o)
                o["ll"] = [round(v, 3) for v in ll]
                o["w"] = round(wsum, 2)
                o["shape"] = _SHAPES[int(np.argmax(ll))] if wsum > 0 else "unknown"
                zs = [f["zmax"] for f in feats if f.get("n", 0) >= 12 and "zmax" in f]
                o["top_z"] = float(min(zs)) if zs else 0.04
                hi = Q[Q[:, 2] > 0.008]
                if len(hi) < 8:
                    hi = Q
                cells = np.unique(np.round(hi[:, :2] / 0.003).astype(np.int64), axis=0) * 0.003
                o["cells"] = cells
                o["bbox"] = [float(np.percentile(hi[:, 0], 3)), float(np.percentile(hi[:, 0], 97)),
                             float(np.percentile(hi[:, 1], 3)), float(np.percentile(hi[:, 1], 97))]
                objs.append(o)
    return objs
