"""Perception helpers for RememberShape9: blue-object segmentation, world-frame
point clouds from both RGB-D cameras, top-view maps and shape features."""

CELL_G = 0.002      # top-view grid cell (m)
MAP_N = 48          # top-view grid size (cells)
TABLE_Z = 0.0015    # points above this height belong to objects


def _blue_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (b > 50) & (b > r + 30) & (b > g + 30)


def _backproject(depth, K, E, mask):
    vs, us = np.nonzero(mask)
    z = depth[vs, us].astype(np.float64) / 1000.0
    ok = z > 0
    vs = vs[ok]
    us = us[ok]
    z = z[ok]
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _frame_points(frame, cams=("wrist", "front")):
    """World points of blue pixels above the table, with camera id (0 wrist, 1 front)."""
    P = []
    S = []
    for ci, cam in enumerate(cams):
        rgb = getattr(frame, cam + "_rgb")
        d = getattr(frame, cam + "_depth")
        K = getattr(frame, cam + "_K")
        E = getattr(frame, cam + "_extrinsic")
        w = _backproject(d, K, E, _blue_mask(rgb))
        keep = (w[:, 2] > TABLE_Z) & (w[:, 2] < 0.09) & (np.abs(w[:, 0]) < 0.6) & (np.abs(w[:, 1]) < 0.6)
        P.append(w[keep])
        S.append(np.full(int(keep.sum()), ci))
    return np.concatenate(P, 0), np.concatenate(S, 0)


def _clusters(P, cell=0.005, min_pts=8):
    """Connected components of points on an xy grid (8-connected)."""
    if len(P) == 0:
        return []
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    keys = {}
    for n in range(len(ij)):
        k = (int(ij[n, 0]), int(ij[n, 1]))
        if k in keys:
            keys[k].append(n)
        else:
            keys[k] = [n]
    seen = set()
    out = []
    for k in keys:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        members = []
        while stack:
            c = stack.pop()
            members.extend(keys[c])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(members) >= min_pts:
            out.append(np.array(members))
    return out


def _split_big(P, idx, rmax_limit=0.042, depth=0):
    """Split a cluster whose xy radius is too large for one object (2-means)."""
    Q = P[idx, :2]
    c = Q.mean(0)
    r = np.sqrt(((Q - c) ** 2).sum(1)).max()
    if r <= rmax_limit or depth > 2 or len(idx) < 20:
        return [idx]
    a = Q[np.argmax(((Q - c) ** 2).sum(1))]
    b = Q[np.argmax(((Q - a) ** 2).sum(1))]
    lab = np.zeros(len(Q), dtype=bool)
    for _ in range(15):
        lab = ((Q - a) ** 2).sum(1) > ((Q - b) ** 2).sum(1)
        if lab.all() or (~lab).all():
            return [idx]
        a = Q[~lab].mean(0)
        b = Q[lab].mean(0)
    return (_split_big(P, idx[~lab], rmax_limit, depth + 1)
            + _split_big(P, idx[lab], rmax_limit, depth + 1))


def _dilate(A):
    B = A.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            B = np.maximum(B, np.roll(np.roll(A, di, 0), dj, 1))
    return B


def _maps(Pw, centre):
    """Top-view occupancy (closed) and max-height maps around `centre`."""
    n = MAP_N
    g = CELL_G
    ij = np.floor((Pw[:, :2] - centre) / g).astype(np.int64) + n // 2
    ok = (ij[:, 0] >= 1) & (ij[:, 0] < n - 1) & (ij[:, 1] >= 1) & (ij[:, 1] < n - 1)
    H = np.zeros((n, n))
    np.maximum.at(H, (ij[ok, 0], ij[ok, 1]), Pw[ok, 2])
    occ = H > 0
    dil = _dilate(occ.astype(float)) > 0
    ero = ~(_dilate((~dil).astype(float)) > 0)
    Hd = _dilate(H)
    Hc = np.where(ero, np.where(H > 0, H, Hd), 0.0)
    return ero, Hc


def _hull_area(pts):
    P = sorted(set((float(a), float(b)) for a, b in pts))
    if len(P) < 3:
        return 0.0

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo = []
    up = []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    h = lo[:-1] + up[:-1]
    a = 0.0
    for i in range(len(h)):
        x1, y1 = h[i]
        x2, y2 = h[(i + 1) % len(h)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _holes(occ):
    n = occ.shape[0]
    empty = ~occ
    reach = np.zeros_like(empty)
    stack = []
    for i in range(n):
        for j in (0, n - 1):
            stack.append((i, j))
            stack.append((j, i))
    for p in stack:
        reach[p] = True
    while stack:
        i, j = stack.pop()
        for a, c in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < n and 0 <= c < n and empty[a, c] and not reach[a, c]:
                reach[a, c] = True
                stack.append((a, c))
    return int((empty & ~reach).sum())


def _object_features(Pw):
    """Shape features of one object's world points (metres); None if too few."""
    if len(Pw) < 8:
        return None
    z = Pw[:, 2]
    hmax = float(np.percentile(z, 95))
    top = Pw[z > hmax - max(0.006, 0.3 * hmax)]
    centre = top[:, :2].mean(0)
    occ, H = _maps(Pw, Pw[:, :2].mean(0))
    cells = np.argwhere(occ)
    if len(cells) < 5:
        return None
    g = CELL_G
    hz = H[occ]
    cen = cells.mean(0)
    d = cells - cen
    r = np.sqrt((d ** 2).sum(1)) + 0.5
    th = np.arctan2(d[:, 1], d[:, 0])
    nb = 36
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(np.int64) % nb
    sig = np.zeros(nb)
    np.maximum.at(sig, b, r)
    for k in range(nb):
        if sig[k] == 0:
            sig[k] = max(sig[(k - 1) % nb], sig[(k + 1) % nb])
    F = np.abs(np.fft.rfft(sig)) / (nb * max(sig.mean(), 1e-9))
    pts = np.concatenate([cells, cells + [1, 0], cells + [0, 1], cells + [1, 1]])
    ha = _hull_area(pts)
    # principal extents (cm)
    cov = np.cov(d.T) if len(d) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    proj = d @ evecs
    ext = (proj.max(0) - proj.min(0) + 1.0) * g * 100
    return {
        "hmax": hmax,
        "area": len(cells) * g * g * 1e4,
        "solid": len(cells) / max(ha, 1e-9),
        "rmax": float(sig.max() * g * 100),
        "circ": float(len(cells) / (np.pi * sig.max() ** 2)),
        "f1": float(F[1]), "f2": float(F[2]), "f3": float(F[3]),
        "f4": float(F[4]), "f5": float(F[5]),
        "top": float((hz > 0.8 * hmax).mean()),
        "hrel": float(hz.mean() / max(hmax, 1e-6)),
        "hole": _holes(occ),
        "ext_major": float(ext[1]),
        "ext_minor": float(ext[0]),
        "x": float(centre[0]),
        "y": float(centre[1]),
        "xm": float(Pw[:, 0].mean()),
        "ym": float(Pw[:, 1].mean()),
        "npts": int(len(Pw)),
    }


def _scene_objects(frames, cams=("wrist", "front")):
    """Detect blue objects in the given frames (fused), returning feature dicts."""
    Ps = []
    for fr in frames:
        P, _ = _frame_points(fr, cams)
        Ps.append(P)
    P = np.concatenate(Ps, 0) if Ps else np.zeros((0, 3))
    objs = []
    for idx in _clusters(P):
        for part in _split_big(P, idx):
            f = _object_features(P[part])
            if f is not None and f["npts"] >= 25:
                objs.append(f)
    return objs
