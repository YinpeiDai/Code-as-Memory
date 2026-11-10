"""Perception helpers: back-project agentview RGB-D to world points and locate
the wine bottle (dark, tall object) and the mug (light object near the bottle)."""

TABLE_Z = 0.431


def _cloud(K, E, depth):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return ((P - t) @ R).reshape(H, W, 3), (depth > 0)


def _components(mask):
    """4-connected components of a boolean image; returns list of (rows, cols)."""
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=int)
    comps = []
    vs, us = np.nonzero(mask)
    for k in range(len(vs)):
        sv, su = int(vs[k]), int(us[k])
        if lab[sv, su] >= 0:
            continue
        cid = len(comps)
        stack = [(sv, su)]
        lab[sv, su] = cid
        rr = []
        cc = []
        while stack:
            a, b = stack.pop()
            rr.append(a)
            cc.append(b)
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                na, nb = a + da, b + db
                if 0 <= na < H and 0 <= nb < W and mask[na, nb] and lab[na, nb] < 0:
                    lab[na, nb] = cid
                    stack.append((na, nb))
        comps.append((np.array(rr), np.array(cc)))
    return comps


def _detect_objects(frame):
    """Return dict with bottle/mug raw world estimates (median of visible surface
    points) and diagnostic counts. Values are None when not found."""
    K = np.asarray(frame.agentview_K)
    E = np.asarray(frame.agentview_extrinsic)
    D = np.asarray(frame.agentview_depth)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    Wp, valid = _cloud(K, E, D)
    X, Y, Z = Wp[..., 0], Wp[..., 1], Wp[..., 2]
    on_table = valid & (X > -0.40) & (X < 0.35) & (np.abs(Y) < 0.75)
    above = on_table & (Z > TABLE_Z + 0.014) & (Z < TABLE_Z + 0.32)
    dark = rgb.sum(-1) < 60
    out = {"bottle": None, "mug": None, "bottle_n": 0, "mug_n": 0}
    # bottle: dark component with the largest vertical extent (tall and thin)
    best = None
    for rr, cc in _components(above & dark):
        if len(rr) < 15:
            continue
        P = Wp[rr, cc]
        ext = P[:, 2].max() - P[:, 2].min()
        span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
        if span > 0.15:
            continue
        score = ext
        if best is None or score > best[0]:
            best = (score, P)
    if best is not None and best[0] > 0.05:
        P = best[1]
        out["bottle"] = [float(np.median(P[:, 0])), float(np.median(P[:, 1])), float(np.median(P[:, 2]))]
        out["bottle_n"] = int(len(P))
        out["bottle_top"] = float(P[:, 2].max())
    # mug: light object points in a window near the bottle side of the table,
    # choose component with highest top among compact ones (excluding bottle area)
    light = above & ~dark & (Z < TABLE_Z + 0.16)
    best = None
    for rr, cc in _components(light):
        if len(rr) < 40:
            continue
        P = Wp[rr, cc]
        span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
        if span > 0.16:
            continue
        if out["bottle"] is not None:
            d = np.hypot(np.median(P[:, 0]) - out["bottle"][0], np.median(P[:, 1]) - out["bottle"][1])
            if d < 0.05:
                continue
        top = P[:, 2].max()
        if top < TABLE_Z + 0.06:
            continue
        score = len(P)
        if best is None or score > best[0]:
            best = (score, P)
    if best is not None:
        P = best[1]
        out["mug"] = [float(np.median(P[:, 0])), float(np.median(P[:, 1])), float(P[:, 2].max())]
        out["mug_n"] = int(len(P))
    return out
