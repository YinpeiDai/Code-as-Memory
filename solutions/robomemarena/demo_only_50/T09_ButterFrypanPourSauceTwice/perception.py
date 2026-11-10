"""Perception helpers: back-project agentview RGB-D into world points and locate
the butter, tomato-sauce bottle, frypan and bowl drainer on the table.
All offsets/thresholds were calibrated on the 50 assigned demonstrations
(see /workspace/task_analysis.md)."""

PAN_RADIUS = 0.0975          # fitted rim radius (m), std < 1 mm over 50 demos
PRIOR_TABLE_Z = 0.4309
PRIOR_BUTTER = [0.115, -0.205]
PRIOR_SAUCE = [-0.173, -0.204]
PRIOR_PAN = [0.002, 0.258]
PRIOR_DRAINER = [-0.125, 0.53]


def _world_points(frame):
    """World XYZ of every valid agentview depth pixel, shape (N, 3)."""
    depth = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1).reshape(-1, 3)
    ok = depth.reshape(-1) > 0
    pw = (pc[ok] - E[:, 3]) @ E[:, :3]
    return pw


def _table_z(P):
    m = (np.abs(P[:, 0]) < 0.3) & (np.abs(P[:, 1]) < 0.4) & (P[:, 2] > 0.35) & (P[:, 2] < 0.5)
    zs = P[m, 2]
    if len(zs) < 100:
        return PRIOR_TABLE_Z
    hist, edges = np.histogram(zs, bins=np.arange(0.35, 0.5, 0.002))
    i = int(np.argmax(hist))
    sel = zs[(zs >= edges[i] - 0.002) & (zs < edges[i + 1] + 0.002)]
    z0 = float(np.median(sel))
    if abs(z0 - PRIOR_TABLE_Z) > 0.03:
        return PRIOR_TABLE_Z
    return z0


def _box(P, h, x0, x1, y0, y1, h0, h1):
    m = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (h > h0) & (h < h1)
    return P[m]


def _robust_center(Q, radius):
    """Median XY, refined twice on points within `radius` of the current estimate (rejects stray points)."""
    c = np.median(Q[:, :2], axis=0)
    for _ in range(3):
        sel = Q[np.linalg.norm(Q[:, :2] - c, axis=1) < radius]
        if len(sel) < 3:
            break
        c = np.median(sel[:, :2], axis=0)
    sel = Q[np.linalg.norm(Q[:, :2] - c, axis=1) < radius]
    if len(sel) < 3:
        sel = Q
    return [float(c[0]), float(c[1]), float(np.percentile(sel[:, 2], 98))], int(len(sel))


def _find_pan(P, h):
    Q = _box(P, h, -0.25, 0.25, 0.05, 0.55, 0.020, 0.040)[:, :2]
    if len(Q) < 20:
        return None, 0
    xs = np.arange(-0.15, 0.15, 0.004)
    ys = np.arange(0.12, 0.45, 0.004)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    best = None
    score = None
    for start in range(0, len(G), 2000):
        g = G[start:start + 2000]
        d = np.sqrt(((g[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
        sc = (np.abs(d - PAN_RADIUS) < 0.007).sum(1)
        k = int(np.argmax(sc))
        if score is None or sc[k] > score:
            score = int(sc[k])
            best = g[k].copy()
    c = best
    inl = Q[np.abs(np.linalg.norm(Q - c, axis=1) - PAN_RADIUS) < 0.01]
    if len(inl) >= 10:
        A = np.concatenate([2 * inl, np.ones((len(inl), 1))], 1)
        b = (inl ** 2).sum(1)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        r2 = float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0)))
        if abs(r2 - PAN_RADIUS) < 0.015 and np.linalg.norm(sol[:2] - c) < 0.02:
            c = sol[:2]
    return [float(c[0]), float(c[1])], score


def _detect_scene(frame):
    """Locate the task objects in one agentview frame (robot assumed clear)."""
    P = _world_points(frame)
    z0 = _table_z(P)
    h = P[:, 2] - z0
    out = {'table_z': z0, 'flags': []}
    S = _box(P, h, -0.40, 0.0, -0.45, -0.06, 0.01, 0.2)
    if len(S) >= 10:
        out['sauce'], _ = _robust_center(S, 0.05)
    else:
        out['sauce'] = [PRIOR_SAUCE[0], PRIOR_SAUCE[1], z0 + 0.074]
        out['flags'].append('sauce_prior')
    B = _box(P, h, 0.0, 0.30, -0.45, -0.02, 0.006, 0.06)
    if len(B) >= 5:
        out['butter'], _ = _robust_center(B, 0.04)
    else:
        out['butter'] = [PRIOR_BUTTER[0], PRIOR_BUTTER[1], z0 + 0.012]
        out['flags'].append('butter_prior')
    pan, score = _find_pan(P, h)
    if pan is None or score < 40:
        pan = list(PRIOR_PAN)
        out['flags'].append('pan_prior')
    out['pan'] = pan
    out['pan_score'] = score
    Dd = _box(P, h, -0.45, 0.15, 0.35, 0.8, 0.05, 0.25)
    if len(Dd) >= 20:
        out['drainer'] = [float(0.5 * (np.percentile(Dd[:, 0], 2) + np.percentile(Dd[:, 0], 98))),
                          float(0.5 * (np.percentile(Dd[:, 1], 2) + np.percentile(Dd[:, 1], 98))),
                          float(np.percentile(Dd[:, 2], 98))]
    else:
        out['drainer'] = [PRIOR_DRAINER[0], PRIOR_DRAINER[1], z0 + 0.163]
        out['flags'].append('drainer_prior')
    return out


def _butter_in_pan(frame, pan, z0):
    """Number of depth points 1.2-6 cm above the table inside the pan rim (empty pan: 0, butter: ~55)."""
    P = _world_points(frame)
    h = P[:, 2] - z0
    r = np.linalg.norm(P[:, :2] - np.asarray(pan[:2]), axis=1)
    return int(((r < 0.075) & (h > 0.012) & (h < 0.06)).sum())


def _find_small_object(frame, z0, avoid, x0, x1, y0, y1, h0, h1, min_pts=5):
    """Median XYZ-top of points in a box, ignoring discs avoid=[[x, y, radius], ...]; None when too few."""
    P = _world_points(frame)
    h = P[:, 2] - z0
    Q = _box(P, h, x0, x1, y0, y1, h0, h1)
    for c in avoid or []:
        if len(Q):
            Q = Q[np.linalg.norm(Q[:, :2] - np.asarray(c[:2]), axis=1) > c[2]]
    if len(Q) < min_pts:
        return None
    c, n = _robust_center(Q, 0.05)
    if n < min_pts:
        return None
    return c
