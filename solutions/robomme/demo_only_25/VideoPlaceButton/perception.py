"""Perception helpers for VideoPlaceButton (no imports; np and math are supplied)."""

TABLE_Z = 0.0
TARGET_TOP_Z = 0.002
CUBE_CENTER_Z = 0.02
CLUSTER_R = 0.055


def _cam(frame):
    """Return (K, E) of the front camera, or (None, None)."""
    K = getattr(frame, 'front_K', None)
    E = getattr(frame, 'front_extrinsic', None)
    if K is None or E is None:
        return None, None
    return np.asarray(K, float), np.asarray(E, float)


def _pix_to_world(frame, us, vs, plane_z, K=None, E=None):
    """Back-project pixels to world using depth when valid, else intersect with z=plane_z."""
    if K is None or E is None:
        K, E = _cam(frame)
    if K is None or E is None:
        raise ValueError('no front-camera calibration')
    R = E[:, :3]
    t = E[:, 3]
    us = np.asarray(us, float)
    vs = np.asarray(vs, float)
    rays = np.stack([(us + 0.5 - K[0, 2]) / K[0, 0], (vs + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(us)], 0)
    cc = -R.T @ t
    dirs = R.T @ rays
    s = (plane_z - cc[2]) / dirs[2]
    pts = (cc[:, None] + dirs * s[None, :]).T
    depth = getattr(frame, 'front_depth', None)
    if depth is not None:
        d = np.asarray(depth)[vs.astype(int), us.astype(int)].astype(float) / 1000.0
        ok = d > 0
        if ok.any():
            cam = rays[:, ok] * d[ok][None, :]
            w = (R.T @ (cam - t[:, None])).T
            pts[ok] = w
    return pts


def _proj(K, E, p):
    c = E[:, :3] @ np.asarray(p, float) + E[:, 3]
    u = K @ c
    return u[:2] / u[2]


def _purple_mask(rgb):
    im = np.asarray(rgb).astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (b > 150) & (r > 120) & (g < 140) & (b - g > 70) & (r - g > 40)


def _cube_mask(rgb, color):
    im = np.asarray(rgb).astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if color == 'red':
        return (r > 120) & (g < 60) & (b < 60)
    if color == 'green':
        return (g > 120) & (r < 80) & (b < 80)
    return (b > 120) & (r < 60) & (g < 60)


def _cluster(pts, radius=CLUSTER_R, min_n=6):
    """Greedy mean-shift-like clustering of Nx2 points; returns list of (center, count)."""
    pts = np.asarray(pts, float)
    left = np.ones(len(pts), bool)
    out = []
    while left.any():
        i = int(np.flatnonzero(left)[0])
        c = pts[i]
        for _ in range(6):
            m = left & (np.sum((pts - c) ** 2, 1) < radius ** 2)
            if not m.any():
                break
            c = pts[m].mean(0)
        m = left & (np.sum((pts - c) ** 2, 1) < radius ** 2)
        m[i] = True
        left &= ~m
        if m.sum() >= min_n:
            out.append((c, int(m.sum())))
    return out


def _target_points(frame, step=1, calib=None):
    """World XY points of purple target pixels (table area only). calib=(K, E) overrides the frame's."""
    mask = _purple_mask(frame.front_rgb)
    vs, us = np.nonzero(mask)
    if step > 1:
        vs, us = vs[::step], us[::step]
    if len(us) == 0:
        return np.zeros((0, 2))
    K, E = calib if calib is not None else (None, None)
    pts = _pix_to_world(frame, us, vs, TARGET_TOP_Z, K, E)
    keep = (np.abs(pts[:, 2] - TARGET_TOP_Z) < 0.02) & (np.abs(pts[:, 0]) < 0.35) & (np.abs(pts[:, 1]) < 0.35)
    return pts[keep, :2]


def _detect_targets(frame, calib=None):
    """List of target centres [x, y] in world metres."""
    return [c.tolist() for c, n in _cluster(_target_points(frame, 1, calib), CLUSTER_R, 15)]


def _detect_cube(frame, color, calib=None):
    """World [x, y, z, n_pixels] of the largest blob of the cube colour (top-face centroid), or None."""
    mask = _cube_mask(frame.front_rgb, color)
    vs, us = np.nonzero(mask)
    if len(us) < 4:
        return None
    K, E = calib if calib is not None else (None, None)
    pts = _pix_to_world(frame, us, vs, CUBE_CENTER_Z, K, E)
    cl = _cluster(pts[:, :2], 0.03, 4)
    if not cl:
        return None
    c, n = max(cl, key=lambda a: a[1])
    m = np.sum((pts[:, :2] - c) ** 2, 1) < 0.03 ** 2
    p = pts[m]
    top = p[p[:, 2] > p[:, 2].max() - 0.003]
    if len(top) >= 3:
        c = top[:, :2].mean(0)
        z = float(np.median(top[:, 2]))
    else:
        z = float(np.median(p[:, 2]))
    return [float(c[0]), float(c[1]), z, int(n)]


def _square_yaw(xy):
    """Yaw (mod pi/2, in [-pi/4, pi/4)) of a square from its top-face points, by min bounding-box area."""
    xy = np.asarray(xy, float)
    if len(xy) < 6:
        return 0.0
    xy = xy - xy.mean(0)
    best, ba = 0.0, 1e9
    for k in range(90):
        a = -math.pi / 4 + k * math.pi / 180
        c, s = math.cos(a), math.sin(a)
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        area = (u.max() - u.min()) * (v.max() - v.min())
        if area < ba:
            ba, best = area, a
    return best


def _cube_top_points(frame, color):
    """World points on the top face of the cube colour blob (largest blob), or None."""
    mask = _cube_mask(frame.front_rgb, color)
    vs, us = np.nonzero(mask)
    if len(us) < 4:
        return None
    pts = _pix_to_world(frame, us, vs, CUBE_CENTER_Z)
    cl = _cluster(pts[:, :2], 0.03, 4)
    if not cl:
        return None
    c, n = max(cl, key=lambda a: a[1])
    p = pts[np.sum((pts[:, :2] - c) ** 2, 1) < 0.03 ** 2]
    return p[p[:, 2] > p[:, 2].max() - 0.003]


RING_IN = 0.025
RING_OUT = 0.043


def _ring_fit(pts, pred, search, pull=150.0):
    """Best ring centre near pred: maximise annulus point count minus a pull to pred.

    Coarse-to-fine grid search. Returns (centre, count)."""
    pred = np.asarray(pred, float)
    if len(pts) == 0:
        return pred, 0
    near = pts[np.sum((pts - pred) ** 2, 1) < (search + RING_OUT + 0.005) ** 2]
    if len(near) < 3:
        return pred, 0
    best_c, best_s, best_n = pred, -1e9, 0
    center = pred
    for spacing, half in ((max(search / 4.0, 0.002), 4), (0.002, 2), (0.001, 1)):
        offs = np.arange(-half, half + 1) * spacing
        gx, gy = np.meshgrid(offs, offs)
        cand = np.stack([gx.ravel() + center[0], gy.ravel() + center[1]], 1)
        d = np.sqrt(((near[None, :, :] - cand[:, None, :]) ** 2).sum(2))
        cnt = ((d > RING_IN) & (d < RING_OUT)).sum(1)
        dist = np.sqrt(((cand - pred) ** 2).sum(1))
        score = cnt - pull * dist * (cnt.max() / 100.0 + 0.1)
        score = np.where(dist <= search + 1e-9, score, -1e9)
        k = int(np.argmax(score))
        if score[k] > best_s:
            best_s, best_c, best_n = score[k], cand[k], int(cnt[k])
        center = best_c
    return np.asarray(best_c), best_n


def _track_step(tracks, vels, pts, gate=0.024, min_n=8, ref_n=None, max_move=None):
    """Advance target tracks one frame using purple world points (Nx2).

    Each track fits a ring (annulus of the purple target) near its constant-velocity
    prediction; the most-supported track is fixed first and its ring points are removed
    before the others are fitted. Tracks with too little support coast on a decaying
    velocity. vels entries are [vx, vy, lost]. Returns (tracks, vels, counts)."""
    T = np.asarray(tracks, float)
    n = len(T)
    VL = np.asarray(vels, float).reshape(n, -1)
    V = VL[:, :2]
    lost = VL[:, 2] if VL.shape[1] > 2 else np.zeros(n)
    pred = T + V
    pts = np.asarray(pts, float).reshape(-1, 2)
    remaining = list(range(n))
    res = {}
    P = pts
    while remaining:
        fits = []
        for i in remaining:
            search = gate + min(0.03, 0.004 * lost[i])
            c, cnt = _ring_fit(P, pred[i], search)
            need = min_n if ref_n is None else max(min_n, 0.3 * ref_n[i])
            fits.append((cnt / max(1.0, need), i, c, cnt, need))
        fits.sort(key=lambda a: -a[0])
        _, i, c, cnt, need = fits[0]
        res[i] = (c, cnt, cnt >= need)
        remaining.remove(i)
        if cnt >= need and len(P):
            d = np.sqrt(((P - c) ** 2).sum(1))
            P = P[(d < RING_IN - 0.004) | (d > RING_OUT + 0.002)]
    newT, newV, counts = [], [], []
    for i in range(n):
        c, cnt, ok = res[i]
        counts.append(int(cnt))
        if ok:
            step = np.asarray(c) - T[i]
            v = 0.6 * V[i] + 0.4 * step
            newT.append([float(c[0]), float(c[1])])
            newV.append([float(v[0]), float(v[1]), 0.0])
        else:
            v = 0.8 * V[i]
            newT.append([float(pred[i][0]), float(pred[i][1])])
            newV.append([float(v[0]), float(v[1]), float(lost[i] + 1)])
    return newT, newV, counts


def _site_support(pts, sites):
    """Annulus point count of each fixed target site (ring centre) in this frame."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    S = np.asarray(sites, float)
    if len(pts) == 0:
        return np.zeros(len(S)), pts
    d = np.sqrt(((pts[:, None, :] - S[None, :, :]) ** 2).sum(2))
    sup = ((d > RING_IN) & (d < RING_OUT)).sum(0)
    off = pts[d.min(1) > RING_OUT + 0.006]
    return sup, off
