"""Perception helpers for ShellGameShuffleTouch: mug/ball segmentation in the front
RGB-D camera, back-projection to world coordinates, 3-cup tracking and rim fitting."""

MUG_MIN_POINTS = 600      # front-camera mug points needed to declare the cups visible
STILL_EPS = 0.002         # m per frame: cup-centre motion below this counts as still
RIM_Z = 0.103             # m: mug-top points used to fit the rim circle (mug top ~0.106)


def _backproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(float) / 1000.0
    ok = z > 0
    v, u, z = v[ok], u[ok], z[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (pc - t) @ R


def _mug_mask(rgb):
    im = rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 40) & (g * 10 < r * 5) & (b * 10 < r * 6) & (r < 215)


def _ball_mask(rgb):
    im = rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > 200) & (g < 40) & (b < 40)


def _frame_points(frame, mask_fn):
    P = _backproject(frame.front_depth, frame.front_K, frame.front_extrinsic, mask_fn(frame.front_rgb))
    return P


def _mug_points(frame):
    P = _frame_points(frame, _mug_mask)
    if len(P) == 0:
        return P
    return P[(P[:, 2] > 0.01) & (P[:, 2] < 0.3)]


def _ball_point(frame):
    P = _frame_points(frame, _ball_mask)
    if len(P) < 4:
        return None
    return P.mean(0)


def _kmeans(P, C, iters):
    C = np.array(C, float)
    a = np.zeros(len(P), int)
    for _ in range(iters):
        d = ((P[:, None, :2] - C[None, :, :2]) ** 2).sum(-1)
        a = d.argmin(1)
        for k in range(len(C)):
            if (a == k).sum() > 0:
                C[k] = P[a == k].mean(0)
    return C, a


def _init_cups(P):
    q = np.percentile(P[:, 1], [16, 50, 84])
    C0 = np.array([[np.median(P[:, 0]), qq, np.median(P[:, 2])] for qq in q])
    return _kmeans(P, C0, 20)


def _rim_center(P, a, k):
    """Circle fit to the mug-top points of cluster k; returns (cx, cy, radius, n)."""
    Q = P[a == k]
    top = Q[Q[:, 2] > RIM_Z]
    if len(top) >= 30:
        A = np.c_[2 * top[:, 0], 2 * top[:, 1], np.ones(len(top))]
        b = (top[:, :2] ** 2).sum(1)
        s = np.linalg.lstsq(A, b, rcond=None)[0]
        c = s[:2]
        rad2 = s[2] + c @ c
        rad = math.sqrt(rad2) if rad2 > 0 else 0.0
        if 0.025 < rad < 0.05 and np.linalg.norm(c - Q[:, :2].mean(0)) < 0.04:
            return float(c[0]), float(c[1]), float(rad), int(len(top))
    m = Q[:, :2].mean(0) if len(Q) else np.zeros(2)
    # visible-surface centroid is biased toward the camera (+x); measured offset in demos
    return float(m[0] - 0.017), float(m[1] + 0.006), -1.0, int(len(top))
