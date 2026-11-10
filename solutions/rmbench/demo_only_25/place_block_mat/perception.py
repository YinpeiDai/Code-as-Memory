"""Perception helpers for place_block_mat (head RGB-D, world-frame geometry).

All helpers are private (start with _). They use only numpy and the frame's
public sensor arrays: head_rgb, head_depth, head_K, head_extrinsic.
"""

TABLE_Z = 0.741          # observed table height (white table back-projection, all demos)
BLOCK_SIZE = 0.042       # observed block edge (top at ~0.783 m)
GREEN_DEFAULT = [0.0, -0.25]


def _backproject(depth, K, E, mask):
    """World points (N,3) of mask pixels with valid depth."""
    v, u = np.nonzero(mask)
    if len(v) == 0:
        return np.zeros((0, 3))
    d = depth[v, u].astype(float) / 1000.0
    ok = d > 0
    u = u[ok].astype(float)
    v = v[ok].astype(float)
    d = d[ok]
    xc = (u - K[0, 2]) * d / K[0, 0]
    yc = (v - K[1, 2]) * d / K[1, 1]
    pc = np.stack([xc, yc, d], 1)
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return (pc - t) @ R


def _red_mask(rgb):
    c = rgb.astype(int)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return (r > 120) & (r - g > 70) & (r - b > 70)


def _green_mask(rgb):
    c = rgb.astype(int)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return (g > 110) & (r < 110) & (g - r > 50) & (b > 90) & (b < 200) & (g > b + 10)


def _cluster_xy(P, radius=0.03, min_pts=12):
    """Greedy xy clustering; returns a list of point arrays, largest first."""
    out = []
    rest = P
    for _ in range(12):
        if len(rest) < min_pts:
            break
        xy = rest[:, :2]
        # seed: densest point approximated by the median of remaining points' nearest mode
        seed = xy[0]
        cnt_best = -1
        step = max(1, len(xy) // 60)
        for i in range(0, len(xy), step):
            c = int(np.sum(np.sum((xy - xy[i]) ** 2, 1) < radius * radius))
            if c > cnt_best:
                cnt_best = c
                seed = xy[i]
        centre = seed
        for _it in range(4):
            m = np.sum((xy - centre) ** 2, 1) < (radius * 1.2) ** 2
            if not np.any(m):
                break
            centre = np.median(xy[m], 0)
        m = np.sum((xy - centre) ** 2, 1) < (radius * 1.25) ** 2
        if int(m.sum()) < min_pts:
            break
        out.append(rest[m])
        rest = rest[~m]
    return out


def _box_fit(xy):
    """Minimum-perimeter rotated box: (cx, cy, yaw_deg in [-45,45), w, h)."""
    best = None
    for k in range(90):
        a = math.radians(k - 45.0)
        ca, sa = math.cos(a), math.sin(a)
        u = xy[:, 0] * ca + xy[:, 1] * sa
        v = -xy[:, 0] * sa + xy[:, 1] * ca
        u0, u1 = np.percentile(u, 3), np.percentile(u, 97)
        v0, v1 = np.percentile(v, 3), np.percentile(v, 97)
        score = (u1 - u0) + (v1 - v0)
        if best is None or score < best[0]:
            cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            best = (score, cu * ca - cv * sa, cu * sa + cv * ca, k - 45.0, u1 - u0, v1 - v0)
    return best[1], best[2], best[3], best[4], best[5]


def _block_from_points(pts):
    """Estimate block centre/yaw from its world points (top face preferred)."""
    zmax = float(np.percentile(pts[:, 2], 95))
    top = pts[pts[:, 2] > zmax - 0.010]
    if len(top) < 6:
        top = pts
    cx, cy, yaw, w, h = _box_fit(top[:, :2])
    # Top-face box is small in the head view; also use all points' x/y spread
    # as a fallback for the centre when the top face is tiny.
    return {"x": float(cx), "y": float(cy), "z_top": zmax, "yaw": float(yaw),
            "n": int(len(pts)), "n_top": int(len(top)), "w": float(w), "h": float(h)}


def _detect_blocks(frame):
    """Red blocks visible in the head camera: list of dicts sorted by x."""
    rgb = frame.head_rgb
    P = _backproject(frame.head_depth, frame.head_K, frame.head_extrinsic, _red_mask(rgb))
    if len(P) == 0:
        return []
    keep = (P[:, 2] > TABLE_Z - 0.01) & (P[:, 2] < TABLE_Z + 0.08) & (P[:, 1] > -0.45) & (P[:, 1] < 0.1)
    P = P[keep]
    blocks = []
    for c in _cluster_xy(P):
        b = _block_from_points(c)
        if b["z_top"] < TABLE_Z + 0.02:
            continue
        blocks.append(b)
    blocks.sort(key=lambda b: b["x"])
    return blocks


def _detect_green(frame):
    """Green mat centre [x, y] from the head camera (fallback to default)."""
    P = _backproject(frame.head_depth, frame.head_K, frame.head_extrinsic, _green_mask(frame.head_rgb))
    if len(P) < 30:
        return list(GREEN_DEFAULT)
    P = P[np.abs(P[:, 2] - TABLE_Z) < 0.02]
    if len(P) < 30:
        return list(GREEN_DEFAULT)
    x0, x1 = np.percentile(P[:, 0], 2), np.percentile(P[:, 0], 98)
    y0, y1 = np.percentile(P[:, 1], 2), np.percentile(P[:, 1], 98)
    return [float(0.5 * (x0 + x1)), float(0.5 * (y0 + y1))]
