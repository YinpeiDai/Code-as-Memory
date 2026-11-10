"""Perception helpers for swap_T: segment the red / blue T-blocks in a static
camera (head by default), back-project RGB-D to world, and fit a planar T
template (position + yaw) to the top-face points."""

# T-block template in its own frame (metres). Origin = middle of the T along its
# symmetry (stem) axis, on the centreline; +A points from stem end toward the bar.
# Dimensions measured from 50 demos (head-camera top faces in the expert grasp frame).
T_LEN_HALF = 0.048
T_BAR_A0 = 0.022
T_BAR_HALF_W = 0.0365
T_STEM_HALF_W = 0.0125
T_CENTROID_A = 0.0123
TABLE_Z = 0.741


def _color_mask(rgb, color):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    if color == "red":
        return (r > 110) & (r - g > 55) & (r - b > 55)
    return (b > 110) & (b - r > 55) & (b - g > 35)


def _backproject(frame, cam, mask):
    dep = np.asarray(getattr(frame, cam + "_depth"))
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    m = mask & (dep > 0)
    v, u = np.nonzero(m)
    if len(v) == 0:
        return np.zeros((0, 3))
    z = dep[v, u].astype(np.float64) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _template_dist(A, B):
    """Distance (m) from block-frame points to the T region (0 inside)."""
    # stem rectangle
    sc = (-T_LEN_HALF + T_BAR_A0) / 2.0
    sh = (T_BAR_A0 + T_LEN_HALF) / 2.0
    dx1 = np.maximum(np.abs(A - sc) - sh, 0.0)
    dy1 = np.maximum(np.abs(B) - T_STEM_HALF_W, 0.0)
    d1 = np.sqrt(dx1 * dx1 + dy1 * dy1)
    bc = (T_BAR_A0 + T_LEN_HALF) / 2.0
    bh = (T_LEN_HALF - T_BAR_A0) / 2.0
    dx2 = np.maximum(np.abs(A - bc) - bh, 0.0)
    dy2 = np.maximum(np.abs(B) - T_BAR_HALF_W, 0.0)
    d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)
    return np.minimum(d1, d2)


def _template_samples(step=0.003):
    pts = []
    a = -T_LEN_HALF + step / 2
    while a < T_LEN_HALF:
        b = -T_BAR_HALF_W + step / 2
        while b < T_BAR_HALF_W:
            if a >= T_BAR_A0 or abs(b) <= T_STEM_HALF_W:
                pts.append((a, b))
            b += step
        a += step
    return np.array(pts)


def _largest_cluster(xy, cell=0.004):
    """Keep the largest 8-connected cluster of occupied grid cells."""
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    ij = np.floor(xy / cell).astype(np.int64)
    ij0 = ij.min(0)
    ij = ij - ij0
    W = int(ij[:, 0].max()) + 1
    H = int(ij[:, 1].max()) + 1
    if W * H > 400000:
        return np.ones(len(xy), dtype=bool)
    occ = np.zeros((W, H), dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True
    lab = -np.ones((W, H), dtype=np.int64)
    cur = 0
    sizes = []
    for i0, j0 in zip(*np.nonzero(occ)):
        if lab[i0, j0] >= 0:
            continue
        stack = [(int(i0), int(j0))]
        lab[i0, j0] = cur
        n = 0
        while stack:
            i, j = stack.pop()
            n += 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < W and 0 <= b < H and occ[a, b] and lab[a, b] < 0:
                        lab[a, b] = cur
                        stack.append((a, b))
        sizes.append(n)
        cur += 1
    best = int(np.argmax(sizes))
    return lab[ij[:, 0], ij[:, 1]] == best


def _fit_T(xy):
    """Fit the T template to top-face xy points. Returns (x, y, yaw, cost, cover)."""
    n = len(xy)
    if n > 1500:
        xy = xy[np.linspace(0, n - 1, 1500).astype(int)]
    cen = xy.mean(0)
    # occupancy grid of data for coverage lookup
    cell = 0.002
    lo = xy.min(0) - 0.02
    ij = np.floor((xy - lo) / cell).astype(np.int64)
    W = int(ij[:, 0].max()) + 12
    H = int(ij[:, 1].max()) + 12
    occ = np.zeros((W, H), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            occ[np.clip(ij[:, 0] + di, 0, W - 1), np.clip(ij[:, 1] + dj, 0, H - 1)] = True
    S = _template_samples()

    def cost(x, y, th):
        c, s = math.cos(th), math.sin(th)
        q = xy - np.array([x, y])
        A = q[:, 0] * c + q[:, 1] * s
        B = -q[:, 0] * s + q[:, 1] * c
        d = _template_dist(A, B)
        wx = x + S[:, 0] * c - S[:, 1] * s
        wy = y + S[:, 0] * s + S[:, 1] * c
        ii = np.floor((wx - lo[0]) / cell).astype(np.int64)
        jj = np.floor((wy - lo[1]) / cell).astype(np.int64)
        inside = (ii >= 0) & (ii < W) & (jj >= 0) & (jj < H)
        cov = np.zeros(len(S), dtype=bool)
        cov[inside] = occ[ii[inside], jj[inside]]
        unc = 1.0 - cov.mean()
        return float(d.mean() + 0.01 * unc), float(cov.mean())

    best = None
    for k in range(120):
        th = k * 2.0 * math.pi / 120
        x = cen[0] - T_CENTROID_A * math.cos(th)
        y = cen[1] - T_CENTROID_A * math.sin(th)
        cst, _ = cost(x, y, th)
        if best is None or cst < best[0]:
            best = (cst, x, y, th)
    cst, x, y, th = best
    steps = [(0.004, math.radians(2.0)), (0.002, math.radians(1.0)),
             (0.001, math.radians(0.5)), (0.0005, math.radians(0.25))]
    for dp, dth in steps:
        improved = True
        it = 0
        while improved and it < 30:
            improved = False
            it += 1
            for ddx, ddy, ddt in ((dp, 0, 0), (-dp, 0, 0), (0, dp, 0), (0, -dp, 0),
                                  (0, 0, dth), (0, 0, -dth)):
                c2, _ = cost(x + ddx, y + ddy, th + ddt)
                if c2 < cst - 1e-9:
                    cst, x, y, th = c2, x + ddx, y + ddy, th + ddt
                    improved = True
    cst, cov = cost(x, y, th)
    th = math.atan2(math.sin(th), math.cos(th))
    return x, y, th, cst, cov


def _detect_block(frame, color, cam="head"):
    """Pose of one T block: dict(x, y, z_top, yaw, n, cost, cover) or None."""
    rgb = np.asarray(getattr(frame, cam + "_rgb"))
    P = _backproject(frame, cam, _color_mask(rgb, color))
    if len(P) < 30:
        return None
    P = P[(P[:, 2] > TABLE_Z + 0.015) & (P[:, 2] < TABLE_Z + 0.07)
          & (np.abs(P[:, 0]) < 0.28) & (P[:, 1] > -0.32) & (P[:, 1] < 0.15)]
    if len(P) < 30:
        return None
    keep = _largest_cluster(P[:, :2])
    P = P[keep]
    zt = float(np.percentile(P[:, 2], 90))
    if zt > TABLE_Z + 0.055:
        return None
    top = P[P[:, 2] > zt - 0.005]
    if len(top) < 30:
        return None
    x, y, th, cst, cov = _fit_T(top[:, :2])
    return {"x": x, "y": y, "z_top": zt, "yaw": th, "n": int(len(top)),
            "cost": cst, "cover": cov}
