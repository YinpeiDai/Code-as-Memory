"""Perception helpers for RememberShape3: blue-object segmentation, 3D back-projection and shape fitting.

All objects are blue, ~4 cm in size, resting on the table (z in [0, 0.04]).  Shapes seen in all 50 assigned
demonstrations: sphere (r=0.02), cylinder (r=0.02, length 0.04, standing or lying) and cube (side 0.04).
Shape is decided by the RMS of a signed-distance fit of each model to the visible surface points.
"""

HALF = 0.02
SHAPES = ("sphere", "cylinder", "cube")


def _blue_points(rgb, depth, K, E):
    """World-frame 3D points of blue pixels with valid depth (N x 3)."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    m = (b > 100) & (r < 60) & (g < 60) & (depth > 0)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.zeros((0, 3))
    d = depth[ys, xs].astype(np.float64) / 1000.0
    Ki = np.linalg.inv(np.asarray(K, dtype=np.float64))
    uv1 = np.vstack([xs + 0.5, ys + 0.5, np.ones(len(xs))])
    pc = (Ki @ uv1) * d
    E = np.asarray(E, dtype=np.float64)
    R = E[:, :3]
    t = E[:, 3]
    return (R.T @ (pc - t[:, None])).T


def _cluster(P, rad=0.05, min_pts=8):
    """Greedy xy clustering of table-level points; returns list of point arrays."""
    if len(P) == 0:
        return []
    P = P[(P[:, 2] > -0.005) & (P[:, 2] < 0.07) & (np.abs(P[:, 1]) < 0.45) & (P[:, 0] > -0.35) & (P[:, 0] < 0.3)]
    groups = []
    rem = P
    while len(rem) > 0:
        c = rem[0, :2]
        for _ in range(5):
            sel = np.linalg.norm(rem[:, :2] - c, axis=1) < rad
            c = np.median(rem[sel, :2], axis=0)
        sel = np.linalg.norm(rem[:, :2] - c, axis=1) < rad
        if sel.sum() == 0:
            sel = np.zeros(len(rem), dtype=bool)
            sel[0] = True
        if sel.sum() >= min_pts:
            groups.append(rem[sel])
        rem = rem[~sel]
    return groups


def _frame_points(frame, use_wrist=True):
    P = _blue_points(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic)
    if use_wrist:
        W = _blue_points(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
        P = np.vstack([P, W]) if len(W) else P
    return P


def _count_objects(frame):
    """Number of blue table objects visible in the front camera (cheap)."""
    P = _blue_points(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic)
    return len(_cluster(P, min_pts=6))


def _sdf_box(U, V, Z):
    dx = np.abs(U) - HALF
    dy = np.abs(V) - HALF
    dz = np.abs(Z) - HALF
    out = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2 + np.maximum(dz, 0) ** 2)
    return out + np.minimum(np.maximum(np.maximum(dx, dy), dz), 0)


def _sdf_cyl(radial, axial):
    d1 = radial - HALF
    d2 = np.abs(axial) - HALF
    out = np.sqrt(np.maximum(d1, 0) ** 2 + np.maximum(d2, 0) ** 2)
    return out + np.minimum(np.maximum(d1, d2), 0)


def _fit_shapes(P):
    """Fit sphere / cylinder (upright or lying, any yaw) / cube (any yaw), all resting on the table.
    Returns {shape: [rms_m, cx, cy, theta]}."""
    if len(P) > 360:
        idx = np.linspace(0, len(P) - 1, 360).astype(int)
        P = P[idx]
    c0 = np.median(P[:, :2], axis=0)
    off = np.arange(-0.016, 0.0161, 0.002)
    n = len(off)
    cx = c0[0] + off[:, None, None]
    cy = c0[1] + off[None, :, None]
    X = P[None, None, :, 0] - cx
    Y = P[None, None, :, 1] - cy
    Z = np.broadcast_to(P[None, None, :, 2] - HALF, (n, n, len(P)))
    res = {}

    def best(err):
        i = np.unravel_index(int(np.argmin(err)), err.shape)
        return float(err[i]), float(cx[i[0], 0, 0]), float(cy[0, i[1], 0])

    e, bx, by = best(np.sqrt(((np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - HALF) ** 2).mean(-1)))
    res["sphere"] = [e, bx, by, 0.0]
    e, bx, by = best(np.sqrt((_sdf_cyl(np.sqrt(X ** 2 + Y ** 2), Z) ** 2).mean(-1)))
    cyl = [e, bx, by, -1.0]  # theta -1 = upright
    box = [1e9, 0.0, 0.0, 0.0]
    for k in range(36):
        th = k * math.pi / 36.0
        c = math.cos(th)
        s = math.sin(th)
        U = c * X + s * Y
        V = -s * X + c * Y
        if k < 18:
            e, bx, by = best(np.sqrt((_sdf_box(U, V, Z) ** 2).mean(-1)))
            if e < box[0]:
                box = [e, bx, by, th]
        e, bx, by = best(np.sqrt((_sdf_cyl(np.sqrt(V ** 2 + Z ** 2), U) ** 2).mean(-1)))
        if e < cyl[0]:
            cyl = [e, bx, by, th]
    res["cylinder"] = cyl
    res["cube"] = box
    return res


def _classify(fit):
    """Best shape and log-ratio margin to the runner-up."""
    order = sorted(SHAPES, key=lambda s: fit[s][0])
    margin = math.log(max(fit[order[1]][0], 1e-6) / max(fit[order[0]][0], 1e-6))
    return order[0], margin


def _assign(fits):
    """Assign distinct shapes to up to 3 candidates minimising total log residual.
    Returns list of shape labels (same order as fits)."""
    n = len(fits)
    if n == 0:
        return []
    perms = []
    for a in SHAPES:
        for b in SHAPES:
            for c in SHAPES:
                if len({a, b, c}) == 3:
                    perms.append((a, b, c))
    if n == 3:
        bestp = None
        bestc = 1e18
        for p in perms:
            cost = sum(math.log(max(fits[i][p[i]][0], 1e-6)) for i in range(3))
            if cost < bestc:
                bestc = cost
                bestp = p
        return list(bestp)
    return [_classify(f)[0] for f in fits]
