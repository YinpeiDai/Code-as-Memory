"""Perception helpers: agentview RGB-D back-projection, cabinet/handle and object localisation."""

TABLE_Z = 0.430
COOKIES_NOMINAL = (0.221, -0.328)
BUTTER_NOMINAL = (0.113, -0.205)
CABINET_NOMINAL_HANDLE = (0.137, 0.452, 0.607)
# measured on 50 demos: handle grasp TCP relative to the cabinet top surface extents
HANDLE_DX_FROM_TOP_CENTER = -0.0023
HANDLE_DY_FROM_TOP_YMIN = -0.0172
HANDLE_Z = 0.607


def _cloud(frame):
    """World XYZ (H, W, 3) of agentview depth pixels and a validity mask."""
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    world = (pc - t) @ R
    return world.reshape(h, w, 3), (d > 0.05)


def _cabinet(frame):
    """Locate the dark cabinet from its top surface; returns dict or None."""
    W, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    dark = rgb.max(-1) < 90
    m = valid & dark & (W[..., 2] > 0.695) & (W[..., 2] < 0.735) & (W[..., 1] > 0.30) & (W[..., 1] < 0.80) \
        & (W[..., 0] > -0.20) & (W[..., 0] < 0.45)
    P = W[m]
    if len(P) < 30:
        return None
    # discard stray points far from the main cluster (median-based)
    med = np.median(P, 0)
    keep = (np.abs(P[:, 0] - med[0]) < 0.20) & (np.abs(P[:, 1] - med[1]) < 0.20)
    P = P[keep]
    if len(P) < 30:
        return None
    xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
    ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
    hx = 0.5 * (xmin + xmax) + HANDLE_DX_FROM_TOP_CENTER
    hy = ymin + HANDLE_DY_FROM_TOP_YMIN
    return {"handle": [hx, hy, HANDLE_Z], "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "n": int(len(P)), "zt": float(np.median(P[:, 2]))}


def _object_on_table(frame, nominal, radius=0.045, exclude_xy=None):
    """Centroid of above-table points near a nominal table position; None when nothing is there."""
    W, valid = _cloud(frame)
    dx = W[..., 0] - nominal[0]
    dy = W[..., 1] - nominal[1]
    m = valid & (W[..., 2] > TABLE_Z + 0.007) & (W[..., 2] < TABLE_Z + 0.07) & (dx * dx + dy * dy < radius * radius)
    P = W[m]
    if len(P) < 8:
        return None
    c = P.mean(0)
    return {"xy": [float(c[0]), float(c[1])], "ztop": float(np.percentile(P[:, 2], 90)), "n": int(len(P))}


def _drawer_open_points(frame, cab):
    """Number of points occupying the space in front of the cabinet at drawer height (open drawer body)."""
    W, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    dark = rgb.max(-1) < 90
    m = valid & dark & (W[..., 1] > cab["ymin"] - 0.20) & (W[..., 1] < cab["ymin"] - 0.04) \
        & (W[..., 2] > 0.50) & (W[..., 2] < 0.69) \
        & (W[..., 0] > cab["xmin"] + 0.03) & (W[..., 0] < cab["xmax"] - 0.03)
    return int(m.sum())


def _drawer_front_y(frame, cab, exclude=None):
    """Robust front (min-y) of the dark middle-drawer body in the cabinet's x range, or None."""
    W, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    dark = rgb.max(-1) < 90
    m = valid & dark & (W[..., 1] > cab["ymin"] - 0.32) & (W[..., 1] < cab["ymin"] + 0.05) \
        & (W[..., 2] > 0.55) & (W[..., 2] < 0.68) \
        & (W[..., 0] > cab["xmin"] + 0.03) & (W[..., 0] < cab["xmax"] - 0.03)
    if exclude is not None:
        ex = np.asarray(exclude, dtype=float)
        dd = (W[..., 0] - ex[0]) ** 2 + (W[..., 1] - ex[1]) ** 2 + (W[..., 2] - ex[2]) ** 2
        m = m & (dd > 0.06 ** 2)
    P = W[m]
    if len(P) < 20:
        return None
    return float(np.percentile(P[:, 1], 3))


def _drawer_levels(frame, cab):
    """Dark points in front of the cabinet split by height: 'middle' (z 0.50-0.625, open middle drawer body,
    rim ~0.64) and 'top' (z 0.655-0.73, open top drawer body, rim ~0.71). Measured on rollouts R001 (middle open)
    and R007 (top drawer opened by mistake)."""
    W, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    dark = rgb.max(-1) < 90
    m = valid & dark & (W[..., 1] > cab["ymin"] - 0.30) & (W[..., 1] < cab["ymin"] - 0.04) \
        & (W[..., 0] > cab["xmin"] + 0.03) & (W[..., 0] < cab["xmax"] - 0.03)
    z = W[..., 2]
    return {"middle": int((m & (z > 0.50) & (z < 0.625)).sum()), "top": int((m & (z > 0.655) & (z < 0.73)).sum())}
