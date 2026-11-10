"""Perception helpers: agentview RGB-D back-projection and object detection (butter, frypan, milk)."""

# Offsets calibrated on the 25 assigned demonstrations (expert action point minus detected feature).
BUTTER_GRASP_OFF = (-0.0174, 0.0019)
BUTTER_GRASP_DZ = 0.003      # tcp height above table at butter grasp (expert 0.433 with table 0.430)
MILK_GRASP_OFF = (-0.010, 0.000)
MILK_GRASP_DZ = 0.013        # tcp height above table at milk grasp (expert 0.438..0.447)
PAN_PLACE_OFF = (-0.0061, 0.0087)
PAN_POUR_OFF = (0.0115, -0.0104)
DEFAULT_TABLE_Z = 0.430
# Fallbacks = mean detections over the assigned demonstrations (used only if detection fails).
DEFAULT_BUTTER = (0.1269, -0.2044)
DEFAULT_MILK_TOP = (-0.0982, -0.1741)
DEFAULT_PAN = (0.0073, 0.1283, 0.0992)


def _frame_attr(frame, name):
    return getattr(frame, name, None)


def _backproject(depth, K, E):
    """World points (H, W, 3) and validity mask from int16 mm depth, 3x3 K and 3x4 world-to-camera E."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    pw = (pc - t) @ R
    return pw, z > 0


def _fit_circle(x, y):
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = s[0] / 2.0, s[1] / 2.0
    r2 = s[2] + cx * cx + cy * cy
    return float(cx), float(cy), float(math.sqrt(max(r2, 1e-9)))


def _detect_scene(frame):
    """Detect table height, milk top centre, frypan circle and butter centroid in world metres.

    Returns a dict; each object entry is None when it was not found."""
    depth = _frame_attr(frame, "agentview_depth")
    K = _frame_attr(frame, "agentview_K")
    E = _frame_attr(frame, "agentview_extrinsic")
    rgb = _frame_attr(frame, "agentview_rgb")
    out = {"table_z": DEFAULT_TABLE_Z, "milk": None, "pan": None, "butter": None,
           "counts": [0, 0, 0]}
    if depth is None or K is None or E is None or rgb is None:
        return out
    pw, valid = _backproject(np.asarray(depth), np.asarray(K), np.asarray(E))
    X, Y, Z = pw[..., 0], pw[..., 1], pw[..., 2]
    region = valid & (X > -0.32) & (X < 0.38) & (np.abs(Y) < 0.48)
    tcp = _frame_attr(frame, "tcp")
    if tcp is not None:
        tcp = np.asarray(tcp, dtype=float)
        near_arm = (np.hypot(X - tcp[0], Y - tcp[1]) < 0.08) & (Z > tcp[2] - 0.04)
        region = region & ~near_arm
    tab = region & (Z > 0.33) & (Z < 0.53)
    h = DEFAULT_TABLE_Z
    if tab.sum() > 500:
        zs = Z[tab]
        h0 = float(np.median(zs))
        zs2 = zs[np.abs(zs - h0) < 0.004]
        if len(zs2) > 200:
            h = float(np.median(zs2))
    if abs(h - DEFAULT_TABLE_Z) > 0.05:
        h = DEFAULT_TABLE_Z
    out["table_z"] = h
    img = np.asarray(rgb).astype(int)
    above = region & (Z > h + 0.006) & (Z < h + 0.19)

    # milk: the only tall object (> 7 cm above the table)
    tall = above & (Z > h + 0.07)
    milk = None
    if tall.sum() >= 8:
        pz = Z[tall]
        px, py = X[tall], Y[tall]
        # keep the biggest group near the highest points
        zt = float(np.max(pz))
        sel = pz > zt - 0.012
        cx, cy = float(np.mean(px[sel])), float(np.mean(py[sel]))
        grp = np.hypot(px - cx, py - cy) < 0.06
        if grp.sum() >= 8 and sel.sum() >= 4:
            zt = float(np.max(pz[grp]))
            sel = grp & (pz > zt - 0.012)
            milk = [float(np.mean(px[sel])), float(np.mean(py[sel])), zt]
    out["milk"] = milk
    out["counts"][0] = int(tall.sum())

    def away_from_milk(mask):
        if milk is None:
            return mask
        return mask & (np.hypot(X - milk[0], Y - milk[1]) > 0.065)

    # frypan: dark grey points low above the table, robust circle fit (handle rejected as outliers)
    mx = img.max(-1)
    mn = img.min(-1)
    dark = away_from_milk(above & (Z < h + 0.07) & (mx < 100) & ((mx - mn) < 30))
    pan = None
    if dark.sum() >= 40:
        px, py = X[dark], Y[dark]
        keep = np.ones(len(px), bool)
        cx = cy = r = 0.0
        for _ in range(8):
            if keep.sum() < 20:
                break
            cx, cy, r = _fit_circle(px[keep], py[keep])
            d = np.abs(np.hypot(px - cx, py - cy) - r)
            keep = d < max(0.012, 2.5 * float(np.median(d[keep])))
        if keep.sum() >= 30 and 0.05 < r < 0.16:
            pan = [cx, cy, r]
    out["pan"] = pan
    out["counts"][1] = int(dark.sum())

    # butter: short, reddish object outside the pan and away from the milk
    low = away_from_milk(above & (Z < h + 0.035) & ((img[..., 0] - img[..., 2]) > 20))
    if pan is not None:
        low = low & (np.hypot(X - pan[0], Y - pan[1]) > pan[2] + 0.02)
    butter = None
    if low.sum() >= 8:
        px, py = X[low], Y[low]
        mxy = (float(np.median(px)), float(np.median(py)))
        grp = np.hypot(px - mxy[0], py - mxy[1]) < 0.045
        if grp.sum() >= 8:
            butter = [float(np.mean(px[grp])), float(np.mean(py[grp]))]
    out["butter"] = butter
    out["counts"][2] = int(low.sum())
    return out


def perceive_scene(obs):
    """Detect butter, frypan and milk in the latest agentview RGB-D frame (world metres)."""
    return _detect_scene(obs.current)
