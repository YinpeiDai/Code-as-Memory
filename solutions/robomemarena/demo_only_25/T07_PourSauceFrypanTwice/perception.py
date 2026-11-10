"""Perception helpers: back-project agentview RGB-D to world points and locate
the sauce bottle, the frypan and the bowl drainer (geometry only, no RGB truth)."""

TABLE_Z_DEFAULT = 0.4307
BOTTLE_GRASP_BIAS = [-0.0127, 0.0019]   # grasp xy minus visible-surface centroid (25 demos)
GRASP_HEIGHT = 0.037                     # TCP height above table at expert grasp
PAN_DEFAULT = [0.0, 0.265]
DRAINER_DEFAULT = [-0.128, 0.475]
BOTTLE_DEFAULT = [-0.002, -0.002]


def _world_points(frame, cam="agentview"):
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    d = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    xc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    R = E[:, :3]
    t = E[:, 3]
    pts = (xc - t) @ R
    return pts.reshape(-1, 3), (d > 0).reshape(-1)


def _table_height(P, valid):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    m = valid & (np.abs(x) < 0.3) & (np.abs(y) < 0.6) & (z > 0.3) & (z < 0.55)
    if m.sum() < 200:
        return TABLE_Z_DEFAULT
    zs = z[m]
    hist, edges = np.histogram(zs, bins=125, range=(0.3, 0.55))
    k = int(np.argmax(hist))
    lo, hi = edges[k] - 0.004, edges[k + 1] + 0.004
    sel = zs[(zs >= lo) & (zs <= hi)]
    if len(sel) < 50:
        return TABLE_Z_DEFAULT
    tz = float(np.median(sel))
    if abs(tz - TABLE_Z_DEFAULT) > 0.08:
        return TABLE_Z_DEFAULT
    return tz


def _circle_fit(Q):
    A = np.stack([2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))], 1)
    b = (Q ** 2).sum(1)
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    r2 = s[2] + s[0] ** 2 + s[1] ** 2
    return np.array([s[0], s[1]]), float(math.sqrt(max(r2, 1e-9)))


def _locate(frame):
    """Returns dict with table_z, bottle [x,y], pan [x,y], drainer [x,y] and flags."""
    P, valid = _world_points(frame)
    tz = _table_height(P, valid)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    hgt = z - tz
    out = {"table_z": tz, "notes": []}
    # bottle: compact object 1..12 cm tall left of the pan
    mb = valid & (hgt > 0.01) & (hgt < 0.12) & (np.abs(x) < 0.2) & (y > -0.2) & (y < 0.13)
    B = P[mb]
    if len(B) >= 20:
        c = np.median(B[:, :2], 0)
        for _ in range(2):
            near = np.linalg.norm(B[:, :2] - c, axis=1) < 0.045
            if near.sum() < 15:
                break
            c = B[near, :2].mean(0)
        bottle = [float(c[0] + BOTTLE_GRASP_BIAS[0]), float(c[1] + BOTTLE_GRASP_BIAS[1])]
        out["bottle_points"] = int(near.sum()) if len(B) else 0
    else:
        bottle = list(BOTTLE_DEFAULT)
        out["notes"].append("bottle default")
        out["bottle_points"] = int(len(B))
    out["bottle"] = bottle
    # frypan: low ring (0.4..6 cm) right of the bottle; exclude handle side and the bottle
    mp = valid & (hgt > 0.004) & (hgt < 0.06) & (np.abs(x) < 0.22) & (y > 0.08) & (y < 0.352)
    Q = P[mp][:, :2]
    if len(Q) >= 1:
        Q = Q[np.linalg.norm(Q - np.array(bottle), axis=1) > 0.06]
    pan = list(PAN_DEFAULT)
    if len(Q) >= 30:
        c, r = _circle_fit(Q)
        for _ in range(4):
            dd = np.abs(np.linalg.norm(Q - c, axis=1) - r)
            keep = dd < 0.015
            if keep.sum() < 30:
                break
            Q = Q[keep]
            c, r = _circle_fit(Q)
        if 0.075 < r < 0.115 and np.linalg.norm(c - np.array(PAN_DEFAULT)) < 0.12:
            pan = [float(c[0]), float(c[1])]
            out["pan_r"] = r
        else:
            out["notes"].append("pan fit rejected r=%.3f" % r)
    else:
        out["notes"].append("pan default")
    out["pan"] = pan
    # bowl drainer: tall walls (>12 cm) beyond the pan
    md = valid & (hgt > 0.12) & (hgt < 0.25) & (x > -0.45) & (x < 0.15) & (y > 0.36) & (y < 0.85)
    D = P[md]
    if len(D) >= 30:
        x0, x1 = np.percentile(D[:, 0], [2, 98])
        y0, y1 = np.percentile(D[:, 1], [2, 98])
        drainer = [float(0.5 * (x0 + x1)), float(y0 + 0.25 * (y1 - y0))]
        out["drainer_box"] = [float(x0), float(x1), float(y0), float(y1)]
        if (x1 - x0) < 0.1 or (y1 - y0) < 0.12:
            out["notes"].append("drainer box small")
    else:
        drainer = list(DRAINER_DEFAULT)
        out["notes"].append("drainer default")
    out["drainer"] = drainer
    return out


def perceive_scene(obs):
    """Locate table height, bottle grasp xy, frypan centre xy and drainer slot xy from the latest agentview RGB-D."""
    return _locate(obs.current)
