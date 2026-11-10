"""Perception helpers: agentview RGB-D back-projection and scene measurements."""

TABLE_Z = 0.43


def _cloud(frame, cam="agentview"):
    """World XYZ (H, W, 3) from a camera's depth image, or None when depth is absent."""
    d = getattr(frame, cam + "_depth", None)
    if d is None:
        return None
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    z = d.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    c = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    w = (c - t) @ R
    w[z.reshape(-1) <= 0] = np.nan
    return w.reshape(H, W, 3)


def _two_clusters(P):
    """Split points into two clusters along their main horizontal spread (simple 2-means)."""
    if len(P) < 2:
        return [P]
    xy = P[:, :2]
    c = xy.mean(0)
    cov = np.cov((xy - c).T)
    w, V = np.linalg.eigh(cov)
    ax = V[:, int(np.argmax(w))]
    s = (xy - c) @ ax
    c1, c2 = s.min(), s.max()
    for _ in range(20):
        lab = np.abs(s - c1) <= np.abs(s - c2)
        if lab.all() or (~lab).all():
            break
        c1, c2 = s[lab].mean(), s[~lab].mean()
    lab = np.abs(s - c1) <= np.abs(s - c2)
    return [P[lab], P[~lab]]


def _scene(frame):
    """Microwave centroid and small-object (cookies, chocolate) grasp estimates from agentview depth.

    Returns dict with microwave [x, y], cookies [x, y, top_z], chocolate [x, y, top_z], pixel counts
    and an ok flag. Object assignment: the taller of the two small boxes is the chocolate pudding
    (22 mm vs 14 mm in every demo); the brighter is the cookies box (secondary check)."""
    out = {"ok": False, "microwave": None, "cookies": None, "chocolate": None,
           "n_mw": 0, "n_obj": [0, 0], "assign": "none"}
    W = _cloud(frame)
    if W is None:
        return out
    rgb = frame.agentview_rgb.astype(float)
    x, y, z = W[..., 0], W[..., 1], W[..., 2]
    good = np.isfinite(z)
    x = np.where(good, x, -9.0)
    y = np.where(good, y, -9.0)
    z = np.where(good, z, -9.0)
    on = (z > 0.445) & (z < 0.9) & (x > -0.35) & (x < 0.45) & (y > -0.5) & (y < 0.7)
    mw = on & (y > 0.2)
    out["n_mw"] = int(mw.sum())
    if mw.sum() > 300:
        out["microwave"] = [float(x[mw].mean()), float(y[mw].mean())]
    ob = (z > TABLE_Z + 0.007) & (z < 0.49) & (x > -0.1) & (x < 0.4) & (y > -0.4) & (y < 0.15)
    P = np.stack([x[ob], y[ob], z[ob]], 1)
    C = rgb[ob]
    if len(P) >= 30:
        cl = _two_clusters(np.concatenate([P, C], 1))
        stats = []
        for Q in cl:
            if len(Q) < 10:
                continue
            stats.append({"x": float(Q[:, 0].mean()), "y": float(Q[:, 1].mean()),
                          "h": float(np.median(Q[:, 2])), "bright": float(Q[:, 3:6].mean()),
                          "n": int(len(Q))})
        if len(stats) == 2:
            a, b = stats
            # chocolate is taller; cookies brighter
            if abs(a["h"] - b["h"]) > 0.003:
                choc, cook = (a, b) if a["h"] > b["h"] else (b, a)
                out["assign"] = "height"
            else:
                cook, choc = (a, b) if a["bright"] > b["bright"] else (b, a)
                out["assign"] = "brightness"
            out["cookies"] = [cook["x"], cook["y"], cook["h"]]
            out["chocolate"] = [choc["x"], choc["y"], choc["h"]]
            out["n_obj"] = [cook["n"], choc["n"]]
    out["ok"] = out["microwave"] is not None and out["cookies"] is not None
    return out


def _count_box(W, lo, hi):
    """Number of valid world points inside the axis-aligned box [lo, hi]."""
    if W is None:
        return -1
    P = W.reshape(-1, 3)
    P = P[np.isfinite(P[:, 2])]
    m = np.all((P >= np.array(lo)) & (P <= np.array(hi)), axis=1)
    return int(m.sum())


def _door_state(frame, mw):
    """Evidence of the microwave door state from agentview depth, given microwave centroid mw=[x, y].

    handle_pts: points where the closed-door handle stands (many when closed);
    leaf_pts: points of an opened door leaf standing left (-y) of the microwave at the hinge side."""
    W = _cloud(frame)
    hx, hy = mw[0] - 0.019, mw[1] - 0.115
    handle = _count_box(W, [hx - 0.02, hy - 0.07, 0.56], [hx + 0.06, hy - 0.005, 0.635])
    leaf = _count_box(W, [mw[0] - 0.36, mw[1] - 0.33, 0.60], [mw[0] - 0.22, mw[1] - 0.12, 0.66])
    return {"handle_pts": handle, "leaf_pts": leaf}


def _objects_left(frame, pos, r=0.03):
    """Count raised points (above table) around a table location pos=[x, y] in agentview depth."""
    W = _cloud(frame)
    return _count_box(W, [pos[0] - r, pos[1] - r, TABLE_Z + 0.007], [pos[0] + r, pos[1] + r, 0.49])
