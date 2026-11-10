"""Perception helpers: agentview RGB-D back-projection and scene anchors."""

TABLE_Z_DEFAULT = 0.4307
# Fallback anchors (means over the 25 assigned demonstrations) used only when perception fails.
DEFAULT_HANDLE = [0.0485, 0.4100, 0.5820]
DEFAULT_COOKIES = [0.1637, -0.0927, 0.4380]
DEFAULT_POPCORN = [0.1637, -0.1922, 0.4380]


def _cloud(frame, cam):
    """World XYZ (H*W x 3), validity mask, and flat RGB for one camera of a frame."""
    D = np.asarray(getattr(frame, cam + "_depth")).astype(float) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    h, w = D.shape
    v, u = np.mgrid[0:h, 0:w]
    z = D
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    c = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    W = (c - t) @ R
    valid = D.reshape(-1) > 0.05
    rgb = np.asarray(getattr(frame, cam + "_rgb")).reshape(-1, 3).astype(float)
    return W, valid, rgb


def _table_z(W, valid):
    sel = valid & (W[:, 2] > 0.35) & (W[:, 2] < 0.50) & (W[:, 0] > -0.2) & (W[:, 0] < 0.45) \
        & (W[:, 1] > -0.45) & (W[:, 1] < 0.3)
    if sel.sum() < 200:
        return TABLE_Z_DEFAULT
    z = W[sel, 2]
    # the table is the dominant flat surface: take the mode of a fine histogram
    hist, edges = np.histogram(z, bins=150, range=(0.35, 0.50))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def _microwave(W, valid, table_z):
    """Handle grasp anchor from the microwave point cloud (box above the table at +y)."""
    sel = valid & (W[:, 2] > table_z + 0.03) & (W[:, 2] < 0.80) & (W[:, 1] > 0.12) & (W[:, 1] < 0.85) \
        & (W[:, 0] > -0.40) & (W[:, 0] < 0.50)
    n = int(sel.sum())
    if n < 300:
        return None, n
    Q = W[sel]
    zmax = float(np.percentile(Q[:, 2], 99.5))
    top = Q[Q[:, 2] > zmax - 0.012]
    xmax = float(np.percentile(Q[:, 0], 99.5))
    ymin = float(np.percentile(Q[:, 1], 0.5))
    ymax = float(np.percentile(Q[:, 1], 99.5))
    tx98 = float(np.percentile(top[:, 0], 98)) if len(top) > 50 else xmax + 0.0009
    hx = 0.5 * ((xmax + MW_OFF["xmax"]) + (tx98 + MW_OFF["tx98"]))
    hy = 0.5 * ((ymin + MW_OFF["ymin"]) + (ymax + MW_OFF["ymax"]))
    hz = zmax + MW_OFF["zmax"]
    feats = {"xmax": xmax, "tx98": tx98, "ymin": ymin, "ymax": ymax, "zmax": zmax, "n": n}
    return [hx, hy, hz], feats


# Offsets fitted on the 25 demonstrations (scratch/fit_perception.py): handle = feature + offset
MW_OFF = {"xmax": -0.1241, "tx98": -0.1232, "ymin": 0.0154, "ymax": -0.2433, "zmax": -0.0629}
# grasp point = bounding-box centre (x, y) + offset; z = box top + offset
OBJ_OFF = {"cookies": [-0.0064, 0.0016], "popcorn": [-0.0046, 0.0029], "z": -0.007}


def _boxes(W, valid, rgb, table_z):
    """Two flat boxes on the table left of the microwave; labelled by colour (cookies are red/orange)."""
    sel = valid & (W[:, 2] > table_z + 0.0065) & (W[:, 2] < table_z + 0.10) & (W[:, 0] > -0.15) \
        & (W[:, 0] < 0.40) & (W[:, 1] > -0.45) & (W[:, 1] < 0.15)
    Q = W[sel]
    C = rgb[sel]
    if len(Q) < 20:
        return None
    order = np.argsort(Q[:, 1])
    ys = Q[order, 1]
    gaps = np.diff(ys)
    if len(gaps) == 0:
        return None
    k = int(np.argmax(gaps))
    if gaps[k] < 0.02:
        # no clear separation along y; try x
        order = np.argsort(Q[:, 0])
        xs = Q[order, 0]
        gx = np.diff(xs)
        k = int(np.argmax(gx))
        if gx[k] < 0.02:
            return None
    a = order[: k + 1]
    b = order[k + 1:]
    out = []
    for idx in (a, b):
        q = Q[idx]
        if len(q) < 8:
            return None
        lo = np.percentile(q, 2, axis=0)
        hi = np.percentile(q, 98, axis=0)
        col = C[idx].mean(0)
        out.append({"center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
                    "top": float(np.percentile(q[:, 2], 95)), "n": int(len(q)),
                    "redness": float(col[0] - col[2])})
    # cookies box is clearly redder than the blue/white popcorn box; fall back to y order (cookies at larger y)
    if abs(out[0]["redness"] - out[1]["redness"]) > 12:
        ci = 0 if out[0]["redness"] > out[1]["redness"] else 1
        how = "colour"
    else:
        ci = 0 if out[0]["center"][1] > out[1]["center"][1] else 1
        how = "y-order"
    cook, pop = out[ci], out[1 - ci]
    res = {"how": how}
    for name, o in (("cookies", cook), ("popcorn", pop)):
        off = OBJ_OFF[name]
        z = o["top"] + OBJ_OFF["z"]
        z = min(max(z, table_z + 0.004), table_z + 0.03)
        res[name] = [o["center"][0] + off[0], o["center"][1] + off[1], z]
        res[name + "_n"] = o["n"]
    return res


def _scene(frame):
    """All anchors from one agentview frame. Returns dict with handle/cookies/popcorn and flags."""
    W, valid, rgb = _cloud(frame, "agentview")
    tz = _table_z(W, valid)
    handle, feats = _microwave(W, valid, tz)
    boxes = _boxes(W, valid, rgb, tz)
    return {"table_z": tz, "handle": handle, "mw": feats, "boxes": boxes}


def _door_open_points(frame, handle):
    """Count agentview points where the opened door stands (front-left of the microwave)."""
    W, valid, rgb = _cloud(frame, "agentview")
    xmax = handle[0] - MW_OFF["xmax"]
    ymin = handle[1] - MW_OFF["ymin"]
    rx = W[:, 0] - xmax
    ry = W[:, 1] - ymin
    d = valid & (rx < -0.36) & (rx > -0.60) & (ry > -0.25) & (ry < 0.05) & (W[:, 2] > 0.46) & (W[:, 2] < 0.70)
    return int(d.sum())


def _single_box(frame, near, table_z, radius=0.05):
    """Re-locate one flat box near its remembered grasp point (agentview). Returns grasp point or None."""
    W, valid, rgb = _cloud(frame, "agentview")
    gx = near[0] - OBJ_OFF["popcorn"][0]
    gy = near[1] - OBJ_OFF["popcorn"][1]
    sel = valid & (W[:, 2] > table_z + 0.0065) & (W[:, 2] < table_z + 0.10) \
        & (np.abs(W[:, 0] - gx) < radius) & (np.abs(W[:, 1] - gy) < radius)
    q = W[sel]
    if len(q) < 25:
        return None
    lo = np.percentile(q, 2, axis=0)
    hi = np.percentile(q, 98, axis=0)
    if hi[0] - lo[0] > 0.09 or hi[1] - lo[1] > 0.09:
        return None
    z = float(np.percentile(q[:, 2], 95)) + OBJ_OFF["z"]
    z = min(max(z, table_z + 0.004), table_z + 0.03)
    return [float((lo[0] + hi[0]) / 2) + OBJ_OFF["popcorn"][0], float((lo[1] + hi[1]) / 2) + OBJ_OFF["popcorn"][1], z]
