"""Perception helpers for T03_CreamPuddingBasket (no imports; np and math are provided)."""

TABLE_Z_DEFAULT = 0.430


def _world_points(depth, K, E):
    """Deproject an int16 millimetre depth image into an (H, W, 3) array of world points."""
    d = np.asarray(depth).astype(float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    c = np.stack([x, y, d], -1).reshape(-1, 3)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    P = (c - t) @ R
    return P.reshape(h, w, 3), d > 0


def _table_z(P, valid):
    """Estimate the table height as the dominant low horizontal surface."""
    z = P[..., 2][valid]
    z = z[(z > 0.3) & (z < 0.6)]
    if len(z) < 100:
        return TABLE_Z_DEFAULT
    hist, edges = np.histogram(z, bins=np.arange(0.3, 0.6, 0.002))
    k = int(np.argmax(hist))
    tz = float(0.5 * (edges[k] + edges[k + 1]))
    if abs(tz - TABLE_Z_DEFAULT) > 0.05:
        return TABLE_Z_DEFAULT
    return tz


def _dilate(mask, r):
    """Binary dilation by a (2r+1) square using shifts."""
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    h, w = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.zeros_like(m)
            ys0, ys1 = max(0, dy), h + min(0, dy)
            xs0, xs1 = max(0, dx), w + min(0, dx)
            sh[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            out |= sh
    return out


def _blobs(mask, max_cells=6000):
    """All 4-connected components of a boolean mask (pure-Python flood fill), largest first."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros_like(m)
    comps = []
    ys, xs = np.nonzero(m)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp = []
        while stack:
            y, x = stack.pop()
            if len(comp) < max_cells:
                comp.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    comps.sort(key=lambda c: -len(c))
    return comps


def _object_masks(frame):
    """Colour+height masks of the cream box and chocolate box in the wrist view.

    Returns (P, table_z, cream_mask, choc_mask)."""
    P, valid = _world_points(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    tz = _table_z(P, valid)
    h = P[..., 2] - tz
    rgb = np.asarray(frame.wrist_rgb).astype(float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = (r + g + b) / 3.0
    tall = valid & (h > 0.07)
    near_tall = _dilate(tall, 4)
    low = valid & (h > 0.004) & (h < 0.06) & (~near_tall)
    # exclude the gripper body (very close to the camera) and far-off points
    low &= np.abs(P[..., 0]) < 0.45
    low &= np.abs(P[..., 1]) < 0.45
    cream = low & (b - r > 10.0) & (bright < 200)
    choc = low & (r - b > 12.0) & (bright < 140)
    return P, tz, cream, choc


def _blob_list(P, mask, min_px):
    """World centroid / extent of every plausibly box-sized blob in mask, largest first."""
    out = []
    for comp in _blobs(mask):
        if len(comp) < min_px:
            break
        ys = np.array([c[0] for c in comp])
        xs = np.array([c[1] for c in comp])
        Q = P[ys, xs]
        lo = np.percentile(Q, 5, axis=0)
        hi = np.percentile(Q, 95, axis=0)
        sx, sy = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        # a box seen from above: both footprint sides between 1.5 cm and 15 cm
        if min(sx, sy) < 0.015 or max(sx, sy) > 0.15:
            continue
        ctr = 0.5 * (lo + hi)
        out.append({
            "xy": [float(ctr[0]), float(ctr[1])],
            "top": float(np.percentile(Q[:, 2], 90)),
            "size": [sx, sy],
            "px": int(len(comp)),
            "uv": [float(xs.mean()), float(ys.mean())],
        })
    return out


def _pick_blob(blobs, near=None, radius=0.04):
    """Largest blob, or the largest within `radius` of the prior xy `near` (None if none)."""
    if not blobs:
        return None
    if near is None:
        return blobs[0]
    for b in blobs:
        if math.hypot(b["xy"][0] - near[0], b["xy"][1] - near[1]) < radius:
            return b
    return None


def _detect_objects(frame, near=None, radius=0.04):
    """Detect cream and chocolate boxes in a wrist RGB-D frame.

    near: optional {"cream": [x, y], "chocolate": [x, y]} priors; blobs farther than radius are ignored."""
    P, tz, cream, choc = _object_masks(frame)
    near = near or {}
    return {
        "table_z": tz,
        "cream": _pick_blob(_blob_list(P, cream, 30), near.get("cream"), radius),
        "chocolate": _pick_blob(_blob_list(P, choc, 30), near.get("chocolate"), radius),
        "cream_all": [b["xy"] for b in _blob_list(P, cream, 30)],
        "chocolate_all": [b["xy"] for b in _blob_list(P, choc, 30)],
    }


def _detect_baskets(frame):
    """Basket rim clusters from the agentview depth: list of dicts sorted by x (robot side first)."""
    P, valid = _world_points(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    tz = _table_z(P, valid)
    h = P[..., 2] - tz
    m = valid & (h > 0.06) & (h < 0.2) & (np.abs(P[..., 0]) < 0.4) & (np.abs(P[..., 1]) < 0.5)
    # the arm hangs over the table at the start; exclude points near the gripper
    Q = P[m]
    if len(Q) < 50:
        return []
    xs = Q[:, 0]
    bins = np.arange(-0.4, 0.41, 0.01)
    hist, _ = np.histogram(xs, bins=bins)
    occ = hist > 3
    clusters = []
    i = 0
    n = len(occ)
    while i < n:
        if not occ[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j + 1 < n and (occ[j + 1] or (j + 2 < n and occ[j + 2])):
            j += 1
        clusters.append((bins[i], bins[j + 1]))
        i = j + 1
    out = []
    for lo, hi in clusters:
        S = Q[(xs >= lo) & (xs < hi)]
        if len(S) < 40:
            continue
        a = np.percentile(S, 3, axis=0)
        b = np.percentile(S, 97, axis=0)
        sx, sy = b[0] - a[0], b[1] - a[1]
        if sx < 0.06 or sy < 0.06 or sx > 0.3 or sy > 0.3:
            continue
        out.append({
            "xy": [float(0.5 * (a[0] + b[0])), float(0.5 * (a[1] + b[1]))],
            "size": [float(sx), float(sy)],
            "rim_z": float(np.percentile(S[:, 2], 90)),
            "n": int(len(S)),
        })
    out.sort(key=lambda d: d["xy"][0])
    return out
