"""Perception helpers: RGB-D back-projection, cabinet and object detection."""

TOP_FRAC = 0.5
STEEP_DZ = 0.003       # height jump (m) between neighbouring pixels treated as an edge
TOP_Z_NOMINAL = 0.663   # cabinet top height seen in all 25 demos (agentview depth)


def _cloud(depth, K, E):
    """World points (H,W,3) and valid mask from int16 optical-axis depth (mm)."""
    z = np.asarray(depth).astype(float) / 1000.0
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w]
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    W = (P - t) @ R
    return W.reshape(h, w, 3), z > 0


def _runs(flags):
    """List of (start, end_exclusive) runs of True values."""
    out = []
    start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def _top_height(W, m, x_rng=(-0.3, 0.3), y_rng=(-0.6, 0.8)):
    """Height of the dominant horizontal plane near the nominal cabinet top."""
    s = m & (W[..., 0] > x_rng[0]) & (W[..., 0] < x_rng[1]) & (W[..., 1] > y_rng[0]) & (W[..., 1] < y_rng[1])
    z = W[..., 2][s]
    z = z[(z > TOP_Z_NOMINAL - 0.06) & (z < TOP_Z_NOMINAL + 0.06)]
    if len(z) < 50:
        return TOP_Z_NOMINAL
    edges = np.arange(TOP_Z_NOMINAL - 0.06, TOP_Z_NOMINAL + 0.0605, 0.002)
    h, e = np.histogram(z, bins=edges)
    k = int(np.argmax(h))
    sel = z[(z >= e[k] - 0.002) & (z < e[k + 1] + 0.002)]
    return float(np.median(sel))


def _detect_cabinets(W, m, tcp, top=None, size=(0.25, 0.20)):
    """Find cabinet top rectangles (world xy) from a point cloud.

    Returns (top_z, list of dicts {cx, cy, x0, x1, y0, y1, n, nobj, full}) sorted by cy.
    A cabinet top is a run of y-bins holding points on the top plane or on objects
    resting on it.  Points within 0.09 m (xy) of a low gripper are ignored; a run
    cut by that exclusion zone is extended to the nominal width `size[1]`.
    """
    if top is None:
        top = _top_height(W, m)
    X, Y, Z = W[..., 0], W[..., 1], W[..., 2]
    region = m & (np.abs(X) < 0.3) & (Y > -0.6) & (Y < 0.8)
    ex = None
    if tcp is not None and tcp[2] < top + 0.2:
        ex = (float(tcp[1]) - 0.09, float(tcp[1]) + 0.09)
        near = (np.abs(X - tcp[0]) < 0.09) & (np.abs(Y - tcp[1]) < 0.09)
        region = region & ~near
    plane = region & (np.abs(Z - top) < 0.006)
    above = region & (Z > top + 0.012) & (Z < top + 0.10)
    P = W[plane]
    Q = W[above]
    if len(P) < 20:
        return top, []
    y_edges = np.arange(-0.6, 0.8001, 0.01)
    hy, _ = np.histogram(np.concatenate([P[:, 1], Q[:, 1]]), bins=y_edges)
    hp, _ = np.histogram(P[:, 1], bins=y_edges)
    flags = hy >= max(3, 0.2 * np.median(hy[hy > 0]))
    runs = _runs(flags)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= 6 and hp[r[0]:r[1]].sum() >= 30]
    cabs = []
    for a, b in runs:
        y0, y1 = float(y_edges[a]), float(y_edges[b])
        Pc = P[(P[:, 1] >= y0) & (P[:, 1] < y1)]
        x0, x1 = float(np.percentile(Pc[:, 0], 2)), float(np.percentile(Pc[:, 0], 98))
        full = (y1 - y0) >= size[1] - 0.03
        if not full and ex is not None:
            if abs(y1 - ex[0]) < 0.025:
                y1 = y0 + size[1]
            elif abs(y0 - ex[1]) < 0.025:
                y0 = y1 - size[1]
        inside = (Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 1] > y0) & (Q[:, 1] < y1)
        cabs.append({"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                     "x0": x0, "x1": x1, "y0": y0, "y1": y1, "full": bool(full),
                     "n": int(len(Pc)), "nobj": int(inside.sum())})
    cabs.sort(key=lambda c: -c["n"])
    cabs = cabs[:2]
    cabs.sort(key=lambda c: c["cy"])
    return top, cabs


def _label(mask, Z, thr=0.005, iters=400):
    """4-connected components of mask where neighbours differ in height < thr."""
    h, w = mask.shape
    lab = np.where(mask, np.arange(h * w).reshape(h, w), -1).astype(np.int64)
    big = h * w + 10
    cr = mask[:, 1:] & mask[:, :-1] & (np.abs(Z[:, 1:] - Z[:, :-1]) < thr)
    cd = mask[1:, :] & mask[:-1, :] & (np.abs(Z[1:, :] - Z[:-1, :]) < thr)
    for _ in range(iters):
        L = np.where(lab < 0, big, lab)
        new = L.copy()
        new[:, :-1] = np.where(cr, np.minimum(new[:, :-1], L[:, 1:]), new[:, :-1])
        new[:, 1:] = np.where(cr, np.minimum(new[:, 1:], L[:, :-1]), new[:, 1:])
        new[:-1, :] = np.where(cd, np.minimum(new[:-1, :], L[1:, :]), new[:-1, :])
        new[1:, :] = np.where(cd, np.minimum(new[1:, :], L[:-1, :]), new[1:, :])
        new = np.where(mask, new, -1)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _detect_objects(W, m, rgb, top, cab, margin=0.015, min_px=60, zcut=None):
    """Objects resting on a cabinet top (wrist or agentview cloud).

    Returns list of dicts {x, y, zmax, h, n, xlen, ylen, orange, kind} sorted by n desc.
    kind is 'chocolate' (thick box, top >= 2.6 cm above cabinet) or 'butter'.
    """
    X, Y, Z = W[..., 0], W[..., 1], W[..., 2]
    inb = (X > cab["x0"] - margin) & (X < cab["x1"] + margin) & (Y > cab["y0"] - margin) & (Y < cab["y1"] + margin)
    zhi = top + 0.10 if zcut is None else min(top + 0.10, zcut)
    mask = m & inb & (Z > top + 0.008) & (Z < zhi)
    # drop steep pixels (box side faces seen in perspective) so stacked/adjacent boxes separate
    Zf = np.where(m, Z, 0.0)
    steep = np.zeros(mask.shape, dtype=bool)
    d = np.abs(Zf[:, 1:] - Zf[:, :-1]) > STEEP_DZ
    steep[:, 1:] |= d
    steep[:, :-1] |= d
    d = np.abs(Zf[1:, :] - Zf[:-1, :]) > STEEP_DZ
    steep[1:, :] |= d
    steep[:-1, :] |= d
    core = mask & ~steep
    lab = _label(core, Z, thr=STEEP_DZ)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    rgbi = np.asarray(rgb).astype(int)
    out = []
    for i, c in zip(ids, counts):
        if c < min_px:
            continue
        sel = lab == i
        P = W[sel]
        col = rgbi[sel]
        hz = float(np.percentile(P[:, 2], 90) - top)
        # footprint from the upper part of the object (drops side faces seen in perspective)
        Pu = P[P[:, 2] >= top + TOP_FRAC * hz]
        if len(Pu) < 20:
            Pu = P
        orange = float(np.mean((col[:, 0] > 110) & (col[:, 2] < 60) & (col[:, 0] - col[:, 1] > 30)))
        out.append({"x": float(np.median(P[:, 0])), "y": float(np.median(P[:, 1])),
                    "xc": float((np.percentile(Pu[:, 0], 3) + np.percentile(Pu[:, 0], 97)) / 2),
                    "yc": float((np.percentile(Pu[:, 1], 3) + np.percentile(Pu[:, 1], 97)) / 2),
                    "zmax": float(np.percentile(P[:, 2], 95)), "h": hz, "n": int(c),
                    "xlen": float(np.percentile(P[:, 0], 97) - np.percentile(P[:, 0], 3)),
                    "ylen": float(np.percentile(P[:, 1], 97) - np.percentile(P[:, 1], 3)),
                    "orange": orange, "kind": "chocolate" if hz > 0.026 else "butter"})
    out.sort(key=lambda o: -o["n"])
    return out
