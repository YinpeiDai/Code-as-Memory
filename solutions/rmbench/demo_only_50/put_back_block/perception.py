"""Perception helpers: colour segmentation of the head RGB-D view, back-projected to world metres."""


def _world_points(frame, cam, mask):
    """Back-project masked pixels of camera `cam` to world xyz (N x 3); invalid depth dropped."""
    depth = np.asarray(getattr(frame, cam + "_depth")).astype(float)
    K = np.asarray(getattr(frame, cam + "_K")).astype(float)
    E = np.asarray(getattr(frame, cam + "_extrinsic")).astype(float)
    ys, xs = np.nonzero(mask & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs] / 1000.0
    xc = (xs - K[0, 2]) * z / K[0, 0]
    yc = (ys - K[1, 2]) * z / K[1, 1]
    pc = np.stack([xc, yc, z], 1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R  # R^T (pc - t)


def _masks(frame, cam):
    im = np.asarray(getattr(frame, cam + "_rgb")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (r > 150) & (g < 110) & (b < 110) & (r - g > 90)
    blue = (b > 150) & (r < 120) & (b - r > 80)
    return red, blue


def _kmeans(pts, k, iters=20):
    """Tiny deterministic k-means on 2-D points; returns centres and labels."""
    n = len(pts)
    if n < k:
        return pts.copy(), np.arange(n)
    # farthest-point initialisation
    cs = [pts[np.argmin(pts[:, 0])]]
    for _ in range(k - 1):
        d = np.min(np.stack([np.sum((pts - c) ** 2, 1) for c in cs], 1), 1)
        cs.append(pts[int(np.argmax(d))])
    cs = np.array(cs, dtype=float)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.stack([np.sum((pts - c) ** 2, 1) for c in cs], 1)
        lab = np.argmin(d, 1)
        for j in range(k):
            if np.any(lab == j):
                cs[j] = pts[lab == j].mean(0)
    return cs, lab


def _scene(frame, cam="head", zmax=0.80):
    """Detect mats (blue), block and button (red) in world coordinates from one frame.

    Returns dict with mats (list of [x,y]), centre [x,y], block [x,y,ztop] or None,
    button [x,y,ztop] or None, counts. Points above 0.70 m world z near the table only.
    """
    red, blue = _masks(frame, cam)
    bp = _world_points(frame, cam, blue)
    rp = _world_points(frame, cam, red)
    # only table-level objects (excludes the red logo on the arms); table top is ~0.74 m
    if len(rp):
        rp = rp[(rp[:, 2] > 0.735) & (rp[:, 2] < zmax)]
    if len(bp):
        bp = bp[(bp[:, 2] > 0.72) & (bp[:, 2] < 0.78)]
    out = {"mats": [], "centre": None, "block": None, "button": None,
           "n_blue": int(len(bp)), "n_red": int(len(rp)), "objs": []}
    if len(bp) >= 40:
        cs, lab = _kmeans(bp[:, :2], 4)
        mats = []
        for j in range(len(cs)):
            if np.sum(lab == j) >= 10:
                mats.append([float(cs[j, 0]), float(cs[j, 1])])
        out["mats"] = mats
        if mats:
            out["centre"] = [float(np.mean([m[0] for m in mats])), float(np.mean([m[1] for m in mats]))]
    if len(rp) >= 10:
        if len(rp) >= 20:
            cs, lab = _kmeans(rp[:, :2], 2)
            groups = [rp[lab == j] for j in range(len(cs)) if np.sum(lab == j) >= 5]
            # merge if the two clusters are close (single object split)
            if len(groups) == 2 and np.linalg.norm(groups[0][:, :2].mean(0) - groups[1][:, :2].mean(0)) < 0.06:
                groups = [rp]
        else:
            groups = [rp]
        objs = []
        for gpts in groups:
            ztop = float(np.percentile(gpts[:, 2], 95))
            top = gpts[gpts[:, 2] > ztop - 0.006]
            if len(top) < 5:
                top = gpts
            # top face is seen whole from the overhead head camera: use its bounding-box centre
            xy = 0.5 * (np.percentile(top[:, :2], 5, axis=0) + np.percentile(top[:, :2], 95, axis=0))
            objs.append([float(xy[0]), float(xy[1]), ztop, int(len(gpts))])
        ref = out["centre"]
        if ref is None and out["mats"]:
            ref = [float(np.mean([m[0] for m in out["mats"]])), float(np.mean([m[1] for m in out["mats"]]))]
        if ref is not None:
            objs.sort(key=lambda o: (o[0] - ref[0]) ** 2 + (o[1] - ref[1]) ** 2)
        else:
            objs.sort(key=lambda o: -o[0])
        out["objs"] = objs
        out["block"] = objs[0]
        if len(objs) > 1:
            out["button"] = objs[1]
    return out


def _mat_centre(mats, block_xy):
    """Centre of the four-mat cross. The mat under the block is partly hidden (biased), so use the
    midpoint of the two mats perpendicular to the block's arm of the cross; fall back to the mean."""
    if len(mats) != 4:
        if not mats:
            return None
        return [float(np.mean([m[0] for m in mats])), float(np.mean([m[1] for m in mats]))]
    M = np.array(mats, dtype=float)
    if block_xy is None:
        return [float(M[:, 0].mean()), float(M[:, 1].mean())]
    b = np.array(block_xy[:2], dtype=float)
    i = int(np.argmin(np.sum((M - b) ** 2, 1)))
    j = int(np.argmax(np.sum((M - M[i]) ** 2, 1)))
    rest = [k for k in range(4) if k not in (i, j)]
    c = M[rest].mean(0)
    return [float(c[0]), float(c[1])]


def _low_red_near(frame, xy, radius=0.05, cam="head"):
    """Number of red table-level pixels (z 0.735-0.80 m) within `radius` of xy: block resting there."""
    red, _ = _masks(frame, cam)
    rp = _world_points(frame, cam, red)
    if len(rp) == 0:
        return 0
    m = (rp[:, 2] > 0.735) & (rp[:, 2] < 0.80)
    m = m & (np.hypot(rp[:, 0] - xy[0], rp[:, 1] - xy[1]) < radius)
    return int(np.sum(m))


def _block_excluding(scene, button, ref):
    """Red table-level object that is not the remembered button, nearest to ref [x, y]; None if absent."""
    best = None
    for o in scene.get("objs", []):
        if button is not None and math.hypot(o[0] - button[0], o[1] - button[1]) < 0.05:
            continue
        if o[3] < 15:
            continue
        d = math.hypot(o[0] - ref[0], o[1] - ref[1])
        if best is None or d < best[0]:
            best = (d, o)
    return None if best is None else best[1]
