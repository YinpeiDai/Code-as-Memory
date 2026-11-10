"""Perception helpers for RememberShape5: blue-object segmentation in the wrist
camera, back-projection to world coordinates and rule-based shape classes."""

SHAPES = ["cube", "cylinder", "sphere", "torus", "cross"]


def _blue_mask(rgb):
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    return (b > 80) & (b > r + 40) & (b > g + 40)


def _components(mask, min_px=20):
    """4-connected components of a boolean mask (pure python BFS over set pixels)."""
    ys, xs = np.nonzero(mask)
    todo = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while todo:
        seed = todo.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            y, x = stack.pop()
            for nb in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if nb in todo:
                    todo.remove(nb)
                    stack.append(nb)
                    comp.append(nb)
        if len(comp) >= min_px:
            comps.append(comp)
    return comps


def _world_points(frame, cam, pix):
    """pix: list of (row, col). Returns Nx3 world points (metres) for pixels with valid depth."""
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    depth = np.asarray(getattr(frame, cam + "_depth"))
    rows = np.array([p[0] for p in pix])
    cols = np.array([p[1] for p in pix])
    d = depth[rows, cols].astype(float) / 1000.0
    ok = d > 0
    rows, cols, d = rows[ok], cols[ok], d[ok]
    x = (cols - K[0, 2]) / K[0, 0] * d
    y = (rows - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], axis=1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


def _shape_features(P):
    P = P[P[:, 2] > 0.003]
    if len(P) < 15:
        return None
    zmax = float(np.percentile(P[:, 2], 98))
    top = P[P[:, 2] > zmax - 0.004]
    c = top[:, :2].mean(axis=0)
    qt = top[:, :2] - c
    radt = np.sqrt((qt ** 2).sum(axis=1))
    ang = np.arctan2(qt[:, 1], qt[:, 0])
    nb = 24
    rm = np.zeros(nb)
    for k in range(nb):
        lo = -math.pi + k * 2 * math.pi / nb
        sel = radt[(ang >= lo) & (ang < lo + 2 * math.pi / nb)]
        rm[k] = sel.max() if len(sel) else 0.0
    return {
        "x": float(c[0]), "y": float(c[1]),
        "zmax": zmax,
        "zmed": float(np.median(P[:, 2])),
        "hole": float(np.mean(radt < 0.005)),
        "roundness": float(np.percentile(rm, 10) / max(rm.max(), 1e-6)),
        "rmax": float(np.percentile(rm, 90)),
        "n": int(len(P)),
    }


def _classify(f):
    """Rule thresholds from 50 demos (ranges: cube roundness .66-.75 / rmax .025-.027;
    cylinder .87-.96 / .020-.021; sphere zmed .033-.036; torus hole 0; cross hole >= .038)."""
    if f["zmax"] < 0.028:
        return "torus" if f["hole"] < 0.015 else "cross"
    if f["zmed"] < 0.0378:
        return "sphere"
    if f["roundness"] < 0.81 and f["rmax"] > 0.0225:
        return "cube"
    if f["roundness"] >= 0.81 and f["rmax"] <= 0.0225:
        return "cylinder"
    return "cube" if (0.81 - f["roundness"]) / 0.05 + (f["rmax"] - 0.0225) / 0.002 > 0 else "cylinder"


def _feat_dist(a, b):
    return (abs(a["zmax"] - b["zmax"]) / 0.005 + abs(a["zmed"] - b["zmed"]) / 0.002
            + abs(a["roundness"] - b["roundness"]) / 0.05 + abs(a["rmax"] - b["rmax"]) / 0.002
            + abs(a["hole"] - b["hole"]) / 0.02)


def _blob_count_fast(frame):
    m = _blue_mask(np.asarray(frame.wrist_rgb))
    return int(m.sum()), m


def _scene_objects(frame, cam="wrist"):
    """Segment blue objects and return a list of feature dicts with a shape label."""
    m = _blue_mask(np.asarray(getattr(frame, cam + "_rgb")))
    out = []
    for comp in _components(m):
        P = _world_points(frame, cam, comp)
        if len(P) < 15:
            continue
        f = _shape_features(P)
        if f is None:
            continue
        f["shape"] = _classify(f)
        f["px"] = len(comp)
        out.append(f)
    return out
