"""Perception helpers for RememberColor: depth-based cube segmentation in the fixed front camera."""

_REGION = (-0.35, 0.35, -0.45, 0.45)   # world x/y window searched for cubes (m)
_ZBAND = (0.008, 0.075)                # world z band of cube surfaces (cube side ~0.04 m)
_TOP_Z = 0.03                          # pixels above this are (mostly) the cube top face
_CLUSTER_R = 0.06                      # cubes are >= ~0.12 m apart; a cube spans <= 0.057 m diagonally
_MIN_PIX = 12                          # minimum pixels for a cluster to count as a cube
_PIX_OFF = 0.5                         # pixel-centre convention (u + 0.5); see scratch/cmp_est.py


def _world_points(frame):
    """Deproject every front-camera pixel to world XYZ; returns (pts Nx3, rgb Nx3, valid mask Nx)."""
    depth = np.asarray(frame.front_depth, dtype=float) / 1000.0
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=float) + _PIX_OFF, np.arange(h, dtype=float) + _PIX_OFF)
    x = (us - K[0, 2]) / K[0, 0] * depth
    y = (vs - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x.ravel(), y.ravel(), depth.ravel()], 1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    valid = depth.ravel() > 0
    rgb = np.asarray(frame.front_rgb, dtype=float).reshape(-1, 3)
    return pw, rgb, valid


def _cluster(pts):
    """Greedy xy clustering; returns list of index arrays."""
    n = len(pts)
    labels = -np.ones(n, dtype=int)
    centers = []
    for i in range(n):
        p = pts[i, :2]
        best = -1
        bd = _CLUSTER_R
        for j in range(len(centers)):
            d = math.hypot(p[0] - centers[j][0], p[1] - centers[j][1])
            if d < bd:
                bd = d
                best = j
        if best < 0:
            centers.append([p[0], p[1], 1.0])
            labels[i] = len(centers) - 1
        else:
            labels[i] = best
            c = centers[best]
            c[2] += 1.0
            c[0] += (p[0] - c[0]) / c[2]
            c[1] += (p[1] - c[1]) / c[2]
    return [np.flatnonzero(labels == j) for j in range(len(centers))]


def _detect_cubes(frame):
    """Cubes on the table seen by the front camera.

    Returns a list of dicts {xyz, rgb, chroma, npix, ntop} sorted by world y.
    xyz is the top-face centroid in x/y with z set to the cube-centre height estimate.
    """
    pw, rgb, valid = _world_points(frame)
    x0, x1, y0, y1 = _REGION
    m = (valid & (pw[:, 0] > x0) & (pw[:, 0] < x1) & (pw[:, 1] > y0) & (pw[:, 1] < y1)
         & (pw[:, 2] > _ZBAND[0]) & (pw[:, 2] < _ZBAND[1]))
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return []
    if len(idx) > 4000:   # something big (e.g. the arm) is in the band; subsample for speed
        idx = idx[:: int(len(idx) // 4000) + 1]
    pts = pw[idx]
    cols = rgb[idx]
    cubes = []
    for g in _cluster(pts):
        if len(g) < _MIN_PIX:
            continue
        p = pts[g]
        c = cols[g]
        zmax = float(np.max(p[:, 2]))
        top = p[:, 2] > max(_TOP_Z, zmax - 0.003)
        if top.sum() >= 4:
            ptop = p[top]
            ctop = c[top]
        else:
            ptop = p
            ctop = c
        span = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
        mean_rgb = np.median(ctop, 0)
        s = float(np.sum(mean_rgb)) + 1e-6
        cubes.append({
            "xyz": [0.5 * float(np.min(ptop[:, 0]) + np.max(ptop[:, 0])),
                    0.5 * float(np.min(ptop[:, 1]) + np.max(ptop[:, 1])), 0.5 * zmax],
            "xyz_mean": [float(np.mean(ptop[:, 0])), float(np.mean(ptop[:, 1]))],
            "rgb": [float(v) for v in mean_rgb],
            "chroma": [float(v) / s for v in mean_rgb],
            "npix": int(len(g)),
            "ntop": int(top.sum()),
            "zmax": zmax,
            "span": span,
        })
    cubes.sort(key=lambda d: d["xyz"][1])
    return cubes


def _color_dist(a, b):
    """Distance between two colour descriptors (chromaticity plus a weak brightness term)."""
    ca = np.asarray(a, dtype=float)
    cb = np.asarray(b, dtype=float)
    sa = float(np.sum(ca)) + 1e-6
    sb = float(np.sum(cb)) + 1e-6
    chroma = float(np.linalg.norm(ca / sa - cb / sb))
    bright = abs(sa - sb) / (sa + sb)
    return chroma + 0.1 * bright


def _match(cue_rgb, cubes):
    """Index of the cube whose colour best matches the cue, and a confidence margin."""
    if not cubes or cue_rgb is None:
        return -1, 0.0, []
    d = [_color_dist(cue_rgb, c["rgb"]) for c in cubes]
    order = sorted(range(len(d)), key=lambda i: d[i])
    best = order[0]
    margin = (d[order[1]] - d[best]) if len(order) > 1 else 1.0
    return best, float(margin), [float(v) for v in d]
