"""Perception helpers for GatherAndRecall5: world back-projection of the front camera and
colour/height segmentation of cubes, disc, buttons and lamp."""

COLORS = ["red", "green", "blue"]


def _world_points(frame, cam="front"):
    """Back-project a camera depth image to world XYZ (H,W,3) and a valid mask."""
    depth = np.asarray(getattr(frame, cam + "_depth")).astype(np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    return pw, depth > 0


def _project(frame, pts, cam="front"):
    """Project world points (N,3) to pixel (u, v)."""
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pc = pts @ E[:, :3].T + E[:, 3]
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return u, v


def _color_masks(rgb):
    """Saturated red/green/blue masks."""
    rgb = rgb.astype(np.int32)
    out = []
    for c in range(3):
        others = [k for k in range(3) if k != c]
        m = (rgb[..., c] > 120) & (rgb[..., c] - rgb[..., others].max(-1) > 80)
        out.append(m)
    return out


def _cluster(points, radius):
    """Greedy xy clustering; returns list of index arrays."""
    n = len(points)
    if n == 0:
        return []
    left = np.ones(n, dtype=bool)
    groups = []
    xy = points[:, :2]
    while left.any():
        i = int(np.flatnonzero(left)[0])
        c = xy[i]
        members = left & (np.linalg.norm(xy - c, axis=1) < radius)
        for _ in range(4):
            c = xy[members].mean(0)
            members = left & (np.linalg.norm(xy - c, axis=1) < radius)
        if not members.any():
            members = np.zeros(n, dtype=bool)
            members[i] = True
        groups.append(np.flatnonzero(members))
        left &= ~members
    return groups


def _cube_yaw(xy):
    """Yaw (mod 90 deg, in [-45, 45) deg) of a square from its top-face points via min-extent search."""
    if len(xy) < 6:
        return 0.0
    p = xy - xy.mean(0)
    best, best_a = None, 0.0
    for a in np.linspace(-math.pi / 4, math.pi / 4, 31)[:-1]:
        ca, sa = math.cos(a), math.sin(a)
        u = p[:, 0] * ca + p[:, 1] * sa
        w = -p[:, 0] * sa + p[:, 1] * ca
        ext = (u.max() - u.min()) * (w.max() - w.min())
        if best is None or ext < best:
            best, best_a = ext, a
    return float(best_a)


def _scene(frame, known_buttons=None):
    """Detect disc, buttons, lamp and cubes in the front camera of one frame.
    known_buttons (colour -> [x, y, z]) from an earlier frame also exclude button pixels from cubes."""
    pw, ok = _world_points(frame, "front")
    rgb = np.asarray(frame.front_rgb)
    X, Y, Z = pw[..., 0], pw[..., 1], pw[..., 2]
    table = ok & (X > -0.35) & (X < 0.45) & (np.abs(Y) < 0.6)
    masks = _color_masks(rgb)
    ri = rgb.astype(np.int32)
    out = {"cubes": [], "buttons": {}, "disc": None, "lamp": None}
    # disc: light grey, flat, near table height
    grey = table & (np.abs(ri[..., 0] - ri[..., 2]) < 20) & (np.abs(ri[..., 0] - ri[..., 1]) < 20) \
        & (ri.min(-1) > 140) & (Z > -0.01) & (Z < 0.025) & (X < 0.25)
    P = pw[grey]
    if len(P) > 100:
        xy = P[:, :2]
        c = np.median(xy, 0)
        for _ in range(5):
            r = np.linalg.norm(xy - c, axis=1)
            keep = r < 0.16
            if keep.sum() < 50:
                break
            # algebraic circle fit
            q = xy[keep]
            A = np.stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))], 1)
            b = (q ** 2).sum(1)
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            c = sol[:2]
            # filled-disc fit: mean r^2 = R^2 / 2
            rad = math.sqrt(2.0 * max(sol[2] + c @ c, 1e-6))
        out["disc"] = [float(c[0]), float(c[1]), float(rad), float(np.median(P[:, 2]))]
    # buttons: coloured caps at z ~ 0.06, x < -0.1
    for ci, name in enumerate(COLORS):
        m = table & masks[ci] & (Z > 0.052) & (Z < 0.075)
        Pb = pw[m]
        if len(Pb) >= 5:
            out["buttons"][name] = [float(np.median(Pb[:, 0])), float(np.median(Pb[:, 1])), float(np.max(Pb[:, 2]))]
    # lamp: bright white blob above table, far from robot base
    white = ok & (ri.min(-1) > 190) & (Z > 0.03) & (Z < 0.2) & (X > 0.12) & (X < 0.5)
    vs, us = np.nonzero(white)
    if len(vs) >= 20:
        out["lamp"] = [int(vs.min()), int(vs.max()), int(us.min()), int(us.max())]
        out["lamp_xyz"] = [float(v) for v in pw[white].mean(0)]
    # cubes: coloured tops at z ~ 0.04 (also lower side faces)
    for ci, name in enumerate(COLORS):
        m = table & masks[ci] & (Z > 0.028) & (Z < 0.052)
        Pc = pw[m]
        if len(Pc) < 4:
            continue
        for g in _cluster(Pc, 0.03):
            if len(g) < 8:
                continue
            q = Pc[g]
            near_button = False
            blist = list(out["buttons"].values()) + list((known_buttons or {}).values())
            for bxyz in blist:
                if math.hypot(q[:, 0].mean() - bxyz[0], q[:, 1].mean() - bxyz[1]) < 0.045:
                    near_button = True
            if near_button:
                continue
            top = q[q[:, 2] > q[:, 2].max() - 0.006]
            cx, cy = top[:, :2].mean(0)
            out["cubes"].append({"color": name, "x": float(cx), "y": float(cy), "z": float(top[:, 2].mean()),
                                 "yaw": _cube_yaw(top[:, :2]), "n": int(len(g))})
    return out
