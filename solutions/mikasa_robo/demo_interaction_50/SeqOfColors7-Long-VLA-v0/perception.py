"""Perception helpers (front camera RGB-D -> world points, cube clusters, colour names)."""

PALETTE = {
    "red": (1.0, 0.0, 0.0),
    "dark_red": (0.73, 0.0, 0.0),
    "orange": (1.0, 0.73, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "teal": (0.0, 0.73, 0.73),
    "blue": (0.0, 0.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
}


def _world_points(depth, K, E):
    """Back-project an int16 mm optical-axis depth image to world XYZ (H,W,3) plus a valid mask."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    pw = (pc - t) @ R
    return pw.reshape(H, W, 3), depth > 0


def _colour_name(rgb):
    """Name a saturated cube colour from a mean top-face RGB (0-255)."""
    r, g, b = [float(c) for c in rgb]
    m = max(r, g, b, 1.0)
    chans = [r / m, g / m, b / m]
    bright = m > 188.0
    hi = [c > 0.5 for c in chans]
    if hi == [True, False, False]:
        return "red" if bright else "dark_red"
    if hi == [False, True, False]:
        return "green"
    if hi == [False, False, True]:
        return "blue"
    if hi == [True, False, True]:
        return "magenta"
    if hi == [False, True, True]:
        return "cyan" if bright else "teal"
    if hi == [True, True, False]:
        return "yellow" if chans[1] > 0.87 else "orange"
    return "unknown"


def _cluster(P, mask, rgb, rad=0.035, minn=20):
    """Greedy xy clustering of masked world points; returns list of cube candidates."""
    pts = P[mask]
    cols = rgb[mask].astype(np.float64)
    out = []
    rem = np.ones(len(pts), dtype=bool)
    guard = 0
    while rem.sum() >= minn and guard < 40:
        guard += 1
        i = int(np.flatnonzero(rem)[0])
        c = pts[i, :2]
        m = rem.copy()
        for _ in range(4):
            d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
            m = rem & (d < rad)
            if m.sum() == 0:
                break
            c = pts[m, :2].mean(0)
        if m.sum() == 0:
            rem[i] = False
            continue
        rem &= ~m
        if m.sum() < minn:
            continue
        zt = pts[m, 2]
        top = m & (pts[:, 2] > zt.max() - 0.005)
        tc = cols[top].mean(0)
        # xy centre from the top face (side faces bias towards the camera)
        tp = pts[top, :2].mean(0)
        out.append({"x": float(tp[0]), "y": float(tp[1]), "n": int(m.sum()),
                    "ztop": float(zt.max()), "rgb": [float(v) for v in tc],
                    "colour": _colour_name(tc)})
    return out


def _front_objects(frame, zmin=0.015, zmax=0.07):
    """Cube-like objects (top between zmin and zmax) on the table seen by the front camera,
    excluding the central button region and the robot base."""
    P, valid = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    z = P[..., 2]
    m = valid & (z > zmin) & (z < zmax) & (P[..., 0] > -0.45)
    m &= np.hypot(P[..., 0] + 0.11, P[..., 1]) > 0.075
    return _cluster(P, m, frame.front_rgb)


def _cue_patch(frame):
    """Colour of whatever cube sits on the cue spot in front of the button (or None)."""
    P, valid = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    z = P[..., 2]
    m = valid & (z > 0.015) & (z < 0.07) & (np.abs(P[..., 0] - 0.005) < 0.03) & (np.abs(P[..., 1]) < 0.03)
    if m.sum() < 15:
        return None
    zt = P[..., 2][m]
    top = m & (z > zt.max() - 0.005)
    rgb = frame.front_rgb[top].astype(np.float64).mean(0)
    return _colour_name(rgb)


def _project(frame, pt):
    """Project a world point into the front image; returns (u, v) pixel floats."""
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    K = np.asarray(frame.front_K, dtype=np.float64)
    pc = E[:, :3] @ np.asarray(pt, dtype=np.float64) + E[:, 3]
    return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]


CUE_SPOT = (0.005, 0.0)


def _cue_colour(frame):
    """Colour name of the cue cube on the spot in front of the button, from a small front-camera
    window only (cheap enough to run on every frame). None when the spot is empty."""
    u, v = _project(frame, (CUE_SPOT[0], CUE_SPOT[1], 0.03))
    u0, v0 = int(max(0, u - 22)), int(max(0, v - 22))
    u1, v1 = int(min(255, u + 22)), int(min(255, v + 22))
    d = np.asarray(frame.front_depth)[v0:v1, u0:u1]
    rgb = np.asarray(frame.front_rgb)[v0:v1, u0:u1]
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    z = d.astype(np.float64) / 1000.0
    pc = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], -1).reshape(-1, 3)
    P = ((pc - E[:, 3]) @ E[:, :3]).reshape(d.shape + (3,))
    m = (d > 0) & (P[..., 2] > 0.015) & (P[..., 2] < 0.07)
    m &= (np.abs(P[..., 0] - CUE_SPOT[0]) < 0.03) & (np.abs(P[..., 1] - CUE_SPOT[1]) < 0.03)
    if m.sum() < 15:
        return None
    zt = P[..., 2][m].max()
    top = m & (P[..., 2] > zt - 0.005)
    name = _colour_name(rgb[top].astype(np.float64).mean(0))
    return None if name == "unknown" else name


def _arc_cubes(frame):
    """Coloured cubes on the table (>= 7 only once the response arc has appeared; the cue phase shows one)."""
    objs = _front_objects(frame)
    return [o for o in objs if o["colour"] != "unknown"]


def _button_centre(frame):
    """Pink button cap centre from the front camera (world xy), or None."""
    P, valid = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgb = np.asarray(frame.front_rgb).astype(np.int32)
    m = valid & (P[..., 2] > 0.04) & (P[..., 2] < 0.075)
    m &= np.hypot(P[..., 0] + 0.11, P[..., 1]) < 0.08
    m &= (rgb[..., 0] - rgb[..., 1]) > 35
    if m.sum() < 20:
        return None
    return [float(P[..., 0][m].mean()), float(P[..., 1][m].mean())]
