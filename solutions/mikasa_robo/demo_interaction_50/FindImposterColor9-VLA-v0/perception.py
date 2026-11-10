"""Perception helpers: cube detection by depth (top faces at ~0.04 m) and palette colour."""

PALETTE = [
    ["red", 231, 5, 5],
    ["darkred", 169, 4, 4],
    ["green", 4, 231, 4],
    ["blue", 5, 5, 231],
    ["yellow", 231, 231, 4],
    ["magenta", 231, 4, 231],
    ["cyan", 4, 231, 231],
    ["teal", 4, 169, 169],
    ["orange", 231, 169, 7],
]
COLOR_TOL = 40
TOP_Z_LO = 0.033
TOP_Z_HI = 0.052
MIN_PIX = {"front": 8, "wrist": 20}
SAME_PLACE = 0.03        # m; detections closer than this are the same cube


def _world_points(depth, K, E):
    z = depth.astype(np.float64) / 1000.0
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    pw = (pc - t) @ R
    return pw, z > 0


def _expected_area(K, E, c):
    """Pixel area of a full 4 cm cube top face (z=0.04) centred at world xy c."""
    h = 0.02
    corners = np.array([[c[0] - h, c[1] - h, 0.04], [c[0] + h, c[1] - h, 0.04],
                        [c[0] + h, c[1] + h, 0.04], [c[0] - h, c[1] + h, 0.04]])
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    pc = corners @ R.T + t
    if (pc[:, 2] <= 0.01).any():
        return 0.0
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    a = 0.0
    for i in range(4):
        j = (i + 1) % 4
        a += u[i] * v[j] - u[j] * v[i]
    return abs(a) / 2.0


def _detect_cam(frame, cam):
    """Per palette colour: pixel count and median world xy of top-face pixels in one camera."""
    depth = getattr(frame, cam + "_depth")
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    rgb = getattr(frame, cam + "_rgb").astype(np.int32)
    pw, valid = _world_points(depth, K, E)
    m = valid & (pw[..., 2] > TOP_Z_LO) & (pw[..., 2] < TOP_Z_HI)
    m = m & (pw[..., 0] > -0.40) & (pw[..., 0] < 0.25) & (np.abs(pw[..., 1]) < 0.40)
    out = {}
    if not m.any():
        return out
    px = rgb[m]
    xy = pw[m][:, :2]
    for name, r, g, b in PALETTE:
        d = np.abs(px - np.array([r, g, b])).max(axis=1)
        sel = d < COLOR_TOL
        n = int(sel.sum())
        if n < MIN_PIX[cam]:
            continue
        pts = xy[sel]
        med = np.median(pts, axis=0)
        near = np.linalg.norm(pts - med, axis=1) < 0.03
        n2 = int(near.sum())
        if n2 < MIN_PIX[cam]:
            continue
        c = pts[near].mean(axis=0)
        area = _expected_area(K, E, c)
        out[name] = {"n": n2, "xy": [float(c[0]), float(c[1])],
                     "comp": float(n2 / area) if area > 1 else 0.0}
    # dedupe colours that sit at the same place (shaded side faces); keep larger count
    names = sorted(out, key=lambda k: -out[k]["n"])
    keep = {}
    for k in names:
        p = np.array(out[k]["xy"])
        clash = False
        for k2 in keep:
            if np.linalg.norm(p - np.array(keep[k2]["xy"])) < 0.022:
                clash = True
        if not clash:
            keep[k] = out[k]
    return keep


def _detect(frame):
    """Merge both cameras: {colour: {"xy": [x, y], "n_front": int, "n_wrist": int}}."""
    res = {}
    for cam in ("front", "wrist"):
        for k, v in _detect_cam(frame, cam).items():
            entry = {"xy": v["xy"], "n": v["n"], "comp": v["comp"], "src": cam}
            if k not in res:
                res[k] = entry
            else:
                # keep the estimate whose top face is most completely visible
                # (partial occlusion biases the centroid); wrist wins near-ties
                ca = min(res[k]["comp"], 1.0)
                cb = min(v["comp"], 1.0)
                if cb > ca - (0.1 if cam == "wrist" else 0.0):
                    res[k] = entry
    # Cross-camera dedupe: two colours at (nearly) the same place cannot both be cubes (4 cm cubes, centres
    # >= 5 cm apart in all 100 recorded layouts). R046: red's shaded side face (front view, comp 0.22) was read
    # as 'darkred' 1.7 cm from red -> 9 cue colours -> no imposter. Keep the more complete detection.
    order = sorted(res, key=lambda k: -min(res[k]["comp"], 1.0))
    keep = {}
    for k in order:
        p = np.array(res[k]["xy"])
        if all(np.linalg.norm(p - np.array(keep[k2]["xy"])) >= SAME_PLACE for k2 in keep):
            keep[k] = res[k]
    return keep
