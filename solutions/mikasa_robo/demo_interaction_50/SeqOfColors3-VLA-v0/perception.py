"""Perception helpers for SeqOfColors3: front RGB-D -> coloured cube tops and button cap (world frame)."""

PALETTE_NAMES = ["red", "darkred", "green", "blue", "magenta", "yellow", "orange", "cyan", "teal"]
PALETTE_RGB = [[231, 5, 5], [169, 5, 5], [5, 231, 5], [5, 5, 231], [231, 5, 231],
               [231, 231, 5], [231, 169, 5], [5, 231, 231], [5, 169, 169]]
CAP_RGB = [211, 136, 136]
CUBE_TOP_Z = (0.036, 0.046)   # measured cube top z ~0.0401-0.041 in all demos
COLOR_TOL = 30.0
MIN_PIX = 8
CUE_XY = [0.0, 0.0]           # cue cube appears at world (0.002, 0.000) in every demo


def _world_points(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0][2]) / K[0][0] * z
    y = (v + 0.5 - K[1][2]) / K[1][1] * z
    c = np.stack([x, y, z], -1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    w = (c - t) @ R
    return w, z > 0


def _detect(frame):
    """Return dict color -> {'xy':[x,y], 'n':pixels} for cube top faces, plus 'button' cap estimate."""
    w, ok = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    im = frame.front_rgb.astype(np.float64)
    zz = w[..., 2]
    region = ok & (w[..., 0] > -0.45) & (w[..., 0] < 0.25) & (np.abs(w[..., 1]) < 0.55)
    top = region & (zz > CUBE_TOP_Z[0]) & (zz < CUBE_TOP_Z[1])
    pal = np.asarray(PALETTE_RGB, dtype=np.float64)
    cubes = {}
    vs, us = np.nonzero(top)
    if len(vs) > 0:
        cols = im[vs, us]
        d = np.sqrt(((cols[:, None, :] - pal[None, :, :]) ** 2).sum(-1))
        k = d.argmin(1)
        dm = d.min(1)
        pts = w[vs, us]
        for i, name in enumerate(PALETTE_NAMES):
            sel = (k == i) & (dm < COLOR_TOL)
            if int(sel.sum()) < MIN_PIX:
                continue
            P = pts[sel][:, :2]
            med = np.median(P, 0)
            keep = np.sqrt(((P - med) ** 2).sum(1)) < 0.03
            if int(keep.sum()) < MIN_PIX:
                continue
            c = P[keep].mean(0)
            cubes[name] = {"xy": [float(c[0]), float(c[1])], "n": int(keep.sum())}
    cap = region & (zz > 0.05) & (zz < 0.07) & (np.sqrt(((im - np.asarray(CAP_RGB, dtype=np.float64)) ** 2).sum(-1)) < 40)
    button = None
    if int(cap.sum()) >= 15:
        P = w[cap][:, :2]
        med = np.median(P, 0)
        keep = np.sqrt(((P - med) ** 2).sum(1)) < 0.05
        c = P[keep].mean(0)
        button = {"xy": [float(c[0]), float(c[1])], "n": int(keep.sum())}
    return {"cubes": cubes, "button": button}


def _is_array(det):
    """True once the response array (several coloured cubes away from the cue spot) is visible."""
    far = [k for k, v in det["cubes"].items()
           if math.hypot(v["xy"][0] - CUE_XY[0], v["xy"][1] - CUE_XY[1]) > 0.08 and v["n"] >= 15]
    return len(far) >= 4


def _cue_color(det):
    """Dominant colour at the cue spot during the cue phase (edge pixels of a red cue can look dark red)."""
    near = [(v["n"], k) for k, v in det["cubes"].items()
            if math.hypot(v["xy"][0] - CUE_XY[0], v["xy"][1] - CUE_XY[1]) < 0.04]
    if not near:
        return None
    near.sort(reverse=True)
    n, name = near[0]
    rest = sum(m for m, _ in near[1:])
    if n >= 20 and rest < 0.35 * n:
        return name
    return None


def _array_cubes(det):
    """Cube colour -> xy for the response array; drops small clusters lying on a bigger cube (edge shading)."""
    items = sorted(det["cubes"].items(), key=lambda kv: -kv[1]["n"])
    out = {}
    for k, v in items:
        if v["n"] < 12:
            continue
        clash = False
        for k2, v2 in out.items():
            if math.hypot(v["xy"][0] - v2[0], v["xy"][1] - v2[1]) < 0.03:
                clash = True
        if not clash:
            out[k] = v["xy"]
    return out
