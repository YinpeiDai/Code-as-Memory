"""Perception helpers for GatherAndRecall3 (front camera, numpy only)."""

COLORS = ("red", "green", "blue")

# Empirical offsets (metres, world x/y) between the perceived object centre and the
# TCP position the expert demonstrations used (measured over 50 assigned demos).
GRASP_OFFSET = (-0.011, 0.003)
BUTTON_OFFSET = (-0.008, 0.001)
# Slot targets for the TCP relative to the perceived disc centre (demo means).
SLOTS = ((-0.022, 0.072), (-0.086, -0.039), (0.045, -0.038))


def _deproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    pw = (pc - t) @ R
    return pw, z > 0


def _color_masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return {
        "red": (r > 120) & (g < 60) & (b < 60),
        "green": (g > 120) & (r < 80) & (b < 80),
        "blue": (b > 120) & (r < 60) & (g < 60),
    }


def _cluster(P, radius):
    """Greedy xy clustering; returns list of index arrays (largest first)."""
    n = len(P)
    left = np.ones(n, dtype=bool)
    groups = []
    while left.any():
        i = int(np.flatnonzero(left)[0])
        c = P[i, :2]
        mem = None
        for _ in range(4):
            d = np.linalg.norm(P[:, :2] - c, axis=1)
            mem = left & (d < radius)
            if not mem.any():
                mem = np.zeros(n, dtype=bool)
                mem[i] = True
                break
            c = P[mem, :2].mean(0)
        left &= ~mem
        left[i] = False
        groups.append(np.flatnonzero(mem))
    groups.sort(key=lambda g: -len(g))
    return groups


def _scene(frame):
    """Detect disc, cubes, buttons and lamp in a front-camera frame."""
    rgb = frame.front_rgb
    pw, ok = _deproject(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgbi = rgb.astype(np.int32)
    sat = rgbi.max(-1) - rgbi.min(-1)
    br = rgbi.mean(-1)
    out = {"cubes": {}, "buttons": {}, "disc": None, "disc_r": None, "lamp_box": None}
    # disc: bright low-saturation flat region on the table
    dm = ok & (sat < 40) & (br > 150) & (pw[..., 2] < 0.012) & (pw[..., 2] > -0.01)
    dm &= (pw[..., 0] > -0.3) & (pw[..., 0] < 0.4) & (np.abs(pw[..., 1]) < 0.6)
    P = pw[dm]
    if len(P) > 200:
        groups = _cluster(P, 0.2)
        Q = P[groups[0]]
        c = Q[:, :2].mean(0)
        out["disc"] = [float(c[0]), float(c[1])]
        out["disc_r"] = float(np.percentile(np.linalg.norm(Q[:, :2] - c, axis=1), 99))
    # coloured blobs: cubes (top below 5 cm) and buttons (cap above 5 cm)
    for col, m in _color_masks(rgb).items():
        m = m & ok & (pw[..., 2] > -0.01) & (pw[..., 2] < 0.3)
        P = pw[m]
        if len(P) < 5:
            continue
        for g in _cluster(P, 0.035):
            if len(g) < 8:
                continue
            Q = P[g]
            ztop = float(Q[:, 2].max())
            if 0.05 < ztop < 0.08 and float(Q[:, 2].min()) > 0.03:
                if col not in out["buttons"]:
                    c = Q.mean(0)
                    out["buttons"][col] = [float(c[0]), float(c[1]), ztop]
            elif ztop <= 0.05:
                if col not in out["cubes"]:
                    cx = (Q[:, 0].min() + Q[:, 0].max()) / 2.0
                    cy = (Q[:, 1].min() + Q[:, 1].max()) / 2.0
                    out["cubes"][col] = [float(cx), float(cy), ztop, int(len(g))]
    # lamp: bright low-saturation elevated blob in front of the work area
    lm = ok & (sat < 40) & (br > 150) & (pw[..., 2] > 0.03) & (pw[..., 2] < 0.2) & (pw[..., 0] > 0.1)
    rows, cols = np.nonzero(lm)
    if len(rows) > 30:
        P = pw[lm]
        g = _cluster(P, 0.06)[0]
        rr, cc = rows[g], cols[g]
        out["lamp_box"] = [int(max(rr.min() - 2, 0)), int(min(rr.max() + 3, 256)),
                           int(max(cc.min() - 2, 0)), int(min(cc.max() + 3, 256)), int(len(g))]
        c = P[g].mean(0)
        out["lamp"] = [float(c[0]), float(c[1]), float(c[2])]
    return out


def _lamp_color(rgb, box):
    """Return (color or None, count) of saturated lamp pixels in the stored box."""
    r0, r1, c0, c1, n = box
    crop = rgb[r0:r1, c0:c1].astype(np.int32)
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    counts = {
        "red": int(((r > 150) & (g < 90) & (b < 90)).sum()),
        "green": int(((g > 150) & (r < 90) & (b < 90)).sum()),
        "blue": int(((b > 150) & (r < 90) & (g < 90)).sum()),
    }
    best = max(counts, key=lambda k: counts[k])
    if counts[best] >= max(15, 0.3 * n):
        return best, counts[best]
    return None, counts[best]
