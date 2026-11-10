"""Perception helpers for T07 (pour sauce over frypan twice).

All geometry is world metres. The table surface is near z=0.43 in every
demonstration; objects are found as connected regions of agentview depth
points that rise above it.
"""

TABLE_Z = 0.437          # points above this are objects on the table
OBJ_ZMAX = 0.62          # ignore points higher than this (robot arm)
GRASP_OFFSET = (-0.018, 0.001)   # expert grasp TCP minus visible-bottle median (50 demos, std ~2 mm)
GRASP_Z = 0.468          # expert TCP height at grasp (all 50 demos)
POUR_DY = 0.0635         # expert pour TCP y minus pan near-edge y (std 1.7 mm)
POUR_DX = -0.002         # expert pour TCP x minus pan body x-centre (std 1.6 mm)
DEFAULT_POUR = (-0.002, 0.222)   # mean expert pour TCP xy (fallback only)
DEFAULT_DRAINER = (-0.124, 0.54)  # mean drainer bbox centre (fallback only)


def _cloud(depth, K, E):
    z = np.asarray(depth, dtype=float) / 1000.0
    H, Wd = z.shape
    v, u = np.mgrid[0:H, 0:Wd]
    K = np.asarray(K, dtype=float)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], -1)
    E = np.asarray(E, dtype=float)
    R = E[:, :3]
    t = E[:, 3]
    return (P - t) @ R, z > 0


def _components(mask, min_size=8):
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=int)
    comps = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if lab[sy, sx] >= 0:
            continue
        cid = len(comps)
        stack = [(sy, sx)]
        lab[sy, sx] = cid
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = cid
                    stack.append((ny, nx))
        comps.append(pix)
    return [c for c in comps if len(c) >= min_size]


def _objects(frame):
    """Segment table objects from agentview RGB-D. Returns a list of dicts."""
    W, valid = _cloud(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    Z = W[..., 2]
    mask = valid & (Z > TABLE_Z) & (Z < OBJ_ZMAX) & (np.abs(W[..., 0]) < 0.45) & (np.abs(W[..., 1]) < 0.75)
    # drop anything near the gripper (arm may hang low over the table)
    tcp = np.asarray(frame.tcp, dtype=float)
    near = (np.hypot(W[..., 0] - tcp[0], W[..., 1] - tcp[1]) < 0.07) & (Z > tcp[2] - 0.10)
    mask = mask & ~near
    out = []
    for c in _components(mask):
        idx = tuple(np.array(c).T)
        P = W[idx]
        out.append({"n": len(c), "P": P, "min": P.min(0), "max": P.max(0), "med": np.median(P, 0)})
    return out


def _scene(frame):
    """Locate bottle grasp point, pan pour point and drainer centre.

    Returns dict with bottle_xy/pour_xy/drainer_xy (None when not seen) and notes.
    """
    objs = _objects(frame)
    notes = []
    bottle = None
    cands = []
    for o in objs:
        ext = o["max"][:2] - o["min"][:2]
        if 0.475 < o["max"][2] < 0.56 and ext.max() < 0.10 and o["n"] >= 25:
            cands.append(o)
    if cands:
        b = max(cands, key=lambda o: o["n"])
        bottle = [float(b["med"][0] + GRASP_OFFSET[0]), float(b["med"][1] + GRASP_OFFSET[1]), float(b["max"][2])]
        notes.append("bottle n=%d top=%.3f" % (b["n"], b["max"][2]))
    else:
        notes.append("bottle not seen")
    drainer = None
    dr = [o for o in objs if o["max"][2] > 0.56 and o["n"] > 200]
    if dr:
        d = max(dr, key=lambda o: o["n"])
        drainer = [float((d["min"][0] + d["max"][0]) / 2), float((d["min"][1] + d["max"][1]) / 2), float(d["max"][2])]
    pour = None
    low = [o for o in objs if o["max"][2] < 0.49 and o["n"] >= 15
           and not (bottle is not None and np.hypot(o["med"][0] - bottle[0], o["med"][1] - bottle[1]) < 0.06)]
    if low:
        main = max(low, key=lambda o: o["n"])
        if main["n"] >= 100:
            group = [o for o in low if np.hypot(o["med"][0] - main["med"][0], o["med"][1] - main["med"][1]) < 0.3]
            P = np.concatenate([o["P"] for o in group], 0)
            P = P[P[:, 2] < 0.49]
            y0 = float(P[:, 1].min())
            body = P[P[:, 1] < y0 + 0.10]
            xc = float((body[:, 0].min() + body[:, 0].max()) / 2)
            pour = [xc + POUR_DX, y0 + POUR_DY]
            notes.append("pan n=%d y0=%.3f xc=%.3f" % (len(P), y0, xc))
    if pour is None:
        notes.append("pan not seen")
    return {"bottle": bottle, "pour": pour, "drainer": drainer, "notes": "; ".join(notes)}


def _in_hand(frame):
    """Wrist-camera check: bottle between fingers gives ~70 mm depth below the
    camera (demo frames); an empty hand sees the table (>300 mm) or the scene."""
    d = np.asarray(frame.wrist_depth, dtype=float)[150:195, 95:160]
    med = float(np.median(d))
    f = np.asarray(frame.fingers, dtype=float)
    width = float(f[0] - f[1])
    return {"wrist_med_mm": med, "finger_width": width,
            "held": bool(med < 120 and width > 0.03)}
