"""Perception helpers: front-camera RGB-D -> world points, cube colours, container tracking."""

REGION = (-0.13, 0.17, -0.32, 0.32)   # x0, x1, y0, y1 world metres of the container area
CELL = 0.01
COLORS = ("red", "green", "blue")


def _cloud(depth, K, E):
    """World XYZ (HxWx3) of every pixel and valid-depth mask."""
    z = np.asarray(depth, dtype=np.float64) / 1000.0
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    P = np.stack([x, y, z], -1)
    E = np.asarray(E, dtype=np.float64)
    R = E[:, :3]
    t = E[:, 3]
    return (P - t) @ R, z > 0


def _frame_cloud(f):
    return _cloud(f.front_depth, np.asarray(f.front_K, dtype=np.float64), f.front_extrinsic)


def _container_points(f, tcp_exclude=True):
    """XY of points 2-12 cm above the table in the container region (robot hand near TCP removed)."""
    Wp, ok = _frame_cloud(f)
    x0, x1, y0, y1 = REGION
    m = ok & (Wp[..., 2] > 0.02) & (Wp[..., 2] < 0.12) & (Wp[..., 0] > x0) & (Wp[..., 0] < x1)
    m = m & (Wp[..., 1] > y0) & (Wp[..., 1] < y1)
    P = Wp[m]
    if tcp_exclude and len(P):
        tcp = np.asarray(f.tcp, dtype=np.float64)
        d = np.hypot(P[:, 0] - tcp[0], P[:, 1] - tcp[1])
        keep = ~((d < 0.07) & (P[:, 2] > tcp[2] - 0.06))
        P = P[keep]
    return P


def _cubes(f):
    """Mean world position of red/green/blue cube pixels (cubes visible before covering)."""
    Wp, ok = _frame_cloud(f)
    rgb = np.asarray(f.front_rgb).astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = ok & (Wp[..., 2] > 0.01) & (Wp[..., 2] < 0.08)
    masks = {"red": (r > 120) & (g < 50) & (b < 50),
             "green": (g > 120) & (r < 90) & (b < 90),
             "blue": (b > 120) & (r < 70) & (g < 70)}
    out = {}
    for name in COLORS:
        m = masks[name] & h
        if int(m.sum()) > 5:
            out[name] = [float(v) for v in Wp[m].mean(0)]
    return out


def _components(P, min_pts=60):
    """Connected blobs of an XY occupancy grid; returns list of [x, y, npts]."""
    if len(P) == 0:
        return []
    x0, x1, y0, y1 = REGION
    nx = int(round((x1 - x0) / CELL)) + 1
    ny = int(round((y1 - y0) / CELL)) + 1
    ix = np.clip(((P[:, 0] - x0) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / CELL).astype(int), 0, ny - 1)
    G = np.zeros((nx, ny), dtype=np.int32)
    np.add.at(G, (ix, iy), 1)
    occ = G >= 2
    lab = -np.ones((nx, ny), dtype=np.int32)
    blobs = []
    cells = np.argwhere(occ)
    for c in cells:
        a, b = int(c[0]), int(c[1])
        if lab[a, b] >= 0:
            continue
        k = len(blobs)
        stack = [(a, b)]
        lab[a, b] = k
        members = []
        while stack:
            p, q = stack.pop()
            members.append((p, q))
            for dp in (-1, 0, 1):
                for dq in (-1, 0, 1):
                    pp, qq = p + dp, q + dq
                    if 0 <= pp < nx and 0 <= qq < ny and occ[pp, qq] and lab[pp, qq] < 0:
                        lab[pp, qq] = k
                        stack.append((pp, qq))
        blobs.append(members)
    out = []
    cid = lab[ix, iy]
    for k in range(len(blobs)):
        sel = cid == k
        n = int(sel.sum())
        if n >= min_pts:
            out.append([float(P[sel, 0].mean()), float(P[sel, 1].mean()), n])
    return out


def _kmeans_update(P, C, it=4):
    C = np.array(C, dtype=np.float64)
    if len(P) == 0:
        return C
    for _ in range(it):
        d = ((P[:, None, :2] - C[None, :, :2]) ** 2).sum(-1)
        a = d.argmin(1)
        dist = np.sqrt(d[np.arange(len(P)), a])
        for k in range(len(C)):
            sel = (a == k) & (dist < 0.06)
            if int(sel.sum()) > 15:
                C[k] = P[sel, :2].mean(0)
    return C


def _container_top(f, center, radius=0.045):
    """Centre XY and yaw (mod pi/2) of the container top near `center`, from points above 6 cm."""
    Wp, ok = _frame_cloud(f)
    m = ok & (Wp[..., 2] > 0.05) & (Wp[..., 2] < 0.12)
    P = Wp[m]
    if len(P) == 0:
        return None
    d = np.hypot(P[:, 0] - center[0], P[:, 1] - center[1])
    P = P[d < radius]
    if len(P) < 20:
        return None
    ztop = float(np.percentile(P[:, 2], 90))
    T = P[P[:, 2] > ztop - 0.012]
    if len(T) < 15:
        T = P
    best = None
    for k in range(45):
        a = k * (math.pi / 2) / 45
        c, s = math.cos(a), math.sin(a)
        u = T[:, 0] * c + T[:, 1] * s
        v = -T[:, 0] * s + T[:, 1] * c
        lo_u, hi_u = np.percentile(u, [3, 97])
        lo_v, hi_v = np.percentile(v, [3, 97])
        area = (hi_u - lo_u) * (hi_v - lo_v)
        if best is None or area < best[0]:
            cu, cv = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2
            best = (area, a, cu * c - cv * s, cu * s + cv * c, hi_u - lo_u, hi_v - lo_v)
    return {"x": float(best[2]), "y": float(best[3]), "yaw": float(best[1]), "ztop": ztop,
            "size": [float(best[4]), float(best[5])], "n": int(len(T))}


BUTTON_REGION = (-0.34, -0.12, -0.3, 0.3)


def _buttons(f):
    """XY of the button bodies (low objects behind the containers), sorted by descending y."""
    Wp, ok = _frame_cloud(f)
    x0, x1, y0, y1 = BUTTON_REGION
    m = ok & (Wp[..., 2] > 0.004) & (Wp[..., 2] < 0.06) & (Wp[..., 0] > x0) & (Wp[..., 0] < x1)
    m = m & (Wp[..., 1] > y0) & (Wp[..., 1] < y1)
    P = Wp[m]
    out = []
    while len(P) > 8 and len(out) < 4:
        # grow a blob around the highest-density seed
        H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=[np.arange(x0, x1 + 0.01, 0.02), np.arange(y0, y1 + 0.01, 0.02)])
        i, j = np.unravel_index(int(H.argmax()), H.shape)
        if H[i, j] < 6:
            break
        c = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
        for _ in range(4):
            d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
            sel = d < 0.035
            c = P[sel, :2].mean(0)
        d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
        sel = d < 0.035
        top = float(np.percentile(P[sel, 2], 95))
        out.append([float(c[0]), float(c[1]), top, int(sel.sum())])
        P = P[d >= 0.05]
    out = [b for b in out if b[3] >= 15]
    out.sort(key=lambda b: -b[1])
    return out
