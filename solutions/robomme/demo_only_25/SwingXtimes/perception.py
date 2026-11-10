"""Perception helpers for SwingXtimes: RGB-D back-projection, colour cube, targets and button."""

CUBE_TOP_Z = 0.04


def _world_points(depth, K, E):
    """Back-project an aligned depth image (int16 mm, optical axis) to world XYZ (H,W,3) and validity mask."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u + 0.5 - K[0][2]) / K[0][0] * z
    y = (v + 0.5 - K[1][2]) / K[1][1] * z
    c = np.stack([x, y, z], -1)
    E = np.asarray(E, dtype=float)
    R = E[:, :3]
    t = E[:, 3]
    w = (c - t) @ R
    return w, depth > 0


def _components(mask, min_pixels=3):
    """4-connected components of a boolean image; list of (rows, cols) arrays, largest first."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    pts = np.argwhere(mask)
    comps = []
    for r0, c0 in pts:
        if seen[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        seen[r0, c0] = True
        rows, cols = [], []
        while stack:
            r, c = stack.pop()
            rows.append(r)
            cols.append(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if len(rows) >= min_pixels:
            comps.append((np.array(rows), np.array(cols)))
    comps.sort(key=lambda rc: -len(rc[0]))
    return comps


def _color_mask(rgb, color):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    if color == 'red':
        return (r > 90) & (g < 50) & (b < 50) & (r > 3 * (g + 5))
    if color == 'green':
        return (g > 90) & (r < 60) & (b < 60) & (g > 3 * (r + 5))
    if color == 'blue':
        return (b > 90) & (r < 60) & (g < 60) & (b > 3 * (r + 5))
    return np.zeros(r.shape, dtype=bool)


def _square_yaw(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of a square from planar points via minimal bounding-square search."""
    if len(xy) < 4:
        return 0.0
    c = xy - xy.mean(0)
    best, best_a = None, 0.0
    for k in range(90):
        a = math.radians(k) - math.pi / 4
        ca, sa = math.cos(a), math.sin(a)
        p = c[:, 0] * ca + c[:, 1] * sa
        q = -c[:, 0] * sa + c[:, 1] * ca
        s = max(p.max() - p.min(), q.max() - q.min())
        if best is None or s < best - 1e-9:
            best, best_a = s, a
    return float(best_a)


def _find_cube(w, ok, rgb, color, region=None):
    """Top face of the goal-coloured cube: dict(xy, yaw, n) or None. region=(x,y,r) limits the search."""
    z = w[..., 2]
    m = ok & _color_mask(rgb, color) & (z > CUBE_TOP_Z - 0.012) & (z < CUBE_TOP_Z + 0.02)
    m &= (w[..., 0] > -0.45) & (w[..., 0] < 0.3) & (np.abs(w[..., 1]) < 0.45)
    if region is not None:
        m &= np.hypot(w[..., 0] - region[0], w[..., 1] - region[1]) < region[2]
    comps = _components(m, 3)
    if not comps:
        return None
    rows, cols = comps[0]
    pts = w[rows, cols]
    xy = pts[:, :2]
    return {'xy': [float(v) for v in xy.mean(0)], 'z': float(np.median(pts[:, 2])),
            'yaw': _square_yaw(xy), 'n': int(len(rows))}


def _find_fixtures(w, ok, rgb):
    """Grey/white flat targets and the button. Returns dict(targets=[[x,y],...], button=[x,y] or None)."""
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    z = w[..., 2]
    grey = (np.abs(r - g) < 14) & (np.abs(g - b) < 14) & (r > 85)
    # activated targets turn red/white; include saturated red at target height
    redring = (r > 150) & (g < 80) & (b < 80) & (z < 0.02)
    m = ok & (grey | redring) & (z > 0.003) & (z < 0.05)
    m &= (w[..., 0] > -0.45) & (w[..., 0] < 0.3) & (np.abs(w[..., 1]) < 0.45)
    comps = _components(m, 6)
    blobs = []
    for rows, cols in comps:
        pts = w[rows, cols]
        blobs.append({'xy': pts[:, :2].mean(0), 'zmax': float(np.percentile(pts[:, 2], 95)),
                      'n': len(rows), 'pts': pts})
    # merge blobs closer than 4 cm (occlusion splits)
    merged = []
    for bl in blobs:
        for mg in merged:
            if np.hypot(*(mg['xy'] - bl['xy'])) < 0.04:
                allp = np.concatenate([mg['pts'], bl['pts']])
                mg['pts'] = allp
                mg['xy'] = allp[:, :2].mean(0)
                mg['zmax'] = max(mg['zmax'], bl['zmax'])
                mg['n'] += bl['n']
                break
        else:
            merged.append(dict(bl))
    buttons = [m_ for m_ in merged if m_['zmax'] > 0.022 and m_['n'] >= 10]
    flats = [m_ for m_ in merged if m_['zmax'] <= 0.022 and m_['n'] >= 10]
    button = None
    if buttons:
        bt = max(buttons, key=lambda d: d['n'])
        # the button cap (raised part) gives the press point
        p = bt['pts']
        top = p[p[:, 2] > bt['zmax'] - 0.006]
        button = [float(v) for v in (top[:, :2].mean(0) if len(top) >= 3 else bt['xy'])]
    flats.sort(key=lambda d: -d['n'])
    targets = [[float(v) for v in d['xy']] for d in flats[:2]]
    return {'targets': targets, 'button': button,
            'blobs': [[round(float(d['xy'][0]), 3), round(float(d['xy'][1]), 3), round(d['zmax'], 3), d['n']] for d in merged]}


def _parse_goal(goal):
    """Colour and number of right/left passes from the goal sentence."""
    g = (goal or '').lower()
    color = 'red'
    best = None
    for c in ('red', 'green', 'blue'):
        i = g.find(c + ' cube')
        if i < 0:
            i = g.find(c)
        if i >= 0 and (best is None or i < best):
            best, color = i, c
    words = {'one': 1, 'once': 1, 'two': 2, 'twice': 2, 'three': 3, 'thrice': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    count = 1
    i = g.find('repeat')
    if i >= 0:
        tail = g[i:].replace(',', ' ').split()
        for tok in tail[:12]:
            if tok in words:
                count = words[tok]
                break
            if tok.isdigit():
                count = int(tok)
                break
            if tok.endswith('x') and tok[:-1].isdigit():
                count = int(tok[:-1])
                break
    return {'color': color, 'count': max(1, min(count, 10))}
