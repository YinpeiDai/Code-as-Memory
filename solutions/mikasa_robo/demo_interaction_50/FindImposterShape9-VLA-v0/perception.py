"""Perception for FindImposterShape9: fuse wrist+front RGB-D into a world top-down height grid,
segment the blue shapes, compare each with canonical shape templates, and infer the imposter."""

CELL = 0.0025
GRID_X0 = -0.40
GRID_Y0 = -0.40
GRID_N = 320
SHAPE_CLASSES = ["C", "T", "plus", "star", "gear", "cube", "sphere", "cyl", "nested"]
TEMPLATE_W_PREC = 2.0   # weight of (1 - fraction of observed cells explained by the template)
TEMPLATE_W_REC = 1.0    # weight of (1 - fraction of template cells observed)
HEIGHT_W = 0.8          # one-sided penalty for being taller than the class template
HEIGHT_SLACK = 0.004    # metres of tolerated excess height

# Canonical top-down height templates (2.5 mm cells, rows = +x, cols = +y, world-fixed orientation),
# median of aligned full views in the assigned demos. "." = empty, a..z = 1..26 mm, A..Z = 27..52 mm.
SHAPE_TEMPLATES = {
    'T': [
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '.................AAAAAA.....',
        '.................AAAAAA.....',
        '.................AAAAAA.....',
        '.................AAAAAA.....',
        '................AAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '.......AAAAAAAAAAAAAAAA.....',
        '................AAAAAAA.....',
        '.................AAAAAA.....',
        '.................AAAAAA.....',
        '.................AAAAAA.....',
        '.................AAAAAy.....',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
    ],
    'sphere': [
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '.............ABBz...........',
        '...........zCEFFFEC.........',
        '..........BEGHIIHHGEA.......',
        '.........BEGIJKKJJIGE.......',
        '.........DGIKLLLLKJIHC......',
        '........CFIKLMMMMMKJHF......',
        '........DGJKMMNNNMLKIGC.....',
        '........EIJLMNNNNNMLJHC.....',
        '.......AEHKLMNNNNNMLJGC.....',
        '........DHJLMMNNNMMKJHC.....',
        '........DGJKLMMMMMLKIGC.....',
        '........AEIJKLLMLLKJHE......',
        '.........CFIJKKKKJIHFC......',
        '.........CDFHIIIIIHFD.......',
        '..........BCEFFGFEEC........',
        '............BCCDBBA.........',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
    ],
    'nested': [
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '.......xyyyyy.yy.y.yxx......',
        '.......yyyyyyyyyyyyyyyy.....',
        '.......yyBHHHIHIIHDzzy......',
        '.......yCHIIIIIIIIIHHy......',
        '.......yzHIIIIJJIIIHzy......',
        '.......yyIJRSSSSRRJHGy......',
        '.......yyHJRSSSSSRQHzy......',
        '.......yyHJRSSSSSRJHzy......',
        '.......yyHJRSSSSSRJHFx......',
        '.......yyHJRSSSSSSJIHy......',
        '.......yyHJRSSSSSRRIGy......',
        '.......yzHIORSRSRPJIHy......',
        '.......yzHIIIIIIIIIHHy......',
        '.......yzHIIIIIIIIIHHyy.....',
        '.......yyzCCDBDCDCAzyyy.....',
        '.......yyyyyyyyyyyyyyyy.....',
        '.......xxxxxxxxxyxyxyx......',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
    ],
    'cyl': [
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '...........NNOOOON..........',
        '.........NOOOOOOOOON........',
        '........NOOOOOOOOOOON.......',
        '.......NOOOOOOOOOOONNM......',
        '.......NOOOOOOOOOOOONN......',
        '......NOOOOOOOOOOOONNN......',
        '......NOOOOOOOONOOONNNM.....',
        '......NONONOOOOOOOONONN.....',
        '......NOOOOOOOOOOOOOOON.....',
        '......NOOOOOOOOOOOOOON......',
        '.......OOOOOOOOOOOOOON......',
        '.......NOOOOOOOOOOOONN......',
        '........OOOOONOOOOOON.......',
        '........MNOOOOOOOOONL.......',
        '.........INNOOOOOONI........',
        '...........KNNNNNK..........',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
    ],
    'C': [
        '............................',
        '............................',
        '............................',
        '........wwwwwww.............',
        '.......wwwwwwwwwwv..........',
        '......wwwwwwwwwwwww.........',
        '......wwwwwwwwwwwwww........',
        '......wwwwwwwwwwwwwww.......',
        '.......wwwwwwwwwwwwwww......',
        '.......wwwwwwwwwwwwwww......',
        '........vwwwwwwwwwwwwww.....',
        '.............vwwwwwwwww.....',
        '..............wwwwwwwww.....',
        '...............wwwwwwww.....',
        '...............wwwwwwwwv....',
        '...............wwwwwwww.....',
        '...............wwwwwwww.....',
        '..............wwwwwwwww.....',
        '............wwwwwwwwwww.....',
        '..........wwwwwwwwwwww......',
        '.........wwwwwwwwwwwww......',
        '.........wwwwwwwwwwww.......',
        '........wwwwwwwwwwwww.......',
        '.........wwwwwwwwwww........',
        '.........wwwwwwwww..........',
        '..........wwwwwww...........',
        '...........vwv..............',
        '............................',
    ],
    'cube': [
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
        '.......NNNNNNNNNNNNNNNN.....',
        '......NNOOONNNONOOOOONN.....',
        '......NOONOOOOOOOONOOON.....',
        '......NONOOONNOOOONONON.....',
        '......NONOOOOOOONNOONON.....',
        '......NOOOOOOOOOONOOOON.....',
        '......OOOOOOOOOOOOOOOON.....',
        '......NOOOOOOOOOOOOOOON.....',
        '......OOOOOOOOOOOOOOOON.....',
        '......NOOOOOOOOOOOONOON.....',
        '......NOOOOOOOOOOOOOOON.....',
        '......NOOOOOOOOOOOOOOON.....',
        '......NNOOOOOOOOONNNNNN.....',
        '......NOOOOOOOOOOOOOOON.....',
        '......NONNNOOOOOOOONOON.....',
        '......NOOOOOOOOOOOOOOON.....',
        '......LNNNNNNNNNNNNNNNM.....',
        '............................',
        '............................',
        '............................',
        '............................',
        '............................',
    ],
    'star': [
        '............................',
        '............................',
        '............................',
        '.........AA........A........',
        '........ABA.......ABAA......',
        '.......ABBBA.....ABBBBA.....',
        '.....AABABAAA....ABBBBAA....',
        '......AAAABAAA..ABABBBAA....',
        '......AABBAABAAABAAAAAA.....',
        '.......AABAAAAABAAAAAAA.....',
        '........ABBBAAABBAAAAA......',
        '........AAAABAABBBAAA.......',
        '........BAABAABBBABAB.......',
        '.......AAAABABABABBBAA......',
        '.....AABBABBBBBBBBBABAAAA...',
        '..AAABBBBBBBBABABBBAABABAAA.',
        '...ABABBBAABBABBBBBAAABBAAA.',
        '...ABABABBBBBABBBBBBBBBBAAA.',
        '...ABBABBBAAABBBBABAAABBAA..',
        '...AABAAAAAAABBAAAAAAAAABA..',
        '....AAAA...AABAAAAB...zAAA..',
        '............AAAAAA..........',
        '............ABAAAA..........',
        '............ABAAAA..........',
        '............ABAAAA..........',
        '...........AAAABBA..........',
        '............AAAAAz..........',
        '............................',
    ],
    'gear': [
        '............................',
        '............................',
        '............................',
        '............................',
        '..........ABBA.AABA.........',
        '.........BBBBBBBBBBBA.......',
        '........ABBBBBBBBBBBB.......',
        '.......BBBBBBBBBBBBBBA......',
        '......BBBBBBBBBBBBBBBBA.....',
        '....BBBBBBBBBBBBBBBBBBBBA...',
        '....BBBBBBBBBBBBBBBBBBBBA...',
        '....BBBBBBBBAAAABBBBBBBBB...',
        '....BBBBBBBAA...ABBBBBBBB...',
        '....ABBBBBBA.....ABBBBBBA...',
        '....ABBBBBA.......BBBBBBA...',
        '....BBBBBBA.......BBBBBBA...',
        '....BBBBBBAA.....ABBBBBBA...',
        '...ABBBBBBBAB...BBBBBBBBBA..',
        '....BBBBBBBBBABBBBBBBBBBBA..',
        '....ABBBBBBBBBBBBBBBBBBBA...',
        '....ABBBBBBBBBBBBBBBBBBBA...',
        '.....yABBBBBBBBBBBBBBAAz....',
        '.......ABBBBBBBBBBBBBB......',
        '........BBBBBBBBBBBBB.......',
        '........ABBBBBABBBBBA.......',
        '.........yAAAABAAAAz........',
        '............................',
        '............................',
    ],
    'plus': [
        '............................',
        '............................',
        '............AAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAABAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAAA.........',
        '..AAAAAAAAAAABAAAAAAAAAAAAA.',
        '..AAAAAAAAAAAAAAAAAAAAAAAAA.',
        '..AAAAAAAAAAAAAAAAAAAAAAAAA.',
        '..AAAAAAAAAAAAAAAAAABAAAAAA.',
        '..AAAAAAAAAAAAAAAAAAAAAAAAA.',
        '..AAAAAAAAAAAAAAAAAAAABAAAA.',
        '...zAAAAAAAAAAAAAAAAAAAAAAz.',
        '..........AAAAAAAAA.........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........AAAAAAA..........',
        '...........yzzAAAy..........',
        '............................',
    ],
}


def _blue_mask(rgb):
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (b > 60) & (b > 2 * r + 20) & (b > 2 * g + 20)


def _label(mask):
    """4-connected component labels (-1 background) by iterative min-propagation."""
    h, w = mask.shape
    big = h * w + 10
    lab = np.where(mask, np.arange(h * w).reshape(h, w), big)
    for _ in range(400):
        p = lab.copy()
        p[1:, :] = np.minimum(p[1:, :], lab[:-1, :])
        p[:-1, :] = np.minimum(p[:-1, :], lab[1:, :])
        p[:, 1:] = np.minimum(p[:, 1:], lab[:, :-1])
        p[:, :-1] = np.minimum(p[:, :-1], lab[:, 1:])
        p = np.where(mask, p, big)
        if np.array_equal(p, lab):
            break
        lab = p
    out = -np.ones((h, w), dtype=np.int64)
    vals = np.unique(lab[mask])
    for i, v in enumerate(vals):
        out[lab == v] = i
    return out, len(vals)


def _backproject(u, v, depth_mm, K, E):
    z = depth_mm.astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R


def _grid(frame, cams=("wrist", "front"), cell=0.0025, x0=-0.40, y0=-0.40, n=320):
    """Top-down max-height grid (metres) of blue points from several cameras."""
    pts = []
    for cam in cams:
        rgb = getattr(frame, cam + "_rgb")
        depth = getattr(frame, cam + "_depth")
        K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
        E = getattr(frame, cam + "_extrinsic")
        m = _blue_mask(rgb) & (depth > 0)
        v, u = np.nonzero(m)
        if len(u) == 0:
            continue
        pts.append(_backproject(u.astype(np.float64), v.astype(np.float64), depth[v, u], K, E))
    grid = np.zeros((n, n))
    if not pts:
        return grid
    pw = np.concatenate(pts, 0)
    pw = pw[(pw[:, 2] > 0.0025) & (pw[:, 2] < 0.08)]
    ix = np.floor((pw[:, 0] - x0) / cell).astype(np.int64)
    iy = np.floor((pw[:, 1] - y0) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    np.maximum.at(grid, (ix[ok], iy[ok]), pw[ok, 2])
    return _fill_gaps(grid)


def _split2(ii, jj):
    """Split a merged blob into two by 2-means on cell coordinates."""
    P = np.stack([ii, jj], 1).astype(np.float64)
    a = P[np.argmax(((P - P.mean(0)) ** 2).sum(1))]
    b = P[np.argmax(((P - a) ** 2).sum(1))]
    lab = np.zeros(len(P), dtype=bool)
    for _ in range(20):
        lab = ((P - a) ** 2).sum(1) > ((P - b) ** 2).sum(1)
        if lab.all() or (~lab).all():
            break
        a2, b2 = P[~lab].mean(0), P[lab].mean(0)
        if np.allclose(a2, a) and np.allclose(b2, b):
            break
        a, b = a2, b2
    return ~lab, lab


def _objects(frame, expected=None, cell=0.0025, x0=-0.40, y0=-0.40, n=320):
    """Segment the fused height grid into objects; use the expected count to split merges / absorb fragments."""
    grid = _grid(frame, cell=cell, x0=x0, y0=y0, n=n)
    lab, k = _label(grid > 0)
    blobs = []
    for i in range(k):
        ii, jj = np.nonzero(lab == i)
        blobs.append([ii, jj])
    blobs = [b for b in blobs if len(b[0]) >= 8]
    # absorb small fragments into a near neighbour (gap <= 4 cells)
    changed = True
    while changed:
        changed = False
        blobs.sort(key=lambda b: len(b[0]))
        for s in range(len(blobs)):
            if len(blobs[s][0]) >= 100:
                break
            si, sj = blobs[s]
            best, bd = None, 4.5
            for t in range(len(blobs)):
                if t == s:
                    continue
                ti, tj = blobs[t]
                d = np.sqrt(((si[:, None] - ti[None, :]) ** 2 + (sj[:, None] - tj[None, :]) ** 2).min())
                if d < bd:
                    best, bd = t, d
            if best is not None:
                blobs[best] = [np.concatenate([blobs[best][0], si]), np.concatenate([blobs[best][1], sj])]
                blobs.pop(s)
                changed = True
                break
    blobs = [b for b in blobs if len(b[0]) >= 12]
    if expected is not None:
        for _ in range(4):
            if len(blobs) >= expected:
                break
            blobs.sort(key=lambda b: -len(b[0]))
            if len(blobs[0][0]) < 400:
                break
            ii, jj = blobs.pop(0)
            m1, m2 = _split2(ii, jj)
            blobs.append([ii[m1], jj[m1]])
            blobs.append([ii[m2], jj[m2]])
    objs = []
    for ii, jj in blobs:
        objs.append({"ii": ii, "jj": jj, "z": grid[ii, jj],
                     "cxy": np.array([x0 + (ii.mean() + 0.5) * cell, y0 + (jj.mean() + 0.5) * cell])})
    return objs, grid


def _fill_gaps(grid, need=5, rounds=2):
    """Fill empty cells surrounded by at least `need` of 8 occupied neighbours (sampling gaps)."""
    g = grid.copy()
    for _ in range(rounds):
        occ = (g > 0).astype(np.int32)
        p = np.pad(occ, 1)
        hp = np.pad(g, 1)
        cnt = np.zeros_like(occ)
        mx = np.zeros_like(g)
        n0, n1 = g.shape
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                cnt += p[1 + di:1 + di + n0, 1 + dj:1 + dj + n1]
                mx = np.maximum(mx, hp[1 + di:1 + di + n0, 1 + dj:1 + dj + n1])
        fill = (occ == 0) & (cnt >= need)
        if not fill.any():
            break
        g = np.where(fill, mx, g)
    return g


def _heightmap(obj, half=14):
    """Max-height map (2*half cells square) of one object, centred on its bounding box.
    Returns (hm, origin_xy of cell [0,0] centre, zt = 97th percentile height)."""
    ii, jj, z = obj["ii"], obj["jj"], obj["z"]
    ci = (ii.min() + ii.max()) // 2
    cj = (jj.min() + jj.max()) // 2
    n = 2 * half
    hm = np.zeros((n, n))
    a = ii - ci + half
    b = jj - cj + half
    ok = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    hm[a[ok], b[ok]] = z[ok]
    zt = float(np.percentile(z, 97))
    org = [GRID_X0 + (ci - half + 0.5) * CELL, GRID_Y0 + (cj - half + 0.5) * CELL]
    return hm, org, zt


TEMPLATE_ALPHABET = ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _decode_template(rows):
    out = np.zeros((len(rows), len(rows[0])))
    for i, r in enumerate(rows):
        for j, ch in enumerate(r):
            out[i, j] = TEMPLATE_ALPHABET.index(ch) / 1000.0
    return out


def _touch_point(hm, org):
    """Most interior occupied cell (max erosion depth; ties -> nearest the map centre) in world xy and its height."""
    occ = hm > 0.0025
    depth = np.zeros(hm.shape)
    cur = occ.copy()
    k = 0
    while cur.any() and k < 20:
        depth[cur] += 1
        p = np.pad(cur, 1)
        n0, n1 = cur.shape
        cur = cur & p[:-2, 1:-1][:n0, :n1] & p[2:, 1:-1][:n0, :n1] & p[1:-1, :-2][:n0, :n1] & p[1:-1, 2:][:n0, :n1]
        k += 1
    if not occ.any():
        return [org[0] + 14 * CELL, org[1] + 14 * CELL], 0.0, 0
    n = hm.shape[0]
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dc = (I - (n - 1) / 2.0) ** 2 + (J - (n - 1) / 2.0) ** 2
    score = depth * 1000.0 - dc
    score[~occ] = -1e9
    a, b = np.unravel_index(int(np.argmax(score)), hm.shape)
    lo_a, hi_a = max(0, a - 2), min(n, a + 3)
    lo_b, hi_b = max(0, b - 2), min(n, b + 3)
    ztop = float(hm[lo_a:hi_a, lo_b:hi_b].max())
    return [org[0] + a * CELL, org[1] + b * CELL], ztop, int(depth[a, b])


def _template_cost(hm, zt=None):
    """Cost of explaining a height map by each class template (lower = better): thickness-invariant shape
    overlap plus a one-sided height prior (an object may look lower than its class - sunk or partly hidden -
    but never clearly taller)."""
    out = {}
    nh = _norm_hm(hm)
    for c in SHAPE_CLASSES:
        tpl = _decode_template(SHAPE_TEMPLATES[c])
        pa, pb = _occ_overlap(nh, _norm_hm(tpl))
        cost = TEMPLATE_W_PREC * (1.0 - pa) + TEMPLATE_W_REC * (1.0 - pb)
        if zt is not None:
            occ = tpl > 0.0025
            zc = float(np.percentile(tpl[occ], 97))
            cost += HEIGHT_W * min(1.0, max(0.0, (zt - zc - HEIGHT_SLACK) / 0.008))
        out[c] = cost
    return out


def _describe(frame, expected):
    """Segment the blue objects of one frame and describe each: xy, height, cells, class costs, touch point."""
    objs, _ = _objects(frame, expected)
    out = []
    for o in objs:
        hm, org, zt = _heightmap(o)
        cost = _template_cost(hm, zt)
        tp, ztop, dep = _touch_point(hm, org)
        best = min(cost, key=cost.get)
        ctr = [float(org[0] + 14 * CELL), float(org[1] + 14 * CELL)]  # footprint bounding-box centre
        out.append({"xy": [float(o["cxy"][0]), float(o["cxy"][1])], "center": ctr, "zt": zt,
                    "ncell": int(len(o["ii"])),
                    "cost": cost, "best": best, "touch": [float(tp[0]), float(tp[1])], "ztop": ztop,
                    "depth": dep})
    return out


def _set1_scores(costs1):
    """For each candidate imposter class, the min cost of assigning set-1 objects to the other 8 classes."""
    sc = {}
    for miss in SHAPE_CLASSES:
        rest = [c for c in SHAPE_CLASSES if c != miss]
        if len(costs1) > len(rest):
            # more blobs than classes: drop the worst-explained blob(s)
            costs1 = sorted(costs1, key=lambda d: min(d.values()))[:len(rest)]
        sc[miss] = _assign(costs1, rest)[0]
    return sc


def _infer_imposter(scores1, costs2):
    """Joint inference: set 1 shows 8 distinct classes, set 2 shows all 9. For each set-2 object j and class c,
    total = scores1[c] + cost2[j][c] + best assignment of the other set-2 objects to the remaining classes.
    Returns the best object index, class, margin to the runner-up object and the per-object totals."""
    n = len(costs2)
    tot = []
    cls = []
    for j in range(n):
        rest = [x for k, x in enumerate(costs2) if k != j]
        best, bc = 1e18, None
        for c in SHAPE_CLASSES:
            others = [k for k in SHAPE_CLASSES if k != c]
            if len(rest) > len(others):
                rest_use = sorted(rest, key=lambda d: min(d.values()))[:len(others)]
            else:
                rest_use = rest
            v = scores1[c] + costs2[j][c] + _assign(rest_use, others)[0]
            if v < best:
                best, bc = v, c
        tot.append(best)
        cls.append(bc)
    if n == 0:
        return None, None, 0.0, []
    order = sorted(range(n), key=lambda k: tot[k])
    margin = tot[order[1]] - tot[order[0]] if n > 1 else 0.0
    return order[0], cls[order[0]], float(margin), [float(t) for t in tot]


def _norm_hm(hm, occ_thr=0.0025):
    """Height map divided by its 97th-percentile occupied height (thickness-invariant profile)."""
    occ = hm > occ_thr
    if occ.sum() < 1:
        return hm * 0.0
    zt = float(np.percentile(hm[occ], 97))
    return np.where(occ, hm / max(zt, 1e-4), 0.0)


def _occ_overlap(ha, hb, shift=6, rtol=0.22):
    """Best translation overlap between two normalised height maps (same cell size).
    A cell matches when both are occupied and normalised heights differ by < rtol.
    Returns (inter/|A|, inter/|B|) at the shift maximising the matched-cell count."""
    a = ha > 0
    b = hb > 0
    na, nb_ = float(a.sum()), float(b.sum())
    if na < 1 or nb_ < 1:
        return 0.0, 0.0
    n = ha.shape[0]
    P = shift
    hbp = np.zeros((n + 2 * P, n + 2 * P))
    hbp[P:P + n, P:P + n] = hb
    best, bi = -1.0, (0.0, 0.0)
    for di in range(-P, P + 1):
        for dj in range(-P, P + 1):
            win = hbp[P + di:P + di + n, P + dj:P + dj + n]
            ok = a & (win > 0) & (np.abs(win - ha) < rtol)
            s = float(ok.sum())
            if s > best:
                best = s
                bi = (s / na, s / nb_)
    return bi


def _assign(costs, classes):
    """Min-cost assignment of objects (list of cost dicts) to distinct classes; DP over class subsets.
    Returns (total, list of class per object)."""
    n = len(costs)
    m = len(classes)
    INF = 1e18
    dp = {0: (0.0, [])}
    for i in range(n):
        nd = {}
        for mask, (val, lab) in dp.items():
            for j in range(m):
                if mask & (1 << j):
                    continue
                nm = mask | (1 << j)
                v = val + costs[i][classes[j]]
                if nm not in nd or v < nd[nm][0]:
                    nd[nm] = (v, lab + [classes[j]])
        dp = nd
    if not dp:
        return INF, []
    best = min(dp.values(), key=lambda t: t[0])
    return best[0], best[1]


