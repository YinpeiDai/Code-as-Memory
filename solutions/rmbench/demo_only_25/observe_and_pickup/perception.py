"""Perception helpers for observe_and_pickup (head RGB-D based)."""

TABLE_Z = 0.741
SHELF_Z = 0.8766
SHELF_X = (-0.175, 0.170)
SHELF_Y = (0.085, 0.215)
GRID = 0.005
TABLE_REGION = (-0.62, 0.62, -0.48, 0.0)   # x0, x1, y0, y1 (screen stands at y~0.046)


def _cloud(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    pw = (pc - t) @ R
    return pw.reshape(H, W, 3), depth > 0


def _label(mask, max_iter=400):
    """4-connected components on a 2D bool grid by min-label propagation. Returns int labels (-1 background)."""
    H, W = mask.shape
    big = H * W + 10
    lab = np.where(mask, np.arange(H * W).reshape(H, W), big)
    for _ in range(max_iter):
        prev = lab
        p = np.pad(lab, 1, constant_values=big)
        n = np.minimum(np.minimum(p[:-2, 1:-1], p[2:, 1:-1]), np.minimum(p[1:-1, :-2], p[1:-1, 2:]))
        lab = np.where(mask, np.minimum(lab, n), big)
        # pointer jumping to accelerate
        flat = lab.reshape(-1)
        ok = flat < big
        jumped = flat.copy()
        jumped[ok] = np.minimum(flat[ok], flat[np.minimum(flat[ok], H * W - 1)])
        lab = jumped.reshape(H, W)
        if np.array_equal(lab, prev):
            break
    out = np.full((H, W), -1)
    uniq = np.unique(lab[mask])
    for i, u in enumerate(uniq):
        out[lab == u] = i
    return out


def _label_pix(mask, P, thr=0.008, max_iter=600):
    """Pixel components: 4-neighbours joined when both are in mask and their 3D distance < thr."""
    H, W = mask.shape
    big = H * W + 10
    dv = np.linalg.norm(P[1:, :] - P[:-1, :], axis=-1) < thr
    dh = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1) < thr
    up = np.zeros((H, W), dtype=bool); up[1:, :] = dv & mask[1:, :] & mask[:-1, :]
    dn = np.zeros((H, W), dtype=bool); dn[:-1, :] = dv & mask[1:, :] & mask[:-1, :]
    lf = np.zeros((H, W), dtype=bool); lf[:, 1:] = dh & mask[:, 1:] & mask[:, :-1]
    rt = np.zeros((H, W), dtype=bool); rt[:, :-1] = dh & mask[:, 1:] & mask[:, :-1]
    lab = np.where(mask, np.arange(H * W).reshape(H, W), big)
    for _ in range(max_iter):
        prev = lab
        p = np.pad(lab, 1, constant_values=big)
        n = lab
        n = np.minimum(n, np.where(up, p[:-2, 1:-1], big))
        n = np.minimum(n, np.where(dn, p[2:, 1:-1], big))
        n = np.minimum(n, np.where(lf, p[1:-1, :-2], big))
        n = np.minimum(n, np.where(rt, p[1:-1, 2:], big))
        flat = n.reshape(-1).copy()
        ok = flat < big
        flat[ok] = np.minimum(flat[ok], flat[flat[ok]])
        lab = np.where(mask, flat.reshape(H, W), big)
        if np.array_equal(lab, prev):
            break
    out = np.full((H, W), -1)
    uniq = np.unique(lab[mask])
    for i, u in enumerate(uniq):
        out[lab == u] = i
    return out


def _dilate(mask, r=1):
    m = mask.copy()
    for _ in range(r):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def _segment_table(P, valid, min_cells=12):
    """Cluster points above the table in a world xy grid. Returns list of dicts with pixel mask."""
    x0, x1, y0, y1 = TABLE_REGION
    z = P[..., 2]
    m = valid & (z > TABLE_Z + 0.006) & (z < TABLE_Z + 0.16)
    m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    gx = int(round((x1 - x0) / GRID)) + 1
    gy = int(round((y1 - y0) / GRID)) + 1
    ix = np.clip(((P[..., 0] - x0) / GRID).astype(int), 0, gx - 1)
    iy = np.clip(((P[..., 1] - y0) / GRID).astype(int), 0, gy - 1)
    occ = np.zeros((gy, gx), dtype=bool)
    occ[iy[m], ix[m]] = True
    occ_d = _dilate(occ, 1)
    lab = _label(occ_d)
    pix_lab = np.where(m, lab[iy, ix], -1)
    objs = []
    for k in range(int(lab.max()) + 1):
        cells = int((occ & (lab == k)).sum())
        if cells < min_cells:
            continue
        pm = pix_lab == k
        objs.append({"mask": pm, "cells": cells})
    return objs


def _segment_table_pix(P, valid, min_pix=40):
    x0, x1, y0, y1 = TABLE_REGION
    z = P[..., 2]
    m = valid & (z > TABLE_Z + 0.004) & (z < TABLE_Z + 0.16)
    m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    lab = _label_pix(m, P)
    objs = []
    for k in range(int(lab.max()) + 1):
        pm = lab == k
        n = int(pm.sum())
        if n < min_pix:
            continue
        objs.append({"mask": pm, "pix": n})
    return objs


def _grow_labels(lab, mask, P, thr=0.008, max_iter=300):
    """Propagate non-negative labels into unlabeled pixels of mask through 3D-adjacent neighbours."""
    H, W = mask.shape
    dv = np.zeros((H, W), dtype=bool)
    dv[1:, :] = np.linalg.norm(P[1:, :] - P[:-1, :], axis=-1) < thr
    dh = np.zeros((H, W), dtype=bool)
    dh[:, 1:] = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1) < thr
    lab = lab.copy()
    for _ in range(max_iter):
        un = mask & (lab < 0)
        if not un.any():
            break
        new = lab.copy()
        # from above neighbour (v-1)
        cand = np.full((H, W), -1)
        cand[1:, :] = np.where(dv[1:, :], lab[:-1, :], -1)
        take = un & (new < 0) & (cand >= 0)
        new[take] = cand[take]
        cand = np.full((H, W), -1)
        cand[:-1, :] = np.where(dv[1:, :], lab[1:, :], -1)
        take = un & (new < 0) & (cand >= 0)
        new[take] = cand[take]
        cand = np.full((H, W), -1)
        cand[:, 1:] = np.where(dh[:, 1:], lab[:, :-1], -1)
        take = un & (new < 0) & (cand >= 0)
        new[take] = cand[take]
        cand = np.full((H, W), -1)
        cand[:, :-1] = np.where(dh[:, 1:], lab[:, 1:], -1)
        take = un & (new < 0) & (cand >= 0)
        new[take] = cand[take]
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _segment_table_ws(P, valid, hi=0.012, lo=0.004, min_pix=40, region=None):
    x0, x1, y0, y1 = region if region is not None else TABLE_REGION
    z = P[..., 2] - TABLE_Z
    base = valid & (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1) & (z < 0.16)
    m_lo = base & (z > lo)
    m_hi = base & (z > hi)
    seeds = _label_pix(m_hi, P, thr=0.006)
    keep = np.full(int(seeds.max()) + 2, -1)
    k2 = 0
    for k in range(int(seeds.max()) + 1):
        if int((seeds == k).sum()) >= 15:
            keep[k] = k2
            k2 += 1
    seeds = np.where(seeds >= 0, keep[np.maximum(seeds, 0)], -1)
    lab = _grow_labels(seeds, m_lo, P)
    objs = []
    for k in range(k2):
        pm = lab == k
        n = int(pm.sum())
        if n < 15:
            continue
        objs.append({"mask": pm, "pix": n})
    return _merge_fragments(objs, P, min_pix=min_pix)


def _segment_shelf(P, valid):
    z = P[..., 2]
    m = valid & (z > SHELF_Z + 0.004) & (z < SHELF_Z + 0.2)
    m &= (P[..., 0] > SHELF_X[0] - 0.02) & (P[..., 0] < SHELF_X[1] + 0.02)
    m &= (P[..., 1] > SHELF_Y[0] - 0.03) & (P[..., 1] < SHELF_Y[1] + 0.03)
    return m


def _merge_fragments(objs, P, min_pix=40):
    """Merge components whose xy footprints overlap (e.g. stapler top seen over its base)."""
    x0, x1, y0, y1 = TABLE_REGION
    cellsets = []
    for o in objs:
        pts = P[o["mask"]]
        ix = ((pts[:, 0] - x0) / 0.006).astype(int)
        iy = ((pts[:, 1] - y0) / 0.006).astype(int)
        cellsets.append(set(zip(ix.tolist(), iy.tolist())))
    n = len(objs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cellsets[i], cellsets[j]
            small = a if len(a) <= len(b) else b
            other = b if small is a else a
            inter = len(small & other)
            if inter >= max(3, 0.25 * len(small)):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        m = objs[g[0]]["mask"].copy()
        for k in g[1:]:
            m |= objs[k]["mask"]
        if int(m.sum()) >= min_pix:
            out.append({"mask": m, "pix": int(m.sum())})
    return out


def _rgb_to_hsv(rgb):
    c = rgb.astype(np.float64) / 255.0
    mx = c.max(-1)
    mn = c.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([h, s, mx], -1)


def _features(rgb, P, mask, support_z):
    pts = P[mask]
    col = rgb[mask].astype(np.float64)
    z = pts[:, 2] - support_z
    height = float(np.percentile(z, 98))
    xy = pts[:, :2]
    c = xy.mean(0)
    cov = np.cov((xy - c).T) if len(xy) > 2 else np.eye(2) * 1e-6
    w, V = np.linalg.eigh(cov)
    ax = V[:, 1]
    proj1 = (xy - c) @ ax
    proj2 = (xy - c) @ V[:, 0]
    length = float(np.percentile(proj1, 98) - np.percentile(proj1, 2))
    width = float(np.percentile(proj2, 98) - np.percentile(proj2, 2))
    hsv = _rgb_to_hsv(col)
    return {
        "center": [float(c[0]), float(c[1])],
        "height": height,
        "length": length,
        "width": width,
        "axis": [float(ax[0]), float(ax[1])],
        "mean_rgb": col.mean(0).tolist(),
        "hsv": hsv,
        "rgb": col,
        "z": z,
        "n": int(mask.sum()),
    }


def _split_two(P, mask):
    """Split one merged component into two by 2-means on xy (PCA-extreme initialisation)."""
    vs, us = np.nonzero(mask)
    xy = P[vs, us, :2]
    m = xy.mean(0)
    w, V = np.linalg.eigh(np.cov((xy - m).T))
    proj = (xy - m) @ V[:, 1]
    c1 = xy[proj <= np.percentile(proj, 25)].mean(0)
    c2 = xy[proj >= np.percentile(proj, 75)].mean(0)
    a = np.zeros(len(xy), dtype=bool)
    for _ in range(15):
        a = np.linalg.norm(xy - c1, axis=1) > np.linalg.norm(xy - c2, axis=1)
        if a.all() or (~a).all():
            break
        c1 = xy[~a].mean(0)
        c2 = xy[a].mean(0)
    out = []
    for sel in (~a, a):
        mm = np.zeros(mask.shape, dtype=bool)
        mm[vs[sel], us[sel]] = True
        out.append(mm)
    return out


def _table_objects(P, valid, region=None, split_len=0.125):
    objs = _segment_table_ws(P, valid, region=region)
    out = []
    for o in objs:
        pts = P[o["mask"]][:, :2]
        if len(pts) > 60:
            c = pts.mean(0)
            w, V = np.linalg.eigh(np.cov((pts - c).T))
            q = (pts - c) @ V[:, 1]
            L = np.percentile(q, 98) - np.percentile(q, 2)
            if L > split_len:
                for mm in _split_two(P, o["mask"]):
                    if mm.sum() >= 40:
                        out.append({"mask": mm, "pix": int(mm.sum()), "split": True})
                continue
        out.append(o)
    return out
