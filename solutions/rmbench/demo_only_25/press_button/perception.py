"""Perception helpers for press_button (head camera digit reading, button localisation)."""

DIGIT_ROI = (50, 150, 60, 240)   # rows r0,r1 / cols c0,c1 of the 240x320 head image holding both tiles
FEAT_H = 20
FEAT_W = 16


def _components(mask):
    """4-connected components of a boolean mask -> list of pixel lists."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not lab[r, c]:
                stack = [(r, c)]
                lab[r, c] = True
                pts = []
                while stack:
                    y, x = stack.pop()
                    pts.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy = y + dy
                        xx = x + dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = True
                            stack.append((yy, xx))
                comps.append(pts)
    return comps


def _tile_boxes(rgb):
    """Two beige number tiles -> [(r0,r1,c0,c1) left, (...) right] in image coords, or None."""
    r0, r1, c0, c1 = DIGIT_ROI
    im = np.asarray(rgb)[r0:r1, c0:c1].astype(float)
    R = im[..., 0]
    B = im[..., 2]
    beige = (R - B > 25) & (R > 120)
    cols = beige.sum(0)
    on = np.flatnonzero(cols >= 3)
    if len(on) < 20:
        return None
    gaps = np.diff(on)
    k = int(np.argmax(gaps))
    if gaps[k] < 3:
        return None
    boxes = []
    for g in (on[:k + 1], on[k + 1:]):
        if len(g) < 15:
            return None
        sub = beige[:, int(g.min()):int(g.max()) + 1]
        rows = np.flatnonzero(sub.sum(1) >= 3)
        if len(rows) < 15:
            return None
        boxes.append((int(rows.min()) + r0, int(rows.max()) + r0 + 1, int(g.min()) + c0, int(g.max()) + c0 + 1))
    return boxes


def _digit_mask(rgb, box, thr=110):
    """Dark glyph pixels inside a tile, excluding the tile's printed frame line."""
    r0, r1, c0, c1 = box
    im = np.asarray(rgb)[r0:r1, c0:c1].astype(float)
    gray = im.mean(2)
    dark = gray < thr
    H, W = dark.shape
    m = np.zeros((H, W), dtype=bool)
    for pts in _components(dark):
        if len(pts) < 8:
            continue
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        h = max(ys) - min(ys) + 1
        w = max(xs) - min(xs) + 1
        cy = (max(ys) + min(ys)) / 2.0
        cx = (max(xs) + min(xs)) / 2.0
        if h > 0.85 * H or w > 0.75 * W:
            continue
        if h < 0.3 * H:
            continue
        if abs(cx - W / 2.0) > 0.3 * W or abs(cy - H / 2.0) > 0.3 * H:
            continue
        for (y, x) in pts:
            m[y, x] = True
    return m, gray


def _raw_feature(rgb, box):
    """Height-normalised, blurred glyph darkness map (FEAT_H x FEAT_W), or None."""
    m, gray = _digit_mask(rgb, box)
    ys, xs = np.nonzero(m)
    if len(ys) < 10:
        return None
    y0 = int(ys.min()); y1 = int(ys.max()) + 1
    x0 = int(xs.min()); x1 = int(xs.max()) + 1
    h = y1 - y0
    dk = np.clip((200.0 - gray) / 150.0, 0, 1) * m
    s = FEAT_H / float(h)
    F = np.zeros((FEAT_H, FEAT_W))
    cx = (x0 + x1) / 2.0
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            v = dk[yy, xx]
            if v <= 0:
                continue
            iy = int((yy + 0.5 - y0) * s)
            ix = int((xx + 0.5 - cx) * s + FEAT_W / 2.0)
            if 0 <= iy < FEAT_H and 0 <= ix < FEAT_W:
                F[iy, ix] += v
    P = np.pad(F, 1)
    Bm = np.zeros((FEAT_H, FEAT_W))
    for dy in range(3):
        for dx in range(3):
            Bm = Bm + P[dy:dy + FEAT_H, dx:dx + FEAT_W]
    return Bm / 9.0


def _normalise(v):
    v = np.asarray(v, dtype=float).flatten()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _decode_template(s):
    return np.array([int(ch) for ch in s], dtype=float)


def _classify(feat):
    """Nearest-template digit -> (digit, best score, margin to the next digit)."""
    q = _normalise(feat)
    best = {}
    for digit, side, code in DIGIT_TEMPLATES:
        sc = float(q @ _normalise(_decode_template(code)))
        if sc > best.get(digit, -9.0):
            best[digit] = sc
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    return int(ranked[0][0]), float(ranked[0][1]), float(ranked[0][1] - second)


def _read_numbers(rgb):
    """Read (left_number, right_number, info) from a head RGB frame; numbers None if unreadable."""
    boxes = _tile_boxes(rgb)
    if boxes is None:
        return None, None, "tiles not found"
    out = []
    info = []
    for box in boxes:
        f = _raw_feature(rgb, box)
        if f is None:
            out.append(None)
            info.append("no glyph")
            continue
        d, sc, mg = _classify(f)
        out.append(d)
        info.append("%d(score %.3f margin %.3f)" % (d, sc, mg))
    return out[0], out[1], "; ".join(info)


def _red_button_centres(rgb, depth, K, E):
    """World centres (x, y, top_z) of red blobs seen by the head camera, left to right."""
    im = np.asarray(rgb).astype(float)
    R = im[..., 0]; G = im[..., 1]; B = im[..., 2]
    m = (R > 150) & (G < 110) & (B < 110)
    m[:60, :] = False
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    order = np.argsort(xs)
    xs_s = xs[order]
    cuts = np.flatnonzero(np.diff(xs_s) > 4)
    groups = np.split(order, cuts + 1)
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    Rm = E[:, :3]
    t = E[:, 3]
    dep = np.asarray(depth)
    out = []
    for g in groups:
        if len(g) < 60:
            continue
        u = xs[g]; v = ys[g]
        d = dep[v, u].astype(float) / 1000.0
        ok = d > 0
        if ok.sum() < 30:
            continue
        u = u[ok]; v = v[ok]; d = d[ok]
        X = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 0)
        W = Rm.T @ (X - t[:, None])
        out.append([float(W[0].mean()), float(W[1].mean()), float(np.percentile(W[2], 95))])
    return out
