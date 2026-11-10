"""Perception helpers for ChainOfColors7: colour classification, cue reading, ring localisation."""

COLOR_NAMES = ["red", "darkred", "orange", "yellow", "green", "cyan", "teal", "blue", "magenta"]


def _hue_class(px):
    """Map Nx3 int RGB to hue-class ids: 0 red-hue, 1 orange, 2 yellow, 3 green, 4 cyan-hue, 5 blue, 6 magenta, -1 none."""
    px = px.astype(np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx = px.max(1)
    mn = px.min(1)
    ok = (mn < 14) & (mx > 45)
    m = np.maximum(mx, 1.0)
    rn, gn, bn = r / m, g / m, b / m
    out = np.full(len(px), -1, dtype=np.int32)
    lo = 0.25
    out[ok & (rn > 0.99) & (gn < lo) & (bn < lo)] = 0
    out[ok & (rn > 0.99) & (gn >= 0.55) & (gn < 0.86) & (bn < lo)] = 1
    out[ok & (gn >= 0.86) & (rn >= 0.86) & (bn < lo)] = 2
    out[ok & (gn > 0.99) & (rn < lo) & (bn < lo)] = 3
    out[ok & (gn >= 0.86) & (bn >= 0.86) & (rn < lo)] = 4
    out[ok & (bn > 0.99) & (rn < lo) & (gn < lo)] = 5
    out[ok & (rn >= 0.86) & (bn >= 0.86) & (gn < lo)] = 6
    return out


def _cue_label(img):
    """Classify the single cue cube shown in front of the button in the wrist image (static home pose).
    Returns (label or None, pixel_count, brightness)."""
    reg = img[60:160, 90:180].reshape(-1, 3).astype(np.int32)
    hc = _hue_class(reg)
    sel = hc >= 0
    n = int(sel.sum())
    if n < 40:
        return None, n, 0.0
    vals = hc[sel]
    counts = np.bincount(vals, minlength=7)
    h = int(np.argmax(counts))
    px = reg[sel][vals == h]
    bright = float(np.percentile(px.max(1), 75))
    if h == 0:
        lab = "red" if bright > 200 else "darkred"
    elif h == 4:
        lab = "cyan" if bright > 200 else "teal"
    else:
        lab = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"][h]
    return lab, n, bright


def _ring_visible(img):
    """True when the response-phase ring of cubes is visible in the wrist image (many saturated pixels)."""
    px = img[::2, ::2].reshape(-1, 3).astype(np.int32)
    n = int(((px.min(1) < 30) & (px.max(1) > 80)).sum())
    return n * 4 > 1500, n * 4


def _components(mask):
    """4-connected components of a boolean mask; returns list of (ys, xs) arrays."""
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while pts:
        seed = pts.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            y, x = stack.pop()
            for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1), (y + 1, x + 1), (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1)):
                if q in pts:
                    pts.remove(q)
                    stack.append(q)
                    comp.append(q)
        a = np.array(comp)
        comps.append((a[:, 0], a[:, 1]))
    return comps


def _backproject(us, vs, depth_m, K, E):
    """Pixels + optical-axis depth (m) -> world Nx3 using OpenCV K and world->camera extrinsic E (3x4)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = (us - cx) / fx * depth_m
    yc = (vs - cy) / fy * depth_m
    pc = np.stack([xc, yc, depth_m], 1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t[None, :]) @ R


def _locate_ring(frame):
    """Find the nine coloured cubes with the front camera. Returns dict label -> [x, y, z_top, npix]."""
    img = frame.front_rgb.astype(np.int32)
    dep = frame.front_depth.astype(np.float32) / 1000.0
    K = np.array(frame.front_K, dtype=np.float64)
    E = np.array(frame.front_extrinsic, dtype=np.float64)
    H, W = img.shape[0], img.shape[1]
    hc = _hue_class(img.reshape(-1, 3)).reshape(H, W)
    found = {}
    names = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]
    for h in range(7):
        comps = _components(hc == h)
        cands = []
        for ys, xs in comps:
            if len(ys) < 4:
                continue
            d = dep[ys, xs]
            good = d > 0.05
            if good.sum() < 3:
                continue
            P = _backproject(xs[good].astype(np.float64), ys[good].astype(np.float64), d[good].astype(np.float64), K, E)
            px = img[ys[good], xs[good]]
            bright = float(np.percentile(px.max(1), 90))
            ztop = float(np.percentile(P[:, 2], 90))
            top = P[:, 2] > ztop - 0.006
            if top.sum() < 2:
                top = P[:, 2] > -1.0
            cands.append([float(np.median(P[top, 0])), float(np.median(P[top, 1])), ztop, int(len(ys)), bright])
        # merge fragments closer than 3 cm
        merged = []
        for c in sorted(cands, key=lambda c: -c[3]):
            hit = False
            for m in merged:
                if (m[0] - c[0]) ** 2 + (m[1] - c[1]) ** 2 < 0.03 ** 2:
                    m[3] += c[3]
                    m[4] = max(m[4], c[4])
                    hit = True
                    break
            if not hit:
                merged.append(list(c))
        # keep plausible tabletop cubes only
        merged = [m for m in merged if m[2] < 0.12 and m[3] >= 6]
        merged.sort(key=lambda c: -c[3])
        if h == 0 or h == 4:
            two = merged[:2]
            if len(two) == 2:
                two.sort(key=lambda c: -c[4])
                bright_name, dark_name = ("red", "darkred") if h == 0 else ("cyan", "teal")
                found[bright_name] = two[0][:4] + [two[0][4]]
                found[dark_name] = two[1][:4] + [two[1][4]]
            elif len(two) == 1:
                c = two[0]
                if h == 0:
                    lab = "red" if c[4] > 185 else "darkred"
                else:
                    lab = "cyan" if c[4] > 185 else "teal"
                found[lab] = c[:4] + [c[4]]
        else:
            if merged:
                c = merged[0]
                found[names[h]] = c[:4] + [c[4]]
    return found


def _locate_button(frame):
    """Pink button cap from the front camera: returns [x, y, z_top, npix] or None."""
    img = frame.front_rgb.astype(np.int32)
    dep = frame.front_depth.astype(np.float32) / 1000.0
    K = np.array(frame.front_K, dtype=np.float64)
    E = np.array(frame.front_extrinsic, dtype=np.float64)
    pink = (np.abs(img - np.array([211, 136, 136])).sum(-1) < 45) & (dep > 0.05)
    ys, xs = np.nonzero(pink)
    if len(ys) < 5:
        return None
    P = _backproject(xs.astype(np.float64), ys.astype(np.float64), dep[ys, xs].astype(np.float64), K, E)
    zt = float(np.percentile(P[:, 2], 90))
    top = P[:, 2] > zt - 0.006
    return [float(np.median(P[top, 0])), float(np.median(P[top, 1])), zt, int(len(ys))]


def _wrist_locate(frame, label):
    """Locate the cube of colour `label` in the wrist camera (top-face pixels). Returns [x, y, z_top, npix] or None."""
    img = frame.wrist_rgb.astype(np.int32)
    dep = frame.wrist_depth.astype(np.float32) / 1000.0
    K = np.array(frame.wrist_K, dtype=np.float64)
    E = np.array(frame.wrist_extrinsic, dtype=np.float64)
    H, W = img.shape[0], img.shape[1]
    hmap = {"red": 0, "darkred": 0, "orange": 1, "yellow": 2, "green": 3, "cyan": 4, "teal": 4, "blue": 5, "magenta": 6}
    h = hmap.get(label)
    if h is None:
        return None
    hc = _hue_class(img.reshape(-1, 3)).reshape(H, W)
    best = None
    for ys, xs in _components(hc == h):
        if len(ys) < 30:
            continue
        d = dep[ys, xs]
        good = d > 0.01
        if good.sum() < 20:
            continue
        P = _backproject(xs[good].astype(np.float64), ys[good].astype(np.float64), d[good].astype(np.float64), K, E)
        zt = float(np.percentile(P[:, 2], 90))
        if zt > 0.08 or zt < 0.02:
            continue
        px = img[ys[good], xs[good]]
        bright = float(np.percentile(px.max(1), 90))
        if h == 0 or h == 4:
            want_bright = label in ("red", "cyan")
            if want_bright and bright < 195:
                continue
            if (not want_bright) and bright >= 205:
                continue
        top = P[:, 2] > zt - 0.005
        c = [float(np.median(P[top, 0])), float(np.median(P[top, 1])), zt, int(len(ys))]
        if best is None or c[3] > best[3]:
            best = c
    return best
