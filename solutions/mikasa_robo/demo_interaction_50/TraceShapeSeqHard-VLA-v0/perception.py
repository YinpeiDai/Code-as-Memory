"""Perception helpers for TraceShapeSeqHard: colour masks, RGB-D back-projection and shape fitting."""

SHAPE_PERIOD = 64  # frames the red cue cube needs for one shape loop (measured in all 50 demos)


def _backproject(depth, K, ext, ys, xs):
    z = depth[ys, xs].astype(float) / 1000.0
    ok = z > 0
    ys = ys[ok]
    xs = xs[ok]
    z = z[ok]
    x = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(ext)[:, :3]
    t = np.asarray(ext)[:, 3]
    return (pc - t) @ R


def _mask(im, kind):
    im = im.astype(int)
    r = im[..., 0]
    g = im[..., 1]
    b = im[..., 2]
    if kind == 'red':  # cue cube (216,110,110) and its shaded faces
        ratio = g / np.maximum(r, 1)
        return (r > 100) & (np.abs(g - b) < 6) & (ratio > 0.44) & (ratio < 0.57)
    if kind == 'green':  # graspable cube (110,216,110)
        ratio = r / np.maximum(g, 1)
        return (g > 100) & (np.abs(r - b) < 6) & (ratio > 0.44) & (ratio < 0.57)
    if kind == 'button':  # pink cap (211,136,136)
        ratio = g / np.maximum(r, 1)
        return (r > 120) & (np.abs(g - b) < 6) & (ratio > 0.6) & (ratio < 0.7)
    if kind == 'lamp_green':
        return (g > 200) & (r < 40) & (b < 40)
    if kind == 'lamp_red':
        return (r > 200) & (g < 40) & (b < 40)
    return np.zeros(r.shape, bool)


def _locate_top(frame, cam, kind, min_px=5):
    """World XYZ of the top face centre of a coloured object seen by camera `cam`, and pixel count."""
    im = getattr(frame, cam + '_rgb')
    d = getattr(frame, cam + '_depth')
    K = getattr(frame, cam + '_K')
    ext = getattr(frame, cam + '_extrinsic')
    ys, xs = np.nonzero(_mask(im, kind))
    if len(ys) < min_px:
        return None, int(len(ys))
    P = _backproject(d, K, ext, ys, xs)
    if len(P) < min_px:
        return None, 0
    zt = np.percentile(P[:, 2], 90)
    top = P[P[:, 2] > zt - 0.004]
    return np.median(top, 0), int(len(top))


def _lamp_state(frame):
    g = 0
    r = 0
    for cam in ('front', 'wrist'):
        im = getattr(frame, cam + '_rgb')
        g += int(_mask(im, 'lamp_green').sum())
        r += int(_mask(im, 'lamp_red').sum())
    if g >= 15 and r < 5:
        return 'green'
    if r >= 15 and g < 5:
        return 'red'
    return 'other'


def _locate_both(frame, kind, min_px=5):
    """Average of wrist and front estimates when they agree, else the one with more top pixels."""
    pw, nw = _locate_top(frame, 'wrist', kind, min_px)
    pf, nf = _locate_top(frame, 'front', kind, min_px)
    if pw is None and pf is None:
        return None
    if pw is None:
        return pf
    if pf is None:
        return pw
    if np.linalg.norm(pw[:2] - pf[:2]) < 0.01:
        return (pw + pf) / 2.0
    return pw if nw >= nf else pf


def _design(model, tt):
    w = 2 * np.pi / SHAPE_PERIOD
    if model == 'circle':
        return np.stack([np.ones_like(tt), np.cos(w * tt), np.sin(w * tt)], 1)
    n = int(model[4:])
    M = np.zeros((len(tt), n))
    for i in range(len(tt)):
        u = (tt[i] % SHAPE_PERIOD) / SHAPE_PERIOD * n
        k = int(np.floor(u)) % n
        s = u - np.floor(u)
        M[i, k] += 1 - s
        M[i, (k + 1) % n] += s
    return M


def _robust_fit(tt, P, model):
    M = _design(model, tt)
    keep = np.ones(len(tt), bool)
    C = None
    r = None
    for it in range(4):
        if keep.sum() < M.shape[1] + 2:
            break
        C = np.linalg.lstsq(M[keep], P[keep], rcond=None)[0]
        r = np.linalg.norm(M @ C - P, axis=1)
        thr = max(0.006, 3 * np.median(r[keep]))
        keep = r < thr
    if C is None:
        return None, 1.0, 0
    return C, float(np.sqrt(np.mean(r[keep] ** 2))) if keep.sum() else 1.0, int(keep.sum())


def _fit_shapes(samples, green_step, start_step=0):
    """samples: [step, x, y] red-cube observations. Shapes occupy consecutive 64-frame windows that end
    at the frame the lamp turns green. Returns a list of dicts with type, residual and path waypoints."""
    S = np.array(samples, float) if len(samples) else np.zeros((0, 3))
    # demos: the first loop starts 3-8 frames after reset, so whole periods fit between reset and the lamp
    n = int((green_step - start_step) // SHAPE_PERIOD)
    n = max(1, min(n, 8))
    shapes = []
    for k in range(n):
        a = green_step - SHAPE_PERIOD * (n - k)
        sel = (S[:, 0] >= a) & (S[:, 0] < a + SHAPE_PERIOD)
        tt = S[sel, 0] - a
        P = S[sel, 1:3]
        if len(tt) < 12:
            shapes.append({'type': 'missing', 'res': 1.0, 'n': int(len(tt)), 'pts': [], 'corner': []})
            continue
        fits = {}
        for m in ('circle', 'poly3', 'poly4', 'poly5'):
            fits[m] = _robust_fit(tt, P, m)
        best = 'circle'
        for m in fits:
            if fits[m][0] is not None and fits[m][1] < fits[best][1] - 1e-6:
                best = m
        # prefer the simpler polygon when a richer one is not clearly better
        if best == 'poly5' and fits['poly3'][1] < 1.5 * fits['poly5'][1] + 0.0005:
            best = 'poly3'
        if fits[best][1] > 0.004:
            alt = _robust_fit(tt, P, 'poly16')
            if alt[0] is not None and alt[1] < fits[best][1]:
                fits['poly16'] = alt
                best = 'poly16'
        C, res, nk = fits[best]
        if best == 'circle':
            w = 2 * np.pi / SHAPE_PERIOD
            pts = []
            for j in range(0, 49):
                t = j * SHAPE_PERIOD / 48.0
                p = C[0] + C[1] * np.cos(w * t) + C[2] * np.sin(w * t)
                pts.append([float(p[0]), float(p[1])])
            corner = [False] * len(pts)
            corner[-1] = True
        else:
            V = [[float(v[0]), float(v[1])] for v in C]
            pts = V + [V[0]]
            corner = [True] * len(pts)
            if best == 'poly16':
                corner = [False] * len(pts)
                corner[-1] = True
        shapes.append({'type': best, 'res': round(res, 5), 'n': nk, 'pts': pts, 'corner': corner,
                       'start_step': int(a)})
    return shapes
