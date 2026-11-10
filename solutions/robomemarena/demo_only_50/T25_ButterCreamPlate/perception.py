"""Perception helpers for T25 (butter and cream from plate1 to plate2).

All geometry is in world metres. Camera pixels are back-projected with the
supplied intrinsics/extrinsics and depth (optical-axis mm)."""

TABLE_REGION = (-0.05, 0.45, -0.42, 0.42)  # x_min, x_max, y_min, y_max (world) searched for objects


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    return pw, z > 0


def _project(p, K, E):
    pc = E[:, :3] @ np.asarray(p, float) + E[:, 3]
    return [K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2], pc[2]]


def _components(mask, min_size=1):
    """4-connected components of a boolean mask; returns list of (rows, cols) arrays, largest first."""
    H, W = mask.shape
    lab = np.zeros((H, W), int)
    ys, xs = np.nonzero(mask)
    comps = []
    cur = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if len(pts) >= min_size:
            a = np.array(pts)
            comps.append((a[:, 0], a[:, 1]))
    comps.sort(key=lambda c: -len(c[0]))
    return comps


def _scene_masks(rgb, depth, K, E, region=TABLE_REGION):
    pw, valid = _backproject(depth, K, E)
    im = rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    x0, x1, y0, y1 = region
    reg = valid & (pw[..., 0] > x0) & (pw[..., 0] < x1) & (pw[..., 1] > y0) & (pw[..., 1] < y1)
    zs = pw[..., 2][reg & (pw[..., 2] < 0.6)]
    table_z = float(np.median(zs)) if len(zs) > 100 else 0.431
    h = pw[..., 2] - table_z
    reg = reg & (h < 0.12)
    mn = np.minimum(np.minimum(r, g), b)
    white = mn > 185
    pink = (r > 140) & (r - b > 12) & (r - b < 80) & (r - g > 8) & (r - g < 65)
    plateish = white | pink
    obj = reg & (h > 0.006) & ~plateish
    cream = obj & (b > r + 6)
    butter = obj & (r > b + 30) & (r - g > 25)
    plate_white = reg & white & (h < 0.008)
    return dict(pw=pw, h=h, table_z=table_z, reg=reg, obj=obj, cream=cream, butter=butter,
                plate_white=plate_white)


def _blob_xy(pw, comp):
    pts = pw[comp[0], comp[1]]
    return [float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))], len(pts)


def _detect(rgb, depth, K, E):
    """Detect butter, cream and the two plates in one RGB-D view; returns dict of world xy."""
    m = _scene_masks(rgb, depth, K, E)
    pw = m["pw"]
    out = {"table_z": m["table_z"]}
    for name in ("butter", "cream"):
        comps = _components(m[name], 3)
        if comps:
            # merge components close to the largest one (labels split one box)
            main = comps[0]
            xy, n = _blob_xy(pw, main)
            rows = list(main[0]); cols = list(main[1])
            for c in comps[1:]:
                cxy, cn = _blob_xy(pw, c)
                if math.hypot(cxy[0] - xy[0], cxy[1] - xy[1]) < 0.04:
                    rows += list(c[0]); cols += list(c[1])
            xy, n = _blob_xy(pw, (np.array(rows), np.array(cols)))
            out[name] = xy
            out[name + "_n"] = n
        else:
            out[name] = None
            out[name + "_n"] = 0
    plates = []
    for c in _components(m["plate_white"], 8)[:4]:
        xy, n = _blob_xy(pw, c)
        pts = pw[c[0], c[1]]
        plates.append({"xy": [float(pts[:, 0].mean()), float(pts[:, 1].mean())], "n": n})
    out["plates"] = plates
    return out
