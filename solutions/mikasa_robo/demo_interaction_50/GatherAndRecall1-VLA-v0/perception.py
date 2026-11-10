"""Scene perception helpers for GatherAndRecall (front camera RGB-D, world frame)."""

COLOR_NAMES = ["red", "green", "blue"]


def _backproject(depth, K, ext):
    """World XYZ for every pixel of a depth image (H, W, 3) and a validity mask."""
    H, W = depth.shape
    z = np.asarray(depth, dtype=np.float64) / 1000.0
    u = np.arange(W, dtype=np.float64)[None, :]
    v = np.arange(H, dtype=np.float64)[:, None]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(ext[:, :3], dtype=np.float64)
    t = np.asarray(ext[:, 3], dtype=np.float64)
    pw = (pc - t) @ R
    return pw.reshape(H, W, 3), z > 0


def _grid_clusters(xy, cell=0.01):
    """Label points by 8-connected occupancy of a 2-D grid; returns labels array."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=int)
    ij = np.floor(xy / cell).astype(int)
    keys = {}
    cell_of = np.zeros(n, dtype=int)
    for k in range(n):
        key = (int(ij[k, 0]), int(ij[k, 1]))
        if key not in keys:
            keys[key] = len(keys)
        cell_of[k] = keys[key]
    lab = [-1] * len(keys)
    cur = 0
    for key, idx in keys.items():
        if lab[idx] >= 0:
            continue
        stack = [key]
        lab[idx] = cur
        while stack:
            a, b = stack.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nb = (a + da, b + db)
                    j = keys.get(nb)
                    if j is not None and lab[j] < 0:
                        lab[j] = cur
                        stack.append(nb)
        cur += 1
    lab = np.array(lab, dtype=int)
    return lab[cell_of]


def _scene(frame, known_buttons=None):
    """Detect cubes, disc, buttons and lamp from one front RGB-D frame.

    Coloured (saturated) and grey pixels are clustered separately on a 1 cm world
    grid so a cube resting on the grey disc stays a separate object."""
    rgb = np.asarray(frame.front_rgb).astype(np.int32)
    pw, valid = _backproject(frame.front_depth, frame.front_K, frame.front_extrinsic)
    X, Y, Z = pw[..., 0], pw[..., 1], pw[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)
    area = valid & (X > -0.40) & (X < 0.50) & (np.abs(Y) < 0.50)
    elev = area & (Z > 0.004) & (Z < 0.16)
    out = {"cubes": [], "disc": None, "buttons": [], "lamp": None}
    # ---- grey objects: disc (flat), lamp (tall)
    gv, gu = np.nonzero(elev & (sat < 45))
    gp = pw[gv, gu]
    glab = _grid_clusters(gp[:, :2], 0.01)
    best_disc = None
    for c in range(int(glab.max()) + 1 if len(glab) else 0):
        sel = glab == c
        cnt = int(sel.sum())
        if cnt < 30:
            continue
        p = gp[sel]
        zmax = float(np.percentile(p[:, 2], 98))
        if zmax < 0.02:
            flat = p[p[:, 2] < 0.02]
            if best_disc is None or cnt > best_disc[0]:
                cen = flat[:, :2].mean(0)
                rad = float(np.percentile(np.sqrt(((flat[:, :2] - cen) ** 2).sum(1)), 97))
                best_disc = (cnt, [float(cen[0]), float(cen[1]), float(np.median(flat[:, 2]))], rad)
        elif 0.085 <= zmax < 0.16 and cnt > 100:
            uu = gu[sel]
            vv = gv[sel]
            if out["lamp"] is None or cnt > out["lamp"]["n"]:
                out["lamp"] = {"bbox": [int(uu.min()), int(uu.max()), int(vv.min()), int(vv.max())],
                               "pos": [float(p[:, 0].mean()), float(p[:, 1].mean()), zmax], "n": cnt}
    if best_disc is not None:
        out["disc"] = {"pos": best_disc[1], "radius": best_disc[2], "n": best_disc[0]}
    # ---- coloured objects: button caps (top ~0.06) and cubes (top ~0.04)
    cv, cu = np.nonzero(elev & (sat > 80) & (Z < 0.08))
    cp = pw[cv, cu]
    ccol = rgb[cv, cu]
    clab = _grid_clusters(cp[:, :2], 0.01)
    for c in range(int(clab.max()) + 1 if len(clab) else 0):
        sel = clab == c
        cnt = int(sel.sum())
        if cnt < 12:
            continue
        p = cp[sel]
        col = ccol[sel]
        zmax = float(np.percentile(p[:, 2], 98))
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        ch = int(np.argmax(np.median(col, 0)))
        top = p[p[:, 2] > zmax - 0.004]
        if len(top) < 4:
            top = p
        cen = (top[:, :2].max(0) + top[:, :2].min(0)) / 2.0
        near_btn = False
        if known_buttons:
            for b in known_buttons:
                if math.hypot(cen[0] - b[0], cen[1] - b[1]) < 0.04:
                    near_btn = True
        if zmax >= 0.054 and zmax < 0.075 and known_buttons is None:
            mcen = top[:, :2].mean(0)
            out["buttons"].append({"color": COLOR_NAMES[ch], "pos": [float(mcen[0]), float(mcen[1]), zmax], "n": cnt})
        elif 0.02 <= zmax < 0.054 and ext.max() < 0.09 and not near_btn:
            out["cubes"].append({"pos": [float(cen[0]), float(cen[1]), zmax], "color": COLOR_NAMES[ch], "n": cnt})
    out["buttons"] = sorted(out["buttons"], key=lambda b: b["pos"][1])
    return out


def _wrist_cube(frame, guess, z_top, radius=0.035):
    """Refine a cube's top-face centre from the wrist RGB-D view (bbox centre of top face)."""
    rgb = np.asarray(frame.wrist_rgb).astype(np.int32)
    pw, valid = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    sat = rgb.max(-1) - rgb.min(-1)
    d = np.sqrt((pw[..., 0] - guess[0]) ** 2 + (pw[..., 1] - guess[1]) ** 2)
    m = valid & (sat > 80) & (np.abs(pw[..., 2] - z_top) < 0.006) & (d < radius)
    n = int(m.sum())
    if n < 30:
        return None
    p = pw[m]
    cen = (p[:, :2].max(0) + p[:, :2].min(0)) / 2.0
    ext = p[:, :2].max(0) - p[:, :2].min(0)
    return {"pos": [float(cen[0]), float(cen[1])], "n": n, "ext": [float(ext[0]), float(ext[1])]}


def _lamp_state(frame, bbox):
    """Colour shown by the lamp in this frame: 'red'/'green'/'blue' or 'off'.

    Only pixels inside the lamp bounding box that lie above the table (from
    depth) are considered, so the orange table never counts as a flash."""
    u0, u1, v0, v1 = bbox
    u0 = max(0, u0 - 2)
    v0 = max(0, v0 - 2)
    patch = np.asarray(frame.front_rgb)[v0:v1 + 3, u0:u1 + 3].astype(np.int32)
    dep = np.asarray(frame.front_depth)[v0:v1 + 3, u0:u1 + 3]
    if patch.size == 0:
        return "off", 0
    K = frame.front_K
    ext = np.asarray(frame.front_extrinsic, dtype=np.float64)
    z = dep.astype(np.float64) / 1000.0
    H, W = dep.shape
    uu = (np.arange(W, dtype=np.float64)[None, :] + u0 - K[0, 2]) / K[0, 0] * z
    vv = (np.arange(H, dtype=np.float64)[:, None] + v0 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([uu, vv, z], -1).reshape(-1, 3)
    pw = (pc - ext[:, 3]) @ ext[:, :3]
    above = (pw[:, 2].reshape(H, W) > 0.015) & (dep > 0)
    mx = patch.max(-1)
    srt = np.sort(patch, -1)
    second = srt[..., 1]
    strong = above & (mx > 150) & (second < 0.45 * mx)
    n = int(strong.sum())
    total = int(above.sum())
    if total == 0 or n < max(12, 0.15 * total):
        return "off", n
    ch = np.argmax(patch[strong], -1)
    counts = [int((ch == k).sum()) for k in range(3)]
    return COLOR_NAMES[int(np.argmax(counts))], n
