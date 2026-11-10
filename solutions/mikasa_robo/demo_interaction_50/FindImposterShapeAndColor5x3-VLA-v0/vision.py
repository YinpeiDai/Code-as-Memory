"""Perception helpers for FindImposterShapeAndColor5x3: colour segmentation,
3-D back-projection and shape classification of the 15 table objects."""

SHAPES = ["sphere", "cube", "T", "cross", "torus"]
COLORS = ["red", "green", "blue"]


def _color_masks(rgb):
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    red = (r > 90) & (g < 50) & (b < 50)
    green = (g > 90) & (r < 80) & (b < 80)
    blue = (b > 90) & (r < 60) & (g < 80)
    return {"red": red, "green": green, "blue": blue}


def _lowsat(rgb):
    """Robot / gripper / background pixels (grey, unsaturated)."""
    im = rgb.astype(np.int32)
    return (im.max(axis=2) - im.min(axis=2)) < 40


def _components(mask, minpix):
    ys, xs = np.nonzero(mask)
    todo = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while todo:
        seed = todo.pop()
        stack = [seed]
        pix = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in todo:
                        todo.discard(q)
                        stack.append(q)
                        pix.append(q)
        if len(pix) >= minpix:
            comps.append(np.array(pix, dtype=np.int64))
    return comps


def _backproject(pix, depth, K, E):
    """World XYZ of pixel list (N,2 rows/cols); returns (P, valid)."""
    v = pix[:, 0].astype(np.float64)
    u = pix[:, 1].astype(np.float64)
    z = depth[pix[:, 0], pix[:, 1]].astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0][2]) / K[0][0] * z
    y = (v + 0.5 - K[1][2]) / K[1][1] * z
    Pc = np.stack([x, y, z], axis=1)
    R = np.array(E, dtype=np.float64)[:, :3]
    t = np.array(E, dtype=np.float64)[:, 3]
    Pw = (Pc - t) @ R
    return Pw, z > 0


def _classify(cam, zmax, area, sx, sy, fill):
    """Shape label from metric features; 'merged' for blobs too large for one object."""
    a = area * 1e4
    if cam == "wrist":
        if zmax > 0.034:
            if a > 45.0 or max(sx, sy) > 0.052:
                return "merged"
            return "sphere" if a < 19.5 else "cube"
        if a > 31.5 or max(sx, sy) > 0.068:
            return "merged"
        if a < 17.8:
            return "T"
        return "cross" if max(sx, sy) > 0.0565 else "torus"
    # front camera (oblique view)
    if zmax > 0.034:
        if a > 32.0 or max(sx, sy) > 0.062:
            return "merged"
        return "sphere" if a < 17.3 else "cube"
    if a > 31.0 or max(sx, sy) > 0.07:
        return "merged"
    if a < 17.0:
        return "T"
    return "cross" if max(sx, sy) > 0.0555 else "torus"


def _detect(frame, cam):
    """List of colour blobs with world features and a shape label."""
    rgb = frame.wrist_rgb if cam == "wrist" else frame.front_rgb
    depth = frame.wrist_depth if cam == "wrist" else frame.front_depth
    K = np.array(frame.wrist_K if cam == "wrist" else frame.front_K, dtype=np.float64)
    E = frame.wrist_extrinsic if cam == "wrist" else frame.front_extrinsic
    low = _lowsat(rgb)
    H, W = low.shape
    out = []
    masks = _color_masks(rgb)
    for col in COLORS:
        for pix in _components(masks[col], 4):
            P, ok = _backproject(pix, depth, K, E)
            if ok.sum() < 4:
                continue
            P = P[ok]
            pv = pix[ok]
            z = P[:, 2]
            if np.percentile(z, 50) > 0.08:
                continue
            zmax = float(np.percentile(z, 95))
            d = depth[pv[:, 0], pv[:, 1]].astype(np.float64) / 1000.0
            area = float(np.sum((d / K[0][0]) ** 2))
            sx = float(np.ptp(P[:, 0]))
            sy = float(np.ptp(P[:, 1]))
            bb = float((np.ptp(pix[:, 0]) + 1) * (np.ptp(pix[:, 1]) + 1))
            fill = len(pix) / bb
            top = P[z > zmax - 0.008]
            # occlusion / border checks
            r0, r1 = int(pix[:, 0].min()), int(pix[:, 0].max())
            c0, c1 = int(pix[:, 1].min()), int(pix[:, 1].max())
            edge = r0 <= 1 or c0 <= 1 or r1 >= H - 2 or c1 >= W - 2
            ys = np.clip(np.concatenate([pix[:, 0] + dy for dy in (-2, 0, 2) for dx in (-2, 0, 2)]), 0, H - 1)
            xs = np.clip(np.concatenate([pix[:, 1] + dx for dy in (-2, 0, 2) for dx in (-2, 0, 2)]), 0, W - 1)
            occl = bool(low[ys, xs].any())
            label = _classify(cam, zmax, area, sx, sy, fill)
            out.append({
                "color": col, "shape": label, "cam": cam,
                "x": float(top[:, 0].mean()), "y": float(top[:, 1].mean()),
                "zmax": zmax, "area": area, "sx": sx, "sy": sy, "n": int(len(pix)),
                "reliable": bool(label in SHAPES and not edge and not occl),
                "partial": bool(edge or occl),
                "pts": P[:, :2],
                "P3": P,
            })
    return out


def _fuse(wrist, front):
    """Combine both views into one object list. Wrist (top-down) labels win."""
    objs = [dict(b) for b in wrist]
    for f in front:
        best = None
        bd = 1e9
        for o in objs:
            if o["color"] != f["color"] or o["shape"] == "merged" or o["cam"] != "wrist":
                continue
            d = math.hypot(o["x"] - (f["x"] - 0.004), o["y"] - f["y"])
            if d < bd:
                bd, best = d, o
        if best is not None and bd < 0.03 and f["shape"] != "merged":
            if (not best["reliable"]) and f["reliable"]:
                best["shape"] = f["shape"]
                best["reliable"] = True
                best["src"] = "front"
            elif best["reliable"] and f["reliable"] and f["shape"] != best["shape"]:
                best["conflict"] = f["shape"]
            continue
        if f["shape"] == "merged":
            dup = False
            for o in objs:
                if o["color"] == f["color"] and o["shape"] == "merged" and o["cam"] == "wrist" and \
                        math.hypot(o["x"] - f["x"], o["y"] - f["y"]) < 0.04:
                    dup = True
            if dup:
                continue
        objs.append(dict(f))
    keep = []
    for o in objs:
        if o["shape"] == "merged":
            P = o["pts"]
            x0, y0 = P[:, 0].min() - 0.01, P[:, 1].min() - 0.01
            x1, y1 = P[:, 0].max() + 0.01, P[:, 1].max() + 0.01
            inside = 0
            for p in objs:
                if p is o or p["color"] != o["color"] or p["shape"] == "merged" or p["cam"] == o["cam"]:
                    continue
                if not p["reliable"]:
                    continue
                if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1:
                    inside += 1
            if inside >= 2:
                continue
        keep.append(o)
    return keep


def _scene(frame):
    """Detect, fuse and summarise objects visible in one frame (both cameras).
    Merged same-colour blobs are decomposed into shape templates."""
    w = _detect(frame, "wrist")
    f = _detect(frame, "front")
    objs = _fuse(w, f)
    out = []
    for o in objs:
        if o["shape"] != "merged":
            out.append(o)
            continue
        parts, rem = _decompose(o["P3"])
        if len(parts) == 1 and parts[0]["frac"] > 0.85 and rem < 40:
            p = parts[0]
            q = dict(o)
            q["shape"] = p["shape"]
            q["reliable"] = True
            q["single_fit"] = True
            q["zmax"] = 0.04 if p["shape"] in TALL_SHAPES else o["zmax"]
            out.append(q)
            continue
        if len(parts) < 2:
            o["parts"] = len(parts)
            out.append(o)
            continue
        for p in parts:
            out.append({"color": o["color"], "shape": p["shape"], "cam": o["cam"] + "-split",
                        "x": p["x"], "y": p["y"], "zmax": (0.04 if p["shape"] in TALL_SHAPES else 0.027),
                        "area": 0.0, "sx": 0.0, "sy": 0.0, "n": 0, "reliable": True, "partial": False,
                        "split": True, "frac": p["frac"], "pts": o["pts"], "P3": o["P3"]})
    final = []
    for o in out:
        dup = False
        for p in final:
            if p["color"] == o["color"] and p["shape"] == o["shape"] and \
                    math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 0.025:
                dup = True
        if not dup:
            final.append(o)
    return final, len(w), len(f)


def _count_visible(frame):
    """Number of coloured blobs in the front view (cheap phase detector)."""
    m = _color_masks(frame.front_rgb)
    return int(sum(int(v.sum()) for v in m.values()))
