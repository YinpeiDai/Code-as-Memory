"""Perception helpers for observe_and_pickup (head RGB-D, world frame, metres)."""

TABLE_Z = 0.741          # measured table-top height (median of head depth points)
SHELF_Z = 0.8765         # measured shelf-top height
SHELF_H_BIAS = 0.004     # shelf objects measure ~4 mm taller than the same model on the table
HIST_BINS = 5


def _cloud(depth, K, E):
    d = np.asarray(depth).astype(np.float64) / 1000.0
    K = np.asarray(K, np.float64)
    E = np.asarray(E, np.float64)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw.reshape(H, W, 3), d > 0


def _components(mask, pw, min_size, rgb=None, dmax=0.010, cmax=60.0):
    H, W = mask.shape
    lab = -np.ones((H, W), np.int32)
    comps = []
    for start in np.flatnonzero(mask.ravel()):
        sv, su = divmod(int(start), W)
        if lab[sv, su] >= 0:
            continue
        cid = len(comps)
        stack = [(sv, su)]
        lab[sv, su] = cid
        pix = []
        while stack:
            v, u = stack.pop()
            pix.append((v, u))
            p = pw[v, u]
            pc = rgb[v, u] if rgb is not None else None
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a = v + dv
                b = u + du
                if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] < 0:
                    q = pw[a, b]
                    ok = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2 < dmax * dmax
                    if ok and pc is not None:
                        qc = rgb[a, b]
                        ok = abs(qc[0] - pc[0]) + abs(qc[1] - pc[1]) + abs(qc[2] - pc[2]) < cmax
                    if ok:
                        lab[a, b] = cid
                        stack.append((a, b))
        comps.append(np.array(pix))
    return [c for c in comps if len(c) >= min_size]


def _soft_hist(C, bins):
    x = np.clip(np.asarray(C, np.float64) / 255.0 * (bins - 1), 0, bins - 1 - 1e-9)
    i0 = np.floor(x).astype(int)
    f = x - i0
    h = np.zeros((bins, bins, bins))
    for a in (0, 1):
        wa = f[:, 0] if a else 1 - f[:, 0]
        for b in (0, 1):
            wb = f[:, 1] if b else 1 - f[:, 1]
            for c in (0, 1):
                wc = f[:, 2] if c else 1 - f[:, 2]
                np.add.at(h, (np.minimum(i0[:, 0] + a, bins - 1), np.minimum(i0[:, 1] + b, bins - 1),
                              np.minimum(i0[:, 2] + c, bins - 1)), wa * wb * wc)
    s = h.sum()
    return (h / s).ravel() if s > 0 else h.ravel()


def _describe(P, C, base):
    xy = P[:, :2]
    cen = xy.mean(0)
    if len(P) > 3:
        w, V = np.linalg.eigh(np.cov((xy - cen).T))
    else:
        V = np.eye(2)
    ax = V[:, 1]
    p1 = (xy - cen) @ ax
    p2 = (xy - cen) @ V[:, 0]
    z = P[:, 2]
    ztop = float(np.percentile(z, 98))
    top = P[z > ztop - 0.012]
    return {
        "n": int(len(P)),
        "cen": [float(cen[0]), float(cen[1])],
        "top_cen": [float(top[:, 0].mean()), float(top[:, 1].mean())],
        "h": ztop - base,
        "L": float(np.percentile(p1, 98) - np.percentile(p1, 2)),
        "W": float(np.percentile(p2, 98) - np.percentile(p2, 2)),
        "yaw": float(math.atan2(ax[1], ax[0])),
        "hist": _soft_hist(C, HIST_BINS),
        "rgb": [float(v) for v in np.median(C, 0)],
    }


def _cam(frame, name):
    return (getattr(frame, name + "_rgb"), getattr(frame, name + "_depth"),
            getattr(frame, name + "_K"), getattr(frame, name + "_extrinsic"))


def _shelf_object(frame):
    """Descriptor of the single object standing on the shelf (head camera), or None."""
    rgb, depth, K, E = _cam(frame, "head")
    pw, valid = _cloud(depth, K, E)
    x = pw[..., 0]
    y = pw[..., 1]
    z = pw[..., 2]
    m = valid & (z > SHELF_Z + 0.004) & (z < 1.10) & (y > 0.05) & (y < 0.36) & (x > -0.35) & (x < 0.30)
    comps = _components(m, pw, 5)
    if not comps:
        return None
    pix = np.concatenate(comps, 0)
    d = _describe(pw[pix[:, 0], pix[:, 1]], rgb[pix[:, 0], pix[:, 1]], SHELF_Z)
    d["h"] = d["h"] - SHELF_H_BIAS
    return d


def _dilate(mask, n):
    rd = mask.copy()
    for _ in range(n):
        g = rd.copy()
        g[1:] |= rd[:-1]
        g[:-1] |= rd[1:]
        g[:, 1:] |= rd[:, :-1]
        g[:, :-1] |= rd[:, 1:]
        rd = g
    return rd


def _table_scene(frame):
    """Coarse geometric clusters of points standing on the table (head camera) and context arrays."""
    rgb, depth, K, E = _cam(frame, "head")
    rgb = np.asarray(rgb).astype(np.float64)
    pw, valid = _cloud(depth, K, E)
    y = pw[..., 1]
    z = pw[..., 2]
    m = valid & (z > TABLE_Z + 0.006) & (z < 0.90) & (y < 0.04) & (y > -0.45)
    comps = _components(m, pw, 25, None, 0.010)
    robot = _dilate(valid & (z >= 0.90) & (y < 0.05), 3)
    info = []
    for c in comps:
        P = pw[c[:, 0], c[:, 1]]
        info.append({"pix": c, "touch": bool(robot[c[:, 0], c[:, 1]].any()), "cen": P[:, :2].mean(0),
                     "hist": _soft_hist(rgb[c[:, 0], c[:, 1]], HIST_BINS), "P": P})
    # merge fragments split by a gripper (both touch the robot silhouette) or by an overhang
    parent = list(range(len(info)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            a = info[i]
            b = info[j]
            if float(np.linalg.norm(a["cen"] - b["cen"])) > 0.11:
                continue
            if float(np.minimum(a["hist"], b["hist"]).sum()) < 0.45:
                continue
            pa = a["P"][::3, :2]
            pb = b["P"][::3, :2]
            dmin = float(np.min(np.linalg.norm(pa[:, None] - pb[None], axis=2)))
            if dmin < 0.008 or (a["touch"] and b["touch"] and dmin < 0.05):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(info)):
        groups.setdefault(find(i), []).append(i)
    clusters = []
    for g in groups.values():
        clusters.append({"pix": np.concatenate([info[i]["pix"] for i in g], 0),
                         "occluded": any(info[i]["touch"] for i in g), "parts": len(g)})
    return clusters, pw, rgb


def _cluster_desc(cl, pw, rgb):
    pix = cl["pix"]
    d = _describe(pw[pix[:, 0], pix[:, 1]], rgb[pix[:, 0], pix[:, 1]], TABLE_Z)
    d["occluded"] = cl["occluded"]
    d["parts"] = cl["parts"]
    return d


def _split_for_target(cl, pw, rgb, s, min_frag=5):
    """Inside an oversized cluster, grow the region that best resembles the shelf object within its size."""
    pix = cl["pix"]
    H, W = pw.shape[:2]
    m = np.zeros((H, W), bool)
    m[pix[:, 0], pix[:, 1]] = True
    frags = _components(m, pw, min_frag, rgb, 0.010, 45.0)
    if len(frags) < 2:
        return None
    lab = -np.ones((H, W), np.int32)
    for i, f in enumerate(frags):
        lab[f[:, 0], f[:, 1]] = i
    adj = set()
    for dv, du in ((1, 0), (0, 1)):
        a = lab[:H - dv, :W - du]
        b = lab[dv:, du:]
        sel = (a >= 0) & (b >= 0) & (a != b)
        for x, yv in zip(a[sel].tolist(), b[sel].tolist()):
            adj.add((min(x, yv), max(x, yv)))
    nb = [set() for _ in frags]
    for x, yv in adj:
        nb[x].add(yv)
        nb[yv].add(x)
    th = np.asarray(s["hist"])
    sims = [float(np.minimum(th, _soft_hist(rgb[f[:, 0], f[:, 1]], HIST_BINS)).sum()) for f in frags]
    seeds = [i for i in range(len(frags)) if len(frags[i]) >= 3 * min_frag]
    if not seeds:
        return None
    seed = max(seeds, key=lambda i: sims[i] * min(1.0, len(frags[i]) / (12.0 * min_frag)))
    region = {seed}
    limL = s["L"] + 0.02
    limW = s["W"] + 0.02
    while True:
        cand = set()
        for r in region:
            cand |= nb[r]
        cand -= region
        best = None
        for c in sorted(cand, key=lambda i: -sims[i]):
            if sims[c] < 0.25 and len(frags[c]) >= 6 * min_frag:
                continue
            trial = np.concatenate([frags[i] for i in region | {c}], 0)
            d = _describe(pw[trial[:, 0], trial[:, 1]], rgb[trial[:, 0], trial[:, 1]], TABLE_Z)
            if d["L"] <= limL and d["W"] <= limW:
                best = c
                break
        if best is None:
            break
        region.add(best)
    sub = np.concatenate([frags[i] for i in region], 0)
    return {"pix": sub, "occluded": cl["occluded"], "parts": cl["parts"]}


def _table_objects(frame, s=None):
    """Descriptors of object candidates on the table; oversized clusters are split using the shelf object's size."""
    clusters, pw, rgb = _table_scene(frame)
    out = []
    for cl in clusters:
        d = _cluster_desc(cl, pw, rgb)
        if s is not None and (d["L"] > s["L"] + 0.035 or d["W"] > s["W"] + 0.035):
            sub = _split_for_target(cl, pw, rgb, s)
            if sub is not None:
                ds = _cluster_desc(sub, pw, rgb)
                ds["split"] = True
                out.append(ds)
                continue
        out.append(d)
    return out


def _chroma(rgb):
    return np.array([rgb[0] - rgb[2], rgb[1] - rgb[2]], np.float64)


def _match_cost(s, o):
    """Lower is better: colour-histogram distance, chroma distance, height mismatch (one-sided when the
    candidate is partly hidden by a gripper) and a one-sided size penalty for unoccluded candidates."""
    col = 1.0 - float(np.minimum(np.asarray(s["hist"]), np.asarray(o["hist"])).sum())
    cost = 10.0 * col
    if "rgb" in s and "rgb" in o:
        co = _chroma(o["rgb"])
        cost += 1.0 * float(np.linalg.norm((_chroma(s["rgb"]) - co) / (8.0 + 0.1 * np.abs(co))))
    dh = o["h"] - s["h"]
    if o.get("occluded", False):
        cost += 1.0 * max(0.0, dh - 0.003) / 0.01 + 0.3 * max(0.0, -dh) / 0.01
    else:
        cost += 1.0 * abs(dh) / 0.01
        cost += 0.25 * max(0.0, o["L"] - s["L"] - 0.03) / 0.01 + 0.25 * max(0.0, s["L"] - o["L"] - 0.03) / 0.01
    return cost


def _wrist_clusters(frame, arm, zmax_world=None, s=None):
    """Object clusters below a top-down wrist camera (world frame), excluding the gripper's own fingers.
    With the shelf descriptor `s`, clusters much larger than the target are split like the head clusters."""
    rgb, depth, K, E = _cam(frame, arm + "_wrist")
    rgb = np.asarray(rgb).astype(np.float64)
    pw, valid = _cloud(depth, K, E)
    z = pw[..., 2]
    dm = np.asarray(depth).astype(np.float64) / 1000.0
    ztop = TABLE_Z + 0.12 if zmax_world is None else zmax_world
    m = valid & (z > TABLE_Z + 0.006) & (z < ztop) & (dm > 0.06)
    comps = _components(m, pw, 40, None, 0.008)
    out = []
    for c in comps:
        d = _describe(pw[c[:, 0], c[:, 1]], rgb[c[:, 0], c[:, 1]], TABLE_Z)
        if s is not None and (d["L"] > s["L"] + 0.04 or d["W"] > s["W"] + 0.04):
            sub = _split_for_target({"pix": c, "occluded": False, "parts": 1}, pw, rgb, s, 40)
            if sub is not None:
                ds = _describe(pw[sub["pix"][:, 0], sub["pix"][:, 1]], rgb[sub["pix"][:, 0], sub["pix"][:, 1]],
                               TABLE_Z)
                ds["split"] = True
                out.append(ds)
        out.append(d)
    return out


def _table_points_near(frame, arm, xy, radius=0.03):
    """Number of wrist-camera points standing on the table (below 7 cm) within `radius` of xy."""
    rgb, depth, K, E = _cam(frame, arm + "_wrist")
    pw, valid = _cloud(depth, K, E)
    z = pw[..., 2]
    d2 = (pw[..., 0] - xy[0]) ** 2 + (pw[..., 1] - xy[1]) ** 2
    m = valid & (z > TABLE_Z + 0.006) & (z < TABLE_Z + 0.07) & (d2 < radius * radius)
    return int(m.sum())
