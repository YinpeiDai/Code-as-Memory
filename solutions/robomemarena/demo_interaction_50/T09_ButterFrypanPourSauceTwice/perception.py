"""Perception helpers: agentview RGB-D back-projection and table-top object detection."""

BOTTLE_GRASP_OFFSET = [-0.012, -0.013]   # visible-centroid -> grasp point (calibrated on 50 demos)
BUTTER_GRASP_OFFSET = [-0.009, 0.001]
PAN_RADIUS = 0.09                        # pan rim min-y -> pan centre


def _cloud(frame, cam="agentview"):
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    d = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    X = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    W = (X - E[:, 3]) @ E[:, :3]
    W[d <= 0] = np.nan
    return W


def _table_z(W):
    z = W[..., 2]
    ok = np.isfinite(z) & (np.abs(W[..., 0]) < 0.45) & (np.abs(W[..., 1]) < 0.6) & (z > 0.2) & (z < 0.8)
    zz = z[ok]
    if zz.size < 100:
        return 0.4307
    hist, edges = np.histogram(zz, bins=300, range=(0.2, 0.8))
    i = int(np.argmax(hist))
    sel = zz[(zz >= edges[i] - 0.003) & (zz <= edges[i + 1] + 0.003)]
    return float(np.median(sel))


def _components(P, res=0.01):
    """Connected components of points P (N,3) on an xy grid; returns list of index arrays."""
    if len(P) == 0:
        return []
    gi = np.floor((P[:, 0] + 1.0) / res).astype(int)
    gj = np.floor((P[:, 1] + 1.0) / res).astype(int)
    keys = gi * 1000 + gj
    order = np.argsort(keys)
    sk = keys[order]
    uniq, starts = np.unique(sk, return_index=True)
    cells = {}
    for n in range(len(uniq)):
        end = starts[n + 1] if n + 1 < len(uniq) else len(sk)
        cells[int(uniq[n])] = order[starts[n]:end]
    lab = {}
    out = []
    for key in cells:
        if key in lab:
            continue
        lab[key] = len(out)
        stack = [key]
        members = []
        while stack:
            c = stack.pop()
            members.append(cells[c])
            a, b = c // 1000, c % 1000
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nb = (a + da) * 1000 + (b + db)
                    if nb in cells and nb not in lab:
                        lab[nb] = len(out)
                        stack.append(nb)
        out.append(np.concatenate(members))
    return out


def _detect(frame, exclude_tcp=True):
    """Detect butter, sauce bottle, frypan and drainer on the table from one agentview frame."""
    W = _cloud(frame, "agentview")
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    tz = _table_z(W)
    z = W[..., 2]
    fin = np.isfinite(z)
    zf = np.where(fin, z, -1.0)
    xf = np.where(fin, W[..., 0], 9.0)
    yf = np.where(fin, W[..., 1], 9.0)
    region = (zf > tz + 0.008) & (zf < tz + 0.35) & (xf > -0.5) & (xf < 0.45) & (np.abs(yf) < 0.65)
    if exclude_tcp:
        tcp = np.asarray(frame.tcp, dtype=float)
        near = (np.abs(xf - tcp[0]) < 0.13) & (np.abs(yf - tcp[1]) < 0.13) & (zf > tcp[2] - 0.16)
        region &= ~near
    light = (rgb.min(-1) > 80) & ((rgb.max(-1) - rgb.min(-1)) < 20)
    res = {"table_z": tz, "butter": None, "bottle": None, "pan": None, "drainer": None}
    # dark / coloured objects
    m = region & ~light
    P = W[m]
    C = rgb[m]
    cands = []
    for idx in _components(P):
        if len(idx) < 12:
            continue
        p = P[idx]
        c = C[idx].mean(0)
        o = {
            "n": int(len(idx)),
            "c": [float(p[:, 0].mean()), float(p[:, 1].mean())],
            "x0": float(p[:, 0].min()), "x1": float(p[:, 0].max()),
            "y0": float(p[:, 1].min()), "y1": float(p[:, 1].max()),
            "h": float(np.percentile(p[:, 2], 95) - tz),
            "zlo": float(np.percentile(p[:, 2], 3) - tz),
            "rgb": [float(c[0]), float(c[1]), float(c[2])],
            "pts": p,
        }
        o["ex"] = o["x1"] - o["x0"]
        o["ey"] = o["y1"] - o["y0"]
        cands.append(o)
    # butter: low, small, orange
    best = None
    for o in cands:
        if o["h"] < 0.035 and o["ex"] < 0.09 and o["ey"] < 0.09 and o["rgb"][0] - o["rgb"][2] > 25:
            if best is None or o["n"] > best["n"]:
                best = o
    if best is not None:
        q = best["pts"][:, :2] - np.array(best["c"])
        ev, evec = np.linalg.eigh(q.T @ q / max(1, len(q)))
        ax = evec[:, 1]
        yaw = math.degrees(math.atan2(float(ax[1]), float(ax[0])))
        while yaw > 90.0:
            yaw -= 180.0
        while yaw <= -90.0:
            yaw += 180.0
        elong = float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12)))
        res["butter"] = {"c": best["c"], "h": best["h"], "n": best["n"], "yaw": yaw, "elong": elong,
                         "grasp": [best["c"][0] + BUTTER_GRASP_OFFSET[0], best["c"][1] + BUTTER_GRASP_OFFSET[1]]}
    # bottle: tall, small footprint
    best = None
    for o in cands:
        if 0.045 < o["h"] < 0.14 and o["ex"] < 0.11 and o["ey"] < 0.11 and o["n"] >= 30 and o["zlo"] < 0.03:
            if best is None or o["n"] > best["n"]:
                best = o
    if best is not None:
        res["bottle"] = {"c": best["c"], "h": best["h"], "n": best["n"],
                         "grasp": [best["c"][0] + BOTTLE_GRASP_OFFSET[0], best["c"][1] + BOTTLE_GRASP_OFFSET[1]]}
    # frypan: dark, low (<8cm), wide.  The rim's min-y edge plus PAN_RADIUS gives the
    # centre y; the centre x is the mid x-extent of all dark low points within the pan disc band.
    darklow = region & (rgb.max(-1) < 95) & (zf < tz + 0.07)
    Q = W[darklow]
    best = None
    for idx in _components(Q):
        if len(idx) < 80:
            continue
        p = Q[idx]
        ey = float(np.percentile(p[:, 1], 99) - np.percentile(p[:, 1], 1))
        ex = float(np.percentile(p[:, 0], 99) - np.percentile(p[:, 0], 1))
        if max(ex, ey) > 0.12 and (best is None or len(idx) > len(best)):
            best = p
    if best is not None:
        y0 = float(np.percentile(best[:, 1], 1))
        mx = float(np.median(best[:, 0]))
        band = Q[(Q[:, 1] > y0 - 0.005) & (Q[:, 1] < y0 + 2 * PAN_RADIUS - 0.01) & (np.abs(Q[:, 0] - mx) < 0.2)]
        cx = 0.5 * (float(np.percentile(band[:, 0], 2)) + float(np.percentile(band[:, 0], 98)))
        res["pan"] = {"c": [cx, y0 + PAN_RADIUS], "n": int(len(best)), "y0": y0}
    # drainer: light, tall
    ml = region & light & (zf > tz + 0.03)
    PL = W[ml]
    bestl = None
    for idx in _components(PL):
        if len(idx) < 60:
            continue
        p = PL[idx]
        h = float(np.percentile(p[:, 2], 95) - tz)
        if float(np.percentile(p[:, 2], 3) - tz) > 0.09:
            continue  # not standing on the table (e.g. robot links)
        ex = float(p[:, 0].max() - p[:, 0].min())
        ey = float(p[:, 1].max() - p[:, 1].min())
        if h > 0.08 and ex < 0.4 and ey < 0.4:
            if bestl is None or len(idx) > bestl[0]:
                bestl = (len(idx), p, h)
    if bestl is not None:
        p = bestl[1]
        c0 = [float(np.median(p[:, 0])), float(np.median(p[:, 1]))]
        rim = region & (zf > tz + 0.12) & (zf < tz + 0.22) & (np.abs(xf - c0[0]) < 0.2) & (np.abs(yf - c0[1]) < 0.2)
        R = W[rim]
        if len(R) >= 30:
            cx = 0.5 * (float(np.percentile(R[:, 0], 1)) + float(np.percentile(R[:, 0], 99)))
            cy = 0.5 * (float(np.percentile(R[:, 1], 1)) + float(np.percentile(R[:, 1], 99)))
        else:
            cx, cy = c0
        res["drainer"] = {"c": [cx, cy], "h": bestl[2], "n": int(bestl[0])}
    for o in cands:
        o.pop("pts", None)
    return res
