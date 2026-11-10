"""Perception helpers for T15 (butter -> frypan, pour milk twice).

All geometry is in world metres, computed from the agentview RGB-D image and
its calibration (no hidden simulator state)."""


def _rodrigues(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _log_rot(Rm):
    """Axis-angle vector of a rotation matrix."""
    c = (np.trace(Rm) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: use the symmetric part
        M = (Rm + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v * th
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def _world_cloud(frame):
    """(N,3) world points and (N,3) colours of valid agentview depth pixels."""
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    D = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    ok = D > 0.05
    d = D[ok]
    x = (u[ok] - K[0, 2]) / K[0, 0] * d
    y = (v[ok] - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1) - E[:, 3]
    W = pc @ E[:, :3]
    return W, rgb[ok]


def _circle_fit(xy):
    A = np.concatenate([2.0 * xy, np.ones((len(xy), 1))], 1)
    b = (xy ** 2).sum(1)
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    c = s[:2]
    r = math.sqrt(max(float(s[2] + c @ c), 1e-8))
    return c, r


def _pan_hough(xy):
    """Robust rim-circle fit: vote over a centre grid and radii, then refine
    by least squares on the inliers. Returns (cx, cy, r, inliers) or None."""
    lo = xy.min(0) - 0.02
    hi = xy.max(0) + 0.02
    best = None
    for step, span in ((0.006, None), (0.0015, 0.012)):
        if span is None:
            xs = np.arange(lo[0], hi[0], step)
            ys = np.arange(lo[1], hi[1], step)
            rs = np.arange(0.07, 0.15, 0.004)
        else:
            xs = np.arange(best[0] - span, best[0] + span, step)
            ys = np.arange(best[1] - span, best[1] + span, step)
            rs = np.arange(best[2] - 0.008, best[2] + 0.008, 0.001)
        cx, cy = np.meshgrid(xs, ys)
        C = np.stack([cx.ravel(), cy.ravel()], 1)
        d = np.sqrt(((C[:, None, :] - xy[None, :, :]) ** 2).sum(2))
        for r in rs:
            cnt = (np.abs(d - r) < 0.006).sum(1)
            i = int(np.argmax(cnt))
            if best is None or cnt[i] > best[3]:
                best = (float(C[i, 0]), float(C[i, 1]), float(r), int(cnt[i]))
    if best is None or best[3] < 30:
        return None
    c = np.array(best[:2])
    r = best[2]
    inl = xy[np.abs(np.linalg.norm(xy - c, axis=1) - r) < 0.008]
    if len(inl) >= 30:
        c2, r2 = _circle_fit(inl)
        if np.linalg.norm(c2 - c) < 0.01 and abs(r2 - r) < 0.01:
            c, r = c2, r2
    return (float(c[0]), float(c[1]), float(r), len(inl))


def _scene(frame, tcp=None):
    """Locate table height, milk carton, frypan and butter.

    Returns dict with table_z, milk [x,y], milk_n, pan [x,y], pan_r, pan_n,
    butter [x,y], butter_n.  Missing objects are None."""
    W, C = _world_cloud(frame)
    inside = (W[:, 0] > -0.35) & (W[:, 0] < 0.35) & (np.abs(W[:, 1]) < 0.45)
    zs = W[inside, 2]
    zs = zs[(zs > 0.2) & (zs < 0.8)]
    table = 0.43
    if len(zs) > 500:
        h, edges = np.histogram(zs, bins=120, range=(0.2, 0.8))
        i = int(np.argmax(h))
        cand = zs[(zs >= edges[i] - 0.01) & (zs <= edges[i + 1] + 0.01)]
        if len(cand):
            table = float(np.median(cand))
    keep = inside & (W[:, 2] > table + 0.006) & (W[:, 2] < table + 0.19)
    if tcp is not None:
        t = np.asarray(tcp[:3], dtype=float)
        near = np.linalg.norm(W[:, :2] - t[:2], axis=1) < 0.09
        keep = keep & ~(near & (W[:, 2] > t[2] - 0.12))
    P = W[keep]
    Q = C[keep]
    out = {"table_z": table, "milk": None, "milk_n": 0, "pan": None,
           "pan_r": None, "pan_n": 0, "butter": None, "butter_n": 0}
    used = np.zeros(len(P), dtype=bool)
    tall = P[:, 2] > table + 0.06
    if tall.sum() >= 15:
        seed = np.median(P[tall, :2], 0)
        m = np.linalg.norm(P[:, :2] - seed, axis=1) < 0.045
        seed = np.median(P[m, :2], 0)
        m = np.linalg.norm(P[:, :2] - seed, axis=1) < 0.045
        out["milk"] = [float(v) for v in P[m, :2].mean(0)]
        out["milk_n"] = int(m.sum())
        used = used | m
    dark = (~used) & (Q.max(1) < 85) & (np.abs(Q[:, 0] - Q[:, 2]) < 25) & (P[:, 2] < table + 0.07)
    if dark.sum() >= 40:
        fit = _pan_hough(P[dark, :2])
        if fit is not None:
            out["pan"] = [float(fit[0]), float(fit[1])]
            out["pan_r"] = float(fit[2])
            out["pan_n"] = int(fit[3])
    orange = (~used) & (Q[:, 0] > Q[:, 1] + 20) & (Q[:, 0] > Q[:, 2] + 30) & (P[:, 2] < table + 0.04)
    if out["pan"] is not None:
        orange = orange & (np.linalg.norm(P[:, :2] - np.array(out["pan"]), axis=1) > out["pan_r"] + 0.02)
    if orange.sum() >= 8:
        seed = np.median(P[orange, :2], 0)
        m = orange & (np.linalg.norm(P[:, :2] - seed, axis=1) < 0.04)
        sm = (~used) & (P[:, 2] < table + 0.04) & (np.linalg.norm(P[:, :2] - P[m, :2].mean(0), axis=1) < 0.035)
        if out["pan"] is not None:
            sm = sm & (np.linalg.norm(P[:, :2] - np.array(out["pan"]), axis=1) > out["pan_r"] + 0.02)
        if sm.sum() >= 8:
            out["butter"] = [float(v) for v in P[sm, :2].mean(0)]
            out["butter_n"] = int(sm.sum())
    return out
