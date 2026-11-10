"""Ball perception helpers: red-sphere segmentation in RGB, back-projection with
aligned depth, and a known-radius sphere fit in the world frame."""

BALL_R = 0.02


def _seg_red(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r > 60) & (r > 2.5 * g + 10) & (r > 2.5 * b + 10)


def _ball_from_cam(rgb, depth, K, E, min_px):
    """Return (centre_world[3], n_pixels, rms_residual) or None."""
    m = _seg_red(rgb)
    n = int(m.sum())
    if n < min_px:
        return None
    v, u = np.nonzero(m)
    z = depth[v, u].astype(np.float64) / 1000.0
    ok = z > 0.0
    if int(ok.sum()) < min_px:
        return None
    u = u[ok].astype(np.float64)
    v = v[ok].astype(np.float64)
    z = z[ok]
    # keep the nearest surface layer (drop mixed edge pixels hitting background)
    zmin = np.percentile(z, 5)
    keep = z < zmin + 2.5 * BALL_R
    u, v, z = u[keep], v[keep], z[keep]
    if len(z) < min_px:
        return None
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    ray = pc / np.linalg.norm(pc, axis=1, keepdims=True)
    c = np.median(pc + BALL_R * ray, axis=0)
    # Gauss-Newton sphere fit with known radius, with outlier rejection
    w = np.ones(len(pc))
    for it in range(6):
        d = pc - c
        dist = np.linalg.norm(d, axis=1) + 1e-9
        res = dist - BALL_R
        if it >= 2:
            w = (np.abs(res) < 0.004).astype(np.float64)
            if w.sum() < 4:
                w = np.ones(len(pc))
        J = -d / dist[:, None]
        A = (J * w[:, None]).T @ J + 1e-6 * np.eye(3)
        g = (J * w[:, None]).T @ res
        step = -np.linalg.solve(A, g)
        sn = np.linalg.norm(step)
        if sn > 0.01:
            step = step * (0.01 / sn)
        c = c + step
    d = pc - c
    res = np.linalg.norm(d, axis=1) - BALL_R
    rms = float(np.sqrt(np.mean(res[w > 0] ** 2))) if w.sum() > 0 else 1.0
    R = E[:, :3]
    t = E[:, 3]
    cw = R.T @ (c - t)
    return cw, n, rms


def _detect_ball(frame):
    """Detect the ball in both cameras. Returns dict with 'pos' (world xyz),
    'src', 'n' or None. Prefers the wrist view when it has a large blob."""
    out = {}
    fr = _ball_from_cam(frame.front_rgb, frame.front_depth, frame.front_K,
                        frame.front_extrinsic, 6)
    wr = _ball_from_cam(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K,
                        frame.wrist_extrinsic, 40)
    cands = []
    if fr is not None and fr[2] < 0.006:
        cands.append(("front", fr))
    if wr is not None and wr[2] < 0.006:
        cands.append(("wrist", wr))
    if not cands:
        return None
    # sanity: ball sits on or slightly above the table
    good = [c for c in cands if -0.01 < c[1][0][2] < 0.25]
    if not good:
        return None
    if len(good) == 2:
        pf, pw = good[0][1][0], good[1][1][0]
        if np.linalg.norm(pf - pw) < 0.03:
            nw = good[1][1][1]
            nf = good[0][1][1]
            ww = min(1.0, nw / 400.0)
            wf = min(1.0, nf / 40.0) * 0.5
            p = (pw * ww + pf * wf) / (ww + wf)
            return {"pos": p, "src": "both", "n": int(nf + nw)}
        # disagree: trust the bigger blob relative to its usual size
        best = good[1] if good[1][1][1] > 300 else good[0]
    else:
        best = good[0]
    return {"pos": best[1][0], "src": best[0], "n": int(best[1][1])}
