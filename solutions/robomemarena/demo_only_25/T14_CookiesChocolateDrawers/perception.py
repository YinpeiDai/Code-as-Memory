"""Perception helpers: back-project agentview depth into world points and locate
the cabinet's top-drawer handle, the cookies box and the chocolate pudding box.
Offsets were fitted on the 25 assigned demonstrations (see task_analysis.md)."""

TABLE_OBJ_ZMIN = 0.436
TOP_HANDLE_Z = 0.6734
MID_HANDLE_DZ = -0.0669
MID_HANDLE_DY = -0.0060
PRIOR_HANDLE = [0.1321, 0.4578, 0.6734]
PRIOR_COOKIE_CENTROID = [0.2244, -0.3280]
PRIOR_CHOC_CENTROID = [-0.2137, -0.3260]
COOKIE_GRASP_OFF = [-0.0063, 0.0003]
CHOC_GRASP_OFF = [-0.0160, 0.0057]
GRASP_Z = 0.4335   # expert TCP rests at 0.4332 (fingertips on table) while still commanding down


def _world_points(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    c = np.stack([x, y, d], -1)
    pts = (c - E[:, 3]) @ E[:, :3]
    return pts, d > 0


def _locate_handle(pts, valid, rgb):
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    dark = np.asarray(rgb, dtype=float).sum(-1) < 200
    m = valid & (Z > 0.47) & (Z < 0.69) & (X > -0.12) & (X < 0.42) & (Y > 0.33) & (Y < 0.78) & dark
    body = m & (Y > PRIOR_HANDLE[1] + 0.07)
    if int(body.sum()) < 150 or int(m.sum()) < 400:
        return list(PRIOR_HANDLE), False, int(m.sum())
    xf = float(np.percentile(X[body], 98))
    sel = m & (X < xf - 0.03) & (X > xf - 0.25)
    if int(sel.sum()) < 150:
        return list(PRIOR_HANDLE), False, int(m.sum())
    y5 = float(np.percentile(Y[sel], 5))
    hx, hy = xf - 0.1279, y5 + 0.0210
    if abs(hx - PRIOR_HANDLE[0]) > 0.10 or abs(hy - PRIOR_HANDLE[1]) > 0.10:
        return list(PRIOR_HANDLE), False, int(m.sum())
    return [hx, hy, TOP_HANDLE_Z], True, int(m.sum())


def _locate_object(pts, valid, prior, off, radius=0.08):
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    m = valid & (Z > TABLE_OBJ_ZMIN) & (Z < 0.56) & (np.hypot(X - prior[0], Y - prior[1]) < radius)
    n = int(m.sum())
    if n < 20:
        return [prior[0] + off[0], prior[1] + off[1], GRASP_Z], False, n
    cx, cy = float(np.mean(X[m])), float(np.mean(Y[m]))
    return [cx + off[0], cy + off[1], GRASP_Z], True, n


def _perceive_scene(frame):
    pts, valid = _world_points(frame)
    handle, ok_h, nh = _locate_handle(pts, valid, frame.agentview_rgb)
    cookie, ok_c, nc = _locate_object(pts, valid, PRIOR_COOKIE_CENTROID, COOKIE_GRASP_OFF)
    choc, ok_k, nk = _locate_object(pts, valid, PRIOR_CHOC_CENTROID, CHOC_GRASP_OFF)
    return {"handle": handle, "cookie": cookie, "choc": choc,
            "ok": [ok_h, ok_c, ok_k], "n": [nh, nc, nk]}
