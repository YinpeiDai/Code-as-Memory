"""Geometry, perception and low-level motion helpers shared by the policy tools."""

PI = 3.141592653589793
# Drawer handle heights relative to the measured cabinet top surface (50 demos, std <= 2 mm).
HANDLE_DZ = {"top": -0.0425, "middle": -0.109, "bottom": -0.182}
# Drawer floor heights relative to the handle (agentview depth inside open drawers).
FLOOR_DZ = {"top": -0.021, "middle": -0.025, "bottom": -0.021}
# Handle grasp point relative to cabinet top-surface extents (demo regression, std ~1 mm).
HANDLE_DX_FROM_XMAX = -0.130
HANDLE_DY_FROM_YMIN = -0.006
BUTTER_PRIOR = (0.11, -0.20)


def _skew(k):
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def _rv2m(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    K = _skew(v / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _m2rv(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > PI - 1e-3:
        d = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        k = np.sqrt(d)
        i = int(np.argmax(k))
        if i == 0:
            k[1] = math.copysign(k[1], R[0, 1] + R[1, 0])
            k[2] = math.copysign(k[2], R[0, 2] + R[2, 0])
        elif i == 1:
            k[0] = math.copysign(k[0], R[0, 1] + R[1, 0])
            k[2] = math.copysign(k[2], R[1, 2] + R[2, 1])
        else:
            k[0] = math.copysign(k[0], R[0, 2] + R[2, 0])
            k[1] = math.copysign(k[1], R[1, 2] + R[2, 1])
        k = k / max(1e-9, float(np.linalg.norm(k)))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _R_drawer():
    """Gripper pointing +y (into the cabinet), fingers closing vertically (demo: tcp rot ~ (4.71,0,0))."""
    return _rx(-PI / 2)


def _R_down(yaw=0.0):
    """Top-down gripper (demo butter grasp: tcp rot ~ (3.14,0,0))."""
    return _rz(yaw) @ _rx(PI)


def _cloud(frame, cam):
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    D = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    H, W = D.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * D
    y = (v - K[1, 2]) / K[1, 1] * D
    pc = np.stack([x, y, D], -1).reshape(-1, 3)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw.reshape(H, W, 3), D


def _perceive_cabinet(frame):
    """Cabinet top surface from agentview RGB-D: dark points about 0.6-0.8 m high at y > 0.2."""
    P, D = _cloud(frame, "agentview")
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.sum(2) < 150
    base = (D > 0) & dark & (P[..., 1] > 0.2) & (P[..., 1] < 0.95) & (P[..., 0] > -0.2) & (P[..., 0] < 0.6)
    cand = base & (P[..., 2] > 0.6) & (P[..., 2] < 0.85)
    if cand.sum() < 200:
        return None
    z = P[..., 2][cand]
    hist, edges = np.histogram(z, bins=np.arange(0.6, 0.851, 0.005))
    topz = float(edges[int(np.argmax(hist))] + 0.0025)
    top = base & (np.abs(P[..., 2] - topz) < 0.012)
    pts = P[top]
    if len(pts) < 150:
        return None
    zt = float(np.median(pts[:, 2]))
    xmax = float(np.percentile(pts[:, 0], 99))
    xmin = float(np.percentile(pts[:, 0], 1))
    ymin = float(np.percentile(pts[:, 1], 1))
    ymax = float(np.percentile(pts[:, 1], 99))
    return {"hx": xmax + HANDLE_DX_FROM_XMAX, "hy": ymin + HANDLE_DY_FROM_YMIN, "topz": zt,
            "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "n": int(len(pts))}


def _perceive_butter(frame, prior=None, win=0.10, table_z=None):
    """Butter = largest low object (6 mm - 9 cm above the table) near the prior table location. The table
    height is the z-histogram mode of the window (or the given table_z), so the gripper/arm hovering above
    the window cannot bias it. Returns centre, extents, PCA major-axis angle and table height."""
    px, py = prior if prior is not None else BUTTER_PRIOR
    obs = []
    tzs = []
    for cam in ("wrist", "agentview"):
        P, D = _cloud(frame, cam)
        m = (D > 0) & (np.abs(P[..., 0] - px) < win) & (np.abs(P[..., 1] - py) < win) & (P[..., 2] > 0.25) & (P[..., 2] < 0.7)
        if m.sum() < 30:
            continue
        pts = P[m]
        if table_z is None:
            h, e = np.histogram(pts[:, 2], bins=np.arange(0.25, 0.7001, 0.004))
            tz = float(e[int(np.argmax(h))] + 0.002)
            near = pts[np.abs(pts[:, 2] - tz) < 0.006]
            if len(near) > 10:
                tz = float(np.median(near[:, 2]))
        else:
            tz = float(table_z)
        tzs.append(tz)
        keep = (pts[:, 2] > tz + 0.006) & (pts[:, 2] < tz + 0.09)
        tcp = np.asarray(frame.tcp, dtype=float)
        if tcp[2] < tz + 0.20:
            # the open fingers hang about 0-10 cm around/below the tcp: drop points near it
            keep = keep & ~((np.hypot(pts[:, 0] - tcp[0], pts[:, 1] - tcp[1]) < 0.065) & (pts[:, 2] > tcp[2] - 0.01))
        obs.append(pts[keep])
    if not obs:
        return None
    tz = float(np.median(tzs))
    ob = np.concatenate(obs, 0)
    if len(ob) < 15:
        return None
    H, xe, ye = np.histogram2d(ob[:, 0], ob[:, 1], bins=[np.arange(px - win, px + win + 1e-6, 0.02), np.arange(py - win, py + win + 1e-6, 0.02)])
    i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    sel = ob[(np.abs(ob[:, 0] - cx) < 0.07) & (np.abs(ob[:, 1] - cy) < 0.07)]
    if len(sel) < 10:
        return None
    x0, x1 = np.percentile(sel[:, 0], [3, 97])
    y0, y1 = np.percentile(sel[:, 1], [3, 97])
    xy = sel[:, :2] - sel[:, :2].mean(0)
    cov = xy.T @ xy / len(xy)
    ev, evec = np.linalg.eigh(cov)
    major = evec[:, 1]
    ang = math.atan2(major[1], major[0])
    while ang > PI / 2:
        ang -= PI
    while ang <= -PI / 2:
        ang += PI
    return {"x": float((x0 + x1) / 2), "y": float((y0 + y1) / 2), "table_z": tz,
            "top_z": float(np.percentile(sel[:, 2], 95)), "len_x": float(x1 - x0), "len_y": float(y1 - y0),
            "major_angle": float(ang), "elong": float(math.sqrt(max(ev[1], 1e-12) / max(ev[0], 1e-12))), "n": int(len(sel))}


def _drawer_contents(frame, hx, y_open, y_closed, hz):
    """Count saturated-colour pixels whose 3-D point lies inside the exposed part of an open drawer."""
    P, D = _cloud(frame, "agentview")
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    m = ((D > 0) & (np.abs(P[..., 0] - hx) < 0.10) & (P[..., 1] > y_open + 0.03) & (P[..., 1] < y_closed)
         & (P[..., 2] > hz - 0.09) & (P[..., 2] < hz + 0.04))
    c = rgb[m]
    if len(c) == 0:
        return {"region_px": 0, "colour_px": 0, "raised_px": 0}
    sat = (c.max(1) - c.min(1)) > 40
    zz = P[..., 2][m]
    floor = float(np.percentile(zz, 10))
    raised = int(((zz > floor + 0.012) & (np.abs(P[..., 0][m] - hx) < 0.07) & (P[..., 1][m] > y_open + 0.05) & (P[..., 1][m] < y_closed - 0.03)).sum())
    return {"region_px": int(m.sum()), "colour_px": int(sat.sum()), "raised_px": raised}


def _pose(robot):
    f = robot.now()
    t = np.asarray(f.tcp, dtype=float)
    return t[:3].copy(), _rv2m(t[3:6]), f


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _cmd(robot, a, r, g):
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(g))


def _goto(robot, target, R, g, tol=0.006, rtol=0.06, max_steps=150, vmax=1.0, gain=15.0, settle=0):
    """Closed-loop move of the measured TCP to target position/orientation (world frame)."""
    target = np.asarray(target, dtype=float)
    n = 0
    hist = []
    for n in range(max_steps):
        p, Rc, _ = _pose(robot)
        e = target - p
        w = _m2rv(R @ Rc.T)
        en, wn = float(np.linalg.norm(e)), float(np.linalg.norm(w))
        if en < tol and wn < rtol:
            break
        hist.append(p)
        # stalled close to the goal (gravity / reach / contact steady-state error): accept
        if len(hist) > 10 and float(np.max(np.abs(hist[-1] - hist[-9]))) < 0.0015 and en < max(3 * tol, 0.03) and wn < max(3 * rtol, 0.2):
            break
        # stuck far from the goal (e.g. a joint limit): give up this waypoint instead of burning steps
        if len(hist) > 25 and float(np.max(np.abs(hist[-1] - hist[-21]))) < 0.002:
            break
        a = e * gain
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        r = np.clip(w * 2.0, -0.5, 0.5)
        _cmd(robot, a, r, g)
    for _ in range(settle):
        p, Rc, _ = _pose(robot)
        e = target - p
        w = _m2rv(R @ Rc.T)
        _cmd(robot, np.clip(e * gain, -vmax, vmax), np.clip(w * 2.0, -0.5, 0.5), g)
    p, Rc, _ = _pose(robot)
    return {"steps": n, "err": float(np.linalg.norm(target - p)), "rot_err": float(np.linalg.norm(_m2rv(R @ Rc.T)))}


def _gripper(robot, g, steps=14, hold=None, R=None):
    """Open (g=-1) or close (g=+1) while holding the current pose."""
    p, Rc, _ = _pose(robot)
    hold = p if hold is None else np.asarray(hold, dtype=float)
    R = Rc if R is None else R
    for _ in range(steps):
        q, Rq, _ = _pose(robot)
        a = np.clip((hold - q) * 15.0, -0.3, 0.3)
        r = np.clip(_m2rv(R @ Rq.T), -0.3, 0.3)
        _cmd(robot, a, r, g)
    return _width(robot.now())


def _slide(robot, axis_target_y, x, z, R, g, max_steps=140, gain=10.0, vmax=1.0, min_steps=15, stall_win=10, stall_eps=0.0015, min_width=None):
    """Move along y towards axis_target_y while holding x, z; stop when y stalls (drawer limit) or, if
    min_width is given, when the fingers close below it (handle slipped out)."""
    ys = []
    slipped = False
    y_slip = None
    y_ok = None
    for n in range(max_steps):
        p, Rc, f = _pose(robot)
        ys.append(float(p[1]))
        if min_width is not None:
            w = _width(f)
            if w >= 0.015:
                y_ok = float(p[1])
            if w < min_width:
                slipped = True
                y_slip = y_ok if y_ok is not None else float(p[1])
                break
        if n >= min_steps and len(ys) > stall_win and abs(ys[-1] - ys[-1 - stall_win]) < stall_eps:
            break
        if abs(axis_target_y - p[1]) < 0.003:
            break
        a = np.array([(x - p[0]) * 15.0, (axis_target_y - p[1]) * gain, (z - p[2]) * 15.0])
        a = np.clip(a, -vmax, vmax)
        r = np.clip(_m2rv(R @ Rc.T), -0.3, 0.3)
        _cmd(robot, a, r, g)
    p, _, _ = _pose(robot)
    return {"steps": len(ys), "y_end": float(p[1]), "y_min": float(min(ys + [p[1]])), "y_max": float(max(ys + [p[1]])),
            "slipped": slipped, "y_slip": y_slip}
