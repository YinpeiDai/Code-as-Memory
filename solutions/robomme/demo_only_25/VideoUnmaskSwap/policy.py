"""VideoUnmaskSwap policy: track cup-like containers through the instruction video, then lift the ones hiding
the requested cubes (shell game)."""

HOVER_Z = 0.20
GRASP_Z = 0.063
RELEASE_Z = 0.068
TABLE_X = 0.30
TABLE_Y = 0.35
CUBE_COLORS = ("red", "green", "blue")
CLUSTER_R = 8.0      # px, container half-diagonal in the home-pose wrist view is ~7 px
MERGE_R = 10.0       # px
GRASP_OFFSET = (-0.006, 0.0)  # m, median (expert grasp - visual top-face centre) over 38 recorded grasps
TOP_BAND = 0.008     # m, depth band below the highest point treated as the top face


def _goal_colors(goal):
    text = str(goal).lower()
    out = []
    key = "hiding the "
    pos = 0
    while True:
        i = text.find(key, pos)
        if i < 0:
            break
        word = text[i + len(key):].split(" ")[0].strip(",.;")
        if word in CUBE_COLORS:
            out.append(word)
        pos = i + len(key)
    if not out:
        for c in CUBE_COLORS:
            if c in text:
                out.append(c)
    return out


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (P - t) @ R


def _world(frame, view):
    depth = getattr(frame, view + "_depth", None)
    if depth is None:
        return None, None
    d = np.asarray(depth)
    if (d > 0).sum() < 100:
        return None, None
    W = _backproject(d, np.asarray(getattr(frame, view + "_K")), np.asarray(getattr(frame, view + "_extrinsic")))
    return W, d > 0


def _white(im):
    im = np.asarray(im).astype(int)
    return (im.min(-1) > 140) & (im.max(-1) - im.min(-1) < 45)


def _color(im, name):
    im = np.asarray(im).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if name == "red":
        return (r > 150) & (g < 80) & (b < 80)
    if name == "green":
        return (g > 130) & (r < 110) & (b < 90)
    return (b > 150) & (r < 80) & (g < 80)


def _roi(W, valid):
    """Pixels on/above the table workspace (excludes robot fingers and background)."""
    return valid & (np.abs(W[..., 0]) < TABLE_X) & (np.abs(W[..., 1]) < TABLE_Y) & (W[..., 2] > -0.02) & (W[..., 2] < 0.12)


def _cluster(P, r, merge, minpts):
    rem = P.copy()
    cents = []
    guard = 0
    while len(rem) > minpts and guard < 50:
        guard += 1
        c = rem[0]
        for _ in range(8):
            m = np.linalg.norm(rem - c, axis=1) < r
            c = rem[m].mean(0)
        m = np.linalg.norm(rem - c, axis=1) < r
        if m.sum() > minpts:
            cents.append(c)
        rem = rem[~m] if m.sum() > 0 else rem[1:]
    # merge fragments
    merged = []
    for c in cents:
        for k in range(len(merged)):
            if np.linalg.norm(merged[k][0] - c) < merge:
                merged[k] = ((merged[k][0] * merged[k][1] + c) / (merged[k][1] + 1), merged[k][1] + 1)
                break
        else:
            merged.append((c, 1))
    return np.array([m[0] for m in merged]) if merged else np.zeros((0, 2))


def _assign(P, cent, iters=5):
    c = cent.copy()
    lab = np.zeros(len(P), dtype=int)
    for _ in range(iters):
        lab = np.argmin(((P[:, None, :] - c[None]) ** 2).sum(-1), 1)
        c = np.array([P[lab == k].mean(0) if (lab == k).sum() > 5 else c[k] for k in range(len(c))])
    return c, lab


def _components(mask, min_px=40):
    """Connected components (4-neighbour min-label propagation). Returns list of (N, 2) [u, v] point arrays."""
    H, W = mask.shape
    big = H * W + 1
    lab = np.where(mask, np.arange(H * W).reshape(H, W), big)
    for _ in range(200):
        p = np.pad(lab, 1, constant_values=big)
        nb = np.minimum(np.minimum(p[:-2, 1:-1], p[2:, 1:-1]), np.minimum(p[1:-1, :-2], p[1:-1, 2:]))
        new = np.where(mask, np.minimum(lab, nb), big)
        if (new == lab).all():
            break
        lab = new
    vs, us = np.nonzero(mask)
    ids = lab[vs, us]
    out = []
    for i in np.unique(ids):
        sel = ids == i
        if sel.sum() >= min_px:
            out.append(np.stack([us[sel], vs[sel]], 1).astype(float))
    return out


def _init_centroids(mask):
    """One centroid per container: components, with oversized blobs (touching containers) split by k-means."""
    comps = _components(mask)
    if not comps:
        return np.zeros((0, 2))
    sizes = np.array([len(c) for c in comps], dtype=float)
    unit = float(np.median(sizes))
    cents = []
    for c in comps:
        k = int(round(len(c) / unit)) if unit > 0 else 1
        k = max(1, min(k, 4))
        if k == 1:
            cents.append(c.mean(0))
            continue
        # k-means seeded along the principal axis
        mu = c.mean(0)
        u, sv, vt = np.linalg.svd(c - mu, full_matrices=False)
        proj = (c - mu) @ vt[0]
        seeds = np.array([mu + vt[0] * q for q in np.linspace(proj.min(), proj.max(), k + 2)[1:-1]])
        cc, _ = _assign(c, seeds, iters=10)
        for x in cc:
            cents.append(x)
    return np.array(cents)


def _prune(P, cent):
    """Drop fragment clusters much smaller than a typical container, then re-fit."""
    for _ in range(3):
        c, lab = _assign(P, cent)
        cnt = np.bincount(lab, minlength=len(c))
        med = float(np.median(cnt))
        keep = cnt >= 0.55 * med
        if keep.all() or keep.sum() == 0:
            return c
        cent = c[keep]
    return _assign(P, cent)[0]


def _orientation(P2):
    """Yaw (rad, in (-pi/4, pi/4]) of a square footprint from xy points via min-area box."""
    best = None
    for a in range(0, 90):
        r = math.radians(a)
        R = np.array([[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]])
        Q = P2 @ R.T
        ext = Q.max(0) - Q.min(0)
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            best = (area, a)
    a = best[1]
    if a > 45:
        a -= 90
    return math.radians(a)


def _track_video(frames, colors):
    """Track containers in the static home-pose wrist view in pixel space. Returns dict with final pixel centroids
    and the container index owning each cube color, or None."""
    ref = None
    for f in list(frames)[::-1]:
        W, valid = _world(f, "wrist")
        if W is not None:
            ref = (W, valid)
            break
    if ref is None:
        return None
    roi = _roi(ref[0], ref[1])
    vs, us = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    cube_px = {}
    cent = None
    vel = None
    first = None
    n_seen = 0
    owner = {}
    for f in frames:
        im = f.wrist_rgb
        m = _white(im) & roi
        n = int(m.sum())
        if cent is None:
            for c in CUBE_COLORS:
                cm = _color(im, c) & roi
                if cm.sum() > 4:
                    cube_px[c] = np.array([us[cm].mean(), vs[cm].mean()])
        if n < 30:
            continue
        P = np.stack([us[m], vs[m]], 1).astype(float)
        if cent is None:
            n_seen += 1
            if n_seen < 3:
                continue
            cent = _init_centroids(m)
            if len(cent) == 0:
                cent = None
                continue
            vel = np.zeros_like(cent)
            for c, p in cube_px.items():
                owner[c] = int(np.argmin(np.linalg.norm(cent - p, axis=1)))
        c2, lab = _assign(P, cent + vel)
        vel = 0.5 * vel + 0.5 * (c2 - cent)
        cent = c2
    if cent is None:
        return None
    return {"cent_px": cent, "owner": owner, "cubes_px": cube_px}


def _px_to_world(frame, cent_px, yaw_too=True):
    """Map pixel centroids in the wrist view of `frame` to world xy using that frame's depth (top faces)."""
    W, valid = _world(frame, "wrist")
    out = []
    if W is None:
        return out
    roi = _roi(W, valid) & _white(frame.wrist_rgb) & (W[..., 2] > 0.02)
    vs, us = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
    for c in cent_px:
        near = roi & ((us - c[0]) ** 2 + (vs - c[1]) ** 2 < (CLUSTER_R + 2) ** 2)
        Q = W[near]
        if len(Q) < 10:
            out.append(None)
            continue
        top = Q[Q[:, 2] > Q[:, 2].max() - TOP_BAND]
        xy = top[:, :2].mean(0)
        out.append([float(xy[0]), float(xy[1]), float(_orientation(top[:, :2])) if len(top) > 8 else 0.0])
    return out


def _locate_near(frame, x, y, radius=0.045):
    """Refine a container's world xy/yaw from the wrist view near expected (x, y)."""
    W, valid = _world(frame, "wrist")
    if W is None:
        return None
    m = _roi(W, valid) & _white(frame.wrist_rgb) & (W[..., 2] > 0.02)
    Q = W[m]
    if len(Q) < 20:
        return None
    Q = Q[np.linalg.norm(Q[:, :2] - np.array([x, y]), axis=1) < radius]
    if len(Q) < 20:
        return None
    zt = Q[:, 2].max()
    top = Q[Q[:, 2] > zt - TOP_BAND]
    if len(top) < 15:
        return None
    xy = top[:, :2].mean(0)
    # recentre once with the new estimate
    Q2 = top[np.linalg.norm(top[:, :2] - xy, axis=1) < 0.032]
    if len(Q2) >= 15:
        top = Q2
        xy = top[:, :2].mean(0)
    return [float(xy[0]), float(xy[1]), float(_orientation(top[:, :2])), float(zt), int(len(top))]


def _plan_targets(frames, goal):
    colors = _goal_colors(goal)
    demo = [f for f in frames if getattr(f, "phase", "exec") == "demo"]
    cur = frames[-1]
    tr = _track_video(demo + [cur] if demo else list(frames), colors)
    targets = []
    info = {"colors": colors, "n_demo": len(demo)}
    if tr is None:
        info["error"] = "no video tracking"
        return targets, info
    world = _px_to_world(cur, tr["cent_px"])
    info["n_containers"] = len(tr["cent_px"])
    for c in colors:
        k = tr["owner"].get(c)
        if k is None or world[k] is None:
            continue
        targets.append(world[k])
    return targets, info


def _fallback_targets(frame):
    W, valid = _world(frame, "wrist")
    if W is None:
        return []
    m = _roi(W, valid) & _white(frame.wrist_rgb) & (W[..., 2] > 0.02)
    Q = W[m]
    if len(Q) < 20:
        return []
    cents = _cluster(Q[:, :2], 0.035, 0.045, 15)
    return [[float(c[0]), float(c[1]), 0.0] for c in cents]


def _last_text(last):
    if last is None:
        return ""
    return str(last).lower().replace(" ", "")


def perceive_decide(obs, goal, memory, last):
    """Shell-game policy. On the first decision, track white containers through the task video (wrist camera, home
    pose), assign each cube colour to the container placed over it, and convert final container positions to world
    coordinates with depth. Then pick up the container hiding each requested colour in goal order; all but the last
    are put back down. A missed grasp (fingers fully closed) is retried after re-perception."""
    frames = list(obs.frames)
    mem = dict(memory)
    targets = mem.get("targets")
    if targets is None:
        targets, info = _plan_targets(frames, goal)
        if not targets:
            fb = _fallback_targets(obs.current)
            targets = fb[:max(1, len(_goal_colors(goal)))] if fb else []
        mem["targets"] = targets
        mem["next_index"] = 0
        mem["attempts"] = 0
        mem["note"] = str(info)[:200]
    nxt = mem.get("next_index") or 0
    attempts = mem.get("attempts") or 0
    text = _last_text(last)
    if mem.get("pending") == "pick" and text:
        if '"grasped":true' in text:
            nxt += 1
            attempts = 0
        else:
            attempts += 1
            if attempts >= 3:
                nxt += 1
                attempts = 0
    mem["next_index"] = nxt
    mem["attempts"] = attempts
    if not targets:
        mem["pending"] = "hold"
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"gripper": 1.0}}}
    if nxt >= len(targets):
        # all targets lifted but the episode continues: keep holding the last container where it is
        mem["pending"] = "hold"
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"gripper": -1.0}}}
    else:
        idx = nxt
        final = idx == len(targets) - 1
    t = targets[idx]
    mem["pending"] = "pick"
    return {"memory": mem, "act": {"tool": "act_pick_container",
                                   "args": {"x": float(t[0]), "y": float(t[1]), "yaw": float(t[2]),
                                            "put_back": (not final)}}}


def _move(robot, target, yaw, grip, max_lin=0.012, max_yaw=0.06, tol=0.003, settle=25):
    f = robot.now()
    p = np.array(f.tcp[:3], dtype=float)
    y0 = float(f.tcp[5])
    goal = np.array(target, dtype=float)
    dy = (yaw - y0 + math.pi) % (2 * math.pi) - math.pi
    n = int(max(1, math.ceil(max(np.linalg.norm(goal - p) / max_lin, abs(dy) / max_yaw))))
    for i in range(1, n + 1):
        q = p + (goal - p) * i / n
        robot.step(q[0], q[1], q[2], math.pi, 0.0, y0 + dy * i / n, grip)
    for _ in range(settle):
        f = robot.now()
        if np.linalg.norm(np.array(f.tcp[:3]) - goal) < tol:
            break
        robot.step(goal[0], goal[1], goal[2], math.pi, 0.0, y0 + dy, grip)
    return robot.now()


def _hold(robot, n, grip):
    f = robot.now()
    p = f.tcp
    for _ in range(n):
        robot.step(p[0], p[1], p[2], math.pi, 0.0, p[5], grip)
    return robot.now()


def act_pick_container(robot, x, y, yaw, put_back=False):
    """Hover above (x, y) at 0.2 m, refine the container centre/yaw from the wrist depth view, descend to grasp
    height, close, lift to 0.2 m. If put_back, lower it again, open and retreat to 0.15 m. Returns grasped flag."""
    _move(robot, [x, y, HOVER_Z], yaw, 1.0)
    ref = _locate_near(robot.now(), x, y)
    if ref is not None and math.hypot(ref[0] - x, ref[1] - y) < 0.04:
        x, y, yaw = ref[0], ref[1], ref[2]
        _move(robot, [x, y, HOVER_Z], yaw, 1.0)
        ref2 = _locate_near(robot.now(), x, y, radius=0.035)
        if ref2 is not None and math.hypot(ref2[0] - x, ref2[1] - y) < 0.02:
            x, y, yaw = ref2[0], ref2[1], ref2[2]
    x = x + GRASP_OFFSET[0]
    y = y + GRASP_OFFSET[1]
    _move(robot, [x, y, HOVER_Z], yaw, 1.0)
    _move(robot, [x, y, GRASP_Z + 0.04], yaw, 1.0)
    _move(robot, [x, y, GRASP_Z], yaw, 1.0, max_lin=0.006, tol=0.002)
    f = _hold(robot, 15, -1.0)
    width = float(np.mean(f.fingers))
    grasped = width > 0.004
    if not grasped:
        _hold(robot, 8, 1.0)
        _move(robot, [x, y, HOVER_Z], yaw, 1.0)
        return {"grasped": False, "finger": width, "x": x, "y": y, "yaw": yaw}
    _move(robot, [x, y, HOVER_Z], yaw, -1.0)
    f = _hold(robot, 10, -1.0)
    width2 = float(np.mean(f.fingers))
    if width2 < 0.004:
        _hold(robot, 8, 1.0)
        return {"grasped": False, "dropped": True, "finger": width2, "x": x, "y": y}
    if put_back:
        _move(robot, [x, y, RELEASE_Z], yaw, -1.0, max_lin=0.01)
        _hold(robot, 12, 1.0)
        _move(robot, [x, y, 0.15], yaw, 1.0)
    return {"grasped": True, "finger": width2, "x": x, "y": y, "yaw": yaw, "put_back": bool(put_back)}


def act_hold(robot, gripper=1.0):
    """Hold the current pose for a few steps with the given gripper command (-1 keeps a grasp closed)."""
    _hold(robot, 10, -1.0 if gripper < 0 else 1.0)
    return {"held": True}
