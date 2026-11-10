"""VideoUnmask policy: remember where each coloured cube was in the instruction
video, then pick up the container that now covers it (and, for two-target goals,
put the first container back down before picking the second)."""

COLORS = ("red", "green", "blue")
GRASP_X_OFFSET = -0.006      # rect-centre -> expert TCP offset (demo median, metres)
GRASP_DEPTH = 0.015          # TCP below container top when closing
SAFE_Z = 0.20                # transfer height used by the expert
RETREAT_Z = 0.145            # height after putting a container back down
ROLL, PITCH = math.pi, 0.0
MISS_WIDTH = 0.008           # finger width below this after closing = empty grasp
TABLE_LIMIT = 0.35


def _goal_colors(goal):
    """Colours in the order they are mentioned in the goal text."""
    text = str(goal).lower()
    found = []
    for c in COLORS:
        start = 0
        while True:
            i = text.find(c, start)
            if i < 0:
                break
            found.append((i, c))
            start = i + 1
    found.sort()
    return [c for _, c in found]


def _color_mask(rgb, color):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    if color == "red":
        return (r > 120) & (g < 70) & (b < 70)
    if color == "green":
        return (g > 100) & (r < 90) & (b < 90)
    return (b > 100) & (r < 70) & (g < 70)


def _world_cloud(frame):
    """HxWx3 world points from front depth and calibration, plus validity mask."""
    depth = getattr(frame, "front_depth", None)
    if depth is None:
        return None, None
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    d = depth.astype(float) / 1000.0
    x = (u + 0.5 - K[0, 2]) * d / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    w = (pc - E[:, 3]) @ E[:, :3]
    return w.reshape(H, W, 3), d > 0


def _pixel_to_plane(frame, u, v, z):
    """Intersect the camera ray through pixel (u, v) with the plane Z = z."""
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    R, t = E[:, :3], E[:, 3]
    ray_c = np.array([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], 1.0])
    origin = -R.T @ t
    ray = R.T @ ray_c
    if abs(ray[2]) < 1e-9:
        return None
    s = (z - origin[2]) / ray[2]
    return origin + s * ray


def _cube_positions(frame):
    """World XY of each visible coloured cube (top-face centroid)."""
    w, valid = _world_cloud(frame)
    out = {}
    for c in COLORS:
        m = _color_mask(frame.front_rgb, c)
        if w is not None:
            mm = m & valid & (w[..., 2] > 0.005) & (w[..., 2] < 0.15)
            if mm.sum() >= 5:
                p = w[mm]
                top = p[p[:, 2] > p[:, 2].max() - 0.006]
                out[c] = [float(top[:, 0].mean()), float(top[:, 1].mean())]
                continue
        if m.sum() >= 5:
            ys, xs = np.nonzero(m)
            mu, mv = np.median(xs), np.median(ys)
            keep = np.hypot(xs - mu, ys - mv) < 8
            if keep.sum() < 5:
                continue
            p = _pixel_to_plane(frame, xs[keep].mean(), ys[keep].mean(), 0.02)
            if p is not None:
                out[c] = [float(p[0]), float(p[1])]
    return out


def _rect_fit(xy):
    """Minimum-area rectangle: (angle in [0, pi/2), centre xy, side lengths)."""
    mean = xy.mean(0)
    q = xy - mean
    best = None
    for deg in range(0, 90):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        Rm = np.array([[c, -s], [s, c]])
        r = q @ Rm
        lo, hi = r.min(0), r.max(0)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if best is None or area < best[0]:
            mid = ((lo + hi) / 2.0) @ Rm.T + mean
            best = (area, a, mid, hi - lo)
    # refine around the best angle
    a0 = best[1]
    for k in range(-10, 11):
        a = a0 + math.radians(k * 0.1)
        c, s = math.cos(a), math.sin(a)
        Rm = np.array([[c, -s], [s, c]])
        r = q @ Rm
        lo, hi = r.min(0), r.max(0)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if area < best[0]:
            mid = ((lo + hi) / 2.0) @ Rm.T + mean
            best = (area, a, mid, hi - lo)
    return best[1], best[2], best[3]


def _wrap_quarter(a):
    """Map an angle to (-pi/4, pi/4] (square container symmetry)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def _container_near(frame, xy):
    """Container top-face pose near xy: [x, y, ztop, yaw, npoints] or None."""
    w, valid = _world_cloud(frame)
    if w is None:
        return None
    p = w[valid]
    p = p[(p[:, 2] > 0.045) & (p[:, 2] < 0.12)]
    centre = np.array(xy, dtype=float)
    result = None
    for rad in (0.04, 0.032, 0.03):
        sel = p[np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) < rad]
        if len(sel) < 25:
            return result
        ztop = float(np.percentile(sel[:, 2], 95))
        top = sel[sel[:, 2] > ztop - 0.008]
        if len(top) < 20:
            return result
        ang, mid, sides = _rect_fit(top[:, :2])
        centre = mid
        result = [float(mid[0]), float(mid[1]), ztop, float(_wrap_quarter(ang)), int(len(top))]
    return result


def perceive_scene(obs, goal):
    """Facts for the goal: ordered target colours, cube XY remembered from the
    instruction video, and the covering container pose in the latest frame."""
    colors = _goal_colors(goal)
    cubes = {}
    for f in obs.frames:
        if f.phase != "demo":
            continue
        found = _cube_positions(f)
        if all(c in found for c in colors) or len(found) > len(cubes):
            cubes = found
            if all(c in found for c in colors):
                break
    if not cubes:  # no instruction video: maybe cubes are still visible
        cubes = _cube_positions(obs.current)
    targets = []
    for c in colors:
        if c not in cubes:
            targets.append(None)
            continue
        cont = _container_near(obs.current, cubes[c])
        if cont is None:
            targets.append([cubes[c][0], cubes[c][1], 0.074, 0.0])
        else:
            targets.append(cont[:4])
    return {"colors": colors, "cubes": cubes, "targets": targets}


def perceive_decide(obs, goal, memory, last):
    """Plan once from the instruction video and first execution frame, then pick
    each goal container in order; retry a missed grasp a little lower."""
    mem = dict(memory)
    for k in ("colors", "targets", "yaws", "stage", "attempts", "status"):
        mem.setdefault(k, None)
    if mem["targets"] is None:
        scene = perceive_scene(obs, goal)
        tg = [t for t in scene["targets"]]
        # unknown target: fall back to any detected cube location not already used
        known = [t for t in tg if t is not None]
        tg = [t if t is not None else (known[0] if known else [0.0, 0.0, 0.074, 0.0]) for t in tg]
        if not tg:
            tg = [[0.0, 0.0, 0.074, 0.0]]
        mem["colors"] = scene["colors"]
        mem["targets"] = [[round(t[0], 4), round(t[1], 4), round(t[2], 4)] for t in tg]
        mem["yaws"] = ",".join("%.4f" % t[3] for t in tg)
        mem["stage"] = 0
        mem["attempts"] = 0
        mem["status"] = "planned"
    elif last is not None:
        text = str(last).lower().replace("'", '"').replace(" ", "")
        grasped = '"grasped":true' in text
        if mem["status"] == "picking":
            if grasped:
                mem["stage"] = int(mem["stage"]) + 1
                mem["attempts"] = 0
            else:
                mem["attempts"] = int(mem["attempts"]) + 1
                if mem["attempts"] > 3:
                    mem["stage"] = int(mem["stage"]) + 1
                    mem["attempts"] = 0
    n = len(mem["targets"])
    stage = int(mem["stage"])
    if stage >= n:
        mem["status"] = "done"
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 30}}}
    t = mem["targets"][stage]
    yaw = float(str(mem["yaws"]).split(",")[stage])
    mem["status"] = "picking"
    args = {"x": float(t[0]) + GRASP_X_OFFSET, "y": float(t[1]), "z_top": float(t[2]),
            "yaw": yaw, "put_back": stage < n - 1,
            "extra_depth": min(0.01, 0.005 * int(mem["attempts"]))}
    return {"memory": mem, "act": {"tool": "act_pick_container", "args": args}}


def _go(robot, x, y, z, yaw, grip, speed=0.01, yaw_speed=0.05, settle=20, tol=0.004):
    """Linear Cartesian interpolation from the measured pose, then settle."""
    f = robot.now()
    p0 = np.array(f.tcp[:3], dtype=float)
    y0 = float(f.tcp[5])
    dyaw = (yaw - y0 + math.pi) % (2 * math.pi) - math.pi
    goal = np.array([x, y, z], dtype=float)
    n = int(max(1, math.ceil(max(np.linalg.norm(goal - p0) / speed, abs(dyaw) / yaw_speed))))
    for i in range(1, n + 1):
        s = i / n
        p = p0 + (goal - p0) * s
        robot.step(p[0], p[1], p[2], ROLL, PITCH, y0 + dyaw * s, grip)
    for _ in range(settle):
        f = robot.now()
        err = np.linalg.norm(np.array(f.tcp[:3], dtype=float) - goal)
        if err < tol:
            break
        robot.step(x, y, z, ROLL, PITCH, yaw, grip)
    return robot.now()


def _hold(robot, grip, steps, pose=None):
    t = list(robot.now().tcp) if pose is None else list(pose)
    for _ in range(steps):
        robot.step(t[0], t[1], t[2], ROLL, PITCH, t[5], grip)
    return robot.now()


def act_pick_container(robot, x, y, z_top, yaw, put_back=False, extra_depth=0.0):
    """Open, move above (x, y) at transfer height with gripper yaw `yaw`, descend
    to z_top - 0.015 - extra_depth, close, lift. If put_back, lower it again,
    release and retreat. Returns grasped (finger width check) and widths."""
    x = float(np.clip(x, -TABLE_LIMIT, TABLE_LIMIT))
    y = float(np.clip(y, -TABLE_LIMIT, TABLE_LIMIT))
    zg = float(max(0.04, z_top - GRASP_DEPTH - extra_depth))
    f = robot.now()
    if float(f.tcp[2]) < SAFE_Z - 0.02:
        _go(robot, f.tcp[0], f.tcp[1], SAFE_Z, f.tcp[5], 1.0, settle=5)
    _go(robot, x, y, SAFE_Z, yaw, 1.0, speed=0.012)
    _go(robot, x, y, zg + 0.04, yaw, 1.0, speed=0.01)
    _go(robot, x, y, zg, yaw, 1.0, speed=0.005, settle=25, tol=0.003)
    f = _hold(robot, -1.0, 15, [x, y, zg, 0, 0, yaw])
    width_closed = float(np.mean(f.fingers))
    grasped = width_closed > MISS_WIDTH
    if not grasped:
        _hold(robot, 1.0, 10, [x, y, zg, 0, 0, yaw])
        _go(robot, x, y, SAFE_Z, yaw, 1.0, speed=0.012, settle=5)
        return {"grasped": False, "width": width_closed, "z_grasp": zg}
    f = _go(robot, x, y, SAFE_Z, yaw, -1.0, speed=0.01, settle=10)
    width_lift = float(np.mean(f.fingers))
    grasped = width_lift > MISS_WIDTH
    if grasped and put_back:
        _go(robot, x, y, zg + 0.005, yaw, -1.0, speed=0.01, settle=15)
        _hold(robot, 1.0, 12, [x, y, zg + 0.005, 0, 0, yaw])
        _go(robot, x, y, RETREAT_Z, yaw, 1.0, speed=0.01, settle=5)
    elif grasped:
        _hold(robot, -1.0, 10, [x, y, SAFE_Z, 0, 0, yaw])
    return {"grasped": bool(grasped), "width": width_lift, "z_grasp": zg, "put_back": bool(put_back)}


def act_hold(robot, steps=30):
    """Hold the current pose and gripper state for `steps` control steps."""
    f = robot.now()
    grip = -1.0 if float(np.mean(f.fingers)) < 0.03 else 1.0
    _hold(robot, grip, int(max(1, min(steps, 200))))
    return {"held": int(steps)}
