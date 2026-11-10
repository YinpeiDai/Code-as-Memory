"""ButtonUnmask policy: memorize cube colours before the containers drop, press the
button, then lift the container(s) covering the goal-coloured cube(s)."""

COLORS = ("red", "green", "blue")
BIN_TOP_LO = 0.066          # container top surface is at z~0.074 m
BIN_TOP_HI = 0.09
BIN_RADIUS = 0.028          # half-diagonal of the 0.039 m square top
GRASP_DX = -0.0085          # expert grasp TCP sits ~8-9 mm -x of the measured container centre
PRESS_DX = -0.0125          # expert press TCP sits ~12 mm -x of the measured button base centre
GRASP_Z = 0.063
HOVER_Z = 0.20
PRESS_HOVER_Z = 0.15
PRESS_Z = 0.010
DEFAULT_BUTTON = [-0.19, 0.0]


def _cloud(depth, K, E):
    """Back-project an aligned optical-axis depth image to world XYZ (H,W,3) and validity mask."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u + 0.5 - K[0][2]) / K[0][0] * z
    y = (v + 0.5 - K[1][2]) / K[1][1] * z
    pc = np.stack([x, y, z], -1)
    E = np.asarray(E, dtype=float)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, z > 0


def _color_masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return {"red": (r > 120) & (g < 45) & (b < 45),
            "green": (g > 120) & (r < 45) & (b < 45),
            "blue": (b > 120) & (r < 45) & (g < 45)}


def _detect_cubes(frame):
    """World XY of saturated red/green/blue cube top faces in the front view."""
    P, ok = _cloud(frame.front_depth, frame.front_K, frame.front_extrinsic)
    out = {}
    for name, m in _color_masks(frame.front_rgb).items():
        m = m & ok & (P[..., 2] > 0.026) & (P[..., 2] < 0.05)
        if int(m.sum()) >= 4:
            q = P[m]
            out[name] = [float(q[:, 0].mean()), float(q[:, 1].mean())]
    return out


def _detect_button(frame):
    """World XY of the white button base (flat top at z~0.016) near the robot side of the table."""
    P, ok = _cloud(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgb = frame.front_rgb
    m = (rgb.min(2) > 170) & ok & (P[..., 2] > 0.004) & (P[..., 2] < 0.03) \
        & (P[..., 0] > -0.40) & (P[..., 0] < -0.06) & (np.abs(P[..., 1]) < 0.15)
    if int(m.sum()) < 15:
        return None
    q = P[m]
    zt = float(np.percentile(q[:, 2], 95))
    top = q[q[:, 2] > zt - 0.004]
    c = np.median(top[:, :2], 0)
    near = top[np.linalg.norm(top[:, :2] - c, axis=1) < 0.06]
    if len(near) < 10:
        near = top
    return [float(near[:, 0].mean()), float(near[:, 1].mean())]


def _rect_angle(xy):
    """Yaw (rad, in [-pi/4, pi/4)) of the minimum-area square fitting the points."""
    c = xy.mean(0)
    best = None
    for deg in range(0, 90):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        p0 = (xy[:, 0] - c[0]) * ca + (xy[:, 1] - c[1]) * sa
        p1 = -(xy[:, 0] - c[0]) * sa + (xy[:, 1] - c[1]) * ca
        e0 = np.percentile(p0, 97) - np.percentile(p0, 3)
        e1 = np.percentile(p1, 97) - np.percentile(p1, 3)
        area = e0 * e1
        if best is None or area < best[0]:
            best = (area, a)
    a = best[1]
    if a >= math.pi / 4:
        a -= math.pi / 2
    return a


def _bin_from_cloud(P, ok, guess, min_pts):
    """Mean-shift on container-top points around guess; returns (xy, yaw, npts) or None."""
    m = ok & (P[..., 2] > BIN_TOP_LO) & (P[..., 2] < BIN_TOP_HI)
    pts = P[m][:, :2]
    if len(pts) < min_pts:
        return None
    c = np.array(guess, dtype=float)
    sel = pts[np.linalg.norm(pts - c, axis=1) < 0.045]
    if len(sel) < min_pts:
        return None
    for radius in (0.04, 0.032, BIN_RADIUS, BIN_RADIUS):
        sel = pts[np.linalg.norm(pts - c, axis=1) < radius]
        if len(sel) < min_pts:
            return None
        c = sel.mean(0)
    sel = pts[np.linalg.norm(pts - c, axis=1) < BIN_RADIUS + 0.002]
    return [float(c[0]), float(c[1])], _rect_angle(sel), int(len(sel))


def _bin_front(frame, guess):
    P, ok = _cloud(frame.front_depth, frame.front_K, frame.front_extrinsic)
    return _bin_from_cloud(P, ok, guess, 25)


def _bin_wrist(frame, guess):
    P, ok = _cloud(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    return _bin_from_cloud(P, ok, guess, 150)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _move(robot, target, grip, step=0.012, rstep=0.06, settle=40, tol=0.004):
    """Interpolate the TCP to target [x,y,z,yaw] with roll=pi, pitch=0, then settle."""
    f = robot.now()
    cur = np.array(f.tcp[:3], dtype=float)
    yaw0 = float(f.tcp[5])
    tgt = np.array(target[:3], dtype=float)
    dyaw = _wrap(float(target[3]) - yaw0)
    n = int(max(1, math.ceil(max(np.linalg.norm(tgt - cur) / step, abs(dyaw) / rstep))))
    for i in range(1, n + 1):
        s = i / n
        p = cur + (tgt - cur) * s
        robot.step(p[0], p[1], p[2], math.pi, 0.0, yaw0 + dyaw * s, grip)
    for _ in range(settle):
        f = robot.now()
        err = np.linalg.norm(np.array(f.tcp[:3]) - tgt)
        if err < tol and abs(_wrap(float(f.tcp[5]) - float(target[3]))) < 0.03:
            break
        robot.step(tgt[0], tgt[1], tgt[2], math.pi, 0.0, float(target[3]), grip)
    return robot.now()


def _hold(robot, grip, n):
    f = robot.now()
    t = f.tcp
    for _ in range(n):
        robot.step(t[0], t[1], t[2], math.pi, 0.0, t[5], grip)
    return robot.now()


def _parse_goal(goal):
    """Colours of the cubes whose containers must be picked, in goal order."""
    g = (goal or "").lower()
    found = []
    pos = 0
    while True:
        k = g.find("hiding the", pos)
        if k < 0:
            break
        rest = g[k + 10:]
        best = None
        for c in COLORS:
            j = rest.find(c)
            if j >= 0 and (best is None or j < best[0]):
                best = (j, c)
        if best is not None:
            found.append(best[1])
        pos = k + 10
    if not found:
        idx = sorted((g.find(c), c) for c in COLORS if g.find(c) >= 0)
        found = [c for _, c in idx]
    return found


def act_press_button(robot, x, y):
    """Close the gripper above the button base centre (x,y), push down until blocked, rise again."""
    f = robot.now()
    yaw = 0.0
    px, py = x + PRESS_DX, y
    _move(robot, [px, py, PRESS_HOVER_Z, yaw], 1.0)
    _hold(robot, -1.0, 8)
    last_z = robot.now().tcp[2]
    stall = 0
    z = float(last_z)
    for _ in range(80):
        z = max(PRESS_Z, z - 0.008)
        f = robot.step(px, py, z, math.pi, 0.0, yaw, -1.0)
        tz = float(f.tcp[2])
        if tz < 0.035 and abs(tz - last_z) < 0.0015:
            stall += 1
        else:
            stall = 0
        last_z = tz
        if stall >= 3 or (z <= PRESS_Z and tz < 0.03):
            break
    _hold(robot, -1.0, 6)
    bottom = float(robot.now().tcp[2])
    _move(robot, [px, py, PRESS_HOVER_Z, yaw], -1.0)
    _hold(robot, 1.0, 8)
    return {"pressed": True, "bottom_z": bottom}


def act_pick_container(robot, x, y, put_back=False):
    """Lift the container covering world point (x,y). Measures the container top in the
    front view, hovers, refines centre/yaw with the wrist view, descends, grasps and lifts.
    put_back=True lowers it again and releases (used before a further pick)."""
    guess = [float(x), float(y)]
    est = _bin_front(robot.now(), guess)
    src = "front"
    if est is None:
        est = (guess, 0.0, 0)
        src = "cube_only"
    xy, yaw, npts = est
    _move(robot, [xy[0] + GRASP_DX, xy[1], HOVER_Z, yaw], 1.0)
    if npts < 60:
        # front view weak or occluded: refine with the top-down wrist view at hover
        for _ in range(2):
            w = _bin_wrist(robot.now(), xy)
            if w is None or np.linalg.norm(np.array(w[0]) - np.array(guess)) > 0.04:
                break
            xy, yaw = w[0], w[1]
            src = "wrist"
            _move(robot, [xy[0] + GRASP_DX, xy[1], HOVER_Z, yaw], 1.0, settle=15)
    gx, gy = xy[0] + GRASP_DX, xy[1]
    _move(robot, [gx, gy, GRASP_Z + 0.03, yaw], 1.0, step=0.01)
    _move(robot, [gx, gy, GRASP_Z, yaw], 1.0, step=0.005, settle=20)
    f = _hold(robot, -1.0, 12)
    width = float(np.sum(f.fingers))
    held = width > 0.01
    _move(robot, [gx, gy, HOVER_Z, yaw], -1.0, step=0.01, settle=10)
    f = robot.now()
    width2 = float(np.sum(f.fingers))
    held = held and width2 > 0.01
    if put_back and held:
        _move(robot, [gx, gy, GRASP_Z + 0.007, yaw], -1.0, step=0.01, settle=10)
        _hold(robot, 1.0, 10)
        _move(robot, [gx, gy, PRESS_HOVER_Z, yaw], 1.0, step=0.01, settle=5)
    elif not held:
        _hold(robot, 1.0, 8)
    return {"held": bool(held), "finger_width": width2, "xy": xy,
            "yaw": yaw, "source": src}


def act_observe(robot):
    """Hold the current pose with an open gripper for a few steps."""
    _hold(robot, 1.0, 5)
    return {"observed": True}


def perceive_decide(obs, goal, memory, last):
    """State machine: (1) memorize cube colours/positions from the earliest frame and the
    button; (2) press the button; (3) pick each goal container (putting back all but the
    last), retrying a missed grasp up to 2 times."""
    mem = dict(memory)
    for k in ("stage", "targets", "cube_red", "cube_green", "cube_blue", "button",
              "next_index", "retries", "notes"):
        mem.setdefault(k, None)
    if mem["targets"] is None:
        mem["targets"] = _parse_goal(goal)
    # memorize cubes from the earliest frames that show them
    for fr in list(obs.frames[:3]) + [obs.current]:
        cubes = _detect_cubes(fr)
        for c in COLORS:
            if mem["cube_" + c] is None and c in cubes:
                mem["cube_" + c] = cubes[c]
        if mem["button"] is None:
            b = _detect_button(fr)
            if b is not None:
                mem["button"] = b
    if mem["stage"] is None:
        mem["stage"] = "press"
        mem["next_index"] = 0
        mem["retries"] = 0
        bx, by = mem["button"] if mem["button"] is not None else DEFAULT_BUTTON
        return {"memory": mem, "act": {"tool": "act_press_button", "args": {"x": bx, "y": by}}}
    targets = mem["targets"] or []
    i = int(mem["next_index"] or 0)
    lt = str(last) if last is not None else ""
    if mem["stage"] == "pick" and "held" in lt:
        if ('"held": true' in lt) or ("'held': True" in lt):
            i += 1
            mem["retries"] = 0
        else:
            mem["retries"] = int(mem["retries"] or 0) + 1
            if mem["retries"] > 2:
                i += 1
                mem["retries"] = 0
        mem["next_index"] = i
    mem["stage"] = "pick"
    if i >= len(targets):
        mem["notes"] = "all targets attempted"
        return {"memory": mem, "act": {"tool": "act_observe", "args": {}}}
    c = targets[i]
    pos = mem.get("cube_" + c)
    if pos is None:
        known = [mem["cube_" + k] for k in COLORS if mem["cube_" + k] is not None]
        pos = known[0] if known else [0.0, 0.0]
        mem["notes"] = "cube " + c + " not seen; guessing"
    put_back = i < len(targets) - 1
    return {"memory": mem, "act": {"tool": "act_pick_container",
                                   "args": {"x": float(pos[0]), "y": float(pos[1]), "put_back": put_back}}}
