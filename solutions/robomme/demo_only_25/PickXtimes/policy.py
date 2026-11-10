"""PickXtimes policy: perceive colored cube / target / button with RGB-D, pick-place N times, press button."""

XOFF = -0.005          # measured TCP minus perceived world x at expert grasp/press (25 demos, std 1-4 mm)
YOFF = 0.0
HOVER_Z = 0.15
GRASP_Z = 0.02
PLACE_Z_MIN = 0.008
PRESS_Z_MIN = 0.004
HOME = (-0.06, 0.0, 0.52)
COUNT_WORDS = {"once": 1, "one": 1, "twice": 2, "two": 2, "three": 3, "thrice": 3, "four": 4, "five": 5,
               "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _parse_goal(goal):
    g = str(goal).lower()
    color = "red"
    for c in ("red", "green", "blue"):
        if (c + " cube") in g:
            color = c
    n = 1
    words = g.replace(",", " ").split()
    for i, w in enumerate(words):
        if w == "times" and i > 0:
            p = words[i - 1]
            if p.isdigit():
                n = int(p)
            elif p in COUNT_WORDS:
                n = COUNT_WORDS[p]
        if w in ("twice", "thrice"):
            n = COUNT_WORDS[w]
    return color, max(1, n)


def _backproject(frame, cam, mask):
    depth = getattr(frame, cam + "_depth", None)
    if depth is None:
        return np.zeros((0, 3))
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    v, u = np.nonzero(mask)
    d = np.asarray(depth)[v, u].astype(float) / 1000.0
    ok = d > 0
    u, v, d = u[ok], v[ok], d[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], 1)
    return (P - E[:, 3]) @ E[:, :3]


def _color_mask(rgb, name):
    rgb = np.asarray(rgb)
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    if name == "red":
        return (r > 120) & (g < 60) & (b < 60)
    if name == "green":
        return (g > 120) & (r < 80) & (b < 80)
    if name == "blue":
        return (b > 120) & (r < 60) & (g < 60)
    if name == "target":
        return (r > 140) & (b > 170) & (g < 140)
    return (np.abs(r - g) < 10) & (np.abs(g - b) < 10) & (r > 100)


def _largest_cluster(P, rad):
    best = None
    rem = P
    for _ in range(6):
        if len(rem) == 0:
            break
        c = rem[0, :2]
        for _ in range(6):
            sel = np.linalg.norm(rem[:, :2] - c, axis=1) < rad
            c = rem[sel, :2].mean(0)
        sel = np.linalg.norm(rem[:, :2] - c, axis=1) < rad
        if best is None or sel.sum() > len(best):
            best = rem[sel]
        rem = rem[~sel]
    return best


def _yaw_square(xy):
    xy = xy - xy.mean(0)
    best = None
    for a in np.arange(0.0, 90.0, 0.5):
        th = math.radians(a)
        c, s = math.cos(th), math.sin(th)
        q = xy @ np.array([[c, -s], [s, c]])
        ar = float(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
        if best is None or ar < best[0]:
            best = (ar, th)
    return _wrap90(best[1])


def _wrap90(a):
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def _detect(frame, color, near=None):
    """World estimates (perceived frame, before TCP offset) of cube, target, button."""
    out = {"cube": None, "cube_yaw": None, "target": None, "button": None}
    for cam in ("wrist", "front"):
        rgb = getattr(frame, cam + "_rgb", None)
        if rgb is None:
            continue
        if out["cube"] is None:
            P = _backproject(frame, cam, _color_mask(rgb, color))
            P = P[(P[:, 2] > -0.02) & (P[:, 2] < 0.07) & (P[:, 0] > -0.45)] if len(P) else P
            if near is not None and len(P):
                P = P[np.linalg.norm(P[:, :2] - np.asarray(near[:2]), axis=1) < 0.08]
            if len(P) >= 15:
                C = _largest_cluster(P, 0.035)
                if C is not None and len(C) >= 15:
                    top = C[C[:, 2] > C[:, 2].max() - 0.006]
                    if len(top) >= 8:
                        out["cube"] = [float(top[:, 0].mean()), float(top[:, 1].mean()), float(C[:, 2].max())]
                        out["cube_yaw"] = float(_yaw_square(top[:, :2]))
        if out["target"] is None:
            P = _backproject(frame, cam, _color_mask(rgb, "target"))
            P = P[(P[:, 2] > -0.02) & (P[:, 2] < 0.03) & (P[:, 0] > -0.45)] if len(P) else P
            if len(P) >= 15:
                C = _largest_cluster(P, 0.06)
                if C is not None and len(C) >= 15:
                    c = (C[:, :2].min(0) + C[:, :2].max(0)) / 2
                    out["target"] = [float(c[0]), float(c[1])]
        if out["button"] is None:
            P = _backproject(frame, cam, _color_mask(rgb, "gray"))
            P = P[(P[:, 2] > 0.02) & (P[:, 2] < 0.06) & (P[:, 0] > -0.45)] if len(P) else P
            if len(P) >= 30:
                C = _largest_cluster(P, 0.05)
                if C is not None and len(C) >= 30:
                    c = (C[:, :2].min(0) + C[:, :2].max(0)) / 2
                    out["button"] = [float(c[0]), float(c[1])]
    return out


def _ang(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _move(robot, x, y, z, yaw, grip, vmax=0.012, tol=0.004, settle=40):
    """Gradual straight-line move of the setpoint, then settle on measured TCP."""
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=float).copy()
    cy = float(f.tcp[5])
    goal = np.array([x, y, z], dtype=float)
    for _ in range(400):
        d = goal - p
        n = float(np.linalg.norm(d))
        dy = _ang(yaw - cy)
        if n < 1e-6 and abs(dy) < 1e-4:
            break
        if n > vmax:
            d = d * (vmax / n)
        p = p + d
        cy = cy + max(-0.06, min(0.06, dy))
        robot.step(p[0], p[1], p[2], math.pi, 0.0, cy, grip)
    for _ in range(settle):
        m = np.asarray(robot.now().tcp[:3], dtype=float)
        if np.linalg.norm(m - goal) < tol:
            break
        robot.step(goal[0], goal[1], goal[2], math.pi, 0.0, yaw, grip)
    return robot.now()


def _descend_until(robot, x, y, z_min, yaw, grip, vmax=0.008):
    """Lower the setpoint toward z_min; stop on reaching it or on stall (contact)."""
    z_set = float(robot.now().tcp[2])
    hist = []
    for _ in range(200):
        z_set = max(z_min, z_set - vmax)
        robot.step(x, y, z_set, math.pi, 0.0, yaw, grip)
        mz = float(robot.now().tcp[2])
        hist.append(mz)
        if mz < z_min + 0.003:
            break
        if len(hist) > 6 and hist[-6] - mz < 0.001 and (z_set <= z_min + 1e-6 or z_set < mz - 0.003):
            break
    return float(robot.now().tcp[2])


def _hold(robot, x, y, z, yaw, grip, n):
    for _ in range(n):
        robot.step(x, y, z, math.pi, 0.0, yaw, grip)


def act_pick(robot, x, y, yaw):
    """Pick the cube whose TCP grasp point is (x, y) with gripper yaw (rad): hover, descend, close, lift.
    Returns finger widths after lifting and whether something is held."""
    f = robot.now()
    if f.tcp[2] < HOVER_Z - 0.02:
        _move(robot, f.tcp[0], f.tcp[1], HOVER_Z, float(f.tcp[5]), 1.0)
    _move(robot, x, y, HOVER_Z, yaw, 1.0)
    _hold(robot, x, y, HOVER_Z, yaw, 1.0, 3)
    _move(robot, x, y, GRASP_Z + 0.03, yaw, 1.0, vmax=0.010)
    _move(robot, x, y, GRASP_Z, yaw, 1.0, vmax=0.006, tol=0.003)
    _hold(robot, x, y, GRASP_Z, yaw, -1.0, 15)
    _move(robot, x, y, HOVER_Z, yaw, -1.0, vmax=0.010)
    fw = [float(v) for v in robot.now().fingers]
    return {"fingers": fw, "holding": bool(sum(fw) > 0.01), "tcp": [float(v) for v in robot.now().tcp]}


def act_place(robot, x, y, yaw):
    """Carry the held cube over TCP point (x, y), lower until contact, open, lift back to hover."""
    _move(robot, x, y, HOVER_Z, yaw, -1.0)
    _hold(robot, x, y, HOVER_Z, yaw, -1.0, 3)
    _move(robot, x, y, 0.05, yaw, -1.0, vmax=0.010)
    z = _descend_until(robot, x, y, PLACE_Z_MIN, yaw, -1.0, vmax=0.004)
    _hold(robot, x, y, z, yaw, 1.0, 12)
    _move(robot, x, y, HOVER_Z, yaw, 1.0, vmax=0.010)
    _hold(robot, x, y, HOVER_Z, yaw, 1.0, 3)
    return {"released_z": z, "tcp": [float(v) for v in robot.now().tcp]}


def act_press(robot, x, y):
    """Close the gripper, go above button TCP point (x, y), press down until contact, then lift."""
    f = robot.now()
    yaw = float(f.tcp[5])
    if f.tcp[2] < HOVER_Z - 0.02:
        _move(robot, f.tcp[0], f.tcp[1], HOVER_Z, yaw, -1.0)
    _hold(robot, f.tcp[0], f.tcp[1], max(HOVER_Z, float(f.tcp[2])), yaw, -1.0, 8)
    _move(robot, x, y, HOVER_Z, yaw, -1.0)
    _move(robot, x, y, 0.06, yaw, -1.0, vmax=0.010)
    z = _descend_until(robot, x, y, PRESS_Z_MIN, yaw, -1.0, vmax=0.006)
    _hold(robot, x, y, max(PRESS_Z_MIN, z - 0.01), yaw, -1.0, 6)
    _move(robot, x, y, HOVER_Z, yaw, -1.0, vmax=0.010)
    return {"pressed_z": z, "tcp": [float(v) for v in robot.now().tcp]}


def act_observe(robot):
    """Move to the high home viewpoint (gripper open) so the wrist camera sees the whole table."""
    f = robot.now()
    grip = 1.0
    if f.tcp[2] < HOVER_Z - 0.02:
        _move(robot, f.tcp[0], f.tcp[1], HOVER_Z, float(f.tcp[5]), grip)
    _move(robot, HOME[0], HOME[1], HOME[2], 0.0, grip, tol=0.01)
    _hold(robot, HOME[0], HOME[1], HOME[2], 0.0, grip, 5)
    return {"tcp": [float(v) for v in robot.now().tcp]}


def _choose_yaw(cube_yaw, cur):
    if cube_yaw is None:
        return _wrap90(cur)
    best = None
    for k in range(-4, 5):
        a = cube_yaw + k * math.pi / 2
        if -math.pi * 0.75 <= a <= math.pi * 0.75:
            if best is None or abs(_ang(a - cur)) < abs(_ang(best - cur)):
                best = a
    return best


def perceive_decide(obs, goal, memory, last):
    """Stage machine: pick target-colored cube -> place on target, repeated N times (from goal text),
    then press the button. Perception: wrist (fallback front) RGB-D back-projected to world."""
    m = dict(memory)
    color, n = _parse_goal(goal)
    f = obs.current
    stage = m.get("stage") or "start"
    done = int(m.get("placed") or 0)
    fails = int(m.get("fails") or 0)
    det = _detect(f, color, near=None)
    tgt = m.get("target") or det["target"]
    btn = m.get("button") or det["button"]
    if det["target"] is not None and m.get("target") is None:
        tgt = det["target"]
    if det["button"] is not None and m.get("button") is None:
        btn = det["button"]
    m["target"] = tgt
    m["button"] = btn
    m["n_required"] = n
    cur_yaw = float(f.tcp[5])

    def pick_act(cube, cyaw):
        yaw = _choose_yaw(cyaw, cur_yaw)
        m["grasp_yaw"] = yaw
        return {"tool": "act_pick", "args": {"x": cube[0] + XOFF, "y": cube[1] + YOFF, "yaw": yaw}}

    act = None
    if stage == "picking":
        holding = False
        if isinstance(last, dict):
            holding = bool(last.get("holding"))
        elif isinstance(last, str):
            holding = '"holding": true' in last.replace(" ", "").replace(":", ": ")
        if holding:
            m["stage"] = "placing"
            t = tgt if tgt is not None else [f.tcp[0] - XOFF, f.tcp[1]]
            act = {"tool": "act_place", "args": {"x": t[0] + XOFF, "y": t[1] + YOFF,
                                                 "yaw": float(m.get("grasp_yaw") or cur_yaw)}}
        else:
            fails += 1
            stage = "start"
    elif stage == "placing":
        done += 1
        stage = "start"
        if tgt is not None:
            m["last_cube"] = [tgt[0], tgt[1], 0.04]
    elif stage == "pressing":
        fails += 1
        m["stage"] = "pressing" if fails < 4 else "observing"
        b = btn if btn is not None else [f.tcp[0] - XOFF, f.tcp[1]]
        act = {"tool": "act_press", "args": {"x": b[0] + XOFF, "y": b[1] + YOFF}}
    if act is None and stage in ("start", "observing"):
        if done >= n:
            if btn is None:
                m["stage"] = "observing"
                act = {"tool": "act_observe", "args": {}}
            else:
                m["stage"] = "pressing"
                act = {"tool": "act_press", "args": {"x": btn[0] + XOFF, "y": btn[1] + YOFF}}
        else:
            near = m.get("last_cube")
            cube, cyaw = det["cube"], det["cube_yaw"]
            if near is not None and cube is not None:
                d2 = _detect(f, color, near=near)
                if d2["cube"] is not None:
                    cube, cyaw = d2["cube"], d2["cube_yaw"]
            if cube is None or tgt is None:
                if stage == "observing" and cube is None and near is not None:
                    m["stage"] = "picking"
                    act = pick_act(near, None)
                else:
                    m["stage"] = "observing"
                    act = {"tool": "act_observe", "args": {}}
            else:
                m["last_cube"] = cube
                m["stage"] = "picking"
                act = pick_act(cube, cyaw)
    if act is None:
        m["stage"] = "observing"
        act = {"tool": "act_observe", "args": {}}
    m["placed"] = done
    m["fails"] = fails
    keys = ["stage", "placed", "fails", "n_required", "target", "button", "last_cube", "grasp_yaw"]
    return {"memory": {k: m.get(k) for k in keys}, "act": act}
