"""VideoRepick policy: track the block picked in the instruction video, repick it N times, press the button."""

GRASP_DX = -0.006      # expert grasp TCP sits ~6 mm -x of the camera-measured cube top centre (25/25 episodes)
GRASP_Z = 0.019        # expert grasp / release TCP height
HOVER_Z = 0.15         # expert transit height
TOP_Z = 0.037          # cube tops are at ~0.041 m (4 cm cubes)
BUTTON_DX = 0.0
STEP_XY = 0.008        # max translation per control step
STEP_YAW = 0.06
ROLL = math.pi
NUMBERS = {"once": 1, "one": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3, "four": 4, "five": 5, "six": 6}


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    E = np.asarray(E, float)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw.reshape(H, W, 3), z > 0


def _colormasks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    red = (r > 90) & (g * 100 < 35 * r) & (b * 100 < 35 * r)
    green = (g > 90) & (r * 2 < g) & (b * 2 < g)
    blue = (b > 90) & (r * 2 < b) & (g * 2 < b)
    return [red, green, blue]


def _components(mask):
    """Connected-component labels (4-neighbour) of a boolean mask, cropped for speed."""
    ys, xs = np.nonzero(mask)
    out = np.full(mask.shape, -1)
    if len(ys) == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1]
    big = m.size + 1
    lab = np.where(m, np.arange(m.size).reshape(m.shape), big)
    for _ in range(400):
        p = np.pad(lab, 1, constant_values=big)
        n = np.minimum(np.minimum(np.minimum(p[1:-1, 1:-1], p[:-2, 1:-1]), np.minimum(p[2:, 1:-1], p[1:-1, :-2])), p[1:-1, 2:])
        n = np.where(m, n, big)
        if (n == lab).all():
            break
        lab = n
    out[y0:y1, x0:x1] = np.where(m, lab, -1)
    return out


def _has_depth(frame, cam):
    d = getattr(frame, cam + "_depth", None)
    return d is not None


def _cam(frame, cam):
    rgb = np.asarray(getattr(frame, cam + "_rgb"))
    K = np.asarray(getattr(frame, cam + "_K"), float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), float)
    if _has_depth(frame, cam):
        P, ok = _backproject(np.asarray(getattr(frame, cam + "_depth")), K, E)
    else:
        P, ok = _plane_points(rgb.shape[0], rgb.shape[1], K, E, 0.04)
    return rgb, P, ok


def _plane_points(H, W, K, E, zp):
    """Fallback without depth: intersect pixel rays with the horizontal plane z=zp."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1], np.ones((H, W))], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    c = -R.T @ t
    dw = d @ R
    s = (zp - c[2]) / np.where(np.abs(dw[:, 2]) < 1e-9, 1e-9, dw[:, 2])
    pw = c + dw * s[:, None]
    pw[:, 2] = zp + 0.001
    return pw.reshape(H, W, 3), (s > 0).reshape(H, W)


def _top_points(frame, cam, color, zmin=TOP_Z, zmax=0.06):
    rgb, P, ok = _cam(frame, cam)
    m = _colormasks(rgb)[color] & ok & (P[..., 2] > zmin) & (P[..., 2] < zmax)
    return P[m][:, :2]


def _detect_blocks(frame, cam="front", colors=(0, 1, 2), zmin=TOP_Z, zmax=0.06):
    """Cube top faces: list of (color, x, y, npix)."""
    rgb, P, ok = _cam(frame, cam)
    top = ok & (P[..., 2] > zmin) & (P[..., 2] < zmax)
    masks = _colormasks(rgb)
    out = []
    for c in colors:
        m = masks[c] & top
        if m.sum() < 4:
            continue
        lab = _components(m)
        ls = lab[m]
        pts = P[m]
        for l in np.unique(ls):
            s = pts[ls == l]
            if len(s) < 4:
                continue
            xy = s[:, :2].mean(0)
            out.append((int(c), float(xy[0]), float(xy[1]), int(len(s))))
    return out


def _detect_button(frame, cam="front"):
    """White button base: returns [x, y] (midpoint of robust extent) or None."""
    rgb, P, ok = _cam(frame, cam)
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    m = ok & (mn > 150) & (mx - mn < 25) & (P[..., 2] > 0.003) & (P[..., 2] < 0.05)
    m = m & (P[..., 0] > -0.45) & (P[..., 0] < 0.4) & (np.abs(P[..., 1]) < 0.4)
    if m.sum() < 15:
        return None
    lab = _components(m)
    ls = lab[m]
    pts = P[m]
    vals, counts = np.unique(ls, return_counts=True)
    s = pts[ls == vals[int(np.argmax(counts))]]
    if len(s) < 15:
        return None
    lo = np.percentile(s[:, :2], 4, axis=0)
    hi = np.percentile(s[:, :2], 96, axis=0)
    c = (lo + hi) / 2.0
    return [float(c[0]), float(c[1])]


def _square_yaw(pts):
    """Yaw (radians, in [-pi/4, pi/4)) of a square from its top-face points (min-extent search)."""
    if len(pts) < 6:
        return None
    c = pts.mean(0)
    q0 = pts - c
    best = None
    for k in range(90):
        a = math.radians(k)
        ca, sa = math.cos(a), math.sin(a)
        u = q0[:, 0] * ca + q0[:, 1] * sa
        w = -q0[:, 0] * sa + q0[:, 1] * ca
        ext = (np.percentile(u, 97) - np.percentile(u, 3)) + (np.percentile(w, 97) - np.percentile(w, 3))
        if best is None or ext < best[0]:
            best = (ext, a)
    a = best[1]
    return ((a + math.pi / 4) % (math.pi / 2)) - math.pi / 4


def _parse_count(goal):
    g = (goal or "").lower().replace(",", " ").replace(".", " ")
    words = g.split()
    for i, w in enumerate(words):
        if w in ("times", "time") and i > 0:
            p = words[i - 1]
            if p.isdigit():
                return int(p)
            if p in NUMBERS:
                return NUMBERS[p]
    for w in words:
        if w in ("twice", "thrice", "once"):
            return NUMBERS[w]
    return 1


def _fingers(frame):
    f = getattr(frame, "fingers", None)
    if f is None:
        return None
    return float(np.mean(np.asarray(f, float)))


def _track_demo(frames):
    """Find the block grasped in the demo video and follow it through any shuffles.

    Returns dict(found, xy, color, grasp_xy, note)."""
    demo = [f for f in frames if getattr(f, "phase", "") == "demo"]
    if len(demo) < 5:
        return {"found": False, "note": "no demo frames"}
    g = None
    rel = None
    for i, f in enumerate(demo):
        fw = _fingers(f)
        tcp = getattr(f, "tcp", None)
        if fw is None or tcp is None:
            continue
        if g is None and fw < 0.03 and tcp[2] < 0.06:
            g = i
        if g is not None and i > g and fw > 0.035:
            rel = i
            break
    note = ""
    if g is None or rel is None:
        start, p, color = _fallback_target(demo)
        if start is None:
            return {"found": False, "note": "no grasp found in demo"}
        note = "fallback-vanish"
        gxy = p
    else:
        start = rel
        p = np.array(demo[rel].tcp[:2], float)
        gxy = [float(demo[g].tcp[0]), float(demo[g].tcp[1])]
        color = None
    p = np.array(p, float)
    v = np.zeros(2)
    last_step = demo[start].step
    for f in demo[start:]:
        gap = max(1, int(f.step) - int(last_step)) if isinstance(f.step, (int, float)) else 1
        cols = (color,) if color is not None else (0, 1, 2)
        dets = _detect_blocks(f, "front", cols)
        if not dets:
            continue
        pred = p + v * gap
        dist = [math.hypot(d[1] - pred[0], d[2] - pred[1]) for d in dets]
        i = int(np.argmin(dist))
        if dist[i] < 0.035 + 0.006 * (gap - 1):
            if color is None:
                tcp = getattr(f, "tcp", None)
                if tcp is None or tcp[2] > 0.12:
                    color = dets[i][0]
                else:
                    continue
            newp = np.array([dets[i][1], dets[i][2]])
            v = 0.5 * v + 0.5 * (newp - p) / gap
            p = newp
            last_step = f.step
        else:
            v = v * 0.8
    if color is None:
        dets = _detect_blocks(demo[-1], "front")
        if dets:
            dist = [math.hypot(d[1] - p[0], d[2] - p[1]) for d in dets]
            i = int(np.argmin(dist))
            if dist[i] < 0.04:
                color = dets[i][0]
                p = np.array([dets[i][1], dets[i][2]])
    return {"found": True, "xy": [float(p[0]), float(p[1])], "color": color, "grasp_xy": gxy, "note": note}


def _fallback_target(demo):
    """Without proprioception: the initial block that vanishes longest in the first half of the demo."""
    init = _detect_blocks(demo[0], "front")
    if not init:
        return None, None, None
    half = demo[: max(2, len(demo) // 2)]
    missing = [0] * len(init)
    lastseen = [0] * len(init)
    for j, f in enumerate(half[::2]):
        dets = _detect_blocks(f, "front")
        for k, b in enumerate(init):
            ok = any(d[0] == b[0] and math.hypot(d[1] - b[1], d[2] - b[2]) < 0.02 for d in dets)
            if not ok:
                missing[k] += 1
            else:
                lastseen[k] = 2 * j
    k = int(np.argmax(missing))
    if missing[k] == 0:
        return None, None, None
    b = init[k]
    # resume tracking once it is back on the table
    return min(len(demo) - 1, len(half)), [b[1], b[2]], b[0]


def _locate(frame, xy, color, cam, radius=0.03):
    """Refine a cube position/yaw near xy using cube-top points of one camera."""
    if color is None:
        return None
    pts = _top_points(frame, cam, int(color))
    if len(pts) == 0:
        return None
    d = np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1])
    s = pts[d < radius]
    if len(s) < 6:
        return None
    # one recentring pass, in case the circle clipped a neighbour
    c = s.mean(0)
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    s = pts[d < 0.026]
    if len(s) < 6:
        return None
    c = s.mean(0)
    return {"xy": [float(c[0]), float(c[1])], "yaw": _square_yaw(s), "n": int(len(s))}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _move(robot, x, y, z, yaw, grip, tol=0.004, settle=40, max_steps=300):
    """Gradual straight-line motion to an absolute TCP target with TCP feedback."""
    f = robot.now()
    cur = np.array(f.tcp[:3], float)
    cyaw = float(f.tcp[5])
    goal = np.array([x, y, z], float)
    cmd = cur.copy()
    cmd_yaw = cyaw
    n = 0
    while n < max_steps:
        dvec = goal - cmd
        dist = float(np.linalg.norm(dvec))
        if dist > STEP_XY:
            cmd = cmd + dvec * (STEP_XY / dist)
        else:
            cmd = goal.copy()
        dy = _wrap(yaw - cmd_yaw)
        cmd_yaw = cmd_yaw + max(-STEP_YAW, min(STEP_YAW, dy))
        f = robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), ROLL, 0.0, float(cmd_yaw), grip)
        n += 1
        if dist <= STEP_XY and abs(_wrap(yaw - cmd_yaw)) < 1e-6:
            break
    for _ in range(settle):
        err = float(np.linalg.norm(np.array(f.tcp[:3], float) - goal))
        if err < tol:
            break
        f = robot.step(float(goal[0]), float(goal[1]), float(goal[2]), ROLL, 0.0, float(yaw), grip)
        n += 1
    return f, n


def _grip(robot, grip, steps=12):
    f = robot.now()
    t = f.tcp
    for _ in range(steps):
        f = robot.step(float(t[0]), float(t[1]), float(t[2]), ROLL, 0.0, float(t[5]), grip)
    return f


def _choose_yaw(yaw, current):
    """Equivalent square yaw (mod pi/2) closest to the current TCP yaw, kept in [-pi/2, pi/2]."""
    best = yaw
    for k in (-2, -1, 0, 1, 2):
        c = yaw + k * math.pi / 2
        if abs(c) <= math.pi / 2 + 1e-6 and abs(_wrap(c - current)) < abs(_wrap(best - current)):
            best = c
    return best


def _num_after(txt, key):
    i = txt.find('"' + key + '":')
    if i < 0:
        return None
    j = i + len(key) + 3
    k = j
    while k < len(txt) and txt[k] in " -+.0123456789eE":
        k += 1
    t = txt[j:k].strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_last(last):
    """The prior tool outcome, as a dict (dict passthrough, or the JSON text the harness returns)."""
    if isinstance(last, dict):
        return last
    if not isinstance(last, str):
        return None
    out = {}
    if '"grasped": true' in last:
        out["grasped"] = True
    elif '"grasped": false' in last:
        out["grasped"] = False
    for key in ("x", "y", "yaw"):
        out[key] = _num_after(last, key)
    return out


def perceive_decide(obs, goal, memory, last):
    """VideoRepick state machine.

    First call: parse the repeat count from the goal, find the block grasped in the demo video (from TCP/finger
    proprioception), follow it through the video's block shuffles with front-camera RGB-D cube-top tracking, and
    locate the white button. Then issue act_repick (one pick-and-put-down per call, retried if the grasp misses)
    until the count is met, then act_press_button."""
    mem = dict(memory)
    for k in ("stage", "target", "color", "button", "count", "done", "attempts", "yaw", "notes"):
        mem.setdefault(k, None)
    cur = obs.current
    if mem["stage"] is None:
        mem["count"] = _parse_count(goal)
        tr = _track_demo(list(obs.frames))
        notes = [tr.get("note", "")]
        tgt = tr.get("xy") if tr.get("found") else None
        color = tr.get("color")
        # snap to the current scene
        if tgt is not None:
            dets = _detect_blocks(cur, "front")
            cand = [d for d in dets if color is None or d[0] == color]
            if cand:
                dist = [math.hypot(d[1] - tgt[0], d[2] - tgt[1]) for d in cand]
                i = int(np.argmin(dist))
                if dist[i] < 0.04:
                    tgt = [cand[i][1], cand[i][2]]
                    color = cand[i][0]
        if tgt is None:
            dets = _detect_blocks(cur, "front")
            if dets:
                tgt = [dets[0][1], dets[0][2]]
                color = dets[0][0]
                notes.append("no demo target; guessed first block")
            else:
                tgt = [0.0, 0.0]
                notes.append("no blocks seen")
        loc = _locate(cur, tgt, color, "front")
        mem["yaw"] = loc["yaw"] if loc and loc["yaw"] is not None else 0.0
        mem["target"] = [float(tgt[0]), float(tgt[1])]
        mem["color"] = "none" if color is None else str(int(color))
        mem["button"] = _detect_button(cur, "front")
        mem["done"] = 0
        mem["attempts"] = 0
        mem["stage"] = "repick"
        mem["notes"] = "; ".join(n for n in notes if n)
    else:
        res = _parse_last(last)
        if mem["stage"] == "repick" and res is not None and "grasped" in res:
            mem["attempts"] = int(mem["attempts"] or 0) + 1
            if res["grasped"]:
                mem["done"] = int(mem["done"] or 0) + 1
            if res.get("x") is not None and res.get("y") is not None:
                if math.hypot(res["x"] - mem["target"][0], res["y"] - mem["target"][1]) < 0.04:
                    mem["target"] = [float(res["x"]), float(res["y"])]
            if res.get("yaw") is not None:
                mem["yaw"] = float(res["yaw"])
        if mem["stage"] == "repick" and (int(mem["done"] or 0) >= int(mem["count"] or 1) or int(mem["attempts"] or 0) >= int(mem["count"] or 1) + 2):
            mem["stage"] = "press"
            if mem["button"] is None:
                mem["button"] = _detect_button(cur, "front")
        elif mem["stage"] == "press":
            mem["stage"] = "press"
    if mem["stage"] == "repick":
        color = -1 if mem["color"] in (None, "none") else int(mem["color"])
        act = {"tool": "act_repick", "args": {"x": mem["target"][0], "y": mem["target"][1],
                                              "yaw": float(mem["yaw"] or 0.0), "color": color}}
    else:
        b = mem["button"] or [-0.2, 0.0]
        act = {"tool": "act_press_button", "args": {"x": b[0], "y": b[1]}}
    return {"memory": mem, "act": act}


def act_repick(robot, x, y, yaw=0.0, color=-1):
    """Pick the cube near world (x, y) (cube-top centre, metres; yaw radians; color 0 red/1 green/2 blue, -1 unknown)
    and put it back down at the same place. Refines position/yaw with the wrist RGB-D on the way down.
    Returns {"grasped": bool, "x", "y", "fingers"}."""
    col = None if color is None or int(color) < 0 else int(color)
    f = robot.now()
    yaw_c = _choose_yaw(float(yaw), float(f.tcp[5]))
    cx, cy = float(x), float(y)
    # transit above the cube
    f, _ = _move(robot, cx + GRASP_DX, cy, HOVER_Z, yaw_c, 1.0)
    for hz in (0.11,):
        loc = _locate(f, [cx, cy], col, "wrist", radius=0.03)
        if loc is not None and math.hypot(loc["xy"][0] - cx, loc["xy"][1] - cy) < 0.03:
            cx, cy = loc["xy"]
            if loc["yaw"] is not None:
                yaw_c = _choose_yaw(loc["yaw"], float(f.tcp[5]))
        f, _ = _move(robot, cx + GRASP_DX, cy, hz, yaw_c, 1.0)
        loc = _locate(f, [cx, cy], col, "wrist", radius=0.03)
        if loc is not None and math.hypot(loc["xy"][0] - cx, loc["xy"][1] - cy) < 0.02:
            cx, cy = loc["xy"]
            if loc["yaw"] is not None:
                yaw_c = _choose_yaw(loc["yaw"], float(f.tcp[5]))
    gx = cx + GRASP_DX
    f, _ = _move(robot, gx, cy, 0.05, yaw_c, 1.0)
    f, _ = _move(robot, gx, cy, GRASP_Z, yaw_c, 1.0, tol=0.003)
    f = _grip(robot, -1.0, 15)
    fw = _fingers(f)
    if fw is not None and fw < 0.006:
        # closed on nothing: reopen, back off and report a miss (not counted as a repetition)
        f = _grip(robot, 1.0, 10)
        f, _ = _move(robot, gx, cy, HOVER_Z, yaw_c, 1.0, tol=0.01, settle=10)
        return {"grasped": False, "x": cx, "y": cy, "yaw": yaw_c, "fingers": fw, "fingers_closed": fw}
    f, _ = _move(robot, gx, cy, HOVER_Z, yaw_c, -1.0, tol=0.01, settle=10)
    fw2 = _fingers(f)
    grasped = fw2 is None or fw2 > 0.006
    # put it back down where it was
    f, _ = _move(robot, gx, cy, 0.05, yaw_c, -1.0, tol=0.01, settle=5)
    f, _ = _move(robot, gx, cy, GRASP_Z, yaw_c, -1.0, tol=0.003, settle=20)
    f = _grip(robot, 1.0, 12)
    f, _ = _move(robot, gx, cy, HOVER_Z, yaw_c, 1.0, tol=0.01, settle=10)
    return {"grasped": bool(grasped), "x": cx, "y": cy, "yaw": yaw_c, "fingers": fw2, "fingers_closed": fw}


def act_press_button(robot, x, y):
    """Close the gripper, move above the button at world (x, y), press down onto it and lift."""
    f = robot.now()
    bx, by = float(x), float(y)
    f, _ = _move(robot, bx, by, HOVER_Z, 0.0, -1.0, tol=0.006)
    loc = _detect_button(f, "wrist")
    if loc is not None and math.hypot(loc[0] - bx, loc[1] - by) < 0.03:
        bx, by = loc
        f, _ = _move(robot, bx, by, HOVER_Z, 0.0, -1.0, tol=0.006)
    f, _ = _move(robot, bx + BUTTON_DX, by, 0.05, 0.0, -1.0, tol=0.006)
    zs = []
    for k in range(60):
        f = robot.step(bx + BUTTON_DX, by, 0.005, ROLL, 0.0, 0.0, -1.0)
        zs.append(float(f.tcp[2]))
        if len(zs) > 8 and abs(zs[-1] - zs[-8]) < 0.0007:
            break
    zmin = min(zs)
    f, _ = _move(robot, bx, by, HOVER_Z, 0.0, -1.0, tol=0.01, settle=5)
    for _ in range(20):
        f = robot.step(bx, by, HOVER_Z, ROLL, 0.0, 0.0, -1.0)
    return {"pressed_to_z": zmin, "x": bx, "y": by}
