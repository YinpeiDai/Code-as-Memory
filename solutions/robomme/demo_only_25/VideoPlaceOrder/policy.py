"""VideoPlaceOrder policy: watch the video, track targets, place cube on k-th visited target."""

TARGET_Z = 0.008          # table-top height of the flat target discs (m)
CUBE_HALF = 0.02          # half edge of the cubes (m)
GRASP_Z = 0.019           # TCP height used by the expert to grasp / release a cube
SAFE_Z = 0.15             # transit height
TARGET_MATCH = 0.05       # max distance between a release point and a target centre
CUBE_BIAS = 0.007          # measured top-centroid bias toward the front camera (m)
TRACK_GATE = 0.055        # pixel-to-target association radius while tracking
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
COLORS = ("red", "green", "blue")


def _parse_goal(goal):
    text = str(goal or "").lower().replace(",", " ").replace(".", " ")
    words = text.split()
    color = None
    for c in COLORS:
        if c in words:
            color = c
            break
    k = None
    for i, w in enumerate(words):
        if w in ORDINALS and i + 1 < len(words) and words[i + 1].startswith("target"):
            k = ORDINALS[w]
    if k is None:
        for w in words:
            if w in ORDINALS:
                k = ORDINALS[w]
                break
    if k is None and "last" in words:
        k = -1
    return color, (k if k is not None else 1)


def _purple(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r > 140) & (b > 160) & (g < 130) & (r - g > 60)


def _color_mask(rgb, color):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    if color == "red":
        return (r > 120) & (g < 70) & (b < 70) & (r - g > 70)
    if color == "green":
        return (g > 120) & (r < 90) & (b < 90) & (g - r > 60)
    return (b > 120) & (r < 70) & (g < 70) & (b - r > 70)


def _calib(frame, fallback):
    K = getattr(frame, "front_K", None)
    E = getattr(frame, "front_extrinsic", None)
    if K is None or E is None:
        return fallback
    return (np.asarray(K, float), np.asarray(E, float))


def _plane_points(vs, us, K, E, z0):
    """World XY of pixels (rows vs, cols us) intersected with plane z=z0."""
    R = E[:, :3]
    t = E[:, 3]
    C = -R.T @ t
    dc = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))], 1)
    dw = dc @ R
    s = (z0 - C[2]) / np.where(np.abs(dw[:, 2]) < 1e-9, -1e-9, dw[:, 2])
    p = C[None, :] + s[:, None] * dw
    return p[:, :2], s > 0


def _depth_points(frame, mask, K, E):
    d = getattr(frame, "front_depth", None)
    if d is None:
        return None
    d = np.asarray(d)
    vs, us = np.nonzero(mask & (d > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))
    z = d[vs, us].astype(float) / 1000.0
    pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _purple_xy(frame, calib):
    K, E = calib
    vs, us = np.nonzero(_purple(np.asarray(frame.front_rgb)))
    if len(vs) == 0:
        return np.zeros((0, 2))
    xy, ok = _plane_points(vs.astype(float), us.astype(float), K, E, TARGET_Z)
    xy = xy[ok]
    keep = (np.abs(xy[:, 0]) < 0.6) & (np.abs(xy[:, 1]) < 0.6)
    return xy[keep]


def _cluster(pts, radius=0.05, min_n=30):
    out = []
    pts = np.asarray(pts)
    left = np.ones(len(pts), bool)
    while left.any():
        idx = np.flatnonzero(left)
        # seed at the densest remaining point (approximately)
        c = pts[idx[0]]
        for _ in range(8):
            sel = left & (np.sum((pts - c) ** 2, 1) < radius * radius)
            if not sel.any():
                break
            c = pts[sel].mean(0)
        sel = left & (np.sum((pts - c) ** 2, 1) < radius * radius)
        left = left & ~sel
        left[idx[0]] = False
        if sel.sum() >= min_n:
            out.append([float(c[0]), float(c[1]), int(sel.sum())])
    # merge near duplicates
    merged = []
    for c in sorted(out, key=lambda q: -q[2]):
        if all((c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2 > (0.6 * radius) ** 2 for m in merged):
            merged.append(c)
    return merged


def _detect_targets(frame, calib):
    return _cluster(_purple_xy(frame, calib))


def _release_events(frames):
    """Places in the video where the gripper released a held cube near the table: list of (index, x, y)."""
    out = []
    prev = None
    for i, f in enumerate(frames):
        fg = getattr(f, "fingers", None)
        tcp = getattr(f, "tcp", None)
        if fg is None or tcp is None:
            return None
        w = float(np.mean(np.asarray(fg, float)))
        if prev is not None:
            pw, ptcp = prev
            if 0.008 < pw < 0.03 and w > pw + 0.004 and float(tcp[2]) < 0.08:
                if not out or i - out[-1][0] > 15:
                    out.append((i, float(tcp[0]), float(tcp[1])))
        prev = (w, tcp)
    return out


def _cube_xyz(frame, color, calib):
    """Goal cube centre estimate (x, y, z, yaw, n_pixels) from the front camera or None."""
    K, E = calib
    mask = _color_mask(np.asarray(frame.front_rgb), color)
    if mask.sum() < 8:
        return None
    pts = _depth_points(frame, mask, K, E)
    if pts is not None and len(pts) >= 8:
        # keep the largest blob
        c = np.median(pts[:, :2], 0)
        near = np.sum((pts[:, :2] - c) ** 2, 1) < 0.04 ** 2
        pts = pts[near]
        top = pts[pts[:, 2] > pts[:, 2].max() - 0.006]
        if len(top) >= 4:
            xy = top[:, :2].mean(0)
            yaw = _square_yaw(top[:, :2])
            cam = -E[:, :3].T @ E[:, 3]
            dv = cam[:2] - xy
            nv = float(np.linalg.norm(dv))
            if nv > 1e-6:
                xy = xy - CUBE_BIAS * dv / nv   # visible front-face rim biases the top centroid toward the camera
            return [float(xy[0]), float(xy[1]), float(pts[:, 2].max()) - CUBE_HALF, yaw, int(len(pts))]
    vs, us = np.nonzero(mask)
    xy, ok = _plane_points(vs.astype(float), us.astype(float), K, E, 2 * CUBE_HALF)
    xy = xy[ok]
    if len(xy) < 4:
        return None
    c = np.median(xy, 0)
    return [float(c[0]), float(c[1]), CUBE_HALF, None, int(len(xy))]


def _square_yaw(xy):
    """Yaw (mod pi/2, in [-pi/4, pi/4)) of a square from its points: minimal bounding-box area."""
    if len(xy) < 6:
        return None
    p = xy - xy.mean(0)
    best, by = None, 0.0
    for k in range(45):
        a = (k * 2.0 - 45.0) * math.pi / 180.0
        ca, sa = math.cos(a), math.sin(a)
        u = p[:, 0] * ca + p[:, 1] * sa
        v = -p[:, 0] * sa + p[:, 1] * ca
        area = (u.max() - u.min()) * (v.max() - v.min())
        if best is None or area < best:
            best, by = area, a
    return float(by)


def _visual_releases(frames, color, calib_of):
    """Fallback when robot state is missing: cube comes to rest (low) and stays >= 8 frames."""
    out = []
    track = []
    for i, f in enumerate(frames):
        c = _cube_xyz(f, color, calib_of(f))
        if c is None or c[2] > 0.035:
            track = []
            continue
        if track and (c[0] - track[-1][1]) ** 2 + (c[1] - track[-1][2]) ** 2 > 0.01 ** 2:
            track = []
        track.append((i, c[0], c[1]))
        if len(track) == 8:
            if not out or (out[-1][1] - c[0]) ** 2 + (out[-1][2] - c[1]) ** 2 > 0.03 ** 2:
                out.append((i, c[0], c[1]))
    return out[1:] if out else out   # first rest is the cube's initial pose


def _track_targets(frames, calib_of, init):
    """Track target centres through the whole video; returns per-frame positions array [T, N, 2]."""
    pos = np.array([[c[0], c[1]] for c in init], float)
    base = np.array([max(c[2], 1) for c in init], float)
    vel = np.zeros_like(pos)
    hist = []
    for f in frames:
        xy = _purple_xy(f, calib_of(f))
        pred = pos + vel
        newpos = pos.copy()
        if len(xy) and len(pos):
            d2 = np.sum((xy[:, None, :] - pred[None, :, :]) ** 2, 2)
            own = np.argmin(d2, 1)
            dmin = d2[np.arange(len(xy)), own]
            for j in range(len(pos)):
                sel = (own == j) & (dmin < TRACK_GATE ** 2)
                n = int(sel.sum())
                if n >= max(12, 0.35 * base[j]):
                    c = xy[sel].mean(0)
                    # refine around the new centre
                    sel2 = np.sum((xy - c) ** 2, 1) < 0.045 ** 2
                    if sel2.sum() >= n * 0.8:
                        c = xy[sel2].mean(0)
                    if np.sum((c - pos[j]) ** 2) > 0.002 ** 2 or n >= 0.7 * base[j]:
                        newpos[j] = c
        vel = 0.5 * vel + 0.5 * (newpos - pos)
        pos = newpos
        hist.append(pos.copy())
    return hist


def _analyze_video(frames, goal):
    color, k = _parse_goal(goal)
    demo = [f for f in frames if getattr(f, "phase", "exec") == "demo"]
    cur = frames[-1]
    base_cal = _calib(cur, None)
    if base_cal is None:
        for f in frames:
            base_cal = _calib(f, None)
            if base_cal is not None:
                break

    def calib_of(f):
        return _calib(f, base_cal)

    info = {"color": color, "k": k, "n_demo": len(demo)}
    if not demo:
        return info
    init = _detect_targets(demo[0], calib_of(demo[0]))
    info["targets0"] = [[round(c[0], 4), round(c[1], 4)] for c in init]
    if not init:
        return info
    hist = _track_targets(demo, calib_of, init)
    rel = _release_events(demo)
    if rel is None:
        rel = _visual_releases(demo, color, calib_of)
        info["release_source"] = "visual"
    else:
        info["release_source"] = "robot_state"
    seq = []
    for (i, x, y) in rel:
        P = hist[i]
        d = np.sqrt(np.sum((P - np.array([x, y])) ** 2, 1))
        j = int(np.argmin(d))
        if d[j] < TARGET_MATCH:
            if info["release_source"] == "visual" and seq and seq[-1] == j:
                continue
            seq.append(j)
    info["sequence"] = seq
    info["releases"] = [[i, round(x, 3), round(y, 3)] for (i, x, y) in rel]
    final = hist[-1]
    info["targets_final"] = [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in final]
    if seq:
        kk = k - 1 if k > 0 else len(seq) - 1
        kk = max(0, min(kk, len(seq) - 1))
        tid = seq[kk]
        tx, ty = float(final[tid][0]), float(final[tid][1])
        # snap to a target visible in the current frame
        now = _detect_targets(cur, calib_of(cur))
        if now:
            dd = [(c[0] - tx) ** 2 + (c[1] - ty) ** 2 for c in now]
            jj = int(np.argmin(dd))
            if dd[jj] < 0.03 ** 2:
                tx, ty = now[jj][0], now[jj][1]
        info["target"] = [tx, ty]
        info["target_id"] = tid
    return info


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _grasp_yaw(cube_yaw, cur_yaw):
    """Gripper yaw aligned with cube faces, the one of four equivalent closest to the current yaw."""
    if cube_yaw is None:
        return cur_yaw
    best = None
    for q in range(-4, 5):
        y = cube_yaw + q * math.pi / 2
        if best is None or abs(_wrap(y - cur_yaw)) < abs(_wrap(best - cur_yaw)):
            best = y
    return cur_yaw + _wrap(best - cur_yaw)


def _move(robot, x, y, z, yaw, grip, max_step=0.01, settle=40, tol=0.003):
    f = robot.now()
    p = np.asarray(f.tcp, float)
    roll, pitch = float(p[3]), 0.0
    if abs(abs(roll) - math.pi) < 0.5:
        roll = -math.pi if roll < 0 else math.pi
    start = p[:3].copy()
    goal = np.array([x, y, z], float)
    dist = float(np.linalg.norm(goal - start))
    y0 = float(p[5])
    dyaw = _wrap(yaw - y0)
    n = max(1, int(math.ceil(max(dist / max_step, abs(dyaw) / 0.05))))
    for i in range(1, n + 1):
        s = i / n
        q = start + (goal - start) * s
        robot.step(q[0], q[1], q[2], roll, pitch, y0 + dyaw * s, grip)
    for _ in range(settle):
        cur = np.asarray(robot.now().tcp, float)
        if np.linalg.norm(cur[:3] - goal) < tol:
            break
        robot.step(x, y, z, roll, pitch, y0 + dyaw, grip)
    cur = np.asarray(robot.now().tcp, float)
    return float(np.linalg.norm(cur[:3] - goal))


def _hold(robot, grip, n):
    f = robot.now()
    p = np.asarray(f.tcp, float)
    roll = float(p[3])
    if abs(abs(roll) - math.pi) < 0.5:
        roll = -math.pi if roll < 0 else math.pi
    for _ in range(n):
        robot.step(p[0], p[1], p[2], roll, 0.0, p[5], grip)


def act_pick(robot, x, y, yaw, z=GRASP_Z):
    """Open gripper, go above (x, y), descend to grasp height z, close, lift to SAFE_Z.
    Returns finger width after lifting (about 0.018 when a cube is held, ~0 when empty)."""
    _move(robot, robot.now().tcp[0], robot.now().tcp[1], max(SAFE_Z, float(robot.now().tcp[2])), yaw, 1.0)
    _move(robot, x, y, SAFE_Z, yaw, 1.0)
    _move(robot, x, y, z + 0.03, yaw, 1.0)
    err = _move(robot, x, y, z, yaw, 1.0, max_step=0.006)
    _hold(robot, -1.0, 12)
    _move(robot, x, y, SAFE_Z, yaw, -1.0)
    w = float(np.mean(np.asarray(robot.now().fingers, float)))
    return {"finger_width": w, "holding": bool(w > 0.006), "reach_error": err}


def act_place(robot, x, y, yaw, z=GRASP_Z):
    """Carry the held cube at SAFE_Z to above (x, y), descend to z, open, and retreat upward."""
    _move(robot, robot.now().tcp[0], robot.now().tcp[1], SAFE_Z, yaw, -1.0)
    _move(robot, x, y, SAFE_Z, yaw, -1.0)
    _move(robot, x, y, z + 0.03, yaw, -1.0)
    err = _move(robot, x, y, z, yaw, -1.0, max_step=0.005)
    _hold(robot, 1.0, 10)
    _move(robot, x, y, SAFE_Z, yaw, 1.0)
    return {"placed": True, "reach_error": err}


def act_observe(robot):
    """Hold the current pose (gripper open) for a few control steps."""
    _hold(robot, 1.0, 5)
    return {"observed": True}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: analyse the video once (target order + target swaps), pick the goal-colour cube,
    then place it on the remembered k-th target; retry the pick if the grasp is empty."""
    mem = dict(memory)
    stage = mem.get("stage") or "analyse"
    cur = obs.current
    frames = list(obs.frames)
    tcp = np.asarray(cur.tcp, float)
    if stage == "analyse":
        info = _analyze_video(frames, goal)
        mem["color"] = info.get("color") or "red"
        mem["order"] = [str(s) for s in info.get("sequence", [])]
        mem["target_pos"] = info.get("target")
        mem["notes"] = str({k: info[k] for k in info if k in ("k", "release_source", "sequence", "target_id", "n_demo")})
        mem["attempts"] = 0
        stage = "pick"
    color = mem.get("color") or "red"
    cal = _calib(cur, None)
    if stage == "pick" and last is not None and mem.get("attempts", 0) > 0:
        held = float(np.mean(np.asarray(cur.fingers, float))) > 0.006
        if held:
            stage = "place"
    if stage == "pick":
        if mem.get("attempts", 0) >= 4:
            stage = "done"
        else:
            c = _cube_xyz(cur, color, cal) if cal is not None else None
            if c is None:
                c = mem.get("cube_pos") or [-0.1, 0.0, CUBE_HALF]
                cy = None
            else:
                cy = c[3]
            mem["cube_pos"] = [float(c[0]), float(c[1]), float(c[2])]
            yaw = _grasp_yaw(cy, float(tcp[5]))
            mem["attempts"] = int(mem.get("attempts", 0)) + 1
            mem["stage"] = "pick"
            return {"memory": _mem_out(mem), "act": {"tool": "act_pick",
                    "args": {"x": float(c[0]), "y": float(c[1]), "yaw": float(yaw)}}}
    if stage == "place":
        held = float(np.mean(np.asarray(cur.fingers, float))) > 0.006
        if not held:
            stage = "pick"
            mem["stage"] = "pick"
            return _retry(obs, goal, mem, last)
        t = mem.get("target_pos")
        if t is None:
            now = _detect_targets(cur, cal) if cal is not None else []
            t = [now[0][0], now[0][1]] if now else [0.0, 0.0]   # no usable video: guess
        mem["stage"] = "done"
        return {"memory": _mem_out(mem), "act": {"tool": "act_place",
                "args": {"x": float(t[0]), "y": float(t[1]), "yaw": float(tcp[5])}}}
    # done: verify the cube rests on the chosen target; otherwise pick it again (bounded attempts)
    t = mem.get("target_pos")
    if t is not None and cal is not None and int(mem.get("attempts", 0)) < 4 and float(tcp[2]) > 0.08:
        c = _cube_xyz(cur, color, cal)
        if c is not None and c[4] >= 20 and c[2] < 0.035 and \
                (c[0] - t[0]) ** 2 + (c[1] - t[1]) ** 2 > 0.035 ** 2:
            mem["notes"] = str(mem.get("notes")) + " | re-pick: cube off target"
            return _retry(obs, goal, mem, last)
    mem["stage"] = "done"
    return {"memory": _mem_out(mem), "act": {"tool": "act_observe", "args": {}}}


def _retry(obs, goal, memory, last):
    """Re-enter the pick stage after a lost grasp (internal helper exposed as a perceive tool)."""
    mem = dict(memory)
    cur = obs.current
    color = mem.get("color") or "red"
    cal = _calib(cur, None)
    c = _cube_xyz(cur, color, cal) if cal is not None else None
    if c is None or int(mem.get("attempts", 0)) >= 4:
        mem["stage"] = "done"
        return {"memory": _mem_out(mem), "act": {"tool": "act_observe", "args": {}}}
    mem["cube_pos"] = [float(c[0]), float(c[1]), float(c[2])]
    mem["attempts"] = int(mem.get("attempts", 0)) + 1
    mem["stage"] = "pick"
    yaw = _grasp_yaw(c[3], float(np.asarray(cur.tcp, float)[5]))
    return {"memory": _mem_out(mem), "act": {"tool": "act_pick",
            "args": {"x": float(c[0]), "y": float(c[1]), "yaw": float(yaw)}}}


MEMORY_FIELDS = ("stage", "color", "order", "target_pos", "cube_pos", "attempts", "notes")


def _mem_out(mem):
    return {k: mem.get(k) for k in MEMORY_FIELDS}
