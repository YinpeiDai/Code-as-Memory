"""TraceShapeHard policy: watch the red cube trace a closed shape while the lamp is red,
record its path from the front RGB-D camera, then (lamp green) grasp the green cube,
lift it and carry it along the recorded path in the same order and direction."""

Z_TABLE_MAX = 0.08      # cube points must lie below this world height (m)
TRACE_Z = 0.06          # TCP height while tracing (expert: 0.059-0.062)
GRASP_Z = 0.033         # TCP height at grasp (expert: 0.030-0.036)
HOVER_Z = 0.10
PATH_SPACING = 0.008    # resampling spacing of the recorded path (m)


def _masks(rgb):
    im = rgb.astype(np.int32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    red = (R - G > 50) & (np.abs(G - B) <= 12) & (G > 40)
    green = (G - R > 40) & (np.abs(R - B) <= 12) & (R > 40)
    lamp_r = (R > 200) & (G < 60) & (B < 60)
    lamp_g = (G > 200) & (R < 80) & (B < 80)
    return red, green, lamp_r, lamp_g


def _lamp(frame):
    _, _, lr, lg = _masks(frame.front_rgb)
    nr, ng = int(lr.sum()), int(lg.sum())
    if ng > 100 and ng > nr:
        return "green"
    if nr > 100:
        return "red"
    return "off"


def _world_points(frame, mask):
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return np.zeros((0, 3))
    d = frame.front_depth[vs, us].astype(np.float64)
    ok = d > 0
    vs, us, d = vs[ok], us[ok], d[ok] / 1000.0
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    return (pc - E[:, 3]) @ E[:, :3]


def _top_center(frame, mask, min_pts=6):
    """XY centre of the visible top face of a cube (points within 4 mm of the highest)."""
    p = _world_points(frame, mask)
    if len(p) == 0:
        return None
    p = p[(p[:, 2] < Z_TABLE_MAX) & (p[:, 2] > -0.01)]
    if len(p) < min_pts:
        return None
    top = p[p[:, 2] > p[:, 2].max() - 0.004]
    if len(top) < 3:
        return None
    c = top.mean(0)
    return [float(c[0]), float(c[1]), float(p[:, 2].max())]


def _red_center(frame):
    red, _, _, _ = _masks(frame.front_rgb)
    return _top_center(frame, red)


def _green_center(frame):
    _, green, _, _ = _masks(frame.front_rgb)
    return _top_center(frame, green)


def _resample(points, spacing):
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return P.tolist()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-4:
            keep.append(i)
    P = P[keep]
    if len(P) < 2:
        return P.tolist()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    q = np.linspace(0.0, s[-1], n)
    x = np.interp(q, s, P[:, 0])
    y = np.interp(q, s, P[:, 1])
    return [[float(a), float(b)] for a, b in zip(x, y)]


def _extract_path(pts):
    """Per-frame red-cube top centres (cue order) -> outlier rejection, light smoothing, resampling."""
    pts = [p for p in pts if p is not None]
    if len(pts) < 5:
        return pts
    P = np.asarray([[p[0], p[1]] for p in pts], dtype=np.float64)
    T = np.asarray([p[2] if len(p) > 2 else i for i, p in enumerate(pts)], dtype=np.float64)
    # reject outliers: jump larger than the cube could move in the elapsed frames (<= 2 cm/frame + 3 cm)
    good = [0]
    for i in range(1, len(P)):
        gap = max(1.0, T[i] - T[good[-1]])
        if np.linalg.norm(P[i] - P[good[-1]]) < 0.03 + 0.02 * gap:
            good.append(i)
    P = P[good]
    # 3-tap smoothing keeping endpoints
    if len(P) >= 3:
        S = P.copy()
        S[1:-1] = (P[:-2] + P[1:-1] + P[2:]) / 3.0
        P = S
    return _resample(P.tolist(), PATH_SPACING)


def _path_length(path):
    P = np.asarray(path)
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def perceive_decide(obs, goal, memory, last):
    """Stage machine: watch -> grasp -> trace -> (retrace / regrasp)."""
    mem = {
        "stage": memory.get("stage") or "watch",
        "red_raw": memory.get("red_raw"),
        "red_path": memory.get("red_path"),
        "cube_xy": memory.get("cube_xy"),
        "grasp_offset": memory.get("grasp_offset"),
        "trace_count": memory.get("trace_count") or 0,
        "grasp_count": memory.get("grasp_count") or 0,
        "log": memory.get("log") or "",
    }
    cur = obs.current
    stage = mem["stage"]
    if stage == "watch":
        raw = list(mem["red_raw"] or [])
        for f in obs.frames:
            if _lamp(f) == "green":
                break
            c = _red_center(f)
            if c is not None:
                raw.append([round(c[0], 4), round(c[1], 4), int(f.step)])
            g = _green_center(f)
            if g is not None:
                mem["cube_xy"] = [g[0], g[1]]
        mem["red_raw"] = raw[-400:]
        if _lamp(cur) != "green":
            mem["log"] = (mem["log"] + " watch@%d lamp=%s n=%d;" % (cur.step, _lamp(cur), len(raw)))[-600:]
            return {"memory": mem, "act": {"tool": "act_watch", "args": {}}}
        path = _extract_path(raw)
        mem["red_path"] = path
        mem["log"] = (mem["log"] + " path n=%d len=%.3f raw=%d;" % (len(path), _path_length(path), len(raw)))[-600:]
        stage = "grasp"
    if stage == "grasp":
        c = _green_center(cur)
        if c is None:
            c = mem["cube_xy"]
        if c is None:
            mem["stage"] = "grasp"
            mem["log"] = (mem["log"] + " no green cube@%d;" % cur.step)[-600:]
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 3, "gripper": 1.0}}}
        mem["cube_xy"] = [c[0], c[1]]
        mem["stage"] = "grasped?"
        mem["grasp_count"] = mem["grasp_count"] + 1
        return {"memory": mem, "act": {"tool": "act_grasp", "args": {"x": c[0], "y": c[1]}}}
    if stage == "grasped?":
        fing = float(np.sum(cur.fingers))
        held = 0.02 < fing < 0.07 and cur.tcp[2] > 0.045
        mem["log"] = (mem["log"] + " grasp#%d fingers=%.4f z=%.3f held=%s;" % (
            mem["grasp_count"], fing, cur.tcp[2], held))[-600:]
        if not held:
            if mem["grasp_count"] >= 3:
                mem["stage"] = "giveup"
                return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 50, "gripper": 1.0}}}
            mem["stage"] = "grasp"
            return {"memory": mem, "act": {"tool": "act_release_up", "args": {}}}
        # cube centre relative to TCP (y is centred by the fingers; x keeps the approach offset)
        ox = float(np.clip(mem["cube_xy"][0] - cur.tcp[0], -0.012, 0.012))
        mem["grasp_offset"] = [ox, 0.0]
        stage = "trace"
    if stage in ("trace", "traced"):
        if stage == "traced":
            fing = float(np.sum(cur.fingers))
            if fing < 0.02:  # cube lost
                mem["stage"] = "grasp"
                mem["log"] = (mem["log"] + " cube lost after trace;")[-600:]
                return {"memory": mem, "act": {"tool": "act_release_up", "args": {}}}
        mem["stage"] = "traced"
        mem["trace_count"] = mem["trace_count"] + 1
        return {"memory": mem, "act": {"tool": "act_trace", "args": {
            "path": mem["red_path"], "offset": mem["grasp_offset"],
            "close_loop": mem["trace_count"] >= 2}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 50, "gripper": -1.0}}}


def _clipnorm(v, cap):
    n = float(np.linalg.norm(v))
    if n > cap:
        return v * (cap / n)
    return v


def _servo(robot, target, cap, gripper, tol, max_steps, settle_v=0.002, offset=None):
    """Closed-loop translation toward a world target (TCP or TCP+offset). Returns (steps, err)."""
    off = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    prev = None
    n = 0
    err = 0.0
    for n in range(1, max_steps + 1):
        p = np.asarray(robot.now().tcp[:3], dtype=np.float64) + off
        e = tgt - p
        err = float(np.linalg.norm(e))
        v = 1.0 if prev is None else float(np.linalg.norm(p - prev))
        if err < tol and v < settle_v:
            return n - 1, err
        a = _clipnorm(e / 0.1, cap)
        prev = p
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, gripper)
    return n, err


def act_watch(robot, max_steps=140):
    """Hold still with the gripper open until the lamp turns green (cue phase). Returns steps and lamp."""
    n = 0
    red_seen = 0
    for n in range(1, max_steps + 1):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        f = robot.now()
        if _red_center(f) is not None:
            red_seen += 1
        if _lamp(f) == "green":
            break
    return {"steps": n, "lamp": _lamp(robot.now()), "red_frames": red_seen, "step": int(robot.now().step)}


def act_hold(robot, steps=10, gripper=1.0):
    """Send zero motion for `steps` steps with the given gripper target."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))
    return {"held": int(steps), "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_release_up(robot):
    """Open the gripper and rise to hover height (used before a re-grasp)."""
    for _ in range(6):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    t = robot.now().tcp
    n, e = _servo(robot, [t[0], t[1], HOVER_Z + 0.03], 0.5, 1.0, 0.01, 30, settle_v=1.0)
    return {"steps": n + 6, "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_grasp(robot, x, y, z_grasp=GRASP_Z, z_lift=TRACE_Z):
    """Open, move above (x, y), descend, close and lift to the tracing height. Reports measured fingers/TCP."""
    out = {}
    t0 = robot.now().tcp
    chk = robot.dry_run(float(x), float(y), float(z_grasp), float(t0[3]), float(t0[4]), float(t0[5]), mode="absolute")
    out["dry_ok"] = bool(chk.get("within_joint_limits", True))
    n1, e1 = _servo(robot, [x, y, HOVER_Z], 0.6, 1.0, 0.008, 60, settle_v=0.01)
    n2, e2 = _servo(robot, [x, y, z_grasp], 0.35, 1.0, 0.003, 50, settle_v=0.002)
    for _ in range(10):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    f_closed = float(np.sum(robot.now().fingers))
    t = robot.now().tcp
    n3, e3 = _servo(robot, [t[0], t[1], z_lift], 0.35, -1.0, 0.004, 30, settle_v=0.003)
    tt = robot.now().tcp
    out.update({"approach_steps": n1, "descend_steps": n2, "descend_err": round(e2, 4),
                "lift_steps": n3, "fingers_closed": round(f_closed, 4),
                "fingers_lifted": round(float(np.sum(robot.now().fingers)), 4),
                "tcp": [round(float(v), 4) for v in tt[:3]], "step": int(robot.now().step)})
    return out


def act_trace(robot, path, offset, z=TRACE_Z, cap=0.32, lookahead=3, hold_steps=12, close_loop=False):
    """Carry the held cube along `path` ([[x, y], ...], cube-centre coordinates; cube = TCP + offset)
    at height z: move to the start, pursue the path in order, stop and hold at the end
    (optionally then continue back to the start). Reports progress, max deviation and steps."""
    P = np.asarray(path, dtype=np.float64)
    off = np.array([float(offset[0]), float(offset[1]), 0.0])
    N = len(P)
    out = {"n": N}
    # 1) go to start (lifted)
    n0, e0 = _servo(robot, [P[0, 0], P[0, 1], z], 0.5, -1.0, 0.004, 60, settle_v=0.003, offset=off)
    out["start_steps"] = n0
    out["start_err"] = round(e0, 4)
    k = 0
    steps = 0
    maxdev = 0.0
    while k < N - 1 and steps < 260:
        t = robot.now().tcp
        c = np.array([t[0], t[1]]) + off[:2]
        # progress: nearest point within a forward window
        hi = min(N, k + 12)
        d = np.linalg.norm(P[k:hi] - c, axis=1)
        j = int(np.argmin(d))
        k = k + j
        dev = float(d[j])
        if steps > 3:
            maxdev = max(maxdev, dev)
        tgt = P[min(N - 1, k + lookahead)]
        e = np.array([tgt[0] - c[0], tgt[1] - c[1], z - t[2]])
        a = _clipnorm(e / 0.1 * 1.2, cap)
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, -1.0)
        steps += 1
        if float(np.sum(robot.now().fingers)) < 0.01:
            out["dropped_at"] = k
            break
    out["trace_steps"] = steps
    out["progress"] = k
    out["max_dev"] = round(maxdev, 4)
    n1, e1 = _servo(robot, [P[-1, 0], P[-1, 1], z], 0.25, -1.0, 0.003, 30, settle_v=0.002, offset=off)
    out["end_err"] = round(e1, 4)
    for _ in range(int(hold_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    if close_loop:
        n2, e2 = _servo(robot, [P[0, 0], P[0, 1], z], 0.25, -1.0, 0.003, 30, settle_v=0.002, offset=off)
        out["loop_err"] = round(e2, 4)
        for _ in range(int(hold_steps)):
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    out["step"] = int(robot.now().step)
    out["tcp"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    return out
