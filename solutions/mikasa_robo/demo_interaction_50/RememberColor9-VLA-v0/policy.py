"""RememberColor9: remember the cue cube's colour, then reach/hover over the same-coloured cube and hold still.

Perception uses the fixed front RGB-D camera: pixels are back-projected to world points with front_K and
front_extrinsic, points 3-6 cm above the table are clustered into cubes, and each cube's colour is the median RGB
of its top face. The cue is the frame(s) where exactly one cube is visible; candidates are frames with >= 5 cubes.
"""

TABLE_ZMIN = 0.03          # cube top faces are at ~0.041 m (cube edge 0.04 m)
TABLE_ZMAX = 0.065
CLUSTER_R = 0.035          # cubes are ~0.04 wide and >= 0.07 apart
MIN_CANDIDATES = 5
GAIN = 1.5                 # P gain on position error (command units: 0.1 m)
HOVER = 0.06               # extra height while still far in xy (z goal = z + min(HOVER, xy_err))
Z_LEVELS = [0.045, 0.035, 0.025]  # TCP heights tried over the target cube centre (top face at 0.04)
HOLD_STEPS = [3, 3, 6]
STEP_LIMIT = 25
MATCH_OK = 40.0           # demos: true match L1 RGB distance 0-6, wrong colours >= 58


def _points(frame, cam):
    depth = getattr(frame, cam + "_depth").astype(np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    Pc = np.stack([x, y, depth], -1)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    return Pw, depth > 0


def _detect_cubes(frame, cam="front"):
    """List of cubes {xy, ztop, rgb, n} visible above the table."""
    P, valid = _points(frame, cam)
    img = getattr(frame, cam + "_rgb").astype(np.float64)
    z = P[..., 2]
    m = valid & (z > TABLE_ZMIN) & (z < TABLE_ZMAX) & (P[..., 0] > -0.45) & (P[..., 0] < 0.35) & (np.abs(P[..., 1]) < 0.45)
    pts = P[m]
    cols = img[m]
    centers = []
    members = []
    for i in range(len(pts)):
        p = pts[i, :2]
        hit = -1
        for j in range(len(centers)):
            c = centers[j]
            if (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 < CLUSTER_R ** 2:
                hit = j
                break
        if hit < 0:
            centers.append([float(p[0]), float(p[1])])
            members.append([i])
        else:
            members[hit].append(i)
            idx = members[hit]
            centers[hit] = [float(np.mean(pts[idx, 0])), float(np.mean(pts[idx, 1]))]
    out = []
    for idx in members:
        if len(idx) < 6:
            continue
        q = pts[idx]
        zt = float(q[:, 2].max())
        top = q[:, 2] > zt - 0.004
        if top.sum() < 4:
            continue
        rgb = np.median(cols[idx][top], axis=0)
        out.append({"xy": [float(q[top, 0].mean()), float(q[top, 1].mean())], "ztop": zt,
                    "rgb": [float(rgb[0]), float(rgb[1]), float(rgb[2])], "n": int(len(idx))})
    return out


def _rgb_dist(a, b):
    return float(sum(abs(float(a[k]) - float(b[k])) for k in range(3)))


def _mem(memory, name, default):
    v = memory.get(name)
    return default if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Track cue colour, candidate cubes and the reach stage; choose the next act."""
    mem = {
        "stage": _mem(memory, "stage", "cue"),
        "cue_rgb": memory.get("cue_rgb"),
        "cue_step": memory.get("cue_step"),
        "cue_frames": _mem(memory, "cue_frames", 0),
        "cubes_step": memory.get("cubes_step"),
        "candidates": _mem(memory, "candidates", []),
        "candidate_rgbs": _mem(memory, "candidate_rgbs", []),
        "target": memory.get("target"),
        "target_rgb": memory.get("target_rgb"),
        "match_dist": memory.get("match_dist"),
        "match_margin": memory.get("match_margin"),
        "level": _mem(memory, "level", 0),
        "log": _mem(memory, "log", ""),
        "decisions": _mem(memory, "decisions", 0) + 1,
    }
    cur = obs.current
    steps_left = STEP_LIMIT - int(cur.step)

    # 1. cue and candidate perception on every new frame (cheap enough), until a target is chosen
    if mem["target"] is None:
        cands = None
        for f in obs.frames:
            if "front_depth" not in f.sensors:
                continue
            cubes = _detect_cubes(f)
            if len(cubes) == 1:
                if mem["cue_rgb"] is None:
                    mem["cue_rgb"] = cubes[0]["rgb"]
                    mem["cue_step"] = int(f.step)
                mem["cue_frames"] = int(mem["cue_frames"]) + 1
            elif len(cubes) >= MIN_CANDIDATES:
                cands = cubes
                if mem["cubes_step"] is None:
                    mem["cubes_step"] = int(f.step)
        if cands is not None and mem["cue_rgb"] is not None:
            dbest = min(_rgb_dist(c["rgb"], mem["cue_rgb"]) for c in cands)
            if dbest > MATCH_OK and "wrist_depth" in cur.sensors:
                # target possibly occluded in the front view: add wrist-camera cubes not already listed
                for w in _detect_cubes(cur, "wrist"):
                    if all(math.hypot(w["xy"][0] - c["xy"][0], w["xy"][1] - c["xy"][1]) > 0.03 for c in cands):
                        cands.append(w)
                mem["log"] = "wrist fallback used (front best match %.0f)" % dbest
        if cands is not None:
            mem["candidates"] = [[c["xy"][0], c["xy"][1], c["ztop"]] for c in cands]
            mem["candidate_rgbs"] = [c["rgb"] for c in cands]
            if mem["cue_rgb"] is not None:
                d = [_rgb_dist(c["rgb"], mem["cue_rgb"]) for c in cands]
                order = sorted(range(len(d)), key=lambda k: d[k])
                best = order[0]
                mem["match_dist"] = d[best]
                mem["match_margin"] = (d[order[1]] - d[best]) if len(order) > 1 else 999.0
            else:
                best = 0  # no cue was seen: no information, pick any candidate
                mem["match_dist"] = None
                mem["match_margin"] = 0.0
            c = cands[best]
            mem["target"] = [c["xy"][0], c["xy"][1], c["ztop"] - 0.02]
            mem["target_rgb"] = c["rgb"]
            mem["stage"] = "reach"
            mem["level"] = 0
        else:
            mem["stage"] = "wait"
            return {"memory": mem, "act": {"tool": "act_wait_for_candidates", "args": {"max_steps": max(1, steps_left)}}}

    # 2. reach/hold ladder over the chosen cube
    lvl = int(mem["level"])
    if lvl >= len(Z_LEVELS):
        lvl = len(Z_LEVELS) - 1
    t = mem["target"]
    mem["level"] = lvl + 1
    mem["stage"] = "reach_level_%d" % lvl
    return {"memory": mem, "act": {"tool": "act_reach_hold", "args": {
        "x": t[0], "y": t[1], "z": Z_LEVELS[lvl], "hold_steps": HOLD_STEPS[lvl], "max_steps": max(1, steps_left)}}}


def act_wait_for_candidates(robot, max_steps=25):
    """Hold still (gripper open) until >= 5 cubes are visible in the front camera or max_steps elapse.
    Returns the step at which candidates appeared and how many cubes were seen."""
    n = 0
    for k in range(int(max_steps)):
        f = robot.now()
        cubes = _detect_cubes(f)
        n = len(cubes)
        if n >= MIN_CANDIDATES:
            return {"appeared": True, "step": int(f.step), "n_cubes": n, "waited": k}
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    n = len(_detect_cubes(f))
    return {"appeared": n >= MIN_CANDIDATES, "step": int(f.step), "n_cubes": n, "waited": int(max_steps)}


def act_reach_hold(robot, x, y, z, hold_steps=4, max_steps=20, brake_tol=0.012):
    """Move the TCP (P control, no rotation command) to (x, y, z) via a hover that shrinks with xy error. Once the
    error is below brake_tol (m), send zero arm commands (target = measured pose) so the arm comes to rest, and keep
    braking for hold_steps steps. Returns measured errors, braking step and a short trace."""
    goal = np.array([float(x), float(y), float(z)])
    prev = np.array(robot.now().tcp[:3], dtype=np.float64)
    q_prev = np.array(robot.now().joints, dtype=np.float64)
    brake_at = None
    held = 0
    log = []
    for k in range(int(max_steps)):
        f = robot.now()
        p = np.array(f.tcp[:3], dtype=np.float64)
        q = np.array(f.joints, dtype=np.float64)
        v = p - prev
        qv = float(np.linalg.norm(q - q_prev) * 20.0)
        prev = p
        q_prev = q
        e = goal - p
        exy = float(math.hypot(e[0], e[1]))
        err = float(np.linalg.norm(e))
        log.append([int(f.step), round(err, 4), round(float(np.linalg.norm(v)), 4), round(qv, 3)])
        if brake_at is None and err < float(brake_tol):
            brake_at = int(f.step)
        if brake_at is not None:
            if held >= int(hold_steps):
                break
            held += 1
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            continue
        zg = goal[2] + min(HOVER, exy)
        tgt = np.array([goal[0], goal[1], max(zg, goal[2])])
        a = np.clip(GAIN * (tgt - p) / 0.1, -1.0, 1.0)
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, 1.0)
    p = np.array(robot.now().tcp[:3], dtype=np.float64)
    return {"brake_at": brake_at, "held": held, "final_err": [round(float(q), 4) for q in (goal - p)],
            "step": int(robot.now().step), "trace": log[-8:]}
