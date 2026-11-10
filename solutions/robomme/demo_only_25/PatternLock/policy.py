"""PatternLock policy: recover the node sequence shown in the instruction video, then retrace it with the stick."""

GRID = 0.1           # lattice spacing of the pattern buttons (m), measured in all 25 demonstrations
SNAP_TOL = 0.02      # TCP within this xy distance of a lattice node counts as visiting it
BUTTON_Z = 0.02      # button top height (m) from front-depth back-projection
STEP_LEN = 0.005     # per-control-step Cartesian advance (m); expert moves ~4-6 mm/frame
RED_FRAC = 0.2       # fraction of red pixels near a projected node to call it lit
WRIST_RED_FRAC = 0.25  # red fraction in a 15x15 wrist-camera patch around a projected node


def _snap(p, origin):
    i = round((p[0] - origin[0]) / GRID)
    j = round((p[1] - origin[1]) / GRID)
    n = (origin[0] + GRID * i, origin[1] + GRID * j)
    d = math.hypot(p[0] - n[0], p[1] - n[1])
    return (i, j), n, d


def _nodes_from_path(path, origin):
    """Ordered distinct lattice nodes the TCP path passes over (consecutive duplicates removed)."""
    seq = []
    for p in path:
        key, n, d = _snap(p, origin)
        if d < SNAP_TOL and (not seq or seq[-1][0] != key):
            seq.append((key, n))
    return [list(n) for _, n in seq]


def _project(K, E, p):
    E = np.asarray(E, dtype=float)
    K = np.asarray(K, dtype=float)
    pc = E[:, :3] @ np.asarray(p, dtype=float) + E[:, 3]
    if pc[2] <= 1e-6:
        return None
    uv = K @ pc
    return uv[0] / uv[2], uv[1] / uv[2]


def _patch_stats(im, uv, r=3):
    if uv is None:
        return None
    h, w = im.shape[0], im.shape[1]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if u < r or v < r or u >= w - r or v >= h - r:
        return None
    p = im[v - r:v + r + 1, u - r:u + r + 1].astype(int)
    red = (p[..., 0] > 150) & (p[..., 1] < 90) & (p[..., 2] < 90)
    white = (p[..., 0] > 170) & (p[..., 1] > 170) & (p[..., 2] > 170)
    return float(red.mean()), float(white.mean())


def _vision_nodes(frames, origin):
    """Fallback: order in which lattice buttons first light up red in the front camera during the video.

    Candidate nodes are lattice points around origin whose projection looks like a button
    (white or red) in the first video frame. A node is accepted when lit in two consecutive
    sampled frames, not already in the sequence, and adjacent to the previous accepted node.
    """
    if not frames:
        return []
    f0 = frames[0]
    K, E = f0.front_K, f0.front_extrinsic
    if K is None or E is None:
        return []
    cands = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            n = (origin[0] + GRID * i, origin[1] + GRID * j)
            uv = _project(K, E, (n[0], n[1], BUTTON_Z))
            for fs in frames[::max(1, len(frames) // 8)]:
                s = _patch_stats(fs.front_rgb, uv)
                if s is not None and s[0] + s[1] > 0.15:
                    cands.append((n, uv))
                    break
    seq = []
    prev_lit = set()
    for f in frames:
        lit = set()
        wk, we = getattr(f, "wrist_K", None), getattr(f, "wrist_extrinsic", None)
        for k, (n, uv) in enumerate(cands):
            s = _patch_stats(f.front_rgb, uv)
            if s is not None and s[0] > RED_FRAC:
                lit.add(k)
                continue
            if wk is not None and we is not None:
                s = _patch_stats(f.wrist_rgb, _project(wk, we, (n[0], n[1], BUTTON_Z)), 7)
                if s is not None and s[0] > WRIST_RED_FRAC:
                    lit.add(k)
        for k in sorted(lit & prev_lit):
            n = cands[k][0]
            if any(abs(n[0] - q[0]) < 1e-6 and abs(n[1] - q[1]) < 1e-6 for q in seq):
                continue
            if seq and math.hypot(n[0] - seq[-1][0], n[1] - seq[-1][1]) > GRID * 1.5:
                continue
            seq.append([n[0], n[1]])
        prev_lit = lit
    # The final node lights up only at the very end of the video; accept it from the last frame alone.
    for k in sorted(prev_lit):
        n = cands[k][0]
        if any(abs(n[0] - q[0]) < 1e-6 and abs(n[1] - q[1]) < 1e-6 for q in seq):
            continue
        if seq and math.hypot(n[0] - seq[-1][0], n[1] - seq[-1][1]) > GRID * 1.5:
            continue
        seq.append([n[0], n[1]])
    if not seq:
        return []
    # The video starts with the stick on the start node; make sure it leads the sequence.
    if math.hypot(seq[0][0] - origin[0], seq[0][1] - origin[1]) > 1e-6:
        seq.insert(0, [origin[0], origin[1]])
    return seq


def perceive_pattern(obs):
    """Measure the demonstrated pattern from instruction-video frames ("demo" phase).

    Returns {"tcp_nodes": [[x,y],...] from the recorded TCP path, "vision_nodes": [[x,y],...]
    from red button highlights in the front camera, "n_demo": count of demo frames}.
    """
    demo = [f for f in obs.frames if f.phase == "demo"]
    if not demo:
        return {"tcp_nodes": [], "vision_nodes": [], "n_demo": 0}
    origin = (float(demo[0].tcp[0]), float(demo[0].tcp[1]))
    execs = [f for f in obs.frames if f.phase != "demo"]
    # The execution starts on the same start node as the video; its TCP anchors the vision lattice.
    v_origin = (float(execs[0].tcp[0]), float(execs[0].tcp[1])) if execs else origin
    path = [(float(f.tcp[0]), float(f.tcp[1])) for f in demo]
    span = max(math.hypot(p[0] - origin[0], p[1] - origin[1]) for p in path)
    tcp_nodes = _nodes_from_path(path, origin) if span > GRID * 0.5 else []
    vis = _vision_nodes(demo, v_origin)
    return {"tcp_nodes": tcp_nodes, "vision_nodes": vis, "n_demo": len(demo)}


def _choose(tcp_nodes, vision_nodes):
    if tcp_nodes and len(tcp_nodes) >= 2:
        return tcp_nodes
    if vision_nodes and len(vision_nodes) >= 2:
        return vision_nodes
    return []


def perceive_decide(obs, goal, memory, last):
    """Decide the next action for PatternLock.

    First decision with instruction frames: extract the demonstrated node sequence (TCP path,
    vision fallback), store it as memory.plan, then call act_trace_pattern. Afterwards hold.
    """
    mem = {k: memory.get(k) for k in ("plan", "vision_plan", "traced", "decisions")}
    mem["decisions"] = int(mem["decisions"] or 0) + 1
    cur = obs.current
    if not mem["plan"]:
        info = perceive_pattern(obs)
        if info["n_demo"] > 0:
            mem["vision_plan"] = info["vision_nodes"]
            mem["plan"] = _choose(info["tcp_nodes"], info["vision_nodes"])
    if cur.phase == "demo":
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 1}}}
    if mem["plan"] and not mem["traced"]:
        mem["traced"] = True
        return {"memory": mem, "act": {"tool": "act_trace_pattern", "args": {"nodes": mem["plan"]}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


def _safe_step(robot, x, y, z, rpy):
    """One control step; an IK/motion rejection of a single target is reported, not fatal."""
    try:
        robot.step(x, y, z, rpy[0], rpy[1], rpy[2], -1.0)
        return True
    except Exception:
        return False


def _move_to(robot, target, z, rpy, settle=3, max_steps=200):
    f = robot.now()
    start = np.array([float(f.tcp[0]), float(f.tcp[1])])
    goal = np.array([float(target[0]), float(target[1])])
    dist = float(np.linalg.norm(goal - start))
    n = max(1, int(math.ceil(dist / STEP_LEN)))
    used = 0
    for k in range(1, n + 1):
        p = start + (goal - start) * (k / n)
        _safe_step(robot, p[0], p[1], z, rpy)
        used += 1
    for _ in range(max_steps):
        f = robot.now()
        err = math.hypot(float(f.tcp[0]) - goal[0], float(f.tcp[1]) - goal[1])
        if err < 0.004 and settle <= 0:
            break
        _safe_step(robot, goal[0], goal[1], z, rpy)
        used += 1
        if err < 0.004:
            settle -= 1
    f = robot.now()
    return used, math.hypot(float(f.tcp[0]) - goal[0], float(f.tcp[1]) - goal[1])


def act_trace_pattern(robot, nodes):
    """Retrace the node sequence [[x,y],...] with the stick in straight segments at the start height.

    Skips leading nodes the stick already sits on. Keeps the starting z and orientation.
    Returns per-node final xy errors (m) and steps used.
    """
    f = robot.now()
    z = float(f.tcp[2])
    rpy = (float(f.tcp[3]), float(f.tcp[4]), float(f.tcp[5]))
    errs = []
    total = 0
    for idx, nd in enumerate(nodes):
        f = robot.now()
        d = math.hypot(float(f.tcp[0]) - nd[0], float(f.tcp[1]) - nd[1])
        if idx == 0 and d < 0.01:
            errs.append(d)
            continue
        used, err = _move_to(robot, nd, z, rpy)
        total += used
        errs.append(err)
    return {"node_errors": errs, "steps": total, "n_nodes": len(nodes)}


def act_hold(robot, steps=10):
    """Hold the current pose for `steps` control steps (stick pressed, gripper closed)."""
    f = robot.now()
    p = [float(v) for v in f.tcp]
    for _ in range(int(max(1, min(steps, 200)))):
        robot.step(p[0], p[1], p[2], p[3], p[4], p[5], -1.0)
    return {"held": int(steps)}
