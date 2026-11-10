"""MoveCube policy: classify the demonstrated manner (pick / push / peg) from the
task video, perceive cube, target and stick with front RGB-D + calibration, then
reproduce the same manner with closed-loop Cartesian steps."""

ROLL = math.pi
PITCH = 0.0
# Geometry constants derived from the 25 assigned demonstrations (expert waypoints vs perceived objects).
PICK_HOVER_Z = 0.15
PICK_GRASP_Z = 0.02
PICK_PLACE_Z = 0.008
PUSH_Z = 0.02
PUSH_START_BACK = 0.05      # TCP start = cube - 0.05*u
PUSH_END_BACK = 0.02        # TCP end = target - 0.02*u
PEG_HOVER_Z = 0.2
PEG_GRASP_Z = 0.01
PEG_PUSH_Z = 0.02
PEG_GRASP_OFFSET = 0.05     # grasp 0.05 m from stick centre, toward the stick end with smaller x
PEG_ALONG = 0.10            # cube contacts stick 0.10 m from TCP along the stick
PEG_START_BACK = 0.10       # stick line 0.10 m behind cube at the start
PEG_END_BACK = 0.03         # stick line 0.03 m behind target at the end (cube half + stick half)


def _wrap(a, period):
    """Wrap angle into (-period/2, period/2]."""
    return a - period * math.floor(a / period + 0.5)


def _world_points(K, E, depth):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    E = np.asarray(E, dtype=float)
    R = E[:, :3]
    t = E[:, 3]
    P = ((pc - t) @ R).reshape(H, W, 3)
    P[depth <= 0] = np.nan
    return P


def _masks(rgb, P):
    a = rgb.astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    with np.errstate(invalid="ignore"):
        Z = np.nan_to_num(P[..., 2], nan=-1.0)
        X = np.nan_to_num(P[..., 0], nan=9.0)
        Y = np.nan_to_num(P[..., 1], nan=9.0)
    inws = (np.abs(X) < 0.35) & (np.abs(Y) < 0.4) & (Z > -0.01) & (Z < 0.3)
    red = (R > 90) & (G < 0.35 * R) & (B < 0.35 * R) & inws & (Z > 0.003)
    stick = (B > 0.72 * G) & (R > 1.5 * G) & (G > 40) & (~red) & inws & (Z > 0.003)
    purple = (B > R - 20) & (B > G + 40) & inws & (Z < 0.02)
    white = (R > 190) & (G > 190) & (B > 190) & inws & (Z < 0.02)
    return red, stick, purple, white


def _cube(P, red):
    pts = P[red]
    if len(pts) < 15:
        return None
    zt = float(np.max(pts[:, 2]))
    top = pts[pts[:, 2] > zt - 0.006][:, :2]
    if len(top) < 5:
        return None
    c = top.mean(0)
    best = None
    for ang in np.arange(0.0, math.pi / 2, math.pi / 90):
        d = np.array([math.cos(ang), math.sin(ang)])
        n = np.array([-d[1], d[0]])
        q = top - c
        ext = np.ptp(q @ d) * np.ptp(q @ n)
        if best is None or ext < best[0]:
            best = (ext, ang)
    return {"xy": [float(c[0]), float(c[1])], "top_z": zt, "yaw": float(_wrap(best[1], math.pi / 2)),
            "n": int(len(pts))}


def _target(P, purple, white):
    pts = np.concatenate([P[purple], P[white]])[:, :2]
    if len(purple.nonzero()[0]) < 15:
        return None
    c = np.median(P[purple][:, :2], 0)
    for _ in range(3):
        r = np.linalg.norm(pts - c, axis=1)
        keep = pts[r < 0.06]
        if len(keep) < 10:
            break
        c = (keep.min(0) + keep.max(0)) / 2
    return {"xy": [float(c[0]), float(c[1])], "n": int(len(pts))}


def _stick(P, m):
    pts = P[m]
    pts = pts[(pts[:, 2] > 0.013) & (pts[:, 2] < 0.03)][:, :2]
    if len(pts) < 30:
        return None
    c = np.median(pts, 0)
    pts = pts[np.linalg.norm(pts - c, axis=1) < 0.16]
    d = np.array([1.0, 0.0])
    n = np.array([0.0, 1.0])
    for _ in range(3):
        if len(pts) < 20:
            return None
        c = pts.mean(0)
        q = pts - c
        w, v = np.linalg.eigh(q.T @ q)
        d = v[:, 1]
        n = np.array([-d[1], d[0]])
        pts = pts[np.abs(q @ n) < 0.02]
    q = pts - c
    s = q @ d
    lo, hi = float(np.percentile(s, 1)), float(np.percentile(s, 99))
    c = c + d * (lo + hi) / 2 + n * float(np.median(q @ n))
    return {"center": [float(c[0]), float(c[1])], "dir": [float(d[0]), float(d[1])],
            "length": hi - lo, "n": int(len(pts))}


def _scene(frame):
    depth = getattr(frame, "front_depth", None)
    if depth is None:
        return None
    P = _world_points(np.asarray(frame.front_K, dtype=float), frame.front_extrinsic, np.asarray(depth))
    red, st, pu, wh = _masks(np.asarray(frame.front_rgb), P)
    return {"cube": _cube(P, red), "target": _target(P, pu, wh), "stick": _stick(P, st)}


def _classify_demo(frames):
    """Return (manner, evidence) from task-video frames: 'pick', 'push' or 'peg'."""
    demo = [f for f in frames if getattr(f, "phase", "") == "demo"]
    if not demo:
        return None, "no demo frames"
    # 1) proprioception: finger width while closed (push 0.0, peg ~0.009, pick ~0.018)
    fw = []
    for f in demo:
        g = getattr(f, "fingers", None)
        if g is not None:
            g = np.asarray(g, dtype=float).ravel()
            if g.size and np.all(np.isfinite(g)):
                fw.append(float(np.mean(g[:2])))
    if len(fw) > 5 and max(fw) - min(fw) > 0.01 and min(fw) < 0.03:
        mn = min(fw)
        if mn < 0.0045:
            return "push", "fingers min %.4f" % mn
        if mn < 0.0135:
            return "peg", "fingers min %.4f" % mn
        return "pick", "fingers min %.4f" % mn
    stride = max(1, len(demo) // 80)
    sel = demo[::stride]
    # 2) depth: did the stick or the cube rise during the video?
    if getattr(sel[0], "front_depth", None) is not None:
        smax, cmax = -1.0, -1.0
        for f in sel:
            P = _world_points(np.asarray(f.front_K, dtype=float), f.front_extrinsic, np.asarray(f.front_depth))
            red, st, pu, wh = _masks(np.asarray(f.front_rgb), P)
            if st.sum() > 20:
                smax = max(smax, float(np.percentile(P[st][:, 2], 90)))
            if red.sum() > 5:
                cmax = max(cmax, float(np.percentile(P[red][:, 2], 95)))
        if smax > 0.08:
            return "peg", "stick lifted %.3f" % smax
        if cmax > 0.08:
            return "pick", "cube lifted %.3f" % cmax
        return "push", "nothing lifted (stick %.3f cube %.3f)" % (smax, cmax)
    # 3) RGB only: pixel-area growth (objects lifted toward the camera)
    sc, rc = [], []
    for f in sel:
        a = np.asarray(f.front_rgb).astype(int)
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        red = (R > 90) & (G < 0.35 * R) & (B < 0.35 * R)
        st = (B > 0.72 * G) & (R > 1.5 * G) & (G > 40) & (~red)
        sc.append(float(st.sum()))
        rc.append(float(red.sum()))
    s0 = max(np.median(sc[:3]), 1.0)
    r0 = max(np.median(rc[:3]), 1.0)
    if max(sc) / s0 > 1.2:
        return "peg", "stick area x%.2f cube x%.2f" % (max(sc) / s0, max(rc) / r0)
    if max(rc) / r0 > 1.2:
        return "pick", "cube area x%.2f" % (max(rc) / r0)
    return "push", "rgb no lift (stick x%.2f cube x%.2f)" % (max(sc) / s0, max(rc) / r0)


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _plan(manner, scene):
    """Waypoint parameters for one attempt; pure function of perceived scene."""
    cube = scene.get("cube")
    tgt = scene.get("target")
    if cube is None or tgt is None:
        return None
    c = np.array(cube["xy"])
    t = np.array(tgt["xy"])
    if manner == "pick":
        return {"manner": "pick", "grasp": [float(c[0]), float(c[1])], "yaw": cube["yaw"],
                "place": [float(t[0]), float(t[1])]}
    dv = t - c
    dist = float(np.linalg.norm(dv))
    if dist < 1e-4:
        return None
    u = dv / dist
    ang = math.atan2(u[1], u[0])
    if manner == "push":
        s = c - PUSH_START_BACK * u
        e = t - PUSH_END_BACK * u
        return {"manner": "push", "start": [float(s[0]), float(s[1])], "end": [float(e[0]), float(e[1])],
                "yaw": float(_wrap(ang, math.pi))}
    st = scene.get("stick")
    if st is None:
        return None
    sc = np.array(st["center"])
    sd = np.array(st["dir"])
    if sd[0] > 0:
        sd = -sd                      # sd points to the end with smaller x (grasped end)
    g = sc + PEG_GRASP_OFFSET * sd
    yaw_g = float(_wrap(math.atan2(sd[1], sd[0]), math.pi))
    L0 = -sd                          # from grasp toward the long part of the stick
    yaw_f = float(_wrap(ang + math.pi / 2, math.pi))
    L = _rot(L0, yaw_f - yaw_g)
    s = c - PEG_ALONG * L - PEG_START_BACK * u
    e = t - PEG_ALONG * L - PEG_END_BACK * u
    return {"manner": "peg", "grasp": [float(g[0]), float(g[1])], "yaw_g": yaw_g, "yaw_f": yaw_f,
            "start": [float(s[0]), float(s[1])], "end": [float(e[0]), float(e[1])]}


def _goto(robot, x, y, z, yaw, grip, lead=0.016, yaw_lead=0.08, tol=0.004, settle=40, max_steps=400):
    """Carrot-following move: command measured TCP + clipped error; returns final error."""
    goal = np.array([x, y, z], dtype=float)
    n = 0
    near = 0
    err = 9.0
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        p = tcp[:3]
        dv = goal - p
        dist = float(np.linalg.norm(dv))
        dyaw = _wrap(yaw - float(tcp[5]), 2 * math.pi)
        err = dist
        if dist < tol and abs(dyaw) < 0.03:
            near += 1
            if near >= 3:
                break
        if dist > lead:
            cmd = p + dv * (lead / dist)
        else:
            cmd = goal + 0.5 * dv     # small overshoot term against steady-state lag
        if abs(dyaw) > yaw_lead:
            cy = float(tcp[5]) + yaw_lead * (1 if dyaw > 0 else -1)
        else:
            cy = yaw
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), ROLL, PITCH, cy, grip)
        n += 1
        if dist < tol * 3:
            settle -= 1
            if settle <= 0:
                break
    return err


def _hold(robot, grip, steps):
    tcp = np.asarray(robot.now().tcp, dtype=float)
    for _ in range(steps):
        robot.step(float(tcp[0]), float(tcp[1]), float(tcp[2]), ROLL, PITCH, float(tcp[5]), grip)


def act_pick_place(robot, grasp, yaw, place):
    """Pick the cube at grasp [x,y] with gripper yaw, carry it and place it at place [x,y], then release and lift."""
    gx, gy = grasp
    px, py = place
    _goto(robot, gx, gy, PICK_HOVER_Z, yaw, 1.0)
    _goto(robot, gx, gy, PICK_GRASP_Z, yaw, 1.0, lead=0.012, tol=0.003)
    _hold(robot, -1.0, 10)
    _goto(robot, gx, gy, PICK_HOVER_Z, yaw, -1.0)
    _goto(robot, px, py, PICK_HOVER_Z, yaw, -1.0)
    _goto(robot, px, py, PICK_PLACE_Z, yaw, -1.0, lead=0.012, tol=0.006, settle=15)
    _hold(robot, 1.0, 10)
    _goto(robot, px, py, PICK_HOVER_Z, yaw, 1.0)
    f = np.asarray(robot.now().fingers, dtype=float).tolist()
    return {"done": "pick_place", "fingers": f}


def act_push(robot, start, end, yaw):
    """Close the gripper and push the cube along the straight line start [x,y] -> end [x,y] at table height."""
    sx, sy = start
    ex, ey = end
    tcp = np.asarray(robot.now().tcp, dtype=float)
    _hold(robot, -1.0, 5)
    _goto(robot, sx, sy, max(0.10, PUSH_Z), yaw, -1.0)
    _goto(robot, sx, sy, PUSH_Z, yaw, -1.0, lead=0.012, tol=0.003)
    _goto(robot, ex, ey, PUSH_Z, yaw, -1.0, lead=0.012, tol=0.003, settle=20, max_steps=300)
    _hold(robot, 1.0, 10)
    _goto(robot, ex, ey, 0.10, yaw, 1.0)
    return {"done": "push", "tcp": np.asarray(robot.now().tcp, dtype=float).tolist()}


def act_peg_push(robot, grasp, yaw_g, yaw_f, start, end):
    """Grasp the stick at grasp [x,y] (yaw_g), lift, rotate to yaw_f, then sweep the cube start [x,y] -> end [x,y] with the stick and release."""
    gx, gy = grasp
    sx, sy = start
    ex, ey = end
    _goto(robot, gx, gy, PEG_HOVER_Z, yaw_g, 1.0)
    _goto(robot, gx, gy, PEG_GRASP_Z, yaw_g, 1.0, lead=0.012, tol=0.003)
    _hold(robot, -1.0, 10)
    _goto(robot, gx, gy, PEG_HOVER_Z, yaw_g, -1.0)
    _goto(robot, sx, sy, PEG_HOVER_Z, yaw_f, -1.0, yaw_lead=0.05)
    _goto(robot, sx, sy, PEG_PUSH_Z, yaw_f, -1.0, lead=0.012, tol=0.003)
    _goto(robot, ex, ey, PEG_PUSH_Z, yaw_f, -1.0, lead=0.012, tol=0.003, settle=20, max_steps=300)
    _hold(robot, 1.0, 10)
    _goto(robot, ex, ey, PEG_HOVER_Z, yaw_f, 1.0)
    return {"done": "peg_push", "fingers": np.asarray(robot.now().fingers, dtype=float).tolist()}


def act_observe(robot, steps=5):
    """Hold the current pose (gripper open) for a few control steps."""
    _hold(robot, 1.0, int(max(1, min(steps, 50))))
    return {"observed": True}


def perceive_decide(obs, goal, memory, last):
    """Classify the demonstrated manner once (task video), perceive cube/target/stick in the current
    exec frame, and run that manner. If the episode continues afterwards, wait once, then retry the
    same manner (at most 3 attempts in total) from fresh perception; the first target estimate is kept."""
    manner = memory.get("manner")
    evidence = memory.get("evidence")
    attempts = memory.get("attempts") or 0
    waited = bool(memory.get("waited"))
    tgt_mem = memory.get("target_xy")
    if manner is None:
        manner, evidence = _classify_demo(list(obs.frames))
        if manner is None:
            manner = "pick"
            evidence = "default (no demo frames)"
    if getattr(obs.current, "phase", "exec") == "demo":
        mem = {"manner": manner, "evidence": str(evidence)[:200], "attempts": attempts, "last_plan": "",
               "target_xy": tgt_mem, "waited": waited}
        return {"memory": mem, "act": {"tool": "act_observe", "args": {"steps": 1}}}
    scene = _scene(obs.current)
    if scene is not None:
        if tgt_mem is None and scene.get("target") is not None:
            tgt_mem = scene["target"]["xy"]
        if tgt_mem is not None:
            scene["target"] = {"xy": tgt_mem, "n": -1}
    plan = _plan(manner, scene) if scene is not None else None
    cube = scene.get("cube") if scene else None
    observe = plan is None or attempts >= 3
    if not observe and attempts > 0 and not waited:
        near = cube is not None and float(np.linalg.norm(np.array(cube["xy"]) - np.array(tgt_mem))) < 0.02
        if near:
            observe = True
            waited = True
    mem = {"manner": manner, "evidence": str(evidence)[:200], "attempts": attempts + (0 if observe else 1),
           "last_plan": str(plan)[:400], "target_xy": tgt_mem, "waited": waited}
    if observe:
        return {"memory": mem, "act": {"tool": "act_observe", "args": {"steps": 20}}}
    mem["waited"] = False
    if plan["manner"] == "pick":
        act = {"tool": "act_pick_place", "args": {"grasp": plan["grasp"], "yaw": plan["yaw"], "place": plan["place"]}}
    elif plan["manner"] == "push":
        act = {"tool": "act_push", "args": {"start": plan["start"], "end": plan["end"], "yaw": plan["yaw"]}}
    else:
        act = {"tool": "act_peg_push", "args": {"grasp": plan["grasp"], "yaw_g": plan["yaw_g"], "yaw_f": plan["yaw_f"],
                                                "start": plan["start"], "end": plan["end"]}}
    return {"memory": mem, "act": act}
