"""T16_PourMilkMugTwice: pick the milk, pour it into the mug twice, then place it in the bowl drainer.

Perception: agentview depth + calibration -> world points -> 1 cm grid clusters above the table
(table top z ~0.43). Clusters are identified as drainer (tallest, wide), milk (nearest nominal
milk prior) and mug (nearest nominal mug prior). Control: closed-loop PD on measured tcp
(world-frame OSC delta), constants fitted offline on the 50 demonstrations.
"""

TABLE_Z = 0.43
NOMINAL_MILK = [-0.077, -0.174]   # median of visible milk points (demo mean)
NOMINAL_MUG = [0.189, -0.029]     # median of visible mug points (demo mean)
NOMINAL_DRAINER = [-0.13, 0.47]   # near compartment centre
GRASP_OFFSET = [-0.0201, -0.0010] # expert grasp tcp - milk feature (std < 1 mm over 50 demos)
GRASP_BELOW_TOP = 0.041           # expert grasp z = visible milk top - 0.041 (z = 0.523)
POUR_OFFSET = [0.032, -0.020]     # expert tilted-hold tcp - mug feature (mean over hold)
POUR_Z = 0.815
POUR_TILT_DEG = 62.0              # expert reaches 60.5 deg about world -y
POUR_HOLD = 60                    # steps held above 50 deg (expert: ~48 first pour; success 30 after 2nd)
CARRY_Z = 0.80
DRAINER_TRAVEL_Z = 0.78
DRAINER_RELEASE_Z = 0.605
POUR_COUNT_DEG = 50.0


def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _R2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        k = R[:, i] + np.eye(3)[:, i]
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _target_R(tilt_deg):
    """Gripper pointing down (x=world x, z=world -z) tilted about world y by -tilt (z swings to +x)."""
    th = -math.radians(tilt_deg)
    c, s = math.cos(th), math.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Ry @ np.diag([1.0, -1.0, -1.0])


def _tilt_deg(tcp):
    R = _aa2R(tcp[3:6])
    return math.degrees(math.acos(min(1.0, max(-1.0, -float(R[2, 2])))))


def _world_points(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    D = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * D, (v - K[1, 2]) / K[1, 1] * D, D], -1)
    W = (pc - E[:, 3]) @ E[:, :3]
    return W[D > 0]


def _clusters(P, cell=0.01):
    """Connected components of points on a 1 cm xy grid (8-neighbour)."""
    if len(P) == 0:
        return []
    ij = np.floor(P[:, :2] / cell).astype(int)
    keys = {}
    for n, (i, j) in enumerate(ij.tolist()):
        keys.setdefault((i, j), []).append(n)
    seen = set()
    out = []
    for start in keys:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            c = stack.pop()
            members.extend(keys[c])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(P[np.array(members)])
    return out


def _locate(frame, exclude_xy=None):
    """Object features from one agentview RGB-D frame. Returns dict with milk/mug/drainer (or None)."""
    W = _world_points(frame)
    m = (W[:, 2] > TABLE_Z + 0.02) & (W[:, 2] < 0.62) & (W[:, 0] > -0.4) & (W[:, 0] < 0.5) & \
        (W[:, 1] > -0.5) & (W[:, 1] < 0.8)
    P = W[m]
    cl = [c for c in _clusters(P) if len(c) >= 15]
    res = {"milk": None, "mug": None, "drainer": None, "n_clusters": len(cl)}
    # drainer: wide cluster reaching ~0.59
    best = None
    for c in cl:
        ext = c[:, :2].max(0) - c[:, :2].min(0)
        if c[:, 2].max() > 0.575 and max(ext) > 0.12:
            if best is None or len(c) > len(best):
                best = c
    if best is not None:
        rim = best[best[:, 2] > best[:, 2].max() - 0.015]
        x0, x1 = np.percentile(rim[:, 0], 2), np.percentile(rim[:, 0], 98)
        y0, y1 = np.percentile(rim[:, 1], 2), np.percentile(rim[:, 1], 98)
        res["drainer"] = [float((x0 + x1) / 2), float(y0 + 0.055), float(best[:, 2].max()),
                          float(x0), float(x1), float(y0), float(y1)]
    small = []
    for c in cl:
        if best is not None and c is best:
            continue
        ext = c[:, :2].max(0) - c[:, :2].min(0)
        if max(ext) > 0.15 or len(c) < 30:
            continue
        if float(c[:, 2].min()) > TABLE_Z + 0.05:
            continue  # floating cluster (e.g. carton held in the gripper), not resting on the table
        med = np.median(c[:, :2], 0)
        small.append((c, med))
    used = set()
    for name, prior in (("mug", NOMINAL_MUG), ("milk", NOMINAL_MILK)):
        bi, bd = None, 1e9
        for k, (c, med) in enumerate(small):
            if k in used:
                continue
            if name == "milk" and exclude_xy is not None:
                if math.hypot(med[0] - exclude_xy[0], med[1] - exclude_xy[1]) < 0.06:
                    continue
            d = math.hypot(med[0] - prior[0], med[1] - prior[1])
            if d < bd:
                bi, bd = k, d
        if bi is not None and bd < 0.15:
            used.add(bi)
            c, med = small[bi]
            res[name] = [float(med[0]), float(med[1]), float(c[:, 2].max()), float(bd)]
    return res


def _cur_tcp(robot):
    return np.asarray(robot.now().tcp, dtype=float)


def _pd_step(robot, pos, Rt, prev, grip, vmax):
    """One PD step toward (pos, Rt); `prev` is the previous tcp (for velocity). Returns (tcp_before, e, er)."""
    tcp = _cur_tcp(robot)
    v = tcp[:3] - prev[:3]
    Rc = _aa2R(tcp[3:6])
    w = _R2aa(Rc @ _aa2R(prev[3:6]).T)
    e = np.asarray(pos, dtype=float) - tcp[:3]
    er = _R2aa(Rt @ Rc.T)
    a = np.clip(25.0 * e - 60.0 * v, -vmax, vmax)
    r = np.clip(4.0 * er - 8.0 * w, -0.8, 0.8)
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return tcp


def _summary(robot, pos, Rt, n):
    tcp = _cur_tcp(robot)
    f = robot.now().fingers
    return {"steps": n, "pos_err": float(np.linalg.norm(np.asarray(pos) - tcp[:3])),
            "rot_err_deg": math.degrees(float(np.linalg.norm(_R2aa(Rt @ _aa2R(tcp[3:6]).T)))),
            "tcp": [float(x) for x in tcp[:3]], "tilt_deg": _tilt_deg(tcp),
            "fingers": float(abs(f[0]) + abs(f[1]))}


def _servo(robot, pos, tilt_deg, grip, max_steps, tol_pos, tol_deg, vmax=1.0, settle=3):
    """PD servo of tcp position + orientation (world frame) until within tolerance for `settle` checks."""
    pos = np.asarray(pos, dtype=float)
    Rt = _target_R(tilt_deg)
    prev = _cur_tcp(robot)
    ok = 0
    n = 0
    while n < max_steps:
        tcp = _cur_tcp(robot)
        e = float(np.linalg.norm(pos - tcp[:3]))
        erd = math.degrees(float(np.linalg.norm(_R2aa(Rt @ _aa2R(tcp[3:6]).T))))
        ok = ok + 1 if (e < tol_pos and erd < tol_deg) else 0
        if ok >= settle:
            break
        prev = _pd_step(robot, pos, Rt, prev, grip, vmax)
        n += 1
    out = _summary(robot, pos, Rt, n)
    out["reached"] = bool(out["pos_err"] < max(2 * tol_pos, 0.008) and out["rot_err_deg"] < max(2 * tol_deg, 4.0))
    return out


def act_move(robot, x, y, z, tilt_deg=0.0, gripper=-1.0, max_steps=250, tol=0.004, vmax=1.0):
    """Closed-loop move of the tcp to world (x, y, z) with the gripper pointing down tilted by tilt_deg
    about world y (z axis toward +x). gripper: -1 open, +1 closed (held every step)."""
    return _servo(robot, [x, y, z], tilt_deg, float(gripper), int(max_steps), float(tol), 2.0, float(vmax))


def act_grasp(robot, steps=25):
    """Hold the current pose and close the gripper for `steps` steps; returns finger opening."""
    tcp = _cur_tcp(robot)
    r = _servo(robot, tcp[:3], 0.0, 1.0, int(steps), 0.0, 0.0, 0.3, 10 ** 6)
    return r


def act_release(robot, steps=20):
    """Hold the current pose and open the gripper for `steps` steps."""
    tcp = _cur_tcp(robot)
    return _servo(robot, tcp[:3], _tilt_deg(tcp), -1.0, int(steps), 0.0, 0.0, 0.3, 10 ** 6)


def act_pour(robot, x, y, z, tilt_deg=62.0, hold=60, settle_up=15):
    """One pour at tcp (x, y, z): tilt to tilt_deg (gripper closed), keep tilted until `hold` steps were
    observed above 50 deg, then return upright and settle. Reports the steps observed above 50 deg."""
    pos = np.asarray([x, y, z], dtype=float)
    Rt = _target_R(tilt_deg)
    prev = _cur_tcp(robot)
    above = 0
    n = 0
    max_tilt = 0.0
    while above < int(hold) and n < int(hold) + 150:
        prev = _pd_step(robot, pos, Rt, prev, 1.0, 0.6)
        n += 1
        t = _tilt_deg(_cur_tcp(robot))
        max_tilt = max(max_tilt, t)
        if t > POUR_COUNT_DEG:
            above += 1
    r2 = _servo(robot, pos, 0.0, 1.0, 80, 0.006, 2.0, 0.6, 1)
    r3 = _servo(robot, pos, 0.0, 1.0, int(settle_up), 0.0, 0.0, 0.6, 10 ** 6)
    return {"held_above_50deg": above, "steps": n, "max_tilt": max_tilt, "upright_tilt": r3["tilt_deg"],
            "fingers": r3["fingers"], "tcp": r3["tcp"]}


def act_wait(robot, steps=40, gripper=-1.0):
    """Hold the current position upright for `steps` steps."""
    tcp = _cur_tcp(robot)
    return _servo(robot, tcp[:3], 0.0, float(gripper), int(steps), 0.0, 0.0, 0.3, 10 ** 6)


def _fingers(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def _count_pour(frames):
    n = 0
    for fr in frames:
        if _tilt_deg(np.asarray(fr.tcp, dtype=float)) > POUR_COUNT_DEG:
            n += 1
    return n


def perceive_decide(obs, goal, memory, last):
    """Phase machine: locate objects once (memory), grasp milk, pour twice (counted from observed tilt),
    place milk into the drainer, then wait."""
    mem = dict(memory)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    phase = mem.get("phase") or "init"
    pours = int(mem.get("pours_done") or 0)
    attempts = int(mem.get("grasp_attempts") or 0)
    retries = int(mem.get("retries") or 0)
    notes = mem.get("notes") or ""
    holding = _fingers(cur) > 0.012 and _fingers(cur) < 0.07

    if phase in ("init", "relocate"):
        loc = _locate(cur, exclude_xy=None)
        milk = loc["milk"]
        if phase == "init":
            mug = loc["mug"] or (NOMINAL_MUG + [0.562, 0.0])
            dr = loc["drainer"] or (NOMINAL_DRAINER + [0.594])
            mem["mug_xy"] = [mug[0], mug[1]]
            mem["drainer_xyz"] = [dr[0], dr[1], dr[2]]
            notes += "mug=%s dr=%s milk=%s; " % (str(loc["mug"] is not None), str(loc["drainer"] is not None),
                                                str(milk is not None))
        if milk is None:
            milk = NOMINAL_MILK + [0.564, 0.0]
            if phase == "relocate" and mem.get("milk_grasp"):
                g = mem["milk_grasp"]
                milk = [g[0] - GRASP_OFFSET[0], g[1] - GRASP_OFFSET[1], g[2] + GRASP_BELOW_TOP, 0.0]
        gz = min(0.531, max(0.455, milk[2] - GRASP_BELOW_TOP))
        mem["milk_grasp"] = [milk[0] + GRASP_OFFSET[0], milk[1] + GRASP_OFFSET[1], gz]
        g = mem["milk_grasp"]
        phase = "descend"
        act = {"tool": "act_move", "args": {"x": g[0], "y": g[1], "z": g[2] + 0.12, "gripper": -1.0,
                                            "max_steps": 200, "tol": 0.004}}
    elif phase == "descend":
        g = mem["milk_grasp"]
        phase = "grasp"
        act = {"tool": "act_move", "args": {"x": g[0], "y": g[1], "z": g[2], "gripper": -1.0,
                                            "max_steps": 150, "tol": 0.003, "vmax": 0.5}}
    elif phase == "grasp":
        phase = "check_grasp"
        act = {"tool": "act_grasp", "args": {"steps": 25}}
    elif phase == "check_grasp":
        g = mem["milk_grasp"]
        if holding:
            phase = "to_mug"
            act = {"tool": "act_move", "args": {"x": g[0], "y": g[1], "z": CARRY_Z, "gripper": 1.0,
                                                "max_steps": 200, "tol": 0.01, "vmax": 0.8}}
        else:
            attempts += 1
            notes += "grasp miss f=%.3f; " % _fingers(cur)
            phase = "relocate" if attempts < 3 else "done"
            act = {"tool": "act_move", "args": {"x": float(tcp[0]), "y": float(tcp[1]), "z": 0.70,
                                                "gripper": -1.0, "max_steps": 120, "tol": 0.01}}
    elif phase in ("to_mug", "pour", "pour_check", "to_drainer", "lower") and not holding:
        attempts += 1
        notes += "lost milk in %s; " % phase
        phase = "relocate" if attempts < 3 else "done"
        act = {"tool": "act_move", "args": {"x": float(tcp[0]), "y": float(tcp[1]), "z": max(0.70, float(tcp[2])),
                                            "gripper": -1.0, "max_steps": 120, "tol": 0.01}}
    elif phase == "to_mug":
        mx, my = mem["mug_xy"]
        phase = "pour"
        act = {"tool": "act_move", "args": {"x": mx + POUR_OFFSET[0], "y": my + POUR_OFFSET[1], "z": POUR_Z,
                                            "gripper": 1.0, "max_steps": 250, "tol": 0.004, "vmax": 0.8}}
    elif phase in ("pour", "pour_check"):
        if phase == "pour_check":
            held = _count_pour(obs.frames)
            if held >= POUR_HOLD - 5:
                pours += 1
                notes += "pour %d held %d; " % (pours, held)
            else:
                retries += 1
                notes += "short pour %d; " % held
        mx, my = mem["mug_xy"]
        if pours < 2 and retries < 4:
            phase = "pour_check"
            act = {"tool": "act_pour", "args": {"x": mx + POUR_OFFSET[0], "y": my + POUR_OFFSET[1], "z": POUR_Z,
                                                "tilt_deg": POUR_TILT_DEG, "hold": POUR_HOLD}}
        else:
            phase = "to_drainer"
            act = {"tool": "act_move", "args": {"x": float(tcp[0]), "y": float(tcp[1]), "z": DRAINER_TRAVEL_Z,
                                                "gripper": 1.0, "max_steps": 120, "tol": 0.01, "vmax": 0.8}}
    elif phase == "to_drainer":
        d = mem["drainer_xyz"]
        phase = "lower"
        act = {"tool": "act_move", "args": {"x": d[0], "y": d[1], "z": DRAINER_TRAVEL_Z, "gripper": 1.0,
                                            "max_steps": 300, "tol": 0.004, "vmax": 0.8}}
    elif phase == "lower":
        d = mem["drainer_xyz"]
        phase = "release"
        rz = max(DRAINER_RELEASE_Z, d[2] + 0.011)
        act = {"tool": "act_move", "args": {"x": d[0], "y": d[1], "z": rz, "gripper": 1.0,
                                            "max_steps": 200, "tol": 0.004, "vmax": 0.4}}
    elif phase == "release":
        phase = "retreat"
        act = {"tool": "act_release", "args": {"steps": 20}}
    elif phase == "retreat":
        phase = "done"
        act = {"tool": "act_move", "args": {"x": float(tcp[0]), "y": float(tcp[1]), "z": DRAINER_TRAVEL_Z,
                                            "gripper": -1.0, "max_steps": 150, "tol": 0.01, "vmax": 0.5}}
    else:
        phase = "done"
        act = {"tool": "act_wait", "args": {"steps": 100, "gripper": -1.0}}

    mem["phase"] = phase
    mem["pours_done"] = pours
    mem["grasp_attempts"] = attempts
    mem["retries"] = retries
    mem["notes"] = notes[-600:]
    for k in ("milk_grasp", "mug_xy", "drainer_xyz"):
        mem.setdefault(k, None)
    return {"memory": {k: mem.get(k) for k in ("phase", "pours_done", "grasp_attempts", "retries", "notes",
                                                "milk_grasp", "mug_xy", "drainer_xyz")}, "act": act}
