"""T06 PourSauceCookiesTwice: grasp the tomato-sauce bottle, pour it over the
cookies box twice (tilt / hold / return upright, counted in memory), then put
the bottle into the bowl drainer as a fallback if the scorer has not already
ended the episode.  Perception: agentview RGB-D back-projected to world points.
"""

TABLE_Z = 0.431            # measured table-top height (agentview depth, all 25 demos)
GRASP_OFFSET = (-0.008, -0.012)   # expert grasp xy minus visible bottle centroid (std 1 mm)
POUR_OFFSET = (-0.027, 0.015)     # expert pour-start tcp xy minus cookies centroid (std 4 mm)
POUR_Z = 0.736             # expert pour-start tcp height (std 1 mm)
HOVER_Z = 0.62
LIFT_Z = 0.80
POUR_REL = 0.873           # rad (50 deg) about world -y from straight-down (expert 45 deg)
POUR_HOLD = 60             # steps held tilted (expert ~35-40 above 40 deg)
DEFAULT_BOTTLE = (-0.04, -0.08, 0.506)
DEFAULT_COOKIES = (0.224, -0.328)
DEFAULT_DRAINER = (-0.157, 0.529)
DRAIN_DROP_Z = 0.66


def _aa2R(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    s = math.sin(th)
    if abs(s) < 1e-6:
        # angle near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
    return w * th


def _world_points(depth, K, E):
    z = depth.astype(float) / 1000.0
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    Pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    return ((Pc - t) @ R).reshape(H, W, 3)


def _clusters(P, cols, cell=0.01):
    """Connected components of points on an xy grid (8-neighbour)."""
    if len(P) == 0:
        return []
    gi = np.floor(P[:, 0] / cell).astype(int)
    gj = np.floor(P[:, 1] / cell).astype(int)
    cells = {}
    for idx in range(len(P)):
        key = (int(gi[idx]), int(gj[idx]))
        if key in cells:
            cells[key].append(idx)
        else:
            cells[key] = [idx]
    seen = set()
    out = []
    for key in cells:
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        members = []
        while stack:
            c = stack.pop()
            members.extend(cells[c])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        Q = P[members]
        C = cols[members]
        out.append({
            "n": len(members),
            "xy": [float(Q[:, 0].mean()), float(Q[:, 1].mean())],
            "lo": [float(Q[:, 0].min()), float(Q[:, 1].min())],
            "hi": [float(Q[:, 0].max()), float(Q[:, 1].max())],
            "zmax": float(Q[:, 2].max()),
            "rgb": [float(C[:, 0].mean()), float(C[:, 1].mean()), float(C[:, 2].mean())],
        })
    return out


def _detect(frame, near=None):
    """Locate bottle, cookies and drainer in world coordinates from agentview RGB-D.
    `near`: previous bottle xy; the closest bottle candidate is preferred."""
    depth = frame.agentview_depth
    P = _world_points(depth, frame.agentview_K, frame.agentview_extrinsic)
    rgb = frame.agentview_rgb
    m = ((P[..., 2] > TABLE_Z + 0.006) & (P[..., 2] < 0.63) & (np.abs(P[..., 1]) < 0.8)
         & (P[..., 0] > -0.6) & (P[..., 0] < 0.6) & (depth > 0))
    cl = _clusters(P[m], rgb[m].astype(float))
    bottle = None
    cookies = None
    drainer = None
    for c in cl:
        ext = max(c["hi"][0] - c["lo"][0], c["hi"][1] - c["lo"][1])
        r, g, b = c["rgb"]
        if c["n"] >= 300 and c["zmax"] > 0.55:
            if drainer is None or c["n"] > drainer["n"]:
                drainer = c
        elif (c["n"] >= 15 and 0.47 < c["zmax"] < 0.56 and ext < 0.10
              and -0.3 < c["xy"][0] < 0.3 and -0.4 < c["xy"][1] < 0.3):
            if near is not None:
                better = bottle is None or (math.hypot(c["xy"][0] - near[0], c["xy"][1] - near[1])
                                            < math.hypot(bottle["xy"][0] - near[0], bottle["xy"][1] - near[1]))
            else:
                better = bottle is None or sum(c["rgb"]) < sum(bottle["rgb"])
            if better:
                bottle = c
        elif c["n"] >= 15 and c["zmax"] <= 0.47 and ext < 0.10 and r > g + 10:
            if cookies is None or c["n"] > cookies["n"]:
                cookies = c
    res = {"n_clusters": len(cl)}
    if bottle is not None:
        res["bottle"] = [bottle["xy"][0], bottle["xy"][1], bottle["zmax"]]
    if cookies is not None:
        res["cookies"] = [cookies["xy"][0], cookies["xy"][1]]
    if drainer is not None:
        res["drainer"] = [(drainer["lo"][0] + drainer["hi"][0]) / 2.0,
                          (drainer["lo"][1] + drainer["hi"][1]) / 2.0]
    return res


def _upright_aa(aa):
    """Gripper pointing straight down, keeping the initial yaw (expert approach pose)."""
    R = _aa2R(aa)
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Rd = Rz @ np.diag([1.0, -1.0, -1.0])
    return [float(v) for v in _R2aa(Rd)]


def _holding(frame):
    f = frame.fingers
    gap = float(f[0] - f[1])
    return 0.03 < gap < 0.076, gap


def _num(v, d):
    return d if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Phase machine for: grasp sauce bottle, pour over cookies twice, place in drainer."""
    cur = obs.current
    phase = memory.get("phase")
    bottle = memory.get("bottle")
    cookies = memory.get("cookies")
    drainer = memory.get("drainer")
    up_aa = memory.get("up_aa")
    pours = int(_num(memory.get("pours_done"), 0))
    tries = int(_num(memory.get("grasp_tries"), 0))
    notes = _num(memory.get("notes"), "")
    tcp = [float(v) for v in cur.tcp]

    def out(ph, tool, args, note=None):
        n = notes if note is None else (notes + " | " + note)[-1500:]
        return {"memory": {"phase": ph, "bottle": bottle, "cookies": cookies,
                           "drainer": drainer, "up_aa": up_aa, "pours_done": pours,
                           "grasp_tries": tries, "notes": n},
                "act": {"tool": tool, "args": args}}

    if phase is None:
        det = _detect(cur)
        bottle = det.get("bottle") or list(DEFAULT_BOTTLE)
        cookies = det.get("cookies") or list(DEFAULT_COOKIES)
        drainer = det.get("drainer") or list(DEFAULT_DRAINER)
        up_aa = _upright_aa(tcp[3:6])
        gx, gy = bottle[0] + GRASP_OFFSET[0], bottle[1] + GRASP_OFFSET[1]
        return out("hover", "act_move", {"x": gx, "y": gy, "z": HOVER_Z, "gripper": -1.0,
                                         "up_aa": up_aa, "tol": 0.004},
                   "det " + str({k: (np.round(v, 3).tolist() if isinstance(v, list) else v)
                                 for k, v in det.items()}))
    gx, gy = bottle[0] + GRASP_OFFSET[0], bottle[1] + GRASP_OFFSET[1]
    gz = min(0.49, max(0.455, bottle[2] - 0.035))
    px, py = cookies[0] + POUR_OFFSET[0], cookies[1] + POUR_OFFSET[1]
    if phase == "hover":
        return out("descend", "act_move", {"x": gx, "y": gy, "z": gz, "gripper": -1.0,
                                           "up_aa": up_aa, "tol": 0.003, "max_cmd": 0.5})
    if phase == "descend":
        return out("grasp", "act_gripper", {"close": True, "steps": 30})
    if phase == "grasp":
        return out("lift", "act_move", {"x": gx, "y": gy, "z": LIFT_Z, "gripper": 1.0,
                                        "up_aa": up_aa, "tol": 0.01})
    if phase == "lift":
        ok, gap = _holding(cur)
        if not ok and tries < 3:
            tries += 1
            det = _detect(cur, bottle)
            if det.get("bottle") is not None:
                bottle = det["bottle"]
            gx, gy = bottle[0] + GRASP_OFFSET[0], bottle[1] + GRASP_OFFSET[1]
            return out("hover", "act_move", {"x": gx, "y": gy, "z": HOVER_Z, "gripper": -1.0,
                                             "up_aa": up_aa, "tol": 0.004},
                       "grasp miss gap=%.3f retry %d" % (gap, tries))
        return out("to_pour", "act_move", {"x": px, "y": py, "z": POUR_Z, "gripper": 1.0,
                                           "up_aa": up_aa, "tol": 0.004},
                   "holding gap=%.3f" % gap)
    if phase == "to_pour":
        return out("pour", "act_rotate", {"rel": POUR_REL, "up_aa": up_aa,
                                          "hold": POUR_HOLD, "tol_deg": 2.0})
    if phase == "pour":
        pours += 1
        return out("unpour", "act_rotate", {"rel": 0.0, "up_aa": up_aa, "hold": 5,
                                            "tol_deg": 1.5},
                   "pour %d done gap=%.3f" % (pours, _holding(cur)[1]))
    if phase == "unpour":
        if pours < 2:
            return out("to_pour", "act_move", {"x": px, "y": py, "z": POUR_Z, "gripper": 1.0,
                                               "up_aa": up_aa, "tol": 0.004})
        return out("drain_up", "act_move", {"x": tcp[0], "y": tcp[1], "z": LIFT_Z,
                                            "gripper": 1.0, "up_aa": up_aa, "tol": 0.01})
    dx, dy = drainer[0], drainer[1] - 0.055  # centre of the -y compartment (divider at bbox centre)
    if phase == "drain_up":
        return out("drain_over", "act_move", {"x": dx, "y": dy, "z": LIFT_Z, "gripper": 1.0,
                                              "up_aa": up_aa, "tol": 0.005})
    if phase == "drain_over":
        return out("drain_down", "act_move", {"x": dx, "y": dy, "z": DRAIN_DROP_Z,
                                              "gripper": 1.0, "up_aa": up_aa, "tol": 0.004,
                                              "max_cmd": 0.5})
    if phase == "drain_down":
        return out("released", "act_gripper", {"close": False, "steps": 25})
    if phase == "released":
        return out("retreat", "act_move", {"x": dx, "y": dy, "z": LIFT_Z, "gripper": -1.0,
                                           "up_aa": up_aa, "tol": 0.01})
    return out("done", "act_wait", {"steps": 100})


def act_move(robot, x, y, z, gripper=-1.0, up_aa=None, tol=0.004, max_cmd=1.0,
             max_steps=400, gain=10.0):
    """Move the tcp to world (x, y, z) with P-control; hold orientation up_aa
    (axis-angle) if given; gripper sign held. Returns final tcp and error."""
    target = np.array([x, y, z], dtype=float)
    Rt = None if up_aa is None else _aa2R(up_aa)
    good = 0
    steps = 0
    err = None
    hist = []
    for _ in range(int(max_steps)):
        fr = robot.now()
        p = np.asarray(fr.tcp[:3], dtype=float)
        e = target - p
        err = float(np.linalg.norm(e))
        if err < tol:
            good += 1
            if good >= 3:
                break
        else:
            good = 0
        hist.append(p)
        if len(hist) > 20 and err < 3.0 * tol and float(np.linalg.norm(hist[-1] - hist[-20])) < 0.002:
            break
        cmd = np.clip(e * gain, -max_cmd, max_cmd)
        rot = np.zeros(3)
        if Rt is not None:
            rot = np.clip(2.0 * _R2aa(Rt @ _aa2R(fr.tcp[3:6]).T), -0.5, 0.5)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                   float(rot[0]), float(rot[1]), float(rot[2]), float(gripper))
        steps += 1
    fr = robot.now()
    return {"steps": steps, "tcp": [round(float(v), 4) for v in fr.tcp[:3]],
            "err": err, "fingers": [round(float(v), 4) for v in fr.fingers]}


def act_gripper(robot, close=True, steps=30):
    """Hold the arm still and close (True) or open (False) the gripper for `steps`."""
    g = 1.0 if close else -1.0
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, g)
    f = robot.now().fingers
    return {"fingers": [round(float(v), 4) for v in f], "gap": round(float(f[0] - f[1]), 4)}


def act_rotate(robot, rel, up_aa, hold=0, tol_deg=2.0, max_steps=150, gripper=1.0):
    """Rotate to orientation R = Ry(-rel) @ R(up_aa) (world frame) with zero
    translation command (as the expert), then hold `hold` steps. Returns tilt."""
    Rt = _aa2R([0.0, -float(rel), 0.0]) @ _aa2R(up_aa)
    steps = 0
    ang = None
    for _ in range(int(max_steps)):
        fr = robot.now()
        e = _R2aa(Rt @ _aa2R(fr.tcp[3:6]).T)
        ang = math.degrees(float(np.linalg.norm(e)))
        if ang < tol_deg:
            break
        r = np.clip(2.0 * e, -0.8, 0.8)
        robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), gripper)
        steps += 1
    for _ in range(int(hold)):
        fr = robot.now()
        e = _R2aa(Rt @ _aa2R(fr.tcp[3:6]).T)
        r = np.clip(1.0 * e, -0.2, 0.2)
        robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), gripper)
    fr = robot.now()
    Rn = _aa2R(fr.tcp[3:6])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(Rn[2, 2])))))
    return {"rot_steps": steps, "err_deg": ang, "tilt_deg": round(tilt, 2),
            "tcp": [round(float(v), 4) for v in fr.tcp[:3]],
            "fingers": [round(float(v), 4) for v in fr.fingers]}


def act_wait(robot, steps=50, gripper=0.0):
    """Hold still for `steps` native steps (lets the scorer settle)."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, gripper)
    return {"tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}
