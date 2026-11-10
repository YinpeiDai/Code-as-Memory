"""Butter-then-popcorn into the same basket (RoboMemArena T02).

Perception: agentview depth + calibration -> world points; table plane by
median height; small objects = clusters 0.6-5 cm above the table at y < 0.08;
butter = cluster with the highest mean (R - B) (orange), popcorn = the other.
Baskets = clusters > 4 cm above the table at y > 0.05, split along world x.
The target basket is the basket nearest to the butter; it is stored in memory
and reused for the popcorn ("the same basket").
Control: closed-loop P control on measured TCP with a fixed top-down grasp
orientation (axis-angle (pi,0,0)); grasp success from finger width.
"""

# ---- constants measured from the 50 assigned demonstrations ----
NOM_BUTTER = [-0.0209, -0.0179]
NOM_POPCORN = [-0.0195, -0.2375]
NOM_BASKET = [-0.0198, 0.1985]
NOM_TABLE_Z = 0.4306
OBJ_OFFSET = [0.0027, -0.0009]      # detection minus expert grasp xy (mean)
BASKET_OFFSET = [0.0036, 0.0015]    # detection minus expert release xy (mean)
GRASP_DZ = 0.0026                   # expert grasp TCP z minus table z
RELEASE_Z = 0.655                   # expert release TCP z (basket rim ~0.565)
TRAVEL_Z = 0.82                     # expert transport height
APPROACH_Z = 0.55
MAX_TRIES = 3
NOM_AV_K = [[231.7645, 0.0, 127.5], [0.0, 309.0193, 127.5], [0.0, 0.0, 1.0]]


def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        d = np.clip(np.diag(M), 0.0, None)
        i = int(np.argmax(d))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _rot_cmd(tcp):
    """World-frame rotation command toward the top-down grasp orientation."""
    Rt = np.diag([1.0, -1.0, -1.0])
    w = _R2aa(Rt @ _aa2R(tcp[3:6]).T)
    return np.clip(4.0 * w, -0.5, 0.5)


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def _world_points(frame):
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    K = np.asarray(getattr(frame, "agentview_K", NOM_AV_K), dtype=float)
    D = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    x = (u - K[0, 2]) / K[0, 0] * D
    y = (v - K[1, 2]) / K[1, 1] * D
    C = np.stack([x, y, D], -1)
    R = E[:, :3]
    t = E[:, 3]
    W = (C - t) @ R
    W[D <= 0] = np.nan
    return W


def _groups(vals, gap):
    o = np.argsort(vals)
    s = vals[o]
    br = np.flatnonzero(np.diff(s) > gap)
    return np.split(o, br + 1)


def _detect(frame):
    """Table height, small objects and baskets from one agentview frame."""
    if not hasattr(frame, "agentview_depth") or not hasattr(frame, "agentview_extrinsic"):
        return {"table_z": NOM_TABLE_Z, "objs": [], "baskets": []}
    W = _world_points(frame)
    im = np.asarray(frame.agentview_rgb, dtype=float)
    X, Y, Z = W[..., 0], W[..., 1], W[..., 2]
    ok = np.isfinite(Z)
    tm = ok & (np.abs(X) < 0.3) & (Y > -0.4) & (Y < 0.05)
    tz = float(np.median(Z[tm])) if np.any(tm) else NOM_TABLE_Z
    if not (0.38 < tz < 0.48):
        tz = NOM_TABLE_Z
    objs = []
    m = ok & (Z > tz + 0.006) & (Z < tz + 0.05) & (X > -0.2) & (X < 0.25) & (Y > -0.4) & (Y < 0.08)
    vs, us = np.nonzero(m)
    if len(vs):
        P = W[vs, us]
        Cc = im[vs, us]
        for g in _groups(P[:, 1], 0.02):
            if len(g) < 6:
                continue
            for g2 in _groups(P[g, 0], 0.03):
                gg = g[g2]
                if len(gg) < 6:
                    continue
                objs.append({
                    "xy": [float(np.median(P[gg, 0])), float(np.median(P[gg, 1]))],
                    "n": int(len(gg)),
                    "rb": float(np.mean(Cc[gg, 0] - Cc[gg, 2])),
                })
    baskets = []
    m = ok & (Z > tz + 0.04) & (Z < tz + 0.2) & (X > -0.3) & (X < 0.45) & (Y > 0.05) & (Y < 0.45)
    vs, us = np.nonzero(m)
    if len(vs):
        P = W[vs, us]
        for g in _groups(P[:, 0], 0.015):
            if len(g) < 20:
                continue
            q = P[g]
            x0, x1 = np.percentile(q[:, 0], [2, 98])
            y0, y1 = np.percentile(q[:, 1], [2, 98])
            if (x1 - x0) < 0.08 or (y1 - y0) < 0.08:
                continue
            baskets.append({"xy": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
                            "ext": [float(x1 - x0), float(y1 - y0)], "n": int(len(g))})
    return {"table_z": tz, "objs": objs, "baskets": baskets}


def _classify(objs):
    """Return (butter_xy or None, popcorn_xy or None) in grasp coordinates."""
    big = [o for o in objs if o["n"] >= 10]
    butter = None
    popcorn = None
    big = sorted(big, key=lambda o: -o["n"])[:4]
    if len(big) >= 2:
        srt = sorted(big, key=lambda o: -o["rb"])
        orange = [srt[0]] if srt[0]["rb"] > 30.0 else []
        other = sorted([o for o in big if o is not srt[0] and o["rb"] <= 30.0], key=lambda o: -o["n"])
        if not orange:
            other = sorted([o for o in big if o["rb"] <= 30.0], key=lambda o: -o["n"])
    else:
        orange = [o for o in big if o["rb"] > 30.0]
        other = [o for o in big if o["rb"] <= 30.0]
    if orange:
        b = orange[0]["xy"]
        butter = [b[0] - OBJ_OFFSET[0], b[1] - OBJ_OFFSET[1]]
    if other:
        p = other[0]["xy"]
        popcorn = [p[0] - OBJ_OFFSET[0], p[1] - OBJ_OFFSET[1]]
    return butter, popcorn


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pick_basket(baskets, butter):
    if not baskets:
        return list(NOM_BASKET)
    ref = butter if butter is not None else NOM_BUTTER
    best = min(baskets, key=lambda b: _dist(b["xy"], ref))
    return [best["xy"][0] - BASKET_OFFSET[0], best["xy"][1] - BASKET_OFFSET[1]]


def _out(mem, tool, args):
    return {"memory": mem, "act": {"tool": tool, "args": args}}


def perceive_decide(obs, goal, memory, last):
    """State machine: perceive once, pick butter, place in nearest basket,
    pick popcorn, place in the remembered basket, verify table is clear."""
    mem = dict(memory)
    cur = obs.current
    phase = mem.get("phase") or "start"
    log = mem.get("log") or ""
    for k in ("butter_tries", "popcorn_tries"):
        if mem.get(k) is None:
            mem[k] = 0
    if mem.get("placed") is None:
        mem["placed"] = []
    width = _width(cur)
    holding = 0.008 < width < 0.07

    if phase == "start":
        det = _detect(cur)
        b, p = _classify(det["objs"])
        mem["table_z"] = det["table_z"]
        mem["butter"] = b if b is not None else list(NOM_BUTTER)
        mem["popcorn"] = p if p is not None else list(NOM_POPCORN)
        mem["basket"] = _pick_basket(det["baskets"], mem["butter"])
        log += "det objs=%d baskets=%d; " % (len(det["objs"]), len(det["baskets"]))
        phase = "pick_butter"

    zg = (mem.get("table_z") or NOM_TABLE_Z) + GRASP_DZ
    bx = mem["basket"]

    if phase == "pick_butter":
        mem["butter_tries"] += 1
        mem["phase"] = "place_butter"
        mem["log"] = log + "pick butter; "
        x, y = mem["butter"]
        return _out(mem, "act_pick", {"x": x, "y": y, "z_grasp": zg})

    if phase == "place_butter":
        if not holding:
            # missed: re-detect the butter on the table and retry
            det = _detect(cur)
            b, _p = _classify(det["objs"])
            if b is not None:
                mem["butter"] = b
            if mem["butter_tries"] < MAX_TRIES:
                mem["butter_tries"] += 1
                mem["log"] = log + "butter miss w=%.3f retry; " % width
                x, y = mem["butter"]
                return _out(mem, "act_pick", {"x": x, "y": y, "z_grasp": zg})
            phase = "pick_popcorn"
        else:
            mem["phase"] = "pick_popcorn"
            mem["placed"] = sorted(set(mem["placed"]) | {"butter"})
            mem["log"] = log + "place butter; "
            return _out(mem, "act_place", {"x": bx[0], "y": bx[1], "z_release": RELEASE_Z})

    if phase == "pick_popcorn":
        det = _detect(cur)
        b, p = _classify(det["objs"])
        if b is not None and _dist(b, mem["butter"]) < 0.06 and mem["butter_tries"] < MAX_TRIES:
            # butter still on the table near its start -> pick it again first
            mem["butter"] = b
            mem["placed"] = [o for o in mem["placed"] if o != "butter"]
            mem["butter_tries"] += 1
            mem["phase"] = "place_butter"
            mem["log"] = log + "butter still on table; "
            return _out(mem, "act_pick", {"x": b[0], "y": b[1], "z_grasp": zg})
        if p is not None and _dist(p, mem["popcorn"]) < 0.10:
            mem["popcorn"] = p
        mem["popcorn_tries"] += 1
        mem["phase"] = "place_popcorn"
        mem["log"] = log + "pick popcorn; "
        x, y = mem["popcorn"]
        return _out(mem, "act_pick", {"x": x, "y": y, "z_grasp": zg})

    if phase == "place_popcorn":
        if not holding:
            if mem["popcorn_tries"] < MAX_TRIES:
                mem["phase"] = "pick_popcorn"
                mem["log"] = log + "popcorn miss w=%.3f; " % width
                return _out(mem, "act_lift", {"z": APPROACH_Z})
            phase = "verify"
        else:
            mem["phase"] = "verify"
            mem["placed"] = sorted(set(mem["placed"]) | {"popcorn"})
            mem["log"] = log + "place popcorn; "
            return _out(mem, "act_place", {"x": bx[0], "y": bx[1], "z_release": RELEASE_Z})

    # verify: task should already be scored; re-pick anything left on the table
    det = _detect(cur)
    b, p = _classify(det["objs"])
    if b is not None and mem["butter_tries"] < MAX_TRIES:
        mem["butter"] = b
        mem["butter_tries"] += 1
        mem["phase"] = "place_butter"
        mem["log"] = log + "verify: butter on table; "
        return _out(mem, "act_pick", {"x": b[0], "y": b[1], "z_grasp": zg})
    if p is not None and mem["popcorn_tries"] < MAX_TRIES:
        mem["popcorn"] = p
        mem["popcorn_tries"] += 1
        mem["phase"] = "place_popcorn"
        mem["log"] = log + "verify: popcorn on table; "
        return _out(mem, "act_pick", {"x": p[0], "y": p[1], "z_grasp": zg})
    mem["phase"] = "verify"
    mem["log"] = log[-400:]
    return _out(mem, "act_hold", {"steps": 40})


# ---------------- control ----------------

def _goto(robot, target, grip, tol=0.004, max_steps=300, vmax=1.0, gain=10.0):
    """P-control of the TCP to a world xyz target; holds the top-down orientation."""
    tgt = np.asarray(target, dtype=float)
    hist = []
    n = 0
    near = 0
    for n in range(max_steps):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = tgt - tcp[:3]
        hist.append(tcp[:3])
        e = float(np.linalg.norm(err))
        if e < tol:
            break
        near = near + 1 if e < 2.5 * tol else 0
        if near > 30:
            break  # close enough; residual offset is not shrinking
        if len(hist) > 25 and float(np.linalg.norm(hist[-1] - hist[-21])) < 0.0015 and float(np.linalg.norm(err)) < 0.03:
            break  # stalled (contact)
        a = np.clip(gain * err, -vmax, vmax)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "err": float(np.linalg.norm(tgt - tcp[:3]))}


def _hold(robot, grip, steps):
    tgt = np.asarray(robot.now().tcp, dtype=float)[:3]
    for _ in range(steps):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a = np.clip(10.0 * (tgt - tcp[:3]), -0.3, 0.3)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)


def _travel(robot, x, y, z, grip):
    """Move to (x, y, z): rise first if low, then translate, then settle."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if tcp[2] < z - 0.03:
        _goto(robot, [tcp[0], tcp[1], z], grip, tol=0.01, max_steps=120)
    r = _goto(robot, [x, y, z], grip, tol=0.003, max_steps=220)
    return r


def act_pick(robot, x, y, z_grasp):
    """Open, travel above (x, y), descend to z_grasp, close, lift to the travel height.
    Returns finger width after closing and after lifting, and final TCP."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if _width(robot.now()) < 0.07:
        _hold(robot, -1.0, 15)
    zh = max(float(tcp[2]), APPROACH_Z)
    r1 = _travel(robot, x, y, zh, -1.0)
    _goto(robot, [x, y, APPROACH_Z], -1.0, tol=0.004, max_steps=120)
    _goto(robot, [x, y, z_grasp + 0.04], -1.0, tol=0.003, max_steps=120, vmax=0.6)
    r2 = _goto(robot, [x, y, z_grasp], -1.0, tol=0.0015, max_steps=100, vmax=0.3)
    _hold(robot, 1.0, 25)
    w_closed = _width(robot.now())
    _goto(robot, [x, y, TRAVEL_Z], 1.0, tol=0.01, max_steps=150)
    _hold(robot, 1.0, 5)
    now = robot.now()
    return {"width_closed": w_closed, "width_lifted": _width(now),
            "tcp": [float(v) for v in now.tcp[:3]], "approach_err": r1["err"], "grasp_err": r2["err"]}


def act_place(robot, x, y, z_release):
    """Carry the held object at travel height to (x, y), descend to z_release, open, rise."""
    r1 = _travel(robot, x, y, TRAVEL_Z, 1.0)
    w_before = _width(robot.now())
    r2 = _goto(robot, [x, y, z_release], 1.0, tol=0.004, max_steps=150, vmax=0.8)
    _hold(robot, -1.0, 25)
    _goto(robot, [x, y, TRAVEL_Z], -1.0, tol=0.01, max_steps=150)
    now = robot.now()
    return {"width_carry": w_before, "width_end": _width(now),
            "tcp": [float(v) for v in now.tcp[:3]], "travel_err": r1["err"], "release_err": r2["err"]}


def act_lift(robot, z):
    """Open the gripper and rise to height z at the current xy."""
    _hold(robot, -1.0, 15)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    r = _goto(robot, [tcp[0], tcp[1], max(z, float(tcp[2]))], -1.0, tol=0.01, max_steps=200)
    return {"tcp": [float(v) for v in robot.now().tcp[:3]], "err": r["err"]}


def act_hold(robot, steps=40):
    """Hold still with the gripper open for a number of steps (monitoring period)."""
    _hold(robot, -1.0, int(steps))
    return {"held": int(steps), "tcp": [float(v) for v in robot.now().tcp[:3]]}
