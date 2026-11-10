"""T20_CookiesChocolateMicrowave policy.

Open the microwave door by its handle, put the cookies box inside, then put
the chocolate pudding box at the same place (inside the microwave).

Perception (first decision only): back-project the agentview depth with the
supplied calibration; find the table height, the two flat boxes on the table
(the brighter one is cookies, the darker one chocolate) and the microwave
handle (the vertical bar sticking out of the door front, at the -y side).
Control: a fixed sequence of pose segments whose targets are offsets from the
perceived anchors (offsets measured in all 50 demonstrations), tracked with
a = clip(6 * position error, +-1) and clip(2 * rotation-vector error, +-0.5),
each for the demonstrated number of steps.
"""

# ---------------------------------------------------------------- constants
_RH = [-1.2092, 1.2092, 1.2092]   # gripper approach along +y (handle / insertion)
_TD = [3.14159265, 0.0, 0.0]      # gripper pointing down (table grasp)
_KP = 6.0
_KR = 2.0
_DEFAULT_TABLE = 0.4309
# demo means of the perceived anchors (only used if perception fails)
_DEFAULT_HANDLE = [0.0960, 0.3980, 0.5560]
_DEFAULT_COOKIES = [0.1720, -0.0940, 0.4540]
_DEFAULT_CHOC = [0.1700, -0.1940, 0.4520]

# offsets of expert targets from perceived anchors (scratch/percept_offsets*.py)
_HD_OFF = [-0.0225, 0.0304]        # handle grasp target xy = handle feature + this
_HD_Z_ABOVE_TABLE = 0.1511         # handle grasp z = table + this (0.582)
_CK_OFF = [-0.0030, 0.0000, -0.0303]   # cookies grasp = box top centroid + this
_CH_OFF = [-0.0028, -0.0010, -0.0350]  # chocolate grasp = box top centroid + this
_CK_T1 = [-0.1513, 0.1995, 0.1447]     # cookies transfer pose rel. grasp
_CK_T2 = [-0.1481, 0.1922, 0.1435]     # cookies rotate pose rel. grasp
_CH_T1 = [-0.2013, 0.1998, 0.1446]     # chocolate transfer/rotate pose rel. grasp


# ---------------------------------------------------------------- geometry
def _rotm(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _logm(Rm):
    c = (float(np.trace(Rm)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        M = (Rm + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v * th
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return v * (th / (2.0 * math.sin(th)))


def _cloud(frame):
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    w = (pc - E[:, 3]) @ E[:, :3]
    return w, d > 0


def _components(pts, cell=0.012):
    """Connected components of xy points on a grid; returns index arrays."""
    if len(pts) == 0:
        return []
    ij = np.floor(pts[:, :2] / cell).astype(int)
    cells = {}
    for n in range(len(ij)):
        key = (int(ij[n, 0]), int(ij[n, 1]))
        cells.setdefault(key, []).append(n)
    seen = set()
    comps = []
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
        comps.append(np.array(members))
    comps.sort(key=lambda m: -len(m))
    return comps


def _perceive_scene(frame):
    """Table height, handle feature, cookies / chocolate box top centroids."""
    w, valid = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    X, Y, Z = w[..., 0], w[..., 1], w[..., 2]
    notes = []
    tab = valid & (X > 0.0) & (X < 0.3) & (Y > -0.4) & (Y < 0.2) & (Z > 0.38) & (Z < 0.48)
    table = float(np.median(Z[tab])) if tab.sum() > 200 else _DEFAULT_TABLE
    if abs(table - _DEFAULT_TABLE) > 0.03:
        notes.append("table height odd %.3f, default used" % table)
        table = _DEFAULT_TABLE
    # ---- boxes
    bm = valid & (Z > table + 0.007) & (Z < table + 0.09) & (X > -0.15) & (X < 0.45) & (Y > -0.45) & (Y < 0.2)
    bp = w[bm]
    bc = rgb[bm]
    boxes = []
    for comp in _components(bp):
        if len(comp) < 25:
            continue
        q = bp[comp]
        if q[:, 2].max() > table + 0.045:
            continue  # taller than a flat box (e.g. gripper fingers)
        top = q[q[:, 2] > q[:, 2].max() - 0.006]
        boxes.append({"top": top.mean(0), "bright": float(bc[comp].mean()), "n": int(len(comp))})
    if len(boxes) == 1 and boxes[0]["n"] > 120:
        # two boxes touching: split at the largest y gap
        q = bp
        order = np.argsort(q[:, 1])
        g = np.diff(q[order, 1])
        k = int(np.argmax(g))
        cut = (q[order[k], 1] + q[order[k + 1], 1]) / 2.0
        boxes = []
        for sel in (q[:, 1] < cut, q[:, 1] >= cut):
            qq = q[sel]
            top = qq[qq[:, 2] > qq[:, 2].max() - 0.006]
            boxes.append({"top": top.mean(0), "bright": float(bc[sel].mean()), "n": int(sel.sum())})
        notes.append("boxes split by y gap")
    if len(boxes) >= 2:
        two = boxes[:2]
        two.sort(key=lambda b: -b["bright"])
        cookies = [float(v) for v in two[0]["top"]]
        choc = [float(v) for v in two[1]["top"]]
        notes.append("box brightness cookies %.0f choc %.0f" % (two[0]["bright"], two[1]["bright"]))
    else:
        cookies, choc = list(_DEFAULT_COOKIES), list(_DEFAULT_CHOC)
        notes.append("boxes not found, defaults used")
    # ---- microwave handle
    mm = valid & (Z > table + 0.012) & (X > -0.25) & (X < 0.5) & (Y > 0.2) & (Y < 0.85)
    mp = w[mm]
    handle = list(_DEFAULT_HANDLE)
    if len(mp) > 500:
        body = mp[(mp[:, 2] > table + 0.05) & (mp[:, 2] < table + 0.2)]
        ys = body[:, 1]
        bins = np.arange(0.2, 0.85, 0.002)
        hist = np.histogram(ys, bins=bins)[0]
        dense = np.flatnonzero(hist >= 40)
        if len(dense):
            door_y = float(bins[dense[0]])
            h = mp[(mp[:, 1] < door_y - 0.004) & (mp[:, 1] > door_y - 0.06)]
            if len(h) >= 20:
                handle = [float(np.median(h[:, 0])), float(np.percentile(h[:, 1], 10)),
                          float((h[:, 2].min() + h[:, 2].max()) / 2.0)]
                notes.append("door_y %.3f handle pts %d" % (door_y, len(h)))
            else:
                notes.append("handle points few, default used")
        else:
            notes.append("door face not found, default used")
    else:
        notes.append("microwave not found, default used")
    return {"table": table, "handle": handle, "cookies": cookies, "chocolate": choc, "notes": notes}


def _box_near(frame, table, old, radius=0.06):
    """Top centroid of the table-box component nearest `old` (xy), or None."""
    w, valid = _cloud(frame)
    X, Y, Z = w[..., 0], w[..., 1], w[..., 2]
    bm = valid & (Z > table + 0.007) & (Z < table + 0.09) & (X > -0.15) & (X < 0.45) & (Y > -0.45) & (Y < 0.2)
    bp = w[bm]
    best = None
    for comp in _components(bp):
        if len(comp) < 25:
            continue
        q = bp[comp]
        if q[:, 2].max() > table + 0.045:
            continue  # taller than a flat box (e.g. gripper fingers)
        top = q[q[:, 2] > q[:, 2].max() - 0.006].mean(0)
        dist = math.hypot(top[0] - old[0], top[1] - old[1])
        if dist < radius and (best is None or dist < best[0]):
            best = (dist, [float(v) for v in top])
    return None if best is None else best[1]


# ---------------------------------------------------------------- planning
def _add(a, b):
    return [float(a[i] + b[i]) for i in range(3)]


def _hold(grip, steps):
    return {"target": None, "rot": None, "grip": grip, "steps": steps}


def _go(target, rot, grip, steps):
    return {"target": [float(v) for v in target], "rot": list(rot), "grip": grip, "steps": steps}


def _plan(memory):
    """Stages: list of (name, segments). Targets from the perceived anchors."""
    table = memory["table_z"]
    h = memory["handle"]
    hd = [h[0] + _HD_OFF[0], h[1] + _HD_OFF[1], table + _HD_Z_ABOVE_TABLE]
    ck = _add(memory["cookies"], _CK_OFF)
    ch = _add(memory["chocolate"], _CH_OFF)
    return [
        ("approach_handle", [_go(_add(hd, [0, -0.15, 0]), _RH, -1, 30), _go(hd, _RH, -1, 30)]),
        ("grasp_handle", [_hold(1, 20)]),
        ("pull_door", [_go(_add(hd, [-0.5, -0.2, 0]), _RH, 1, 40)]),
        ("release_handle", [_hold(-1, 10)]),
        ("approach_cookies", [_go(_add(ck, [0, 0, 0.27]), _TD, -1, 40), _go(ck, _TD, -1, 30)]),
        ("grasp_cookies", [_hold(1, 20)]),
        ("lift_cookies", [_go(_add(ck, [0, 0, 0.15]), _TD, 1, 40)]),
        ("transfer_cookies", [_go(_add(ck, _CK_T1), _TD, 1, 40), _go(_add(ck, _CK_T2), _RH, 1, 30)]),
        ("insert_cookies", [_go(_add(hd, [-0.05, 0.03, 0.02]), _RH, 1, 75), _go(_add(hd, [0, 0.08, -0.01]), _RH, 1, 25)]),
        ("release_cookies", [_hold(-1, 20)]),
        ("retreat_1", [_go(_add(hd, [0, -0.12, -0.01]), _RH, -1, 30), _go(_add(hd, [0, -0.12, 0.14]), _RH, -1, 30)]),
        ("approach_chocolate", [_go(_add(ch, [0, 0, 0.27]), _TD, -1, 40), _go(ch, _TD, -1, 30)]),
        ("grasp_chocolate", [_hold(1, 20)]),
        ("lift_chocolate", [_go(_add(ch, [0, 0, 0.15]), _TD, 1, 40)]),
        ("transfer_chocolate", [_go(_add(ch, _CH_T1), _TD, 1, 40), _go(_add(ch, _CH_T1), _RH, 1, 40)]),
        ("insert_chocolate", [_go(_add(hd, [-0.05, 0.03, 0.02]), _RH, 1, 75)]),
        ("release_chocolate", [_hold(-1, 20)]),
        ("retreat_2", [_go(_add(hd, [0, -0.12, -0.01]), _RH, -1, 30), _go(_add(hd, [0, -0.12, 0.14]), _RH, -1, 30)]),
    ]


def _grip_width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def perceive_decide(obs, goal, memory, last):
    """Perceive anchors once, then run the next stage of the plan; retries a failed box grasp."""
    mem = dict(memory)
    cur = obs.current
    if mem.get("stage") is None:
        sc = _perceive_scene(cur)
        mem["stage"] = 0
        mem["table_z"] = sc["table"]
        mem["handle"] = sc["handle"]
        mem["cookies"] = sc["cookies"]
        mem["chocolate"] = sc["chocolate"]
        mem["retries"] = 0
        mem["pending"] = ""
        mem["log"] = "; ".join(sc["notes"])
    if mem.get("pending"):
        # re-locate a box after an empty grasp (gripper lifted out of the way)
        box = mem["pending"]
        new = _box_near(cur, mem["table_z"], mem[box])
        if new is not None:
            mem[box] = new
        mem["log"] = (mem.get("log") or "")[-300:] + "; re-perceived %s %.3f %.3f" % (box, mem[box][0], mem[box][1])
        mem["pending"] = ""
    stages = _plan(mem)
    stage = int(mem["stage"])
    names = [s[0] for s in stages]
    if 0 < stage <= len(stages) and names[stage - 1] in ("grasp_cookies", "grasp_chocolate"):
        width = _grip_width(cur)
        if width < 0.012 and int(mem["retries"]) < 2:
            box = "cookies" if names[stage - 1] == "grasp_cookies" else "chocolate"
            mem["retries"] = int(mem["retries"]) + 1
            mem["stage"] = stage - 2
            mem["pending"] = box
            mem["log"] = (mem.get("log") or "")[-300:] + "; empty %s grasp (w=%.3f), retry" % (box, width)
            grasp = _add(mem[box], _CK_OFF if box == "cookies" else _CH_OFF)
            return {"memory": mem, "act": {"tool": "act_segments", "args": {
                "segments": [_hold(-1, 15), _go(_add(grasp, [0, 0, 0.27]), _TD, -1, 40)]}}}
    if stage >= len(stages):
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 100}}}
    mem["stage"] = stage + 1
    return {"memory": mem, "act": {"tool": "act_segments", "args": {"segments": stages[stage][1]}}}


# ---------------------------------------------------------------- acting
def _step_toward(robot, seg):
    fr = robot.now()
    g = 1.0 if seg["grip"] > 0 else -1.0
    if seg["target"] is None:
        robot.step(0, 0, 0, 0, 0, 0, g)
        return
    tcp = np.asarray(fr.tcp, dtype=float)
    dp = _KP * (np.asarray(seg["target"], dtype=float) - tcp[:3])
    dp = np.clip(dp, -1.0, 1.0)
    dr = _KR * _logm(_rotm(seg["rot"]) @ _rotm(tcp[3:6]).T)
    dr = np.clip(dr, -0.5, 0.5)
    robot.step(float(dp[0]), float(dp[1]), float(dp[2]), float(dr[0]), float(dr[1]), float(dr[2]), g)


def act_segments(robot, segments):
    """Track each segment {target [x,y,z] or None (hold), rot axis-angle, grip +1 close/-1 open, steps}
    with a = clip(6*pos error, 1), clip(2*rotvec error, 0.5). Returns final tcp and finger width."""
    n = 0
    for seg in segments:
        for _ in range(int(seg["steps"])):
            _step_toward(robot, seg)
            n += 1
    fr = robot.now()
    out = {"steps": n, "tcp": [round(float(v), 4) for v in np.asarray(fr.tcp)[:3]],
           "grip_width": round(_grip_width(fr), 4)}
    return out


def act_wait(robot, steps=100):
    """Hold still with the gripper open for `steps` native steps (lets the scorer settle)."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    fr = robot.now()
    return {"steps": int(steps), "tcp": [round(float(v), 4) for v in np.asarray(fr.tcp)[:3]]}
