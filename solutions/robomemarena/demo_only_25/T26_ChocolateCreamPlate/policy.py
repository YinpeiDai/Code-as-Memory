"""T26 ChocolateCreamPlate: pick chocolate then cream from plate1 and place each on plate2.

Perception: agentview RGB-D back-projected to world (coarse object / plate locations),
wrist RGB-D at a hover pose above the object (fine location). Control: closed-loop
OSC delta servoing on the measured TCP. Constants were derived from the 25 assigned
demonstrations (see task_analysis.md).
"""

HOVER_Z = 0.68          # hover / carry height (expert used ~0.69-0.70)
GRASP_YAW = -45.0       # gripper yaw at grasp, degrees (expert: -45.0 in 25/25 demos)
PLACE_YAW = 0.0         # gripper yaw at release (expert: 0.0 in 25/25 demos)
RELEASE_Z = 0.524       # TCP height when opening over plate2 (expert 0.514-0.535)
TABLE_Z_DEFAULT = 0.431
# grasp point = estimate + offset (offsets = mean(expert grasp - estimate) over demos)
AGENT_OFFSET = {"chocolate": [-0.0158, -0.0036], "cream": [-0.002, 0.000]}
WRIST_OFFSET = {"chocolate": [-0.0014, -0.0031], "cream": [0.0042, 0.0001]}
PLATE_OFFSET = [0.0, -0.003]
GRASP_BELOW_TOP = 0.021
GRASP_ABOVE_TABLE = 0.009
ON_PLATE_RADIUS = 0.065
HOME = [-0.05, 0.0, 0.67]


# ---------------------------------------------------------------- geometry
def _rotm(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the diagonal
        d = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        k = np.sqrt(d)
        i = int(np.argmax(k))
        if k[i] < 1e-9:
            return np.zeros(3)
        for j in range(3):
            if j != i:
                k[j] = k[j] * (1.0 if (R[i, j] + R[j, i]) >= 0 else -1.0)
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _target_R(yaw_deg):
    a = math.radians(yaw_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


def _yaw_of(tcp):
    R = _rotm(tcp[3:6])
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    pw = (pc - t) @ R
    return pw, z > 0


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


# ---------------------------------------------------------------- perception
def _robust_xy(P, radius=0.05):
    if len(P) == 0:
        return None
    c = np.median(P[:, :2], 0)
    for _ in range(2):
        d = np.linalg.norm(P[:, :2] - c, axis=1)
        sel = P[d < radius]
        if len(sel) < 5:
            break
        c = np.median(sel[:, :2], 0)
    return c


def _agent_detect(frame):
    """Locate chocolate, cream and the two plates from the fixed agentview RGB-D."""
    pw, ok = _backproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    im = np.asarray(frame.agentview_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    z = pw[..., 2]
    tab = ok & (pw[..., 0] > -0.3) & (pw[..., 0] < 0.5) & (np.abs(pw[..., 1]) < 0.45) & (z > 0.40) & (z < 0.56)
    s = r + g + b
    blue = tab & (b > r + 12) & (b > g + 5) & (z > 0.433)
    dark = tab & (s < 330) & (~blue) & (z > 0.433)
    white = tab & (r > 185) & (g > 180) & (b > 175) & (z < 0.45)
    flat = tab & (z < 0.445) & (~white) & (s > 330) & (s < 560)
    out = {"n_choc": int(dark.sum()), "n_cream": int(blue.sum()), "n_white": int(white.sum())}
    tz = z[flat]
    out["table_z"] = float(np.median(tz)) if len(tz) > 100 else TABLE_Z_DEFAULT
    out["chocolate"] = None
    out["cream"] = None
    if dark.sum() >= 15:
        P = pw[dark]
        c = _robust_xy(P)
        near = P[np.linalg.norm(P[:, :2] - c, axis=1) < 0.05]
        out["chocolate"] = [float(c[0]), float(c[1])]
        out["choc_top"] = float(np.percentile(near[:, 2], 90)) if len(near) else None
    if blue.sum() >= 15:
        P = pw[blue]
        c = _robust_xy(P)
        near = P[np.linalg.norm(P[:, :2] - c, axis=1) < 0.05]
        out["cream"] = [float(c[0]), float(c[1])]
        out["cream_top"] = float(np.percentile(near[:, 2], 90)) if len(near) else None
    # two plates: 2-means on white plate-centre pixels
    out["plates"] = []
    Wp = pw[white][:, :2]
    if len(Wp) >= 20:
        c = np.array([Wp[int(Wp[:, 1].argmin())], Wp[int(Wp[:, 1].argmax())]])
        lab = np.zeros(len(Wp), dtype=int)
        for _ in range(10):
            lab = np.argmin(((Wp[:, None, :] - c[None]) ** 2).sum(-1), 1)
            newc = []
            for k in range(2):
                newc.append(Wp[lab == k].mean(0) if (lab == k).sum() > 0 else c[k])
            c = np.array(newc)
        for k in range(2):
            n = int((lab == k).sum())
            if n >= 8:
                out["plates"].append([float(c[k][0]), float(c[k][1]), n])
    return out


def _wrist_detect(frame, obj, near_xy, radius=0.07):
    """Fine top-down object location from the wrist RGB-D (used at the hover pose)."""
    pw, ok = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    im = np.asarray(frame.wrist_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    z = pw[..., 2]
    base = ok & (z < 0.56)
    if near_xy is not None:
        d = np.sqrt((pw[..., 0] - near_xy[0]) ** 2 + (pw[..., 1] - near_xy[1]) ** 2)
        base = base & (d < radius)
    blue = base & (z > 0.445) & (b > r + 12) & (b > g + 5)
    if obj == "cream":
        m = blue
    else:
        m = base & (z > 0.45) & (~blue) & (((r + g + b) < 330) | ((r > g + 25) & (r > b + 25) & ((r + g + b) < 450)))
    n = int(m.sum())
    if n < 40:
        return None, None, n
    P = pw[m]
    c = np.median(P[:, :2], 0)
    top = float(np.percentile(P[:, 2], 90))
    return [float(c[0]), float(c[1])], top, n


# ---------------------------------------------------------------- control
def _servo(robot, target, yaw, grip, tol=0.004, max_steps=150, vmax=1.0, gain=10.0,
           settle=2, stall_steps=0, rot_tol_deg=3.0):
    """Closed-loop move of the measured TCP to target xyz with gripper yaw; returns a summary."""
    target = np.asarray(target, dtype=float)
    Rt = _target_R(yaw)
    good = 0
    still = 0
    prev = None
    steps = 0
    stalled = False
    f = robot.now()
    for _ in range(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        p = tcp[:3]
        e = target - p
        er = _rotvec(Rt @ _rotm(tcp[3:6]).T)
        if np.linalg.norm(e) < tol and math.degrees(np.linalg.norm(er)) < rot_tol_deg:
            good += 1
            if good >= settle:
                break
        else:
            good = 0
        if stall_steps and prev is not None:
            if np.linalg.norm(p - prev) < 0.0003:
                still += 1
            else:
                still = 0
            if still >= stall_steps:
                stalled = True
                break
        prev = p
        a = np.clip(gain * e, -vmax, vmax)
        ar = np.clip(2.0 * er, -1.0, 1.0)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
        steps += 1
    f = robot.now()
    p = np.asarray(f.tcp, dtype=float)[:3]
    return {"steps": steps, "pos": [float(v) for v in p], "err": float(np.linalg.norm(target - p)),
            "stalled": stalled}


def _hold(robot, grip, steps):
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))


def _gripper(robot, grip, max_steps=40, min_steps=8):
    """Command the gripper until the finger width stops changing."""
    prev = _width(robot.now())
    still = 0
    n = 0
    for i in range(max_steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
        n += 1
        w = _width(robot.now())
        if abs(w - prev) < 0.0004:
            still += 1
        else:
            still = 0
        prev = w
        if i + 1 >= min_steps and still >= 4:
            break
    return n


def _descend(robot, xy, z, yaw, grip, max_steps=160):
    """Fast descent to 3 cm above z, then slow approach with stall detection."""
    r1 = _servo(robot, [xy[0], xy[1], z + 0.03], yaw, grip, tol=0.006, max_steps=max_steps, vmax=1.0, settle=1)
    r2 = _servo(robot, [xy[0], xy[1], z], yaw, grip, tol=0.003, max_steps=80, vmax=0.35, settle=2, stall_steps=6)
    return {"steps": r1["steps"] + r2["steps"], "pos": r2["pos"], "stalled": r2["stalled"]}


# ---------------------------------------------------------------- acts
def act_pick(robot, obj, x, y, top=None, table_z=0.431):
    """Pick `obj` ('chocolate' or 'cream') near world (x, y): hover, refine with the wrist
    camera, descend at yaw -45 deg, close, lift. Returns holding flag, finger width and points."""
    total = 0
    p0 = robot.now().tcp
    r = _servo(robot, [float(p0[0]), float(p0[1]), max(HOVER_Z, float(p0[2]))],
               GRASP_YAW, -1.0, tol=0.01, max_steps=60, settle=1, rot_tol_deg=180.0)
    total += r["steps"]
    goal = [float(x), float(y)]
    refined = None
    wtop = None
    for it in range(2):
        r = _servo(robot, [goal[0], goal[1], HOVER_Z], GRASP_YAW, -1.0, tol=0.003, max_steps=150, settle=3)
        total += r["steps"]
        _hold(robot, -1.0, 3)
        total += 3
        xy, wt, n = _wrist_detect(robot.now(), obj, goal)
        if xy is None:
            break
        refined = [xy[0] + WRIST_OFFSET[obj][0], xy[1] + WRIST_OFFSET[obj][1]]
        wtop = wt
        moved = math.hypot(refined[0] - goal[0], refined[1] - goal[1])
        goal = refined
        if moved < 0.004:
            break
    top_used = wtop if wtop is not None else top
    z_floor = float(table_z if table_z is not None else TABLE_Z_DEFAULT) + GRASP_ABOVE_TABLE
    zg = z_floor if top_used is None else max(float(top_used) - GRASP_BELOW_TOP, z_floor)
    d = _descend(robot, goal, zg, GRASP_YAW, -1.0)
    total += d["steps"]
    total += _gripper(robot, 1.0, max_steps=45, min_steps=28)
    _hold(robot, 1.0, 5)
    total += 5
    w_closed = _width(robot.now())
    lift = _servo(robot, [goal[0], goal[1], HOVER_Z], GRASP_YAW, 1.0, tol=0.01, max_steps=120, settle=1)
    total += lift["steps"]
    _hold(robot, 1.0, 3)
    total += 3
    w = _width(robot.now())
    holding = bool(w > 0.008 and w < 0.077)
    return {"obj": obj, "holding": holding, "width": w, "width_at_close": w_closed,
            "grasp_xy": goal, "wrist_refined": refined is not None, "grasp_z": zg,
            "reached_z": d["pos"][2], "steps": total}


def act_place(robot, x, y, release_z=0.524):
    """Carry the held object at hover height to above world (x, y) at yaw 0, descend to
    release_z (stops early on contact), open, and lift back to hover height."""
    total = 0
    p = robot.now().tcp
    r = _servo(robot, [p[0], p[1], HOVER_Z], GRASP_YAW, 1.0, tol=0.01, max_steps=60, settle=1)
    total += r["steps"]
    r = _servo(robot, [float(x), float(y), HOVER_Z], PLACE_YAW, 1.0, tol=0.004, max_steps=200, settle=3)
    total += r["steps"]
    w_before = _width(robot.now())
    d = _descend(robot, [float(x), float(y)], float(release_z), PLACE_YAW, 1.0)
    total += d["steps"]
    total += _gripper(robot, -1.0, max_steps=30, min_steps=10)
    _hold(robot, -1.0, 5)
    total += 5
    r = _servo(robot, [float(x), float(y), HOVER_Z], PLACE_YAW, -1.0, tol=0.01, max_steps=120, settle=1)
    total += r["steps"]
    return {"released_at": d["pos"], "width_before_release": w_before, "stalled": d["stalled"],
            "steps": total}


def act_retreat(robot):
    """Open the gripper, rise and move to the home pose so the agentview sees the table."""
    total = 0
    p = robot.now().tcp
    r = _servo(robot, [p[0], p[1], max(HOVER_Z, float(p[2]))], _yaw_of(p), -1.0, tol=0.01, max_steps=60, settle=1,
               rot_tol_deg=180.0)
    total += r["steps"]
    r = _servo(robot, HOME, PLACE_YAW, -1.0, tol=0.01, max_steps=200, settle=2)
    total += r["steps"]
    _hold(robot, -1.0, 5)
    total += 5
    return {"pos": r["pos"], "steps": total}


def act_wait(robot, steps=40):
    """Hold still with the gripper open for `steps` native steps (lets the checker settle)."""
    _hold(robot, -1.0, int(steps))
    return {"waited": int(steps)}


# ---------------------------------------------------------------- decision
def _num_after(text, key):
    """Read the number following '"key": ' in a JSON-like act result string; None if absent."""
    tag = '"' + key + '": '
    i = text.find(tag)
    if i < 0:
        return None
    j = i + len(tag)
    k = j
    while k < len(text) and (text[k].isdigit() or text[k] in "-+.eE"):
        k += 1
    try:
        return float(text[j:k])
    except ValueError:
        return None


def _parse_last(last):
    if not isinstance(last, str):
        return None, ""
    tool = last.split(" -> ")[0].strip()
    return tool, last


def perceive_decide(obs, goal, memory, last):
    """Stage machine: detect -> pick chocolate -> place -> pick cream -> place -> verify -> wait."""
    names = ["stage", "choc_xy", "cream_xy", "plate2_xy", "table_z", "choc_top", "cream_top",
             "attempts", "placed", "verify_count", "note"]
    m = {k: memory.get(k) for k in names}
    if m["stage"] is None:
        m["stage"] = "detect"
    if m["attempts"] is None:
        m["attempts"] = 0
    if m["placed"] is None:
        m["placed"] = []
    if m["verify_count"] is None:
        m["verify_count"] = 0
    if m["table_z"] is None:
        m["table_z"] = TABLE_Z_DEFAULT
    f = obs.current
    tool, text = _parse_last(last)
    order = ["chocolate", "cream"]

    def pick_args(obj):
        xy = m["choc_xy"] if obj == "chocolate" else m["cream_xy"]
        top = m["choc_top"] if obj == "chocolate" else m["cream_top"]
        return {"obj": obj, "x": float(xy[0]), "y": float(xy[1]),
                "top": (None if top is None else float(top)), "table_z": float(m["table_z"])}

    def next_obj():
        for o in order:
            if o not in m["placed"]:
                return o
        return None

    def refresh(det, objs):
        # update coarse positions from a clean agentview detection
        for o in objs:
            if o == "chocolate" and det["chocolate"] is not None and det["n_choc"] >= 25:
                m["choc_xy"] = [det["chocolate"][0] + AGENT_OFFSET[o][0], det["chocolate"][1] + AGENT_OFFSET[o][1]]
                m["choc_top"] = det.get("choc_top")
            if o == "cream" and det["cream"] is not None and det["n_cream"] >= 25:
                m["cream_xy"] = [det["cream"][0] + AGENT_OFFSET[o][0], det["cream"][1] + AGENT_OFFSET[o][1]]
                m["cream_top"] = det.get("cream_top")

    act = {"tool": "act_wait", "args": {"steps": 20}}
    stage = m["stage"]

    if stage == "detect":
        det = _agent_detect(f)
        m["table_z"] = float(det["table_z"])
        refresh(det, order)
        if m["plate2_xy"] is None and m["choc_xy"] is not None and len(det["plates"]) >= 1:
            objc = np.array(m["choc_xy"] if m["cream_xy"] is None else
                            [(m["choc_xy"][0] + m["cream_xy"][0]) / 2, (m["choc_xy"][1] + m["cream_xy"][1]) / 2])
            best = None
            for pl in det["plates"]:
                dd = math.hypot(pl[0] - objc[0], pl[1] - objc[1])
                if dd > 0.12 and (best is None or dd > best[0]):
                    best = (dd, pl)
            if best is not None:
                m["plate2_xy"] = [best[1][0] + PLATE_OFFSET[0], best[1][1] + PLATE_OFFSET[1]]
        if m["choc_xy"] is None or m["cream_xy"] is None or m["plate2_xy"] is None:
            m["attempts"] = int(m["attempts"]) + 1
            m["note"] = "detection incomplete: " + str({k: det[k] for k in ("n_choc", "n_cream", "n_white")})
            if m["attempts"] > 3:
                # fall back to demo-typical layout for whatever is missing
                if m["choc_xy"] is None:
                    m["choc_xy"] = [0.20, -0.23]
                if m["cream_xy"] is None:
                    m["cream_xy"] = [0.21, -0.15]
                if m["plate2_xy"] is None:
                    m["plate2_xy"] = [0.17, 0.13]
            else:
                act = {"tool": "act_retreat", "args": {}}
                return {"memory": m, "act": act}
        m["attempts"] = 0
        m["stage"] = "pick"
        act = {"tool": "act_pick", "args": pick_args("chocolate")}
        m["note"] = "picking chocolate"
        return {"memory": m, "act": act}

    if stage == "pick":
        obj = next_obj()
        holding = (tool == "act_pick") and ('"holding": true' in text)
        w = _width(f)
        holding = holding and (0.008 < w < 0.077)
        if holding:
            m["stage"] = "place"
            m["attempts"] = 0
            act = {"tool": "act_place", "args": {"x": float(m["plate2_xy"][0]), "y": float(m["plate2_xy"][1]),
                                                 "release_z": RELEASE_Z}}
            m["note"] = "placing " + str(obj)
            return {"memory": m, "act": act}
        # failed grasp: retreat, re-detect, retry
        m["attempts"] = int(m["attempts"]) + 1
        m["stage"] = "redetect"
        m["note"] = "grasp failed for " + str(obj) + " attempt " + str(m["attempts"])
        return {"memory": m, "act": {"tool": "act_retreat", "args": {}}}

    if stage == "redetect":
        obj = next_obj()
        if obj is None:
            m["stage"] = "verify"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 10}}}
        det = _agent_detect(f)
        refresh(det, [obj])
        # no give-up: an unplaced object means failure anyway, so keep retrying until decisions run out
        m["stage"] = "pick"
        m["note"] = "retry pick " + obj
        return {"memory": m, "act": {"tool": "act_pick", "args": pick_args(obj)}}

    if stage == "place":
        obj = next_obj()
        wb = _num_after(text, "width_before_release") if tool == "act_place" else None
        if wb is not None and wb < 0.008:
            # the object was dropped before reaching plate2: look again and re-pick it
            m["attempts"] = int(m["attempts"]) + 1
            m["stage"] = "redetect"
            m["note"] = str(obj) + " dropped in transit (width " + str(round(wb, 4)) + ")"
            return {"memory": m, "act": {"tool": "act_retreat", "args": {}}}
        if obj is not None and obj not in m["placed"]:
            m["placed"] = list(m["placed"]) + [obj]
        nxt = next_obj()
        if nxt is not None:
            m["stage"] = "pick"
            m["attempts"] = 0
            m["note"] = "picking " + nxt
            return {"memory": m, "act": {"tool": "act_pick", "args": pick_args(nxt)}}
        m["stage"] = "verify"
        m["note"] = "both placed; retreating to verify"
        return {"memory": m, "act": {"tool": "act_retreat", "args": {}}}

    if stage == "verify":
        # arm is at home: check that neither object is clearly away from plate2
        det = _agent_detect(f)
        m["verify_count"] = int(m["verify_count"]) + 1
        p2 = m["plate2_xy"]
        bad = []
        for o, key, nkey in (("chocolate", "chocolate", "n_choc"), ("cream", "cream", "n_cream")):
            if det[key] is not None and det[nkey] >= 25:
                dd = math.hypot(det[key][0] + AGENT_OFFSET[o][0] - p2[0], det[key][1] + AGENT_OFFSET[o][1] - p2[1])
                if dd > ON_PLATE_RADIUS + 0.03:
                    bad.append(o)
        m["note"] = "verify " + str(m["verify_count"]) + " off-plate: " + str(bad)
        if bad and int(m["verify_count"]) <= 2:
            m["placed"] = [o for o in m["placed"] if o not in bad]
            refresh(det, bad)
            m["attempts"] = 0
            m["stage"] = "pick"
            return {"memory": m, "act": {"tool": "act_pick", "args": pick_args(next_obj())}}
        m["stage"] = "done"
        return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 40}}}

    # done: hold still and let the checker settle
    m["stage"] = "done"
    return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 60}}}
