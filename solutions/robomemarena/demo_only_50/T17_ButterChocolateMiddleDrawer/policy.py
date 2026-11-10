"""Policy for T17: open the middle drawer, put the butter in it, then the chocolate pudding.

All numeric constants were measured from the 50 assigned expert demonstrations
(see /workspace/task_analysis.md). Perception uses the agentview RGB-D point
cloud of the current frame; control is closed-loop OSC delta servoing on the
measured TCP pose.
"""

# ---- constants measured from the demonstrations -------------------------------
HANDLE_X_FROM_XMAX = -0.1272   # handle grasp x = cabinet far-x edge (98th pct) + this
HANDLE_Y_FROM_P5 = 0.0118      # handle grasp y = 5th pct y of dark front band + this
HANDLE_Z = 0.6067              # handle grasp height (constant in all demos)
HANDLE_PRIOR = [0.1355, 0.4512, 0.6067]
PULL_DIST = 0.22               # pull distance along -y (drawer stops at ~0.18, then gripper slips off)
BUTTER_PRIOR = [0.1092, -0.2025]
CHOC_PRIOR = [-0.2285, -0.3235]
BUTTER_DET_OFFSET = [-0.0038, 0.0024]   # expert grasp xy - detected top-cluster centroid
CHOC_DET_OFFSET = [-0.0161, 0.0033]
BUTTER_GRASP_Z = 0.431   # below the 0.433 contact limit seen in demos; stall/tol stops it
CHOC_GRASP_Z = 0.434
BDROP_REL = [-0.0029, -0.0958]  # butter release xy relative to the (closed-drawer) handle grasp
BDROP_Z = 0.726
CDROP_REL = [-0.043, -0.0406]
CDROP_Z = 0.736
TABLE_Z = 0.430
OPEN_FRONT_REL = -0.10          # drawer counted open when front 5th-pct y < handle_y + this


# ---- geometry helpers ---------------------------------------------------------
def _aa_to_mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat_to_aa(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    v = np.array([x, y, z])
    if w < 0:
        w, v = -w, -v
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n * (2.0 * math.atan2(n, w))


def _rx(angle):
    return _aa_to_mat([angle, 0.0, 0.0])


def _rot_err(R_target, aa_current):
    return _mat_to_aa(R_target @ _aa_to_mat(aa_current).T)


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


# ---- perception ---------------------------------------------------------------
def _world_points(frame):
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    w = (pc - E[:, 3]) @ E[:, :3]
    w[d <= 0] = np.nan
    return w


def _cabinet(frame):
    """Handle grasp point from the dark cabinet point cloud; None when not seen."""
    w = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.max(-1) < 70
    with_np = np.nan_to_num(w, nan=-9.0)
    x, y, z = with_np[..., 0], with_np[..., 1], with_np[..., 2]
    m = dark & (z > 0.45) & (z < 0.9) & (y > 0.25) & (y < 0.9) & (x > -0.2) & (x < 0.5)
    p = with_np[m]
    if len(p) < 300:
        return None
    xmax = float(np.percentile(p[:, 0], 98))
    q = p[(p[:, 0] > xmax - 0.2) & (p[:, 0] < xmax - 0.02) & (p[:, 2] > 0.48) & (p[:, 2] < 0.7)]
    if len(q) < 100:
        return None
    y5 = float(np.percentile(q[:, 1], 5))
    return [xmax + HANDLE_X_FROM_XMAX, y5 + HANDLE_Y_FROM_P5, HANDLE_Z, len(p)]


def _drawer_front_rel(frame, handle):
    """5th-percentile y of dark points in the drawer band, relative to the closed handle y."""
    w = np.nan_to_num(_world_points(frame), nan=-9.0)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.max(-1) < 70
    m = (dark & (np.abs(w[..., 0] - handle[0]) < 0.12) & (w[..., 2] > 0.5) & (w[..., 2] < 0.68)
         & (w[..., 1] > handle[1] - 0.35) & (w[..., 1] < handle[1] + 0.1))
    p = w[m]
    if len(p) < 50:
        return None
    return float(np.percentile(p[:, 1], 5)) - handle[1]


def _object_xy(frame, prior, offset):
    """Grasp xy of a flat box near `prior`: centroid of points above the table, refined twice."""
    w = np.nan_to_num(_world_points(frame), nan=-9.0)
    c = np.array(prior[:2], dtype=float) - np.array(offset)
    found = False
    for _ in range(3):
        m = ((np.abs(w[..., 0] - c[0]) < 0.06) & (np.abs(w[..., 1] - c[1]) < 0.06)
             & (w[..., 2] > TABLE_Z + 0.006) & (w[..., 2] < 0.52))
        q = w[m]
        if len(q) < 30:
            break
        c = np.array([q[:, 0].mean(), q[:, 1].mean()])
        found = True
    if not found:
        return [float(prior[0]), float(prior[1])], False
    g = c + np.array(offset)
    if np.linalg.norm(g - np.array(prior[:2])) > 0.08:
        return [float(prior[0]), float(prior[1])], False
    return [float(g[0]), float(g[1])], True


# ---- control ------------------------------------------------------------------
def _move(robot, pos, R, grip, tol=0.004, rtol=0.04, max_steps=250, gain=10.0,
          vmax=1.0, vmin=0.12, watch=None, stall_dist=0.012):
    """Servo the TCP to pos/R. watch: min finger width to keep (returns early if lost)."""
    pos = np.asarray(pos, dtype=float)
    steps = 0
    hist = []
    for steps in range(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        if watch is not None and _width(f) < watch:
            return {"reached": False, "lost": True, "steps": steps}
        e = pos - tcp[:3]
        er = _rot_err(R, tcp[3:6])
        en = float(np.linalg.norm(e))
        ren = float(np.linalg.norm(er))
        if en < tol and ren < rtol:
            return {"reached": True, "lost": False, "steps": steps}
        hist.append(tcp[:3])
        if len(hist) > 10:
            hist = hist[-10:]
            if (float(np.linalg.norm(hist[-1] - hist[0])) < 0.0008 and en < stall_dist
                    and ren < max(rtol, 0.06)):
                return {"reached": False, "stalled": True, "lost": False, "steps": steps}
        v = e * gain
        n = float(np.linalg.norm(v))
        if n > vmax:
            v = v * (vmax / n)
        elif n < vmin and en >= tol:
            v = v * (vmin / max(n, 1e-9))
        r = np.clip(er * 2.0, -0.5, 0.5)
        robot.step(float(v[0]), float(v[1]), float(v[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return {"reached": False, "lost": False, "steps": max_steps}


def _hold(robot, pos, R, grip, min_steps=12, max_steps=35):
    """Hold the pose while the gripper moves; stop once the fingers settle."""
    pos = np.asarray(pos, dtype=float)
    prev = _width(robot.now())
    still = 0
    for i in range(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        v = np.clip((pos - tcp[:3]) * 10.0, -0.3, 0.3)
        r = np.clip(_rot_err(R, tcp[3:6]) * 2.0, -0.3, 0.3)
        robot.step(float(v[0]), float(v[1]), float(v[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        wdt = _width(robot.now())
        still = still + 1 if abs(wdt - prev) < 2e-4 else 0
        prev = wdt
        if i + 1 >= min_steps and still >= 4:
            break
    return _width(robot.now())


def _tcp(robot):
    return [float(v) for v in np.asarray(robot.now().tcp, dtype=float)[:3]]


# ---- act tools ----------------------------------------------------------------
def act_open_drawer(robot, handle, pull_to_y):
    """Grasp the middle-drawer handle from the -y side with a horizontal gripper and pull it to
    world y = pull_to_y, then release and back away to a high pose.
    handle: [x, y, z] grasp point. Returns status ok / missed and measured finger widths."""
    hx, hy, hz = float(handle[0]), float(handle[1]), float(handle[2])
    Rh = _rx(1.5 * math.pi)
    r1 = _move(robot, [hx, hy - 0.08, hz], Rh, -1.0, tol=0.008, rtol=0.05, max_steps=250)
    r2 = _move(robot, [hx, hy, hz], Rh, -1.0, tol=0.003, rtol=0.03, max_steps=150, vmax=0.5)
    w = _hold(robot, [hx, hy, hz], Rh, 1.0, min_steps=15, max_steps=40)
    if w < 0.006:
        _hold(robot, [hx, hy, hz], Rh, -1.0, min_steps=10, max_steps=25)
        _move(robot, [hx, hy - 0.10, hz], Rh, -1.0, tol=0.01, max_steps=120)
        p = _tcp(robot)
        _move(robot, [p[0], p[1], 0.78], Rh, -1.0, tol=0.01, max_steps=120)
        return {"status": "missed", "grip_width": w, "approach": [r1["reached"], r2["reached"]]}
    pull = _move(robot, [hx, float(pull_to_y), hz], Rh, 1.0, tol=0.005, rtol=0.08,
                 max_steps=220, gain=8.0, vmax=0.8, watch=0.005, stall_dist=1.0)
    end = _tcp(robot)
    _hold(robot, end, Rh, -1.0, min_steps=10, max_steps=25)
    _move(robot, [end[0], end[1] - 0.10, end[2]], Rh, -1.0, tol=0.01, max_steps=120)
    p = _tcp(robot)
    _move(robot, [p[0], p[1], 0.78], Rh, -1.0, tol=0.01, max_steps=120)
    return {"status": "ok", "grip_width": w, "pulled_to_y": end[1], "slipped": pull["lost"],
            "pull_steps": pull["steps"]}


def act_pick_place(robot, pick, grasp_z, place, place_z, transit_z=0.88):
    """Pick a flat box at xy `pick` (top-down grasp at grasp_z) and release it at xy `place`,
    height place_z, travelling at transit_z. Returns status ok / missed / dropped."""
    Rd = _rx(math.pi)
    px, py = float(pick[0]), float(pick[1])
    cur = _tcp(robot)
    if cur[2] < 0.70:
        _move(robot, [cur[0], cur[1], 0.75], _aa_to_mat(robot.now().tcp[3:6]), -1.0, tol=0.01,
              rtol=0.5, max_steps=100)
    _move(robot, [px, py, 0.75], Rd, -1.0, tol=0.01, rtol=0.05, max_steps=300)
    _move(robot, [px, py, grasp_z + 0.06], Rd, -1.0, tol=0.006, rtol=0.03, max_steps=200)
    rg = _move(robot, [px, py, grasp_z], Rd, -1.0, tol=0.003, rtol=0.03, max_steps=150, vmax=0.4)
    w = _hold(robot, [px, py, grasp_z], Rd, 1.0, min_steps=15, max_steps=40)
    if w < 0.01:
        _hold(robot, [px, py, grasp_z], Rd, -1.0, min_steps=10, max_steps=25)
        _move(robot, [px, py, 0.70], Rd, -1.0, tol=0.01, max_steps=150)
        return {"status": "missed", "grip_width": w, "reached_grasp": rg["reached"]}
    lift = _move(robot, [px, py, 0.75], Rd, 1.0, tol=0.01, rtol=0.05, max_steps=200, watch=0.008)
    if lift["lost"]:
        _move(robot, [px, py, 0.75], Rd, -1.0, tol=0.01, max_steps=100)
        return {"status": "dropped", "where": "lift", "grip_width": _width(robot.now())}
    dx, dy = float(place[0]), float(place[1])
    tr = _move(robot, [dx, dy, transit_z], Rd, 1.0, tol=0.006, rtol=0.04, max_steps=300, watch=0.008)
    if tr["lost"]:
        return {"status": "dropped", "where": "transit", "tcp": _tcp(robot)}
    dn = _move(robot, [dx, dy, float(place_z)], Rd, 1.0, tol=0.004, rtol=0.04, max_steps=200,
               watch=0.008)
    held = _width(robot.now())
    _hold(robot, [dx, dy, float(place_z)], Rd, -1.0, min_steps=12, max_steps=30)
    _move(robot, [dx, dy, float(place_z) + 0.15], Rd, -1.0, tol=0.01, max_steps=120)
    return {"status": "ok", "grip_width": w, "held_at_release": held, "lost_on_descent": dn["lost"],
            "release_tcp": [dx, dy, float(place_z)]}


def act_wait(robot, steps=40):
    """Hold still with the gripper open for `steps` native steps (lets the checker settle)."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    return {"waited": int(steps), "tcp": _tcp(robot)}


# ---- decision -----------------------------------------------------------------
def _status(last, tool):
    if not isinstance(last, str) or not last.startswith(tool):
        return None
    for s in ("ok", "missed", "dropped"):
        if '"status": "' + s + '"' in last:
            return s
    return "failed"


def _num(text, key):
    """Read the number after '"key": ' in an act result string; None if absent."""
    if not isinstance(text, str):
        return None
    tag = '"' + key + '": '
    i = text.find(tag)
    if i < 0:
        return None
    j = i + len(tag)
    k = j
    while k < len(text) and text[k] in "+-.0123456789eE":
        k += 1
    try:
        return float(text[j:k])
    except ValueError:
        return None


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive cabinet/objects once, open the middle drawer, place butter, then
    chocolate, verifying each stage from the observation and the act result."""
    mem = dict(memory)
    for k in ("stage", "handle", "butter", "chocolate", "drawer_tries", "butter_tries",
              "choc_tries", "notes"):
        mem.setdefault(k, None)
    frame = obs.current
    notes = mem["notes"] or ""
    if mem["stage"] is None:
        cab = _cabinet(frame)
        if cab is None:
            mem["handle"] = list(HANDLE_PRIOR)
            notes += "cabinet not detected, prior handle; "
        else:
            mem["handle"] = [cab[0], cab[1], cab[2]]
            notes += "cabinet pts %d; " % cab[3]
        b, okb = _object_xy(frame, BUTTER_PRIOR, BUTTER_DET_OFFSET)
        c, okc = _object_xy(frame, CHOC_PRIOR, CHOC_DET_OFFSET)
        mem["butter"], mem["chocolate"] = b, c
        notes += "butter det %s choc det %s; " % (okb, okc)
        mem["stage"] = "open_drawer"
        mem["drawer_tries"], mem["butter_tries"], mem["choc_tries"] = 0, 0, 0
    h = mem["handle"]
    hy0 = float(h[1])
    if mem["stage"] == "open_drawer":
        st = _status(last, "act_open_drawer")
        target = [h[0], hy0, h[2]]
        if st is not None:
            rel = _drawer_front_rel(frame, h)
            notes += "drawer try %d %s front_rel %s; " % (
                mem["drawer_tries"], st, None if rel is None else round(rel, 3))
            if (rel is not None and rel < OPEN_FRONT_REL) or (rel is None and st == "ok") \
                    or mem["drawer_tries"] >= 3:
                mem["stage"] = "butter"
            else:
                shift = 0.0
                if rel is not None and rel < -0.03:
                    # partially opened: the handle moved by the same amount as the front
                    shift = -(rel + 0.012)
                elif st == "ok":
                    py = _num(last, "pulled_to_y")
                    if py is not None and py < hy0 - 0.02:
                        shift = hy0 - py      # grip slipped while pulling: handle left near there
                elif st == "missed":
                    shift = [0.0, -0.008, 0.008][mem["drawer_tries"] % 3]
                target = [h[0], hy0 - shift, h[2]]
        if mem["stage"] == "open_drawer":
            mem["drawer_tries"] += 1
            return {"memory": _out(mem, notes),
                    "act": {"tool": "act_open_drawer",
                            "args": {"handle": target, "pull_to_y": hy0 - PULL_DIST}}}
    if mem["stage"] == "butter":
        st = _status(last, "act_pick_place")
        if st == "ok" and mem["butter_tries"] > 0:
            mem["stage"] = "chocolate"
        elif st is not None and mem["butter_tries"] > 0:
            b, okb = _object_xy(frame, BUTTER_PRIOR, BUTTER_DET_OFFSET)
            mem["butter"] = b
            notes += "butter %s, redetect %s; " % (st, okb)
            if mem["butter_tries"] >= 3:
                mem["stage"] = "chocolate"
        if mem["stage"] == "butter":
            mem["butter_tries"] += 1
            return {"memory": _out(mem, notes),
                    "act": {"tool": "act_pick_place",
                            "args": {"pick": mem["butter"], "grasp_z": BUTTER_GRASP_Z,
                                     "place": [h[0] + BDROP_REL[0], hy0 + BDROP_REL[1]],
                                     "place_z": BDROP_Z}}}
    if mem["stage"] == "chocolate":
        st = _status(last, "act_pick_place")
        if st == "ok" and mem["choc_tries"] > 0:
            mem["stage"] = "done"
        elif st is not None and mem["choc_tries"] > 0:
            c, okc = _object_xy(frame, CHOC_PRIOR, CHOC_DET_OFFSET)
            mem["chocolate"] = c
            notes += "choc %s, redetect %s; " % (st, okc)
            if mem["choc_tries"] >= 3:
                mem["stage"] = "done"
        if mem["stage"] == "chocolate":
            mem["choc_tries"] += 1
            return {"memory": _out(mem, notes),
                    "act": {"tool": "act_pick_place",
                            "args": {"pick": mem["chocolate"], "grasp_z": CHOC_GRASP_Z,
                                     "place": [h[0] + CDROP_REL[0], hy0 + CDROP_REL[1]],
                                     "place_z": CDROP_Z}}}
    return {"memory": _out(mem, notes), "act": {"tool": "act_wait", "args": {"steps": 40}}}


def _out(mem, notes):
    return {"stage": mem["stage"], "handle": mem["handle"], "butter": mem["butter"],
            "chocolate": mem["chocolate"], "drawer_tries": mem["drawer_tries"],
            "butter_tries": mem["butter_tries"], "choc_tries": mem["choc_tries"],
            "notes": notes[-1500:]}
