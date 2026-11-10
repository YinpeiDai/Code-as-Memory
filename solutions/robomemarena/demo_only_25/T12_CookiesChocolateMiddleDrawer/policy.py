"""Policy for T12: put cookies into the middle drawer, then chocolate pudding into the same drawer.

Pipeline (one act tool per phase; memory carries the perceived geometry and the stage):
  init        perceive cabinet / cookies / chocolate from the agentview RGB-D point cloud -> act_open_drawer
  check_open  measure the open drawer in the point cloud; retry open if needed -> act_pick (cookies)
  check_ck    finger width says cookies held? -> act_place (drawer) / retry pick
  placed_ck   -> act_pick (chocolate)
  check_ch    finger width says chocolate held? -> act_place (drawer) / retry pick
  done        -> act_wait (the scorer ends the episode once both objects are in the drawer)
"""

# ---------------------------------------------------------------- constants (from assigned demos)
TABLE_Z = 0.4307           # measured table top height (all 25 demos)
HANDLE_Z = 0.607           # expert tcp height when grasping the middle-drawer handle (all demos)
HANDLE_DX = -0.1271        # handle x = cabinet front edge x (98th pct) + HANDLE_DX   (resid 0.3 mm)
HANDLE_DY = -0.1969        # handle y = cabinet far side y (97th pct) + HANDLE_DY      (resid 0.7 mm)
PRIOR_HANDLE = [0.140, 0.457, 0.607]
PRIOR_COOKIE = [0.217, -0.328]   # mean expert grasp
PRIOR_CHOC = [-0.231, -0.323]
COOKIE_OFF = [-0.0062, 0.0008]   # expert grasp - top-face centroid (std 1 mm)
CHOC_OFF = [-0.010, 0.003]       # expert grasp - top-face centroid (expert std ~7-9 mm)
GRASP_Z = 0.435
PLACE_CK = [0.002, -0.093, 0.122]   # cookies release point relative to the handle grasp point
PLACE_CH = [-0.025, -0.085, 0.100]  # chocolate release point relative to the handle grasp point
TRANSIT_Z = 0.95
DOWN = [math.pi, 0.0, 0.0]               # gripper pointing down (axis-angle)
SIDE = [1.5 * math.pi, 0.0, 0.0]         # gripper pointing +y into the cabinet face (axis-angle)
AV_K = [[231.76450199, 0.0, 127.5], [0.0, 309.01933598, 127.5], [0.0, 0.0, 1.0]]
AV_E = [[-7.26818450e-08, 1.0, -1.52115266e-07, 3.39354405e-07],
        [6.28266450e-01, -7.26818450e-08, -7.77998244e-01, 4.47718863e-01],
        [-7.77998244e-01, -1.52115266e-07, -6.28266450e-01, 2.01697949e+00]]


# ---------------------------------------------------------------- geometry helpers
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        k = k / max(np.linalg.norm(k), 1e-12)
        # pick the sign consistent with the antisymmetric part
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(np.dot(w, k)) < 0:
            k = -k
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _sensor(frame, name, default=None):
    if name in tuple(frame.sensors):
        return getattr(frame, name)
    return default


def _cloud(frame):
    """World XYZ (256x256x3) of every agentview pixel, plus RGB; None when depth is missing."""
    d = _sensor(frame, "agentview_depth")
    if d is None:
        return None, None
    K = np.asarray(_sensor(frame, "agentview_K", AV_K), dtype=float)
    E = np.asarray(_sensor(frame, "agentview_extrinsic", AV_E), dtype=float)
    z = np.asarray(d, dtype=float) / 1000.0
    v, u = np.mgrid[0:256, 0:256]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    w = (pc - E[:, 3]) @ E[:, :3]
    w[z <= 0] = np.nan
    rgb = _sensor(frame, "agentview_rgb")
    rgb = None if rgb is None else np.asarray(rgb).astype(int)
    return w, rgb


def _find_cabinet(w, rgb):
    """Handle grasp point of the middle drawer from the dark cabinet body."""
    if w is None or rgb is None:
        return None, 0
    with np.errstate(invalid="ignore"):
        m = ((rgb.max(-1) < 45) & (w[..., 2] > 0.45) & (w[..., 2] < 0.9) & (w[..., 1] > 0.2)
             & (w[..., 1] < 0.9) & (w[..., 0] > -0.3) & (w[..., 0] < 0.5))
    n = int(m.sum())
    if n < 300:
        return None, n
    P = w[m]
    hx = float(np.percentile(P[:, 0], 98)) + HANDLE_DX
    hy = float(np.percentile(P[:, 1], 97)) + HANDLE_DY
    if abs(hx - PRIOR_HANDLE[0]) > 0.12 or abs(hy - PRIOR_HANDLE[1]) > 0.12:
        return None, n
    return [hx, hy, HANDLE_Z], n


def _find_box(w, prior, radius=0.06, zmin_off=0.008, zmax_off=0.08):
    """Top-face centroid (x, y, top z, n) of the object nearest `prior` on the table, or None."""
    if w is None:
        return None
    px, py = prior[0], prior[1]
    with np.errstate(invalid="ignore"):
        m = ((np.abs(w[..., 0] - px) < radius) & (np.abs(w[..., 1] - py) < radius)
             & (w[..., 2] > TABLE_Z + zmin_off) & (w[..., 2] < TABLE_Z + zmax_off))
    if int(m.sum()) < 10:
        return None
    P = w[m]
    # keep the cluster around the highest dense level (object top face)
    zt = float(np.percentile(P[:, 2], 95))
    top = P[P[:, 2] > zt - 0.006]
    if len(top) < 5:
        return None
    c = top.mean(0)
    # re-centre once on the found object to drop neighbours at the window edge
    d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    P2 = P[d < 0.05]
    if len(P2) >= 10:
        zt = float(np.percentile(P2[:, 2], 95))
        top = P2[P2[:, 2] > zt - 0.006]
        c = top.mean(0)
    return [float(c[0]), float(c[1]), float(zt), int(len(top))]


def _drawer_state(w, rgb, handle):
    """Dark drawer-body points in front of the cabinet face and the drawer front face y."""
    if w is None or rgb is None or handle is None:
        return 0, None
    hx, hy, hz = handle
    with np.errstate(invalid="ignore"):
        m = ((w[..., 0] > hx - 0.12) & (w[..., 0] < hx + 0.12) & (w[..., 1] > hy - 0.25)
             & (w[..., 1] < hy - 0.06) & (w[..., 2] > hz - 0.10) & (w[..., 2] < hz + 0.06)
             & (rgb.max(-1) < 60))
    n = int(m.sum())
    if n < 30:
        return n, None
    return n, float(np.percentile(w[m][:, 1], 3))


# ---------------------------------------------------------------- control helpers
def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _servo(robot, pos, rot, grip, max_steps=200, tol=0.004, rtol=0.03, vmax=1.0, k=12.0, settle=2):
    """Closed-loop move of the tcp to `pos` (world m) and axis-angle `rot`, gripper sign `grip`."""
    target = np.asarray(pos, dtype=float)
    Rt = _aa2mat(rot)
    ok = 0
    n = 0
    err = 1.0
    rerr = 1.0
    for n in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        e = target - tcp[:3]
        er = _mat2aa(Rt @ _aa2mat(tcp[3:6]).T)
        err = float(np.linalg.norm(e))
        rerr = float(np.linalg.norm(er))
        if err < tol and rerr < rtol:
            ok += 1
            if ok >= settle:
                break
        else:
            ok = 0
        a = np.clip(k * e, -vmax, vmax)
        b = np.clip(3.0 * er, -1.0, 1.0)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(b[0]), float(b[1]), float(b[2]), float(grip))
    return {"err": round(err, 4), "rerr": round(rerr, 4), "steps": n + 1}


def _hold(robot, grip, steps):
    """Hold the current pose (servo) while commanding the gripper for `steps` steps."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    pos = tcp[:3].copy()
    rot = tcp[3:6].copy()
    Rt = _aa2mat(rot)
    for _ in range(int(steps)):
        cur = np.asarray(robot.now().tcp, dtype=float)
        e = pos - cur[:3]
        er = _mat2aa(Rt @ _aa2mat(cur[3:6]).T)
        a = np.clip(12.0 * e, -1.0, 1.0)
        b = np.clip(3.0 * er, -1.0, 1.0)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(b[0]), float(b[1]), float(b[2]), float(grip))
    return _width(robot.now())


def _pose(robot):
    return [round(float(v), 4) for v in np.asarray(robot.now().tcp, dtype=float)[:3]]


# ---------------------------------------------------------------- act tools
def act_open_drawer(robot, handle):
    """Grasp the middle-drawer handle at `handle` [x,y,z] with the gripper pointing +y, pull it out
    along -y, release, back off and turn the gripper down again. Returns the finger width when closed
    on the handle and the final tcp."""
    hx, hy, hz = float(handle[0]), float(handle[1]), float(handle[2])
    _servo(robot, [hx, hy - 0.10, hz + 0.03], SIDE, -1, max_steps=160, tol=0.01, rtol=0.06)
    _servo(robot, [hx, hy - 0.10, hz], SIDE, -1, max_steps=80, tol=0.006, rtol=0.04)
    r = _servo(robot, [hx, hy, hz], SIDE, -1, max_steps=120, tol=0.004, rtol=0.04, vmax=0.5)
    w_closed = _hold(robot, 1, 25)
    _servo(robot, [hx, hy - 0.22, hz], SIDE, 1, max_steps=160, tol=0.006, rtol=0.06, vmax=0.6, k=8.0)
    _hold(robot, -1, 18)
    _servo(robot, [hx, hy - 0.36, hz], SIDE, -1, max_steps=100, tol=0.01, rtol=0.08)
    # turn down and stay clear of the drawer region the open-check looks at
    _servo(robot, [hx, hy - 0.38, hz + 0.10], DOWN, -1, max_steps=150, tol=0.01, rtol=0.06)
    return {"approach_err": r["err"], "width_on_handle": round(w_closed, 4), "tcp": _pose(robot)}


def act_pick(robot, target, grasp_z=0.435, lift_z=0.75, approach_z=0.63):
    """Pick a box lying on the table at `target` [x,y] with a top-down grasp: move above at a safe
    height, descend to `grasp_z`, close, lift to `lift_z`. Returns the finger width (m) after lifting."""
    x, y = float(target[0]), float(target[1])
    cur = np.asarray(robot.now().tcp, dtype=float)
    if cur[2] < approach_z - 0.02:
        _servo(robot, [cur[0], cur[1], approach_z], DOWN, -1, max_steps=80, tol=0.015, rtol=0.1)
    _servo(robot, [x, y, approach_z], DOWN, -1, max_steps=200, tol=0.006, rtol=0.03)
    r = _servo(robot, [x, y, grasp_z], DOWN, -1, max_steps=200, tol=0.004, rtol=0.03, vmax=0.5)
    w0 = _hold(robot, 1, 25)
    _servo(robot, [x, y, lift_z], DOWN, 1, max_steps=150, tol=0.01, rtol=0.05, vmax=0.8)
    w1 = _width(robot.now())
    return {"descend_err": r["err"], "width_closed": round(w0, 4), "width_lifted": round(w1, 4), "tcp": _pose(robot)}


def act_place(robot, place, transit_z=0.95, release=True, retreat_z=0.90):
    """Carry the held object high over the drawer to `place` [x,y,z], descend to it, open the gripper
    (if `release`) and rise to `retreat_z`. Returns the width before release and the final tcp."""
    x, y, z = float(place[0]), float(place[1]), float(place[2])
    cur = np.asarray(robot.now().tcp, dtype=float)
    _servo(robot, [cur[0], cur[1], max(cur[2], 0.75)], DOWN, 1, max_steps=60, tol=0.015, rtol=0.08)
    _servo(robot, [x, y, transit_z], DOWN, 1, max_steps=250, tol=0.01, rtol=0.05)
    r = _servo(robot, [x, y, z], DOWN, 1, max_steps=200, tol=0.005, rtol=0.04, vmax=0.5)
    w = _width(robot.now())
    if release:
        _hold(robot, -1, 20)
        _servo(robot, [x, y, retreat_z], DOWN, -1, max_steps=120, tol=0.015, rtol=0.08)
    return {"place_err": r["err"], "width_before_release": round(w, 4), "tcp": _pose(robot)}


def act_move(robot, pos, grip=-1.0, side=False):
    """Move the tcp to `pos` [x,y,z] (gripper down, or pointing +y when `side`) with gripper sign `grip`."""
    r = _servo(robot, pos, SIDE if side else DOWN, grip, max_steps=250, tol=0.008, rtol=0.05)
    return {"err": r["err"], "tcp": _pose(robot), "width": round(_width(robot.now()), 4)}


def act_wait(robot, steps=40, grip=-1.0):
    """Hold the pose for `steps` steps (lets the scorer settle). Returns the tcp."""
    _hold(robot, grip, steps)
    return {"tcp": _pose(robot)}


# ---------------------------------------------------------------- decision
def _fresh(memory):
    m = {}
    m["stage"] = memory.get("stage") if memory.get("stage") is not None else "init"
    m["handle"] = memory.get("handle") if memory.get("handle") is not None else list(PRIOR_HANDLE)
    m["cookie"] = memory.get("cookie") if memory.get("cookie") is not None else list(PRIOR_COOKIE)
    m["choc"] = memory.get("choc") if memory.get("choc") is not None else list(PRIOR_CHOC)
    m["tries"] = int(memory.get("tries") or 0)
    m["log"] = memory.get("log") or ""
    return m


def _note(m, s):
    m["log"] = (m["log"] + " | " + s)[-600:]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive at the start, then open drawer, pick+place cookies, pick+place chocolate."""
    m = _fresh(memory if isinstance(memory, dict) else {})
    frame = obs.current
    w, rgb = _cloud(frame)
    stage = m["stage"]

    if stage == "init":
        h, n = _find_cabinet(w, rgb)
        if h is None:
            h = list(PRIOR_HANDLE)
            _note(m, "cabinet not found (%d px), prior handle" % n)
        ck = _find_box(w, PRIOR_COOKIE)
        ch = _find_box(w, PRIOR_CHOC)
        m["cookie"] = ([ck[0] + COOKIE_OFF[0], ck[1] + COOKIE_OFF[1]] if ck is not None else list(PRIOR_COOKIE))
        m["choc"] = ([ch[0] + CHOC_OFF[0], ch[1] + CHOC_OFF[1]] if ch is not None else list(PRIOR_CHOC))
        if ck is None:
            _note(m, "cookies not seen, prior")
        if ch is None:
            _note(m, "chocolate not seen, prior")
        m["handle"] = [float(v) for v in h]
        m["stage"] = "check_open"
        m["tries"] = 0
        return {"memory": m, "act": {"tool": "act_open_drawer", "args": {"handle": m["handle"]}}}

    handle = [float(v) for v in m["handle"]]
    hx, hy, hz = handle

    if stage == "check_open":
        n, front = _drawer_state(w, rgb, handle)
        is_open = n >= 150 and front is not None and front < hy - 0.11
        _note(m, "drawer px %d front %s" % (n, "none" if front is None else "%.3f" % (front - hy)))
        if not is_open and m["tries"] < 2:
            m["tries"] += 1
            h2 = list(handle)
            if front is not None and n >= 60:
                h2[1] = front - 0.018   # partially open: handle sits ~1.8 cm in front of the drawer face
            return {"memory": m, "act": {"tool": "act_open_drawer", "args": {"handle": h2}}}
        m["tries"] = 0
        m["stage"] = "check_ck"
        return {"memory": m, "act": {"tool": "act_pick", "args": {"target": m["cookie"], "grasp_z": GRASP_Z}}}

    if stage == "check_ck":
        wd = _width(frame)
        if wd < 0.02 and m["tries"] < 2:
            m["tries"] += 1
            _note(m, "cookie grasp missed w=%.3f" % wd)
            m["stage"] = "regrasp_ck"
            return {"memory": m, "act": {"tool": "act_move", "args": {"pos": [0.05, 0.05, 0.80], "grip": -1.0}}}
        m["tries"] = 0
        m["stage"] = "placed_ck"
        p = [hx + PLACE_CK[0], hy + PLACE_CK[1], hz + PLACE_CK[2]]
        return {"memory": m, "act": {"tool": "act_place", "args": {"place": p}}}

    if stage == "regrasp_ck":
        ck = _find_box(w, m["cookie"], radius=0.10)
        if ck is not None:
            m["cookie"] = [ck[0] + COOKIE_OFF[0], ck[1] + COOKIE_OFF[1]]
        m["stage"] = "check_ck"
        return {"memory": m, "act": {"tool": "act_pick", "args": {"target": m["cookie"], "grasp_z": GRASP_Z}}}

    if stage == "placed_ck":
        # cookies still lying at their table spot -> the grasp/carry failed; pick them again
        ck = _find_box(w, m["cookie"], radius=0.04)
        if ck is not None and m["tries"] < 2:
            m["tries"] += 1
            m["cookie"] = [ck[0] + COOKIE_OFF[0], ck[1] + COOKIE_OFF[1]]
            _note(m, "cookies still on table")
            m["stage"] = "check_ck"
            return {"memory": m, "act": {"tool": "act_pick", "args": {"target": m["cookie"], "grasp_z": GRASP_Z}}}
        m["tries"] = 0
        m["stage"] = "check_ch"
        return {"memory": m, "act": {"tool": "act_pick", "args": {"target": m["choc"], "grasp_z": GRASP_Z, "approach_z": 0.66}}}

    if stage == "check_ch":
        wd = _width(frame)
        if wd < 0.008 and m["tries"] < 2:
            m["tries"] += 1
            _note(m, "choc grasp missed w=%.3f" % wd)
            m["stage"] = "regrasp_ch"
            return {"memory": m, "act": {"tool": "act_move", "args": {"pos": [0.05, 0.05, 0.80], "grip": -1.0}}}
        m["stage"] = "verify"
        p = [hx + PLACE_CH[0], hy + PLACE_CH[1], hz + PLACE_CH[2]]
        return {"memory": m, "act": {"tool": "act_place", "args": {"place": p}}}

    if stage == "regrasp_ch":
        ch = _find_box(w, m["choc"], radius=0.10)
        if ch is not None:
            m["choc"] = [ch[0] + CHOC_OFF[0], ch[1] + CHOC_OFF[1]]
        m["stage"] = "check_ch"
        return {"memory": m, "act": {"tool": "act_pick", "args": {"target": m["choc"], "grasp_z": GRASP_Z, "approach_z": 0.66}}}

    if stage == "verify":
        ch = _find_box(w, m["choc"], radius=0.04)
        if ch is not None and m["tries"] < 3:
            m["tries"] += 1
            m["choc"] = [ch[0] + CHOC_OFF[0], ch[1] + CHOC_OFF[1]]
            _note(m, "chocolate still on table")
            m["stage"] = "check_ch"
            return {"memory": m, "act": {"tool": "act_pick", "args": {"target": m["choc"], "grasp_z": GRASP_Z, "approach_z": 0.66}}}
        m["stage"] = "done"
        return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 60, "grip": -1.0}}}

    # done: hold still; the official checker ends the episode on success
    m["stage"] = "done"
    return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 60, "grip": -1.0}}}
