"""T22 Pour tomato sauce over cookies twice, then put the cookies into the microwave.

Stage machine (see policy.md).  Perception: agentview depth back-projected to
world points; objects are clusters above the table (z ~ 0.43 m) near their
expected regions.  Control: world-frame P-control on the measured TCP pose
(position gain 6 /m, rotation gain 2, as measured in the expert actions).
"""

TABLE_Z = 0.43
# Offsets learned from 50 assigned demonstrations (see task_analysis.md).
SAUCE_PRIOR = [-0.03, -0.20]
COOKIE_PRIOR = [0.17, -0.195]
MW_PRIOR = [0.17, 0.39]          # [x_max, y_min] of microwave points
SAUCE_ABOVE_OFF = [-0.0125, -0.0006]
SAUCE_GRASP_OFF = [-0.0125, -0.0106]
POUR_OFF = [-0.0039, -0.0010]
COOKIE_GRASP_OFF = [-0.0033, -0.0011]
DOOR_PRE_OFF = [-0.1227, -0.1177]
DOOR_OFF = [-0.1227, 0.0322]
PULL_Y_OFF = -0.1678
RETREAT_OFF = [-0.6226, -0.2678]
CARRY_OFF = [-0.1477, 0.1998]
INSERT_X_OFF = -0.1661
Z_SAUCE_ABOVE = 0.668
Z_SAUCE_GRASP = 0.468
Z_SAUCE_LIFT = 0.818
Z_POUR = 0.683
Z_DOOR = 0.582
Z_COOKIE_ABOVE = 0.633
Z_COOKIE_PRESS = 0.403
Z_COOKIE_LIFT = 0.733
Z_CARRY = 0.728
Z_INSERT = 0.582
PLACE_ABOVE = [0.0, -0.2, 0.6]
PLACE_DOWN = [0.0, -0.2, 0.5]
UP_AFTER_PLACE = [-0.05, -0.1, 0.7]
PULL_X = -0.435
HOLD_WIDTH_MIN = 0.04     # sauce / cookies held: ~0.062 m measured in demos
HANDLE_WIDTH_MIN = 0.008  # door handle held: ~0.023-0.028 m in demos
OPEN_WIDTH = 0.072        # fully open fingers measure ~0.078-0.080 m
MAX_RETRY = 2


def _rotx(a):
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _named_rot(name):
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    if name == "down":
        return down
    if name == "pour":
        return _rotx(math.pi / 4) @ down
    if name == "side":
        # gripper approach axis along world +y, fingers close along world x
        return np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    return None


def _aa2mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if math.pi - th < 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _world_points(frame):
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    Pc = np.stack([x, y, d], -1).reshape(-1, 3)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    ok = d.reshape(-1) > 0
    return Pw[ok]


def _cluster(P, prior, zlo, zhi, r1=0.08, r2=0.06):
    m = (P[:, 2] > zlo) & (P[:, 2] < zhi)
    Q = P[m]
    c = np.asarray(prior, dtype=float)
    sel = Q[np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]) < r1]
    if len(sel) < 15:
        return None, int(len(sel))
    c = sel[:, :2].mean(0)
    sel = Q[np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]) < r2]
    if len(sel) < 15:
        return None, int(len(sel))
    return [float(sel[:, 0].mean()), float(sel[:, 1].mean())], int(len(sel))


def _detect_scene(frame):
    P = _world_points(frame)
    sauce, ns = _cluster(P, SAUCE_PRIOR, TABLE_Z + 0.007, 0.62)
    cookie, nc = _cluster(P, COOKIE_PRIOR, TABLE_Z + 0.006, 0.50)
    m = (P[:, 1] > 0.25) & (P[:, 2] > 0.44) & (P[:, 2] < 0.8) & (P[:, 0] > -0.4) & (P[:, 0] < 0.4)
    Q = P[m]
    mw = None
    if len(Q) > 300:
        mw = [float(Q[:, 0].max()), float(Q[:, 1].min())]
    return {"sauce": sauce, "n_sauce": ns, "cookie": cookie, "n_cookie": nc,
            "mw": mw, "n_mw": int(len(Q))}


def _door_open_count(frame, mw):
    P = _world_points(frame)
    # open door panel swings to x in [M_x-0.50, M_x-0.33], y in [M_y-0.20, M_y]; the closed
    # microwave has no points with y < M_y - 0.02 (checked on 50 demos and 26 rollouts)
    m = ((P[:, 0] > mw[0] - 0.56) & (P[:, 0] < mw[0] - 0.28) & (P[:, 1] > mw[1] - 0.30)
         & (P[:, 1] < mw[1] - 0.02) & (P[:, 2] > 0.45) & (P[:, 2] < 0.75))
    return int(m.sum())


def perceive_scene(obs):
    """Detect sauce [x,y], cookies [x,y] and microwave [x_max, y_min] from agentview depth."""
    return _detect_scene(obs.current)


# ---------------------------------------------------------------- control

def _servo(robot, wp):
    """Run one waypoint; returns a summary dict."""
    grip = float(wp.get("grip", 0.0))
    kp = float(wp.get("kp", 6.0))
    kr = float(wp.get("kr", 2.0))
    tol = float(wp.get("tol", 0.003))
    rtol = float(wp.get("rtol", 0.02))
    nmin = int(wp.get("min", 0))
    nmax = int(wp.get("max", 100))
    stall = int(wp.get("stall", 0))
    x_below = wp.get("x_below")
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    pos = wp.get("pos")
    tgt = tcp[:3].copy() if pos is None else np.asarray(pos, dtype=float)
    for i in range(3):
        if pos is not None and pos[i] is None:
            tgt[i] = tcp[i]
    rot = wp.get("rot")
    Rt = _aa2mat(tcp[3:]) if rot is None else _named_rot(rot)
    hist = []
    n = 0
    done = False
    while n < nmax:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        ep = tgt - tcp[:3]
        er = _mat2aa(Rt @ _aa2mat(tcp[3:]).T)
        hist.append(tcp[:3].copy())
        conv = float(np.linalg.norm(ep)) < tol and float(np.linalg.norm(er)) < rtol
        if n >= nmin:
            if conv:
                done = True
                break
            if x_below is not None and tcp[0] < float(x_below):
                done = True
                break
            if stall > 0 and len(hist) > stall:
                if float(np.linalg.norm(hist[-1] - hist[-1 - stall])) < 0.001:
                    done = True
                    break
        a = np.clip(kp * ep, -1.0, 1.0)
        b = np.clip(kr * er, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(b[0]), float(b[1]), float(b[2]), grip)
        n += 1
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"name": wp.get("name", ""), "steps": n, "reached": bool(done),
            "err": float(np.linalg.norm(tgt - tcp[:3])), "tcp": [round(float(v), 4) for v in tcp[:3]],
            "width": round(_width(fr), 4)}


def act_waypoints(robot, waypoints):
    """Execute waypoints in order. Each: {name, pos [x,y,z] (None entries keep current), rot
    ('down'|'pour'|'side'|None keep), grip (+1 close, -1 open, 0 hold), min, max steps, tol m,
    rtol rad, kp, kr, stall (steps w/o 1mm motion ends), x_below}. Returns per-waypoint
    measured TCP, finger width, steps and whether the target/stop condition was reached."""
    out = []
    for wp in waypoints:
        out.append(_servo(robot, wp))
    fr = robot.now()
    return {"results": out, "final_tcp": [round(float(v), 4) for v in fr.tcp], "width": round(_width(fr), 4)}


def act_observe(robot, steps=5, grip=0.0):
    """Hold the current pose for `steps` native steps with the given gripper command."""
    r = _servo(robot, {"name": "hold", "pos": None, "rot": None, "grip": grip, "min": steps, "max": steps})
    return r


# ---------------------------------------------------------------- policy

def _off(base, off):
    return [base[0] + off[0], base[1] + off[1]]


def _plan_sauce_grasp(s):
    a = _off(s, SAUCE_ABOVE_OFF)
    g = _off(s, SAUCE_GRASP_OFF)
    return [
        {"name": "above_sauce", "pos": [a[0], a[1], Z_SAUCE_ABOVE], "rot": "down", "grip": -1, "max": 120, "tol": 0.004},
        {"name": "descend_sauce", "pos": [g[0], g[1], Z_SAUCE_GRASP], "rot": "down", "grip": -1, "max": 120, "tol": 0.002},
        {"name": "close_sauce", "pos": None, "rot": None, "grip": 1, "min": 30, "max": 30},
    ]


def _plan_lift_to_pour(s, c):
    a = _off(s, SAUCE_ABOVE_OFF)
    p = _off(c, POUR_OFF)
    return [
        {"name": "lift_sauce", "pos": [a[0], a[1], Z_SAUCE_LIFT], "rot": "down", "grip": 1, "max": 80, "tol": 0.01},
        {"name": "to_pour", "pos": [p[0], p[1], Z_POUR], "rot": "down", "grip": 1, "max": 120, "tol": 0.003},
    ]


def _plan_pour(c):
    p = _off(c, POUR_OFF)
    return [
        {"name": "tilt", "pos": [p[0], p[1], Z_POUR], "rot": "pour", "grip": 1, "max": 60, "tol": 0.01, "rtol": 0.02},
        {"name": "hold_tilt", "pos": None, "rot": None, "grip": 1, "min": 30, "max": 30},
        {"name": "untilt", "pos": [p[0], p[1], Z_POUR], "rot": "down", "grip": 1, "max": 80, "tol": 0.004, "rtol": 0.02},
    ]


def _plan_place():
    return [
        {"name": "place_above", "pos": PLACE_ABOVE, "rot": "down", "grip": 1, "max": 100, "tol": 0.004},
        {"name": "place_down", "pos": PLACE_DOWN, "rot": "down", "grip": 1, "max": 60, "tol": 0.008, "stall": 8, "min": 10},
        {"name": "release_sauce", "pos": None, "rot": None, "grip": -1, "min": 20, "max": 20},
        {"name": "up_after_place", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 80, "tol": 0.006},
    ]


def _plan_door_grasp(mw, dy=0.0):
    pre = _off(mw, DOOR_PRE_OFF)
    d = _off(mw, DOOR_OFF)
    return [
        {"name": "door_pre", "pos": [pre[0], pre[1] + dy, Z_DOOR], "rot": "side", "grip": -1, "max": 120, "tol": 0.004, "rtol": 0.03},
        {"name": "door_in", "pos": [d[0], d[1] + dy, Z_DOOR], "rot": "side", "grip": -1, "max": 50, "tol": 0.003, "stall": 6, "min": 10},
        {"name": "close_handle", "pos": None, "rot": None, "grip": 1, "min": 20, "max": 20},
    ]


def _plan_door_pull(mw):
    r = _off(mw, RETREAT_OFF)
    return [
        {"name": "pull", "pos": [PULL_X, mw[1] + PULL_Y_OFF, Z_DOOR], "rot": "side", "grip": 1, "max": 50, "min": 20, "x_below": -0.318, "tol": 0.003},
        {"name": "release_handle", "pos": None, "rot": None, "grip": -1, "min": 10, "max": 10},
        {"name": "retreat", "pos": [r[0], r[1], Z_DOOR], "rot": "side", "grip": -1, "max": 30, "min": 30},
        {"name": "rot_down", "pos": None, "rot": "down", "grip": -1, "max": 60, "rtol": 0.03, "tol": 1.0},
    ]


def _plan_door_backoff(mw):
    pre = _off(mw, DOOR_PRE_OFF)
    return [
        {"name": "open_handle", "pos": None, "rot": None, "grip": -1, "min": 15, "max": 15},
        {"name": "door_back", "pos": [pre[0], pre[1], Z_DOOR], "rot": "side", "grip": -1, "max": 80, "tol": 0.006, "rtol": 0.05},
    ]


def _plan_cookie_above(c):
    a = _off(c, COOKIE_GRASP_OFF)
    return [
        {"name": "above_cookie", "pos": [a[0], a[1], Z_COOKIE_ABOVE], "rot": "down", "grip": -1, "max": 150, "tol": 0.003},
    ]


def _plan_cookie_grasp(c):
    a = _off(c, COOKIE_GRASP_OFF)
    return [
        {"name": "above_cookie", "pos": [a[0], a[1], Z_COOKIE_ABOVE], "rot": "down", "grip": -1, "max": 60, "tol": 0.003},
        {"name": "press_cookie", "pos": [a[0], a[1], Z_COOKIE_PRESS], "rot": "down", "grip": -1, "max": 90, "min": 30, "stall": 10, "tol": 0.001},
        {"name": "close_cookie", "pos": None, "rot": None, "grip": 1, "min": 30, "max": 30},
    ]


def _plan_carry(g):
    cy = _off(g, CARRY_OFF)
    return [
        {"name": "lift_cookie", "pos": [g[0], g[1], Z_COOKIE_LIFT], "rot": "down", "grip": 1, "max": 80, "tol": 0.008},
        {"name": "carry", "pos": [cy[0], cy[1], Z_CARRY], "rot": "down", "grip": 1, "max": 100, "tol": 0.004},
        {"name": "rot_side", "pos": [cy[0], cy[1], Z_CARRY], "rot": "side", "grip": 1, "max": 80, "tol": 0.006, "rtol": 0.03},
    ]


def _plan_insert(mw, g, c, ydepth):
    x = mw[0] + INSERT_X_OFF + (g[0] - c[0])
    return [
        {"name": "insert", "pos": [x, mw[1] + ydepth, Z_INSERT], "rot": "side", "grip": 1, "max": 90, "tol": 0.004},
        {"name": "hold_in", "pos": None, "rot": None, "grip": 1, "min": 20, "max": 20},
    ]


def _mem(memory, names):
    out = {}
    for n in names:
        out[n] = memory.get(n) if isinstance(memory, dict) else None
    return out


FIELDS = ["stage", "sauce", "cookie", "mw", "pour_count", "retries", "total_retries",
          "cookie_grasp", "door_open", "insert_depth", "notes"]


def _act(memory, tool, args):
    return {"memory": memory, "act": {"tool": tool, "args": args}}


def perceive_decide(obs, goal, memory, last):
    """Stage machine for: grasp sauce, pour over cookies twice, place sauce, open microwave
    door, grasp cookies, insert them into the microwave. Verifies grasps by finger width and
    the door by agentview depth; bounded retries."""
    m = _mem(memory, FIELDS)
    fr = obs.current
    w = _width(fr)
    tcp = [float(v) for v in fr.tcp]
    if m["pour_count"] is None:
        m["pour_count"] = 0
    if m["retries"] is None:
        m["retries"] = 0
    if m["total_retries"] is None:
        m["total_retries"] = 0
    if m["notes"] is None:
        m["notes"] = ""
    st = m["stage"]
    if st is None or st == "perceive":
        det = _detect_scene(fr)
        miss = []
        for k, prior in (("sauce", SAUCE_PRIOR), ("cookie", COOKIE_PRIOR), ("mw", MW_PRIOR)):
            if det[k] is None:
                miss.append(k)
        if miss and m["retries"] < 2:
            m["retries"] += 1
            m["stage"] = "perceive"
            m["notes"] = "missing " + ",".join(miss)
            return _act(m, "act_observe", {"steps": 5, "grip": -1})
        m["sauce"] = det["sauce"] if det["sauce"] is not None else list(SAUCE_PRIOR)
        m["cookie"] = det["cookie"] if det["cookie"] is not None else list(COOKIE_PRIOR)
        m["mw"] = det["mw"] if det["mw"] is not None else list(MW_PRIOR)
        m["notes"] = "det n=%d,%d,%d miss=%s" % (det["n_sauce"], det["n_cookie"], det["n_mw"], ",".join(miss))
        m["retries"] = 0
        m["stage"] = "sauce_grasp"
        return _act(m, "act_waypoints", {"waypoints": _plan_sauce_grasp(m["sauce"])})

    if st == "sauce_grasp":
        # the grasp plan has been executed; verify the bottle is between the fingers
        if not (HOLD_WIDTH_MIN < w < OPEN_WIDTH):
            if m["retries"] >= MAX_RETRY:
                m["notes"] = "sauce grasp failed repeatedly; continuing without pour"
                m["stage"] = "door_grasp_go"
                m["retries"] = 0
                return _act(m, "act_waypoints", {"waypoints": [
                    {"name": "open", "pos": None, "rot": None, "grip": -1, "min": 15, "max": 15},
                    {"name": "up", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 120, "tol": 0.006}]})
            m["retries"] += 1
            m["total_retries"] += 1
            m["stage"] = "sauce_regrasp"
            m["notes"] = "sauce grasp missed (width %.3f)" % w
            return _act(m, "act_waypoints", {"waypoints": [
                {"name": "open", "pos": None, "rot": None, "grip": -1, "min": 15, "max": 15},
                {"name": "up", "pos": [tcp[0], tcp[1], Z_SAUCE_ABOVE], "rot": "down", "grip": -1, "max": 80, "tol": 0.006},
                {"name": "clear", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 80, "tol": 0.01}]})
        m["retries"] = 0
        m["stage"] = "to_pour"
        return _act(m, "act_waypoints", {"waypoints": _plan_lift_to_pour(m["sauce"], m["cookie"])})

    if st == "sauce_regrasp":
        det = _detect_scene(fr)
        if det["sauce"] is not None:
            m["sauce"] = det["sauce"]
        m["stage"] = "sauce_grasp"
        return _act(m, "act_waypoints", {"waypoints": _plan_sauce_grasp(m["sauce"])})

    if st in ("to_pour", "pour"):
        if not (HOLD_WIDTH_MIN < w < OPEN_WIDTH):
            # bottle lost: try to find it again on the table and regrasp
            m["notes"] = "sauce lost before pour %d (width %.3f)" % (m["pour_count"], w)
            m["total_retries"] += 1
            if m["total_retries"] > 4:
                m["stage"] = "door_grasp_go"
                return _act(m, "act_waypoints", {"waypoints": [
                    {"name": "open", "pos": None, "rot": "down", "grip": -1, "min": 10, "max": 40},
                    {"name": "up", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 120, "tol": 0.006}]})
            m["stage"] = "sauce_regrasp"
            return _act(m, "act_waypoints", {"waypoints": [
                {"name": "open", "pos": None, "rot": "down", "grip": -1, "min": 10, "max": 40},
                {"name": "clear", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 120, "tol": 0.01}]})
        if st == "pour":
            m["pour_count"] = int(m["pour_count"]) + 1
        if int(m["pour_count"]) >= 2:
            m["stage"] = "place"
            return _act(m, "act_waypoints", {"waypoints": _plan_place()})
        m["stage"] = "pour"
        return _act(m, "act_waypoints", {"waypoints": _plan_pour(m["cookie"])})

    if st in ("place", "door_grasp_go"):
        m["retries"] = 0
        m["stage"] = "door_grasp"
        return _act(m, "act_waypoints", {"waypoints": _plan_door_grasp(m["mw"])})

    if st == "door_grasp":
        if not (HANDLE_WIDTH_MIN < w < OPEN_WIDTH):
            if m["retries"] < MAX_RETRY:
                m["retries"] += 1
                m["total_retries"] += 1
                m["notes"] = "handle grasp missed (width %.3f)" % w
                m["stage"] = "door_regrasp"
                return _act(m, "act_waypoints", {"waypoints": _plan_door_backoff(m["mw"])})
            m["notes"] = "handle grasp missed; pulling anyway"
        m["stage"] = "door_pull"
        return _act(m, "act_waypoints", {"waypoints": _plan_door_pull(m["mw"])})

    if st == "door_regrasp":
        dy = 0.0
        m["stage"] = "door_grasp"
        return _act(m, "act_waypoints", {"waypoints": _plan_door_grasp(m["mw"], dy)[1:]})

    if st == "door_pull":
        m["stage"] = "door_check"
        return _act(m, "act_waypoints", {"waypoints": _plan_cookie_above(m["cookie"])})

    if st == "door_check":
        cnt = _door_open_count(fr, m["mw"])
        m["door_open"] = bool(cnt > 100)
        m["notes"] = "door count %d" % cnt
        if not m["door_open"] and m["retries"] < MAX_RETRY:
            m["retries"] += 1
            m["total_retries"] += 1
            m["stage"] = "door_grasp"
            wps = [{"name": "up", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 120, "tol": 0.01}]
            return _act(m, "act_waypoints", {"waypoints": wps + _plan_door_grasp(m["mw"])})
        m["retries"] = 0
        m["stage"] = "cookie_grasp"
        return _act(m, "act_waypoints", {"waypoints": _plan_cookie_grasp(m["cookie"])})

    if st == "cookie_grasp":
        if not (HOLD_WIDTH_MIN < w < OPEN_WIDTH):
            if m["retries"] < MAX_RETRY:
                m["retries"] += 1
                m["total_retries"] += 1
                m["notes"] = "cookie grasp missed (width %.3f)" % w
                m["stage"] = "cookie_regrasp"
                return _act(m, "act_waypoints", {"waypoints": [
                    {"name": "open", "pos": None, "rot": None, "grip": -1, "min": 15, "max": 15},
                    {"name": "up", "pos": [tcp[0], tcp[1], Z_COOKIE_ABOVE], "rot": "down", "grip": -1, "max": 80, "tol": 0.006},
                    {"name": "clear", "pos": [tcp[0] - 0.15, tcp[1], Z_COOKIE_ABOVE + 0.05], "rot": "down", "grip": -1, "max": 80, "tol": 0.01}]})
        m["cookie_grasp"] = [tcp[0], tcp[1]]
        m["retries"] = 0
        m["stage"] = "carry"
        return _act(m, "act_waypoints", {"waypoints": _plan_carry(m["cookie_grasp"])})

    if st == "cookie_regrasp":
        P = _world_points(fr)
        c, n = _cluster(P, m["cookie"], TABLE_Z + 0.006, 0.50)
        if c is not None:
            m["cookie"] = c
        m["stage"] = "cookie_grasp"
        return _act(m, "act_waypoints", {"waypoints": _plan_cookie_grasp(m["cookie"])})

    if st == "carry":
        if not (HOLD_WIDTH_MIN < w < OPEN_WIDTH):
            m["notes"] = "cookies lost during carry (width %.3f)" % w
            m["total_retries"] += 1
            m["stage"] = "cookie_regrasp"
            return _act(m, "act_waypoints", {"waypoints": [
                {"name": "open", "pos": None, "rot": "down", "grip": -1, "min": 10, "max": 60},
                {"name": "clear", "pos": UP_AFTER_PLACE, "rot": "down", "grip": -1, "max": 120, "tol": 0.01}]})
        m["insert_depth"] = 0.03
        m["stage"] = "insert"
        return _act(m, "act_waypoints", {"waypoints": _plan_insert(m["mw"], m["cookie_grasp"], m["cookie"], 0.03)})

    if st == "insert":
        # no success yet: push a little deeper, then release inside
        d = float(m["insert_depth"] or 0.03)
        if d < 0.07:
            m["insert_depth"] = d + 0.03
            return _act(m, "act_waypoints", {"waypoints": _plan_insert(m["mw"], m["cookie_grasp"], m["cookie"], d + 0.03)})
        m["stage"] = "release"
        return _act(m, "act_waypoints", {"waypoints": [
            {"name": "release_cookie", "pos": None, "rot": None, "grip": -1, "min": 20, "max": 20},
            {"name": "back_out", "pos": [tcp[0], tcp[1] - 0.2, tcp[2]], "rot": "side", "grip": -1, "max": 60, "tol": 0.005}]})

    m["stage"] = "done"
    return _act(m, "act_observe", {"steps": 40, "grip": -1})
