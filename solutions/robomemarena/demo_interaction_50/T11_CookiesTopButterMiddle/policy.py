"""T11: put cookies into the top drawer, then butter into the middle drawer.

Perception: agentview RGB-D back-projected to world points (calibrated K, [R|t]).
  * cabinet top plane (z~0.713) -> top-handle grasp point (front edge / far edge offsets)
  * cookies / butter: raised blobs (z > table+7 mm) near their nominal table spots
Control: closed-loop Cartesian servo on measured TCP (OSC delta, world frame).
Memory: stage machine; every dependent step re-checks fingers / TCP evidence.
"""

TABLE_Z = 0.430
CAB_TOP_Z = 0.713
DRAWER_ROT = [-1.5707963, 0.0, 0.0]   # gripper points +y (toward the drawer fronts)
DOWN_ROT = [3.1415926, 0.0, 0.0]      # gripper points down
# Cookies (grasp and drop) lie at the arm's reach limit (shoulder-wrist 0.719 m = Panda max in demos);
# tilting the hand 0.35 rad about world y (fingertips toward +x) brings the wrist ~6 cm closer.
# Finger closing axis stays along world y.  Axis-angle of Ry(-0.35) @ Rx(pi):
TILT_ROT = [3.0936, 0.0, 0.5470]
NOMINAL_H1 = [0.135, 0.452, 0.675]    # fallback top-handle grasp point
NOMINAL_COOKIE = [0.218, -0.328]
NOMINAL_BUTTER = [0.110, -0.202]
COOKIE_OFF = [-0.007, 0.001]          # demo grasp minus visible-blob centroid
BUTTER_OFF = [-0.003, 0.003]
TOP_Z_OFF = -0.038                    # top handle z below cabinet top
MID_Z_OFF = -0.106                    # middle handle z below cabinet top
OPEN_TRAVEL = 0.157                   # handle travel of an opened drawer (demo push-start)
PULL_DIST = 0.225
GRASP_Z = 0.434
MAX_RETRY = 3


# ---------------------------------------------------------------- geometry
def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if math.pi - th < 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _world_points(frame):
    d = frame.agentview_depth.astype(np.float64) / 1000.0
    K = frame.agentview_K
    E = frame.agentview_extrinsic
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    X = (u - K[0, 2]) / K[0, 0] * d
    Y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([X, Y, d], -1).reshape(-1, 3)
    Wp = (P - E[:, 3]) @ E[:, :3]
    ok = d.reshape(-1) > 0
    return Wp[ok]


# ---------------------------------------------------------------- perception
def _perceive_cabinet(P):
    sel = (np.abs(P[:, 2] - CAB_TOP_Z) < 0.008) & (P[:, 1] > 0.15) & (P[:, 0] > -0.4) & (P[:, 0] < 0.6)
    Q = P[sel]
    if len(Q) < 300:
        return None
    ztop = float(np.median(Q[:, 2]))
    xmax = float(np.percentile(Q[:, 0], 99))
    ymin = float(np.percentile(Q[:, 1], 1))
    ymax = float(np.percentile(Q[:, 1], 99))
    hx = xmax - 0.128
    hy = (ymin + ymax) / 2.0 - 0.1121
    if not (-0.05 < hx < 0.30 and 0.30 < hy < 0.60 and 0.15 < ymax - ymin < 0.23):
        return None
    return [hx, hy, ztop]


def _perceive_blob(P, nominal, offset, win=0.06):
    sel = (np.abs(P[:, 0] - nominal[0]) < win) & (np.abs(P[:, 1] - nominal[1]) < win) & \
          (P[:, 2] > TABLE_Z + 0.007) & (P[:, 2] < TABLE_Z + 0.08)
    Q = P[sel]
    if len(Q) < 15:
        return None
    c = np.median(Q[:, :2], axis=0)
    # re-centre on the cluster nearest the median
    near = np.linalg.norm(Q[:, :2] - c, axis=1) < 0.05
    if near.sum() >= 10:
        c = Q[near, :2].mean(0)
    return [float(c[0] + offset[0]), float(c[1] + offset[1]), float(Q[:, 2].max())]


def _empty_memory():
    return {"stage": "init", "handle": None, "cookie": None, "butter": None,
            "retries": 0, "top_slip_dy": None, "mid_slip_dy": None,
            "grasp_width": None, "note": "", "adjust": 0.0, "target": None, "redo": None,
            "recoveries": 0}


def _fill(memory):
    m = _empty_memory()
    for k in m:
        if memory.get(k) is not None:
            m[k] = memory[k]
    return m


def _go(points, rot, grip, tol=0.005, max_steps=400, vmax=1.0, stall=25):
    return {"tool": "act_move", "args": {"points": points, "rot": rot, "grip": grip,
                                          "tol": tol, "max_steps": max_steps, "vmax": vmax,
                                          "stall": stall}}


def _grip(close, steps=12):
    return {"tool": "act_gripper", "args": {"close": close, "steps": steps}}


def _slip_dy(frames, y0):
    """TCP y displacement (from y0) at which the fingers lost the handle during a pull."""
    seen = False
    for fr in frames:
        w = _width(fr)
        if w > 0.009:
            seen = True
        if seen and w < 0.006:
            return float(fr.tcp[1] - y0)
    return None


# stages whose move must end at a precise pose before the next dependent action
PRECISE = {"approach_top": 0.012, "descend_cookie": 0.012, "carry_cookie": 0.03, "to_push": 0.03,
           "push_grasp": 0.012, "push_back": 0.012, "approach_mid": 0.012, "descend_butter": 0.012, "carry_butter": 0.03,
           "to_cookie": 0.03, "to_butter": 0.03}
SAFE_POS = [0.0, 0.12, 0.78]
SHOULDER = [-0.51, 0.0, 0.742]   # Panda joint-2 axis in world (FK fit to measured TCP, scratch/fk.py)


def _command(st, m, tcp):
    """The act that executes stage `st`, and the final TCP target it should reach (or None)."""
    hx, hy, zt = m["handle"]
    tz = zt + TOP_Z_OFF
    mz = zt + MID_Z_OFF
    h2y = hy + 0.003
    py = hy - OPEN_TRAVEL
    adj = m["adjust"] or 0.0
    cx, cy = m["cookie"]
    bx, by = m["butter"]
    near_cab = tcp[1] > hy - 0.30 and tcp[2] < 0.72
    if st == "approach_top":
        t = [hx, hy + adj, tz]
        return _go([[hx, hy - 0.10, tz], t], DRAWER_ROT, -1, tol=0.004), t
    if st == "regrasp_top":
        # after a slip the handle stayed ~3.5 cm ahead of the fingertips; without a slip it is still at the TCP
        y_h = (hy + m["top_slip_dy"] + 0.035) if m["top_slip_dy"] is not None else tcp[1]
        t = [hx, y_h, tz]
        return _go([[hx, t[1] - 0.08, tz], t], DRAWER_ROT, -1, tol=0.004), t
    if st in ("grasp_top", "push_close", "grasp_mid", "grasp_cookie", "grasp_butter"):
        return _grip(True), None
    if st in ("release_top", "release_push", "release_mid"):
        return _grip(False), None
    if st in ("drop_cookie", "drop_butter"):
        return _grip(False, 14), None
    if st == "pull_top":
        return _go([[hx, hy - PULL_DIST, tz]], DRAWER_ROT, 1, tol=0.006, max_steps=120, stall=15), None
    if st == "to_cookie":
        pts = []
        if near_cab:
            pts = [[hx, tcp[1] - 0.08, tcp[2]] + DRAWER_ROT, [hx, hy - 0.33, tcp[2] + 0.02]]
        t = [cx, cy, 0.60]
        return _go(pts + [t], TILT_ROT, -1, tol=0.01), t
    if st == "descend_cookie":
        t = [cx, cy, GRASP_Z + 0.003]
        return _go([[cx, cy, 0.60], t], TILT_ROT, -1, tol=0.004, max_steps=200), t
    if st == "carry_cookie":
        dx, dy = hx + 0.012, hy - 0.09
        t = [dx, dy, tz + 0.052]
        return _go([[tcp[0], tcp[1], 0.74], [dx, dy, 0.80], t], TILT_ROT, 1, tol=0.006), t
    if st == "to_push":
        t = [hx, py - 0.10, tz]
        pts = []
        if tcp[1] > hy - 0.20 and tcp[2] < 0.78:   # just released above the open drawer
            pts = [[tcp[0], tcp[1], 0.78] + TILT_ROT]
        return _go(pts + [[hx, py - 0.10, 0.74], t], DRAWER_ROT, -1, tol=0.006), t
    if st == "push_grasp":
        t = [hx, py + adj, tz]
        return _go([[hx, py - 0.10, tz], t], DRAWER_ROT, -1, tol=0.004, max_steps=150), t
    if st == "pushing":
        return _go([[hx, hy + 0.004, tz]], DRAWER_ROT, 1, tol=0.005, max_steps=120, stall=15), None
    if st == "push_back":
        # back off with an open hand and re-approach the handle 6 cm further in
        t = [hx, tcp[1], tz]
        return _go([[hx, tcp[1] - 0.06, tz], t], DRAWER_ROT, -1, tol=0.005, max_steps=120), t
    if st == "approach_mid":
        t = [hx, h2y + adj, mz]
        pts = [[hx, min(tcp[1], hy) - 0.10, max(tcp[2], tz)], [hx, h2y - 0.10, mz], t]
        return _go(pts, DRAWER_ROT, -1, tol=0.004), t
    if st == "regrasp_mid":
        y_h = (h2y + m["mid_slip_dy"] + 0.035) if m["mid_slip_dy"] is not None else tcp[1]
        t = [hx, y_h, mz]
        return _go([[hx, t[1] - 0.08, mz], t], DRAWER_ROT, -1, tol=0.004), t
    if st == "pull_mid":
        return _go([[hx, h2y - PULL_DIST, mz]], DRAWER_ROT, 1, tol=0.006, max_steps=120, stall=15), None
    if st == "to_butter":
        pts = []
        if near_cab:
            pts = [[hx, tcp[1] - 0.08, tcp[2]] + DRAWER_ROT, [hx, h2y - 0.33, tcp[2] + 0.03]]
        t = [bx, by, 0.60]
        return _go(pts + [t], DOWN_ROT, -1, tol=0.01), t
    if st == "descend_butter":
        t = [bx, by, GRASP_Z]
        return _go([[bx, by, 0.60], t], DOWN_ROT, -1, tol=0.004, max_steps=200), t
    if st == "carry_butter":
        dx, dy = hx + 0.015, h2y - 0.10
        t = [dx, dy, mz + 0.06]
        return _go([[tcp[0], tcp[1], 0.74], [dx, dy, 0.79], t], TILT_ROT, 1, tol=0.006), t
    if st in ("lift_away_cookie", "lift_away_butter"):
        r = TILT_ROT if st == "lift_away_cookie" else DOWN_ROT
        return _go([[tcp[0], tcp[1], max(tcp[2], 0.62)]], r, -1, tol=0.01, max_steps=80), None
    if st in ("lift_drop_cookie", "lift_drop_butter"):
        return _go([[tcp[0], tcp[1], max(tcp[2], 0.78)]], TILT_ROT, -1, tol=0.02, max_steps=60), None
    if st == "done":
        return _go([[tcp[0], tcp[1], 0.78]], TILT_ROT, -1, tol=0.01, max_steps=60), None
    return {"tool": "act_wait", "args": {"steps": 30}}, None


def _next(st, m, obs, tcp, w):
    """Evaluate the measured outcome of the stage that just ran; return the next stage."""
    hx, hy, zt = m["handle"]
    h2y = hy + 0.003
    if st in ("approach_top", "regrasp_top"):
        return "grasp_top"
    if st == "grasp_top":
        m["grasp_width"] = w
        if 0.008 < w < 0.04:
            return "pull_top"
        m["retries"] += 1
        m["note"] = "top handle grasp missed (w=%.4f)" % w
        m["adjust"] = [0.0, 0.008, -0.006, 0.014][m["retries"] % 4]
        return "approach_top"
    if st == "pull_top":
        sd = _slip_dy(obs.frames, hy)
        m["top_slip_dy"] = sd
        travel = (hy - tcp[1]) if sd is None else -sd
        if travel < 0.12:
            m["note"] = "top drawer pull short (%.3f)" % travel
            m["retries"] += 1
            return "release_top_retry"
        return "release_top"
    if st == "release_top_retry":
        return "regrasp_top"
    if st == "release_top":
        m["adjust"] = 0.0
        return "to_cookie"
    if st == "to_cookie":
        return "descend_cookie"
    if st == "descend_cookie":
        return "grasp_cookie"
    if st == "grasp_cookie":
        m["grasp_width"] = w
        if w > 0.03:
            return "carry_cookie"
        m["retries"] += 1
        m["note"] = "cookie grasp missed (w=%.4f)" % w
        return "lift_away_cookie"
    if st == "lift_away_cookie":
        P = _world_points(obs.current)
        ck = _perceive_blob(P, m["cookie"], [0.0, 0.0], win=0.08)
        if ck is not None:
            m["cookie"] = [ck[0] + COOKIE_OFF[0], ck[1] + COOKIE_OFF[1]]
        return "descend_cookie"
    if st == "carry_cookie":
        if w < 0.02:
            m["note"] = "cookies lost during carry (w=%.4f)" % w
            m["retries"] += 1
            return "lift_drop_cookie"
        return "drop_cookie"
    if st == "lift_drop_cookie":
        return "to_cookie"
    if st == "drop_cookie":
        m["adjust"] = 0.0
        return "to_push"
    if st == "to_push":
        return "push_grasp"
    if st == "push_grasp":
        return "push_close"
    if st == "push_close":
        m["grasp_width"] = w
        if 0.008 < w < 0.04:
            return "pushing"
        m["retries"] += 1
        m["note"] = "open top handle grasp missed (w=%.4f)" % w
        m["adjust"] = [0.0, 0.012, -0.01, 0.024][m["retries"] % 4]
        return "push_grasp"
    if st == "pushing":
        if tcp[1] < hy - 0.03 and m["retries"] < 8:
            m["note"] = "top drawer push stopped at dy=%.3f" % (tcp[1] - hy)
            m["retries"] += 1
            return "release_push_retry"
        return "release_push"
    if st == "release_push_retry":
        return "push_back"
    if st == "push_back":
        return "push_close"
    if st == "release_push":
        m["adjust"] = 0.0
        return "approach_mid"
    if st in ("approach_mid", "regrasp_mid"):
        return "grasp_mid"
    if st == "grasp_mid":
        m["grasp_width"] = w
        if 0.008 < w < 0.04:
            return "pull_mid"
        m["retries"] += 1
        m["note"] = "middle handle grasp missed (w=%.4f)" % w
        m["adjust"] = [0.0, 0.008, -0.006, 0.014][m["retries"] % 4]
        return "approach_mid"
    if st == "pull_mid":
        sd = _slip_dy(obs.frames, h2y)
        m["mid_slip_dy"] = sd
        travel = (h2y - tcp[1]) if sd is None else -sd
        if travel < 0.12:
            m["note"] = "middle drawer pull short (%.3f)" % travel
            m["retries"] += 1
            return "release_mid_retry"
        return "release_mid"
    if st == "release_mid_retry":
        return "regrasp_mid"
    if st == "release_mid":
        return "to_butter"
    if st == "to_butter":
        return "descend_butter"
    if st == "descend_butter":
        return "grasp_butter"
    if st == "grasp_butter":
        m["grasp_width"] = w
        if w > 0.02:
            return "carry_butter"
        m["retries"] += 1
        m["note"] = "butter grasp missed (w=%.4f)" % w
        return "lift_away_butter"
    if st == "lift_away_butter":
        P = _world_points(obs.current)
        bt = _perceive_blob(P, m["butter"], [0.0, 0.0], win=0.08)
        if bt is not None:
            m["butter"] = [bt[0] + BUTTER_OFF[0], bt[1] + BUTTER_OFF[1]]
        return "descend_butter"
    if st == "carry_butter":
        if w < 0.015:
            m["note"] = "butter lost during carry (w=%.4f)" % w
            m["retries"] += 1
            return "lift_drop_butter"
        return "drop_butter"
    if st == "lift_drop_butter":
        return "to_butter"
    if st == "drop_butter":
        return "done"
    return "wait"


ALIAS = {"release_top_retry": "release_top", "release_push_retry": "release_push",
         "release_mid_retry": "release_mid"}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: open top drawer, cookies in, close top drawer, open middle drawer, butter in.
    Each decision evaluates the measured outcome (TCP, finger width, slip point) of the stage that
    just ran, then commands the next stage (or a recovery)."""
    m = _fill(memory or {})
    cur = obs.current
    tcp = [float(x) for x in cur.tcp[:3]]
    w = _width(cur)
    st = m["stage"]

    if st == "init" or m["handle"] is None:
        P = _world_points(cur)
        cab = _perceive_cabinet(P)
        if cab is None:
            m["handle"] = [NOMINAL_H1[0], NOMINAL_H1[1], CAB_TOP_Z]
            m["note"] = "cabinet not detected; nominal handle"
        else:
            m["handle"] = cab
            m["note"] = "cabinet detected"
        ck = _perceive_blob(P, NOMINAL_COOKIE, COOKIE_OFF)
        bt = _perceive_blob(P, NOMINAL_BUTTER, BUTTER_OFF)
        m["cookie"] = ck[:2] if ck else list(NOMINAL_COOKIE)
        m["butter"] = bt[:2] if bt else list(NOMINAL_BUTTER)
        if ck is None or bt is None:
            m["note"] += "; object blob missing -> nominal"
        nxt = "approach_top"
    elif st == "recover":
        nxt = m["redo"] or "wait"
    else:
        # did the last precise move reach its target?  (joint-limit / collision stalls)
        tgt = m["target"]
        # elbow at its limit (joint 4 upper limit -0.07) only matters together with a missed target
        elbow_straight = float(cur.joints[3]) > -0.10
        if st in PRECISE and tgt is not None and m["recoveries"] < 4 and (
                float(np.linalg.norm(np.asarray(tcp) - np.asarray(tgt))) > (PRECISE[st] * 0.5 if elbow_straight else PRECISE[st])):
            m["recoveries"] += 1
            m["note"] = "%s missed target by %.3f (elbow %.2f); recovering via safe pose" % (
                st, float(np.linalg.norm(np.asarray(tcp) - np.asarray(tgt))), float(cur.joints[3]))
            m["redo"] = st
            m["stage"] = "recover"
            cmd, _ = _command(st, m, tcp)
            rot = cmd["args"]["rot"] if cmd["tool"] == "act_move" else DOWN_ROT
            grip = cmd["args"]["grip"] if cmd["tool"] == "act_move" else -1
            pts = [[tcp[0] - 0.05, tcp[1], tcp[2] + 0.03], SAFE_POS]
            if float(cur.joints[3]) > -0.30:
                # elbow on the over-stretched branch: first reach outward past the singularity so the
                # controller's posture term can fold the elbow back (seen in R001/R003 carry), then go home
                d = np.asarray(tcp) - np.asarray(SHOULDER)
                d = d / max(float(np.linalg.norm(d)), 1e-6)
                pts = [[tcp[0] + 0.10 * float(d[0]), tcp[1] + 0.10 * float(d[1]), tcp[2] + 0.10 * float(d[2])]] + pts
            m["target"] = None
            return {"memory": m, "act": _go(pts, rot, grip, tol=0.03, max_steps=240, stall=20)}
        nxt = _next(st, m, obs, tcp, w)
    m["stage"] = nxt
    cmd, tgt = _command(ALIAS.get(nxt, nxt), m, tcp)
    m["target"] = tgt
    return {"memory": m, "act": cmd}


# ---------------------------------------------------------------- actions
def act_move(robot, points, rot, grip, tol=0.005, max_steps=400, vmax=1.0, stall=25, rmax=0.5):
    """Servo the TCP through world waypoints `points` ([[x,y,z],...] or [x,y,z,rx,ry,rz] to override
    the orientation for that point) at axis-angle `rot` (rotation action clipped to `rmax`),
    gripper sign `grip` (+1 close, -1 open). Intermediate waypoints use a 2 cm tolerance.
    Stops a segment if no progress for `stall` steps. Returns measured TCP, finger width, steps."""
    Rt = _aa2R(rot)
    steps = 0
    reached = []
    R_default = Rt
    for j, pt in enumerate(points):
        target = np.asarray(pt[:3], dtype=float)
        Rt = _aa2R(pt[3:6]) if len(pt) >= 6 else R_default
        seg_tol = tol if j == len(points) - 1 else max(tol, 0.02)
        rtol = 0.04 if j == len(points) - 1 else 0.15
        best = 1e9
        still = 0
        ok = False
        while steps < max_steps:
            f = robot.now()
            p = np.asarray(f.tcp[:3], dtype=float)
            e = target - p
            er = _R2aa(Rt @ _aa2R(f.tcp[3:6]).T)
            d = float(np.linalg.norm(e))
            dr = float(np.linalg.norm(er))
            if d < seg_tol and dr < rtol:
                ok = True
                break
            score = d + 0.05 * dr
            if score < best - 0.0008:
                best = score
                still = 0
            else:
                still += 1
                if still > stall:
                    break
            a = 1.6 * e / 0.05
            n = float(np.max(np.abs(a)))
            if n > vmax:
                a = a * (vmax / n)
            ar = np.clip(2.0 * er / 0.5, -rmax, rmax)
            robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
            steps += 1
        reached.append(ok)
    f = robot.now()
    return {"reached": reached, "steps": steps, "tcp": [round(float(x), 4) for x in f.tcp[:3]],
            "width": round(_width(f), 4)}


def act_gripper(robot, close, steps=12):
    """Close (close=True) or open the gripper for `steps` steps while holding the current TCP pose.
    Returns the finger width (m) after the motion."""
    f0 = robot.now()
    p0 = np.asarray(f0.tcp[:3], dtype=float)
    R0 = _aa2R(f0.tcp[3:6])
    g = 1.0 if close else -1.0
    for _ in range(int(steps)):
        f = robot.now()
        e = p0 - np.asarray(f.tcp[:3], dtype=float)
        er = _R2aa(R0 @ _aa2R(f.tcp[3:6]).T)
        a = np.clip(1.6 * e / 0.05, -0.3, 0.3)
        ar = np.clip(2.0 * er / 0.5, -0.3, 0.3)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), g)
    f = robot.now()
    return {"width": round(_width(f), 4), "tcp": [round(float(x), 4) for x in f.tcp[:3]]}


def act_wait(robot, steps=30):
    """Hold position with the gripper open for `steps` steps (lets the success checker settle)."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    return {"waited": int(steps)}
