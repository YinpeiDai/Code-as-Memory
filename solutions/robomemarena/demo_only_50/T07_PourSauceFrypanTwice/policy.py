"""T07 PourSauceFrypanTwice: grasp the tomato-sauce bottle, pour over the frypan twice,
(fallback) place the bottle into the bowl drainer.

Controller structure, gains, thresholds and target offsets were identified from the 50 assigned
expert recordings (exact P-controller fits, see task_analysis.md). Object positions come from
the agentview depth image back-projected with the supplied calibration.
"""

TABLE_Z = 0.43
# grasp target = visible-bottle-point centroid + offset (fit over 50 demos, std < 1 mm)
GRASP_OFF = [-0.0111, 0.0010]
HOVER_Z = 0.6682
DESCEND_Z = 0.4582
LIFT_Z = 0.8182
# pour target = frypan rim-circle centre + offset (fit over 50 demos, std < 0.5 mm)
POUR_OFF = [-0.0015, -0.0074]
POUR_Z = 0.7232
TILT_RAD = 0.7853981633974483  # 45 deg about -world y (bottle top tips toward -x)
DEFAULT_BOTTLE = [0.0106, -0.0010, 0.48]
DEFAULT_PAN = [0.0014, 0.2576, 0.465]
DEFAULT_DRAINER = [-0.14, 0.46, 0.59]
MEMORY_FIELDS = ["phase", "bottle", "frypan", "drainer", "pours", "grasp_tries", "log"]


# ---------------------------------------------------------------- geometry helpers
def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        return V[:, int(np.argmax(w))] * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return k * th


def _R_down():
    return np.diag([1.0, -1.0, -1.0])


def _R_tilt():
    return _aa2R([0.0, -TILT_RAD, 0.0]) @ _R_down()


def _rot_err(R_target, tcp):
    """World-frame axis-angle error R_target * R_current^T."""
    return _R2aa(R_target @ _aa2R(tcp[3:6]).T)


def _clip(v, lim):
    return np.clip(np.asarray(v, dtype=float), -lim, lim)


# ---------------------------------------------------------------- perception
def _cloud(frame):
    K = np.asarray(getattr(frame, "agentview_K"), dtype=float)
    E = np.asarray(getattr(frame, "agentview_extrinsic"), dtype=float)
    d = np.asarray(getattr(frame, "agentview_depth"), dtype=float) / 1000.0
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1)
    W = (P - E[:, 3]) @ E[:, :3]
    return W[d > 0]


def _circfit(p):
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    b = (p[:, :2] ** 2).sum(1)
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(s[0]), float(s[1]), float(math.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0.0)))


def _find_bottle(W):
    m = (W[:, 2] > TABLE_Z + 0.015) & (W[:, 2] < 0.56) & (np.abs(W[:, 0]) < 0.15) & (np.abs(W[:, 1]) < 0.13)
    p = W[m]
    if len(p) < 30:
        return None, int(len(p))
    med = np.median(p[:, :2], 0)
    near = p[np.hypot(p[:, 0] - med[0], p[:, 1] - med[1]) < 0.07]
    if len(near) < 30:
        return None, int(len(near))
    c = near.mean(0)
    return [float(c[0]), float(c[1]), float(near[:, 2].max())], int(len(near))


def _find_pan(W):
    m = (W[:, 2] > TABLE_Z + 0.007) & (W[:, 2] < 0.5) & (W[:, 0] > -0.2) & (W[:, 0] < 0.2) & \
        (W[:, 1] > 0.09) & (W[:, 1] < 0.38)
    p = W[m]
    if len(p) < 40:
        return None, int(len(p))
    cx, cy, r = _circfit(p)
    q = p
    for _ in range(4):
        dd = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        q = p[dd < 0.015]
        if len(q) < 40:
            return None, int(len(q))
        cx, cy, r = _circfit(q)
    if not (0.07 < r < 0.12):
        return None, int(len(q))
    return [cx, cy, float(q[:, 2].max())], int(len(q))


def _find_drainer(W):
    m = (W[:, 2] > 0.56) & (W[:, 2] < 0.65) & (W[:, 0] > -0.45) & (W[:, 0] < 0.15) & \
        (W[:, 1] > 0.36) & (W[:, 1] < 0.85)
    p = W[m]
    if len(p) < 30:
        return None, int(len(p))
    lo = np.percentile(p, 3, 0)
    hi = np.percentile(p, 97, 0)
    # two compartments split along y; use the one nearer the frypan (lower y)
    return [float((lo[0] + hi[0]) / 2), float(lo[1] + 0.25 * (hi[1] - lo[1])), float(hi[2])], int(len(p))


def perceive_scene(obs):
    """Locate bottle, frypan and drainer (world metres) from the latest agentview depth frame."""
    try:
        W = _cloud(obs.current)
    except (AttributeError, ValueError, TypeError, IndexError) as e:
        return {"bottle": None, "frypan": None, "drainer": None, "points": [0, 0, 0], "error": str(e)[:80]}
    b, nb = _find_bottle(W)
    p, npn = _find_pan(W)
    d, nd = _find_drainer(W)
    return {"bottle": b, "frypan": p, "drainer": d, "points": [nb, npn, nd]}


# ---------------------------------------------------------------- control law (pure, testable)
def _command(kind, tcp, target, gain, rot_gain, grip):
    """One expert-style P-control command. kind: approach | carry | orient | hold."""
    tcp = np.asarray(tcp, dtype=float)
    a = np.zeros(7)
    a[6] = grip
    if kind in ("approach", "carry"):
        a[0:3] = _clip(gain * (np.asarray(target, dtype=float) - tcp[0:3]), 1.0)
        if kind == "approach":
            a[3:6] = _clip(rot_gain * _rot_err(_R_down(), tcp), 1.0)
    elif kind == "orient":
        R = _R_tilt() if target == "tilt" else _R_down()
        a[3:6] = _clip(rot_gain * _rot_err(R, tcp), 0.8)
    return a


def _run(robot, kind, target, gain, rot_gain, grip, stop_dist, max_steps):
    n = 0
    f = robot.now()
    while n < max_steps:
        tcp = np.asarray(f.tcp, dtype=float)
        if stop_dist > 0 and kind in ("approach", "carry") and \
                float(np.linalg.norm(np.asarray(target, dtype=float) - tcp[0:3])) < stop_dist:
            break
        a = _command(kind, tcp, target, gain, rot_gain, grip)
        f = robot.step(*[float(x) for x in a])
        n += 1
    tcp = np.asarray(f.tcp, dtype=float)
    return {"steps": n, "tcp": [round(float(x), 4) for x in tcp], "fingers": [round(float(x), 4) for x in f.fingers]}


# ---------------------------------------------------------------- act tools
def act_approach(robot, x, y, z, gain=6.0, rot_gain=4.0, stop_dist=0.02, max_steps=150):
    """Open gripper, P-servo TCP to (x,y,z) with gripper pointing down; stop when within stop_dist (0 = never) or after max_steps."""
    return _run(robot, "approach", [x, y, z], gain, rot_gain, -1.0, stop_dist, max_steps)


def act_carry(robot, x, y, z, gain=6.0, stop_dist=0.03, max_steps=200, grip=1.0):
    """P-servo TCP position to (x,y,z) without rotation commands, gripper sign grip; stop within stop_dist or after max_steps."""
    return _run(robot, "carry", [x, y, z], gain, 0.0, grip, stop_dist, max_steps)


def act_orient(robot, mode="tilt", steps=30, hold=0, rot_gain=2.0, grip=1.0):
    """Zero position command; servo orientation to 'tilt' (pour, 45 deg) or 'down' for `steps` steps, then send `hold` all-zero (still) steps."""
    r = _run(robot, "orient", mode, 0.0, rot_gain, grip, 0.0, steps)
    if hold > 0:
        h = _run(robot, "hold", None, 0.0, 0.0, grip, 0.0, hold)
        r = {"steps": r["steps"] + h["steps"], "tcp": h["tcp"], "fingers": h["fingers"]}
    return r


def act_gripper(robot, close=True, steps=30):
    """Hold still and close (close=True) or open the gripper for `steps` steps."""
    return _run(robot, "hold", None, 0.0, 0.0, 1.0 if close else -1.0, 0.0, steps)


# ---------------------------------------------------------------- decision
def _grasp_xy(mem):
    b = mem["bottle"] or DEFAULT_BOTTLE
    return [b[0] + GRASP_OFF[0], b[1] + GRASP_OFF[1]]


def _pour_xyz(mem):
    p = mem["frypan"] or DEFAULT_PAN
    return [p[0] + POUR_OFF[0], p[1] + POUR_OFF[1], POUR_Z]


def _holding(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(np.abs(f).mean()) > 0.01


def _plan(mem, frame):
    """Return (act, next_phase) for mem['phase']; may update mem."""
    ph = mem["phase"]
    g = _grasp_xy(mem)
    if ph == "hover":
        return {"tool": "act_approach", "args": {"x": g[0], "y": g[1], "z": HOVER_Z, "gain": 6.0,
                                                  "rot_gain": 4.0, "stop_dist": 0.02, "max_steps": 150}}, "descend"
    if ph == "descend":
        return {"tool": "act_approach", "args": {"x": g[0], "y": g[1], "z": DESCEND_Z, "gain": 2.5,
                                                  "rot_gain": 4.0, "stop_dist": 0.0, "max_steps": 100}}, "grasp"
    if ph == "grasp":
        return {"tool": "act_gripper", "args": {"close": True, "steps": 30}}, "lift"
    if ph == "lift":
        return {"tool": "act_carry", "args": {"x": g[0], "y": g[1], "z": LIFT_Z, "gain": 5.0,
                                               "stop_dist": 0.02, "max_steps": 150}}, "move"
    if ph == "move":
        t = _pour_xyz(mem)
        return {"tool": "act_carry", "args": {"x": t[0], "y": t[1], "z": t[2], "gain": 6.0,
                                               "stop_dist": 0.03, "max_steps": 200}}, "tilt"
    if ph == "tilt":
        # expert: 30 rotation steps + 30 still steps, then untilt; on the second pour success
        # arrived 16-17 still steps after the 30 rotation steps in every demo -> hold longer before fallback
        hold = 30 if mem["pours"] == 0 else 150
        return {"tool": "act_orient", "args": {"mode": "tilt", "steps": 30, "hold": hold, "rot_gain": 2.0}}, "untilt"
    if ph == "untilt":
        return {"tool": "act_orient", "args": {"mode": "down", "steps": 30, "rot_gain": 2.0}}, \
            ("move" if mem["pours"] < 2 else "to_drainer")
    d = mem["drainer"] or DEFAULT_DRAINER
    if ph == "to_drainer":
        return {"tool": "act_carry", "args": {"x": d[0], "y": d[1], "z": LIFT_Z, "gain": 5.0,
                                               "stop_dist": 0.01, "max_steps": 250}}, "lower"
    if ph == "lower":
        return {"tool": "act_carry", "args": {"x": d[0], "y": d[1], "z": max(d[2] - 0.08, 0.5),
                                               "gain": 3.0, "stop_dist": 0.01, "max_steps": 150}}, "release"
    if ph == "release":
        return {"tool": "act_gripper", "args": {"close": False, "steps": 25}}, "retreat"
    if ph == "retreat":
        return {"tool": "act_carry", "args": {"x": d[0], "y": d[1], "z": LIFT_Z, "gain": 5.0,
                                               "stop_dist": 0.01, "max_steps": 100, "grip": -1.0}}, "idle"
    # idle: wait for the checker's monitoring period
    return {"tool": "act_gripper", "args": {"close": False, "steps": 200}}, "idle"


def perceive_decide(obs, goal, memory, last):
    """Phase machine: perceive once, grasp bottle, pour over frypan twice, fallback drainer placement."""
    mem = {k: (memory or {}).get(k) for k in MEMORY_FIELDS}
    frame = obs.current
    log = []
    if mem["phase"] is None:
        s = perceive_scene(obs)
        mem["bottle"], mem["frypan"], mem["drainer"] = s["bottle"], s["frypan"], s["drainer"]
        mem["pours"], mem["grasp_tries"], mem["phase"] = 0, 0, "hover"
        log.append("scene " + str(s["points"]))
    ph = mem["phase"]
    # outcome-driven transitions
    if ph == "move" and mem["pours"] == 0 and not _holding(frame):
        # grasp failed after lift: open, re-perceive and retry (at most 2 retries)
        if mem["grasp_tries"] < 2:
            mem["grasp_tries"] = mem["grasp_tries"] + 1
            s = perceive_scene(obs)
            if s["bottle"] is not None:
                mem["bottle"] = s["bottle"]
            mem["phase"] = "reopen"
            log.append("grasp missed; retry")
        else:
            mem["phase"] = "idle"
            log.append("grasp failed")
    if mem["phase"] == "reopen":
        mem["phase"] = "hover"
        act = {"tool": "act_gripper", "args": {"close": False, "steps": 15}}
        mem["log"] = "; ".join(log) or (mem["log"] or "")
        return {"memory": mem, "act": act}
    if mem["phase"] == "untilt":
        mem["pours"] = int(mem["pours"]) + 1
    act, nxt = _plan(mem, frame)
    log.append(mem["phase"] + "->" + nxt)
    mem["phase"] = nxt
    mem["log"] = "; ".join(log)
    return {"memory": mem, "act": act}
