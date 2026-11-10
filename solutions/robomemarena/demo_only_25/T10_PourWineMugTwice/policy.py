"""Pour wine into the mug twice (RoboMemArena T10_PourWineMugTwice).

Perception: agentview depth -> world points; the wine bottle is the tallest
thin object (neck top above 0.55 m), the mug the object whose rim is at
~0.51-0.545 m.  Control: closed-loop servo on measured TCP pose (OSC delta).
Memory holds the perceived object positions, the phase and the pour count.
"""

TABLE_Z = 0.43
# Canonical gripper orientations (world rotation matrices) taken from the expert.
R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]   # approach +y, bottle upright
R_POUR = [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]  # R_x(-90deg) @ R_SIDE, neck -> +y
# Offsets calibrated on the 25 demonstrations (tcp - perceived object), see task_analysis.md.
GRASP_OFF = [-0.0126, -0.0050]
GRASP_Z = 0.462
LIFT_Z = 0.75
PREPOUR_OFF = [-0.0289, -0.0742, 0.670]
POUR1_OFF = [-0.0125, -0.1177, 0.655]
MID_OFF = [-0.0852, 0.0004, 0.617]
PREPOUR2_OFF = [-0.0317, -0.0414, 0.657]
POUR2_OFF = [-0.0097, -0.1022, 0.638]
DEFAULT_BOTTLE = [-0.159, 0.238]
DEFAULT_MUG = [0.040, 0.261]
POUR_HOLD = 50
FINAL_HOLD = 160
MAX_POURS = 4


def _aa2mat(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _mat2aa(r):
    r = np.asarray(r, dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        m = (r + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(m)))
        v = m[:, i] / math.sqrt(max(m[i, i], 1e-12))
        return v * th
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _world_points(frame):
    """Agentview depth back-projected to world XYZ, shape (256,256,3), plus validity mask."""
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    k = np.asarray(frame.agentview_K, dtype=float)
    e = np.asarray(frame.agentview_extrinsic, dtype=float)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - k[0, 2]) / k[0, 0] * d
    y = (v - k[1, 2]) / k[1, 1] * d
    p = np.stack([x, y, d], -1)
    wp = (p - e[:, 3]) @ e[:, :3]
    return wp, d > 0


def _detect(frame):
    """Return dict with bottle_xy, mug_xy (or None) from one agentview frame."""
    wp, ok = _world_points(frame)
    tcp = np.asarray(frame.tcp, dtype=float)
    X, Y, Z = wp[..., 0], wp[..., 1], wp[..., 2]
    valid = ok & (np.abs(X) < 0.4) & (np.abs(Y) < 0.7) & (Z > TABLE_Z + 0.015) & (Z < 0.75)
    near = (np.hypot(X - tcp[0], Y - tcp[1]) < 0.12) & (Z > tcp[2] - 0.15)
    valid = valid & (~near)
    out = {"bottle_xy": None, "mug_xy": None, "n_bottle": 0, "n_mug": 0}
    seed = valid & (Z > 0.55)
    if int(seed.sum()) >= 3:
        s = np.array([np.median(X[seed]), np.median(Y[seed])])
        sel = valid & (Z > 0.50) & (np.hypot(X - s[0], Y - s[1]) < 0.04)
        if int(sel.sum()) >= 3:
            out["bottle_xy"] = [float(np.median(X[sel])), float(np.median(Y[sel]))]
            out["n_bottle"] = int(sel.sum())
    b = out["bottle_xy"]
    far_b = np.ones(X.shape, dtype=bool) if b is None else (np.hypot(X - b[0], Y - b[1]) > 0.05)
    mseed = valid & (Z > 0.512) & (Z < 0.545) & far_b
    if int(mseed.sum()) >= 5:
        s = np.array([np.median(X[mseed]), np.median(Y[mseed])])
        for _ in range(2):
            sel = mseed & (np.hypot(X - s[0], Y - s[1]) < 0.06)
            if int(sel.sum()) < 5:
                break
            s = np.array([np.median(X[sel]), np.median(Y[sel])])
        sel = valid & far_b & (np.hypot(X - s[0], Y - s[1]) < 0.11)
        if int(sel.sum()) >= 10:
            out["mug_xy"] = [float(np.median(X[sel])), float(np.median(Y[sel]))]
            out["n_mug"] = int(sel.sum())
    return out


def _targets(bottle, mug):
    gx, gy = bottle[0] + GRASP_OFF[0], bottle[1] + GRASP_OFF[1]
    def off(o):
        return [mug[0] + o[0], mug[1] + o[1], o[2]]
    return {
        "hover": [gx, gy - 0.02, 0.78],
        "above": [gx, gy, 0.62],
        "grasp": [gx, gy, GRASP_Z],
        "lift": [gx, gy, LIFT_Z],
        "prepour": off(PREPOUR_OFF),
        "pour1": off(POUR1_OFF),
        "mid": off(MID_OFF),
        "prepour2": off(PREPOUR2_OFF),
        "pour2": off(POUR2_OFF),
    }


def _servo_args(pos, rot, grip, max_steps, pos_tol=0.006, rot_tol=0.05, vmax=1.0, hold=0):
    return {"pos": [float(v) for v in pos], "rot": rot, "grip": float(grip),
            "max_steps": int(max_steps), "pos_tol": float(pos_tol), "rot_tol": float(rot_tol),
            "vmax": float(vmax), "hold": int(hold)}


def _get(memory, key, default):
    v = memory.get(key) if isinstance(memory, dict) else None
    return default if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Finite-state pouring program; memory: phase, bottle/mug positions, pours done, retries."""
    phase = _get(memory, "phase", "start")
    bottle = _get(memory, "bottle_xy", None)
    mug = _get(memory, "mug_xy", None)
    pours = int(_get(memory, "pours_done", 0))
    retries = int(_get(memory, "grasp_retries", 0))
    note = str(_get(memory, "note", ""))
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    fing = np.asarray(cur.fingers, dtype=float)
    gap = float(abs(fing[0]) + abs(fing[1]))

    if phase in ("start", "redetect"):
        det = _detect(cur)
        if det["bottle_xy"] is not None:
            bottle = det["bottle_xy"]
        if det["mug_xy"] is not None:
            mug = det["mug_xy"]
        if bottle is None:
            bottle = list(DEFAULT_BOTTLE)
        if mug is None:
            mug = list(DEFAULT_MUG)
        note = "det b=%s m=%s nb=%d nm=%d" % (str(det["bottle_xy"]), str(det["mug_xy"]), det["n_bottle"], det["n_mug"])
    tg = _targets(bottle, mug)
    cur_rot = _aa2mat(tcp[3:6]).tolist()
    nxt, act = phase, None

    if phase in ("start", "redetect"):
        act = _servo_args(tg["hover"], cur_rot, -1, 120, 0.01, 0.1)
        nxt = "rotate_side"
    elif phase == "rotate_side":
        act = _servo_args(tg["hover"], R_SIDE, -1, 120, 0.01, 0.04)
        nxt = "above"
    elif phase == "above":
        act = _servo_args(tg["above"], R_SIDE, -1, 120, 0.005, 0.04)
        nxt = "descend"
    elif phase == "descend":
        act = _servo_args(tg["grasp"], R_SIDE, -1, 150, 0.003, 0.04, vmax=0.5, hold=5)
        nxt = "close"
    elif phase == "close":
        act = _servo_args(tg["grasp"], R_SIDE, 1, 25, 0.0, 10.0, vmax=0.3, hold=20)
        nxt = "lift"
    elif phase == "lift":
        if gap < 0.008 and retries < 2:
            # nothing between the fingers: open, back off upward, re-detect
            act = _servo_args([tcp[0], tcp[1], 0.78], R_SIDE, -1, 120, 0.01, 0.1)
            retries += 1
            nxt = "back_home"
        else:
            act = _servo_args(tg["lift"], R_SIDE, 1, 150, 0.01, 0.05)
            nxt = "prepour"
    elif phase == "back_home":
        act = _servo_args([-0.058, 0.0, 0.671], [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], -1, 150, 0.01, 0.06)
        nxt = "redetect"
    elif phase == "prepour":
        act = _servo_args(tg["prepour"], R_SIDE, 1, 150, 0.006, 0.05)
        nxt = "pour"
    elif phase == "pour":
        p = tg["pour1"] if pours == 0 else tg["pour2"]
        hold = POUR_HOLD if pours == 0 else FINAL_HOLD
        act = _servo_args(p, R_POUR, 1, 150 + hold, 0.006, 0.05, hold=hold)
        pours += 1
        nxt = "unpour" if pours < MAX_POURS else "idle"
    elif phase == "unpour":
        act = _servo_args(tg["mid"], R_SIDE, 1, 120, 0.008, 0.06, hold=10)
        nxt = "prepour2"
    elif phase == "prepour2":
        act = _servo_args(tg["prepour2"], R_SIDE, 1, 120, 0.006, 0.05)
        nxt = "pour"
    else:
        act = _servo_args(tcp[:3].tolist(), cur_rot, 1, 200, 0.0, 0.0, hold=200)
        nxt = "idle"

    mem = {"phase": nxt, "bottle_xy": [float(v) for v in bottle], "mug_xy": [float(v) for v in mug],
           "pours_done": int(pours), "grasp_retries": int(retries), "note": note[:200]}
    return {"memory": mem, "act": {"tool": "act_servo", "args": act}}


def act_servo(robot, pos, rot, grip, max_steps, pos_tol=0.006, rot_tol=0.05, vmax=1.0, hold=0):
    """Closed-loop servo of the TCP to world position `pos` and orientation `rot` (3x3 world
    rotation matrix) with gripper command `grip` (+1 close, -1 open). Stops when within
    pos_tol (m) and rot_tol (rad) after first converging and then holding `hold` further
    steps, or after max_steps. Returns final pose errors, steps and finger gap."""
    target = np.asarray(pos, dtype=float)
    rt = np.asarray(rot, dtype=float)
    kp, kr = 12.0, 2.0
    steps, held, reached = 0, 0, False
    perr, rerr = 1.0, 1.0
    while steps < int(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        e = target - tcp[:3]
        w = _mat2aa(rt @ _aa2mat(tcp[3:6]).T)
        perr, rerr = float(np.linalg.norm(e)), float(np.linalg.norm(w))
        if perr <= pos_tol and rerr <= rot_tol:
            reached = True
        if reached:
            if held >= int(hold):
                break
            held += 1
        a = e * kp
        n = float(np.max(np.abs(a)))
        if n > vmax:
            a = a * (vmax / n)
        r = w * kr
        m = float(np.max(np.abs(r)))
        if m > 1.0:
            r = r / m
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]),
                   1.0 if grip > 0 else (-1.0 if grip < 0 else 0.0))
        steps += 1
    f = robot.now()
    fg = np.asarray(f.fingers, dtype=float)
    return {"steps": steps, "reached": reached, "held": held, "pos_err": perr, "rot_err": rerr,
            "tcp": [float(v) for v in np.asarray(f.tcp, dtype=float)],
            "finger_gap": float(abs(fg[0]) + abs(fg[1]))}
