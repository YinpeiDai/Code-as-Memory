"""Policy for: put cookies into the middle drawer, then put butter into the same drawer.

Plan (derived from 25 assigned expert demonstrations, see policy.md):
  1. perceive the cabinet's middle-drawer handle, the cookies box and the butter box
     in the first agentview RGB-D frame (world coordinates via calibration);
  2. grasp the middle handle with the gripper turned sideways, pull the drawer 0.24 m;
  3. pick cookies, carry high, lower over the open drawer, release;
  4. pick butter, carry, release into the same drawer.
Memory keeps the measured positions and the stage, plus retry counters.
"""

# ----------------------------------------------------------------- constants
TABLE_GRASP_Z = 0.433          # expert grasp TCP height for both boxes (table top ~0.430)
HANDLE_X_FROM_NEAR_END = -0.035  # grasp x = 98th pct x of visible middle-handle points + this
HANDLE_Y_OFFSET = 0.005        # grasp y = median handle y + this
HANDLE_Z_OFFSET = -0.001
PULL_DIST = 0.24               # expert pulled exactly 0.24 m in all 25 demos
BACKOFF = 0.09                 # extra -y retreat after releasing the handle
CARRY_Z = 0.86                 # transport height (cabinet top 0.714)
COOKIE_PRIOR = [0.216, -0.328]
BUTTER_PRIOR = [0.110, -0.202]
COOKIE_OFF = [-0.013, 0.000]   # grasp - refined median of detected points (demo mean)
BUTTER_OFF = [-0.001, 0.002]
COOKIE_DROP = [-0.035, -0.035, 0.135]  # relative to handle grasp point (demo mean)
BUTTER_DROP = [-0.002, -0.045, 0.120]
SAFE_POSE = [-0.05, 0.0, 0.75]
HOLD_MIN = 0.008               # finger opening sum (m) above which an object is held
HOLD_MAX = 0.074               # ... and below which (open gripper ~0.078; cookies 0.062, butter 0.039)


# ----------------------------------------------------------------- geometry
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _mat2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _target_rot(name):
    down = np.diag([1.0, -1.0, -1.0])            # gripper pointing -z, fingers along world y
    if name == "side":                           # rotated +90 deg about world x: points +y
        rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        return rx @ down
    return down


def _backproject(frame):
    depth = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]               # R^T (pc - t)
    return pw, depth > 0.05


# ----------------------------------------------------------------- perception
def _gray_mask(rgb):
    rgb = np.asarray(rgb).astype(int)
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    return (mx - mn < 15) & (mx > 60) & (mx < 150)


def _detect_handle(frame):
    """Middle-drawer handle grasp point from gray metallic points at middle-drawer height."""
    pw, valid = _backproject(frame)
    gray = _gray_mask(frame.agentview_rgb) & valid
    reg = ((pw[..., 0] > -0.05) & (pw[..., 0] < 0.40) & (pw[..., 1] > 0.28)
           & (pw[..., 1] < 0.65) & (pw[..., 2] > 0.58) & (pw[..., 2] < 0.64))
    P = pw[gray & reg]
    if len(P) < 8:
        return None
    # keep the dominant y cluster (handle bar) to reject stray gray pixels
    ymed = float(np.median(P[:, 1]))
    P = P[np.abs(P[:, 1] - ymed) < 0.03]
    if len(P) < 8:
        return None
    x98 = float(np.percentile(P[:, 0], 98))
    pos = [x98 + HANDLE_X_FROM_NEAR_END, float(np.median(P[:, 1])) + HANDLE_Y_OFFSET,
           float(np.median(P[:, 2])) + HANDLE_Z_OFFSET]
    return {"pos": pos, "n": int(len(P))}


def _handle_points_near(frame, handle):
    """Number of gray handle points still at the closed-drawer handle location."""
    pw, valid = _backproject(frame)
    gray = _gray_mask(frame.agentview_rgb) & valid
    reg = ((np.abs(pw[..., 0] - handle[0] + 0.02) < 0.09) & (np.abs(pw[..., 1] - handle[1]) < 0.025)
           & (pw[..., 2] > 0.58) & (pw[..., 2] < 0.64))
    return int(np.sum(gray & reg))


def _detect_box(frame, prior, radius, off):
    """Box on the table: points at table height whose colour differs from the table."""
    pw, valid = _backproject(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(float)
    onz = valid & (pw[..., 2] > 0.425) & (pw[..., 2] < 0.50)
    tab = onz & (pw[..., 2] < 0.436)
    if np.sum(tab) < 50:
        return None
    tc = np.median(rgb[tab], 0)
    diff = np.abs(rgb - tc).sum(-1)
    win = onz & (np.hypot(pw[..., 0] - prior[0], pw[..., 1] - prior[1]) < radius) & (diff > 60)
    P = pw[win]
    if len(P) < 10:
        return None
    med = np.median(P, 0)
    # refine around the median to drop neighbouring clutter
    P2 = P[np.hypot(P[:, 0] - med[0], P[:, 1] - med[1]) < 0.05]
    if len(P2) >= 10:
        med = np.median(P2, 0)
    return {"pos": [float(med[0]) + off[0], float(med[1]) + off[1], TABLE_GRASP_Z],
            "n": int(len(P))}


def _held(frame):
    s = _fingers_sum(frame)
    return bool(HOLD_MIN < s < HOLD_MAX)


def _fingers_sum(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


# ----------------------------------------------------------------- decision
def _num(v, d):
    return d if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Stage machine for the cookies-then-butter middle-drawer task.

    Reads the latest frame, updates memory (stage, measured positions, retry counts)
    and chooses the next act tool with explicit waypoints."""
    cur = obs.current
    mem = {
        "stage": memory.get("stage") or "start",
        "handle": memory.get("handle"),
        "cookies": memory.get("cookies"),
        "butter": memory.get("butter"),
        "open_tries": int(_num(memory.get("open_tries"), 0)),
        "pick_tries": int(_num(memory.get("pick_tries"), 0)),
        "decisions": int(_num(memory.get("decisions"), 0)) + 1,
        "placed": list(memory.get("placed") or []),
        "note": "",
    }
    st = mem["stage"]
    tcp = np.asarray(cur.tcp, dtype=float)

    if st == "start":
        h = _detect_handle(cur)
        c = _detect_box(cur, COOKIE_PRIOR, 0.06, COOKIE_OFF)
        b = _detect_box(cur, BUTTER_PRIOR, 0.06, BUTTER_OFF)
        mem["handle"] = h["pos"] if h else [0.135, 0.447, 0.607]
        mem["cookies"] = c["pos"] if c else [COOKIE_PRIOR[0] + COOKIE_OFF[0], COOKIE_PRIOR[1], TABLE_GRASP_Z]
        mem["butter"] = b["pos"] if b else [BUTTER_PRIOR[0] + BUTTER_OFF[0], BUTTER_PRIOR[1] + BUTTER_OFF[1], TABLE_GRASP_Z]
        mem["note"] = "detected handle=%s cookies=%s butter=%s" % (bool(h), bool(c), bool(b))
        st = "open"

    if st == "check_open":
        n = _handle_points_near(cur, mem["handle"])
        if n >= 10 and mem["open_tries"] < 3:
            h = _detect_handle(cur)
            if h:
                mem["handle"] = h["pos"]
            mem["note"] = "drawer still closed (%d handle pts); retry" % n
            st = "open"
        else:
            mem["note"] = "drawer open check pts=%d" % n
            st = "pick_cookies" if "cookies" not in mem["placed"] else "pick_butter"
            mem["pick_tries"] = 0

    if st in ("check_cookies", "check_butter"):
        obj = "cookies" if st == "check_cookies" else "butter"
        held = _held(cur)
        if held:
            st = "carry_" + obj
        else:
            mem["note"] = obj + " grasp missed; re-observe"
            st = "reobserve_" + obj

    if st in ("after_carry_cookies", "after_carry_butter"):
        obj = "cookies" if st == "after_carry_cookies" else "butter"
        ok = isinstance(last, str) and '"held_before_release": true' in last
        if ok:
            if obj not in mem["placed"]:
                mem["placed"].append(obj)
            todo = [o for o in ("cookies", "butter") if o not in mem["placed"]]
            st = ("pick_" + todo[0]) if todo else "wait"
            mem["pick_tries"] = 0
        else:
            mem["note"] = obj + " lost during transport; re-observe"
            st = "reobserve_" + obj

    if st in ("detect_cookies", "detect_butter"):
        obj = "cookies" if st == "detect_cookies" else "butter"
        prior = mem[obj]
        off = COOKIE_OFF if obj == "cookies" else BUTTER_OFF
        d = _detect_box(cur, [prior[0] - off[0], prior[1] - off[1]], 0.09, off)
        if d:
            mem[obj] = d["pos"]
        mem["note"] = "re-detected %s: %s" % (obj, bool(d))
        st = "pick_" + obj

    h = mem["handle"]
    if st == "open":
        mem["open_tries"] += 1
        pre = [h[0], h[1] - 0.10, h[2]]
        # on a retry reach slightly deeper (the first try may have stopped short of the bar)
        grasp = [h[0], h[1] + 0.005 * (mem["open_tries"] - 1), h[2]]
        pulled = [h[0], h[1] - PULL_DIST, h[2]]
        back = [h[0], h[1] - PULL_DIST - BACKOFF, h[2]]
        plan = [
            {"op": "move", "pos": pre, "rot": "side", "grip": -1, "vmax": 1.0, "tol": 0.006},
            {"op": "move", "pos": grasp, "rot": "side", "grip": -1, "vmax": 0.4, "tol": 0.003, "stall": True},
            {"op": "grip", "grip": 1, "n": 15},
            {"op": "move", "pos": pulled, "rot": "side", "grip": 1, "vmax": 0.5, "tol": 0.005, "stall": True},
            {"op": "grip", "grip": -1, "n": 14},
            {"op": "move", "pos": back, "rot": "side", "grip": -1, "vmax": 0.6, "tol": 0.01},
        ]
        mem["stage"] = "check_open"
        return {"memory": mem, "act": {"tool": "act_plan", "args": {"plan": plan}}}

    if st in ("pick_cookies", "pick_butter"):
        obj = "cookies" if st == "pick_cookies" else "butter"
        if mem["pick_tries"] >= 4:
            mem["note"] = "too many pick attempts for " + obj
            mem["stage"] = "verify"
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"n": 50, "grip": -1}}}
        return _pick_plan(mem, obj, tcp)

    if st in ("carry_cookies", "carry_butter"):
        obj = "cookies" if st == "carry_cookies" else "butter"
        off = COOKIE_DROP if obj == "cookies" else BUTTER_DROP
        drop = [h[0] + off[0], h[1] + off[1], h[2] + off[2]]
        plan = [
            {"op": "move", "pos": [float(tcp[0]), float(tcp[1]), CARRY_Z], "rot": "down", "grip": 1, "vmax": 0.8, "tol": 0.015},
            {"op": "move", "pos": [drop[0], drop[1], CARRY_Z], "rot": "down", "grip": 1, "vmax": 0.8, "tol": 0.008},
            {"op": "move", "pos": drop, "rot": "down", "grip": 1, "vmax": 0.5, "tol": 0.005, "stall": True},
            {"op": "check_held"},
            {"op": "grip", "grip": -1, "n": 20},
            {"op": "move", "pos": [drop[0], drop[1], CARRY_Z], "rot": "down", "grip": -1, "vmax": 0.8, "tol": 0.015},
        ]
        mem["stage"] = "after_carry_" + obj
        return {"memory": mem, "act": {"tool": "act_plan", "args": {"plan": plan}}}

    if st in ("reobserve_cookies", "reobserve_butter"):
        obj = "cookies" if st == "reobserve_cookies" else "butter"
        plan = [
            {"op": "grip", "grip": -1, "n": 10},
            {"op": "move", "pos": [float(tcp[0]), float(tcp[1]), max(float(tcp[2]), 0.75)], "rot": "down", "grip": -1, "vmax": 0.8, "tol": 0.02, "maxn": 80},
            {"op": "move", "pos": SAFE_POSE, "rot": "down", "grip": -1, "vmax": 1.0, "tol": 0.02},
        ]
        mem["stage"] = "detect_" + obj
        return {"memory": mem, "act": {"tool": "act_plan", "args": {"plan": plan}}}

    if st == "verify":
        # the episode did not end after both releases: look for a box left on the table
        for obj, off in (("cookies", COOKIE_OFF), ("butter", BUTTER_OFF)):
            p = mem[obj]
            d = _detect_box(cur, [p[0] - off[0], p[1] - off[1]], 0.05, off)
            if d and d["n"] >= 15 and mem["pick_tries"] < 4:
                mem[obj] = d["pos"]
                mem["placed"] = [o for o in mem["placed"] if o != obj]
                mem["note"] = obj + " still on the table; pick again"
                mem["stage"] = "check_" + obj
                return _pick_plan(mem, obj, tcp)
        mem["note"] = "nothing left on the table; holding"

    # wait / fallback: hold still (task success is scored by the simulator)
    mem["stage"] = "verify"
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"n": 40, "grip": -1}}}


def _pick_plan(mem, obj, tcp):
    mem["pick_tries"] += 1
    p = mem[obj]
    above_z = max(0.55, min(float(tcp[2]), CARRY_Z))
    plan = [
        {"op": "move", "pos": [float(tcp[0]), float(tcp[1]), above_z], "rot": "down", "grip": -1, "vmax": 1.0, "tol": 0.02, "maxn": 60},
        {"op": "move", "pos": [p[0], p[1], 0.55], "rot": "down", "grip": -1, "vmax": 1.0, "tol": 0.006},
        {"op": "move", "pos": [p[0], p[1], p[2]], "rot": "down", "grip": -1, "vmax": 0.5, "tol": 0.003, "stall": True},
        {"op": "grip", "grip": 1, "n": 18},
        {"op": "move", "pos": [p[0], p[1], 0.60], "rot": "down", "grip": 1, "vmax": 0.6, "tol": 0.01},
    ]
    mem["stage"] = "check_" + obj
    return {"memory": mem, "act": {"tool": "act_plan", "args": {"plan": plan}}}


# ----------------------------------------------------------------- control
def _control(frame, pos, rot, vmax):
    tcp = np.asarray(frame.tcp, dtype=float)
    perr = np.asarray(pos, dtype=float) - tcp[:3]
    a = np.clip(perr / 0.05, -vmax, vmax)
    Rc = _aa2mat(tcp[3:6])
    rerr = _mat2aa(_target_rot(rot) @ Rc.T)
    r = np.clip(rerr / 0.5, -0.8, 0.8)
    return a, r, float(np.linalg.norm(perr)), float(np.linalg.norm(rerr))


def _move(robot, step, budget):
    pos, rot, grip = step["pos"], step.get("rot", "down"), float(step.get("grip", -1))
    vmax = float(step.get("vmax", 1.0))
    tol = float(step.get("tol", 0.005))
    maxn = int(step.get("maxn", 250))
    stall = bool(step.get("stall", False))
    n = 0
    hist = []
    pe, re = 1.0, 1.0
    while n < maxn and n < budget:
        a, r, pe, re = _control(robot.now(), pos, rot, vmax)
        if pe < tol and re < 0.05:
            break
        hist.append(pe + 0.05 * re)
        if stall and len(hist) > 25 and hist[-25] - hist[-1] < 0.002:
            break
        if len(hist) > 30 and hist[-30] - hist[-1] < 0.001 and pe < max(3.0 * tol, 0.01):
            break
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)
        n += 1
    return n, pe, re


def _grip(robot, grip, count, budget):
    f = robot.now()
    hold = np.asarray(f.tcp, dtype=float)[:3].tolist()
    rot_now = np.asarray(f.tcp, dtype=float)[3:6]
    n = 0
    while n < count and n < budget:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a = np.clip((np.asarray(hold) - tcp[:3]) / 0.05, -0.3, 0.3)
        rerr = _mat2aa(_aa2mat(rot_now) @ _aa2mat(tcp[3:6]).T)
        r = np.clip(rerr / 0.5, -0.5, 0.5)
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], float(grip))
        n += 1
    return n


def act_plan(robot, plan):
    """Execute a list of plan steps. Each step is a dict with "op":
    "move" (pos [x,y,z] world m, rot "down"|"side", grip -1|1, vmax, tol, maxn, stall),
    "grip" (grip -1 open | 1 close, n steps while holding pose) or "check_held".
    Returns step counts, final errors, tcp, fingers and whether an object was held before release."""
    budget = 2400
    used = 0
    log = []
    held_before = None
    for s in plan:
        op = s.get("op")
        if op == "move":
            n, pe, re = _move(robot, s, budget - used)
            used += n
            log.append([op, n, round(pe, 4), round(re, 3)])
        elif op == "grip":
            n = _grip(robot, s.get("grip", -1), int(s.get("n", 12)), budget - used)
            used += n
            log.append([op, n])
        elif op == "check_held":
            held_before = _held(robot.now())
            log.append([op, held_before])
    f = robot.now()
    return {"held_before_release": held_before, "steps": used, "tcp": np.asarray(f.tcp)[:3].tolist(),
            "fingers": np.asarray(f.fingers).tolist(), "log": log}


def act_hold(robot, n=40, grip=-1):
    """Hold the current pose for n steps with the given gripper command (-1 open, 1 close)."""
    used = _grip(robot, grip, int(n), 2400)
    f = robot.now()
    return {"steps": used, "tcp": np.asarray(f.tcp)[:3].tolist(), "fingers": np.asarray(f.fingers).tolist()}
