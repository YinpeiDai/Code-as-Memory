"""T04_ButterNonEmptyDrawer policy.

Stages: locate cabinet + butter (agentview RGB-D, frame 0) -> for drawer in
(top, middle, bottom): grasp handle, pull open, observe contents (agentview
point cloud inside the open drawer), push closed, release -> open the drawer
remembered as non-empty -> release handle -> pick butter -> carry above the open
drawer and lower it in -> release.
"""

LEVELS = ["top", "middle", "bottom"]
# Handle-centre heights relative to the measured cabinet top surface (expert
# grasp heights 0.680/0.606/0.532 with cabinet top 0.713 in all 25 demos).
LEVEL_DZ = {"top": -0.033, "middle": -0.107, "bottom": -0.181}
# Expert handle grasp TCP = (x_max(top surface) - 0.130, y_min(top surface) - 0.020)
HANDLE_DX = -0.130
HANDLE_DY = -0.021
PRIOR_HANDLE = [0.138, 0.448]
PRIOR_CAB_TOP = 0.713
PRIOR_BUTTER = [0.111, -0.202]
BUTTER_GRASP_OFFSET = [-0.004, 0.003]  # expert grasp - elevated-point median
TABLE_Z_PRIOR = 0.430
PULL_DIST = 0.20       # commanded pull; drawer stops after ~0.165 m
BACKOFF = 0.06
PLACE_REL = [-0.017, 0.104]   # place xy relative to the open handle (expert end pose)
TRAVERSE_Z = 0.93
RELEASE_DZ = 0.065     # release TCP height above the open handle centre (drop ~2-4 cm onto drawer contents)
PICK_RETRY_OFFSETS = [[0.0, 0.0], [0.01, 0.0], [-0.01, 0.0], [0.0, 0.01], [0.0, -0.01]]
ROT_DRAWER = [3.0 * math.pi / 2.0, 0.0, 0.0]   # gripper pointing +y (fingers close vertically)
ROT_DOWN = [math.pi, 0.0, 0.0]                 # top-down


# ----------------------------------------------------------------- math helpers
def _rv2m(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _m2rv(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0] - f[1]))


def _cloud(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    dep = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    H, W = dep.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([u, v, np.ones_like(u)], -1).astype(float) @ np.linalg.inv(K).T
    P = (rays * dep[..., None] - E[:, 3]) @ E[:, :3]
    return P, dep > 0


# ----------------------------------------------------------------- perception
def _locate_cabinet(frame):
    P, valid = _cloud(frame)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    reg = valid & (x > -0.15) & (x < 0.5) & (y > 0.2) & (y < 0.95) & (z > 0.62) & (z < 0.80)
    if int(reg.sum()) < 150:
        return None
    ztop = float(np.percentile(z[reg], 90))
    top = reg & (np.abs(z - ztop) < 0.012)
    if int(top.sum()) < 150:
        return None
    xs, ys = x[top], y[top]
    hx = float(np.percentile(xs, 98)) + HANDLE_DX
    hy = float(np.percentile(ys, 2)) + HANDLE_DY
    if abs(hx - PRIOR_HANDLE[0]) > 0.12 or abs(hy - PRIOR_HANDLE[1]) > 0.15:
        return None
    return {"hx": hx, "hy": hy, "ztop": ztop, "n": int(top.sum())}


def _locate_butter(frame, prior):
    P, valid = _cloud(frame)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    near = valid & (np.abs(x - prior[0]) < 0.08) & (np.abs(y - prior[1]) < 0.08)
    if int(near.sum()) < 50:
        return None
    tab = float(np.median(z[near]))
    obj = near & (z > tab + 0.006) & (z < tab + 0.10)
    if int(obj.sum()) < 15:
        return None
    bx = float(np.median(x[obj])) + BUTTER_GRASP_OFFSET[0]
    by = float(np.median(y[obj])) + BUTTER_GRASP_OFFSET[1]
    return {"x": bx, "y": by, "table": tab, "n": int(obj.sum())}


def _drawer_contents(frame, h):
    """Bright / raised points inside the open drawer box relative to the held handle."""
    P, valid = _cloud(frame)
    rel = P - np.asarray(h, dtype=float)
    box = (valid & (rel[..., 0] > -0.08) & (rel[..., 0] < 0.08) & (rel[..., 1] > 0.04)
           & (rel[..., 1] < 0.17) & (rel[..., 2] > -0.07) & (rel[..., 2] < 0.06))
    n = int(box.sum())
    if n < 20:
        return {"n": n, "bright": 0, "high": 0, "score": 0.0}
    zz = rel[..., 2][box]
    floor = float(np.percentile(zz, 15))
    high = int(((zz > floor + 0.008) & (zz < floor + 0.08)).sum())
    c = np.asarray(frame.agentview_rgb)[box].astype(int)
    bright = int((c.max(1) > 90).sum())
    return {"n": n, "bright": bright, "high": high, "score": float(bright + 0.2 * high)}


# ----------------------------------------------------------------- decision
def _num(v, d):
    return d if v is None else float(v)


def perceive_decide(obs, goal, memory, last):
    """Plan the next stage of: check top/middle/bottom drawers (open, look, close),
    then put the butter into the drawer that already contains an object."""
    m = dict(memory)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    phase = m.get("phase")
    if m.get("retries") is None:
        m["retries"] = 0
    if m.get("check_index") is None:
        m["check_index"] = 0
    if m.get("note") is None:
        m["note"] = ""
    for k in ("score_top", "score_middle", "score_bottom"):
        if k not in m:
            m[k] = None

    if phase is None:
        cab = _locate_cabinet(cur)
        if cab is None:
            m["handle_x"], m["handle_y"], m["cab_top"] = PRIOR_HANDLE[0], PRIOR_HANDLE[1], PRIOR_CAB_TOP
            m["note"] = "cabinet not found; prior used"
        else:
            m["handle_x"], m["handle_y"], m["cab_top"] = cab["hx"], cab["hy"], cab["ztop"]
            m["note"] = "cabinet n=%d" % cab["n"]
        b = _locate_butter(cur, PRIOR_BUTTER)
        m["butter"] = [PRIOR_BUTTER[0], PRIOR_BUTTER[1]] if b is None else [b["x"], b["y"]]
        m["table_z"] = TABLE_Z_PRIOR if b is None else b["table"]
        m["target"] = None
        m["open_handle"] = None
        m["check_index"] = 0
        m["retries"] = 0
        m["score_top"] = None
        m["score_middle"] = None
        m["score_bottom"] = None
        phase = "start"

    hx = _num(m.get("handle_x"), PRIOR_HANDLE[0])
    hy = _num(m.get("handle_y"), PRIOR_HANDLE[1])
    ztop = _num(m.get("cab_top"), PRIOR_CAB_TOP)
    idx = int(m["check_index"])

    def level_z(name):
        return ztop + LEVEL_DZ[name]

    def open_act(name):
        deeper = 0.006 * min(int(m["retries"]), 2)
        return {"tool": "act_open_drawer",
                "args": {"handle_x": hx, "handle_y": hy, "handle_z": level_z(name), "deeper": deeper}}

    act = None
    if phase == "start":
        phase = "check_open"
        act = open_act(LEVELS[idx])
    elif phase == "check_open":
        pulled = hy - float(tcp[1])
        held = _gap(cur) > 0.005
        name = LEVELS[idx]
        if pulled > 0.10 and held:
            c = _drawer_contents(cur, tcp[:3])
            m["score_" + name] = c["score"]
            m["note"] = "%s drawer: bright=%d high=%d n=%d" % (name, c["bright"], c["high"], c["n"])
            m["retries"] = 0
            phase = "check_close"
            act = {"tool": "act_close_drawer",
                   "args": {"handle_x": hx, "handle_y": hy, "handle_z": level_z(name)}}
        elif int(m["retries"]) < 1:
            m["retries"] = int(m["retries"]) + 1
            m["note"] = "%s open failed (pulled=%.3f held=%s); retry" % (name, pulled, held)
            act = open_act(name)
        else:
            m["note"] = "%s open failed; skipping" % name
            m["retries"] = 0
            phase = "check_close"
            act = {"tool": "act_close_drawer",
                   "args": {"handle_x": hx, "handle_y": hy, "handle_z": level_z(name)}}
    elif phase == "check_close":
        idx += 1
        m["check_index"] = idx
        if idx < 3:
            phase = "check_open"
            act = open_act(LEVELS[idx])
        else:
            scores = [m.get("score_" + n) for n in LEVELS]
            vals = [(-1.0 if s is None else float(s)) for s in scores]
            best = int(np.argmax(vals))
            # Evidence threshold: occupied drawers gave >= 29 bright points in all demos,
            # empty ones 0 bright and <= 26 raised (score <= ~5). Fallback: top (all demos).
            target = LEVELS[best] if vals[best] >= 8.0 else "top"
            m["target"] = target
            m["note"] = "scores=%s target=%s" % (vals, target)
            phase = "target_open"
            act = open_act(target)
    elif phase == "target_open":
        target = m.get("target") or "top"
        pulled = hy - float(tcp[1])
        held = _gap(cur) > 0.005
        if (pulled > 0.10 and held) or int(m["retries"]) >= 2:
            if pulled > 0.10:
                m["open_handle"] = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
            else:  # drawer did not open: nominal open-handle pose (expert pull 0.165-0.175 m)
                m["open_handle"] = [hx, hy - 0.168, level_z(target)]
            m["retries"] = 0
            phase = "release"
            act = {"tool": "act_release_handle", "args": {}}
        else:
            m["retries"] = int(m["retries"]) + 1
            m["note"] = "target open failed (pulled=%.3f); retry" % pulled
            act = open_act(target)
    elif phase in ("release", "pick"):
        if phase == "pick":
            holding = _gap(cur) > 0.01 and float(tcp[2]) > float(_num(m.get("table_z"), TABLE_Z_PRIOR)) + 0.15
            if holding:
                oh = m.get("open_handle") or [hx, hy - 0.165, level_z(m.get("target") or "top")]
                phase = "place"
                m["retries"] = 0
                act = {"tool": "act_place",
                       "args": {"x": float(oh[0]) + PLACE_REL[0], "y": float(oh[1]) + PLACE_REL[1],
                                "release_z": float(oh[2]) + RELEASE_DZ}}
            else:
                m["retries"] = int(m["retries"]) + 1
                prev = m.get("butter") or PRIOR_BUTTER
                b = _locate_butter(cur, prev)
                # accept a re-detection only if consistent (arm/gripper can contaminate it)
                if b is not None and b["n"] >= 30 and math.hypot(b["x"] - prev[0], b["y"] - prev[1]) < 0.04:
                    m["butter"] = [b["x"], b["y"]]
                m["note"] = "pick failed (gap=%.4f); retry %d" % (_gap(cur), m["retries"])
        if act is None:
            if int(m["retries"]) > 4:
                phase = "done"
                act = {"tool": "act_hold", "args": {"steps": 40}}
            else:
                phase = "pick"
                bt = m.get("butter") or PRIOR_BUTTER
                off = PICK_RETRY_OFFSETS[min(int(m["retries"]), len(PICK_RETRY_OFFSETS) - 1)]
                act = {"tool": "act_pick",
                       "args": {"x": float(bt[0]) + off[0], "y": float(bt[1]) + off[1],
                                "table_z": float(_num(m.get("table_z"), TABLE_Z_PRIOR))}}
    else:  # place finished or done: hold still and let the checker settle
        phase = "done"
        act = {"tool": "act_hold", "args": {"steps": 40}}

    m["phase"] = phase
    out = {}
    for k in FIELDS:
        out[k] = m.get(k)
    return {"memory": out, "act": act}


FIELDS = ["phase", "handle_x", "handle_y", "cab_top", "butter", "table_z", "check_index",
          "score_top", "score_middle", "score_bottom", "target", "open_handle", "retries", "note"]


# ----------------------------------------------------------------- control
def _step(robot, a, r, g):
    return robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(g))


def _goto(robot, target, rot, grip, tol=0.004, rtol=0.04, max_steps=200, kp=15.0, vmax=1.0, kr=1.0):
    target = np.asarray(target, dtype=float)
    Rt = _rv2m(rot)
    n = 0
    e = target - np.asarray(robot.now().tcp[:3], dtype=float)
    for n in range(max_steps):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        R = _rv2m(f.tcp[3:6])
        e = target - p
        er = _m2rv(Rt @ R.T)
        if np.linalg.norm(e) < tol and np.linalg.norm(er) < rtol:
            return {"reached": True, "steps": n, "err": float(np.linalg.norm(e))}
        a = kp * e
        mx = float(np.max(np.abs(a)))
        if mx > vmax:
            a = a * (vmax / mx)
        r = np.clip(kr * er, -0.5, 0.5)
        _step(robot, a, r, grip)
    return {"reached": False, "steps": n + 1, "err": float(np.linalg.norm(e))}


def _hold(robot, steps, grip):
    """Hold the current pose (small P-correction) while the gripper acts."""
    f0 = robot.now()
    p0 = np.asarray(f0.tcp[:3], dtype=float)
    R0 = _rv2m(f0.tcp[3:6])
    for _ in range(int(steps)):
        f = robot.now()
        e = p0 - np.asarray(f.tcp[:3], dtype=float)
        er = _m2rv(R0 @ _rv2m(f.tcp[3:6]).T)
        _step(robot, np.clip(10.0 * e, -0.3, 0.3), np.clip(er, -0.2, 0.2), grip)


def _push_axis(robot, target, rot, grip, max_steps=120, kp=12.0, vmin=0.25):
    """Move toward target along y while holding a handle; stop on stall."""
    target = np.asarray(target, dtype=float)
    Rt = _rv2m(rot)
    hist = []
    for n in range(max_steps):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        hist.append(float(p[1]))
        e = target - p
        if abs(e[1]) < 0.004:
            return {"stalled": False, "steps": n}
        if n > 15 and abs(hist[-1] - hist[-8]) < 0.0015:
            return {"stalled": True, "steps": n}
        a = kp * e
        a[0] = np.clip(a[0], -0.3, 0.3)
        a[2] = np.clip(a[2], -0.3, 0.3)
        ay = float(np.clip(a[1], -1.0, 1.0))
        if abs(ay) < vmin:
            ay = vmin if ay >= 0 else -vmin
        a[1] = ay
        er = _m2rv(Rt @ _rv2m(f.tcp[3:6]).T)
        _step(robot, a, np.clip(er, -0.3, 0.3), grip)
    return {"stalled": False, "steps": max_steps}


def act_open_drawer(robot, handle_x, handle_y, handle_z, deeper=0.0):
    """Release anything held, approach the drawer handle at height handle_z from the front
    (gripper pointing +y, `deeper` metres past the nominal grasp y), grasp it and pull the
    drawer open until it stops. Ends holding the open handle so the drawer interior is
    visible to the agentview camera. Aborts (gripper reopened) if the grasp closes on nothing."""
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=float)
    if _gap(f) < 0.07:
        _hold(robot, 12, -1.0)
    near_cab = p[1] > handle_y - 0.22 and abs(p[0] - handle_x) < 0.1
    if near_cab:
        q = robot.now().tcp[:3]
        _goto(robot, [q[0], min(float(q[1]) - 0.02, handle_y - BACKOFF), q[2]], ROT_DRAWER, -1.0, tol=0.008, max_steps=80)
    pre = [handle_x, handle_y - BACKOFF, handle_z]
    r1 = _goto(robot, pre, ROT_DRAWER, -1.0, tol=0.005, rtol=0.03, max_steps=250)
    r2 = _goto(robot, [handle_x, handle_y + deeper, handle_z], ROT_DRAWER, -1.0, tol=0.004, rtol=0.03, max_steps=120, vmax=0.6)
    _hold(robot, 15, 1.0)
    g = _gap(robot.now())
    if g < 0.005:
        _hold(robot, 10, -1.0)
        f = robot.now()
        return {"grasped": False, "pre_reached": r1["reached"], "handle_reached": r2["reached"],
                "grip_gap": g, "tcp": [float(v) for v in f.tcp[:3]]}
    y0 = float(robot.now().tcp[1])
    zc = float(robot.now().tcp[2])
    r3 = _push_axis(robot, [handle_x, handle_y - PULL_DIST, zc], ROT_DRAWER, 1.0, max_steps=110)
    _hold(robot, 4, 1.0)
    f = robot.now()
    return {"grasped": True, "pre_reached": r1["reached"], "handle_reached": r2["reached"], "grip_gap": g,
            "pulled": float(y0 - f.tcp[1]), "pull": r3, "tcp": [float(v) for v in f.tcp[:3]],
            "gap_now": _gap(f)}


def act_close_drawer(robot, handle_x, handle_y, handle_z):
    """Push the held drawer closed (to the closed handle y), release the handle and back off."""
    f = robot.now()
    zc = float(f.tcp[2])
    r = _push_axis(robot, [handle_x, handle_y + 0.005, zc], ROT_DRAWER, 1.0, max_steps=110)
    _hold(robot, 12, -1.0)
    q = robot.now().tcp[:3]
    _goto(robot, [q[0], float(q[1]) - BACKOFF, q[2]], ROT_DRAWER, -1.0, tol=0.008, max_steps=80)
    f = robot.now()
    return {"push": r, "closed_y": float(q[1]), "tcp": [float(v) for v in f.tcp[:3]]}


def act_release_handle(robot):
    """Open the gripper on the (open) drawer handle and back straight out and up, leaving the drawer open."""
    _hold(robot, 12, -1.0)
    q = np.asarray(robot.now().tcp[:3], dtype=float)
    _goto(robot, [q[0], q[1] - 0.08, q[2]], ROT_DRAWER, -1.0, tol=0.008, max_steps=80)
    _goto(robot, [q[0], q[1] - 0.08, max(0.80, q[2] + 0.12)], ROT_DRAWER, -1.0, tol=0.01, max_steps=80)
    f = robot.now()
    return {"tcp": [float(v) for v in f.tcp[:3]]}


def act_pick(robot, x, y, table_z):
    """Top-down grasp of the butter at (x, y) on the table and lift it to 0.78 m."""
    if _gap(robot.now()) < 0.07:
        _hold(robot, 10, -1.0)
    q = robot.now().tcp[:3]
    if float(q[2]) < 0.6:
        _goto(robot, [q[0], q[1], 0.66], ROT_DOWN, -1.0, tol=0.01, max_steps=80)
    r1 = _goto(robot, [x, y, 0.64], ROT_DOWN, -1.0, tol=0.005, rtol=0.03, max_steps=250)
    r2 = _goto(robot, [x, y, table_z + 0.007], ROT_DOWN, -1.0, tol=0.003, max_steps=150, vmax=0.6)
    _hold(robot, 15, 1.0)
    g = _gap(robot.now())
    q = robot.now().tcp[:3]
    r3 = _goto(robot, [q[0], q[1], 0.78], ROT_DOWN, 1.0, tol=0.01, max_steps=120)
    f = robot.now()
    return {"above": r1["reached"], "down": r2["reached"], "gap_after_close": g,
            "gap_now": _gap(f), "tcp": [float(v) for v in f.tcp[:3]]}


def act_place(robot, x, y, release_z):
    """Carry the held butter high over the cabinet front to (x, y), lower it into the open
    drawer to release_z, open the gripper and rise again."""
    q = robot.now().tcp[:3]
    _goto(robot, [0.5 * (float(q[0]) + x), y - 0.12, TRAVERSE_Z], ROT_DOWN, 1.0, tol=0.015, max_steps=200)
    _goto(robot, [x, y, TRAVERSE_Z], ROT_DOWN, 1.0, tol=0.006, max_steps=120)
    r = _goto(robot, [x, y, release_z], ROT_DOWN, 1.0, tol=0.005, max_steps=150, vmax=0.5)
    _hold(robot, 15, -1.0)
    _hold(robot, 15, -1.0)
    q = robot.now().tcp[:3]
    _goto(robot, [q[0], q[1], release_z + 0.12], ROT_DOWN, -1.0, tol=0.01, max_steps=60)
    f = robot.now()
    return {"lowered": r["reached"], "tcp": [float(v) for v in f.tcp[:3]], "gap": _gap(f)}


def act_hold(robot, steps=40):
    """Stay still with the gripper open for `steps` native steps."""
    _hold(robot, int(steps), -1.0)
    return {"held": int(steps)}
