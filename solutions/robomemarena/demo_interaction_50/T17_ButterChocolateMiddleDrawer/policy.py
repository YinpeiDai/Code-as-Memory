"""Policy for T17: open the middle drawer, put butter in it, then put chocolate in the same drawer.

All geometry is in world metres.  Perception uses agentview / wrist RGB-D back-projected
with the supplied calibration.  Constants marked DEMO were fitted on the 50 assigned demos
(see task_analysis.md)."""

PI = 3.141592653589793
# DEMO fits: handle grasp TCP = (cab_xmax - 0.127, cab_yface - 0.018, 0.607)
HANDLE_DX = -0.127
HANDLE_DY = -0.018
HANDLE_Z = 0.607
PULL = 0.24
# DEMO: release points relative to the (closed) handle grasp point
# DEMO release points were (-0.003,-0.096,+0.119) butter, (-0.043,-0.041,+0.129) chocolate; they put the
# arm within ~6 mm of its maximum reach (R002-R004), so both are moved ~2.5 cm toward the robot and lowered
# (still inside the opened part of the drawer, side by side in y; footprints 7x3.6 / 7.7x4.5 cm).
BUTTER_DROP = [-0.020, -0.120, 0.103]
CHOC_DROP = [-0.050, -0.070, 0.108]
# reach model (fitted with Panda FK on demo/rollout joints): shoulder at world (-0.51,0,0.748),
# wrist centre ~ TCP + (-0.075,-0.05,0.208) when top-down; singular at ~0.719 m
SHOULDER = [-0.51, -0.002, 0.748]
WRIST_OFF = [-0.075, -0.05, 0.208]
REACH_MAX = 0.69
# DEMO priors (wrist top-view centroids) and grasp offsets (grasp - centroid)
BUTTER_PRIOR = [0.110, -0.205]
CHOC_PRIOR = [-0.218, -0.330]
BUTTER_OFF = [-0.001, 0.003]
CHOC_OFF = [-0.009, 0.005]
BUTTER_DZ = 0.002   # grasp z above table
CHOC_DZ = 0.005
TABLE_Z = 0.431
HOVER_Z = 0.60
TRAVEL_Z = 0.80
APPROACH_BACK = 0.12  # place approach starts this far in -y of the drop point (toward the robot)
HANDLE_AA = [4.712389, 0.0, 0.0]
TOPDOWN_AA = [PI, 0.0, 0.0]
DEFAULT_HANDLE = [0.137, 0.452, 0.607]
DEFAULT_CAB = [0.264, 0.470, 0.714]


# ---------------------------------------------------------------- geometry
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > PI - 1e-3:
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        ax = np.zeros(3)
        ax[i] = d[i]
        for j in range(3):
            if j != i:
                ax[j] = (R[i, j] + R[j, i]) / (4.0 * d[i])
        return ax / np.linalg.norm(ax) * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return ax * th


def _cloud(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    W_ = (P - t) @ R
    valid = depth > 0
    return W_, valid


def _agent_cloud(f):
    return _cloud(f.agentview_depth, f.agentview_K, f.agentview_extrinsic)


def _wrist_cloud(f):
    return _cloud(f.wrist_depth, f.wrist_K, f.wrist_extrinsic)


def _gap(f):
    return float(f.fingers[0] - f.fingers[1])


# ---------------------------------------------------------------- perception
def _cabinet(f):
    """Dark cabinet right of the robot: returns [xmax, yface, n] or None."""
    P, valid = _agent_cloud(f)
    rgb = f.agentview_rgb.astype(int)
    dark = rgb.max(-1) < 60
    m = valid & dark & (P[..., 1] > 0.15) & (P[..., 2] > 0.45) & (P[..., 2] < 0.85) \
        & (P[..., 0] > -0.2) & (P[..., 0] < 0.45)
    Q = P[m]
    if len(Q) < 200:
        return None
    xmax = float(np.percentile(Q[:, 0], 98))
    b = Q[(Q[:, 0] > xmax - 0.08) & (Q[:, 0] < xmax - 0.04) & (Q[:, 2] > 0.48) & (Q[:, 2] < 0.70)]
    if len(b) < 30:
        return None
    yface = float(np.percentile(b[:, 1], 5))
    return [xmax, yface, int(len(Q))]


def _drawer_open_count(f, cab):
    """Dark points in front of the cabinet face at middle-drawer height (open drawer walls)."""
    P, valid = _agent_cloud(f)
    rgb = f.agentview_rgb.astype(int)
    dark = rgb.max(-1) < 60
    xmax, yface = cab[0], cab[1]
    m = valid & dark & (P[..., 0] > xmax - 0.12) & (P[..., 0] < xmax - 0.01) \
        & (P[..., 2] > 0.50) & (P[..., 2] < 0.66) \
        & (P[..., 1] > yface - 0.18) & (P[..., 1] < yface - 0.05)
    return int(m.sum())


def _agent_object(f, prior, r=0.06):
    """Centroid of above-table points near prior in the agentview cloud (front-face biased)."""
    P, valid = _agent_cloud(f)
    m = valid & (P[..., 2] > TABLE_Z + 0.006) & (P[..., 2] < 0.60) \
        & (np.abs(P[..., 0] - prior[0]) < r) & (np.abs(P[..., 1] - prior[1]) < r)
    Q = P[m]
    if len(Q) < 15:
        return None
    c = np.median(Q[:, :2], 0)
    # agentview sees the camera-facing side; shift toward the far side (DEMO: ~5 mm)
    return [float(c[0]) - 0.005, float(c[1]), int(len(Q))]


def _wrist_object(f, est, r=0.06):
    """Top-view centroid of the object under the hovering wrist camera, plus table z."""
    P, valid = _wrist_cloud(f)
    ring = valid & (np.abs(P[..., 0] - est[0]) < 0.12) & (np.abs(P[..., 1] - est[1]) < 0.12) \
        & (P[..., 2] > 0.40) & (P[..., 2] < 0.46)
    tz = TABLE_Z
    if ring.sum() > 200:
        zs = P[ring][:, 2]
        tz = float(np.median(zs))
        if abs(tz - TABLE_Z) > 0.01:
            tz = TABLE_Z
    c = [est[0], est[1]]
    n = 0
    for rr in (r, 0.045):
        m = valid & (P[..., 2] > tz + 0.006) & (P[..., 2] < tz + 0.10) \
            & (np.abs(P[..., 0] - c[0]) < rr) & (np.abs(P[..., 1] - c[1]) < rr)
        Q = P[m]
        n = int(len(Q))
        if n < 40:
            return None, tz, n
        c = [float(np.mean(Q[:, 0])), float(np.mean(Q[:, 1]))]
    return c, tz, n


def _reach(p):
    return float(np.linalg.norm(np.asarray(p, dtype=float) + np.asarray(WRIST_OFF) - np.asarray(SHOULDER))) + 0.008


def _safe_drop(p):
    """Shift a drop point toward the robot (-y, then -x) until the reach model is below REACH_MAX (<=3 cm)."""
    q = [float(p[0]), float(p[1]), float(p[2])]
    for i in range(6):
        if _reach(q) <= REACH_MAX:
            break
        q[1] -= 0.004
        q[0] -= 0.001
    return q


# ---------------------------------------------------------------- control
def _servo(robot, pos, aa, grip, tol=0.004, rtol=0.04, max_steps=200, vmax=1.0, kp=25.0, kr=2.5,
           watch_gap=None):
    """Closed-loop OSC delta servo to a world pose. Returns measured final state."""
    pos = np.asarray(pos, dtype=float)
    Rt = _aa2mat(aa)
    n = 0
    reached = False
    lost = False
    stalled = False
    hist = []
    for n in range(max_steps):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        hist.append(p)
        if len(hist) > 20 and np.linalg.norm(hist[-1] - hist[-21]) < 0.002 and np.linalg.norm(pos - p) > 2 * tol:
            stalled = True
            break
        e = pos - p
        er = _mat2aa(Rt @ _aa2mat(f.tcp[3:]).T)
        if np.linalg.norm(e) < tol and np.linalg.norm(er) < rtol:
            reached = True
            break
        if watch_gap is not None and _gap(f) < watch_gap:
            lost = True
            break
        a = e * kp
        mx = float(np.max(np.abs(a)))
        if mx > vmax:
            a = a * (vmax / mx)
        ar = np.clip(er * kr, -1.0, 1.0)
        robot.step(a[0], a[1], a[2], ar[0], ar[1], ar[2], grip)
    f = robot.now()
    return {"reached": reached, "lost": lost, "stalled": stalled, "steps": n,
            "j4": round(float(f.joints[3]), 3),
            "err": float(np.linalg.norm(pos - np.asarray(f.tcp[:3]))),
            "tcp": [round(float(v), 4) for v in f.tcp[:3]], "gap": round(_gap(f), 4)}


def _grip(robot, grip, steps, aa):
    """Hold the current pose while opening (-1) or closing (+1) the gripper."""
    f0 = robot.now()
    hold = np.asarray(f0.tcp[:3], dtype=float)
    Rt = _aa2mat(aa)
    prev = _gap(f0)
    still = 0
    for i in range(steps):
        f = robot.now()
        e = (hold - np.asarray(f.tcp[:3])) * 25.0
        e = np.clip(e, -0.5, 0.5)
        er = np.clip(_mat2aa(Rt @ _aa2mat(f.tcp[3:]).T) * 2.5, -1, 1)
        robot.step(e[0], e[1], e[2], er[0], er[1], er[2], grip)
        g = _gap(robot.now())
        still = still + 1 if abs(g - prev) < 0.0003 else 0
        prev = g
        if i >= 6 and still >= 4:
            break
    return round(_gap(robot.now()), 4)


# ---------------------------------------------------------------- act tools
def act_open_drawer(robot, handle, pull=0.24, park=None):
    """Grasp the drawer handle at world [x,y,z] (gripper pointing +y), pull it -y by `pull` m,
    release, back off, rise and park top-down at `park` [x,y,z]. Returns measured TCP at each stage."""
    hx, hy, hz = float(handle[0]), float(handle[1]), float(handle[2])
    out = {}
    out["pre"] = _servo(robot, [hx, hy - 0.10, hz + 0.01], HANDLE_AA, -1, tol=0.012, rtol=0.08, max_steps=160)
    out["pre2"] = _servo(robot, [hx, hy - 0.10, hz], HANDLE_AA, -1, tol=0.004, rtol=0.03, max_steps=80)
    out["in"] = _servo(robot, [hx, hy, hz], HANDLE_AA, -1, tol=0.003, rtol=0.03, max_steps=90, vmax=0.5)
    out["close_gap"] = _grip(robot, 1, 25, HANDLE_AA)
    out["pull"] = _servo(robot, [hx, hy - pull, hz], HANDLE_AA, 1, tol=0.005, rtol=0.05, max_steps=160, vmax=0.6)
    out["open_gap"] = _grip(robot, -1, 20, HANDLE_AA)
    out["back"] = _servo(robot, [hx, hy - pull - 0.11, hz], HANDLE_AA, -1, tol=0.01, rtol=0.08, max_steps=80)
    out["up"] = _servo(robot, [hx, hy - pull - 0.11, 0.78], HANDLE_AA, -1, tol=0.02, rtol=0.2, max_steps=80)
    if park is not None:
        out["park"] = _servo(robot, park, TOPDOWN_AA, -1, tol=0.005, rtol=0.03, max_steps=220)
    return out


def act_move(robot, pos, gripper=-1.0, max_steps=250, tol=0.005):
    """Move the TCP top-down to world [x,y,z] holding the gripper command; returns measured pose."""
    return _servo(robot, pos, TOPDOWN_AA, gripper, tol=tol, rtol=0.03, max_steps=max_steps)


def act_pick(robot, xy, z, lift_z=0.75):
    """Top-down pick: from hover descend to (x,y,z), close, and lift to lift_z. Returns gaps."""
    x, y = float(xy[0]), float(xy[1])
    out = {}
    cur = robot.now().tcp
    out["hover"] = _servo(robot, [x, y, max(HOVER_Z, float(cur[2]))], TOPDOWN_AA, -1, tol=0.004, rtol=0.03,
                          max_steps=150)
    out["descend"] = _servo(robot, [x, y, z], TOPDOWN_AA, -1, tol=0.003, rtol=0.03, max_steps=150, vmax=0.7)
    out["close_gap"] = _grip(robot, 1, 30, TOPDOWN_AA)
    out["lift"] = _servo(robot, [x, y, lift_z], TOPDOWN_AA, 1, tol=0.01, rtol=0.05, max_steps=120, vmax=0.8)
    out["held_gap"] = round(_gap(robot.now()), 4)
    return out


def act_place(robot, drop, travel_z=0.80, park=None):
    """Carry the held object at travel_z to a point APPROACH_BACK m in -y of drop [x,y,z], move
    diagonally down to drop (keeps the arm well inside its reach), verify it is still held, open,
    back out the same way and optionally park top-down at `park`.
    status: released / dropped_in_transit / released_stalled."""
    dx, dy, dz = float(drop[0]), float(drop[1]), float(drop[2])
    out = {}
    cur = robot.now().tcp
    w1 = [dx, dy - APPROACH_BACK, travel_z]
    out["rise"] = _servo(robot, [float(cur[0]), float(cur[1]), travel_z], TOPDOWN_AA, 1, tol=0.015, rtol=0.05,
                         max_steps=100, watch_gap=0.006)
    if out["rise"]["lost"]:
        out["status"] = "dropped_in_transit"
        return out
    out["over"] = _servo(robot, w1, TOPDOWN_AA, 1, tol=0.008, rtol=0.04, max_steps=200, watch_gap=0.006)
    if out["over"]["lost"]:
        out["status"] = "dropped_in_transit"
        return out
    out["down"] = _servo(robot, [dx, dy, dz], TOPDOWN_AA, 1, tol=0.004, rtol=0.03, max_steps=150, vmax=0.5,
                         watch_gap=0.006)
    if out["down"]["lost"]:
        out["status"] = "dropped_in_transit"
        return out
    out["pre_release_gap"] = round(_gap(robot.now()), 4)
    out["open_gap"] = _grip(robot, -1, 20, TOPDOWN_AA)
    out["status"] = "released" if out["down"]["reached"] else "released_stalled"
    out["back"] = _servo(robot, w1, TOPDOWN_AA, -1, tol=0.015, rtol=0.05, max_steps=100)
    if park is not None:
        out["park"] = _servo(robot, park, TOPDOWN_AA, -1, tol=0.005, rtol=0.03, max_steps=250)
    return out


def act_wait(robot, steps=30):
    """Hold still with the gripper open for `steps` steps (lets the checker settle)."""
    for i in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    return {"tcp": [round(float(v), 4) for v in robot.now().tcp[:3]], "gap": round(_gap(robot.now()), 4)}


# ---------------------------------------------------------------- decision
FIELDS = ["phase", "cabinet", "handle", "drawer_open", "butter_est", "choc_est", "butter_done",
          "choc_done", "drawer_tries", "butter_tries", "choc_tries", "log"]


def _parse_goal(goal):
    g = (goal or "").lower()
    order = []
    for name in ("butter", "chocolate"):
        if name in g:
            order.append((g.index(name), name))
    order = [n for _, n in sorted(order)] or ["butter", "chocolate"]
    drawer = "middle"
    for d in ("top", "middle", "bottom"):
        if d + " drawer" in g:
            drawer = d
    return order, drawer


def _out(mem, tool, args):
    return {"memory": {k: mem.get(k) for k in FIELDS}, "act": {"tool": tool, "args": args}}


def _note(mem, s):
    lg = (mem.get("log") or "") + s + "; "
    mem["log"] = lg[-1500:]


def _obj_info(name):
    if name == "butter":
        return BUTTER_PRIOR, BUTTER_OFF, BUTTER_DZ, BUTTER_DROP, "butter_est", "butter_tries", "butter_done"
    return CHOC_PRIOR, CHOC_OFF, CHOC_DZ, CHOC_DROP, "choc_est", "choc_tries", "choc_done"


def perceive_decide(obs, goal, memory, last):
    """Phase machine: open middle drawer -> verify open -> pick/place butter -> pick/place chocolate."""
    mem = {k: (memory or {}).get(k) for k in FIELDS}
    for k in ("drawer_tries", "butter_tries", "choc_tries"):
        if mem[k] is None:
            mem[k] = 0
    for k in ("drawer_open", "butter_done", "choc_done"):
        if mem[k] is None:
            mem[k] = False
    f = obs.current
    order, _drawer = _parse_goal(goal)
    last = last or ""
    phase = mem["phase"] or "start"

    if phase == "start":
        cab = _cabinet(f)
        if cab is None:
            cab = list(DEFAULT_CAB)
            _note(mem, "cabinet not detected, using default")
        mem["cabinet"] = [round(cab[0], 4), round(cab[1], 4), 0.714]
        mem["handle"] = [round(cab[0] + HANDLE_DX, 4), round(cab[1] + HANDLE_DY, 4), HANDLE_Z]
        for name in ("butter", "chocolate"):
            prior, _o, _dz, _dr, ek, _t, _d = _obj_info(name)
            est = _agent_object(f, prior)
            mem[ek] = [round(est[0], 4), round(est[1], 4)] if est else list(prior)
            if est is None:
                _note(mem, name + " not seen in agentview, prior used")
        _note(mem, "handle %s" % mem["handle"])
        first = mem["butter_est"] if order[0] == "butter" else mem["choc_est"]
        mem["phase"] = "verify_drawer"
        mem["drawer_tries"] = 1
        return _out(mem, "act_open_drawer", {"handle": mem["handle"], "pull": PULL,
                                             "park": [first[0], first[1], HOVER_Z]})

    if phase == "verify_drawer":
        cnt = _drawer_open_count(f, mem["cabinet"])
        _note(mem, "drawer count %d" % cnt)
        if cnt >= 80:
            mem["drawer_open"] = True
            mem["phase"] = "pick"
        elif cnt >= 15:
            # partly open: the handle is no longer at the closed position, a blind re-grasp there would
            # hit the drawer front; continue and place (objects are dropped well inside the front)
            _note(mem, "drawer partly open; continuing")
            mem["drawer_open"] = True
            mem["phase"] = "pick"
        elif mem["drawer_tries"] < 3:
            mem["drawer_tries"] += 1
            h = mem["handle"]
            # retry with a slightly lower grasp and the handle re-measured from the current view
            cab = _cabinet(f)
            if cab is not None and abs(cab[0] - mem["cabinet"][0]) < 0.02 and abs(cab[1] - mem["cabinet"][1]) < 0.03:
                h = [round(cab[0] + HANDLE_DX, 4), round(cab[1] + HANDLE_DY, 4), HANDLE_Z]
            h = [h[0], h[1], HANDLE_Z - 0.004 * (mem["drawer_tries"] - 1)]
            first = mem["butter_est"] if order[0] == "butter" else mem["choc_est"]
            return _out(mem, "act_open_drawer", {"handle": h, "pull": PULL,
                                                 "park": [first[0], first[1], HOVER_Z]})
        else:
            _note(mem, "drawer not confirmed open; continuing")
            mem["phase"] = "pick"
        phase = mem["phase"]

    # which object is next
    todo = [n for n in order if not mem["butter_done" if n == "butter" else "choc_done"]]
    if not todo:
        mem["phase"] = "done"
        return _out(mem, "act_wait", {"steps": 40})
    name = todo[0]
    prior, off, dz, drop, ek, tk, dk = _obj_info(name)
    h = mem["handle"] or DEFAULT_HANDLE
    est = mem[ek] or list(prior)

    if phase == "place_check":
        if '"status": "released' in last:
            mem[dk] = True
            _note(mem, name + " released in drawer")
            todo = [n for n in order if not mem["butter_done" if n == "butter" else "choc_done"]]
            if not todo:
                mem["phase"] = "done"
                return _out(mem, "act_wait", {"steps": 40})
            name = todo[0]
            prior, off, dz, drop, ek, tk, dk = _obj_info(name)
            est = mem[ek] or list(prior)
            mem["phase"] = "pick"
            return _out(mem, "act_move", {"pos": [est[0], est[1], HOVER_Z], "gripper": -1})
        # dropped in transit: re-detect from agentview around the prior
        _note(mem, name + " lost in transit")
        e2 = _agent_object(f, est, r=0.12)
        if e2 is not None:
            mem[ek] = [round(e2[0], 4), round(e2[1], 4)]
        mem["phase"] = "pick"
        est = mem[ek]
        return _out(mem, "act_move", {"pos": [est[0], est[1], HOVER_Z], "gripper": -1})

    if phase == "grasp_check":
        g = _gap(f)
        if g > 0.008 and f.tcp[2] > 0.6:
            mem["phase"] = "place_check"
            dp = _safe_drop([h[0] + drop[0], h[1] + drop[1], h[2] + drop[2]])
            park = None
            rest = [n for n in todo if n != name]
            if rest:
                pe = mem[_obj_info(rest[0])[4]] or _obj_info(rest[0])[0]
                park = [pe[0], pe[1], HOVER_Z]
            return _out(mem, "act_place", {"drop": [round(v, 4) for v in dp], "travel_z": TRAVEL_Z, "park": park})
        _note(mem, "%s grasp miss gap %.4f" % (name, g))
        mem["phase"] = "pick"
        return _out(mem, "act_move", {"pos": [est[0], est[1], HOVER_Z], "gripper": -1})

    # phase pick: ensure hovering top-down above estimate, then refine with wrist and pick
    p = f.tcp[:3]
    rerr = float(np.linalg.norm(_mat2aa(_aa2mat(TOPDOWN_AA) @ _aa2mat(f.tcp[3:]).T)))
    near = abs(p[0] - est[0]) < 0.03 and abs(p[1] - est[1]) < 0.03 and abs(p[2] - HOVER_Z) < 0.03 and rerr < 0.08
    if not near:
        mem["phase"] = "pick"
        return _out(mem, "act_move", {"pos": [est[0], est[1], HOVER_Z], "gripper": -1})
    if mem[tk] >= 4:
        _note(mem, name + " retries exhausted")
        mem[dk] = False
        mem["phase"] = "done"
        return _out(mem, "act_wait", {"steps": 40})
    c, tz, n = _wrist_object(f, est)
    if c is None:
        _note(mem, "%s not seen by wrist (n=%d), using estimate" % (name, n))
        c = est
    else:
        mem[ek] = [round(c[0], 4), round(c[1], 4)]
    k = mem[tk]
    mem[tk] = k + 1
    # later tries: small deterministic perturbation of grasp height
    gz = tz + dz - 0.002 * min(k, 2)
    gx = c[0] + off[0]
    gy = c[1] + off[1]
    mem["phase"] = "grasp_check"
    _note(mem, "%s pick %d at %.3f %.3f %.3f" % (name, k + 1, gx, gy, gz))
    return _out(mem, "act_pick", {"xy": [round(gx, 4), round(gy, 4)], "z": round(gz, 4), "lift_z": 0.75})
