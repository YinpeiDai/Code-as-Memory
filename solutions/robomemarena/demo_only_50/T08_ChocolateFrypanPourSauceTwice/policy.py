"""Chocolate -> frypan, pour sauce twice, sauce bottle -> bowl drainer.

Perception: agentview depth + calibration -> world points; fixed table-top windows
locate the chocolate box, the sauce bottle cap, the frypan rim (circle fit) and the
drainer.  Control: closed-loop pose servo on the measured TCP (OSC delta steps).
A phase counter in memory sequences the waypoints; object positions measured on the
first frame persist in memory.
"""

TABLE_Z = 0.43
NOM_CHOC = [-0.1246, -0.1746]
NOM_BOTTLE = [0.0760, -0.1769]
NOM_PAN = [0.002, 0.125]
NOM_DRAIN = [-0.14, 0.54, 0.43, 0.65]   # x centre, y centre, ymin, ymax
K_AGENT = [[231.76450199, 0.0, 127.5], [0.0, 309.01933598, 127.5], [0.0, 0.0, 1.0]]

# Expert-derived offsets (see task_analysis.md)
CHOC_GRASP_OFF = [0.0016, 0.0006]
CHOC_GRASP_Z = 0.438
BOT_GRASP_OFF = [0.004, -0.010]
BOT_GRASP_Z = 0.468
DROP_OFF = [-0.002, -0.012]
DROP_Z = 0.60
POUR_OFF = [0.030, -0.010]
POUR_Z = 0.77
POUR_TILT = 0.84       # rad (48 deg); experts reach 46 deg
POUR_HOLD = 60         # steps held at full tilt (experts: success ~30-36 steps)
TRAVEL_Z = 0.80
VIEW_POSE = [-0.06, 0.0, 0.75]
CHOC_CLOSED_MIN = 0.008
BOT_CLOSED_MIN = 0.010


def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _mat2aa(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-8:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        return V[:, int(np.argmax(w))] * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return k * th


def _r_des(tilt):
    """Gripper pointing down (axis-angle [pi,0,0]) rotated about world y by -tilt."""
    return _aa2mat([0.0, -float(tilt), 0.0]) @ _aa2mat([math.pi, 0.0, 0.0])


def _tilt_deg(tcp):
    R = _aa2mat(tcp[3:6])
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 2])))))


def _world_points(frame):
    depth = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    K = np.asarray(frame.agentview_K, dtype=float) if "agentview_K" in frame.sensors else np.asarray(K_AGENT)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1).reshape(-1, 3)
    ok = depth.reshape(-1) > 0
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw[ok]


def _win(P, x0, x1, y0, y1, z0, z1):
    m = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > z0) & (P[:, 2] < z1)
    return P[m]


def _table_z(P):
    T = _win(P, -0.3, 0.3, -0.45, 0.45, 0.38, 0.48)
    if len(T) < 200:
        return TABLE_Z
    hist, edges = np.histogram(T[:, 2], bins=100, range=(0.38, 0.48))
    i = int(np.argmax(hist))
    z = float((edges[i] + edges[i + 1]) / 2.0)
    if abs(z - TABLE_Z) > 0.02:
        return TABLE_Z
    return z


def _near(p, q, r):
    return math.hypot(p[0] - q[0], p[1] - q[1]) < r


def _perceive_scene(frame):
    P = _world_points(frame)
    tz = _table_z(P)
    out = {"table_z": tz, "fallback": []}
    # chocolate: low box, far-left region
    C = _win(P, NOM_CHOC[0] - 0.08, NOM_CHOC[0] + 0.08, NOM_CHOC[1] - 0.08, NOM_CHOC[1] + 0.08, tz + 0.007, tz + 0.045)
    choc = list(NOM_CHOC)
    if len(C) >= 15:
        lo = np.percentile(C[:, :2], 2, axis=0)
        hi = np.percentile(C[:, :2], 98, axis=0)
        cand = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)]
        if _near(cand, NOM_CHOC, 0.06):
            choc = cand
        else:
            out["fallback"].append("choc")
    else:
        out["fallback"].append("choc")
    # bottle: cylinder, cap = highest points
    B = _win(P, NOM_BOTTLE[0] - 0.09, NOM_BOTTLE[0] + 0.09, NOM_BOTTLE[1] - 0.09, NOM_BOTTLE[1] + 0.09, tz + 0.007, tz + 0.12)
    bot = list(NOM_BOTTLE)
    bot_top = tz + 0.0758
    if len(B) >= 20:
        zt = float(B[:, 2].max())
        T = B[B[:, 2] > zt - 0.006]
        cand = [float(T[:, 0].mean()), float(T[:, 1].mean())]
        if _near(cand, NOM_BOTTLE, 0.06) and zt > tz + 0.05:
            bot = cand
            bot_top = zt
        else:
            out["fallback"].append("bottle")
    else:
        out["fallback"].append("bottle")
    # frypan: rim circle fit (handle excluded by y-window from rim's near edge)
    pan = list(NOM_PAN)
    F = _win(P, NOM_PAN[0] - 0.17, NOM_PAN[0] + 0.17, NOM_PAN[1] - 0.17, NOM_PAN[1] + 0.25, tz + 0.017, tz + 0.07)
    pan_r = 0.097
    if len(F) >= 60:
        Q2 = F[:, :2]
        ymin = float(np.percentile(Q2[:, 1], 1))
        Q = Q2[Q2[:, 1] < ymin + 0.205]
        if len(Q) >= 40:
            A = np.concatenate([2.0 * Q, np.ones((len(Q), 1))], 1)
            b = (Q ** 2).sum(1)
            s = np.linalg.lstsq(A, b, rcond=None)[0]
            c = [float(s[0]), float(s[1])]
            r2 = float(s[2] + s[0] ** 2 + s[1] ** 2)
            r = math.sqrt(r2) if r2 > 0 else 0.0
            if 0.07 < r < 0.12 and _near(c, NOM_PAN, 0.10):
                pan = c
                pan_r = r
            else:
                out["fallback"].append("pan")
        else:
            out["fallback"].append("pan")
    else:
        out["fallback"].append("pan")
    # drainer: tall walls
    D = _win(P, -0.45, 0.3, 0.28, 0.85, tz + 0.12, tz + 0.2)
    drain = list(NOM_DRAIN)
    if len(D) >= 100:
        lo = np.percentile(D[:, :2], 1, axis=0)
        hi = np.percentile(D[:, :2], 99, axis=0)
        cand = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(lo[1]), float(hi[1])]
        if _near(cand, NOM_DRAIN, 0.12):
            drain = cand
        else:
            out["fallback"].append("drain")
    else:
        out["fallback"].append("drain")
    out.update({"choc": choc, "bottle": bot, "bottle_top": bot_top, "pan": pan, "pan_r": pan_r, "drain": drain})
    return out


def _servo(robot, target, tilt, grip, tol=0.004, rtol=0.02, max_steps=250, settle=3, vmax=1.0,
           allow_stall=True, log=None):
    """Closed-loop pose servo. Returns (converged, steps, final error m)."""
    tgt = np.asarray(target, dtype=float)
    Rd = _r_des(tilt)
    integ = np.zeros(3)
    good = 0
    n = 0
    err = 1.0
    hist = []
    while n < max_steps:
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        e = tgt - tcp[:3]
        err = float(np.linalg.norm(e))
        R = _aa2mat(tcp[3:6])
        er = _mat2aa(Rd @ R.T)
        rerr = float(np.linalg.norm(er))
        if err < tol and rerr < rtol:
            good += 1
            if good >= settle:
                return True, n, err
        else:
            good = 0
        if err < 0.05 and grip > 0:
            # integral only while holding an object (compensates its uncompensated weight)
            integ = np.clip(integ + e, -0.25, 0.25)
        hist.append(err)
        if allow_stall and len(hist) > 25 and err < 2.0 * tol + 0.002 and rerr < 2.0 * rtol and hist[-25] - err < 0.0005:
            return True, n, err   # stalled close to target (e.g. contact): accept
        a = 12.0 * e + 1.5 * integ
        na = float(np.max(np.abs(a)))
        if na > vmax:
            a = a * (vmax / na)
        ar = 1.5 * er
        nr = float(np.linalg.norm(ar))
        if nr > 0.8:
            ar = ar * (0.8 / nr)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
        n += 1
        if log is not None:
            log.append(_tilt_deg(np.asarray(robot.now().tcp, dtype=float)))
    return False, n, err


def _state(robot):
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    fing = np.asarray(f.fingers, dtype=float)
    return {"tcp": [round(float(v), 4) for v in tcp[:3]], "tilt_deg": round(_tilt_deg(tcp), 1),
            "finger": round(float(abs(fing[0]) + abs(fing[1])) / 2.0, 4)}


def act_move(robot, x, y, z, tilt=0.0, grip=-1.0, tol=0.004, max_steps=250, vmax=1.0):
    """Servo the TCP to world (x,y,z) metres with the gripper pointing down tilted by
    `tilt` rad about world -y; gripper command grip (+1 close, -1 open) held throughout.
    Returns convergence, steps and the measured final state."""
    ok, n, err = _servo(robot, [x, y, z], tilt, grip, tol=tol, max_steps=max_steps, vmax=vmax)
    out = _state(robot)
    out.update({"converged": bool(ok), "steps": n, "err": round(err, 4)})
    return out


def act_grip(robot, x, y, z, close=True, steps=25, tilt=0.0):
    """Hold pose (x,y,z,tilt) while closing (close=True) or opening the gripper for `steps` steps."""
    g = 1.0 if close else -1.0
    _servo(robot, [x, y, z], tilt, g, tol=0.0, max_steps=int(steps), settle=10 ** 6, allow_stall=False)
    out = _state(robot)
    out["steps"] = int(steps)
    return out


def act_pour(robot, x, y, z, tilt=POUR_TILT, hold=POUR_HOLD):
    """Tilt the held bottle to `tilt` rad above (x,y,z), hold it for `hold` steps, then
    return it upright at the same position. Gripper stays closed."""
    ok, n, err = _servo(robot, [x, y, z], tilt, 1.0, tol=0.01, rtol=0.03, max_steps=150, settle=2)
    tilts = []
    _servo(robot, [x, y, z], tilt, 1.0, tol=0.0, max_steps=int(hold), settle=10 ** 6,
           allow_stall=False, log=tilts)
    ok2, n2, err2 = _servo(robot, [x, y, z], 0.0, 1.0, tol=0.01, rtol=0.03, max_steps=150, settle=2)
    out = _state(robot)
    out.update({"tilt_reached": bool(ok), "min_hold_tilt_deg": round(min(tilts), 1) if tilts else None,
                "upright": bool(ok2), "steps": n + int(hold) + n2})
    return out


def act_observe(robot):
    """Hold position for one step with the gripper unchanged (open)."""
    robot.step(0, 0, 0, 0, 0, 0, -1)
    return _state(robot)


def _plan(scene):
    """Ordered list of (tool, args, tag) for the whole task given perceived scene."""
    tz = scene["table_z"]
    dz = max(-0.005, min(0.005, tz - TABLE_Z))
    cx = scene["choc"][0] + CHOC_GRASP_OFF[0]
    cy = scene["choc"][1] + CHOC_GRASP_OFF[1]
    cz = CHOC_GRASP_Z + dz
    bx = scene["bottle"][0] + BOT_GRASP_OFF[0]
    by = scene["bottle"][1] + BOT_GRASP_OFF[1]
    bz = BOT_GRASP_Z + dz
    px, py = scene["pan"]
    dx, dy = px + DROP_OFF[0], py + DROP_OFF[1]
    ox, oy = px + POUR_OFF[0], py + POUR_OFF[1]
    dr = scene["drain"]
    # near (lower-y) compartment of the drainer
    rx = dr[0]
    ry = dr[2] + 0.27 * (dr[3] - dr[2])
    top = tz + 0.164
    return [
        ("act_move", {"x": cx, "y": cy, "z": cz + 0.20, "grip": -1.0}, "choc_above"),
        ("act_move", {"x": cx, "y": cy, "z": cz + 0.03, "grip": -1.0, "tol": 0.004, "vmax": 0.7}, "choc_approach"),
        ("act_move", {"x": cx, "y": cy, "z": cz, "grip": -1.0, "tol": 0.002, "vmax": 0.25}, "choc_descend"),
        ("act_grip", {"x": cx, "y": cy, "z": cz, "close": True, "steps": 25}, "choc_close"),
        ("act_move", {"x": cx, "y": cy, "z": TRAVEL_Z, "grip": 1.0, "tol": 0.01, "vmax": 0.8}, "choc_lift"),
        ("act_move", {"x": dx, "y": dy, "z": TRAVEL_Z, "grip": 1.0, "tol": 0.006, "vmax": 0.8}, "pan_above"),
        ("act_move", {"x": dx, "y": dy, "z": DROP_Z, "grip": 1.0, "tol": 0.005, "vmax": 0.8}, "pan_lower"),
        ("act_grip", {"x": dx, "y": dy, "z": DROP_Z, "close": False, "steps": 25}, "choc_release"),
        ("act_move", {"x": dx, "y": dy, "z": TRAVEL_Z, "grip": -1.0, "tol": 0.01}, "pan_retreat"),
        ("act_move", {"x": bx, "y": by, "z": bz + 0.20, "grip": -1.0}, "bottle_above"),
        ("act_move", {"x": bx, "y": by, "z": bz + 0.03, "grip": -1.0, "tol": 0.004, "vmax": 0.7}, "bottle_approach"),
        ("act_move", {"x": bx, "y": by, "z": bz, "grip": -1.0, "tol": 0.002, "vmax": 0.25}, "bottle_descend"),
        ("act_grip", {"x": bx, "y": by, "z": bz, "close": True, "steps": 25}, "bottle_close"),
        ("act_move", {"x": bx, "y": by, "z": POUR_Z, "grip": 1.0, "tol": 0.01, "vmax": 0.8}, "bottle_lift"),
        ("act_move", {"x": ox, "y": oy, "z": POUR_Z, "grip": 1.0, "tol": 0.006, "vmax": 0.8}, "pour_pos"),
        ("act_pour", {"x": ox, "y": oy, "z": POUR_Z}, "pour1"),
        ("act_grip", {"x": ox, "y": oy, "z": POUR_Z, "close": True, "steps": 10}, "pour_pause"),
        ("act_pour", {"x": ox, "y": oy, "z": POUR_Z}, "pour2"),
        ("act_move", {"x": rx, "y": ry, "z": top + 0.15, "grip": 1.0, "tol": 0.008}, "drain_above"),
        ("act_move", {"x": rx, "y": ry, "z": top + 0.04, "grip": 1.0, "tol": 0.006}, "drain_lower"),
        ("act_grip", {"x": rx, "y": ry, "z": top + 0.04, "close": False, "steps": 30}, "bottle_release"),
        ("act_move", {"x": rx, "y": ry, "z": top + 0.20, "grip": -1.0, "tol": 0.01}, "drain_retreat"),
    ]


def _finger(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1])) / 2.0


def _at_view(cur):
    tcp = np.asarray(cur.tcp, dtype=float)
    return abs(tcp[0] - VIEW_POSE[0]) < 0.03 and abs(tcp[1] - VIEW_POSE[1]) < 0.03 and tcp[2] > VIEW_POSE[2] - 0.04


def _scene_of(mem):
    return {"choc": mem["choc"], "bottle": mem["bottle"], "pan": mem["pan"],
            "drain": mem["drain"], "table_z": float(mem["table_z"])}


def perceive_decide(obs, goal, memory, last):
    """Sequence the task: perceive the scene once (memory), then step through the plan,
    checking grasps by finger opening and retrying a failed grasp (re-perceiving)."""
    mem = dict(memory or {})
    cur = obs.current
    phase = int(mem.get("phase") or 0)
    retries = int(mem.get("retries") or 0)
    if mem.get("pan") is None:
        scene = _perceive_scene(cur)
        for k in ("choc", "bottle", "pan", "drain", "table_z"):
            mem[k] = scene[k]
        mem["notes"] = "fallback:" + ",".join(scene["fallback"])
        mem["pending"] = None
        phase = 0
    plan = _plan(_scene_of(mem))
    tags = [p[2] for p in plan]
    fing = _finger(cur)
    view_move = {"tool": "act_move", "args": {"x": VIEW_POSE[0], "y": VIEW_POSE[1], "z": VIEW_POSE[2],
                                              "grip": -1.0, "tol": 0.01}}
    if not mem.get("pending") and 0 < phase <= len(plan):
        prev = tags[phase - 1]
        failed = None
        if prev in ("choc_close", "choc_lift", "pan_above", "pan_lower") and fing < CHOC_CLOSED_MIN:
            failed = "choc"
        if prev in ("bottle_close", "bottle_lift", "pour_pos") and fing < BOT_CLOSED_MIN:
            failed = "bottle"
        if failed is not None and retries < 3:
            retries += 1
            mem["pending"] = failed
            mem["notes"] = "grasp_failed_" + failed + "_after_" + prev
    if mem.get("pending"):
        failed = mem["pending"]
        if not _at_view(cur):
            mem["retries"] = retries
            mem["phase"] = phase
            mem["stage"] = "retreat_view"
            tcp = np.asarray(cur.tcp, dtype=float)
            if tcp[2] < VIEW_POSE[2] - 0.08:
                # rise vertically first so the open gripper clears nearby objects
                return {"memory": _mem_out(mem), "act": {"tool": "act_move", "args": {
                    "x": float(tcp[0]), "y": float(tcp[1]), "z": VIEW_POSE[2], "grip": -1.0, "tol": 0.015}}}
            return {"memory": _mem_out(mem), "act": view_move}
        scene2 = _perceive_scene(cur)
        if failed == "choc":
            mem["choc"] = scene2["choc"]
            phase = tags.index("choc_above")
        else:
            mem["bottle"] = scene2["bottle"]
            phase = tags.index("bottle_above")
        mem["pending"] = None
        mem["notes"] = "repick_" + failed + " fallback:" + ",".join(scene2["fallback"])
        plan = _plan(_scene_of(mem))
    mem["retries"] = retries
    if phase >= len(plan):
        mem["phase"] = phase
        mem["stage"] = "done_idle"
        tcp = np.asarray(cur.tcp, dtype=float)
        return {"memory": _mem_out(mem), "act": {"tool": "act_grip",
                "args": {"x": float(tcp[0]), "y": float(tcp[1]), "z": float(tcp[2]), "close": False, "steps": 100}}}
    tool, args, tag = plan[phase]
    mem["phase"] = phase + 1
    mem["stage"] = tag
    return {"memory": _mem_out(mem), "act": {"tool": tool, "args": args}}


def _mem_out(mem):
    keys = ["phase", "stage", "retries", "choc", "bottle", "pan", "drain", "table_z", "pending", "notes"]
    return {k: mem.get(k) for k in keys}
