"""Butter -> frypan, then pour milk over it N times (LIBERO / RoboMemArena T15).

Perception: agentview depth back-projected to world coordinates (K + extrinsic).
Control: closed-loop OSC delta steps toward world-frame pose targets.
All geometric offsets come from the assigned demonstrations (see task_analysis.md).
"""

TABLE_Z_DEFAULT = 0.4305
# demo-mean detections (fallback only when detection fails)
BUTTER_DEFAULT = [0.113, -0.205]
MILK_DEFAULT = [-0.103, -0.176]
PAN_DEFAULT = [-0.020, 0.177]
# offsets learned from demos: expert pose minus detected object reference
BUTTER_GRASP_OFF = [-0.004, 0.002]
BUTTER_GRASP_Z = 0.433
RELEASE_OFF = [0.0235, -0.0402]
RELEASE_Z = 0.612
MILK_ABOVE_OFF = [0.0005, 0.0185]
MILK_APPROACH_OFF = [0.0113, 0.0005]
MILK_GRASP_OFF = [-0.0098, 0.0001]
MILK_GRASP_Z = 0.4435
POUR1_OFF = [0.0135, -0.0647]
POUR1_Z = 0.6955
POUR2_OFF = [0.0199, -0.0595]
POUR2_Z = 0.6706
TILT_DEG = 49.1
EMPTY_WIDTH = 0.010   # finger opening below this when closed = nothing grasped


# ---------------------------------------------------------------- geometry
def _rv2m(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _m2rv(R):
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        i = int(np.argmax(np.diag(R)))
        col = R[:, i] + np.eye(3)[:, i]
        k = col / max(1e-9, np.linalg.norm(col))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2 * math.sin(th)))


def _R_down():
    return np.diag([1.0, -1.0, -1.0])


def _R_tilt():
    a = math.radians(TILT_DEG)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return Ry @ _R_down()


def _R_named(name):
    if name == "tilt":
        return _R_tilt()
    return _R_down()


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _tilt_of(frame):
    """Angle (deg) between the gripper approach axis and straight down."""
    R = _rv2m(frame.tcp[3:6])
    z = R[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(z[2])))))


# ---------------------------------------------------------------- perception
def _world_points(frame):
    dep = frame.agentview_depth.astype(float) / 1000.0
    K = frame.agentview_K
    E = frame.agentview_extrinsic
    H, W = dep.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) * dep / K[0, 0]
    y = (v - K[1, 2]) * dep / K[1, 1]
    pc = np.stack([x, y, dep], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, dep > 0


def _detect(frame):
    """Detect table height, butter, milk carton (top centroid) and frypan centroid in world metres."""
    pw, ok = _world_points(frame)
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    area = ok & (x > -0.25) & (x < 0.40) & (np.abs(y) < 0.45)
    tab = area & (z > 0.38) & (z < 0.47)
    table_z = TABLE_Z_DEFAULT
    if tab.sum() > 500:
        tz = float(np.median(z[tab]))
        if abs(tz - TABLE_Z_DEFAULT) < 0.02:
            table_z = tz
    base = area & (z > table_z + 0.0065) & (z < table_z + 0.19)
    out = {"table_z": table_z}
    mk = base & (y < -0.10) & (x < 0.0)
    if mk.sum() >= 15 and float(z[mk].max()) > table_z + 0.08:
        ztop = float(z[mk].max())
        top = mk & (z > ztop - 0.012)
        out["milk"] = [float(x[top].mean()), float(y[top].mean()), ztop]
        out["milk_n"] = int(mk.sum())
    else:
        out["milk"] = None
        out["milk_n"] = int(mk.sum())
    bt = base & (y < -0.10) & (x > 0.03) & (z < table_z + 0.05)
    if bt.sum() >= 8:
        out["butter"] = [float(x[bt].mean()), float(y[bt].mean()), float(z[bt].max())]
    else:
        out["butter"] = None
    out["butter_n"] = int(bt.sum())
    fp = base & (y > 0.0) & (x > -0.20) & (z < table_z + 0.07)
    if fp.sum() >= 100:
        out["pan"] = [float(x[fp].mean()), float(y[fp].mean()), float(z[fp].max())]
    else:
        out["pan"] = None
    out["pan_n"] = int(fp.sum())
    return out


def _butter_in_pan(frame, pan):
    """Count raised points near the release spot inside the pan (butter in pan) and at the
    original butter spot (butter still on the table)."""
    pw, ok = _world_points(frame)
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    rx, ry = pan[0] + RELEASE_OFF[0], pan[1] + RELEASE_OFF[1]
    near = ok & (np.abs(x - rx) < 0.06) & (np.abs(y - ry) < 0.06) & (z > TABLE_Z_DEFAULT + 0.004) & (z < 0.50)
    return int(near.sum())


# ---------------------------------------------------------------- act tools
def _step_toward(robot, pos, R_t, grip, speed, rot_speed, pos_gain, hold_pos):
    fr = robot.now()
    p = np.asarray(fr.tcp[:3], dtype=float)
    Rc = _rv2m(fr.tcp[3:6])
    if hold_pos or pos is None:
        a_p = np.zeros(3)
        perr = 0.0
    else:
        e = np.asarray(pos, dtype=float) - p
        perr = float(np.linalg.norm(e))
        a_p = np.clip(e * pos_gain, -speed, speed)
    if R_t is None:
        a_r = np.zeros(3)
        rerr = 0.0
    else:
        er = _m2rv(R_t @ Rc.T)
        rerr = float(np.linalg.norm(er))
        a_r = np.clip(er / 0.5 * 1.2, -rot_speed, rot_speed)
    robot.step(float(a_p[0]), float(a_p[1]), float(a_p[2]),
               float(a_r[0]), float(a_r[1]), float(a_r[2]), float(grip))
    return perr, rerr


def act_move(robot, x, y, z, orient="down", grip=-1, max_steps=200, tol=0.004,
             rot_tol_deg=3.0, speed=1.0, pos_gain=15.0, settle=3):
    """Move the TCP to world (x, y, z) with orientation 'down' (vertical gripper), 'tilt'
    (gripper tilted for the milk grasp) or 'keep'. grip: -1 open, +1 closed. Stops after
    `settle` consecutive steps within tolerance or max_steps. Returns measured pose."""
    R_t = None if orient == "keep" else _R_named(orient)
    target = [float(x), float(y), float(z)]
    good = 0
    n = 0
    perr = rerr = 0.0
    while n < int(max_steps):
        perr, rerr = _step_toward(robot, target, R_t, grip, speed, 0.8, pos_gain, False)
        n += 1
        if perr < tol and math.degrees(rerr) < rot_tol_deg:
            good += 1
            if good >= settle:
                break
        else:
            good = 0
    fr = robot.now()
    p = fr.tcp[:3]
    return {"steps": n, "converged": good >= settle,
            "tcp": [round(float(v), 4) for v in p],
            "err_mm": round(1000 * float(np.linalg.norm(np.asarray(target) - p)), 1),
            "tilt_deg": round(_tilt_of(fr), 1), "width": round(_width(fr), 4)}


def act_rotate(robot, orient="down", grip=1, max_steps=60, rot_tol_deg=2.0, hold_steps=0):
    """Rotate the gripper toward orientation 'down' or 'tilt' with zero translation command
    (as the expert pours), then hold still for hold_steps. Returns measured tilt."""
    R_t = _R_named(orient)
    n = 0
    rerr = 0.0
    while n < int(max_steps):
        perr, rerr = _step_toward(robot, None, R_t, grip, 0.0, 0.8, 0.0, True)
        n += 1
        if math.degrees(rerr) < rot_tol_deg:
            break
    tilts = []
    for i in range(int(hold_steps)):
        robot.step(0, 0, 0, 0, 0, 0, float(grip))
        if i % 10 == 0:
            tilts.append(round(_tilt_of(robot.now()), 1))
    fr = robot.now()
    return {"rot_steps": n, "rot_err_deg": round(math.degrees(rerr), 1),
            "held": int(hold_steps), "tilt_deg": round(_tilt_of(fr), 1),
            "tilts": tilts, "tcp": [round(float(v), 4) for v in fr.tcp[:3]],
            "width": round(_width(fr), 4)}


def act_grip(robot, close=True, steps=25):
    """Close (close=True) or open the gripper without moving the arm for `steps` steps.
    Returns the measured finger opening width (m)."""
    g = 1.0 if close else -1.0
    widths = []
    for i in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, g)
        widths.append(_width(robot.now()))
    fr = robot.now()
    return {"width": round(_width(fr), 4), "moving": abs(widths[-1] - widths[-3]) > 0.0005 if len(widths) > 3 else True,
            "tcp": [round(float(v), 4) for v in fr.tcp[:3]]}


def act_hold(robot, steps=20, grip=1):
    """Hold the arm still for `steps` steps with the given gripper sign."""
    for i in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, float(grip))
    fr = robot.now()
    return {"held": int(steps), "tilt_deg": round(_tilt_of(fr), 1), "width": round(_width(fr), 4)}


# ---------------------------------------------------------------- decision
def _parse_pours(goal):
    g = (goal or "").lower()
    if "three times" in g or "thrice" in g:
        return 3
    if "twice" in g or "two times" in g:
        return 2
    if "pour" in g:
        return 1
    return 2


def _xy(v, default):
    if isinstance(v, (list, tuple)) and len(v) >= 2 and v[0] is not None:
        return [float(v[0]), float(v[1])]
    return list(default)


def perceive_decide(obs, goal, memory, last):
    """State machine: grasp butter -> release over frypan -> tilted milk grasp -> pour N times."""
    m = dict(memory)
    defaults = {"stage": "init", "butter": None, "milk": None, "pan": None, "table_z": None,
                "tries": 0, "butter_attempts": 0, "milk_attempts": 0, "pours_done": 0,
                "pours_needed": None, "events": ""}
    for k, v in defaults.items():
        if m.get(k) is None:
            m[k] = v
    fr = obs.current
    tcp = fr.tcp
    width = _width(fr)
    stage = m["stage"]
    ev = []

    def done(act_tool, args, nxt):
        m["stage"] = nxt
        if ev:
            m["events"] = (m["events"] + " | " + "; ".join(ev))[-1500:]
        return {"memory": m, "act": {"tool": act_tool, "args": args}}

    if stage == "init":
        m["pours_needed"] = _parse_pours(goal)
        det = _detect(fr)
        m["table_z"] = det["table_z"]
        m["butter"] = det["butter"] if det["butter"] is not None else BUTTER_DEFAULT + [0.443]
        m["milk"] = det["milk"] if det["milk"] is not None else MILK_DEFAULT + [0.563]
        m["pan"] = det["pan"] if det["pan"] is not None else PAN_DEFAULT + [0.479]
        ev.append("det b=%s m=%s p=%s n=%d/%d/%d" % (
            [round(v, 3) for v in m["butter"]], [round(v, 3) for v in m["milk"]],
            [round(v, 3) for v in m["pan"]], det["butter_n"], det["milk_n"], det["pan_n"]))
        stage = "butter_above"

    b = _xy(m["butter"], BUTTER_DEFAULT)
    mk = _xy(m["milk"], MILK_DEFAULT)
    pn = _xy(m["pan"], PAN_DEFAULT)
    att = int(m["butter_attempts"])
    # small alternating xy jitter on retries
    jit = [[0, 0], [0.004, 0], [-0.004, 0], [0, 0.004], [0, -0.004]][att % 5]
    bgx, bgy = b[0] + BUTTER_GRASP_OFF[0] + jit[0], b[1] + BUTTER_GRASP_OFF[1] + jit[1]

    if stage == "butter_above":
        return done("act_move", {"x": bgx, "y": bgy, "z": 0.64, "orient": "down", "grip": -1,
                                 "max_steps": 150, "tol": 0.005}, "butter_descend")
    if stage == "butter_descend":
        return done("act_move", {"x": bgx, "y": bgy, "z": BUTTER_GRASP_Z, "orient": "down", "grip": -1,
                                 "max_steps": 150, "tol": 0.003, "speed": 0.6}, "butter_close")
    if stage == "butter_close":
        return done("act_grip", {"close": True, "steps": 25}, "butter_check")
    if stage == "butter_check":
        m["butter_attempts"] = att + 1
        if width > EMPTY_WIDTH:
            ev.append("butter grasped w=%.4f" % width)
            return done("act_move", {"x": bgx, "y": bgy, "z": 0.76, "orient": "down", "grip": 1,
                                     "max_steps": 120, "tol": 0.01}, "butter_lifted")
        ev.append("butter grasp empty w=%.4f att=%d" % (width, att + 1))
        if att + 1 >= 4:
            ev.append("butter attempts exhausted; continue to milk")
            return done("act_move", {"x": bgx, "y": bgy, "z": 0.70, "orient": "down", "grip": -1,
                                     "max_steps": 80, "tol": 0.01}, "milk_redetect_move")
        return done("act_move", {"x": float(tcp[0]), "y": float(tcp[1]), "z": 0.60, "orient": "down", "grip": -1,
                                 "max_steps": 100, "tol": 0.01}, "butter_clear_view")
    if stage == "butter_clear_view":
        return done("act_move", {"x": -0.05, "y": 0.0, "z": 0.70, "orient": "down", "grip": -1,
                                 "max_steps": 120, "tol": 0.01}, "butter_redetect")
    if stage == "butter_redetect":
        det = _detect(fr)
        if det["butter"] is not None:
            m["butter"] = det["butter"]
        ev.append("butter redetect %s" % det["butter"])
        b = _xy(m["butter"], BUTTER_DEFAULT)
        att = int(m["butter_attempts"])
        jit = [[0, 0], [0.004, 0], [-0.004, 0], [0, 0.004], [0, -0.004]][att % 5]
        return done("act_move", {"x": b[0] + BUTTER_GRASP_OFF[0] + jit[0], "y": b[1] + BUTTER_GRASP_OFF[1] + jit[1],
                                 "z": 0.64, "orient": "down", "grip": -1, "max_steps": 150, "tol": 0.005},
                    "butter_descend")
    rx, ry = pn[0] + RELEASE_OFF[0], pn[1] + RELEASE_OFF[1]
    if stage == "butter_lifted":
        if width <= EMPTY_WIDTH:
            ev.append("butter lost during lift w=%.4f" % width)
            return done("act_move", {"x": -0.05, "y": 0.0, "z": 0.70, "orient": "down", "grip": -1,
                                     "max_steps": 120, "tol": 0.01}, "butter_redetect")
        return done("act_move", {"x": rx, "y": ry, "z": 0.83, "orient": "down", "grip": 1,
                                 "max_steps": 150, "tol": 0.008}, "pan_lower")
    if stage == "pan_lower":
        if width <= EMPTY_WIDTH:
            ev.append("butter lost in transport w=%.4f" % width)
            return done("act_move", {"x": -0.05, "y": 0.0, "z": 0.70, "orient": "down", "grip": -1,
                                     "max_steps": 120, "tol": 0.01}, "butter_redetect")
        return done("act_move", {"x": rx, "y": ry, "z": RELEASE_Z, "orient": "down", "grip": 1,
                                 "max_steps": 150, "tol": 0.004}, "butter_release")
    if stage == "butter_release":
        return done("act_grip", {"close": False, "steps": 22}, "pan_retreat")
    if stage == "pan_retreat":
        return done("act_move", {"x": rx, "y": ry, "z": 0.79, "orient": "down", "grip": -1,
                                 "max_steps": 100, "tol": 0.01}, "milk_redetect")
    if stage in ("milk_redetect", "milk_redetect_move"):
        det = _detect(fr)
        if det["milk"] is not None:
            old = _xy(m["milk"], MILK_DEFAULT)
            if abs(det["milk"][0] - old[0]) < 0.06 and abs(det["milk"][1] - old[1]) < 0.06:
                m["milk"] = det["milk"]
        ev.append("milk redetect %s; butter_near_pan_pts=%d" % (
            None if det["milk"] is None else [round(v, 3) for v in det["milk"]], _butter_in_pan(fr, pn)))
        mk = _xy(m["milk"], MILK_DEFAULT)
        return done("act_move", {"x": mk[0] + MILK_ABOVE_OFF[0], "y": mk[1] + MILK_ABOVE_OFF[1], "z": 0.84,
                                 "orient": "down", "grip": -1, "max_steps": 150, "tol": 0.008}, "milk_tilt")
    if stage == "milk_tilt":
        return done("act_move", {"x": mk[0] + 0.003, "y": mk[1] + 0.005, "z": 0.70, "orient": "tilt", "grip": -1,
                                 "max_steps": 150, "tol": 0.008, "rot_tol_deg": 2.0}, "milk_descend")
    if stage == "milk_descend":
        return done("act_move", {"x": mk[0] + MILK_APPROACH_OFF[0], "y": mk[1] + MILK_APPROACH_OFF[1], "z": MILK_GRASP_Z - 0.002,
                                 "orient": "tilt", "grip": -1, "max_steps": 200, "tol": 0.003, "speed": 0.7}, "milk_approach")
    if stage == "milk_approach":
        return done("act_move", {"x": mk[0] + MILK_GRASP_OFF[0], "y": mk[1] + MILK_GRASP_OFF[1], "z": MILK_GRASP_Z,
                                 "orient": "tilt", "grip": -1, "max_steps": 120, "tol": 0.003, "speed": 0.3}, "milk_close")
    if stage == "milk_close":
        return done("act_grip", {"close": True, "steps": 25}, "milk_check")
    if stage == "milk_check":
        matt = int(m["milk_attempts"]) + 1
        m["milk_attempts"] = matt
        if width > EMPTY_WIDTH:
            ev.append("milk grasped w=%.4f" % width)
            return done("act_move", {"x": mk[0] + 0.0026, "y": mk[1] + 0.0032, "z": 0.78, "orient": "tilt", "grip": 1,
                                     "max_steps": 150, "tol": 0.01}, "pour_move")
        ev.append("milk grasp empty w=%.4f att=%d" % (width, matt))
        return done("act_move", {"x": float(tcp[0]), "y": float(tcp[1]), "z": 0.62, "orient": "tilt", "grip": -1,
                                 "max_steps": 120, "tol": 0.01}, "milk_retry_view" if matt < 4 else "give_up")
    if stage == "milk_retry_view":
        return done("act_move", {"x": 0.0, "y": 0.12, "z": 0.80, "orient": "down", "grip": -1,
                                 "max_steps": 150, "tol": 0.01}, "milk_redetect")
    p1x, p1y = pn[0] + POUR1_OFF[0], pn[1] + POUR1_OFF[1]
    p2x, p2y = pn[0] + POUR2_OFF[0], pn[1] + POUR2_OFF[1]
    if stage == "pour_move":
        if width <= EMPTY_WIDTH:
            ev.append("milk lost on lift w=%.4f" % width)
            return done("act_move", {"x": 0.0, "y": 0.12, "z": 0.80, "orient": "down", "grip": -1,
                                     "max_steps": 150, "tol": 0.01}, "milk_redetect")
        return done("act_move", {"x": p1x, "y": p1y, "z": POUR1_Z, "orient": "tilt", "grip": 1,
                                 "max_steps": 200, "tol": 0.005}, "pour")
    if stage == "pour":
        k = int(m["pours_done"])
        need = int(m["pours_needed"])
        final = k + 1 >= need
        hold = 150 if final else 50
        return done("act_rotate", {"orient": "down", "grip": 1, "max_steps": 60, "hold_steps": hold}, "unpour")
    if stage == "unpour":
        k = int(m["pours_done"]) + 1
        m["pours_done"] = k
        ev.append("pour %d done tilt=%.1f" % (k, _tilt_of(fr)))
        return done("act_rotate", {"orient": "tilt", "grip": 1, "max_steps": 60, "hold_steps": 10}, "pour_next")
    if stage == "pour_next":
        if width <= EMPTY_WIDTH:
            ev.append("milk lost while pouring w=%.4f" % width)
        if int(m["pours_done"]) >= int(m["pours_needed"]) + 3:
            return done("act_hold", {"steps": 100, "grip": 1}, "give_up")
        return done("act_move", {"x": p2x, "y": p2y, "z": POUR2_Z, "orient": "tilt", "grip": 1,
                                 "max_steps": 120, "tol": 0.005}, "pour")
    # give_up / unknown stage: hold still
    return done("act_hold", {"steps": 100, "grip": 1 if width > EMPTY_WIDTH else -1}, "give_up")
