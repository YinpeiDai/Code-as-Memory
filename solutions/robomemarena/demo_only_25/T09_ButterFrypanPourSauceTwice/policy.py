"""Butter -> frypan, pour sauce twice, sauce bottle -> bowl drainer.

Phase machine driven by declared memory. Perception: agentview RGB-D back-projected
to world points with the frame calibration; objects segmented by height above the
table, workspace region and colour. Control: closed-loop P control on measured TCP
pose (OSC delta actions), gripper +1 close / -1 open.
"""

TABLE_Z = 0.430
# Nominal fallbacks (median of the 25 assigned demos), used only if perception fails.
NOM_BUTTER = [0.1095, -0.2025]
NOM_PAN = [0.0010, 0.2480]      # butter-release target above the pan
NOM_SAUCE = [-0.191, -0.215]
NOM_DRAINER = [-0.120, 0.537]
HOME = [-0.06, 0.0, 0.67]

# Offsets measured across all 25 demos (std <= 1 mm)
BUTTER_OFF = [-0.0035, 0.0023]  # grasp tcp - butter centroid
PAN_OFF = [0.0004, -0.0122]     # release tcp - fitted rim-circle centre
SAUCE_OFF = [-0.011, -0.013]    # grasp tcp - visible bottle centroid
POUR_OFF = [-0.008, -0.020]     # pour tcp - release target (between the expert's two pours)

Z_BUTTER_GRASP = 0.425   # below reachable 0.433: contact/stall terminates the descent
Z_SAUCE_GRASP = 0.458    # expert stalled at 0.468
Z_RELEASE = 0.643
Z_POUR = 0.715
Z_TRAVEL = 0.80
Z_SAUCE_ABOVE = 0.663
POUR_ANGLE = -0.785      # rotation about world y (rad), as in every demo
MAX_RETRY = 2


# ---------------------------------------------------------------- math helpers
def _rotvec_to_R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _R_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(ax))
        a = np.zeros(3)
        a[i] = ax[i]
        for j in range(3):
            if j != i:
                a[j] = (R[i, j] + R[j, i]) / (4.0 * ax[i])
        a = a / (np.linalg.norm(a) + 1e-12)
        return a * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _base_R():
    """Gripper pointing down, yaw 0 (the expert's working orientation, rotvec (pi,0,0))."""
    return np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])


def _target_R(tilt):
    return _Ry(tilt) @ _base_R()


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


# ---------------------------------------------------------------- perception
def _world_points(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1)
    W = (P - E[:, 3]) @ E[:, :3]
    W[d <= 0] = np.nan
    return W


def _box(W, x0, x1, y0, y1, z0, z1):
    X, Y, Z = W[..., 0], W[..., 1], W[..., 2]
    with np.errstate(invalid="ignore"):
        return (X > x0) & (X < x1) & (Y > y0) & (Y < y1) & (Z > z0) & (Z < z1)


def _fit_ring(P, c, r0=0.09, band=0.025):
    c = np.array(c, dtype=float)
    n = 0
    for _ in range(6):
        d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
        S = P[np.abs(d - r0) < band]
        n = len(S)
        if n < 30:
            return None, n
        x, y = S[:, 0], S[:, 1]
        for _ in range(5):
            dd = np.hypot(x - c[0], y - c[1]) + 1e-9
            J = np.stack([-(x - c[0]) / dd, -(y - c[1]) / dd], 1)
            res = dd - r0
            c = c - np.linalg.lstsq(J, res, rcond=None)[0]
    return c, n


def _perceive_scene(frame):
    W = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    br = rgb.sum(-1) / 3.0
    info = {}
    # butter: low object on the left-front of the table
    m = _box(W, 0.0, 0.30, -0.45, -0.05, TABLE_Z + 0.006, TABLE_Z + 0.07)
    if int(m.sum()) >= 15:
        c = np.nanmean(W[m], 0)
        info["butter"] = [float(c[0] + BUTTER_OFF[0]), float(c[1] + BUTTER_OFF[1])]
        info["butter_n"] = int(m.sum())
    else:
        info["butter"] = list(NOM_BUTTER)
        info["butter_n"] = 0
    # sauce bottle: tall object on the left-back of the table (near the robot)
    m = _box(W, -0.38, -0.02, -0.45, -0.02, TABLE_Z + 0.012, 0.62)
    if int(m.sum()) >= 25:
        c = np.nanmean(W[m], 0)
        info["sauce"] = [float(c[0] + SAUCE_OFF[0]), float(c[1] + SAUCE_OFF[1])]
        info["sauce_n"] = int(m.sum())
    else:
        info["sauce"] = list(NOM_SAUCE)
        info["sauce_n"] = 0
    # frypan: dark rim points on the right; fit a fixed-radius circle
    m = _box(W, -0.15, 0.20, 0.10, 0.50, TABLE_Z + 0.007, TABLE_Z + 0.05) & (br < 110)
    pan = None
    if int(m.sum()) >= 60:
        P = W[m]
        mean = np.nanmean(P, 0)
        c0 = [mean[0] + 0.0256, mean[1] - 0.060]
        c, n = _fit_ring(P, c0)
        if c is not None and abs(c[0] - c0[0]) < 0.06 and abs(c[1] - c0[1]) < 0.06:
            pan = [float(c[0] + PAN_OFF[0]), float(c[1] + PAN_OFF[1])]
        else:
            pan = [float(mean[0] + 0.026), float(mean[1] - 0.072)]
        info["pan_n"] = int(m.sum())
    else:
        info["pan_n"] = 0
    info["pan"] = pan if pan is not None else list(NOM_PAN)
    # bowl drainer: tall light object at the back right
    m = _box(W, -0.32, 0.05, 0.38, 0.72, TABLE_Z + 0.02, 0.66)
    if int(m.sum()) >= 50:
        c = np.nanmean(W[m], 0)
        info["drainer"] = [float(c[0]), float(c[1])]
        info["drainer_top"] = float(np.nanmax(W[m][:, 2]))
    else:
        info["drainer"] = list(NOM_DRAINER)
        info["drainer_top"] = 0.594
    # sanity clamp: keep targets inside the region seen in the demos +- margin
    for key, nom, lim in (("butter", NOM_BUTTER, 0.10), ("sauce", NOM_SAUCE, 0.10),
                          ("pan", NOM_PAN, 0.10), ("drainer", NOM_DRAINER, 0.15)):
        p = info[key]
        if not (np.isfinite(p[0]) and np.isfinite(p[1])) or abs(p[0] - nom[0]) > lim or abs(p[1] - nom[1]) > lim:
            info[key] = list(nom)
            info[key + "_fallback"] = True
    return info


def perceive_scene(obs):
    """Locate butter grasp, pan release target, sauce grasp and drainer (world xy, metres) from the current agentview RGB-D frame."""
    return _perceive_scene(obs.current)


# ---------------------------------------------------------------- decision
PHASES = [
    "butter_above", "butter_descend", "butter_close", "butter_lift",
    "to_pan", "pan_descend", "butter_release", "pan_lift",
    "sauce_above", "sauce_descend", "sauce_close", "sauce_lift",
    "to_pour1", "pour1", "untilt1", "to_pour2", "pour2", "untilt2",
    "to_drainer", "drainer_descend", "drainer_release", "drainer_lift", "done",
]


def _num(v, default):
    return default if v is None else v


def _mv(x, y, z, grip, **kw):
    args = {"x": float(x), "y": float(y), "z": float(z), "grip": float(grip)}
    for k, v in kw.items():
        args[k] = v
    return {"tool": "act_move", "args": args}


def perceive_decide(obs, goal, memory, last):
    """Phase machine: decide the next phase from memory and the latest frame, return the act call."""
    mem = {
        "phase": memory.get("phase"),
        "butter_xy": memory.get("butter_xy"),
        "pan_xy": memory.get("pan_xy"),
        "sauce_xy": memory.get("sauce_xy"),
        "drainer_xy": memory.get("drainer_xy"),
        "pours_done": _num(memory.get("pours_done"), 0),
        "retries": _num(memory.get("retries"), 0),
        "decisions": _num(memory.get("decisions"), 0) + 1,
        "note": memory.get("note"),
    }
    cur = obs.current
    width = _width(cur)
    prev = mem["phase"]
    failed = isinstance(last, str) and ("the tool failed" in last or "error:" in last)

    # ---- first decision (or a requested re-perception): perceive the scene
    if prev is None or prev in ("retry_butter_home", "retry_sauce_home") or mem["butter_xy"] is None:
        info = _perceive_scene(cur)
        if prev is None or mem["butter_xy"] is None:
            mem["butter_xy"] = info["butter"]
            mem["pan_xy"] = info["pan"]
            mem["sauce_xy"] = info["sauce"]
            mem["drainer_xy"] = info["drainer"]
            mem["note"] = "perceived n_butter=%d n_sauce=%d n_pan=%d" % (info["butter_n"], info["sauce_n"], info["pan_n"])
        elif prev == "retry_butter_home":
            mem["butter_xy"] = info["butter"]
        else:
            mem["sauce_xy"] = info["sauce"]
        if prev == "retry_sauce_home":
            nxt = "sauce_above"
        else:
            nxt = "butter_above"
    else:
        idx = PHASES.index(prev) if prev in PHASES else len(PHASES) - 1
        nxt = PHASES[min(idx + 1, len(PHASES) - 1)]
        if failed and prev in ("butter_above", "to_pan", "sauce_above", "to_pour1", "to_pour2", "to_drainer"):
            nxt = prev  # repeat a transit that raised an error
        # grasp verification after lifting
        if prev == "butter_lift" and width < 0.012:
            if mem["retries"] < MAX_RETRY:
                mem["retries"] += 1
                nxt = "retry_butter_home"
            else:
                nxt = "to_pan"
            mem["note"] = "butter grasp missed (width %.4f)" % width
        if prev == "sauce_lift" and width < 0.012:
            if mem["retries"] < MAX_RETRY:
                mem["retries"] += 1
                nxt = "retry_sauce_home"
            mem["note"] = "sauce grasp missed (width %.4f)" % width
        if prev in ("to_pour1", "to_pour2", "pour1", "untilt1") and width < 0.012:
            # bottle lost on the way: re-grasp from the current scene
            if mem["retries"] < MAX_RETRY:
                mem["retries"] += 1
                nxt = "retry_sauce_home"
                mem["note"] = "bottle lost before pouring"
        if prev == "pour1":
            mem["pours_done"] = 1
        if prev == "pour2":
            mem["pours_done"] = 2

    bx, by = mem["butter_xy"]
    px, py = mem["pan_xy"]
    sx, sy = mem["sauce_xy"]
    dx, dy = mem["drainer_xy"]
    qx, qy = px + POUR_OFF[0], py + POUR_OFF[1]

    if nxt == "butter_above":
        act = _mv(bx, by, 0.60, -1, max_steps=400, tol=0.005)
    elif nxt == "butter_descend":
        act = _mv(bx, by, Z_BUTTER_GRASP, -1, max_steps=250, tol=0.003, stall=True, gain=5.0)
    elif nxt == "butter_close":
        act = {"tool": "act_grip", "args": {"grip": 1.0, "steps": 30}}
    elif nxt == "butter_lift":
        act = _mv(bx, by, Z_TRAVEL - 0.02, 1, max_steps=250, tol=0.008)
    elif nxt == "to_pan":
        act = _mv(px, py, Z_TRAVEL, 1, max_steps=400, tol=0.005)
    elif nxt == "pan_descend":
        act = _mv(px, py, Z_RELEASE, 1, max_steps=250, tol=0.004)
    elif nxt == "butter_release":
        act = {"tool": "act_grip", "args": {"grip": -1.0, "steps": 25}}
    elif nxt == "pan_lift":
        act = _mv(px, py, Z_TRAVEL, -1, max_steps=250, tol=0.008)
    elif nxt == "sauce_above":
        act = _mv(sx, sy, Z_SAUCE_ABOVE, -1, max_steps=400, tol=0.004)
    elif nxt == "sauce_descend":
        act = _mv(sx, sy, Z_SAUCE_GRASP, -1, max_steps=250, tol=0.003, stall=True, gain=5.0)
    elif nxt == "sauce_close":
        act = {"tool": "act_grip", "args": {"grip": 1.0, "steps": 30}}
    elif nxt == "sauce_lift":
        act = _mv(sx, sy, Z_TRAVEL - 0.01, 1, max_steps=250, tol=0.008)
    elif nxt in ("to_pour1", "to_pour2"):
        act = _mv(qx, qy, Z_POUR, 1, max_steps=400, tol=0.004)
    elif nxt == "pour1":
        act = {"tool": "act_tilt", "args": {"tilt": POUR_ANGLE, "hold": 30, "grip": 1.0}}
    elif nxt == "pour2":
        act = {"tool": "act_tilt", "args": {"tilt": POUR_ANGLE, "hold": 150, "grip": 1.0}}
    elif nxt in ("untilt1", "untilt2"):
        act = {"tool": "act_tilt", "args": {"tilt": 0.0, "hold": 0, "grip": 1.0}}
    elif nxt == "to_drainer":
        act = _mv(dx, dy, 0.76, 1, max_steps=400, tol=0.005)
    elif nxt == "drainer_descend":
        act = _mv(dx, dy, 0.65, 1, max_steps=250, tol=0.004, stall=True)
    elif nxt == "drainer_release":
        act = {"tool": "act_grip", "args": {"grip": -1.0, "steps": 30}}
    elif nxt == "drainer_lift":
        act = _mv(dx, dy, 0.78, -1, max_steps=200, tol=0.008)
    elif nxt in ("retry_butter_home", "retry_sauce_home"):
        act = _mv(HOME[0], HOME[1], HOME[2], -1, max_steps=400, tol=0.01)
    else:
        nxt = "done"
        act = {"tool": "act_grip", "args": {"grip": -1.0, "steps": 100}}
    mem["phase"] = nxt
    return {"memory": mem, "act": act}


# ---------------------------------------------------------------- actions
def _norm_clip(a, lim):
    m = float(np.max(np.abs(a)))
    if m > lim:
        a = a * (lim / m)
    return a


def act_move(robot, x, y, z, grip, max_steps=300, tol=0.004, stall=False, gain=6.0, rot_gain=2.0):
    """Move the TCP to world (x, y, z) with the gripper pointing down (yaw 0), sending gripper `grip` (+1 close, -1 open) every step.
    Stops when within `tol` metres, on stall (if `stall`, i.e. contact) or after `max_steps`. Returns final tcp, steps and reason."""
    tgt = np.array([x, y, z], dtype=float)
    Rt = _target_R(0.0)
    hist = []
    reason = "timeout"
    n = 0
    for i in range(int(max_steps)):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        R = _rotvec_to_R(f.tcp[3:6])
        e = tgt - p
        er = _R_to_rotvec(Rt @ R.T)
        if float(np.linalg.norm(e)) < tol and float(np.linalg.norm(er)) < 0.05:
            reason = "reached"
            break
        if stall and len(hist) >= 20 and float(np.linalg.norm(p - hist[-10])) < 0.0015:
            reason = "stalled"
            break
        a = _norm_clip(gain * e, 1.0)
        ar = np.clip(rot_gain * er, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
        hist.append(p)
        n += 1
    f = robot.now()
    return {"reason": reason, "steps": n, "tcp": [float(v) for v in f.tcp[:3]],
            "err": float(np.linalg.norm(tgt - np.asarray(f.tcp[:3], dtype=float))), "width": _width(f)}


def act_grip(robot, grip, steps=30):
    """Hold the arm still (zero motion) while commanding the gripper (+1 close, -1 open) for `steps` steps. Returns finger width."""
    for i in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
    f = robot.now()
    return {"width": _width(f), "tcp": [float(v) for v in f.tcp[:3]]}


def act_tilt(robot, tilt, hold=30, grip=1.0, max_steps=120, rot_gain=2.0):
    """Rotate the gripper about world y to `tilt` rad relative to pointing-down (0 = upright, -0.785 = pour) with zero
    translation commands, then hold still for `hold` steps. Returns final tcp, reached angle error and width."""
    Rt = _target_R(float(tilt))
    n = 0
    reason = "timeout"
    for i in range(int(max_steps)):
        f = robot.now()
        R = _rotvec_to_R(f.tcp[3:6])
        er = _R_to_rotvec(Rt @ R.T)
        if float(np.linalg.norm(er)) < 0.02:
            reason = "reached"
            break
        ar = np.clip(rot_gain * er, -0.8, 0.8)
        robot.step(0.0, 0.0, 0.0, float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
        n += 1
    for i in range(int(hold)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
    f = robot.now()
    er = _R_to_rotvec(Rt @ _rotvec_to_R(f.tcp[3:6]).T)
    return {"reason": reason, "steps": n, "hold": int(hold), "tcp": [float(v) for v in f.tcp[:3]],
            "angle_err": float(np.linalg.norm(er)), "width": _width(f)}
