"""Code-as-Memory policy for T08_ChocolateFrypanPourSauceTwice (LIBERO / RoboMemArena).

Task: pick the chocolate, drop it into the frypan, pour the sauce bottle over
the pan twice (tilt ~45 deg about world -y, hold), then place the bottle into
the bowl drainer (fallback stage: every demo ended with success after the
second pour, before any drainer placement).

Perception: agentview depth + calibration -> world points; table plane z~0.43.
Frypan centre: circle fit to elevated rim points. Chocolate / bottle: elevated
blob centroids near their nominal positions. Drainer: elevated tall points at y>0.3.
Control: closed-loop P servo on measured TCP (OSC delta semantics: target =
current + 0.05*a), orientation error in world frame.
"""

# ---------------------------------------------------------------- constants
TABLE_Z = 0.43
# nominal (all 25 demos) visible-blob centroids and expert grasp points
CHOC_BLOB_NOM = [-0.119, -0.174]
CHOC_GRASP_OFF = [-0.0038, 0.0001]          # grasp xy - blob centroid xy
CHOC_GRASP_Z = 0.4378
BOTTLE_BLOB_NOM = [0.087, -0.175]
BOTTLE_GRASP_OFF = [-0.0068, -0.0121]
BOTTLE_GRASP_Z = 0.468
PAN_NOM = [0.003, 0.125]
DRAINER_NOM = [-0.125, 0.535]
# expert offsets relative to the fitted pan centre (std ~1.5 mm over 25 demos)
REL_OFF = [-0.0010, -0.0108]
REL_Z = 0.642
PREPOUR_OFF = [0.0001, -0.0365]
PREPOUR_Z = 0.724
POUR_OFF = [0.0335, -0.0217]
POUR_Z = 0.778
TILT_DEG = 45.0
LIFT_Z = 0.79
STAGES = [
    "init", "choc_above", "choc_descend", "choc_close", "choc_lift",
    "to_pan", "pan_descend", "choc_release", "pan_lift",
    "bottle_above", "bottle_descend", "bottle_close", "bottle_lift",
    "to_prepour", "pour", "upright", "to_drainer", "drainer_descend",
    "drainer_release", "retreat", "done",
]


# ---------------------------------------------------------------- math helpers
def _aa2mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * kx @ kx


def _mat2aa(m):
    c = (np.trace(m) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        b = (m + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(b)))
        ax = b[:, i] / math.sqrt(max(b[i, i], 1e-12))
        return ax * th
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _target_R(tilt_deg):
    """Gripper pointing down (yaw 0), then rotated about world -y by tilt_deg."""
    down = np.diag([1.0, -1.0, -1.0])
    t = -math.radians(float(tilt_deg))
    ry = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
    return ry @ down


def _tilt_of(tcp):
    z = _aa2mat(tcp[3:6])[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(z[2])))))


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


# ---------------------------------------------------------------- perception
def _world_points(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    D = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * D, (v - K[1, 2]) / K[1, 1] * D, D], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, D > 0


def _circle_fit(px, py, cx0, cy0):
    """Grid search for the rim circle maximizing inliers; coarse then fine."""
    best = (-1, cx0, cy0, 0.095)
    for step, span, radii in ((0.005, 0.10, np.arange(0.080, 0.116, 0.005)), (0.001, 0.006, None)):
        bx, by, br = best[1], best[2], best[3]
        rs = radii if radii is not None else np.arange(br - 0.004, br + 0.0041, 0.001)
        cxs = np.arange(bx - span, bx + span + 1e-9, step)
        cys = np.arange(by - span, by + span + 1e-9, step)
        for cx in cxs:
            dx2 = (px - cx) ** 2
            for cy in cys:
                d = np.sqrt(dx2 + (py - cy) ** 2)
                for r in rs:
                    s = int((np.abs(d - r) < 0.01).sum())
                    if s > best[0]:
                        best = (s, float(cx), float(cy), float(r))
    return best


def _detect_scene(frame):
    pw, ok = _world_points(frame)
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    out = {"ok_pan": False, "ok_choc": False, "ok_bottle": False, "ok_drainer": False}
    # frypan rim (handle/pan bottom excluded by the inlier band)
    m = ok & (z > TABLE_Z + 0.007) & (z < 0.52) & (x > -0.25) & (x < 0.30) & (y > -0.08) & (y < 0.42)
    pan = list(PAN_NOM)
    if int(m.sum()) > 60:
        s, cx, cy, r = _circle_fit(x[m], y[m], 0.0, 0.15)
        if s > 80 and math.hypot(cx - PAN_NOM[0], cy - PAN_NOM[1]) < 0.12:
            pan = [cx, cy]
            out["ok_pan"] = True
            out["pan_inliers"] = s
            out["pan_r"] = r
    out["pan"] = pan
    # chocolate and bottle blobs near nominal places
    for key, nom, off, gz, zmax in (("choc", CHOC_BLOB_NOM, CHOC_GRASP_OFF, CHOC_GRASP_Z, 0.50),
                                    ("bottle", BOTTLE_BLOB_NOM, BOTTLE_GRASP_OFF, BOTTLE_GRASP_Z, 0.56)):
        mm = ok & (z > TABLE_Z + 0.007) & (z < zmax) & (np.hypot(x - nom[0], y - nom[1]) < 0.06)
        c = list(nom)
        if int(mm.sum()) > 15:
            c = [float(x[mm].mean()), float(y[mm].mean())]
            out["ok_" + key] = True
        out[key] = [c[0] + off[0], c[1] + off[1], gz]
    # drainer: tall elevated points on the right side of the table
    md = ok & (z > 0.50) & (z < 0.65) & (y > 0.30) & (y < 0.80) & (x > -0.40) & (x < 0.30)
    dr = list(DRAINER_NOM)
    dtop = 0.594
    if int(md.sum()) > 200:
        dr = [float((np.percentile(x[md], 2) + np.percentile(x[md], 98)) / 2),
              float((np.percentile(y[md], 2) + np.percentile(y[md], 98)) / 2)]
        dtop = float(np.percentile(z[md], 99))
        out["ok_drainer"] = True
    out["drainer"] = [dr[0], dr[1], dtop]
    return out


# ---------------------------------------------------------------- control
def _servo(robot, target, tilt_deg, grip, tol=0.005, rot_tol_deg=2.0, max_steps=300,
           settle=3, vmax=1.0, hold_after=0):
    """Closed-loop move of the TCP to `target` with orientation _target_R(tilt_deg)."""
    tgt = np.asarray(target, dtype=float)
    Rt = _target_R(tilt_deg)
    good = 0
    steps = 0
    held = 0
    fr = robot.now()
    while steps < max_steps:
        tcp = np.asarray(fr.tcp, dtype=float)
        ep = tgt - tcp[:3]
        er = _mat2aa(Rt @ _aa2mat(tcp[3:6]).T)
        dist = float(np.linalg.norm(ep))
        rerr = math.degrees(float(np.linalg.norm(er)))
        if dist < tol and rerr < rot_tol_deg:
            good += 1
        else:
            good = 0
        if good >= settle:
            if held >= hold_after:
                break
            held += 1
        a = ep * 20.0
        n = float(np.max(np.abs(a)))
        if n > vmax:
            a = a * (vmax / n)
        r = er / 0.5 * 1.5
        rn = float(np.max(np.abs(r)))
        if rn > 0.8:
            r = r * (0.8 / rn)
        fr = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
        steps += 1
    tcp = np.asarray(fr.tcp, dtype=float)
    return {
        "steps": steps,
        "converged": bool(good >= settle),
        "pos_err": float(np.linalg.norm(tgt - tcp[:3])),
        "tcp": [float(v) for v in tcp[:3]],
        "tilt_deg": _tilt_of(tcp),
        "width": _width(fr),
    }


# ---------------------------------------------------------------- act tools
def act_move(robot, x, y, z, tilt_deg=0.0, grip=-1.0, tol=0.005, max_steps=300, vmax=1.0, hold_after=0):
    """Servo TCP to world (x,y,z) with gripper down tilted by tilt_deg about world -y.
    grip: -1 keep open, +1 keep closed. Returns steps, converged, pos_err, tcp, tilt_deg, width."""
    return _servo(robot, [x, y, z], tilt_deg, grip, tol=tol, max_steps=max_steps, vmax=vmax,
                  hold_after=hold_after)


def act_grip(robot, close=True, steps=25):
    """Hold the current TCP pose and close (close=True) or open the gripper for `steps` steps.
    Returns final finger width (m)."""
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    hold = tcp[:3].copy()
    g = 1.0 if close else -1.0
    Rt = _aa2mat(tcp[3:6])
    for _ in range(int(steps)):
        t = np.asarray(fr.tcp, dtype=float)
        a = np.clip((hold - t[:3]) * 20.0, -0.5, 0.5)
        r = np.clip(_mat2aa(Rt @ _aa2mat(t[3:6]).T) / 0.5 * 1.5, -0.5, 0.5)
        fr = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
    return {"width": _width(fr), "tcp": [float(v) for v in np.asarray(fr.tcp)[:3]]}


def act_pour(robot, x, y, z, tilt_deg=45.0, hold_steps=60, max_steps=400):
    """Keep the gripper closed, tilt the held bottle to tilt_deg at TCP (x,y,z) and hold the
    tilt (within 5 deg) for hold_steps steps. Returns steps, steps held, final tilt."""
    tgt = np.asarray([x, y, z], dtype=float)
    Rt = _target_R(tilt_deg)
    fr = robot.now()
    steps = 0
    held = 0
    while steps < max_steps and held < hold_steps:
        tcp = np.asarray(fr.tcp, dtype=float)
        ep = tgt - tcp[:3]
        er = _mat2aa(Rt @ _aa2mat(tcp[3:6]).T)
        a = ep * 20.0
        n = float(np.max(np.abs(a)))
        if n > 1.0:
            a = a / n
        r = er / 0.5 * 1.5
        rn = float(np.max(np.abs(r)))
        if rn > 0.8:
            r = r * (0.8 / rn)
        fr = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), 1.0)
        steps += 1
        if abs(_tilt_of(np.asarray(fr.tcp, dtype=float)) - tilt_deg) < 5.0:
            held += 1
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"steps": steps, "held": held, "tilt_deg": _tilt_of(tcp),
            "tcp": [float(v) for v in tcp[:3]], "width": _width(fr)}


def act_wait(robot, steps=20, grip=-1.0):
    """Hold the current pose for `steps` steps (gripper -1 open / +1 closed)."""
    fr = robot.now()
    hold = np.asarray(fr.tcp, dtype=float)[:3].copy()
    for _ in range(int(steps)):
        t = np.asarray(fr.tcp, dtype=float)
        a = np.clip((hold - t[:3]) * 20.0, -0.5, 0.5)
        fr = robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, float(grip))
    return {"waited": int(steps), "width": _width(fr)}


# ---------------------------------------------------------------- decision
def _parse_int(text, key):
    """Integer following `key` in the previous act's result text, or None."""
    if not isinstance(text, str):
        return None
    i = text.find(key)
    if i < 0:
        return None
    j = i + len(key)
    digits = ""
    while j < len(text) and text[j].isdigit():
        digits += text[j]
        j += 1
    return int(digits) if digits else None


def _relocate(frame, est, zmax):
    """Centroid [x, y] of elevated table points within 6 cm of the grasp estimate, or None."""
    pw, ok = _world_points(frame)
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    mm = ok & (z > TABLE_Z + 0.007) & (z < zmax) & (np.hypot(x - est[0], y - est[1]) < 0.06)
    if int(mm.sum()) < 15:
        return None
    return [float(x[mm].mean()), float(y[mm].mean())]


def _num(v, d):
    return d if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Stage machine. Memory keeps the scene estimate from the first frame, the current stage,
    pour count and retry counters; each decision issues one closed-loop act tool."""
    mem = {
        "stage": _num(memory.get("stage"), "init"),
        "pan": memory.get("pan"),
        "drainer": memory.get("drainer"),
        "choc": memory.get("choc"),
        "bottle": memory.get("bottle"),
        "pours_done": int(_num(memory.get("pours_done"), 0)),
        "retries": int(_num(memory.get("retries"), 0)),
        "decisions": int(_num(memory.get("decisions"), 0)) + 1,
        "notes": _num(memory.get("notes"), ""),
    }
    fr = obs.current
    tcp = np.asarray(fr.tcp, dtype=float)
    width = _width(fr)
    stage = mem["stage"]
    failed = isinstance(last, str) and ("the tool failed" in last or "-> error" in last)

    if stage == "init" or mem["pan"] is None or mem["choc"] is None or mem["bottle"] is None:
        sc = _detect_scene(fr)
        mem["pan"] = [float(v) for v in sc["pan"]]
        mem["drainer"] = [float(v) for v in sc["drainer"]]
        mem["choc"] = [float(v) for v in sc["choc"]]
        mem["bottle"] = [float(v) for v in sc["bottle"]]
        mem["notes"] = "detect pan=%s choc=%s bottle=%s drainer=%s" % (
            sc["ok_pan"], sc["ok_choc"], sc["ok_bottle"], sc["ok_drainer"])
        if stage == "init":
            stage = "choc_above"

    pan, choc, bottle, dr = mem["pan"], mem["choc"], mem["bottle"], mem["drainer"]
    if dr is None:
        dr = [DRAINER_NOM[0], DRAINER_NOM[1], 0.594]
    rel = [pan[0] + REL_OFF[0], pan[1] + REL_OFF[1]]
    pre = [pan[0] + PREPOUR_OFF[0], pan[1] + PREPOUR_OFF[1], PREPOUR_Z]
    pour = [pan[0] + POUR_OFF[0], pan[1] + POUR_OFF[1], POUR_Z]

    # ---- transitions that depend on the outcome of the previous act (measured now)
    if stage == "choc_close_check":
        if 0.008 < width < 0.07:
            stage = "choc_lift"
        else:
            stage = "choc_regrasp"
    elif stage == "choc_lift_check":
        stage = "to_pan" if 0.008 < width < 0.07 else "choc_regrasp"
    elif stage == "bottle_close_check":
        stage = "bottle_lift" if 0.025 < width < 0.075 else "bottle_regrasp"
    elif stage == "bottle_lift_check":
        stage = "to_prepour" if 0.025 < width < 0.075 else "bottle_regrasp"

    if stage in ("choc_regrasp", "bottle_regrasp"):
        mem["retries"] += 1
        if mem["retries"] > 3:
            stage = "done"
        else:
            nxt = "choc_above" if stage == "choc_regrasp" else "bottle_above"
            mem["stage"] = nxt
            return {"memory": mem, "act": {"tool": "act_move", "args": {
                "x": float(tcp[0]), "y": float(tcp[1]), "z": float(max(tcp[2], 0.60)),
                "tilt_deg": 0.0, "grip": -1.0, "tol": 0.01, "max_steps": 150}}}

    if stage in ("choc_above", "bottle_above") and mem["retries"] > 0:
        # after a failed grasp, re-locate the object near its previous estimate (arm is lifted)
        key = "choc" if stage == "choc_above" else "bottle"
        found = _relocate(fr, mem[key], 0.50 if key == "choc" else 0.56)
        if found is not None:
            off = CHOC_GRASP_OFF if key == "choc" else BOTTLE_GRASP_OFF
            mem[key] = [found[0] + off[0], found[1] + off[1], mem[key][2]]
        choc, bottle = mem["choc"], mem["bottle"]

    act = None
    nxt = stage
    if stage == "choc_above":
        act = ("act_move", {"x": choc[0], "y": choc[1], "z": choc[2] + 0.13, "grip": -1.0, "tol": 0.006})
        nxt = "choc_descend"
    elif stage == "choc_descend":
        act = ("act_move", {"x": choc[0], "y": choc[1], "z": choc[2], "grip": -1.0, "tol": 0.003,
                            "vmax": 0.4, "max_steps": 250})
        nxt = "choc_close"
    elif stage == "choc_close":
        act = ("act_grip", {"close": True, "steps": 25})
        nxt = "choc_close_check"
    elif stage == "choc_lift":
        act = ("act_move", {"x": choc[0], "y": choc[1], "z": LIFT_Z, "grip": 1.0, "tol": 0.01})
        nxt = "choc_lift_check"
    elif stage == "to_pan":
        act = ("act_move", {"x": rel[0], "y": rel[1], "z": LIFT_Z, "grip": 1.0, "tol": 0.006})
        nxt = "pan_descend"
    elif stage == "pan_descend":
        act = ("act_move", {"x": rel[0], "y": rel[1], "z": REL_Z, "grip": 1.0, "tol": 0.004,
                            "vmax": 0.7, "hold_after": 3})
        nxt = "choc_release"
    elif stage == "choc_release":
        act = ("act_grip", {"close": False, "steps": 20})
        nxt = "pan_lift"
    elif stage == "pan_lift":
        act = ("act_move", {"x": rel[0], "y": rel[1], "z": 0.80, "grip": -1.0, "tol": 0.01})
        nxt = "bottle_above"
    elif stage == "bottle_above":
        act = ("act_move", {"x": bottle[0], "y": bottle[1], "z": bottle[2] + 0.15, "grip": -1.0, "tol": 0.005})
        nxt = "bottle_descend"
    elif stage == "bottle_descend":
        act = ("act_move", {"x": bottle[0], "y": bottle[1], "z": bottle[2], "grip": -1.0, "tol": 0.003,
                            "vmax": 0.4, "max_steps": 250})
        nxt = "bottle_close"
    elif stage == "bottle_close":
        act = ("act_grip", {"close": True, "steps": 30})
        nxt = "bottle_close_check"
    elif stage == "bottle_lift":
        act = ("act_move", {"x": bottle[0], "y": bottle[1], "z": LIFT_Z, "grip": 1.0, "tol": 0.01})
        nxt = "bottle_lift_check"
    elif stage == "to_prepour":
        act = ("act_move", {"x": pre[0], "y": pre[1], "z": pre[2], "grip": 1.0, "tol": 0.006})
        nxt = "pour"
    elif stage == "pour":
        if width < 0.02:
            # bottle lost (fingers closed on nothing): nothing left to pour; open and idle
            nxt = "done"
            act = ("act_wait", {"steps": 20, "grip": -1.0})
        else:
            hold = 60 if mem["pours_done"] == 0 else 120
            act = ("act_pour", {"x": pour[0], "y": pour[1], "z": pour[2], "tilt_deg": TILT_DEG,
                                "hold_steps": hold})
            nxt = "upright"
    elif stage == "upright":
        held = _parse_int(last, '"held": ')
        if not failed and held is not None and held >= 30:
            mem["pours_done"] += 1
        act = ("act_move", {"x": pre[0], "y": pre[1], "z": pre[2], "grip": 1.0, "tol": 0.008,
                            "hold_after": 25, "max_steps": 220})
        nxt = "pour" if mem["pours_done"] < 2 else "to_drainer"
    elif stage == "to_drainer":
        act = ("act_move", {"x": dr[0], "y": dr[1], "z": max(dr[2] + 0.17, 0.76), "grip": 1.0, "tol": 0.008})
        nxt = "drainer_descend"
    elif stage == "drainer_descend":
        act = ("act_move", {"x": dr[0], "y": dr[1], "z": dr[2] + 0.07, "grip": 1.0, "tol": 0.005,
                            "vmax": 0.5})
        nxt = "drainer_release"
    elif stage == "drainer_release":
        act = ("act_grip", {"close": False, "steps": 20})
        nxt = "retreat"
    elif stage == "retreat":
        act = ("act_move", {"x": float(tcp[0]), "y": float(tcp[1]), "z": 0.85, "grip": -1.0, "tol": 0.01})
        nxt = "done"
    else:
        act = ("act_wait", {"steps": 50, "grip": -1.0 if width < 0.02 or stage == "done" else 1.0})
        nxt = "done"
    mem["stage"] = nxt
    return {"memory": mem, "act": {"tool": act[0], "args": act[1]}}
