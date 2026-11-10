"""Pour tomato sauce over cookies twice (and place bottle in drainer as fallback).

Perception: agentview RGB-D back-projected to world points (calibration supplied
per frame). Bottle = highest compact cluster above the table near the robot;
cookies = flat box points on the table at the left-front; drainer = tall object
at large +y. Control: proportional OSC_POSE delta control on measured TCP.
"""

TOP_DOWN = [math.pi, 0.0, 0.0]
GRASP_OFF = [0.004, -0.011]       # grasp tcp xy minus bottle-top median xy (50 demos, std 3 mm)
GRASP_BELOW_TOP = 0.034           # grasp tcp z below highest bottle point
POUR_OFF = [-0.027, 0.0145]       # pour-start tcp xy minus cookie-box median xy
POUR_Z = 0.736                    # pour-start tcp height (demos: 0.735-0.738)
DEFAULT_COOKIE = [0.224, -0.328]
DEFAULT_DRAINER = [-0.14, 0.535, 0.594]
TILT_ANGLE = -0.785               # rotation about world y (demos: -0.79 rad)


def _rv2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _mat2rv(R):
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    return th / (2 * math.sin(th)) * w


def _rot_err(target_rv, cur_rv):
    return _mat2rv(_rv2mat(target_rv) @ _rv2mat(cur_rv).T)


def _cloud(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1).reshape(-1, 3)
    W = (P - E[:, 3]) @ E[:, :3]
    return W, (d.reshape(-1) > 0)


def _detect(frame):
    """Return dict with bottle top (x,y,zmax,n), cookie xy, drainer xyz, table z."""
    W, ok = _cloud(frame)
    X, Y, Z = W[:, 0], W[:, 1], W[:, 2]
    out = {"bottle": None, "cookie": None, "drainer": None, "table_z": None}
    tab = ok & (X > -0.3) & (X < 0.35) & (Y > -0.45) & (Y < 0.3) & (Z > 0.3) & (Z < 0.6)
    if tab.sum() > 50:
        out["table_z"] = float(np.percentile(Z[tab], 30))
    reg = ok & (X > -0.35) & (X < 0.4) & (Y > -0.45) & (Y < 0.3) & (Z > 0.455) & (Z < 0.6)
    p = W[reg]
    if len(p) >= 15:
        zmax = float(p[:, 2].max())
        c = np.median(p[p[:, 2] > zmax - 0.015][:, :2], 0)
        # refine on the compact cluster around that estimate (rejects stray points elsewhere)
        cl = p[np.linalg.norm(p[:, :2] - c, axis=1) < 0.05]
        if len(cl) >= 10:
            zmax = float(cl[:, 2].max())
            c = np.median(cl[cl[:, 2] > zmax - 0.015][:, :2], 0)
            out["bottle"] = [float(c[0]), float(c[1]), zmax, int(len(cl))]
    ck = ok & (X > 0.1) & (X < 0.4) & (Y > -0.5) & (Y < -0.15) & (Z > 0.437) & (Z < 0.47)
    if ck.sum() >= 30:
        c = np.median(W[ck][:, :2], 0)
        out["cookie"] = [float(c[0]), float(c[1])]
    dr = ok & (Y > 0.3) & (Y < 0.8) & (X > -0.4) & (X < 0.3) & (Z > 0.45) & (Z < 0.7)
    if dr.sum() >= 100:
        q = W[dr]
        out["drainer"] = [float(np.median(q[:, 0])), float(np.median(q[:, 1])), float(np.percentile(q[:, 2], 95))]
    return out


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _mem_default():
    return {"phase": "start", "bottle": None, "cookie": None, "pour_start": None,
            "grasp_attempts": 0, "pours_done": 0, "extra_pours": 0, "note": ""}


def perceive_decide(obs, goal, memory, last):
    """Phase machine: detect -> approach -> descend -> close -> lift -> transport -> pour x2 -> (fallback drainer)."""
    m = _mem_default()
    for k in m:
        if memory.get(k) is not None:
            m[k] = memory[k]
    fr = obs.current
    tcp = np.asarray(fr.tcp, dtype=float)
    w = _width(fr)
    ph = m["phase"]
    act = {"tool": "act_hold", "args": {"steps": 5, "gripper": -1.0}}

    def mv(x, y, z, g, steps=200, tol=0.004, gain=10.0, vmax=1.0, hold_rot=True):
        return {"tool": "act_move", "args": {"x": float(x), "y": float(y), "z": float(z), "gripper": float(g),
                                              "max_steps": int(steps), "tol": float(tol), "gain": float(gain),
                                              "vmax": float(vmax), "hold_rot": bool(hold_rot)}}

    if ph in ("start", "redetect"):
        det = _detect(fr)
        if det["cookie"] is not None and abs(det["cookie"][0] - DEFAULT_COOKIE[0]) < 0.08 and abs(det["cookie"][1] - DEFAULT_COOKIE[1]) < 0.08:
            m["cookie"] = det["cookie"]
        elif m["cookie"] is None:
            m["cookie"] = list(DEFAULT_COOKIE)
        m["pour_start"] = [m["cookie"][0] + POUR_OFF[0], m["cookie"][1] + POUR_OFF[1], POUR_Z]
        b = det["bottle"]
        if b is None or b[2] < 0.48:
            m["note"] = "bottle not detected upright: %s" % (b,)
            m["phase"] = "redetect"
            m["grasp_attempts"] = int(m["grasp_attempts"]) + 1
            if m["grasp_attempts"] > 4:
                act = {"tool": "act_hold", "args": {"steps": 20, "gripper": -1.0}}
            else:
                act = mv(-0.05, 0.0, 0.70, -1.0, steps=120)
        else:
            gx, gy = b[0] + GRASP_OFF[0], b[1] + GRASP_OFF[1]
            gz = b[2] - GRASP_BELOW_TOP
            m["bottle"] = [gx, gy, gz]
            m["note"] = "bottle top %s cookie %s" % ([round(v, 3) for v in b[:3]], [round(v, 3) for v in m["cookie"]])
            m["phase"] = "approach"
            act = mv(gx, gy, max(gz + 0.10, 0.58), -1.0, steps=150, tol=0.006)
    elif ph == "approach":
        gx, gy, gz = m["bottle"]
        m["phase"] = "descend"
        act = mv(gx, gy, gz, -1.0, steps=150, tol=0.003, gain=8.0, vmax=0.6)
    elif ph == "descend":
        m["phase"] = "close"
        act = {"tool": "act_gripper", "args": {"close": True, "steps": 25}}
    elif ph == "close":
        if w > 0.02:
            m["phase"] = "lift"
            act = mv(tcp[0], tcp[1], 0.80, 1.0, steps=120, tol=0.01)
        else:
            m["note"] = "grasp missed (width %.4f); reopen and retry" % w
            m["phase"] = "redetect_lift"
            act = {"tool": "act_gripper", "args": {"close": False, "steps": 15}}
    elif ph == "redetect_lift":
        m["phase"] = "redetect"
        m["grasp_attempts"] = int(m["grasp_attempts"]) + 1
        act = mv(-0.05, 0.0, 0.70, -1.0, steps=150)
    elif ph == "lift":
        if w > 0.02:
            m["phase"] = "transport"
            ps = m["pour_start"]
            act = mv(ps[0], ps[1], ps[2], 1.0, steps=200, tol=0.003)
        else:
            m["note"] = "bottle lost during lift (width %.4f)" % w
            m["phase"] = "redetect_lift"
            act = {"tool": "act_gripper", "args": {"close": False, "steps": 15}}
    elif ph == "transport":
        if w > 0.02:
            m["phase"] = "pour"
            hold = 35 if int(m["pours_done"]) == 0 else 90
            act = {"tool": "act_pour", "args": {"angle": TILT_ANGLE, "hold": hold, "untilt": int(m["pours_done"]) == 0}}
        else:
            m["note"] = "bottle lost during transport"
            m["phase"] = "redetect_lift"
            act = {"tool": "act_gripper", "args": {"close": False, "steps": 15}}
    elif ph == "pour":
        m["pours_done"] = int(m["pours_done"]) + 1
        if w <= 0.02:
            m["note"] = "bottle lost during pour"
            m["phase"] = "redetect_lift"
            act = {"tool": "act_gripper", "args": {"close": False, "steps": 15}}
        elif int(m["pours_done"]) < 2:
            m["phase"] = "transport"
            ps = m["pour_start"]
            act = mv(ps[0], ps[1], ps[2], 1.0, steps=80, tol=0.003)
        else:
            # two pours done but episode still running: untilt and try once more, then drainer
            m["phase"] = "untilt"
            act = {"tool": "act_rotate", "args": {"rx": TOP_DOWN[0], "ry": TOP_DOWN[1], "rz": TOP_DOWN[2], "gripper": 1.0, "max_steps": 60}}
    elif ph == "untilt":
        if int(m["extra_pours"]) < 1 and w > 0.02:
            m["extra_pours"] = int(m["extra_pours"]) + 1
            m["pours_done"] = 1
            m["phase"] = "transport"
            ps = m["pour_start"]
            act = mv(ps[0], ps[1], ps[2], 1.0, steps=80, tol=0.003)
        else:
            m["phase"] = "to_drainer"
            d = DEFAULT_DRAINER
            act = mv(d[0], d[1], 0.80, 1.0, steps=250, tol=0.01)
    elif ph == "to_drainer":
        m["phase"] = "lower_drainer"
        d = DEFAULT_DRAINER
        act = mv(d[0], d[1], 0.68, 1.0, steps=100, tol=0.01)
    elif ph == "lower_drainer":
        m["phase"] = "release"
        act = {"tool": "act_gripper", "args": {"close": False, "steps": 20}}
    elif ph == "release":
        m["phase"] = "done"
        act = mv(tcp[0], tcp[1], 0.85, -1.0, steps=60, tol=0.01)
    else:
        act = {"tool": "act_hold", "args": {"steps": 50, "gripper": -1.0}}
    return {"memory": m, "act": act}


def _rot_cmd(target_rv, cur_rv, gain=2.0, vmax=0.8):
    e = _rot_err(target_rv, cur_rv)
    return np.clip(gain * e, -vmax, vmax), float(np.linalg.norm(e))


def act_move(robot, x, y, z, gripper=-1.0, max_steps=200, tol=0.004, gain=10.0, vmax=1.0, hold_rot=True):
    """P-control TCP to world (x,y,z); keeps top-down orientation if hold_rot. Returns final tcp, error, steps, finger width."""
    tgt = np.array([x, y, z], dtype=float)
    n = 0
    err = 1.0
    hist = []
    stalled = False
    best = 1e9
    best_n = 0
    for n in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        e = tgt - tcp[:3]
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        hist.append(err)
        # stalled close to the target (steady-state offset under gravity/contact): stop
        if len(hist) > 15 and hist[-16] - err < 0.0005 and err < 0.02:
            stalled = True
            break
        # no new best error for 25 steps (e.g. oscillating at a reach limit): stop
        if err < best - 0.0005:
            best = err
            best_n = n
        elif n - best_n > 25 and err < 0.06:
            stalled = True
            break
        dp = np.clip(gain * e, -vmax, vmax)
        if hold_rot:
            dr, _ = _rot_cmd(TOP_DOWN, tcp[3:], gain=1.5, vmax=0.3)
        else:
            dr = np.zeros(3)
        robot.step(dp[0], dp[1], dp[2], dr[0], dr[1], dr[2], gripper)
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"tcp": tcp[:3], "err": float(np.linalg.norm(tgt - tcp[:3])), "steps": n, "stalled": stalled, "width": _width(fr)}


def act_gripper(robot, close=True, steps=25):
    """Hold the arm still and close (+1) or open (-1) the gripper for `steps` steps. Returns finger width."""
    g = 1.0 if close else -1.0
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, g)
    fr = robot.now()
    return {"width": _width(fr), "tcp": np.asarray(fr.tcp)[:3]}


def act_hold(robot, steps=10, gripper=-1.0):
    """Zero motion for `steps` steps with the given gripper sign."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, gripper)
    fr = robot.now()
    return {"width": _width(fr), "tcp": np.asarray(fr.tcp)}


def act_rotate(robot, rx, ry, rz, gripper=1.0, max_steps=60):
    """Rotate (no translation command) toward world axis-angle (rx,ry,rz)."""
    tgt = [rx, ry, rz]
    n = 0
    for n in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        dr, e = _rot_cmd(tgt, tcp[3:])
        if e < 0.01:
            break
        robot.step(0, 0, 0, dr[0], dr[1], dr[2], gripper)
    fr = robot.now()
    return {"tcp": np.asarray(fr.tcp), "steps": n, "width": _width(fr)}


def act_pour(robot, angle=-0.785, hold=35, untilt=True):
    """Tilt the held bottle about world y by `angle` (rotation-only P control, gain 2, clip 0.8),
    hold `hold` steps, and optionally rotate back to the start orientation. Gripper kept closed."""
    start = np.asarray(robot.now().tcp, dtype=float)
    R0 = _rv2mat(start[3:])
    c, s = math.cos(angle), math.sin(angle)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    tgt = _mat2rv(Ry @ R0)
    n1 = 0
    for n1 in range(60):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        dr, e = _rot_cmd(tgt, tcp[3:])
        if e < 0.01:
            break
        robot.step(0, 0, 0, dr[0], dr[1], dr[2], 1.0)
    tilted = np.asarray(robot.now().tcp, dtype=float)
    for _ in range(int(hold)):
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
    n2 = 0
    if untilt:
        back = _mat2rv(R0)
        for n2 in range(60):
            tcp = np.asarray(robot.now().tcp, dtype=float)
            dr, e = _rot_cmd(back, tcp[3:])
            if e < 0.01:
                break
            robot.step(0, 0, 0, dr[0], dr[1], dr[2], 1.0)
    fr = robot.now()
    return {"tilt_steps": n1, "tilted_tcp": tilted, "untilt_steps": n2, "tcp": np.asarray(fr.tcp), "width": _width(fr)}
