"""Policy for T12: open the middle drawer, put cookies in it, then chocolate.

All geometry is in world metres. Perception uses the fixed agentview RGB-D camera
(back-projected with its calibration). Control is a closed-loop P servo on the
measured TCP pose using OSC delta steps.
"""

DOWN = [math.pi, 0.0, 0.0]            # gripper pointing down (tcp axis-angle)
SIDE = [-math.pi / 2.0, 0.0, 0.0]     # gripper pointing +y (into cabinet front)

# Priors from demonstrations (used only when perception fails).
PRIOR_HANDLE = [0.137, 0.454, 0.607]
PRIOR_COOKIES = [0.217, -0.327]
PRIOR_CHOC = [-0.230, -0.325]
# Calibrated offsets perception -> grasp point (50 demos, see task_analysis.md)
HANDLE_DX = -0.1284     # handle x = cabinet top near-edge x (98th pct) + DX
HANDLE_DY = -0.0184     # handle y = cabinet top front-edge y (2nd pct) + DY
HANDLE_DZ = -0.105      # handle z = cabinet top z + DZ
COOKIE_OFF = [-0.0068, 0.0010]
CHOC_OFF = [-0.0143, 0.0032]
GRASP_Z = 0.428
TRAVEL_Z = 0.78
ELBOW_SLOW = -0.30      # joint-4 value above which motions are slowed
ELBOW_OK = -0.35        # settle until joint 4 is below this (bounded)
CLEAR = {"x": -0.05, "y": 0.0, "z": 0.75}
VIA = [0.0, 0.0]
BASE_XY = [-0.51, 0.0]   # robot base (fit of Panda FK to demo joints/TCP, scratch/fk.py)
R_DROP_COOKIES = 0.725   # horizontal reach limit for drops (R001-R013: r>=0.755 -> elbow at limit)
R_DROP_CHOC = 0.72
REACH_TILT_R = 0.70      # tilt the top-down grasp beyond this horizontal reach
TILT = 0.21              # rad (12 deg)


def _rodrigues(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the diagonal
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(th)) * th


def _yaw_down(yaw):
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
    return _rotvec(Rz @ _rodrigues(DOWN)).tolist()


def _world_points(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    Xc = np.stack([x, y, d], -1).reshape(-1, 3)
    Xw = (Xc - E[:, 3]) @ E[:, :3]
    Xw = Xw.reshape(H, W, 3)
    valid = d > 0.05
    return Xw, valid


def _perceive_scene(frame):
    """Locate the middle-drawer handle, cookies and chocolate from agentview RGB-D."""
    Xw, valid = _world_points(frame)
    X, Y, Z = Xw[..., 0], Xw[..., 1], Xw[..., 2]
    out = {"notes": []}
    # Cabinet top surface: flat patch at z ~0.71, y > 0.3
    m = valid & (Z > 0.69) & (Z < 0.74) & (Y > 0.3) & (Y < 0.8) & (X > -0.2) & (X < 0.5)
    n = int(m.sum())
    if n > 300:
        P = Xw[m]
        x98 = float(np.percentile(P[:, 0], 98))
        y2 = float(np.percentile(P[:, 1], 2))
        ztop = float(np.median(P[:, 2]))
        out["handle"] = [x98 + HANDLE_DX, y2 + HANDLE_DY, ztop + HANDLE_DZ]
        out["notes"].append("cabinet top pts %d" % n)
    else:
        out["handle"] = None
        out["notes"].append("cabinet not found (%d pts)" % n)
    for name, prior, ztab, off in (("cookies", PRIOR_COOKIES, 0.438, COOKIE_OFF),
                                   ("chocolate", PRIOR_CHOC, 0.445, CHOC_OFF)):
        dd = np.hypot(X - prior[0], Y - prior[1])
        mm = valid & (dd < 0.07) & (Z > ztab) & (Z < 0.52)
        k = int(mm.sum())
        if k >= 20:
            P = Xw[mm]
            # second pass: tighten around the found blob
            c = np.median(P[:, :2], 0)
            dd2 = np.hypot(X - c[0], Y - c[1])
            mm2 = valid & (dd2 < 0.05) & (Z > ztab) & (Z < 0.52)
            if int(mm2.sum()) >= 15:
                P = Xw[mm2]
            out[name] = [float(P[:, 0].mean() + off[0]), float(P[:, 1].mean() + off[1])]
            out[name + "_pts"] = int(len(P))
        else:
            out[name] = None
            out[name + "_pts"] = k
    return out


def _drawer_open_pts(frame, handle):
    """Dark (cabinet-coloured) points in the volume an opened drawer occupies."""
    Xw, valid = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.sum(-1) < 200
    hx, hy = handle[0], handle[1]
    m = valid & dark & (np.abs(Xw[..., 0] - hx) < 0.12) & (Xw[..., 1] > hy - 0.22) & \
        (Xw[..., 1] < hy - 0.04) & (Xw[..., 2] > 0.5) & (Xw[..., 2] < 0.68)
    return int(m.sum())


def _object_pts(frame, xy, ztab):
    Xw, valid = _world_points(frame)
    dd = np.hypot(Xw[..., 0] - xy[0], Xw[..., 1] - xy[1])
    m = valid & (dd < 0.05) & (Xw[..., 2] > ztab) & (Xw[..., 2] < 0.52)
    return int(m.sum())


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _servo(robot, pos, rot, grip, n=300, tol=0.006, rtol=0.05, vmax=1.0, kp=12.0, stall=40, ki=1.5):
    """Closed-loop PI move of the TCP to pos/rot (axis-angle). The integral term removes the
    steady-state sag of the OSC controller (seen in R001 while carrying a payload).
    Stops when within tolerance or when the error has not improved for `stall` steps.
    Returns (reached, steps)."""
    Rt = _rodrigues(rot)
    pos = np.asarray(pos, dtype=float)
    best = 1e9
    still = 0
    ie = np.zeros(3)
    iw = np.zeros(3)
    for i in range(int(n)):
        fr = robot.now()
        t = np.asarray(fr.tcp, dtype=float)
        e = pos - t[:3]
        w = _rotvec(Rt @ _rodrigues(t[3:]).T)
        en = float(np.linalg.norm(e))
        wn = float(np.linalg.norm(w))
        if en < tol and wn < rtol:
            return True, i
        score = en + 0.05 * wn
        if stall:
            if score < best - 0.0007:
                best = score
                still = 0
            else:
                still += 1
                if still >= stall:
                    return False, i
        # integrate only near the target (anti-windup)
        if en < 0.04:
            ie = np.clip(ie + e, -0.5 / max(ki, 1e-6), 0.5 / max(ki, 1e-6))
        if wn < 0.3:
            iw = np.clip(iw + w, -1.5, 1.5)
        # vmax limits the proportional (speed) part only; the integral part must be free to
        # cancel steady-state sag (R008: with the total clipped to 0.3 the loaded arm could not
        # descend to the drop height and stalled 6 cm high)
        a = np.clip(np.clip(kp * e, -vmax, vmax) + ki * ie, -1.0, 1.0)
        r = np.clip(2.0 * w + 0.2 * iw, -0.5, 0.5)
        # elbow (joint 4) near its upper limit -0.07 (arm stretched, R007 failure):
        # slow down so the controller's posture term can re-bend the elbow
        q4 = float(np.asarray(fr.joints, dtype=float)[3])
        if q4 > ELBOW_SLOW:
            sc = 0.35 if q4 > ELBOW_SLOW + 0.1 else 0.6
            a = a * sc
            r = r * sc
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return False, int(n)


def _hold(robot, grip, n):
    fr = robot.now()
    t = np.asarray(fr.tcp, dtype=float)
    for i in range(int(n)):
        fr = robot.now()
        e = t[:3] - np.asarray(fr.tcp, dtype=float)[:3]
        a = np.clip(10.0 * e, -0.3, 0.3)
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, float(grip))


def _settle(robot, grip, n=40):
    """Hold still while the elbow is near its limit, letting the posture recover. Returns steps."""
    k = 0
    while k < n and float(np.asarray(robot.now().joints, dtype=float)[3]) > ELBOW_OK:
        _hold(robot, grip, 5)
        k += 5
    return k


def _tcp(robot):
    return [round(float(v), 4) for v in np.asarray(robot.now().tcp, dtype=float)]


def act_open_drawer(robot, hx, hy, hz, pull=0.24):
    """Approach the middle-drawer handle horizontally (gripper pointing +y), grasp it,
    pull it out by `pull` metres along -y, release and back away. Returns the measured
    finger gap on the handle, the TCP travel during the pull and final pose."""
    res = {}
    ok1, _ = _servo(robot, [hx, hy - 0.12, hz + 0.01], SIDE, -1, n=250, tol=0.015, rtol=0.08)
    ok2, _ = _servo(robot, [hx, hy - 0.06, hz], SIDE, -1, n=150, tol=0.008, rtol=0.04)
    ok3, _ = _servo(robot, [hx, hy, hz], SIDE, -1, n=120, tol=0.004, rtol=0.04, vmax=0.5, stall=25)
    res["approach_ok"] = bool(ok1 and ok2)
    res["at_handle"] = _tcp(robot)
    _hold(robot, 1, 20)
    g = _gap(robot.now())
    res["gap_on_handle"] = round(g, 4)
    y0 = float(robot.now().tcp[1])
    _servo(robot, [hx, hy - pull, hz], SIDE, 1, n=200, tol=0.006, rtol=0.05, vmax=0.7)
    res["pull_travel"] = round(y0 - float(robot.now().tcp[1]), 4)
    res["gap_after_pull"] = round(_gap(robot.now()), 4)
    _hold(robot, -1, 15)
    t = np.asarray(robot.now().tcp, dtype=float)
    _servo(robot, [t[0], t[1] - 0.10, t[2] + 0.02], SIDE, -1, n=120, tol=0.01, rtol=0.1)
    res["final_tcp"] = _tcp(robot)
    return res


def _via(robot, x, y, grip, rot):
    """Long transits (|dy| > 0.3) pass over a folded-arm waypoint near the base to keep the
    elbow away from its limit (R007: stretched arm at the drawer could not travel to the
    chocolate)."""
    t = np.asarray(robot.now().tcp, dtype=float)
    if abs(t[1] - y) > 0.3:
        _servo(robot, [VIA[0], VIA[1], TRAVEL_Z], rot, grip, n=200, tol=0.03, rtol=0.15)
        return True
    return False


def _reach_rot(x, y, yaw=0.0):
    """Top-down grasp orientation; for targets near the arm's reach limit (horizontal distance
    from the base > REACH_TILT_R) the gripper is tilted by TILT rad so it points down and away
    from the base. This moves the wrist ~4 cm toward the shoulder. Without it the cookie grasp
    needs a shoulder-wrist distance of 0.717 m vs the 0.719 m maximum (scratch/fk.py) and the
    elbow repeatedly passed its singularity and locked at the joint-4 limit (R007, R013, R017)."""
    R0 = _rodrigues(_yaw_down(yaw))
    dx, dy = x - BASE_XY[0], y - BASE_XY[1]
    r = math.hypot(dx, dy)
    if r <= REACH_TILT_R:
        return _rotvec(R0).tolist(), 0.0
    ax = np.array([dy / r, -dx / r, 0.0])
    return _rotvec(_rodrigues(ax * TILT) @ R0).tolist(), TILT


def act_pick(robot, x, y, yaw=0.0, z=GRASP_Z, hover=0.60):
    """Pick an object lying on the table at (x, y) with a top-down grasp (tilted away from
    the base when near the reach limit): (via a folded waypoint if far), move above, settle,
    descend slowly, close, lift to TRAVEL_Z. Returns finger gap after closing and after lift,
    and elbow (joint 4) readings."""
    rot, tilt = _reach_rot(x, y, yaw)
    res = {"tilt": round(tilt, 3)}
    t = np.asarray(robot.now().tcp, dtype=float)
    if t[2] < TRAVEL_Z - 0.05 and abs(t[1] - y) > 0.3:
        _servo(robot, [t[0], t[1], TRAVEL_Z], rot, -1, n=100, tol=0.02, rtol=0.3, vmax=0.6)
    res["via"] = _via(robot, x, y, -1, rot)
    ok1, _ = _servo(robot, [x, y, hover], rot, -1, n=300, tol=0.008, rtol=0.04)
    res["settle1"] = _settle(robot, -1)
    ok2, _ = _servo(robot, [x, y, z + 0.04], rot, -1, n=200, tol=0.006, rtol=0.04, vmax=0.35)
    ok3, _ = _servo(robot, [x, y, z], rot, -1, n=100, tol=0.005, rtol=0.04, vmax=0.25, stall=15)
    res["above_ok"] = bool(ok1)
    res["at_grasp"] = _tcp(robot)
    _hold(robot, 1, 25)
    res["gap_closed"] = round(_gap(robot.now()), 4)
    t = np.asarray(robot.now().tcp, dtype=float)
    _servo(robot, [t[0], t[1], 0.65], rot, 1, n=150, tol=0.01, rtol=0.1, vmax=0.5)
    # rise the rest of the way while moving halfway toward the folded waypoint: lifting
    # straight up at the far cookie spot stalls at z~0.75 (R015, elbow near limit)
    _servo(robot, [t[0] + 0.5 * (VIA[0] - t[0]), t[1] + 0.5 * (VIA[1] - t[1]), TRAVEL_Z], rot, 1,
           n=150, tol=0.02, rtol=0.1, vmax=0.6)
    res["settle2"] = _settle(robot, 1)
    res["gap_lifted"] = round(_gap(robot.now()), 4)
    res["q4"] = round(float(np.asarray(robot.now().joints, dtype=float)[3]), 3)
    res["final_tcp"] = _tcp(robot)
    return res


def act_place(robot, x, y, z):
    """Carry the held object at TRAVEL_Z (via a folded waypoint if far) to above (x, y),
    settle, descend slowly to z, open the gripper and rise again.
    Returns finger gap before release and final pose."""
    res = {}
    res["via"] = _via(robot, x, y, 1, DOWN)
    ok1, _ = _servo(robot, [x, y, TRAVEL_Z], DOWN, 1, n=300, tol=0.01, rtol=0.06)
    res["settle1"] = _settle(robot, 1)
    ok2, _ = _servo(robot, [x, y, z], DOWN, 1, n=200, tol=0.007, rtol=0.05, vmax=0.3, stall=30)
    res["over_ok"] = bool(ok1)
    res["at_release"] = _tcp(robot)
    res["gap_before_release"] = round(_gap(robot.now()), 4)
    _hold(robot, -1, 20)
    _servo(robot, [x, y, TRAVEL_Z], DOWN, -1, n=150, tol=0.015, rtol=0.1, vmax=0.5)
    res["settle2"] = _settle(robot, -1)
    res["q4"] = round(float(np.asarray(robot.now().joints, dtype=float)[3]), 3)
    res["final_tcp"] = _tcp(robot)
    return res


def act_move(robot, x, y, z, side=False, grip=-1.0, n=200):
    """Move the TCP to (x, y, z) with the gripper pointing down (or +y if side)."""
    ok, k = _servo(robot, [x, y, z], SIDE if side else DOWN, grip, n=n, tol=0.01, rtol=0.06)
    return {"reached": bool(ok), "steps": k, "tcp": _tcp(robot)}


def act_wait(robot, n=40, grip=-1.0):
    """Hold the current pose for n steps (lets the scorer settle)."""
    _hold(robot, grip, n)
    return {"tcp": _tcp(robot), "gap": round(_gap(robot.now()), 4)}


def _drop_xy(h, kind):
    """Drop point inside the opened drawer, kept within the arm's comfortable reach:
    y = handle_y - 0.10 (cookies) / - 0.09 (chocolate); x as close to the nominal as
    the reach limit allows but not beyond the drawer's near side."""
    if kind == "cookies":
        y = h[1] - 0.10
        x_nom, x_min, rmax = h[0], h[0] - 0.06, R_DROP_COOKIES
    else:
        y = h[1] - 0.09
        x_nom, x_min, rmax = h[0] - 0.04, h[0] - 0.08, R_DROP_CHOC
    dy = y - BASE_XY[1]
    x_max = BASE_XY[0] + math.sqrt(max(rmax * rmax - dy * dy, 0.0))
    x = max(min(x_nom, x_max), x_min)
    return float(x), float(y)


def _at_clear(tcp):
    return abs(tcp[0] - CLEAR["x"]) < 0.03 and abs(tcp[1] - CLEAR["y"]) < 0.03 and abs(tcp[2] - CLEAR["z"]) < 0.04


def _mem(memory, **kw):
    keys = ["stage", "handle", "cookies", "chocolate", "drawer_tries", "cookie_tries",
            "choc_tries", "drawer_open_pts", "log"]
    out = {k: memory.get(k) for k in keys}
    out.update(kw)
    if out["stage"] is None:
        out["stage"] = "start"
    for k in ("drawer_tries", "cookie_tries", "choc_tries"):
        if out[k] is None:
            out[k] = 0
    if out["log"] is None:
        out["log"] = ""
    return out


def _note(mem, s):
    mem["log"] = (str(mem.get("log") or "") + " | " + s)[-1500:]
    return mem


def _parse(last):
    """Best-effort read of the previous act result (dict or JSON-ish text)."""
    if isinstance(last, dict):
        r = last.get("result", last)
        return r if isinstance(r, dict) else {}
    return {}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive -> open drawer -> verify -> pick/place cookies ->
    pick/place chocolate -> wait. Uses agentview RGB-D for localisation and
    finger gap / depth for verification."""
    mem = _mem(memory or {})
    fr = obs.current
    stage = mem["stage"]
    gap = _gap(fr)
    tcp = np.asarray(fr.tcp, dtype=float)

    if stage == "start":
        sc = _perceive_scene(fr)
        mem["handle"] = sc["handle"] if sc["handle"] is not None else list(PRIOR_HANDLE)
        mem["cookies"] = sc["cookies"] if sc["cookies"] is not None else list(PRIOR_COOKIES)
        mem["chocolate"] = sc["chocolate"] if sc["chocolate"] is not None else list(PRIOR_CHOC)
        _note(mem, "scene h=%s c=%s ch=%s %s" % (np.round(mem["handle"], 3).tolist(),
              np.round(mem["cookies"], 3).tolist(), np.round(mem["chocolate"], 3).tolist(), sc["notes"]))
        stage = "open_drawer"

    if stage == "open_drawer":
        h = mem["handle"]
        k = int(mem["drawer_tries"])
        # retries vary the approach height slightly
        dz = [0.0, 0.0, 0.006, -0.006][min(k, 3)]
        mem["drawer_tries"] = k + 1
        mem["stage"] = "check_drawer"
        return {"memory": mem, "act": {"tool": "act_open_drawer",
                                       "args": {"hx": h[0], "hy": h[1], "hz": h[2] + dz}}}

    if stage == "check_drawer":
        # need the arm out of the camera's view of the drawer: move toward start pose
        if not _at_clear(tcp):
            mem["stage"] = "check_drawer"
            return {"memory": mem, "act": {"tool": "act_move", "args": dict(CLEAR)}}
        n = _drawer_open_pts(fr, mem["handle"])
        mem["drawer_open_pts"] = n
        _note(mem, "drawer pts %d (try %d)" % (n, mem["drawer_tries"]))
        if n < 250 and mem["drawer_tries"] < 3:
            stage = "open_drawer"
            mem["stage"] = stage
            return _retry_drawer(mem)
        stage = "pick_cookies"

    if stage == "pick_cookies":
        c = mem["cookies"]
        mem["cookie_tries"] = int(mem["cookie_tries"]) + 1
        mem["stage"] = "check_cookies"
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": c[0], "y": c[1]}}}

    if stage == "check_cookies":
        if 0.012 < gap < 0.072:
            dx, dy = _drop_xy(mem["handle"], "cookies")
            mem["stage"] = "after_cookies"
            _note(mem, "cookies held gap %.3f drop (%.3f,%.3f)" % (gap, dx, dy))
            return {"memory": mem, "act": {"tool": "act_place",
                                           "args": {"x": dx, "y": dy, "z": 0.73}}}
        _note(mem, "cookies grasp missed gap %.3f" % gap)
        if mem["cookie_tries"] < 3:
            mem["stage"] = "repercieve_cookies"
            return {"memory": mem, "act": {"tool": "act_move", "args": dict(CLEAR)}}
        stage = "pick_choc"

    if stage == "repercieve_cookies":
        sc = _perceive_scene(fr)
        if sc["cookies"] is not None:
            mem["cookies"] = sc["cookies"]
        _note(mem, "cookies reperceived %s" % (sc["cookies"],))
        c = mem["cookies"]
        mem["cookie_tries"] = int(mem["cookie_tries"]) + 1
        mem["stage"] = "check_cookies"
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": c[0], "y": c[1]}}}

    if stage == "after_cookies":
        # cookies should have left their table spot; re-pick if still there
        n = _object_pts(fr, mem["cookies"], 0.438)
        _note(mem, "cookie spot pts after place %d" % n)
        if n >= 40 and mem["cookie_tries"] < 3:
            mem["stage"] = "repercieve_cookies"
            return perceive_decide(obs, goal, mem, last)
        stage = "pick_choc"

    if stage == "pick_choc":
        c = mem["chocolate"]
        mem["choc_tries"] = int(mem["choc_tries"]) + 1
        mem["stage"] = "check_choc"
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": c[0], "y": c[1]}}}

    if stage == "check_choc":
        if 0.012 < gap < 0.072:
            dx, dy = _drop_xy(mem["handle"], "chocolate")
            mem["stage"] = "done"
            _note(mem, "choc held gap %.3f drop (%.3f,%.3f)" % (gap, dx, dy))
            return {"memory": mem, "act": {"tool": "act_place",
                                           "args": {"x": dx, "y": dy, "z": 0.77}}}
        _note(mem, "choc grasp missed gap %.3f" % gap)
        if mem["choc_tries"] < 3:
            mem["stage"] = "repercieve_choc"
            return {"memory": mem, "act": {"tool": "act_move", "args": dict(CLEAR)}}
        stage = "done"

    if stage == "repercieve_choc":
        sc = _perceive_scene(fr)
        if sc["chocolate"] is not None:
            mem["chocolate"] = sc["chocolate"]
        _note(mem, "choc reperceived %s" % (sc["chocolate"],))
        c = mem["chocolate"]
        mem["choc_tries"] = int(mem["choc_tries"]) + 1
        mem["stage"] = "check_choc"
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": c[0], "y": c[1]}}}

    if stage == "done" and mem.get("chocolate") is not None and 0 < int(mem["choc_tries"]) < 3 \
            and float(tcp[1]) > 0.0:
        # arm is over the drawer: chocolate spot visible; if chocolate still there, retry
        n = _object_pts(fr, mem["chocolate"], 0.445)
        if n >= 30:
            _note(mem, "chocolate still on table (%d pts)" % n)
            mem["stage"] = "repercieve_choc"
            return perceive_decide(obs, goal, mem, last)
    mem["stage"] = "done"
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 60}}}


def _retry_drawer(mem):
    """Re-issue the drawer opening after a failed open check."""
    h = mem["handle"]
    k = int(mem["drawer_tries"])
    dz = [0.0, 0.0, 0.006, -0.006][min(k, 3)]
    mem["drawer_tries"] = k + 1
    mem["stage"] = "check_drawer"
    return {"memory": mem, "act": {"tool": "act_open_drawer",
                                   "args": {"hx": h[0], "hy": h[1], "hz": h[2] + dz}}}
