"""Policy for T13: open the middle drawer, put cookies in it, then put butter in it.

Perception: agentview RGB-D back-projected to world points (calibration supplied per frame).
Control: closed-loop P-control of the measured TCP pose through OSC delta steps.
All geometric offsets were calibrated on the 50 assigned expert demonstrations (see policy.md).
"""

# ---------------------------------------------------------------- constants (from demonstrations)
ROT_DOWN = [math.pi, 0.0, 0.0]          # top-down grasp orientation used by the expert
ROT_DRAWER = [1.5 * math.pi, 0.0, 0.0]  # gripper pointing +y toward the drawer handle
HANDLE_FROM_CAB = [-0.1272, -0.2016]    # handle xy = (cabinet x98, cabinet y98) + this (max err < 1 mm)
HANDLE_Z = 0.607                        # middle drawer handle height (constant in all demos)
HANDLE_PRIOR = [0.136, 0.452]
COOKIE_PRIOR = [0.223, -0.329]          # blob centroid prior (agentview)
COOKIE_BIAS = [-0.0070, 0.0008]         # expert grasp minus blob centroid
BUTTER_PRIOR = [0.113, -0.205]
BUTTER_BIAS = [-0.0030, 0.0030]
GRASP_Z = 0.432                         # expert grasp TCP height for both objects (table ~0.430)
PULL = 0.24                             # expert pull distance (drawer hits its stop before that)
BACKOFF = 0.35
COOKIE_DROP = [-0.050, -0.055, 0.118]   # release offset from handle (expert median -0.05,-0.047,0.118)
BUTTER_DROP = [0.000, -0.048, 0.120]    # expert mean (-0.001,-0.045,0.121)
TRANSIT_DZ = 0.30
AGENT_K = [[231.7645019878, 0.0, 127.5], [0.0, 309.0193359838, 127.5], [0.0, 0.0, 1.0]]   # fallback (demos)
AGENT_E = [[-7.26818450e-08, 1.0, -1.52115266e-07, 3.39354405e-07],
           [0.62826645, -7.26818450e-08, -0.777998244, 0.447718863],
           [-0.777998244, -1.52115266e-07, -0.62826645, 2.016979493]]                    # fallback (static camera)
HOLD_W = 0.006                          # finger width (m) above which an object/handle is held


def _aa2R(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_err(cur_aa, tgt_aa):
    """World-frame axis-angle that turns the current orientation into the target."""
    return _R2aa(_aa2R(tgt_aa) @ _aa2R(cur_aa).T)


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _points(frame):
    """Agentview depth back-projected to world XYZ (H x W x 3) and a validity mask."""
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = getattr(frame, "agentview_K", None)
    K = np.asarray(AGENT_K if K is None else K, dtype=float)
    E = getattr(frame, "agentview_extrinsic", None)
    E = np.asarray(AGENT_E if E is None else E, dtype=float)
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, d > 0


def _dark(frame):
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    return rgb.sum(-1) < 200


def _find_handle(frame):
    pw, ok = _points(frame)
    m = ok & _dark(frame) & (pw[..., 1] > 0.2) & (pw[..., 1] < 0.8) & (pw[..., 0] > -0.2) \
        & (pw[..., 0] < 0.5) & (pw[..., 2] > 0.44) & (pw[..., 2] < 0.9)
    p = pw[m]
    if len(p) < 500:
        return [HANDLE_PRIOR[0], HANDLE_PRIOR[1], HANDLE_Z], False, int(len(p))
    hx = float(np.percentile(p[:, 0], 98)) + HANDLE_FROM_CAB[0]
    hy = float(np.percentile(p[:, 1], 98)) + HANDLE_FROM_CAB[1]
    good = abs(hx - HANDLE_PRIOR[0]) < 0.08 and abs(hy - HANDLE_PRIOR[1]) < 0.08
    if not good:
        return [HANDLE_PRIOR[0], HANDLE_PRIOR[1], HANDLE_Z], False, int(len(p))
    return [hx, hy, HANDLE_Z], True, int(len(p))


def _find_blob(frame, prior, bias, r=0.06):
    pw, ok = _points(frame)
    m = ok & (np.abs(pw[..., 0] - prior[0]) < r) & (np.abs(pw[..., 1] - prior[1]) < r) \
        & (pw[..., 2] > 0.437) & (pw[..., 2] < 0.50)
    p = pw[m]
    if len(p) < 25:
        return None, int(len(p))
    c = p.mean(0)
    return [float(c[0] + bias[0]), float(c[1] + bias[1]), GRASP_Z], int(len(p))


def _drawer_open(frame, handle):
    pw, ok = _points(frame)
    m = ok & _dark(frame) & (np.abs(pw[..., 0] - handle[0]) < 0.12) & (pw[..., 1] > handle[1] - 0.30) \
        & (pw[..., 1] < handle[1] - 0.03) & (pw[..., 2] > 0.5) & (pw[..., 2] < 0.7)
    p = pw[m]
    if len(p) < 50:
        return False, int(len(p)), 0.0
    med = float(np.median(p[:, 1]) - handle[1])
    return (len(p) > 400 and med < -0.07), int(len(p)), med


# ---------------------------------------------------------------- perception / decision

def _mv(target, rot, grip, tol=0.006, vmax=1.0, max_steps=300, stall=False):
    return {"tool": "act_move", "args": {"target": [float(v) for v in target], "rot": rot,
                                         "gripper": float(grip), "tol": tol, "vmax": vmax,
                                         "max_steps": max_steps, "stop_on_stall": stall}}


def _gr(close, steps, rot):
    return {"tool": "act_gripper", "args": {"close": bool(close), "steps": int(steps), "rot": rot}}


def perceive_decide(obs, goal, memory, last):
    """Phase machine: perceive cabinet/cookies/butter at start, then open drawer, place cookies, place butter.
    Memory keeps the perceived positions, the current phase and retry count."""
    mem = dict(memory)
    for k in ("handle", "cookie", "butter", "phase", "retries", "notes"):
        mem.setdefault(k, None)
    fr = obs.current
    tcp = np.asarray(fr.tcp, dtype=float)
    width = _width(fr)
    ph = mem["phase"]
    if mem["retries"] is None:
        mem["retries"] = 0
    note = ""

    if ph is None:
        h, okh, nh = _find_handle(fr)
        c, nc = _find_blob(fr, COOKIE_PRIOR, COOKIE_BIAS)
        b, nb = _find_blob(fr, BUTTER_PRIOR, BUTTER_BIAS)
        if c is None:
            c = [COOKIE_PRIOR[0] + COOKIE_BIAS[0], COOKIE_PRIOR[1] + COOKIE_BIAS[1], GRASP_Z]
        if b is None:
            b = [BUTTER_PRIOR[0] + BUTTER_BIAS[0], BUTTER_PRIOR[1] + BUTTER_BIAS[1], GRASP_Z]
        mem["handle"], mem["cookie"], mem["butter"] = h, c, b
        note = "init handle_ok=%s n=%d cookie_n=%d butter_n=%d" % (okh, nh, nc, nb)
        ph = "start"

    H = mem["handle"]
    hx, hy, hz = H
    act = None
    nxt = ph

    # ---- drawer opening
    if ph == "start":
        act, nxt = _mv([hx, hy - 0.12, hz], "drawer", -1, tol=0.008), "pre_handle"
    elif ph == "pre_handle":
        act, nxt = _mv([hx, hy, hz], "drawer", -1, tol=0.004, vmax=0.4, max_steps=150), "at_handle"
    elif ph == "at_handle":
        act, nxt = _gr(True, 30, "drawer"), "grip_handle"
    elif ph == "grip_handle":
        if width < HOLD_W - 0.002 and mem["retries"] < 3:
            mem["retries"] += 1
            note = "handle grasp missed (w=%.4f), retry" % width
            act, nxt = _mv([hx, hy - 0.10, hz], "drawer", -1, tol=0.01), "pre_handle"
        else:
            act, nxt = _mv([hx, hy - PULL, hz], "drawer", 1, tol=0.008, vmax=1.0, max_steps=150, stall=True), "pulled"
    elif ph == "pulled":
        act, nxt = _gr(False, 15, "drawer"), "released_handle"
    elif ph == "released_handle":
        act, nxt = _mv([hx, hy - BACKOFF, hz], "drawer", -1, tol=0.01), "backed"
    elif ph == "backed":
        okd, nd, med = _drawer_open(fr, H)
        note = "drawer_open=%s n=%d med=%.3f" % (okd, nd, med)
        if (not okd) and mem["retries"] < 3:
            mem["retries"] += 1
            # drawer front position unknown (partially open): approach the handle line slowly from the
            # backed-off pose and stop on contact/stall, then re-grasp and pull again
            act, nxt = _mv([hx, hy, hz], "drawer", -1, tol=0.004, vmax=0.3, max_steps=250, stall=True), "at_handle"
        else:
            c = mem["cookie"]
            act, nxt = _mv([c[0], c[1], 0.63], "down", -1, tol=0.008), "pre_cookie"

    # ---- cookies
    elif ph == "pre_cookie":
        c = mem["cookie"]
        act, nxt = _mv([c[0], c[1], GRASP_Z], "down", -1, tol=0.003, vmax=0.5, max_steps=200, stall=True), "at_cookie"
    elif ph == "at_cookie":
        act, nxt = _gr(True, 30, "down"), "grip_cookie"
    elif ph == "grip_cookie":
        c = mem["cookie"]
        act, nxt = _mv([c[0], c[1], 0.75], "down", 1, tol=0.01), "lift_cookie"
    elif ph == "lift_cookie":
        if width < HOLD_W and mem["retries"] < 4:
            mem["retries"] += 1
            note = "cookie not held (w=%.4f); retry" % width
            act, nxt = _mv([0.0, -0.05, 0.80], "down", -1, tol=0.02), "reperceive_cookie"
        else:
            act, nxt = _mv([hx + COOKIE_DROP[0], hy + COOKIE_DROP[1], hz + TRANSIT_DZ], "down", 1, tol=0.01), "above_drawer_c"
    elif ph == "reperceive_cookie":
        c, nc = _find_blob(fr, mem["cookie"], [0.0, 0.0], r=0.10)
        if c is not None:
            c = [c[0] + COOKIE_BIAS[0], c[1] + COOKIE_BIAS[1], GRASP_Z]
            mem["cookie"] = c
        note = "reperceived cookie n=%d" % nc
        c = mem["cookie"]
        act, nxt = _mv([c[0], c[1], 0.63], "down", -1, tol=0.008), "pre_cookie"
    elif ph == "above_drawer_c":
        act, nxt = _mv([hx + COOKIE_DROP[0], hy + COOKIE_DROP[1], hz + COOKIE_DROP[2]], "down", 1,
                       tol=0.005, vmax=0.4, max_steps=200, stall=True), "place_cookie"
    elif ph == "place_cookie":
        act, nxt = _gr(False, 20, "down"), "drop_cookie"
    elif ph == "drop_cookie":
        act, nxt = _mv([tcp[0], tcp[1], hz + TRANSIT_DZ], "down", -1, tol=0.01), "cookie_done"

    # ---- butter
    elif ph == "cookie_done":
        b, nb = _find_blob(fr, BUTTER_PRIOR, BUTTER_BIAS, r=0.06)
        if b is not None:
            mem["butter"] = b
        note = "butter reperceived n=%d" % nb
        b = mem["butter"]
        act, nxt = _mv([b[0], b[1], 0.63], "down", -1, tol=0.008), "pre_butter"
    elif ph == "pre_butter":
        b = mem["butter"]
        act, nxt = _mv([b[0], b[1], GRASP_Z], "down", -1, tol=0.003, vmax=0.5, max_steps=200, stall=True), "at_butter"
    elif ph == "at_butter":
        act, nxt = _gr(True, 30, "down"), "grip_butter"
    elif ph == "grip_butter":
        b = mem["butter"]
        act, nxt = _mv([b[0], b[1], 0.75], "down", 1, tol=0.01), "lift_butter"
    elif ph == "lift_butter":
        if width < HOLD_W and mem["retries"] < 6:
            mem["retries"] += 1
            note = "butter not held (w=%.4f); retry" % width
            act, nxt = _mv([0.0, -0.05, 0.80], "down", -1, tol=0.02), "reperceive_butter"
        else:
            act, nxt = _mv([hx + BUTTER_DROP[0], hy + BUTTER_DROP[1], hz + TRANSIT_DZ], "down", 1, tol=0.01), "above_drawer_b"
    elif ph == "reperceive_butter":
        b, nb = _find_blob(fr, mem["butter"], [0.0, 0.0], r=0.10)
        if b is not None:
            mem["butter"] = [b[0] + BUTTER_BIAS[0], b[1] + BUTTER_BIAS[1], GRASP_Z]
        note = "reperceived butter n=%d" % nb
        b = mem["butter"]
        act, nxt = _mv([b[0], b[1], 0.63], "down", -1, tol=0.008), "pre_butter"
    elif ph == "above_drawer_b":
        act, nxt = _mv([hx + BUTTER_DROP[0], hy + BUTTER_DROP[1], hz + BUTTER_DROP[2]], "down", 1,
                       tol=0.005, vmax=0.4, max_steps=200, stall=True), "place_butter"
    elif ph == "place_butter":
        act, nxt = _gr(False, 25, "down"), "drop_butter"
    elif ph == "drop_butter":
        act, nxt = _mv([tcp[0], tcp[1], hz + TRANSIT_DZ], "down", -1, tol=0.01, vmax=0.5), "done"
    else:
        # done: hold still with the gripper open while the checker monitors the result
        act, nxt = {"tool": "act_hold", "args": {"steps": 100}}, "done"

    mem["phase"] = nxt
    mem["notes"] = (note or ("%s -> %s" % (ph, nxt)))[:300]
    return {"memory": mem, "act": act}


# ---------------------------------------------------------------- actions

def _step_toward(robot, target, rot_aa, grip, vmax, kp=10.0, kr=2.0):
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    e = np.asarray(target, dtype=float) - tcp[:3]
    a = kp * e
    m = float(np.max(np.abs(a)))
    if m > vmax:
        a = a * (vmax / m)
    r = kr * _rot_err(tcp[3:], rot_aa)
    mr = float(np.max(np.abs(r)))
    if mr > 0.5:
        r = r * (0.5 / mr)
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return float(np.linalg.norm(e)), float(np.linalg.norm(_rot_err(tcp[3:], rot_aa)))


def act_move(robot, target, rot="down", gripper=-1.0, tol=0.006, vmax=1.0, max_steps=300, stop_on_stall=False):
    """Move the TCP to world `target` [x,y,z] (m) with orientation `rot` ('down' or 'drawer') while holding the
    gripper command (+1 close, -1 open). Stops when within `tol` m and 0.05 rad, after `max_steps`, or (if
    stop_on_stall) when the TCP stops moving. Returns measured pose, finger width and why it stopped."""
    rot_aa = ROT_DRAWER if rot == "drawer" else ROT_DOWN
    hist = []
    reason = "max_steps"
    n = 0
    for n in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        pe = float(np.linalg.norm(np.asarray(target, dtype=float) - tcp[:3]))
        re = float(np.linalg.norm(_rot_err(tcp[3:], rot_aa)))
        if pe < tol and re < 0.05:
            reason = "reached"
            break
        hist.append(tcp[:3].copy())
        if len(hist) > 25 and re < 0.08:
            moved = float(np.linalg.norm(hist[-1] - hist[-16]))
            if stop_on_stall and moved < 0.002:
                reason = "stalled"
                break
            if pe < 3.0 * tol and moved < 0.001:
                reason = "settled"
                break
        _step_toward(robot, target, rot_aa, gripper, vmax)
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"reason": reason, "steps": n, "tcp": [round(float(v), 4) for v in tcp],
            "err": round(float(np.linalg.norm(np.asarray(target, dtype=float) - tcp[:3])), 4),
            "width": round(_width(fr), 4)}


def act_gripper(robot, close=True, steps=30, rot="down"):
    """Close (close=True) or open the gripper for `steps` native steps while holding the current TCP pose.
    Returns the final finger width (m)."""
    rot_aa = ROT_DRAWER if rot == "drawer" else ROT_DOWN
    hold = np.asarray(robot.now().tcp, dtype=float)[:3].copy()
    g = 1.0 if close else -1.0
    for _ in range(int(steps)):
        _step_toward(robot, hold, rot_aa, g, 0.3)
    fr = robot.now()
    return {"width": round(_width(fr), 4), "tcp": [round(float(v), 4) for v in np.asarray(fr.tcp)]}


def act_hold(robot, steps=100):
    """Hold the current pose with the gripper open for `steps` native steps (lets the checker settle)."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    return {"width": round(_width(robot.now()), 4)}
