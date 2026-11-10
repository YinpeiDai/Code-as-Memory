"""Policy for T11_CookiesTopButterMiddle: put cookies into the top drawer, close it,
then put butter into the middle drawer.

Stages (memory field `phase`): open_top -> pick_cookies -> place_cookies -> close_top
-> open_mid -> pick_butter -> place_butter -> done.
All geometry is measured from the agentview RGB-D point cloud of the current episode;
offsets between perceived features and grasp points were fitted on the demonstrations
(see task_analysis.md)."""

# ---------------- constants fitted on the 50 demonstrations ----------------
DOWN_RV = [3.141592653589793, 0.0, 0.0]        # gripper pointing down, fingers along world y
HORIZ_RV = [-1.5707963267948966, 0.0, 0.0]     # gripper pointing +y (into drawer fronts), fingers along world z
TOP_X_OFF = -0.1285    # handle x = cabinet top-face x(99th pct) + off   (std 0.8 mm)
TOP_Y_OFF = -0.0181    # handle y = cabinet top-face y(1st pct) + off    (std 0.4 mm)
TOP_HANDLE_DZ = -0.038 # top handle z = cabinet top z (0.713) - 0.038 -> 0.675
MID_HANDLE_DZ = -0.106 # middle handle z = 0.607
PRIOR_HANDLE = [0.137, 0.454, 0.675]
PRIOR_ZTOP = 0.713
PRIOR_COOKIES = [0.217, -0.328]
PRIOR_BUTTER = [0.110, -0.202]
COOKIE_OFF = [-0.005, 0.002]   # expert grasp - bbox centre (mean over demos)
BUTTER_OFF = [0.0, 0.003]
TABLE_Z = 0.430
GRASP_Z = 0.431
PULL_DY = 0.24         # pull target beyond the handle (drawer stops ~0.16-0.19, fingers slip)
DROP_DX = 0.012
DROP_DY_COOKIES = -0.088
DROP_DY_BUTTER = -0.09
RELEASE_Z_COOKIES = 0.716
RELEASE_Z_BUTTER = 0.675
TRAVEL_Z = 0.80


# ---------------- maths helpers ----------------
def _rv2m(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _m2rv(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _pos(frame):
    return np.asarray(frame.tcp, dtype=float)[:3].copy()


def _r(v, d=4):
    return [round(float(x), d) for x in v]


# ---------------- perception ----------------
def _cloud(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    w = (pc - E[:, 3]) @ E[:, :3]
    return w.reshape(H, W, 3), d > 0.05


def _perceive_cabinet(w, valid):
    X, Y, Z = w[..., 0], w[..., 1], w[..., 2]
    m = valid & (Z > 0.69) & (Z < 0.74) & (Y > 0.25) & (Y < 0.9) & (X > -0.4) & (X < 0.6)
    n = int(m.sum())
    if n < 300:
        return None
    ztop = float(np.median(Z[m]))
    xmax = float(np.percentile(X[m], 99))
    ymin = float(np.percentile(Y[m], 1))
    hx = xmax + TOP_X_OFF
    hy = ymin + TOP_Y_OFF
    if abs(ztop - PRIOR_ZTOP) > 0.03:
        ztop = PRIOR_ZTOP
    return {"hx": hx, "hy": hy, "ztop": ztop, "n": n}


def _perceive_box(w, valid, prior, radius):
    X, Y, Z = w[..., 0], w[..., 1], w[..., 2]
    m = valid & (Z > TABLE_Z + 0.006) & (Z < 0.50) & (np.hypot(X - prior[0], Y - prior[1]) < radius)
    n = int(m.sum())
    if n < 15:
        return None
    xs, ys = X[m], Y[m]
    x0, x1 = np.percentile(xs, 2), np.percentile(xs, 98)
    y0, y1 = np.percentile(ys, 2), np.percentile(ys, 98)
    return {"x": float((x0 + x1) / 2), "y": float((y0 + y1) / 2), "sx": float(x1 - x0),
            "sy": float(y1 - y0), "n": n, "ztop": float(np.percentile(Z[m], 95))}


def perceive_scene(obs):
    """Measure cabinet handle and object positions from the current agentview RGB-D frame."""
    w, valid = _cloud(obs.current)
    return {"cabinet": _perceive_cabinet(w, valid),
            "cookies": _perceive_box(w, valid, PRIOR_COOKIES, 0.07),
            "butter": _perceive_box(w, valid, PRIOR_BUTTER, 0.055)}


# ---------------- control helpers ----------------
def _servo(robot, tp, trv, grip, tol=0.006, rtol=0.05, max_steps=250, vmax=1.0, kp=0.7,
           amin=0.08, stall=0, rmax=0.6):
    """Closed-loop move of the tcp to position tp / axis-angle trv. Returns a summary."""
    tp = np.asarray(tp, dtype=float)
    Rt = _rv2m(trv)
    hist = []
    n = 0
    status = "timeout"
    while n < max_steps:
        f = robot.now()
        q = np.asarray(f.tcp, dtype=float)
        p = q[:3]
        ep = tp - p
        er = _m2rv(Rt @ _rv2m(q[3:]).T)
        if float(np.max(np.abs(ep))) < tol and float(np.linalg.norm(er)) < rtol:
            status = "reached"
            break
        hist.append(p.copy())
        if stall and len(hist) > stall + 3 and float(np.linalg.norm(hist[-1] - hist[-1 - stall])) < 0.0015:
            status = "stalled"
            break
        if (tol > 0 and len(hist) > 15 and float(np.max(np.abs(ep))) < 2.0 * tol
                and float(np.linalg.norm(er)) < 2.0 * rtol
                and float(np.linalg.norm(hist[-1] - hist[-11])) < 0.001):
            status = "settled"
            break
        a = ep / 0.05 * kp
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        na = float(np.linalg.norm(a))
        if 1e-9 < na < amin and float(np.max(np.abs(ep))) >= tol:
            a = a * (amin / na)
        r = er / 0.5
        mr = float(np.max(np.abs(r)))
        if mr > rmax:
            r = r * (rmax / mr)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    p = _pos(robot.now())
    return {"status": status, "steps": n, "pos": _r(p), "err": round(float(np.linalg.norm(tp - p)), 4)}


def _gripper(robot, grip, tp, trv, min_steps=8, max_steps=30):
    """Open (-1) or close (+1) the fingers while holding pose; waits until fingers settle."""
    tp = np.asarray(tp, dtype=float)
    Rt = _rv2m(trv)
    ws = []
    for i in range(max_steps):
        f = robot.now()
        q = np.asarray(f.tcp, dtype=float)
        ep = tp - q[:3]
        er = _m2rv(Rt @ _rv2m(q[3:]).T)
        a = np.clip(ep / 0.05 * 0.5, -0.3, 0.3)
        r = np.clip(er / 0.5, -0.3, 0.3)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        ws.append(_width(robot.now()))
        if i >= min_steps and abs(ws[-1] - ws[-4]) < 0.0004:
            break
    return _width(robot.now())


def _total(*parts):
    return int(sum(p["steps"] for p in parts if isinstance(p, dict) and "steps" in p))


# ---------------- act tools ----------------
def act_open_drawer(robot, hx, hy, hz, pull=0.24, pre_dy=0.08, retreat_dy=0.35):
    """Grasp the drawer handle at (hx, hy, hz) with the gripper pointing +y, pull it open
    toward -y, release and retreat to y = hy - retreat_dy. Reports grasp width, the tcp y
    where the fingers slipped off (drawer fully open) and ok."""
    s1 = _servo(robot, [hx, hy - pre_dy, hz], HORIZ_RV, -1, tol=0.005, rtol=0.04, max_steps=260)
    s2 = _servo(robot, [hx, hy + 0.004, hz], HORIZ_RV, -1, tol=0.004, max_steps=90, vmax=0.35, stall=8)
    gp = _pos(robot.now())
    w = _gripper(robot, 1, gp, HORIZ_RV, min_steps=10, max_steps=30)
    if w < 0.006:
        _gripper(robot, -1, gp, HORIZ_RV)
        s4 = _servo(robot, [hx, hy - pre_dy, hz], HORIZ_RV, -1, max_steps=80)
        return {"ok": False, "reason": "handle_missed", "width": round(w, 4), "grasp_pos": _r(gp),
                "steps": _total(s1, s2, s4) + 40}
    # pull, tracking where the fingers slip off the handle
    tp = np.array([hx, hy - pull, hz])
    Rt = HORIZ_RV
    slip_y = None
    n = 0
    while n < 160:
        f = robot.now()
        p = _pos(f)
        if slip_y is None and _width(f) < 0.006:
            slip_y = float(p[1])
        if abs(tp[1] - p[1]) < 0.008:
            break
        if slip_y is not None and p[1] < slip_y - 0.02:
            break
        q = np.asarray(f.tcp, dtype=float)
        ep = tp - p
        er = _m2rv(_rv2m(Rt) @ _rv2m(q[3:]).T)
        a = ep / 0.05 * 0.7
        m = float(np.max(np.abs(a)))
        if m > 1.0:
            a = a / m
        r = np.clip(er / 0.5, -0.6, 0.6)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), 1)
        n += 1
    endp = _pos(robot.now())
    if slip_y is None:
        slip_y = float(endp[1])
    _gripper(robot, -1, endp, HORIZ_RV)
    s5 = _servo(robot, [hx, hy - retreat_dy, hz], HORIZ_RV, -1, tol=0.01, max_steps=120)
    opened = slip_y < hy - 0.12
    return {"ok": bool(opened), "width": round(w, 4), "grasp_pos": _r(gp), "slip_y": round(slip_y, 4),
            "pull_end": _r(endp), "approach": s2["status"], "steps": _total(s1, s2, s5) + n + 50}


def act_close_drawer(robot, hx, hy, hz, standoff=0.27, grab_dy=-0.155, high_z=0.86, retreat_dy=0.10):
    """Close an open drawer whose closed handle position is (hx, hy, hz): come down in front
    of it, slide the open fingers over the handle, close them, push to the closed position,
    release and retreat to y = hy - retreat_dy. ok when the tcp reached the closed position."""
    p0 = _pos(robot.now())
    s0 = _servo(robot, [p0[0], p0[1], high_z], DOWN_RV, -1, tol=0.01, rtol=0.2, max_steps=60)
    s1 = _servo(robot, [hx, hy - standoff, high_z], HORIZ_RV, -1, tol=0.01, rtol=0.05, max_steps=200)
    s2 = _servo(robot, [hx, hy - standoff, hz], HORIZ_RV, -1, tol=0.005, rtol=0.04, max_steps=120)
    s3 = _servo(robot, [hx, hy + grab_dy, hz], HORIZ_RV, -1, tol=0.004, max_steps=80, vmax=0.5, stall=8)
    gp = _pos(robot.now())
    w = _gripper(robot, 1, gp, HORIZ_RV, min_steps=10, max_steps=25)
    s4 = _servo(robot, [hx, hy + 0.006, hz], HORIZ_RV, 1, tol=0.003, max_steps=160, vmax=0.6, stall=10)
    endp = _pos(robot.now())
    _gripper(robot, -1, endp, HORIZ_RV)
    s5 = _servo(robot, [hx, hy - retreat_dy, hz], HORIZ_RV, -1, tol=0.008, max_steps=100)
    closed = endp[1] > hy - 0.015
    return {"ok": bool(closed), "grab_width": round(w, 4), "push_end": _r(endp), "push": s4["status"],
            "steps": _total(s0, s1, s2, s3, s4, s5) + 50}


def act_pick(robot, x, y, z_grasp=0.431, z_above=0.62, lift_z=0.80, min_width=0.012):
    """Pick a flat box lying on the table at (x, y) with the gripper pointing down
    (fingers close along world y), then lift to lift_z. ok when the fingers stay apart."""
    p0 = _pos(robot.now())
    s0 = _servo(robot, [x, y, max(z_above, min(p0[2], 0.70))], DOWN_RV, -1, tol=0.006, rtol=0.04, max_steps=260)
    s1 = _servo(robot, [x, y, z_grasp - 0.004], DOWN_RV, -1, tol=0.004, rtol=0.04, max_steps=120,
                vmax=0.7, stall=6)
    gp = _pos(robot.now())
    w = _gripper(robot, 1, gp, DOWN_RV, min_steps=10, max_steps=30)
    s2 = _servo(robot, [x, y, lift_z], DOWN_RV, 1, tol=0.01, max_steps=120)
    w2 = _width(robot.now())
    ok = w2 > min_width
    return {"ok": bool(ok), "width_closed": round(w, 4), "width_lifted": round(w2, 4),
            "grasp_pos": _r(gp), "steps": _total(s0, s1, s2) + 30}


def act_place(robot, x, y, z_release, z_travel=0.80, z_after=0.86):
    """Carry the held object at height z_travel to above (x, y), lower it to z_release,
    open the fingers and lift to z_after."""
    p0 = _pos(robot.now())
    s0 = _servo(robot, [p0[0], p0[1], z_travel], DOWN_RV, 1, tol=0.01, max_steps=80)
    s1 = _servo(robot, [x, y, z_travel], DOWN_RV, 1, tol=0.008, max_steps=220)
    held = _width(robot.now())
    s2 = _servo(robot, [x, y, z_release], DOWN_RV, 1, tol=0.004, max_steps=100, vmax=0.6, stall=8)
    rp = _pos(robot.now())
    _gripper(robot, -1, rp, DOWN_RV, min_steps=10, max_steps=25)
    s3 = _servo(robot, [x, y, z_after], DOWN_RV, -1, tol=0.01, max_steps=80)
    return {"ok": bool(held > 0.008), "held_width": round(held, 4), "release_pos": _r(rp),
            "steps": _total(s0, s1, s2, s3) + 25}


def act_wait(robot, steps=40):
    """Hold the current pose with the gripper open for a number of steps."""
    p = _pos(robot.now())
    q = np.asarray(robot.now().tcp, dtype=float)
    s = _servo(robot, p, q[3:], -1, tol=-1.0, max_steps=int(steps))
    return {"ok": True, "steps": s["steps"]}


# ---------------- decision logic ----------------
def _flag(last, key):
    return isinstance(last, str) and ('"%s": true' % key) in last


def _num(last, key):
    if not isinstance(last, str):
        return None
    k = '"%s": ' % key
    i = last.find(k)
    if i < 0:
        return None
    j = i + len(k)
    s = ""
    while j < len(last) and last[j] in "-+.0123456789e":
        s += last[j]
        j += 1
    try:
        return float(s)
    except ValueError:
        return None


def _box_or(prior_xy, det, off, mem_val):
    if det is not None and abs(det["sx"] - 0.0) > 0.02:
        return [det["x"] + off[0], det["y"] + off[1]]
    if mem_val is not None:
        return list(mem_val)[:2]
    return [prior_xy[0] + off[0], prior_xy[1] + off[1]]


def perceive_decide(obs, goal, memory, last):
    """State machine over the eight task stages; perception from the current agentview frame."""
    mem = dict(memory)
    phase = mem.get("phase")
    tries = int(mem.get("tries") or 0)
    log = mem.get("log") or ""

    if phase is None:
        sc = perceive_scene(obs)
        cab = sc["cabinet"]
        if cab is None:
            hx, hy, zt = PRIOR_HANDLE[0], PRIOR_HANDLE[1], PRIOR_ZTOP
            log += "cabinet:prior;"
        else:
            hx, hy, zt = cab["hx"], cab["hy"], cab["ztop"]
            log += "cabinet:n%d;" % cab["n"]
        mem["handle"] = [round(hx, 4), round(hy, 4), round(zt + TOP_HANDLE_DZ, 4)]
        mem["mid_z"] = round(zt + MID_HANDLE_DZ, 4)
        mem["cookies"] = [round(v, 4) for v in _box_or(PRIOR_COOKIES, sc["cookies"], COOKIE_OFF, None)]
        mem["butter"] = [round(v, 4) for v in _box_or(PRIOR_BUTTER, sc["butter"], BUTTER_OFF, None)]
        log += "cookies:%s;butter:%s;" % ("det" if sc["cookies"] else "prior", "det" if sc["butter"] else "prior")
        mem["slip_y"] = None
        mem["rechecks"] = 0
        phase = "open_top"
        tries = 0
    else:
        ok = _flag(last, "ok")
        sy = _num(last, "slip_y")
        if sy is not None:
            mem["slip_y"] = round(sy, 4)
        if phase == "open_top":
            if ok or tries >= 2:
                phase = "pick_cookies"
                tries = 0
            else:
                tries += 1
        elif phase == "pick_cookies":
            if ok:
                phase = "place_cookies"
                tries = 0
            elif tries >= 2:
                phase = "close_top"
                tries = 0
            else:
                tries += 1
                det = perceive_scene(obs)["cookies"]
                if det is None:
                    w, valid = _cloud(obs.current)
                    det = _perceive_box(w, valid, PRIOR_COOKIES, 0.15)
                mem["cookies"] = [round(v, 4) for v in _box_or(PRIOR_COOKIES, det, COOKIE_OFF, mem.get("cookies"))]
        elif phase == "place_cookies":
            # verify the cookies have left the table; otherwise pick them again
            det = perceive_scene(obs)["cookies"]
            rechecks = int(mem.get("rechecks") or 0)
            if det is not None and det["n"] > 30 and rechecks < 2:
                phase = "pick_cookies"
                mem["rechecks"] = rechecks + 1
                mem["cookies"] = [round(v, 4) for v in _box_or(PRIOR_COOKIES, det, COOKIE_OFF, mem.get("cookies"))]
            else:
                phase = "close_top"
            tries = 0
        elif phase == "close_top":
            if ok or tries >= 1:
                phase = "open_mid"
                tries = 0
            else:
                tries += 1
        elif phase == "open_mid":
            if ok or tries >= 2:
                phase = "pick_butter"
                tries = 0
                det = perceive_scene(obs)["butter"]
                if det is not None and 0.05 < det["sx"] < 0.10 and 0.025 < det["sy"] < 0.06:
                    nb = [det["x"] + BUTTER_OFF[0], det["y"] + BUTTER_OFF[1]]
                    ob = mem.get("butter") or nb
                    if math.hypot(nb[0] - ob[0], nb[1] - ob[1]) < 0.05:
                        mem["butter"] = [round(v, 4) for v in nb]
            else:
                tries += 1
        elif phase == "pick_butter":
            if ok:
                phase = "place_butter"
                tries = 0
            elif tries >= 3:
                phase = "done"
            else:
                tries += 1
                det = perceive_scene(obs)["butter"]
                if det is None:
                    w, valid = _cloud(obs.current)
                    det = _perceive_box(w, valid, PRIOR_BUTTER, 0.15)
                mem["butter"] = [round(v, 4) for v in _box_or(PRIOR_BUTTER, det, BUTTER_OFF, mem.get("butter"))]
        elif phase == "place_butter" or phase == "done":
            # verify the butter has left the table; otherwise try to pick it again
            det = perceive_scene(obs)["butter"]
            rechecks = int(mem.get("rechecks") or 0)
            if det is not None and det["n"] > 30 and rechecks < 2:
                phase = "pick_butter"
                tries = 0
                mem["rechecks"] = rechecks + 1
                mem["butter"] = [round(v, 4) for v in _box_or(PRIOR_BUTTER, det, BUTTER_OFF, mem.get("butter"))]
            else:
                phase = "done"
        if isinstance(last, str):
            log += phase[:6] + ("+" if ok else "-") + ";"

    hx, hy, hz = mem["handle"]
    pre = 0.08
    if phase in ("open_top", "open_mid") and tries > 0 and mem.get("slip_y") is not None:
        pre = min(0.26, max(0.08, 0.08 + (hy - float(mem["slip_y"])) * 0.8))
    if phase == "open_top":
        act = {"tool": "act_open_drawer", "args": {"hx": hx, "hy": hy, "hz": hz, "pre_dy": round(pre, 4)}}
    elif phase == "pick_cookies":
        c = mem["cookies"]
        act = {"tool": "act_pick", "args": {"x": c[0], "y": c[1], "z_grasp": GRASP_Z, "min_width": 0.02}}
    elif phase == "place_cookies":
        act = {"tool": "act_place", "args": {"x": hx + DROP_DX, "y": hy + DROP_DY_COOKIES,
                                             "z_release": RELEASE_Z_COOKIES, "z_travel": TRAVEL_Z}}
    elif phase == "close_top":
        act = {"tool": "act_close_drawer", "args": {"hx": hx, "hy": hy, "hz": hz}}
    elif phase == "open_mid":
        act = {"tool": "act_open_drawer", "args": {"hx": hx, "hy": hy, "hz": mem["mid_z"], "pre_dy": round(pre, 4)}}
    elif phase == "pick_butter":
        b = mem["butter"]
        act = {"tool": "act_pick", "args": {"x": b[0], "y": b[1], "z_grasp": GRASP_Z, "min_width": 0.012}}
    elif phase == "place_butter":
        act = {"tool": "act_place", "args": {"x": hx + DROP_DX, "y": hy + DROP_DY_BUTTER,
                                             "z_release": RELEASE_Z_BUTTER, "z_travel": TRAVEL_Z}}
    else:
        act = {"tool": "act_wait", "args": {"steps": 60}}

    mem["phase"] = phase
    mem["tries"] = tries
    mem["log"] = log[-400:]
    out = {k: mem.get(k) for k in ("phase", "tries", "rechecks", "handle", "mid_z", "cookies", "butter", "slip_y", "log")}
    return {"memory": out, "act": act}
