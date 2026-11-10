"""Code-as-Memory policy for T01_CookiesTomatoBasket.

Goal: put the cookies box into a basket, then the tomato sauce into the SAME basket.
Perception: RGB-D back-projection into world coordinates (calibrated K/extrinsics),
height-above-table segmentation and connected components. Control: closed-loop
OSC_POSE delta steps toward world waypoints using the measured TCP.
"""

TABLE_Z_DEFAULT = 0.431
DOWN_R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
LOOK_POSE = [-0.058, 0.0, 0.671]
TRAVEL_Z = 0.80
# Priors (mean of assigned demos) used only when perception fails.
PRIOR_COOKIES = [-0.141, 0.083, 0.433]
PRIOR_TOMATO = [-0.066, -0.062, 0.500]
PRIOR_BASKET = [-0.119, 0.551, 0.567]
# Small view-dependent centroid offsets measured on demos (wrist view from LOOK_POSE).
COOKIE_BIAS_WRIST = [-0.0012, -0.0007]
TOMATO_BIAS_WRIST = [-0.0004, 0.0048]
TOMATO_BIAS_AGENT = [-0.0122, 0.0029]
COOKIE_BIAS_AGENT = [-0.0021, -0.0009]
BASKET_BIAS_AGENT = [-0.0016, 0.0003]
COOKIE_RELEASE_Z = 0.656
TOMATO_RELEASE_Z = 0.656
MAX_PICK_TRIES = 4
GRIP_GAP_HOLD = 0.012

FIELDS = ["phase", "cookies_xyz", "tomato_xyz", "basket_xyz", "basket_choice",
          "cookies_tries", "tomato_tries", "holding", "cookies_released",
          "tomato_released", "looks", "notes"]


# ---------------------------------------------------------------- math helpers
def _aa2m(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _m2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        w, v = np.linalg.eigh((R + R.T) / 2.0)
        k = v[:, int(np.argmax(w))]
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def _rot_cmd(tcp, gain=2.0):
    Rc = _aa2m(tcp[3:6])
    e = _m2aa(np.array(DOWN_R) @ Rc.T)
    return np.clip(e / 0.5 * gain, -1.0, 1.0)


# ---------------------------------------------------------------- perception
def _backproject(depth, K, E):
    D = np.asarray(depth).astype(float) / 1000.0
    H, W = D.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([(u - K[0][2]) / K[0][0] * D, (v - K[1][2]) / K[1][1] * D, D], -1).reshape(-1, 3)
    E = np.asarray(E)
    R = E[:, :3]
    t = E[:, 3]
    Wp = ((P - t) @ R).reshape(H, W, 3)
    Wp[D <= 0] = np.nan
    return Wp


def _components(mask, minpix=30, iters=500):
    H, W = mask.shape
    big = H * W + 1
    lab = np.where(mask, np.arange(H * W).reshape(H, W), big)
    for _ in range(iters):
        p = np.pad(lab, 1, constant_values=big)
        n = np.minimum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        n = np.where(mask, n, big)
        if np.array_equal(n, lab):
            break
        lab = n
    ids, counts = np.unique(lab[lab < big], return_counts=True)
    order = sorted(zip(ids.tolist(), counts.tolist()), key=lambda x: -x[1])
    return [lab == i for i, c in order if c >= minpix]


def _table_z(Wp, tcp):
    z = Wp[..., 2]
    ok = np.isfinite(z) & (z > 0.38) & (z < 0.47) & (np.abs(Wp[..., 0]) < 0.45) & (np.abs(Wp[..., 1]) < 0.75)
    if ok.sum() < 200:
        return TABLE_Z_DEFAULT
    zz = z[ok]
    med = float(np.median(zz))
    if abs(med - TABLE_Z_DEFAULT) > 0.03:
        return TABLE_Z_DEFAULT
    return med


def _detect(Wp, tcp, view):
    """Objects on the table in one back-projected view.

    Returns dict with lists of cookies-like, tomato-like and basket-like components."""
    tcp = np.asarray(tcp, dtype=float)
    tz = _table_z(Wp, tcp)
    x = np.nan_to_num(Wp[..., 0], nan=9.0)
    y = np.nan_to_num(Wp[..., 1], nan=9.0)
    z = np.nan_to_num(Wp[..., 2], nan=-9.0)
    region = (np.abs(x) < 0.45) & (np.abs(y) < 0.75)
    # exclude the gripper itself (at or above the fingertip level, close to tcp)
    near_tcp = (np.hypot(x - tcp[0], y - tcp[1]) < 0.13) & (z > tcp[2] - 0.015)
    mask = region & (z > tz + 0.006) & (z < 0.64) & ~near_tcp
    out = {"table_z": tz, "cookies": [], "tomato": [], "basket": []}
    for s in _components(mask, minpix=25):
        P = Wp[s]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 25:
            continue
        top = float(np.percentile(P[:, 2], 98))
        lo = P[:, :2].min(0)
        hi = P[:, :2].max(0)
        ext = hi - lo
        info = {"n": int(len(P)), "top": top, "mean": P[:, :2].mean(0).tolist(),
                "bbox_c": ((lo + hi) / 2).tolist(), "ext": ext.tolist()}
        if top > 0.53 and max(ext) > 0.08:
            continue
        elif top < tz + 0.03 and max(ext) < 0.14 and max(ext) > 0.025:
            out["cookies"].append(info)
        elif tz + 0.035 < top < 0.56 and max(ext) < 0.10:
            sel = P[P[:, 2] > top - 0.01]
            info["top_c"] = sel[:, :2].mean(0).tolist()
            out["tomato"].append(info)
    # baskets: segment the raised rims separately so adjacent baskets do not merge
    rmask = region & (z > 0.53) & (z < 0.64) & ~near_tcp
    for s in _components(rmask, minpix=25):
        P = Wp[s]
        P = P[np.isfinite(P).all(1)]
        if len(P) < 25:
            continue
        lo = P[:, :2].min(0)
        hi = P[:, :2].max(0)
        ext = hi - lo
        if 0.08 < max(ext) < 0.30:
            out["basket"].append({"n": int(len(P)), "top": float(np.percentile(P[:, 2], 98)),
                                  "bbox_c": ((lo + hi) / 2).tolist(), "rim_c": ((lo + hi) / 2).tolist(),
                                  "ext": ext.tolist()})
    return out


def _inside(xy, basket_xyz, margin=0.0):
    if basket_xyz is None:
        return False
    return abs(xy[0] - basket_xyz[0]) < 0.09 + margin and abs(xy[1] - basket_xyz[1]) < 0.09 + margin


def _perceive_all(frame, basket_xyz):
    tcp = np.asarray(frame.tcp, dtype=float)
    res = {"tcp": tcp.tolist(), "fingers": np.asarray(frame.fingers).tolist()}
    wrist = None
    agent = None
    if hasattr(frame, "wrist_depth"):
        Wp = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
        wrist = _detect(Wp, tcp, "wrist")
    if hasattr(frame, "agentview_depth"):
        Wa = _backproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
        agent = _detect(Wa, tcp, "agent")
    at_look = float(np.linalg.norm(tcp[:3] - np.array(LOOK_POSE))) < 0.03
    tz = wrist["table_z"] if wrist else TABLE_Z_DEFAULT
    # baskets from agentview
    baskets = []
    if agent:
        for b in agent["basket"]:
            c = b.get("rim_c", b["bbox_c"])
            baskets.append([c[0] + BASKET_BIAS_AGENT[0], c[1] + BASKET_BIAS_AGENT[1], b["top"]])
    res["baskets"] = baskets
    bref = basket_xyz
    if bref is None and baskets:
        bref = min(baskets, key=lambda b: b[0])
    cookies = None
    tomato = None
    src = {}
    # wrist detections are trusted only when the hand is well above the objects
    wrist_ok = wrist is not None and tcp[2] > 0.60
    if wrist_ok:
        cands = [c for c in wrist["cookies"] if not any(_inside(c["mean"], b) for b in baskets)]
        if cands:
            c = max(cands, key=lambda c: c["n"])
            bx = COOKIE_BIAS_WRIST if at_look else [0.0, 0.0]
            cookies = [c["mean"][0] + bx[0], c["mean"][1] + bx[1], tz + 0.002]
            src["cookies"] = "wrist"
        cands = [c for c in wrist["tomato"] if not any(_inside(c["top_c"], b) for b in baskets)]
        if cands:
            c = max(cands, key=lambda c: c["n"])
            bx = TOMATO_BIAS_WRIST if at_look else [0.0, 0.0]
            tomato = [c["top_c"][0] + bx[0], c["top_c"][1] + bx[1], c["top"] - 0.004]
            src["tomato"] = "wrist"
    if agent:
        if cookies is None:
            cands = [c for c in agent["cookies"] if not any(_inside(c["mean"], b) for b in baskets)]
            if cands:
                c = max(cands, key=lambda c: c["n"])
                cookies = [c["mean"][0] + COOKIE_BIAS_AGENT[0], c["mean"][1] + COOKIE_BIAS_AGENT[1], tz + 0.002]
                src["cookies"] = "agent"
        if tomato is None:
            cands = [c for c in agent["tomato"] if not any(_inside(c["mean"], b) for b in baskets)]
            if cands:
                c = max(cands, key=lambda c: c["n"])
                tomato = [c["mean"][0] + TOMATO_BIAS_AGENT[0], c["mean"][1] + TOMATO_BIAS_AGENT[1], c["top"] - 0.004]
                src["tomato"] = "agent"
    res["cookies"] = cookies
    res["tomato"] = tomato
    res["src"] = src
    res["table_z"] = tz
    res["at_look"] = at_look
    return res


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _r(v, d=4):
    if v is None:
        return None
    return [round(float(a), d) for a in v]


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Perceive objects/baskets from RGB-D, update episode memory and pick the next act.

    Sequence: pick cookies -> place in chosen basket -> look/verify -> pick tomato sauce ->
    place in the SAME (remembered) basket -> look/verify. Retries are bounded."""
    m = {k: memory.get(k) if isinstance(memory, dict) else None for k in FIELDS}
    frame = obs.current
    phase = m["phase"] or "start"
    ct = int(m["cookies_tries"] or 0)
    tt = int(m["tomato_tries"] or 0)
    looks = int(m["looks"] or 0)
    det = _perceive_all(frame, m["basket_xyz"])
    gap = _gap(frame)
    tcp = np.asarray(frame.tcp, dtype=float)
    notes = "phase_in=%s gap=%.4f tcp=%s det_c=%s det_t=%s src=%s" % (
        phase, gap, _r(tcp[:3], 3), _r(det["cookies"], 3), _r(det["tomato"], 3), det["src"])

    # basket choice (remembered once): the basket farther from the front camera (smaller x),
    # the one used in every demonstration; tomato must go to the same one.
    if m["basket_xyz"] is None:
        if det["baskets"]:
            b = min(det["baskets"], key=lambda b: b[0])
            m["basket_xyz"] = _r(b)
            m["basket_choice"] = "far_basket_detected"
        else:
            m["basket_xyz"] = list(PRIOR_BASKET)
            m["basket_choice"] = "far_basket_prior"
    for obj in ("cookies", "tomato"):
        if det[obj] is not None and (m[obj + "_xyz"] is None or det["src"].get(obj) == "wrist"
                                     or det["at_look"]):
            m[obj + "_xyz"] = _r(det[obj])

    holding = gap > GRIP_GAP_HOLD and gap < 0.075
    act = None

    def pick(obj):
        xyz = m[obj + "_xyz"] or (PRIOR_COOKIES if obj == "cookies" else PRIOR_TOMATO)
        return {"tool": "act_pick", "args": {"x": xyz[0], "y": xyz[1], "z": xyz[2], "obj": obj}}

    def place(obj):
        b = m["basket_xyz"]
        rz = COOKIE_RELEASE_Z if obj == "cookies" else TOMATO_RELEASE_Z
        return {"tool": "act_place", "args": {"x": b[0], "y": b[1], "z": rz}}

    look = {"tool": "act_look", "args": {"x": LOOK_POSE[0], "y": LOOK_POSE[1], "z": LOOK_POSE[2]}}

    if phase == "start":
        m["holding"] = "none"
        phase = "picking_cookies"
        ct += 1
        act = pick("cookies")
    elif phase == "picking_cookies":
        if holding:
            m["holding"] = "cookies"
            phase = "placing_cookies"
            act = place("cookies")
        else:
            m["holding"] = "none"
            phase = "relook_cookies"
            looks += 1
            act = look
    elif phase == "relook_cookies":
        if holding:
            m["holding"] = "cookies"
            phase = "placing_cookies"
            act = place("cookies")
        elif ct < MAX_PICK_TRIES:
            phase = "picking_cookies"
            ct += 1
            act = pick("cookies")
        else:
            phase = "relook_tomato"
            looks += 1
            act = look
    elif phase == "placing_cookies":
        m["cookies_released"] = True
        m["holding"] = "none"
        phase = "verify_cookies"
        looks += 1
        act = look
    elif phase == "verify_cookies":
        # cookies still visible on the table (outside the basket) -> not placed
        if det["cookies"] is not None and det["src"].get("cookies") == "wrist" and ct < MAX_PICK_TRIES:
            m["cookies_released"] = False
            phase = "picking_cookies"
            ct += 1
            act = pick("cookies")
        else:
            phase = "picking_tomato"
            tt += 1
            act = pick("tomato")
    elif phase == "picking_tomato":
        if holding:
            m["holding"] = "tomato"
            phase = "placing_tomato"
            act = place("tomato")
        else:
            m["holding"] = "none"
            phase = "relook_tomato"
            looks += 1
            act = look
    elif phase == "relook_tomato":
        if holding:
            m["holding"] = "tomato"
            phase = "placing_tomato"
            act = place("tomato")
        elif tt < MAX_PICK_TRIES:
            phase = "picking_tomato"
            tt += 1
            act = pick("tomato")
        else:
            phase = "done"
            act = {"tool": "act_wait", "args": {"steps": 60}}
    elif phase == "placing_tomato":
        m["tomato_released"] = True
        m["holding"] = "none"
        phase = "verify_tomato"
        looks += 1
        act = look
    elif phase == "verify_tomato":
        if det["tomato"] is not None and det["src"].get("tomato") == "wrist" and tt < MAX_PICK_TRIES:
            m["tomato_released"] = False
            phase = "picking_tomato"
            tt += 1
            act = pick("tomato")
        else:
            phase = "done"
            act = {"tool": "act_wait", "args": {"steps": 100}}
    else:
        phase = "done"
        act = {"tool": "act_wait", "args": {"steps": 100}}

    m["phase"] = phase
    m["cookies_tries"] = ct
    m["tomato_tries"] = tt
    m["looks"] = looks
    m["cookies_released"] = bool(m["cookies_released"])
    m["tomato_released"] = bool(m["tomato_released"])
    m["notes"] = notes
    print("DECIDE", notes, "-> phase", phase, act["tool"], act["args"])
    return {"memory": m, "act": act}


# ---------------------------------------------------------------- control
def _move(robot, target, grip, tol=0.004, max_steps=300, gain=11.0, vmax=1.0, stall_steps=12,
          settle=0.0025):
    """Closed-loop move of the TCP to world target; 'reached' needs the TCP inside tol AND
    moving slower than `settle` m/step (avoids overshoot from controller lag)."""
    target = np.asarray(target, dtype=float)
    best = 1e9
    since = 0
    n = 0
    f = robot.now()
    prev = None
    while n < max_steps:
        tcp = np.asarray(f.tcp, dtype=float)
        err = target - tcp[:3]
        d = float(np.linalg.norm(err))
        speed = 0.0 if prev is None else float(np.linalg.norm(tcp[:3] - prev))
        prev = tcp[:3].copy()
        if d < tol and speed < settle:
            return f, n, "reached"
        if d < best - 0.0005:
            best = d
            since = 0
        else:
            since += 1
            if since >= stall_steps:
                return f, n, "stalled"
        a = err * np.asarray(gain, dtype=float)
        na = float(np.linalg.norm(a))
        if na > vmax:
            a = a / na * vmax
        if na < 0.08:
            a = a / max(na, 1e-9) * 0.08
        r = _rot_cmd(tcp)
        f = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    return f, n, "max_steps"


def _gripper(robot, grip, hold_xyz, min_steps=12, max_steps=45):
    f = robot.now()
    hist = []
    target = np.asarray(hold_xyz, dtype=float)
    for i in range(max_steps):
        tcp = np.asarray(f.tcp, dtype=float)
        a = np.clip((target - tcp[:3]) * 10.0, -0.3, 0.3)
        r = _rot_cmd(tcp)
        f = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        hist.append(_gap(f))
        if i >= min_steps and abs(hist[-1] - hist[-6]) < 0.0003:
            break
    return f, len(hist)


def act_pick(robot, x, y, z, obj="object"):
    """Pick an object at world (x, y) with grasp TCP height z: open, approach from above,
    descend, close, lift. Returns measured finger gap before/after lifting and TCP."""
    log = {}
    above = max(z + 0.09, 0.53)
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    if tcp[2] < above - 0.02:
        f, n, why = _move(robot, [tcp[0], tcp[1], above], -1, tol=0.01, max_steps=120, settle=1.0)
    f, n, why = _move(robot, [x, y, above], -1, tol=0.004, max_steps=250)
    log["approach"] = why
    # make sure gripper is open
    if _gap(f) < 0.07:
        f, k = _gripper(robot, -1, [x, y, above], min_steps=8, max_steps=25)
    # two-stage descent: fast to just above the grasp height (settled), then slow final approach
    f, n, why = _move(robot, [x, y, z + 0.025], -1, tol=0.003, max_steps=200, gain=[25.0, 25.0, 12.0], vmax=0.6)
    log["pre"] = why
    f, n, why = _move(robot, [x, y, z], -1, tol=0.003, max_steps=120, gain=[25.0, 25.0, 8.0], vmax=0.2,
                      stall_steps=10)
    log["descend"] = why
    log["descend_tcp"] = _r(f.tcp[:3])
    f, k = _gripper(robot, 1, [x, y, z], min_steps=15, max_steps=45)
    log["gap_closed"] = round(_gap(f), 4)
    lift = 0.62 if obj == "cookies" else 0.66
    f, n, why = _move(robot, [x, y, lift], 1, tol=0.01, max_steps=150, vmax=0.8)
    log["lift"] = why
    log["gap_lifted"] = round(_gap(f), 4)
    log["tcp"] = _r(f.tcp[:3])
    log["holding"] = bool(GRIP_GAP_HOLD < _gap(f) < 0.075)
    print("PICK", log)
    return log


def act_place(robot, x, y, z):
    """Carry the held object at travel height to above world (x, y), descend to TCP height z,
    open the gripper and retreat upward. Returns measured TCP/finger gap at each stage."""
    log = {}
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    f, n, why = _move(robot, [tcp[0], tcp[1], TRAVEL_Z], 1, tol=0.015, max_steps=150, settle=1.0)
    f, n, why = _move(robot, [x, y, TRAVEL_Z], 1, tol=0.008, max_steps=250)
    log["travel"] = why
    log["gap_travel"] = round(_gap(f), 4)
    f, n, why = _move(robot, [x, y, z], 1, tol=0.004, max_steps=200, gain=[20.0, 20.0, 12.0], vmax=0.7)
    log["descend"] = why
    log["release_tcp"] = _r(f.tcp[:3])
    f, k = _gripper(robot, -1, [x, y, z], min_steps=12, max_steps=30)
    log["gap_open"] = round(_gap(f), 4)
    f, n, why = _move(robot, [x, y, TRAVEL_Z], -1, tol=0.015, max_steps=150, settle=1.0)
    log["tcp"] = _r(f.tcp[:3])
    print("PLACE", log)
    return log


def act_look(robot, x, y, z):
    """Move (gripper open) to a viewing pose at world (x, y, z) above the table and settle,
    so the next decision can re-perceive with the wrist camera."""
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    if tcp[2] < z - 0.03:
        f, n, why = _move(robot, [tcp[0], tcp[1], z], -1, tol=0.01, max_steps=120, settle=1.0)
    f, n, why = _move(robot, [x, y, z], -1, tol=0.004, max_steps=250)
    for i in range(3):
        tcp = np.asarray(f.tcp, dtype=float)
        r = _rot_cmd(tcp)
        f = robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), -1)
    return {"look": why, "tcp": _r(f.tcp[:3]), "gap": round(_gap(f), 4)}


def act_wait(robot, steps=50):
    """Hold still with the gripper open for a monitoring period."""
    f = robot.now()
    for i in range(int(steps)):
        f = robot.step(0, 0, 0, 0, 0, 0, -1)
    return {"waited": int(steps), "tcp": _r(f.tcp[:3])}
