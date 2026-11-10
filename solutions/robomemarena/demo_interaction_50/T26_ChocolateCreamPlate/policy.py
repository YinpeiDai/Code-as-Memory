"""Policy for T26_ChocolateCreamPlate: move chocolate then cream from plate1 onto plate2.

Perception: agentview / wrist RGB-D back-projected to world points with the supplied calibration.
Colours: chocolate box = dark brown (R > B, dark), cream box = blue (B > R), plates = bright white disc.
Control: closed-loop OSC delta steps toward world targets measured with the TCP pose.
"""

GRASP_AA = [2.902, -1.203, 0.0]      # demo grasp orientation (gripper yawed 45 deg), all 50 demos
PLACE_AA = [3.1416, 0.0, 0.0]        # demo transport / release orientation
TRAVEL_Z = 0.64                       # hover / transport height (demo 0.70; lowered, see CHANGELOG R2)
J4_GUARD = -0.58                      # elbow joint (joints[3]); arm singular (full extension) at about -0.47
SHOULDER = [-0.51, 0.0, 0.75]          # world position of Panda joint-2 axis (FK fit on demos: base (-0.51,0,0.417))
WOFF_GRASP = [-0.075, 0.027, 0.214]    # wrist point (frame-5 origin) minus TCP, grasp orientation, extended arm (FK R4)
WOFF_PLACE = [-0.087, -0.020, 0.210]   # same for the place orientation (yaw 0)
WOFF_MIRROR = [-0.047, -0.076, 0.210]  # yaw +45 (WOFF_PLACE rotated by 45 deg; estimate)
D_MAX = 0.719                          # shoulder-wrist distance at full extension (FK)
D_VIEW = 0.700
D_GRASP = 0.717
_DH_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
_DH_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
_DH_AL = [0.0, -1.5707963267948966, 1.5707963267948966, 1.5707963267948966, -1.5707963267948966,
          1.5707963267948966, 1.5707963267948966]
RELEASE_DZ = 0.091                    # demo release TCP height above plate white surface
PLATE_OFF = [0.0, -0.003]             # demo release xy minus plate2 white-centroid xy
KNOWN = {
    "chocolate": {"off": [-0.004, -0.002], "off_low": [-0.007, -0.0025], "coarse": [-0.025, -0.002]},
    "cream": {"off": [0.004, -0.001], "off_low": [0.004, 0.0], "coarse": [-0.005, -0.001]},
}
MAX_TRIES = 4


def _rotm(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        v = R[:, i] + np.eye(3)[:, i]
        v = v / max(np.linalg.norm(v), 1e-9)
        return v * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _cloud(K, E, depth):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    Wp = (P - t) @ R
    return Wp.reshape(H, W, 3), z > 0


def _masks(rgb, Wp, valid, tz):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    X, Y, Z = Wp[..., 0], Wp[..., 1], Wp[..., 2]
    ws = valid & (X > -0.15) & (X < 0.5) & (Y > -0.5) & (Y < 0.5)
    above = ws & (Z > tz + 0.010) & (Z < tz + 0.16)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    return {
        "chocolate": above & (r > b + 10) & (r + g + b < 300),
        "cream": above & (b > r + 12),
        "white": ws & (Z > tz - 0.02) & (Z < tz + 0.03) & (mn > 170) & (mx - mn < 20),
    }


def _robust(P, rad=0.04):
    if P is None or len(P) < 8:
        return None
    m = np.median(P, 0)
    Q = P
    for _ in range(4):
        d = np.linalg.norm(P[:, :2] - m[:2], axis=1)
        Q = P[d < rad]
        if len(Q) < 8:
            return None
        m = np.median(Q, 0)
    return Q


def _table_z(Wp, valid):
    X, Y, Z = Wp[..., 0], Wp[..., 1], Wp[..., 2]
    ws = valid & (X > -0.1) & (X < 0.45) & (Y > -0.45) & (Y < 0.45) & (Z > 0.2) & (Z < 0.7)
    if ws.sum() < 100:
        return None
    return float(np.median(Z[ws]))


def _two_clusters(P):
    if len(P) < 15:
        return []
    xy = P[:, :2]
    a = xy[0]
    c1 = xy[int(np.argmax(np.linalg.norm(xy - a, axis=1)))]
    c2 = xy[int(np.argmax(np.linalg.norm(xy - c1, axis=1)))]
    if np.linalg.norm(c1 - c2) < 0.08:
        return [P]
    lab = None
    for _ in range(8):
        d1 = np.linalg.norm(xy - c1, axis=1)
        d2 = np.linalg.norm(xy - c2, axis=1)
        lab = d1 < d2
        if lab.sum() == 0 or (~lab).sum() == 0:
            return [P]
        c1 = np.median(xy[lab], 0)
        c2 = np.median(xy[~lab], 0)
    out = []
    for m in (lab, ~lab):
        Q = _robust(P[m], 0.07)
        if Q is not None and len(Q) >= 15:
            out.append(Q)
    return out


def _agent_scene(frame, tz=None):
    """World-frame detections from the agentview RGB-D frame."""
    K = frame.agentview_K
    E = frame.agentview_extrinsic
    Wp, valid = _cloud(K, E, frame.agentview_depth)
    if tz is None:
        tz = _table_z(Wp, valid)
    if tz is None:
        return {"table_z": None, "objects": {}, "plates": []}
    m = _masks(frame.agentview_rgb, Wp, valid, tz)
    objs = {}
    for name in KNOWN:
        Q = _robust(Wp[m[name]])
        if Q is not None:
            objs[name] = {"pos": np.median(Q, 0).tolist(), "n": int(len(Q)),
                          "top": float(np.percentile(Q[:, 2], 95))}
    plates = []
    for Q in _two_clusters(Wp[m["white"]]):
        plates.append({"pos": np.median(Q, 0).tolist(), "n": int(len(Q))})
    return {"table_z": tz, "objects": objs, "plates": plates, "Wp": Wp, "valid": valid, "masks": m}


def _count_near(Wp, mask, xy, rad):
    P = Wp[mask]
    if len(P) == 0:
        return 0
    return int((np.linalg.norm(P[:, :2] - np.asarray(xy[:2])[None, :], axis=1) < rad).sum())


def _wrist_find(frame, name, guess, tz, rad=0.10):
    K = frame.wrist_K
    E = frame.wrist_extrinsic
    Wp, valid = _cloud(K, E, frame.wrist_depth)
    m = _masks(frame.wrist_rgb, Wp, valid, tz)[name]
    P = Wp[m]
    if guess is not None and len(P):
        P = P[np.linalg.norm(P[:, :2] - np.asarray(guess[:2])[None, :], axis=1) < rad]
    Q = _robust(P)
    if Q is None or len(Q) < 40:
        return None
    xy = Q[:, :2] - Q[:, :2].mean(0)
    w, v = np.linalg.eigh(xy.T @ xy / len(xy))
    yaw = math.atan2(float(v[1, 1]), float(v[0, 1]))
    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw <= -math.pi / 2:
        yaw += math.pi
    elong = float(math.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12)))
    return {"pos": np.median(Q, 0).tolist(), "top": float(np.percentile(Q[:, 2], 95)), "n": int(len(Q)),
            "yaw": yaw, "elong": elong}


def _neighbors(frame, name, pos, tz, rad=0.10):
    """Pixels of the other known boxes within rad of pos (wrist RGB-D)."""
    Wp, valid = _cloud(frame.wrist_K, frame.wrist_extrinsic, frame.wrist_depth)
    m = _masks(frame.wrist_rgb, Wp, valid, tz)
    Z = Wp[..., 2]
    band = (Z > tz + 0.02) & (Z < tz + 0.07)
    n = 0
    for other in KNOWN:
        if other != name:
            P = Wp[m[other] & band]
            if len(P):
                d = np.linalg.norm(P[:, :2] - np.asarray(pos[:2])[None, :], axis=1)
                n += int(((d > 0.03) & (d < rad)).sum())
    return n


def _finger_hits(frame, center, gz, tz, yaws):
    """For each candidate gripper yaw: wrist-depth points above the fingertip level inside the open-finger
    footprints (inner 3.6 cm .. outer 5.6 cm from the grasp centre along the closing axis, +-1.5 cm across)."""
    Wp, valid = _cloud(frame.wrist_K, frame.wrist_extrinsic, frame.wrist_depth)
    Z = Wp[..., 2]
    sel = valid & (Z > max(gz - 0.004, tz + 0.015)) & (Z < tz + 0.10)
    P = Wp[sel]
    rel = P[:, :2] - np.asarray(center[:2])[None, :]
    near = np.linalg.norm(rel, axis=1) < 0.09
    rel = rel[near]
    out = []
    for yaw in yaws:
        d = np.array([math.sin(yaw), -math.cos(yaw)])
        e = np.array([math.cos(yaw), math.sin(yaw)])
        a = np.abs(rel @ d)
        b = np.abs(rel @ e)
        out.append(int(((a > 0.036) & (a < 0.056) & (b < 0.015)).sum()))
    return out


def _neighbor_centroid(frame, name, pos, tz):
    """World xy centroid of the closest other known box (wrist RGB-D), or None."""
    Wp, valid = _cloud(frame.wrist_K, frame.wrist_extrinsic, frame.wrist_depth)
    m = _masks(frame.wrist_rgb, Wp, valid, tz)
    Z = Wp[..., 2]
    band = (Z > tz + 0.015) & (Z < tz + 0.07)
    best = None
    for other in KNOWN:
        if other == name:
            continue
        P = Wp[m[other] & band]
        if len(P) == 0:
            continue
        d = np.linalg.norm(P[:, :2] - np.asarray(pos[:2])[None, :], axis=1)
        P = P[(d > 0.05) & (d < 0.12)]
        Q = _robust(P, 0.04)
        if Q is not None and len(Q) >= 40:
            c = np.median(Q, 0)
            dd = float(np.linalg.norm(c[:2] - np.asarray(pos[:2])))
            if best is None or dd < best[1]:
                best = (c[:2].tolist(), dd)
    return None if best is None else best[0]


def _yaw_aa(yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return _rotvec(Rz @ _rotm(PLACE_AA)).tolist()


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _reach(p, woff):
    """Shoulder-to-wrist distance for a TCP position p (singular at D_MAX)."""
    return float(np.linalg.norm(np.asarray(p, dtype=float) + np.asarray(woff) - np.asarray(SHOULDER)))


def _clampx(p, woff=None, dmax=D_VIEW):
    """Pull the target back in x (toward the robot) until the shoulder-wrist distance is <= dmax."""
    if woff is None:
        woff = WOFF_GRASP
    q = [float(v) for v in p]
    if _reach(q, woff) <= dmax:
        return q
    dy = q[1] + woff[1] - SHOULDER[1]
    dz = q[2] + woff[2] - SHOULDER[2]
    r = dmax * dmax - dy * dy - dz * dz
    if r > 0:
        q[0] = SHOULDER[0] - woff[0] + math.sqrt(r)
    return q


def _fk_radial(joints):
    """Forward kinematics (base frame = world rotation): shoulder-wrist distance and its unit direction."""
    M = np.eye(4)
    pts = []
    for i in range(5):
        th = float(joints[i])
        ca, sa = math.cos(_DH_AL[i]), math.sin(_DH_AL[i])
        ct, st = math.cos(th), math.sin(th)
        T = np.array([[ct, -st, 0, _DH_A[i]], [st * ca, ct * ca, -sa, -_DH_D[i] * sa],
                      [st * sa, ct * sa, ca, _DH_D[i] * ca], [0, 0, 0, 1.0]])
        M = M @ T
        pts.append(M[:3, 3].copy())
    v = pts[4] - pts[1]
    d = float(np.linalg.norm(v))
    return d, v / max(d, 1e-9)


def _goto(robot, pos, aa, grip, tol=0.004, rtol=0.03, max_steps=200, vmax=1.0, gain=12.0):
    R_t = _rotm(aa)
    pos = np.asarray(pos, dtype=float)
    n = 0
    err = None
    hist = []
    guarded = 0
    for n in range(max_steps):
        tcp = robot.now().tcp
        err = pos - tcp[:3]
        rv = _rotvec(R_t @ _rotm(tcp[3:6]).T)
        if np.linalg.norm(err) < tol and np.linalg.norm(rv) < rtol:
            break
        hist.append(np.array(tcp[:3], dtype=float))
        # stalled close to the target (controller steady-state error or contact): stop pushing
        if len(hist) > 10 and np.linalg.norm(hist[-1] - hist[-9]) < 0.0007 and np.linalg.norm(err) < 0.015 \
                and np.linalg.norm(rv) < max(rtol, 0.05):
            break
        if len(hist) > 25 and np.linalg.norm(hist[-1] - hist[-21]) < 0.001:
            break
        jn = robot.now().joints
        vm = vmax if float(jn[3]) < J4_GUARD - 0.2 else min(vmax, 0.3)
        a = np.clip(err * gain, -vm, vm)
        # near full arm extension: remove any outward (radial) motion and fold back slightly
        if float(jn[3]) > J4_GUARD:
            d, u = _fk_radial(jn)
            out = float(a @ u)
            if out > -0.1:
                a = a - (out + 0.1) * u
            guarded += 1
        r = np.clip(rv * 1.5, -1.0, 1.0)
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)
    tcp = robot.now().tcp
    return float(np.linalg.norm(pos - tcp[:3]))


def _hold(robot, pos, aa, grip, n):
    R_t = _rotm(aa)
    pos = np.asarray(pos, dtype=float)
    for _ in range(n):
        tcp = robot.now().tcp
        err = pos - tcp[:3]
        rv = _rotvec(R_t @ _rotm(tcp[3:6]).T)
        a = np.clip(err * 12.0, -1, 1)
        r = np.clip(rv * 1.5, -1, 1)
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)


def _close(robot, pos, aa):
    last = _width(robot.now())
    still = 0
    for i in range(70):
        _hold(robot, pos, aa, 1.0, 1)
        w = _width(robot.now())
        still = still + 1 if abs(w - last) < 0.0002 else 0
        last = w
        if i >= 15 and still >= 8:
            break
    return _width(robot.now())


def _goal_objects(goal):
    g = (goal or "").lower()
    found = [(g.find(k), k) for k in KNOWN if g.find(k) >= 0]
    found.sort()
    names = [k for _, k in found]
    return names if names else ["chocolate", "cream"]


def _pick_plate2(plates, objs, old):
    if not plates:
        return old
    if old is not None:
        return old
    cents = [o["pos"] for o in objs.values()]
    best = None
    bestd = -1.0
    for p in plates:
        if cents:
            d = min(float(np.linalg.norm(np.asarray(p["pos"][:2]) - np.asarray(c[:2]))) for c in cents)
        else:
            d = 0.0
        if d > bestd:
            bestd = d
            best = p
    if cents and bestd < 0.08:
        return old
    return best["pos"]


def perceive_decide(obs, goal, memory, last):
    """Track task progress (objects placed, held object, plate2 location) and choose the next action."""
    mem = dict(memory or {})
    order = _goal_objects(goal)
    placed = list(mem.get("placed") or [])
    tries = list(mem.get("tries") or [])
    holding = mem.get("holding") or ""
    plate2 = mem.get("plate2")
    tz = mem.get("table_z")
    cpos = {"chocolate": mem.get("choc_pos"), "cream": mem.get("cream_pos")}
    notes = ""
    stage = mem.get("stage") or "start"
    lasts = last or ""
    fr = obs.current
    sc = _agent_scene(fr, tz)
    if tz is None and sc["table_z"] is not None:
        tz = sc["table_z"]
    if tz is None:
        tz = 0.431
    objs = sc["objects"]
    plate2 = _pick_plate2(sc["plates"], objs, plate2)
    width = _width(fr)

    # outcome of the previous action
    fail_mode = mem.get("fail_mode") or ""
    if lasts.startswith("act_pick"):
        if '"grasped": true' in lasts and width > 0.008:
            pass
        else:
            k = lasts.find('"mode": "')
            if k >= 0 and holding:
                fail_mode = holding + ":" + lasts[k + 9:].split('"')[0]
            holding = ""
    if lasts.startswith("act_place"):
        obj = holding
        holding = ""
        if obj and '"on_plate": true' in lasts:
            if obj not in placed:
                placed.append(obj)
            notes = obj + " verified on plate2"
        elif obj:
            o = objs.get(obj)
            far = o is not None and plate2 is not None and o["n"] >= 15 and \
                float(np.linalg.norm(np.asarray(o["pos"][:2]) - np.asarray(plate2[:2]))) > 0.08
            if far or '"reason": "object lost' in lasts:
                notes = obj + " place failed; object seen elsewhere"
            else:
                if obj not in placed:
                    placed.append(obj)
                notes = obj + " place not verified but not seen elsewhere; assume on plate2"
    if holding and width < 0.006:
        notes = "lost " + holding
        holding = ""

    # re-check placed objects that are visible elsewhere (dropped / knocked off)
    if plate2 is not None:
        for name in list(placed):
            o = objs.get(name)
            if o is not None and o["n"] > 25:
                d = float(np.linalg.norm(np.asarray(o["pos"][:2]) - np.asarray(plate2[:2])))
                if d > 0.12:
                    placed.remove(name)
                    notes = name + " seen off plate2; retry"

    # an object seen resting on plate2 counts as placed (agentview check, independent of the act report)
    if plate2 is not None and not holding:
        for name in order:
            o = objs.get(name)
            if o is not None and name not in placed and o["n"] >= 15:
                d = float(np.linalg.norm(np.asarray(o["pos"][:2]) - np.asarray(plate2[:2])))
                if d < 0.08:
                    placed.append(name)
                    notes = notes + "; " + name + " seen on plate2"
    for name in KNOWN:
        o = objs.get(name)
        if o is not None and name not in placed and name != holding:
            co = KNOWN[name].get("coarse", [0.0, 0.0])
            cpos[name] = [o["pos"][0] + co[0], o["pos"][1] + co[1], o["pos"][2]]

    remaining = [n for n in order if n not in placed]
    if holding:
        act = {"tool": "act_place", "args": {"name": holding, "plate": plate2 if plate2 else [0.17, 0.12, tz],
                                              "table_z": tz}}
        stage = "place_" + holding
    elif remaining:
        name = remaining[0]
        if tries.count(name) >= MAX_TRIES and len(remaining) > 1:
            name = remaining[1]
        k = tries.count(name)
        guess = cpos.get(name)
        avoid = fail_mode.split(":", 1)[1] if fail_mode.startswith(name + ":") else ""
        act = {"tool": "act_pick", "args": {"name": name, "guess": guess, "table_z": tz, "attempt": k,
                                             "avoid": avoid}}
        tries.append(name)
        holding = name
        stage = "pick_" + name
    else:
        act = {"tool": "act_wait", "args": {"steps": 40}}
        stage = "monitor"
    out = {
        "stage": stage,
        "plate2": plate2,
        "table_z": tz,
        "choc_pos": cpos.get("chocolate"),
        "cream_pos": cpos.get("cream"),
        "placed": placed,
        "holding": holding,
        "tries": tries,
        "notes": notes,
        "fail_mode": fail_mode,
    }
    return {"memory": out, "act": act}


def _view_point(x, y):
    """Viewing / travel point above (x, y): TRAVEL_Z when comfortably reachable, else lower (z 0.56) and clamped."""
    p = [x, y, TRAVEL_Z]
    if _reach(p, WOFF_GRASP) <= D_VIEW - 0.015:
        return p
    return _clampx([x, y, 0.56], WOFF_GRASP, D_VIEW)


def act_pick(robot, name, guess=None, table_z=0.431, attempt=0, avoid=""):
    """Hover above the named object (coarse world xy guess), refine its position with the wrist RGB-D,
    descend with the demo 45-degree grasp orientation, close the gripper, lift. Returns measured grasp result."""
    tz = float(table_z)
    off = KNOWN.get(name, {"off": [0.0, 0.0]})["off"]
    if guess is None:
        guess = [0.2, -0.2 if name == "chocolate" else -0.15, tz + 0.03]
    start = robot.now().tcp
    if start[2] < 0.56:
        _goto(robot, _clampx([start[0], start[1], 0.60], WOFF_GRASP, D_VIEW), GRASP_AA, -1.0, tol=0.02, rtol=0.5,
              max_steps=80, vmax=0.7)
    tgt = _view_point(guess[0] + off[0], guess[1] + off[1])
    # fast to a point 5 cm short (toward the robot), then creep out: the arm overshoots ~1.4 cm (R8/R9)
    _goto(robot, [tgt[0] - 0.05, tgt[1], tgt[2]], GRASP_AA, -1.0, tol=0.01, rtol=0.06, max_steps=200, vmax=0.8)
    _goto(robot, tgt, GRASP_AA, -1.0, tol=0.005, rtol=0.03, max_steps=150, vmax=0.3)
    seen = None
    shifts = []
    est = list(guess)
    for it in range(4):
        _hold(robot, tgt, GRASP_AA, -1.0, 4)
        f = _wrist_find(robot.now(), name, est, tz, rad=0.10 if it == 0 else 0.06)
        if f is None:
            break
        seen = f
        est = f["pos"]
        new = _view_point(est[0] + off[0], est[1] + off[1])
        sh = float(np.linalg.norm(np.asarray(new[:2]) - np.asarray(tgt[:2])))
        shifts.append(round(sh, 4))
        tgt = new
        _goto(robot, tgt, GRASP_AA, -1.0, tol=0.004, rtol=0.03, max_steps=120, vmax=0.6)
        if sh < 0.003:
            break
    if seen is None:
        return {"grasped": False, "reason": "object not seen by wrist camera", "name": name,
                "tcp": robot.now().tcp[:3].tolist(), "j4": float(robot.now().joints[3])}
    gx = est[0] + off[0]
    gy = est[1] + off[1]
    nb = _neighbors(robot.now(), name, est, tz, 0.10)
    # second look from (nearly) directly above at lower height
    low = _clampx([gx, gy, 0.56], WOFF_GRASP, 0.708)
    _goto(robot, low, GRASP_AA, -1.0, tol=0.004, rtol=0.03, max_steps=150, vmax=0.5)
    _hold(robot, low, GRASP_AA, -1.0, 4)
    f2 = _wrist_find(robot.now(), name, est, tz, rad=0.04)
    low_shift = None
    if f2 is not None and f2["n"] >= 150:
        d = float(np.linalg.norm(np.asarray(f2["pos"][:2]) - np.asarray(est[:2])))
        low_shift = round(d, 4)
        if d < 0.02:
            est = f2["pos"]
            seen = f2
            offl = KNOWN.get(name, {}).get("off_low", off)
            gx = est[0] + offl[0]
            gy = est[1] + offl[1]
    top = seen["top"]
    # orientation: fingers across the box short side ("aligned", most stable) when both open fingers have free
    # space; else the 45-degree diagonal (demo) whose finger footprints hit fewer points (neighbour box, rim)
    byaw = seen.get("yaw", 0.0) if seen.get("elong", 1.0) > 1.4 else 0.0
    gz_a = min(max(top - 0.024 - 0.004 * attempt, tz + 0.008), tz + 0.05)
    gz_d = min(max(top - 0.030 - 0.004 * attempt, tz + 0.008), tz + 0.05)
    hits = _finger_hits(robot.now(), [gx, gy], gz_d, tz, [byaw, -math.pi / 4, math.pi / 4])
    if hits[0] <= 3:
        mode = "aligned"
        aa = _yaw_aa(byaw)
        woff = WOFF_PLACE
        gz = gz_a
    else:
        mode = "diag45_mirror" if (hits[1] > hits[2] + 30 and hits[1] > 1.5 * hits[2]) else "diag45"
        # neighbour box offset: put the finger on the neighbour's side at the x-side away from it
        # (-45° puts that finger at +x,+y; +45° at -x,+y). R14/R28/R36: -45° failed, +45° held (cream at +x).
        nbc = _neighbor_centroid(robot.now(), name, [gx, gy], tz)
        if nbc is not None:
            ndx, ndy = nbc[0] - gx, nbc[1] - gy
            side = ndx * (1.0 if ndy > 0 else -1.0)
            if side > 0.01:
                mode = "diag45_mirror"
            elif side < -0.01:
                mode = "diag45"
        if avoid == mode:
            mode = "diag45" if mode == "diag45_mirror" else "diag45_mirror"
        aa = GRASP_AA if mode == "diag45" else _yaw_aa(math.pi / 4)
        woff = WOFF_GRASP if mode == "diag45" else WOFF_MIRROR
        gz = gz_d
    pre = _clampx([gx, gy, gz + 0.06], woff, 0.712)
    gpos = np.array(_clampx([gx, gy, gz], woff, D_GRASP))
    _goto(robot, pre, aa, -1.0, tol=0.005, max_steps=150, vmax=0.5)
    _goto(robot, gpos, aa, -1.0, tol=0.003, max_steps=120, vmax=0.3)
    _hold(robot, gpos, aa, -1.0, 4)
    reached = robot.now().tcp[:3].tolist()
    blocked = []
    if reached[2] - gz > 0.007:
        blocked.append(round(reached[2] - gz, 4))   # finger resting on something (plate rim / box); close anyway
    w = _close(robot, gpos, aa)
    lift = _clampx([gpos[0], gpos[1], TRAVEL_Z], woff, D_VIEW)
    _goto(robot, [gpos[0], gpos[1], gpos[2] + 0.02], aa, 1.0, tol=0.004, max_steps=30, vmax=0.5)
    w1 = _width(robot.now())
    _goto(robot, [0.5 * (gpos[0] + lift[0]), gpos[1], 0.54], aa, 1.0, tol=0.01, max_steps=80, vmax=0.5)
    _goto(robot, lift, aa, 1.0, tol=0.01, max_steps=150, vmax=0.7)
    w2 = _width(robot.now())
    ok = w2 > 0.008
    return {"grasped": bool(ok), "name": name, "mode": mode, "yaw": seen.get("yaw"), "neighbors": nb, "hits": hits,
            "blocked": blocked,
            "obj_est": est, "obj_top": top, "grasp_z": gz, "width_2cm": w1,
            "target": gpos.tolist(), "reached": reached, "width_closed": w, "width_lifted": w2, "shifts": shifts,
            "low_shift": low_shift, "j4": float(robot.now().joints[3])}


def act_place(robot, name, plate, table_z=0.431):
    """Carry the held object at travel height to above plate2 (world xyz of its white surface), lower to the
    demo release height, open, lift, then check with the agentview/wrist RGB-D that the object is on plate2."""
    tz = float(table_z)
    px = plate[0] + PLATE_OFF[0]
    py = plate[1] + PLATE_OFF[1]
    pz = plate[2] if len(plate) > 2 and plate[2] is not None else tz
    tcp = robot.now().tcp
    if tcp[2] < TRAVEL_Z - 0.03:
        _goto(robot, _clampx([tcp[0], tcp[1], TRAVEL_Z], WOFF_GRASP, D_VIEW), tcp[3:6].tolist(), 1.0, tol=0.01,
              rtol=0.5, max_steps=100, vmax=0.7)
    above = _clampx([px, py, TRAVEL_Z], WOFF_PLACE, D_VIEW)
    _goto(robot, above, PLACE_AA, 1.0, tol=0.006, rtol=0.03, max_steps=250, vmax=0.8)
    if _width(robot.now()) < 0.006:
        return {"on_plate": False, "reason": "object lost during transport", "name": name}
    rz = pz + RELEASE_DZ
    rel = _clampx([px, py, rz], WOFF_PLACE, D_GRASP)
    _goto(robot, rel, PLACE_AA, 1.0, tol=0.004, max_steps=150, vmax=0.6)
    _hold(robot, rel, PLACE_AA, 1.0, 3)
    _hold(robot, rel, PLACE_AA, -1.0, 15)
    _goto(robot, above, PLACE_AA, -1.0, tol=0.01, max_steps=120, vmax=0.7)
    _hold(robot, above, PLACE_AA, -1.0, 6)
    fr = robot.now()
    sc = _agent_scene(fr, tz)
    n_agent = _count_near(sc["Wp"], sc["masks"][name], [px, py], 0.08) if "Wp" in sc else 0
    wf = _wrist_find(fr, name, [px, py], tz, rad=0.09)
    n_wrist = wf["n"] if wf else 0
    ok = n_agent >= 10 or n_wrist >= 60
    return {"on_plate": bool(ok), "name": name, "agent_px": n_agent, "wrist_px": n_wrist,
            "wrist_pos": wf["pos"] if wf else None, "released_at": rel}


def act_wait(robot, steps=40):
    """Hold the arm still with the gripper open for a number of steps (lets the scorer settle)."""
    tcp = robot.now().tcp
    _hold(robot, tcp[:3], PLACE_AA, -1.0, int(steps))
    return {"waited": int(steps)}
