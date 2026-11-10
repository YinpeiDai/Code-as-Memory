"""T24 CookiesPopcornMicrowave: open microwave door, put cookies in, then put popcorn
where the cookies were placed (inside the microwave).

Perception: agentview depth -> world point cloud (calibration supplied per frame).
Control: waypoint P-controller on measured TCP (per-axis clip, as in the expert demos).
"""

GAIN = 8.0          # action units per metre of position error (expert ~8)
ROT_GAIN = 2.0      # action units per radian of orientation error (expert ~2)
ROT_CLIP = 0.5
# Orientation targets (rotation matrices, columns = gripper axes in world)
Q_DOOR = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]   # approach +y, fingers along x
Q_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]   # approach -z, fingers along y
GRASP_Z = 0.438     # TCP height for grasping the flat boxes (demo median 0.4382)
# Offsets regressed on 50 demos (frame 0 perception -> expert key poses)
H_FROM_YMIN = 0.0122      # handle y = 1st percentile y of microwave points + this (std 0.3 mm)
H_FROM_HX = -0.0230       # handle x = mean x of front-most (handle) points + this (std 0.2 mm)
H_FROM_BODYX = -0.1241    # handle x = max x of microwave body + this (std 0.1 mm)
H_Z_DEFAULT = 0.582
OBJ_OFF = [-0.0055, 0.0022]   # grasp xy = cluster bbox centre + this
DEFAULT_H = [0.047, 0.414, 0.582]
DEFAULT_C = [0.164, -0.094]
DEFAULT_P = [0.164, -0.192]
CAVITY_PRE = [-0.048, 0.034, 0.018]   # pre-place pose inside the cavity mouth, relative to handle
PLACE_COOKIE = [-0.028, 0.056, -0.0045]  # expert cookie release pose relative to handle
PLACE_POP = [-0.055, 0.050, 0.010]       # popcorn release pose (beside the cookies)
PULL = [-0.46, -0.186, 0.004]          # door pull target relative to handle


def _rv2mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _mat2rv(m):
    c = (float(np.trace(m)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        mm = (m + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(mm)))
        v = mm[:, i] / math.sqrt(max(float(mm[i, i]), 1e-12))
        return v * th
    w = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_target(name):
    if name == "door":
        return np.array(Q_DOOR)
    if name == "down":
        return np.array(Q_DOWN)
    return None


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _world_points(frame):
    """Agentview depth -> (256,256,3) world points and validity mask."""
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    v, u = np.mgrid[0:256, 0:256]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, d], -1)
    w = (pc - E[:, 3]) @ E[:, :3]
    return w, d > 0


def _perceive_microwave(frame):
    """Handle pose from the front-most (min-y) microwave points. None if not found."""
    w, ok = _world_points(frame)
    m = ok & (w[..., 2] > 0.45) & (w[..., 2] < 0.75) & (w[..., 1] > 0.2) & (w[..., 1] < 0.9) & (np.abs(w[..., 0]) < 0.6)
    p = w[m]
    if len(p) < 300:
        return None
    ymin = float(np.percentile(p[:, 1], 1))
    hp = p[p[:, 1] < ymin + 0.015]
    body = p[p[:, 1] > ymin + 0.03]
    if len(hp) < 15 or len(body) < 200:
        return None
    hx1 = float(hp[:, 0].mean()) + H_FROM_HX
    hx2 = float(body[:, 0].max()) + H_FROM_BODYX
    hx = hx2 if abs(hx1 - hx2) > 0.02 else 0.5 * (hx1 + hx2)
    hz_meas = float(hp[:, 2].mean()) + 0.0201
    hz = H_Z_DEFAULT if abs(hz_meas - H_Z_DEFAULT) < 0.02 else hz_meas
    return {"handle": [hx, ymin + H_FROM_YMIN, hz], "n": int(len(p)),
            "hx_pair": [hx1, hx2], "body_ymin": float(body[:, 1].min())}


def _clusters(pts, cell=0.01):
    """Grid flood-fill clustering of xy points; returns list of index arrays."""
    if len(pts) == 0:
        return []
    ij = np.floor(pts[:, :2] / cell).astype(int)
    keys = {}
    for n in range(len(ij)):
        keys.setdefault((int(ij[n, 0]), int(ij[n, 1])), []).append(n)
    seen = set()
    out = []
    for k in keys:
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        members = []
        while stack:
            c = stack.pop()
            members.extend(keys[c])
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    nb = (c[0] + a, c[1] + b)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(np.array(members))
    return out


def _perceive_objects(frame, tcp=None):
    """Small boxes on the table: returns list of {xy, rb, n, ztop} sorted by size."""
    w, ok = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb, dtype=float)
    m = ok & (w[..., 2] > 0.437) & (w[..., 2] < 0.50) & (w[..., 0] > -0.15) & (w[..., 0] < 0.40) \
        & (w[..., 1] > -0.45) & (w[..., 1] < 0.25)
    p = w[m]
    c = rgb[m]
    if tcp is not None and len(p):
        far = np.linalg.norm(p[:, :2] - np.asarray(tcp[:2])[None, :], axis=1) > 0.06
        far = far | (float(tcp[2]) > 0.55)
        p = p[far]
        c = c[far]
    objs = []
    for idx in _clusters(p):
        if len(idx) < 12:
            continue
        q = p[idx]
        mid = (q[:, :2].min(0) + q[:, :2].max(0)) / 2.0
        ext = q[:, :2].max(0) - q[:, :2].min(0)
        if ext[0] > 0.15 or ext[1] > 0.15:
            continue
        rb = float((c[idx, 0] - c[idx, 2]).mean())
        objs.append({"xy": [float(mid[0] + OBJ_OFF[0]), float(mid[1] + OBJ_OFF[1])],
                     "rb": rb, "n": int(len(idx)), "ztop": float(q[:, 2].max())})
    objs.sort(key=lambda o: -o["n"])
    return objs


def _label_objects(objs):
    """Assign cookies (red/orange, R-B high) and popcorn (blue/white, R-B low)."""
    cookie, pop = None, None
    cand = [o for o in objs if o["n"] >= 25]
    if len(cand) >= 2:
        a, b = sorted(cand[:2], key=lambda o: -o["rb"])
        cookie, pop = a, b
    elif len(cand) == 1:
        o = cand[0]
        if o["rb"] > 30:
            cookie = o
        else:
            pop = o
    return cookie, pop


def _door_open_count(frame, handle):
    w, ok = _world_points(frame)
    hx, hy = handle[0], handle[1]
    m = ok & (w[..., 2] > 0.47) & (w[..., 2] < 0.63) & (w[..., 1] > hy - 0.22) & (w[..., 1] < hy - 0.02) \
        & (w[..., 0] > hx - 0.32) & (w[..., 0] < hx - 0.2)
    return int(m.sum())


def _near(xy, ref, tol):
    return xy is not None and ref is not None and math.hypot(xy[0] - ref[0], xy[1] - ref[1]) < tol


def _add(a, b):
    return [float(a[0] + b[0]), float(a[1] + b[1]), float(a[2] + b[2])]


def _wp(pos, rot, grip, steps, tol=0.004, rtol=0.05):
    return {"pos": [float(v) for v in pos], "rot": rot, "grip": float(grip), "steps": int(steps),
            "tol": float(tol), "rtol": float(rtol)}


FIELDS = ["stage", "handle", "cookie", "popcorn", "place", "door_open", "tries", "notes"]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive at the first frame, then open door, place cookies, place popcorn."""
    mem = {k: memory.get(k) for k in FIELDS}
    fr = obs.current
    tcp = [float(v) for v in fr.tcp]
    width = _width(fr)
    stage = mem["stage"] or "init"
    tries = int(mem["tries"] or 0)
    notes = mem["notes"] or ""
    act = None

    def note(s):
        return (notes + "|" + s)[-600:]

    if stage == "init":
        mw = _perceive_microwave(fr)
        objs = _perceive_objects(fr)
        ck, pp = _label_objects(objs)
        mem["handle"] = mw["handle"] if mw else list(DEFAULT_H)
        mem["cookie"] = ck["xy"] if ck else list(DEFAULT_C)
        mem["popcorn"] = pp["xy"] if pp else list(DEFAULT_P)
        mem["door_open"] = False
        notes = note("init mw=%s ck=%s pp=%s" % (mw is not None, ck is not None, pp is not None))
        stage = "approach_handle"
        tries = 0

    H = mem["handle"] or list(DEFAULT_H)
    C = mem["cookie"] or list(DEFAULT_C)
    P = mem["popcorn"] or list(DEFAULT_P)

    if stage == "approach_handle":
        dy = 0.008 * tries  # small extra depth on retries
        act = {"tool": "act_path", "args": {"waypoints": [
            _wp([H[0], H[1] - 0.17, H[2]], "door", -1, 70, 0.012, 0.06),
            _wp([H[0], H[1] + dy, H[2]], "door", -1, 50, 0.003, 0.04)]}}
        stage = "grip_handle"
    elif stage == "grip_handle":
        act = {"tool": "act_gripper", "args": {"close": True, "steps": 20}}
        stage = "check_handle"
    elif stage == "check_handle":
        if width > 0.008:
            act = {"tool": "act_pull", "args": {"target": _add(H, PULL), "steps": 40, "release_steps": 12}}
            stage = "goto_cookie"
            notes = note("handle w=%.3f" % width)
        else:
            notes = note("handle miss w=%.3f" % width)
            tries += 1
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp(tcp[:3], None, -1, 12, 0.003, 9.0),
                _wp([H[0], H[1] - 0.12, H[2]], "door", -1, 40, 0.01, 0.06)]}}
            stage = "approach_handle" if tries < 3 else "goto_cookie"
    if stage == "goto_cookie" and act is None:
        act = {"tool": "act_path", "args": {"waypoints": [
            _wp([C[0], C[1], 0.67], "down", -1, 90, 0.01, 0.05)]}}
        stage = "check_door"
    elif stage == "check_door":
        cnt = _door_open_count(fr, H)
        mem["door_open"] = cnt > 60
        notes = note("door=%d" % cnt)
        if cnt <= 60 and tries < 2:
            # door evidence missing: if the closed-door handle is still seen, redo the opening
            mw = _perceive_microwave(fr)
            if mw is not None and _near(mw["handle"], H, 0.04):
                H = mw["handle"]
                mem["handle"] = H
                tries += 1
                notes = note("door closed, retry open")
                act = {"tool": "act_path", "args": {"waypoints": [
                    _wp([H[0], H[1] - 0.17, H[2] + 0.05], "door", -1, 80, 0.015, 0.06)]}}
                stage = "approach_handle"
        objs = _perceive_objects(fr, tcp)
        ck, pp = _label_objects(objs)
        if ck and _near(ck["xy"], C, 0.05):
            C = ck["xy"]
            mem["cookie"] = C
        if act is None:
            stage = "grasp_cookie"
            tries = 0
    if stage == "grasp_cookie" and act is None:
        act = {"tool": "act_grasp", "args": {"xy": [C[0], C[1]], "z": GRASP_Z, "close_steps": 20}}
        stage = "check_cookie"
    elif stage == "check_cookie":
        if width > 0.02:
            pre = _add(H, CAVITY_PRE)
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([C[0], C[1], 0.56], "down", 1, 45, 0.01, 0.05),
                _wp([0.03, 0.05, 0.56], "down", 1, 50, 0.01, 0.05),
                _wp([0.03, 0.05, 0.56], "door", 1, 50, 0.01, 0.03),
                _wp([pre[0], pre[1] - 0.15, pre[2]], "door", 1, 60, 0.01, 0.04),
                _wp(pre, "door", 1, 60, 0.005, 0.04),
                _wp(_add(H, PLACE_COOKIE), "door", 1, 25, 0.003, 0.04)]}}
            stage = "release_cookie"
            notes = note("cookie w=%.3f" % width)
        else:
            notes = note("cookie miss w=%.3f" % width)
            tries += 1
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([tcp[0], tcp[1], 0.60], "down", -1, 40, 0.01, 0.08)]}}
            stage = "regrasp_cookie"
    elif stage == "regrasp_cookie":
        objs = _perceive_objects(fr, tcp)
        ck, pp = _label_objects(objs)
        if ck:
            C = ck["xy"]
            mem["cookie"] = C
        if tries >= 4:
            stage = "goto_popcorn"
        else:
            act = {"tool": "act_grasp", "args": {"xy": [C[0], C[1]], "z": GRASP_Z, "close_steps": 20}}
            stage = "check_cookie"
    elif stage == "release_cookie":
        held = width > 0.02
        mem["place"] = _add(H, PLACE_COOKIE)
        notes = note("at place w=%.3f held=%s tcp=%s" % (width, held, [round(v, 3) for v in tcp[:3]]))
        act = {"tool": "act_release_retreat", "args": {
            "open_steps": 20, "back": [H[0], H[1] - 0.10, H[2] - 0.01], "up": [H[0], H[1] - 0.11, 0.71]}}
        stage = "check_cookie_placed"
    elif stage == "check_cookie_placed":
        objs = _perceive_objects(fr, tcp)
        ck, pp = _label_objects(objs)
        if pp and _near(pp["xy"], P, 0.05):
            P = pp["xy"]
            mem["popcorn"] = P
        if ck and ck["rb"] > 30 and tries < 3:
            # cookies still on the table: retry from its observed spot
            C = ck["xy"]
            mem["cookie"] = C
            tries += 1
            notes = note("cookie still on table")
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([C[0], C[1], 0.67], "down", -1, 90, 0.01, 0.05)]}}
            stage = "grasp_cookie"
        else:
            stage = "goto_popcorn"
            tries = 0
    if stage == "goto_popcorn" and act is None:
        act = {"tool": "act_path", "args": {"waypoints": [
            _wp([P[0], P[1], 0.70], "down", -1, 90, 0.01, 0.05)]}}
        stage = "grasp_popcorn"
    elif stage == "grasp_popcorn" and act is None:
        act = {"tool": "act_grasp", "args": {"xy": [P[0], P[1]], "z": GRASP_Z, "close_steps": 20}}
        stage = "check_popcorn"
    elif stage == "check_popcorn":
        if width > 0.015:
            pre = _add(H, CAVITY_PRE)
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([P[0], P[1], 0.56], "down", 1, 45, 0.01, 0.05),
                _wp([0.0, 0.0, 0.56], "down", 1, 50, 0.01, 0.05),
                _wp([0.0, 0.0, 0.56], "door", 1, 50, 0.01, 0.03),
                _wp([pre[0], pre[1] - 0.15, pre[2]], "door", 1, 60, 0.01, 0.04),
                _wp(pre, "door", 1, 60, 0.005, 0.04),
                _wp(_add(H, PLACE_POP), "door", 1, 25, 0.003, 0.04)]}}
            stage = "release_popcorn"
            notes = note("pop w=%.3f" % width)
        else:
            notes = note("pop miss w=%.3f" % width)
            tries += 1
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([tcp[0], tcp[1], 0.60], "down", -1, 40, 0.01, 0.08)]}}
            stage = "regrasp_popcorn"
    elif stage == "regrasp_popcorn":
        objs = _perceive_objects(fr, tcp)
        ck, pp = _label_objects(objs)
        if pp:
            P = pp["xy"]
            mem["popcorn"] = P
        if tries >= 4:
            stage = "done"
        else:
            act = {"tool": "act_grasp", "args": {"xy": [P[0], P[1]], "z": GRASP_Z, "close_steps": 20}}
            stage = "check_popcorn"
    elif stage == "release_popcorn":
        notes = note("pop at place w=%.3f tcp=%s" % (width, [round(v, 3) for v in tcp[:3]]))
        act = {"tool": "act_release_retreat", "args": {
            "open_steps": 20, "back": [H[0], H[1] - 0.10, H[2] - 0.01], "up": [H[0], H[1] - 0.11, 0.71]}}
        stage = "check_popcorn_placed"
    elif stage == "check_popcorn_placed":
        objs = _perceive_objects(fr, tcp)
        ck, pp = _label_objects(objs)
        if pp and pp["rb"] <= 30 and tries < 3:
            P = pp["xy"]
            mem["popcorn"] = P
            tries += 1
            notes = note("popcorn still on table")
            act = {"tool": "act_path", "args": {"waypoints": [
                _wp([P[0], P[1], 0.70], "down", -1, 90, 0.01, 0.05)]}}
            stage = "grasp_popcorn"
        else:
            stage = "done"
    if act is None:
        # done / fallback: hold still and observe (monitoring period)
        act = {"tool": "act_wait", "args": {"steps": 40}}
        stage = "done" if stage == "done" else stage
    mem["stage"] = stage
    mem["tries"] = tries
    mem["notes"] = notes
    return {"memory": mem, "act": act}


def _step_towards(robot, pos, rot, grip):
    fr = robot.now()
    cur = np.asarray(fr.tcp, dtype=float)
    err = np.asarray(pos, dtype=float) - cur[:3]
    a = np.clip(GAIN * err, -1.0, 1.0)
    r = np.zeros(3)
    rerr = 0.0
    tgt = _rot_target(rot) if isinstance(rot, str) else (None if rot is None else _rv2mat(rot))
    if tgt is not None:
        e = _mat2rv(tgt @ _rv2mat(cur[3:]).T)
        rerr = float(np.linalg.norm(e))
        r = np.clip(ROT_GAIN * e, -ROT_CLIP, ROT_CLIP)
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return float(np.linalg.norm(err)), rerr


def _pose_err(robot, pos, rot):
    cur = np.asarray(robot.now().tcp, dtype=float)
    pe = float(np.linalg.norm(np.asarray(pos, dtype=float) - cur[:3]))
    tgt = _rot_target(rot) if isinstance(rot, str) else (None if rot is None else _rv2mat(rot))
    re = 0.0 if tgt is None else float(np.linalg.norm(_mat2rv(tgt @ _rv2mat(cur[3:]).T)))
    return pe, re


def act_path(robot, waypoints):
    """Follow waypoints [{pos:[x,y,z], rot:'door'|'down'|None, grip:+1 close/-1 open, steps, tol, rtol}]
    with a P-controller on measured TCP. Returns per-waypoint steps/convergence, final tcp and finger width."""
    report = []
    for wp in waypoints:
        n = 0
        conv = False
        for _ in range(int(wp["steps"])):
            pe, re = _pose_err(robot, wp["pos"], wp.get("rot"))
            if pe < wp.get("tol", 0.004) and re < wp.get("rtol", 0.05):
                conv = True
                break
            _step_towards(robot, wp["pos"], wp.get("rot"), wp.get("grip", 0))
            n += 1
        pe, re = _pose_err(robot, wp["pos"], wp.get("rot"))
        report.append({"steps": n, "conv": conv, "pos_err": pe, "rot_err": re})
    fr = robot.now()
    return {"waypoints": report, "tcp": [float(v) for v in fr.tcp[:3]], "width": _width(fr)}


def act_gripper(robot, close, steps=20):
    """Hold the current pose and close (close=True) or open the gripper for `steps` steps.
    Returns finger width before/after."""
    w0 = _width(robot.now())
    g = 1.0 if close else -1.0
    hold = [float(v) for v in robot.now().tcp]
    for _ in range(int(steps)):
        _step_towards(robot, hold[:3], hold[3:], g)
    fr = robot.now()
    return {"width_before": w0, "width_after": _width(fr), "tcp": [float(v) for v in fr.tcp[:3]]}


def act_pull(robot, target, steps=40, release_steps=12):
    """With the gripper closed on the handle, drive toward `target` for `steps` steps (door swings
    open), then open the gripper. Returns widths during the pull and final tcp."""
    widths = []
    for i in range(int(steps)):
        _step_towards(robot, target, "door", 1.0)
        if i % 5 == 4:
            widths.append(_width(robot.now()))
    end = [float(v) for v in robot.now().tcp]
    for _ in range(int(release_steps)):
        _step_towards(robot, end[:3], end[3:], -1.0)
    fr = robot.now()
    return {"pull_widths": widths, "tcp_end_pull": end[:3], "width": _width(fr)}


def act_grasp(robot, xy, z=0.438, close_steps=20):
    """Top-down grasp: descend at xy to height z (open), close, hold. Returns final width/tcp."""
    pos = [float(xy[0]), float(xy[1]), float(z)]
    res = act_path(robot, [_wp([xy[0], xy[1], z + 0.08], "down", -1, 60, 0.008, 0.05),
                           _wp(pos, "down", -1, 60, 0.003, 0.05)])
    for _ in range(int(close_steps)):
        _step_towards(robot, pos, "down", 1.0)
    fr = robot.now()
    return {"descend": res["waypoints"], "width": _width(fr), "tcp": [float(v) for v in fr.tcp[:3]]}


def act_release_retreat(robot, open_steps, back, up):
    """Open the gripper in place, then back out to `back` and rise to `up` (door orientation)."""
    hold = [float(v) for v in robot.now().tcp]
    for _ in range(int(open_steps)):
        _step_towards(robot, hold[:3], hold[3:], -1.0)
    res = act_path(robot, [_wp(back, "door", -1, 40, 0.01, 0.06), _wp(up, "door", -1, 40, 0.01, 0.06)])
    return {"released_at": hold[:3], "retreat": res["waypoints"], "width": res["width"], "tcp": res["tcp"]}


def act_wait(robot, steps=40):
    """Hold pose (gripper open) for `steps` steps to let the checker settle."""
    hold = [float(v) for v in robot.now().tcp]
    for _ in range(int(steps)):
        _step_towards(robot, hold[:3], hold[3:], -1.0)
    return {"tcp": [float(v) for v in robot.now().tcp[:3]], "width": _width(robot.now())}
