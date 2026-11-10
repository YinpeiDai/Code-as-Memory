"""T25_ButterCreamPlate: pick butter then cream from plate1 and drop both on plate2.

Perception uses the agentview RGB-D image back-projected into world coordinates
with the supplied calibration.  Control is closed-loop on the measured TCP pose.
"""

TABLE_Z_DEFAULT = 0.430      # measured table top in all 25 demos (agentview depth)
SAFE_DZ = 0.27               # travel height above table (expert travelled at ~0.70 m)
GRASP_DZ = 0.003             # grasp target above table; descent stops early on contact
RELEASE_DZ = 0.094           # expert released both objects at tcp z ~0.524 (table + 0.094)
GRASP_YAW = -0.785398        # expert always grasped at yaw -45 deg (axis-angle ~(2.90,-1.20,0))
PLACE_YAW = 0.0              # expert placed at yaw 0, axis-angle (pi,0,0)
# agentview calibration recorded in every demo (fallback if a frame lacks it)
AGENT_K_DEFAULT = [[231.765, 0.0, 127.5], [0.0, 309.019, 127.5], [0.0, 0.0, 1.0]]
BUTTER_DX = -0.012           # red-pixel centroid is biased +x (front face); demo-calibrated
PLATE2_DEFAULT = [0.175, 0.12]
HELD_WIDTH = 0.008           # finger opening (m) above which an object is held
OPEN_WIDTH = 0.065           # fully open is ~0.078-0.080; held boxes measured 0.038-0.044
RETRY_DX = [0.0, -0.015, 0.015, -0.03, 0.03]  # x offsets for successive grasp attempts


# ---------------------------------------------------------------- geometry
def _aa_to_R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _is_held(width):
    return bool(HELD_WIDTH < width < OPEN_WIDTH)


def _R_down(yaw):
    """Gripper pointing straight down (expert place pose axis-angle (pi,0,0)), yawed about world z."""
    return _Rz(yaw) @ np.diag([1.0, -1.0, -1.0])


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


# ---------------------------------------------------------------- perception
def _world_points(frame):
    d = np.asarray(frame.agentview_depth)
    K = getattr(frame, "agentview_K", None)
    K = np.asarray(AGENT_K_DEFAULT if K is None else K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    valid = d > 0
    z = d.astype(float) / 1000.0
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, valid


def _robust_centroid(pts, radius):
    if len(pts) == 0:
        return None
    med = np.median(pts, axis=0)
    keep = np.linalg.norm(pts - med, axis=1) < radius
    if keep.sum() == 0:
        return med
    return pts[keep].mean(axis=0)


def _scene(frame, plate2=None):
    """Segment the agentview frame.  Returns plain python values."""
    pw, valid = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    X, Y, Z = pw[..., 0], pw[..., 1], pw[..., 2]
    ws = valid & (X > -0.15) & (X < 0.45) & (np.abs(Y) < 0.45)
    tabl = ws & (np.abs(Z - TABLE_Z_DEFAULT) < 0.02)
    tz = float(np.median(Z[tabl])) if tabl.sum() > 100 else TABLE_Z_DEFAULT
    h = Z - tz
    el = ws & (h > 0.006) & (h < 0.08)
    red = el & (R - B > 60) & (R - G > 40)
    blue = el & (B - R >= 8)
    white = ws & (h < 0.012) & (h > -0.01) & (R > 170) & (G > 170) & (B > 170)
    out = {"table_z": tz}
    red_xy = pw[red][:, :2]
    blue_xy = pw[blue][:, :2]
    white_xy = pw[white][:, :2]
    # plate2: white plate region far from the objects
    if plate2 is None:
        objs = np.concatenate([red_xy, blue_xy], 0)
        p2 = None
        if len(objs) >= 10 and len(white_xy) >= 20:
            oc = np.median(objs, axis=0)
            far = white_xy[np.linalg.norm(white_xy - oc, axis=1) > 0.15]
            if len(far) >= 20:
                c = _robust_centroid(far, 0.08)
                near = white_xy[np.linalg.norm(white_xy - c, axis=1) < 0.07]
                if len(near) >= 20:
                    c = near.mean(axis=0)
                p2 = [float(c[0]), float(c[1])]
        out["plate2"] = p2
        p2c = np.array(p2) if p2 is not None else None
    else:
        p2c = np.array(plate2[:2], dtype=float)
        out["plate2"] = [float(p2c[0]), float(p2c[1])]
    # butter
    if p2c is not None and len(red_xy):
        dr = np.linalg.norm(red_xy - p2c, axis=1)
        red_on, red_off = red_xy[dr < 0.075], red_xy[dr >= 0.075]
    else:
        red_on, red_off = red_xy[:0], red_xy
    out["butter_on_n"] = int(len(red_on))
    out["butter_off_n"] = int(len(red_off))
    bc = None
    if len(red_off) >= 5:
        bc = _robust_centroid(red_off, 0.05)
        out["butter"] = [float(bc[0] + BUTTER_DX), float(bc[1])]
    else:
        out["butter"] = None
    # cream (exclude bluish label pixels on the butter box and anything on plate2)
    if p2c is not None and len(blue_xy):
        db = np.linalg.norm(blue_xy - p2c, axis=1)
        blue_on, blue_off = blue_xy[db < 0.085], blue_xy[db >= 0.085]
    else:
        blue_on, blue_off = blue_xy[:0], blue_xy
    if bc is not None and len(blue_off):
        blue_off = blue_off[blue_off[:, 1] > bc[1] + 0.025]
    out["cream_on_n"] = int(len(blue_on))
    out["cream_off_n"] = int(len(blue_off))
    if len(blue_off) >= 10:
        cc = _robust_centroid(blue_off, 0.06)
        out["cream"] = [float(cc[0]), float(cc[1])]
    else:
        out["cream"] = None
    return out


# ---------------------------------------------------------------- decision
def _parse_last(last):
    if not isinstance(last, str):
        return ""
    return last.split(" -> ")[0].strip()


def _gone(now_xy, before_xy):
    """True when the object is no longer seen near where it was (it was lifted)."""
    if now_xy is None:
        return True
    if before_xy is None:
        return False
    d = math.hypot(now_xy[0] - before_xy[0], now_xy[1] - before_xy[1])
    return d > 0.06


def perceive_decide(obs, goal, memory, last):
    """Detect butter, cream and plate2 from agentview RGB-D; advance the
    pick butter -> place butter -> pick cream -> place cream program."""
    frame = obs.current
    mem = dict(memory) if isinstance(memory, dict) else {}
    stage = mem.get("stage")
    plate2 = mem.get("plate2")
    attempts = mem.get("attempts")
    decisions = mem.get("decisions")
    prev_butter = mem.get("butter_xy")
    prev_cream = mem.get("cream_xy")
    target_xy = mem.get("target_xy")
    attempts = 0 if attempts is None else int(attempts)
    decisions = 1 if decisions is None else int(decisions) + 1
    prev = _parse_last(last)

    if plate2 is None:
        sc = _scene(frame, None)
        if sc.get("plate2") is not None:
            plate2 = [sc["plate2"][0], sc["plate2"][1], sc["table_z"]]
    else:
        sc = _scene(frame, plate2)
    table_z = float(sc["table_z"])
    butter_now = sc.get("butter")
    cream_now = sc.get("cream")
    butter_xy = [butter_now[0], butter_now[1], table_z] if butter_now is not None else prev_butter
    cream_xy = [cream_now[0], cream_now[1], table_z] if cream_now is not None else prev_cream
    width = _width(frame)
    if stage is None:
        stage = "pick_butter"
        attempts = 0
    p2 = plate2 if plate2 is not None else [PLATE2_DEFAULT[0], PLATE2_DEFAULT[1], table_z]
    note = ""
    act = None

    # ---- transitions driven by the previous act and the current observation
    if stage in ("pick_butter", "pick_cream") and prev == "act_pick":
        now = butter_now if stage == "pick_butter" else cream_now
        lifted = _gone(now, target_xy)
        # widths of 0.002-0.06 were all seen while holding in demos; a fully
        # open hand (~0.078) never holds.  Vision decides small widths.
        held = (0.015 < width < OPEN_WIDTH) or (lifted and width < OPEN_WIDTH)
        if held:
            stage = "place_butter" if stage == "pick_butter" else "place_cream"
            note = "grasp ok (width %.4f, lifted=%s)" % (width, lifted)
        else:
            attempts += 1
            note = "grasp missed (width %.4f, lifted=%s)" % (width, lifted)
    elif stage in ("place_butter", "place_cream") and prev == "act_place":
        # verified by absence: the object is no longer detected off plate2
        # (plate2 itself is often occluded by the lifted arm, see task_analysis.md)
        if stage == "place_butter":
            if butter_now is None:
                stage = "pick_cream"
                attempts = 0
                note = "butter placed"
            else:
                stage = "pick_butter"
                attempts += 1
                note = "butter still seen off plate2; repick"
        else:
            if cream_now is None:
                stage = "monitor"
                note = "cream placed"
            else:
                stage = "pick_cream"
                attempts += 1
                note = "cream still seen off plate2; repick"
    elif stage in ("place_butter", "place_cream") and prev != "act_place":
        # a place was decided but did not complete (tool failure)
        if not (0.0005 < width < OPEN_WIDTH):
            stage = "pick_butter" if stage == "place_butter" else "pick_cream"
    elif stage == "monitor" and prev == "act_wait" and attempts < 3:
        # episode still running after both places: repair an object seen off plate2
        if butter_now is not None and not (0.0005 < width < OPEN_WIDTH):
            stage = "pick_butter"
            attempts += 1
            note = "monitor: butter off plate2"
        elif cream_now is not None and not (0.0005 < width < OPEN_WIDTH):
            stage = "pick_cream"
            attempts += 1
            note = "monitor: cream off plate2"

    # ---- choose the act for the current stage
    if stage in ("place_butter", "place_cream"):
        act = {"tool": "act_place",
               "args": {"x": float(p2[0]), "y": float(p2[1]), "table_z": table_z}}
    if act is None and stage == "pick_butter" and attempts >= 5:
        stage = "pick_cream"
        attempts = 0
        note = "butter abandoned after 5 misses"
    if act is None and stage == "pick_cream" and attempts >= 6:
        stage = "monitor"
        note = "cream abandoned after 6 misses"
    if act is None and stage in ("pick_butter", "pick_cream"):
        target = butter_now if stage == "pick_butter" else cream_now
        remembered = butter_xy if stage == "pick_butter" else cream_xy
        if target is None:
            if remembered is not None and prev == "act_home":
                target = remembered[:2]
                note = note + " using remembered target"
            else:
                act = {"tool": "act_home", "args": {"table_z": table_z}}
                note = note + " target not visible; clear view"
                if remembered is None:
                    attempts += 1
        if act is None and target is not None:
            # retries probe along the boxes' long (world x) axis
            off = RETRY_DX[attempts % len(RETRY_DX)]
            target_xy = [float(target[0]), float(target[1])]
            act = {"tool": "act_pick",
                   "args": {"x": float(target[0]) + off, "y": float(target[1]),
                            "table_z": table_z}}
    if act is None:
        stage = "monitor"
        act = {"tool": "act_wait", "args": {"steps": 100}}

    new_mem = {
        "stage": stage,
        "plate2": plate2,
        "attempts": int(attempts),
        "decisions": int(decisions),
        "butter_xy": butter_xy,
        "cream_xy": cream_xy,
        "target_xy": target_xy,
        "note": (note + " | scene b_on=%d b_off=%d c_on=%d c_off=%d w=%.4f" % (
            sc["butter_on_n"], sc["butter_off_n"], sc["cream_on_n"], sc["cream_off_n"], width)).strip(),
    }
    return {"memory": new_mem, "act": act}


# ---------------------------------------------------------------- control
def _servo(robot, target, R_des, grip, max_steps, tol=0.004, rot_tol=0.04,
           gain=12.0, vmax=1.0, stall_z=False, min_steps=0):
    """Closed-loop move of the TCP to `target` (world xyz) and orientation R_des."""
    target = np.asarray(target, dtype=float)
    hist = []
    n = 0
    for i in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        p = tcp[:3]
        Rc = _aa_to_R(tcp[3:6])
        e = target - p
        w = _R_to_rotvec(R_des @ Rc.T)
        hist.append(float(p[2]))
        if i >= min_steps and np.linalg.norm(e) < tol and np.linalg.norm(w) < rot_tol:
            break
        if stall_z and len(hist) > 12 and abs(hist[-1] - hist[-11]) < 0.0015 and e[2] < 0:
            break
        v = e * gain
        m = float(np.max(np.abs(v)))
        if m > vmax:
            v = v * (vmax / m)
        r = np.clip(w * 1.5, -1.0, 1.0)
        robot.step(float(v[0]), float(v[1]), float(v[2]),
                   float(r[0]), float(r[1]), float(r[2]), float(grip))
        n += 1
    f = robot.now()
    return n, [float(x) for x in np.asarray(f.tcp, dtype=float)[:3]]


def _hold(robot, R_des, grip, steps):
    f = robot.now()
    p = np.asarray(f.tcp, dtype=float)[:3].copy()
    _servo(robot, p, R_des, grip, steps, tol=-1.0, gain=8.0, vmax=0.3)


def _go_high(robot, x, y, safe_z, R_des, grip):
    """Rise to the travel height, then translate at that height."""
    p = np.asarray(robot.now().tcp, dtype=float)[:3]
    if p[2] < safe_z - 0.03:
        _servo(robot, [p[0], p[1], safe_z], R_des, grip, 80, tol=0.01, rot_tol=0.3)
    _servo(robot, [x, y, safe_z], R_des, grip, 150, tol=0.006, rot_tol=0.05)


def act_pick(robot, x, y, table_z):
    """Top-down grasp at world (x, y): travel high with yaw -45 deg, open,
    descend until near the table or contact, close, lift. Returns finger width."""
    Rg = _R_down(GRASP_YAW)
    safe_z = table_z + SAFE_DZ
    _go_high(robot, x, y, safe_z, Rg, -1.0)
    _servo(robot, [x, y, table_z + 0.08], Rg, -1.0, 120, tol=0.004, rot_tol=0.03)
    n, p = _servo(robot, [x, y, table_z + GRASP_DZ], Rg, -1.0, 150, tol=0.003,
                  gain=8.0, vmax=0.5, stall_z=True)
    _hold(robot, Rg, 1.0, 30)
    _servo(robot, [x, y, safe_z], Rg, 1.0, 120, tol=0.01, rot_tol=0.1, vmax=0.8)
    f = robot.now()
    return {"grasp_tcp": p, "width": _width(f),
            "held": _is_held(_width(f))}


def act_place(robot, x, y, table_z):
    """Carry the held object high to plate2 (x, y), rotate to home yaw, lower
    to the expert release height, open, wait, and lift."""
    Rp = _R_down(PLACE_YAW)
    safe_z = table_z + SAFE_DZ
    _go_high(robot, x, y, safe_z, Rp, 1.0)
    _hold(robot, Rp, 1.0, 8)
    n, p = _servo(robot, [x, y, table_z + RELEASE_DZ], Rp, 1.0, 150, tol=0.003,
                  gain=8.0, vmax=0.6, stall_z=True)
    _hold(robot, Rp, 1.0, 10)
    _hold(robot, Rp, -1.0, 20)
    _servo(robot, [x, y, safe_z], Rp, -1.0, 100, tol=0.01, rot_tol=0.1)
    f = robot.now()
    return {"release_tcp": p, "width": _width(f)}


def act_home(robot, table_z):
    """Lift and move to a view-clearing pose (x=0, y=0) at travel height."""
    R0 = _R_down(PLACE_YAW)
    _go_high(robot, 0.0, 0.0, table_z + SAFE_DZ, R0, -1.0)
    return {"tcp": [float(v) for v in np.asarray(robot.now().tcp, dtype=float)[:3]]}


def act_wait(robot, steps):
    """Hold the current pose with the gripper open for `steps` steps (lets the
    success monitor settle)."""
    R0 = _R_down(PLACE_YAW)
    _hold(robot, R0, -1.0, int(steps))
    return {"waited": int(steps)}
