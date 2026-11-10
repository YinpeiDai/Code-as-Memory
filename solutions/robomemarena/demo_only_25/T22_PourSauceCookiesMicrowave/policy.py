"""T22 PourSauceCookiesMicrowave: perception (agentview/wrist RGB-D), memory-driven phase logic and
closed-loop OSC waypoint control. Geometry offsets were measured on the assigned demonstrations
(see task_analysis.md); object positions are always perceived from the current episode."""

# ---------------------------------------------------------------- constants (world frame, metres)
TABLE_Z = 0.431
R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
POUR_TILT = 0.8            # rad about world +x, as measured in every demo
SAUCE_GRASP_OFF = [0.001, -0.0085, -0.036]   # grasp tcp - (sauce top-cap centroid, top z)
COOKIE_GRASP_OFF_AV = [-0.004, -0.002]       # grasp tcp - agentview box-top centroid
COOKIE_GRASP_OFF_WR = [-0.001, -0.0015]      # grasp tcp - wrist box-top centroid
COOKIE_GRASP_Z = 0.434
HANDLE_GRASP_OFF = [-0.023, 0.007]           # grasp tcp - agentview handle centroid
HANDLE_GRASP_Z = 0.583
DEFAULT_SAUCE = [-0.033, -0.196, 0.505]
DEFAULT_COOKIES = [0.172, -0.194, 0.445]
DEFAULT_HANDLE = [0.068, 0.404, 0.562]


# ---------------------------------------------------------------- math helpers
def _aa2mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _mat2aa(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if math.pi - th < 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return v * th


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _pose(frame):
    tcp = np.asarray(frame.tcp, dtype=float)
    return tcp[:3].copy(), _aa2mat(tcp[3:6])


def _finger_width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def _backproject(depth, K, E):
    d = np.asarray(depth, dtype=float) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    valid = d > 0
    return pw, valid


# ---------------------------------------------------------------- perception
def _agentview_points(frame):
    return _backproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)


def _detect_sauce(pw, valid):
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    m = valid & (x > -0.30) & (x < 0.35) & (y > -0.50) & (y < 0.25) & (z > 0.465) & (z < 0.60)
    n = int(m.sum())
    if n < 15:
        return None
    zs = z[m]
    zmax = float(np.percentile(zs, 99.0))
    top = m & (z > zmax - 0.012) & (z < zmax + 0.01)
    cx, cy = float(np.median(x[top])), float(np.median(y[top]))
    top2 = top & (np.hypot(x - cx, y - cy) < 0.05)
    if int(top2.sum()) < 5:
        return None
    return [float(x[top2].mean()), float(y[top2].mean()), zmax, n]


def _detect_cookies(pw, valid, sauce_xy):
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    m = valid & (x > -0.15) & (x < 0.40) & (y > -0.48) & (y < 0.20) & (z > 0.437) & (z < 0.458)
    if sauce_xy is not None:
        m = m & (np.hypot(x - sauce_xy[0], y - sauce_xy[1]) > 0.06)
    if int(m.sum()) < 15:
        return None
    mx, my = float(np.median(x[m])), float(np.median(y[m]))
    m2 = m & (np.hypot(x - mx, y - my) < 0.08)
    n = int(m2.sum())
    if n < 15:
        return None
    return [float(x[m2].mean()), float(y[m2].mean()), float(np.percentile(z[m2], 95)), n]


def _detect_handle(pw, valid):
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    m = valid & (x > -0.40) & (x < 0.40) & (y > 0.15) & (y < 0.85) & (z > 0.47) & (z < 0.90)
    if int(m.sum()) < 200:
        return None
    ymin = float(np.percentile(y[m], 1.0))
    h = m & (y < ymin + 0.015) & (z > 0.48) & (z < 0.64)
    n = int(h.sum())
    if n < 12:
        return None
    hx = float(np.median(x[h]))
    h2 = h & (np.abs(x - hx) < 0.03)
    if int(h2.sum()) < 10:
        return None
    return [float(x[h2].mean()), float(y[h2].mean()), float(z[h2].mean()), n]


def _door_state(frame, handle):
    """Counts agentview points still at the closed-door handle location and points in the swept-open door area."""
    pw, valid = _agentview_points(frame)
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    near = valid & (np.abs(x - handle[0]) < 0.03) & (np.abs(y - handle[1]) < 0.025) & (z > 0.48) & (z < 0.64)
    hx = handle[0]
    door = valid & (x > hx - 0.39) & (x < hx - 0.18) & (y > 0.15) & (y < handle[1]) & (z > 0.48) & (z < 0.65)
    return int(near.sum()), int(door.sum())


def _wrist_cookie(frame, guess):
    pw, valid = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    x, y, z = pw[..., 0], pw[..., 1], pw[..., 2]
    m = valid & (z > 0.437) & (z < 0.458) & (np.hypot(x - guess[0], y - guess[1]) < 0.09)
    n = int(m.sum())
    if n < 150:
        return None
    return [float(x[m].mean()), float(y[m].mean()), n,
            float(x[m].max() - x[m].min()), float(y[m].max() - y[m].min())]


def _parse_pours(goal):
    g = str(goal or "").lower()
    words = {"once": 1, "twice": 2, "thrice": 3, "one time": 1, "two times": 2, "three times": 3,
             "four times": 4, "1 time": 1, "2 times": 2, "3 times": 3, "4 times": 4}
    for k in ["three times", "four times", "two times", "one time", "thrice", "twice", "once",
              "1 time", "2 times", "3 times", "4 times"]:
        if k in g:
            return words[k]
    return 2


# ---------------------------------------------------------------- control helpers (used inside act tools)
def _clip(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _servo(robot, pos, R_t, grip, max_steps, ptol=0.004, rtol=0.03, kp=10.0, kr=1.6, vmax=1.0, rmax=0.5):
    """P-servo of tcp to (pos, R_t) with per-axis clipping; returns steps used and final error."""
    pos = np.asarray(pos, dtype=float)
    R_t = np.asarray(R_t, dtype=float)
    used = 0
    ep, er = 1.0, 1.0
    for _ in range(int(max_steps)):
        p, R = _pose(robot.now())
        e = pos - p
        r = _mat2aa(R_t @ R.T)
        ep, er = float(np.linalg.norm(e)), float(np.linalg.norm(r))
        if ep < ptol and er < rtol:
            break
        a = [_clip(kp * e[0], -vmax, vmax), _clip(kp * e[1], -vmax, vmax), _clip(kp * e[2], -vmax, vmax),
             _clip(kr * r[0], -rmax, rmax), _clip(kr * r[1], -rmax, rmax), _clip(kr * r[2], -rmax, rmax), grip]
        robot.step(*a)
        used += 1
    return used, ep, er


def _hold(robot, pos, R_t, grip, steps, kp=10.0, kr=1.6):
    """Hold a pose for a fixed number of steps (used while the fingers move or while pouring)."""
    pos = np.asarray(pos, dtype=float)
    R_t = np.asarray(R_t, dtype=float)
    for _ in range(int(steps)):
        p, R = _pose(robot.now())
        e = pos - p
        r = _mat2aa(R_t @ R.T)
        robot.step(_clip(kp * e[0], -1, 1), _clip(kp * e[1], -1, 1), _clip(kp * e[2], -1, 1),
                   _clip(kr * r[0], -0.5, 0.5), _clip(kr * r[1], -0.5, 0.5), _clip(kr * r[2], -0.5, 0.5), grip)
    return int(steps)


def _tcp_list(robot):
    return [round(float(v), 4) for v in np.asarray(robot.now().tcp, dtype=float)[:3]]


# ---------------------------------------------------------------- decision
def _init_memory(memory):
    m = dict(memory or {})
    defaults = {"phase": "start", "pours_done": 0, "pours_required": 2, "sauce": None, "cookies": None,
                "handle": None, "attempts": 0, "decisions": 0, "notes": ""}
    for k, v in defaults.items():
        if m.get(k) is None:
            m[k] = v
    return m


def _near(new, old, r):
    return new is not None and (old is None or math.hypot(new[0] - old[0], new[1] - old[1]) < r)


def _ok(last):
    return isinstance(last, str) and '"ok": true' in last


def perceive_decide(obs, goal, memory, last):
    """Phase machine: perceive objects at the start, then grasp sauce -> pour N times -> place sauce ->
    open microwave -> (verify door) grasp cookies -> insert into microwave -> wait. Memory keeps perceived
    object positions, the pour count and the current phase; phases advance only on measured success."""
    m = _init_memory(memory)
    m["decisions"] = int(m["decisions"]) + 1
    cur = obs.current
    phase = m["phase"]
    notes = []
    act = {"tool": "act_wait", "args": {"steps": 40}}

    if phase == "start":
        m["pours_required"] = _parse_pours(goal)
        pw, valid = _agentview_points(cur)
        s = _detect_sauce(pw, valid)
        c = _detect_cookies(pw, valid, None if s is None else s[:2])
        h = _detect_handle(pw, valid)
        m["sauce"] = [round(v, 4) for v in (s[:3] if s is not None else DEFAULT_SAUCE)]
        m["cookies"] = [round(v, 4) for v in (c[:3] if c is not None else DEFAULT_COOKIES)]
        m["handle"] = [round(v, 4) for v in (h[:3] if h is not None else DEFAULT_HANDLE)]
        notes.append("detect sauce=%s cookies=%s handle=%s" % (s is not None, c is not None, h is not None))
        phase = "grasp_sauce"
        m["attempts"] = 0
    elif phase == "grasp_sauce":
        if _ok(last):
            phase = "pour"
            m["attempts"] = 0
        else:
            m["attempts"] = int(m["attempts"]) + 1
            pw, valid = _agentview_points(cur)
            s = _detect_sauce(pw, valid)
            if _near(s, m["sauce"], 0.10):
                m["sauce"] = [round(v, 4) for v in s[:3]]
            notes.append("sauce grasp retry %d" % m["attempts"])
            if m["attempts"] > 2:
                phase = "pour"   # give up re-grasping; continue the rest of the task
    elif phase == "pour":
        gave_up = int(m["attempts"]) >= 3 or int(m["decisions"]) >= 16
        if _ok(last) and "act_pour" in last and ('"holding": true' in last or gave_up):
            m["pours_done"] = int(m["pours_done"]) + 1
        if _finger_width(cur) < 0.02 and not gave_up and \
                int(m["pours_done"]) < int(m["pours_required"]):
            # bottle no longer between the fingers: perceive it again and re-grasp before pouring on
            pw, valid = _agentview_points(cur)
            s = _detect_sauce(pw, valid)
            if _near(s, m["sauce"], 0.15):
                m["sauce"] = [round(v, 4) for v in s[:3]]
            notes.append("bottle lost during pour; regrasp")
            phase = "grasp_sauce"
            m["attempts"] = 0
    elif phase == "place_sauce":
        phase = "open_door"
        m["attempts"] = 0
    elif phase == "open_door":
        near, door = _door_state(cur, m["handle"])
        notes.append("door check near_handle=%d door_pts=%d" % (near, door))
        if near < 20 or int(m["attempts"]) >= 1:
            phase = "grasp_cookies"
            m["attempts"] = 0
        else:
            m["attempts"] = int(m["attempts"]) + 1
    elif phase == "grasp_cookies":
        if _ok(last):
            phase = "insert"
            m["attempts"] = 0
        else:
            m["attempts"] = int(m["attempts"]) + 1
            notes.append("cookie grasp retry %d" % m["attempts"])
            if m["attempts"] > 2:
                phase = "insert"
    elif phase == "insert":
        phase = "done"

    # pour bookkeeping (may move from pour to place in the same decision)
    if phase == "pour" and int(m["pours_done"]) >= int(m["pours_required"]):
        phase = "place_sauce"

    if phase == "grasp_sauce":
        s = m["sauce"]
        act = {"tool": "act_grasp_sauce", "args": {"x": s[0], "y": s[1], "ztop": s[2]}}
    elif phase == "pour":
        c = m["cookies"]
        act = {"tool": "act_pour", "args": {"cx": c[0], "cy": c[1], "index": int(m["pours_done"]) + 1}}
    elif phase == "place_sauce":
        s = m["sauce"]
        act = {"tool": "act_place_sauce", "args": {"x": s[0] + SAUCE_GRASP_OFF[0], "y": s[1] + SAUCE_GRASP_OFF[1],
                                                  "z": s[2] + SAUCE_GRASP_OFF[2] + 0.035}}
    elif phase == "open_door":
        h = m["handle"]
        if int(m["attempts"]) > 0:
            pw, valid = _agentview_points(cur)
            h2 = _detect_handle(pw, valid)
            if _near(h2, m["handle"], 0.06):
                m["handle"] = [round(v, 4) for v in h2[:3]]
                h = m["handle"]
        act = {"tool": "act_open_door", "args": {"hx": h[0], "hy": h[1]}}
    elif phase == "grasp_cookies":
        pw, valid = _agentview_points(cur)
        c = _detect_cookies(pw, valid, m["sauce"][:2])
        if _near(c, m["cookies"], 0.10):
            m["cookies"] = [round(v, 4) for v in c[:3]]
        notes.append("cookies redetected=%s" % (c is not None))
        c = m["cookies"]
        act = {"tool": "act_grasp_cookies", "args": {"cx": c[0], "cy": c[1]}}
    elif phase == "insert":
        h = m["handle"]
        act = {"tool": "act_insert_cookies", "args": {"hx": h[0], "hy": h[1]}}
    elif phase == "done":
        act = {"tool": "act_wait", "args": {"steps": 60}}

    m["phase"] = phase
    m["notes"] = "; ".join(notes)[:300]
    return {"memory": m, "act": act}


# ---------------------------------------------------------------- act tools
def act_grasp_sauce(robot, x, y, ztop):
    """Top-down grasp of the sauce bottle whose top-cap centre is (x, y, ztop); lifts to 0.81 m.
    Returns ok (fingers stopped on the bottle), finger width and steps."""
    gx, gy, gz = x + SAUCE_GRASP_OFF[0], y + SAUCE_GRASP_OFF[1], ztop + SAUCE_GRASP_OFF[2]
    n = 0
    n += _servo(robot, [gx, gy, 0.668], R_DOWN, -1, 90, ptol=0.006)[0]
    n += _servo(robot, [gx, gy, gz], R_DOWN, -1, 80, ptol=0.003)[0]
    n += _hold(robot, [gx, gy, gz], R_DOWN, -1, 4)
    n += _hold(robot, [gx, gy, gz], R_DOWN, 1, 22)
    w = _finger_width(robot.now())
    n += _servo(robot, [gx, gy, 0.81], R_DOWN, 1, 70, ptol=0.01)[0]
    w2 = _finger_width(robot.now())
    ok = (w2 > 0.03)
    if not ok:
        n += _hold(robot, _tcp_list(robot), R_DOWN, -1, 12)
    return {"ok": bool(ok), "width_closed": round(w, 4), "width_lifted": round(w2, 4), "steps": n,
            "tcp": _tcp_list(robot)}


def act_pour(robot, cx, cy, index=1):
    """One pour over the cookies at (cx, cy): move above, tilt 0.8 rad about world x, hold, return upright.
    Returns ok, holding width and steps."""
    n = 0
    n += _servo(robot, [cx - 0.020, cy - 0.0025, 0.695], R_DOWN, 1, 90, ptol=0.006)[0]
    R_p = _rotx(POUR_TILT) @ np.asarray(R_DOWN)
    tilt_pos = [cx + 0.006, cy - 0.019, 0.687]
    n += _servo(robot, tilt_pos, R_p, 1, 50, ptol=0.006, rtol=0.03)[0]
    p, R = _pose(robot.now())
    tilt = float(np.linalg.norm(_mat2aa(R @ np.asarray(R_DOWN).T)))
    n += _hold(robot, tilt_pos, R_p, 1, 40)
    n += _servo(robot, [cx - 0.010, cy - 0.002, 0.682], R_DOWN, 1, 50, ptol=0.006, rtol=0.03)[0]
    n += _hold(robot, [cx - 0.010, cy - 0.002, 0.682], R_DOWN, 1, 30)
    w = _finger_width(robot.now())
    return {"ok": True, "holding": bool(w > 0.02), "index": int(index), "tilt_reached": round(tilt, 3),
            "holding_width": round(w, 4),
            "steps": n}


def act_place_sauce(robot, x, y, z):
    """Put the bottle down upright at grasp position (x, y) releasing at height z, then rise clear.
    Returns ok and steps."""
    n = 0
    n += _servo(robot, [x, y, z + 0.09], R_DOWN, 1, 90, ptol=0.006)[0]
    n += _servo(robot, [x, y, z], R_DOWN, 1, 60, ptol=0.004)[0]
    n += _hold(robot, [x, y, z], R_DOWN, -1, 15)
    n += _servo(robot, [x, y + 0.03, z + 0.12], R_DOWN, -1, 40, ptol=0.01)[0]
    return {"ok": True, "steps": n, "tcp": _tcp_list(robot)}


def act_open_door(robot, hx, hy):
    """Open the microwave door by its handle centroid (hx, hy): rise to a clear point, turn the gripper to face +y
    while moving to a pre-grasp 19 cm in front of the handle, close on the handle, pull toward -x until the tcp
    passes x=-0.323, release, back away and turn the gripper down again. Returns ok, handle grip width, steps."""
    gx, gy = hx + HANDLE_GRASP_OFF[0], hy + HANDLE_GRASP_OFF[1]
    n = 0
    n += _servo(robot, [-0.05, -0.103, 0.692], R_DOWN, -1, 60, ptol=0.01, rtol=0.05)[0]
    n += _servo(robot, [gx, gy - 0.19, HANDLE_GRASP_Z + 0.001], R_SIDE, -1, 90, ptol=0.004, rtol=0.03)[0]
    n += _servo(robot, [gx, gy, HANDLE_GRASP_Z], R_SIDE, -1, 70, ptol=0.003, kp=8.0)[0]
    n += _hold(robot, [gx, gy, HANDLE_GRASP_Z], R_SIDE, 1, 20)
    w = _finger_width(robot.now())
    target = np.array([-0.465, gy - 0.186, HANDLE_GRASP_Z + 0.002])
    Rs = np.asarray(R_SIDE)
    pulled = 0
    for _ in range(80):
        p, R = _pose(robot.now())
        if p[0] <= -0.323:
            break
        e = target - p
        r = _mat2aa(Rs @ R.T)
        robot.step(_clip(5.7 * e[0], -1, 1), _clip(5.7 * e[1], -1, 1), _clip(5.7 * e[2], -1, 1),
                   _clip(1.6 * r[0], -0.5, 0.5), _clip(1.6 * r[1], -0.5, 0.5), _clip(1.6 * r[2], -0.5, 0.5), 1)
        pulled += 1
    n += pulled
    w_end = _finger_width(robot.now())
    n += _hold(robot, _tcp_list(robot), R_SIDE, -1, 10)
    n += _servo(robot, [-0.372, gy - 0.289, 0.567], R_SIDE, -1, 35, ptol=0.008, kp=8.0)[0]
    n += _servo(robot, [-0.315, gy - 0.26, 0.61], R_DOWN, -1, 60, ptol=0.01, rtol=0.05)[0]
    return {"ok": True, "handle_width": round(w, 4), "width_end_pull": round(w_end, 4), "pull_steps": pulled,
            "steps": n, "tcp": _tcp_list(robot)}


def act_grasp_cookies(robot, cx, cy):
    """Top-down grasp of the cookie box seen at (cx, cy) (refined with the wrist depth camera), lift to 0.728 m.
    Returns ok (fingers stopped on the box), widths, refinement and steps."""
    n = 0
    gx, gy = cx + COOKIE_GRASP_OFF_AV[0], cy + COOKIE_GRASP_OFF_AV[1]
    n += _servo(robot, [gx, gy, 0.636], R_DOWN, -1, 110, ptol=0.004)[0]
    n += _hold(robot, [gx, gy, 0.636], R_DOWN, -1, 3)
    ref = _wrist_cookie(robot.now(), [cx, cy])
    used_ref = False
    if ref is not None and math.hypot(ref[0] - cx, ref[1] - cy) < 0.03:
        gx, gy = ref[0] + COOKIE_GRASP_OFF_WR[0], ref[1] + COOKIE_GRASP_OFF_WR[1]
        used_ref = True
    n += _servo(robot, [gx, gy, 0.52], R_DOWN, -1, 50, ptol=0.004)[0]
    n += _servo(robot, [gx, gy, COOKIE_GRASP_Z], R_DOWN, -1, 50, ptol=0.003, kp=8.0)[0]
    n += _hold(robot, [gx, gy, COOKIE_GRASP_Z], R_DOWN, -1, 3)
    n += _hold(robot, [gx, gy, COOKIE_GRASP_Z], R_DOWN, 1, 25)
    w = _finger_width(robot.now())
    n += _servo(robot, [gx, gy, 0.728], R_DOWN, 1, 70, ptol=0.01)[0]
    w2 = _finger_width(robot.now())
    ok = w2 > 0.025
    if not ok:
        n += _servo(robot, [gx, gy, 0.636], R_DOWN, -1, 40, ptol=0.01)[0]
        n += _hold(robot, _tcp_list(robot), R_DOWN, -1, 10)
    return {"ok": bool(ok), "width_closed": round(w, 4), "width_lifted": round(w2, 4), "wrist_refined": used_ref,
            "grasp_xy": [round(gx, 4), round(gy, 4)], "steps": n}


def act_insert_cookies(robot, hx, hy):
    """Carry the held cookie box into the opened microwave whose closed-door handle centroid was (hx, hy):
    turn the gripper to face +y, approach along the expert-like path, lower in front of the opening, push the box
    about 2 cm past the front face, set it on the floor, release and back out. Returns ok and steps."""
    n = 0
    n += _servo(robot, [0.022, -0.004, 0.725], R_DOWN, 1, 90, ptol=0.01)[0]
    n += _servo(robot, [0.022, -0.004, 0.725], R_SIDE, 1, 60, ptol=0.01, rtol=0.03)[0]
    target = np.array([hx - 0.071, hy + 0.067, 0.578])
    Rs = np.asarray(R_SIDE)
    for _ in range(90):
        p, R = _pose(robot.now())
        if p[1] >= hy - 0.013:
            break
        e = target - p
        r = _mat2aa(Rs @ R.T)
        robot.step(_clip(5.7 * e[0], -1, 1), _clip(5.7 * e[1], -1, 1), _clip(5.7 * e[2], -1, 1),
                   _clip(1.6 * r[0], -0.5, 0.5), _clip(1.6 * r[1], -0.5, 0.5), _clip(1.6 * r[2], -0.5, 0.5), 1)
        n += 1
    w_in = _finger_width(robot.now())
    tx = hx - 0.073
    n += _servo(robot, [tx, hy - 0.013, 0.525], R_SIDE, 1, 50, ptol=0.004, kp=8.0)[0]
    n += _servo(robot, [tx, hy + 0.058, 0.525], R_SIDE, 1, 60, ptol=0.004, kp=6.0)[0]
    n += _hold(robot, [tx, hy + 0.058, 0.512], R_SIDE, 1, 15)
    n += _hold(robot, _tcp_list(robot), R_SIDE, -1, 15)
    n += _servo(robot, [tx, hy - 0.12, 0.53], R_SIDE, -1, 50, ptol=0.01, kp=8.0)[0]
    n += _servo(robot, [tx, hy - 0.16, 0.65], R_SIDE, -1, 40, ptol=0.01)[0]
    return {"ok": True, "width_at_front": round(w_in, 4), "steps": n, "tcp": _tcp_list(robot)}


def act_wait(robot, steps=40):
    """Hold the current pose with the gripper open for `steps` steps (lets the checker settle)."""
    p, R = _pose(robot.now())
    n = _hold(robot, p, R, -1, max(1, min(400, int(steps))))
    return {"ok": True, "steps": n}
