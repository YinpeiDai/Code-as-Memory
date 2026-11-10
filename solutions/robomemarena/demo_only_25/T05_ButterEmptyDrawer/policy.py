"""T05_ButterEmptyDrawer: check the three drawers in the instructed order (open, look, close),
remember which drawers hold an object, then put the butter into an empty drawer.

All geometry is measured online (agentview RGB-D + calibration, measured TCP and fingers).
Offsets were fitted on the assigned demonstrations (see policy.md / task_analysis.md)."""

LEVELS = ["top", "middle", "bottom"]
# handle grasp height relative to the perceived cabinet top surface (demos: 0.670/0.605/0.530 vs top 0.714)
HANDLE_DZ = {"top": -0.044, "middle": -0.109, "bottom": -0.184}
# placement relative to the closed-drawer handle grasp point / cabinet top:
# [dx, dy of the descent column, dy of the release point, release tcp z relative to cabinet top]
# middle/bottom descent columns follow where the expert's episodes were scored successful
# (middle: y = handle - 0.055..0.084 at tcp z ~0.76; bottom: y = handle - 0.151..0.157 at tcp z ~0.73).
PLACE = {"top": [-0.004, -0.070, -0.070, 0.056],
         "middle": [-0.004, -0.070, -0.075, -0.014],
         "bottom": [-0.005, -0.155, -0.110, -0.014]}
CARRY_Z = 0.87                 # transport height while carrying the butter
OPEN_TRAVEL = 0.183            # drawer travel measured in the demos
PRE_DY = 0.10                  # pre-grasp stand-off in -y in front of the handle
ROT_HANDLE = [4.712389, 0.0, 0.0]   # horizontal gripper facing +y (expert grasp orientation)
ROT_DOWN = [3.141593, 0.0, 0.0]     # gripper pointing down
DEFAULT_CAB = [0.135, 0.465, 0.714]
DEFAULT_BUTTER = [0.110, -0.202, 0.437]
BUTTER_OFFSET = [-0.017, 0.004]     # grasp point minus centroid of coloured butter pixels (demos)


# ---------------------------------------------------------------- math helpers
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _mat2aa(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return np.zeros(3)
    ang = 2.0 * math.atan2(n, w)
    return np.array([x, y, z]) / n * ang


def _rot_error(target_aa, cur_aa):
    """World-frame axis-angle that rotates the current orientation onto the target."""
    return _mat2aa(_aa2mat(target_aa) @ _aa2mat(cur_aa).T)


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _tcp(frame):
    return np.asarray(frame.tcp, dtype=float)


# ---------------------------------------------------------------- perception helpers
def _world_points(frame):
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    W = (P - E[:, 3]) @ E[:, :3]
    return W, d > 0


def _perceive_cabinet(frame):
    """Handle grasp point of the top drawer from the dark cabinet top surface."""
    W, valid = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    m = (valid & (rgb.max(2) < 80) & (W[..., 2] > 0.695) & (W[..., 2] < 0.73)
         & (W[..., 1] > 0.3) & (W[..., 1] < 0.8) & (W[..., 0] > -0.1) & (W[..., 0] < 0.5))
    n = int(m.sum())
    if n < 300:
        return list(DEFAULT_CAB), n, False
    p = W[m]
    hx = 0.5 * ((float(p[:, 0].mean()) - 0.0208) + (float(np.percentile(p[:, 0], 98)) - 0.1325))
    hy = float(np.percentile(p[:, 1], 2)) - 0.0085
    tz = float(np.median(p[:, 2]))
    ok = (-0.05 < hx < 0.35) and (0.3 < hy < 0.65) and (0.69 < tz < 0.74)
    if not ok:
        return list(DEFAULT_CAB), n, False
    return [hx, hy, tz], n, True


def _perceive_butter(frame):
    W, valid = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    m = (valid & (W[..., 0] > 0.0) & (W[..., 0] < 0.25) & (W[..., 1] > -0.35) & (W[..., 1] < -0.05)
         & (W[..., 2] > 0.425) & (W[..., 2] < 0.5) & ((rgb.max(2) - rgb.min(2)) > 40))
    n = int(m.sum())
    if n < 15:
        return list(DEFAULT_BUTTER), n, False
    c = W[m].mean(0)
    bx, by = float(c[0]) + BUTTER_OFFSET[0], float(c[1]) + BUTTER_OFFSET[1]
    if abs(bx - DEFAULT_BUTTER[0]) > 0.08 or abs(by - DEFAULT_BUTTER[1]) > 0.08:
        return list(DEFAULT_BUTTER), n, False
    return [bx, by, DEFAULT_BUTTER[2]], n, True


def _content_score(frame):
    """Bright / coloured pixels inside the interior of the drawer currently held open
    (box relative to the measured TCP that holds the handle)."""
    W, valid = _world_points(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    rel = W - _tcp(frame)[:3]
    m = (valid & (np.abs(rel[..., 0]) < 0.075) & (rel[..., 1] > 0.02) & (rel[..., 1] < 0.17)
         & (rel[..., 2] > -0.045) & (rel[..., 2] < 0.04))
    c = rgb[m]
    if len(c) == 0:
        return 0, 0, 0
    bright = int((c.max(1) > 100).sum())
    sat = int(((c.max(1) - c.min(1)) > 45).sum())
    return int(m.sum()), bright, sat


def _parse_order(goal):
    text = str(goal or "").lower()
    key = text.find("order")
    seg = text[key:] if key >= 0 else text
    found = []
    for lv in LEVELS:
        i = seg.find(lv)
        if i >= 0:
            found.append((i, lv))
    found.sort()
    order = [lv for _, lv in found]
    for lv in LEVELS:
        if lv not in order:
            order.append(lv)
    return order


# ---------------------------------------------------------------- control helpers
def _servo(robot, target, rot, grip, max_steps, tol=0.004, rtol=0.05, vmax=1.0,
           stall=False, stall_min=10, kp=15.0, settle=False):
    """Closed-loop OSC delta servo to a world target. Stops when converged, blocked (stall) or out of steps."""
    target = np.asarray(target, dtype=float)
    hist = []
    n = 0
    blocked = False
    while n < max_steps:
        f = robot.now()
        t = _tcp(f)
        p = t[:3]
        e = target - p
        er = _rot_error(rot, t[3:6])
        if (not settle) and float(np.linalg.norm(e)) < tol and float(np.linalg.norm(er)) < rtol:
            break
        if stall and n >= stall_min and len(hist) >= 8 and float(np.linalg.norm(p - hist[-8])) < 0.0015:
            blocked = True
            break
        if (not settle) and (not stall) and n >= 30 and len(hist) >= 10 \
                and float(np.linalg.norm(p - hist[-10])) < 0.0008 and float(np.linalg.norm(e)) < 0.012 \
                and float(np.linalg.norm(er)) < 0.12:
            break  # converged as far as the controller gets (steady-state error)
        a = kp * e
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        ar = 2.0 * er
        mr = float(np.max(np.abs(ar)))
        if mr > 0.6:
            ar = ar * (0.6 / mr)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
        hist.append(p)
        n += 1
    return n, blocked


def _hold(robot, rot, grip, steps):
    p = _tcp(robot.now())[:3].copy()
    _servo(robot, p, rot, grip, steps, settle=True)


def _set_gripper(robot, rot, grip, steps=12):
    _hold(robot, rot, grip, steps)
    return _gap(robot.now())


def _handle_point(cab, level):
    return [float(cab[0]), float(cab[1]), float(cab[2]) + HANDLE_DZ[level]]


# ---------------------------------------------------------------- act tools
def act_open_drawer(robot, level, cabinet, release=False):
    """Open one drawer ('top'|'middle'|'bottom'): retreat in front of the cabinet, line up with the
    handle (horizontal gripper), push in until blocked, close the gripper, pull until the drawer stops.
    If release is true, let go and back away (drawer stays open). Returns measured travel and grip."""
    hx, hy, hz = _handle_point(cabinet, level)
    t0 = _tcp(robot.now())
    used = 0
    # 1. leave the cabinet front if we are close to it, at the current height
    if t0[1] > hy - PRE_DY + 0.02:
        n, _ = _servo(robot, [t0[0], hy - PRE_DY, t0[2]], t0[3:6], -1, 60, tol=0.01, rtol=1.0)
        used += n
    # 2. pre-grasp pose in front of the handle with the horizontal gripper, gripper open
    n, _ = _servo(robot, [hx, hy - PRE_DY, hz], ROT_HANDLE, -1, 200, tol=0.004, rtol=0.06)
    used += n
    if _gap(robot.now()) < 0.07:
        _set_gripper(robot, ROT_HANDLE, -1, 10)
        used += 10
    # 3. approach until contact
    n, blocked_in = _servo(robot, [hx, hy + 0.012, hz], ROT_HANDLE, -1, 90, tol=0.002, vmax=0.4, stall=True)
    used += n
    y_grasp = float(_tcp(robot.now())[1])
    # 4. grasp
    g = _set_gripper(robot, ROT_HANDLE, 1, 20)
    used += 20
    # 5. pull until the drawer stops
    n, blocked_out = _servo(robot, [hx, y_grasp - OPEN_TRAVEL - 0.035, hz], ROT_HANDLE, 1, 140,
                            tol=0.003, vmax=0.6, stall=True, stall_min=12)
    used += n
    _hold(robot, ROT_HANDLE, 1, 8)
    used += 8
    t1 = _tcp(robot.now())
    travel = y_grasp - float(t1[1])
    holding = _gap(robot.now()) > 0.006
    # opening measured against the perceived closed-handle position (robust to a partly open start)
    opened = ((hy - float(t1[1])) > 0.15) and holding
    res = {"level": level, "opened": bool(opened), "travel": round(travel, 4), "holding": bool(holding),
           "grip_gap_after_close": round(g, 4), "y_grasp": round(y_grasp, 4),
           "handle_now": [round(float(t1[0]), 4), round(float(t1[1]), 4), round(float(t1[2]), 4)],
           "released": False}
    if release:
        _set_gripper(robot, ROT_HANDLE, -1, 14)
        t2 = _tcp(robot.now())
        n, _ = _servo(robot, [t2[0], hy - 0.32, t2[2]], ROT_HANDLE, -1, 80, tol=0.01, rtol=1.0, vmax=0.8)
        t3 = _tcp(robot.now())
        n2, _ = _servo(robot, [t3[0], t3[1], 0.75], ROT_HANDLE, -1, 60, tol=0.01, rtol=1.0, vmax=0.8)
        used += 14 + n + n2
        res["released"] = True
    res["steps"] = used
    return res


def act_close_drawer(robot, level, cabinet, handle_y):
    """Push the open drawer ('top'|'middle'|'bottom') shut; handle_y is the measured handle y while open.
    Re-grasps the handle first if the gripper is not holding it. Releases and backs off afterwards."""
    hx, hy, hz = _handle_point(cabinet, level)
    used = 0
    if _gap(robot.now()) < 0.006 or _gap(robot.now()) > 0.05:
        # not holding: re-approach the handle at its open position
        t = _tcp(robot.now())
        _set_gripper(robot, ROT_HANDLE, -1, 10)
        n, _ = _servo(robot, [t[0], handle_y - 0.09, t[2]], ROT_HANDLE, -1, 60, tol=0.01, rtol=1.0)
        n2, _ = _servo(robot, [hx, handle_y - 0.09, hz], ROT_HANDLE, -1, 120, tol=0.004, rtol=0.04)
        n3, _ = _servo(robot, [hx, handle_y + 0.012, hz], ROT_HANDLE, -1, 90, tol=0.002, vmax=0.4, stall=True)
        _set_gripper(robot, ROT_HANDLE, 1, 20)
        used += 30 + n + n2 + n3
    n, blocked = _servo(robot, [hx, hy + 0.02, hz], ROT_HANDLE, 1, 140, tol=0.003, vmax=0.6,
                        stall=True, stall_min=12)
    used += n
    _hold(robot, ROT_HANDLE, 1, 6)
    y_push = float(_tcp(robot.now())[1])
    closed = y_push > hy - 0.035
    _set_gripper(robot, ROT_HANDLE, -1, 14)
    t = _tcp(robot.now())
    n2, _ = _servo(robot, [t[0], hy - PRE_DY, t[2]], ROT_HANDLE, -1, 80, tol=0.006, rtol=1.0, vmax=0.8)
    used += 20 + n2
    return {"level": level, "closed": bool(closed), "y_after_push": round(y_push, 4), "steps": used}


def act_pick_butter(robot, butter):
    """Rotate to a downward gripper at a safe height, descend onto the butter [x, y, z], grasp and lift."""
    bx, by, bz = float(butter[0]), float(butter[1]), float(butter[2])
    used = 0
    t = _tcp(robot.now())
    if t[2] < 0.70:
        n, _ = _servo(robot, [t[0], t[1], 0.72], _tcp(robot.now())[3:6], -1, 60, tol=0.01, rtol=1.0)
        used += n
    n, _ = _servo(robot, [bx, by, 0.64], ROT_DOWN, -1, 220, tol=0.005, rtol=0.03)
    used += n
    if _gap(robot.now()) < 0.07:
        _set_gripper(robot, ROT_DOWN, -1, 10)
        used += 10
    n, _ = _servo(robot, [bx, by, bz], ROT_DOWN, -1, 120, tol=0.003, rtol=0.03, vmax=0.6, stall=True, stall_min=15)
    used += n
    g = _set_gripper(robot, ROT_DOWN, 1, 16)
    used += 16
    n, _ = _servo(robot, [bx, by, CARRY_Z], ROT_DOWN, 1, 120, tol=0.01, rtol=0.05, vmax=1.0)
    used += n
    g2 = _gap(robot.now())
    holding = 0.01 < g2 < 0.06
    t = _tcp(robot.now())
    return {"holding": bool(holding), "gap_closed": round(g, 4), "gap_lifted": round(g2, 4),
            "tcp": [round(float(v), 4) for v in t[:3]], "steps": used}


def act_place_butter(robot, level, cabinet):
    """Carry the held butter high above the open drawer ('top'|'middle'|'bottom'), descend slowly into the
    drawer opening, shift to the release point over the drawer floor and release; then rise."""
    hx, hy, hz = _handle_point(cabinet, level)
    dx, dy1, dy2, dzr = PLACE[level]
    px = hx + dx
    zf = float(cabinet[2]) + dzr
    used = 0
    t = _tcp(robot.now())
    if t[2] < CARRY_Z - 0.03:
        n, _ = _servo(robot, [t[0], t[1], CARRY_Z], ROT_DOWN, 1, 80, tol=0.01, rtol=0.05)
        used += n
    n, _ = _servo(robot, [px, hy + dy1, CARRY_Z], ROT_DOWN, 1, 160, tol=0.006, rtol=0.03, vmax=1.0)
    used += n
    n, _ = _servo(robot, [px, hy + dy1, zf], ROT_DOWN, 1, 160, tol=0.004, rtol=0.03, vmax=0.3)
    used += n
    n, _ = _servo(robot, [px, hy + dy2, zf], ROT_DOWN, 1, 60, tol=0.004, rtol=0.03, vmax=0.3)
    used += n
    _hold(robot, ROT_DOWN, 1, 10)
    _set_gripper(robot, ROT_DOWN, -1, 20)
    _hold(robot, ROT_DOWN, -1, 20)
    t = _tcp(robot.now())
    n, _ = _servo(robot, [t[0], t[1], CARRY_Z], ROT_DOWN, -1, 80, tol=0.01, rtol=0.05, vmax=0.5)
    used += 50 + n
    return {"released": True, "steps": used, "tcp": [round(float(v), 4) for v in _tcp(robot.now())[:3]]}


def act_wait(robot, steps=40):
    """Hold the current pose with the gripper open for a number of steps (monitoring period)."""
    k = int(max(1, min(200, steps)))
    _hold(robot, _tcp(robot.now())[3:6], -1, k)
    return {"waited": k}


# ---------------------------------------------------------------- decision
def _flag(last, key):
    s = str(last or "")
    return ('"%s": true' % key) in s


def perceive_decide(obs, goal, memory, last):
    """Stage machine: check drawers in instructed order (open -> inspect -> close), remember contents,
    open the first empty drawer, pick the butter and place it there."""
    mem = dict(memory or {})
    cur = obs.current
    stage = mem.get("stage") or "init"
    cab = mem.get("cabinet")
    butter = mem.get("butter")
    order = mem.get("order")
    status = mem.get("drawer_status")
    checked = mem.get("checked")
    target = mem.get("target")
    retries = mem.get("retries")
    handle_y = mem.get("handle_y_open")
    notes = mem.get("notes")
    decisions = mem.get("decisions")
    retries = int(retries) if retries is not None else 0
    decisions = int(decisions) + 1 if decisions is not None else 1
    if notes is None:
        notes = ""
    if checked is None:
        checked = []
    if status is None:
        status = ["unknown", "unknown", "unknown"]
    if order is None:
        order = _parse_order(goal)

    if cab is None:
        cab, n, ok = _perceive_cabinet(cur)
        notes = "cab_pixels=%d ok=%s" % (n, ok)
    if butter is None:
        butter, nb, okb = _perceive_butter(cur)
        notes = notes + "; butter_pixels=%d ok=%s" % (nb, okb)

    act = None
    if stage == "init":
        stage = "opening"
        act = {"tool": "act_open_drawer", "args": {"level": order[0], "cabinet": cab}}
    elif stage == "opening":
        level = order[len(checked)]
        idx = LEVELS.index(level)
        t = _tcp(cur)
        opened = _flag(last, "opened")
        if opened:
            best = (0, 0, 0)
            for fr in list(obs.frames)[-3:]:
                s = _content_score(fr)
                if s[1] + 3 * s[2] > best[1] + 3 * best[2]:
                    best = s
            occupied = best[1] >= 12 or best[2] >= 5
            status[idx] = "occupied" if occupied else "empty"
            notes = notes + "; %s score=%s" % (level, list(best))
        handle_y = float(t[1])
        if opened or retries >= 2:
            retries = 0
            stage = "closing"
        else:
            retries += 1
            stage = "reopen"
        act = {"tool": "act_close_drawer", "args": {"level": level, "cabinet": cab, "handle_y": handle_y}}
    elif stage == "reopen":
        level = order[len(checked)]
        stage = "opening"
        act = {"tool": "act_open_drawer", "args": {"level": level, "cabinet": cab}}
    elif stage == "closing":
        level = order[len(checked)]
        hy = _handle_point(cab, level)[1]
        if (not _flag(last, "closed")) and retries < 2:
            retries += 1
            hyo = float(handle_y) if handle_y is not None else hy - OPEN_TRAVEL
            act = {"tool": "act_close_drawer", "args": {"level": level, "cabinet": cab, "handle_y": hyo}}
        else:
            retries = 0
            checked = checked + [level]
            if len(checked) < len(order):
                stage = "opening"
                act = {"tool": "act_open_drawer", "args": {"level": order[len(checked)], "cabinet": cab}}
            else:
                empties = [lv for lv in order if status[LEVELS.index(lv)] == "empty"]
                if not empties:
                    empties = [lv for lv in order if status[LEVELS.index(lv)] != "occupied"] or [order[-1]]
                target = empties[0]
                stage = "open_target"
                act = {"tool": "act_open_drawer", "args": {"level": target, "cabinet": cab, "release": True}}
    elif stage == "open_target":
        if _flag(last, "opened") or retries >= 2:
            retries = 0
            stage = "pick"
            act = {"tool": "act_pick_butter", "args": {"butter": butter}}
        else:
            # drawer did not open: make sure gripper is free and try again
            retries += 1
            act = {"tool": "act_open_drawer", "args": {"level": target, "cabinet": cab, "release": True}}
    elif stage == "pick":
        holding = _flag(last, "holding") or (0.01 < _gap(cur) < 0.06)
        if holding:
            retries = 0
            stage = "place"
            act = {"tool": "act_place_butter", "args": {"level": target, "cabinet": cab}}
        elif retries < 3:
            retries += 1
            act = {"tool": "act_pick_butter", "args": {"butter": butter}}
        else:
            stage = "place"
            act = {"tool": "act_place_butter", "args": {"level": target, "cabinet": cab}}
    else:
        stage = "done"
        act = {"tool": "act_wait", "args": {"steps": 60}}

    new_mem = {
        "stage": stage,
        "cabinet": [float(v) for v in cab],
        "butter": [float(v) for v in butter],
        "order": list(order),
        "drawer_status": list(status),
        "checked": list(checked),
        "target": target,
        "retries": int(retries),
        "handle_y_open": None if handle_y is None else float(handle_y),
        "decisions": int(decisions),
        "notes": str(notes)[-600:],
    }
    return {"memory": new_mem, "act": act}
