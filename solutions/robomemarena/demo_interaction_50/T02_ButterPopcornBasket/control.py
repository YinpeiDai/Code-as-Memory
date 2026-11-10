"""Closed-loop Cartesian motion, gripper and pick/place act tools."""

KP_POS = 12.0      # normalized action per metre of error (1.0 ~ 1 cm/step)
KP_ROT = 3.0       # normalized action per radian of error
OPEN, CLOSE = -1.0, 1.0
HELD_MIN = 0.006   # finger joint (m) above which a closed hand holds something
SAFE_Z = 0.75


def _tcp(robot):
    return np.asarray(robot.now().tcp, dtype=float)


def _fing(robot):
    f = np.asarray(robot.now().fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1])) / 2.0


def _move(robot, target, grip, tol=0.003, max_steps=250, kp=KP_POS, vmax=1.0, watch_hold=False, yaw=None):
    """Servo the TCP to target xyz with a top-down orientation at yaw (None = keep the current yaw)."""
    target = np.asarray(target, dtype=float)
    if yaw is None:
        yaw = _yaw_of(_tcp(robot))
    good = 0
    steps = 0
    lost = False
    stalled = False
    hist = []
    for _ in range(max_steps):
        tcp = _tcp(robot)
        err = target - tcp[:3]
        dist = float(np.linalg.norm(err))
        hist.append(dist)
        # blocked by contact: no progress over 15 steps while still far from the target
        if len(hist) > 15 and dist > max(3 * tol, 0.01) and hist[-16] - dist < 0.002:
            stalled = True
            break
        rerr = _rot_error(tcp, yaw)
        if float(np.linalg.norm(err)) < tol and float(np.linalg.norm(rerr)) < 0.03:
            good += 1
            if good >= 3:
                break
        else:
            good = 0
        if watch_hold and _fing(robot) < HELD_MIN * 0.7:
            lost = True
            break
        a = np.clip(kp * err, -vmax, vmax)
        r = np.clip(KP_ROT * rerr, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
    tcp = _tcp(robot)
    return {"reached": bool(np.linalg.norm(target - tcp[:3]) < max(tol * 2, 0.006)),
            "err": float(np.linalg.norm(target - tcp[:3])), "steps": steps, "lost": lost, "stalled": stalled,
            "tcp": [float(v) for v in tcp[:3]]}


def _gripper(robot, grip, min_steps=8, max_steps=40):
    hold = _tcp(robot)[:3]
    yaw = _yaw_of(_tcp(robot))
    prev = _fing(robot)
    still = 0
    n = 0
    for n in range(max_steps):
        tcp = _tcp(robot)
        a = np.clip(KP_POS * (hold - tcp[:3]), -0.3, 0.3)
        r = np.clip(KP_ROT * _rot_error(tcp, yaw), -0.3, 0.3)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        cur = _fing(robot)
        still = still + 1 if abs(cur - prev) < 2e-4 else 0
        prev = cur
        if n + 1 >= min_steps and still >= 4:
            break
    return _fing(robot)


def _travel(robot, x, y, z_end, grip, safe_z, watch_hold=False, yaw=None):
    """Rise to safe height if needed, move over (x, y) (turning to yaw), then descend to z_end."""
    log = []
    tcp = _tcp(robot)
    far = math.hypot(x - tcp[0], y - tcp[1]) > 0.03
    if far and tcp[2] < safe_z - 0.02:
        r = _move(robot, [tcp[0], tcp[1], safe_z], grip, tol=0.01, max_steps=150, watch_hold=watch_hold)
        log.append(("rise", r))
        if r["lost"]:
            return log, False
    if far:
        z = max(safe_z, _tcp(robot)[2]) if tcp[2] < safe_z + 0.1 else _tcp(robot)[2]
        r = _move(robot, [x, y, z], grip, tol=0.006, max_steps=200, watch_hold=watch_hold, yaw=yaw)
        log.append(("over", r))
        if r["lost"]:
            return log, False
    r = _move(robot, [x, y, z_end], grip, tol=0.004, max_steps=200, watch_hold=watch_hold, yaw=yaw)
    log.append(("down", r))
    return log, not r["lost"]


def _joint_safe_yaw(robot, yaw):
    """Among yaw and yaw +- pi (equivalent for the symmetric hand) pick the smallest turn that keeps the
    predicted wrist joint (q7 ~ q7 - d_yaw for a downward hand) inside its range."""
    now = _yaw_of(_tcp(robot))
    q7 = float(np.asarray(robot.now().joints)[6])
    best, bs = yaw, 1e9
    for c in (yaw, yaw + math.pi, yaw - math.pi):
        d = c - now
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        q = q7 - d
        score = abs(d) + (10.0 if abs(q) > 2.6 else 0.0)
        if score < bs:
            best, bs = now + d, score
    return best


def act_pick(robot, item, x, y, table_z, basket_boxes=None, yaw=0.0, hover_dz=0.10, grasp_dz=0.003, safe_z=0.75):
    """Pick a small flat item lying near world (x, y): travel at safe_z, hover hover_dz above the table,
    re-detect the item in the wrist view to refine (x, y) and the grasp yaw (fingers across the short axis,
    or turned 90 deg when the wide hand would overlap a basket), descend to table_z+grasp_dz, close, lift.
    A blocked descent is retried once with the other yaw, otherwise abandoned without closing.
    Returns measured finger opening, whether the hand holds something, and the refined target."""
    boxes = basket_boxes or []
    yaw = _joint_safe_yaw(robot, float(yaw))
    log, ok = _travel(robot, x, y, table_z + hover_dz, OPEN, safe_z, yaw=yaw)
    refined = None
    det = None
    for _ in range(3):
        items = _detect_items(robot.now(), table_z, boxes)
        it = _match_item(items, item, near=[x, y], radius=0.06)
        if it is None:
            break
        det = it
        refined = it["xy"]
        new_yaw, ov = _grasp_yaw(it, refined[0], refined[1], boxes)
        new_yaw = _joint_safe_yaw(robot, new_yaw)
        moved = math.hypot(refined[0] - x, refined[1] - y)
        turn = abs(_wrap_half(new_yaw - yaw))
        x, y = refined
        if moved < 0.004 and turn < 0.15:
            break
        yaw = new_yaw if turn >= 0.15 else yaw
        _move(robot, [x, y, table_z + hover_dz], OPEN, tol=0.003, max_steps=80, yaw=yaw)
    tries = []
    pre = None
    for k in range(2):
        pre = _move(robot, [x, y, table_z + 0.03], OPEN, tol=0.004, max_steps=120, yaw=yaw)
        tries.append({"yaw": float(yaw), "stalled": pre["stalled"], "err": pre["err"]})
        if not (pre["stalled"] or pre["err"] > 0.015):
            break
        # blocked (e.g. a basket rim under the wide hand): back up and turn the hand 90 deg
        _move(robot, [x, y, table_z + hover_dz], OPEN, tol=0.006, max_steps=80, yaw=yaw)
        if k == 0:
            alt, ov = _grasp_yaw(det, x, y, boxes, avoid=yaw)
            if abs(_wrap_half(alt - yaw)) < 0.3:
                alt = _wrap_half(yaw + math.pi / 2)
            yaw = _joint_safe_yaw(robot, alt)
            _move(robot, [x, y, table_z + hover_dz], OPEN, tol=0.005, max_steps=80, yaw=yaw)
    if pre["stalled"] or pre["err"] > 0.015:
        up = _move(robot, [pre["tcp"][0], pre["tcp"][1], safe_z], OPEN, tol=0.01, max_steps=150)
        return {"item": item, "target": [float(x), float(y)], "refined": refined is not None, "blocked": True,
                "tries": tries, "held": False, "finger_lifted": _fing(robot), "lift_tcp": up["tcp"]}
    d = _move(robot, [x, y, table_z + grasp_dz], OPEN, tol=0.003, max_steps=120, kp=8.0, vmax=0.5, yaw=yaw)
    f_closed = _gripper(robot, CLOSE, min_steps=10, max_steps=40)
    # settle before lifting (expert also pauses with the hand closed)
    for _ in range(5):
        robot.step(0, 0, 0, 0, 0, 0, CLOSE)
    up = _move(robot, [x, y, safe_z], CLOSE, tol=0.01, max_steps=150, kp=10.0, vmax=0.8, yaw=yaw)
    f_up = _fing(robot)
    return {"item": item, "target": [float(x), float(y)], "refined": refined is not None, "blocked": False,
            "yaw": float(yaw), "tries": tries, "q7": float(np.asarray(robot.now().joints)[6]),
            "grasp_tcp": d["tcp"], "grasp_err": d["err"], "finger_closed": f_closed, "finger_lifted": f_up,
            "held": bool(f_up > HELD_MIN), "lift_tcp": up["tcp"]}


def act_place(robot, x, y, release_z, safe_z=0.75, settle=12):
    """Carry the held item at safe_z over basket centre (x, y), descend to release_z, open the gripper,
    wait for it to fall, then rise. Watches the fingers for a dropped item. Returns measured outcome."""
    log, ok = _travel(robot, x, y, release_z, CLOSE, safe_z, watch_hold=True)
    if not ok:
        return {"released": False, "lost_in_transit": True, "tcp": [float(v) for v in _tcp(robot)[:3]],
                "finger": _fing(robot)}
    f_before = _fing(robot)
    pos = _tcp(robot)[:3]
    f_open = _gripper(robot, OPEN, min_steps=10, max_steps=40)
    for _ in range(settle):
        robot.step(0, 0, 0, 0, 0, 0, OPEN)
    up = _move(robot, [x, y, safe_z], OPEN, tol=0.01, max_steps=100)
    return {"released": bool(f_open > f_before + 0.004 or f_open > 0.035), "lost_in_transit": False,
            "release_tcp": [float(v) for v in pos], "finger_before": f_before, "finger_open": f_open,
            "tcp": up["tcp"]}


def act_goto(robot, x, y, z, gripper=-1.0):
    """Move the TCP (top-down orientation) to world (x, y, z) holding the gripper command; for re-observation."""
    r = _move(robot, [x, y, z], gripper, tol=0.005, max_steps=200)
    return r


def act_wait(robot, steps=10, gripper=0.0):
    """Hold still for a number of steps (e.g. to let the scorer settle); returns the step count."""
    n = int(max(1, min(200, steps)))
    for _ in range(n):
        robot.step(0, 0, 0, 0, 0, 0, gripper)
    return {"waited": n, "finger": _fing(robot)}
