"""Closed-loop end-effector control helpers and act tools (world-frame OSC_POSE delta steps)."""

_POS_GAIN = 15.0      # normalized command per metre of position error (1.0 ~ 1.4 cm/step steady)
_POS_DAMP = 100.0     # normalized command per (metre/step) of measured TCP velocity (removes overshoot)
_ROT_GAIN = 2.5       # normalized command per radian of orientation error
_ROT_MAX = 0.5        # the expert never exceeded 0.5 on rotation components


def _rv2mat(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _mat2rv(R):
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        ax = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_error(target_rv, current_rv):
    """World-frame axis-angle that rotates the current orientation onto the target."""
    return _mat2rv(_rv2mat(target_rv) @ _rv2mat(current_rv).T)


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _command(frame, pos, rot, vmax, rmax, vel=None):
    tcp = np.asarray(frame.tcp, dtype=float)
    e = np.asarray(pos, dtype=float) - tcp[:3]
    d = _POS_GAIN * e
    if vel is not None:
        d = d - _POS_DAMP * np.asarray(vel, dtype=float)
    n = float(np.max(np.abs(d)))
    if n > vmax:
        d = d * (vmax / n)
    if rot is None:
        r = np.zeros(3)
        re = 0.0
    else:
        er = _rot_error(rot, tcp[3:6])
        re = float(np.linalg.norm(er))
        r = _ROT_GAIN * er
        m = float(np.max(np.abs(r)))
        if m > rmax:
            r = r * (rmax / m)
    return d, r, float(np.linalg.norm(e)), re


def act_move(robot, pos, rot=None, grip=-1.0, tol=0.004, rot_tol=0.05, max_steps=200,
             vmax=1.0, rmax=0.5, hold=0, stall_stop=False, stall_err=0.03):
    """Servo the TCP to world position `pos` [x,y,z] (m; None = stay where it is) and axis-angle `rot` (rad, None keeps
    orientation free) while sending gripper `grip` (+1 close, -1 open, 0 hold) every step.
    Stops when within tol/rot_tol (then keeps servoing `hold` more steps), after max_steps, or,
    with stall_stop, when the TCP stops moving near the goal (contact). Returns measured errors,
    steps, final TCP and finger gap statistics."""
    if pos is None:
        pos = [float(v) for v in np.asarray(robot.now().tcp)[:3]]
    steps = 0
    inside = 0
    held = 0
    hist = []
    gaps = []
    reached = False
    stalled = False
    pe = 1.0
    re = 0.0
    while steps < max_steps:
        f = robot.now()
        hist.append(np.asarray(f.tcp, dtype=float)[:3])
        vel = hist[-1] - hist[-2] if len(hist) > 1 else None
        d, r, pe, re = _command(f, pos, rot, vmax, min(rmax, _ROT_MAX), vel)
        gaps.append(_gap(f))
        if pe < tol and re < rot_tol:
            inside += 1
        else:
            inside = 0
        if inside >= 2:
            reached = True
        if reached:
            if held >= hold:
                break
            held += 1
        if stall_stop and len(hist) > 12 and not reached:
            moved = float(np.linalg.norm(hist[-1] - hist[-9]))
            if moved < 0.002 and pe < stall_err and re < max(rot_tol, 0.1):
                stalled = True
                if held >= hold:
                    break
                held += 1
        robot.step(float(d[0]), float(d[1]), float(d[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
        steps += 1
    f = robot.now()
    return {"reached": reached, "stalled": stalled, "steps": steps, "pos_err": pe, "rot_err": re,
            "tcp": [float(v) for v in np.asarray(f.tcp)], "gap": _gap(f),
            "min_gap": float(min(gaps)) if gaps else _gap(f)}


def act_path(robot, points, rot=None, grip=1.0, pass_tol=0.02, tol=0.006, max_steps=400,
             vmax=0.8, min_gap=-1.0):
    """Follow a list of world waypoints [[x,y,z],...] with fixed orientation `rot`, switching to
    the next point within pass_tol and stopping at the last within tol. With min_gap > 0 the
    motion aborts when the finger gap drops below it (grasp lost). Returns progress and gaps."""
    steps = 0
    idx = 0
    gaps = []
    lost = False
    pe = 1.0
    n = len(points)
    inside = 0
    hist = []
    since = 0
    stalls = 0
    while steps < max_steps:
        f = robot.now()
        g = _gap(f)
        gaps.append(g)
        if min_gap > 0 and steps > 5 and g < min_gap:
            lost = True
            break
        hist.append(np.asarray(f.tcp, dtype=float)[:3])
        vel = hist[-1] - hist[-2] if len(hist) > 1 else None
        d, r, pe, re = _command(f, points[idx], rot, vmax, _ROT_MAX, vel if idx == n - 1 else None)
        if idx < n - 1 and pe < pass_tol:
            idx += 1
            since = 0
            continue
        since += 1
        stuck = since > 12 and float(np.linalg.norm(hist[-1] - hist[-11])) < 0.004
        if idx < n - 1 and stuck and pe < 4 * pass_tol:
            # the constraint (door arc) does not let the TCP reach this point: aim at the next one
            idx += 1
            since = 0
            stalls += 1
            continue
        if idx == n - 1 and (pe < tol or (stuck and pe < 4 * pass_tol)):
            inside += 1
            if inside >= 2:
                break
        robot.step(float(d[0]), float(d[1]), float(d[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
        steps += 1
    f = robot.now()
    return {"index": idx, "of": n, "lost": lost, "steps": steps, "pos_err": pe, "stalls": stalls,
            "tcp": [float(v) for v in np.asarray(f.tcp)], "gap": _gap(f),
            "min_gap": float(min(gaps)) if gaps else _gap(f)}


def act_observe(robot, steps=1, grip=0.0):
    """Hold still for `steps` native steps with gripper command `grip`; returns the final TCP."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, float(grip))
    f = robot.now()
    return {"tcp": [float(v) for v in np.asarray(f.tcp)], "gap": _gap(f)}


def act_sequence(robot, segments):
    """Run a list of servo segments in order. Each segment is a dict of act_move arguments
    (pos, rot, grip, tol, rot_tol, max_steps, vmax, rmax, hold, stall_stop) plus optional
    `name` and `need_gap` (abort the sequence if the finger gap is below it before the segment
    starts, i.e. the held object was lost). Returns per-segment results and whether it completed."""
    results = []
    keys = ("pos", "rot", "grip", "tol", "rot_tol", "max_steps", "vmax", "rmax", "hold", "stall_stop", "stall_err")
    for seg in segments:
        need = float(seg.get("need_gap", -1.0))
        g = _gap(robot.now())
        if need > 0 and g < need:
            return {"completed": False, "aborted_at": seg.get("name", ""), "gap": g, "results": results}
        kw = {k: seg[k] for k in keys if k in seg}
        r = act_move(robot, **kw)
        results.append({"name": seg.get("name", ""), "reached": r["reached"], "stalled": r["stalled"],
                        "steps": r["steps"], "pos_err": round(r["pos_err"], 4), "rot_err": round(r["rot_err"], 3),
                        "gap": round(r["gap"], 4), "min_gap": round(r["min_gap"], 4)})
    f = robot.now()
    return {"completed": True, "results": results, "tcp": [float(v) for v in np.asarray(f.tcp)], "gap": _gap(f)}
