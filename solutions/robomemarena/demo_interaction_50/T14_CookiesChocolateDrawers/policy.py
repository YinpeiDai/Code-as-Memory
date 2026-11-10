"""Policy for T14 'Put cookies into the top drawer and put chocolate into another drawer'.

Sequence (inferred from 50 demonstrations): open top drawer (horizontal grasp of handle, pull -y),
pick cookies, drop above open top drawer, push top drawer closed, open middle drawer, pick chocolate,
drop above open middle drawer.  Geometry comes from agentview RGB-D at the first decision.
"""

DOWN = 0
HANDLE = 1
KP = 15.0
KR = 2.0
KI = 1.0
KRI = 0.15
ELBOW_LIMIT = -0.2          # joint 4 value treated as reach limit
DROP_DX = -0.03              # drop on the base side of the drawer centre (reach margin, R007-R024)
CARRY_VIA = [0.10, -0.05, 0.72]  # R019: straight carry from the far cookie locked the elbow; demo carries pass x~0.14,y~-0.08
COOKIE_DX = -0.02            # grasp the 11 cm-long cookie box 2 cm nearer the base (R013/R020: elbow at reach limit)
RMAX = 0.5                  # rotation command cap (demos never exceed 0.5); R004 elbow lock at 1.0
# offsets measured on 50 demos (see task_analysis.md)
CAB_X_TO_HANDLE = -0.1292   # handle_x = cabinet 99th pct x + this
CAB_Y_TO_HANDLE = -0.1947   # handle_y = cabinet 99th pct y + this
CAB_TOP_TO_HZ = -0.041      # top handle z = cabinet top z + this
MID_DX = 0.0006
MID_DY = -0.0057
MID_DZ = -0.0663
PRIOR_HANDLE = [0.135, 0.458, 0.673]
PRIOR_COOKIE = [0.218, -0.328]
PRIOR_CHOC = [-0.228, -0.320]
OBJ_X_BIAS = -0.008         # top-surface centroid is biased towards the camera (+x)
GRASP_Z = 0.434
TABLE_Z = 0.431


def _aa2R(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-9))
        return v * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _target_R(code):
    if int(code) == HANDLE:
        return _aa2R([1.5 * math.pi, 0.0, 0.0])
    return _aa2R([math.pi, 0.0, 0.0])


def _gap(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _cloud(frame):
    K = frame.agentview_K
    E = frame.agentview_extrinsic
    D = frame.agentview_depth.astype(float) / 1000.0
    v, u = np.mgrid[0:256, 0:256]
    x = (u - K[0, 2]) / K[0, 0] * D
    y = (v - K[1, 2]) / K[1, 1] * D
    c = np.stack([x, y, D], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    w = (c - t) @ R
    w = w.reshape(256, 256, 3)
    valid = (D > 0.05)
    return w, valid


def _find_cabinet(frame):
    w, valid = _cloud(frame)
    rgb = frame.agentview_rgb.astype(int)
    dark = rgb.max(-1) < 70
    m = valid & dark & (w[..., 1] > 0.25) & (w[..., 2] > 0.43) & (w[..., 2] < 1.0) & (np.abs(w[..., 0]) < 0.5)
    P = w[m]
    if len(P) < 300:
        return None
    xmax = float(np.percentile(P[:, 0], 99))
    ymax = float(np.percentile(P[:, 1], 99))
    ztop = float(np.percentile(P[:, 2], 99))
    hx = xmax + CAB_X_TO_HANDLE
    hy = ymax + CAB_Y_TO_HANDLE
    hz = ztop + CAB_TOP_TO_HZ
    return {"handle": [hx, hy, hz], "n": int(len(P)), "ztop": ztop}


def _find_object(frame, prior, r=0.06):
    w, valid = _cloud(frame)
    d = np.hypot(w[..., 0] - prior[0], w[..., 1] - prior[1])
    m = valid & (d < r) & (w[..., 2] > TABLE_Z + 0.006) & (w[..., 2] < 0.6)
    P = w[m]
    if len(P) < 15:
        return None
    zt = float(np.percentile(P[:, 2], 95))
    Q = P[P[:, 2] > zt - 0.008]
    if len(Q) < 5:
        return None
    return [float(Q[:, 0].mean()) + OBJ_X_BIAS, float(Q[:, 1].mean()), zt, int(len(P))]


def _table_z(frame):
    w, valid = _cloud(frame)
    m = valid & (w[..., 2] > 0.3) & (w[..., 2] < 0.47) & (np.abs(w[..., 0]) < 0.3) & (np.abs(w[..., 1]) < 0.2)
    P = w[m]
    if len(P) < 100:
        return None
    return float(np.median(P[:, 2]))


def _servo(robot, pos, code, grip, tol=0.004, rtol=0.05, max_steps=200, vmax=1.0, hold=0):
    """PI-control TCP to pos/orientation code; returns (steps, reached, final_err).
    The integral term (only accumulated within 2 cm / 0.3 rad of the target) removes the steady-state
    sag observed when the gripper carries an object (rollout R001); a stall exit stops waiting when
    the error no longer shrinks."""
    pos = np.asarray(pos, dtype=float)
    Rt = _target_R(code)
    n = 0
    reached = False
    err = 1.0
    rerr = 0.0
    ip = np.zeros(3)
    ir = np.zeros(3)
    best = 1e9
    best_n = 0
    stalled = False
    while n < max_steps:
        fr = robot.now()
        p = np.asarray(fr.tcp[:3], dtype=float)
        Rc = _aa2R(fr.tcp[3:6])
        e = pos - p
        re = _R2aa(Rt @ Rc.T)
        err = float(np.linalg.norm(e))
        rerr = float(np.linalg.norm(re))
        if err < tol and rerr < rtol:
            reached = True
            break
        if float(fr.joints[3]) > ELBOW_LIMIT and (rerr > 0.12 or n - best_n > 5):
            # elbow almost straight (Panda joint 4 upper limit -0.07): reach limit, R004/R005 lock-ups
            if err < 0.04 and rerr < 0.2:
                break
            stalled = True
            break
        score = err + 0.05 * rerr
        if score < best - 0.0005:
            best = score
            best_n = n
        if n - best_n > 15 and err < max(3 * tol, 0.012) and rerr < 0.15:
            break
        if n - best_n > 30:
            stalled = True
            break
        if err < 0.02:
            ip = ip + e
        else:
            ip = ip * 0.0
        if rerr < 0.3:
            ir = ir + re
        else:
            ir = ir * 0.0
        iterm = np.clip(KI * ip, -0.3, 0.3)
        a = KP * e
        na = float(np.linalg.norm(a))
        if na > vmax:
            a = a * (vmax / na)
        elif na < 0.12 and err >= tol:
            a = a * (0.12 / max(na, 1e-9))
        a = a + iterm
        r = KR * re
        nr = float(np.linalg.norm(r))
        if nr > RMAX:
            r = r * (RMAX / nr)
        r = r + np.clip(KRI * ir, -0.2, 0.2)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
        n += 1
    if max_steps > 0 and not reached and (err > max(3 * tol, 0.012) or rerr > 0.15):
        return n, None, err
    k = 0
    while k < hold:
        # zero arm command during gripper actuation, as in the demos (keeps the controller target at the
        # current pose; P-corrections during holds coincided with joint-4 drift in R004-R010)
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
        k += 1
        n += 1
    return n, reached, err


def act_path(robot, waypoints):
    """Follow waypoints [x, y, z, orient(0 down,1 handle), grip(+1 close,-1 open), hold_steps, tol, vmax].
    Each waypoint is PI-servoed (position + orientation) with the given gripper command, then held
    hold_steps steps (gripper actuation).  Stops early with aborted=true when a waypoint stalls far from
    its target (blocked / joint limit).  Returns per-waypoint reach flags, final tcp and finger gap."""
    out = []
    total = 0
    for wp in waypoints:
        x, y, z, code, grip = wp[0], wp[1], wp[2], wp[3], wp[4]
        hold = int(wp[5]) if len(wp) > 5 else 0
        tol = float(wp[6]) if len(wp) > 6 else 0.006
        vmax = float(wp[7]) if len(wp) > 7 else 1.0
        n, ok, err = _servo(robot, [x, y, z], code, grip, tol=tol, max_steps=220, vmax=vmax, hold=hold)
        total += n
        out.append([bool(ok), round(err, 4)])
        if ok is None:
            fr = robot.now()
            return {"aborted": True, "at": len(out) - 1, "reached": out, "steps": total,
                    "tcp": [round(float(v), 4) for v in fr.tcp[:3]], "gap": round(_gap(fr), 4)}
    fr = robot.now()
    return {"aborted": False, "reached": out, "steps": total, "tcp": [round(float(v), 4) for v in fr.tcp[:3]], "gap": round(_gap(fr), 4)}


def act_grasp_slide(robot, target_y, close_steps=25, max_steps=150, require_grasp=True, vmax=0.6):
    """Close the gripper at the current pose (horizontal handle grasp), check the finger gap, then slide
    along world y towards target_y with the gripper closed, holding x, z and orientation, until the
    motion stalls.  Returns start/end y, finger gaps, and whether a handle was held."""
    fr = robot.now()
    p0 = np.asarray(fr.tcp[:3], dtype=float)
    _servo(robot, p0, HANDLE, 1.0, tol=0.0, max_steps=0, hold=close_steps)
    g_closed = _gap(robot.now())
    held = g_closed > 0.004
    if require_grasp and not held:
        return {"held": False, "gap": round(g_closed, 4), "y0": round(float(p0[1]), 4), "y1": round(float(robot.now().tcp[1]), 4), "moved": 0.0}
    Rt = _target_R(HANDLE)
    ys = []
    n = 0
    lost_at = None
    while n < max_steps:
        fr = robot.now()
        p = np.asarray(fr.tcp[:3], dtype=float)
        ys.append(float(p[1]))
        if lost_at is None and _gap(fr) < 0.003 and held:
            lost_at = float(p[1])
        if abs(target_y - p[1]) < 0.003:
            break
        if n > 25 and abs(ys[-1] - ys[-11]) < 0.0015:
            break
        e = np.array([p0[0] - p[0], target_y - p[1], p0[2] - p[2]])
        a = KP * e
        a[1] = max(-vmax, min(vmax, a[1]))
        if abs(a[1]) < 0.28:
            a[1] = 0.28 if a[1] >= 0 else -0.28
        a = np.clip(a, -1, 1)
        r = np.clip(KR * _R2aa(Rt @ _aa2R(fr.tcp[3:6]).T), -1, 1)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), 1.0)
        n += 1
    fr = robot.now()
    y1 = float(fr.tcp[1])
    return {"held": bool(held), "gap": round(g_closed, 4), "gap_end": round(_gap(fr), 4), "y0": round(float(p0[1]), 4),
            "y1": round(y1, 4), "moved": round(y1 - float(p0[1]), 4), "lost_at": lost_at, "steps": n}


def act_grasp_lift(robot, lift_z, close_steps=22, back_x=0.0):
    """Close the gripper at the current pose (top-down grasp), then lift to lift_z while shifting back_x
    in world x (negative = towards the robot base, relieves the reach limit).
    Returns the finger gap after closing and after lifting."""
    fr = robot.now()
    p0 = np.asarray(fr.tcp[:3], dtype=float)
    _servo(robot, p0, DOWN, 1.0, tol=0.0, max_steps=0, hold=close_steps)
    g1 = _gap(robot.now())
    _servo(robot, [p0[0] + back_x, p0[1], lift_z], DOWN, 1.0, tol=0.01, max_steps=120)
    fr = robot.now()
    return {"gap_closed": round(g1, 4), "gap_lifted": round(_gap(fr), 4), "tcp": [round(float(v), 4) for v in fr.tcp[:3]]}


def act_wait(robot, steps=40, grip=-1.0):
    """Hold the current pose for a number of steps (lets the scorer settle / objects drop)."""
    fr = robot.now()
    p0 = np.asarray(fr.tcp[:3], dtype=float)
    R = _aa2R(fr.tcp[3:6])
    code = HANDLE if abs(R[2, 2]) < 0.5 else DOWN
    _servo(robot, p0, code, grip, tol=0.0, max_steps=0, hold=int(steps))
    fr = robot.now()
    return {"gap": round(_gap(fr), 4), "tcp": [round(float(v), 4) for v in fr.tcp[:3]]}


def _at_pose(frame, target, code, ptol=0.012, rtol=0.12):
    """True when the measured TCP is within ptol of target and rtol of the orientation code."""
    p = np.asarray(frame.tcp[:3], dtype=float)
    re = _R2aa(_target_R(code) @ _aa2R(frame.tcp[3:6]).T)
    return bool(np.linalg.norm(np.asarray(target, dtype=float) - p) < ptol and np.linalg.norm(re) < rtol)


def _reapproach(frame, h):
    """Waypoints: open, back off to a central pose, rotate there, then approach handle h along +y."""
    p = [float(v) for v in frame.tcp[:3]]
    hx, hy, hz = h
    return [
        [p[0], p[1] - 0.05, p[2], HANDLE if abs(_aa2R(frame.tcp[3:6])[2, 2]) < 0.5 else DOWN, -1, 10, 0.02],
        [0.0, 0.12, 0.70, DOWN, -1, 0, 0.03],
        [0.0, 0.12, 0.69, HANDLE, -1, 0, 0.02],
        [hx, hy - 0.12, hz, HANDLE, -1, 0, 0.01],
        [hx, hy - 0.03, hz, HANDLE, -1, 0, 0.004, 0.6],
        [hx, hy, hz, HANDLE, -1, 3, 0.003, 0.4]]


def _parse_goal(goal):
    g = (goal or "").lower()
    first = "cookies"
    if "chocolate" in g and "cookie" in g and g.find("chocolate") < g.find("cookie"):
        first = "chocolate"
    return first


def _mem_default(memory):
    keys = ["stage", "handle_top", "handle_mid", "cookie", "choc", "top_pulled_y", "mid_pulled_y",
            "top_closed_y", "attempts", "drop_retries", "issued", "recoveries", "events", "progress", "uncertain"]
    m = {}
    for k in keys:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "start"
    if m["attempts"] is None:
        m["attempts"] = 0
    if m["drop_retries"] is None:
        m["drop_retries"] = 0
    if m["recoveries"] is None:
        m["recoveries"] = 0
    if m["events"] is None:
        m["events"] = []
    if m["progress"] is None:
        m["progress"] = []
    if m["uncertain"] is None:
        m["uncertain"] = ""
    return m


def _ev(m, s):
    ev = list(m["events"])
    ev.append(s)
    m["events"] = ev[-30:]


def _done(m, s):
    pr = list(m["progress"])
    if s not in pr:
        pr.append(s)
    m["progress"] = pr


def _out(m, tool, args):
    return {"memory": m, "act": {"tool": tool, "args": args}}


def _outs(st, m, tool, args):
    m["issued"] = st
    return {"memory": m, "act": {"tool": tool, "args": args}}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive cabinet/objects once, then run open-place-close-open-place with checks.
    If the previous act_path aborted (stalled far from a waypoint), first back off to a central safe pose,
    then re-issue the stage that was interrupted (bounded)."""
    m = _mem_default(memory or {})
    cur = obs.current
    tcp = [float(v) for v in cur.tcp[:3]]
    lt = last if isinstance(last, str) else ""
    if '"aborted": true' in lt and m["issued"] is not None and int(m["recoveries"]) < 3:
        m["recoveries"] = int(m["recoveries"]) + 1
        _ev(m, "act aborted in %s at %s; recovery" % (m["issued"], [round(v, 3) for v in tcp]))
        m["stage"] = m["issued"]
        grip = 1 if 0.02 < _gap(cur) < 0.075 else -1
        if m["issued"] == "place_cookie" and grip < 0:
            m["stage"] = "check_cookie_drop"      # already released (R023 looped re-placing an empty gripper)
        if m["issued"] == "place_choc" and grip < 0:
            m["stage"] = "final_wait"
        Rc = _aa2R(cur.tcp[3:6])
        code = HANDLE if abs(Rc[2, 2]) < 0.5 else DOWN
        sy = 0.12 if tcp[1] > 0 else -0.12
        return _out(m, "act_path", {"waypoints": [
            [0.5 * tcp[0], 0.7 * tcp[1], min(max(tcp[2], 0.62), 0.72), code, grip, 0, 0.03],
            [0.0, sy, 0.70, DOWN, grip, 0, 0.03]]})
    return _perceive_decide_core(obs, goal, m, last)


def _perceive_decide_core(obs, goal, m, last):
    """Stage machine body (see perceive_decide)."""
    cur = obs.current
    tcp = [float(v) for v in cur.tcp[:3]]
    first = obs.frames[0]
    st = m["stage"]
    step = int(cur.step)

    if st == "start":
        cab = _find_cabinet(cur)
        if cab is None:
            h = list(PRIOR_HANDLE)
            m["uncertain"] = "cabinet not detected; prior handle used"
        else:
            h = cab["handle"]
        m["handle_top"] = [h[0], h[1], h[2]]
        m["handle_mid"] = [h[0] + MID_DX, h[1] + MID_DY, h[2] + MID_DZ]
        c = _find_object(cur, PRIOR_COOKIE)
        k = _find_object(cur, PRIOR_CHOC)
        m["cookie"] = [c[0], c[1], c[2]] if c else [PRIOR_COOKIE[0], PRIOR_COOKIE[1], 0.445]
        m["choc"] = [k[0], k[1], k[2]] if k else [PRIOR_CHOC[0], PRIOR_CHOC[1], 0.453]
        _ev(m, "perceived cab=%s cookie=%s choc=%s" % (cab is not None, c is not None, k is not None))
        st = "approach_top"

    ht = m["handle_top"]
    hm = m["handle_mid"]

    if st == "approach_top":
        m["stage"] = "pull_top"
        hx, hy, hz = ht
        return _outs(st, m, "act_path", {"waypoints": [
            [hx, hy - 0.12, hz, HANDLE, -1, 0, 0.01],
            [hx, hy - 0.03, hz, HANDLE, -1, 0, 0.004, 0.6],
            [hx, hy, hz, HANDLE, -1, 3, 0.003, 0.4]]})

    if st == "pull_top":
        if not _at_pose(cur, ht, HANDLE) and int(m["attempts"]) < 3:
            m["attempts"] = int(m["attempts"]) + 1
            _ev(m, "not at top handle before grasp %s" % [round(v, 3) for v in tcp])
            m["stage"] = "pull_top"
            return _outs("approach_top", m, "act_path", {"waypoints": _reapproach(cur, ht)})
        m["stage"] = "check_top_open"
        return _outs(st, m, "act_grasp_slide", {"target_y": ht[1] - 0.185})

    if st == "check_top_open":
        moved = tcp[1] - float(first.tcp[1])
        held = _gap(cur) > 0.003
        if moved < -0.13:
            m["top_pulled_y"] = tcp[1]
            _done(m, "top_open")
            _ev(m, "top drawer pulled %.3f" % moved)
            m["attempts"] = 0
            st = "to_cookie"
        else:
            _ev(m, "top pull failed moved=%.3f held=%s" % (moved, held))
            m["attempts"] = int(m["attempts"]) + 1
            if int(m["attempts"]) > 2:
                m["uncertain"] = "top drawer could not be confirmed open"
                m["top_pulled_y"] = tcp[1]
                st = "to_cookie"
            else:
                # handle now displaced by the partial pull; re-approach from the front
                hx, hy, hz = ht
                ny = hy + min(0.0, moved)
                m["handle_top"] = [hx, ny, hz]
                m["stage"] = "pull_top"
                return _outs(st, m, "act_path", {"waypoints": [
                    [tcp[0], tcp[1], tcp[2], HANDLE, -1, 12, 0.01],
                    [hx, tcp[1] - 0.06, hz, HANDLE, -1, 0, 0.008],
                    [hx, ny - 0.03, hz, HANDLE, -1, 0, 0.004, 0.6],
                    [hx, ny + 0.003 * int(m["attempts"]), hz, HANDLE, -1, 3, 0.003, 0.4]]})

    if st == "to_cookie":
        m["stage"] = "grasp_cookie"
        c = m["cookie"]
        yb = float(m["top_pulled_y"]) - 0.11
        return _outs(st, m, "act_path", {"waypoints": [
            [tcp[0], tcp[1], tcp[2], HANDLE, -1, 12, 0.01],
            [ht[0], yb, tcp[2], HANDLE, -1, 0, 0.015],
            [ht[0], yb, 0.76, HANDLE, -1, 0, 0.02],
            [c[0] + COOKIE_DX, c[1], 0.60, DOWN, -1, 0, 0.01],
            [c[0] + COOKIE_DX, c[1], GRASP_Z, DOWN, -1, 2, 0.003, 0.5]]})

    if st == "grasp_cookie":
        m["stage"] = "check_cookie"
        return _outs(st, m, "act_grasp_lift", {"lift_z": 0.62, "back_x": -0.05})

    if st == "check_cookie":
        g = _gap(cur)
        if 0.02 < g < 0.075 and tcp[2] > 0.55:
            _done(m, "cookie_held")
            m["attempts"] = 0
            st = "place_cookie"
        else:
            _ev(m, "cookie grasp failed gap=%.3f" % g)
            m["attempts"] = int(m["attempts"]) + 1
            if int(m["attempts"]) > 2 or step > 1700:
                m["uncertain"] = "cookie grasp failed repeatedly"
                st = "close_top"
            else:
                m["stage"] = "regrasp_cookie"
                return _outs(st, m, "act_path", {"waypoints": [
                    [tcp[0], tcp[1], 0.62, DOWN, -1, 12, 0.02]]})

    if st == "regrasp_cookie":
        c = _find_object(cur, m["cookie"])
        if c:
            m["cookie"] = [c[0], c[1], c[2]]
        _ev(m, "cookie re-perceived %s" % (c is not None))
        m["stage"] = "grasp_cookie"
        cc = m["cookie"]
        return _outs(st, m, "act_path", {"waypoints": [
            [cc[0] + COOKIE_DX, cc[1], 0.60, DOWN, -1, 0, 0.01],
            [cc[0] + COOKIE_DX, cc[1], GRASP_Z, DOWN, -1, 2, 0.003, 0.5]]})

    if st == "place_cookie":
        m["stage"] = "check_cookie_drop"
        py = float(m["top_pulled_y"])
        # drop over the near half of the open drawer (interior ~ py+0.015 .. py+0.15): R010/R019/R020/R023/R024 showed
        # elbow (joint 4) lock-ups and drift at the far drop poses (y ~0.41-0.42, x ~0.12-0.15).
        dx = ht[0] + DROP_DX   # drawer interior x ~ [hx-0.10, hx+0.11] (demo ep0 depth); box <= 11 cm long
        dy = py + 0.075
        return _outs(st, m, "act_path", {"waypoints": [
            [tcp[0], tcp[1], max(tcp[2], 0.62), DOWN, 1, 0, 0.02],
            [CARRY_VIA[0], CARRY_VIA[1], CARRY_VIA[2], DOWN, 1, 0, 0.03],
            [dx, dy, 0.76, DOWN, 1, 0, 0.012],
            [dx, dy, 0.755, DOWN, 1, 0, 0.008, 0.5],
            [dx, dy, 0.755, DOWN, -1, 12, 0.03, 0.3],
            [dx - 0.02, py - 0.04, 0.755, DOWN, -1, 0, 0.03]]})

    if st == "check_cookie_drop":
        g = _gap(cur)
        _ev(m, "cookie released gap=%.3f" % g)
        _done(m, "cookie_released")
        st = "close_top"

    if st == "close_top":
        m["stage"] = "push_top"
        hx, hy, hz = ht
        py = float(m["top_pulled_y"]) if m["top_pulled_y"] is not None else hy - 0.166
        # rotate DOWN->HANDLE only after moving towards the base (elbow bent); R004/R006 rotated near the
        # elbow limit and lost the orientation.
        return _outs(st, m, "act_path", {"waypoints": [
            [min(tcp[0], hx - 0.03, 0.10), min(tcp[1], py - 0.03), max(tcp[2], 0.745), DOWN, -1, 0, 0.02],
            [hx - 0.08, py - 0.12, 0.72, DOWN, -1, 0, 0.02],
            [hx, py - 0.10, hz + 0.02, HANDLE, -1, 0, 0.01],
            [hx, py - 0.03, hz, HANDLE, -1, 0, 0.004, 0.6],
            [hx, py, hz, HANDLE, -1, 3, 0.003, 0.4]]})

    if st == "push_top":
        py = float(m["top_pulled_y"]) if m["top_pulled_y"] is not None else ht[1] - 0.166
        tgt = [ht[0], py, ht[2]]
        if not _at_pose(cur, tgt, HANDLE) and int(m["attempts"]) < 3:
            m["attempts"] = int(m["attempts"]) + 1
            _ev(m, "not at open top handle before push %s" % [round(v, 3) for v in tcp])
            m["stage"] = "push_top"
            return _outs("close_top", m, "act_path", {"waypoints": _reapproach(cur, tgt)})
        m["stage"] = "check_top_closed"
        return _outs(st, m, "act_grasp_slide", {"target_y": ht[1] + 0.01, "require_grasp": False})

    if st == "check_top_closed":
        if tcp[1] > ht[1] - 0.02:
            m["top_closed_y"] = tcp[1]
            _done(m, "top_closed")
            _ev(m, "top closed y=%.3f" % tcp[1])
            m["attempts"] = 0
            st = "approach_mid"
        else:
            _ev(m, "top push stalled y=%.3f" % tcp[1])
            m["attempts"] = int(m["attempts"]) + 1
            if int(m["attempts"]) > 2:
                m["uncertain"] = "top drawer close not confirmed"
                st = "approach_mid"
            else:
                m["stage"] = "check_top_closed"
                return _outs(st, m, "act_grasp_slide", {"target_y": ht[1] + 0.02, "require_grasp": False, "close_steps": 5})

    if st == "approach_mid":
        m["stage"] = "pull_mid"
        hx, hy, hz = hm
        return _outs(st, m, "act_path", {"waypoints": [
            [tcp[0], tcp[1], tcp[2], HANDLE, -1, 12, 0.01],
            [hx, tcp[1] - 0.12, ht[2], HANDLE, -1, 0, 0.01],
            [hx, hy - 0.12, hz, HANDLE, -1, 0, 0.006],
            [hx, hy - 0.03, hz, HANDLE, -1, 0, 0.004, 0.6],
            [hx, hy, hz, HANDLE, -1, 3, 0.003, 0.4]]})

    if st == "pull_mid":
        if not _at_pose(cur, hm, HANDLE) and int(m["attempts"]) < 3:
            m["attempts"] = int(m["attempts"]) + 1
            _ev(m, "not at mid handle before grasp %s" % [round(v, 3) for v in tcp])
            m["stage"] = "pull_mid"
            return _outs("approach_mid", m, "act_path", {"waypoints": _reapproach(cur, hm)})
        m["stage"] = "check_mid_open"
        return _outs(st, m, "act_grasp_slide", {"target_y": hm[1] - 0.20})

    if st == "check_mid_open":
        moved = tcp[1] - float(first.tcp[1])
        if moved < -0.13:
            m["mid_pulled_y"] = tcp[1]
            _done(m, "mid_open")
            _ev(m, "mid drawer pulled %.3f" % moved)
            m["attempts"] = 0
            st = "to_choc"
        else:
            _ev(m, "mid pull failed moved=%.3f" % moved)
            m["attempts"] = int(m["attempts"]) + 1
            if int(m["attempts"]) > 2:
                m["uncertain"] = "middle drawer could not be confirmed open"
                m["mid_pulled_y"] = tcp[1]
                st = "to_choc"
            else:
                hx, hy, hz = hm
                ny = hy + min(0.0, moved)
                m["handle_mid"] = [hx, ny, hz]
                m["stage"] = "pull_mid"
                return _outs(st, m, "act_path", {"waypoints": [
                    [tcp[0], tcp[1], tcp[2], HANDLE, -1, 12, 0.01],
                    [hx, tcp[1] - 0.06, hz, HANDLE, -1, 0, 0.008],
                    [hx, ny - 0.03, hz, HANDLE, -1, 0, 0.004, 0.6],
                    [hx, ny + 0.003 * int(m["attempts"]), hz, HANDLE, -1, 3, 0.003, 0.4]]})

    if st == "to_choc":
        m["stage"] = "grasp_choc"
        c = m["choc"]
        yb = float(m["mid_pulled_y"]) - 0.14
        return _outs(st, m, "act_path", {"waypoints": [
            [tcp[0], tcp[1], tcp[2], HANDLE, -1, 12, 0.01],
            [hm[0], yb, tcp[2], HANDLE, -1, 0, 0.015],
            [hm[0], yb, 0.66, HANDLE, -1, 0, 0.02],
            [c[0], c[1], 0.62, DOWN, -1, 0, 0.01],
            [c[0], c[1], GRASP_Z, DOWN, -1, 2, 0.003, 0.5]]})

    if st == "grasp_choc":
        m["stage"] = "check_choc"
        return _outs(st, m, "act_grasp_lift", {"lift_z": 0.78})

    if st == "check_choc":
        g = _gap(cur)
        if 0.012 < g < 0.075 and tcp[2] > 0.6:
            _done(m, "choc_held")
            m["attempts"] = 0
            st = "place_choc"
        else:
            _ev(m, "choc grasp failed gap=%.3f" % g)
            m["attempts"] = int(m["attempts"]) + 1
            if int(m["attempts"]) > 3:
                m["uncertain"] = "chocolate grasp failed repeatedly"
                m["stage"] = "final_wait"
                return _outs(st, m, "act_wait", {"steps": 40})
            m["stage"] = "regrasp_choc"
            return _outs(st, m, "act_path", {"waypoints": [
                [tcp[0], tcp[1], 0.78, DOWN, -1, 12, 0.02]]})

    if st == "regrasp_choc":
        c = _find_object(cur, m["choc"])
        if c:
            m["choc"] = [c[0], c[1], c[2]]
        _ev(m, "choc re-perceived %s" % (c is not None))
        m["stage"] = "grasp_choc"
        cc = m["choc"]
        return _outs(st, m, "act_path", {"waypoints": [
            [cc[0], cc[1], 0.62, DOWN, -1, 0, 0.01],
            [cc[0], cc[1], GRASP_Z, DOWN, -1, 2, 0.003, 0.5]]})

    if st == "place_choc":
        m["stage"] = "final_wait"
        _done(m, "choc_released")
        dx = hm[0] + DROP_DX
        dy = float(m["mid_pulled_y"]) + 0.09
        return _outs(st, m, "act_path", {"waypoints": [
            [tcp[0], tcp[1], 0.75, DOWN, 1, 0, 0.02],
            [dx, dy, 0.74, DOWN, 1, 0, 0.012],
            [dx, dy, 0.72, DOWN, 1, 0, 0.008, 0.5],
            [dx, dy, 0.72, DOWN, -1, 20, 0.03],
            [dx, dy - 0.03, 0.74, DOWN, -1, 0, 0.03]]})

    # final_wait / fallback: observe; the official checker ends the episode on success
    if st == "final_wait" and "choc_released" in m["progress"] and step < 2150:
        k = _find_object(cur, m["choc"], r=0.05)
        if k is not None and k[2] > TABLE_Z + 0.012 and int(m["drop_retries"]) < 1:
            # chocolate still (or again) on the table: the release did not take it away -> retry pick
            _ev(m, "choc still on table after release; retry")
            m["drop_retries"] = int(m["drop_retries"]) + 1
            pr = [p for p in m["progress"] if p not in ("choc_held", "choc_released")]
            m["progress"] = pr
            m["choc"] = [k[0], k[1], k[2]]
            m["stage"] = "grasp_choc"
            return _outs(st, m, "act_path", {"waypoints": [
                [tcp[0], tcp[1], 0.78, DOWN, -1, 0, 0.03],
                [k[0], k[1], 0.62, DOWN, -1, 0, 0.01],
                [k[0], k[1], GRASP_Z, DOWN, -1, 2, 0.003, 0.5]]})
    m["stage"] = "final_wait"
    _ev(m, "waiting step=%d" % step)
    return _outs(st, m, "act_wait", {"steps": 40})
