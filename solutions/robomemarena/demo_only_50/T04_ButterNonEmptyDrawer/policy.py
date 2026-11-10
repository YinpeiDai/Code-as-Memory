"""T04_ButterNonEmptyDrawer policy: check top/middle/bottom drawers (open, look, close),
remember which one holds an object, reopen it, pick the butter and drop it inside.

Geometry constants were measured from the 32 assigned demonstrations (see task_analysis.md).
"""

R_DRAWER_RV = [4.714, 0.004, -0.002]   # expert hand orientation while working the drawers
R_DOWN_RV = [3.141, -0.001, -0.002]    # expert hand orientation for the butter pick
HANDLE_DZ = {"top": -0.019, "middle": -0.104, "bottom": -0.183}  # handle z - cabinet top z
HANDLE_DX = -0.005      # handle x - cabinet top-surface x centre
HANDLE_DY = -0.024      # handle y - cabinet front edge y
PRE_DY = 0.07           # stand-off in -y before grasping a handle
PULL_DY = 0.20          # pull request; the drawer stops at its limit (~0.17 m)
BUTTER_PRIOR = [0.111, -0.204, 0.442]
CABINET_PRIOR = [0.14, 0.465, 0.713]
OCC_MIN = 15            # raised-point count that marks a drawer as non-empty
DRAWERS = ["top", "middle", "bottom"]
RETRY_DZ = [-0.008, 0.008]   # handle height offsets tried on grasp retries


# ---------------------------------------------------------------- math helpers
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
        d = np.sqrt(np.clip(np.diag(M), 0.0, None))
        i = int(np.argmax(d))
        k = M[i] / max(d[i], 1e-9)
        return k / max(float(np.linalg.norm(k)), 1e-9) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _cloud(frame):
    """World points (N,3) of valid agentview depth pixels."""
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    ok = d > 0
    z = d[ok]
    x = (u[ok] - K[0, 2]) / K[0, 0] * z
    y = (v[ok] - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    return (pc - E[:, 3]) @ E[:, :3]


# ---------------------------------------------------------------- perception
def _perceive_cabinet(frame):
    """Cabinet top surface -> [x centre, front edge y, top z]; prior when not found."""
    P = _cloud(frame)
    m = (P[:, 0] > -0.15) & (P[:, 0] < 0.45) & (P[:, 1] > 0.2) & (P[:, 1] < 0.8) & (P[:, 2] > 0.695) & (P[:, 2] < 0.73)
    Q = P[m]
    if len(Q) < 200:
        return list(CABINET_PRIOR), False
    zt = float(np.median(Q[:, 2]))
    Q = Q[np.abs(Q[:, 2] - zt) < 0.006]
    if len(Q) < 200:
        return list(CABINET_PRIOR), False
    xc = 0.5 * (float(np.percentile(Q[:, 0], 1)) + float(np.percentile(Q[:, 0], 99)))
    yf = float(np.percentile(Q[:, 1], 1))
    if abs(xc - CABINET_PRIOR[0]) > 0.12 or abs(yf - CABINET_PRIOR[1]) > 0.12:
        return list(CABINET_PRIOR), False
    return [xc, yf, zt], True


def _perceive_butter(frame):
    """Low box near the expected butter spot -> [x, y, top z]; prior when not found."""
    P = _cloud(frame)
    bx, by = BUTTER_PRIOR[0], BUTTER_PRIOR[1]
    near = (np.abs(P[:, 0] - bx) < 0.12) & (np.abs(P[:, 1] - by) < 0.12)
    tab = near & (P[:, 2] > 0.38) & (P[:, 2] < 0.47)
    if np.sum(tab) < 50:
        return list(BUTTER_PRIOR), False
    ztab = float(np.percentile(P[tab, 2], 30))
    m = (np.abs(P[:, 0] - bx) < 0.07) & (np.abs(P[:, 1] - by) < 0.06) & (P[:, 2] > ztab + 0.005) & (P[:, 2] < ztab + 0.05)
    Q = P[m]
    if len(Q) < 15:
        return list(BUTTER_PRIOR), False
    x = 0.5 * (float(np.percentile(Q[:, 0], 2)) + float(np.percentile(Q[:, 0], 98)))
    y = 0.5 * (float(np.percentile(Q[:, 1], 2)) + float(np.percentile(Q[:, 1], 98)))
    zt = float(np.percentile(Q[:, 2], 90))
    if abs(x - bx) > 0.06 or abs(y - by) > 0.06:
        return list(BUTTER_PRIOR), False
    return [x, y, zt], True


def _drawer_occupancy(frame):
    """Count depth points raised above the floor of the drawer currently held open.
    The window is relative to the hand, which holds the open drawer's handle."""
    P = _cloud(frame)
    hx, hy, hz = [float(v) for v in frame.tcp[:3]]
    m = (P[:, 0] > hx - 0.085) & (P[:, 0] < hx + 0.035) & (P[:, 1] > hy + 0.06) & (P[:, 1] < hy + 0.175) \
        & (P[:, 2] > hz - 0.08) & (P[:, 2] < hz + 0.06)
    Z = P[m, 2]
    if len(Z) < 20:
        return 0.0, len(Z)
    floor = float(np.percentile(Z, 20))
    n = int(np.sum((Z > floor + 0.012) & (Z < floor + 0.05)))
    return float(n), len(Z)


def _parse_goal(goal):
    g = (goal or "").lower()
    order = list(DRAWERS)
    k = g.find("order")
    if k >= 0:
        seg = g[k:]
        end = seg.find(".")
        if end > 0:
            seg = seg[:end]
        found = [(seg.find(d), d) for d in DRAWERS if seg.find(d) >= 0]
        if len(found) == 3:
            order = [d for _, d in sorted(found)]
    want = "occupied"
    if ("empty" in g) and not ("non-empty" in g or "nonempty" in g or "contains" in g or "not empty" in g):
        want = "empty"
    return order, want


def _handle(cab, name):
    return [cab[0] + HANDLE_DX, cab[1] + HANDLE_DY, cab[2] + HANDLE_DZ[name]]


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Stage machine. Memory keeps the cabinet/butter estimates, the order of checked drawers,
    each drawer's observed occupancy, the chosen target drawer and where it was held open."""
    mem = dict(memory)
    for key in ("stage", "cabinet", "butter", "order", "checked", "occ_top", "occ_middle", "occ_bottom",
                "target", "open_handle", "retries", "notes"):
        mem.setdefault(key, None)
    cur = obs.current
    stage = mem["stage"]
    last = last or ""

    def out(act, args):
        return {"memory": mem, "act": {"tool": act, "args": args}}

    if stage is None:
        order, want = _parse_goal(goal)
        cab, okc = _perceive_cabinet(cur)
        but, okb = _perceive_butter(cur)
        mem.update(stage="open", cabinet=[round(v, 4) for v in cab], butter=[round(v, 4) for v in but],
                   order=order, checked=[], retries=0,
                   notes="want=%s cabinet_seen=%s butter_seen=%s" % (want, okc, okb))
        return out("act_open_drawer", {"handle": _handle(cab, order[0]), "release": False})

    cab = mem["cabinet"]
    order = mem["order"] or list(DRAWERS)
    checked = list(mem["checked"] or [])
    tcp = [float(v) for v in cur.tcp[:3]]
    width = _width(cur)

    if stage == "open":
        name = order[len(checked)]
        h = _handle(cab, name)
        pulled = h[1] - tcp[1]
        holding = 0.006 < width < 0.032
        if (not holding or pulled < 0.10) and (mem["retries"] or 0) < 2:
            mem["retries"] = (mem["retries"] or 0) + 1
            mem["notes"] = "retry open %s (width %.4f pulled %.3f)" % (name, width, pulled)
            h2 = [h[0], h[1], h[2] + RETRY_DZ[mem["retries"] - 1]]
            return out("act_open_drawer", {"handle": h2, "release": False, "fresh": True})
        n, tot = _drawer_occupancy(cur)
        mem["occ_" + name] = n
        mem["stage"] = "close"
        mem["retries"] = 0
        mem["notes"] = "%s open by %.3f, raised points %d of %d" % (name, pulled, int(n), tot)
        return out("act_close_drawer", {"handle": h})

    if stage == "close":
        name = order[len(checked)]
        checked.append(name)
        mem["checked"] = checked
        if len(checked) < len(order):
            mem["stage"] = "open"
            return out("act_open_drawer", {"handle": _handle(cab, order[len(checked)]), "release": False})
        _, want = _parse_goal(goal)
        occ = [(float(mem["occ_" + d] or 0.0), d) for d in DRAWERS]
        if want == "empty":
            target = min(occ)[1]
        else:
            best = max(occ)
            target = best[1] if best[0] >= OCC_MIN else "top"
        mem["target"] = target
        mem["stage"] = "reopen"
        mem["notes"] = "occupancy %s -> target %s" % (occ, target)
        return out("act_open_drawer", {"handle": _handle(cab, target), "release": True})

    if stage == "reopen":
        h = _handle(cab, mem["target"])
        rec = '"opened_y":'
        oy = None
        k = last.find(rec)
        if k >= 0:
            txt = last[k + len(rec):].strip()
            num = ""
            for ch in txt:
                if ch in "-0123456789.":
                    num += ch
                else:
                    break
            if num not in ("", "-", "."):
                oy = float(num)
        if (oy is None or h[1] - oy < 0.10) and (mem["retries"] or 0) < 2:
            mem["retries"] = (mem["retries"] or 0) + 1
            mem["notes"] = "retry reopen %s (opened_y %s)" % (mem["target"], oy)
            h2 = [h[0], h[1], h[2] + RETRY_DZ[mem["retries"] - 1]]
            return out("act_open_drawer", {"handle": h2, "release": True, "fresh": True})
        if oy is None:
            oy = h[1] - 0.17
        mem["open_handle"] = [h[0], oy, h[2]]
        mem["stage"] = "pick"
        mem["retries"] = 0
        return out("act_pick_butter", {"butter": mem["butter"]})

    if stage == "pick":
        if width < 0.02 and (mem["retries"] or 0) < 2:
            mem["retries"] = (mem["retries"] or 0) + 1
            mem["notes"] = "retry pick (width %.4f)" % width
            return out("act_pick_butter", {"butter": mem["butter"]})
        oh = mem["open_handle"]
        top = mem["target"] == "top"
        target = [oh[0] - 0.02, oh[1] + (0.105 if top else 0.095)]
        rel_z = cab[2] + (0.03 if top else 0.0)
        mem["stage"] = "placed"
        mem["retries"] = 0
        return out("act_place_butter", {"target": target, "release_z": rel_z})

    mem["stage"] = "wait"
    return out("act_wait", {"steps": 60})


# ---------------------------------------------------------------- control
def _servo(robot, pos, R, grip, tol=0.005, rtol=0.05, max_steps=200, umax=1.0, rmax=0.6,
           stall_axis=None, stall_min=0.0, slip_abort=False):
    """Closed-loop move of the measured TCP to pos/R (world). Returns a summary dict."""
    pos = np.asarray(pos, dtype=float)
    prev_p = None
    prev_R = None
    ok = 0
    hist = []
    phist = []
    steps = 0
    reason = "max_steps"
    while steps < max_steps:
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        Rc = _rv2m(f.tcp[3:6])
        e = pos - p
        er = _m2rv(R @ Rc.T)
        if float(np.linalg.norm(e)) < tol and float(np.linalg.norm(er)) < rtol:
            ok += 1
            if ok >= 2:
                reason = "reached"
                break
        else:
            ok = 0
        phist.append(p)
        if tol > 0 and len(phist) > 15 and float(np.linalg.norm(e)) < 3.0 * tol \
                and float(np.linalg.norm(phist[-1] - phist[-11])) < 0.001 and float(np.linalg.norm(er)) < 2.0 * rtol:
            reason = "settled"
            break
        if stall_axis is not None:
            hist.append(float(p[stall_axis]))
            start = hist[0]
            if len(hist) > 10 and abs(hist[-1] - hist[-10]) < 0.0015 and abs(hist[-1] - start) >= stall_min:
                reason = "stalled"
                break
        if slip_abort and _width(f) < 0.004:
            reason = "slipped"
            break
        v = (p - prev_p) if prev_p is not None else np.zeros(3)
        w = _m2rv(Rc @ prev_R.T) if prev_R is not None else np.zeros(3)
        u = e / 0.02 - 0.5 * v / 0.01
        m = float(np.max(np.abs(u)))
        if m > umax:
            u = u * (umax / m)
        ur = er / 0.2 - 0.5 * w / 0.1
        mr = float(np.max(np.abs(ur)))
        if mr > rmax:
            ur = ur * (rmax / mr)
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(ur[0]), float(ur[1]), float(ur[2]), grip)
        prev_p, prev_R = p, Rc
        steps += 1
    f = robot.now()
    return {"reason": reason, "steps": steps, "tcp": [round(float(x), 4) for x in f.tcp[:3]],
            "err": round(float(np.linalg.norm(pos - np.asarray(f.tcp[:3], dtype=float))), 4)}


def _hold(robot, R, grip, n):
    p = [float(x) for x in robot.now().tcp[:3]]
    return _servo(robot, p, R, grip, tol=-1.0, rtol=-1.0, max_steps=n)


def act_open_drawer(robot, handle, release=False, fresh=False):
    """Approach the drawer handle [x,y,z] from -y with the hand horizontal, grasp it, pull the
    drawer to its open limit and hold still (release=False), or let go and retreat up (release=True).
    fresh=True first opens the hand and backs off before re-approaching. Returns measured outcomes."""
    R = _rv2m(R_DRAWER_RV)
    hx, hy, hz = [float(v) for v in handle]
    log = {}
    cur = [float(v) for v in robot.now().tcp[:3]]
    if fresh:
        _hold(robot, R, -1, 10)
        cur = [float(v) for v in robot.now().tcp[:3]]
        _servo(robot, [cur[0], min(cur[1], hy - PRE_DY) - 0.03, cur[2]], R, -1, tol=0.01, max_steps=60)
        cur = [float(v) for v in robot.now().tcp[:3]]
    if cur[1] > hy - PRE_DY + 0.01:
        _servo(robot, [cur[0], hy - PRE_DY, cur[2]], R, -1, tol=0.01, max_steps=60)
    log["pre"] = _servo(robot, [hx, hy - PRE_DY, hz], R, -1, tol=0.005, rtol=0.04, max_steps=220)
    log["approach"] = _servo(robot, [hx, hy, hz], R, -1, tol=0.004, rtol=0.04, max_steps=80, umax=0.5,
                             stall_axis=1, stall_min=0.04)
    _hold(robot, R, 1, 15)
    log["grip_width"] = round(_width(robot.now()), 4)
    log["pull"] = _servo(robot, [hx, hy - PULL_DY, hz], R, 1, tol=0.003, max_steps=90, umax=0.6,
                         stall_axis=1, stall_min=0.05, slip_abort=True)
    _hold(robot, R, 1, 6)
    f = robot.now()
    log["opened_y"] = round(float(f.tcp[1]), 4)
    log["holding_width"] = round(_width(f), 4)
    if release:
        _hold(robot, R, -1, 12)
        p = [float(v) for v in robot.now().tcp[:3]]
        _servo(robot, [p[0], p[1] - 0.08, p[2]], R, -1, tol=0.01, max_steps=60)
        p = [float(v) for v in robot.now().tcp[:3]]
        log["retreat"] = _servo(robot, [p[0], p[1], 0.80], R, -1, tol=0.015, max_steps=80)
    return log


def act_close_drawer(robot, handle):
    """Push the held drawer back to its closed handle position [x,y,z] (stops when it stalls),
    release the handle and back off in -y. Returns measured outcomes."""
    R = _rv2m(R_DRAWER_RV)
    hx, hy, hz = [float(v) for v in handle]
    log = {}
    log["push"] = _servo(robot, [hx, hy + 0.01, hz], R, 1, tol=0.004, max_steps=110, umax=0.8,
                         stall_axis=1, stall_min=0.08)
    y = float(robot.now().tcp[1])
    log["closed_y"] = round(y, 4)
    log["closed"] = bool(y > hy - 0.03)
    _hold(robot, R, -1, 12)
    p = [float(v) for v in robot.now().tcp[:3]]
    log["backoff"] = _servo(robot, [p[0], hy - PRE_DY, p[2]], R, -1, tol=0.008, max_steps=70)
    return log


def act_pick_butter(robot, butter):
    """Move above the butter [x,y,top z], turn the hand to point down, descend, grasp and lift
    to 0.60 m. Returns measured outcomes including the final finger width."""
    R = _rv2m(R_DOWN_RV)
    bx, by, bz = [float(v) for v in butter]
    log = {}
    f = robot.now()
    p = [float(v) for v in f.tcp[:3]]
    if p[2] < 0.75 and p[1] > by + 0.1:
        _servo(robot, [p[0], p[1], 0.80], _rv2m(f.tcp[3:6]), -1, tol=0.02, max_steps=60)
    log["above"] = _servo(robot, [bx, by, 0.60], R, -1, tol=0.008, rtol=0.04, max_steps=250)
    log["pre"] = _servo(robot, [bx, by, bz + 0.05], R, -1, tol=0.005, rtol=0.03, max_steps=120)
    log["down"] = _servo(robot, [bx, by, bz - 0.005], R, -1, tol=0.003, rtol=0.03, max_steps=80, umax=0.4)
    _hold(robot, R, 1, 18)
    log["lift"] = _servo(robot, [bx, by, 0.60], R, 1, tol=0.01, max_steps=100, umax=0.8)
    log["width"] = round(_width(robot.now()), 4)
    return log


def act_place_butter(robot, target, release_z):
    """Carry the held butter high over the cabinet to target [x,y], descend to release_z, open the
    hand, then rise a little and hold still. Returns measured outcomes."""
    R = _rv2m(R_DOWN_RV)
    tx, ty = float(target[0]), float(target[1])
    log = {}
    p = [float(v) for v in robot.now().tcp[:3]]
    _servo(robot, [p[0], p[1], 0.80], R, 1, tol=0.015, max_steps=60)
    log["carry"] = _servo(robot, [tx, ty, 0.92], R, 1, tol=0.01, rtol=0.04, max_steps=200)
    log["descend"] = _servo(robot, [tx, ty, float(release_z)], R, 1, tol=0.005, rtol=0.04, max_steps=150, umax=0.5)
    _hold(robot, R, -1, 20)
    p = [float(v) for v in robot.now().tcp[:3]]
    log["rise"] = _servo(robot, [p[0], p[1], p[2] + 0.08], R, -1, tol=0.01, max_steps=60, umax=0.5)
    _hold(robot, R, -1, 30)
    return log


def act_wait(robot, steps=60):
    """Hold the current pose with the hand open for `steps` native steps (lets the checker settle)."""
    R = _rv2m(robot.now().tcp[3:6])
    return _hold(robot, R, -1, int(steps))
