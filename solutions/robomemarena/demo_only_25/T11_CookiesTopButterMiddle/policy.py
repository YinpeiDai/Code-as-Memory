"""T11 Cookies->top drawer, Butter->middle drawer.  Code-as-Memory policy.

Plan (derived from 25 expert demonstrations, see policy.md / task_analysis.md):
  open top drawer -> cookies into top drawer -> close top drawer ->
  open middle drawer -> butter into middle drawer.
Perception: agentview depth + calibration -> world points.  The cabinet top
surface (z ~ 0.713) gives xmax (front edge) and ymin (handle side face).
Handles, pulls and drop points are fixed offsets from (xmax, ymin).
Cookies / butter are detected as points > 8 mm above the local table height
near prior locations.
"""

# ---------------------------------------------------------------- constants
CAB_DEFAULT = [0.265, 0.468, 0.713]       # mean of demo detections
COOKIE_PRIOR = [0.222, -0.329]
BUTTER_PRIOR = [0.112, -0.205]
COOKIE_OFF = [-0.0061, 0.0015]            # expert grasp - detected centroid
BUTTER_OFF = [-0.0027, 0.0030]
H1 = [-0.1285, -0.0179, 0.675]            # top handle: (xmax+dx, ymin+dy, z)
H2 = [-0.1275, -0.0160, 0.607]            # middle handle
PULL_DY = -0.258                          # pull target y offset from ymin
RETREAT_DY = -0.37
OPEN_HANDLE_DY = -0.174                   # handle y offset when drawer fully open
PRECLOSE_DY = -0.28
DROP1 = [-0.118, -0.107, 0.717]           # cookies release (top drawer)
DROP2 = [-0.112, -0.113, 0.720]           # butter release (middle drawer)
GRASP_Z = 0.434
ABOVE_Z = 0.63
LIFT_Z = 0.80
TRANSIT_Z = 0.87
GAP_EMPTY = 0.008                         # finger gap below this = nothing held


def _rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ori(name):
    if name == "side":
        return _rx(-math.pi / 2)   # gripper points +y (towards the drawers)
    return _rx(math.pi)            # gripper points down


def _aa2R(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        d = np.clip((np.diag(R) + 1.0) / 2.0, 0, None)
        ax = np.sqrt(d)
        i = int(np.argmax(ax))
        for j in range(3):
            if j != i and R[i, j] + R[j, i] < 0:
                ax[j] = -ax[j]
        return ax / max(np.linalg.norm(ax), 1e-9) * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * math.sin(th)) * th


def _rot_err(tcp, name):
    Rc = _aa2R(np.asarray(tcp)[3:6])
    return _R2aa(_ori(name) @ Rc.T)


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0] - f[1]))


def _world_points(frame):
    D = getattr(frame, "agentview_depth", None)
    K = getattr(frame, "agentview_K", None)
    E = getattr(frame, "agentview_extrinsic", None)
    if D is None or K is None or E is None:
        return None, None
    D = np.asarray(D)
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    z = D.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    c = np.stack([x, y, z], -1)
    P = (c - E[:, 3]) @ E[:, :3]
    return P, D > 0


def _cabinet(P, valid):
    """Cabinet top-surface extents -> [xmax, ymin, ztop, npts]."""
    if P is None:
        return list(CAB_DEFAULT) + [0]
    m = valid & (P[..., 2] > 0.69) & (P[..., 2] < 0.75) & (P[..., 1] > 0.2)
    m = m & (P[..., 0] > -0.3) & (P[..., 0] < 0.6)
    m[:, :150] = False
    pts = P[m]
    if len(pts) < 200:
        return list(CAB_DEFAULT) + [int(len(pts))]
    xmin, xmax = np.percentile(pts[:, 0], [1, 99])
    ymin, ymax = np.percentile(pts[:, 1], [1, 99])
    ztop = float(np.median(pts[:, 2]))
    wx, wy = xmax - xmin, ymax - ymin
    if not (0.18 < wx < 0.34 and 0.13 < wy < 0.25):
        return list(CAB_DEFAULT) + [-int(len(pts))]
    return [float(xmax), float(ymin), ztop, int(len(pts))]


def _detect_obj(P, valid, prior, rad=0.07, dz=0.008):
    """Centroid xy of points > dz above the local table near prior."""
    if P is None:
        return [float(prior[0]), float(prior[1])], 0
    d = np.sqrt((P[..., 0] - prior[0]) ** 2 + (P[..., 1] - prior[1]) ** 2)
    m = valid & (d < rad) & (P[..., 2] > 0.3) & (P[..., 2] < 0.6)
    if int(m.sum()) < 50:
        return [float(prior[0]), float(prior[1])], 0
    zt = float(np.median(P[m, 2]))
    o = m & (P[..., 2] > zt + dz) & (P[..., 2] < zt + 0.08)
    n = int(o.sum())
    if n < 15:
        return [float(prior[0]), float(prior[1])], n
    q = P[o]
    return [float(q[:, 0].mean()), float(q[:, 1].mean())], n


def _seg_move(p, o, g, tol=0.005, n=300, vmax=1.0, rtol=0.06):
    return {"p": [float(p[0]), float(p[1]), float(p[2])], "o": o, "g": g,
            "tol": tol, "n": n, "vmax": vmax, "rtol": rtol}


def _seg_grip(g, n=25):
    return {"p": None, "o": None, "g": g, "n": n}


def _handle(cab, h):
    zoff = min(0.02, max(-0.02, cab[2] - 0.713))
    return [cab[0] + h[0], cab[1] + h[1], h[2] + zoff]


def _drop(cab, d):
    zoff = min(0.02, max(-0.02, cab[2] - 0.713))
    return [cab[0] + d[0], cab[1] + d[1], d[2] + zoff]


def _slip_info(frames, y_start):
    """From frames of a pull act: tcp displacement (m) when the handle slipped.
    0.0 if the handle was not held at the start of the pull; 1.0 if never slipped."""
    if not frames:
        return 0.0, None
    if _gap(frames[0]) <= GAP_EMPTY:
        return 0.0, None
    for fr in frames:
        if _gap(fr) <= GAP_EMPTY:
            y = float(np.asarray(fr.tcp)[1])
            return float(y_start - y), y
    return 1.0, None


def _pull_segments(cab, h, gz):
    hp = _handle(cab, h)
    return [
        _seg_move([hp[0], cab[1] + PULL_DY, gz], "side", 1, tol=0.004, n=200),
        _seg_grip(-1, 20),
        _seg_move([hp[0], cab[1] + RETREAT_DY, gz], "side", -1, tol=0.01, n=150),
    ]


def _grasp_handle_segments(cab, h, y_handle=None):
    hp = _handle(cab, h)
    yh = hp[1] if y_handle is None else y_handle
    return [
        _seg_move([hp[0], yh - 0.11, hp[2]], "side", -1, tol=0.008, n=250),
        _seg_move([hp[0], yh, hp[2]], "side", -1, tol=0.003, n=150, vmax=0.5, rtol=0.04),
        _seg_grip(1, 25),
    ]


def _pick_segments(xy, cur):
    s = []
    if float(np.asarray(cur.tcp)[2]) < ABOVE_Z - 0.03:
        s.append(_seg_move([float(np.asarray(cur.tcp)[0]), float(np.asarray(cur.tcp)[1]), ABOVE_Z], None, -1, tol=0.01, n=100))
    s += [
        _seg_move([xy[0], xy[1], ABOVE_Z], "down", -1, tol=0.006, n=300, rtol=0.04),
        _seg_move([xy[0], xy[1], GRASP_Z], "down", -1, tol=0.003, n=200, vmax=0.5, rtol=0.04),
        _seg_grip(1, 25),
    ]
    return s


def _out(memory, stage, tries, note, act_args, tool="act_run"):
    mem = dict(memory)
    mem["stage"] = stage
    mem["tries"] = int(tries)
    old = mem.get("log") or ""
    mem["log"] = (old + " | " + note)[-1500:] if note else old
    return {"memory": mem, "act": {"tool": tool, "args": act_args}}


def perceive_decide(obs, goal, memory, last):
    """Stage machine for: open top drawer, cookies in, close it, open middle drawer, butter in."""
    mem = {k: memory.get(k) for k in ("stage", "cab", "cookie", "butter", "tries", "log")}
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    gap = _gap(cur)
    stage = mem["stage"] or "start"
    tries = mem["tries"] if isinstance(mem["tries"], int) else 0

    if mem["cab"] is None or mem["cookie"] is None or mem["butter"] is None:
        P, valid = _world_points(cur)
        cab = _cabinet(P, valid)
        ck, nck = _detect_obj(P, valid, COOKIE_PRIOR)
        bt, nbt = _detect_obj(P, valid, BUTTER_PRIOR)
        mem["cab"] = [round(cab[0], 4), round(cab[1], 4), round(cab[2], 4)]
        mem["cookie"] = [round(ck[0] + COOKIE_OFF[0], 4), round(ck[1] + COOKIE_OFF[1], 4)]
        mem["butter"] = [round(bt[0] + BUTTER_OFF[0], 4), round(bt[1] + BUTTER_OFF[1], 4)]
        mem["log"] = "perceived cab=%s n=%d cookie=%s n=%d butter=%s n=%d" % (
            mem["cab"], cab[3], mem["cookie"], nck, mem["butter"], nbt)
    cab = mem["cab"]
    ck = mem["cookie"]
    bt = mem["butter"]
    h1 = _handle(cab, H1)
    h2 = _handle(cab, H2)

    # ---------------- top drawer: grasp handle
    if stage == "start":
        return _out(mem, "top_grasp", 0, "approach top handle",
                    {"segments": _grasp_handle_segments(cab, H1)})
    if stage == "top_grasp":
        if GAP_EMPTY < gap < 0.04:
            return _out(mem, "top_pull", 0, "top handle held gap=%.4f" % gap,
                        {"segments": _pull_segments(cab, H1, h1[2] + 0.001)})
        if tries < 3:
            segs = [_seg_grip(-1, 15), _seg_move([h1[0], tcp[1] - 0.08, h1[2]], "side", -1, tol=0.01, n=100)]
            segs += _grasp_handle_segments(cab, H1)
            return _out(mem, "top_grasp", tries + 1, "top handle miss gap=%.4f retry" % gap, {"segments": segs})
        return _out(mem, "top_pull", 0, "top handle miss, pull anyway",
                    {"segments": _pull_segments(cab, H1, h1[2] + 0.001)})
    if stage == "top_pull":
        disp, yslip = _slip_info(obs.frames, h1[1])
        if disp < 0.15 and tries < 2:
            yh = h1[1] - max(0.0, disp - 0.025) if disp < 0.99 else h1[1]
            segs = _grasp_handle_segments(cab, H1, y_handle=yh)
            return _out(mem, "top_regrasp", tries + 1, "top pull short disp=%.3f regrasp y=%.3f" % (disp, yh),
                        {"segments": segs})
        return _out(mem, "cookie_pick", 0, "top drawer open disp=%.3f" % disp,
                    {"segments": _pick_segments(ck, cur)})
    if stage == "top_regrasp":
        segs = _pull_segments(cab, H1, h1[2] + 0.001)
        return _out(mem, "top_pull", tries, "re-pull top gap=%.4f" % gap, {"segments": segs})

    # ---------------- cookies
    if stage == "cookie_pick":
        if gap > GAP_EMPTY:
            return _out(mem, "cookie_lift", tries, "cookie grasped gap=%.4f" % gap,
                        {"segments": [_seg_move([tcp[0], tcp[1], LIFT_Z], "down", 1, tol=0.01, n=150)]})
        return _cookie_retry(mem, cur, tries, "cookie miss gap=%.4f" % gap)
    if stage == "cookie_relocate":
        P, valid = _world_points(cur)
        c2, n2 = _detect_obj(P, valid, [ck[0] - COOKIE_OFF[0], ck[1] - COOKIE_OFF[1]], rad=0.09)
        if n2 >= 15:
            mem["cookie"] = [round(c2[0] + COOKIE_OFF[0], 4), round(c2[1] + COOKIE_OFF[1], 4)]
        return _out(mem, "cookie_pick", tries, "cookie re-detected %s n=%d" % (mem["cookie"], n2),
                    {"segments": _pick_segments(mem["cookie"], cur)})
    if stage == "cookie_lift":
        if gap > GAP_EMPTY:
            d1 = _drop(cab, DROP1)
            segs = [
                _seg_move([d1[0], d1[1], TRANSIT_Z], "down", 1, tol=0.01, n=300),
                _seg_move(d1, "down", 1, tol=0.004, n=200, rtol=0.05),
                _seg_grip(-1, 20),
                _seg_move([d1[0], d1[1], TRANSIT_Z], "down", -1, tol=0.01, n=150),
            ]
            return _out(mem, "cookie_drop", 0, "cookie lifted gap=%.4f" % gap, {"segments": segs})
        return _cookie_retry(mem, cur, tries, "cookie lost on lift gap=%.4f" % gap)

    # ---------------- close top drawer
    if stage == "cookie_drop":
        segs = [
            _seg_move([h1[0], cab[1] + PRECLOSE_DY, TRANSIT_Z], "side", -1, tol=0.01, n=250),
            _seg_move([h1[0], cab[1] + PRECLOSE_DY, h1[2] + 0.002], "side", -1, tol=0.005, n=200),
            _seg_move([h1[0], cab[1] + OPEN_HANDLE_DY, h1[2] + 0.002], "side", -1, tol=0.003, n=150, vmax=0.5),
            _seg_grip(1, 25),
        ]
        return _out(mem, "top_close", 0, "cookie released; go close top", {"segments": segs})
    if stage == "top_close":
        segs = [
            _seg_move([h1[0], h1[1] + 0.004, h1[2]], "side", 1, tol=0.004, n=200),
            _seg_grip(-1, 20),
            _seg_move([h1[0], cab[1] - 0.14, h1[2] + 0.002], "side", -1, tol=0.01, n=120),
        ]
        return _out(mem, "top_closed", tries, "push top closed (gap=%.4f)" % gap, {"segments": segs})
    if stage == "top_closed":
        ymax = max(float(np.asarray(f.tcp)[1]) for f in obs.frames)
        if ymax < h1[1] - 0.012 and tries < 2:
            segs = [
                _seg_move([h1[0], ymax - 0.06, h1[2]], "side", -1, tol=0.006, n=120),
                _seg_move([h1[0], ymax - 0.005, h1[2]], "side", -1, tol=0.004, n=120, vmax=0.5),
                _seg_grip(1, 25),
            ]
            return _out(mem, "top_close", tries + 1, "top not closed ymax=%.3f retry" % ymax, {"segments": segs})
        return _out(mem, "mid_grasp", 0, "top closed ymax=%.3f; approach middle" % ymax,
                    {"segments": _grasp_handle_segments(cab, H2)})

    # ---------------- middle drawer
    if stage == "mid_grasp":
        if GAP_EMPTY < gap < 0.04:
            return _out(mem, "mid_pull", 0, "middle handle held gap=%.4f" % gap,
                        {"segments": _pull_segments(cab, H2, h2[2] + 0.001)})
        if tries < 3:
            segs = [_seg_grip(-1, 15), _seg_move([h2[0], tcp[1] - 0.08, h2[2]], "side", -1, tol=0.01, n=100)]
            segs += _grasp_handle_segments(cab, H2)
            return _out(mem, "mid_grasp", tries + 1, "middle handle miss gap=%.4f retry" % gap, {"segments": segs})
        return _out(mem, "mid_pull", 0, "middle handle miss, pull anyway",
                    {"segments": _pull_segments(cab, H2, h2[2] + 0.001)})
    if stage == "mid_pull":
        disp, yslip = _slip_info(obs.frames, h2[1])
        if disp < 0.15 and tries < 2:
            yh = h2[1] - max(0.0, disp - 0.025) if disp < 0.99 else h2[1]
            return _out(mem, "mid_regrasp", tries + 1, "middle pull short disp=%.3f regrasp y=%.3f" % (disp, yh),
                        {"segments": _grasp_handle_segments(cab, H2, y_handle=yh)})
        return _out(mem, "butter_pick", 0, "middle drawer open disp=%.3f" % disp,
                    {"segments": _pick_segments(bt, cur)})
    if stage == "mid_regrasp":
        return _out(mem, "mid_pull", tries, "re-pull middle gap=%.4f" % gap,
                    {"segments": _pull_segments(cab, H2, h2[2] + 0.001)})

    # ---------------- butter
    if stage == "butter_pick":
        if gap > GAP_EMPTY:
            return _out(mem, "butter_lift", tries, "butter grasped gap=%.4f" % gap,
                        {"segments": [_seg_move([tcp[0], tcp[1], LIFT_Z], "down", 1, tol=0.01, n=150)]})
        return _butter_retry(mem, cur, tries, "butter miss gap=%.4f" % gap)
    if stage == "butter_relocate":
        P, valid = _world_points(cur)
        b2, n2 = _detect_obj(P, valid, [bt[0] - BUTTER_OFF[0], bt[1] - BUTTER_OFF[1]], rad=0.09)
        if n2 >= 15:
            mem["butter"] = [round(b2[0] + BUTTER_OFF[0], 4), round(b2[1] + BUTTER_OFF[1], 4)]
        return _out(mem, "butter_pick", tries, "butter re-detected %s n=%d" % (mem["butter"], n2),
                    {"segments": _pick_segments(mem["butter"], cur)})
    if stage == "butter_lift":
        if gap > GAP_EMPTY:
            d2 = _drop(cab, DROP2)
            segs = [
                _seg_move([d2[0], d2[1], TRANSIT_Z], "down", 1, tol=0.01, n=300),
                _seg_move([d2[0], d2[1], 0.76], "down", 1, tol=0.005, n=150, rtol=0.05),
                _seg_move(d2, "down", 1, tol=0.004, n=120, rtol=0.05),
                _seg_grip(-1, 20),
                _seg_move([d2[0], d2[1], TRANSIT_Z], "down", -1, tol=0.01, n=150),
            ]
            return _out(mem, "done", 0, "butter lifted gap=%.4f; place" % gap, {"segments": segs})
        return _butter_retry(mem, cur, tries, "butter lost on lift gap=%.4f" % gap)

    # ---------------- finished: hold still while the checker monitors
    return _out(mem, "done", tries + 1, "", {"segments": [_seg_grip(-1, 60)]})


def _cookie_retry(mem, cur, tries, note):
    if tries >= 3:
        # give up on re-grasping; continue with the rest of the task
        return _out(mem, "cookie_drop", 0, note + " giving up", {"segments": [_seg_grip(-1, 10)]})
    t = np.asarray(cur.tcp, dtype=float)
    segs = [_seg_grip(-1, 15),
            _seg_move([t[0], t[1], ABOVE_Z], "down", -1, tol=0.01, n=100),
            _seg_move([t[0] - 0.22, t[1], 0.72], "down", -1, tol=0.02, n=150)]
    return _out(mem, "cookie_relocate", tries + 1, note + " retry", {"segments": segs})


def _butter_retry(mem, cur, tries, note):
    if tries >= 3:
        return _out(mem, "done", 0, note + " giving up", {"segments": [_seg_grip(-1, 10)]})
    t = np.asarray(cur.tcp, dtype=float)
    segs = [_seg_grip(-1, 15),
            _seg_move([t[0], t[1], ABOVE_Z], "down", -1, tol=0.01, n=100),
            _seg_move([t[0] - 0.22, t[1], 0.72], "down", -1, tol=0.02, n=150)]
    return _out(mem, "butter_relocate", tries + 1, note + " retry", {"segments": segs})


def act_run(robot, segments):
    """Execute segments in order.  A segment is either a move
    {"p":[x,y,z], "o":"down"|"side"|None, "g":+1 close/-1 open, "tol":m, "n":max steps,
     "vmax":<=1, "rtol":rad} (P-control of the OSC delta target until within tol / rtol,
    or stalled, or n steps) or a gripper wait {"p":None, "g":+-1, "n":steps}.
    Returns per-segment step counts, final errors, finger gap and tcp."""
    report = []
    for seg in segments:
        g = float(seg.get("g") or 0)
        n = int(seg.get("n") or 1)
        p = seg.get("p")
        if p is None:
            for _ in range(n):
                robot.step(0, 0, 0, 0, 0, 0, g)
            fr = robot.now()
            report.append({"grip": g, "steps": n, "gap": round(_gap(fr), 4)})
            continue
        o = seg.get("o")
        tol = float(seg.get("tol") or 0.005)
        rtol = float(seg.get("rtol") or 0.06)
        vmax = float(seg.get("vmax") or 1.0)
        target = np.asarray(p, dtype=float)
        steps, ok, stall = 0, 0, 0
        best = 1e9
        perr, rerr = 1.0, 0.0
        for i in range(n):
            fr = robot.now()
            t = np.asarray(fr.tcp, dtype=float)
            e = target - t[:3]
            perr = float(np.linalg.norm(e))
            if o is not None:
                re = _rot_err(t, o)
                rerr = float(np.linalg.norm(re))
            else:
                re = np.zeros(3)
                rerr = 0.0
            if perr < tol and rerr < rtol:
                ok += 1
                if ok >= 2:
                    break
            else:
                ok = 0
            metric = perr + 0.05 * rerr
            if metric < best - 0.0005:
                best = metric
                stall = 0
            else:
                stall += 1
                if stall > 40:
                    break
            a = np.clip(e * 12.0, -vmax, vmax)
            r = np.clip(re * 2.0, -0.5, 0.5)
            robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
            steps += 1
        fr = robot.now()
        report.append({"to": [round(float(v), 3) for v in target], "steps": steps,
                       "err": round(perr, 4), "rerr": round(rerr, 3), "gap": round(_gap(fr), 4)})
    fr = robot.now()
    return {"segments": report, "tcp": [round(float(v), 4) for v in np.asarray(fr.tcp)[:3]],
            "gap": round(_gap(fr), 4)}
