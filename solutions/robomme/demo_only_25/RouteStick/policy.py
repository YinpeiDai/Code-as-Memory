"""RouteStick policy: reproduce the route shown in the task video.

The instruction video (phase 'demo' frames) shows the stick robot weaving around
sticks on the table. Execution in every recorded episode retraces exactly the
same TCP route from the same start pose, at constant height z~0.07 m.  The
policy therefore extracts the demonstrated TCP route from the video frames and
tracks it with a pure-pursuit controller using measured TCP feedback.

Route sources, in order of preference:
  1. tcp of demo frames (native TCP, same frame as execution);
  2. wrist_extrinsic of demo frames (TCP is fixed in the wrist-camera frame);
  3. front depth + calibration: the lowest moving 3D point in the work volume.
"""

TCP_IN_WRIST_CAM = [0.02, 0.0465, 0.114]   # measured, constant over all 25 episodes
FRONT_TIP_BIAS = [0.0045, -0.003]          # mean (estimate - tcp) of depth tracker, metres
WORK_X = [-0.40, 0.20]                      # table work volume for the depth tracker
WORK_Y = 0.40
MIN_EXTENT = 0.02                           # a route must span at least 2 cm
SPACING = 0.004                             # resampled route spacing, metres
LOOKAHEAD = 0.009                           # pure-pursuit lookahead, metres
CORNER_DEG = 40.0                           # turn sharper than this is a waypoint to reach
CORNER_TOL = 0.003                          # must pass within this of each corner
MAX_LEAD = 0.015                            # commanded target at most this far from tcp


def _as_xyz(v):
    a = np.asarray(v, dtype=float).reshape(-1)
    return a[:3]


def _extent(P):
    if len(P) < 2:
        return 0.0
    P = np.asarray(P, dtype=float)
    return float(np.max(np.linalg.norm(P[:, :2] - P[0, :2], axis=1)))


def _route_from_tcp(frames):
    out = []
    for f in frames:
        tcp = getattr(f, "tcp", None)
        if tcp is None:
            continue
        a = np.asarray(tcp, dtype=float).reshape(-1)
        if a.size >= 3 and np.all(np.isfinite(a[:3])):
            out.append(a[:3])
    return out


def _route_from_wrist(frames):
    out = []
    c = np.array(TCP_IN_WRIST_CAM)
    for f in frames:
        E = getattr(f, "wrist_extrinsic", None)
        if E is None:
            continue
        E = np.asarray(E, dtype=float)
        if E.shape[0] < 3 or E.shape[1] < 4 or not np.all(np.isfinite(E)):
            continue
        R = E[:3, :3]
        t = E[:3, 3]
        out.append(R.T @ (c - t))
    return out


def _cloud(D, K, E):
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    z = D.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    C = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:3, :3]
    t = E[:3, 3]
    return ((C - t) @ R).reshape(H, W, 3)


def _route_from_front_depth(frames):
    fr = [f for f in frames if getattr(f, "front_depth", None) is not None]
    if len(fr) < 5:
        return []
    K = np.asarray(fr[0].front_K, dtype=float)
    E = np.asarray(fr[0].front_extrinsic, dtype=float)
    Ds = np.array([np.asarray(f.front_depth, dtype=float) for f in fr])
    med = np.median(Ds, 0)
    out = []
    for D in Ds:
        P = _cloud(D, K, E)
        m = ((np.abs(D - med) > 15) & (D > 0) & (P[..., 2] > 0.03) & (P[..., 2] < 0.15)
             & (P[..., 0] > WORK_X[0]) & (P[..., 0] < WORK_X[1]) & (np.abs(P[..., 1]) < WORK_Y))
        if m.sum() < 3:
            continue
        zmin = np.percentile(P[..., 2][m], 2)
        sel = m & (P[..., 2] < zmin + 0.012)
        xy = P[sel][:, :2].mean(0) - np.array(FRONT_TIP_BIAS)
        out.append(np.array([xy[0], xy[1], 0.07]))
    if len(out) >= 5:  # light median smoothing of the noisy visual track
        A = np.array(out)
        S = A.copy()
        for i in range(len(A)):
            S[i, :2] = np.median(A[max(0, i - 2):i + 3, :2], 0)
        out = list(S)
    return out


def _resample(P):
    """Drop stationary duplicates and resample the polyline at SPACING."""
    P = [np.asarray(p, dtype=float) for p in P]
    if not P:
        return []
    clean = [P[0]]
    for p in P[1:]:
        if np.linalg.norm(p[:2] - clean[-1][:2]) > 0.0008:
            clean.append(p)
    if len(clean) < 2:
        return [clean[0].tolist()]
    A = np.array(clean)
    seg = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / SPACING)) + 1)
    q = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(q, s, A[:, k]) for k in range(3)], 1)
    return out.tolist()


def perceive_route(obs):
    """Extract the demonstrated TCP route from instruction-video frames.

    Returns {"route": [[x,y,z],...] resampled at 4 mm, "source": label, "n_demo": count}.
    """
    demo = [f for f in obs.frames if getattr(f, "phase", "") == "demo"]
    if not demo:
        return {"route": [], "source": "none", "n_demo": 0}
    for name, fn in (("tcp", _route_from_tcp), ("wrist_extrinsic", _route_from_wrist),
                     ("front_depth", _route_from_front_depth)):
        P = fn(demo)
        if len(P) >= 2 and _extent(P) >= MIN_EXTENT:
            Z = np.array(P)
            if not (np.all(Z[:, 2] > 0.02) and np.all(Z[:, 2] < 0.3)):
                continue
            return {"route": _resample(P), "source": name, "n_demo": len(demo)}
    return {"route": [], "source": "none", "n_demo": len(demo)}


def _parse_last(last):
    """(progress, finished) from the previous act outcome (dict or JSON text)."""
    if isinstance(last, dict):
        p = last.get("progress")
        return (int(p) if isinstance(p, (int, float)) else None), bool(last.get("finished"))
    if not isinstance(last, str) or '"progress"' not in last:
        return None, False
    i = last.find('"progress"') + len('"progress"')
    rest = last[i:].lstrip(" :")
    num = ""
    for ch in rest:
        if ch.isdigit():
            num += ch
        else:
            break
    fin = ('"finished": true' in last) or ('"finished":true' in last)
    return (int(num) if num else None), fin


def perceive_decide(obs, goal, memory, last):
    """Extract the video route once, then follow it from the stored progress, then hold.

    Memory: route (positions), source (label), progress (count index into route),
    follow_calls (count), done (boolean).
    """
    mem = dict(memory)
    for k in ("route", "source", "progress", "follow_calls", "done"):
        mem.setdefault(k, None)
    if mem["route"] is None:
        r = perceive_route(obs)
        mem["route"] = r["route"]
        mem["source"] = r["source"]
        mem["progress"] = 0
        mem["follow_calls"] = 0
        mem["done"] = False
    prog, fin = _parse_last(last)
    if prog is not None:
        mem["progress"] = prog
    if fin:
        mem["done"] = True
    route = mem["route"] or []
    calls = int(mem["follow_calls"] or 0)
    if len(route) >= 2 and not mem["done"] and calls < 4:
        mem["follow_calls"] = calls + 1
        return {"memory": mem, "act": {"tool": "act_follow_route",
                                       "args": {"route": route, "start": int(mem["progress"] or 0)}}}
    mem["done"] = True
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 40}}}


def _cmd(robot, p, rpy):
    robot.step(float(p[0]), float(p[1]), float(p[2]), float(rpy[0]), float(rpy[1]), float(rpy[2]), -1.0)
    return np.asarray(robot.now().tcp, dtype=float).reshape(-1)


def _corners(R):
    """Indices of sharp turns (> CORNER_DEG over +-2 points) in a resampled route."""
    N = len(R)
    out = []
    for i in range(2, N - 2):
        a = R[i, :2] - R[i - 2, :2]
        b = R[i + 2, :2] - R[i, :2]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        c = float(np.dot(a, b) / (na * nb))
        if c < math.cos(math.radians(CORNER_DEG)):
            out.append(i)
    # keep the sharpest index of each cluster
    merged = []
    for i in out:
        if merged and i - merged[-1][-1] <= 3:
            merged[-1].append(i)
        else:
            merged.append([i])
    res = []
    for grp in merged:
        best, bc = grp[0], 2.0
        for i in grp:
            a = R[i, :2] - R[i - 2, :2]
            b = R[i + 2, :2] - R[i, :2]
            c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if c < bc:
                best, bc = i, c
        res.append(best)
    return res


def act_follow_route(robot, route, start=0):
    """Track route ([[x,y,z],...]) from index `start` with pure pursuit and TCP feedback.

    Keeps the initial orientation. Approaches the first route point gradually if needed.
    Returns {"progress": index reached, "finished": bool, "steps": n, "final_err": m, "error": text}.
    """
    R = np.asarray(route, dtype=float)
    N = len(R)
    tcp = np.asarray(robot.now().tcp, dtype=float).reshape(-1)
    rpy = tcp[3:6].copy()
    seg = np.linalg.norm(np.diff(R[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    j = max(0, min(int(start), N - 1))
    steps = 0
    err = ""
    # gradual approach to the start/resume point
    while steps < 200 and np.linalg.norm(tcp[:3] - R[j]) > 0.006:
        d = R[j] - tcp[:3]
        L = np.linalg.norm(d)
        tgt = tcp[:3] + d * min(1.0, MAX_LEAD / max(L, 1e-9))
        try:
            tcp = _cmd(robot, tgt, rpy)
        except Exception as e:
            return {"progress": j, "finished": False, "steps": steps, "final_err": -1, "error": str(e)[:200]}
        steps += 1
    corners = [c for c in _corners(R) if c > j] + [N - 1]
    stall = 0
    while steps < 1400:
        while len(corners) > 1 and (j > corners[0] or (j >= corners[0] - 1 and np.linalg.norm(tcp[:2] - R[corners[0], :2]) < CORNER_TOL)):
            corners = corners[1:]
        lo, hi = j, min(N, j + 12, corners[0] + 1)
        dist = np.linalg.norm(R[lo:hi, :2] - tcp[:2], axis=1)
        jn = lo + int(np.argmin(dist))
        stall = stall + 1 if jn == j else 0
        j = jn
        if stall > 6 and j < corners[0]:   # not progressing: push the index forward
            j += 1
            stall = 0
        if j >= N - 1 and np.linalg.norm(tcp[:2] - R[-1, :2]) < 0.003:
            break
        k = j
        while k < corners[0] and s[k] - s[j] < LOOKAHEAD:
            k += 1
        tgt = R[k].copy()
        d = tgt - tcp[:3]
        L = np.linalg.norm(d)
        if L > MAX_LEAD:
            tgt = tcp[:3] + d * (MAX_LEAD / L)
        try:
            tcp = _cmd(robot, tgt, rpy)
        except Exception as e:
            err = str(e)[:200]
            break
        steps += 1
        if j >= N - 1 and stall > 20:
            break
        if j >= corners[0] and len(corners) > 1 and stall > 20:
            corners = corners[1:]
            stall = 0
    fe = float(np.linalg.norm(tcp[:2] - R[-1, :2]))
    return {"progress": int(j), "finished": bool(j >= N - 1 and fe < 0.01 and not err),
            "steps": steps, "final_err": fe, "error": err}


def act_hold(robot, steps=40):
    """Hold the current pose for `steps` control steps (after the route is done)."""
    tcp = np.asarray(robot.now().tcp, dtype=float).reshape(-1)
    n = 0
    for _ in range(int(steps)):
        _cmd(robot, tcp[:3], tcp[3:6])
        n += 1
    return {"held": n}
