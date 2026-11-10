"""T18 ChocolateButterCabinet: move chocolate then butter from cabinet1 top to cabinet2 top.

Perception: agentview depth -> world cloud -> two dark cabinet tops (cabinet1 = the one
carrying objects); wrist RGB-D above a cabinet -> object segments (depth above the top)
labelled chocolate (dark maroon) or butter (bright red/orange/yellow).
Control: closed-loop OSC delta servo on measured TCP, gripper sign +1 close / -1 open.
"""

R0 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]   # top-down gripper, yaw 0
HOVER_Z = 0.82          # TCP height for wrist scans (camera ~0.10 m higher)
TRAVEL_Z = 0.88         # TCP height for transport
PLACE_DY = {"chocolate": 0.05, "butter": -0.05}   # slot on destination top (world y)
ORDER = ["chocolate", "butter"]


# ---------------------------------------------------------------- geometry helpers
def _aa2R(a):
    a = np.asarray(a, float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        v = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return v * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _Rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _target_R(yaw):
    return _Rz(yaw) @ np.array(R0)


def _cloud(depth, K, E):
    d = np.asarray(depth).astype(float) / 1000.0
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    Pw = (P - t) @ R
    Pw[d <= 0] = np.nan
    return Pw


def _cam_offset_local(frame):
    """Wrist camera centre relative to the TCP, in the gripper frame."""
    E = np.asarray(frame.wrist_extrinsic)
    c = -E[:, :3].T @ E[:, 3]
    tcp = np.asarray(frame.tcp)
    return _aa2R(tcp[3:]).T @ (c - tcp[:3])


# ---------------------------------------------------------------- perception
def _find_cabinets(frame):
    """Dark horizontal cabinet tops in the agentview cloud; returns (ztop, list)."""
    Pw = _cloud(frame.agentview_depth, np.asarray(frame.agentview_K), frame.agentview_extrinsic)
    X, Y, Z = Pw[..., 0], Pw[..., 1], Pw[..., 2]
    ok = ~np.isnan(Z)
    Zs = np.where(ok, Z, -1.0)
    Xs = np.nan_to_num(X)
    Ys = np.nan_to_num(Y)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.max(-1) < 90
    cand = ok & (Zs > 0.55) & (Zs < 0.80) & dark & (np.abs(Xs) < 0.35)
    zz = Zs[cand]
    if zz.size < 100:
        return None, []
    h, b = np.histogram(zz, bins=50, range=(0.55, 0.80))
    k = int(np.argmax(h))
    sel = zz[(zz >= b[k] - 0.005) & (zz < b[k + 1] + 0.005)]
    ztop = float(np.median(sel))
    top = cand & (np.abs(Zs - ztop) < 0.006)
    ys = Ys[top]
    xs = Xs[top]
    bins = np.arange(-0.7, 0.7101, 0.01)
    hy, _ = np.histogram(ys, bins=bins)
    runs = []
    cur = None
    for i in range(len(hy)):
        if hy[i] >= 3:
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        else:
            if cur is not None:
                runs.append(cur)
                cur = None
    if cur is not None:
        runs.append(cur)
    cabs = []
    for r in runs:
        lo, hi = bins[r[0]], bins[r[1] + 1]
        s = (ys >= lo) & (ys < hi)
        if s.sum() < 60:
            continue
        ylo, yhi = np.percentile(ys[s], [1, 99])
        xlo, xhi = np.percentile(xs[s], [1, 99])
        if yhi - ylo < 0.08:
            continue
        cx, cy = float((xlo + xhi) / 2), float((ylo + yhi) / 2)
        ex, ey = float(xhi - xlo), float(yhi - ylo)
        m = ok & (Zs > ztop + 0.01) & (Zs < ztop + 0.10) & (np.abs(Xs - cx) < ex / 2 - 0.005) & (np.abs(Ys - cy) < ey / 2 - 0.005)
        cabs.append({"c": [cx, cy, ztop], "ext": [ex, ey], "n": int(s.sum()), "nobj": int(m.sum())})
    return ztop, cabs


def _components(mask, min_size):
    H, W = mask.shape
    lab = -np.ones((H, W), int)
    comps = []
    vs, us = np.nonzero(mask)
    for idx in range(len(vs)):
        v0, u0 = int(vs[idx]), int(us[idx])
        if lab[v0, u0] >= 0:
            continue
        k = len(comps)
        stack = [(v0, u0)]
        lab[v0, u0] = k
        pts = []
        while stack:
            v, u = stack.pop()
            pts.append((v, u))
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = v + dv, u + du
                if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] < 0:
                    lab[a, b] = k
                    stack.append((a, b))
        comps.append(np.array(pts))
    return [c for c in comps if len(c) >= min_size]


def _segment_stats(P, Zv, label, nb, nc):
    lo = np.percentile(P, 3, 0)
    hi = np.percentile(P, 97, 0)
    ctr = (lo + hi) / 2
    Q = P - P.mean(0)
    w, v = np.linalg.eigh(Q.T @ Q)
    ax = v[:, 1]
    ang = math.atan2(float(ax[1]), float(ax[0]))
    if ang > math.pi / 2:
        ang -= math.pi
    if ang < -math.pi / 2:
        ang += math.pi
    elong = math.sqrt(max(w[1], 1e-12) / max(w[0], 1e-12))
    return {"label": label, "x": float(ctr[0]), "y": float(ctr[1]), "ztop": float(np.percentile(Zv, 90)),
            "ext": [float(hi[0] - lo[0]), float(hi[1] - lo[1])], "n": int(len(P)),
            "yaw": float(ang), "elong": float(elong), "nb": int(nb), "nc": int(nc)}


def _detect_objects(frame, cab, ztop):
    """Objects on one cabinet top from the wrist RGB-D frame."""
    Pw = _cloud(frame.wrist_depth, np.asarray(frame.wrist_K), frame.wrist_extrinsic)
    X, Y, Z = Pw[..., 0], Pw[..., 1], Pw[..., 2]
    ok = ~np.isnan(Z)
    Zs = np.where(ok, Z, -1.0)
    Xs = np.nan_to_num(X)
    Ys = np.nan_to_num(Y)
    rgb = np.asarray(frame.wrist_rgb).astype(int)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cx, cy = cab[0], cab[1]
    obj = ok & (Zs > ztop + 0.008) & (Zs < ztop + 0.12) & (np.abs(Xs - cx) < 0.14) & (np.abs(Ys - cy) < 0.11)
    bpx = obj & (R >= 115) & (B < 45) & (R - G >= 25)
    cpx = obj & (R >= 30) & (R < 110) & (R - G >= 18) & (B <= G + 12)
    segs = []
    for comp in _components(obj, 40):
        vv, uu = comp[:, 0], comp[:, 1]
        P = np.stack([Xs[vv, uu], Ys[vv, uu]], 1)
        Zv = Zs[vv, uu]
        cb = bpx[vv, uu]
        cc = cpx[vv, uu]
        nb, nc = int(cb.sum()), int(cc.sum())
        span = float(np.max(np.percentile(P, 97, 0) - np.percentile(P, 3, 0)))
        if nb >= 15 and nc >= 30 and (span > 0.09 or len(P) > 1300):
            sb = np.median(P[cb], 0)
            sc = np.median(P[cc], 0)
            lb = None
            for it in range(10):
                lb = np.linalg.norm(P - sb, axis=1) < np.linalg.norm(P - sc, axis=1)
                if lb.sum() < 5 or (~lb).sum() < 5:
                    break
                sb = np.median(P[lb], 0)
                sc = np.median(P[~lb], 0)
            if lb is not None and lb.sum() >= 20 and (~lb).sum() >= 20:
                segs.append(_segment_stats(P[lb], Zv[lb], "butter", cb[lb].sum(), cc[lb].sum()))
                segs.append(_segment_stats(P[~lb], Zv[~lb], "chocolate", cb[~lb].sum(), cc[~lb].sum()))
                continue
        if nb + nc < 10:
            label = "unknown"
        else:
            label = "butter" if nb > nc else "chocolate"
        segs.append(_segment_stats(P, Zv, label, nb, nc))
    return segs


def _pick_seg(segs, label):
    best = None
    for s in segs:
        if s["label"] == label and (best is None or s["n"] > best["n"]):
            best = s
    return best


# ---------------------------------------------------------------- control helpers
def _servo(robot, target, grip, yaw=0.0, tol=0.004, max_steps=300, vmax=1.0, settle=2, gain=0.015):
    """Closed-loop move of the measured TCP to target xyz (world) with top-down yaw."""
    target = np.asarray(target, float)
    Rt = _target_R(yaw)
    best = 1e9
    stall = 0
    ok_count = 0
    n = 0
    for n in range(1, max_steps + 1):
        f = robot.now()
        tcp = np.asarray(f.tcp)
        e = target - tcp[:3]
        err = float(np.linalg.norm(e))
        rerr = _R2aa(Rt @ _aa2R(tcp[3:]).T)
        if err < tol and float(np.linalg.norm(rerr)) < 0.05:
            ok_count += 1
            if ok_count >= settle:
                break
        else:
            ok_count = 0
        if err < best - 0.0005:
            best = err
            stall = 0
        else:
            stall += 1
        if stall > 25:
            break
        a = e / gain
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        r = np.clip(rerr * 5.0, -1, 1)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    tcp = np.asarray(robot.now().tcp)
    return {"reached": bool(np.linalg.norm(target - tcp[:3]) < max(tol * 2, 0.008)),
            "err": float(np.linalg.norm(target - tcp[:3])), "steps": n, "tcp": tcp[:3].tolist()}


def _hold(robot, grip, steps):
    """Hold the current target pose with a gripper command for some steps."""
    for i in range(steps):
        robot.step(0, 0, 0, 0, 0, 0, grip)


def _finger_gap(frame):
    f = np.asarray(frame.fingers)
    return float(f[0] - f[1])


def _to_height(robot, z, grip, yaw=0.0):
    tcp = np.asarray(robot.now().tcp)
    return _servo(robot, [tcp[0], tcp[1], z], grip, yaw=yaw, tol=0.01, max_steps=150, settle=1)


def _view_tcp(frame, x, y, z):
    """TCP position that puts the wrist camera above (x, y)."""
    off = np.array(R0) @ _cam_offset_local(frame)
    return [x - float(off[0]), y - float(off[1]), z]


# ---------------------------------------------------------------- policy
def _fields():
    return ["stage", "target", "done", "cab_src", "cab_dst", "obj", "fails", "scans", "notes"]


def perceive_decide(obs, goal, memory, last):
    """Decide the next act from the latest frame and the declared memory."""
    mem = {k: memory.get(k) for k in _fields()}
    if mem["done"] is None:
        mem["done"] = []
    if mem["fails"] is None:
        mem["fails"] = 0
    if mem["scans"] is None:
        mem["scans"] = 0
    if mem["stage"] is None:
        mem["stage"] = "locate"
    f = obs.current
    note = ""

    def out(tool, args):
        mem["notes"] = (note + " | " + str(last)[:200]) if last else note
        print("DECIDE step=%s stage=%s target=%s done=%s -> %s %s :: %s" % (f.step, mem["stage"], mem["target"], mem["done"], tool, args, note))
        print("LAST", str(last)[:1500])
        return {"memory": mem, "act": {"tool": tool, "args": args}}

    remaining = [o for o in ORDER if o not in mem["done"]]
    if not remaining:
        mem["stage"] = "finished"
        note = "both objects verified on destination; waiting"
        if mem["cab_dst"] is not None:
            return out("act_view", {"x": mem["cab_dst"][0], "y": mem["cab_dst"][1], "wait": 40})
        return out("act_wait", {"steps": 40})
    if mem["target"] not in remaining:
        mem["target"] = remaining[0]
        mem["fails"] = 0
        mem["scans"] = 0

    # 1) cabinets
    if mem["cab_src"] is None or mem["cab_dst"] is None:
        ztop, cabs = _find_cabinets(f)
        if len(cabs) >= 2:
            cabs = sorted(cabs, key=lambda c: -c["n"])[:2]
            cabs = sorted(cabs, key=lambda c: (-c["nobj"], c["c"][1]))
            if cabs[0]["nobj"] < 20:
                cabs = sorted(cabs, key=lambda c: c["c"][1])   # assumption: cabinet1 at lower y
            mem["cab_src"] = [round(v, 4) for v in cabs[0]["c"]]
            mem["cab_dst"] = [round(v, 4) for v in cabs[1]["c"]]
            mem["stage"] = "scan"
            note = "cabinets src %s dst %s nobj %d/%d" % (mem["cab_src"], mem["cab_dst"], cabs[0]["nobj"], cabs[1]["nobj"])
            return out("act_view", {"x": mem["cab_src"][0], "y": mem["cab_src"][1], "wait": 8})
        mem["scans"] += 1
        note = "cabinets not found (%d candidates)" % len(cabs)
        return out("act_move", {"x": 0.0, "y": 0.0, "z": 1.0, "grip": -1.0})

    src, dst = mem["cab_src"], mem["cab_dst"]
    ztop = src[2]
    tgt = mem["target"]

    # 2) holding something? -> place
    gap = _finger_gap(f)
    tcp = np.asarray(f.tcp)
    if mem["stage"] == "picked":
        if 0.012 < gap < 0.072:
            mem["stage"] = "placing"
            note = "holding %s gap %.3f; placing" % (tgt, gap)
            return out("act_place", {"x": dst[0], "y": dst[1] + PLACE_DY[tgt], "zsurf": dst[2],
                                     "view_x": dst[0], "view_y": dst[1]})
        mem["fails"] += 1
        mem["stage"] = "scan"
        note = "grasp of %s failed (gap %.3f); rescanning source" % (tgt, gap)
        return out("act_view", {"x": src[0], "y": src[1], "wait": 8})

    if mem["stage"] == "placing":
        # act_place ends with the wrist camera above the destination: verify
        segs = _detect_objects(f, dst, dst[2])
        seg = _pick_seg(segs, tgt)
        if seg is not None:
            mem["done"] = mem["done"] + [tgt]
            note = "%s verified on destination at (%.3f, %.3f)" % (tgt, seg["x"], seg["y"])
            rem = [o for o in ORDER if o not in mem["done"]]
            if rem:
                mem["target"] = rem[0]
                mem["fails"] = 0
                mem["scans"] = 0
                mem["stage"] = "scan"
                return out("act_view", {"x": src[0], "y": src[1], "wait": 8})
            mem["stage"] = "finished"
            return out("act_view", {"x": dst[0], "y": dst[1], "wait": 40})
        mem["stage"] = "scan"
        mem["fails"] += 1
        note = "%s not seen on destination after place (segs %s); rescanning source" % (tgt, [s["label"] for s in segs])
        return out("act_view", {"x": src[0], "y": src[1], "wait": 8})

    # 3) scan source for the target
    if mem["stage"] == "scan":
        segs = _detect_objects(f, src, ztop)
        seg = _pick_seg(segs, tgt)
        cam_ok = abs(float(tcp[2]) - HOVER_Z) < 0.05
        if seg is None and not cam_ok:
            note = "not at scan pose; moving to view source"
            return out("act_view", {"x": src[0], "y": src[1], "wait": 8})
        if seg is None:
            mem["scans"] += 1
            if mem["scans"] >= 3:
                # check destination, maybe it is already there
                mem["stage"] = "check_dst"
                note = "%s not found on source after %d scans" % (tgt, mem["scans"])
                return out("act_view", {"x": dst[0], "y": dst[1], "wait": 8})
            note = "%s not found on source (segs %s); rescan offset" % (tgt, [s["label"] for s in segs])
            off = 0.04 * (1 if mem["scans"] % 2 else -1)
            return out("act_view", {"x": src[0] + off, "y": src[1], "wait": 8})
        if mem["fails"] >= 4:
            note = "too many failures for %s" % tgt
        yaw = 0.0
        if seg["elong"] > 1.3 and seg["n"] > 150:
            yaw = max(-0.45, min(0.45, seg["yaw"]))
            if abs(yaw) < 0.12:
                yaw = 0.0
        # small deterministic perturbation on retries
        dx = [0.0, 0.0, 0.01, -0.01, 0.0][min(mem["fails"], 4)]
        mem["obj"] = [round(seg["x"], 4), round(seg["y"], 4), round(seg["ztop"], 4)]
        mem["stage"] = "picked"
        note = "%s at (%.3f,%.3f) top %.3f yaw %.2f n %d; picking" % (tgt, seg["x"], seg["y"], seg["ztop"], yaw, seg["n"])
        return out("act_pick", {"x": seg["x"] + dx, "y": seg["y"], "zsurf": ztop, "yaw": yaw})

    if mem["stage"] == "check_dst":
        segs = _detect_objects(f, dst, dst[2])
        seg = _pick_seg(segs, tgt)
        if seg is not None:
            mem["done"] = mem["done"] + [tgt]
            note = "%s found already on destination" % tgt
        else:
            note = "%s not on destination either; rescanning source" % tgt
        mem["scans"] = 0
        mem["stage"] = "scan"
        return out("act_view", {"x": src[0], "y": src[1], "wait": 8})

    mem["stage"] = "scan"
    note = "unknown stage; rescanning"
    return out("act_view", {"x": src[0], "y": src[1], "wait": 8})


# ---------------------------------------------------------------- acts
def act_view(robot, x, y, wait=8):
    """Raise to scan height and put the wrist camera above world (x, y); gripper opened unless holding."""
    f = robot.now()
    grip = -1.0
    tcp = np.asarray(f.tcp)
    r1 = None
    if tcp[2] < HOVER_Z - 0.03:
        r1 = _to_height(robot, HOVER_Z, grip)
    goal = _view_tcp(robot.now(), x, y, HOVER_Z)
    r = _servo(robot, goal, grip, tol=0.004, max_steps=300)
    _hold(robot, grip, int(wait))
    return {"lift": r1, "view": r, "gap": _finger_gap(robot.now())}


def act_move(robot, x, y, z, grip=-1.0):
    """Lift to z first if lower, then move the TCP to (x, y, z) with top-down orientation."""
    tcp = np.asarray(robot.now().tcp)
    if tcp[2] < z:
        _to_height(robot, z, grip)
    r = _servo(robot, [x, y, z], grip, tol=0.005, max_steps=300)
    return r


def act_wait(robot, steps=20):
    """Hold position with the gripper open for a number of steps."""
    _hold(robot, -1.0, int(steps))
    return {"tcp": np.asarray(robot.now().tcp)[:3].tolist()}


def act_pick(robot, x, y, zsurf, yaw=0.0):
    """Top-down grasp of an object centred at (x, y) on a surface at height zsurf, then lift."""
    log = {}
    tcp = np.asarray(robot.now().tcp)
    if tcp[2] < HOVER_Z - 0.03:
        _to_height(robot, HOVER_Z, -1.0)
    log["above"] = _servo(robot, [x, y, HOVER_Z], -1.0, yaw=yaw, tol=0.004, max_steps=300)
    zg = zsurf + 0.013
    log["pre"] = _servo(robot, [x, y, zsurf + 0.06], -1.0, yaw=yaw, tol=0.004, max_steps=200, vmax=0.8)
    log["down"] = _servo(robot, [x, y, zg], -1.0, yaw=yaw, tol=0.003, max_steps=150, vmax=0.4)
    _hold(robot, 1.0, 14)
    log["gap_closed"] = _finger_gap(robot.now())
    log["lift"] = _servo(robot, [x, y, HOVER_Z], 1.0, yaw=yaw, tol=0.01, max_steps=200, vmax=0.6, settle=1)
    _hold(robot, 1.0, 4)
    log["gap_lifted"] = _finger_gap(robot.now())
    log["holding"] = bool(0.012 < log["gap_lifted"] < 0.072)
    return log


def act_place(robot, x, y, zsurf, view_x, view_y):
    """Carry the held object to (x, y) above a surface at zsurf, lower, release, retreat, view (view_x, view_y)."""
    log = {}
    tcp = np.asarray(robot.now().tcp)
    if tcp[2] < TRAVEL_Z - 0.03:
        _servo(robot, [tcp[0], tcp[1], TRAVEL_Z], 1.0, tol=0.01, max_steps=150, vmax=0.7, settle=1)
    log["carry"] = _servo(robot, [x, y, TRAVEL_Z], 1.0, tol=0.005, max_steps=300, vmax=0.8)
    log["gap_carry"] = _finger_gap(robot.now())
    log["pre"] = _servo(robot, [x, y, zsurf + 0.07], 1.0, tol=0.004, max_steps=200, vmax=0.7)
    log["down"] = _servo(robot, [x, y, zsurf + 0.018], 1.0, tol=0.003, max_steps=150, vmax=0.35)
    _hold(robot, -1.0, 14)
    log["gap_open"] = _finger_gap(robot.now())
    _servo(robot, [x, y, zsurf + 0.10], -1.0, tol=0.01, max_steps=100, vmax=0.6, settle=1)
    goal = _view_tcp(robot.now(), view_x, view_y, HOVER_Z)
    log["view"] = _servo(robot, goal, -1.0, tol=0.004, max_steps=300)
    _hold(robot, -1.0, 6)
    return log
