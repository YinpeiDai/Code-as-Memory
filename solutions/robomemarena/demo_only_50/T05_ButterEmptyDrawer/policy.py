"""T05_ButterEmptyDrawer policy: check top/middle/bottom drawers in order, remember which are occupied,
then put the butter into an empty drawer.  Perception: agentview RGB-D + calibration; control: closed-loop
OSC delta steps on the measured TCP."""

# ---------------------------------------------------------------- constants (derived from the 50 demos)
DEFAULT_CAB = [0.135, 0.465]          # handle x,y of the top drawer when perception fails
DEFAULT_BUTTER = [0.1105, -0.2021, 0.437]
# grasp offsets relative to (hx, hy) and absolute z of each handle (expert measured grasp points)
HANDLE = {"top": [0.0, 0.0, 0.670], "middle": [0.006, 0.001, 0.605], "bottom": [0.003, -0.019, 0.531]}
ORDER = ["top", "middle", "bottom"]
PREFER = ["middle", "bottom", "top"]   # tie-break among empty drawers (demo frequency 40/10/0)
FWD = [-1.5707963, 0.0, 0.0]          # gripper pointing +y, fingers close vertically (= rotvec (3pi/2,0,0))
DOWN = [3.1415926, 0.0, 0.0]          # gripper pointing down
PULL = 0.20                           # pull target beyond the drawer limit (~0.18)
OCC_POINTS = 10                       # raised depth points above the drawer floor => occupied
FLOOR_Z = {"top": 0.640, "middle": 0.575, "bottom": 0.500}
RETRY_DZ = [0.0, 0.006, -0.006]       # grasp height perturbation for handle-grasp retries


def _rv2m(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _m2rv(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        d = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        k = np.sqrt(d)
        i = int(np.argmax(k))
        for j in range(3):
            if j != i and R[i, j] + R[j, i] < 0:
                k[j] = -k[j]
        return k / max(np.linalg.norm(k), 1e-9) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def _rot_err(target_rv, cur_rv):
    return _m2rv(_rv2m(target_rv) @ _rv2m(cur_rv).T)


def _world_points(frame, cam):
    depth = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    pc = np.stack([x, y, depth], -1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, depth > 0


def _find_cabinet(frame):
    """Top-drawer handle (hx, hy) from the cabinet's dark top surface (z 0.695..0.73)."""
    pw, ok = _world_points(frame, "agentview")
    rgb = np.asarray(frame.agentview_rgb, dtype=float).sum(2)
    m = ok & (rgb < 200) & (pw[..., 2] > 0.695) & (pw[..., 2] < 0.73) & (pw[..., 1] > 0.2) \
        & (pw[..., 1] < 0.9) & (pw[..., 0] > -0.2) & (pw[..., 0] < 0.5)
    n = int(m.sum())
    if n < 200:
        return list(DEFAULT_CAB), n
    P = pw[m]
    hy = float(np.percentile(P[:, 1], 2)) - 0.0085
    hx = float(np.percentile(P[:, 0], 98)) - 0.1323
    if not (-0.1 < hx < 0.35 and 0.3 < hy < 0.65):
        return list(DEFAULT_CAB), n
    return [hx, hy], n


def _find_butter(frame):
    """Butter grasp point from agentview depth: flat box ~1.3 cm above the table near its usual place."""
    pw, ok = _world_points(frame, "agentview")
    m = ok & (np.abs(pw[..., 0] - 0.113) < 0.07) & (np.abs(pw[..., 1] + 0.205) < 0.07) \
        & (pw[..., 2] > 0.436) & (pw[..., 2] < 0.47)
    n = int(m.sum())
    if n < 20 or n > 400:
        return list(DEFAULT_BUTTER), n
    P = pw[m]
    c = P.mean(0)
    top = float(np.percentile(P[:, 2], 90))
    g = [float(c[0]) - 0.0024, float(c[1]) + 0.0027, min(max(top - 0.005, 0.432), 0.445)]
    if abs(g[0] - DEFAULT_BUTTER[0]) > 0.05 or abs(g[1] - DEFAULT_BUTTER[1]) > 0.05:
        return list(DEFAULT_BUTTER), n
    return g, n


def _drawer_content(frame):
    """Count depth points clearly above the floor of the drawer currently held open by the gripper."""
    pw, ok = _world_points(frame, "agentview")
    h = np.asarray(frame.tcp, dtype=float)[:3]
    rel = (pw - h)[ok]
    m = (rel[:, 0] > -0.07) & (rel[:, 0] < 0.05) & (rel[:, 1] > 0.07) & (rel[:, 1] < 0.15) \
        & (rel[:, 2] > -0.08) & (rel[:, 2] < 0.06)
    z = rel[m, 2]
    if len(z) < 30:
        return -1, int(len(z))
    floor = float(np.percentile(z, 20))
    return int((z > floor + 0.012).sum()), int(len(z))


def _grip_width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


# ---------------------------------------------------------------- sequence builders (JSON segments)
def _mv(p, r, g, tol=0.005, mx=150, stall=False, vmax=1.0, rtol=0.05, minsteps=0, name=""):
    return {"k": "move", "p": p, "r": r, "g": g, "tol": tol, "rtol": rtol, "max": mx, "stall": stall,
            "vmax": vmax, "min": minsteps, "name": name}


def _grip(g, n, name=""):
    return {"k": "grip", "g": g, "n": n, "name": name}


def _check(kind, value, fail):
    return {"k": "check", "c": kind, "v": value, "fail": fail}


def _handle(cab, d):
    o = HANDLE[d]
    return [cab[0] + o[0], cab[1] + o[1], o[2]]


def _seq_open(cab, d, from_handle=None, dz=0.0):
    """From anywhere in front of the cabinet (gripper open): go to drawer d, grasp, pull to its limit.
    dz perturbs the grasp height on retries."""
    hp = _handle(cab, d)
    hp[2] = hp[2] + dz
    back = cab[1] - 0.085
    segs = []
    if from_handle is not None:
        fp = _handle(cab, from_handle)
        segs.append(_mv([fp[0], back, fp[2]], FWD, -1, tol=0.008, mx=80, name="retreat"))
    else:
        segs.append(_mv([hp[0], cab[1] - 0.13, hp[2]], FWD, -1, tol=0.008, rtol=0.04, mx=200, name="pre"))
    segs.append(_mv([hp[0], back, hp[2]], FWD, -1, tol=0.005, mx=80, name="align"))
    segs.append(_mv([hp[0], hp[1] + 0.004, hp[2]], FWD, -1, tol=0.003, mx=60, stall=True, vmax=0.6,
                    minsteps=6, name="approach"))
    segs.append(_grip(1, 16, name="grasp"))
    segs.append(_check("width_gt", 0.006, "grasp"))
    segs.append(_mv([None, hp[1] - PULL, None], FWD, 1, tol=0.004, mx=160, stall=True, minsteps=15,
                    name="pull"))
    segs.append(_grip(1, 4, name="hold"))
    segs.append(_check("y_lt", hp[1] - 0.12, "pull"))
    segs.append(_check("width_gt", 0.006, "pull"))
    return segs


def _seq_close(cab, d, release=True):
    hp = _handle(cab, d)
    segs = [_mv([None, hp[1] + 0.006, None], FWD, 1, tol=0.003, mx=160, stall=True, minsteps=15, name="push"),
            _grip(1, 6, name="hold_closed"),
            _check("y_gt", hp[1] - 0.035, "push")]
    if release:
        segs.append(_grip(-1, 12, name="release"))
    return segs


def _seq_after_target_open(cab, d):
    """Release the opened target drawer and back away, ending high and pointing forward."""
    hp = _handle(cab, d)
    return [_grip(-1, 12, name="release"),
            _mv([None, cab[1] - 0.32, None], FWD, -1, tol=0.01, mx=80, name="backoff"),
            _mv([hp[0], cab[1] - 0.32, 0.73], FWD, -1, tol=0.01, mx=80, name="rise")]


def _seq_butter(b):
    return [_mv([b[0], b[1], 0.64], DOWN, -1, tol=0.008, rtol=0.04, mx=200, name="above_butter"),
            _mv([b[0], b[1], b[2] + 0.04], DOWN, -1, tol=0.004, rtol=0.03, mx=80, name="near_butter"),
            _mv([b[0], b[1], b[2]], DOWN, -1, tol=0.0025, rtol=0.03, mx=80, vmax=0.4, name="at_butter"),
            _grip(1, 18, name="grasp_butter"),
            _check("width_gt", 0.012, "butter"),
            _mv([b[0], b[1], 0.76], DOWN, 1, tol=0.01, mx=80, name="lift"),
            _check("width_gt", 0.012, "butter")]


def _seq_place(cab, d):
    x, y = cab[0] - 0.01, cab[1] - 0.10
    zr = 0.73 if d == "top" else 0.70
    return [_mv([x, y, 0.88], DOWN, 1, tol=0.01, rtol=0.05, mx=200, name="carry"),
            _mv([x, y, zr], DOWN, 1, tol=0.004, mx=150, vmax=0.3, name="descend"),
            _grip(1, 10, name="settle"),
            _grip(-1, 15, name="drop"),
            _mv([x, y, 0.88], DOWN, -1, tol=0.01, mx=80, vmax=0.5, name="clear"),
            _grip(-1, 60, name="wait")]


def _seq_recover(cab):
    """Open the gripper and back away from the cabinet front (keeps current height)."""
    return [_grip(-1, 12, name="open"),
            _mv([None, cab[1] - 0.12, None], FWD, -1, tol=0.01, mx=80, name="away")]


def _choose(occ):
    empty = [d for d in PREFER if occ.get(d) is not None and 0 <= occ[d] < OCC_POINTS]
    if empty:
        return empty[0]
    unknown = [d for d in PREFER if occ.get(d) is None or occ[d] < 0]
    if unknown:
        return unknown[0]
    return min(PREFER, key=lambda d: occ[d])


def _status(last):
    if not last:
        return None
    for s in ("ok", "fail_grasp", "fail_pull", "fail_push", "fail_butter"):
        if '"status": "%s"' % s in last:
            return s
    return "error"


def perceive_decide(obs, goal, memory, last):
    """State machine. Memory: stage, cabinet handle xy, butter grasp point, occupancy counts per drawer,
    chosen target drawer, retry counter and a short log."""
    mem = dict(memory)
    cur = obs.current
    stage = mem.get("stage") or "start"
    cab = mem.get("cabinet")
    occ = {"top": mem.get("occ_top"), "middle": mem.get("occ_middle"), "bottom": mem.get("occ_bottom")}
    retries = int(mem.get("retries") or 0)
    log = mem.get("log") or ""
    st = _status(last)
    act = None

    if stage == "start":
        cab, n = _find_cabinet(cur)
        b, nb = _find_butter(cur)
        mem["cabinet"] = [round(cab[0], 4), round(cab[1], 4)]
        mem["butter"] = [round(b[0], 4), round(b[1], 4), round(b[2], 4)]
        cab = mem["cabinet"]
        log += "cab%s n%d butter%s n%d;" % (mem["cabinet"], n, mem["butter"], nb)
        stage = "open_top"
        act = _seq_open(cab, "top")
    else:
        if stage.startswith("open_"):
            d = stage[5:]
            if st == "ok":
                c, n = _drawer_content(cur)
                occ[d] = c
                log += "%s:%d/%d;" % (d, c, n)
                retries = 0
                i = ORDER.index(d)
                if i < 2:
                    nxt = ORDER[i + 1]
                    stage = "open_" + nxt
                    act = _seq_close(cab, d) + _seq_open(cab, nxt, from_handle=d)
                else:
                    tgt = _choose(occ)
                    mem["target"] = tgt
                    log += "target=%s;" % tgt
                    if tgt == "bottom":
                        stage = "target_open"
                        act = _seq_close(cab, d, release=False) + [_grip(1, 8, name="closed_pause")] + [
                            _mv([None, _handle(cab, d)[1] - PULL, None], FWD, 1, tol=0.004, mx=160, stall=True,
                                minsteps=15, name="pull_again"),
                            _check("y_lt", _handle(cab, d)[1] - 0.12, "pull")]
                    else:
                        stage = "target_open"
                        act = _seq_close(cab, d) + _seq_open(cab, tgt, from_handle=d)
            else:
                retries += 1
                log += "%s_%s;" % (stage, st)
                if retries > 3:
                    # give up checking this drawer; move on
                    occ[d] = -1
                    i = ORDER.index(d)
                    retries = 0
                    if i < 2:
                        stage = "open_" + ORDER[i + 1]
                        act = _seq_recover(cab) + _seq_open(cab, ORDER[i + 1])
                    else:
                        tgt = _choose(occ)
                        mem["target"] = tgt
                        stage = "target_open"
                        act = _seq_recover(cab) + _seq_open(cab, tgt)
                elif st == "fail_push":
                    act = _seq_close(cab, ORDER[max(0, ORDER.index(d) - 1)])
                    # retry the close of the previous drawer, then reopen this one
                    act = act + _seq_open(cab, d, from_handle=ORDER[max(0, ORDER.index(d) - 1)])
                else:
                    act = _seq_recover(cab) + _seq_open(cab, d, dz=RETRY_DZ[retries % 3])
        elif stage == "target_open":
            tgt = mem.get("target") or "middle"
            if st == "ok":
                stage = "fetch_butter"
                retries = 0
                act = _seq_after_target_open(cab, tgt) + _seq_butter(mem.get("butter") or DEFAULT_BUTTER)
            else:
                retries += 1
                log += "target_%s;" % st
                if retries > 3:
                    stage = "fetch_butter"
                    act = _seq_after_target_open(cab, tgt) + _seq_butter(mem.get("butter") or DEFAULT_BUTTER)
                else:
                    act = _seq_recover(cab) + _seq_open(cab, tgt, dz=RETRY_DZ[retries % 3])
        elif stage == "fetch_butter":
            tgt = mem.get("target") or "middle"
            b = mem.get("butter") or DEFAULT_BUTTER
            if st == "ok":
                stage = "place"
                retries = 0
                act = _seq_place(cab, tgt)
            else:
                retries += 1
                log += "butter_%s;" % st
                tcp = np.asarray(cur.tcp, dtype=float)
                act = [_grip(-1, 12, name="open"),
                       _mv([float(tcp[0]), float(tcp[1]), 0.64], DOWN, -1, tol=0.01, mx=80, name="up")] \
                    + _seq_butter(b)
                if retries > 4:
                    stage = "place"
                    act = _seq_place(cab, tgt)
        else:  # place / done: keep the scene still while the checker monitors
            stage = "done"
            act = [_grip(-1, 100, name="idle")]

    mem["stage"] = stage
    mem["occ_top"], mem["occ_middle"], mem["occ_bottom"] = occ["top"], occ["middle"], occ["bottom"]
    mem["retries"] = retries
    mem["log"] = log[-1500:]
    for k in ("cabinet", "butter", "target"):
        mem.setdefault(k, None)
    return {"memory": {k: mem.get(k) for k in MEMORY_FIELDS},
            "act": {"tool": "act_run", "args": {"segments": act, "label": stage}}}


MEMORY_FIELDS = ["stage", "cabinet", "butter", "occ_top", "occ_middle", "occ_bottom", "target", "retries", "log"]


def _run_move(robot, s):
    start = np.asarray(robot.now().tcp, dtype=float)
    p = s.get("p")
    tgt = np.array([start[i] if (p is None or p[i] is None) else float(p[i]) for i in range(3)])
    rv = s.get("r") if s.get("r") is not None else list(start[3:6])
    g = float(s.get("g", 0))
    hist = []
    n = 0
    reason = "max"
    for n in range(1, int(s.get("max", 100)) + 1):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = tgt - tcp[:3]
        re = _rot_err(rv, tcp[3:6])
        if np.linalg.norm(err) < s.get("tol", 0.005) and np.linalg.norm(re) < s.get("rtol", 0.05):
            reason = "reached"
            break
        hist.append(tcp[:3].copy())
        if s.get("stall") and n > max(8, int(s.get("min", 0))) and np.linalg.norm(hist[-1] - hist[-8]) < 0.002:
            reason = "stall"
            break
        dp = err * 10.0
        nm = float(np.linalg.norm(dp))
        vmax = float(s.get("vmax", 1.0))
        if nm > vmax:
            dp = dp * (vmax / nm)
        dr = re * 1.0
        rn = float(np.linalg.norm(dr))
        if rn > 0.5:
            dr = dr * (0.5 / rn)
        robot.step(float(dp[0]), float(dp[1]), float(dp[2]), float(dr[0]), float(dr[1]), float(dr[2]), g)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"seg": s.get("name", ""), "end": reason, "n": n, "pos": [round(float(v), 4) for v in tcp[:3]]}


def _run_grip(robot, s):
    start = np.asarray(robot.now().tcp, dtype=float)
    g = float(s.get("g", 0))
    for _ in range(int(s.get("n", 10))):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        dp = np.clip((start[:3] - tcp[:3]) * 10.0, -0.3, 0.3)
        dr = np.clip(_rot_err(start[3:6], tcp[3:6]), -0.2, 0.2)
        robot.step(float(dp[0]), float(dp[1]), float(dp[2]), float(dr[0]), float(dr[1]), float(dr[2]), g)
    return {"seg": s.get("name", ""), "width": round(_grip_width(robot.now()), 4)}


def act_run(robot, segments, label=""):
    """Execute a list of segments in order. Segment kinds: move {p:[x,y,z] (None keeps start), r: rotvec,
    g: gripper sign, tol, rtol, max, stall, vmax, min}, grip {g, n} (holds pose), check {c: width_gt|y_lt|y_gt,
    v, fail}. Returns status ok or fail_<reason> at the first failing check, plus per-segment summaries."""
    out = []
    for s in segments:
        k = s.get("k")
        if k == "move":
            out.append(_run_move(robot, s))
        elif k == "grip":
            out.append(_run_grip(robot, s))
        elif k == "check":
            f = robot.now()
            tcp = np.asarray(f.tcp, dtype=float)
            w = _grip_width(f)
            c = s.get("c")
            v = float(s.get("v"))
            okk = (c == "width_gt" and w > v) or (c == "y_lt" and tcp[1] < v) or (c == "y_gt" and tcp[1] > v)
            if not okk:
                return {"status": "fail_" + s.get("fail", "check"), "label": label, "width": round(w, 4),
                        "pos": [round(float(x), 4) for x in tcp[:3]], "segs": out[-6:]}
    f = robot.now()
    return {"status": "ok", "label": label, "width": round(_grip_width(f), 4),
            "pos": [round(float(x), 4) for x in np.asarray(f.tcp, dtype=float)[:3]], "segs": out[-8:]}
