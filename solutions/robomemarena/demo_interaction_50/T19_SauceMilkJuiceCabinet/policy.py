"""T19 SauceMilkJuiceCabinet: move tomato sauce, milk, orange juice from cabinet1 (top) to cabinet2 (top), in order.

Perception: RGB-D back-projection (camera K + world-to-camera extrinsic) to world points.
Cabinet tops are dark flat surfaces at z ~0.663; cabinet1 holds the objects (y<0 in all demos),
cabinet2 is the empty one (y>0).  Objects on the top are classified per pixel:
sauce = blue-grey can lid, orange juice = yellow, milk = white; red pixels join the nearest
carton seed.  Control: closed-loop OSC delta steps on measured TCP, top-down orientation held.
"""

ORDER = ["sauce", "milk", "oj"]
CAB_TOP_DEFAULT = 0.663
CAB_HX = 0.125   # half size of cabinet top along world x
CAB_HY = 0.094   # half size along world y
# release offsets (tcp xy relative to cabinet2 top centre), measured from all demos
PLACE_OFF = {"sauce": [-0.03, 0.053], "milk": [0.063, 0.033], "oj": [-0.03, -0.03]}
# demo grasp tcp minus wrist-detected centre (median over 140 demo grasps)
GRASP_BIAS = [0.0, 0.0]
TRANSIT_Z = 0.90
# reach model fitted on demo joints (r=0.97): elbow joint q4 ~ 5.44*d - 5.63, d = |wrist - SHOULDER|,
# wrist ~ tcp + 0.107 z. The straight-elbow singularity (q4 -> -0.07) traps the OSC (seen in R2, R4).
SHOULDER = [-0.56, 0.0, 0.58]
D_MAX = 0.84
HOVER_Z = 0.88
SURVEY_Z = 0.95


# ---------------------------------------------------------------- geometry helpers
def _cloud(frame, cam):
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    d = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1).reshape(-1, 3)
    W = (P - E[:, 3]) @ E[:, :3]
    return W.reshape(h, w, 3), d > 0


def _classes(rgb):
    r = rgb[..., 0].astype(float)
    g = rgb[..., 1].astype(float)
    b = rgb[..., 2].astype(float)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    lid = (b > r + 6) & (b > 45)
    yellow = (r > 70) & (g > 0.72 * r) & (b < 0.45 * r)
    white = (mn > 90) & ((mx - mn) < 35)
    red = (r > 70) & (g < 0.62 * r) & (b < 0.5 * r)
    return lid, yellow, white, red


def _detect_objects(frame, cam, foot, top):
    """Object centres [x, y, top_z] on a cabinet footprint foot=[cx, cy] from one camera."""
    W, valid = _cloud(frame, cam)
    rgb = np.asarray(getattr(frame, cam + "_rgb"))
    z = W[..., 2]
    m = valid & (z > top + 0.012) & (z < top + 0.16)
    m &= (np.abs(W[..., 0] - foot[0]) < CAB_HX + 0.01) & (np.abs(W[..., 1] - foot[1]) < CAB_HY + 0.01)
    lid, yel, wh, red = _classes(rgb)
    P = W[m]
    L, Y, Wh, R = lid[m], yel[m], wh[m], red[m]
    res = {}
    counts = {"lid": int(L.sum()), "yellow": int(Y.sum()), "white": int(Wh.sum()), "red": int(R.sum())}
    if L.sum() >= 12:
        Q = P[L]
        c = np.median(Q[:, :2], 0)
        Q = Q[np.linalg.norm(Q[:, :2] - c, axis=1) < 0.05]
        if len(Q) >= 8:
            res["sauce"] = _top_centre(Q)
    seeds = []
    if Y.sum() >= 6:
        seeds.append(("oj", np.median(P[Y][:, :2], 0)))
    if Wh.sum() >= 6:
        seeds.append(("milk", np.median(P[Wh][:, :2], 0)))
    C = Y | Wh | R
    if seeds and C.sum() > 0:
        S = np.array([s[1] for s in seeds])
        Q = P[C]
        dd = np.linalg.norm(Q[:, None, :2] - S[None], axis=2)
        lab = np.argmin(dd, 1)
        ok = dd.min(1) < 0.06
        if "sauce" in res:  # red label of the can is not a carton
            ok &= np.linalg.norm(Q[:, :2] - np.array(res["sauce"][:2]), axis=1) > 0.035
        for k, s in enumerate(seeds):
            sel = ok & (lab == k)
            if sel.sum() >= 8:
                res[s[0]] = _top_centre(Q[sel])
    return res, counts


def _top_centre(Q):
    """[x, y, top_z, n]: centre of the upper face only (side faces seen obliquely bias the centroid)."""
    zt = float(np.percentile(Q[:, 2], 90))
    T = Q[Q[:, 2] > zt - 0.012]
    if len(T) < 5:
        T = Q
    return [float(np.median(T[:, 0])), float(np.median(T[:, 1])), zt, int(len(Q))]


def _object_points(frame, cam, foot, top):
    """All points standing above a cabinet top inside its footprint (any colour)."""
    W, valid = _cloud(frame, cam)
    z = W[..., 2]
    m = valid & (z > top + 0.012) & (z < top + 0.16)
    m &= (np.abs(W[..., 0] - foot[0]) < CAB_HX + 0.03) & (np.abs(W[..., 1] - foot[1]) < CAB_HY + 0.03)
    return W[m]


def _finger_hits(Q, gx, gy, yaw):
    """Points inside the two open-finger footprints of a top-down grasp at (gx, gy) with gripper yaw."""
    fx, fy = -math.sin(yaw), math.cos(yaw)      # finger closing axis (world y at yaw 0)
    px, py = math.cos(yaw), math.sin(yaw)       # finger width axis
    dx = Q[:, 0] - gx
    dy = Q[:, 1] - gy
    along = dx * fx + dy * fy
    across = dx * px + dy * py
    hit = (np.abs(across) < 0.017) & (np.abs(along) > 0.033) & (np.abs(along) < 0.06)
    return int(hit.sum())


def _choose_grasp(P, centre, grasp_z, yaws=(0.0,)):
    """Search small xy offsets and gripper yaws so that both open fingers (inner faces +-0.04 m) land
    on free space. Returns (gx, gy, yaw, cost, cost_at_centre_yaw0)."""
    cx, cy = float(centre[0]), float(centre[1])
    if P is None or len(P) == 0:
        return cx, cy, 0.0, 0, 0
    Q = P[P[:, 2] > grasp_z - 0.02]
    best = None
    c0 = _finger_hits(Q, cx, cy, 0.0)
    for k, yaw in enumerate(yaws):
        for dx in [0.0, -0.004, 0.004, -0.008, 0.008, -0.012, 0.012, -0.016, 0.016]:
            for dy in [0.0, -0.004, 0.004, -0.008, 0.008, -0.012, 0.012, -0.016, 0.016]:
                # offsets are expressed along the finger-width axis mostly; keep the closing axis centred
                gx, gy = cx + dx, cy + dy
                fx, fy = -math.sin(yaw), math.cos(yaw)
                off_close = abs(dx * fx + dy * fy)
                off_width = abs(dx * math.cos(yaw) + dy * math.sin(yaw))
                if off_close > 0.0065:
                    continue
                cost = _finger_hits(Q, gx, gy, yaw)
                score = cost + 30.0 * off_width / 0.016 + 60.0 * off_close / 0.006 + (0.0 if k == 0 else 12.0)
                if best is None or score < best[0]:
                    best = (score, gx, gy, yaw, cost)
    return best[1], best[2], best[3], best[4], c0


def _detect_cabinets(frame):
    """Cabinet top centres from agentview: returns (cab_neg_y, cab_pos_y, top_z) or Nones."""
    W, valid = _cloud(frame, "agentview")
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    m = valid & (W[..., 2] > 0.62) & (W[..., 2] < 0.70) & (rgb.max(-1) < 70)
    m &= (np.abs(W[..., 0]) < 0.35) & (np.abs(W[..., 1]) < 0.7)
    P = W[m]
    out = []
    tops = []
    for s in (-1.0, 1.0):
        Q = P[P[:, 1] * s > 0.02]
        if len(Q) < 60:
            out.append(None)
            continue
        zt = float(np.median(Q[:, 2]))
        Q = Q[np.abs(Q[:, 2] - zt) < 0.008]
        if len(Q) < 60:
            out.append(None)
            continue
        x0, x1 = np.percentile(Q[:, 0], [1, 99])
        y0, y1 = np.percentile(Q[:, 1], [1, 99])
        cx = float(x1 - CAB_HX) if (x1 - x0) < 0.24 else float((x0 + x1) / 2)
        if (y1 - y0) > 0.18:
            cy = float((y0 + y1) / 2)
        else:  # inner edge occluded (arm): use the outer edge and the known size
            cy = float(y0 + CAB_HY) if s < 0 else float(y1 - CAB_HY)
        out.append([cx, cy, zt])
        tops.append(zt)
    top = float(np.median(tops)) if tops else CAB_TOP_DEFAULT
    return out[0], out[1], top


# ---------------------------------------------------------------- rotation helpers
def _reach(p):
    w = np.array([p[0], p[1], p[2] + 0.107]) - np.array(SHOULDER)
    return float(np.linalg.norm(w)), w


def _reach_z(x, y, dmax=D_MAX):
    """Highest TCP z at (x, y) keeping the wrist within dmax of the shoulder."""
    h2 = (x - SHOULDER[0]) ** 2 + (y - SHOULDER[1]) ** 2
    if h2 >= dmax * dmax:
        return SHOULDER[2] - 0.107
    return SHOULDER[2] - 0.107 + math.sqrt(dmax * dmax - h2)


def _rotmat(rv):
    rv = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(rv))
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _rot_cmd(tcp, yaw=0.0):
    """Rotation command that holds the gripper pointing down with the given yaw about world z
    (yaw 0: fingers close along world y)."""
    Rc = _rotmat(tcp[3:6])
    Rt = _rotmat([0.0, 0.0, yaw]) @ _rotmat([math.pi, 0.0, 0.0])
    err = _rotvec(Rt @ Rc.T)
    return np.clip(err / 0.5 * 1.0, -0.3, 0.3)


def _held(frame):
    """Fingers stopped apart by an object: neither closed (<0.01) nor fully open (~0.08)."""
    g = _gap(frame)
    return 0.01 < g < 0.072


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


# ---------------------------------------------------------------- motion primitive
def _move(robot, target, grip, vmax=1.0, tol=0.005, max_steps=150, min_steps=0, stall=0, watch_drop=False,
          slow_below=None, yaw=0.0, rot_tol=None, q7_band=None):
    """Closed-loop move of the TCP to target xyz; returns a summary dict.
    slow_below: (z, vmax2) limits speed once the TCP is below z. Stops early if the elbow nears
    its straight-arm limit (joint 4 > -0.25 rad), reported as limit=True."""
    target = np.asarray(target, dtype=float)
    n = 0
    hist = []
    dropped = False
    limit = False
    prev = None
    while n < max_steps:
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        err = target - tcp[:3]
        dist = float(np.linalg.norm(err))
        r = _rot_cmd(tcp, yaw)
        if dist < tol and n >= min_steps and (rot_tol is None or float(np.max(np.abs(r))) < rot_tol):
            break
        if watch_drop and _gap(fr) < 0.008:
            dropped = True
            break
        q = np.asarray(fr.joints, dtype=float)
        dreach, wv = _reach(tcp)
        if q[3] > -0.55 and float(np.dot(err, wv)) > 0.0:
            # elbow nearly straight and the move goes outward: stop before the singularity
            limit = True
            break
        if q7_band is not None and (q[6] < q7_band[0] or q[6] > q7_band[1]):
            limit = True
            break
        hist.append(tcp[2])
        vel = np.zeros(3) if prev is None else tcp[:3] - prev
        prev = tcp[:3].copy()
        if stall and n > stall and abs(hist[-1] - hist[-1 - stall]) < 0.002 and dist < 0.08:
            break
        a = (2.0 * err - 3.0 * vel) / 0.05
        vm = vmax
        if dreach > 0.80 or q[3] > -0.8:
            vm = min(vm, 0.5)   # extended arm: fast lateral moves swing the elbow straight (R22)
        elif dreach > 0.76:
            vm = min(vm, 0.75)
        if slow_below is not None and tcp[2] < slow_below[0]:
            vm = min(vm, slow_below[1])
        mx = float(np.max(np.abs(a)))
        if mx > vm:
            a = a * (vm / mx)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"steps": n, "tcp": [round(float(v), 4) for v in tcp[:3]],
            "err": round(float(np.linalg.norm(target - tcp[:3])), 4), "gap": round(_gap(fr), 4), "dropped": dropped,
            "limit": limit}


def _hold(robot, grip, steps, yaw=0.0):
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    target = tcp[:3].copy()
    for _ in range(steps):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        a = np.clip((target - tcp[:3]) / 0.05 * 2.0, -0.5, 0.5)
        r = _rot_cmd(tcp, yaw)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return _gap(robot.now())


# ---------------------------------------------------------------- act tools
def _yaw_of(tcp):
    R = _rotmat(np.asarray(tcp[3:6], dtype=float))
    return math.atan2(R[1, 0], R[0, 0])


def act_survey(robot, x, y, z=0.95, grip=-1.0):
    """Lift if low, then move the TCP to (x, y, z) with the gripper open and yaw 0; settle so the
    wrist camera looks straight down on the area. Returns final tcp, error and finger gap."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    zt = min(z, max(float(tcp[2]), 0.86))
    if tcp[2] < zt - 0.03:
        _move(robot, [tcp[0], tcp[1], zt], grip, tol=0.02, max_steps=60, yaw=_yaw_of(tcp))
    if math.hypot(x - tcp[0], y - tcp[1]) > 0.1:
        # long moves at the lower of the two heights (keeps the elbow bent), then rise
        _move(robot, [x, y, zt], grip, tol=0.02, max_steps=150)
    res = _move(robot, [x, y, z], grip, tol=0.004, max_steps=150, rot_tol=0.08)
    _hold(robot, grip, 4)
    res["gap"] = round(_gap(robot.now()), 4)
    res["yaw"] = round(_yaw_of(robot.now().tcp), 3)
    return res


def act_pick(robot, x, y, grasp_z, yaw=0.0, hover_z=0.88, lift_z=0.97):
    """Top-down grasp at (x, y, grasp_z) with gripper yaw (rad about world z; 0 = fingers close
    along world y): hover and turn, descend slowly (stops on stall), close, lift.
    Returns gaps after close/lift; held=True when the fingers stay apart (0.01-0.072 m)."""
    out = {}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if tcp[2] < hover_z - 0.03:
        _move(robot, [tcp[0], tcp[1], hover_z], -1.0, tol=0.02, max_steps=60, yaw=_yaw_of(tcp))
    # wrist joint 7 grows by ~0.9 rad when carrying to cabinet2: keep it inside [-2.5, 1.5] here and
    # otherwise use the equivalent yaw rotated by pi (the parallel gripper is symmetric)
    r = _move(robot, [x, y, hover_z], -1.0, tol=0.006, max_steps=150, yaw=yaw, rot_tol=0.06, q7_band=(-2.5, 1.5))
    if r["limit"] and abs(yaw) > 0.3:
        yaw = yaw - math.pi if yaw > 0 else yaw + math.pi
        out["yaw_switched"] = round(yaw, 3)
        r = _move(robot, [x, y, hover_z], -1.0, tol=0.006, max_steps=180, yaw=yaw, rot_tol=0.06)
    out["hover"] = r
    out["q7"] = round(float(robot.now().joints[6]), 3)
    _hold(robot, -1.0, 3, yaw)
    out["descend"] = _move(robot, [x, y, grasp_z], -1.0, vmax=0.6, tol=0.004, max_steps=150, stall=8,
                           slow_below=(grasp_z + 0.04, 0.3), yaw=yaw)
    _hold(robot, -1.0, 3, yaw)
    g_prev = _gap(robot.now())
    g = g_prev
    for k in range(30):
        g = _hold(robot, 1.0, 1, yaw)
        if k >= 10 and abs(g - g_prev) < 0.0003:
            break
        g_prev = g
    out["gap_closed"] = round(g, 4)
    out["lift"] = _move(robot, [x, y, lift_z], 1.0, vmax=0.8, tol=0.01, max_steps=120, yaw=yaw)
    _hold(robot, 1.0, 5, yaw)
    gl = _gap(robot.now())
    out["gap_lift"] = round(gl, 4)
    out["held"] = bool(0.01 < gl < 0.072)
    return out


def act_place(robot, x, y, release_z, transit_z=0.97):
    """Carry the held object at transit_z (keeping the current gripper yaw) to above (x, y), descend
    to release_z (stops on stall), open, and retreat upward. Aborts with dropped=True if the fingers
    close (object lost) or limit=True if the elbow nears full extension before arrival."""
    out = {}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    yaw = _yaw_of(tcp)
    if tcp[2] < transit_z - 0.03:
        r = _move(robot, [tcp[0], tcp[1], transit_z], 1.0, tol=0.02, max_steps=60, watch_drop=True, yaw=yaw)
        if r["dropped"]:
            out["dropped"] = True
            out["where"] = r["tcp"]
            return out
    r = _move(robot, [x, y, transit_z], 1.0, tol=0.008, max_steps=200, watch_drop=True, yaw=yaw)
    out["transit"] = r
    if r["dropped"]:
        out["dropped"] = True
        out["where"] = r["tcp"]
        return out
    if r["limit"]:
        out["limit"] = True
        out["dropped"] = False
        return out
    r = _move(robot, [x, y, release_z], 1.0, vmax=0.6, tol=0.005, max_steps=150, stall=8, watch_drop=True,
              slow_below=(release_z + 0.05, 0.35), yaw=yaw)
    out["descend"] = r
    out["gap_before_release"] = round(_gap(robot.now()), 4)
    out["dropped"] = bool(r["dropped"])
    _hold(robot, -1.0, 12, yaw)
    out["gap_open"] = round(_gap(robot.now()), 4)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["retreat"] = _move(robot, [tcp[0], tcp[1], release_z + 0.15], -1.0, vmax=0.8, tol=0.01, max_steps=60,
                           yaw=yaw)
    return out


def act_release(robot, lift=0.1):
    """Open the gripper in place and move up by `lift` metres."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    yaw = _yaw_of(tcp)
    _hold(robot, -1.0, 12, yaw)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return _move(robot, [tcp[0], tcp[1], tcp[2] + lift], -1.0, tol=0.01, max_steps=60, yaw=yaw)


def act_wait(robot, steps=20):
    """Hold the current pose with the gripper open for `steps` native steps (lets the scorer settle)."""
    g = _hold(robot, -1.0, int(steps), _yaw_of(robot.now().tcp))
    return {"gap": round(g, 4), "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}
