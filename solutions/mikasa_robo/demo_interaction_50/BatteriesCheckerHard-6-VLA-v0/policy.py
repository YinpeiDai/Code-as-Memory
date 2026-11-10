"""BatteriesCheckerHard: test every battery in the socket, return it to its slot, press the button.

Layout model (from 50 demos, see task_analysis.md): the rack, socket, lamp and button
form one rigid layout that is only translated between episodes.  Its origin `org`
is the mean of the six detected green battery tops (front camera, depth
back-projection).  All targets are fixed offsets from `org`.
"""

# ---------------- constants learned from the demonstrations -----------------
SLOT_GRASP = [[0.0458, -0.0267], [-0.0034, -0.0269], [-0.0548, -0.0266],
              [0.0464, 0.0255], [-0.0034, 0.0251], [-0.0545, 0.0255]]
SLOT_RELEASE = [[0.0427, -0.0267], [-0.0080, -0.0265], [-0.0579, -0.0265],
                [0.0428, 0.0249], [-0.0081, 0.0251], [-0.0578, 0.0251]]
SOCKET_OFF = [0.1919, 0.0777]
BUTTON_CENTROID_OFF = [-0.0029, 0.3145]     # pink centroid relative to org
BUTTON_PRESS_OFF = [-0.0125, 0.3105]        # where the expert pressed
LAMP_OFF = [0.273, 0.075, 0.10]
Z_TRANSIT = 0.21
Z_GRASP = 0.043
Z_INSERT = 0.043
Z_REGRASP = 0.047
Z_SLOT_RELEASE = 0.047
Z_PRESS = 0.052
G_APPROACH = 0.1
G_REGRASP = 0.22
G_SLOT_OPEN = 0.28


# ---------------- small math helpers -----------------
def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _backproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(float) / 1000.0
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (pc - t) @ R


def _project(K, E, P):
    E = np.asarray(E)
    pc = E[:, :3] @ np.asarray(P, dtype=float) + E[:, 3]
    return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]


def _detect_layout(frame):
    """Green battery tops and pink button from the front camera -> layout origin."""
    im = frame.front_rgb.astype(int)
    d = frame.front_depth
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    green = (g > r + 30) & (g > b + 20)
    pink = (r > 150) & (r > g + 25) & (r > b + 20) & (np.abs(g - b) < 30)
    P = _backproject(d, K, E, green)
    P = P[(P[:, 2] > 0.03) & (P[:, 2] < 0.12)] if len(P) else P
    clusters = []
    pts = P.copy()
    while len(pts):
        c = pts[0]
        for _ in range(5):
            m = np.linalg.norm(pts[:, :2] - c[:2], axis=1) < 0.02
            c = pts[m].mean(0)
        m = np.linalg.norm(pts[:, :2] - c[:2], axis=1) < 0.02
        if m.sum() >= 3:
            clusters.append([float(c[0]), float(c[1]), int(m.sum())])
        pts = pts[~m]
    Q = _backproject(d, K, E, pink)
    Q = Q[(Q[:, 2] > 0.02) & (Q[:, 2] < 0.12)] if len(Q) else Q
    btn = [float(Q[:, 0].mean()), float(Q[:, 1].mean())] if len(Q) >= 10 else None
    org = None
    method = "none"
    can = np.array(SLOT_GRASP) + np.array([0.004, 0.001])   # detection bias of the green tops
    if len(clusters) == 6:
        org = np.array(clusters)[:, :2].mean(0)
        method = "six_batteries"
    elif btn is not None:
        org = np.array(btn) - np.array(BUTTON_CENTROID_OFF)
        method = "button"
        # refine by snapping detected clusters onto the canonical grid
        if len(clusters) >= 2:
            ests = []
            for c in clusters:
                rel = np.array(c[:2]) - org
                j = int(np.argmin(((can - rel) ** 2).sum(1)))
                if np.linalg.norm(can[j] - rel) < 0.015:
                    ests.append(np.array(c[:2]) - can[j] + np.array([0.0, 0.0]))
            if len(ests) >= 2:
                org = np.mean(np.array(ests), 0)
                method = "button+%d_batteries" % len(ests)
    elif len(clusters) >= 3:
        method = "partial_batteries"
        # assume the mean of present clusters; crude
        org = np.array(clusters)[:, :2].mean(0)
    # consistency check between batteries and button
    check = None
    if org is not None and btn is not None:
        check = float(np.linalg.norm(np.array(btn) - np.array(BUTTON_CENTROID_OFF) - org))
    return {"org": None if org is None else [float(org[0]), float(org[1])],
            "n_clusters": len(clusters), "button": btn, "method": method, "btn_residual": check}


def _green_clusters(frame):
    """Green battery-top clusters [x, y, z_top, n] from the front camera."""
    im = frame.front_rgb.astype(int)
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    green = (g > r + 30) & (g > b + 20)
    P = _backproject(frame.front_depth, K, E, green)
    P = P[(P[:, 2] > 0.025) & (P[:, 2] < 0.12)] if len(P) else P
    out = []
    pts = P.copy()
    while len(pts):
        c = pts[0]
        for _ in range(5):
            m = np.linalg.norm(pts[:, :2] - c[:2], axis=1) < 0.02
            c = pts[m].mean(0)
        m = np.linalg.norm(pts[:, :2] - c[:2], axis=1) < 0.02
        if m.sum() >= 3:
            out.append([float(c[0]), float(c[1]), float(pts[m][:, 2].max()), int(m.sum())])
        pts = pts[~m]
    return out


def _slot_check(frame, rx):
    """Is a battery top seen where a battery seated at release xy `rx` should appear?"""
    cl = _green_clusters(frame)
    exp = [rx[0] + 0.0075, rx[1] + 0.001]
    best = None
    for c in cl:
        d = math.hypot(c[0] - exp[0], c[1] - exp[1])
        if best is None or d < best[0]:
            best = [d, c]
    if best is None:
        return {"ok": False, "d": None, "n": 0, "cand": None}
    return {"ok": best[0] < 0.010, "d": float(best[0]), "n": len(cl), "cand": best[1]}


def _bad_slots(frame, org):
    """Slots without a seated battery top; each with the nearest unmatched cluster and its distance."""
    cl = _green_clusters(frame)
    exp = [[org[0] + r[0] + 0.0075, org[1] + r[1] + 0.001] for r in SLOT_RELEASE]
    used = set()
    good = set()
    for i, e in enumerate(exp):
        for j, c in enumerate(cl):
            if j not in used and math.hypot(c[0] - e[0], c[1] - e[1]) < 0.010:
                used.add(j)
                good.add(i)
                break
    bad = []
    for i, e in enumerate(exp):
        if i in good:
            continue
        best = None
        for j, c in enumerate(cl):
            if j in used:
                continue
            d = math.hypot(c[0] - e[0], c[1] - e[1])
            if best is None or d < best[1]:
                best = [c, d]
        bad.append([i, None if best is None else best[0], 1.0 if best is None else best[1]])
    return bad


def _reseat(robot, rx, c, rpy, log):
    """One re-seat attempt: grasp a displaced battery at its seen top `c` and put it into slot `rx`."""
    gxy = [c[0] - 0.004, c[1] - 0.001]
    zg = max(c[2] - 0.017, 0.036)
    fs = _pick(robot, gxy, zg, rpy, log, "reseat_pick")
    if fs >= 0.006:
        _move(robot, [gxy[0], gxy[1], 0.12], -1.0, rpy, tol=0.01, vtol=0.01, log=log, name="reseat_lift")
        _descend_into(robot, rx, Z_SLOT_RELEASE, 0.078, -1.0, rpy, log, "reseat")
        _grip(robot, G_SLOT_OPEN, robot.now().tcp[:3], rpy, n_min=5, n_max=12, log=log, name="reseat_release")
        p = robot.now().tcp
        _move(robot, [rx[0], rx[1], float(p[2]) + 0.035], G_SLOT_OPEN, rpy, tol=0.006, vtol=0.01, max_steps=20, vzmax=0.3, log=log, name="reseat_up_slow")
    else:
        _grip(robot, G_APPROACH, robot.now().tcp[:3], rpy, n_min=4, n_max=10, log=log, name="reseat_reopen")
    _move(robot, [rx[0], rx[1], Z_TRANSIT], G_SLOT_OPEN, rpy, tol=0.02, vtol=0.02, log=log, name="reseat_up")
    chk = _slot_check(robot.now(), rx)
    log.append("slot_recheck ok=%s d=%s" % (chk["ok"], None if chk["d"] is None else round(chk["d"], 4)))
    return chk


def _lamp_count(frame, org):
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    L = [org[0] + LAMP_OFF[0], org[1] + LAMP_OFF[1], LAMP_OFF[2]]
    u, v = _project(K, E, L)
    u, v = int(u), int(v)
    win = frame.front_rgb[max(v - 10, 0):v + 10, max(u - 10, 0):u + 10].astype(int)
    if win.size == 0:
        return 0
    return int((win.min(2) >= 250).sum())


# ---------------- motion primitives -----------------
def _rot_cmd(frame, Rdes, kr=0.6):
    t = frame.tcp
    R = _rpy2R(float(t[3]), float(t[4]), float(t[5]))
    w = _logR(Rdes @ R.T)
    c = -kr * w / 0.1
    return [float(min(1.0, max(-1.0, x))) for x in c], float(np.linalg.norm(w))


def _move(robot, target, grip, rpy, tol=0.004, vtol=0.0015, max_steps=60, k=1.0,
          vmax=1.0, stall=False, log=None, name="", rtol=None, kxy=None, vzmax=None):
    """Closed-loop move of the TCP to target xyz with the given gripper command.

    rtol: also require orientation error (rad) below rtol to arrive.
    kxy: separate lateral gain; vzmax: separate clip for the vertical command."""
    Rdes = _rpy2R(rpy[0], rpy[1], rpy[2])
    target = np.array(target, dtype=float)
    prev = np.array(robot.now().tcp[:3], dtype=float)
    n = 0
    slow = 0
    reason = "max_steps"
    while n < max_steps:
        f = robot.now()
        p = np.array(f.tcp[:3], dtype=float)
        v = float(np.linalg.norm(p - prev))
        pz_prev = float(prev[2])
        prev = p
        err = target - p
        en = float(np.linalg.norm(err))
        rc, rerr = _rot_cmd(f, Rdes)
        if en < tol and v < vtol and (rtol is None or rerr < rtol):
            reason = "arrived"
            break
        if stall and n > 4:
            # vertical progress only: lateral stick-slip jitter in contact must not mask a stall
            slow = slow + 1 if abs(float(p[2]) - float(pz_prev)) < 0.0007 else 0
            if slow >= 4:
                reason = "stalled"
                break
        c = np.clip(k * err / 0.1, -vmax, vmax)
        if kxy is not None:
            c[0] = min(vmax, max(-vmax, kxy * err[0] / 0.1))
            c[1] = min(vmax, max(-vmax, kxy * err[1] / 0.1))
        if vzmax is not None:
            c[2] = min(vzmax, max(-vzmax, c[2]))
        robot.step(float(c[0]), float(c[1]), float(c[2]), rc[0], rc[1], rc[2], float(grip))
        n += 1
    p = np.array(robot.now().tcp[:3], dtype=float)
    if log is not None:
        e = target - p
        log.append("%s:%s n=%d exy=(%.4f,%.4f) z=%.3f" % (name, reason, n, float(e[0]), float(e[1]), float(p[2])))
    return reason, n, p


def _descend_into(robot, xy, z_goal, z_pre, grip, rpy, log, name, tries=3, accept=0.009):
    """Settle above a hole (xy, orientation), then descend slowly; if the held battery
    stalls clearly above z_goal, lift back to z_pre and retry. Returns (ok, z)."""
    p = robot.now().tcp
    for i in range(tries):
        _move(robot, [xy[0], xy[1], z_pre], grip, rpy, tol=0.002, vtol=0.001, max_steps=40,
              rtol=0.015, kxy=1.3, log=log, name="%s_pre%d" % (name, i))
        reason, n, p = _move(robot, [xy[0], xy[1], z_goal], grip, rpy, tol=0.003, vtol=0.0012,
                             max_steps=50, stall=True, kxy=1.5, vzmax=0.3, log=log, name="%s_down%d" % (name, i))
        if p[2] <= z_goal + accept:
            return True, float(p[2])
    return False, float(p[2])


def _grip(robot, grip, hold, rpy, n_min=4, n_max=15, log=None, name=""):
    """Command the gripper while holding position; stop when the fingers stop moving."""
    Rdes = _rpy2R(rpy[0], rpy[1], rpy[2])
    hold = np.array(hold, dtype=float)
    last = float(np.sum(robot.now().fingers))
    n = 0
    still = 0
    while n < n_max:
        f = robot.now()
        p = np.array(f.tcp[:3], dtype=float)
        c = np.clip(0.8 * (hold - p) / 0.1, -0.3, 0.3)
        rc, _ = _rot_cmd(f, Rdes)
        robot.step(float(c[0]), float(c[1]), float(c[2]), rc[0], rc[1], rc[2], float(grip))
        n += 1
        fs = float(np.sum(robot.now().fingers))
        still = still + 1 if abs(fs - last) < 0.0006 else 0
        last = fs
        if n >= n_min and still >= 2:
            break
    if log is not None:
        log.append("%s: n=%d fingers=%.4f" % (name, n, last))
    return last, n


def _pick(robot, xy, z, rpy, log, name):
    """Approach from above with partly open fingers, descend, close. Returns finger sum."""
    _move(robot, [xy[0], xy[1], Z_TRANSIT], G_APPROACH, rpy, tol=0.006, vtol=0.004, log=log, name=name + "_above")
    _move(robot, [xy[0], xy[1], z], G_APPROACH, rpy, tol=0.003, vtol=0.0012, max_steps=50, stall=True, log=log, name=name + "_down")
    fs, _ = _grip(robot, -1.0, [xy[0], xy[1], z], rpy, n_min=6, n_max=15, log=log, name=name + "_close")
    return fs


# ---------------- act tools -----------------
def act_cycle(robot, index, org, rpy, start="pick"):
    """Handle battery `index` (0..5, demo order) of the rack whose origin is `org`.

    Stages: pick from slot -> insert into socket -> release & observe lamp ->
    regrasp -> return to the original slot -> release -> press the button.
    `start` may be 'pick', 'socket' (battery already held), 'regrasp' (battery
    released in the socket), 'return' (held, tested),
    or 'button' (battery already back).  Returns a code string 'OK', or
    'FAIL_<stage>' plus measurements and a log of every motion segment.
    """
    log = []
    org = [float(org[0]), float(org[1])]
    rpy = [float(rpy[0]), float(rpy[1]), float(rpy[2])]
    gx = [org[0] + SLOT_GRASP[index][0], org[1] + SLOT_GRASP[index][1]]
    rx = [org[0] + SLOT_RELEASE[index][0], org[1] + SLOT_RELEASE[index][1]]
    sk = [org[0] + SOCKET_OFF[0], org[1] + SOCKET_OFF[1]]
    bt = [org[0] + BUTTON_PRESS_OFF[0], org[1] + BUTTON_PRESS_OFF[1]]
    step0 = int(robot.now().step)
    lamp_max = -1
    lamp_counts = []
    zmin = None
    slot_ok = None
    stage = start
    if stage == "pick":
        fs = _pick(robot, gx, Z_GRASP, rpy, log, "pick")
        if fs < 0.006:
            # missed: reopen, retry once slightly lower
            _grip(robot, G_APPROACH, robot.now().tcp[:3], rpy, n_min=4, n_max=10, log=log, name="pick_reopen")
            _move(robot, [gx[0], gx[1], Z_GRASP + 0.04], G_APPROACH, rpy, tol=0.006, vtol=0.004, log=log, name="pick_up")
            fs = _pick(robot, gx, Z_GRASP - 0.004, rpy, log, "pick2")
            if fs < 0.006:
                _grip(robot, G_APPROACH, robot.now().tcp[:3], rpy, n_min=4, n_max=10, log=log, name="pick2_reopen")
                return {"code": "FAIL_pick", "fingers": fs, "steps": int(robot.now().step) - step0, "log": log}
        _move(robot, [gx[0], gx[1], Z_TRANSIT], -1.0, rpy, tol=0.02, vtol=0.02, log=log, name="lift")
        stage = "socket"
    if stage == "socket":
        _move(robot, [sk[0], sk[1], Z_TRANSIT], -1.0, rpy, tol=0.005, vtol=0.003, log=log, name="to_socket")
        ok, zz = _descend_into(robot, sk, Z_INSERT, 0.078, -1.0, rpy, log, "insert", accept=0.011)
        fsum = float(np.sum(robot.now().fingers))
        if not ok:
            _move(robot, [sk[0], sk[1], Z_TRANSIT], -1.0, rpy, tol=0.02, vtol=0.02, log=log, name="insert_abort")
            return {"code": "FAIL_insert", "z": zz, "fingers": fsum, "steps": int(robot.now().step) - step0, "log": log}
        # release and observe the lamp
        _grip(robot, 1.0, [sk[0], sk[1], Z_INSERT], rpy, n_min=6, n_max=12, log=log, name="release")
        for i in range(10):
            f = robot.now()
            c = _lamp_count(f, org)
            lamp_counts.append(c)
            lamp_max = max(lamp_max, c)
            rc, _ = _rot_cmd(f, _rpy2R(rpy[0], rpy[1], rpy[2]))
            robot.step(0.0, 0.0, 0.0, rc[0], rc[1], rc[2], 1.0)
        c = _lamp_count(robot.now(), org)
        lamp_counts.append(c)
        lamp_max = max(lamp_max, c)
        stage = "regrasp"
    if stage == "regrasp":
        if start == "regrasp":
            # resume: open IN PLACE first (lifting with half-closed fingers drags/tilts the battery)
            _grip(robot, 1.0, robot.now().tcp[:3], rpy, n_min=6, n_max=12, log=log, name="re_open")
            p = robot.now().tcp
            if math.hypot(float(p[0]) - sk[0], float(p[1]) - sk[1]) > 0.01:
                _move(robot, [float(p[0]), float(p[1]), Z_TRANSIT], 1.0, rpy, tol=0.01, vtol=0.004, log=log, name="re_up")
                _move(robot, [sk[0], sk[1], Z_TRANSIT], 1.0, rpy, tol=0.006, vtol=0.004, log=log, name="re_to_socket")
        fs = 0.0
        for k in range(2):
            _move(robot, [sk[0], sk[1], Z_REGRASP], 1.0, rpy, tol=0.003, vtol=0.0015, max_steps=40, log=log, name="regrasp_pos%d" % k)
            fs, _ = _grip(robot, -1.0, [sk[0], sk[1], Z_REGRASP], rpy, n_min=6, n_max=15, log=log, name="regrasp%d" % k)
            if fs >= 0.006:
                break
            _grip(robot, 1.0, robot.now().tcp[:3], rpy, n_min=6, n_max=12, log=log, name="regrasp_reopen%d" % k)
        if fs < 0.006:
            return {"code": "FAIL_regrasp", "fingers": fs, "lamp_max": lamp_max, "steps": int(robot.now().step) - step0, "log": log}
        _move(robot, [sk[0], sk[1], Z_TRANSIT], -1.0, rpy, tol=0.02, vtol=0.02, log=log, name="extract")
        stage = "return"
    if stage == "return":
        _move(robot, [rx[0], rx[1], Z_TRANSIT], -1.0, rpy, tol=0.005, vtol=0.003, log=log, name="to_slot")
        ok, zz = _descend_into(robot, rx, Z_SLOT_RELEASE, 0.078, -1.0, rpy, log, "slot")
        if not ok:
            _move(robot, [rx[0], rx[1], Z_TRANSIT], -1.0, rpy, tol=0.02, vtol=0.02, log=log, name="slot_abort")
            return {"code": "FAIL_return", "z": zz, "lamp_max": lamp_max, "steps": int(robot.now().step) - step0, "log": log}
        _grip(robot, G_SLOT_OPEN, [rx[0], rx[1], Z_SLOT_RELEASE], rpy, n_min=5, n_max=12, log=log, name="slot_release")
        # slow first lift: a full-speed start lurches the TCP ~9 mm sideways while the open fingers
        # still surround the released battery top (R040), which can knock it into a lean
        _move(robot, [rx[0], rx[1], Z_SLOT_RELEASE + 0.035], G_SLOT_OPEN, rpy, tol=0.006, vtol=0.01, max_steps=20, vzmax=0.3, log=log, name="slot_up_slow")
        _move(robot, [rx[0], rx[1], Z_TRANSIT], G_SLOT_OPEN, rpy, tol=0.02, vtol=0.02, log=log, name="slot_up")
        chk = _slot_check(robot.now(), rx)
        log.append("slot_check ok=%s d=%s n=%d" % (chk["ok"], None if chk["d"] is None else round(chk["d"], 4), chk["n"]))
        if (not chk["ok"]) and chk["cand"] is not None and chk["d"] < 0.035:
            chk = _reseat(robot, rx, chk["cand"], rpy, log)
        slot_ok = chk["ok"]
        stage = "button"

    if stage == "button":
        _move(robot, [bt[0], bt[1], Z_TRANSIT], -1.0, rpy, tol=0.006, vtol=0.003, log=log, name="to_button")
        if index == 5 and start != "button":
            # before the final confirmation press, with the arm beside the rack (no occlusion):
            # every slot must show a seated battery; re-seat clearly displaced ones once
            bad = _bad_slots(robot.now(), org)
            log.append("final_check bad=%s" % str([[b[0], round(b[2], 4)] for b in bad]))
            fixed = False
            for b in bad:
                if b[1] is not None and 0.015 <= b[2] < 0.035:
                    rxb = [org[0] + SLOT_RELEASE[b[0]][0], org[1] + SLOT_RELEASE[b[0]][1]]
                    _reseat(robot, rxb, b[1], rpy, log)
                    fixed = True
            if fixed:
                _move(robot, [bt[0], bt[1], Z_TRANSIT], -1.0, rpy, tol=0.006, vtol=0.003, log=log, name="to_button2")
        reason, n, p = _move(robot, [bt[0], bt[1], Z_PRESS], -1.0, rpy, tol=0.003, vtol=0.0015, max_steps=40, stall=True, log=log, name="press")
        zmin = float(p[2])
        _move(robot, [bt[0], bt[1], Z_TRANSIT - 0.03], -1.0, rpy, tol=0.02, vtol=0.02, log=log, name="press_up")
    return {"code": "OK", "index": index, "lamp": "on" if lamp_max > 20 else ("off" if lamp_max >= 0 else "unseen"),
            "lamp_max": lamp_max, "press_z": zmin, "slot_ok": slot_ok, "steps": int(robot.now().step) - step0,
            "end_step": int(robot.now().step), "log": log}


def act_wait(robot, n=5):
    """Hold still with open gripper for n native steps (used when perception fails)."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"code": "WAITED", "step": int(robot.now().step)}


def act_press_button(robot, org, rpy):
    """Extra confirmation press of the button (closed gripper)."""
    return act_cycle(robot, 0, org, rpy, start="button")


# ---------------- decision -----------------
def perceive_decide(obs, goal, memory, last):
    """Detect the layout once, then process the six batteries in demo order."""
    mem = dict(memory)
    for k in ("org", "rpy", "next_index", "stage", "lamps", "retries", "log", "detect_info"):
        mem.setdefault(k, None)
    if mem["next_index"] is None:
        mem["next_index"] = 0
        mem["lamps"] = []
        mem["retries"] = 0
        mem["stage"] = "pick"
        mem["log"] = ""
    cur = obs.current
    if mem["org"] is None:
        info = _detect_layout(obs.frames[0] if len(obs.frames) else cur)
        if info["org"] is None:
            info = _detect_layout(cur)
        mem["detect_info"] = "%s n=%d res=%s" % (info["method"], info["n_clusters"], info["btn_residual"])
        if info["org"] is None:
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 5}}}
        mem["org"] = info["org"]
        t = cur.tcp
        mem["rpy"] = [float(t[3]), float(t[4]), float(t[5])]
    elif last is not None:
        text = str(last)
        if text.startswith("act_press_button") or text.startswith("act_wait"):
            mem["retries"] = int(mem["retries"] or 0) + 1
        elif '"code": "OK"' in text:
            lamp = "on" if '"lamp": "on"' in text else ("off" if '"lamp": "off"' in text else "unseen")
            mem["lamps"] = list(mem["lamps"] or []) + [lamp]
            mem["next_index"] = int(mem["next_index"]) + 1
            mem["stage"] = "pick"
            mem["retries"] = 0
        elif "FAIL_" in text:
            mem["retries"] = int(mem["retries"] or 0) + 1
            fing = float(np.sum(cur.fingers))
            if "FAIL_pick" in text:
                mem["stage"] = "pick"
            elif "FAIL_insert" in text:
                mem["stage"] = "socket" if fing > 0.006 else "pick"
            elif "FAIL_regrasp" in text:
                mem["stage"] = "regrasp"
            elif "FAIL_return" in text:
                mem["stage"] = "return" if fing > 0.006 else "button"
        mem["log"] = (str(mem["log"] or "") + " | " + text[:160])[-1500:]
    if int(mem["next_index"]) >= 6:
        # all handled but no native success yet: at most two extra confirmation presses
        extra = int(mem["retries"] or 0)
        if extra < 2:
            return {"memory": mem, "act": {"tool": "act_press_button", "args": {"org": mem["org"], "rpy": mem["rpy"]}}}
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 20}}}
    stage = mem["stage"] if mem["stage"] in ("pick", "socket", "regrasp", "return", "button") else "pick"
    if int(mem["retries"] or 0) > 3:
        # give up on this battery and move on
        mem["next_index"] = int(mem["next_index"]) + 1
        mem["retries"] = 0
        stage = "pick"
        if int(mem["next_index"]) >= 6:
            return {"memory": mem, "act": {"tool": "act_press_button", "args": {"org": mem["org"], "rpy": mem["rpy"]}}}
    return {"memory": mem, "act": {"tool": "act_cycle", "args": {
        "index": int(mem["next_index"]), "org": mem["org"], "rpy": mem["rpy"], "start": stage}}}
