"""TraceShapeSeqMedium policy: watch red cube trace shapes, re-trace them with the green cube, press button.

Stages (memory.stage): watch -> grasp -> trace -> press -> done.
Perception is colour segmentation + depth deprojection on the fixed front camera.
"""

SHAPE_FRAMES = 64          # cue frames per shape (measured in all 50 demos)
CARRY_Z = 0.06             # TCP height while tracing (expert used ~0.06)
GRASP_Z = 0.031            # TCP height for grasping (expert used ~0.031)
HOVER_Z = 0.12
PRESS_Z = 0.072            # first press height (TCP); success seen at TCP z~0.083 in demos
PRESS_Z_MIN = 0.045
POS_SCALE = 0.1            # metres per unit translation command
SS_GAIN = 0.041            # steady-state m/step per unit command (fit on demos)
TRACE_SPEED = 0.009        # carrot speed m/step along shapes
TRANSIT_SPEED = 0.012


# ---------------------------------------------------------------- geometry helpers
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    s = math.sin(ang)
    if s < 1e-4:  # near pi
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        ax = V[:, int(np.argmax(w))]
        return ax * ang
    return v * (ang / (2.0 * s))


def _rot_cmd(tcp, Rt, gain=0.5):
    """Rotation command (drx,dry,drz) that turns the measured orientation toward Rt.
    Native convention (checked on demos): world angular offset = -0.1*r."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _rotvec(Rt @ Rc.T)
    r = -gain * w / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _deproject(K, E, us, vs, dmm):
    z = dmm / 1000.0
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


# ---------------------------------------------------------------- perception helpers
def _masks(img):
    im = img.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (np.abs(g - b) <= 8) & (g > 30) & (r * 100 > 175 * g) & (r * 100 < 225 * g)
    green = (np.abs(r - b) <= 8) & (r > 30) & (g * 100 > 175 * r) & (g * 100 < 225 * r)
    button = (np.abs(g - b) <= 8) & (g > 40) & (r * 100 > 140 * g) & (r * 100 < 172 * g)
    return red, green, button


def _lamp(img):
    im = img.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = int(((r > 230) & (g < 40) & (b < 40)).sum())
    green = int(((g > 230) & (r < 40) & (b < 40)).sum())
    return red, green


def _top_center(mask, depth, K, E, band=0.004, min_px=3):
    d = depth.astype(np.float64)
    m = mask & (d > 0)
    n = int(m.sum())
    if n < min_px:
        return None, n
    vs, us = np.nonzero(m)
    P = _deproject(K, E, us.astype(np.float64), vs.astype(np.float64), d[vs, us])
    zt = float(np.percentile(P[:, 2], 90))
    top = P[P[:, 2] > zt - band]
    return top.mean(0), n


def _frame_objects(fr, which=("red", "green", "button")):
    img = fr.front_rgb
    red, green, button = _masks(img)
    K, E, D = fr.front_K, fr.front_extrinsic, fr.front_depth
    out = {}
    if "red" in which:
        out["red"] = _top_center(red, D, K, E)[0]
    if "green" in which:
        out["green"] = _top_center(green, D, K, E)[0]
    if "button" in which:
        out["button"] = _top_center(button, D, K, E, band=0.003, min_px=8)[0]
    return out


def _smooth_shape(P):
    """Local linear fit (window +-2 frames) of an equally timed track; keeps corners sharp-ish."""
    n = len(P)
    out = []
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        idx = np.arange(lo, hi, dtype=np.float64)
        Q = P[lo:hi]
        if len(Q) < 3:
            out.append(P[i])
            continue
        A = np.stack([np.ones_like(idx), idx - i], 1)
        c = np.linalg.lstsq(A, Q, rcond=None)[0]
        out.append(c[0])
    return np.array(out)


def _extract_shapes(frames):
    """Red-cube track from cue frames -> list of shapes (each a list of [x,y])."""
    track = []
    g_idx = None
    for i, fr in enumerate(frames):
        lr, lg = _lamp(fr.front_rgb)
        if lg > 30:
            g_idx = i
            break
        red, _, _ = _masks(fr.front_rgb)
        p, n = _top_center(red, fr.front_depth, fr.front_K, fr.front_extrinsic)
        track.append(None if p is None else [float(p[0]), float(p[1])])
    if g_idx is None:
        return None, "lamp never green"
    # fill gaps by nearest valid neighbour
    valid = [i for i, p in enumerate(track) if p is not None]
    if len(valid) < 10:
        return None, "red cube not tracked"
    for i in range(len(track)):
        if track[i] is None:
            j = min(valid, key=lambda k: abs(k - i))
            track[i] = list(track[j])
    T = np.array(track)
    # motion start: first frame whose successor moved > 3 mm and keeps moving
    sp = np.linalg.norm(np.diff(T, axis=0), axis=1)
    s = None
    for i in range(len(sp) - 3):
        if sp[i] > 0.003 and sp[i + 1] > 0.003 and sp[i + 2] > 0.003:
            s = i
            break
    if s is None:
        return None, "no red motion"
    n_sh = int(round((g_idx - s) / float(SHAPE_FRAMES)))
    n_sh = max(1, min(8, n_sh))
    first = g_idx - n_sh * SHAPE_FRAMES
    bounds = [max(0, first + k * SHAPE_FRAMES) for k in range(n_sh)] + [g_idx]
    # refine internal boundaries to the largest jump nearby (if a clear teleport exists)
    for k in range(1, n_sh):
        b = bounds[k]
        best, bj = 0.0, None
        for j in range(max(1, b - 3), min(len(T), b + 4)):
            jump = float(np.linalg.norm(T[j] - T[j - 1]))
            if jump > best:
                best, bj = jump, j
        if bj is not None and best > 0.03:
            bounds[k] = bj
    shapes = []
    for k in range(n_sh):
        P = T[bounds[k]:bounds[k + 1]]
        if len(P) < 5:
            continue
        Ps = _smooth_shape(P)
        pts = [Ps[i] for i in range(0, len(Ps), 2)]
        if (len(Ps) - 1) % 2 != 0:
            pts.append(Ps[-1])
        shapes.append([[round(float(q[0]), 4), round(float(q[1]), 4)] for q in pts])
    info = "g_idx=%d start=%d n_shapes=%d bounds=%s" % (g_idx, s, n_sh, bounds)
    return shapes, info


# ---------------------------------------------------------------- motion helpers
def _tcp(robot):
    return np.array(robot.now().tcp, dtype=np.float64)


def _fingers(robot):
    return float(np.mean(robot.now().fingers))


def _step_toward(robot, target, Rt, grip, kp=0.7, vmax=0.6, ff=None):
    """One control step toward target xyz (P control in command units) with orientation hold."""
    tcp = _tcp(robot)
    e = np.array(target, dtype=np.float64) - tcp[:3]
    a = kp * e / POS_SCALE
    if ff is not None:
        a = a + np.array(ff, dtype=np.float64) / SS_GAIN
    n = float(np.linalg.norm(a))
    if n > vmax:
        a = a * (vmax / n)
    r, _ = _rot_cmd(tcp, Rt)
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    return float(np.linalg.norm(e))


def _move_to(robot, target, Rt, grip, tol=0.004, max_steps=150, vmax=0.6, settle=2):
    ok = 0
    n = 0
    prev = _tcp(robot)[:3]
    for n in range(1, max_steps + 1):
        err = _step_toward(robot, target, Rt, grip, vmax=vmax)
        cur = _tcp(robot)[:3]
        spd = float(np.linalg.norm(cur - prev))
        prev = cur
        err = float(np.linalg.norm(np.array(target) - cur))
        if err < tol and spd < 0.002:
            ok += 1
            if ok >= settle:
                return True, n, err
        else:
            ok = 0
    return False, n, err


def _hold(robot, Rt, grip, steps):
    tgt = _tcp(robot)[:3].copy()
    for _ in range(steps):
        _step_toward(robot, tgt, Rt, grip)


def _home_R(tcp):
    # keep the yaw measured, pitch->0 roll->pi (vertical gripper)
    return _rpy_to_R(math.pi, 0.0, float(tcp[5]))


NEAR_X = -0.31             # closer to the base than this, tilt fingers toward the base (relieves J4 limit)


def _near_pitch(x):
    """Positive pitch (fingers toward base) for targets near the robot; offline IK: J4 margin 0.02 rad at
    x=-0.33 vertical vs 0.14 rad with pitch 0.2."""
    return float(min(0.3, max(0.0, (NEAR_X - float(x)) * 6.0)))


def _R_at(x, yaw):
    return _rpy_to_R(math.pi, _near_pitch(x), yaw)


# ---------------------------------------------------------------- policy
def perceive_decide(obs, goal, memory, last):
    """Decide the next stage from observations and memory."""
    mem = dict(memory)
    for k in ("stage", "path", "shape_starts", "green_cube", "button", "notes", "grasp_tries", "n_shapes",
              "press_count"):
        mem.setdefault(k, None)
    cur = obs.current
    stage = mem["stage"] or "watch"
    notes = mem["notes"] or ""
    # remember button / cube from the first frame where they are visible
    if mem["button"] is None or mem["green_cube"] is None:
        o = _frame_objects(obs.frames[0])
        if mem["button"] is None and o["button"] is not None:
            mem["button"] = [round(float(v), 4) for v in o["button"]]
        if mem["green_cube"] is None and o["green"] is not None:
            mem["green_cube"] = [round(float(v), 4) for v in o["green"]]
    if stage == "watch":
        lr, lg = _lamp(cur.front_rgb)
        greens = [i for i, f in enumerate(obs.frames) if _lamp(f.front_rgb)[1] > 30]
        if not greens:
            mem["stage"] = "watch"
            mem["notes"] = notes
            mem["grasp_tries"] = mem["grasp_tries"] or 0
            mem["n_shapes"] = mem["n_shapes"] or 0
            return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 800}}}
        shapes, info = _extract_shapes(list(obs.frames))
        notes += " cue:" + info + ";"
        if shapes is None:
            mem["stage"] = "watch"
            mem["notes"] = notes
            return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 1}}}
        path, starts = [], []
        for sh in shapes:
            starts.append(str(len(path)))
            path.extend(sh)
        mem["path"] = path
        mem["shape_starts"] = starts
        mem["n_shapes"] = len(shapes)
        stage = "grasp"
    if stage == "grasp":
        tries = int(mem["grasp_tries"] or 0)
        if last is not None and last.startswith("act_grasp") and '"grasp_ok": true' in last:
            stage = "trace"
        else:
            if tries > 0:
                o = _frame_objects(cur, ("green",))
                if o["green"] is not None:
                    mem["green_cube"] = [round(float(v), 4) for v in o["green"]]
            if tries >= 4 or mem["green_cube"] is None:
                stage = "press"
            else:
                mem["grasp_tries"] = tries + 1
                mem["stage"] = "grasp"
                mem["notes"] = notes
                g = mem["green_cube"]
                return {"memory": mem, "act": {"tool": "act_grasp", "args": {"x": g[0], "y": g[1]}}}
    if stage == "trace":
        if last is not None and last.startswith("act_trace"):
            if '"lost_cube": true' in last:
                stage = "grasp"
                mem["grasp_tries"] = 1
                mem["stage"] = stage
                mem["notes"] = notes + " cube lost during trace;"
                o = _frame_objects(cur, ("green",))
                if o["green"] is not None:
                    mem["green_cube"] = [round(float(v), 4) for v in o["green"]]
                g = mem["green_cube"]
                return {"memory": mem, "act": {"tool": "act_grasp", "args": {"x": g[0], "y": g[1]}}}
            stage = "press"
        else:
            mem["stage"] = "trace"
            mem["notes"] = notes
            starts = [int(s) for s in mem["shape_starts"]]
            return {"memory": mem, "act": {"tool": "act_trace", "args": {"path": mem["path"], "shape_starts": starts}}}
    if stage == "press":
        b = mem["button"]
        if b is None:
            o = _frame_objects(cur, ("button",))
            b = None if o["button"] is None else [float(v) for v in o["button"]]
        mem["stage"] = "press"
        mem["notes"] = notes
        if b is None:
            return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 5}}}
        presses = int(mem["press_count"] or 0)
        mem["press_count"] = presses + 1
        return {"memory": mem, "act": {"tool": "act_press", "args": {"x": b[0], "y": b[1], "z_top": b[2], "deeper": presses}}}
    mem["stage"] = stage
    mem["notes"] = notes
    return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 5}}}


def act_watch(robot, max_steps=800):
    """Hold still (zero motion, gripper open command) until the lamp turns green or max_steps pass."""
    n = 0
    for n in range(1, int(max_steps) + 1):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        if _lamp(robot.now().front_rgb)[1] > 30:
            break
    lr, lg = _lamp(robot.now().front_rgb)
    return {"steps": n, "lamp_green_px": lg, "lamp_red_px": lr, "tcp": _tcp(robot).round(4).tolist()}


def act_grasp(robot, x, y):
    """Open, hover above (x,y), refine with wrist camera, descend, close, lift to carry height."""
    tcp0 = _tcp(robot)
    Rt = _R_at(x, float(tcp0[5]))
    log = {}
    ok, n1, e1 = _move_to(robot, [x, y, HOVER_Z], Rt, 1.0, tol=0.006, max_steps=120, vmax=0.8)
    log["hover"] = [ok, n1, round(e1, 4)]
    # refine cube position with the wrist camera (closer, higher resolution)
    fr = robot.now()
    _, gm, _ = _masks(fr.wrist_rgb)
    p, npx = _top_center(gm, fr.wrist_depth, fr.wrist_K, fr.wrist_extrinsic)
    if p is not None and npx > 30 and math.hypot(p[0] - x, p[1] - y) < 0.04:
        log["wrist_refine"] = [round(float(p[0] - x), 4), round(float(p[1] - y), 4), npx]
        x, y = float(p[0]), float(p[1])
        ok, n1b, e1 = _move_to(robot, [x, y, HOVER_Z], Rt, 1.0, tol=0.004, max_steps=60)
    ok, n2, e2 = _move_to(robot, [x, y, GRASP_Z + 0.03], Rt, 1.0, tol=0.005, max_steps=80)
    ok, n3, e3 = _move_to(robot, [x, y, GRASP_Z], Rt, 1.0, tol=0.003, max_steps=80, vmax=0.3)
    log["descend"] = [ok, n2 + n3, round(e3, 4)]
    for _ in range(12):
        _step_toward(robot, [x, y, GRASP_Z], Rt, -1.0)
    f_closed = _fingers(robot)
    ok, n4, e4 = _move_to(robot, [x, y, CARRY_Z], Rt, -1.0, tol=0.004, max_steps=80, vmax=0.4)
    f_lift = _fingers(robot)
    grasp_ok = bool(f_lift > 0.008)
    log.update({"fingers_closed": round(f_closed, 4), "fingers_lift": round(f_lift, 4), "grasp_ok": grasp_ok,
                "tcp": _tcp(robot).round(4).tolist(), "x": round(x, 4), "y": round(y, 4)})
    return log


def act_trace(robot, path, shape_starts):
    """Carry the grasped cube along each shape polyline in order (carrot follower at CARRY_Z)."""
    tcp0 = _tcp(robot)
    yaw0 = float(tcp0[5])
    P = np.array(path, dtype=np.float64)
    stalls = 0
    starts = list(shape_starts) + [len(P)]
    report = []
    lost = False
    total = 0
    for k in range(len(starts) - 1):
        S = P[starts[k]:starts[k + 1]]
        if len(S) == 0:
            continue
        ok, n0, e0 = _move_to(robot, [S[0, 0], S[0, 1], CARRY_Z], _R_at(S[0, 0], yaw0), -1.0, tol=0.004,
                              max_steps=150, vmax=0.5)
        total += n0
        seg = np.linalg.norm(np.diff(S, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(cum[-1])
        s = 0.0
        maxerr = 0.0
        steps = 0
        wait = 0
        while True:
            tcp = _tcp(robot)
            if s >= L:
                break
            # carrot position
            j = int(np.searchsorted(cum, s, side="right") - 1)
            j = min(max(j, 0), len(S) - 2)
            u = (s - cum[j]) / max(seg[j], 1e-9)
            c = S[j] + u * (S[j + 1] - S[j])
            dirv = (S[j + 1] - S[j]) / max(seg[j], 1e-9)
            err = float(np.linalg.norm(c - tcp[:2]))
            maxerr = max(maxerr, err)
            v = TRACE_SPEED if err < 0.008 else (TRACE_SPEED * 0.4 if err < 0.015 else 0.0)
            if v == 0.0:
                wait += 1
                if wait > 30:          # arm cannot follow (e.g. joint limit): keep advancing slowly
                    v = TRACE_SPEED * 0.3
                    stalls += 1
            else:
                wait = 0
            s_next = min(L, s + v)
            j2 = int(np.searchsorted(cum, s_next, side="right") - 1)
            j2 = min(max(j2, 0), len(S) - 2)
            u2 = (s_next - cum[j2]) / max(seg[j2], 1e-9)
            c2 = S[j2] + u2 * (S[j2 + 1] - S[j2])
            ffv = (c2 - c)
            _step_toward(robot, [c2[0], c2[1], CARRY_Z], _R_at(c2[0], yaw0), -1.0, kp=0.7, vmax=0.6,
                         ff=[ffv[0] * 0.8, ffv[1] * 0.8, 0.0])
            s = s_next
            steps += 1
            if _fingers(robot) < 0.006:
                lost = True
                break
            if steps > 600:
                break
        if lost:
            report.append({"shape": k, "lost": True})
            break
        ok, n2, e2 = _move_to(robot, [S[-1, 0], S[-1, 1], CARRY_Z], _R_at(S[-1, 0], yaw0), -1.0, tol=0.004,
                              max_steps=40, vmax=0.3)
        total += steps + n2
        report.append({"shape": k, "pts": int(len(S)), "len": round(L, 3), "steps": steps,
                       "max_err": round(maxerr, 4), "end_err": round(e2, 4)})
    return {"lost_cube": lost, "shapes": report, "steps": total, "stall_steps": stalls, "fingers": round(_fingers(robot), 4),
            "tcp": _tcp(robot).round(4).tolist()}


def act_press(robot, x, y, z_top=0.06, deeper=0):
    """Release cube, lift, move above the button at (x,y), close gripper and press down."""
    tcp0 = _tcp(robot)
    Rt = _home_R(tcp0)
    log = {}
    if _fingers(robot) > 0.004 and tcp0[2] < 0.12:
        _hold(robot, Rt, 1.0, 10)          # open: release the cube near the table
    ok, n1, e1 = _move_to(robot, [tcp0[0], tcp0[1], 0.16], Rt, 1.0, tol=0.01, max_steps=60, vmax=0.6)
    _hold(robot, Rt, -1.0, 3)
    # choose an orientation that can reach the press point
    pz = PRESS_Z if deeper == 0 else PRESS_Z - 0.012 * deeper
    pz = max(pz, PRESS_Z_MIN)
    pitch_used = 0.0
    Rp = Rt
    for pitch in (0.0, -0.25, -0.45, -0.65):
        yaw = float(tcp0[5])
        chk = robot.dry_run(float(x), float(y), float(pz), math.pi, pitch, yaw)
        if chk.get("reachable"):
            pitch_used = pitch
            Rp = _rpy_to_R(math.pi, pitch, yaw)
            break
    log["press_pitch"] = pitch_used
    ok, n2, e2 = _move_to(robot, [x, y, 0.16], Rp, -1.0, tol=0.006, max_steps=150, vmax=0.8)
    log["above"] = [ok, n2, round(e2, 4)]
    ok, n3, e3 = _move_to(robot, [x, y, pz + 0.03], Rp, -1.0, tol=0.006, max_steps=60, vmax=0.6)
    ok, n4, e4 = _move_to(robot, [x, y, pz], Rp, -1.0, tol=0.003, max_steps=60, vmax=0.4)
    log["press"] = [ok, n3 + n4, round(e4, 4), _tcp(robot).round(4).tolist()]
    _hold(robot, Rp, -1.0, 15)
    log["after_hold"] = _tcp(robot).round(4).tolist()
    # come back up a little (in case success requires release)
    ok, n5, e5 = _move_to(robot, [x, y, pz + 0.05], Rp, -1.0, tol=0.006, max_steps=40, vmax=0.5)
    log["lifted"] = _tcp(robot).round(4).tolist()
    return log
