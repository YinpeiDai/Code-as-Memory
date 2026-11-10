"""InterceptMedium policy: perceive the target disc and the rolling red ball in the
front RGB-D camera, predict when the ball reaches a chosen contact point on its
straight path, and sweep the front face of one (vertical, open) finger through that
point with the ball-relative timing and offsets measured in the expert demonstrations,
so the ball is deflected toward the target disc. All state lives in declared memory."""

# ---------------------------------------------------------------- constants
BALL_R = 0.02            # measured by sphere fits on demo depth (0.0197)
TARGET_R = 0.1           # outer ring radius measured in demos (0.097-0.100)
LINE_DX = -0.200         # ball spawn line x relative to target centre (demo: -0.197..-0.202)
DY_NOMINAL = 0.15        # default contact point: y = target_y - dy (before the ball is tracked)
DY_MIN = 0.07            # range of contact offsets searched (expert contacts: 0.06..0.19)
DY_MAX = 0.17
MIN_CONTACT_STEP = 16.0  # contact earlier than this cannot be reached in time (descent+approach)
ROT_RATE = 0.036         # rad/step achievable in free space at full rotation command (demos)
NORMAL_DEG = -45.8       # face-normal (TCP x axis) angle in xy for ORI
OUT_SPEED = 0.40         # deflected speed / incoming speed (demos 0.33-0.74, median 0.49; conservative)
END_BUDGET = 56.0        # native step limit is 60
PHI_BIAS = 87.0          # demos: outgoing ball angle - face-normal angle, mean 87 deg (sd ~11)
ORI = [3.14159, 0.0, -0.80]   # vertical fingers; face normal (TCP x) at -46 deg in xy (expert: -43 deg, tilted)
Z_LOW = 0.020            # TCP height while sweeping (fingertips just above the table)
Z_RETREAT = 0.12
KP = 22.0                # translation gain: command units per metre of error
KR = 22.0                # rotation gain: command units per radian of error
TRANS_NORM_MAX = 1.0     # max norm of the translation command (expert median start ~1.1)
V_REF = 0.0225           # median free-rolling ball speed (m/step) of the demos
FIT_POINTS = 10          # latest detections used for the line fit
# expert median TCP position relative to the ball contact point (ball centre at contact),
# for phase = steps relative to contact, -6..5 (scale applied around phase 0)
OFF_PHASE = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
OFF_X = [-0.037, -0.027, -0.018, -0.013, -0.004, 0.003, 0.012, 0.021, 0.033, 0.048, 0.060, 0.069]
OFF_Y = [-0.019, -0.005, 0.009, 0.022, 0.035, 0.048, 0.060, 0.076, 0.092, 0.106, 0.117, 0.127]
RETREAT_V = [0.009, 0.010]   # keep the sweep velocity for 2 steps while lifting, then hold xy
RETREAT_XY_STEPS = 2.0
RETREAT_DZ = 0.025       # lift so the fingertips clear the ball top (0.04 m) in ~1-2 steps
WAIT_PHASE = -4.0        # hold at the phase -4 offset until the sweep starts (less reach toward the base)
BASE_XY = [-0.615, 0.0]  # robot base axis (from demo kinematics: tcp = base + FK(q))
REACH_MIN = 0.28
REACH_HIGH_EXTRA = 0.03


# ---------------------------------------------------------------- geometry helpers
def _rpy2R(r, p, y):
    cx, sx = math.cos(r), math.sin(r)
    cy, sy = math.cos(p), math.sin(p)
    cz, sz = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    if ang > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / math.sqrt(max(A[k, k], 1e-9))
        return ax * ang
    return v * (ang / (2.0 * math.sin(ang)))


def _unproject(us, vs, d_mm, K, E):
    z = d_mm / 1000.0
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


def _cam_pos(E):
    R = E[:, :3]
    t = E[:, 3]
    return -(R.T @ t)


def _sphere_fit(P):
    A = np.concatenate([2.0 * P, np.ones((len(P), 1))], 1)
    b = (P ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    r2 = sol[3] + float(c @ c)
    return c, (math.sqrt(r2) if r2 > 0 else 0.0)


# ---------------------------------------------------------------- perception
def _ball_from_camera(frame, cam):
    rgb = getattr(frame, cam + "_rgb", None)
    dep = getattr(frame, cam + "_depth", None)
    K = getattr(frame, cam + "_K", None)
    E = getattr(frame, cam + "_extrinsic", None)
    if rgb is None or dep is None or K is None or E is None:
        return None
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 90) & (g < 45) & (b < 45) & (r > 3 * g + 20) & (dep > 0)
    n = int(m.sum())
    if n < 6:
        return None
    vs, us = np.nonzero(m)
    H, W = m.shape
    if us.min() <= 1 or vs.min() <= 1 or us.max() >= W - 2 or vs.max() >= H - 2:
        return None  # cut by the image border: centroid/sphere fit would be biased
    P = _unproject(us.astype(float), vs.astype(float), dep[vs, us].astype(float),
                   np.asarray(K, float), np.asarray(E, float))
    P = P[(P[:, 2] > -0.01) & (P[:, 2] < 0.06)]
    if len(P) < 6:
        return None
    c, rad = _sphere_fit(P)
    if not (0.014 < rad < 0.027 and 0.0 < c[2] < 0.045):
        mean = P.mean(0)
        u = mean - _cam_pos(np.asarray(E, float))
        nu = float(np.linalg.norm(u))
        c = mean + 0.012 * u / max(nu, 1e-6)
    return [float(c[0]), float(c[1]), float(c[2]), n]


def _detect_ball(frame):
    out = _ball_from_camera(frame, "front")
    if out is None:
        out = _ball_from_camera(frame, "wrist")
    return out


def _detect_target(frame):
    rgb = getattr(frame, "front_rgb", None)
    dep = getattr(frame, "front_depth", None)
    if rgb is None or dep is None:
        return None
    K = np.asarray(frame.front_K, float)
    E = np.asarray(frame.front_extrinsic, float)
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 150) & (np.abs(b - g) < 25) & (r - g > 90) & (dep > 0)
    if int(m.sum()) < 30:
        return None
    vs, us = np.nonzero(m)
    P = _unproject(us.astype(float), vs.astype(float), dep[vs, us].astype(float), K, E)
    P = P[(P[:, 2] < 0.015) & (P[:, 2] > -0.01)]
    if len(P) < 30:
        return None
    lo = P.min(0)
    hi = P.max(0)
    cx = 0.5 * (lo[0] + hi[0])
    cy = 0.5 * (lo[1] + hi[1])
    # partial occlusion: trust the side whose extent matches the known radius
    if hi[0] - lo[0] < 2 * TARGET_R - 0.01:
        med = float(np.median(P[:, 0]))
        cx = lo[0] + TARGET_R if med > cx else hi[0] - TARGET_R
    if hi[1] - lo[1] < 2 * TARGET_R - 0.01:
        med = float(np.median(P[:, 1]))
        cy = lo[1] + TARGET_R if med > cy else hi[1] - TARGET_R
    return [float(cx), float(cy), len(P)]


# ---------------------------------------------------------------- memory state
FIELDS = ["target", "target_px", "track", "post", "contact_dy", "contact_step",
          "contact_point", "stage", "first_step", "last_step", "off_count", "notes", "decisions"]


def _fresh_state():
    return {"target": None, "target_px": 0, "track": [], "post": [], "contact_dy": None,
            "contact_step": None, "contact_point": None, "stage": "approach",
            "first_step": None, "last_step": -1, "off_count": 0, "notes": "", "decisions": 0}


def _state_from_memory(memory):
    st = _fresh_state()
    if isinstance(memory, dict):
        for k in FIELDS:
            v = memory.get(k)
            if v is not None:
                st[k] = v
    st["track"] = [list(p) for p in (st["track"] or [])]
    st["post"] = [list(p) for p in (st["post"] or [])]
    return st


def _fit(track):
    """Robust linear fit x(s), y(s) of the latest free-rolling detections.
    Returns [s_ref, x(s_ref), y(s_ref), vx, vy] or None."""
    if not track:
        return None
    arr = np.array(track, dtype=float)[-FIT_POINTS:]
    s = arr[:, 0]
    if len(arr) == 1:
        return [float(s[0]), float(arr[0, 1]), float(arr[0, 2]), 0.0, V_REF]
    s_ref = float(s[-1])
    keep = np.ones(len(arr), dtype=bool)
    cx = np.array([arr[-1, 1], 0.0])
    cy = np.array([arr[-1, 2], V_REF])
    for it in range(3):
        A = np.stack([np.ones(int(keep.sum())), s[keep] - s_ref], 1)
        if int(keep.sum()) < 2 or float(np.ptp(s[keep])) < 1:
            break
        cx = np.linalg.lstsq(A, arr[keep, 1], rcond=None)[0]
        cy = np.linalg.lstsq(A, arr[keep, 2], rcond=None)[0]
        rx = arr[:, 1] - (cx[0] + cx[1] * (s - s_ref))
        ry = arr[:, 2] - (cy[0] + cy[1] * (s - s_ref))
        new_keep = (np.abs(rx) < 0.006) & (np.abs(ry) < 0.006)
        if int(new_keep.sum()) < max(2, len(arr) // 2) or bool(np.all(new_keep == keep)):
            break
        keep = new_keep
    vy = float(cy[1])
    vx = float(cx[1])
    if int(keep.sum()) < 3 or float(np.ptp(s[keep])) < 2:
        vx = 0.0
        vy = min(max(vy, 0.015), 0.035)
    return [s_ref, float(cx[0]), float(cy[0]), vx, vy]


def _predict(fit, s):
    return [fit[1] + fit[3] * (s - fit[0]), fit[2] + fit[4] * (s - fit[0])]


def _update(st, frame):
    """Fold one observation frame into the episode state (idempotent per step)."""
    s = int(frame.step)
    if s <= int(st["last_step"]):
        return st
    st["last_step"] = s
    if st["first_step"] is None:
        st["first_step"] = s
    # target: detect early; keep the most complete detection
    if st["target"] is None or int(st["target_px"] or 0) < 500:
        t = _detect_target(frame)
        if t is not None and t[2] > int(st["target_px"] or 0):
            st["target"] = [t[0], t[1]]
            st["target_px"] = int(t[2])
    ball = _detect_ball(frame)
    if ball is not None:
        if st["contact_step"] is None:
            fit = _fit(st["track"])
            ok = True
            if fit is not None and len(st["track"]) >= 3:
                px, py = _predict(fit, s)
                ok = abs(px - ball[0]) < 0.015 and abs(py - ball[1]) < 0.02
            if ok:
                st["track"].append([s, round(ball[0], 5), round(ball[1], 5)])
                st["track"] = st["track"][-25:]
                st["off_count"] = 0
            else:
                st["off_count"] = int(st["off_count"] or 0) + 1
                st["notes"] = "ball off predicted line at step %d (dx %.3f, dy %.3f)" % (
                    s, ball[0] - px, ball[1] - py)
                tcp = np.asarray(frame.tcp, float)
                near = math.hypot(float(tcp[0]) - ball[0], float(tcp[1]) - ball[1]) < 0.15
                deflected = (ball[0] - px > 0.008) or (py - ball[1] > 0.015)
                if near and deflected and st["off_count"] >= 2 and len(st["track"]) >= 5:
                    # deflected before the planned contact: freeze contact now
                    st["contact_step"] = float(s)
                    st["contact_point"] = [round(ball[0], 5), round(ball[1], 5)]
                    st["stage"] = "early_contact"
                elif not near and st["off_count"] >= 2:
                    # paddle far away: the fit was wrong, restart it from recent points
                    st["track"] = st["track"][-1:] + [[s, round(ball[0], 5), round(ball[1], 5)]]
                    st["off_count"] = 0
                    st["notes"] += "; track reset"
        else:
            st["post"].append([s, round(ball[0], 5), round(ball[1], 5)])
            st["post"] = st["post"][-20:]
    _plan_contact(st, s, frame)
    return st


def _rot_error(frame):
    tcp = np.asarray(frame.tcp, float)
    Rc = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy2R(ORI[0], ORI[1], ORI[2])
    return float(np.linalg.norm(_rotvec(Rd @ Rc.T)))


def _dy_margin(T, fit, dy, s, rot_err, s_first):
    """Angular margin (deg) of the predicted deflection for contact offset dy.
    Model (from demos): outgoing angle = face-normal angle + PHI_BIAS; the face normal
    lags its goal by the rotation still left at contact (rotation rate ROT_RATE)."""
    y_c = T[1] - dy
    s_c = fit[0] + (y_c - fit[2]) / fit[4]
    left = max(0.0, rot_err - ROT_RATE * max(0.0, s_c - s))
    nang = NORMAL_DEG + math.degrees(left)
    phi = nang + PHI_BIAS
    x_c = fit[1] + fit[3] * (s_c - fit[0])
    px, py = T[0] - x_c, dy
    dist = math.hypot(px, py)
    ideal = math.degrees(math.atan2(py, px))
    tol = math.degrees(math.asin(min(1.0, TARGET_R / max(dist, 1e-6))))
    margin = tol - abs(phi - ideal)
    # time budget: the deflected ball rolls at ~half its speed (demos 0.33-0.74)
    s_end = s_c + max(0.0, dist - 0.09) / max(OUT_SPEED * fit[4], 1e-4)
    if s_end > s_first + END_BUDGET:
        margin -= 3.0 * (s_end - s_first - END_BUDGET)
    if s_c < s_first + MIN_CONTACT_STEP:
        margin -= 100.0 + (s_first + MIN_CONTACT_STEP - s_c)
    return margin, s_c


def _plan_contact(st, s, frame):
    T = st["target"]
    if T is None or st["contact_step"] is not None:
        return
    fit = _fit(st["track"])
    if fit is None or len(st["track"]) < 2:
        return
    vy = fit[4]
    if vy < 0.008:
        return
    s_first = int(st["first_step"] or 0)
    dy = st["contact_dy"]
    s_c_cur = None
    if dy is not None:
        s_c_cur = fit[0] + (T[1] - dy - fit[2]) / vy
    # (re)choose the contact offset every frame while the current plan is > 4 steps
    # away; only switch to candidates whose contact is still > 6 steps away
    if dy is None or s_c_cur - s > 4:
        rot_err = _rot_error(frame)
        best = None
        for i in range(int(round((DY_MAX - DY_MIN) / 0.01)) + 1):
            cand = DY_MIN + 0.01 * i
            m, sc = _dy_margin(T, fit, cand, s, rot_err, s_first)
            if dy is not None and sc - s <= 6 and abs(cand - dy) > 1e-6:
                continue
            if dy is not None and abs(cand - dy) < 1e-6:
                m += 0.5  # hysteresis
            if best is None or m > best[0] + 1e-9:
                best = (m, cand)
        if best is not None:
            st["contact_dy"] = round(float(best[1]), 5)
    y_c = T[1] - float(st["contact_dy"])
    s_c = fit[0] + (y_c - fit[2]) / vy
    if s + 1 >= s_c:
        x_c = fit[1] + fit[3] * (s_c - fit[0])
        st["contact_step"] = round(float(s_c), 4)
        st["contact_point"] = [round(float(x_c), 5), round(float(y_c), 5)]
        st["stage"] = "contact"


# ---------------------------------------------------------------- reference & control
def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            w = (x - xs[i]) / float(xs[i + 1] - xs[i])
            return ys[i] * (1 - w) + ys[i + 1] * w
    return ys[-1]


def _offset(phase, kappa):
    ox0, oy0 = OFF_X[6], OFF_Y[6]
    ox = _interp(OFF_PHASE, OFF_X, phase)
    oy = _interp(OFF_PHASE, OFF_Y, phase)
    return [ox0 + kappa * (ox - ox0), oy0 + kappa * (oy - oy0)]


def _reference(st, s_next):
    """Reference TCP position for frame s_next; returns (xyz, mode, phase)."""
    T = st["target"]
    if T is None:
        return None, "no_target", None
    fit = _fit(st["track"])
    vy = fit[4] if fit is not None else V_REF
    kappa = min(1.3, max(0.7, vy / V_REF))
    dy = st["contact_dy"] if st["contact_dy"] is not None else DY_NOMINAL
    if st["contact_point"] is not None:
        cb = st["contact_point"]
    else:
        y_c = T[1] - dy
        if fit is not None and len(st["track"]) >= 3 and fit[4] > 0.008:
            sc = fit[0] + (y_c - fit[2]) / fit[4]
            cb = [fit[1] + fit[3] * (sc - fit[0]), y_c]
        elif fit is not None:
            cb = [fit[1], y_c]
        else:
            cb = [T[0] + LINE_DX, y_c]
    phase = None
    if st["contact_step"] is not None:
        phase = s_next - float(st["contact_step"])
    elif fit is not None and len(st["track"]) >= 2 and fit[4] > 0.008:
        phase = s_next - (fit[0] + (cb[1] - fit[2]) / fit[4])
    if phase is None or phase <= 5.0:
        ph = WAIT_PHASE if phase is None else max(WAIT_PHASE, phase)
        o = _offset(ph, kappa)
        return [cb[0] + o[0], cb[1] + o[1], Z_LOW], ("sweep" if ph > WAIT_PHASE else "wait"), phase
    o = _offset(5.0, kappa)
    k = phase - 5.0
    kxy = min(k, RETREAT_XY_STEPS)
    z = min(Z_RETREAT, Z_LOW + RETREAT_DZ * k)
    return [cb[0] + o[0] + kappa * RETREAT_V[0] * kxy, cb[1] + o[1] + kappa * RETREAT_V[1] * kxy, z], "retreat", phase


def _clamp_reach(xy, z_now=0.0):
    """Keep TCP targets outside the elbow (J4) fold limit near the robot base. Offline IK
    with the vertical gripper (scratch/ik.py): J4 margin ~0.05 at 0.28 m from the base axis
    for z 0.02, but z 0.08-0.16 needs ~0.31 m for the same margin."""
    dmin = REACH_MIN + REACH_HIGH_EXTRA * min(1.0, max(0.0, (z_now - 0.02) / 0.04))
    dx = xy[0] - BASE_XY[0]
    dy = xy[1] - BASE_XY[1]
    d = math.hypot(dx, dy)
    if d < dmin and d > 1e-6:
        k = dmin / d
        return [BASE_XY[0] + dx * k, BASE_XY[1] + dy * k]
    return xy


def _command(st, frame):
    tcp = np.asarray(frame.tcp, float)
    ref, mode, phase = _reference(st, int(frame.step) + 1)
    if ref is None:
        ref = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
    else:
        xy = _clamp_reach([ref[0], ref[1]], max(float(tcp[2]), ref[2]))
        ref = [xy[0], xy[1], ref[2]]
    e = np.array(ref) - tcp[:3]
    cmd = np.clip(KP * e, -1.0, 1.0)
    nt = float(np.linalg.norm(cmd))
    if nt > TRANS_NORM_MAX:
        cmd = cmd * (TRANS_NORM_MAX / nt)  # limit joint-target jumps (drive torque demand)
    Rc = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy2R(ORI[0], ORI[1], ORI[2])
    w = _rotvec(Rd @ Rc.T)
    rot = -KR * w
    nr = float(np.linalg.norm(rot))
    if nr > 1.0:
        rot = rot / nr
    out = [float(cmd[0]), float(cmd[1]), float(cmd[2]),
           float(rot[0]), float(rot[1]), float(rot[2]), 1.0]
    info = {"mode": mode, "phase": None if phase is None else round(float(phase), 2),
            "ref": [round(float(v), 4) for v in ref],
            "rot_err": round(float(np.linalg.norm(w)), 3)}
    return out, info


def _memory_out(st):
    return {k: st[k] for k in FIELDS}


# ---------------------------------------------------------------- public tools
def perceive_decide(obs, goal, memory, last):
    """Fold the frames since the last action into memory (target, ball track, contact
    plan) and run the interception controller for the next few native steps."""
    st = _state_from_memory(memory)
    for f in obs.frames:
        st = _update(st, f)
    st["decisions"] = int(st["decisions"] or 0) + 1
    n = 2
    if st["target"] is None:
        n = 1
    args = {"state": _memory_out(st), "n_steps": n}
    return {"memory": _memory_out(st), "act": {"tool": "act_intercept", "args": args}}


def act_intercept(robot, state, n_steps=2):
    """Run the closed-loop intercept controller for n_steps native steps. Each step
    re-detects the ball/target on the newest frame, updates the contact plan, and sends
    one pd_ee_delta_pose command (gripper open). Returns measured progress."""
    st = _state_from_memory(state)
    infos = []
    for i in range(int(n_steps)):
        f = robot.now()
        st = _update(st, f)
        cmd, info = _command(st, f)
        info["step"] = int(f.step)
        info["tcp"] = [round(float(v), 4) for v in np.asarray(f.tcp, float)[:3]]
        infos.append(info)
        robot.step(*cmd)
    f = robot.now()
    st = _update(st, f)
    return {"stage": st["stage"], "target": st["target"], "contact_dy": st["contact_dy"],
            "contact_step": st["contact_step"], "contact_point": st["contact_point"],
            "n_track": len(st["track"]), "post_last": st["post"][-1] if st["post"] else None,
            "tcp": [round(float(v), 4) for v in np.asarray(f.tcp, float)],
            "steps": infos, "notes": st["notes"]}
