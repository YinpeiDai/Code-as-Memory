"""InterceptGrabFast policy: intercept a ball rolling along the table and grasp it.

Strategy (from demonstrations): the red ball rolls in a straight line (almost pure +y,
~0.027-0.036 m per control step) at constant height.  The arm moves to a hover point
above the predicted ball line ahead of the ball, waits, accelerates along +y to match the
ball velocity when it arrives, descends around it with the fingers straddling the ball
along y (yaw 0), closes the gripper, then decelerates to rest, lifts slightly and holds.
"""

R_BALL = 0.02
# Per-axis first-order response fitted on all 50 demos: d[t+1] = AL*0.1*u + BE*d[t]
AL = (0.318, 0.249, 0.268)
BE = (0.257, 0.348, 0.291)
Z_HOVER = 0.065
Z_GRASP = 0.020
Z_CLOSE = 0.032
Z_LIFT = 0.045
Y_WAIT = -0.18
LEAD = 0.03
KP = 0.6
U_DESC = 0.6


def _rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-4:
        # near pi: use diagonal
        a = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return a * th
    return v * (th / (2.0 * s))


def _rot_cmd(tcp, yaw_des=0.0, gain=5.0):
    """Normalized rotation command keeping the gripper pointing down with yaw yaw_des."""
    R = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_to_R(math.pi, 0.0, yaw_des)
    w = _logR(Rd @ R.T)
    r = -gain * w
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _detect(rgb, dep, K, E, minpix):
    """Red-ball centre estimate in world coordinates from one RGB-D camera."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    m = (r > 100) & (g < 45) & (b < 45) & (dep > 0)
    n = int(m.sum())
    if n < minpix:
        return None, n
    vs, us = np.nonzero(m)
    # keep the main blob (pixels near the median pixel)
    mu, mv = np.median(us), np.median(vs)
    rad = max(6.0, 2.5 * math.sqrt(n / math.pi))
    keep = (np.abs(us - mu) < rad) & (np.abs(vs - mv) < rad)
    us, vs = us[keep], vs[keep]
    if len(us) < minpix:
        return None, int(len(us))
    z = dep[vs, us].astype(np.float64) / 1000.0
    x = (us + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (vs + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    Rm = E[:, :3].astype(np.float64)
    t = E[:, 3].astype(np.float64)
    pts = (pc - t) @ Rm
    c = np.median(pts, 0)
    C = -Rm.T @ t
    u = c - C
    u = u / max(1e-9, float(np.linalg.norm(u)))
    c = c + R_BALL * u
    edge = bool(us.min() <= 0 or vs.min() <= 0 or us.max() >= rgb.shape[1] - 1 or vs.max() >= rgb.shape[0] - 1)
    return [float(c[0]), float(c[1]), float(c[2]), edge], int(len(us))


def _observe_ball(f):
    """Return (pos, source, npix) for the ball in frame f, or (None, '', 0)."""
    pw, nw = _detect(f.wrist_rgb, f.wrist_depth, f.wrist_K, f.wrist_extrinsic, 60)
    if pw is not None and (not pw[3]) and 0.0 < pw[2] < 0.06:
        return pw[:3], "wrist", nw
    pf, nf = _detect(f.front_rgb, f.front_depth, f.front_K, f.front_extrinsic, 8)
    if pf is not None and (not pf[3]) and -0.01 < pf[2] < 0.06:
        return pf[:3], "front", nf
    return None, "", 0


def _fit(hist, t_now, window=8):
    """Constant-velocity ball model from recent observations [step, x, y, z, src(0 front,1 wrist)].

    Returns dict with x, y (at t_now), vx, vy, n, or None."""
    if not hist:
        return None
    recent = [h for h in hist if h[0] >= t_now - 12]
    if not recent:
        recent = hist[-3:]
    wr = [h for h in recent if h[4] == 1 and h[0] >= t_now - 4]
    use = wr if len(wr) >= 3 else [h for h in recent if h[4] == 0]
    if len(use) < 2:
        use = recent
    use = use[-window:]
    T = np.array([h[0] for h in use], dtype=np.float64)
    X = np.array([h[1] for h in use], dtype=np.float64)
    Y = np.array([h[2] for h in use], dtype=np.float64)
    if len(use) >= 3 and T.max() - T.min() >= 2:
        A = np.stack([np.ones_like(T), T - t_now], -1)
        cy = np.linalg.lstsq(A, Y, rcond=None)[0]
        cx = np.linalg.lstsq(A, X, rcond=None)[0]
        vy, vx = float(cy[1]), float(cx[1])
        vx = max(-0.004, min(0.004, vx))
        y0 = float(cy[0])
        x0 = float(np.mean(X + vx * (t_now - T)))
        # the ball x is nearly constant: once the (more accurate) wrist camera has seen the
        # ball, take x from the latest wrist observations only (front view is occluded by the
        # gripper during the final descent and becomes biased)
        allw = [h for h in hist if h[4] == 1]
        if len(allw) >= 2:
            ww = allw[-4:]
            x0 = float(np.mean([h[1] + vx * (t_now - h[0]) for h in ww]))
        return {"x": x0, "y": y0, "vx": vx, "vy": vy, "n": len(use)}
    # velocity unknown yet: use prior speed along +y
    vy = 0.031
    h = use[-1]
    return {"x": float(h[1]), "y": float(h[2]) + vy * (t_now - h[0]), "vx": 0.0, "vy": vy, "n": len(use)}


def _u_from_d(d_des, d_prev):
    u = []
    for k in range(3):
        u.append(max(-1.0, min(1.0, (d_des[k] - BE[k] * d_prev[k]) / (0.1 * AL[k]))))
    return u


def _clipn(v, lim):
    n = float(np.linalg.norm(v))
    if n > lim:
        return [float(a) * lim / n for a in v]
    return [float(a) for a in v]


def perceive_decide(obs, goal, memory, last):
    """Detect the ball in the current frame, keep stage memory and choose the next act."""
    f = obs.current
    pos, src, n = _observe_ball(f)
    stage = memory.get("stage") or "start"
    attempts = int(memory.get("attempts") or 0)
    notes = memory.get("notes") or ""
    if isinstance(last, str) and last:
        notes = (notes + " | " + last[:300])[-1500:]
    step = int(f.step)
    ball = pos if pos is not None else (memory.get("ball_last") or [])
    fingers = [float(a) for a in f.fingers]
    if stage in ("start", "missed", "returned"):
        act = {"tool": "act_catch", "args": {"retry": attempts}}
        attempts += 1
        stage = "catching"
    else:
        act = {"tool": "act_catch", "args": {"retry": attempts}}
        attempts += 1
    return {
        "memory": {
            "stage": stage,
            "attempts": attempts,
            "ball_last": [float(a) for a in ball] if ball else [],
            "notes": notes + " | step %d src %s n %d fingers %.3f %.3f" % (step, src, n, fingers[0], fingers[1]),
        },
        "act": act,
    }


def act_catch(robot, retry=0, max_hold=200):
    """Closed-loop intercept: hover ahead of the ball, match its velocity, descend, close,
    decelerate, lift slightly and hold.  Returns stage log if it ends without episode end."""
    hist = []
    stage = "approach"
    log = []
    d_prev = [0.0, 0.0, 0.0]
    last_tcp = None
    t_close = -1
    hold_n = 0
    grip = 1.0
    y_wait = None
    events = []
    f = robot.now()
    checked = False
    for it in range(2000):
        f = robot.now()
        t = int(f.step)
        tcp = [float(a) for a in f.tcp]
        if last_tcp is not None:
            d_prev = [tcp[k] - last_tcp[k] for k in range(3)]
        last_tcp = tcp
        pos, src, npx = _observe_ball(f)
        if pos is not None:
            hist.append([t, pos[0], pos[1], pos[2], 1 if src == "wrist" else 0])
        m = _fit(hist, t)
        fing = float(f.fingers[0] + f.fingers[1]) / 2.0
        rcmd, rerr = _rot_cmd(f.tcp, 0.0)
        p = tcp[:3]
        dd = [0.0, 0.0, 0.0]
        if m is None:
            # ball not yet seen: move to a generic hover pose
            tgt = [-0.19, Y_WAIT, Z_HOVER]
            dd = _clipn([(tgt[k] - p[k]) * 0.5 for k in range(3)], 0.04)
        else:
            xb, yb, vy = m["x"], m["y"], m["vy"]
            if stage == "approach":
                if y_wait is None or m["n"] < 3:
                    y_wait = Y_WAIT
                    # never wait behind the ball: at least 0.15 m ahead of it
                    y_wait = max(y_wait, yb + 0.15)
                if m["n"] >= 3 and not checked:
                    chk = robot.dry_run(xb, y_wait, Z_HOVER, math.pi, 0.0, 0.0)
                    events.append(["dry_run", t, bool(chk.get("reachable")), chk.get("joint_margin")])
                    checked = True
                tgt = [xb, y_wait, Z_HOVER]
                dd = _clipn([(tgt[k] - p[k]) * 0.5 for k in range(3)], 0.04)
                if m["n"] >= 3 and yb + vy >= p[1] - LEAD:
                    stage = "track"
                    events.append(["track", t, round(yb, 4), round(p[1], 4), round(vy, 4)])
            if stage in ("track", "descend", "closing"):
                ey = yb - p[1]
                ex = xb - p[0]
                dd[0] = m["vx"] + 0.5 * ex
                dd[1] = vy + KP * ey
                if stage == "track":
                    dd[2] = (Z_HOVER - p[2]) * 0.5
                    if abs(ey) < 0.01 and abs(ex) < 0.008 and abs(d_prev[1] - vy) < 0.006:
                        stage = "descend"
                        events.append(["descend", t, round(ex, 4), round(ey, 4), round(d_prev[1] - vy, 4)])
                if stage == "descend":
                    dd[2] = max(Z_GRASP - p[2], -0.04)
                    if p[2] < Z_CLOSE and abs(ey) < 0.015 and abs(ex) < 0.012:
                        stage = "closing"
                        t_close = t
                        grip = -1.0
                        events.append(["close", t, round(ex, 4), round(ey, 4), round(p[2], 4)])
                if stage == "closing":
                    dd[2] = Z_GRASP - p[2]
                    grip = -1.0
                    if t - t_close >= 2:
                        if fing > 0.010:
                            stage = "hold"
                            events.append(["grasped", t, round(fing, 4)])
                        elif fing < 0.006:
                            stage = "missed"
                            events.append(["missed", t, round(fing, 4)])
                if abs(ey) > 0.06 and stage in ("track", "descend"):
                    events.append(["lost", t, round(ey, 4)])
                    stage = "approach"
                    y_wait = max(yb + 0.12, p[1] + 0.02)
        if stage == "hold":
            hold_n += 1
            # decelerate smoothly, lift a little, keep gripper closed
            dd = [d_prev[0] * 0.5, d_prev[1] * 0.5, 0.0]
            dd[2] = max(-0.02, min(0.02, (Z_LIFT - p[2]) * 0.5))
            grip = -1.0
            if hold_n > max_hold:
                break
        if stage == "missed":
            grip = 1.0
            # reopen and go back to approach mode ahead of the ball if any time is left
            dd = [0.0, 0.0, (Z_HOVER - p[2]) * 0.5]
            if fing > 0.035:
                stage = "approach"
                y_wait = None if m is None else max(m["y"] + 0.12, p[1])
        u = _u_from_d(dd, d_prev)
        if stage in ("descend", "closing"):
            # R001-R003: u_z=-1 while tracking coupled +1.7..2.1 cm into x; limit descent rate
            u[2] = max(-U_DESC, u[2])
        if len(log) < 70:
            log.append([t, stage[0], round(p[0], 3), round(p[1], 3), round(p[2], 3),
                        None if m is None else round(m["y"], 3), round(fing, 3)])
        robot.step(u[0], u[1], u[2], float(rcmd[0]), float(rcmd[1]), float(rcmd[2]), grip)
    return {"stage": stage, "events": events, "log": log[-20:]}
