"""InterceptMedium policy: perceive ball + bullseye with the front RGB-D camera,
fit the ball's straight rolling line, plan an angled finger 'wall' on that line so the
ball glances off along the wall toward the bullseye, servo there, hold, let it roll.

Conventions (all measured in the assigned demos, see task_analysis.md):
- world frame = front_extrinsic frame = tcp frame; robot base at (-0.615, 0, 0).
- ball radius 0.02 m (sphere fit), centre z = 0.02; it rolls ~+y at 0.018-0.028 m/step.
- bullseye radius ~0.1 m; success observed when ball centre enters ~0.1 m of its centre.
- one step realises ~38% of the commanded offset (0.1 m / 0.1 rad per unit command).
"""

BALL_R = 0.02
FACE_N = 0.029          # ball-centre distance from TCP axis plane at contact (finger half width 0.009 + r)
Z_LOW = 0.022           # TCP height at the wall (fingertip ~0.009 below TCP)
Z_SAFE = 0.06           # transit height (fingertip ~0.051, above ball top 0.04)
GRIP = -1.0             # fingers closed: one flat 2-finger face ~0.026 m long, no gap
REALISE = 0.38          # fraction of commanded offset realised per step (free space fit)
KP = 0.75
KR = 0.75
ROT_RATE = 0.036        # achievable yaw rad/step (0.1 rad * 0.38, minus margin)
LIN_RATE = 0.034        # achievable m/step per axis
TH_MIN = 22.0
TH_MAX = 50.0
POST_W = 0.5            # weight of the expert regression in the post-contact direction model
DELTA_PREF = 0.02       # preferred signed miss distance of predicted path from bullseye centre
PITCH_NEAR = 0.15       # rad, pitch used when the wall is close to the robot base (x < -0.25)
SLIDE_FRAC = 0.8        # wall slides along itself at this fraction of the ball speed along it
PRIOR_DX = 0.200        # bullseye x - ball line x (demos: 0.194..0.207)
PRIOR_VY = 0.022
PRIOR_Y0 = -0.75


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotvec(m):
    c = (np.trace(m) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    ang = math.acos(c)
    if ang < 1e-7:
        return np.zeros(3)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        # near pi: axis from the symmetric part
        d = np.sqrt(np.maximum((np.diag(m) + 1.0) / 2.0, 0.0))
        return d * ang
    return v / (2.0 * s) * ang


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _front_points(frame, mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 3))
    depth = np.asarray(frame.front_depth)
    z = depth[ys, xs].astype(float) / 1000.0
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    k = np.asarray(frame.front_K, dtype=float)
    e = np.asarray(frame.front_extrinsic, dtype=float)
    xc = (xs + 0.5 - k[0, 2]) * z / k[0, 0]
    yc = (ys + 0.5 - k[1, 2]) * z / k[1, 1]
    pc = np.stack([xc, yc, z], 1)
    return (pc - e[:, 3]) @ e[:, :3]


def _front_cam_pos(frame):
    e = np.asarray(frame.front_extrinsic, dtype=float)
    return -e[:, :3].T @ e[:, 3]


def _detect_ball(frame):
    """Return [x, y, npix] of the red ball centre in world metres, or None."""
    im = np.asarray(frame.front_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mask = (r > 100) & (g < 50) & (b < 50) & (r > 3 * g)
    n = int(mask.sum())
    if n < 8:
        return None
    p = _front_points(frame, mask)
    p = p[(p[:, 2] > -0.01) & (p[:, 2] < 0.06)]
    if len(p) < 8:
        return None
    c = np.median(p, 0).copy()
    c[2] = BALL_R
    cam = _front_cam_pos(frame)
    d = c[:2] - cam[:2]
    nd = float(np.linalg.norm(d))
    if nd > 1e-6:
        c[:2] = c[:2] + 0.5 * BALL_R * d / nd
    for _ in range(8):
        diff = p - c
        dist = np.linalg.norm(diff, axis=1) + 1e-9
        res = dist - BALL_R
        jac = -diff[:, :2] / dist[:, None]
        step = np.linalg.lstsq(jac, -res, rcond=None)[0]
        c[:2] = c[:2] + step
        if float(np.linalg.norm(step)) < 1e-5:
            break
    if not np.all(np.isfinite(c)):
        return None
    return [float(c[0]), float(c[1]), n]


def _detect_target(frame):
    """Return [x, y, radius] of the red/white bullseye on the table, or None."""
    im = np.asarray(frame.front_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mask = (r > 150) & (g < 120) & (b > g + 2) & (r - g > 80)
    if int(mask.sum()) < 50:
        return None
    p = _front_points(frame, mask)
    p = p[np.abs(p[:, 2]) < 0.015]
    if len(p) < 50:
        return None
    lo = np.percentile(p[:, :2], 0.5, axis=0)
    hi = np.percentile(p[:, :2], 99.5, axis=0)
    c = (lo + hi) / 2.0
    rad = float(np.mean(hi - lo) / 2.0)
    if rad < 0.07 or rad > 0.13:
        c = np.mean(p[:, :2], 0)
    return [float(c[0]), float(c[1]), rad]


def _fit_ball(track):
    """Least-squares straight line x(t), y(t) through [x, y, t, npix] samples (t >= 2)."""
    pts = [q for q in track if q is not None and len(q) >= 4 and q[2] >= 2 and q[3] >= 15]
    if len(pts) < 2:
        pts = [q for q in track if q is not None and len(q) >= 4 and q[2] >= 1 and q[3] >= 10]
    if len(pts) < 2:
        return None
    pts = pts[-14:]
    a = np.array(pts, dtype=float)
    t = a[:, 2]
    if float(t.max() - t.min()) < 1.0:
        return None
    m = np.stack([np.ones_like(t), t], 1)
    cx = np.linalg.lstsq(m, a[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(m, a[:, 1], rcond=None)[0]
    res = float(np.sqrt(np.mean((m @ cx - a[:, 0]) ** 2 + (m @ cy - a[:, 1]) ** 2)))
    return {"x0": float(cx[0]), "vx": float(cx[1]), "y0": float(cy[0]), "vy": float(cy[1]),
            "n": len(pts), "res": res}


def _plan(fit, goal, t_now, tcp, prev_th, rot_rate=ROT_RATE, lin_rate=LIN_RATE):
    """Choose wall angle theta (deg from +y toward +x) and contact point on the ball line.

    Maximises the angular margin of the predicted post-contact path (it must pass within
    ~0.09 m of the bullseye centre) subject to being in place before the ball arrives and
    to an estimated entry time before the 60-step limit."""
    gx, gy = float(goal[0]), float(goal[1])
    x0, vx, y0, vy = fit["x0"], fit["vx"], fit["y0"], fit["vy"]
    if vy < 0.008:
        vy = 0.008
    gvec = np.array([gx, gy])
    yaw_now = float(tcp[5])
    best = None
    for thd in np.arange(TH_MIN, TH_MAX + 0.1, 1.0):
        th = math.radians(thd)
        ub = np.array([math.sin(th), math.cos(th)])
        nb = np.array([ub[1], -ub[0]])
        thp = th + math.radians(_post_bias(thd))
        up = np.array([math.sin(thp), math.cos(thp)])
        npp = np.array([up[1], -up[0]])
        yaw_d = -th
        erot = abs(_wrap(yaw_d - yaw_now))
        trot = erot / rot_rate + (1.0 if erot > 0.05 else 0.0)
        for dl in np.arange(-0.04, 0.0801, 0.01):
            f0 = float((gvec - np.array([x0, y0])) @ npp) - dl
            f1 = -float(np.array([vx, vy]) @ npp)
            if abs(f1) < 1e-6:
                continue
            tc = -f0 / f1
            pc = np.array([x0 + vx * tc, y0 + vy * tc])
            wall = pc - FACE_N * nb
            exy = max(abs(wall[0] - tcp[0]), abs(wall[1] - tcp[1]))
            txy = exy / lin_rate + (1.0 if exy > 0.01 else 0.0)
            ez = abs(Z_LOW - tcp[2])
            tz = ez / lin_rate + (1.0 if ez > 0.01 else 0.0)
            tready = t_now + max(trot, txy, tz) + 1.0
            dist = float(np.linalg.norm(gvec - pc))
            proj = float((gvec - pc) @ up)
            lent = proj - math.sqrt(max(1e-6, 0.1 ** 2 - min(dl * dl, 0.0099)))
            spd = 0.65 * math.cos(th) * vy
            ttot = tc + max(lent, 0.0) / max(spd, 1e-3)
            hi = (0.09 - dl) / max(dist, 0.1)
            lo = (0.09 + dl) / max(dist, 0.1)
            marg = math.degrees(math.asin(max(-1.0, min(1.0, min(hi, lo)))))
            slack = tc - tready
            score = -4.0 * marg + 0.2 * ttot
            if ttot > 52.0:
                score += 3.0 * (ttot - 52.0)
            if slack < 0:
                score += 1000.0 - 20.0 * slack
            if prev_th is not None:
                score += 0.3 * abs(thd - prev_th)
            if best is None or score < best["score"]:
                best = {"score": float(score), "th": float(thd), "dl": float(dl), "tc": float(tc),
                        "tready": float(tready), "ttot": float(ttot), "pc": [float(pc[0]), float(pc[1])],
                        "wall": [float(wall[0]), float(wall[1])], "yaw": float(yaw_d),
                        "feasible": bool(slack >= 0), "margin_deg": float(marg)}
    return best


def _post_bias(thd):
    """Predicted post-contact direction minus wall direction (deg). Expert demos (moving finger bar):
    theta_post = 0.68 theta_bar + 17 (sd 7 deg); an ideal inelastic wall gives 0. Use a blend."""
    return POST_W * (17.0 - 0.32 * float(thd))


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else []


def _num(v, default):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(default)


def perceive_decide(obs, goal, memory, last):
    """Track ball and bullseye, keep the wall plan in memory, and command the next servo chunk."""
    mem = dict(memory) if isinstance(memory, dict) else {}
    frames = list(obs.frames)
    cur = frames[-1]
    steps_done = mem.get("steps_done")
    first = steps_done is None
    if first:
        steps_done = 0
    else:
        steps_done = int(steps_done) + len(frames)
    # time index of the current frame: native steps since reset. Our own counter; the frame
    # 'step' field is only recorded for diagnosis.
    t_now = int(steps_done)
    mem["frame_step"] = int(cur.step) if isinstance(cur.step, (int, np.integer)) else -1
    track = _as_list(mem.get("ball_track"))
    goal_xy = mem.get("goal_xy")
    n_fr = len(frames)
    for i, fr in enumerate(frames):
        t_f = t_now - (n_fr - 1 - i)
        if goal_xy is None:
            tg = _detect_target(fr)
            if tg is not None:
                goal_xy = [tg[0], tg[1]]
                mem["goal_radius"] = tg[2]
        bd = _detect_ball(fr)
        if bd is not None:
            track.append([bd[0], bd[1], float(t_f), float(bd[2])])
    track = track[-40:]
    # online estimate of achievable rotation / translation per step from saturated steps
    rot_rate = _num(mem.get("rot_rate"), ROT_RATE)
    lin_rate = _num(mem.get("lin_rate"), LIN_RATE)
    prev_tcp = mem.get("prev_tcp")
    prev_cmd = mem.get("prev_cmd")
    if not first and isinstance(prev_tcp, list) and len(prev_tcp) == 6 and isinstance(prev_cmd, list) \
            and len(prev_cmd) == 6:
        rd_prev = _rot([math.pi, prev_cmd[4], prev_cmd[5]])
        pz_prev = prev_cmd[2]
        a_tcp = np.asarray(prev_tcp, dtype=float)
        r_s, l_s = [], []
        for fr in frames:
            b_tcp = np.asarray(fr.tcp, dtype=float)
            ra = _rot(a_tcp[3:6])
            err_before = float(np.linalg.norm(_rotvec(rd_prev @ ra.T)))
            moved = float(np.linalg.norm(_rotvec(_rot(b_tcp[3:6]) @ ra.T)))
            if err_before > 0.15:
                r_s.append(moved)
            if abs(pz_prev - a_tcp[2]) > 0.08:
                l_s.append(abs(b_tcp[2] - a_tcp[2]))
            a_tcp = b_tcp
        if r_s:
            rot_rate = min(0.06, max(0.015, 0.7 * rot_rate + 0.3 * 0.95 * float(np.mean(r_s))))
        if l_s:
            lin_rate = min(0.05, max(0.015, 0.7 * lin_rate + 0.3 * 0.95 * float(np.mean(l_s))))
    phase = mem.get("phase") or "approach"
    contact_t = mem.get("contact_t")

    # ball line fit from pre-contact samples
    pre = [q for q in track if contact_t is None or q[2] < contact_t]
    fit = _fit_ball(pre)
    cue_src = "observed"
    if fit is None:
        gx = goal_xy[0] if goal_xy is not None else -0.0
        fit = {"x0": gx - PRIOR_DX, "vx": 0.0, "y0": PRIOR_Y0, "vy": PRIOR_VY, "n": 0, "res": -1.0}
        cue_src = "prior"
    g_use = goal_xy if goal_xy is not None else [fit["x0"] + PRIOR_DX, 0.0]

    # contact detection: latest sample deviates from the pre-contact line
    if contact_t is None and fit["n"] >= 4 and len(track) > 0:
        q = track[-1]
        px = fit["x0"] + fit["vx"] * q[2]
        py = fit["y0"] + fit["vy"] * q[2]
        dev = math.hypot(q[0] - px, q[1] - py)
        tc_prev = _num(mem.get("t_contact"), 99.0)
        if dev > 0.010 and q[2] >= max(5.0, tc_prev - 6.0):
            contact_t = float(q[2]) - 1.0
            phase = "deflected"
    tcp = np.asarray(cur.tcp, dtype=float)
    plan_frozen = bool(mem.get("plan_frozen")) if mem.get("plan_frozen") is not None else False
    wall = mem.get("wall")
    yaw_d = mem.get("yaw_target")
    tc = mem.get("t_contact")
    th = mem.get("theta_deg")
    if not plan_frozen and phase == "approach":
        prev = th if (isinstance(th, (int, float)) and cue_src == "observed" and fit["n"] >= 3) else None
        pl = _plan(fit, g_use, float(t_now), tcp, prev, rot_rate, lin_rate)
        if pl is not None:
            wall, yaw_d, tc, th = pl["wall"], pl["yaw"], pl["tc"], pl["th"]
            mem["plan_feasible"] = pl["feasible"]
            mem["plan_ttot"] = pl["ttot"]
            mem["plan_delta"] = pl["dl"]
            mem["plan_margin"] = pl["margin_deg"]
            mem["contact_point"] = pl["pc"]
            if fit["n"] >= 3 and float(t_now) >= pl["tc"] - 4.0:
                plan_frozen = True
    if wall is None:
        wall = [float(tcp[0]), float(tcp[1])]
        yaw_d = float(tcp[5])
        tc = 99.0
        th = 0.0
    tc = _num(tc, 99.0)
    force_low = float(t_now) >= tc - 3.5
    # slide the wall along its own direction from contact on (keeps contact during spin conversion)
    thr = math.radians(_num(th, 0.0))
    v_along = SLIDE_FRAC * math.cos(thr) * max(fit["vy"], 0.008)
    slide = [math.sin(thr) * v_along, math.cos(thr) * v_along]
    # small outward tilt (fingertips toward the robot) only near the base, for joint-4 margin
    pitch_d = PITCH_NEAR * min(1.0, max(0.0, (-0.25 - float(wall[0])) / 0.08))
    if t_now >= 56:
        n_steps = 4
    elif t_now < 12:
        n_steps = 1
    elif t_now < 32:
        n_steps = 2
    else:
        n_steps = 4
    mem.update({
        "rot_rate": rot_rate, "lin_rate": lin_rate, "prev_tcp": [float(v) for v in tcp],
        "prev_cmd": [float(wall[0]), float(wall[1]), Z_LOW if force_low else Z_SAFE, math.pi,
                     float(pitch_d), float(yaw_d)],
        "steps_done": steps_done, "ball_track": track, "goal_xy": goal_xy, "phase": phase,
        "contact_t": contact_t, "wall": [float(wall[0]), float(wall[1])], "yaw_target": float(yaw_d),
        "t_contact": float(tc), "theta_deg": float(_num(th, 0.0)), "plan_frozen": plan_frozen,
        "fit_source": cue_src,
        "ball_fit": [fit["x0"], fit["vx"], fit["y0"], fit["vy"]],
    })
    out = {}
    for k in MEMORY_FIELDS:
        out[k] = mem.get(k)
    args = {"x": float(wall[0]), "y": float(wall[1]), "yaw": float(yaw_d), "pitch": float(pitch_d),
            "steps": int(n_steps), "t_now": int(t_now), "t_contact": float(tc),
            "ball_line": [float(fit["x0"]), float(fit["vx"]), float(fit["y0"]), float(fit["vy"])],
            "slide_vx": float(slide[0]), "slide_vy": float(slide[1]), "slide_steps": 6,
            "gripper": GRIP}
    return {"memory": out, "act": {"tool": "act_servo", "args": args}}


MEMORY_FIELDS = ["steps_done", "goal_xy", "goal_radius", "ball_track", "ball_fit", "fit_source",
                 "phase", "contact_t", "theta_deg", "yaw_target", "wall", "contact_point", "t_contact",
                 "plan_frozen", "plan_feasible", "plan_ttot", "plan_delta", "plan_margin",
                 "rot_rate", "lin_rate", "prev_tcp", "prev_cmd", "frame_step"]


def act_servo(robot, x, y, yaw, pitch=0.0, steps=1, t_now=0, t_contact=99.0, ball_line=None,
              slide_vx=0.0, slide_vy=0.0, slide_steps=6, gripper=-1.0):
    """Closed-loop servo of the TCP toward the wall pose (x, y, z, roll=pi, pitch, yaw) for `steps`
    native steps (t_now is the step index of the current frame). z is Z_SAFE in transit and Z_LOW
    once the TCP is within 3 cm / 0.2 rad of the pose and the predicted ball (ball_line =
    [x0, vx, y0, vy], position at step t = (x0 + vx t, y0 + vy t)) is not under the gripper,
    and always Z_LOW from t_contact - 3.5. From t_contact - 0.5 the target slides along
    (slide_vx, slide_vy) m/step for slide_steps steps. Returns measured TCP and per-step errors."""
    rd = _rot([math.pi, float(pitch), float(yaw)])
    line = ball_line if isinstance(ball_line, (list, tuple)) and len(ball_line) == 4 else None
    log = []
    fr = robot.now()
    for k in range(int(steps)):
        t = int(t_now) + k
        tcp = np.asarray(fr.tcp, dtype=float)
        ds = min(max(t - (float(t_contact) - 0.5), 0.0), float(slide_steps))
        px = float(x) + float(slide_vx) * ds
        py = float(y) + float(slide_vy) * ds
        exy = math.hypot(px - tcp[0], py - tcp[1])
        rc = _rot(tcp[3:6])
        w = _rotvec(rd @ rc.T)
        ball_far = True
        if line is not None:
            bx = line[0] + line[1] * t
            by = line[2] + line[3] * t
            ball_far = math.hypot(bx - px, by - py) > 0.07 or by > py + 0.05
        force_low = t >= float(t_contact) - 3.5
        low = force_low or (ball_far and exy < 0.03 and float(np.linalg.norm(w)) < 0.2)
        pz = Z_LOW if low else Z_SAFE
        dxyz = np.array([px - tcp[0], py - tcp[1], pz - tcp[2]])
        a = np.clip(KP * dxyz / (0.1 * REALISE), -1.0, 1.0)
        r = -KR * w / (0.1 * REALISE)
        nr = float(np.linalg.norm(r))
        if nr > 1.0:
            r = r / nr
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(gripper))
        fr = robot.now()
        log.append([t, round(float(exy), 4), round(float(np.linalg.norm(w)), 3), round(float(pz), 3)])
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"tcp": [round(float(v), 4) for v in tcp], "steps": int(steps), "log": log}
