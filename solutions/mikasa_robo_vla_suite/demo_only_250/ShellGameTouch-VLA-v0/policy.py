"""ShellGameTouch policy: remember where the ball is during the cue, then touch that cup.

Evidence (237 assigned demos, see task_analysis.md):
- Frames 0..n-1 (n = 1..5) show only the red ball; commands are zeroed. At frame n three
  inverted mugs appear (row along world y, spacing ~0.19 m, top at z~0.108) and the
  command issued at frame n is the first one that moves the arm.
- The cup hiding the ball is centred on the ball (ball-to-nearest-cup <= 0.02 m; next >= 0.18 m).
- Expert final TCP relative to the front-camera ball estimate: dx +0.021, dy -0.010, dz +0.067
  (TCP z 0.068..0.121); the arm is usually decelerating at success.
"""

BALL_MIN_PIXELS = 15
CUP_MIN_PIXELS = 500
TOUCH_OFFSET = (0.021, -0.010)   # expert mean final TCP offset from ball estimate (x, y)
Z_TOUCH = 0.103                  # expert mean final TCP height
Z_HOVER = 0.16                   # safe travel height above cup tops (0.108)
GRIP_OPEN = 1.0


def _ball_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r > 150) & (g < 60) & (b < 60)


def _cup_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r > 40) & (r <= 150) & (g < 45) & (b < 45) & (r > 2 * g + 15)


def _backproject(K, E, us, vs, depth_m):
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    us = np.asarray(us, dtype=float)
    vs = np.asarray(vs, dtype=float)
    z = np.asarray(depth_m, dtype=float)
    pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], axis=-1)
    Rm = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ Rm   # rows: R^T (pc - t)


def _ball_in_frame(frame, cam):
    """Return [x, y, z, pixels] of the red ball seen by camera `cam`, or None."""
    rgb = getattr(frame, cam + "_rgb", None)
    dep = getattr(frame, cam + "_depth", None)
    K = getattr(frame, cam + "_K", None)
    E = getattr(frame, cam + "_extrinsic", None)
    if rgb is None or dep is None or K is None or E is None:
        return None
    m = _ball_mask(rgb)
    cnt = int(m.sum())
    if cnt < BALL_MIN_PIXELS:
        return None
    vs, us = np.nonzero(m)
    d = np.asarray(dep, dtype=float)[vs, us]
    good = d > 0
    if int(good.sum()) < 5:
        return None
    p = _backproject(K, E, [us.mean()], [vs.mean()], [float(np.median(d[good])) / 1000.0])[0]
    if not np.all(np.isfinite(p)):
        return None
    return [float(p[0]), float(p[1]), float(p[2]), cnt]


def _cup_pixels(frame):
    rgb = getattr(frame, "front_rgb", None)
    if rgb is None:
        return 0
    return int(_cup_mask(rgb).sum())


def _cup_centres(frame):
    """Front-camera cup clusters (x, y) sorted by y; may be empty."""
    rgb = frame.front_rgb
    dep = np.asarray(frame.front_depth, dtype=float) / 1000.0
    m = _cup_mask(rgb) & (dep > 0)
    vs, us = np.nonzero(m)
    if len(vs) < CUP_MIN_PIXELS:
        return []
    P = _backproject(frame.front_K, frame.front_extrinsic, us, vs, dep[vs, us])
    P = P[(P[:, 2] > 0.01) & (P[:, 2] < 0.15)]
    if len(P) < 100:
        return []
    ys = np.sort(P[:, 1])
    gaps = np.diff(ys)
    big = np.flatnonzero(gaps > 0.04)
    cuts = list(ys[big] + gaps[big] / 2.0)
    labels = np.digitize(P[:, 1], cuts)
    out = []
    for k in range(len(cuts) + 1):
        C = P[labels == k]
        if len(C) >= 50:
            out.append([float(np.median(C[:, 0])), float(np.median(C[:, 1])), int(len(C))])
    return out


def _rpy_to_R(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(Rm):
    c = (float(np.trace(Rm)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:
        # near pi: axis from the symmetric part
        A = (Rm + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / math.sqrt(max(A[k, k], 1e-9))
        return axis * ang
    return v / (2.0 * s) * ang


def _rot_cmd(tcp, R_goal, gain=1.0, cap=0.3):
    """Rotation arguments driving the measured orientation toward R_goal.
    Native semantics: the angular offset is -0.1*r (world rotation vector)."""
    if R_goal is None:
        return [0.0, 0.0, 0.0]
    w = _rotvec(R_goal @ _rpy_to_R(tcp[3:6]).T)
    r = -gain * w / 0.1
    n = float(np.linalg.norm(r))
    if n > cap:
        r = r * (cap / n)
    return [float(r[0]), float(r[1]), float(r[2])]


def _clip(v, lo, hi):
    return float(max(lo, min(hi, v)))


def _qvel(frames):
    if len(frames) < 2:
        return None
    a = np.asarray(frames[-1].joints, dtype=float)
    b = np.asarray(frames[-2].joints, dtype=float)
    return float(np.max(np.abs(a - b)) * 20.0)


def _num(x, default):
    return default if x is None else x


def perceive_decide(obs, goal, memory, last):
    """Track the ball cue, then select the touch/fallback stage for the remembered cup."""
    mem = dict(memory)
    stage = _num(mem.get("stage"), "cue")
    decisions = int(_num(mem.get("decisions"), 0)) + 1
    ball = mem.get("ball_xyz")
    ball_pixels = _num(mem.get("ball_pixels"), 0)
    ball_step = _num(mem.get("ball_seen_step"), -1)
    ball_source = _num(mem.get("ball_source"), "none")
    cups_seen = bool(_num(mem.get("cups_seen"), False))
    target = mem.get("target_xy")
    contact_z = mem.get("contact_z")
    orient = mem.get("hold_rpy")
    cur = obs.current

    if orient is None:
        orient = [float(v) for v in np.asarray(cur.tcp, dtype=float)[3:6]]

    # Observed event: ball visible (only during the cue). Keep the latest sighting.
    if stage == "cue":
        for f in obs.frames:
            est = _ball_in_frame(f, "front")
            src = "front"
            if est is None:
                est = _ball_in_frame(f, "wrist")
                src = "wrist"
            if est is not None and (src == "front" or ball_source != "front"):
                ball = est[:3]
                ball_pixels = est[3]
                ball_step = int(f.step)
                ball_source = src

    cups_now = _cup_pixels(cur) >= CUP_MIN_PIXELS
    cups_seen = cups_seen or cups_now
    note = ""

    if stage == "cue":
        ball_now = _ball_in_frame(cur, "front") is not None
        waited = last is not None and "act_wait_cue" in str(last)
        if (not cups_now or ball_now) and not waited:
            act = {"tool": "act_wait_cue", "args": {"max_steps": 8}}
            note = "waiting for cups"
        else:
            if ball is not None:
                target = [float(ball[0]) + TOUCH_OFFSET[0], float(ball[1]) + TOUCH_OFFSET[1]]
                note = "target from remembered ball (%s)" % ball_source
            else:
                cs = _cup_centres(cur) if cups_now else []
                if cs:
                    mid = cs[len(cs) // 2]
                    target = [mid[0] + 0.02, mid[1] - 0.005]
                    note = "no ball cue; guessing middle cup"
                else:
                    target = [0.03, 0.0]
                    note = "no ball cue, no cups; default guess"
            stage = "touch"
            act = {"tool": "act_touch", "args": {"tx": target[0], "ty": target[1],
                                                 "z_touch": Z_TOUCH, "hold_steps": 6,
                                                 "max_steps": 22, "rpy": orient}}
    elif stage == "touch":
        # act_touch returned without episode end: try the expert-like press/tilt, then settle.
        if last is not None and "contact_z" in str(last):
            txt = str(last)
            i = txt.find('"contact_z": ')
            if i >= 0:
                frag = txt[i + 13:i + 30].split(",")[0].split("}")[0]
                try:
                    contact_z = float(frag)
                except ValueError:
                    contact_z = contact_z
        stage = "tilt"
        act = {"tool": "act_press_tilt", "args": {"tx": target[0] if target else 0.03,
                                                  "ty": target[1] if target else 0.0,
                                                  "steps": 4}}
        note = "hold produced no success; expert-like press/tilt"
    else:
        stage = "settle"
        act = {"tool": "act_settle", "args": {"steps": 40}}
        note = "settling still"

    new_mem = {
        "stage": stage,
        "decisions": decisions,
        "ball_xyz": ball,
        "ball_pixels": ball_pixels,
        "ball_seen_step": ball_step,
        "ball_source": ball_source,
        "cups_seen": cups_seen,
        "target_xy": target,
        "contact_z": contact_z,
        "hold_rpy": orient,
        "note": note,
    }
    return {"memory": new_mem, "act": act}


def act_wait_cue(robot, max_steps=8):
    """Send zero (suppressed) commands while the ball cue is shown; stop once cups appear."""
    seen = []
    n = 0
    for _ in range(int(max_steps)):
        f = robot.now()
        cups = _cup_pixels(f) >= CUP_MIN_PIXELS
        ball = _ball_in_frame(f, "front")
        if cups and ball is None:
            return {"cups_visible": True, "steps": n, "step": int(f.step), "ball_frames": len(seen)}
        if ball is not None:
            seen.append([round(v, 4) for v in ball[:3]])
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP_OPEN)
        n += 1
    f = robot.now()
    return {"cups_visible": _cup_pixels(f) >= CUP_MIN_PIXELS, "steps": n, "step": int(f.step),
            "ball_frames": len(seen), "timeout": True}


def act_touch(robot, tx, ty, z_touch=0.103, hold_steps=6, max_steps=22, rpy=None):
    """Move above (tx, ty), descend through a funnel to z_touch or first contact, then hold still."""
    R_goal = _rpy_to_R(rpy) if rpy is not None else _rpy_to_R(np.asarray(robot.now().tcp, dtype=float)[3:6])
    stage = "move"
    z_hold = None
    contact = False
    held = 0
    prev_z = None
    prev_az = 0.0
    log = []
    steps = 0
    for _ in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        p = tcp[:3]
        ex, ey = tx - p[0], ty - p[1]
        hd = math.hypot(ex, ey)
        if stage == "move":
            frac = _clip((hd - 0.012) / 0.038, 0.0, 1.0)
            zg = z_touch + frac * (Z_HOVER - z_touch)
            dz_meas = None if prev_z is None else p[2] - prev_z
            if (hd < 0.02 and frac < 0.3 and dz_meas is not None and p[2] < 0.13
                    and dz_meas > 0.2 * 0.03 * prev_az and p[2] > zg + 0.004 and prev_az < -0.04):
                stage, z_hold, contact = "hold", p[2] - 0.001, True
            elif hd < 0.006 and abs(p[2] - z_touch) < 0.004:
                stage, z_hold = "hold", z_touch
        if stage == "move":
            ax = _clip(1.5 * ex / 0.1, -1.0, 1.0)
            ay = _clip(1.5 * ey / 0.1, -1.0, 1.0)
            az = _clip(1.5 * (zg - p[2]) / 0.1, -1.0, 1.0)
            if p[2] < 0.15:
                az = max(az, -0.5)
        else:
            ax = _clip(ex / 0.1, -0.3, 0.3)
            ay = _clip(ey / 0.1, -0.3, 0.3)
            az = _clip((z_hold - p[2]) / 0.1, -0.05, 0.3)
            held += 1
        rot = _rot_cmd(tcp, R_goal)
        prev_z, prev_az = p[2], az
        robot.step(ax, ay, az, rot[0], rot[1], rot[2], GRIP_OPEN)
        steps += 1
        log.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3), stage[0]])
        if stage == "hold" and held >= int(hold_steps):
            break
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    return {"stage": stage, "contact": contact, "contact_z": None if z_hold is None else round(float(z_hold), 4),
            "steps": steps, "step": int(f.step), "tcp": [round(float(v), 4) for v in tcp[:3]],
            "xy_err": round(math.hypot(tx - tcp[0], ty - tcp[1]), 4),
            "qvel_est": _qvel(robot.frames()), "trace": log[-8:]}


def act_press_tilt(robot, tx, ty, steps=4):
    """Expert-like pressing: push down/back with a pitch tilt while staying over (tx, ty)."""
    n = 0
    for _ in range(int(steps)):
        f = robot.now()
        p = np.asarray(f.tcp, dtype=float)
        ax = _clip(-0.3 + 2.0 * (tx - p[0]) / 0.1, -0.6, 0.3)
        ay = _clip((ty - p[1]) / 0.1, -0.3, 0.3)
        robot.step(ax, ay, -0.6, 0.2, -1.0, 0.1, 0.8)
        n += 1
    f = robot.now()
    return {"steps": n, "step": int(f.step), "tcp": [round(float(v), 4) for v in np.asarray(f.tcp)[:6]],
            "qvel_est": _qvel(robot.frames())}


def act_settle(robot, steps=40):
    """Send zero arm commands (hold the measured pose) so the arm comes to rest."""
    n = 0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8)
        n += 1
    f = robot.now()
    return {"steps": n, "step": int(f.step), "qvel_est": _qvel(robot.frames())}
