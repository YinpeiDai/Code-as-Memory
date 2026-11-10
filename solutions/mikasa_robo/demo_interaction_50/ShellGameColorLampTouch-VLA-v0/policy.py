"""ShellGameColorLampTouch policy.

Observed in demos: at reset three coloured balls (red/green/blue) lie in a row on
the table; after 1-5 cue frames identical red mugs cover them and a lamp (in front
of the row, ~0.25 m further +x, ~0.13 m higher) lights in one of the three colours.
The arm is frozen (commands zeroed) while the balls are visible.  The expert then
moves the closed gripper onto the mug covering the ball whose colour matches the
lamp, ends with TCP ~0.05 m above the table at the mug centre and slows to rest;
success is registered after a small final command (hypothesis: TCP-in-target
region + robot static).
"""

COLORS = ("red", "green", "blue")
BALL_MAX_Z = 0.08        # balls deproject to z ~0.036 m
LAMP_MIN_Z = 0.10        # lamp deprojects to z ~0.165 m
LAMP_OFFSET = (0.248, 0.0, 0.129)   # lamp minus middle ball (demo mean, std 3 mm)
TARGET_DY = 0.004        # median expert final TCP offset from ball visible centre
TOUCH_Z = 0.07           # descent goal; fingertips stall on the 0.105 m mug top (TCP ~0.116) first
TRANSIT_Z = 0.14         # clear the 0.105 m mug tops while moving sideways
NATIVE_LIMIT = 30


def _color_masks(rgb):
    f = rgb.astype(np.int32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return {
        "red": (r > 150) & (g < 60) & (b < 60),
        "green": (g > 150) & (r < 80) & (b < 80),
        "blue": (b > 150) & (r < 80) & (g < 80),
    }


def _deproject(frame, ys, xs):
    """World xyz of front-camera pixels (rows ys, cols xs); drops zero depth."""
    d = frame.front_depth[ys, xs].astype(np.float64) / 1000.0
    ok = d > 0
    ys, xs, d = ys[ok], xs[ok], d[ok]
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    xc = np.stack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d], -1)
    return (xc - E[:, 3]) @ E[:, :3]


def _detect(frame, balls=None):
    """Balls (z<0.08) and lit lamp (z>0.10) per colour in one front frame."""
    m = _color_masks(frame.front_rgb)
    found, lamp = {}, {}
    for name in COLORS:
        ys, xs = np.nonzero(m[name])
        if len(ys) < 5:
            continue
        W = _deproject(frame, ys, xs)
        if len(W) == 0:
            continue
        low = W[W[:, 2] < BALL_MAX_Z]
        if len(low) >= 8:
            c = np.median(low, axis=0)
            close = low[np.linalg.norm(low[:, :2] - c[:2], axis=1) < 0.04]
            if len(close) >= 8:
                found[name] = [float(v) for v in close.mean(axis=0)]
        high = W[W[:, 2] > LAMP_MIN_Z]
        if balls and len(balls) == 3:
            mid = np.mean(np.array([balls[k] for k in COLORS]), axis=0)
            exp = mid + np.array(LAMP_OFFSET)
            high = high[np.linalg.norm(high - exp, axis=1) < 0.10]
        if len(high) >= 10:
            lamp[name] = int(len(high))
    return found, lamp


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _log_R(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _rot_cmd(tcp, ref_rpy, gain=0.5, cap=0.3):
    """Normalised (drx,dry,drz) turning the measured orientation toward ref_rpy.
    Native offset is -0.1*r (base axes, left-multiplied), so r = -gain*w/0.1."""
    if ref_rpy is None:
        return [0.0, 0.0, 0.0]
    Rc = _rpy_to_R(*[float(v) for v in tcp[3:6]])
    Rd = _rpy_to_R(*[float(v) for v in ref_rpy])
    w = _log_R(Rd @ Rc.T)
    r = -gain * w / 0.1
    return [float(np.clip(v, -cap, cap)) for v in r]


def _qspeed(frames):
    """Max |dq| per step (rad) between the last two frames; None if unknown."""
    if len(frames) < 2:
        return None
    a = np.asarray(frames[-1].joints, dtype=np.float64)
    b = np.asarray(frames[-2].joints, dtype=np.float64)
    return float(np.abs(a - b).max())


def perceive_decide(obs, goal, memory, last):
    """Record ball colours/positions while visible, read the lamp colour once the
    mugs cover the balls, then touch the mug over the matching ball and settle."""
    mem = {
        "balls": memory.get("balls") or {},
        "ball_step": memory.get("ball_step", -1),
        "lamp_color": memory.get("lamp_color") or "",
        "lamp_step": memory.get("lamp_step", -1),
        "stage": memory.get("stage") or "cue",
        "attempt": memory.get("attempt") or 0,
        "ref_rpy": memory.get("ref_rpy") or [],
        "log": memory.get("log") or [],
    }
    frames = list(obs.frames)
    cur = frames[-1]
    if not mem["ref_rpy"]:
        mem["ref_rpy"] = [float(v) for v in frames[0].tcp[3:6]]
    for fr in frames:
        found, lamp = _detect(fr, mem["balls"] if len(mem["balls"]) == 3 else None)
        if len(found) == 3 and (len(mem["balls"]) < 3 or not mem["lamp_color"]):
            mem["balls"] = found
            mem["ball_step"] = int(fr.step)
        if lamp and not mem["lamp_color"] and len(found) < 3:
            best = max(lamp, key=lambda k: lamp[k])
            mem["lamp_color"] = best
            mem["lamp_step"] = int(fr.step)
    if last is not None:
        mem["log"] = (mem["log"] + [str(last)[:300]])[-8:]
    step = int(cur.step)
    if not mem["lamp_color"]:
        mem["stage"] = "cue"
        return {"memory": mem, "act": {"tool": "act_wait_for_lamp", "args": {"max_steps": 12}}}
    color = mem["lamp_color"]
    if color in mem["balls"]:
        b = mem["balls"][color]
        tx, ty = b[0], b[1] + TARGET_DY
    elif len(mem["balls"]) > 0:
        # fallback: demos always order red, green, blue along +y with 0.196 m spacing
        k = list(mem["balls"].keys())[0]
        b = mem["balls"][k]
        dy = (COLORS.index(color) - COLORS.index(k)) * 0.196
        tx, ty = b[0], b[1] + dy + TARGET_DY
    else:
        tx, ty = 0.0, 0.0 + TARGET_DY
    # retry variants if the arm comes to rest without the episode ending
    variants = [
        {"mode": "top", "tz": TOUCH_Z},
        {"mode": "side", "tz": TOUCH_Z},
        {"mode": "top", "tz": TOUCH_Z - 0.03},
        {"mode": "side", "tz": TOUCH_Z - 0.02},
    ]
    a = min(int(mem["attempt"]), len(variants) - 1)
    v = variants[a]
    mem["stage"] = "touch_%d_%s" % (a, v["mode"])
    mem["attempt"] = int(mem["attempt"]) + 1
    return {"memory": mem, "act": {"tool": "act_touch", "args": {
        "tx": float(tx), "ty": float(ty), "tz": float(v["tz"]), "mode": v["mode"],
        "ref_rpy": mem["ref_rpy"]}}}


def act_wait_for_lamp(robot, max_steps=12):
    """Hold the arm (zero motion, gripper closing) until the lamp lights and the
    balls are covered; returns detected lamp colour and the step it was seen."""
    seen = {}
    balls = None
    n = 0
    for n in range(1, max_steps + 1):
        fr = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        found, lamp = _detect(fr, None)
        if len(found) == 3:
            balls = found
        if lamp and len(found) < 3:
            seen = lamp
            break
    fr = robot.now()
    return {"steps": n, "step": int(fr.step), "lamp_pixels": seen,
            "balls_last_seen": balls is not None}


def act_touch(robot, tx, ty, tz, mode="top", side_dx=0.048, side_dy=0.006, side_z=0.112,
              above_z=0.15, ref_rpy=None, gain=1.2, plunge_gain=2.0, descend_cap=0.5,
              side_tol=0.012, arrive_tol=0.008, still_rad=0.004, still_steps=3):
    """Touch the mug whose top centre is above (tx,ty).
    mode "top": move (orientation held, lifted while far) to (tx,ty,above_z), descend
    vertically with zero rotation and capped speed toward tz (below the 0.105 m mug top),
    and on contact stall (or arrival) send zero commands so the arm comes to rest.
    mode "side": expert-style side point (+x of the mug at mug-top height) then a
    diagonal plunge.  Returns diagnostics only if the arm rests without the episode
    ending."""
    trace = []
    stage = "transit"
    still = 0
    stall = 0
    start = int(robot.now().step)
    if mode == "side":
        way = np.array([tx + side_dx, ty + side_dy, side_z])
    else:
        way = np.array([tx, ty, above_z])
    goal = np.array([tx, ty, tz])
    for i in range(NATIVE_LIMIT + 5):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=np.float64)
        if stage == "transit":
            exy = math.hypot(way[0] - tcp[0], way[1] - tcp[1])
            lift = min(0.05, max(0.0, exy - 0.02) * 0.6)   # stay above neighbouring mugs
            err = way - tcp[:3] + np.array([0.0, 0.0, lift])
            if abs(err[0]) < side_tol and abs(err[1]) < side_tol and abs(err[2]) < side_tol:
                stage = "plunge"
        if stage == "plunge":
            err = goal - tcp[:3]
            if np.linalg.norm(err) < arrive_tol or stall >= 1:
                stage = "hold"
        if stage == "transit":
            u = np.clip(gain * err / 0.1, -1.0, 1.0)
            cmd = [float(u[0]), float(u[1]), float(u[2])] + _rot_cmd(tcp, ref_rpy)
        elif stage == "plunge":
            if mode == "side":
                u = np.clip(plunge_gain * err / 0.1, -1.0, 1.0)
            else:
                u = np.clip(gain * err / 0.1, -1.0, 1.0)
                u[2] = max(-descend_cap, u[2])
            cmd = [float(u[0]), float(u[1]), float(u[2]), 0.0, 0.0, 0.0]
        else:
            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        fr2 = robot.step(*(cmd + [-1.0]))
        frs = robot.frames()
        qs = _qspeed(frs)
        t2 = np.asarray(fr2.tcp, dtype=np.float64)
        dz = float(t2[2] - tcp[2])
        if stage == "plunge" and t2[2] < 0.13 and dz > -0.002:
            stall += 1
        elif stage == "plunge":
            stall = 0
        trace.append([int(fr2.step), stage[0]] + [round(float(v), 3) for v in t2[:3]]
                     + [round(float(t2[4]), 2)]
                     + [round(float(v), 2) for v in cmd[:3]] + [None if qs is None else round(qs, 4)])
        if stage == "hold" and qs is not None and qs < still_rad:
            still += 1
        else:
            still = 0
        if stage == "hold" and still >= still_steps:
            break
    t = np.asarray(robot.now().tcp, dtype=np.float64)
    return {"result": "rested_without_success", "mode": mode, "start_step": start,
            "end_step": int(robot.now().step), "tcp": [round(float(v), 4) for v in t],
            "target": [round(tx, 4), round(ty, 4), round(tz, 4)],
            "on_top": bool(t[2] > 0.095), "trace_tail": trace[-8:]}
