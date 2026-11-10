"""BlinkCountButtonPressEasy policy: count blue blinks until the status lamp turns green, press the red
button that many times, then press the black button. Perception: front RGB-D deprojection."""

GRIP = -1.0            # expert keeps the gripper closed
HOVER_Z = 0.15         # approach height (expert ~0.135-0.145 before descending; cap top ~0.060)
LIFT_Z = 0.197         # lift ends at >0.185. R004: lifts 0.146/0.152 FAILED; R002 lifts 0.178 OK (expert min 0.167)
DWELL = 4              # steps after first contact (=5 frames below 0.058). R005: 5 frames + high lift OK
CONTACT_Z = 0.058      # TCP below this = on the cap (cap top 0.060, hard stop ~0.053)
PRESS_CMD = -0.6       # dz command while descending onto / first pushing the cap (expert ~ -0.7)
HARD_PUSH = 3          # steps of PRESS_CMD after first contact (expert ~4 steps of -0.7, then ~0)
HOLD_CMD = -0.15       # gentle dz while holding the cap down (R002: continuous -0.6 tilted the arm, x slid 3.8 cm)
BLACK_OFFSET = (0.0, 0.1595)   # black centre - red centre, measured in all 50 demos


# ---------------------------------------------------------------- perception helpers
def _masks(img):
    r = img[..., 0].astype(np.int32)
    g = img[..., 1].astype(np.int32)
    b = img[..., 2].astype(np.int32)
    blue = (b > 150) & (r < 80) & (g < 80)
    green = (g > 180) & (r < 100) & (b < 100)
    return blue, green


def _lamp_state(frame):
    blue, green = _masks(frame.front_rgb)
    return int(blue.sum()), int(green.sum())


def _deproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    d = depth[v, u].astype(np.float64) / 1000.0
    ok = d > 0
    u, v, d = u[ok], v[ok], d[ok]
    if len(d) == 0:
        return np.zeros((0, 3))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    cam = np.stack([x, y, d], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (cam - t) @ R


def _cap_points(img, depth, K, E, kind):
    r = img[..., 0].astype(np.int32)
    g = img[..., 1].astype(np.int32)
    b = img[..., 2].astype(np.int32)
    if kind == "red":
        m = (r > 110) & (g > 60) & (np.abs(g - b) <= 8) & (r - g > 40)
    else:
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        m = (mx - mn <= 4) & (mx < 90) & (mx > 20)
    P = _deproject(depth, K, E, m)
    P = P[(P[:, 2] > 0.005) & (P[:, 2] < 0.1)]
    if len(P) < 10:
        return None
    top = P[P[:, 2] > P[:, 2].max() - 0.004]
    return top


def _locate_buttons(frame):
    """Red and black cap top centres (world) from the front camera; wrist camera as fallback."""
    info = {}
    red = None
    for cam in ("front", "wrist"):
        img = getattr(frame, cam + "_rgb")
        dep = getattr(frame, cam + "_depth")
        K = getattr(frame, cam + "_K")
        E = getattr(frame, cam + "_extrinsic")
        top = _cap_points(img, dep, K, E, "red")
        if top is not None and len(top) >= 20:
            red = top.mean(0)
            info["red_src"] = cam
            info["red_px"] = int(len(top))
            break
    if red is None:
        return None, None, info
    black = None
    top = _cap_points(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic, "black")
    pred = np.array([red[0] + BLACK_OFFSET[0], red[1] + BLACK_OFFSET[1], red[2]])
    if top is not None:
        near = top[np.hypot(top[:, 0] - pred[0], top[:, 1] - pred[1]) < 0.04]
        if len(near) >= 20:
            black = near.mean(0)
            info["black_px"] = int(len(near))
    if black is None or np.hypot(black[0] - pred[0], black[1] - pred[1]) > 0.02:
        info["black_src"] = "offset"
        black = pred
    else:
        info["black_src"] = "front"
    return [float(v) for v in red], [float(v) for v in black], info


def _count_cue(frames, blue_on, blinks):
    """Rising edges of the blue lamp, and whether the status lamp is green, over frames."""
    green_step = None
    for f in frames:
        nb, ng = _lamp_state(f)
        on = nb > 20
        if on and not blue_on:
            blinks += 1
        blue_on = on
        if ng > 20 and green_step is None:
            green_step = int(f.step)
    return blue_on, blinks, green_step


def _count_presses(frames, below):
    """Contact events (TCP z below CONTACT_Z, then above CONTACT_Z+0.03) in frames."""
    n = 0
    for f in frames:
        z = float(f.tcp[2])
        if not below and z < CONTACT_Z:
            below = True
            n += 1
        elif below and z > CONTACT_Z + 0.03:
            below = False
    return n, below


# ---------------------------------------------------------------- rotation helpers
def _rpy_R(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return v * (ang / (2.0 * s))


# ---------------------------------------------------------------- control helpers
def _command(tcp, prev, target, Rt, kp=0.7, kd=0.5, cap=0.85, zcap=None, aprev=None, rate=0.35):
    p = np.asarray(tcp[:3], dtype=np.float64)
    v = np.zeros(3) if prev is None else p - np.asarray(prev[:3], dtype=np.float64)
    err = np.asarray(target, dtype=np.float64) - p
    a = (kp * err - kd * v) / 0.1
    a = np.clip(a, -cap, cap)
    if aprev is not None:                         # ramp like the expert (limits drive demand)
        ap = np.asarray(aprev[:3], dtype=np.float64)
        a = np.clip(a, ap - rate, ap + rate)
    if zcap is not None:
        a[2] = max(min(a[2], zcap[1]), zcap[0])
    w = _rotvec(Rt @ _rpy_R(tcp[3:6]).T)        # world-frame rotation still needed
    r = np.clip(-0.5 * w / 0.1, -0.3, 0.3)       # step rotation is ~ -0.1*r about world axes
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])], err, w


def _step(robot, cmd):
    return robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], GRIP)


# ---------------------------------------------------------------- policy
def perceive_decide(obs, goal, memory, last):
    """Track the blink cue and press progress from frames; choose the next stage action."""
    m = dict(memory)
    frames = obs.frames
    cur = obs.current
    if m.get("stage") is None:
        red, black, info = _locate_buttons(frames[0])
        if red is None:
            red, black, info = _locate_buttons(cur)
        m = {"stage": "watch", "blink_count": 0, "blue_on": False, "green_step": -1,
             "red_button": red, "black_button": black, "red_presses": 0, "below": False,
             "black_tries": 0, "rot_target": [float(v) for v in cur.tcp[3:6]],
             "log": "init " + str(info)}
    if m["red_button"] is None:
        red, black, info = _locate_buttons(cur)
        m["red_button"], m["black_button"] = red, black
        m["log"] = m["log"] + " | relocate " + str(info)

    # cue counting (only before green)
    if m["green_step"] < 0:
        blue_on, blinks, gstep = _count_cue(frames, bool(m["blue_on"]), int(m["blink_count"]))
        m["blue_on"], m["blink_count"] = blue_on, blinks
        if gstep is not None:
            m["green_step"] = gstep
            m["log"] = m["log"] + " | green@%d blinks=%d" % (gstep, blinks)
    # press progress from measured TCP z
    if m["stage"] in ("press_red", "press_black", "after_black"):
        n, below = _count_presses(frames, bool(m["below"]))
        if m["stage"] == "press_red":
            m["red_presses"] = int(m["red_presses"]) + n
        m["below"] = below

    if m["green_step"] < 0:
        m["stage"] = "watch"
        return {"memory": m, "act": {"tool": "act_watch", "args": {"max_steps": 60}}}

    target_n = max(1, int(m["blink_count"]))
    if m["red_button"] is None:
        return {"memory": m, "act": {"tool": "act_hold", "args": {"steps": 2}}}
    if m["stage"] == "watch":
        m["stage"] = "press_red"
    if m["stage"] == "press_red" and int(m["red_presses"]) >= target_n:
        m["stage"] = "press_black"
    if m["stage"] == "press_red":
        b = m["red_button"]
        return {"memory": m, "act": {"tool": "act_press", "args": {
            "x": b[0], "y": b[1], "rot_target": m["rot_target"], "lift": True}}}
    if m["stage"] == "press_black" and int(m["black_tries"]) < 2:
        m["black_tries"] = int(m["black_tries"]) + 1
        b = m["black_button"]
        return {"memory": m, "act": {"tool": "act_press", "args": {
            "x": b[0], "y": b[1], "rot_target": m["rot_target"], "lift": True}}}
    m["stage"] = "after_black"
    return {"memory": m, "act": {"tool": "act_hold", "args": {"steps": 10}}}


def act_watch(robot, max_steps=60):
    """Hold still (zero arm command, gripper closed) until the status lamp turns green or max_steps.
    Returns steps used, blue on-frame count and the green detection step."""
    n_blue = 0
    for i in range(max_steps):
        f = robot.now()
        nb, ng = _lamp_state(f)
        if ng > 20:
            return {"green": True, "steps": i, "green_step": int(f.step), "blue_frames": n_blue}
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP)
        if _lamp_state(robot.now())[0] > 20:
            n_blue += 1
    return {"green": False, "steps": max_steps, "blue_frames": n_blue}


def act_hold(robot, steps=5):
    """Send zero arm commands for `steps` steps (gripper closed)."""
    for i in range(steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP)
    return {"held": steps, "tcp": [round(float(v), 4) for v in robot.now().tcp]}


def act_press(robot, x, y, rot_target, lift=True, hover_z=HOVER_Z, lift_z=LIFT_Z,
              dwell=DWELL, max_steps=80):
    """Move above (x, y) at hover_z, descend onto the cap, keep pushing for `dwell` steps after the
    first frame below CONTACT_Z, then lift to lift_z. Returns measured phase data."""
    Rt = _rpy_R(rot_target)
    out = {"target": [round(x, 4), round(y, 4)]}
    prev = None
    aprev = [0.0, 0.0, 0.0]
    used = 0
    start = robot.now()
    # optional kinematic preview of the hover pose (diagnostic only)
    try:
        chk = robot.dry_run(x, y, hover_z, rot_target[0], rot_target[1], rot_target[2], mode="absolute")
        out["hover_check"] = {"ok": bool(chk.get("within_joint_limits")),
                              "margin": chk.get("min_joint_margin_rad")}
    except Exception as e:
        out["hover_check"] = "unavailable"
    # phase 1: approach hover
    phase_steps = 0
    while used < max_steps:
        f = robot.now()
        p = f.tcp
        exy = math.hypot(x - float(p[0]), y - float(p[1]))
        if exy < 0.006 and hover_z - 0.02 < float(p[2]) < hover_z + 0.05:
            break
        cmd, err, w = _command(p, prev, [x, y, hover_z], Rt, aprev=aprev)
        prev = p
        _step(robot, cmd)
        aprev = cmd
        used += 1
        phase_steps += 1
    out["approach_steps"] = phase_steps
    out["approach_xy_err"] = round(math.hypot(x - float(robot.now().tcp[0]), y - float(robot.now().tcp[1])), 4)
    # phase 2: descend and press
    phase_steps = 0
    stall = 0
    zmin = 1.0
    contact_step = -1
    pushed = 0
    prevz = float(robot.now().tcp[2])
    while used < max_steps:
        f = robot.now()
        p = f.tcp
        z = float(p[2])
        zmin = min(zmin, z)
        if z < CONTACT_Z:
            stall += 1
            if contact_step < 0:
                contact_step = phase_steps
                out["contact_xy_err"] = round(math.hypot(x - float(p[0]), y - float(p[1])), 4)
        if contact_step >= 0:
            exy_c = math.hypot(x - float(p[0]), y - float(p[1]))
            out["contact_max_xy_err"] = round(max(out.get("contact_max_xy_err", 0.0), exy_c), 4)
            rerr = float(np.linalg.norm(_rotvec(Rt @ _rpy_R(p[3:6]).T)))
            out["contact_max_rot_err"] = round(max(out.get("contact_max_rot_err", 0.0), rerr), 4)
            pushed += 1
            if pushed > dwell:
                break
        if z > 0.075:
            cmd, err, w = _command(p, prev, [x, y, 0.03], Rt, zcap=(-0.7, 1.0), aprev=aprev)
        else:
            cmd, err, w = _command(p, prev, [x, y, 0.03], Rt, aprev=aprev)
            cmd[2] = PRESS_CMD if pushed <= HARD_PUSH else HOLD_CMD
        prev = p
        prevz = z
        _step(robot, cmd)
        aprev = cmd
        used += 1
        phase_steps += 1
        if phase_steps > 40:
            break
    out["press_steps"] = phase_steps
    out["frames_below"] = stall
    out["contact_step"] = contact_step
    out["zmin"] = round(zmin, 4)
    out["pressed"] = bool(zmin < CONTACT_Z)
    # phase 3: lift
    phase_steps = 0
    aprev = [aprev[0], aprev[1], 0.0]     # release the downward push immediately
    if lift:
        while used < max_steps:
            f = robot.now()
            p = f.tcp
            if float(p[2]) > lift_z - 0.012:
                break
            cmd, err, w = _command(p, prev, [x, y, lift_z + 0.04], Rt, aprev=aprev)
            prev = p
            _step(robot, cmd)
            aprev = cmd
            used += 1
            phase_steps += 1
    out["lift_steps"] = phase_steps
    end = robot.now()
    out["end_tcp"] = [round(float(v), 4) for v in end.tcp]
    out["rot_err"] = round(float(np.linalg.norm(_rotvec(Rt @ _rpy_R(end.tcp[3:6]).T))), 4)
    out["steps"] = used
    out["end_step"] = int(end.step)
    return out
