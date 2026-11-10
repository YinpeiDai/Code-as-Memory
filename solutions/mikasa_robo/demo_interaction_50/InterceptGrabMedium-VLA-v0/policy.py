"""InterceptGrab policy: detect the rolling red ball in the front/wrist RGB-D
views, predict its constant-velocity path, wait above the path with a top-down
open gripper (fingers spread along the rolling direction y), match the ball's
velocity, descend around it, close, then lift slightly and hold still."""

Z_HOVER = 0.08     # TCP height while matching velocity (finger tips above ball top 0.04)
Z_GRASP = 0.018    # TCP height for closing (ball centre is at z ~0.02)
Y_WAIT = -0.28     # y where the gripper waits for the incoming ball
W_SMOOTH = 0.03    # smoothing width of the wait->track hand-over
VZ_DESCENT = 0.022 # max vertical speed (m/step) in catch; R001/R002 showed a 16 mm x jump at full-speed descent


def _smax(a, b, w):
    d = a - b
    s = math.sqrt(d * d + w * w)
    return 0.5 * (a + b + s), 0.5 * (1.0 + d / s)


def _margin(joints):
    lo = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    hi = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    q = np.asarray(joints, dtype=np.float64)
    m = np.minimum(q - lo, hi - q)
    return float(m.min()), int(np.argmin(m)) + 1


def act_catch(robot, y_wait=-0.28, z_hover=0.08, z_grasp=0.018, max_steps=50):
    """Intercept and grasp the rolling ball. Runs a closed loop (one native step
    per iteration): detect ball -> fit constant-velocity track -> wait at y_wait
    above the path -> match velocity -> descend -> close gripper -> check finger
    width. Returns status grasped/missed/lost/timeout with measured quantities."""
    hist = []
    trace = []
    events = []
    f = robot.now()
    p_last = np.array(f.tcp[:3], dtype=np.float64)
    dp_prev = np.zeros(3)
    phase = "acquire"
    aligned = 0
    close_steps = 0
    grip = 1.0
    status = "timeout"
    b = None
    v = np.zeros(3)
    for i in range(max_steps):
        f = robot.now()
        t = int(f.step)
        tcp = np.array(f.tcp, dtype=np.float64)
        p = tcp[:3]
        if i > 0:
            dp_prev = p - p_last
        p_last = p.copy()
        fing = float(np.mean(f.fingers))
        det = _detect_ball(f)
        if det is not None:
            hist.append([t, float(det["pos"][0]), float(det["pos"][1]), float(det["pos"][2])])
        est = _fit_ball(hist, t)
        if est is not None:
            b, v = est[0], est[1]
        # ---- targets -----------------------------------------------------
        if b is None:
            target = np.array([-0.15, -0.15, 0.15])
            tvel = np.zeros(3)
            k = 0.35
        else:
            yd, s = _smax(float(b[1]), y_wait, W_SMOOTH)
            target = np.array([float(b[0]), yd, z_hover])
            tvel = np.array([float(v[0]), float(v[1]) * s, 0.0])
            k = 0.6
            e_xy = float(np.linalg.norm(b[:2] - p[:2]))
            rel_v = float(np.linalg.norm(dp_prev[:2] - v[:2]))
            tracking = float(b[1]) > y_wait + 0.01
            if phase in ("acquire", "hover"):
                phase = "hover"
                if tracking and len(hist) >= 3 and e_xy < 0.012 and rel_v < 0.008:
                    aligned += 1
                else:
                    aligned = 0
                if aligned >= 2:
                    phase = "descend"
                    events.append(["descend", t, round(e_xy, 4), round(rel_v, 4)])
            if phase == "descend":
                target[2] = z_grasp
                if e_xy > 0.025 and p[2] > 0.045:
                    phase = "hover"
                    aligned = 0
                    events.append(["abort_descent", t, round(e_xy, 4)])
                elif p[2] < z_grasp + 0.012 and e_xy < 0.015:
                    phase = "closing"
                    grip = -1.0
                    events.append(["close", t, round(e_xy, 4), [round(x, 3) for x in p]])
            if phase == "closing":
                target[2] = z_grasp
                close_steps += 1
                if close_steps >= 2 and 0.008 < fing < 0.026:
                    status = "grasped"
                    events.append([status, t, round(fing, 4)])
                    break
                if close_steps > 4:
                    if 0.008 < fing < 0.034:
                        status = "grasped"
                    else:
                        status = "missed"
                    events.append([status, t, round(fing, 4)])
                    break
            if float(b[1]) > 0.3 and phase != "closing":
                status = "lost"
                break
        a = _track_cmd(p, dp_prev, target, tvel, k=k, vzmax=VZ_DESCENT)
        r, rot_err = _orient_cmd(tcp, 0.0)
        mmin, mj = _margin(f.joints)
        trace.append([t, phase[:3], [round(x, 3) for x in p],
                      None if b is None else [round(float(x), 3) for x in b[:2]],
                      round(fing, 3), round(rot_err, 3), round(mmin, 2)])
        robot.step(float(a[0]), float(a[1]), float(a[2]),
                   float(r[0]), float(r[1]), float(r[2]), grip)
    f = robot.now()
    return {"status": status, "phase": phase, "events": events,
            "ball": None if b is None else [round(float(x), 4) for x in b],
            "ball_vel": [round(float(x), 4) for x in v],
            "n_det": len(hist), "fingers": [round(float(x), 4) for x in f.fingers],
            "tcp": [round(float(x), 4) for x in f.tcp], "step": int(f.step),
            "trace": trace[-12:]}


def act_hold(robot, z_lift=0.03, max_steps=60, still_steps=8):
    """Keep the gripper closed and bring the held ball to rest: zero translation
    commands (controller target = measured pose) for `still_steps` steps, then
    alternate very slow lift pulses (<=~0.06 m/s, up to z_lift above the grasp
    height) with still periods. Evidence R009: success registered only once the
    held ball was nearly at rest, not while it moved at ~0.1 m/s.
    Stops early if the fingers close fully (ball lost)."""
    f = robot.now()
    p_last = np.array(f.tcp[:3], dtype=np.float64)
    z0 = float(p_last[2])
    dp_prev = np.zeros(3)
    trace = []
    status = "held"
    for i in range(max_steps):
        f = robot.now()
        tcp = np.array(f.tcp, dtype=np.float64)
        p = tcp[:3]
        if i > 0:
            dp_prev = p - p_last
        p_last = p.copy()
        fing = float(np.mean(f.fingers))
        if fing < 0.006:
            status = "dropped"
            break
        az = 0.0
        if i >= still_steps:
            cyc = (i - still_steps) % 10
            if cyc < 3 and p[2] < z0 + z_lift:
                az = 0.08
        r = np.zeros(3)
        trace.append([int(f.step), [round(x, 3) for x in p], round(fing, 4),
                      round(float(np.linalg.norm(dp_prev)), 4), az])
        robot.step(0.0, 0.0, az, float(r[0]), float(r[1]), float(r[2]), -1.0)
    f = robot.now()
    return {"status": status, "fingers": [round(float(x), 4) for x in f.fingers],
            "tcp": [round(float(x), 4) for x in f.tcp], "step": int(f.step),
            "trace": trace[-10:]}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: catch -> hold; retry catch after a miss/drop."""
    stage = memory.get("stage") or "start"
    attempts = int(memory.get("attempts") or 0)
    grasped = bool(memory.get("grasped") or False)
    notes = memory.get("notes") or ""
    f = obs.current
    det = _detect_ball(f)
    ball = memory.get("ball_pos")
    if det is not None:
        ball = [round(float(x), 4) for x in det["pos"]]
    last = last or ""
    if stage == "start":
        nxt = "act_catch"
    elif '"status": "grasped"' in last or '"status": "held"' in last:
        nxt = "act_hold"
        grasped = True
    else:
        grasped = False
        nxt = "act_catch"
    if nxt == "act_catch":
        attempts += 1
        stage = "catch"
        act = {"tool": "act_catch", "args": {"y_wait": Y_WAIT if attempts == 1 else -0.05,
                                             "z_hover": Z_HOVER, "z_grasp": Z_GRASP}}
    else:
        stage = "hold"
        act = {"tool": "act_hold", "args": {"z_lift": 0.03}}
    notes = (notes + " | " if notes else "") + f"step{int(f.step)}:{stage}"
    return {"memory": {"stage": stage, "attempts": attempts, "grasped": grasped,
                       "ball_pos": ball, "notes": notes[-400:]},
            "act": act}
