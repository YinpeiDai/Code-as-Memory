"""BlinkCountButtonPress policy: count blue-lamp blinks during the cue, wait for the
status lamp to turn green, press the red (pink-capped) button that many times, then
press the black button. Perception uses exact flat-shaded colours seen in the fixed
front camera plus its depth/calibration; control is closed-loop on measured TCP."""

PINK_CAP = (211, 136, 136)   # red button cap, top face (front camera, flat shading)
BLACK_CAP = (73, 73, 73)     # black button cap, top face
BLACK_OFFSET = (0.0, 0.16, 0.0)  # black cap minus red cap in all 50 demos (std 1 mm)
CAP_TOP_Z = 0.060            # measured cap top height in demos
HOVER_Z = 0.19               # expert lifts to ~0.198 between presses
PRESS_Z = 0.053              # expert TCP bottom during red presses
PUSH_DZ = -0.7               # expert command while pushing the cap down
ROT_GAIN = 1.0               # rotation offset = ROT_GAIN * orientation error
ROT_MAX = 0.5                # max normalized rotation command norm
BLUE_MIN_PIXELS = 60         # blue lamp on: 178..239 px; off: 0 px
GREEN_MIN_PIXELS = 60        # green lamp: >=179 px once on


def _lamp_state(frame):
    f = frame.front_rgb.astype(np.int32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    blue = int(((b > 150) & (r < 80) & (g < 80)).sum())
    green = int(((g > 180) & (r < 100) & (b < 100)).sum())
    red = int(((r > 180) & (g < 60) & (b < 60)).sum())
    return blue, green, red


def _backproject(u, v, depth_mm, K, ext):
    z = depth_mm / 1000.0
    xc = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    R = ext[:, :3]
    t = ext[:, 3]
    return R.T.dot(xc - t)


def _cap_world(frame, colour, tol):
    f = frame.front_rgb.astype(np.int32)
    m = np.abs(f - np.array(colour)).max(axis=-1) <= tol
    ys, xs = np.nonzero(m)
    if len(xs) < 15:
        return None, int(len(xs))
    D = frame.front_depth
    dd = D[ys, xs]
    ok = dd > 0
    if ok.sum() < 10:
        return None, int(len(xs))
    K = frame.front_K
    ext = frame.front_extrinsic
    pts = np.array([_backproject(float(x), float(y), float(z), K, ext)
                    for x, y, z in zip(xs[ok], ys[ok], dd[ok])])
    return [float(v) for v in pts.mean(axis=0)], int(len(xs))


def _blank_memory():
    return {
        "phase": "cue",
        "blink_count": 0,
        "blue_on": False,
        "last_step": -1,
        "blink_steps": [],
        "go_step": -1,
        "red_pos": [],
        "black_pos": [],
        "presses_done": 0,
        "black_attempts": 0,
        "log": "",
    }


def perceive_decide(obs, goal, memory, last):
    """Track blue blinks and the go (green) lamp from every new frame, locate the
    buttons once, and choose the next stage: wait for go, one red press, black press."""
    mem = _blank_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    log = []
    # 1. button localisation from the first frame where both caps are visible
    if not mem["red_pos"]:
        cur = obs.frames[0]
        red, nr = _cap_world(cur, PINK_CAP, 8)
        blk, nb = _cap_world(cur, BLACK_CAP, 3)
        if red is not None:
            mem["red_pos"] = red
            if blk is not None and abs(blk[0] - red[0]) < 0.02 and abs(blk[1] - red[1] - 0.16) < 0.02:
                mem["black_pos"] = blk
                log.append("black cap seen")
            else:
                mem["black_pos"] = [red[0] + BLACK_OFFSET[0], red[1] + BLACK_OFFSET[1], red[2]]
                log.append("black from offset (n=%d)" % nb)
            log.append("red px=%d at %.3f,%.3f,%.3f" % (nr, red[0], red[1], red[2]))
    # 2. cue tracking over all new frames
    for fr in obs.frames:
        if fr.step <= mem["last_step"]:
            continue
        blue, green, redl = _lamp_state(fr)
        on = blue >= BLUE_MIN_PIXELS
        if mem["go_step"] < 0:
            if on and not mem["blue_on"]:
                mem["blink_count"] = int(mem["blink_count"]) + 1
                mem["blink_steps"] = list(mem["blink_steps"]) + [int(fr.step)]
            if green >= GREEN_MIN_PIXELS:
                mem["go_step"] = int(fr.step)
                log.append("go at step %d with %d blinks" % (fr.step, mem["blink_count"]))
        mem["blue_on"] = bool(on)
        mem["last_step"] = int(fr.step)
    # 3. stage selection
    if last is not None and isinstance(last, str):
        log.append("last: " + last[:200])
    if mem["phase"] == "cue" and mem["go_step"] >= 0:
        mem["phase"] = "press_red" if mem["blink_count"] > 0 else "press_black"
    if mem["phase"] == "press_red" and mem["presses_done"] >= mem["blink_count"]:
        mem["phase"] = "press_black"
    if mem["log"]:
        mem["log"] = (mem["log"] + " | " + "; ".join(log))[-1500:] if log else mem["log"]
    else:
        mem["log"] = "; ".join(log)
    if mem["phase"] == "cue" or not mem["red_pos"]:
        act = {"tool": "act_wait_for_go", "args": {"max_steps": 300}}
    elif mem["phase"] == "press_red":
        rp = mem["red_pos"]
        mem["presses_done"] = int(mem["presses_done"]) + 1
        act = {"tool": "act_press", "args": {"x": rp[0], "y": rp[1], "index": mem["presses_done"]}}
    else:
        bp = mem["black_pos"]
        mem["black_attempts"] = int(mem["black_attempts"]) + 1
        act = {"tool": "act_press", "args": {"x": bp[0], "y": bp[1], "index": -mem["black_attempts"]}}
    return {"memory": mem, "act": act}


def act_wait_for_go(robot, max_steps=300):
    """Hold still (zero motion, gripper open) until the status lamp turns green or
    max_steps elapse. Returns steps used, blue-on frame count and whether go was seen."""
    n = 0
    blue_frames = 0
    go = False
    blue, green, redl = _lamp_state(robot.now())
    if green >= GREEN_MIN_PIXELS:
        return {"steps": 0, "go_seen": True, "blue_frames": 0}
    while n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        blue, green, redl = _lamp_state(robot.now())
        if blue >= BLUE_MIN_PIXELS:
            blue_frames += 1
        if green >= GREEN_MIN_PIXELS:
            go = True
            break
    return {"steps": n, "go_seen": go, "blue_frames": blue_frames, "step": int(robot.now().step)}


def _rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz.dot(Ry).dot(Rx)


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        # near pi: use diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return axis * ang
    return v / (2.0 * s) * ang


def _rot_cmd(tcp, goal_rpy, k=ROT_GAIN, max_norm=ROT_MAX):
    """Rotation command toward goal orientation. Native CPU semantics (verified sign from
    demo regression: world-frame increment ~ -c*r): R_target = R(-0.1 r) R_meas, so
    r = -k * rotvec(R_goal R_meas^T) / 0.1."""
    Rm = _rpy_mat(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rg = _rpy_mat(float(goal_rpy[0]), float(goal_rpy[1]), float(goal_rpy[2]))
    w = _rotvec(Rg.dot(Rm.T))
    r = -k * w / 0.1
    n = float(np.linalg.norm(r))
    if n > max_norm:
        r = r * (max_norm / n)
    return r, float(np.linalg.norm(w))


GOAL_RPY = (math.pi, 0.0, 0.0)  # vertical tool; initial demo pose is within ~0.05 rad


def _move_to(robot, target, tol, max_steps, gain=10.0, grip=-1.0, goal_rpy=None):
    """P-control of TCP position toward target (world) with orientation hold."""
    n = 0
    err = None
    while n < max_steps:
        tcp = robot.now().tcp
        p = np.array(tcp[:3], dtype=float)
        err = np.array(target, dtype=float) - p
        r = np.zeros(3)
        rerr = 0.0
        if goal_rpy is not None:
            r, rerr = _rot_cmd(tcp, goal_rpy)
        if np.abs(err).max() < tol and rerr < 0.03:
            break
        cmd = np.clip(err * gain, -1.0, 1.0)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    p = np.array(robot.now().tcp[:3], dtype=float)
    return n, [float(v) for v in (np.array(target) - p)]


def act_press(robot, x, y, index=1):
    """One press of the button whose cap centre is at world (x, y): move to hover
    height above it, descend, push the cap down until the TCP stalls near the
    expert bottom height, hold briefly, then lift back to hover height.
    Returns measured hover error, bottom z, stall/hold steps and steps used."""
    out = {"index": index, "target": [round(x, 4), round(y, 4)]}
    s0 = int(robot.now().step)
    tcp0 = [float(v) for v in robot.now().tcp]
    out["tcp0"] = [round(v, 4) for v in tcp0]
    # close fingers so a single compact fingertip presses the cap
    if float(robot.now().fingers[0]) > 0.005:
        for _ in range(3):
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    # hover above target
    n1, e1 = _move_to(robot, [x, y, HOVER_Z], 0.006, 60, goal_rpy=GOAL_RPY)
    out["hover_steps"] = n1
    out["hover_err"] = [round(v, 4) for v in e1]
    # descend fast to just above the cap
    n2, e2 = _move_to(robot, [x, y, CAP_TOP_Z + 0.02], 0.006, 40, goal_rpy=GOAL_RPY)
    out["descend_steps"] = n2
    # push down with a constant command, correcting xy, until stall or bottom
    zs = []
    rot_errs = []
    stall = 0
    n3 = 0
    while n3 < 25:
        p = np.array(robot.now().tcp[:3], dtype=float)
        zs.append(float(p[2]))
        if len(zs) >= 2 and abs(zs[-1] - zs[-2]) < 0.001 and p[2] < CAP_TOP_Z:
            stall += 1
        if stall >= 3 or (p[2] <= PRESS_Z - 0.004 and n3 >= 2):
            break
        exy = np.clip((np.array([x, y]) - p[:2]) * 10.0, -0.3, 0.3)
        r, rerr = _rot_cmd(robot.now().tcp, GOAL_RPY)
        rot_errs.append(rerr)
        robot.step(float(exy[0]), float(exy[1]), PUSH_DZ, float(r[0]), float(r[1]), float(r[2]), -1.0)
        n3 += 1
    pb = [float(v) for v in robot.now().tcp]
    out["push_steps"] = n3
    out["bottom"] = [round(v, 4) for v in pb[:3]]
    out["min_z"] = round(min(zs + [pb[2]]), 4)
    out["stall"] = stall
    out["push_rot_err_max"] = round(max(rot_errs + [0.0]), 4)
    # lift back to hover
    n4, e4 = _move_to(robot, [x, y, HOVER_Z], 0.01, 40, goal_rpy=GOAL_RPY)
    out["lift_steps"] = n4
    out["lift_z"] = round(float(robot.now().tcp[2]), 4)
    out["rpy"] = [round(float(v), 3) for v in robot.now().tcp[3:]]
    out["steps"] = int(robot.now().step) - s0
    return out
