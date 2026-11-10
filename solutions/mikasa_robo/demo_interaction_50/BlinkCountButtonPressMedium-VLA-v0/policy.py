"""BlinkCountButtonPress policy: count blue-lamp blinks, wait for green, press red N times, press black."""

RED_PRESS_OFFSET = (0.010, 0.0015)   # commanded; live x lags ~+6 mm at the bottom, so measured press ~= expert demo offset (+0.016, +0.0025)
BLACK_PRESS_OFFSET = (0.001, -0.0005)  # demo-measured: expert final black xy minus perceived black cap-top centre
BLACK_FROM_RED = (0.0012, 0.1595)    # demo-measured rigid layout offset (fallback if black not segmented)
HOVER_Z = 0.165
PRESS_BOTTOM_Z = 0.057               # demo: pressed TCP z bottoms out at ~0.053
GAIN = 8.0                           # action units per metre of error (P control, offline-modelled)
ROT_GAIN = 0.5                       # fraction of orientation error commanded per step
ROT_DEADBAND = 0.01                  # rad


def _deproject(K, E, u, v, d_mm):
    d = d_mm / 1000.0
    p = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (p - t) @ R


def _lamps(frame):
    im = np.asarray(frame.front_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    blue = int(((b > 120) & (r < 80) & (g < 80)).sum())
    green = int(((g > 150) & (r < 120) & (b < 120)).sum())
    red = int(((r > 150) & (g < 60) & (b < 60)).sum())
    return blue, green, red


def _find_buttons(frame):
    img = np.asarray(frame.front_rgb).astype(int)
    dep = np.asarray(frame.front_depth).astype(float)
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    v, u = np.mgrid[0:256, 0:256]
    P = _deproject(K, E, u.astype(float), v.astype(float), dep)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    valid = dep > 0
    pink = (r > 150) & (b > 90) & (np.abs(g - b) < 35) & ((r - g) > 40) & valid
    if pink.sum() < 15:
        return None, None, {"pink_px": int(pink.sum())}
    Pp = P[pink]
    top = Pp[Pp[:, 2] > Pp[:, 2].max() - 0.004]
    red = top.mean(0)
    gray = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & valid
    rel = P - red
    win = gray & (np.abs(rel[..., 0]) < 0.05) & (rel[..., 1] > 0.09) & (rel[..., 1] < 0.21) \
        & (rel[..., 2] > -0.004) & (rel[..., 2] < 0.01)
    info = {"pink_px": int(pink.sum()), "red_top_px": int(len(top)), "black_px": int(win.sum())}
    if win.sum() >= 15:
        black = P[win].mean(0)
        d = black - red
        info["black_minus_red"] = [round(float(d[0]), 4), round(float(d[1]), 4)]
        if abs(d[0] - BLACK_FROM_RED[0]) > 0.02 or abs(d[1] - BLACK_FROM_RED[1]) > 0.02:
            black = np.array([red[0] + BLACK_FROM_RED[0], red[1] + BLACK_FROM_RED[1], red[2]])
            info["black_source"] = "layout_fallback"
        else:
            info["black_source"] = "segmented"
    else:
        black = np.array([red[0] + BLACK_FROM_RED[0], red[1] + BLACK_FROM_RED[1], red[2]])
        info["black_source"] = "layout_fallback"
    return [float(red[0]), float(red[1]), float(red[2])], [float(black[0]), float(black[1]), float(black[2])], info


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_err(tcp):
    """World-frame rotation vector taking the measured orientation to gripper-straight-down (roll=pi, pitch=0, yaw=0)."""
    R = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_to_R(math.pi, 0.0, 0.0)
    Re = Rd @ R.T
    c = max(-1.0, min(1.0, (np.trace(Re) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2.0 * math.sin(ang))
    return w * ang


def _rcmd(tcp, k=ROT_GAIN):
    """Rotation args: native offset is -0.1*r rad (world frame), so r = -k*err/0.1, norm-limited to 1."""
    w = _rot_err(tcp)
    if float(np.linalg.norm(w)) < ROT_DEADBAND:
        return [0.0, 0.0, 0.0]
    r = -k * w / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return [float(r[0]), float(r[1]), float(r[2])]


def _cmd(tcp, target, gain=GAIN, zmax=1.0):
    a = []
    for k in range(3):
        a.append(float(np.clip(gain * (target[k] - tcp[k]), -1.0, 1.0)))
    a[2] = float(np.clip(a[2], -zmax, zmax))
    return a


def perceive_decide(obs, goal, memory, last):
    """Track cue (blink count, green go signal) over all frames, perceive buttons, and choose the next stage."""
    m = dict(memory)
    for k in ("stage", "red_xyz", "black_xyz", "blinks", "blue_on", "go_seen", "presses_done",
              "black_presses", "notes", "frames_seen"):
        if k not in m:
            m[k] = None
    if m["stage"] is None:
        m["stage"] = "wait_cue"
        m["blinks"] = 0
        m["blue_on"] = False
        m["go_seen"] = False
        m["presses_done"] = 0
        m["black_presses"] = 0
        m["frames_seen"] = 0
        m["notes"] = ""
    # cue tracking over every frame since the previous action
    for f in obs.frames:
        blue, green, red = _lamps(f)
        on = blue > 60
        if on and not m["blue_on"]:
            m["blinks"] = int(m["blinks"]) + 1
        m["blue_on"] = bool(on)
        if green > 60 and not on:
            m["go_seen"] = True
        m["frames_seen"] = int(m["frames_seen"]) + 1
    if m["red_xyz"] is None:
        red, black, info = _find_buttons(obs.current)
        if red is not None:
            m["red_xyz"] = red
            m["black_xyz"] = black
            m["notes"] = "buttons " + str(info)
    if m["red_xyz"] is None:
        return {"memory": m, "act": {"tool": "act_wait", "args": {"max_steps": 1, "stop_on_go": False}}}
    rx = m["red_xyz"][0] + RED_PRESS_OFFSET[0]
    ry = m["red_xyz"][1] + RED_PRESS_OFFSET[1]
    bx = m["black_xyz"][0] + BLACK_PRESS_OFFSET[0]
    by = m["black_xyz"][1] + BLACK_PRESS_OFFSET[1]
    n = int(m["blinks"])
    if not m["go_seen"]:
        m["stage"] = "wait_cue"
        return {"memory": m, "act": {"tool": "act_wait", "args": {"max_steps": 120, "stop_on_go": True}}}
    if int(m["presses_done"]) < n:
        m["stage"] = "press_red_" + str(int(m["presses_done"]) + 1) + "_of_" + str(n)
        m["presses_done"] = int(m["presses_done"]) + 1   # counts commanded presses (not native acceptance)
        return {"memory": m, "act": {"tool": "act_press", "args": {"x": rx, "y": ry, "dwell": 3,
                                                                      "lift_z": HOVER_Z, "label": m["stage"]}}}
    if int(m["black_presses"]) < 2:
        m["stage"] = "press_black_" + str(int(m["black_presses"]) + 1)
        m["black_presses"] = int(m["black_presses"]) + 1
        return {"memory": m, "act": {"tool": "act_press", "args": {"x": bx, "y": by, "dwell": 3,
                                                                      "lift_z": HOVER_Z, "label": m["stage"]}}}
    m["stage"] = "done_hold"
    return {"memory": m, "act": {"tool": "act_wait", "args": {"max_steps": 10, "stop_on_go": False}}}


def act_wait(robot, max_steps=120, stop_on_go=True):
    """Hold still (gripper closed) up to max_steps; stop early once the green go lamp is seen. Reports lamp sequence."""
    seq = []
    blinks = 0
    prev_on = False
    go_at = None
    for i in range(int(max_steps)):
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        blue, green, red = _lamps(f)
        on = blue > 60
        if on and not prev_on:
            blinks += 1
        prev_on = on
        seq.append("B" if on else ("G" if green > 60 else ("R" if red > 60 else ".")))
        if stop_on_go and green > 60 and not on:
            go_at = int(f.step)
            break
    return {"steps": len(seq), "lamp_seq": "".join(seq), "blinks_in_call": blinks, "go_step": go_at,
            "tcp": [round(float(v), 4) for v in robot.now().tcp]}


def act_press(robot, x, y, dwell=3, lift_z=0.165, push=-0.2, label=""):
    """Move above (x, y), descend until the TCP bottoms out on the button, hold `dwell` steps, lift to lift_z."""
    out = {"label": label}
    tcp = robot.now().tcp
    # 1) approach: reach hover height first if far in xy
    n = 0
    while n < 40:
        tcp = robot.now().tcp
        dxy = math.hypot(x - tcp[0], y - tcp[1])
        if dxy < 0.006 and abs(tcp[2] - lift_z) < 0.03:
            break
        if dxy > 0.03 and tcp[2] < lift_z - 0.04:
            tgt = [tcp[0], tcp[1], lift_z]
        else:
            tgt = [x, y, lift_z]
        a = _cmd(tcp, tgt)
        rr = _rcmd(tcp)
        robot.step(a[0], a[1], a[2], rr[0], rr[1], rr[2], -1.0)
        n += 1
    out["approach_steps"] = n
    out["hover_err_xy"] = round(math.hypot(x - tcp[0], y - tcp[1]), 4)
    # 2) descend until bottomed (z <= PRESS_BOTTOM_Z or stalled low)
    n = 0
    zs = []
    while n < 30:
        tcp = robot.now().tcp
        zs.append(float(tcp[2]))
        if tcp[2] <= PRESS_BOTTOM_Z:
            break
        if len(zs) >= 3 and tcp[2] < 0.07 and abs(zs[-1] - zs[-3]) < 0.0015:
            break
        a = _cmd(tcp, [x, y, 0.03])
        a[2] = float(np.clip(a[2], -0.8, -0.2))
        rr = _rcmd(tcp)
        robot.step(a[0], a[1], a[2], rr[0], rr[1], rr[2], -1.0)
        n += 1
    out["descend_steps"] = n
    # 3) dwell pushing down
    zmin = float(robot.now().tcp[2])
    for i in range(int(dwell)):
        tcp = robot.now().tcp
        a = _cmd(tcp, [x, y, 0.03])
        a[2] = float(push)
        rr = _rcmd(tcp)
        robot.step(a[0], a[1], a[2], rr[0], rr[1], rr[2], -1.0)
        zmin = min(zmin, float(robot.now().tcp[2]))
    tcp = robot.now().tcp
    out["press_xy_err"] = round(math.hypot(x - tcp[0], y - tcp[1]), 4)
    out["zmin"] = round(zmin, 4)
    # 4) lift
    n = 0
    while n < 30:
        tcp = robot.now().tcp
        if tcp[2] >= lift_z - 0.01:
            break
        a = _cmd(tcp, [x, y, lift_z + 0.01])
        rr = _rcmd(tcp)
        robot.step(a[0], a[1], a[2], rr[0], rr[1], rr[2], -1.0)
        n += 1
    out["lift_steps"] = n
    f = robot.now()
    out["rot_err_rad"] = round(float(np.linalg.norm(_rot_err(f.tcp))), 4)
    out["step"] = int(f.step)
    out["tcp"] = [round(float(v), 4) for v in f.tcp]
    return out
