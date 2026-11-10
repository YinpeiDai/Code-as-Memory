"""BlinkCountButtonPress policy: count blue-lamp blinks, wait for the red lamp to turn
green, press the red (pink) button that many times, then press the black button.

Perception uses the fixed front camera (RGB + depth + calibration). Button positions are
measured once from the start frame (arm away from the buttons). Progress (blinks counted,
red presses physically completed) is measured from the observed frames in perceive_decide.
"""

BLUE_MIN_PX = 60        # front-camera blue-lamp pixels when lit (demos: 0 when off, >=178 when on)
GREEN_MIN_PX = 60       # front-camera green-lamp pixels once the go signal shows (demos ~252)
BLACK_OFFSET = (-0.002, 0.160)   # black button minus red button (xy), measured in all 50 demos
BUTTON_TOP_Z = 0.0587   # measured button cap top height (both buttons)
Z_HOVER = 0.20          # expert lifts to ~0.20 between presses
Z_PUSH = 0.03           # descent target (below the ~0.053 mechanical bottom)
DOWN_Z = 0.062          # tracker: TCP below this over the button = pressed down
UP_Z = 0.12             # tracker: TCP above this after a down = press completed (released)
KP = 0.8                # proportional gain on (error / 0.1 m)
KR = 0.4                # orientation correction gain
MAX_DECISIONS = 40


def _masks(img):
    r = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    b = img[..., 2].astype(int)
    blue = (b > 150) & (r < 80) & (g < 80)
    green = (g > 150) & (r < 100) & (b < 100)
    return blue, green


def _lamps(frame):
    blue, green = _masks(frame.front_rgb)
    return int(blue.sum()), int(green.sum())


def _backproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    d = depth[v, u].astype(float) / 1000.0
    ok = d > 0
    u = u[ok]
    v = v[ok]
    d = d[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    R = np.asarray(E)[:, :3].astype(float)
    t = np.asarray(E)[:, 3].astype(float)
    return (pc - t) @ R


def _detect_buttons(frame):
    """Front camera first; wrist camera as fallback (agrees within 3 mm in all 50 demos)."""
    out = _detect_buttons_cam(frame, "front")
    if out["red"] is None:
        alt = _detect_buttons_cam(frame, "wrist")
        alt["fallback"] = "wrist (front n_red=%d)" % out["n_red"]
        return alt
    return out


def _detect_buttons_cam(frame, cam):
    """Red (pink cap) and black button cap centres in world coordinates from one camera."""
    img = getattr(frame, cam + "_rgb")
    r = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    b = img[..., 2].astype(int)
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    D = getattr(frame, cam + "_depth")
    pink = (r > 100) & (r > 1.3 * g) & (np.abs(b - g) < 10) & (g > 50)
    P = _backproject(D, K, E, pink)
    P = P[(P[:, 2] > 0.052) & (P[:, 2] < 0.075)] if len(P) else P
    out = {"n_red": int(len(P)), "red": None, "black": None, "n_black": 0, "black_source": None,
           "cam": cam}
    if len(P) < 15:
        return out
    # keep the largest cluster around the median (robust to stray pixels)
    med = np.median(P[:, :2], 0)
    P = P[np.linalg.norm(P[:, :2] - med, axis=1) < 0.03]
    red = P.mean(0)
    out["red"] = [float(red[0]), float(red[1]), float(red[2])]
    prior = np.array([red[0] + BLACK_OFFSET[0], red[1] + BLACK_OFFSET[1]])
    gray = (np.abs(r - g) < 5) & (np.abs(g - b) < 5) & (r >= 58) & (r <= 88)
    Q = _backproject(D, K, E, gray)
    if len(Q):
        Q = Q[(Q[:, 2] > 0.052) & (Q[:, 2] < 0.075)]
        Q = Q[np.linalg.norm(Q[:, :2] - prior, axis=1) < 0.03]
    out["n_black"] = int(len(Q))
    if len(Q) >= 15 and np.linalg.norm(Q[:, :2].mean(0) - prior) < 0.015:
        bc = Q.mean(0)
        out["black"] = [float(bc[0]), float(bc[1]), float(bc[2])]
        out["black_source"] = "seen"
    else:
        out["black"] = [float(prior[0]), float(prior[1]), float(red[2])]
        out["black_source"] = "offset_prior"
    return out


def _rpy_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return w / (2 * s) * ang


def _command(tcp, target, yaw0, kp=KP, vmax=1.0):
    """Six arm arguments that move toward target xyz and orientation (roll=pi, pitch=0, yaw0)."""
    err = np.array(target, dtype=float) - np.array(tcp[:3], dtype=float)
    d = kp * err / 0.1
    n = np.abs(d).max()
    if n > vmax:
        d = d * (vmax / n)
    Rc = _rpy_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_R(math.pi, 0.0, yaw0)
    w = _rotvec(Rt @ Rc.T)               # base-frame rotation still needed
    rcmd = -KR * w / 0.1                  # applied offset is -0.1*r
    rcmd = np.clip(rcmd, -0.3, 0.3)
    return [float(d[0]), float(d[1]), float(d[2]), float(rcmd[0]), float(rcmd[1]), float(rcmd[2])]


def _yaw0(tcp):
    y = float(tcp[5])
    return max(-0.3, min(0.3, y))


def _move_to(robot, target, yaw0, tol_xy=0.005, tol_z=0.008, max_steps=40, grip=-1.0):
    n = 0
    tcp = robot.now().tcp
    for n in range(1, max_steps + 1):
        cmd = _command(tcp, target, yaw0)
        robot.step(*cmd, grip)
        tcp = robot.now().tcp
        exy = math.hypot(target[0] - tcp[0], target[1] - tcp[1])
        if exy < tol_xy and abs(target[2] - tcp[2]) < tol_z:
            break
    return n, [float(v) for v in tcp[:3]]


def _press_once(robot, x, y, yaw0, z_hover, hold=3, max_down=40, max_up=40, grip=-1.0):
    """Descend on (x, y) until the TCP stalls at the button bottom, hold, then lift."""
    tcp = robot.now().tcp
    rec = {"start_z": float(tcp[2])}
    # align xy above the button first if needed
    if math.hypot(x - tcp[0], y - tcp[1]) > 0.008:
        n, p = _move_to(robot, [x, y, max(float(tcp[2]), z_hover - 0.03)], yaw0, tol_xy=0.005, tol_z=0.05)
        rec["align_steps"] = n
    zmin = 10.0
    prev = float(robot.now().tcp[2])
    bottom = False
    steps = 0
    for steps in range(1, max_down + 1):
        tcp = robot.now().tcp
        robot.step(*_command(tcp, [x, y, Z_PUSH], yaw0), grip)
        z = float(robot.now().tcp[2])
        zmin = min(zmin, z)
        if z < DOWN_Z and (prev - z) < 0.0015:
            bottom = True
            break
        prev = z
    rec["down_steps"] = steps
    rec["bottom"] = bottom
    for _ in range(hold):
        tcp = robot.now().tcp
        cmd = _command(tcp, [x, y, Z_PUSH], yaw0)
        cmd[2] = -0.5
        robot.step(*cmd, grip)
        zmin = min(zmin, float(robot.now().tcp[2]))
    tb = robot.now().tcp
    rec["zmin"] = zmin
    rec["xy_err_bottom"] = float(math.hypot(x - tb[0], y - tb[1]))
    n, p = _move_to(robot, [x, y, z_hover], yaw0, tol_xy=0.01, tol_z=0.012, max_steps=max_up)
    rec["up_steps"] = n
    rec["end_z"] = p[2]
    return rec


def act_watch(robot, max_steps=900):
    """Hold still (gripper open) and watch the lamps each step until the red lamp turns green
    or max_steps pass. Returns the blink count seen in this call and the go-signal step."""
    rises = []
    prev_on = False
    f = robot.now()
    blue, green = _lamps(f)
    prev_on = blue >= BLUE_MIN_PX
    n = 0
    green_step = None
    if green >= GREEN_MIN_PX:
        return {"steps": 0, "green_step": int(f.step), "rises": rises}
    for n in range(1, max_steps + 1):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        f = robot.now()
        blue, green = _lamps(f)
        on = blue >= BLUE_MIN_PX
        if on and not prev_on:
            rises.append(int(f.step))
        prev_on = on
        if green >= GREEN_MIN_PX:
            green_step = int(f.step)
            break
    return {"steps": n, "green_step": green_step, "rises_in_call": rises, "count_in_call": len(rises)}


def act_press_red(robot, x, y, n=1, z_hover=Z_HOVER, yaw=None):
    """Press the red button at world (x, y) n times with a closed gripper: descend until the
    TCP stalls at the button bottom, hold briefly, lift to z_hover. yaw: fixed hand-yaw
    reference (rad) for the whole episode. Returns measured depths."""
    yaw0 = _yaw0(robot.now().tcp) if yaw is None else max(-0.3, min(0.3, float(yaw)))
    recs = []
    for _ in range(int(n)):
        recs.append(_press_once(robot, float(x), float(y), yaw0, float(z_hover)))
    return {"presses_commanded": len(recs), "records": recs, "step": int(robot.now().step)}


def act_press_black(robot, x, y, z_hover=Z_HOVER, yaw=None):
    """Move above the black button at (x, y) and press it once (the episode should end with
    success during the descent). yaw: fixed hand-yaw reference. Returns measurements if the
    episode is still running."""
    yaw0 = _yaw0(robot.now().tcp) if yaw is None else max(-0.3, min(0.3, float(yaw)))
    tcp = robot.now().tcp
    n1, p = _move_to(robot, [float(x), float(y), max(float(tcp[2]), z_hover - 0.05)], yaw0,
                     tol_xy=0.005, tol_z=0.06)
    rec = _press_once(robot, float(x), float(y), yaw0, float(z_hover))
    rec["transfer_steps"] = n1
    rec["note"] = "black press finished without episode end"
    rec["step"] = int(robot.now().step)
    return rec


def act_hold(robot, steps=20):
    """Hold the current pose for a number of steps (gripper closed)."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"held": int(steps), "step": int(robot.now().step)}


def _init_memory(memory):
    m = dict(memory)
    defaults = {
        "stage": "watch", "blink_count": 0, "blue_on": False, "last_step": -1,
        "blink_steps": [], "green_step": None, "red_xyz": None, "black_xyz": None,
        "button_info": "", "presses_done": 0, "press_down": False, "press_zmins": [],
        "black_attempts": 0, "decisions": 0, "notes": "", "yaw_ref": None,
    }
    for k, v in defaults.items():
        if m.get(k) is None:
            m[k] = v
    return m


def _set_yaw_ref(m, frame):
    if m.get("yaw_ref") is None:
        m["yaw_ref"] = round(_yaw0(frame.tcp), 4)


def perceive_decide(obs, goal, memory, last):
    """Track lamps and press progress from every observed frame, then choose the next act."""
    m = _init_memory(memory)
    _set_yaw_ref(m, obs.frames[0])
    m["decisions"] = int(m["decisions"]) + 1
    frames = list(obs.frames)
    red = m["red_xyz"]
    for f in frames:
        if int(f.step) <= int(m["last_step"]):
            continue
        m["last_step"] = int(f.step)
        if m["stage"] == "watch":
            blue, green = _lamps(f)
            on = blue >= BLUE_MIN_PX
            if on and not m["blue_on"]:
                m["blink_count"] = int(m["blink_count"]) + 1
                m["blink_steps"] = list(m["blink_steps"]) + [int(f.step)]
            m["blue_on"] = bool(on)
            if green >= GREEN_MIN_PX:
                m["stage"] = "press"
                m["green_step"] = int(f.step)
        elif red is not None:
            tcp = f.tcp
            near = math.hypot(float(tcp[0]) - red[0], float(tcp[1]) - red[1]) < 0.03
            z = float(tcp[2])
            if near and z < DOWN_Z and not m["press_down"]:
                m["press_down"] = True
                m["press_zmins"] = list(m["press_zmins"]) + [round(z, 4)]
            elif m["press_down"] and near:
                zs = list(m["press_zmins"])
                zs[-1] = round(min(zs[-1], z), 4)
                m["press_zmins"] = zs
            if m["press_down"] and z > UP_Z:
                m["press_down"] = False
                m["presses_done"] = int(m["presses_done"]) + 1
    # locate buttons while the arm is still at its start pose
    if m["red_xyz"] is None:
        info = _detect_buttons(obs.current)
        if info["red"] is not None:
            m["red_xyz"] = [round(v, 4) for v in info["red"]]
            m["black_xyz"] = [round(v, 4) for v in info["black"]]
        m["button_info"] = "cam=%s n_red=%d n_black=%d black=%s %s" % (
            info["cam"], info["n_red"], info["n_black"], info["black_source"], info.get("fallback", ""))
    left = MAX_DECISIONS - int(m["decisions"])
    if m["stage"] == "watch":
        act = {"tool": "act_watch", "args": {"max_steps": 900}}
    elif m["red_xyz"] is None:
        m["notes"] = "red button not found"
        act = {"tool": "act_hold", "args": {"steps": 5}}
    elif m["stage"] == "press" and int(m["presses_done"]) < int(m["blink_count"]):
        remaining = int(m["blink_count"]) - int(m["presses_done"])
        slots = max(1, left - 4)
        n = max(1, -(-remaining // slots))
        act = {"tool": "act_press_red", "args": {"x": m["red_xyz"][0], "y": m["red_xyz"][1], "n": n,
                                                  "yaw": m["yaw_ref"]}}
    else:
        m["stage"] = "black"
        if int(m["black_attempts"]) < 3:
            m["black_attempts"] = int(m["black_attempts"]) + 1
            act = {"tool": "act_press_black", "args": {"x": m["black_xyz"][0], "y": m["black_xyz"][1],
                                                        "yaw": m["yaw_ref"]}}
        else:
            act = {"tool": "act_hold", "args": {"steps": 50}}
    return {"memory": m, "act": act}
