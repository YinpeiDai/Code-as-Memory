"""BlinkCountButtonPressHard policy: count blue-lamp blinks, wait for the red lamp
to turn green, press the red (pink) button that many times, then press the black
button.  Perception uses RGB colour masks back-projected with depth into world
coordinates; control is closed-loop P control on measured TCP position."""

# ---------------- constants (from demo measurements, see task_analysis.md) -------------
CAP_TOP_Z = 0.059          # measured button cap top height (demos, both cameras)
STALL_Z = 0.058            # TCP below this = pressing the cap (demo stall z = 0.053)
HOVER_Z = 0.20             # lift between presses; R006: 0.15 fails, 0.175 works, expert 0.167-0.178 -> margin
APPROACH_Z = 0.14          # first approach height (demo ~0.142)
RELEASE_Z = 0.09           # TCP above this after a dip = press released (perception hysteresis)
BLACK_OFFSET = (-0.001, 0.157)   # fallback black-pink offset (demo mean, std 3/6 mm)
BLUE_MIN_PIX = 20
GREEN_MIN_PIX = 20
DZ_PRESS_FAST = -0.7       # descent command above the cap (expert ~-0.75)
DZ_PRESS_NEAR = -0.5       # descent command within 12 mm of the cap top
DZ_HOLD = -0.3             # push command while stalled on the cap
LATERAL_NEAR = 0.1         # lateral correction cap near/at contact (reduce shear)
HOLD_STEPS = 8             # stalled steps held before lifting (demo: 8-11 frames below 0.058)
ROT_GAIN = 0.7             # orientation hold gain (fraction of error per command)


def _K(frame, cam):
    k = getattr(frame, cam + "_K", None)
    if k is None:
        return np.array([[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1.0]])
    return np.asarray(k, dtype=float)


def _world_img(frame, cam):
    dp = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    ext = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    K = _K(frame, cam)
    vs, us = np.mgrid[0:dp.shape[0], 0:dp.shape[1]]
    x = (us - K[0, 2]) / K[0, 0] * dp
    y = (vs - K[1, 2]) / K[1, 1] * dp
    pc = np.stack([x, y, dp], -1).reshape(-1, 3)
    R = ext[:, :3]
    t = ext[:, 3]
    W = ((pc - t) @ R).reshape(dp.shape[0], dp.shape[1], 3)
    W[dp == 0] = np.nan
    return W


def _buttons_cam(frame, cam):
    rgb = np.asarray(getattr(frame, cam + "_rgb")).astype(int)
    W = _world_img(frame, cam)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    wz = np.nan_to_num(W[..., 2], nan=-1.0)
    pink = (r > 150) & (r - g > 40) & (np.abs(g - b) < 15) & (g > 90) & (wz > 0.04) & (wz < 0.08)
    out = {"cam": cam, "pink_px": int(pink.sum()), "red": None, "black": None, "black_px": 0}
    if pink.sum() < 15:
        return out
    P = W[pink]
    pc = np.nanmedian(P, 0)
    out["red"] = [float(pc[0]), float(pc[1]), float(pc[2])]
    neutral = (np.abs(r - g) < 8) & (np.abs(g - b) < 8) & (r < 95)
    dy = np.nan_to_num(W[..., 1] - pc[1], nan=-9.0)
    dx = np.nan_to_num(W[..., 0] - pc[0], nan=-9.0)
    blk = neutral & (wz > 0.045) & (wz < 0.075) & (dy > 0.10) & (dy < 0.21) & (np.abs(dx) < 0.05)
    out["black_px"] = int(blk.sum())
    if blk.sum() >= 15:
        B = W[blk]
        bc = np.nanmedian(B, 0)
        out["black"] = [float(bc[0]), float(bc[1]), float(bc[2])]
    return out


def _locate_buttons(frame):
    """Average front and wrist estimates of the red(pink) and black button cap centres."""
    ests = []
    for cam in ("front", "wrist"):
        if hasattr(frame, cam + "_rgb") and hasattr(frame, cam + "_depth"):
            ests.append(_buttons_cam(frame, cam))
    reds = [e["red"] for e in ests if e["red"] is not None]
    blacks = [e["black"] for e in ests if e["black"] is not None]
    red = None
    black = None
    if reds:
        red = [float(np.mean([p[0] for p in reds])), float(np.mean([p[1] for p in reds]))]
    if blacks:
        black = [float(np.mean([p[0] for p in blacks])), float(np.mean([p[1] for p in blacks]))]
    elif red is not None:
        black = [red[0] + BLACK_OFFSET[0], red[1] + BLACK_OFFSET[1]]
    info = {e["cam"]: {"pink_px": e["pink_px"], "black_px": e["black_px"], "red": e["red"], "black": e["black"]} for e in ests}
    return red, black, info


def _lamp_state(frame):
    """Blue lamp lit / red lamp turned green, from the fixed front camera."""
    rgb = np.asarray(frame.front_rgb).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = int(((b > 150) & (r < 80) & (g < 80)).sum())
    green = int(((g > 150) & (r < 100) & (b < 100)).sum())
    return blue >= BLUE_MIN_PIX, green >= GREEN_MIN_PIX, blue, green


def _tcp(frame):
    return np.asarray(frame.tcp, dtype=float)


def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def _move_cmd(tcp, target, k=1.0, vmax=0.8, zmin=-0.8, zmax=0.8):
    e = np.asarray(target, dtype=float) - tcp[:3]
    a = k * e / 0.1
    return [_clip(a[0], -vmax, vmax), _clip(a[1], -vmax, vmax), _clip(a[2], zmin, zmax)]


def _rpy_mat(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(max(c, -1.0), 1.0)
    ang = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * w
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return w * (ang / (2.0 * s))


def _rot_cmd(tcp, yaw_des, k=ROT_GAIN):
    """Rotation command that drives the gripper back to top-down (roll=pi, pitch=0, yaw=yaw_des).
    Native CPU IK: target = Rx(-0.1 r0) Ry(-0.1 r1) Rz(-0.1 r2) R_meas, so r = -k * rotvec(R_des R_meas^T) / 0.1."""
    Rd = _rpy_mat([math.pi, 0.0, yaw_des])
    e = _rotvec(Rd @ _rpy_mat(tcp[3:6]).T)
    r = -k * e / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return [float(r[0]), float(r[1]), float(r[2])]


def _step(robot, cmd, gripper=-1.0, yaw_des=None):
    if yaw_des is None:
        robot.step(cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, gripper)
    else:
        rc = _rot_cmd(_tcp(robot.now()), yaw_des)
        robot.step(cmd[0], cmd[1], cmd[2], rc[0], rc[1], rc[2], gripper)
    return robot.now()


def _go_to(robot, target, yaw, max_steps=30, tol=0.006, k=1.0):
    """Closed-loop move to an xyz target holding top-down orientation; ends when within tol and nearly stopped."""
    n = 0
    for n in range(1, max_steps + 1):
        cur = _tcp(robot.now())
        _step(robot, _move_cmd(cur, target, k=k), yaw_des=yaw)
        new = _tcp(robot.now())
        err = float(np.linalg.norm(np.asarray(target) - new[:3]))
        spd = float(np.linalg.norm(new[:3] - cur[:3]))
        if err < tol and spd < 0.004:
            break
    fin = _tcp(robot.now())
    return {"steps": n, "err_m": float(np.linalg.norm(np.asarray(target) - fin[:3])), "tcp": [round(float(v), 4) for v in fin]}


def _press_down(robot, x, y, yaw, hold_steps=2, max_steps=20):
    """Descend on (x, y) (fast above the cap, softer near it) until the TCP stalls below STALL_Z, then hold."""
    n = 0
    zmin = 9.0
    stalled = False
    held = 0
    contact_xy = None
    for n in range(1, max_steps + 1):
        cur = _tcp(robot.now())
        near = cur[2] < CAP_TOP_Z + 0.012
        cmd = _move_cmd(cur, [x, y, cur[2]], k=(1.0 if near else 2.5), vmax=(LATERAL_NEAR if near else 0.3))
        cmd[2] = DZ_PRESS_NEAR if near else DZ_PRESS_FAST
        if stalled:
            cmd[2] = DZ_HOLD
        _step(robot, cmd, yaw_des=yaw)
        new = _tcp(robot.now())
        zmin = min(zmin, float(new[2]))
        if contact_xy is None and new[2] < CAP_TOP_Z + 0.002:
            contact_xy = [round(float(new[0] - x), 4), round(float(new[1] - y), 4)]
        if new[2] < STALL_Z and abs(new[2] - cur[2]) < 0.002:
            stalled = True
        if stalled:
            held += 1
            if held >= hold_steps:
                break
    fin = _tcp(robot.now())
    return {"steps": n, "stalled": stalled, "zmin": round(zmin, 4), "contact_xy_err": contact_xy,
            "end_xy_err": [round(float(fin[0] - x), 4), round(float(fin[1] - y), 4)],
            "rpy": [round(float(v), 3) for v in fin[3:6]]}


def _lift(robot, x, y, z, yaw, max_steps=20):
    n = 0
    for n in range(1, max_steps + 1):
        cur = _tcp(robot.now())
        cmd = _move_cmd(cur, [x, y, z], k=1.0, vmax=0.3)
        if cur[2] < CAP_TOP_Z + 0.01:
            cmd[0] = 0.0
            cmd[1] = 0.0
        _step(robot, cmd, yaw_des=yaw)
        new = _tcp(robot.now())
        if new[2] > z - 0.012:
            break
    fin = _tcp(robot.now())
    return {"steps": n, "z": round(float(fin[2]), 4), "rpy": [round(float(v), 3) for v in fin[3:6]]}


def _scan_frames(frames, mem):
    """Update blink count / green flag / press dips from every observed frame."""
    for f in frames:
        blue_on, green_on, nb, ng = _lamp_state(f)
        if not mem["green_seen"]:
            if blue_on and not mem["blue_on"]:
                mem["blink_count"] = mem["blink_count"] + 1
                mem["blink_steps"] = mem["blink_steps"] + [int(f.step)]
            mem["blue_on"] = bool(blue_on)
            if green_on:
                mem["green_seen"] = True
                mem["green_step"] = int(f.step)
        z = float(np.asarray(f.tcp)[2])
        if mem["tcp_low"]:
            if z > RELEASE_Z:
                mem["tcp_low"] = False
        elif z < STALL_Z:
            mem["tcp_low"] = True
            if mem["stage"] == "press_red":
                mem["red_dips"] = mem["red_dips"] + 1
            elif mem["stage"] == "press_black":
                mem["black_dips"] = mem["black_dips"] + 1
        mem["last_step"] = int(f.step)
    return mem


MEM_DEFAULT = {
    "stage": "init", "blink_count": 0, "blink_steps": [], "blue_on": False,
    "green_seen": False, "green_step": -1, "red_xy": [], "black_xy": [],
    "red_dips": 0, "black_dips": 0, "tcp_low": False, "last_step": 0,
    "black_attempts": 0, "notes": "", "yaw0": 0.0,
}


def perceive_decide(obs, goal, memory, last):
    """Count blue blinks until the red lamp turns green, then press red N times, then black."""
    mem = dict(MEM_DEFAULT)
    for k in MEM_DEFAULT:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    cur = obs.current
    if mem["stage"] == "init":
        red, black, info = _locate_buttons(cur)
        if red is not None:
            mem["red_xy"] = red
            mem["black_xy"] = black
        mem["notes"] = "buttons " + repr(info)[:600]
        mem["yaw0"] = float(np.asarray(cur.tcp)[5])
        mem["stage"] = "wait_cue"
    mem = _scan_frames(obs.frames, mem)
    if mem["stage"] == "wait_cue":
        if not mem["green_seen"]:
            return {"memory": mem, "act": {"tool": "act_wait_cue", "args": {"max_steps": 150}}}
        if not mem["red_xy"]:
            red, black, info = _locate_buttons(cur)
            mem["red_xy"] = red if red is not None else []
            mem["black_xy"] = black if black is not None else []
            mem["notes"] = "relocate " + repr(info)[:600]
        if not mem["red_xy"]:
            # buttons not visible (e.g. occluded): back off upward and look again next decision
            return {"memory": mem, "act": {"tool": "act_retreat_up", "args": {"z": 0.30}}}
        mem["stage"] = "press_red" if mem["blink_count"] > 0 else "press_black"
    if mem["stage"] == "press_red":
        if mem["red_dips"] >= mem["blink_count"]:
            mem["stage"] = "press_black"
        else:
            first = mem["red_dips"] == 0
            return {"memory": mem, "act": {"tool": "act_press_button", "args": {
                "x": mem["red_xy"][0], "y": mem["red_xy"][1], "yaw": mem["yaw0"],
                "approach_z": APPROACH_Z if first else HOVER_Z, "lift_z": HOVER_Z, "final": False}}}
    if mem["stage"] == "press_black":
        if mem["black_attempts"] < 2:
            mem["black_attempts"] = mem["black_attempts"] + 1
            return {"memory": mem, "act": {"tool": "act_press_button", "args": {
                "x": mem["black_xy"][0], "y": mem["black_xy"][1], "yaw": mem["yaw0"],
                "approach_z": HOVER_Z, "lift_z": HOVER_Z, "final": True}}}
        mem["stage"] = "done"
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


def act_wait_cue(robot, max_steps=150):
    """Hold still (zero motion, gripper closed) until the red lamp turns green or max_steps; reports blinks seen."""
    blinks = 0
    was_on = False
    n = 0
    start = int(robot.now().step)
    blue_on, green_on, nb, ng = _lamp_state(robot.now())
    while not green_on and n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
        blue_on, green_on, nb, ng = _lamp_state(robot.now())
        if blue_on and not was_on:
            blinks += 1
        was_on = blue_on
    return {"steps": n, "start_step": start, "end_step": int(robot.now().step), "green": bool(green_on), "blinks_seen_in_act": blinks}


def act_press_button(robot, x, y, yaw=0.0, approach_z=0.15, lift_z=0.15, final=False):
    """Move above (x, y) holding a top-down gripper (yaw), descend until the TCP stalls on the button cap,
    hold, and lift to lift_z. final=True (submit button): hold longer at the bottom (episode should end)."""
    cur = _tcp(robot.now())
    report = {"start_tcp": [round(float(v), 4) for v in cur]}
    need_xy = float(np.hypot(cur[0] - x, cur[1] - y))
    tilt = float(np.linalg.norm(_rotvec(_rpy_mat([math.pi, 0.0, yaw]) @ _rpy_mat(cur[3:6]).T)))
    if need_xy > 0.004 or cur[2] < approach_z - 0.03 or tilt > 0.05:
        check = robot.dry_run(x, y, approach_z, math.pi, 0.0, yaw, mode="absolute")
        report["dry_run_ok"] = bool(check.get("within_joint_limits", False))
        report["approach"] = _go_to(robot, [x, y, approach_z], yaw, max_steps=25, tol=0.004)
    report["down"] = _press_down(robot, x, y, yaw, hold_steps=8 if final else HOLD_STEPS, max_steps=25 if final else 20)
    report["lift"] = _lift(robot, x, y, lift_z, yaw)
    report["end_step"] = int(robot.now().step)
    return report


def act_retreat_up(robot, z=0.30):
    """Move straight up to height z (top-down gripper) so the cameras can re-observe the buttons."""
    cur = _tcp(robot.now())
    return _go_to(robot, [float(cur[0]), float(cur[1]), float(z)], float(cur[5]), max_steps=15, tol=0.01)


def act_hold(robot, steps=20):
    """Hold position with zero motion for a number of steps."""
    for i in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"held": int(steps), "end_step": int(robot.now().step)}
