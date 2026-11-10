"""FindImposterColor5: remember the cue colours, then touch the cube whose colour is new.

Observed structure (50 demos): frames 0..k (k=0..4) show 4 cubes of distinct pure colours
(from R,G,B,Y,M); then 1..5 empty frames; then 5 cubes appear (new layout) and robot commands
start to take effect on that frame.  The imposter is the one colour absent from the cue.
"""

COLOR_CODES = {"R": (1, 0, 0), "G": (0, 1, 0), "B": (0, 0, 1), "Y": (1, 1, 0), "M": (1, 0, 1)}
MIN_PIXELS = 4
SAFE_Z = 0.065         # transit height of the TCP (cube tops at ~0.042 m, fingertips ~1 cm below TCP)
XY_FRAC = 0.7          # fraction of the xy error to cover per step (model-based)
TOUCH_GRIPPER = 1.0    # +1 open (expert style), -1 closed
FINAL_Z = 0.021        # final TCP height = cube centre height (top 0.041-0.042); fingertips ~1 cm above table
ROT_GAIN = 5.0


def _color_masks(img):
    im = img.astype(np.float32)
    mx = im.max(axis=2)
    r = im / np.maximum(mx, 1.0)[..., None]
    pure = (mx > 90) & np.all((r < 0.2) | (r > 0.85), axis=2)
    hi = r > 0.5
    out = {}
    for name, code in COLOR_CODES.items():
        m = pure & (hi[..., 0] == bool(code[0])) & (hi[..., 1] == bool(code[1])) & (hi[..., 2] == bool(code[2]))
        n = int(m.sum())
        if n >= MIN_PIXELS:
            out[name] = m
    return out


def _backproject(K, E, us, vs, d_mm):
    z = d_mm.astype(np.float64) / 1000.0
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], axis=1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t[None, :]) @ R


def _cube_top(frame, mask):
    """World xy of the visible top face and its height, from the front depth."""
    D = frame.front_depth
    vs, us = np.nonzero(mask & (D > 0))
    if len(vs) < 3:
        return None
    pts = _backproject(frame.front_K, frame.front_extrinsic, us.astype(np.float64), vs.astype(np.float64), D[vs, us])
    zmax = float(pts[:, 2].max())
    top = pts[pts[:, 2] > zmax - 0.006]
    c = top.mean(axis=0)
    return [float(c[0]), float(c[1]), zmax, int(len(top)), int(len(pts))]


def _rpy_to_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return w * ang / (2.0 * s)


def _clip(v, lo=-1.0, hi=1.0):
    return float(min(hi, max(lo, v)))


def perceive_decide(obs, goal, memory, last):
    """Track cue colours across frames; once the 5-cube test layout appears, locate the
    cube with the colour absent from the cue (front depth) and touch it."""
    cue = list(memory.get("cue_colors") or [])
    seen_blank = bool(memory.get("seen_blank") or False)
    phase = memory.get("phase") or "cue"
    target = memory.get("target") or []
    imposter = memory.get("imposter") or ""
    log = memory.get("log") or ""
    test_step = memory.get("test_step")
    if test_step is None:
        test_step = -1
    act = None
    if phase in ("cue", "blank"):
        for f in obs.frames:
            masks = _color_masks(f.front_rgb)
            names = sorted(masks.keys())
            is_test = len(names) == 5 or (seen_blank and len(names) > 0 and any(n not in cue for n in names) and len(cue) >= 3)
            if not is_test:
                if len(names) == 0:
                    if len(cue) > 0:
                        seen_blank = True
                        phase = "blank"
                else:
                    if not seen_blank:
                        # cue colours: front view, plus the wrist view as a second witness against occlusion
                        wnames = sorted(_color_masks(f.wrist_rgb).keys())
                        for n in names + wnames:
                            if n not in cue:
                                cue.append(n)
                continue
            # test layout visible
            cands = [n for n in names if n not in cue]
            if len(cands) == 0:
                cands = names
            imposter = cands[0]
            top = _cube_top(f, masks[imposter])
            if top is None:
                continue
            target = [top[0], top[1], top[2]]
            phase = "touch"
            test_step = int(f.step)
            log = log + "test@%d cue=%s seen=%s imp=%s top=%.3f,%.3f,%.3f n=%d/%d; " % (
                f.step, "".join(sorted(cue)), "".join(names), imposter, top[0], top[1], top[2], top[3], top[4])
            break
    if phase == "touch":
        act = {"tool": "act_touch", "args": {"x": target[0], "y": target[1], "top_z": target[2], "color": imposter,
                                            "gripper": TOUCH_GRIPPER, "final_z": FINAL_Z}}
    elif phase == "done":
        act = {"tool": "act_hold", "args": {}}
    else:
        act = {"tool": "act_wait", "args": {}}
    if act["tool"] == "act_touch":
        log = log + "touch_called; "
    mem = {"cue_colors": sorted(cue), "seen_blank": seen_blank, "phase": phase, "target": target,
           "imposter": imposter, "test_step": int(test_step), "log": log[-1500:]}
    return {"memory": mem, "act": act}


def act_wait(robot):
    """One zero-motion step (cue / delay phase; commands are suppressed by the task anyway)."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TOUCH_GRIPPER)
    f = robot.now()
    return {"waited": 1, "step": int(f.step)}


def act_hold(robot):
    """Hold position for one step."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TOUCH_GRIPPER)
    return {"held": 1, "step": int(robot.now().step)}


def _plan_cmd(err, vel, frac, max_step=0.04):
    """One-axis command from the fitted step model d[t+1] ~ 0.25 d[t] + 0.030 a[t] (demo fit, rms 3 mm)."""
    want = max(-max_step, min(max_step, frac * err))
    return _clip((want - 0.25 * vel) / 0.030)


def _wrist_cube(frame, color):
    """World xy of the visible top face of `color` from the wrist camera (None if not visible)."""
    masks = _color_masks(frame.wrist_rgb)
    if color not in masks:
        return None
    D = frame.wrist_depth
    vs, us = np.nonzero(masks[color] & (D > 0))
    if len(vs) < 20:
        return None
    pts = _backproject(frame.wrist_K, frame.wrist_extrinsic, us.astype(np.float64), vs.astype(np.float64), D[vs, us])
    zmax = float(pts[:, 2].max())
    top = pts[pts[:, 2] > zmax - 0.004]
    if len(top) < 10:
        return None
    c = top.mean(axis=0)
    return [round(float(c[0]), 4), round(float(c[1]), 4), round(zmax, 4)]


def act_touch(robot, x, y, top_z, color="", gripper=1.0, final_z=0.025, push=0.03, squeeze_dz=0.014, max_steps=30):
    """Closed-loop touch of the cube at (x, y) (top face height top_z) with the OPEN gripper.
    transit: TCP z >= SAFE_Z until |ey| < 1 cm and |ex| < 1.5 cm (fingers straddle the cube along y);
    descend: to final_z (cube between the fingers);
    squeeze: once the TCP is below top_z + squeeze_dz (fingertips ~1 cm lower, beside the cube), close the
    gripper on the cube while finishing the descent;
    push: after 3 squeeze steps without the episode ending, reopen and shift the TCP by `push` along y towards
    y=0 so the inner face of one finger presses the cube side (produced native success in R004/R006/R007).
    Returns the measured trace (stage, tcp, command, wrist cube estimate)."""
    f0 = robot.now()
    t0 = np.asarray(f0.tcp, dtype=np.float64)
    R_hold = _rpy_to_R(t0[3], t0[4], t0[5])
    prev = t0[:3].copy()
    trace = []
    stage = "transit"
    pushed = 0
    squeezed = 0
    grip = float(gripper)
    sgn = -1.0 if y > 0.03 else 1.0
    tx, ty = float(x), float(y)
    for i in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=np.float64)
        vel = tcp[:3] - prev if i > 0 else np.zeros(3)
        prev = tcp[:3].copy()
        if stage == "transit" and abs(tx - tcp[0]) < 0.015 and abs(ty - tcp[1]) < 0.01:
            stage = "descend"
        if stage == "descend" and tcp[2] < top_z + float(squeeze_dz):
            stage = "squeeze"
            grip = -1.0
        elif stage == "squeeze":
            squeezed += 1
            if squeezed >= 3:
                stage = "push"
                grip = float(gripper)
                ty = float(y) + sgn * float(push)
        elif stage == "push":
            pushed += 1
        zt = max(SAFE_Z, final_z) if stage == "transit" else final_z
        ex, ey, ez = tx - tcp[0], ty - tcp[1], zt - tcp[2]
        R = _rpy_to_R(tcp[3], tcp[4], tcp[5])
        w = _rotvec(R_hold @ R.T)
        cmd = [_plan_cmd(ex, vel[0], XY_FRAC), _plan_cmd(ey, vel[1], XY_FRAC), _plan_cmd(ez, vel[2], 0.7),
               _clip(-ROT_GAIN * w[0]), _clip(-ROT_GAIN * w[1]), _clip(-ROT_GAIN * w[2])]
        wc = _wrist_cube(f, color) if color else None
        trace.append([int(f.step), stage] + [round(float(v), 3) for v in tcp[:3]] + [round(v, 2) for v in cmd[:3]] + [wc])
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
    f = robot.now()
    return {"touch_returned": True, "stage": stage,
            "final_tcp": [round(float(v), 3) for v in f.tcp], "fingers": [round(float(v), 3) for v in f.fingers],
            "trace": trace[-12:]}
