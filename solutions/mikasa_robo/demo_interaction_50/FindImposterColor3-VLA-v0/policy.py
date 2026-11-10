"""FindImposterColor3: remember the cue cube colours, wait for the test layout,
then touch (straddle + close the gripper on) the cube whose colour is new."""

COLOR_NAMES = ("red", "green", "blue")
MIN_PIX = 8            # pixels needed to call a colour present in the front image
Z_HOVER = 0.075        # m, TCP height to finish lateral alignment before final descent
Z_GRASP = 0.020        # m, TCP height (cube centre for a 4 cm cube) where fingers close
XY_TOL = 0.012         # m, lateral error allowed before descending below Z_HOVER
POS_GAIN = 22.0        # action units per metre of error (one unit ~ 0.035-0.04 m/step measured)
VEL_DAMP = 0.35        # steps of current velocity subtracted from the error (R002: carry-over ~0.37 v)
ROT_GAIN = 12.0        # action units per radian of orientation error (1 unit ~ 0.037 rad/step)
NATIVE_LIMIT = 25
X_BRAKE_COMP = 0.010   # m, R005-R013: hold point ends -1.1+-0.2 cm in x from the aim (pitch +0.037)
SETTLE_ERR = 0.015     # m, start holding still once this close to the target ...
SETTLE_SPEED = 0.030   # m/step ... and already braking (R003/R004: 0.021-0.026 m/step after braking step)


def _masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return {
        "red": (r > 120) & (g < 50) & (b < 50),
        "green": (g > 120) & (r < 80) & (b < 80),
        "blue": (b > 120) & (r < 60) & (g < 60),
    }


def _colors_present(rgb):
    out = {}
    for k, m in _masks(rgb).items():
        n = int(m.sum())
        if n >= MIN_PIX:
            out[k] = n
    return out


def _backproject(depth, K, E, us, vs):
    z = depth[vs, us].astype(np.float64) / 1000.0
    # pixel centres at +0.5 (front/wrist estimates agree to <2 mm with this convention)
    x = (us + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (vs + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R


def _cube_estimates(frame):
    """World estimates of each coloured cube: top-face centroid and top height."""
    rgb = frame.front_rgb
    depth = frame.front_depth
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = frame.front_extrinsic
    out = {}
    for k, m in _masks(rgb).items():
        vs, us = np.nonzero(m)
        if len(us) < MIN_PIX:
            continue
        ok = depth[vs, us] > 0
        if ok.sum() < 3:
            continue
        P = _backproject(depth, K, E, us[ok], vs[ok])
        zt = float(np.percentile(P[:, 2], 90))
        top = P[P[:, 2] > zt - 0.006]
        if len(top) < 3:
            top = P
        c = top.mean(0)
        out[k] = {"xy": [float(c[0]), float(c[1])], "ztop": zt, "n": int(len(us))}
    return out


def _euler_to_R(roll, pitch, yaw):
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
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:   # near pi: use the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * ang
    return v / (2.0 * s) * ang


def _rot_command(tcp, target_rpy):
    """Normalised rotation args that turn the measured orientation toward target_rpy.
    Measured in demos: world rotation vector per step ~= -0.037 * r."""
    R = _euler_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _euler_to_R(target_rpy[0], target_rpy[1], target_rpy[2])
    w = _rotvec(Rt @ R.T)
    r = -ROT_GAIN * w
    return [float(np.clip(v, -1.0, 1.0)) for v in r], float(np.linalg.norm(w))


def _pos_command(tcp, target):
    e = np.asarray(target, dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64)
    a = np.clip(POS_GAIN * e, -1.0, 1.0)
    return [float(v) for v in a], e


def perceive_decide(obs, goal, memory, last):
    """Track cue colours, the blank delay and the test layout; pick the next act."""
    def g(key, default):
        v = memory.get(key) if isinstance(memory, dict) else None
        return default if v is None else v
    mem = {
        "stage": str(g("stage", "cue")),
        "cue_colors": list(g("cue_colors", [])),
        "cue_step": g("cue_step", -1),
        "blank_seen": bool(g("blank_seen", False)),
        "target_color": str(g("target_color", "")),
        "target_xyz": list(g("target_xyz", [])),
        "test_step": g("test_step", -1),
        "touch_tries": int(g("touch_tries", 0)),
        "log": str(g("log", "")),
    }
    for fr in obs.frames:
        if mem["target_color"]:
            break
        cols = _colors_present(fr.front_rgb)
        names = sorted(cols)
        if not mem["cue_colors"]:
            if names:
                mem["cue_colors"] = names
                mem["cue_step"] = int(fr.step)
            continue
        if not names:
            mem["blank_seen"] = True
            continue
        new = [c for c in names if c not in mem["cue_colors"]]
        is_test = mem["blank_seen"] or (len(names) >= 3 and len(names) > len(mem["cue_colors"]))
        if not is_test:
            # still the cue display: accumulate every colour shown
            mem["cue_colors"] = sorted(set(mem["cue_colors"]) | set(names))
            continue
        if not new:
            continue
        # the imposter: the new colour (largest blob if several)
        new.sort(key=lambda c: -cols[c])
        tc = new[0]
        est = _cube_estimates(fr)
        if tc in est:
            xyz = [est[tc]["xy"][0], est[tc]["xy"][1], est[tc]["ztop"]]
            how = "depth"
        else:
            xyz = _ray_plane_estimate(fr, tc, 0.028)
            if xyz is not None:
                xyz[2] = 0.04   # assumed cube top (all demos: 0.041-0.042)
            how = "ray"
        if xyz is None:
            continue
        mem["target_color"] = tc
        mem["target_xyz"] = xyz
        mem["test_step"] = int(fr.step)
        mem["stage"] = "touch"
        mem["log"] += "test@%d cue=%s seen=%s target=%s xyz=%s (%s); " % (
            int(fr.step), ",".join(mem["cue_colors"]), ",".join(names), tc,
            ",".join("%.3f" % v for v in xyz), how)
    if mem["stage"] == "cue" or not mem["target_color"]:
        return {"memory": mem, "act": {"tool": "act_wait", "args": {
            "cue_colors": mem["cue_colors"], "blank_seen": mem["blank_seen"], "max_steps": 30}}}
    mem["touch_tries"] += 1
    x, y, zt = mem["target_xyz"]
    args = {"x": x, "y": y, "ztop": zt}
    if mem["touch_tries"] > 1:
        # first touch did not end the episode: try a slightly lower, shorter hold
        args.update({"settle_steps": 3, "close_steps": 3, "z_offset": -0.003})
    return {"memory": mem, "act": {"tool": "act_touch", "args": args}}


def _ray_plane_estimate(frame, color, z_plane):
    """Fallback target estimate: intersect the camera ray through the colour blob centroid
    with the horizontal plane z = z_plane (cube top)."""
    m = _masks(frame.front_rgb)[color]
    vs, us = np.nonzero(m)
    if len(us) < 3:
        return None
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    R, t = E[:, :3], E[:, 3]
    d_cam = np.array([(us.mean() + 0.5 - K[0, 2]) / K[0, 0], (vs.mean() + 0.5 - K[1, 2]) / K[1, 1], 1.0])
    d = R.T @ d_cam
    c = -R.T @ t
    if abs(d[2]) < 1e-6:
        return None
    s = (z_plane - c[2]) / d[2]
    p = c + s * d
    return [float(p[0]), float(p[1]), float(z_plane)]


def act_wait(robot, cue_colors, blank_seen=False, max_steps=30):
    """Send zero commands (gripper open) until the test layout appears: a colour not in
    cue_colors after an empty table (or more cubes than the cue showed). With empty
    cue_colors, waits until any cube is visible. Steps at least once per call when nothing
    changed at the start. Returns the colour history."""
    hist = []
    blank = bool(blank_seen)
    for i in range(int(max_steps)):
        fr = robot.now()
        names = sorted(_colors_present(fr.front_rgb))
        hist.append("".join(n[0] for n in names) or "-")
        if cue_colors and not names:
            blank = True
        if i > 0:
            if not cue_colors and names:
                return {"stopped": "cue_seen", "history": hist, "step": int(fr.step)}
            new = [n for n in names if n not in cue_colors]
            if cue_colors and new and (blank or (len(names) >= 3 and len(names) > len(cue_colors))):
                return {"stopped": "new_color", "history": hist, "step": int(fr.step)}
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"stopped": "max_steps", "history": hist, "blank": blank, "step": int(robot.now().step)}


def act_touch(robot, x, y, ztop, settle_steps=4, close_steps=5, z_offset=0.005):
    """Move the TCP over the cube at (x, y) (top face at height ztop) with open fingers,
    descend so the open fingers straddle it (TCP at cube-centre height + z_offset) using
    velocity-damped P control, then hold still (zero arm command) for settle_steps, then
    close the gripper for close_steps as a contact fallback.
    Returns phase-by-phase measured TCP/finger trace."""
    rpy = [math.pi, 0.0, 0.0]
    z_final = max(0.012, float(ztop) - 0.020 + float(z_offset))
    x_cube = float(x)
    x = x_cube + X_BRAKE_COMP      # braking pitches the hand and shifts the TCP ~-1.1 cm in x
    log = []
    phase = "approach"
    n_phase = 0
    prev = None
    for i in range(40):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=np.float64)
        v = np.zeros(3) if prev is None else tcp[:3] - prev
        prev = tcp[:3].copy()
        fing = float(np.mean(fr.fingers))
        exy = math.hypot(x - tcp[0], y - tcp[1])
        err = math.sqrt(exy * exy + (tcp[2] - z_final) ** 2)
        speed = float(np.linalg.norm(v))
        rcmd = [0.0, 0.0, 0.0]
        rerr = 0.0
        if phase == "approach":
            if (err < SETTLE_ERR and speed < SETTLE_SPEED) or (err < 0.004) or n_phase >= 16:
                phase, n_phase = "settle", 0
        if phase == "approach":
            tz = z_final
            if exy > XY_TOL or fing < 0.030:
                tz = max(Z_HOVER, z_final)
                if tcp[2] < Z_HOVER - 0.01:
                    tz = tcp[2]
            e = np.array([x, y, tz]) - tcp[:3]
            a = [float(v_) for v_ in np.clip(POS_GAIN * (e - VEL_DAMP * v), -1.0, 1.0)]
            rcmd, rerr = _rot_command(tcp, rpy)
            grip = 1.0
        elif phase == "settle":
            a = [0.0, 0.0, 0.0]      # target = measured pose: brake and hold still
            grip = 1.0
            if n_phase >= int(settle_steps):
                phase, n_phase = "close", 0
        if phase == "close":
            a = [0.0, 0.0, 0.0]
            grip = -1.0
            if n_phase >= int(close_steps):
                break
        log.append([phase[0], int(fr.step), round(float(tcp[0]), 3), round(float(tcp[1]), 3),
                    round(float(tcp[2]), 3), round(fing, 3), round(err, 3), round(speed, 3)])
        n_phase += 1
        robot.step(a[0], a[1], a[2], rcmd[0], rcmd[1], rcmd[2], grip)
    fr = robot.now()
    return {"end_tcp": [round(float(v_), 4) for v_ in fr.tcp[:3]], "cube_xy": [round(x_cube, 4), round(float(y), 4)],
            "fingers": [round(float(v_), 4) for v_ in fr.fingers], "phase": phase,
            "trace": log[-16:]}
