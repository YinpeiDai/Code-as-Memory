"""RememberColor5: observe the single cue cube's colour, wait through the empty delay,
then touch the candidate cube of the same colour.

Perception: front RGB-D (fixed camera) -> saturated pure-colour pixels -> world points
(depth, K, extrinsic) -> one cube estimate per colour (top-face centre, top height).
Memory: cue colour (seen while exactly one cube is visible), candidate list (first frame
with >= 2 cubes), chosen target, stage and step bookkeeping.
Control: closed-loop pd_ee_delta_pose steps on the measured TCP with a descent cone,
vertical gripper, open fingers; settle on the cube, then (fallback) close the fingers,
then (fallback) re-touch variants while steps remain.
"""

COLOR_NAMES = {1: "red", 2: "green", 4: "blue", 3: "yellow", 5: "magenta", 6: "cyan"}
CUBE_HALF = 0.02          # measured: cube top at z~0.041, footprint ~0.04 x 0.04
MIN_PIXELS = 12           # front-camera cubes cover ~80-110 px (cue ~88 px)
NATIVE_STEP_LIMIT = 25
STEP_M = 0.1              # 1.0 command unit = 0.1 m target offset
ROT_STEP = 0.1            # 1.0 rotation unit = 0.1 rad
POS_GAIN = 2.5            # commanded offset = POS_GAIN * error (tuned offline)
ROT_GAIN = 0.5
VEL_DAMP = 0.8            # velocity feedback (m/step -> m offset), cancels coasting
FINGER_BELOW_TCP = 0.008  # fingertips extend ~8 mm below the TCP (wrist depth)


# ----------------------------------------------------------------------------- perception
def _color_codes(rgb):
    im = np.asarray(rgb).astype(np.float32)
    mx = im.max(axis=-1)
    mn = im.min(axis=-1)
    sat = (mx > 50.0) & (mn < 0.2 * mx)
    hi = im > (0.6 * mx)[..., None]
    mid = ((im > (0.2 * mx)[..., None]) & (im <= (0.6 * mx)[..., None])).any(axis=-1)
    code = hi[..., 0].astype(np.int32) + 2 * hi[..., 1].astype(np.int32) + 4 * hi[..., 2].astype(np.int32)
    return np.where(sat & (~mid), code, 0)


def _backproject(depth, K, E, ys, xs):
    z = depth[ys, xs].astype(np.float64) / 1000.0
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    X = (xs.astype(np.float64) + 0.5 - cx) / fx * z
    Y = (ys.astype(np.float64) + 0.5 - cy) / fy * z
    pc = np.stack([X, Y, z], axis=1)
    E = np.asarray(E, dtype=np.float64)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


def _detect_cubes(frame):
    """List of {color, x, y, z_top, pixels} for each pure-coloured cube on the table."""
    rgb = getattr(frame, "front_rgb", None)
    depth = getattr(frame, "front_depth", None)
    K = getattr(frame, "front_K", None)
    E = getattr(frame, "front_extrinsic", None)
    if rgb is None or depth is None or K is None or E is None:
        return []
    depth = np.asarray(depth)
    codes = _color_codes(rgb)
    out = []
    for code in sorted(COLOR_NAMES):
        mask = (codes == code) & (depth > 0)
        ys, xs = np.nonzero(mask)
        if len(ys) < MIN_PIXELS:
            continue
        W = _backproject(depth, K, E, ys, xs)
        ok = (W[:, 2] > -0.02) & (W[:, 2] < 0.12)
        W = W[ok]
        if len(W) < MIN_PIXELS:
            continue
        med = np.median(W[:, :2], axis=0)
        near = np.linalg.norm(W[:, :2] - med, axis=1) < 0.04
        W = W[near]
        if len(W) < MIN_PIXELS:
            continue
        ztop = float(np.percentile(W[:, 2], 95))
        top = W[W[:, 2] > ztop - 0.006]
        if len(top) < 3:
            top = W
        c = top[:, :2].mean(axis=0)
        out.append({"color": COLOR_NAMES[code], "x": float(c[0]), "y": float(c[1]),
                    "z_top": ztop, "pixels": int(len(W))})
    return out


def _parse_goal(goal):
    g = (goal or "").lower()
    if "same color" in g or "same colour" in g or "matching" in g:
        return "match_cue_color"
    return "match_cue_color"


# ----------------------------------------------------------------------------- decision
FIELDS = ("stage", "cue_color", "cue_pixels", "cue_step", "cue_frames", "candidates",
          "candidates_step", "target_color", "target_xyz", "target_confidence",
          "steps_seen", "touch_attempts", "history")


def _init_memory(memory):
    m = {}
    for k in FIELDS:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "observe"
    if m["cue_frames"] is None:
        m["cue_frames"] = 0
    if m["steps_seen"] is None:
        m["steps_seen"] = 0
    if m["touch_attempts"] is None:
        m["touch_attempts"] = 0
    if m["history"] is None:
        m["history"] = ""
    return m


def _log(m, text):
    h = (m["history"] or "") + text + "; "
    m["history"] = h[-900:]


def _touch_plan(m):
    """(x, y, z_goal, gripper, offset label) for the next touch attempt."""
    tx, ty, tz = m["target_xyz"]
    z_top = 2.0 * tz
    z_goal = min(max(z_top - 0.026, 0.010), 0.030)   # ~5 mm below cube centre
    n = int(m["touch_attempts"])
    if n == 0:
        return tx, ty, z_goal, 1.0, "centre_open"
    if n == 1:
        return tx, ty, z_goal, -1.0, "centre_close"
    if n == 2:
        return tx - 0.012, ty, max(z_goal - 0.004, 0.009), 1.0, "minus_x_low_open"
    return tx, ty, z_goal, -1.0, "centre_close"


def perceive_decide(obs, goal, memory, last):
    """Update cue/candidate memory from every new frame, then choose the next act."""
    m = _init_memory(memory)
    _parse_goal(goal)
    frames = list(obs.frames) if obs is not None else []
    for f in frames:
        step = getattr(f, "step", None)
        if isinstance(step, (int, float)) and step > m["steps_seen"]:
            m["steps_seen"] = int(step)
        if m["candidates"] is not None:
            continue
        dets = _detect_cubes(f)
        if len(dets) == 1:
            d = dets[0]
            if m["cue_color"] is None:
                m["cue_color"] = d["color"]
                m["cue_pixels"] = d["pixels"]
                m["cue_step"] = int(step) if isinstance(step, (int, float)) else 0
                _log(m, "cue %s px%d at step %s" % (d["color"], d["pixels"], step))
            if d["color"] == m["cue_color"]:
                m["cue_frames"] = int(m["cue_frames"]) + 1
        elif len(dets) >= 2:
            m["candidates"] = ",".join("%s:%.3f:%.3f:%.3f" % (d["color"], d["x"], d["y"], d["z_top"]) for d in dets)
            m["candidates_step"] = int(step) if isinstance(step, (int, float)) else 0
            _log(m, "candidates %d at step %s" % (len(dets), step))
            match = [d for d in dets if d["color"] == m["cue_color"]]
            if match:
                d = max(match, key=lambda q: q["pixels"])
                m["target_confidence"] = "matched"
            else:
                # cue never seen or colour absent: no evidence; take the largest blob
                d = max(dets, key=lambda q: q["pixels"])
                m["target_confidence"] = "guess"
            m["target_color"] = d["color"]
            m["target_xyz"] = [d["x"], d["y"], 0.5 * d["z_top"]]
            m["stage"] = "touch"
            _log(m, "target %s (%s)" % (d["color"], m["target_confidence"]))

    if m["target_xyz"] is None:
        m["stage"] = "observe" if m["cue_color"] is None else "wait_candidates"
        return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 1}}}

    remaining = NATIVE_STEP_LIMIT - int(m["steps_seen"])
    x, y, zg, grip, label = _touch_plan(m)
    n = int(m["touch_attempts"])
    if last is not None and n > 0:
        _log(m, "attempt %d done: %s" % (n, str(last)[-160:]))
    m["touch_attempts"] = n + 1
    m["stage"] = "touch_" + label
    if n == 0:
        args = {"x": x, "y": y, "z_goal": zg, "gripper": grip, "max_steps": max(4, remaining), "hold_steps": 2}
    elif n == 1:
        args = {"x": x, "y": y, "z_goal": zg, "gripper": grip, "max_steps": max(3, remaining), "hold_steps": 3}
    else:
        args = {"x": x, "y": y, "z_goal": zg, "gripper": grip, "max_steps": max(3, remaining), "hold_steps": 40}
    return {"memory": m, "act": {"tool": "act_touch", "args": args}}


# ----------------------------------------------------------------------------- control
def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(M):
    c = (float(np.trace(M)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    w = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / (2.0 * s)
    return w * th


def _command(tcp, goal, R_des, vel=None, scale=1.0):
    """Per-axis saturated PD command with a descent cone around the target column.
    `vel` is the measured TCP displacement of the previous step (m/step)."""
    p = np.asarray(tcp[:3], dtype=np.float64)
    v = np.zeros(3) if vel is None else np.asarray(vel, dtype=np.float64)
    e = np.asarray(goal, dtype=np.float64) - p
    exy = math.hypot(e[0], e[1])
    # xy error expected after the coasting part of the next step
    exy_pred = max(exy, math.hypot(e[0] - VEL_DAMP * v[0], e[1] - VEL_DAMP * v[1]))
    cone = min(2.5 * max(0.0, exy_pred - 0.005), 0.05 + 1.0 * max(0.0, exy_pred - 0.02))
    z_target = max(goal[2], goal[2] + cone)
    ez = z_target - p[2]
    if ez > 0 and p[2] > 0.07:
        ez = 0.0
    err = np.array([e[0], e[1], ez])
    lin = np.clip(scale * (POS_GAIN * err - VEL_DAMP * v) / STEP_M, -1.0, 1.0)
    w = _rotvec(R_des @ _rot(tcp[3:6]).T)
    rot = np.clip(-ROT_GAIN * w / ROT_STEP, -1.0, 1.0)
    return lin, rot, exy, ez


def _joint_speed(prev, cur):
    if prev is None or cur is None:
        return None
    return float(np.max(np.abs(np.asarray(cur, dtype=np.float64) - np.asarray(prev, dtype=np.float64)))) / 0.05


def act_wait(robot, steps=1):
    """Send `steps` zero-motion native steps (gripper open) while the cue/delay plays."""
    n = max(1, min(int(steps), 30))
    for _ in range(n):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"steps": n, "tcp": [round(float(v), 4) for v in f.tcp[:3]],
            "fingers": [round(float(v), 4) for v in f.fingers]}


def act_touch(robot, x, y, z_goal, gripper=1.0, max_steps=14, hold_steps=2):
    """Move the TCP (vertical gripper) onto world point (x, y, z_goal) with a safe descent
    cone, then hold still there for `hold_steps` steps. `gripper` +1 open, -1 close
    (close is applied only once the TCP is on the target). Returns measured outcome."""
    f0 = robot.now()
    yaw0 = float(f0.tcp[5])
    R_des = _rot([math.pi, 0.0, yaw0])
    goal = np.array([float(x), float(y), float(z_goal)])
    prev_q = np.asarray(f0.joints, dtype=np.float64)
    prev_p = np.asarray(f0.tcp[:3], dtype=np.float64)
    vel = np.zeros(3)
    arrived_at = None
    held = 0
    steps = 0
    speed = None
    exy = None
    ez = None
    max_steps = max(1, min(int(max_steps), 60))
    opened = 0
    scale = 1.0
    prev_e = None
    if float(gripper) > 0:
        # after a closing attempt, release the cube before moving sideways
        while steps < max_steps and opened < 3 and float(np.min(robot.now().fingers)) < 0.03 \
                and float(robot.now().tcp[2]) < 0.06:
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            steps += 1
            opened += 1
        prev_q = np.asarray(robot.now().joints, dtype=np.float64)
        prev_p = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    while steps < max_steps:
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=np.float64)
        vel = tcp[:3] - prev_p
        prev_p = tcp[:3].copy()
        e_now = float(np.linalg.norm(goal - tcp[:3]))
        if prev_e is not None and e_now < 0.06 and e_now > prev_e + 0.002:
            scale = max(0.3, scale * 0.7)   # error grew near the target: damp the loop
        prev_e = e_now
        lin, rot, exy, ez = _command(tcp, goal, R_des, vel, scale)
        dz = goal[2] - tcp[2]
        on_target = exy < 0.006 and abs(dz) < 0.008
        if on_target and arrived_at is None:
            arrived_at = steps
        g = 1.0
        if arrived_at is not None and float(gripper) < 0:
            g = float(gripper)
        robot.step(float(lin[0]), float(lin[1]), float(lin[2]), float(rot[0]), float(rot[1]), float(rot[2]), g)
        steps += 1
        q = np.asarray(robot.now().joints, dtype=np.float64)
        speed = _joint_speed(prev_q, q)
        prev_q = q
        if arrived_at is not None:
            held += 1
            if held >= int(hold_steps) and speed is not None and speed < 0.15:
                break
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=np.float64)
    return {"steps": steps, "opened_first": opened, "gain_scale": round(scale, 3), "arrived_after": arrived_at, "held": held,
            "tcp": [round(float(v), 4) for v in tcp[:3]],
            "err_xyz": [round(float(v), 4) for v in (tcp[:3] - goal)],
            "joint_speed": None if speed is None else round(speed, 3),
            "fingers": [round(float(v), 4) for v in f.fingers]}
