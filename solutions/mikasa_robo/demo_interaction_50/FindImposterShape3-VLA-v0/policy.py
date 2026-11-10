"""FindImposterShape3: remember the two cue shapes, wait for the 3-object reveal,
touch the object whose shape class was not in the cue.

Perception: blue blobs in the wrist RGB (camera looks straight down at reset);
wrist depth classifies each blob: sphere = small flat-top fraction (domed depth),
cube = square flat top (low area / circumscribed-circle ratio), cylinder = round flat top.
Control: closed-loop pd_ee_delta_pose steps toward the target centre, xy first.
"""

SHAPES = ("cube", "cylinder", "sphere")
MIN_BLOB_PIX = 8
TOP_BAND_MM = 4.0
SPHERE_FTOP_MAX = 0.6     # demos: sphere <= 0.48, flat tops >= 0.74
CUBE_RATIO_MAX = 0.72     # demos: cube <= 0.66, cylinder >= 0.76
OBJ_HALF_HEIGHT = 0.02    # objects ~0.04 m tall (top z ~0.041 in demos)
EPISODE_STEP_LIMIT = 25


def _blue_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (b > 80) & (b > r + 50) & (b > g + 50)


def _components(mask, min_pix):
    ys, xs = np.nonzero(mask)
    pix = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while pix:
        seed = pix.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pix:
                        pix.discard(q)
                        stack.append(q)
                        comp.append(q)
        if len(comp) >= min_pix:
            comps.append(comp)
    return comps


def _backproject(u, v, d_m, K, E):
    pc = np.array([(u - K[0, 2]) / K[0, 0] * d_m, (v - K[1, 2]) / K[1, 1] * d_m, d_m])
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return R.T.dot(pc - t)


def _objects(frame):
    """Detect and classify blue objects in a wrist frame. Returns list of dicts."""
    rgb = frame.wrist_rgb
    depth = frame.wrist_depth
    K = np.asarray(frame.wrist_K, dtype=float)
    E = np.asarray(frame.wrist_extrinsic, dtype=float)
    out = []
    for comp in _components(_blue_mask(rgb), MIN_BLOB_PIX):
        ys = np.array([p[0] for p in comp])
        xs = np.array([p[1] for p in comp])
        d = depth[ys, xs].astype(float)
        ok = d > 0
        if ok.sum() < MIN_BLOB_PIX:
            continue
        dmin = d[ok].min()
        top = ok & (d <= dmin + TOP_BAND_MM)
        tx = xs[top].astype(float)
        ty = ys[top].astype(float)
        cx, cy = tx.mean(), ty.mean()
        rmax = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2).max() + 0.5
        ratio = float(top.sum() / (math.pi * rmax * rmax))
        ftop = float(top.sum() / len(xs))
        if ftop < SPHERE_FTOP_MAX:
            shape = "sphere"
            margin = (SPHERE_FTOP_MAX - ftop) / 0.15
        elif ratio < CUBE_RATIO_MAX:
            shape = "cube"
            margin = min((ftop - SPHERE_FTOP_MAX) / 0.15, (CUBE_RATIO_MAX - ratio) / 0.05)
        else:
            shape = "cylinder"
            margin = min((ftop - SPHERE_FTOP_MAX) / 0.15, (ratio - CUBE_RATIO_MAX) / 0.05)
        top_w = _backproject(cx, cy, dmin / 1000.0, K, E)
        out.append({
            "shape": shape, "conf": float(min(1.0, max(0.0, margin))),
            "u": float(xs.mean()), "v": float(ys.mean()), "n": int(len(xs)),
            "ftop": round(ftop, 3), "ratio": round(ratio, 3),
            "top": [float(top_w[0]), float(top_w[1]), float(top_w[2])],
        })
    return out


def _empty_memory():
    return {
        "stage": "observe_cue",
        "cue_shapes": [],
        "cue_votes": "",
        "cue_frames": 0,
        "blank_frames": 0,
        "frames_seen": 0,
        "target_shape": "",
        "target_position": [],
        "reveal_shapes": [],
        "notes": "",
        "last_step": -1,
    }


def perceive_decide(obs, goal, memory, last):
    """Track cue -> blank -> reveal from wrist frames; on reveal pick the imposter
    (shape class absent from the cue) and reach it; otherwise wait one step."""
    mem = _empty_memory()
    if memory:
        for k in mem:
            if k in memory and memory[k] is not None:
                mem[k] = memory[k]
    votes = {}
    if mem["cue_votes"]:
        for item in mem["cue_votes"].split(","):
            if ":" in item:
                k, v = item.split(":")
                votes[k] = int(v)
    frames = list(obs.frames) if obs.frames else [obs.current]
    last_step = int(mem["last_step"])
    frames = [fr for fr in frames if int(fr.step) > last_step]
    reveal = None
    for fr in frames:
        mem["last_step"] = int(fr.step)
        if mem["stage"] == "reaching":
            continue
        objs = _objects(fr)
        mem["frames_seen"] = int(mem["frames_seen"]) + 1
        n = len(objs)
        if mem["stage"] == "observe_cue":
            if n == 2:
                mem["cue_frames"] = int(mem["cue_frames"]) + 1
                for o in objs:
                    votes[o["shape"]] = votes.get(o["shape"], 0) + 1
            elif n == 0 and mem["cue_frames"] > 0:
                mem["stage"] = "wait_reveal"
                mem["blank_frames"] = 1
            elif n >= 3:
                # reveal without a preceding blank (or no cue seen): decide now
                if mem["cue_frames"] == 0:
                    mem["notes"] = "no cue frames seen before reveal"
                mem["stage"] = "wait_reveal"
                reveal = objs
                break
        elif mem["stage"] == "wait_reveal":
            if n >= 3:
                reveal = objs
                break
            mem["blank_frames"] = int(mem["blank_frames"]) + 1
    mem["cue_votes"] = ",".join("%s:%d" % (k, v) for k, v in sorted(votes.items()))
    # cue shapes = the (up to) two most voted classes
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    mem["cue_shapes"] = [k for k, _ in ranked[:2]]
    if mem["stage"] in ("reaching", "done"):
        mem["stage"] = "reaching"
        steps_left = EPISODE_STEP_LIMIT - int(mem["last_step"])
        return {"memory": mem, "act": {"tool": "act_reach", "args": {
            "target": mem["target_position"], "max_steps": max(1, steps_left)}}}
    if reveal is not None:
        cue = mem["cue_shapes"]
        cands = [o for o in reveal if o["shape"] not in cue]
        if len(cands) != 1:
            # fallback: the least-voted / lowest-confidence explanation
            def score(o):
                return votes.get(o["shape"], 0) - 0.5 * (1.0 - o["conf"])
            cands = sorted(reveal, key=score)[:1]
            mem["notes"] = "ambiguous imposter; fallback by votes"
        tgt = cands[0]
        pos = [tgt["top"][0], tgt["top"][1], tgt["top"][2] - OBJ_HALF_HEIGHT]
        mem["target_shape"] = tgt["shape"]
        mem["target_position"] = pos
        mem["reveal_shapes"] = [o["shape"] for o in reveal]
        mem["stage"] = "reaching"
        steps_left = EPISODE_STEP_LIMIT - int(mem["last_step"])
        mem["notes"] = (mem["notes"] + " reveal=" + ";".join(
            "%s@(%.3f,%.3f,%.3f)c%.2f" % (o["shape"], o["top"][0], o["top"][1], o["top"][2], o["conf"])
            for o in reveal)).strip()
        return {"memory": mem, "act": {"tool": "act_reach", "args": {
            "target": pos, "max_steps": max(1, steps_left)}}}
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}


def act_wait(robot, steps=1):
    """Send zero-motion steps (gripper open) while the cue/blank phase runs."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"waited": int(steps), "step": int(f.step), "tcp": [float(v) for v in f.tcp]}


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz.dot(Ry).dot(Rx)


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return w * (ang / (2.0 * s))


RESP_A = 0.029   # fitted per-step response to a unit command (m); demos 0.031, CPU rollout R001 0.028
ROT_OFF_DIST = 0.04  # m; within this of the goal stop orientation corrections (R002: they kept joints moving)
DEADBAND = 0.003     # m
RESP_B = 0.28    # fitted carry-over of the previous step's displacement; demos 0.24, R001 0.32


def _reach_command(tcp, vel, target, R_goal, gain, rot_gain, z_floor, xy_tol):
    p = np.array([float(v) for v in tcp[:3]])
    tg = np.array([float(v) for v in target[:3]])
    goal = tg.copy()
    if np.linalg.norm(tg[:2] - p[:2]) > xy_tol:
        goal[2] = max(tg[2], z_floor)
    e = goal - p
    cmd = np.clip(gain * (e - RESP_B * np.asarray(vel)) / RESP_A, -1.0, 1.0)
    # near the goal: no orientation correction (lets the wrist settle) and a small deadband
    near = np.linalg.norm(tg - p) < ROT_OFF_DIST
    if near:
        for k in range(3):
            if abs(e[k]) < DEADBAND and abs(float(vel[k])) < DEADBAND:
                cmd[k] = 0.0
    w = _rotvec(R_goal.dot(_rot(tcp[3:6]).T))
    r = np.zeros(3) if near else -rot_gain * w / 0.1
    nr = np.linalg.norm(r)
    if nr > 0.5:
        r = r * (0.5 / nr)
    return [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(r[0]), float(r[1]), float(r[2])], e


BRAKE_KB = 0.12      # brake command = -KB*v/A (experts issue ~+0.12..0.2 reverse dz at full speed, success next frame)
BRAKE_BASE = 0.004   # m; brake when |goal error| <= BRAKE_BASE + BRAKE_KV*|v|
BRAKE_KV = 0.3
HOLD_RESUME = 0.03   # m; in hold, resume approach only if further than this from the goal


def act_reach(robot, target, max_steps=20, gain=0.7, rot_gain=0.5, z_floor=0.07, xy_tol=0.02, hover=0.02):
    """Move the TCP (gripper open) to `hover` m above world `target` (object centre):
    velocity-compensated approach (xy first while z >= z_floor, start orientation held
    while far), then one brake step against the measured velocity, then zero-offset holds
    so the arm comes to rest (native success appears to need TCP near the object AND the
    arm at rest). Steps until the episode ends or max_steps."""
    start = robot.now()
    R_goal = _rot(start.tcp[3:6])
    goal = [float(target[0]), float(target[1]), float(target[2]) + float(hover)]
    g_arr = np.array(goal)
    log = []
    modes = []
    n = 0
    prev = np.array([float(v) for v in start.tcp[:3]])
    vel = np.zeros(3)
    mode = "approach"
    for i in range(int(max_steps)):
        f = robot.now()
        cur = np.array([float(v) for v in f.tcp[:3]])
        if i > 0:
            vel = cur - prev
        prev = cur
        dist = float(np.linalg.norm(g_arr - cur))
        speed = float(np.linalg.norm(vel))
        xy_ok = float(np.linalg.norm(g_arr[:2] - cur[:2])) <= xy_tol
        if mode == "approach" and xy_ok and dist <= BRAKE_BASE + BRAKE_KV * speed:
            mode = "brake"
        elif mode == "hold" and dist > HOLD_RESUME:
            mode = "approach"
        if mode == "brake":
            c = np.clip(-BRAKE_KB * vel / RESP_A, -1.0, 1.0)
            cmd = [float(c[0]), float(c[1]), float(c[2]), 0.0, 0.0, 0.0]
        elif mode == "hold":
            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            cmd, e = _reach_command(f.tcp, vel, goal, R_goal, gain, rot_gain, z_floor, xy_tol)
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], 1.0)
        modes.append(mode[0])
        if mode == "brake":
            mode = "hold"
        n += 1
        g = robot.now()
        log.append([round(float(v), 3) for v in g.tcp[:3]])
    f = robot.now()
    err = [float(f.tcp[k]) - float(goal[k]) for k in range(3)]
    return {"steps": n, "final_tcp": [float(v) for v in f.tcp], "final_err": err,
            "dist": float(np.linalg.norm(err)), "modes": "".join(modes), "path": log[-8:], "ended": False}
