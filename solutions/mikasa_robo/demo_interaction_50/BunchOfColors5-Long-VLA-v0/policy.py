"""BunchOfColors5-Long-VLA policy: remember cue colours, wait for the ring of 9
cubes, touch every cued colour (open gripper, TCP lowered over the cube centre),
then press the centre button with a closed gripper and lift."""

PALETTE = {
    "red": (231, 5, 5), "darkred": (169, 5, 5), "green": (5, 231, 5),
    "blue": (5, 5, 231), "yellow": (231, 231, 5), "magenta": (231, 5, 231),
    "cyan": (5, 231, 231), "teal": (5, 169, 169), "orange": (231, 169, 5),
}
COLOR_MAX_DIST = 70.0
CUBE_XY_OFFSET = (-0.009, 0.0)    # demo-measured: expert touch xy minus top-face centroid
BUTTON_XY_OFFSET = (-0.004, 0.0)  # R001/R002: push tilts pitch and slides TCP ~+2.5 cm in x; cap spans x -0.151..-0.095
Z_TOUCH = 0.052
Z_SAFE = 0.15
Z_CLEAR = 0.09
GAIN = 10.0
ROT_GAIN = 3.0
ROT_MAX = 0.3
STEP_LIMIT = 700


# ---------------------------------------------------------------- perception
def _world_points(frame):
    depth = np.asarray(frame.front_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * depth
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    return pw, depth > 0


def _classify(rgb):
    best, bd = None, 1e9
    for name in PALETTE:
        d = float(np.linalg.norm(np.asarray(PALETTE[name], dtype=np.float64) - rgb))
        if d < bd:
            best, bd = name, d
    return best, bd


def _detect(frame):
    """Cubes (colour, xy centre, confidence) and button cap centre from the front camera."""
    pw, valid = _world_points(frame)
    rgb = np.asarray(frame.front_rgb).astype(np.float64)
    X, Y, Z = pw[..., 0], pw[..., 1], pw[..., 2]
    region = valid & (X > -0.45) & (X < 0.40) & (np.abs(Y) < 0.55)
    cap = region & (Z > 0.050) & (Z < 0.075) & (np.abs(Y) < 0.15) & (X > -0.30) & (X < 0.05)
    button = None
    if cap.sum() >= 15:
        button = [float(X[cap].mean()), float(Y[cap].mean())]
    m = region & (Z > 0.008) & (Z < 0.05)
    if button is not None:
        m = m & (((X - button[0]) ** 2 + (Y - button[1]) ** 2) > 0.065 ** 2)
    pts = pw[m]
    cols = rgb[m]
    cubes = []
    free = np.ones(len(pts), dtype=bool)
    guard = 0
    while free.any() and guard < 30:
        guard += 1
        i = int(np.flatnonzero(free)[0])
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        sel = free & (d < 0.04)
        c = pts[sel][:, :2].mean(0)
        d2 = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
        sel2 = free & (d2 < 0.035)
        free = free & ~(sel | sel2)
        n = int(sel2.sum())
        if n < 10:
            continue
        top = sel2 & (pts[:, 2] > 0.034)
        if top.sum() < 3:
            continue
        med = np.median(cols[top], 0)
        name, cd = _classify(med)
        xy = pts[top][:, :2].mean(0)
        cubes.append({"color": name, "dist": round(cd, 1), "x": float(xy[0]), "y": float(xy[1]), "n": n})
    return cubes, button


def _good(cubes):
    return [c for c in cubes if c["dist"] < COLOR_MAX_DIST]


# ---------------------------------------------------------------- rotation
def _R(rpy):
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_cmd(tcp, rpy_des):
    """Normalized rotation args holding rpy_des. CPU IK left-multiplies Rx(-0.1r0)Ry(-0.1r1)Rz(-0.1r2),
    i.e. a world-frame offset of about -0.1*r, so r = -gain * (world rotvec error)."""
    if rpy_des is None:
        return [0.0, 0.0, 0.0]
    w = _rotvec(_R(rpy_des) @ _R(tcp[3:6]).T)
    r = np.clip(-ROT_GAIN * w, -ROT_MAX, ROT_MAX)
    return [float(r[0]), float(r[1]), float(r[2])]


# ---------------------------------------------------------------- motion helpers
def _tcp(robot):
    return np.asarray(robot.now().tcp, dtype=np.float64)


def _xyz_cmd(p, target, amax=1.0):
    a = np.clip(GAIN * (np.asarray(target) - p[:3]), -amax, amax)
    return [float(a[0]), float(a[1]), float(a[2])]


def _transit_target(p, x, y, z_low):
    """Target for one step of a lift / traverse / descend move to (x, y, z_low)."""
    d = math.hypot(x - p[0], y - p[1])
    if d > 0.012:
        tz = Z_SAFE
        if p[2] < Z_CLEAR:
            # lift before moving sideways near the objects
            f = max(0.0, (p[2] - (Z_CLEAR - 0.03)) / 0.03)
            tx = p[0] + f * (x - p[0])
            ty = p[1] + f * (y - p[1])
            return [tx, ty, tz], d
        return [x, y, tz], d
    return [x, y, z_low], d


# ---------------------------------------------------------------- policy
def perceive_decide(obs, goal, memory, last):
    """Track stage: read cue colours, wait for the ring, touch each cued cube, press the button."""
    mem = dict(memory)
    for k, v in (("stage", "cue"), ("cue_colors", []), ("ring_colors", []), ("ring_xy", []),
                 ("touched", []), ("button_xy", None), ("home_rpy", None), ("press_count", 0),
                 ("steps_used", 0), ("log", "")):
        if mem.get(k) is None:
            mem[k] = v
    cur = obs.current
    mem["steps_used"] = int(cur.step)
    if mem["home_rpy"] is None:
        mem["home_rpy"] = [float(v) for v in cur.tcp[3:6]]
    stage = mem["stage"]

    if stage == "cue":
        # cue cubes are visible on the reset frames
        seen = {}
        for fr in obs.sampled(5):
            cubes, button = _detect(fr)
            if button is not None:
                mem["button_xy"] = button
            g = _good(cubes)
            if 1 <= len(g) <= 6:
                for c in g:
                    seen[c["color"]] = seen.get(c["color"], 0) + 1
        if seen:
            mem["cue_colors"] = sorted(seen)
            mem["stage"] = "wait"
            mem["log"] = mem["log"] + "cue@%d:%s;" % (cur.step, ",".join(sorted(seen)))
            return {"memory": mem, "act": {"tool": "act_wait_ring", "args": {"min_count": 7}}}
        # no cue yet: wait for any cube to appear
        return {"memory": mem, "act": {"tool": "act_wait_ring", "args": {"min_count": 1}}}

    if stage == "wait":
        # frames observed while waiting may still show the cue: union their colours
        extra = []
        for fr in obs.sampled(4)[:-1]:
            if fr.step > 60:
                break
            gc = _good(_detect(fr)[0])
            if 1 <= len(gc) <= 6:
                for c in gc:
                    if c["color"] not in mem["cue_colors"] and c["color"] not in extra:
                        extra.append(c["color"])
        if extra:
            mem["cue_colors"] = sorted(mem["cue_colors"] + extra)
            mem["log"] = mem["log"] + "cue+%s;" % ",".join(extra)
        cubes, button = _detect(cur)
        if button is not None:
            mem["button_xy"] = button
        g = _good(cubes)
        names = [c["color"] for c in g]
        missing = [c for c in mem["cue_colors"] if c not in names]
        if len(g) >= 7 and missing and mem["log"].count("ringretry") < 5:
            mem["log"] = mem["log"] + "ringretry@%d:%s;" % (cur.step, ",".join(missing))
            return {"memory": mem, "act": {"tool": "act_wait_ring", "args": {"min_count": 7, "max_steps": 2}}}
        if len(g) >= 7:
            mem["ring_colors"] = [c["color"] for c in g]
            mem["ring_xy"] = [[c["x"] + CUBE_XY_OFFSET[0], c["y"] + CUBE_XY_OFFSET[1]] for c in g]
            mem["stage"] = "touch"
            missing = [c for c in mem["cue_colors"] if c not in mem["ring_colors"]]
            mem["log"] = mem["log"] + "ring@%d:n=%d,missing=%s;" % (cur.step, len(g), ",".join(missing))
            stage = "touch"
        else:
            return {"memory": mem, "act": {"tool": "act_wait_ring", "args": {"min_count": 7}}}

    if stage == "touch":
        if last is not None and last.startswith("act_touch") and '"reached": true' in last:
            col = last.split('"color": "')[1].split('"')[0] if '"color": "' in last else ""
            if col and col not in mem["touched"]:
                mem["touched"] = mem["touched"] + [col]
        todo = [c for c in mem["cue_colors"] if c not in mem["touched"] and c in mem["ring_colors"]]
        if todo:
            p = cur.tcp
            best = None
            # choose next target: shortest remaining tour ending at the button
            bx = mem["button_xy"] if mem["button_xy"] is not None else [-0.12, 0.0]
            pos = {c: mem["ring_xy"][mem["ring_colors"].index(c)] for c in todo}
            perms = [[]]
            for _ in todo:
                perms = [q + [c] for q in perms for c in todo if c not in q]
            for q in perms:
                L, prev = 0.0, (float(p[0]), float(p[1]))
                for c in q:
                    L += math.hypot(pos[c][0] - prev[0], pos[c][1] - prev[1])
                    prev = pos[c]
                L += math.hypot(bx[0] - prev[0], bx[1] - prev[1])
                if best is None or L < best[0]:
                    best = (L, q)
            c = best[1][0]
            mem["log"] = mem["log"] + "touch:%s@%d;" % (c, cur.step)
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": pos[c][0], "y": pos[c][1], "color": c, "rpy": mem["home_rpy"]}}}
        mem["stage"] = "press"
        stage = "press"

    if stage == "press" and int(mem["press_count"]) >= 2 and "RETOUCH" not in mem["log"] \
            and int(cur.step) < STEP_LIMIT - 180:
        # two presses without native success: a touch may not have registered. Re-touching an
        # already touched cube was harmless in R017, so redo every touch once, then press again.
        mem["log"] = mem["log"] + "RETOUCH@%d;" % cur.step
        mem["touched"] = []
        mem["stage"] = "touch"
        return perceive_decide(obs, goal, mem, None)

    if stage == "press":
        bx = mem["button_xy"] if mem["button_xy"] is not None else [-0.125, 0.0]
        mem["press_count"] = int(mem["press_count"]) + 1
        mem["log"] = mem["log"] + "press%d@%d;" % (mem["press_count"], cur.step)
        return {"memory": mem, "act": {"tool": "act_press", "args": {
            "x": bx[0] + BUTTON_XY_OFFSET[0], "y": bx[1] + BUTTON_XY_OFFSET[1], "rpy": mem["home_rpy"]}}}

    return {"memory": mem, "act": {"tool": "act_wait_ring", "args": {"min_count": 99, "max_steps": 20}}}


def act_wait_ring(robot, min_count=7, max_steps=700):
    """Hold still (gripper open) until at least min_count palette cubes are visible on the
    front camera, or max_steps elapse. Returns step counts and what was seen."""
    n = 0
    last_count = 0
    first = int(robot.now().step)
    while n < max_steps:
        fr = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        cubes, _ = _detect(fr)
        last_count = len(_good(cubes))
        if last_count >= min_count:
            return {"seen": True, "count": last_count, "steps": n, "start_step": first, "at_step": int(fr.step)}
    return {"seen": False, "count": last_count, "steps": n, "start_step": first}


def act_touch(robot, x, y, color="", rpy=None, z_touch=Z_TOUCH, hold=3, max_steps=90):
    """Lift to the safe height, move over (x, y), lower the open gripper to z_touch, hold a few
    steps and return (the next move lifts). Reports measured minimum height and xy error."""
    n = 0
    held = 0
    zmin = 9.0
    exy = None
    start = int(robot.now().step)
    margin = None
    while n < max_steps:
        p = _tcp(robot)
        tgt, d = _transit_target(p, x, y, z_touch)
        cmd = _xyz_cmd(p, tgt) + _rot_cmd(p, rpy)
        if n == 0:
            chk = robot.dry_run(*cmd, mode="delta")
            margin = chk.get("min_joint_margin_rad")
        robot.step(*cmd, 1.0)
        n += 1
        q = _tcp(robot)
        if math.hypot(x - q[0], y - q[1]) < 0.02:
            zmin = min(zmin, float(q[2]))
        if math.hypot(x - q[0], y - q[1]) < 0.012 and q[2] < z_touch + 0.006:
            held += 1
            exy = [float(x - q[0]), float(y - q[1])]
            if held >= hold:
                return {"reached": True, "color": color, "steps": n, "start": start,
                        "zmin": round(zmin, 4), "exy": exy, "margin0": margin}
    return {"reached": False, "color": color, "steps": n, "start": start, "zmin": round(zmin, 4),
            "tcp": [float(v) for v in _tcp(robot)[:3]], "margin0": margin}


def act_press(robot, x, y, rpy=None, push_steps=4, z_lift=0.20, wait_steps=8, max_steps=90):
    """Closed gripper: move over the button at the safe height, descend, push down for
    push_steps steps at the bottom (z<0.057, at most 8 push steps), then lift to z_lift and hold. Success ends the episode inside this call."""
    n = 0
    start = int(robot.now().step)
    phase = "approach"
    pushed = 0
    bottom = 0
    zmin = 9.0
    waited = 0
    while n < max_steps:
        p = _tcp(robot)
        rot = _rot_cmd(p, rpy)
        if phase == "approach":
            tgt, d = _transit_target(p, x, y, 0.07)
            cmd = _xyz_cmd(p, tgt)
            if d < 0.012 and p[2] < 0.078:
                phase = "push"
        if phase == "push":
            a = _xyz_cmd(p, [x, y, p[2]])
            cmd = [a[0], a[1], -0.7]
            pushed += 1
            zmin = min(zmin, float(p[2]))
            if p[2] < 0.057:
                bottom += 1
            if bottom > push_steps or pushed > 8:
                phase = "lift"
        if phase == "lift":
            cmd = _xyz_cmd(p, [x, y, z_lift])
            if p[2] > z_lift - 0.01:
                waited += 1
                if waited > wait_steps:
                    break
        robot.step(*(cmd + rot), -1.0)
        n += 1
    return {"pressed": pushed > 0, "push_steps": pushed, "bottom_steps": bottom, "steps": n, "start": start, "zmin": round(zmin, 4),
            "tcp": [float(v) for v in _tcp(robot)[:3]]}
