"""RememberColor9: remember the cue cube colour, wait for the 9-cube ring, press the matching cube.

Perception: front camera RGB-D -> world points; cube top faces are the pixels at
world z in [0.033, 0.05] whose colour is within a tight tolerance of one of nine
palette colours measured in the assigned demonstrations. Control: closed loop on
the measured TCP with pd_ee_delta_pose translation only (gripper closed),
transit at a safe height, then press down on the top-face centre.
"""

PALETTE = {
    "red": (231, 4, 4),
    "darkred": (169, 4, 4),
    "green": (4, 231, 4),
    "blue": (4, 4, 231),
    "yellow": (231, 231, 4),
    "magenta": (231, 4, 231),
    "cyan": (4, 231, 231),
    "teal": (4, 169, 169),
    "orange": (231, 169, 4),
}
TOP_Z_MIN = 0.033
TOP_Z_MAX = 0.05
MIN_PIXELS = 5
RING_MIN_CLASSES = 5
STEP_LIMIT = 25
Z_SAFE = 0.075
Z_PRESS = 0.035
XY_TOL = 0.012
GAIN = 20.0


def _world_points(depth, K, E):
    d = np.asarray(depth, dtype=float) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R, d > 0


def _densest(P, radius=0.025):
    """Mean of the points around the densest point (robust to stray edge pixels)."""
    if len(P) > 400:
        idx = np.linspace(0, len(P) - 1, 400).astype(int)
        S = P[idx]
    else:
        S = P
    D = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    cnt = (D < radius).sum(1)
    seed = S[int(np.argmax(cnt))]
    near = np.sqrt(((P - seed) ** 2).sum(-1)) < radius
    return P[near].mean(0), int(near.sum())


def _detect(frame, tol=15.0):
    """Colour class -> [x, y, pixel_count] of cube top faces seen by the front camera."""
    pw, valid = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgb = np.asarray(frame.front_rgb, dtype=float)
    z = pw[..., 2]
    m = valid & (z > TOP_Z_MIN) & (z < TOP_Z_MAX) & (pw[..., 0] > -0.45)
    P = rgb[m]
    Wp = pw[m][:, :2]
    out = {}
    if len(P) == 0:
        return out
    names = list(PALETTE)
    C = np.array([PALETTE[n] for n in names], dtype=float)
    D = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    k = D.argmin(1)
    dm = D.min(1)
    for i, n in enumerate(names):
        sel = (k == i) & (dm < tol)
        if int(sel.sum()) >= MIN_PIXELS:
            c, cnt = _densest(Wp[sel])
            if cnt >= MIN_PIXELS:
                out[n] = [float(c[0]), float(c[1]), int(cnt)]
    return out


def _f(x, default):
    return default if x is None else x


def _stepno(frame):
    s = getattr(frame, "step", None)
    if isinstance(s, (int, float)) or hasattr(s, "dtype"):
        return int(s)
    return 0


def perceive_decide(obs, goal, memory, last):
    """Track cue colour, wait for the cube ring, then press the matching cube."""
    mem = dict(memory or {})
    stage = _f(mem.get("stage"), "observe")
    cue_votes = _f(mem.get("cue_votes"), "")
    votes = {}
    if cue_votes:
        for part in cue_votes.split(","):
            if ":" in part:
                a, b = part.split(":")
                votes[a] = int(b)
    target_color = mem.get("target_color")
    target_xy = mem.get("target_xy")
    attempts = int(_f(mem.get("attempts"), 0))
    steps_seen = int(_f(mem.get("steps_seen"), 0))
    ring_step = mem.get("ring_step")
    log = _f(mem.get("log"), "")

    frames = list(obs.frames)
    last_step = None
    ring = None
    for fr in frames:
        last_step = _stepno(fr)
        if target_xy is not None and stage in ("touch", "retry", "grasp", "done"):
            continue  # ring already localised; the arm may occlude cubes now
        det = _detect(fr)
        if len(det) >= RING_MIN_CLASSES:
            if ring is None:
                ring = det
                if ring_step is None:
                    ring_step = last_step
        elif ring is None and target_xy is None and 1 <= len(det) <= 2:
            best = max(det, key=lambda n: det[n][2])
            votes[best] = votes.get(best, 0) + 1
    steps_seen = max(steps_seen, last_step if last_step is not None else 0)
    cue_color = max(votes, key=lambda n: votes[n]) if votes else None

    if ring is not None and target_xy is None:
        if cue_color is not None and cue_color in ring:
            target_color = cue_color
        elif cue_color is not None:
            # cue colour not segmented with the tight tolerance: retry wider
            wide = _detect(frames[-1], tol=40.0)
            if cue_color in wide:
                ring = wide
                target_color = cue_color
        if target_color is None:
            target_color = sorted(ring, key=lambda n: -ring[n][2])[0]
            log += "|cue_missing_guess:" + str(target_color)
        target_xy = [ring[target_color][0], ring[target_color][1]]
        log += "|ring@%s n=%d cue=%s tgt=(%.3f,%.3f)" % (
            str(ring_step), len(ring), str(cue_color), target_xy[0], target_xy[1])
        stage = "touch"

    act = {"tool": "act_wait_for_cubes", "args": {"max_steps": 15}}
    if target_xy is not None:
        if stage == "touch" and attempts == 0:
            act = {"tool": "act_press", "args": {"x": target_xy[0], "y": target_xy[1],
                                                 "z_press": Z_PRESS, "gripper": -1.0,
                                                 "lift_first": False}}
            attempts = 1
        elif attempts == 1:
            stage = "retry"
            act = {"tool": "act_press", "args": {"x": target_xy[0], "y": target_xy[1],
                                                 "z_press": 0.02, "gripper": -1.0,
                                                 "lift_first": True}}
            attempts = 2
        elif attempts == 2:
            stage = "grasp"
            act = {"tool": "act_grasp", "args": {"x": target_xy[0], "y": target_xy[1]}}
            attempts = 3
        else:
            stage = "done"
            act = {"tool": "act_hold", "args": {"n": 3}}
        if last:
            log += "|" + str(last)[:160]
    else:
        stage = "wait"
    if len(log) > 1500:
        log = log[-1500:]
    cue_votes = ",".join("%s:%d" % (n, votes[n]) for n in sorted(votes))
    return {
        "memory": {
            "stage": stage,
            "cue_color": cue_color,
            "cue_votes": cue_votes,
            "target_color": target_color,
            "target_xy": target_xy,
            "attempts": attempts,
            "steps_seen": steps_seen,
            "ring_step": ring_step,
            "log": log,
        },
        "act": act,
    }


def act_wait_for_cubes(robot, max_steps=15):
    """Send zero motion (gripper closed) until the cube ring is visible in the front camera or max_steps pass."""
    n = 0
    seen = len(_detect(robot.now())) >= RING_MIN_CLASSES
    while not seen and n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
        seen = len(_detect(robot.now())) >= RING_MIN_CLASSES
    return {"steps": n, "ring_visible": bool(seen), "step": _stepno(robot.now())}


def _servo_cmd(p, goal, gain=GAIN):
    e = np.asarray(goal, dtype=float) - np.asarray(p, dtype=float)
    return [float(v) for v in np.clip(e * gain, -1.0, 1.0)]


def act_press(robot, x, y, z_press=0.035, gripper=-1.0, lift_first=False, max_steps=18):
    """Move above (x, y) at a safe height, descend and press down on the cube top until stalled.

    Returns measured TCP, steps used and whether a stall (contact) was observed.
    """
    n = 0
    if lift_first:
        while n < max_steps:
            p = np.asarray(robot.now().tcp, dtype=float)
            if p[2] > Z_SAFE - 0.01:
                break
            c = _servo_cmd(p[:3], [p[0], p[1], Z_SAFE + 0.01])
            robot.step(c[0], c[1], c[2], 0.0, 0.0, 0.0, gripper)
            n += 1
    stall = 0
    zs = []
    contact = False
    descending = False
    while n < max_steps:
        p = np.asarray(robot.now().tcp, dtype=float)
        dxy = math.hypot(x - p[0], y - p[1])
        if not descending and dxy < XY_TOL:
            descending = True
        elif descending and dxy > 0.025:
            descending = False
        if descending:
            zg = z_press
        else:
            zg = Z_SAFE
        c = _servo_cmd(p[:3], [x, y, zg])
        if not descending and p[2] < Z_SAFE - 0.02:
            c[0] = 0.0  # low and off target: rise before moving sideways
            c[1] = 0.0
        robot.step(c[0], c[1], c[2], 0.0, 0.0, 0.0, gripper)
        n += 1
        q = np.asarray(robot.now().tcp, dtype=float)
        zs.append(round(float(q[2]), 4))
        if descending and q[2] < 0.065 and abs(q[2] - p[2]) < 0.003:
            stall += 1
        else:
            stall = 0
        if stall >= 2:
            contact = True
            break
    q = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "contact_stall": contact, "tcp": [round(float(v), 4) for v in q[:3]],
            "dxy": round(math.hypot(x - q[0], y - q[1]), 4),
            "z_trace": zs[-6:], "step": _stepno(robot.now())}


def act_grasp(robot, x, y, max_steps=12):
    """Fallback: lift, open, descend around the cube (TCP z 0.025) and close the fingers on it."""
    n = 0
    while n < max_steps:
        p = np.asarray(robot.now().tcp, dtype=float)
        dxy = math.hypot(x - p[0], y - p[1])
        f = np.asarray(robot.now().fingers, dtype=float)
        opened = float(f.min()) > 0.035
        if dxy < 0.008 and opened:
            zg = 0.025
        else:
            zg = 0.08
        c = _servo_cmd(p[:3], [x, y, zg])
        robot.step(c[0], c[1], c[2], 0.0, 0.0, 0.0, 1.0)
        n += 1
        if robot.now().tcp[2] < 0.032:
            break
    for _ in range(4):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
    q = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "tcp": [round(float(v), 4) for v in q[:3]],
            "fingers": [round(float(v), 4) for v in robot.now().fingers], "step": _stepno(robot.now())}


def act_hold(robot, n=3):
    """Hold the current pose (zero motion, gripper closed) for n steps."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"held": int(n), "step": _stepno(robot.now())}
