"""RememberColor3-VLA-v0 policy: remember the cue cube colour, wait for the three
candidate cubes, localise the matching one with front RGB-D, then touch it by
lowering the open gripper around it, holding still, and (fallback) closing the
gripper on it. See policy.md for the evidence."""

COLORS = ("red", "green", "blue")
MIN_PIX = 15          # pixels for a cube to count as visible in the front view
CUE_MAX_OTHER = 5     # other colours must be (almost) absent in a cue frame
WAIT_MAX_STEPS = 18   # delay phases in demos last 1..9 steps after the cue
TABLE_Z = 0.0


def _masks(rgb):
    """Boolean masks of saturated red/green/blue cube pixels."""
    a = np.asarray(rgb).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    out = {}
    for name, c, o1, o2 in (("red", r, g, b), ("green", g, r, b), ("blue", b, r, g)):
        out[name] = (c > 60) & (o1 * 10 < c * 3) & (o2 * 10 < c * 3)
    return out


def _backproject(frame, mask):
    """World XYZ (N,3) of masked front-camera pixels with valid depth."""
    d = np.asarray(frame.front_depth).astype(np.float64) / 1000.0
    v, u = np.nonzero(mask)
    if len(v) == 0:
        return np.zeros((0, 3))
    z = d[v, u]
    ok = z > 0
    v, u, z = v[ok], u[ok], z[ok]
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _detect(frame):
    """Per colour: pixel count and, when visible, top-face centre [x, y, top_z]."""
    res = {}
    masks = _masks(frame.front_rgb)
    for name in COLORS:
        m = masks[name]
        n = int(m.sum())
        info = {"pixels": n, "center": None}
        if n >= MIN_PIX:
            P = _backproject(frame, m)
            if len(P) >= 5:
                ztop = float(np.percentile(P[:, 2], 98))
                top = P[P[:, 2] > ztop - 0.006]
                if len(top) < 3:
                    top = P
                info["center"] = [float(top[:, 0].mean()), float(top[:, 1].mean()), ztop]
        res[name] = info
    return res


def _classify(det):
    """'cue' (exactly one colour), 'cubes' (all three), 'empty' or 'other'."""
    vis = [c for c in COLORS if det[c]["pixels"] >= MIN_PIX]
    if len(vis) == 3 and all(det[c]["center"] is not None for c in COLORS):
        return "cubes", vis
    if len(vis) == 1 and all(det[c]["pixels"] <= CUE_MAX_OTHER for c in COLORS if c != vis[0]):
        return "cue", vis
    if not vis and all(det[c]["pixels"] <= CUE_MAX_OTHER for c in COLORS):
        return "empty", vis
    return "other", vis


def _target_color_from_goal(goal, cue):
    """The instruction asks for the cube of the *same* colour as the cue; only an
    explicit single colour word without 'same' would override the cue."""
    g = (goal or "").lower()
    if "same" in g or cue is not None:
        return cue
    words = " " + g.replace(".", " ").replace(",", " ") + " "
    named = [c for c in COLORS if (" " + c + " ") in words]
    return named[0] if len(named) == 1 else cue


def _init(memory):
    m = {}
    m["stage"] = memory.get("stage") or "wait"
    m["cue_color"] = memory.get("cue_color")
    m["cue_step"] = memory.get("cue_step")
    m["cue_pixels"] = memory.get("cue_pixels")
    m["cue_frames"] = memory.get("cue_frames") if memory.get("cue_frames") is not None else 0
    m["cubes_step"] = memory.get("cubes_step")
    m["target_color"] = memory.get("target_color")
    m["target"] = memory.get("target")
    m["cube_positions"] = memory.get("cube_positions")
    m["decisions"] = memory.get("decisions") if memory.get("decisions") is not None else 0
    m["log"] = memory.get("log") or ""
    return m


def perceive_decide(obs, goal, memory, last):
    """Track the cue colour and the response-phase cubes, then choose the act.

    Stages: 'wait' (cue/delay, commands are zeroed by the task) -> 'touch'
    (target localised, run act_touch) -> 'done'."""
    memory = memory or {}
    m = _init(memory)
    m["decisions"] = int(m["decisions"]) + 1
    frames = list(obs.frames)
    cubes_det = None
    for fr in frames:
        det = _detect(fr)
        kind, vis = _classify(det)
        step = int(getattr(fr, "step", 0) or 0)
        if kind == "cue" and m["cubes_step"] is None:
            col = vis[0]
            if m["cue_color"] is None or m["cue_color"] == col:
                m["cue_color"] = col
                if m["cue_step"] is None:
                    m["cue_step"] = step
                m["cue_pixels"] = max(int(m["cue_pixels"] or 0), det[col]["pixels"])
                m["cue_frames"] = int(m["cue_frames"]) + 1
            else:
                # conflicting cue colour: keep the first one seen and log it
                m["log"] = (m["log"] + " cue_conflict@%d:%s" % (step, col))[-600:]
        elif kind == "cubes" and m["cubes_step"] is None:
            m["cubes_step"] = step
            cubes_det = det
    if cubes_det is None and m["cubes_step"] is not None and m["target"] is None:
        # cubes seen earlier but target never fixed: re-detect on current frame
        det = _detect(obs.current)
        if _classify(det)[0] == "cubes":
            cubes_det = det
    if cubes_det is not None and m["target"] is None:
        col = _target_color_from_goal(goal, m["cue_color"])
        if col is None:
            # cue never observed: fall back to the cube nearest the cue location (centre)
            best = None
            for c in COLORS:
                cc = cubes_det[c]["center"]
                dd = abs(cc[1]) + abs(cc[0])
                if best is None or dd < best[0]:
                    best = (dd, c)
            col = best[1]
            m["log"] = (m["log"] + " no_cue_fallback").strip()[-600:]
        m["target_color"] = col
        m["target"] = [round(v, 5) for v in cubes_det[col]["center"]]
        m["cube_positions"] = [[round(v, 4) for v in cubes_det[c]["center"]] for c in COLORS]
        m["stage"] = "touch"
    if m["stage"] == "touch" and m["target"] is not None:
        t = m["target"]
        act = {"tool": "act_touch", "args": {"x": float(t[0]), "y": float(t[1]), "top_z": float(t[2])}}
    else:
        m["stage"] = "wait" if m["stage"] != "done" else m["stage"]
        act = {"tool": "act_wait_for_cubes", "args": {"max_steps": WAIT_MAX_STEPS}}
    return {"memory": m, "act": act}


def act_wait_for_cubes(robot, max_steps=18):
    """Send zero motion with the gripper open, one native step at a time, until
    the front camera shows all three candidate cubes (or max_steps). Returns the
    number of steps, the step index when cubes appeared and cue colours seen."""
    n = 0
    seen = []
    kind = "none"
    for _ in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        fr = robot.now()
        det = _detect(fr)
        kind, vis = _classify(det)
        if kind == "cue":
            seen.append(vis[0])
        if kind == "cubes":
            return {"steps": n, "cubes_visible": True, "frame_step": int(fr.step), "cue_seen": seen}
    return {"steps": n, "cubes_visible": False, "last_kind": kind, "cue_seen": seen}


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return w / (2.0 * s) * ang


def _cmd(goal, pos, vel, R_des, R_meas, rot_gain=0.5, pos_scale=0.045, damp=0.6, vmax=1.0):
    """Normalised delta command towards goal with velocity damping and orientation hold."""
    err = np.asarray(goal, dtype=float) - pos
    a = (err - damp * vel) / pos_scale
    a = np.clip(a, -vmax, vmax)
    e = _rotvec(R_des @ R_meas.T)          # world rotation still needed
    r = np.clip(-e / 0.1 * rot_gain, -1.0, 1.0)   # offset = -0.1 * r
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])]


def act_touch(robot, x, y, top_z, hover_dz=0.025, touch_dz=-0.031, settle_steps=1,
              close_steps=4, max_steps=40):
    """Touch the cube whose top-face centre is (x, y, top_z).

    Moves the open gripper towards the cube while descending to the hover
    height (TCP top_z+hover_dz); once aligned (xy error < 12 mm) descends at
    full speed with the fingers straddling the cube to TCP z = top_z+touch_dz
    (at the table, as the experts do), holds still (settle_steps), then closes
    the gripper on the cube and keeps holding still, re-opening for 2 steps
    after every close_steps+2 closed steps. The episode ends at the native
    success; otherwise returns the stage log and measured TCP errors."""
    f0 = robot.now()
    tcp0 = np.asarray(f0.tcp, dtype=float)
    R_des = _rpy_to_R(math.pi, 0.0, float(tcp0[5]))
    top_z = float(top_z)
    z_touch = min(0.03, max(0.008, top_z + float(touch_dz)))
    z_hover = top_z + float(hover_dz)
    stage = "approach"
    log = []
    prev = tcp0[:3].copy()
    held = 0
    still = 0
    closed = 0
    grip = 1.0
    n = 0
    for _ in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        pos = tcp[:3]
        vel = pos - prev
        prev = pos.copy()
        R_meas = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
        xy_err = math.hypot(x - pos[0], y - pos[1])
        goal = [x, y, z_touch]
        if stage == "approach":
            # descend concurrently to the hover height; go below it only when aligned
            if xy_err < 0.012:
                zg = z_touch
            elif xy_err < 0.03:
                zg = top_z + 0.012      # fingertips stay just above the cube top
            else:
                zg = z_hover
            goal = [x, y, zg]
            low = pos[2] < z_touch + 0.006
            stalled = pos[2] < top_z - 0.015 and abs(vel[2]) < 0.003 and n > 0
            # resting on something above the table (e.g. a fingertip on the cube top)
            if xy_err < 0.012 and pos[2] < top_z + 0.015 and abs(vel[2]) < 0.002 and n > 1:
                still += 1
            else:
                still = 0
            if xy_err < 0.012 and (low or stalled or still >= 2):
                stage = "settle"
                log.append(["settle", n, round(xy_err, 4), round(float(pos[2]), 4)])
        if stage == "settle":
            held += 1
            if held > int(settle_steps):
                stage = "close"
                log.append(["close", n])
        if stage == "close":
            # close on the cube (finger contact) and hold still; every
            # close_steps+2 steps briefly re-open and close again so a fresh
            # contact is made while the TCP stays at the cube centre
            closed += 1
            phase = (closed - 1) % (int(close_steps) + 4)
            grip = 1.0 if phase >= int(close_steps) + 2 else -1.0
            if phase == 0 and closed > 1:
                log.append(["regrip", n])
        a = _cmd(goal, pos, vel, R_des, R_meas)
        if stage == "approach" and goal[2] == z_touch:
            # decisive final descent (the table stops the fingertips, as in the demos)
            a[2] = float(max(-1.0, min(1.0, (z_touch - pos[2]) / 0.025)))
        if stage in ("settle", "close"):
            # hold still: only small corrections
            a = [max(-0.3, min(0.3, v)) for v in a[:3]] + a[3:]
        robot.step(a[0], a[1], a[2], a[3], a[4], a[5], grip)
        n += 1
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    return {"steps": n, "stage": stage, "log": log,
            "final_tcp": [round(float(v), 4) for v in tcp[:3]],
            "xy_err": round(math.hypot(x - tcp[0], y - tcp[1]), 4),
            "fingers": [round(float(v), 4) for v in np.asarray(fr.fingers)]}
