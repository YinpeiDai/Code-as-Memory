"""RememberColor3: observe the cue cube's colour, wait, then touch the cube of the same colour.

Stages (memory.stage):
  observe  - cue / delay phases: robot commands are zeroed by the task; one native step per decision,
             every new front frame is segmented (depth band above the table) and the cue colour is stored.
  touch    - response phase: three cubes visible; target = cube whose colour best matches the cue.
             act_touch drives the TCP to the cube centre (x, y, z=0.02) top-down and holds still.
  probe    - fallback variants if holding did not end the episode (lower, higher, then centre again).
"""

_Z_TOUCH = 0.02        # TCP height for the touch (cube centre; cube side 0.04 m, top at 0.040)
_SAFE_Z = 0.075        # stay above this until x/y are aligned (fingertips clear the cube top)
_ALIGN_XY = 0.012      # per-axis x/y error allowed before descending below _SAFE_Z
_MAX_COLOR_DIST = 0.3  # demos: matching cube ~0.01, others ~1.3 (scratch/test_perc.py)
_FIELDS = ("stage", "cue_rgb", "cue_frames", "cue_step", "response_step", "cubes", "target",
           "match_margin", "frames_seen", "attempts", "log")


def _probe_goal(target, attempt):
    """Touch pose for the n-th touch attempt (0 = primary hypothesis: cube centre)."""
    x, y = float(target[0]), float(target[1])
    variants = [
        (0.0, 0.0, _Z_TOUCH),     # primary: TCP at the cube centre, held still
        (0.009, 0.0, 0.014),      # expert-like end pose (mean expert end offset +9 mm x, low z)
        (0.0, 0.0, 0.03),         # higher, nearer the cube top
        (0.0, 0.0, _Z_TOUCH),     # back to centre
    ]
    dx, dy, z = variants[attempt % len(variants)]
    return [x + dx, y + dy, z]


def perceive_decide(obs, goal, memory, last):
    """Update cue/target memory from all new frames and choose the next act."""
    mem = {k: memory.get(k) for k in _FIELDS}
    if mem["stage"] is None:
        mem["stage"] = "observe"
    if mem["cue_frames"] is None:
        mem["cue_frames"] = 0
    if mem["frames_seen"] is None:
        mem["frames_seen"] = -1
    if mem["attempts"] is None:
        mem["attempts"] = 0
    if mem["log"] is None:
        mem["log"] = ""

    frames = list(obs.frames)
    for f in frames:
        st = int(f.step) if f.step is not None else mem["frames_seen"] + 1
        if st <= mem["frames_seen"]:
            continue
        mem["frames_seen"] = st
        if mem["target"] is not None:
            continue
        cubes = _detect_cubes(f)
        n = len(cubes)
        if n == 1 and mem["response_step"] is None:
            rgb = cubes[0]["rgb"]
            k = int(mem["cue_frames"])
            if mem["cue_rgb"] is None or k == 0:
                mem["cue_rgb"] = [float(v) for v in rgb]
                mem["cue_step"] = st
            else:   # running mean over cue frames
                mem["cue_rgb"] = [(float(a) * k + float(b)) / (k + 1) for a, b in zip(mem["cue_rgb"], rgb)]
            mem["cue_frames"] = k + 1
        elif n >= 2:
            if mem["response_step"] is None:
                mem["response_step"] = st
            mem["cubes"] = [c["xyz"] for c in cubes]
            if mem["cue_rgb"] is not None:
                best, margin, dists = _match(mem["cue_rgb"], cubes)
                if dists[best] > _MAX_COLOR_DIST and st - int(mem["response_step"]) < 2:
                    # no cube resembles the cue (occlusion / partial view?): look again next frame
                    mem["log"] = "no good colour match yet, dists=" + str([round(d, 3) for d in dists])
                    continue
                mem["target"] = [float(v) for v in cubes[best]["xyz"]]
                mem["match_margin"] = margin
                mem["log"] = "target from colour match, dists=" + str([round(d, 3) for d in dists])
            else:
                # cue never seen: no evidence; fall back to the cube closest to the cue position (0, 0)
                order = sorted(range(n), key=lambda i: math.hypot(cubes[i]["xyz"][0], cubes[i]["xyz"][1]))
                mem["target"] = [float(v) for v in cubes[order[0]]["xyz"]]
                mem["match_margin"] = 0.0
                mem["log"] = "cue not observed; fallback target nearest the cue location"
            mem["stage"] = "touch"

    if mem["target"] is None:
        return {"memory": mem, "act": {"tool": "act_wait", "args": {}}}

    cur = obs.current
    d_prev = [0.0, 0.0, 0.0]
    if len(frames) >= 2:
        d_prev = [float(frames[-1].tcp[i]) - float(frames[-2].tcp[i]) for i in range(3)]
    attempt = int(mem["attempts"])
    g = _probe_goal(mem["target"], attempt)
    yaw_goal = _snap_yaw(float(cur.tcp[5]), 0.0)
    mem["attempts"] = attempt + 1
    mem["stage"] = "touch" if attempt == 0 else "probe"
    hold = 3 if attempt == 0 else 2
    args = {"x": g[0], "y": g[1], "z": g[2], "yaw_goal": yaw_goal,
            "d_prev": d_prev, "hold_steps": hold, "max_steps": 24}
    return {"memory": mem, "act": {"tool": "act_touch", "args": args}}


def act_wait(robot):
    """One native step with zero arm motion and the gripper commanded open (cue/delay phases)."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp[:3]]}


def act_touch(robot, x, y, z, yaw_goal=0.0, d_prev=None, hold_steps=3, max_steps=24):
    """Move the TCP top-down to (x, y, z) with the gripper open, then hold still.

    Stays above the safe height until x/y are aligned. Returns after hold_steps consecutive
    settled steps (error < 4 mm, speed < 1.5 mm/step) or max_steps; reports measured errors.
    """
    goal = [float(x), float(y), float(z)]
    preview = None
    try:
        chk = robot.dry_run(goal[0], goal[1], goal[2], math.pi, 0.0, float(yaw_goal), mode="absolute")
        preview = {"within_joint_limits": bool(chk.get("within_joint_limits")),
                   "min_joint_margin_rad": chk.get("min_joint_margin_rad")}
    except Exception as exc:   # diagnostics only; motion does not depend on it
        preview = {"unavailable": str(exc)[:80]}
    f = robot.now()
    prev = [float(v) for v in f.tcp[:3]]
    d = [0.0, 0.0, 0.0] if d_prev is None else [float(v) for v in d_prev]
    a_prev = [0.0] * 6
    settled = 0
    n = 0
    trace = []
    err = 1.0
    speed = 1.0
    while n < int(max_steps):
        tcp = [float(v) for v in robot.now().tcp]
        if n > 0:
            d = [tcp[i] - prev[i] for i in range(3)]
        prev = tcp[:3]
        exy = max(abs(goal[0] - tcp[0]), abs(goal[1] - tcp[1]))
        zg = goal[2] if exy <= _ALIGN_XY else max(goal[2], _SAFE_Z)
        err = math.sqrt(sum((goal[i] - tcp[i]) ** 2 for i in range(3)))
        speed = math.sqrt(sum(v * v for v in d))
        if err < 0.004 and speed < 0.0015:
            settled += 1
        else:
            settled = 0
        if settled > int(hold_steps):
            break
        a = _command(tcp, d, a_prev, [goal[0], goal[1], zg], float(yaw_goal))
        trace.append([round(tcp[0], 3), round(tcp[1], 3), round(tcp[2], 3)])
        robot.step(a[0], a[1], a[2], a[3], a[4], a[5], 1.0)
        a_prev = a
        n += 1
    f = robot.now()
    tcp = [float(v) for v in f.tcp]
    return {"steps": n, "step": int(f.step), "preview": preview, "settled_steps": settled,
            "final_err_m": round(err, 4), "final_speed_m_per_step": round(speed, 4),
            "tcp": [round(v, 4) for v in tcp], "goal": [round(v, 4) for v in goal],
            "fingers": [round(float(v), 4) for v in f.fingers], "trace_tail": trace[-6:]}
