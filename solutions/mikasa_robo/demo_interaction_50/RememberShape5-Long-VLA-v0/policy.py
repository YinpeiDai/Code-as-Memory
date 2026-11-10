"""RememberShape5-Long policy: remember the cue shape, wait for the five candidates,
touch the candidate whose shape class matches the cue (TCP descent above it)."""

HOVER_DZ = 0.06      # hover height above object top before the final descent
TOUCH_DZ = -0.002     # first touch target: TCP this far relative to the object top (fingers straddle the object;
                      # R007/R009 sphere succeeded only at z 0.049/0.044, cube at 0.053-0.057)
MIN_TOUCH_Z = 0.022   # never command the TCP lower than this (finger tips reach ~9 mm below the TCP)
AIM_OFFSET = (-0.005, -0.001)  # expert final TCP minus our top-centroid estimate (demo mean)


def _find_cue(frames):
    """Scan frames (latest first) for a single blue object (cue phase). Returns (features, step) or (None, None)."""
    idx = list(range(len(frames)))
    # check every frame cheaply; full analysis only on the candidate frame
    best = None
    for i in idx:
        cnt, _ = _blob_count_fast(frames[i])
        if 40 <= cnt <= 700:
            best = i
            break
    if best is None:
        return None, None
    objs = _scene_objects(frames[best])
    if len(objs) == 1:
        return objs[0], int(frames[best].step)
    return None, None


def _choose_target(cue, objs):
    same = [o for o in objs if o["shape"] == cue["shape"]]
    if len(same) == 1:
        return same[0], "class_match"
    pool = same if len(same) > 1 else objs
    best = min(pool, key=lambda o: _feat_dist(o, cue))
    return best, ("feature_nearest_of_%d" % len(pool))


def perceive_decide(obs, goal, memory, last):
    """Stage machine: observe cue -> wait for candidates -> touch matching object -> retries."""
    mem = dict(memory)
    for k in ("stage", "cue_shape", "cue_info", "cue_step", "candidates", "candidate_shapes",
              "target", "target_shape", "attempts", "notes"):
        mem.setdefault(k, None)
    stage = mem["stage"] or "start"
    attempts = int(mem["attempts"] or 0)
    notes = mem["notes"] or ""
    cur = obs.current

    # record the cue as soon as it is visible
    if mem["cue_shape"] is None:
        cue, cstep = _find_cue(list(obs.frames))
        if cue is not None:
            mem["cue_shape"] = cue["shape"]
            mem["cue_info"] = "zmax=%.4f zmed=%.4f round=%.3f rmax=%.4f hole=%.3f x=%.4f y=%.4f" % (
                cue["zmax"], cue["zmed"], cue["roundness"], cue["rmax"], cue["hole"], cue["x"], cue["y"])
            mem["cue_step"] = cstep
            notes += "cue %s@%d; " % (cue["shape"], cstep)

    if stage in ("start", "wait"):
        objs = _scene_objects(cur) if _blob_count_fast(cur)[0] > 700 else []
        if len(objs) >= 4:
            stage = "select"
        else:
            mem["stage"] = "wait"
            mem["attempts"] = attempts
            mem["notes"] = notes
            return {"memory": mem, "act": {"tool": "act_wait_for_objects", "args": {"max_steps": 600}}}

    if stage == "select":
        objs = _scene_objects(cur)
        mem["candidates"] = [[round(o["x"], 4), round(o["y"], 4), round(o["zmax"], 4)] for o in objs]
        mem["candidate_shapes"] = [o["shape"] for o in objs]
        if mem["cue_shape"] is not None:
            cue = _parse_cue(mem["cue_info"], mem["cue_shape"])
            tgt, how = _choose_target(cue, objs)
        else:
            tgt, how = objs[0], "no_cue_guess"
        mem["target"] = [round(tgt["x"], 4), round(tgt["y"], 4), round(tgt["zmax"], 4)]
        mem["target_shape"] = tgt["shape"]
        notes += "cands %s target %s via %s at step %d; " % (mem["candidate_shapes"], tgt["shape"], how, cur.step)
        stage = "touch"

    # touch / retries
    tx, ty, top = mem["target"]
    plans = [
        (TOUCH_DZ, 0.0, 0.0),
        (TOUCH_DZ - 0.004, 0.005, 0.0),
        (TOUCH_DZ - 0.004, 0.0, 0.006),
        (TOUCH_DZ - 0.004, 0.0, -0.006),
        (TOUCH_DZ - 0.004, 0.010, 0.0),
        (TOUCH_DZ - 0.004, -0.005, 0.0),
    ]
    if attempts < len(plans):
        dz, ox, oy = plans[attempts]
        dz = max(top + dz, MIN_TOUCH_Z) - top
        mem["stage"] = "touch"
        mem["attempts"] = attempts + 1
        mem["notes"] = notes + "touch#%d dz=%.3f off=(%.3f,%.3f) at step %d; " % (attempts + 1, dz, ox, oy, cur.step)
        return {"memory": mem, "act": {"tool": "act_touch", "args": {
            "x": tx + AIM_OFFSET[0] + ox, "y": ty + AIM_OFFSET[1] + oy, "top": top,
            "touch_dz": dz, "hover_dz": HOVER_DZ if attempts == 0 else 0.03}}}
    mem["stage"] = "hold"
    mem["attempts"] = attempts
    mem["notes"] = notes
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 50}}}


def _parse_cue(info, shape):
    vals = {}
    for tok in info.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            vals[k] = float(v)
    return {"shape": shape, "zmax": vals.get("zmax", 0.04), "zmed": vals.get("zmed", 0.04),
            "roundness": vals.get("round", 0.8), "rmax": vals.get("rmax", 0.02), "hole": vals.get("hole", 0.05)}


def act_wait_for_objects(robot, max_steps=600):
    """Hold still (zero motion, gripper open) until >=4 blue candidate objects are visible in the wrist
    camera for 2 consecutive frames, or 5 at once. Returns the step and blob count."""
    n = 0
    prev_ok = False
    cnt = 0
    while n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        f = robot.now()
        cnt, _ = _blob_count_fast(f)
        if cnt > 700:
            k = len(_scene_objects(f))
            if k >= 5 or (k >= 4 and prev_ok):
                return {"objects_visible": True, "count": k, "step": int(f.step), "waited": n}
            prev_ok = k >= 4
        else:
            prev_ok = False
    return {"objects_visible": False, "blue_px": cnt, "waited": n}


def _goto(robot, goal, vmax, tol, max_steps, settle_n, trace, still_req=True):
    """Closed-loop move of the TCP to goal (xyz) with zero rotation command and gripper open.
    Command = (goal - measured)/0.1 clipped to norm vmax (action units, 1 = 0.1 m target offset)."""
    goal = np.asarray(goal, dtype=float)
    prev = np.asarray(robot.now().tcp[:3], dtype=float)
    still = 0
    for i in range(max_steps):
        pos = np.asarray(robot.now().tcp[:3], dtype=float)
        err = goal - pos
        cmd = err / 0.1
        nrm = float(np.sqrt((cmd ** 2).sum()))
        if nrm > vmax:
            cmd = cmd * (vmax / nrm)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, 1.0)
        f = robot.now()
        p2 = np.asarray(f.tcp[:3], dtype=float)
        speed = float(np.sqrt(((p2 - pos) ** 2).sum()))
        e2 = float(np.sqrt(((goal - p2) ** 2).sum()))
        trace.append([int(f.step), round(float(p2[0]), 4), round(float(p2[1]), 4), round(float(p2[2]), 4)])
        if e2 < tol and (speed < 0.0015 or not still_req):
            still += 1
            if still >= settle_n:
                return True, i + 1, e2
        else:
            still = 0
    return False, max_steps, e2


def act_touch(robot, x, y, top, touch_dz=0.012, hover_dz=0.06):
    """Move above (x, y) to hover height top+hover_dz, descend to TCP z = top+touch_dz and hold there
    until settled. Returns arrival flags, final TCP and a short trace."""
    trace = []
    start = robot.now()
    z_hover = top + hover_dz
    chk = robot.dry_run(x, y, top + touch_dz, float(start.tcp[3]), float(start.tcp[4]), float(start.tcp[5]))
    reach = {"reachable": chk.get("reachable"), "margin": chk.get("joint_margin")} if isinstance(chk, dict) else {}
    ok1, n1, e1 = _goto(robot, [x, y, z_hover], 1.0, 0.012, 60, 1, trace, still_req=False)
    ok2, n2, e2 = _goto(robot, [x, y, top + touch_dz], 0.5, 0.003, 50, 4, trace)
    f = robot.now()
    return {"hover_ok": ok1, "hover_steps": n1, "touch_ok": ok2, "touch_steps": n2, "err": round(e2, 4),
            "tcp": [round(float(v), 4) for v in f.tcp], "step": int(f.step), "dry_run": reach,
            "trace": trace[-12:]}


def act_hold(robot, steps=50):
    """Hold the current pose (zero motion, gripper open) for a number of steps."""
    for _ in range(steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": steps, "tcp": [round(float(v), 4) for v in robot.now().tcp]}
