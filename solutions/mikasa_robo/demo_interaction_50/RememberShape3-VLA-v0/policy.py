"""RememberShape3 policy: remember the cue shape, wait, then touch the candidate of the same shape.

Stages (memory.phase): wait -> cue -> delay -> reach -> hold -> refine -> hold ...
Arm commands are suppressed by the environment until the three candidates appear, so the policy only steps with
zero motion until then (gripper kept open).
"""

Z_SAFE = 0.075       # travel height of the TCP while moving in xy
Z_STRADDLE = 0.030   # TCP target height (object centre z=0.02) with the object between the open fingers
XY_DESCEND = 0.010   # only descend below Z_SAFE once xy error is below this
Z_NEAR = 0.055       # allowed TCP height while within 3 cm xy but not yet aligned


def _mem_defaults():
    return {
        "phase": "wait",
        "n_steps": 0,
        "cue_shape": "",
        "cue_info": "",
        "appear_step": -1,
        "cand_centers": [],
        "cand_shapes": [],
        "target": [],
        "target_info": "",
        "stage_count": 0,
        "log": "",
    }


def _append(mem, text):
    s = mem["log"] + text + "; "
    mem["log"] = s[-1500:]


def perceive_decide(obs, goal, memory, last):
    """Track cue/delay/candidate phases from the front/wrist RGB-D, remember the cue shape, pick the matching
    candidate and choose the next action stage."""
    mem = _mem_defaults()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    frames = list(obs.frames)
    cur = obs.current
    mem["n_steps"] = int(cur.step)

    if mem["phase"] in ("wait", "cue", "delay"):
        for fr in frames:
            if mem["phase"] not in ("wait", "cue", "delay"):
                break
            n = _count_objects(fr)
            if n == 1 and mem["cue_shape"] == "":
                P = _frame_points(fr, True)
                groups = _cluster(P)
                if len(groups) >= 1:
                    g = max(groups, key=lambda a: len(a))
                    fit = _fit_shapes(g)
                    shape, margin = _classify(fit)
                    mem["cue_shape"] = shape
                    mem["cue_info"] = "step %d n=%d res(mm) s=%.2f y=%.2f c=%.2f margin=%.2f" % (
                        int(fr.step), len(g), fit["sphere"][0] * 1000, fit["cylinder"][0] * 1000,
                        fit["cube"][0] * 1000, margin)
                    mem["phase"] = "cue"
                    _append(mem, "cue %s@%d" % (shape, int(fr.step)))
            elif n == 0 and mem["phase"] == "cue":
                mem["phase"] = "delay"
                _append(mem, "delay@%d" % int(fr.step))
            elif n >= 3:
                P = _frame_points(fr, True)
                groups = sorted(_cluster(P), key=lambda a: -len(a))[:3]
                fits = [_fit_shapes(g) for g in groups]
                shapes = _assign(fits)
                centers = [[fits[i][shapes[i]][1], fits[i][shapes[i]][2], HALF] for i in range(len(fits))]
                mem["cand_centers"] = centers
                mem["cand_shapes"] = shapes
                mem["appear_step"] = int(fr.step)
                want = mem["cue_shape"]
                idx = [i for i in range(len(shapes)) if shapes[i] == want]
                if not idx:
                    # cue never classified: fall back to the candidate whose own best fit is weakest-ambiguous
                    idx = [0]
                    _append(mem, "NO CUE MATCH")
                i = idx[0]
                mem["target"] = centers[i]
                mem["target_info"] = "|".join(
                    "%s(%.3f,%.3f) s%.2f y%.2f c%.2f" % (shapes[j], centers[j][0], centers[j][1],
                                                      fits[j]["sphere"][0] * 1000, fits[j]["cylinder"][0] * 1000,
                                                      fits[j]["cube"][0] * 1000) for j in range(len(fits)))
                mem["phase"] = "reach"
                mem["stage_count"] = 0
                _append(mem, "appear@%d target %s" % (int(fr.step), want))

    phase = mem["phase"]
    if phase in ("wait", "cue", "delay"):
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"gripper": 1.0}}}

    tx, ty, tz = mem["target"][0], mem["target"][1], mem["target"][2]
    tcp = cur.tcp
    k = mem["stage_count"]
    if phase == "reach":
        mem["phase"] = "hold"
        mem["stage_count"] = 0
        _append(mem, "reach@%d" % mem["n_steps"])
        return {"memory": mem, "act": {"tool": "act_reach", "args": {
            "x": tx, "y": ty, "z": Z_STRADDLE, "gripper": 1.0, "max_steps": 14}}}
    if phase == "hold":
        err = math.sqrt((tcp[0] - tx) ** 2 + (tcp[1] - ty) ** 2 + (tcp[2] - Z_STRADDLE) ** 2)
        _append(mem, "hold@%d e=%.3f" % (mem["n_steps"], err))
        mem["phase"] = "refine"
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 3, "gripper": 1.0}}}
    if phase == "refine":
        # still no success: re-measure the target near its remembered position and re-approach, then hold again
        mem["stage_count"] = k + 1
        P = _frame_points(cur, True)
        groups = _cluster(P)
        near = [g for g in groups if math.sqrt((np.median(g[:, 0]) - tx) ** 2 + (np.median(g[:, 1]) - ty) ** 2) < 0.05]
        if near:
            g = max(near, key=lambda a: len(a))
            fit = _fit_shapes(g)
            s = mem["cue_shape"] if mem["cue_shape"] in fit else _classify(fit)[0]
            nx, ny = fit[s][1], fit[s][2]
            _append(mem, "refine@%d n=%d (%.3f,%.3f)->(%.3f,%.3f)" % (mem["n_steps"], len(g), tx, ty, nx, ny))
            mem["target"] = [nx, ny, HALF]
            tx, ty = nx, ny
        else:
            _append(mem, "refine@%d no-cluster" % mem["n_steps"])
        mem["phase"] = "hold"
        return {"memory": mem, "act": {"tool": "act_reach", "args": {
            "x": tx, "y": ty, "z": Z_STRADDLE, "gripper": 1.0, "max_steps": 6, "gain": 1.0}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 1, "gripper": 1.0}}}


def act_wait(robot, gripper=1.0):
    """One native step with zero arm motion; gripper is the absolute finger target (+1 open, -1 closed)."""
    f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))
    return {"step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp[:3]]}


def act_reach(robot, x, y, z, gripper=1.0, max_steps=14, gain=1.3, deadband=0.018, still=0.12):
    """Closed-loop TCP move to world (x, y, z) keeping orientation (zero rotation commands).
    Travels at Z_SAFE until the xy error is below XY_DESCEND, then descends to z (never below 0.015).
    Inside `deadband` (3D error, m) it sends zero motion so the arm brakes and settles; stops once the
    finite-difference joint speed is below `still` rad/s there, or after max_steps native steps."""
    z = max(float(z), 0.015)
    trace = []
    prevq = robot.now().joints.copy()
    reached = False
    for i in range(int(max_steps)):
        p = robot.now().tcp
        exy = math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)
        e3 = math.sqrt(exy ** 2 + (z - p[2]) ** 2)
        # fingers open along world y: y misalignment risks landing a finger on the object, x is lenient
        aligned = abs(y - p[1]) < XY_DESCEND and abs(x - p[0]) < 2.0 * XY_DESCEND
        if e3 < deadband:
            cmd = [0.0, 0.0, 0.0]
        else:
            zg = z
            if (not aligned) and z < Z_SAFE:
                # off-target: descend only to the travel height (or, when within 3 cm in xy, to Z_NEAR where the
                # fingertips stay above the object tops); below that, hold the current height (never climb)
                zcap = Z_NEAR if exy < 0.03 else Z_SAFE
                zg = max(z, min(zcap, float(p[2])))
            e = [x - p[0], y - p[1], zg - p[2]]
            cmd = [max(-1.0, min(1.0, gain * v / 0.1)) for v in e]
        f = robot.step(cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, float(gripper))
        q = f.tcp
        qv = float(np.abs(f.joints - prevq).max()) / 0.05
        prevq = f.joints.copy()
        err = math.sqrt((x - q[0]) ** 2 + (y - q[1]) ** 2 + (z - q[2]) ** 2)
        trace.append([int(f.step), round(float(q[0]), 4), round(float(q[1]), 4), round(float(q[2]), 4),
                      round(err, 4), round(qv, 3)])
        if err < deadband and qv < still:
            reached = True
            break
    p = robot.now()
    return {"reached": reached, "final_tcp": [round(float(v), 4) for v in p.tcp],
            "fingers": [round(float(v), 4) for v in p.fingers], "trace": trace[-8:]}


def act_hold(robot, steps=3, gripper=1.0):
    """Hold the arm still (zero motion commands) for a few native steps; returns measured TCP/joint motion."""
    out = []
    prev = robot.now().joints.copy()
    for i in range(int(steps)):
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))
        dq = float(np.abs(f.joints - prev).max()) / 0.05
        prev = f.joints.copy()
        out.append([int(f.step), round(dq, 3), round(float(f.tcp[2]), 4)])
    return {"hold": out, "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_gripper(robot, gripper=-1.0, steps=3):
    """Hold the arm still and command the gripper target for a few native steps; returns finger positions."""
    out = []
    for i in range(int(steps)):
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))
        out.append([int(f.step), round(float(f.fingers[0]), 4), round(float(f.fingers[1]), 4)])
    return {"fingers": out, "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}
