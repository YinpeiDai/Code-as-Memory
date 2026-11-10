"""RememberShapeAndColor policy: remember the cue (colour + shape), wait for the 15 candidates,
pick the matching one by colour and one-to-one shape assignment, and touch it from above with
closed fingers."""

CANDIDATE_PIXELS = 1500   # coloured pixels (wrist+front) above this => candidate phase
CUE_MAX_PIXELS = 1200     # a single cue object gives ~150-400 coloured pixels
T_TIP_OFFSET = 0.0256     # T actor origin (stem tip) lies 2.56 cm from footprint centroid (cue frames)
T_PRESS_FROM_TIP = 0.012  # touch point on the stem, between tip (actor origin) and centroid
FIELDS = ["stage", "cue_color", "cue_shape", "cue_conf", "cue_votes", "cue_frames", "cue_first_step",
          "empty_frames", "cand_step", "n_candidates", "target", "target_origin", "target_top",
          "target_info", "touch_attempts", "last_outcome", "decisions"]


def _mem_defaults():
    return {"stage": "cue", "cue_color": "", "cue_shape": "", "cue_conf": 0.0, "cue_votes": "",
            "cue_frames": 0, "cue_first_step": -1, "empty_frames": 0, "cand_step": -1,
            "n_candidates": 0, "target": [], "target_origin": [], "target_top": 0.0,
            "target_info": "", "touch_attempts": 0, "last_outcome": "", "decisions": 0}


def _parse_votes(text):
    votes = {}
    if text:
        for item in text.split(";"):
            if ":" in item:
                k, v = item.split(":")
                votes[k] = int(v)
    return votes


def _fmt_votes(votes):
    return ";".join("%s:%d" % (k, v) for k, v in sorted(votes.items()))


def _press_point(obj, toward):
    """Where to put the closed fingertips on the chosen object (world x, y)."""
    ox, oy = obj["x"], obj["y"]
    origin = [ox, oy]
    px, py = ox, oy
    if obj["shape"] == "T":
        th = obj.get("t_theta")
        ax = [math.cos(th), math.sin(th)] if th is not None else [0.0, 1.0]
        tip = obj.get("t_tip") or [ox - ax[0] * T_TIP_OFFSET, oy - ax[1] * T_TIP_OFFSET]
        origin = [tip[0], tip[1]]
        px, py = tip[0] + ax[0] * T_PRESS_FROM_TIP, tip[1] + ax[1] * T_PRESS_FROM_TIP
    return [px, py], origin


def _choose_target(objs, color, shape):
    same = [o for o in objs if o["color"] == color]
    hit = [o for o in same if o["shape"] == shape]
    if hit:
        return hit[0], "assigned"
    if same:
        best = min(same, key=lambda o: _shape_cost(o, shape) if shape in SHAPES else 0.0)
        return best, "nearest_cost"
    if objs:
        return min(objs, key=lambda o: _shape_cost(o, shape) if shape in SHAPES else 0.0), "any_color"
    return None, "none"


def perceive_decide(obs, goal, memory, last):
    """Track cue -> delay -> candidates; record the cue (colour, shape) while visible, then
    choose the matching candidate and call act_touch. Returns every declared memory field."""
    m = _mem_defaults()
    for k in FIELDS:
        if k in memory and memory[k] is not None:
            m[k] = memory[k]
    m["decisions"] = int(m["decisions"]) + 1
    if last is not None:
        m["last_outcome"] = str(last)[:600]
    frame = obs.current
    step = int(frame.step)

    if m["stage"] in ("cue", "delay"):
        npx = _scene_counts(frame)
        if npx >= CANDIDATE_PIXELS:
            m["stage"] = "plan"
            m["cand_step"] = step
        elif npx > 40 and npx <= CUE_MAX_PIXELS:
            cue = _cue_object(frame)
            if cue is not None:
                votes = _parse_votes(m["cue_votes"])
                key = cue["color"] + "/" + cue["shape"]
                votes[key] = votes.get(key, 0) + 1
                m["cue_votes"] = _fmt_votes(votes)
                best = max(votes, key=votes.get)
                m["cue_color"], m["cue_shape"] = best.split("/")
                m["cue_conf"] = round(float(votes[best]) / sum(votes.values()) * min(1.0, cue["margin"] / 5.0), 3)
                m["cue_frames"] = int(m["cue_frames"]) + 1
                if int(m["cue_first_step"]) < 0:
                    m["cue_first_step"] = step
        else:
            if int(m["cue_frames"]) > 0:
                m["stage"] = "delay"
                m["empty_frames"] = int(m["empty_frames"]) + 1
        if m["stage"] != "plan":
            return {"memory": m, "act": {"tool": "act_wait", "args": {}}}

    if m["stage"] == "plan":
        objs = _scene_objects(frame)
        for o in objs:
            if o["shape"] == "T":
                o["t_theta"] = None
        target, how = _choose_target(objs, m["cue_color"], m["cue_shape"])
        m["n_candidates"] = len(objs)
        if target is None:
            m["stage"] = "cue"
            return {"memory": m, "act": {"tool": "act_wait", "args": {}}}
        if target["shape"] == "T":
            P = _frame_points(frame)
            xy, z = _cells(P[target["color"]])
            near = np.sqrt(((xy - np.array([target["x"], target["y"]])) ** 2).sum(1)) < 0.035
            if near.sum() >= 6:
                th, score, tip, jun = _fit_t_fixed(xy[near], np.array([target["x"], target["y"]]))
                if score > 1.2:
                    target["t_theta"] = th
                    target["t_tip"] = [float(tip[0]), float(tip[1])]
        tcp = frame.tcp
        press, origin = _press_point(target, [float(tcp[0]), float(tcp[1])])
        top = float(target["h"])
        m["target"] = [round(press[0], 4), round(press[1], 4), round(top, 4)]
        m["target_origin"] = [round(origin[0], 4), round(origin[1], 4)]
        m["target_top"] = round(top, 4)
        m["target_info"] = "%s %s via %s centroid=(%.3f,%.3f) area=%.1f hole=%.2f top=%.2f h=%.3f theta=%s" % (
            target["color"], target["shape"], how, target["x"], target["y"], target["area"], target["hole"],
            target["top"], target["h"], str(target.get("t_theta")))
        m["stage"] = "touch"

    if m["stage"] in ("touch", "retouch", "retouch2"):
        m["touch_attempts"] = int(m["touch_attempts"]) + 1
        tx, ty, top = m["target"]
        if m["stage"] == "touch":
            m["stage"] = "retouch"
            zt = top + 0.012
        elif m["stage"] == "retouch":
            m["stage"] = "retouch2"
            zt = top + 0.004
        else:
            ox, oy = m["target_origin"]
            tx, ty = ox, oy
            zt = top + 0.004
        return {"memory": m, "act": {"tool": "act_touch",
                                     "args": {"x": tx, "y": ty, "z": round(zt, 4), "top": top}}}
    return {"memory": m, "act": {"tool": "act_wait", "args": {}}}


def act_wait(robot):
    """One native step with a zero arm command and the gripper closing (the task zeroes
    commands during the cue/delay phase anyway). Returns the observed step."""
    f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"waited": 1, "step": int(f.step), "tcp_z": float(f.tcp[2])}


def act_touch(robot, x, y, z, top, max_steps=24, hover=0.025, slope=0.15, gain=1.5, damp=0.5, gain_z=2.0,
              gripper=-1.0, tol_xy=0.012, tol_z=0.012, hold_steps=3, slow_band=0.0, slow_cmd=1.0):
    """Open-gripper approach to the touch point (x, y, z): move inside a descent cone
    z >= top + hover + slope*(dxy - tol_xy) (never above the start height), then once within
    tol_xy / tol_z send zero arm commands for hold_steps steps so the arm comes to rest there
    (the task appears to register the touch only when the arm is static near the target).
    Returns the measured trace [step, x, y, z, dxy, phase(0 approach, 1 hold), qvel_max]."""
    trace = []
    z_start = float(robot.now().tcp[2])
    held = 0
    phase = 0
    prev_j = robot.now().joints
    outcome = "max_steps"
    for i in range(int(max_steps)):
        f = robot.now()
        p = f.tcp
        ex, ey, ez = x - float(p[0]), y - float(p[1]), z - float(p[2])
        dxy = math.hypot(ex, ey)
        if phase == 0 and dxy < tol_xy and abs(ez) < tol_z:
            phase = 1
        if phase == 1:
            if held >= int(hold_steps):
                outcome = "held_no_success"
                break
            a = [0.0, 0.0, 0.0]
            held += 1
        else:
            if dxy < tol_xy * 1.5:
                zg = z
            else:
                zg = max(z, min(z_start, top + hover + slope * (dxy - tol_xy)))
            ez = zg - float(p[2])
            vel = [0.0, 0.0, 0.0]
            fr = robot.frames()
            if len(fr) >= 2:
                vel = [float(fr[-1].tcp[j] - fr[-2].tcp[j]) for j in range(3)]
            a = [max(-1.0, min(1.0, (g * e - damp * v) / 0.1)) for g, e, v in zip((gain, gain, gain_z), (ex, ey, ez), vel)]
            if float(p[2]) < top + slow_band and a[2] < -slow_cmd:
                a[2] = -slow_cmd
        f2 = robot.step(a[0], a[1], a[2], 0.0, 0.0, 0.0, gripper)
        q = f2.tcp
        qv = float(np.abs(np.asarray(f2.joints) - np.asarray(prev_j)).max() * 20.0)
        prev_j = f2.joints
        trace.append([int(f2.step), round(float(q[0]), 4), round(float(q[1]), 4), round(float(q[2]), 4),
                      round(math.hypot(x - float(q[0]), y - float(q[1])), 4), phase, round(qv, 3)])
    return {"outcome": outcome, "trace": trace[-8:], "fingers": [round(float(v), 4) for v in robot.now().fingers]}
