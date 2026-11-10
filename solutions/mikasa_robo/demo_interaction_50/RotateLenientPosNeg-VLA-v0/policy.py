"""RotateLenientPosNeg policy: rotate the blue peg by the instructed angle by
pushing one end sideways with the closed gripper, closed-loop on the peg yaw
measured from fused front+wrist RGB-D, then hold still so the arm settles."""

TOL_IN_DEG = 1.2     # stop pushing when |yaw error| <= this
TOL_OUT_DEG = 3.0    # resume / re-plan when |yaw error| exceeds this
DEPTH_GAIN = 0.20    # m of push lead per rad of yaw error (peg pivots ~0.19 m from the contact, R004)
DEPTH_MAX = 0.040   # cap on push lead (m)


BASE_XY = (-0.615, 0.0)   # robot base (FK of the supplied Panda model vs measured TCP, R030)
REACH_MAX = 0.73          # pusher radius from the base kept below this (R030 stalled at 0.78)
REACH_MIN = 0.40


def _reach_pen(p):
    r = math.hypot(p[0] - BASE_XY[0], p[1] - BASE_XY[1])
    return max(0.0, r - REACH_MAX) + max(0.0, REACH_MIN - r)


def _plan(peg, goal_rad, tcp, outward=0.0, forbid=None):
    """Choose the pushed end (sigma) and the pre-push TCP xy beside it.

    Score: smaller x of the pre-push point (front camera at +x sees past the hand), a small
    distance term, and a strong penalty when the start or the predicted end of the push path
    (peg pivots ~0.2 m from the contact) leaves the comfortable reach annulus."""
    yaw = peg["yaw"]
    a = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-a[1], a[0]])
    s = 1.0 if goal_rad >= 0 else -1.0
    c = np.array([peg["cx"], peg["cy"]])
    best = None
    for sigma in (1.0, -1.0):
        d = sigma * s * n
        p = c + sigma * a * LEVER - d * (PEG_WID / 2 + R_PUSHER + GAP + outward)
        q = p + d * (GAP + 0.20 * abs(goal_rad) + 0.02)
        dist = float(np.linalg.norm(p - np.asarray(tcp[:2])))
        score = float(p[0]) + 0.1 * dist + 5.0 * (_reach_pen(p) + _reach_pen(q))
        if forbid is not None and sigma == forbid:
            score += 100.0
        if best is None or score < best[0]:
            best = (score, sigma, p)
    return best[1], s, best[2]


def _clamp_reach(xy):
    v = np.array([xy[0] - BASE_XY[0], xy[1] - BASE_XY[1]])
    r = float(np.linalg.norm(v))
    lim = REACH_MAX + 0.03
    if r > lim:
        v = v * (lim / r)
    return np.array([BASE_XY[0] + v[0], BASE_XY[1] + v[1]])


def _valid(est, prev_yaw):
    if not est.get("ok"):
        return False
    if est["ext_n"] < 0.035 or est["ext_n"] > 0.08 or est["ext_a"] < 0.08:
        return False
    if prev_yaw is not None and abs(est["yaw"] - prev_yaw) > 0.3:
        return False
    return True


def _centre(est, sigma, tcp=None):
    """Peg centre. If the visible length is short (occlusion), anchor on one visible end:
    the candidate whose pushed-contact point (c + sigma*a*LEVER) lies closest along the axis
    to the TCP (which tracks that contact) wins; without a TCP, anchor on the far end."""
    yaw = est["yaw"]
    a = np.array([math.cos(yaw), math.sin(yaw)])
    if est["ext_a"] >= 0.225:
        return np.array([est["cx"], est["cy"]])
    n = np.array([-a[1], a[0]])
    m = np.array([est["mx"], est["my"]])
    lat = float(np.dot(np.array([est["cx"], est["cy"]]) - m, n))
    if sigma > 0:
        far_anchor, near_anchor = est["lo"] + PEG_LEN / 2, est["hi"] - PEG_LEN / 2
    else:
        far_anchor, near_anchor = est["hi"] - PEG_LEN / 2, est["lo"] + PEG_LEN / 2
    along = far_anchor
    if tcp is not None:
        t = float(np.dot(np.asarray(tcp[:2], dtype=np.float64) - m, a))
        e_far = abs(far_anchor + sigma * LEVER - t)
        e_near = abs(near_anchor + sigma * LEVER - t)
        if e_near < e_far:
            along = near_anchor
    return m + a * along + n * lat


def perceive_decide(obs, goal, memory, last):
    """Track stage, initial peg yaw and the latest yaw; choose the next act."""
    mem = dict(memory)
    f = obs.current
    tcp = np.asarray(f.tcp, dtype=np.float64)
    stage = mem.get("stage") or "start"
    if mem.get("decisions") is None:
        mem["decisions"] = 0
    mem["decisions"] = int(mem["decisions"]) + 1
    if mem.get("retries") is None:
        mem["retries"] = 0
    if mem.get("notes") is None:
        mem["notes"] = ""

    if stage == "start":
        g = _parse_goal_deg(goal)
        if g is None:
            g = 0.0
        est = _peg_estimate(f, None)
        if not est.get("ok"):
            mem.update({"stage": "start", "goal_deg": g})
            return {"memory": _mem_out(mem), "act": {"tool": "act_wait", "args": {"n": 1}}}
        sigma, s, p = _plan(est, math.radians(g), tcp)
        mem.update({"stage": "approach", "goal_deg": g, "yaw0": est["yaw"],
                    "yaw_now": est["yaw"], "peg0": [est["cx"], est["cy"]],
                    "sigma": sigma, "push_sign": s, "start_xy": [float(p[0]), float(p[1])],
                    "yaw_keep": float(tcp[5]),
                    "notes": "init peg yaw %.1f deg ext %.3f/%.3f goal %.1f sigma %d" % (
                        math.degrees(est["yaw"]), est["ext_a"], est["ext_n"], g, sigma)})
        return {"memory": _mem_out(mem), "act": {"tool": "act_approach", "args": {
            "sx": float(p[0]), "sy": float(p[1]), "yaw_keep": float(tcp[5])}}}

    goal_deg = float(mem["goal_deg"])
    yaw0 = float(mem["yaw0"])
    # track the yaw through every frame since the last decision (continuity check per step)
    frames = list(obs.frames)
    stride = max(1, len(frames) // 12)
    idx = list(range(0, len(frames) - 1, stride)) + [len(frames) - 1]
    est = {"ok": False}
    for k in idx:
        e = _peg_estimate(frames[k], float(mem["yaw_now"]))
        if _valid(e, float(mem["yaw_now"])):
            mem["yaw_now"] = e["yaw"]
        if k == idx[-1]:
            est = e
    err_deg = goal_deg - math.degrees(float(mem["yaw_now"]) - yaw0)
    sx, sy = mem["start_xy"]

    if stage == "approach":
        dxy = math.hypot(tcp[0] - sx, tcp[1] - sy)
        if dxy < 0.015 and tcp[2] < Z_PUSH + 0.015:
            stage = "push"
        elif int(mem["retries"]) < 2:
            mem["retries"] = int(mem["retries"]) + 1
            sigma, s, p = _plan({"yaw": float(mem["yaw_now"]), "cx": est.get("cx", mem["peg0"][0]),
                                 "cy": est.get("cy", mem["peg0"][1])},
                                math.radians(err_deg), tcp, outward=0.015 * int(mem["retries"]))
            mem.update({"sigma": sigma, "push_sign": s, "start_xy": [float(p[0]), float(p[1])]})
            mem["notes"] = str(mem["notes"]) + "; approach retry %d" % int(mem["retries"])
            return {"memory": _mem_out(mem), "act": {"tool": "act_approach", "args": {
                "sx": float(p[0]), "sy": float(p[1]), "yaw_keep": float(mem["yaw_keep"])}}}
        else:
            stage = "push"

    # push stalled (reach / IK limit): lift and push the other end the opposite way
    if stage == "push" and last is not None and '"stalled"' in str(last):
        mem.update({"stage": "lifted_switch", "last_act": "act_lift"})
        mem["notes"] = str(mem["notes"]) + "; push stalled -> switch end"
        return {"memory": _mem_out(mem), "act": {"tool": "act_lift", "args": {"yaw_keep": float(mem["yaw_keep"])}}}
    if stage == "lifted_switch":
        est2 = est if _valid(est, None) else {"cx": mem["peg0"][0], "cy": mem["peg0"][1]}
        sigma, s, p = _plan({"yaw": float(mem["yaw_now"]), "cx": est2["cx"], "cy": est2["cy"]},
                            math.radians(err_deg), tcp, forbid=float(mem["sigma"]))
        mem.update({"stage": "approach", "sigma": sigma, "push_sign": s,
                    "start_xy": [float(p[0]), float(p[1])], "last_act": "act_approach"})
        mem["notes"] = str(mem["notes"]) + "; replan other end err %.1f" % err_deg
        return {"memory": _mem_out(mem), "act": {"tool": "act_approach", "args": {
            "sx": float(p[0]), "sy": float(p[1]), "yaw_keep": float(mem["yaw_keep"])}}}
    # push / hold stages: if the peg cannot be measured, lift to look again
    if stage in ("push", "hold") and not _valid(est, float(mem["yaw_now"])) \
            and mem.get("last_act") in ("act_push", "act_hold"):
        mem.update({"stage": "lifted", "last_act": "act_lift"})
        mem["notes"] = str(mem["notes"]) + "; peg lost -> lift"
        return {"memory": _mem_out(mem), "act": {"tool": "act_lift", "args": {"yaw_keep": float(mem["yaw_keep"])}}}
    if stage == "lifted":
        est2 = est if _valid(est, None) else {"cx": mem["peg0"][0], "cy": mem["peg0"][1]}
        sigma, s, p = _plan({"yaw": float(mem["yaw_now"]), "cx": est2["cx"], "cy": est2["cy"]},
                            math.radians(err_deg), tcp)
        mem.update({"stage": "approach", "sigma": sigma, "push_sign": s,
                    "start_xy": [float(p[0]), float(p[1])], "last_act": "act_approach"})
        mem["notes"] = str(mem["notes"]) + "; replan after lift err %.1f" % err_deg
        return {"memory": _mem_out(mem), "act": {"tool": "act_approach", "args": {
            "sx": float(p[0]), "sy": float(p[1]), "yaw_keep": float(mem["yaw_keep"])}}}
    # push / hold stages: decide from the measured yaw error
    if abs(err_deg) > TOL_OUT_DEG and err_deg * float(mem["push_sign"]) < 0 and stage in ("push", "hold") \
            and mem.get("last_act") in ("act_push", "act_hold"):
        # overshoot: push back from the other side
        est2 = est if est.get("ok") else {"cx": mem["peg0"][0], "cy": mem["peg0"][1]}
        sigma, s, p = _plan({"yaw": float(mem["yaw_now"]), "cx": est2["cx"], "cy": est2["cy"]},
                            math.radians(err_deg), tcp)
        mem.update({"stage": "approach", "sigma": sigma, "push_sign": s,
                    "start_xy": [float(p[0]), float(p[1])], "last_act": "act_approach"})
        mem["notes"] = str(mem["notes"]) + "; overshoot %.1f -> replan" % err_deg
        return {"memory": _mem_out(mem), "act": {"tool": "act_approach", "args": {
            "sx": float(p[0]), "sy": float(p[1]), "yaw_keep": float(mem["yaw_keep"])}}}
    if abs(err_deg) <= TOL_OUT_DEG and mem.get("last_act") in ("act_push", "act_hold"):
        mem.update({"stage": "hold", "last_act": "act_hold"})
        return {"memory": _mem_out(mem), "act": {"tool": "act_hold", "args": {"n": 8}}}
    mem.update({"stage": "push", "last_act": "act_push"})
    return {"memory": _mem_out(mem), "act": {"tool": "act_push", "args": {
        "yaw0": yaw0, "goal_deg": goal_deg, "sigma": float(mem["sigma"]),
        "sign": float(mem["push_sign"]), "yaw_prev": float(mem["yaw_now"]),
        "yaw_keep": float(mem["yaw_keep"])}}}


def _mem_out(mem):
    keys = ["stage", "goal_deg", "yaw0", "yaw_now", "peg0", "sigma", "push_sign",
            "start_xy", "yaw_keep", "retries", "decisions", "last_act", "notes"]
    return {k: mem.get(k) for k in keys}


def act_wait(robot, n=1):
    """Hold still (zero motion, gripper closed) for n native steps."""
    for _ in range(int(n)):
        robot.step(0, 0, 0, 0, 0, 0, -1.0)
    return {"waited": int(n), "step": int(robot.now().step)}


def act_approach(robot, sx, sy, yaw_keep, max_steps=30):
    """Close the gripper, travel at Z_SAFE to (sx, sy), then descend to Z_PUSH.
    Returns the final TCP, steps used and whether the descent stalled."""
    log = []
    stall = 0
    status = "max_steps"
    for k in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        dxy = math.hypot(tcp[0] - sx, tcp[1] - sy)
        if dxy > 0.012:
            if tcp[2] < Z_SAFE - 0.015 and dxy > 0.03:
                tgt = [tcp[0], tcp[1], Z_SAFE]          # lift straight up first
            else:
                tgt = [sx, sy, Z_SAFE]
        else:
            tgt = [sx, sy, Z_PUSH]
        u, e = _trans_cmd(tcp, tgt)
        r, _ = _rot_cmd(tcp, yaw_keep)
        if dxy <= 0.012 and abs(tcp[2] - Z_PUSH) < 0.006:
            status = "arrived"
            break
        z_before = tcp[2]
        robot.step(u[0], u[1], u[2], r[0], r[1], r[2], -1.0)
        tcp2 = np.asarray(robot.now().tcp, dtype=np.float64)
        log.append([int(robot.now().step), round(float(tcp2[0]), 3), round(float(tcp2[1]), 3), round(float(tcp2[2]), 3)])
        if dxy <= 0.012 and tcp2[2] > Z_PUSH + 0.015 and z_before - tcp2[2] < 0.002 and u[2] < -0.1:
            stall += 1
            if stall >= 3:
                status = "stalled"
                break
        else:
            stall = 0
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return {"status": status, "tcp": [round(float(v), 4) for v in tcp[:3]],
            "step": int(robot.now().step), "trace": log[-12:]}


def act_push(robot, yaw0, goal_deg, sigma, sign, yaw_prev, yaw_keep, max_steps=25,
             hold_max=10):
    """Push the sigma end of the peg along sign*normal until the measured yaw change
    reaches goal_deg (within TOL_IN_DEG), then hold zero motion. Returns a trace."""
    target = yaw0 + math.radians(goal_deg)
    yaw = yaw_prev
    c = None
    mode = "push"
    hold = 0
    bad = 0
    run_bad = 0
    stuck = 0
    last_tcp = None
    trace = []
    status = "max_steps"
    for k in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=np.float64)
        est = _peg_estimate(f, yaw)
        if _valid(est, yaw):
            yaw = est["yaw"]
            c = _centre(est, sigma, tcp)
            run_bad = 0
        else:
            bad += 1
            run_bad += 1
            if run_bad >= 3:
                status = "lost"
                break
        err = target - yaw
        ed = math.degrees(err)
        if mode == "push" and abs(ed) <= TOL_IN_DEG:
            mode = "hold"
        if mode == "hold" and abs(ed) > TOL_OUT_DEG:
            if err * sign > 0:
                mode = "push"
            else:
                status = "overshoot"
                break
        if mode == "push" and c is not None:
            a = np.array([math.cos(yaw), math.sin(yaw)])
            n = np.array([-a[1], a[0]])
            d = sigma * sign * n
            pe = c + sigma * a * LEVER
            depth = min(DEPTH_MAX, max(0.004, DEPTH_GAIN * abs(err)))
            txy = _clamp_reach(pe - d * (PEG_WID / 2 + R_PUSHER) + d * depth)
            u, _ = _trans_cmd(tcp, [txy[0], txy[1], Z_PUSH])
            if last_tcp is not None and float(np.linalg.norm(u[:2])) > 0.15 \
                    and float(np.linalg.norm(tcp[:2] - last_tcp[:2])) < 0.002:
                stuck += 1
            else:
                stuck = 0
            if stuck >= 3:
                status = "stalled"
                break
            r, _ = _rot_cmd(tcp, yaw_keep)
            hold = 0
        else:
            u = np.zeros(3)
            r = np.zeros(3)
            hold += 1
        q = _qvel(robot)
        trace.append([int(f.step), round(ed, 2), mode[0], round(float(tcp[0]), 3), round(float(tcp[1]), 3),
                      round(float(tcp[2]), 3), None if q is None else round(q, 2)])
        if mode == "hold" and hold > hold_max:
            status = "held"
            break
        last_tcp = tcp
        robot.step(u[0], u[1], u[2], r[0], r[1], r[2], -1.0)
    return {"status": status, "err_deg": round(math.degrees(target - yaw), 2), "yaw": yaw,
            "bad_frames": bad, "step": int(robot.now().step), "trace": trace[-20:]}


def act_hold(robot, n=8):
    """Command zero motion (gripper closed) for n steps so the arm comes to rest."""
    qs = []
    for _ in range(int(n)):
        robot.step(0, 0, 0, 0, 0, 0, -1.0)
        q = _qvel(robot)
        qs.append(None if q is None else round(q, 3))
    return {"held": int(n), "qvel": qs, "step": int(robot.now().step)}


def act_lift(robot, yaw_keep, z=0.16, max_steps=8):
    """Move straight up to height z (gripper closed) so the cameras can see the peg again."""
    for _ in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        if tcp[2] > z - 0.01:
            break
        u, _ = _trans_cmd(tcp, [tcp[0], tcp[1], z])
        r, _ = _rot_cmd(tcp, yaw_keep)
        robot.step(u[0], u[1], u[2], r[0], r[1], r[2], -1.0)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return {"tcp": [round(float(v), 4) for v in tcp[:3]], "step": int(robot.now().step)}
