"""Action tools: zero-motion waiting with scene-change detection, and the touch motion."""


def act_wait(robot, until="candidates", max_steps=590):
    """Hold still (zero arm motion, gripper open) one native step at a time until
    `until` happens: 'cue' (exactly one coloured object in front view), 'candidates'
    (>= 8 coloured objects in front view, plus one settling step), 'steps' (just wait
    max_steps) or 'end' (wait until the episode ends). Returns the counts history."""
    hist = []
    last_n = None
    n_steps = 0
    seen = False
    for i in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n_steps += 1
        f = robot.now()
        if until in ("steps", "end"):
            continue
        n = _count_objects(f, "front")
        if n != last_n:
            hist.append([int(getattr(f, "step", -1)), int(n)])
            last_n = n
        if until == "cue" and n == 1:
            return {"event": "cue", "steps": n_steps, "step": int(getattr(f, "step", -1)), "counts": hist[-6:]}
        if until == "candidates" and n >= 8:
            if seen:
                return {"event": "candidates", "steps": n_steps, "step": int(getattr(f, "step", -1)), "counts": hist[-6:]}
            seen = True
    return {"event": "max_steps", "steps": n_steps, "step": int(getattr(robot.now(), "step", -1)), "counts": hist[-6:]}


def _goto(robot, goal, tol=0.006, max_steps=40, vmax=1.0, gain=10.0, gripper=1.0, settle_v=0.004, stall_stop=False):
    """Closed-loop straight-line move of the TCP to world xyz (orientation unchanged).
    Command = gain * error (m -> normalized units, 1 unit = 0.1 m), norm-clipped to vmax.
    Stops when within tol and moving slower than settle_v m/step. With stall_stop, stops
    when the TCP has not moved 1 mm/step for 3 steps while still > 1 cm from the goal
    (blocked by contact) instead of pushing on."""
    goal = np.asarray(goal, dtype=np.float64)
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    n = 0
    slow = 0
    stalled = False
    for i in range(int(max_steps)):
        p = np.asarray(robot.now().tcp[:3], dtype=np.float64)
        err = goal - p
        v = float(np.linalg.norm(p - prev))
        prev = p
        if np.linalg.norm(err) < tol and (i > 0 and v < settle_v):
            break
        if i > 0 and v < 0.001 and np.linalg.norm(err) > 0.01:
            slow += 1
        else:
            slow = 0
        if stall_stop and slow >= 3:
            stalled = True
            break
        c = gain * err
        nc = float(np.linalg.norm(c))
        if nc > vmax:
            c = c * (vmax / nc)
        robot.step(float(c[0]), float(c[1]), float(c[2]), 0.0, 0.0, 0.0, gripper)
        n += 1
    p = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    return {"steps": n, "err": round(float(np.linalg.norm(goal - p)), 4), "tcp": [round(float(q), 4) for q in p],
            "stalled": stalled}


def _refine_target(frame, color, shape, x, y):
    """Re-measure the target in a close wrist view: nearest same-colour object within 4 cm.
    Returns (x, y, cost_for_shape, best_shape, n_pixels, touches_border, best_cost) or None."""
    objs = [o for o in _view_objects(frame, "wrist") if o["color"] == color]
    best = None
    for o in objs:
        d = math.hypot(o["x"] - x, o["y"] - y)
        if d < 0.04 and (best is None or d < best[0]):
            best = (d, o)
    if best is None:
        return None
    o = best[1]
    costs = {s: _shape_cost(o, s) for s in SHAPE_PROTO}
    bs = min(costs, key=lambda s: costs[s])
    tx, ty = o["x"], o["y"]
    if shape == "tee":
        P = _top_points_near(frame, "wrist", color, o["x"], o["y"], 0.032)
        if len(P) >= 8:
            tx, ty, th, sc = _tee_origin(P[:, :2])
    return tx, ty, float(costs[shape]), bs, int(o["pixels"]), bool(o["border"]), float(costs[bs])


def act_touch(robot, x, y, ztop, color, shape, hover=0.10, final_dz=0.015, refine=True, gripper=-1.0, hold=12, verify=True):
    """Touch the object at world (x, y) whose top is at height ztop: close the fingers
    (gripper=-1, so open fingers cannot hit neighbouring objects), move to a hover point
    `hover` m above the top, re-measure the target in the close wrist view (refine xy),
    descend vertically to ztop + final_dz (stops early if blocked) and hold still for
    `hold` steps (success seems to need the arm at rest near the object).
    The refined xy is used only if the object is fully visible (>= 100 px, off the border).
    With verify, the close view must not clearly show a different shape (cue-shape cost > 8,
    another shape cost < 3 and 6 lower, object fully visible, >= 100 px); otherwise the call returns
    verify_failed without descending so the policy can choose another object.
    Episode success ends the call. Returns measured TCP positions and refinement."""
    out = {"x": round(x, 4), "y": round(y, 4), "ztop": ztop}
    hz = ztop + hover
    t0 = robot.now().tcp
    chk = robot.dry_run(x, y, hz, float(t0[3]), float(t0[4]), float(t0[5]), mode="absolute")
    out["dry_ok"] = bool(chk.get("within_joint_limits", True))
    out["dry_margin"] = round(float(chk.get("min_joint_margin_rad", 0.0) or 0.0), 3)
    # stage A: move to hover
    out["hover"] = _goto(robot, [x, y, hz], tol=0.008, max_steps=35, gripper=gripper)
    # stage B: refine from close view
    if refine:
        ref = _refine_target(robot.now(), color, shape, x, y)
        if ref is not None:
            tx, ty, cst, bs, npx, border = ref[:6]
            out["refine"] = {"x": round(tx, 4), "y": round(ty, 4), "cost": round(cst, 2), "best": bs, "px": npx, "border": border}
            if verify and bs != shape and cst > 8.0 and npx >= 100 and not border:
                bcost = ref[6]
                if bcost < 3.0 and bcost < cst - 6.0:
                    out["verify_failed"] = True
                    out["seen_shape"] = bs
                    out["seen_xy"] = [round(ref[0], 4), round(ref[1], 4)]
                    return out
            if math.hypot(tx - x, ty - y) < 0.03 and npx >= 100 and not border:
                x, y = tx, ty
                if math.hypot(tx - out["x"], ty - out["y"]) > 0.004:
                    out["hover2"] = _goto(robot, [x, y, hz], tol=0.006, max_steps=12, gripper=gripper)
        else:
            out["refine"] = None
    # stage C: descend (slower), stop if blocked
    zf = ztop + final_dz
    out["descend"] = _goto(robot, [x, y, zf], tol=0.004, max_steps=30, vmax=0.5, gripper=gripper, stall_stop=True)
    # hold still near the object
    for i in range(int(hold)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper)
    out["final_tcp"] = [round(float(q), 4) for q in robot.now().tcp[:6]]
    out["fingers"] = [round(float(q), 4) for q in robot.now().fingers]
    out["target_used"] = [round(x, 4), round(y, 4), round(zf, 4)]
    return out
