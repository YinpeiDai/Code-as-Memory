"""RememberShape9-Long policy: remember the cue shape shown at the start, wait
for the nine-object arc, pick the object of the same shape and hover the TCP
just above its reference point until the episode reports success."""

ARC_MIN_OBJECTS = 6          # >= this many objects => the arc is visible
ARC_FRONT_BLUE = 200         # front-camera blue pixels that indicate the arc
VIEW_POSE = (-0.10, 0.0, 0.36)   # wrist-camera viewing pose while waiting
TOUCH_ABOVE_CENTRE = 0.030   # final TCP height above the object's centre (m)
HOVER_ABOVE = 0.06           # hover height above the final touch height (m)
MAX_CANDIDATES = 4           # ranked fallback targets kept in memory
HOLD_STEPS = 8               # static hold at the touch point before trying the next candidate
GRIP_OPEN = 1.0


def _front_blue(frame):
    return int(_blue_mask(frame.front_rgb).sum())


def _cue_from_frames(frames):
    """Find the single cue object near the table centre; returns features or None."""
    cand = [frames[0], frames[-1]] if len(frames) > 1 else [frames[0]]
    for fr in cand:
        if _front_blue(fr) > ARC_FRONT_BLUE:
            continue
        objs = _scene_objects([fr])
        near = [o for o in objs if abs(o["x"]) < 0.1 and abs(o["y"]) < 0.1]
        if len(near) == 1 and len(objs) <= 2:
            return near[0], int(fr.step)
    return None, None


def _feat_list(f):
    keys = sorted(SHAPE_SCALE) + ["ext_major", "ext_minor", "x", "y"]
    return [float(f[k]) for k in keys]


def _feat_dict(lst):
    keys = sorted(SHAPE_SCALE) + ["ext_major", "ext_minor", "x", "y"]
    return {k: v for k, v in zip(keys, lst)}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: record cue -> wait for arc -> select target -> touch."""
    m = dict(memory)
    for k in ("stage", "cue_class", "cue_feats", "cue_step", "arc_step", "target",
              "candidates", "target_info", "touch_tries", "notes"):
        m.setdefault(k, None)
    if m["stage"] is None:
        m["stage"] = "cue"
        m["touch_tries"] = 0
        m["notes"] = ""
    cur = obs.current
    if m["cue_feats"] is None:
        cue, step = _cue_from_frames(list(obs.frames))
        if cue is not None:
            cls, costs = _classify_cue(cue)
            m["cue_feats"] = _feat_list(cue)
            m["cue_class"] = cls
            m["cue_step"] = step
            m["notes"] = (m["notes"] or "") + "cue %s cost %.2f at step %d; " % (cls, costs[cls], step)
    arc_visible = _front_blue(cur) > ARC_FRONT_BLUE
    if m["target"] is None and not arc_visible:
        m["stage"] = "wait"
        return {"memory": m, "act": {"tool": "act_wait_for_arc",
                                     "args": {"view": list(VIEW_POSE), "max_steps": 640}}}
    if m["target"] is None:
        objs = _scene_objects([cur])
        m["arc_step"] = int(cur.step)
        if m["cue_feats"] is None or len(objs) == 0:
            m["notes"] += "no cue/objects at arc; "
            m["stage"] = "observe"
            return {"memory": m, "act": {"tool": "act_hold", "args": {"steps": 1}}}
        cue = _feat_dict(m["cue_feats"])
        i, info = _select_target(cue, objs)
        order, deltas = _rank_candidates(cue, objs, i)
        cands = []
        for rank, k in enumerate(order[:MAX_CANDIDATES]):
            o = objs[k]
            # reference point: same shape and yaw as the cue, whose reference sits at the table origin
            cdx = cue["x"] if (rank > 0 or info["how"] == "assigned") else 0.0
            cdy = cue["y"] if (rank > 0 or info["how"] == "assigned") else 0.0
            cands.append([round(o["x"] - cdx, 4), round(o["y"] - cdy, 4), round(0.5 * o["hmax"], 4),
                          round(cdx, 4), round(cdy, 4)])
        m["candidates"] = cands
        m["target"] = cands[0]
        m["target_info"] = "%s via %s cost %s; n=%d; labels=%s; fallback deltas=%s" % (
            info["cue_class"], info["how"], info["cost"], len(objs),
            ",".join(str(l) for l in info["labels"]), deltas[1:MAX_CANDIDATES])
        m["stage"] = "touch"
    tries = int(m["touch_tries"] or 0)
    ncand = len(m["candidates"])
    cand_idx = tries % ncand          # visit each ranked candidate once, then repeat lower
    sub = tries // ncand
    m["target"] = m["candidates"][cand_idx]
    tx, ty, tz, cdx, cdy = m["target"]
    m["touch_tries"] = tries + 1
    dz = max(0.012, TOUCH_ABOVE_CENTRE - 0.012 * sub)
    return {"memory": m, "act": {"tool": "act_touch",
                                 "args": {"x": tx, "y": ty, "z_centre": tz, "above": dz,
                                          "hover": HOVER_ABOVE, "refine": sub == 0,
                                          "cue_dx": cdx, "cue_dy": cdy, "hold_steps": HOLD_STEPS}}}


def _step_towards(robot, goal, gain=1.0, grip=GRIP_OPEN):
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    e = np.asarray(goal, dtype=np.float64) - p
    a = np.clip(e * gain / 0.1, -1.0, 1.0)
    robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, grip)
    return e


def _move_to(robot, goal, tol=0.006, max_steps=40, settle=0.004):
    prev = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    n = 0
    for n in range(1, max_steps + 1):
        _step_towards(robot, goal)
        p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
        speed = float(np.linalg.norm(p - prev))
        prev = p
        if np.linalg.norm(np.asarray(goal) - p) < tol and speed < settle:
            break
    p = np.asarray(robot.now().tcp, dtype=np.float64)[:3]
    return n, [round(float(v), 4) for v in p]


def act_wait_for_arc(robot, view=None, max_steps=640):
    """Optionally move the wrist camera to `view` (TCP xyz), then hold still until the
    object arc becomes visible in the front camera. Returns measured motion and timing."""
    moved = None
    start = [round(float(v), 4) for v in robot.now().tcp[:3]]
    n = 0
    for n in range(1, max_steps + 1):
        fr = robot.now()
        if _front_blue(fr) > ARC_FRONT_BLUE:
            break
        if view is not None:
            e = _step_towards(robot, view)
            if moved is None and np.linalg.norm(e) < 0.005:
                moved = int(fr.step)
        else:
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP_OPEN)
    fr = robot.now()
    return {"steps": n, "start_tcp": start, "tcp": [round(float(v), 4) for v in fr.tcp[:3]],
            "view_reached_step": moved, "arc_step": int(fr.step),
            "front_blue": _front_blue(fr)}


def act_hold(robot, steps=1):
    """Hold the current pose for `steps` control steps."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP_OPEN)
    return {"tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_touch(robot, x, y, z_centre, above=0.03, hover=0.06, refine=True, cue_dx=0.0, cue_dy=0.0,
              hold_steps=12):
    """Move above (x, y), optionally re-measure the target from the wrist camera, descend to
    z_centre + above and hold. The episode ends on native success; otherwise returns
    the measured final pose."""
    z_touch = z_centre + above
    z_hover = max(z_touch + hover, 0.10)
    n1, p1 = _move_to(robot, [x, y, z_hover], tol=0.008, max_steps=40, settle=0.006)
    refined = None
    if refine:
        objs = _scene_objects([robot.now()], ("wrist",))
        best = None
        for o in objs:
            d = math.hypot(o["x"] - cue_dx - x, o["y"] - cue_dy - y)
            if d < 0.025 and (best is None or d < best[0]):
                best = (d, o)
        if best is not None:
            o = best[1]
            x = o["x"] - cue_dx
            y = o["y"] - cue_dy
            refined = [round(x, 4), round(y, 4), round(best[0], 4), round(o["hmax"], 4)]
    n2, p2 = _move_to(robot, [x, y, z_touch + 0.02], tol=0.006, max_steps=25, settle=0.006)
    n3, p3 = _move_to(robot, [x, y, z_touch], tol=0.004, max_steps=25, settle=0.003)
    for _ in range(int(hold_steps)):
        _step_towards(robot, [x, y, z_touch], gain=0.5)
    p = robot.now().tcp
    return {"hover_steps": n1, "hover_tcp": p1, "refined": refined, "descend_steps": [n2, n3],
            "z_touch": round(z_touch, 4), "final_tcp": [round(float(v), 4) for v in p[:3]],
            "rpy": [round(float(v), 3) for v in p[3:6]]}
