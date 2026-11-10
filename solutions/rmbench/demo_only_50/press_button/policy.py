"""press_button policy: read the two digits once from the reset head frame, then press the left button
(left arm) N_left times, the middle button (left arm) N_right times, retract the left arm, and press the right
button (right arm) once. Geometry is taken from the assigned demonstrations (see task_analysis.md)."""

# Nominal button centres (world x, y) seen in all 50 demos (head RGB-D red tops at z=0.78; expert presses here).
_BUTTONS = {"left": [-0.15, -0.15], "middle": [0.0, -0.15], "right": [0.15, -0.15]}
_HOVER_Z = 0.964          # expert hover height of the ee frame above every button
_PRESS_Z = {"left": 0.9255, "right": 0.9206}   # expert press bottoms (left arm presses L/M, right arm presses R)
_QUAT = {"left": [0.7071068, 0.0, 0.7071068, 0.0], "right": [0.0, -0.7071068, 0.0, 0.7071068]}
_LOW_Z = 0.940            # ee z below this near a button counts as "pressed"
_HIGH_Z = 0.952           # ee z above this counts as released
_MAX_PRESSES_PER_CALL = 5
_STAGES = ["start", "goto_left", "press_left", "goto_middle", "press_middle", "retract_left",
           "goto_right", "press_right", "confirm_retry", "idle"]


def _pose(frame, arm):
    return [float(v) for v in np.asarray(getattr(frame, arm + "_ee_pose"), dtype=float).ravel()[:7]]


def _grip(frame, arm):
    return float(np.asarray(getattr(frame, arm + "_gripper"), dtype=float).ravel()[0])


def _send(robot, left7, lg, right7, rg):
    a = [float(v) for v in left7] + [float(lg)] + [float(v) for v in right7] + [float(rg)]
    return robot.step(*a)


def _move(robot, arm, target7, grip):
    """One native step moving `arm` to target7 with gripper `grip`; the other arm holds its current pose."""
    cur = robot.now()
    other = "right" if arm == "left" else "left"
    hold = _pose(cur, other)
    hg = _grip(cur, other)
    if arm == "left":
        return _send(robot, target7, grip, hold, hg)
    return _send(robot, hold, hg, target7, grip)


def _try_move(robot, arm, target7, grip, tries=2):
    errs = []
    for _ in range(tries):
        try:
            _move(robot, arm, target7, grip)
            return True, errs
        except Exception as e:  # MotionError: planning failed, arm stayed still; EpisodeEnd is not caught
            errs.append(str(e)[:120])
    return False, errs


def _count_presses(frames, arm, xy, start_low):
    """Count high->low transitions of the arm's ee z while within 3 cm of the button xy."""
    low = bool(start_low)
    n = 0
    for f in frames:
        p = getattr(f, arm + "_ee_pose", None)
        if p is None:
            continue
        p = np.asarray(p, dtype=float).ravel()
        near = math.hypot(p[0] - xy[0], p[1] - xy[1]) < 0.03
        if near and p[2] < _LOW_Z:
            if not low:
                n += 1
            low = True
        elif p[2] > _HIGH_Z:
            low = False
    return n, low


def _near(frame, arm, target3, tol=0.015):
    p = _pose(frame, arm)
    return math.sqrt(sum((p[i] - target3[i]) ** 2 for i in range(3))) < tol


def _button_layout(frame):
    """Nominal button centres unless head RGB-D shows a clearly different layout (>2 cm), in which case the
    detected red tops are used after removing the detector bias measured on the demos (+0.004 m in y)."""
    nominal = [list(_BUTTONS["left"]), list(_BUTTONS["middle"]), list(_BUTTONS["right"])]
    det = _find_buttons(frame, "head")
    if len(det) != 3:
        return nominal
    out = []
    moved = False
    for d, n in zip(det, nominal):
        c = [d[0], d[1] + 0.004]
        if math.hypot(c[0] - n[0], c[1] - n[1]) > 0.02:
            moved = True
        out.append(c)
    if not moved:
        return nominal
    # sanity: detected layout must still be a left-to-right row reachable on the robot side of the table
    if all(-0.35 < c[0] < 0.35 and -0.3 < c[1] < 0.05 for c in out):
        return out
    return nominal


def perceive_decide(obs, goal, memory, last):
    """Update episode memory from the latest frames and choose the next press-sequence action."""
    mem = dict(memory)
    for k, v in (("stage", "start"), ("left_count", None), ("middle_count", None), ("left_done", 0),
                 ("middle_done", 0), ("right_done", 0), ("retries", 0), ("arm_low", False),
                 ("left_home", None), ("right_home", None), ("reading", ""), ("history", ""),
                 ("buttons", [list(_BUTTONS["left"]), list(_BUTTONS["middle"]), list(_BUTTONS["right"])])):
        if mem.get(k) is None:
            mem[k] = v
    cur = obs.current
    stage = mem["stage"]
    B = {"left": [float(v) for v in mem["buttons"][0]], "middle": [float(v) for v in mem["buttons"][1]],
         "right": [float(v) for v in mem["buttons"][2]]}
    frames = list(obs.frames)

    # ---- interpret the outcome of the previous action from the frames it produced
    if stage in ("press_left", "press_middle", "press_right", "confirm_retry"):
        arm = "right" if stage in ("press_right", "confirm_retry") else "left"
        btn = {"press_left": "left", "press_middle": "middle"}.get(stage, "right")
        n, low = _count_presses(frames, arm, B[btn], mem["arm_low"])
        mem["arm_low"] = low
        mem[btn + "_done"] = int(mem[btn + "_done"]) + n
        if btn == "left" and mem["left_done"] >= mem["left_count"] and not low:
            stage = "goto_middle"
        elif btn == "middle" and mem["middle_done"] >= mem["middle_count"] and not low:
            stage = "retract_left"
        elif btn == "right" and mem["right_done"] >= 1 and not low:
            # the episode should have ended at success; if not, retry the confirmation a little deeper
            stage = "confirm_retry" if mem["right_done"] < 3 else "idle"
    elif stage in ("goto_left", "goto_middle", "goto_right", "retract_left"):
        arm = "right" if stage == "goto_right" else "left"
        if stage == "retract_left":
            tgt = mem["left_home"][:3]
            ok = _near(cur, "left", tgt, 0.03) or mem["retries"] >= 2
        else:
            btn = {"goto_left": "left", "goto_middle": "middle", "goto_right": "right"}[stage]
            pc = _pose(cur, arm)
            xy = B[btn]
            ok = (math.hypot(pc[0] - xy[0], pc[1] - xy[1]) < 0.015 and _HOVER_Z - 0.008 < pc[2] < _HOVER_Z + 0.04
                  and _grip(cur, arm) < 0.2)
        if ok:
            stage = {"goto_left": "press_left", "goto_middle": "press_middle", "goto_right": "press_right",
                     "retract_left": "goto_right"}[stage]
            mem["retries"] = 0
        else:
            mem["retries"] = int(mem["retries"]) + 1

    # ---- first decision: read the numbers and remember the home poses
    if stage == "start":
        r = _read_numbers(cur, "head")
        if not r.get("ok"):
            r2 = _read_numbers(cur, "front")
            if r2.get("ok"):
                r = r2
        if r.get("ok"):
            mem["left_count"] = int(r["left"])
            mem["middle_count"] = int(r["right"])
            mem["reading"] = "L=%d R=%d scores=%s margins=%s" % (r["left"], r["right"], r["scores"], r["margins"])
        else:
            # no reliable reading: fall back to the smallest valid count (assumption, see policy.md)
            mem["left_count"] = 1
            mem["middle_count"] = 1
            mem["reading"] = "FAILED: " + str(r.get("error"))
        mem["buttons"] = _button_layout(cur)
        B = {"left": mem["buttons"][0], "middle": mem["buttons"][1], "right": mem["buttons"][2]}
        mem["left_home"] = _pose(cur, "left")
        mem["right_home"] = _pose(cur, "right")
        stage = "goto_left"

    mem["stage"] = stage
    mem["history"] = (str(mem["history"]) + " " + stage)[-300:]

    # ---- choose the action
    if stage == "goto_left":
        act = {"tool": "act_goto_button", "args": {"arm": "left", "x": B["left"][0], "y": B["left"][1], "hover_z": _HOVER_Z + 0.01 * min(int(mem["retries"]), 3)}}
    elif stage == "press_left":
        n = max(0, min(_MAX_PRESSES_PER_CALL, int(mem["left_count"]) - int(mem["left_done"])))
        act = {"tool": "act_press", "args": {"arm": "left", "x": B["left"][0], "y": B["left"][1], "times": n,
                                             "start_low": bool(mem["arm_low"]), "press_z": _PRESS_Z["left"]}}
    elif stage == "goto_middle":
        act = {"tool": "act_goto_button", "args": {"arm": "left", "x": B["middle"][0], "y": B["middle"][1], "hover_z": _HOVER_Z + 0.01 * min(int(mem["retries"]), 3)}}
    elif stage == "press_middle":
        n = max(0, min(_MAX_PRESSES_PER_CALL, int(mem["middle_count"]) - int(mem["middle_done"])))
        act = {"tool": "act_press", "args": {"arm": "left", "x": B["middle"][0], "y": B["middle"][1], "times": n,
                                             "start_low": bool(mem["arm_low"]), "press_z": _PRESS_Z["left"]}}
    elif stage == "retract_left":
        act = {"tool": "act_retract", "args": {"arm": "left", "home": [float(v) for v in mem["left_home"]],
                                               "attempt": int(mem["retries"])}}
    elif stage == "goto_right":
        act = {"tool": "act_goto_button", "args": {"arm": "right", "x": B["right"][0], "y": B["right"][1], "hover_z": _HOVER_Z + 0.01 * min(int(mem["retries"]), 3)}}
    elif stage == "press_right":
        act = {"tool": "act_press", "args": {"arm": "right", "x": B["right"][0], "y": B["right"][1],
                                             "times": 0 if int(mem["right_done"]) >= 1 else 1,
                                             "start_low": bool(mem["arm_low"]), "press_z": _PRESS_Z["right"]}}
    elif stage == "confirm_retry":
        act = {"tool": "act_press", "args": {"arm": "right", "x": B["right"][0], "y": B["right"][1],
                                             "times": 0 if mem["arm_low"] else 1,
                                             "start_low": bool(mem["arm_low"]),
                                             "press_z": _PRESS_Z["right"] - 0.004 * int(mem["right_done"])}}
    else:
        act = {"tool": "act_hold", "args": {}}
    return {"memory": mem, "act": act}


def act_goto_button(robot, arm, x, y, hover_z=0.964):
    """Move `arm` ('left'/'right') above the button at world (x, y) with the gripper pointing down,
    then close the gripper there. The other arm holds. Returns the reached pose and any planning errors."""
    xy = [float(x), float(y)]
    target = [xy[0], xy[1], float(hover_z)] + _QUAT[arm]
    ok, errs = _try_move(robot, arm, target, 1.0)
    ok2, errs2 = _try_move(robot, arm, target, 0.0)
    p = _pose(robot.now(), arm)
    return {"reached": [round(v, 4) for v in p[:3]], "moved": ok, "closed": ok2, "errors": errs + errs2}


def act_press(robot, arm, x, y, times=1, start_low=False, press_z=0.9255):
    """Press the button at world (x, y) with `arm` `times` times: down to press_z then back to hover height, gripper closed.
    If the arm starts below the release height it is raised first (times=0 only raises). Returns the number of measured presses (ee z below the press threshold near the button)."""
    xy = [float(x), float(y)]
    q = _QUAT[arm]
    hover = [xy[0], xy[1], _HOVER_Z] + q
    down = [xy[0], xy[1], float(press_z)] + q
    errs = []
    zmins = []
    pressed = 0
    if start_low or _pose(robot.now(), arm)[2] < _HIGH_Z:
        ok, e = _try_move(robot, arm, hover, 0.0, tries=3)
        errs += e
    for i in range(int(times)):
        ok, e = _try_move(robot, arm, down, 0.0, tries=2)
        errs += e
        z = _pose(robot.now(), arm)[2]
        zmins.append(round(z, 4))
        if ok and z < _LOW_Z:
            pressed += 1
        ok, e = _try_move(robot, arm, hover, 0.0, tries=3)
        errs += e
        if not ok:
            break
    return {"pressed": pressed, "bottom_z": zmins, "final_z": round(_pose(robot.now(), arm)[2], 4), "errors": errs[:4]}


def act_retract(robot, arm, home, attempt=0):
    """Return `arm` to its remembered home pose (first attempt) or to a safe raised pose beside its button
    (later attempts) so that the other arm can work; gripper stays closed."""
    if int(attempt) == 0:
        target = [float(v) for v in home[:7]]
    else:
        target = [-0.25 if arm == "left" else 0.25, -0.28, 0.98] + _QUAT[arm]
    ok, errs = _try_move(robot, arm, target, 0.0)
    p = _pose(robot.now(), arm)
    return {"reached": [round(v, 4) for v in p[:3]], "moved": ok, "errors": errs}


def act_hold(robot):
    """Hold both arms in place for one native step (used only after the press sequence is exhausted)."""
    cur = robot.now()
    _send(robot, _pose(cur, "left"), _grip(cur, "left"), _pose(cur, "right"), _grip(cur, "right"))
    return {"held": True}
