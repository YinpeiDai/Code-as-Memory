"""press_button policy: read the two tile numbers once at reset, then press left button N_left times and the
middle button N_right times with the left arm, then the right button once with the right arm.

Memory keeps the numbers (they are read only from the reset frame), the verified press counts, and which button
the previous act tried to press (verified next decision from that act's recorded end-effector frames)."""

# Button press points (x, y) in world metres, as pressed by the expert in all 25 demos (identical layout).
NOMINAL_BUTTONS = [[-0.1506, -0.1498], [0.0, -0.1500], [0.1507, -0.1498]]
# Head-camera red-blob centre minus press point, measured on the demos (visible-surface bias of the detector).
DETECT_BIAS = [[0.0013, -0.0046], [-0.0016, -0.0042], [-0.0051, -0.0046]]
NOMINAL_TOP_Z = 0.7796          # 95th percentile height of the red cap points in the demos
HOVER_Z = 0.9637                # expert hover height of the ee frame above every button
LEFT_PRESS_Z = 0.9255           # expert lowest ee z for every left-arm press (all 25 demos)
RIGHT_PRESS_Z = 0.9206          # expert lowest ee z for the right-arm confirm press (all 25 demos)
LEFT_Q = [0.7071, 0.0, 0.7071, 0.0]      # left gripper pointing down (expert)
RIGHT_Q = [0.0, -0.7071, 0.0, 0.7071]    # right gripper pointing down (expert)
PRESS_TOL = 0.012               # a press is verified if the ee got within this of the press depth
XY_TOL = 0.02
RELEASE_DZ = 0.025              # ee must rise this far above press depth before another press is counted
MAX_CONFIRM_TRIES = 3           # verified confirm presses before giving up (episode ends on success)
CONFIRM_DEEPEN = 0.004          # extra depth per verified-but-unsuccessful confirm press
MAX_READ_TRIES = 3


def _pose(frame, arm):
    return [float(v) for v in np.asarray(getattr(frame, arm + "_ee_pose")).flatten()]


def _grip(frame, arm):
    return float(np.asarray(getattr(frame, arm + "_gripper")).flatten()[0])


def _buttons_from(frame):
    """Press targets [[x, y, dz], ...] for left/middle/right buttons; dz shifts press heights."""
    try:
        det = _red_button_centres(frame.head_rgb, frame.head_depth, frame.head_K, frame.head_extrinsic)
    except Exception:
        det = []
    out = []
    note = "nominal"
    if len(det) == 3:
        note = "detected"
        for i in range(3):
            px = det[i][0] - DETECT_BIAS[i][0]
            py = det[i][1] - DETECT_BIAS[i][1]
            dz = det[i][2] - NOMINAL_TOP_Z
            off = math.hypot(px - NOMINAL_BUTTONS[i][0], py - NOMINAL_BUTTONS[i][1])
            if off < 0.008:
                px, py = NOMINAL_BUTTONS[i]
            elif off > 0.06:
                px, py = NOMINAL_BUTTONS[i]
                note = "detected-implausible"
            if abs(dz) < 0.005 or abs(dz) > 0.04:
                dz = 0.0
            out.append([float(px), float(py), float(dz)])
    else:
        out = [[b[0], b[1], 0.0] for b in NOMINAL_BUTTONS]
    return out, note


def _count_presses(start_pose, frames, arm, bx, by, press_z):
    """Descents from clearly above the button (> press_z + RELEASE_DZ) to press depth, near (bx, by), over the
    arm's pose before the act followed by the act's recorded frames. A stale low pose is never re-counted."""
    seq = [start_pose] + [_pose(f, arm) for f in frames]
    n = 0
    armed = start_pose is None or start_pose[2] > press_z + RELEASE_DZ
    for p in seq[1:]:
        near = abs(p[0] - bx) < XY_TOL and abs(p[1] - by) < XY_TOL
        if p[2] > press_z + RELEASE_DZ:
            armed = True
        elif armed and near and p[2] <= press_z + PRESS_TOL:
            n += 1
            armed = False
    return n


def _press_spec(name, buttons, home_left, extra_depth):
    i = {"left": 0, "mid": 1, "right": 2}[name]
    bx, by, dz = buttons[i]
    if name == "right":
        return {"arm": "right", "x": bx, "y": by, "hover_z": HOVER_Z + dz,
                "press_z": RIGHT_PRESS_Z + dz - extra_depth, "quat": RIGHT_Q,
                "other_home": home_left}
    return {"arm": "left", "x": bx, "y": by, "hover_z": HOVER_Z + dz,
            "press_z": LEFT_PRESS_Z + dz - extra_depth, "quat": LEFT_Q, "other_home": None}


def perceive_decide(obs, goal, memory, last):
    """Read numbers at reset, verify the last press from recorded frames, choose the next press."""
    mem = dict(memory)
    cur = obs.current
    if mem.get("stage") is None or mem.get("stage") == "read":
        tries = int(mem.get("read_tries") or 0) + 1
        nl, nr, info = _read_numbers(obs.frames[0].head_rgb)
        if (nl is None or nr is None) and len(obs.frames) > 1:
            nl, nr, info = _read_numbers(cur.head_rgb)
        if mem.get("buttons") is None:
            buttons, bnote = _buttons_from(obs.frames[0])
            mem["buttons"] = buttons
            mem["home_left"] = _pose(obs.frames[0], "left")
            mem["home_right"] = _pose(obs.frames[0], "right")
            mem["note"] = "buttons " + bnote
        mem["read_tries"] = tries
        mem["reading"] = info
        if nl is None or nr is None:
            if tries < MAX_READ_TRIES:
                mem.update({"stage": "read", "left_number": None, "right_number": None,
                            "left_presses": 0, "mid_presses": 0, "right_presses": 0,
                            "pending": "none", "confirm_tries": 0})
                mem["pre_pose"] = None
                return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
            nl = 1 if nl is None else nl
            nr = 1 if nr is None else nr
            mem["reading"] = info + " (fallback 1)"
        mem.update({"stage": "pressing", "left_number": int(nl), "right_number": int(nr),
                    "left_presses": 0, "mid_presses": 0, "right_presses": 0,
                    "pending": "none", "confirm_tries": 0})
    else:
        pend = mem.get("pending") or "none"
        if pend != "none":
            spec = _press_spec(pend, mem["buttons"], mem["home_left"],
                               CONFIRM_DEEPEN * int(mem["right_presses"]) if pend == "right" else 0.0)
            n = _count_presses(mem.get("pre_pose"), obs.frames, spec["arm"], spec["x"], spec["y"], spec["press_z"])
            key = {"left": "left_presses", "mid": "mid_presses", "right": "right_presses"}[pend]
            mem[key] = int(mem[key]) + n
        mem["pending"] = "none"

    if int(mem["left_presses"]) < int(mem["left_number"]):
        nxt = "left"
    elif int(mem["mid_presses"]) < int(mem["right_number"]):
        nxt = "mid"
    elif int(mem["right_presses"]) < MAX_CONFIRM_TRIES:
        nxt = "right"
    else:
        nxt = "none"
    if nxt == "none":
        mem["stage"] = "done"
        mem["pre_pose"] = None
        return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
    extra = 0.0
    if nxt == "right":
        mem["confirm_tries"] = int(mem["confirm_tries"]) + 1
        # still running after a verified confirm press -> the press was too shallow; press a bit deeper
        extra = CONFIRM_DEEPEN * int(mem["right_presses"])
    mem["pending"] = nxt
    mem["stage"] = "pressing"
    mem["pre_pose"] = _pose(cur, "right" if nxt == "right" else "left")
    return {"memory": mem, "act": {"tool": "act_press", "args": _press_spec(nxt, mem["buttons"], mem["home_left"], extra)}}


def _step(robot, lp, lg, rp, rg):
    vals = [float(v) for v in lp[:7]] + [float(lg)] + [float(v) for v in rp[:7]] + [float(rg)]
    robot.step(*vals)


def _move_arm(robot, arm, target, grip):
    """One native action moving `arm` to target pose [x,y,z,qw,qx,qy,qz] while holding the other arm."""
    now = robot.now()
    if arm == "left":
        _step(robot, target, grip, _pose(now, "right"), _grip(now, "right"))
    else:
        _step(robot, _pose(now, "left"), _grip(now, "left"), target, grip)


def act_press(robot, arm, x, y, hover_z, press_z, quat, other_home=None):
    """Press the button at (x, y) once with `arm`: optionally send the other arm to `other_home`, go to the hover
    pose above the button, close the gripper, descend to press_z and return to hover. Returns measured heights."""
    other = "right" if arm == "left" else "left"
    result = {"arm": arm, "errors": []}
    hover = [x, y, hover_z] + list(quat)
    try:
        if other_home is not None:
            op = _pose(robot.now(), other)
            if math.dist(op[:3], other_home[:3]) > 0.02:
                _move_arm(robot, other, list(other_home), _grip(robot.now(), other))
        p = _pose(robot.now(), arm)
        q = p[3:7]
        qdot = abs(sum(a * b for a, b in zip(q, quat)))
        if math.dist(p[:3], hover[:3]) > 0.005 or qdot < 0.999:
            _move_arm(robot, arm, hover, _grip(robot.now(), arm))
        if _grip(robot.now(), arm) > 0.05:
            _move_arm(robot, arm, hover, 0.0)
        _move_arm(robot, arm, [x, y, press_z] + list(quat), 0.0)
        result["bottom_z"] = _pose(robot.now(), arm)[2]
        _move_arm(robot, arm, hover, 0.0)
        result["after_z"] = _pose(robot.now(), arm)[2]
    except Exception as e:  # MotionError: report and let the next decision re-verify from frames
        result["errors"].append(str(e)[:200])
    return result


def act_hold(robot):
    """Hold both arms in place for one native action (used while waiting / after the task)."""
    now = robot.now()
    _step(robot, _pose(now, "left"), _grip(now, "left"), _pose(now, "right"), _grip(now, "right"))
    return {"held": True}
