"""Policy: watch the cue, wait through the delay, find the imposter in the test layout and touch it.

Stages (memory.stage): "start" -> "cue" -> "delay" -> "touch".
Commands are zeroed by the environment during the cue and delay phases, so the
policy only observes (one native step per decision) until the test layout appears.
"""

_VIS_PIXELS = 30        # saturated R/G/B pixels in the front image => objects shown
_HOVER = 0.03           # m above the object top before the final descent
_GRIP = 1.0             # gripper command during the touch (+1 open: fingers straddle the object)
_FINAL_Z = {"T": 0.015, "cube": 0.019, "sphere": 0.019}   # final TCP height = object mid-height (R002/R003 succeeded there)
_XY_TOL = 0.008         # m xy error allowed before descending below hover height


def _n_colored(frame):
    m = _color_masks(frame.front_rgb)
    return int(sum(int(v.sum()) for v in m.values()))


def _detect_frame(frame):
    return _detect([(frame, "front"), (frame, "wrist")])


def _compact(objs):
    return [{"color": o["color"], "shape": o["shape"], "x": round(o["x"], 4), "y": round(o["y"], 4),
             "ll": o["ll"], "w": o["w"], "top_z": round(o["top_z"], 4),
             "bbox": [round(v, 4) for v in o.get("bbox", [o["x"] - 0.02, o["x"] + 0.02, o["y"] - 0.02, o["y"] + 0.02])]}
            for o in objs]


# Open-finger footprint relative to the TCP, measured from wrist depth in R044 (fingers 0.039):
# each finger spans |y| in [0.038, 0.051] and x in [-0.010, 0.011]; finger tips ~9 mm below the TCP.
_FING_IN = 0.038
_FING_OUT = 0.051
_FING_HALF_X = 0.011
_FING_MARGIN = 0.0


def _finger_offset(tgt, others):
    """TCP y-offset (finger axis = world y) so that neither open finger lands on a neighbour while the
    target stays between the fingers. Clearance uses neighbour surface points above 8 mm (3 mm cells).
    Returns (offset, clearance_at_zero, clearance_at_offset); clearance < 0 means predicted overlap."""
    tb = tgt["bbox"]
    m = _FING_MARGIN
    pts = [o["cells"] for o in others if "cells" in o and len(o["cells"])]
    P = np.concatenate(pts, 0) if pts else np.zeros((0, 2))
    if len(P):
        P = P[np.abs(P[:, 0] - tgt["x"]) <= _FING_HALF_X]
        P = P[np.abs(P[:, 1] - tgt["y"]) <= 0.09]

    def clearance(s):
        ty = tgt["y"] + s
        c = min(0.05, tb[2] - (ty - _FING_IN), (ty + _FING_IN) - tb[3])
        if len(P):
            dy = np.abs(P[:, 1] - ty)
            inside = (dy >= _FING_IN) & (dy <= _FING_OUT)
            if inside.any():
                depth = np.minimum(dy[inside] - _FING_IN, _FING_OUT - dy[inside])
                c = min(c, -float(depth.max()) - 0.0001)
            else:
                outside = np.minimum(np.abs(dy - _FING_IN), np.abs(dy - _FING_OUT))
                c = min(c, float(outside.min()))
        return c
    c0 = clearance(0.0)
    if c0 >= m:
        return 0.0, round(c0, 4), round(c0, 4)
    best = (c0, 0.0)
    for k in range(1, 21):
        for s in (0.001 * k, -0.001 * k):
            c = clearance(s)
            if c >= m:
                return s, round(c0, 4), round(c, 4)
            if c > best[0]:
                best = (c, s)
    return best[1], round(c0, 4), round(best[0], 4)


def _blocked_shift(tgt, others):
    """Fallback y shift (12 mm) away from the nearest neighbour roughly on the finger axis."""
    best = None
    for o in others:
        dx, dy = o["x"] - tgt["x"], o["y"] - tgt["y"]
        if abs(dx) < 0.035 and abs(dy) < 0.09:
            if best is None or abs(dy) < abs(best):
                best = dy
    if best is None:
        return 0.0
    return -0.012 if best > 0 else 0.012


def _counts_ok(objs, expected):
    cnt = {}
    for o in objs:
        cnt[o["color"]] = cnt.get(o["color"], 0) + 1
    return sorted(cnt.get(c, 0) for c in ("red", "green", "blue")) == sorted(expected)


def _layout_changed(a, b):
    """Number of objects in b without a same-colour object of a within 2 cm."""
    n = 0
    for o in b:
        if not any(p["color"] == o["color"] and math.hypot(p["x"] - o["x"], p["y"] - o["y"]) < 0.02 for p in a):
            n += 1
    return n


def _mget(memory, key, default):
    v = memory.get(key) if isinstance(memory, dict) else None
    return default if v is None or v == "" else v


def perceive_decide(obs, goal, memory, last):
    """Track cue/delay/test phases from the front camera, remember the cue objects, choose the
    imposter when the test layout appears and touch it."""
    mem = {
        "stage": _mget(memory, "stage", "start"),
        "frames_seen": int(_mget(memory, "frames_seen", 0)),
        "cue_objects": _mget(memory, "cue_objects", []),
        "cue_frame": int(_mget(memory, "cue_frame", -1)),
        "delay_frame": int(_mget(memory, "delay_frame", -1)),
        "test_frame": int(_mget(memory, "test_frame", -1)),
        "target": _mget(memory, "target", []),
        "target_info": _mget(memory, "target_info", ""),
        "touch_calls": int(_mget(memory, "touch_calls", 0)),
        "notes": _mget(memory, "notes", ""),
    }
    frames = list(obs.frames)
    mem["frames_seen"] = mem["frames_seen"] + len(frames)
    cur = obs.current
    vis = _n_colored(cur) > _VIS_PIXELS
    stage = mem["stage"]
    wait = {"tool": "act_wait", "args": {"steps": 1}}

    if stage == "start":
        if vis:
            objs = _detect_frame(cur)
            mem["cue_objects"] = _compact(objs)
            mem["cue_frame"] = mem["frames_seen"] - 1
            mem["stage"] = "cue"
            mem["notes"] = "cue %d objects: %s" % (len(objs), ",".join(o["color"][0] + o["shape"][0] for o in objs))
        return {"memory": mem, "act": wait}

    if stage == "cue":
        if not vis:
            mem["stage"] = "delay"
            mem["delay_frame"] = mem["frames_seen"] - 1
            return {"memory": mem, "act": wait}
        objs = _detect_frame(cur)
        if _layout_changed(mem["cue_objects"], objs) < 3:
            # same cue layout: keep the detection whose colour counts fit the expected {2,3,3}
            if not _counts_ok(mem["cue_objects"], [2, 3, 3]) and _counts_ok(objs, [2, 3, 3]):
                mem["cue_objects"] = _compact(objs)
                mem["notes"] += "; cue re-detected at frame %d" % (mem["frames_seen"] - 1)
            return {"memory": mem, "act": wait}
        mem["notes"] += "; layout changed without blank delay"
        stage = "delay"

    if stage == "delay":
        if not vis:
            return {"memory": mem, "act": wait}
        objs = _detect_frame(cur)
        if not _counts_ok(objs, [3, 3, 3]) and mem["test_frame"] < 0:
            # unexpected test detection: look once more (arm still at rest) before committing
            mem["test_frame"] = mem["frames_seen"] - 1
            mem["notes"] += "; test counts unexpected, re-observing"
            return {"memory": mem, "act": wait}
        tgt, info = _choose_target(mem["cue_objects"], objs)
        if mem["test_frame"] < 0:
            mem["test_frame"] = mem["frames_seen"] - 1
        if tgt is None:
            mem["notes"] += "; no test objects detected"
            return {"memory": mem, "act": wait}
        top = 0.03 if tgt["shape"] == "T" else 0.04
        others = [o for o in objs if o is not tgt]
        if "bbox" in tgt and "cells" in tgt:
            s, c0, c1 = _finger_offset(tgt, others)
        else:
            s, c0, c1 = 0.0, 0.0, 0.0
        info["finger_offset"] = [round(s, 4), c0, c1]
        # The clearance model did not separate R044 (finger landed on a neighbour) from R004/R024/R034
        # (similar predicted overlap, no collision), so the offset is applied only if a stall above the
        # target is actually measured (see act_touch); its sign/size come from the model.
        if s == 0.0:
            s = _blocked_shift(tgt, others)
        mem["target"] = [round(tgt["x"], 4), round(tgt["y"], 4), top, _FINAL_Z.get(tgt["shape"], 0.016), round(s, 4)]
        info["test"] = ",".join("%s%s(%.3f,%.3f)" % (o["color"][0], o["shape"][0], o["x"], o["y"]) for o in objs)
        mem["target_info"] = str(info)
        mem["stage"] = "touch"
        stage = "touch"

    if stage == "touch":
        mem["touch_calls"] += 1
        x, y, top, zf = mem["target"][:4]
        s = mem["target"][4] if len(mem["target"]) > 4 else 0.0
        return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": x, "y": y, "top": top, "z_final": zf,
                                                                    "grip": _GRIP, "shift_if_blocked": s,
                                                                    "max_steps": 40}}}

    return {"memory": mem, "act": wait}


def act_wait(robot, steps=1):
    """Send `steps` zero-motion native steps (gripper at the touch setting) and report the measured TCP."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _GRIP)
    return {"waited": int(steps), "tcp": [round(float(v), 4) for v in robot.now().tcp]}


_KP = 2.0        # proportional gain on (error - predicted coast), command units per 0.1 m
_KP_HOLD = 2.0   # xy gain near contact; 3.0 caused a +-3 mm limit cycle and no success (R033)
_COAST = 0.56    # predicted remaining coast = _COAST * last measured per-step displacement (demo fit)


def act_touch(robot, x, y, top, z_final, grip=1.0, shift_if_blocked=0.0, max_steps=40):
    """Move the gripper (grip: +1 open, -1 closed) above world (x, y) at top+hover, then descend
    vertically to TCP height z_final (open fingers straddle the object). PD-like control on the
    measured TCP: command = KP*(error - COAST*d)/0.1, clipped to [-1, 1], where d averages the
    measured last displacement and the model prediction (per-step response fitted on the demos:
    d' = 0.26*0.1*a + 0.36*d), which damps one-step disturbances. Zero rotation command.
    Descent below hover only once the xy error and xy speed are small. Returns a per-step log; the
    episode normally ends with success inside this call.
    Blocked descent (R044: an open finger landed on a neighbour): if, while descending, the TCP stays
    above z_final + 12 mm for 2 steps despite a down command, lift 8 mm for one step and shift the
    target by shift_if_blocked along y (finger axis), keeping the object between the fingers; after
    3 hold steps at mid-height the shift is removed gradually (re-centring)."""
    log = []
    prev = None
    prev_az = 0.0
    d_pred = np.zeros(3)
    alpha = np.array([0.262, 0.262, 0.30])
    beta = np.array([0.36, 0.36, 0.254])
    descending = False
    hold = 0
    y_offset = 0.0
    stall = 0
    lift = 0
    wiggle = [(0.0, 0.0), (0.008, 0.0), (0.0, 0.008), (-0.008, 0.0), (0.0, -0.008)]
    for i in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=np.float64)
        d_meas = np.zeros(3) if prev is None else tcp[:3] - prev
        prev = tcp[:3].copy()
        d = 0.5 * (d_meas + d_pred)
        s_eff = y_offset * max(0.0, 1.0 - max(0, hold - 3) / 4.0)
        ex, ey = x - tcp[0], y + s_eff - tcp[1]
        exy = math.hypot(ex, ey)
        vxy = math.hypot(d[0], d[1])
        if not descending and exy < _XY_TOL and vxy < 0.012:
            descending = True
        if descending and exy > 2.5 * _XY_TOL and y_offset == 0.0 and stall == 0:
            descending = False
        zt = z_final if descending else top + _HOVER
        if descending and y_offset == 0.0 and shift_if_blocked != 0.0:
            if tcp[2] > z_final + 0.012 and d_meas[2] > -0.003 and prev_az < -0.3:
                stall += 1
            else:
                stall = 0
            if stall >= 2:
                y_offset = float(shift_if_blocked)
                lift = 1
                s_eff = y_offset
                ey = y + s_eff - tcp[1]
        if lift > 0:
            zt = tcp[2] + 0.008
            lift -= 1
        if descending and (tcp[2] < z_final + 0.006 or (abs(d_meas[2]) < 0.002 and tcp[2] < top)):
            hold += 1
        ox, oy = 0.0, 0.0
        if hold > 6 + (4 if y_offset != 0.0 else 0):
            # no success after holding at mid-height: small lateral wiggle around the target
            ox, oy = wiggle[((hold - 4) // 2) % len(wiggle)]
        err = np.array([ex + ox, ey + oy, zt - tcp[2]])
        kp = np.array([_KP, _KP, _KP])
        if descending and tcp[2] < top + 0.01:
            # near/at contact the hand is pushed sideways (~-1 cm x, R005-R032); re-centre faster
            kp[0] = kp[1] = _KP_HOLD
        a = kp * (err - _COAST * d) / 0.1
        a = np.clip(a, -1.0, 1.0)
        d_pred = alpha * 0.1 * a + beta * d
        prev_az = float(a[2])
        log.append([int(getattr(f, "step", i)), round(float(tcp[0]), 4), round(float(tcp[1]), 4),
                    round(float(tcp[2]), 4), round(float(a[0]), 2), round(float(a[1]), 2),
                    round(float(a[2]), 2), int(descending), hold, round(y_offset, 3)])
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, grip)
    f = robot.now()
    return {"done_steps": int(max_steps), "final_tcp": [round(float(v), 4) for v in f.tcp], "log": log[-14:]}
