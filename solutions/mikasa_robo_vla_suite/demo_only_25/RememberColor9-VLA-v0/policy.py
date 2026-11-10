"""RememberColor9: remember the cue cube colour, then touch the same-coloured cube.

Phases observed in all 25 demos (front camera, fixed):
  cue    : one cube at the table origin (1-5 frames)
  delay  : empty table (1-5 frames); robot commands are zeroed until cubes appear
  answer : nine cubes of nine distinct colours in an arc; commands act from here
Control: closed-gripper press on the top face of the target cube (finger contact).
"""

# Top-face colours (flat-shaded, identical in cue and answer phase in all demos).
_COLORS = {
    "yellow": (231, 231, 5), "red": (231, 6, 6), "darkred": (169, 6, 6),
    "magenta": (231, 5, 231), "green": (5, 231, 5), "blue": (5, 5, 231),
    "teal": (5, 169, 169), "cyan": (5, 231, 231), "orange": (231, 169, 4),
}
_CUBE_TOP = 0.040      # measured top-face height (m)
_HOVER_Z = 0.075       # fingertip height while travelling (above all cube tops)
_PRESS_Z = 0.005       # commanded goal height during the press (below the top face)
_GATE_IN = 0.010       # start final descent when xy error below this (m)
_GATE_OUT = 0.018      # abort descent (go back to hover) if xy error exceeds this
_GAIN = 0.6            # command = error / (0.1 * gain), clipped to [-1, 1]
_STEP_LIMIT = 25       # native step limit of the task


def _world_points(depth, K, E):
    """Back-project an optical-axis depth image (mm) to world XYZ (H, W, 3)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return ((pc - t) @ R).reshape(H, W, 3)


def _components(xy, link):
    """Single-linkage components (Chebyshev distance < link). Returns labels, count."""
    n = len(xy)
    lab = -np.ones(n, dtype=int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = k
        stack = [i]
        while stack:
            j = stack.pop()
            nb = np.nonzero((np.abs(xy - xy[j]).max(1) < link) & (lab < 0))[0]
            lab[nb] = k
            stack.extend(nb.tolist())
        k += 1
    return lab, k


def _detect_cubes(frame):
    """Detect cube top faces in the front camera. Returns {colour: {xy, n, ext}}."""
    names = list(_COLORS)
    P = np.array([_COLORS[k] for k in names], dtype=np.float64)
    im = np.asarray(frame.front_rgb).astype(np.float64)
    depth = np.asarray(frame.front_depth)
    w = _world_points(depth, np.asarray(frame.front_K, dtype=np.float64),
                      np.asarray(frame.front_extrinsic, dtype=np.float64))
    d = np.sqrt(((im[:, :, None, :] - P[None, None]) ** 2).sum(-1))
    lab = d.argmin(-1)
    dm = d.min(-1)
    zz = w[..., 2]
    mask = (zz > 0.034) & (zz < 0.048) & (dm < 40.0) & (depth > 0)
    cands = []
    for i, name in enumerate(names):
        mm = mask & (lab == i)
        if int(mm.sum()) < 6:
            continue
        xy = w[mm][:, :2]
        cl, k = _components(xy, 0.013)
        for j in range(k):
            p = xy[cl == j]
            if len(p) >= 6:
                cands.append((len(p), name, p.mean(0), p.max(0) - p.min(0)))
    cands.sort(key=lambda c: -c[0])
    out = {}
    for n, name, c, ext in cands:
        if name in out:
            continue
        if any(float(np.abs(c - np.array(o["xy"])).max()) < 0.03 for o in out.values()):
            continue  # edge-shading fragment of an already accepted cube
        out[name] = {"xy": [float(c[0]), float(c[1])], "n": int(n),
                     "ext": [float(ext[0]), float(ext[1])]}
    return out


def _closest_color(name, available):
    if not available:
        return None
    if name in available:
        return name
    if name not in _COLORS:
        return sorted(available)[0]
    ref = np.array(_COLORS[name], dtype=float)
    return min(available, key=lambda k: float(np.sum((np.array(_COLORS[k]) - ref) ** 2)))


def perceive_decide(obs, goal, memory, last):
    """Track cue/delay/answer phases from the front camera, remember the cue colour,
    and touch the cube of that colour once the nine cubes appear."""
    m = dict(memory)
    defaults = {"phase": "cue", "cue_color": "", "cue_step": -1, "cue_frames": 0,
                "cubes_step": -1, "target_color": "", "target_xy": [],
                "cube_colors": [], "cube_xy": [], "waits_for_target": 0,
                "reach_calls": 0, "notes": ""}
    for k, v in defaults.items():
        if m.get(k) is None:
            m[k] = v

    frames = list(obs.frames)
    cur = obs.current
    cur_step = int(cur.step) if cur.step is not None else 0
    latest_multi = None
    for f in frames:
        if m["phase"] not in ("cue", "delay"):
            break
        det = _detect_cubes(f)
        fstep = int(f.step) if f.step is not None else cur_step
        if len(det) == 1 and m["phase"] == "cue":
            name = list(det)[0]
            if m["cue_color"] in ("", name):
                m["cue_color"] = name
                m["cue_frames"] = int(m["cue_frames"]) + 1
                if int(m["cue_step"]) < 0:
                    m["cue_step"] = fstep
            else:  # conflicting cue reading: keep the colour seen in more frames
                m["notes"] = "conflicting cue %s vs %s at step %d" % (name, m["cue_color"], fstep)
        elif len(det) == 0:
            if m["cue_color"] != "":
                m["phase"] = "delay"
        elif len(det) >= 2:
            latest_multi = (det, fstep)

    if latest_multi is not None and m["phase"] in ("cue", "delay"):
        det, fstep = latest_multi
        m["cube_colors"] = sorted(det)
        m["cube_xy"] = [det[k]["xy"] for k in sorted(det)]
        if int(m["cubes_step"]) < 0:
            m["cubes_step"] = fstep
        want = m["cue_color"]
        if want in det:
            m["target_color"] = want
            m["target_xy"] = det[want]["xy"]
            m["phase"] = "reach"
        elif int(m["waits_for_target"]) >= 2 or len(det) >= 9 or want == "":
            alt = _closest_color(want, list(det))
            m["target_color"] = alt
            m["target_xy"] = det[alt]["xy"]
            m["phase"] = "reach"
            m["notes"] = "cue colour '%s' not detected; using %s" % (want, alt)
        else:
            m["waits_for_target"] = int(m["waits_for_target"]) + 1

    if m["phase"] in ("reach", "pressing") and len(m["target_xy"]) == 2:
        m["reach_calls"] = int(m["reach_calls"]) + 1
        m["phase"] = "pressing"
        act = {"tool": "act_reach_touch",
               "args": {"target_x": float(m["target_xy"][0]),
                        "target_y": float(m["target_xy"][1]),
                        "start_step": cur_step,
                        "fallback_open": int(m["reach_calls"]) > 1}}
    else:
        act = {"tool": "act_wait", "args": {}}
    return {"memory": m, "act": act}


def act_wait(robot):
    """Send one zero-motion step (gripper closing) while the cue/delay phase runs."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    f = robot.now()
    return {"step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp[:3]]}


def act_reach_touch(robot, target_x, target_y, start_step=0, fallback_open=False, max_steps=24):
    """Closed-loop: travel at hover height above (target_x, target_y), descend with a
    closed gripper and press the cube's top face. If the press stalls without the episode
    ending, open the gripper while pressing (fingers slide onto the cube sides)."""
    tgt = np.array([float(target_x), float(target_y)])
    log = []
    descending = False
    stall = 0
    open_grip = bool(fallback_open)
    prev_z = None
    for i in range(int(max_steps)):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        step_now = int(f.step) if f.step is not None else int(start_step) + i
        exy = tgt - p[:2]
        e = float(np.hypot(exy[0], exy[1]))
        remaining = _STEP_LIMIT - step_now
        if not descending and (e < _GATE_IN or (remaining <= 3 and e < _GATE_OUT)):
            descending = True
        elif descending and e > _GATE_OUT and p[2] > _CUBE_TOP + 0.01:
            descending = False
        if descending:
            zg = _PRESS_Z
        else:
            zg = max(_HOVER_Z, _CUBE_TOP + 0.02)
        err = np.array([exy[0], exy[1], zg - p[2]])
        a = np.clip(err / (0.1 * _GAIN), -1.0, 1.0)
        if descending and prev_z is not None and p[2] < _CUBE_TOP + 0.015 and prev_z - p[2] < 0.003:
            stall += 1
        if stall >= 2:
            open_grip = True
        g = 1.0 if open_grip else -1.0
        log.append([step_now, round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3),
                    round(e, 4), int(descending), int(open_grip)])
        prev_z = float(p[2])
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, g)
    f = robot.now()
    return {"done_steps": len(log), "final_tcp": [round(float(v), 4) for v in f.tcp[:3]],
            "stall": stall, "opened": open_grip, "log": log[-12:]}
