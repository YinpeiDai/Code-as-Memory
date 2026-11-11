"""RememberColor9: remember the cue cube colour, wait for the nine-cube line-up,
then touch the cube of the same colour (press its top face with closed fingers).

Perception: colour-palette classification of pixels whose back-projected world
height is on a cube top face (z > 0.038 m, from depth + calibration).
Control: closed-loop pd_ee_delta_pose offsets on the measured TCP, holding the
start orientation, hovering above neighbours until aligned, then pressing down.
"""

_PALETTE = [[231, 7, 7], [169, 5, 5], [231, 169, 8], [231, 231, 5], [4, 231, 4],
            [4, 231, 231], [6, 169, 169], [5, 5, 231], [231, 4, 231]]
_NAMES = ["red", "darkred", "orange", "yellow", "green", "cyan", "teal", "blue", "magenta"]
_ZTOP = 0.038        # world z above which a pixel belongs to a cube top face (cube 0.04 m)
_COLOR_TOL = 45.0    # max RGB distance to a palette colour
_CUBES_MIN_CLASSES = 5
_HOVER = 0.03        # fingertip clearance above cube tops while moving laterally
_CONE = 1.0          # extra height per metre of lateral error
_XYTOL = 0.012       # lateral alignment needed before pressing
_PRESS = 0.012       # press target below the measured top face
_KP = 1.5            # translation gain (one step realises ~40% of 0.1*command)
_KR = 1.0            # orientation-hold gain


def _world_points(depth, K, E):
    d = np.asarray(depth, dtype=np.float64) / 1000.0
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1)
    Pw = (P - E[:, 3]) @ E[:, :3]
    return Pw, d > 0


def _classify(frame, cam):
    """Per-pixel palette class (-1 = none) on cube top faces, plus world points."""
    rgb = np.asarray(getattr(frame, cam + "_rgb"), dtype=np.float64)
    Pw, ok = _world_points(getattr(frame, cam + "_depth"), getattr(frame, cam + "_K"),
                           getattr(frame, cam + "_extrinsic"))
    pal = np.asarray(_PALETTE, dtype=np.float64)
    dist = np.sqrt(((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1))
    cls = dist.argmin(-1)
    good = ok & (Pw[..., 2] > _ZTOP) & (Pw[..., 2] < 0.07) & (dist.min(-1) < _COLOR_TOL)
    cls = np.where(good, cls, -1)
    return cls, Pw


def _class_counts(cls):
    return [int((cls == k).sum()) for k in range(len(_NAMES))]


def _components(mask):
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack = [(y0, x0)]
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        comps.append(np.array(pix))
    return comps


def _locate(frame, cam, k, minsize=6):
    """Largest top-face component of class k: dict with xyz, n, border flag."""
    cls, Pw = _classify(frame, cam)
    best = None
    for c in _components(cls == k):
        if len(c) >= minsize and (best is None or len(c) > len(best)):
            best = c
    if best is None:
        return None
    P = Pw[best[:, 0], best[:, 1]]
    H, W = cls.shape
    border = bool(best[:, 0].min() == 0 or best[:, 1].min() == 0 or
                  best[:, 0].max() == H - 1 or best[:, 1].max() == W - 1)
    return {"xyz": [float(v) for v in P.mean(0)], "n": int(len(best)), "border": border}


def _cue_from_frame(frame):
    """Single-cube cue: returns (class, confidence, pixel count) or None."""
    votes = np.zeros(len(_NAMES))
    total = 0
    for cam, w in (("wrist", 1.0), ("front", 3.0)):
        cls, _ = _classify(frame, cam)
        cnt = np.array(_class_counts(cls), dtype=float)
        if (cnt >= 6).sum() >= 3:      # several cubes -> not the cue phase
            return None
        votes += w * cnt
        total += cnt.sum()
    if votes.sum() < 15:
        return None
    k = int(votes.argmax())
    return k, float(votes[k] / votes.sum()), int(total)


def _cubes_visible(frame):
    cls, _ = _classify(frame, "front")
    cnt = _class_counts(cls)
    return sum(1 for c in cnt if c >= 6) >= _CUBES_MIN_CLASSES, cnt


def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(R):
    c = min(1.0, max(-1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-8:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _command(tcp, target, R0, xy_offset, press):
    p = np.asarray(tcp[:3], dtype=np.float64)
    T = np.asarray(target, dtype=np.float64)
    goal_xy = T[:2] + np.asarray(xy_offset, dtype=np.float64)
    e = goal_xy - p[:2]
    d = float(np.linalg.norm(e))
    if d > _XYTOL:
        zd = T[2] + _HOVER + _CONE * d
        phase = "approach"
    else:
        zd = T[2] - press
        phase = "press"
    des = np.array([goal_xy[0], goal_xy[1], zd])
    u = np.clip(_KP * (des - p) / 0.1, -1.0, 1.0)
    if phase == "approach" and p[2] < T[2] + 0.015:
        u[0] = 0.0          # near/at a cube top: lift before moving sideways
        u[1] = 0.0
    R = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _logR(R0 @ R.T)
    r = -_KR * w / 0.1          # native: signed offset is -0.1*r rad
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return [float(v) for v in u] + [float(v) for v in r], phase, d


def perceive_decide(obs, goal, memory, last):
    """Stage machine: read the cue colour, wait for the cubes, then touch the matching cube."""
    mem = dict(memory or {})
    stage = mem.get("stage") or "cue"
    decisions = int(mem.get("decisions") or 0) + 1
    attempts = int(mem.get("touch_attempts") or 0)
    cue = mem.get("cue_color")
    cue_conf = mem.get("cue_confidence")
    cue_step = mem.get("cue_step")
    target = mem.get("target_xyz")
    home = mem.get("home_rpy")
    notes = mem.get("notes") or ""
    frames = list(obs.frames)
    cur = obs.current
    if home is None:
        home = [float(v) for v in np.asarray(cur.tcp)[3:6]]
    # 1. cue: from any frame of this observation that shows a single cube
    if cue is None:
        for fr in frames:
            res = _cue_from_frame(fr)
            if res is not None:
                cue = _NAMES[res[0]]
                cue_conf = res[1]
                cue_step = int(fr.step)
                break
    if isinstance(last, dict) and cue is None and last.get("cue_color"):
        cue = last.get("cue_color")
        cue_conf = last.get("cue_confidence")
        cue_step = last.get("cue_step")
    visible, counts = _cubes_visible(cur)
    if not visible and target is None:
        stage = "wait"
        act = {"tool": "act_wait_for_cubes", "args": {"max_steps": 16}}
    else:
        if target is None:
            k = _NAMES.index(cue) if cue in _NAMES else None
            if k is None:
                # no cue seen: unavoidable guess, take the class with most front pixels
                k = int(np.argmax(counts))
                notes = notes + "no cue observed; guessed " + _NAMES[k] + ". "
            f = _locate(cur, "front", k)
            w = _locate(cur, "wrist", k)
            pts = []
            if f is not None:
                pts.append(f["xyz"])
            if w is not None and not w["border"] and w["n"] >= 30:
                pts.append(w["xyz"])
            if pts:
                xyz = np.mean(np.array(pts), axis=0)
                target = [float(xyz[0]), float(xyz[1]), float(min(max(xyz[2], 0.035), 0.046))]
            else:
                notes = notes + "target colour not located. "
        if target is None:
            stage = "wait"
            act = {"tool": "act_wait_for_cubes", "args": {"max_steps": 2}}
        else:
            stage = "touch"
            modes = ["closed_center", "open_finger_on_top", "open_straddle"]
            mode = modes[min(attempts, len(modes) - 1)]
            attempts += 1
            act = {"tool": "act_touch", "args": {
                "target": target, "color": cue if cue in _NAMES else _NAMES[int(np.argmax(counts))],
                "mode": mode, "home_rpy": home, "max_steps": 14}}
    if isinstance(last, str):
        notes = (notes + last[:160] + " | ")[-600:]
    elif isinstance(last, dict):
        notes = (notes + str(last.get("summary", "")) + " | ")[-600:]
    new_mem = {"stage": stage, "decisions": decisions, "cue_color": cue,
               "cue_confidence": cue_conf, "cue_step": cue_step, "target_xyz": target,
               "touch_attempts": attempts, "home_rpy": home, "notes": notes}
    return {"memory": new_mem, "act": act}


def act_wait_for_cubes(robot, max_steps=16):
    """Send zero motion (gripper closing) until the nine-cube line-up is visible in the
    front camera, or max_steps elapse. Reports the cue colour if a cue frame is seen."""
    cue = None
    seen = None
    n = 0
    fr = robot.now()
    vis, cnt = _cubes_visible(fr)
    while not vis and n < int(max_steps):
        res = _cue_from_frame(fr)
        if res is not None and cue is None:
            cue = res
            seen = int(fr.step)
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
        fr = robot.now()
        vis, cnt = _cubes_visible(fr)
    out = {"waited_steps": n, "cubes_visible": bool(vis), "class_pixels": cnt,
           "frame_step": int(fr.step),
           "summary": "waited %d steps, cubes %s at step %d" % (n, vis, int(fr.step))}
    if cue is not None:
        out["cue_color"] = _NAMES[cue[0]]
        out["cue_confidence"] = cue[1]
        out["cue_step"] = seen
    return out


def act_touch(robot, target, color, mode="closed_center", home_rpy=None, max_steps=14):
    """Move above the remembered cube (world top-face centre `target`), then press down
    on it. mode: closed_center (closed fingers on top centre), open_finger_on_top
    (open gripper, one finger over the top), open_straddle (open, descend around it).
    Returns the trace and why it stopped (stall after contact or step budget)."""
    k = _NAMES.index(color) if color in _NAMES else -1
    T = [float(v) for v in target]
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=np.float64)
    if home_rpy is None:
        home_rpy = [float(v) for v in tcp[3:6]]
    R0 = _rpy2R(float(home_rpy[0]), float(home_rpy[1]), float(home_rpy[2]))
    if mode == "closed_center":
        grip, off, press = -1.0, [0.0, 0.0], _PRESS
    elif mode == "open_finger_on_top":
        grip, off, press = 1.0, [-0.01, 0.02], _PRESS
    else:
        grip, off, press = 1.0, [0.0, 0.0], 0.03
    check = None
    try:
        dr = robot.dry_run(T[0] + off[0], T[1] + off[1], T[2] + _HOVER,
                           float(home_rpy[0]), float(home_rpy[1]), float(home_rpy[2]), mode="absolute")
        check = {"within_joint_limits": dr.get("within_joint_limits"),
                 "min_joint_margin_rad": dr.get("min_joint_margin_rad")}
    except Exception as e:
        check = {"error": str(e)[:120]}
    trace = []
    stall = 0
    reason = "budget"
    prev_z = None
    for i in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=np.float64)
        # refine the target from the wrist camera while still well above the cubes
        if k >= 0 and tcp[2] > 0.10:
            w = _locate(fr, "wrist", k, minsize=30)
            if w is not None and not w["border"]:
                dxy = math.hypot(w["xyz"][0] - T[0], w["xyz"][1] - T[1])
                if dxy < 0.02:
                    T[0] = 0.5 * T[0] + 0.5 * w["xyz"][0]
                    T[1] = 0.5 * T[1] + 0.5 * w["xyz"][1]
        cmd, phase, d = _command(tcp, T, R0, off, press)
        if phase == "press" and prev_z is not None and abs(prev_z - tcp[2]) < 0.0015 \
                and tcp[2] < T[2] + 0.03:
            stall += 1
        else:
            stall = 0
        trace.append([int(fr.step), round(float(tcp[0]), 4), round(float(tcp[1]), 4),
                      round(float(tcp[2]), 4), phase[0]])
        if stall >= 2:
            reason = "stalled_in_contact_without_success"
            break
        prev_z = float(tcp[2])
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=np.float64)
    return {"mode": mode, "reason": reason, "target": T, "dry_run": check,
            "final_tcp": [float(v) for v in tcp[:3]], "fingers": [float(v) for v in np.asarray(fr.fingers)],
            "trace": trace[-8:],
            "summary": "%s ended %s at step %d tcp z %.3f" % (mode, reason, int(fr.step), float(tcp[2]))}
