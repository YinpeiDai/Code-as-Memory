"""RememberColor3: observe a cue cube's colour, wait, then touch the cube of the same colour.

Phases measured in the 250 assigned demos (front camera):
  cue    : one cube at the table centre, frames 0..k-1 (k = 1..5)
  delay  : empty table, 1..5 frames
  choice : three cubes (distinct colours) appear at frame 2..10; arm commands are replaced by zero
           by the environment until the frame on which the cubes are visible.
Native step limit 25, so 15..23 steps remain after the cubes appear.
"""

CUBE_HALF = 0.02          # measured: top face at z ~ 0.041 (wrist depth), top face ~0.038 m wide
SAT_MIN = 0.8             # saturation threshold for cube pixels (wood ~0.6, arm/floor grey)
VAL_MIN = 100             # brightness threshold (shaded cube faces are ~147)
MIN_BLOB = 12             # pixels
STEP_LIMIT = 25           # native episode step limit (task description)
MOVE_SCALE = 0.1          # metres per unit translation command
ROT_SCALE = 0.1           # radians per unit rotation command, applied with a minus sign


def _rgb_names(chroma):
    r, g, b = float(chroma[0]), float(chroma[1]), float(chroma[2])
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx - mn < 1e-6:
        return "grey"
    if mx == r:
        h = (60.0 * ((g - b) / (mx - mn))) % 360.0
    elif mx == g:
        h = 60.0 * ((b - r) / (mx - mn)) + 120.0
    else:
        h = 60.0 * ((r - g) / (mx - mn)) + 240.0
    names = ["red", "yellow", "green", "cyan", "blue", "magenta"]
    return names[int(((h + 30.0) % 360.0) // 60.0)]


def _hue_bins(f):
    """Per-pixel hue sector 0..5 (red, yellow, green, cyan, blue, magenta) of an int RGB image."""
    r = f[:, :, 0].astype(float)
    g = f[:, :, 1].astype(float)
    b = f[:, :, 2].astype(float)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, (60.0 * (g - b) / d) % 360.0,
                 np.where(mx == g, 60.0 * (b - r) / d + 120.0, 60.0 * (r - g) / d + 240.0))
    return (((h + 30.0) % 360.0) // 60.0).astype(int)


def _components(mask):
    """4-connected components of a boolean mask; returns list of (ys, xs) arrays."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    W = mask.shape[1]
    keys = (ys * W + xs).tolist()
    left = set(keys)
    comps = []
    for k in keys:
        if k not in left:
            continue
        left.discard(k)
        stack = [k]
        members = []
        while stack:
            c = stack.pop()
            members.append(c)
            cy, cx = divmod(c, W)
            for n in (c - 1 if cx > 0 else -1, c + 1 if cx < W - 1 else -1, c - W, c + W):
                if n in left:
                    left.discard(n)
                    stack.append(n)
        m = np.array(members)
        comps.append((m // W, m % W))
    return comps


def _backproject(K, E, xs, ys, depth_mm):
    z = depth_mm.astype(float) / 1000.0
    good = z > 0
    xs, ys, z = xs[good], ys[good], z[good]
    if len(z) == 0:
        return np.zeros((0, 3))
    xc = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)
    R = np.asarray(E, dtype=float)[:, :3]
    t = np.asarray(E, dtype=float)[:, 3]
    return (xc - t) @ R


def _blobs(rgb, depth, K, E, min_pix=MIN_BLOB):
    """Saturated colour blobs (cubes) with world-frame geometry."""
    f = np.asarray(rgb).astype(int)
    mx = f.max(2)
    mn = f.min(2)
    sat = (mx - mn) / np.maximum(mx, 1)
    mask = (sat > SAT_MIN) & (mx > VAL_MIN)
    hue = _hue_bins(f)
    comps = []
    for hb in range(6):
        sub = mask & (hue == hb)
        if sub.sum() >= min_pix:
            comps.extend(_components(sub))
    out = []
    for ys, xs in comps:
        if len(ys) < min_pix:
            continue
        cols = f[ys, xs].astype(float)
        bright = cols[cols.max(1) >= np.percentile(cols.max(1), 50)]
        chroma = bright.mean(0) / max(1.0, float(bright.mean(0).sum()))
        P = _backproject(K, E, xs.astype(float), ys.astype(float), np.asarray(depth)[ys, xs])
        blob = {
            "n": int(len(ys)),
            "px": [float(xs.mean()), float(ys.mean())],
            "chroma": [float(c) for c in chroma],
            "name": _rgb_names(chroma),
            "border": bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= f.shape[1] - 2 or ys.max() >= f.shape[0] - 2),
        }
        if len(P) >= 5:
            lo = np.percentile(P[:, :2], 2, axis=0)
            hi = np.percentile(P[:, :2], 98, axis=0)
            ztop = float(np.percentile(P[:, 2], 98))
            top = P[P[:, 2] > P[:, 2].max() - 0.003]
            blob["xy"] = [float(v) for v in (lo + hi) / 2.0]
            blob["ztop"] = ztop
            blob["top_xy"] = [float(v) for v in (top[:, :2].min(0) + top[:, :2].max(0)) / 2.0]
            blob["top_n"] = int(len(top))
        out.append(blob)
    return out


def _chroma_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _front(frame):
    return _blobs(frame.front_rgb, frame.front_depth, frame.front_K, frame.front_extrinsic)


def _defaults(memory):
    m = dict(memory) if isinstance(memory, dict) else {}
    init = {
        "stage": "cue_wait",
        "cue_color": "",
        "cue_chroma": [],
        "cue_first_step": -1,
        "cue_frames": 0,
        "delay_frames": 0,
        "cubes_step": -1,
        "cube_positions": [],
        "cube_colors": [],
        "target_pos": [],
        "target_conf": 0.0,
        "touch_attempts": 0,
        "last_step": -1,
        "notes": "",
        "y_offset": 0.0,
    }
    for k, v in init.items():
        if m.get(k) is None:
            m[k] = v
    return m


def _finger_clearance_offset(best, blobs):
    """TCP y shift (m) that keeps the fingers (opening along world y, ~0.039 m from the TCP)
    off a neighbouring cube. 0 when no other cube lies in a finger's footprint."""
    shift = 0.0
    for b in blobs:
        if b is best or "xy" not in b:
            continue
        dx = b["xy"][0] - best["xy"][0]
        dy = b["xy"][1] - best["xy"][1]
        if abs(dx) < 0.028 + 0.012 and 0.02 < abs(dy) < 0.039 + 0.012 + 0.028:
            shift = shift - (0.008 if dy > 0 else -0.008)
    return float(max(-0.008, min(0.008, shift)))


def perceive_decide(obs, goal, memory, last):
    """Track cue -> delay -> choice phases from the front camera and choose the next act.

    Memory records the cue colour as soon as a single cube is visible, counts delay frames,
    localises the matching cube once the three cubes appear, and tracks touch attempts.
    """
    m = _defaults(memory)
    for fr in obs.frames:
        step = int(fr.step)
        if step <= m["last_step"]:
            continue
        m["last_step"] = step
        if m["stage"] not in ("cue_wait", "delay"):
            continue
        blobs = [b for b in _front(fr) if "xy" in b]
        if m["stage"] == "cue_wait":
            if len(blobs) == 1:
                b = blobs[0]
                if m["cue_frames"] == 0:
                    m["cue_color"] = b["name"]
                    m["cue_chroma"] = b["chroma"]
                    m["cue_first_step"] = step
                m["cue_frames"] = m["cue_frames"] + 1
                continue
            if len(blobs) == 0 and m["cue_frames"] > 0:
                m["stage"] = "delay"
                m["delay_frames"] = 1
                continue
            if len(blobs) == 0:
                continue
        elif m["stage"] == "delay":
            if len(blobs) == 0:
                m["delay_frames"] = m["delay_frames"] + 1
                continue
            if len(blobs) == 1:
                # a single cube again: unexpected; keep waiting but note it
                m["notes"] = "single blob during delay at step %d" % step
                continue
        if len(blobs) >= 2:
            m["cubes_step"] = step
            m["cube_positions"] = [[b["xy"][0], b["xy"][1], b["ztop"]] for b in blobs]
            m["cube_colors"] = [b["name"] for b in blobs]
            if m["cue_chroma"]:
                d = [_chroma_dist(b["chroma"], m["cue_chroma"]) for b in blobs]
                order = sorted(range(len(d)), key=lambda i: d[i])
                best = blobs[order[0]]
                second = d[order[1]] if len(order) > 1 else 1.0
                m["target_conf"] = float(max(0.0, min(1.0, (second - d[order[0]]) / 0.3)))
            else:
                # cue never observed: no evidence; pick the cube nearest the cue location (table centre)
                best = sorted(blobs, key=lambda b: b["xy"][0] ** 2 + b["xy"][1] ** 2)[0]
                m["target_conf"] = 0.0
                m["notes"] = "cue not observed; guessed centre cube"
            zc = max(0.012, best["ztop"] - CUBE_HALF)
            m["target_pos"] = [best["xy"][0], best["xy"][1], zc]
            m["y_offset"] = _finger_clearance_offset(best, blobs)
            m["stage"] = "touch"
    remaining = STEP_LIMIT - m["last_step"] if m["last_step"] >= 0 else STEP_LIMIT
    if m["stage"] in ("cue_wait", "delay"):
        act = {"tool": "act_wait", "args": {}}
    else:
        attempt = int(m["touch_attempts"])
        m["touch_attempts"] = attempt + 1
        mode = ["grasp", "open_hold", "grasp", "open_hold"][min(attempt, 3)]
        m["stage"] = "touch"
        act = {"tool": "act_touch", "args": {"target": m["target_pos"], "chroma": m["cue_chroma"],
                                             "mode": mode, "max_steps": max(4, remaining + 2),
                                             "reuse_tcp_xy": attempt > 0,
                                             "y_offset": m["y_offset"]}}
    return {"memory": m, "act": act}


def act_wait(robot):
    """Send one zero-motion native step (gripper open) while the cue/delay phases play."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"waited": 1, "step": int(f.step)}


def _rpy_matrix(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:
        # near pi: use the diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return axis * ang
    return v / (2.0 * s) * ang


def _wrist_refine(frame, chroma, tcp_z):
    """Target cube top-face centre from the wrist camera when the whole cube is in view."""
    if not chroma or tcp_z < 0.09:
        return None
    blobs = _blobs(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic, min_pix=150)
    cands = [b for b in blobs if "top_xy" in b and not b["border"] and b.get("top_n", 0) >= 40
             and _chroma_dist(b["chroma"], chroma) < 0.2 and abs(b["ztop"] - 2 * CUBE_HALF) < 0.012]
    if len(cands) != 1:
        return None
    return cands[0]["top_xy"]


def act_touch(robot, target, chroma=None, mode="grasp", max_steps=20, gain=2.0, damping=0.5, reuse_tcp_xy=False, y_offset=0.0):
    """Closed-loop TCP approach to the target cube centre, then touch it.

    target: [x, y, z_centre] world metres. The TCP stays high until it is above the cube,
    descends to the cube centre height with fingers open (cube between the fingers) and then
    mode='grasp' closes the gripper on the cube and holds still; mode='open_hold' holds still
    with the gripper open. Returns measured TCP error, finger opening and step counts.
    """
    tgt = np.array([float(target[0]), float(target[1]), float(target[2])])
    tgt[2] = max(0.015, tgt[2] - 0.002)
    f0 = robot.now()
    p0 = np.array(f0.tcp[:3], dtype=float)
    if reuse_tcp_xy and float(np.linalg.norm(p0[:2] - tgt[:2])) < 0.015:
        # a previous attempt already centred the TCP on the (wrist-refined) cube; stay there
        tgt[0] = p0[0]
        tgt[1] = p0[1]
    R0 = _rpy_matrix(f0.tcp[3:])
    yaw0 = math.atan2(R0[1, 0], R0[0, 0])
    # desired orientation: fingers pointing straight down, current yaw kept (roll = pi)
    Rdes = _rpy_matrix([math.pi, 0.0, yaw0])
    prev = np.array(f0.tcp[:3], dtype=float)
    vel = np.zeros(3)
    closing = False
    arrived_at = -1
    hold = 0
    refined = 0
    log = []
    grip = 1.0
    steps = 0
    hold_needed = 4 if mode == "grasp" else 3
    for i in range(int(max_steps)):
        fr = robot.now()
        p = np.array(fr.tcp[:3], dtype=float)
        if i > 0:
            vel = p - prev
        prev = p
        if arrived_at < 0:
            w = _wrist_refine(fr, chroma, p[2])
            if w is not None and np.linalg.norm(np.array(w) - tgt[:2]) < 0.03:
                tgt[0] = 0.5 * tgt[0] + 0.5 * w[0]
                tgt[1] = 0.5 * tgt[1] + 0.5 * w[1]
                refined += 1
        aim = np.array([tgt[0], tgt[1] + (0.0 if reuse_tcp_xy else float(y_offset))])
        exy = float(np.linalg.norm(aim - p[:2]))
        zgoal = tgt[2] + min(0.12, 2.5 * max(0.0, exy - 0.008))
        goal = np.array([aim[0], aim[1], zgoal])
        e = goal - p
        u = np.clip((gain * e - damping * vel) / MOVE_SCALE, -1.0, 1.0)
        # orientation hold: base-frame rotation error, command = -k*err/0.1 (angular offset is -0.1*r)
        Rc = _rpy_matrix(fr.tcp[3:])
        werr = _rotvec(Rdes @ Rc.T)
        r = np.clip(-0.8 * werr / ROT_SCALE, -0.5, 0.5)
        if np.linalg.norm(werr) > 1.0:
            r = np.zeros(3)
        near = exy < 0.008 and abs(p[2] - tgt[2]) < 0.012
        if near and arrived_at < 0:
            arrived_at = i
        if mode == "grasp" and arrived_at >= 0:
            closing = True
        if (mode == "grasp" and not closing and i >= int(max_steps) - 5 and exy < 0.015
                and p[2] < tgt[2] + 0.02):
            closing = True   # running out of native steps: close on the cube even if not fully settled
        grip = -1.0 if closing else 1.0
        if arrived_at >= 0:
            hold += 1
        log.append([round(float(v), 4) for v in p] + [round(float(fr.fingers[0]), 4)])
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
        if hold >= hold_needed:
            break
    fr = robot.now()
    p = np.array(fr.tcp[:3], dtype=float)
    return {"mode": mode, "steps": steps, "arrived_at": arrived_at, "refined": refined,
            "target": [round(float(v), 4) for v in tgt],
            "tcp": [round(float(v), 4) for v in p], "err": round(float(np.linalg.norm(p - tgt)), 4),
            "fingers": round(float(fr.fingers[0]), 4), "end_step": int(fr.step), "trace": log[-6:]}
