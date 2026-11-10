"""RememberShape5: remember the cue shape, then touch the candidate of the same shape.

Perception: blue-pixel segmentation of the wrist RGB, back-projection with wrist depth,
intrinsics and extrinsic to world points, per-blob 3D features (height, top-face fraction,
max planar extent, enclosed hole) and class prototypes; candidates are jointly assigned
to the five distinct classes. Control: closed-loop delta-pose steps toward a hover point
above the matching candidate, then a press with closed fingers onto its top face.
"""

CLASSES = ["cross", "torus", "sphere", "cube", "cylinder"]
# prototype feature vectors: [top height m, top-face fraction, max planar extent m, hole flag]
# (class index, vector). Resting poses measured in all 50 demos; "torus on its rim" observed in R011
# (ep 43); "cylinder lying" is an untested geometric estimate (diameter 4 cm, length 4 cm).
PROTO = [
    (0, [0.016, 0.95, 0.061, 0.0]),   # cross, flat
    (1, [0.016, 0.95, 0.056, 1.0]),   # torus, flat with hole
    (1, [0.055, 0.55, 0.055, 0.0]),   # torus standing on its rim
    (2, [0.040, 0.40, 0.039, 0.0]),   # sphere
    (3, [0.041, 0.97, 0.055, 0.0]),   # cube (top diagonal ~5.5 cm)
    (4, [0.041, 0.95, 0.040, 0.0]),   # cylinder upright
    (4, [0.040, 0.60, 0.057, 0.0]),   # cylinder lying (hypothesis)
]
SCALE = [0.005, 0.15, 0.004, 0.3]
SETTLE_MARGIN = 4.0  # wait (max 2 steps) if another candidate is within this cost of being the target
MAX_SETTLE_WAITS = 2
MIN_BLOB = 25


def _blue_mask(img):
    im = img.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (b > 90) & (b > 2 * r + 30) & (b > 2 * g + 30)


def _components(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0]:
            continue
        n = len(comps) + 1
        lab[y0, x0] = n
        stack = [(y0, x0)]
        py, px = [], []
        while stack:
            y, x = stack.pop()
            py.append(y)
            px.append(x)
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        comps.append((np.array(py), np.array(px)))
    return comps


def _holes(sub):
    H, W = sub.shape
    m = np.zeros((H + 2, W + 2), dtype=bool)
    m[1:-1, 1:-1] = sub
    seen = np.zeros_like(m)
    seen[0, 0] = True
    stack = [(0, 0)]
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < H + 2 and 0 <= xx < W + 2 and (not m[yy, xx]) and not seen[yy, xx]:
                seen[yy, xx] = True
                stack.append((yy, xx))
    return int(((~m) & (~seen)).sum())


def _to_world(depth, K, E, ys, xs):
    z = depth[ys, xs].astype(np.float64) / 1000.0
    u = (xs - K[0, 2]) / K[0, 0] * z
    v = (ys - K[1, 2]) / K[1, 1] * z
    pc = np.stack([u, v, z], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t) @ R


def _blobs(frame):
    """Blue objects seen by the wrist camera, with world-frame 3D shape features."""
    img, depth = frame.wrist_rgb, frame.wrist_depth
    K, E = frame.wrist_K, frame.wrist_extrinsic
    mask = _blue_mask(img)
    out = []
    for ys, xs in _components(mask):
        if len(ys) < MIN_BLOB:
            continue
        ok = depth[ys, xs] > 0
        if ok.sum() < MIN_BLOB:
            continue
        P = _to_world(depth, K, E, ys[ok], xs[ok])
        zmax = float(np.percentile(P[:, 2], 99))
        top = P[P[:, 2] > zmax - 0.004]
        g = np.unique(np.floor(P[:, :2] / 0.002).astype(np.int64), axis=0)
        gt = np.unique(np.floor(top[:, :2] / 0.002).astype(np.int64), axis=0)
        topfrac = len(gt) / max(len(g), 1)
        xy = P[:, :2] - P[:, :2].mean(0)
        ext = 0.0
        for a in np.arange(0, np.pi, np.pi / 36):
            d = xy @ np.array([np.cos(a), np.sin(a)])
            ext = max(ext, float(d.max() - d.min()))
        sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        # only this component's pixels
        own = np.zeros_like(sub)
        own[ys - ys.min(), xs - xs.min()] = True
        holes = _holes(own)
        touches_edge = bool(ys.min() <= 0 or xs.min() <= 0 or ys.max() >= img.shape[0] - 1 or xs.max() >= img.shape[1] - 1)
        out.append({
            "npix": int(len(ys)), "zmax": zmax, "topfrac": float(topfrac), "ext": ext, "holes": holes,
            "top_xy": [float(v) for v in top[:, :2].mean(0)], "edge": touches_edge,
            "px": [float(xs.mean()), float(ys.mean())],
        })
    return out


def _costs(b):
    """Per-class cost: minimum scaled squared distance to that class's prototypes."""
    f = np.array([b["zmax"], b["topfrac"], b["ext"], 1.0 if b["holes"] >= 2 else 0.0])
    S = np.array(SCALE)
    out = np.full(len(CLASSES), 1e9)
    for k, v in PROTO:
        c = float((((f - np.array(v)) / S) ** 2).sum())
        if c < out[k]:
            out[k] = c
    return out


def _perms(n, k):
    if k == 0:
        return [[]]
    out = []
    for p in _perms(n, k - 1):
        for i in range(n):
            if i not in p:
                out.append(p + [i])
    return out


def _assign(cands, cls=None):
    """Jointly assign distinct shape classes to candidates (minimum total prototype cost).
    With cls given, also return the cost gap to the best assignment that gives cls to a
    different candidate (confidence of the target choice)."""
    C = [_costs(b) for b in cands]
    k = CLASSES.index(cls) if cls in CLASSES else None
    best, bestp = None, None
    alt = {}
    for p in _perms(len(CLASSES), len(cands)):
        s = sum(float(C[i][p[i]]) for i in range(len(cands)))
        if best is None or s < best:
            best, bestp = s, p
        if k is not None:
            who = p.index(k) if k in p else -1
            if who not in alt or s < alt[who]:
                alt[who] = s
    margin = None
    if k is not None:
        mine = bestp.index(k) if k in bestp else -1
        others = [v for w, v in alt.items() if w != mine]
        margin = (min(others) - best) if others else 1e9
    return [CLASSES[i] for i in bestp], C, margin


def _fmt(b):
    return "z%.3f tf%.2f ext%.3f h%d n%d xy(%.3f,%.3f)" % (
        b["zmax"], b["topfrac"], b["ext"], b["holes"], b["npix"], b["top_xy"][0], b["top_xy"][1])


def perceive_decide(obs, goal, memory, last):
    """Record the cue shape when a single object is shown; when several candidates appear,
    choose the one with the same shape and touch it."""
    mem = {
        "stage": memory.get("stage") or "wait_cue",
        "cue_class": memory.get("cue_class") or "",
        "cue_step": memory.get("cue_step", -1) if memory.get("cue_step") is not None else -1,
        "cue_margin": memory.get("cue_margin") or 0.0,
        "cue_info": memory.get("cue_info") or "",
        "target": memory.get("target") or [],
        "target_class": memory.get("target_class") or "",
        "cands_info": memory.get("cands_info") or "",
        "frames_seen": memory.get("frames_seen") or 0,
        "touch_tries": memory.get("touch_tries") or 0,
        "settle_waits": memory.get("settle_waits") or 0,
        "log": memory.get("log") or "",
    }
    frames = list(obs.frames)
    mem["frames_seen"] = int(mem["frames_seen"]) + len(frames)
    if mem["stage"] in ("wait_cue", "wait_cands"):
        for fr in frames:
            bl = _blobs(fr)
            nb = len([b for b in bl if not b["edge"]])
            if nb == 1 and mem["stage"] in ("wait_cue", "wait_cands"):
                # cue frame: keep the most confident classification over all cue frames
                b = [b for b in bl if not b["edge"]][0]
                c = _costs(b)
                order = np.argsort(c)
                margin = float(c[order[1]] - c[order[0]])
                if not mem["cue_class"] or margin > float(mem["cue_margin"]):
                    mem["cue_class"] = CLASSES[int(order[0])]
                    mem["cue_margin"] = margin
                    mem["cue_step"] = int(fr.step)
                    mem["cue_info"] = _fmt(b)
                mem["stage"] = "wait_cands"
            elif nb >= 3 and mem["stage"] == "wait_cands" or (nb >= 3 and fr is frames[-1]):
                cands = bl[:5] if len(bl) <= 5 else sorted(bl, key=lambda b: -b["npix"])[:5]
                labels, C, margin = _assign(cands, mem["cue_class"])
                worst = max(float(C[i][CLASSES.index(l)]) for i, l in enumerate(labels))
                unsure = margin is not None and margin < SETTLE_MARGIN
                if unsure and int(mem["settle_waits"]) < MAX_SETTLE_WAITS and fr is frames[-1]:
                    # objects may still be falling into their resting pose: look again next step
                    mem["settle_waits"] = int(mem["settle_waits"]) + 1
                    mem["log"] = (mem["log"] + " settle-wait@%d(cost %.1f margin %.1f)" % (fr.step, worst, margin))[-400:]
                    mem["stage"] = "wait_cands"
                    continue
                cls = mem["cue_class"]
                idx = labels.index(cls) if cls in labels else None
                if idx is None:
                    # cue unseen or class missing: nearest-cost candidate for the cue class (or first)
                    k = CLASSES.index(cls) if cls in CLASSES else 0
                    idx = int(np.argmin([float(c[k]) for c in C]))
                tb = cands[idx]
                mem["target"] = [tb["top_xy"][0], tb["top_xy"][1], tb["zmax"]]
                mem["target_class"] = labels[idx]
                mem["cands_info"] = "; ".join(l + ":" + _fmt(b) for l, b in zip(labels, cands)) + " @step %d worst %.1f margin %.1f" % (fr.step, worst, margin if margin is not None else -1)
                mem["stage"] = "touch"
                break
    if mem["stage"] == "touch":
        tries = int(mem["touch_tries"])
        mem["touch_tries"] = tries + 1
        x, y, zt = mem["target"]
        return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": x, "y": y, "ztop": zt, "attempt": tries}}}
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 1}}}


def act_wait(robot, n=1):
    """Hold still for n native steps (gripper commanded open). Returns the step index reached."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"step": int(f.step), "tcp": [float(v) for v in f.tcp[:3]]}


GAIN = 1.5          # command = (GAIN*error - DAMP*last_move)/0.1; fit: move ~0.27*0.1*cmd + 0.31*previous move
DAMP = 0.0          # damping 1.0 (with GAIN 2.0) produced x oscillation in R025 -> disabled
XY_GATE = 0.010     # descend below hover height only once the xy error is this small
HOVER_DZ = 0.040    # hover height of the TCP above the target's top face
GRASP_BELOW = 0.020 # straddle height: TCP this far below the top face ...
MIN_Z = 0.012       # ... but never below this (fingertips reach ~9 mm below the TCP)
CLOSE_EARLY_DZ = 0.012  # start closing once the TCP is below min(straddle z + this, top face)
CLOSE_XY = 0.012        # ... and within this xy distance of the target (finger gap is 8 cm)


def _log_step(robot, cmd, grip, log):
    f = robot.now()
    log.append([int(f.step)] + [round(float(v), 4) for v in f.tcp[:3]] + [round(float(v), 2) for v in cmd]
               + [round(float(grip), 1), round(float(f.fingers[0]), 3)])


def _velocity(robot):
    """Measured TCP displacement over the last native step (m/step); zero if unknown."""
    fr = robot.frames()
    if len(fr) >= 2:
        return np.asarray(fr[-1].tcp[:3], dtype=np.float64) - np.asarray(fr[-2].tcp[:3], dtype=np.float64)
    return np.zeros(3)


def _move(robot, goal, grip, log):
    tcp = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    v = _velocity(robot)
    cmd = np.clip((GAIN * (np.asarray(goal, dtype=np.float64) - tcp) - DAMP * v) / 0.1, -1.0, 1.0)
    robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, float(grip))
    _log_step(robot, cmd, grip, log)


def act_touch(robot, x, y, ztop, attempt=0):
    """Open the gripper and move toward (x, y) while descending to hover (ztop + 0.04); once
    within 0.01 m in xy, descend so the open fingers straddle the object (TCP z =
    max(0.012, ztop - 0.02)), then close the fingers on it (lateral finger contact) while
    holding position. Later attempts first open and lift, and use small xy offsets."""
    log = []
    f0 = robot.now()
    a = int(attempt)
    offs = [(0.0, 0.0), (0.008, 0.0), (-0.008, 0.0), (0.0, 0.008), (0.0, -0.008)]
    ox, oy = offs[a % len(offs)]
    gx, gy = float(x) + ox, float(y) + oy
    hover_z = float(ztop) + HOVER_DZ
    grasp_z = max(MIN_Z, float(ztop) - GRASP_BELOW)
    reach = None
    if a == 0:
        try:
            r = robot.dry_run(gx, gy, grasp_z, float(f0.tcp[3]), float(f0.tcp[4]), float(f0.tcp[5]))
            reach = {k: r.get(k) for k in ("reachable", "path_clear", "joint_margin", "message")}
        except Exception as e:  # a failed preview must not stop the touch
            reach = {"error": str(e)[:120]}
    if a > 0:
        for _ in range(2):  # open and lift before retrying
            tcp = robot.now().tcp
            _move(robot, [float(tcp[0]), float(tcp[1]), hover_z], 1.0, log)
    reached = False
    descending = False
    for _ in range(16):
        tcp = np.asarray(robot.now().tcp[:3], dtype=np.float64)
        exy = math.hypot(gx - tcp[0], gy - tcp[1])
        descending = exy < (2.0 * XY_GATE if descending else XY_GATE)
        zg = grasp_z if descending else hover_z
        err = np.array([gx, gy, zg]) - tcp
        if descending and float(np.max(np.abs(err))) < 0.004:
            reached = True
            break
        if descending and tcp[2] < min(grasp_z + CLOSE_EARLY_DZ, float(ztop)) and exy < CLOSE_XY:
            reached = True  # close while finishing the descent (saves the settling steps)
            break
        _move(robot, [gx, gy, zg], 1.0, log)
    for _ in range(4):  # close on the object while holding the straddle pose
        _move(robot, [gx, gy, grasp_z], -1.0, log)
    f = robot.now()
    return {"attempt": a, "goal": [gx, gy, grasp_z], "reach": reach, "reached": reached,
            "final_tcp": [float(v) for v in f.tcp[:3]], "fingers": [float(v) for v in f.fingers],
            "step": int(f.step), "trace": log[-18:]}
