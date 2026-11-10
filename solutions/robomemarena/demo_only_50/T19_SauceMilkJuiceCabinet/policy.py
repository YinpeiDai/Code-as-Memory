"""T19 SauceMilkJuiceCabinet: move tomato sauce, milk, orange juice (in that order)
from the top of cabinet1 (left, -y) to the top of cabinet2 (right, +y).

Perception: agentview depth -> cabinet top boxes; wrist RGB-D -> object tops.
Control: closed-loop OSC delta servo on measured TCP, fixed top-down orientation.
"""

ORDER = ["tomato_sauce", "milk", "orange_juice"]
# release slots relative to the visible-top centre of cabinet2 (from 48 demos, median)
SLOTS = {"tomato_sauce": [-0.029, 0.053, 0.760],
         "milk": [0.063, 0.033, 0.767],
         "orange_juice": [-0.032, -0.030, 0.768]}
# expert grasp point minus perceived object top-centre (wrist view, hover at z~0.86)
GRASP_BIAS = [0.008, 0.002]
DEFAULT_TOP = 0.665
TRANSIT_Z = 0.95
SURVEY_Z = 0.93
HOVER_Z = 0.86
WRIST_CAM_DX = 0.05   # wrist camera sits ~5 cm +x of the TCP


# ---------------------------------------------------------------- geometry
def _cloud(depth, K, E):
    z = np.asarray(depth, dtype=np.float64) / 1000.0
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    pw = (pc - t) @ R
    pw[z <= 0] = np.nan
    return pw


def _classify(rgb):
    rgb = np.asarray(rgb)
    R = rgb[..., 0].astype(np.float64)
    r = R + 1e-3
    g = rgb[..., 1] / r
    b = rgb[..., 2] / r
    yellow = (R > 60) & (g > 0.7) & (b < 0.45)
    red = (R > 60) & (g < 0.3) & (b < 0.2)
    orange = (R > 60) & (g >= 0.3) & (g < 0.7) & (b < 0.2)
    white = (R > 85) & (g > 0.85) & (b > 0.8) & (b < 1.06)
    return yellow, red, orange, white


def _cabinets(frame):
    """Cabinet top boxes from the agentview depth (dark pixels on the top plane)."""
    pw = _cloud(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    rgb = np.asarray(frame.agentview_rgb).astype(np.int64)
    z = pw[..., 2]
    ok = np.isfinite(z) & (np.abs(pw[..., 0]) < 0.4) & (rgb.sum(-1) < 200)
    # top height: mode of dark points in a plausible band
    cand = z[ok & (z > 0.55) & (z < 0.8)]
    top = DEFAULT_TOP
    if cand.size > 200:
        h, edges = np.histogram(cand, bins=np.arange(0.55, 0.80, 0.004))
        k = int(np.argmax(h))
        sel = cand[(cand >= edges[k] - 0.004) & (cand <= edges[k + 1] + 0.004)]
        top = float(np.median(sel))
        if abs(top - DEFAULT_TOP) > 0.04:
            top = DEFAULT_TOP
    m = ok & (np.abs(z - top) < 0.008)
    P = pw[m]
    out = {"top": top}
    # split the two cabinets at the widest empty y-gap between them
    split = 0.06
    edges = np.arange(-0.10, 0.25, 0.005)
    h, _ = np.histogram(P[:, 1], bins=edges)
    best, run, start = 0, 0, 0
    for i in range(len(h)):
        if h[i] <= 2:
            run += 1
            if run > best:
                best = run
                start = i - run + 1
        else:
            run = 0
    if best >= 3:
        split = float(edges[start] + 0.5 * best * 0.005)
    out["split"] = split
    for name, sel in (("c1", P[:, 1] < split), ("c2", P[:, 1] > split)):
        Q = P[sel]
        if len(Q) < 50:
            out[name] = None
            continue
        out[name] = [float(np.percentile(Q[:, 0], 2)), float(np.percentile(Q[:, 1], 2)),
                     float(np.percentile(Q[:, 0], 98)), float(np.percentile(Q[:, 1], 98))]
    return out


def _detect(frame, top, box):
    """Object top-centres on cabinet1 from one wrist RGB-D frame.

    Returns {name: [x, y, ztop, npix]} for detected objects.
    sauce: pixels higher than top+0.072 that are not carton coloured (bottle lid);
    milk: red/white carton pixels; juice: yellow/orange carton pixels.
    """
    pw = _cloud(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    yellow, red, orange, white = _classify(frame.wrist_rgb)
    z = pw[..., 2]
    pad = 0.02
    zmax = min(top + 0.22, float(np.asarray(frame.tcp)[2]) - 0.035)
    rows = np.arange(z.shape[0])[:, None] * np.ones((1, z.shape[1]))
    ok = (np.isfinite(z) & (z < zmax) & (rows < 205) & (pw[..., 0] > box[0] - pad) & (pw[..., 0] < box[2] + pad)
          & (pw[..., 1] > box[1] - pad) & (pw[..., 1] < box[3] + pad)
          & (z > top + 0.012) & (z < top + 0.22))
    juicec = yellow | orange
    milkc = red | white
    res = {}
    rgbi = np.asarray(frame.wrist_rgb).astype(np.float64)
    mx = rgbi.max(-1)
    mn = rgbi.min(-1)
    grey = (mx > 50) & ((mx - mn) < 0.3 * mx + 10) & (rgbi[..., 2] - rgbi[..., 0] >= 6)
    sauce = ok & (z > top + 0.0725) & grey & ~juicec & ~milkc
    for name, m in (("tomato_sauce", sauce), ("milk", ok & milkc), ("orange_juice", ok & juicec)):
        n = int(m.sum())
        if n < (80 if name == "tomato_sauce" else 25):
            continue
        P = pw[m]
        if name != "tomato_sauce" and "tomato_sauce" in res:
            s = res["tomato_sauce"]
            far = np.hypot(P[:, 0] - s[0], P[:, 1] - s[1]) > 0.035
            P = P[far]
            if len(P) < 25:
                continue
        zt = float(np.percentile(P[:, 2], 95))
        Q = P[P[:, 2] > zt - 0.012]
        if len(Q) < 10:
            Q = P
        res[name] = [float(np.median(Q[:, 0])), float(np.median(Q[:, 1])), zt, int(len(P))]
    # milk vs juice: a carton cluster cannot be both; drop the weaker if centres coincide
    if "milk" in res and "orange_juice" in res:
        a, b = res["milk"], res["orange_juice"]
        if math.hypot(a[0] - b[0], a[1] - b[1]) < 0.02:
            if a[3] < b[3]:
                res.pop("milk")
            else:
                res.pop("orange_juice")
    return res


# ---------------------------------------------------------------- rotation
def _aa2mat(v):
    v = np.asarray(v, dtype=np.float64)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _mat2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-4:
        return np.zeros(3)
    return w * th / (2 * s)


def _rot_cmd(tcp):
    """Axis-angle command (normalised) that turns the hand to the top-down grasp pose."""
    Rt = np.diag([1.0, -1.0, -1.0])
    Rc = _aa2mat(tcp[3:6])
    err = _mat2aa(Rt @ Rc.T)
    return np.clip(err / 0.5 * 1.0, -0.3, 0.3)


# ---------------------------------------------------------------- servo
def _servo(robot, target, grip, tol=0.004, max_steps=200, vmax=1.0, kp=1.0, settle=2):
    """P-servo the TCP to `target` (world xyz). Returns (steps, final error)."""
    target = np.asarray(target, dtype=np.float64)
    steps = 0
    good = 0
    err = 1.0
    while steps < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        d = target - tcp[:3]
        err = float(np.linalg.norm(d))
        if err < tol:
            good += 1
            if good >= settle:
                break
        else:
            good = 0
        a = np.clip(d / 0.05 * kp, -vmax, vmax)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
    return steps, err


def _hold(robot, n, grip):
    for _ in range(n):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        r = _rot_cmd(tcp)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), grip)


def _gap(robot):
    f = np.asarray(robot.now().fingers, dtype=np.float64)
    return float(f[0] - f[1])


def _transit(robot, x, y, z, grip, high=TRANSIT_Z):
    """Rise to `high`, move horizontally, then go to z (safe path between cabinets)."""
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    n = 0
    if math.hypot(tcp[0] - x, tcp[1] - y) > 0.03 and tcp[2] < high - 0.02:
        s, _ = _servo(robot, [tcp[0], tcp[1], high], grip, tol=0.02, max_steps=120, settle=1)
        n += s
    if math.hypot(tcp[0] - x, tcp[1] - y) > 0.03:
        s, _ = _servo(robot, [x, y, max(high, z)], grip, tol=0.012, max_steps=200, settle=1)
        n += s
    s, e = _servo(robot, [x, y, z], grip, tol=0.004, max_steps=150)
    n += s
    return n, e


# ---------------------------------------------------------------- act tools
def act_move(robot, x, y, z, gripper=-1.0):
    """Safe move of the TCP to world (x, y, z): rise to transit height, move across, descend.
    Keeps the top-down hand orientation; gripper -1 opens, +1 closes. Returns steps, error, tcp."""
    n, e = _transit(robot, x, y, z, gripper)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return {"steps": n, "error": e, "tcp": tcp[:3].tolist(), "gap": _gap(robot)}


def act_pick(robot, x, y, z_grasp, z_lift=1.0):
    """From a hover above the object: align over (x, y), descend open to z_grasp, close,
    lift to z_lift. Returns whether the fingers stopped on an object (gap)."""
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    _servo(robot, [x, y, tcp[2]], -1.0, tol=0.003, max_steps=60)
    _servo(robot, [x, y, z_grasp + 0.05], -1.0, tol=0.004, max_steps=100, settle=1)
    _servo(robot, [x, y, z_grasp], -1.0, tol=0.003, max_steps=90, kp=0.5)
    zg = float(robot.now().tcp[2])
    _hold(robot, 30, 1.0)
    g1 = _gap(robot)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    _servo(robot, [tcp[0], tcp[1], z_lift], 1.0, tol=0.01, max_steps=120, settle=1)
    _hold(robot, 5, 1.0)
    g2 = _gap(robot)
    return {"held": bool(0.012 < g2 < 0.075), "gap_closed": g1, "gap_lifted": g2, "z_reached": zg,
            "tcp": np.asarray(robot.now().tcp)[:3].tolist()}


def act_place(robot, x, y, z_release, z_lift=0.95):
    """Carry the held object at transit height to (x, y), descend to z_release, open,
    wait, and rise to z_lift. Returns gap before release and final tcp."""
    _transit(robot, x, y, z_release + 0.06, 1.0)
    _servo(robot, [x, y, z_release], 1.0, tol=0.004, max_steps=80, kp=0.8)
    g = _gap(robot)
    _hold(robot, 20, -1.0)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    _servo(robot, [tcp[0], tcp[1], z_lift], -1.0, tol=0.01, max_steps=100, settle=1)
    return {"gap_before_release": g, "tcp": np.asarray(robot.now().tcp)[:3].tolist()}


def act_wait(robot, steps=40):
    """Hold still with the gripper open for `steps` native steps (monitoring period)."""
    _hold(robot, int(steps), -1.0)
    return {"waited": int(steps)}


# ---------------------------------------------------------------- decision
def _blank_memory():
    return {"phase": "init", "stage": 0, "top": DEFAULT_TOP, "cab1": [], "cab2": [],
            "target": [], "fails": 0, "seen": 0, "log": ""}


def perceive_decide(obs, goal, memory, last):
    """State machine: init -> survey -> approach -> pick -> place (x3) -> done.
    Perception faults never end the episode: they are logged and the robot waits briefly."""
    try:
        return _decide(obs, goal, memory, last)
    except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError, ArithmeticError) as exc:
        mem = _blank_memory()
        for k in mem:
            if isinstance(memory, dict) and memory.get(k) is not None:
                mem[k] = memory[k]
        mem["log"] = (str(mem["log"]) + " | error: " + repr(exc)[:140])[-1500:]
        mem["seen"] = int(mem["seen"]) + 1
        if int(mem["seen"]) > 3 and mem["phase"] not in ("init", "done"):
            mem["phase"] = "survey"
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 5}}}


def _decide(obs, goal, memory, last):
    mem = _blank_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    fr = obs.current
    phase = mem["phase"]
    log = []

    def survey_act():
        c1 = mem["cab1"]
        cx = 0.5 * (c1[0][0] + c1[1][0])
        cy = 0.5 * (c1[0][1] + c1[1][1])
        return {"tool": "act_move", "args": {"x": cx - WRIST_CAM_DX, "y": cy, "z": SURVEY_Z, "gripper": -1.0}}

    def done(m, act):
        m["log"] = (m["log"] + " | " + "; ".join(log))[-1500:]
        return {"memory": m, "act": act}

    if phase == "init":
        cab = _cabinets(fr)
        mem["top"] = cab["top"]
        c1 = cab["c1"] or [-0.12, -0.30, 0.12, -0.11]
        c2 = cab["c2"] or [-0.12, 0.17, 0.12, 0.35]
        mem["cab1"] = [[c1[0], c1[1]], [c1[2], c1[3]]]
        mem["cab2"] = [[c2[0], c2[1]], [c2[2], c2[3]]]
        log.append("init top=%.3f c1=%s c2=%s" % (cab["top"], str(np.round(c1, 3).tolist()), str(np.round(c2, 3).tolist())))
        mem["phase"] = "survey"
        return done(mem, survey_act())

    stage = int(mem["stage"])
    if stage >= len(ORDER) or phase == "done":
        mem["phase"] = "done"
        return done(mem, {"tool": "act_wait", "args": {"steps": 60}})
    name = ORDER[stage]
    box = [mem["cab1"][0][0], mem["cab1"][0][1], mem["cab1"][1][0], mem["cab1"][1][1]]
    top = float(mem["top"])

    if phase == "survey":
        det = _detect(fr, top, box)
        log.append("survey s%d %s" % (stage, {k: [round(v, 3) for v in d[:3]] for k, d in det.items()}))
        if name in det:
            d = det[name]
            mem["target"] = [d[0] + GRASP_BIAS[0], d[1] + GRASP_BIAS[1], d[2]]
            mem["phase"] = "approach"
            return done(mem, {"tool": "act_move", "args": {"x": mem["target"][0], "y": mem["target"][1], "z": HOVER_Z, "gripper": -1.0}})
        mem["fails"] = int(mem["fails"]) + 1
        if int(mem["fails"]) > 6:
            log.append("give up on " + name)
            mem["stage"] = stage + 1
            mem["fails"] = 0
            return done(mem, survey_act())
        # look again from a shifted, higher viewpoint
        c1 = mem["cab1"]
        cx = 0.5 * (c1[0][0] + c1[1][0]) - WRIST_CAM_DX + (0.04 if int(mem["fails"]) % 2 else -0.04)
        cy = 0.5 * (c1[0][1] + c1[1][1])
        return done(mem, {"tool": "act_move", "args": {"x": cx, "y": cy, "z": SURVEY_Z + 0.05, "gripper": -1.0}})

    if phase == "approach":
        det = _detect(fr, top, box)
        tgt = list(mem["target"])
        if name in det:
            d = det[name]
            cand = [d[0] + GRASP_BIAS[0], d[1] + GRASP_BIAS[1], d[2]]
            if math.hypot(cand[0] - tgt[0], cand[1] - tgt[1]) < 0.06:
                tgt = cand
        log.append("approach %s tgt=%s" % (name, str(np.round(tgt, 3).tolist())))
        ztop = float(tgt[2])
        zg = max(top + 0.038, ztop - 0.035)
        zg = min(zg, top + 0.12)
        mem["target"] = [tgt[0], tgt[1], ztop]
        mem["phase"] = "pick"
        return done(mem, {"tool": "act_pick", "args": {"x": tgt[0], "y": tgt[1], "z_grasp": zg, "z_lift": 1.0}})

    if phase == "pick":
        g = np.asarray(fr.fingers, dtype=np.float64)
        gap = float(g[0] - g[1])
        if isinstance(last, str) and '"held": true' in last:
            held = True
        elif isinstance(last, str) and '"held": false' in last:
            held = False
        else:
            held = 0.012 < gap < 0.072   # closed on an object (empty ~0, open ~0.078)
        log.append("pick %s held=%s" % (name, held))
        if held:
            c2 = mem["cab2"]
            cx = 0.5 * (c2[0][0] + c2[1][0])
            cy = 0.5 * (c2[0][1] + c2[1][1])
            s = SLOTS[name]
            mem["phase"] = "place"
            return done(mem, {"tool": "act_place", "args": {"x": cx + s[0], "y": cy + s[1], "z_release": top + (s[2] - DEFAULT_TOP), "z_lift": TRANSIT_Z}})
        mem["fails"] = int(mem["fails"]) + 1
        mem["phase"] = "survey"
        if int(mem["fails"]) > 6:
            mem["stage"] = stage + 1
            mem["fails"] = 0
        return done(mem, survey_act())

    if phase == "place":
        lost = False
        if isinstance(last, str) and '"gap_before_release": ' in last:
            txt = last.split('"gap_before_release": ')[1].split(",")[0].split("}")[0]
            try:
                lost = float(txt) < 0.010
            except ValueError:
                lost = False
        if lost:
            log.append("lost %s in transit" % name)
            mem["fails"] = int(mem["fails"]) + 1
            mem["phase"] = "survey"
            if int(mem["fails"]) > 6:
                mem["stage"] = stage + 1
                mem["fails"] = 0
            return done(mem, survey_act())
        mem["stage"] = stage + 1
        mem["fails"] = 0
        log.append("placed %s" % name)
        if stage + 1 >= len(ORDER):
            mem["phase"] = "done"
            return done(mem, {"tool": "act_wait", "args": {"steps": 60}})
        mem["phase"] = "survey"
        return done(mem, survey_act())

    mem["phase"] = "survey"
    return done(mem, survey_act())
