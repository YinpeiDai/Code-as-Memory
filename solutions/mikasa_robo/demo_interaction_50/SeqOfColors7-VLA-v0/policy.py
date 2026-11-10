"""SeqOfColors7-VLA policy: remember the cued colours, touch every matching ring cube, then press the button.

Perception uses only the front RGB-D camera with its calibration; control uses the measured TCP.
"""

NAMES = ["red", "darkred", "yellow", "gold", "green", "teal", "cyan", "blue", "magenta"]
# normalised per-channel palette levels (low=0, mid~0.735, high=1), measured in demos
PALETTE = [[1, 0, 0], [0.735, 0, 0], [1, 1, 0], [1, 0.735, 0], [0, 1, 0],
           [0, 0.735, 0.735], [0, 1, 1], [0, 0, 1], [1, 0, 1]]
# expert touched with TCP ~9 mm on -x of the front-camera top-face centroid (std 1.4 mm, 351 touches)
TOUCH_OFFSET_X = -0.009
# press point relative to the visible cap centre; the push slides the TCP ~+0.025 m in x (R001), so start slightly -x
PRESS_OFFSET_X = -0.005
Z_TRAVEL = 0.10
ORDER_MODE = "cue"   # R012: shortest-path order fails natively (10 clean presses); cue order required
Z_TOUCH = 0.05
KP = 10.0          # a = KP * error(m); 0.1 m error -> full-scale command
KR = 0.5           # orientation-hold gain (fraction of error per step target)
MEM_DEFAULT = {
    "stage": "cue", "cue_colors": [], "cue_labels": [], "ring_step": -1,
    "cubes": [], "cube_labels": [], "targets": [], "target_labels": [],
    "touched": [], "pending": [], "pending_tries": 0, "button": [],
    "press_attempts": 0, "home_rpy": [], "notes": "",
}


# ---------------------------------------------------------------- geometry

def _world_points(frame):
    depth = np.asarray(frame.front_depth)
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    h, w = depth.shape
    z = depth.astype(np.float64) / 1000.0
    u = np.arange(w, dtype=np.float64) + 0.5
    v = np.arange(h, dtype=np.float64) + 0.5
    x = (u[None, :] - K[0, 2]) * z / K[0, 0]
    y = (v[:, None] - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    P = (pc - t) @ R
    return P, depth > 0


def _rgb_sat(frame):
    rgb = np.asarray(frame.front_rgb).astype(np.float64)
    return rgb, rgb.max(2) - rgb.min(2)


def _cue_color(frame):
    """Mean RGB of a coloured object floating in the cue spot in front of the button, else None."""
    P, valid = _world_points(frame)
    rgb, sat = _rgb_sat(frame)
    m = valid & (P[..., 2] > 0.008) & (P[..., 2] < 0.07) & (P[..., 0] > -0.04) & \
        (P[..., 0] < 0.15) & (np.abs(P[..., 1]) < 0.08) & (sat > 40)
    if int(m.sum()) < 15:
        return None
    return [float(c) for c in rgb[m].mean(0)]


def _ring_cubes(frame):
    """Coloured cubes on the table: list of (xy top centroid, p90 rgb, n pixels)."""
    P, valid = _world_points(frame)
    rgb, sat = _rgb_sat(frame)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    btn = (X < -0.045) & (X > -0.21) & (np.abs(Y) < 0.085)
    m = valid & (Z > 0.008) & (Z < 0.06) & (sat > 60) & (~btn) & (X > -0.45) & (X < 0.25) & (np.abs(Y) < 0.5)
    pts = P[m]
    cols = rgb[m]
    out = []
    free = np.ones(len(pts), dtype=bool)
    while free.sum() > 0 and len(out) < 20:
        i = int(np.flatnonzero(free)[0])
        c = pts[i, :2]
        for _ in range(3):
            sel = free & (np.linalg.norm(pts[:, :2] - c, axis=1) < 0.035)
            c = pts[sel, :2].mean(0)
        sel = free & (np.linalg.norm(pts[:, :2] - c, axis=1) < 0.035)
        if not sel.any():
            free[i] = False
            continue
        free[sel] = False
        if int(sel.sum()) < 15:
            continue
        pk = pts[sel]
        top = pk[:, 2] > pk[:, 2].max() - 0.006
        xy = pk[top, :2].mean(0)
        col = np.percentile(cols[sel], 90, axis=0)
        out.append(([float(xy[0]), float(xy[1])], [float(v) for v in col], int(sel.sum())))
    return out


def _button(frame):
    """Button cap centre (x, y) from the raised cap points; fallback to the default layout."""
    P, valid = _world_points(frame)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    reg = valid & (np.abs(Y) < 0.12) & (X < 0.0) & (X > -0.3)
    cap = reg & (Z > 0.045) & (Z < 0.075)
    if int(cap.sum()) < 20:
        return [-0.127, 0.0], False
    pc = P[cap]
    c = (pc[:, :2].min(0) + pc[:, :2].max(0)) / 2.0
    return [float(c[0]), float(c[1])], True


def _classify(colors):
    """Per-channel min/max normalisation inside one set, then nearest palette entry."""
    C = np.asarray(colors, dtype=np.float64)
    if len(C) == 0:
        return []
    lo = C.min(0)
    hi = C.max(0)
    N = (C - lo) / np.maximum(hi - lo, 1.0)
    pal = np.asarray(PALETTE, dtype=np.float64)
    labs = []
    for n in N:
        d = ((pal - n) ** 2).sum(1)
        labs.append(NAMES[int(np.argmin(d))])
    return labs


def _order(start, pts, end):
    """Held-Karp shortest open path start -> all pts -> end. Returns index order."""
    n = len(pts)
    if n == 0:
        return []
    P = [np.asarray(p, dtype=np.float64) for p in pts]
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end, dtype=np.float64)
    INF = 1e9
    dp = {}
    for i in range(n):
        dp[(1 << i, i)] = (float(np.linalg.norm(P[i] - s)), -1)
    for mask in range(1, 1 << n):
        for j in range(n):
            if not (mask >> j) & 1 or (mask, j) not in dp:
                continue
            cost = dp[(mask, j)][0]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                c2 = cost + float(np.linalg.norm(P[k] - P[j]))
                if c2 < dp.get((nm, k), (INF, -1))[0]:
                    dp[(nm, k)] = (c2, j)
    full = (1 << n) - 1
    best, bj = INF, -1
    for j in range(n):
        c = dp[(full, j)][0] + float(np.linalg.norm(e - P[j]))
        if c < best:
            best, bj = c, j
    order = []
    mask, j = full, bj
    while j != -1:
        order.append(j)
        pj = dp[(mask, j)][1]
        mask = mask & ~(1 << j)
        j = pj
    return order[::-1]


# ---------------------------------------------------------------- rotation helpers

def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _log_R(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = math.sin(ang)
    if s < 1e-6:
        return v
    return v * (ang / s)


def _rot_cmd(tcp, home_rpy):
    """Rotation command (drx, dry, drz) that turns the measured orientation toward home_rpy."""
    if not home_rpy:
        return [0.0, 0.0, 0.0], 0.0
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(home_rpy[0], home_rpy[1], home_rpy[2])
    w = _log_R(Rt @ Rc.T)          # world-frame rotation still needed
    err = float(np.linalg.norm(w))
    if err < 0.01:
        return [0.0, 0.0, 0.0], err
    r = -KR * w / 0.1                # applied rotation is about -0.1 * r
    nr = float(np.linalg.norm(r))
    if nr > 0.5:
        r = r * (0.5 / nr)
    return [float(v) for v in r], err


def _cmd(target, tcp, maxa=1.0):
    err = np.asarray(target, dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64)
    a = KP * err
    s = float(np.abs(a).max())
    if s > maxa:
        a = a * (maxa / s)
    return [float(v) for v in a], err


def _step_to(robot, target, grip, home_rpy, maxa=1.0):
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    a, err = _cmd(target, tcp, maxa)
    r, _ = _rot_cmd(tcp, home_rpy)
    robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)
    return err


# ---------------------------------------------------------------- decision

def perceive_decide(obs, goal, memory, last):
    """Track cue colours, detect/classify ring cubes, verify touches from measured TCP, choose next act."""
    mem = {}
    for k, v in MEM_DEFAULT.items():
        mv = memory.get(k) if isinstance(memory, dict) else None
        mem[k] = v if mv is None else mv
    frames = obs.frames
    cur = obs.current
    if not mem["home_rpy"]:
        t0 = frames[0].tcp
        mem["home_rpy"] = [float(t0[3]), float(t0[4]), float(t0[5])]

    if mem["stage"] == "cue":
        ring_seen = False
        for f in frames:
            if len(_ring_cubes(f)) >= 5:
                ring_seen = True
                mem["ring_step"] = int(f.step)
                break
            c = _cue_color(f)
            if c is not None:
                mem["cue_colors"] = list(mem["cue_colors"]) + [[round(v, 1) for v in c]]
        if not ring_seen:
            return {"memory": mem, "act": {"tool": "act_wait_ring", "args": {"max_steps": 80}}}
        # plan from the latest frame (robot still at home, full view)
        cubes = _ring_cubes(cur)
        cue_seq = _classify(mem["cue_colors"])
        cue_labels = []
        for l in cue_seq:            # first-occurrence order = order in which the cue showed them
            if l not in cue_labels:
                cue_labels.append(l)
        cube_labels = _classify([c[1] for c in cubes])
        mem["cue_labels"] = cue_labels
        mem["cubes"] = [c[0] for c in cubes]
        mem["cube_labels"] = cube_labels
        btn, ok = _button(cur)
        mem["button"] = [btn[0] + PRESS_OFFSET_X, btn[1]]
        # touch in the order the cue presented the colours (all 50 demos do exactly this)
        tg, tl = [], []
        for l in cue_labels:
            idx = [i for i, cl in enumerate(cube_labels) if cl == l]
            if idx:
                i = idx[0]
                tg.append([cubes[i][0][0] + TOUCH_OFFSET_X, cubes[i][0][1]])
                tl.append(l)
        if ORDER_MODE == "shortest" and tg:
            od = _order([float(cur.tcp[0]), float(cur.tcp[1])], tg, mem["button"])
            tg = [tg[i] for i in od]
            tl = [tl[i] for i in od]
        mem["targets"] = tg
        mem["target_labels"] = tl
        mem["stage"] = "touch"
        mem["notes"] = "cues=%d labels=%s cubes=%d uniq=%d button_ok=%s" % (
            len(mem["cue_colors"]), ",".join(cue_labels), len(cubes), len(set(cube_labels)), ok)

    if mem["stage"] == "touch":
        # verify pending touch with measured TCP of frames since last action
        if mem["pending"]:
            p = mem["pending"]
            hit = False
            for f in frames:
                t = f.tcp
                if abs(float(t[0]) - p[0]) < 0.012 and abs(float(t[1]) - p[1]) < 0.012 and float(t[2]) < 0.06:
                    hit = True
            if hit:
                mem["touched"] = list(mem["touched"]) + [p]
                mem["targets"] = list(mem["targets"])[1:]
                mem["target_labels"] = list(mem["target_labels"])[1:]
                mem["pending"] = []
                mem["pending_tries"] = 0
            # unverified: retry the same cube immediately (order matters, never re-queue)
        if mem["targets"]:
            p = mem["targets"][0]
            mem["pending"] = p
            mem["pending_tries"] = int(mem["pending_tries"]) + 1
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": p[0], "y": p[1], "home_rpy": mem["home_rpy"]}}}
        mem["stage"] = "press"

    # press stage (repeat until native success ends the episode)
    mem["press_attempts"] = int(mem["press_attempts"]) + 1
    b = mem["button"]
    return {"memory": mem, "act": {"tool": "act_press", "args": {
        "x": b[0], "y": b[1], "home_rpy": mem["home_rpy"], "attempt": mem["press_attempts"]}}}


# ---------------------------------------------------------------- actions

def act_wait_ring(robot, max_steps=80):
    """Hold still (gripper open) through the cue and delay until the cube ring appears; report cue frames seen."""
    n = 0
    cues = 0
    for i in range(max_steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        f = robot.now()
        if len(_ring_cubes(f)) >= 5:
            return {"steps": n, "ring_seen": True, "step": int(f.step), "cue_frames": cues}
        if _cue_color(f) is not None:
            cues += 1
    return {"steps": n, "ring_seen": False, "step": int(robot.now().step), "cue_frames": cues}


def act_touch(robot, x, y, home_rpy=None, z_touch=Z_TOUCH, z_travel=Z_TRAVEL, dwell=2, max_steps=90):
    """Lift to travel height, move above (x, y), descend to z_touch with gripper open, dwell, rise again.
    Returns measured bottom pose, xy error and steps used."""
    home_rpy = home_rpy or []
    n = 0
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    start_step = int(robot.now().step)
    # phase 1: rise if low (so we never drag through a cube)
    while float(tcp[2]) < 0.075 and n < max_steps:
        _step_to(robot, [tcp[0], tcp[1], z_travel], 1.0, home_rpy)
        n += 1
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    # phase 2: travel above target
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        exy = math.hypot(x - tcp[0], y - tcp[1])
        if exy < 0.012 and abs(z_travel - tcp[2]) < 0.03:
            break
        _step_to(robot, [x, y, z_travel], 1.0, home_rpy)
        n += 1
    # phase 3: descend
    zmin = 9.0
    bottom = None
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        if float(tcp[2]) < zmin:
            zmin = float(tcp[2])
            bottom = [float(v) for v in tcp[:3]]
        exy = math.hypot(x - tcp[0], y - tcp[1])
        if float(tcp[2]) < z_touch + 0.006 and exy < 0.008:
            break
        _step_to(robot, [x, y, z_touch], 1.0, home_rpy)
        n += 1
    # phase 4: dwell at the bottom
    for i in range(dwell):
        _step_to(robot, [x, y, z_touch], 1.0, home_rpy)
        n += 1
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        if float(tcp[2]) < zmin:
            zmin = float(tcp[2])
            bottom = [float(v) for v in tcp[:3]]
    # phase 5: rise until clear of the cube
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        if float(tcp[2]) > 0.08:
            break
        _step_to(robot, [x, y, z_travel], 1.0, home_rpy)
        n += 1
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    _, rerr = _rot_cmd(tcp, home_rpy)
    return {"steps": n, "zmin": zmin, "bottom": bottom,
            "bottom_xy_err": None if bottom is None else math.hypot(bottom[0] - x, bottom[1] - y),
            "end_tcp": [float(v) for v in tcp], "rot_err": rerr,
            "start_step": start_step, "end_step": int(robot.now().step)}


def act_press(robot, x, y, home_rpy=None, attempt=1, z_travel=Z_TRAVEL, lift_z=0.22, hold=3, wait=12, max_steps=120):
    """Close the gripper, move above the button (x, y), push down until blocked, hold, lift to lift_z and wait.
    The episode ends on native success; otherwise returns measured press depth and lift height."""
    home_rpy = home_rpy or []
    n = 0
    start_step = int(robot.now().step)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    while float(tcp[2]) < 0.075 and n < max_steps:
        _step_to(robot, [tcp[0], tcp[1], z_travel], -1.0, home_rpy)
        n += 1
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        fing = float(np.asarray(robot.now().fingers).max())
        if math.hypot(x - tcp[0], y - tcp[1]) < 0.008 and abs(z_travel - tcp[2]) < 0.02 and fing < 0.01:
            break
        _step_to(robot, [x, y, z_travel], -1.0, home_rpy)
        n += 1
    # push down: strong downward command (expert used dz ~ -0.7) until motion stalls
    zs = []
    zmin = 9.0
    contact_steps = 0
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        z = float(tcp[2])
        zs.append(z)
        zmin = min(zmin, z)
        stalled = len(zs) >= 3 and abs(zs[-1] - zs[-2]) < 0.0015 and z < 0.065
        if stalled or z < 0.045:
            break
        _step_to(robot, [x, y, -0.02], -1.0, home_rpy, maxa=0.7)
        n += 1
    for i in range(hold):
        _step_to(robot, [x, y, -0.02], -1.0, home_rpy, maxa=0.7)
        n += 1
        zmin = min(zmin, float(robot.now().tcp[2]))
        contact_steps += 1
    bottom = [float(v) for v in robot.now().tcp]
    # lift
    zmax = 0.0
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        zmax = max(zmax, float(tcp[2]))
        if float(tcp[2]) > lift_z - 0.01:
            break
        _step_to(robot, [x, y, lift_z], -1.0, home_rpy)
        n += 1
    for i in range(wait):
        _step_to(robot, [x, y, lift_z], -1.0, home_rpy)
        n += 1
        zmax = max(zmax, float(robot.now().tcp[2]))
    return {"attempt": attempt, "steps": n, "zmin": zmin, "bottom": bottom, "zmax": zmax,
            "start_step": start_step, "end_step": int(robot.now().step), "success": False}
