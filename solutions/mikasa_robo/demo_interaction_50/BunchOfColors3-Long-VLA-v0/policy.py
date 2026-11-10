"""BunchOfColors3-Long-VLA policy: remember cue cube colours, wait for the 9-cube arc,
touch the matching cubes, then press the centre button.

All perception uses the fixed front RGB-D camera: pixels are back-projected to world
coordinates with front_K / front_extrinsic; cubes are compact, strongly saturated blobs
whose top face lies ~0.04 m above the table; their colour is the median of top-face pixels.
"""

PALETTE = {
    "red": [231, 5, 5], "darkred": [169, 8, 8], "teal": [8, 169, 169],
    "green": [5, 231, 5], "cyan": [5, 231, 231], "blue": [5, 5, 231],
    "orange": [231, 169, 5], "yellow": [231, 231, 5], "magenta": [231, 5, 231],
}
TRAVEL_Z = 0.19      # lift/transit height: expert always rose to 0.176-0.182 between touches and
                     # success came when the post-press lift crossed ~0.17-0.18 (failed lifts peaked 0.165)
TOUCH_Z = 0.05       # TCP height at a touch (expert minima 0.045-0.062; cube top 0.041)
PRESS_Z = 0.035      # commanded press target; the button stops the TCP near 0.053
LIFT_Z = 0.21        # lift height after the press


# ---------------------------------------------------------------- geometry helpers
def _backproject(frame):
    depth = np.asarray(frame.front_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * depth   # pixel-centre convention (checked vs wrist camera)
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * depth
    Pc = np.stack([x, y, depth], -1).reshape(-1, 3)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    return Pw, depth.reshape(-1) > 0


def _classify(rgb):
    best, bd = None, 1e9
    for name, ref in PALETTE.items():
        d = float(np.max(np.abs(np.asarray(rgb, dtype=float) - np.asarray(ref, dtype=float))))
        if d < bd:
            best, bd = name, d
    return best, bd


def _detect(frame):
    """Detect saturated cubes and the button in one frame. Returns (cubes, button)."""
    P, valid = _backproject(frame)
    rgb = np.asarray(frame.front_rgb).reshape(-1, 3).astype(np.float64)
    z = P[:, 2]
    region = valid & (z > 0.006) & (z < 0.075) & (P[:, 0] > -0.5) & (P[:, 0] < 0.3) & (np.abs(P[:, 1]) < 0.55)
    sat = rgb.max(1) - rgb.min(1)
    cmask = region & (sat > 90)
    pts = P[cmask]
    cols = rgb[cmask]
    cubes = []
    remaining = np.ones(len(pts), dtype=bool)
    guard = 0
    while remaining.sum() >= 8 and guard < 40:
        guard += 1
        idx = np.flatnonzero(remaining)
        seed = pts[idx[0], :2]
        for _ in range(3):
            dd = np.linalg.norm(pts[idx, :2] - seed, axis=1)
            mem = idx[dd < 0.035]
            if len(mem) == 0:
                break
            seed = pts[mem, :2].mean(0)
        dd = np.linalg.norm(pts[:, :2] - seed, axis=1)
        mem = np.flatnonzero(remaining & (dd < 0.035))
        remaining[mem] = False
        remaining[idx[0]] = False
        if len(mem) < 12:
            continue
        zz = pts[mem, 2]
        ztop = float(zz.max())
        top = mem[zz > ztop - 0.004]
        if len(top) < 3:
            top = mem
        col = np.median(cols[top], 0)
        name, dist = _classify(col)
        cxy = pts[top, :2].mean(0)
        cubes.append({"color": name, "cdist": round(dist, 1), "rgb": [int(c) for c in col],
                      "x": float(cxy[0]), "y": float(cxy[1]), "ztop": round(ztop, 4), "n": int(len(mem))})
    # button: low-saturation object near the table centre with a cap ~0.056-0.06 m high
    bmask = region & (sat < 90) & (np.abs(P[:, 0] + 0.12) < 0.08) & (np.abs(P[:, 1]) < 0.08)
    button = None
    if bmask.sum() > 50:
        bp = P[bmask]
        zt = float(bp[:, 2].max())
        topb = bp[bp[:, 2] > zt - 0.004]
        button = {"x": float(topb[:, 0].mean()), "y": float(topb[:, 1].mean()), "ztop": round(zt, 4), "n": int(bmask.sum())}
    return cubes, button


def _good_cubes(cubes):
    return [c for c in cubes if 0.025 < c["ztop"] < 0.06 and c["cdist"] < 45]


# ---------------------------------------------------------------- rotation helpers
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:
        # near pi: fall back to diagonal-based axis
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return axis * ang
    return v / (2.0 * s) * ang


def _rot_cmd(tcp, goal_rpy, gain=0.5, dead=0.01):
    R = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rg = _rpy_to_R(goal_rpy[0], goal_rpy[1], goal_rpy[2])
    w = _rotvec(Rg @ R.T)  # rotation (base frame) that brings current to goal
    if np.linalg.norm(w) < dead:
        return np.zeros(3), float(np.linalg.norm(w))
    r = -gain * w / 0.1   # native offset is -0.1*r
    n = np.linalg.norm(r)
    if n > 0.5:
        r = r * 0.5 / n
    return r, float(np.linalg.norm(w))


# ---------------------------------------------------------------- motion helpers
def _move(robot, goal, rpy, grip, tol=0.004, max_steps=80, vmax=1.0, settle_speed=0.002, xy_first=False):
    """Closed-loop move of the TCP to goal (x,y,z). Returns stats."""
    goal = np.asarray(goal, dtype=float)
    prev = np.asarray(robot.now().tcp[:3], dtype=float)
    steps = 0
    maxrot = 0.0
    sat = 0
    for k in range(max_steps):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        p = tcp[:3]
        err = goal - p
        speed = float(np.linalg.norm(p - prev))
        if np.linalg.norm(err) < tol and speed < settle_speed:
            break
        cmd = err / 0.1
        m = float(np.max(np.abs(cmd)))
        if m > vmax:
            cmd = cmd * vmax / m
            sat += 1
        rc, rerr = _rot_cmd(tcp, rpy)
        maxrot = max(maxrot, rerr)
        prev = p
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), grip)
        steps += 1
    p = np.asarray(robot.now().tcp[:3], dtype=float)
    return {"steps": steps, "err": round(float(np.linalg.norm(goal - p)), 4), "pos": [round(float(v), 4) for v in p],
            "max_rot_err": round(maxrot, 4), "saturated": sat}


def _transit(robot, x, y, z, rpy, grip, max_steps=90):
    """Rise vertically to z (if lower) at the current xy, then move xy at z; ends near (x,y,z)."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out = {}
    if tcp[2] < z - 0.015:
        out["rise"] = _move(robot, [tcp[0], tcp[1], z], rpy, grip, tol=0.012, max_steps=40, settle_speed=1.0)
    out["xy"] = _move(robot, [x, y, z], rpy, grip, tol=0.008, max_steps=max_steps, settle_speed=0.006)
    return out


def _hold(robot, rpy, grip, n):
    for _ in range(n):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        rc, _ = _rot_cmd(tcp, rpy)
        robot.step(0.0, 0.0, 0.0, float(rc[0]), float(rc[1]), float(rc[2]), grip)


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Stage machine: cue -> wait for arc -> touch matching cubes -> press button -> retry."""
    mem = dict(memory)
    for k, v in (("stage", None), ("cue_colors", []), ("targets", []), ("target_colors", []),
                 ("touched", []), ("button", []), ("arc_step", None), ("presses", 0), ("home_rpy", []),
                 ("log", ""), ("arc_seen", "")):
        if mem.get(k) is None:
            mem[k] = v
    cur = obs.current
    if mem["stage"] is None:
        # read the cue from every frame available now (cubes may vanish later)
        best = []
        for f in obs.frames:
            cs = _good_cubes(_detect(f)[0])
            if len(cs) > len(best) and len(cs) <= 5:
                best = cs
        mem["cue_colors"] = [c["color"] for c in best]
        mem["home_rpy"] = [float(cur.tcp[3]), float(cur.tcp[4]), float(cur.tcp[5])]
        mem["stage"] = "wait"
        mem["log"] = "cue@%s:%s" % (cur.step, ",".join(mem["cue_colors"]))
        return {"memory": mem, "act": {"tool": "act_wait_for_arc", "args": {"cue_known": len(best)}}}
    if mem["stage"] == "wait":
        cubes, button = _detect(cur)
        cubes = _good_cubes(cubes)
        if len(cubes) < 7:
            return {"memory": mem, "act": {"tool": "act_wait_for_arc", "args": {"cue_known": len(mem["cue_colors"])}}}
        mem["arc_step"] = int(cur.step)
        mem["arc_seen"] = ";".join("%s(%.3f,%.3f)" % (c["color"], c["x"], c["y"]) for c in cubes)
        cue = list(mem["cue_colors"])
        chosen = []
        for col in cue:
            cand = [c for c in cubes if c["color"] == col and c not in chosen]
            if not cand:
                # nearest by palette distance fallback
                ref = np.asarray(PALETTE.get(col, [0, 0, 0]), dtype=float)
                cand = sorted([c for c in cubes if c not in chosen],
                              key=lambda c: float(np.max(np.abs(np.asarray(c["rgb"], dtype=float) - ref))))[:1]
            if cand:
                chosen.append(cand[0])
        # order: shortest path from the current TCP through all targets to the button
        bxy = np.array([button["x"], button["y"]]) if button else np.array([-0.125, 0.0])
        start = np.asarray(cur.tcp[:2], dtype=float)
        best_order, best_len = None, 1e9
        n = len(chosen)
        perms = [[]]
        for _ in range(n):
            perms = [p + [i] for p in perms for i in range(n) if i not in p]
        for p in perms:
            pts = [start] + [np.array([chosen[i]["x"], chosen[i]["y"]]) for i in p] + [bxy]
            L = sum(float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1))
            if L < best_len:
                best_order, best_len = p, L
        chosen = [chosen[i] for i in best_order] if best_order else chosen
        mem["targets"] = [[round(c["x"], 4), round(c["y"], 4), round(c["ztop"], 4)] for c in chosen]
        mem["target_colors"] = [c["color"] for c in chosen]
        mem["button"] = [round(float(bxy[0]), 4), round(float(bxy[1]), 4), round(button["ztop"], 4) if button else 0.059]
        mem["stage"] = "touch"
        mem["log"] += " | arc@%s n=%d targets=%s" % (cur.step, len(cubes), ",".join(mem["target_colors"]))
    if mem["stage"] == "touch":
        done = len(mem["touched"])
        if done < len(mem["targets"]):
            t = mem["targets"][done]
            mem["touched"] = mem["touched"] + [mem["target_colors"][done]]
            return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": t[0], "y": t[1], "rpy": mem["home_rpy"]}}}
        mem["stage"] = "button"
    # button stage. If the episode is still running after a press, retry with a longer hold and higher
    # lift; after one failed press re-touch every target (lower, 0.045) once, then press again
    # (R016: touches at TCP z 0.065 did not register; the retouch at 0.045 + press succeeded).
    k = int(mem["presses"])
    if k == 1 and "retouch" not in mem["log"] and int(cur.step) < 570:
        mem["log"] += " | retouch@%s" % cur.step
        mem["touched"] = []
        mem["stage"] = "retouch"
    if mem["stage"] == "retouch":
        done = len(mem["touched"])
        if done < len(mem["targets"]):
            t = mem["targets"][done]
            mem["touched"] = mem["touched"] + [mem["target_colors"][done]]
            return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": t[0], "y": t[1], "rpy": mem["home_rpy"], "touch_z": 0.045, "hold": 5}}}
        mem["stage"] = "button"
    mem["presses"] = k + 1
    b = mem["button"]
    hold = 6 if k == 0 else 10
    lift = LIFT_Z if k == 0 else 0.24
    mem["log"] += " | press%d@%s" % (k + 1, cur.step)
    return {"memory": mem, "act": {"tool": "act_press_button", "args": {"x": b[0], "y": b[1], "rpy": mem["home_rpy"], "hold": hold, "lift_z": lift}}}


# ---------------------------------------------------------------- actions
def act_wait_for_arc(robot, cue_known=3, max_steps=660):
    """Hold still (gripper open) until the 9-cube arc is visible in the front camera for 2 frames."""
    seen = 0
    steps = 0
    last_n = 0
    gone_step = None
    for k in range(max_steps):
        f = robot.now()
        cs = _good_cubes(_detect(f)[0])
        last_n = len(cs)
        if last_n == 0 and gone_step is None:
            gone_step = int(f.step)
        if (last_n >= 7 and gone_step is not None) or last_n >= 10:
            seen += 1
            if seen >= 2:
                break
        else:
            seen = 0
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        steps += 1
    return {"steps": steps, "cubes_now": last_n, "cue_gone_step": gone_step, "step": int(robot.now().step)}


def act_touch(robot, x, y, rpy, touch_z=TOUCH_Z, hold=3, lift_z=TRAVEL_Z):
    """Rise to lift_z, move above (x,y), descend to touch_z with open gripper, hold, rise vertically to lift_z."""
    out = _transit(robot, x, y, lift_z, rpy, 1.0)
    d = _move(robot, [x, y, touch_z], rpy, 1.0, tol=0.004, max_steps=40, settle_speed=0.004)
    zmin = float(robot.now().tcp[2])
    _hold(robot, rpy, 1.0, hold)
    p = np.asarray(robot.now().tcp, dtype=float)
    zmin = min(zmin, float(p[2]))
    up = _move(robot, [x, y, lift_z], rpy, 1.0, tol=0.006, max_steps=40, settle_speed=1.0)
    return {"transit": out, "descend": d, "touch_xy_err": round(float(np.hypot(p[0] - x, p[1] - y)), 4),
            "touch_z": round(zmin, 4), "rise": up, "step": int(robot.now().step)}


def act_press_button(robot, x, y, rpy, hold=6, press_z=PRESS_Z, lift_z=LIFT_Z):
    """Close gripper, move above the button, push down until the TCP stalls, hold, lift."""
    out = _transit(robot, x, y, TRAVEL_Z, rpy, -1.0)
    # descend; the button cap stops the TCP
    zs = []
    for k in range(50):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = np.array([x, y, press_z]) - tcp[:3]
        cmd = err / 0.1
        m = float(np.max(np.abs(cmd)))
        if m > 0.8:
            cmd = cmd * 0.8 / m
        rc, _ = _rot_cmd(tcp, rpy)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), -1.0)
        zs.append(float(robot.now().tcp[2]))
        if len(zs) >= 3 and zs[-1] < 0.07 and abs(zs[-1] - zs[-3]) < 0.0015:
            break
    zbot = min(zs) if zs else None
    # keep pushing for `hold` steps
    for k in range(hold):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = np.array([x, y, press_z]) - tcp[:3]
        cmd = np.clip(err / 0.1, -0.8, 0.8)
        rc, _ = _rot_cmd(tcp, rpy)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), -1.0)
        zs.append(float(robot.now().tcp[2]))
    zbot = min(zs)
    up = _move(robot, [x, y, lift_z], rpy, -1.0, tol=0.006, max_steps=30, settle_speed=0.003)
    _hold(robot, rpy, -1.0, 10)
    p = robot.now().tcp
    return {"transit": out, "press_zmin": round(zbot, 4), "press_steps": len(zs), "lift": up,
            "end_pos": [round(float(v), 4) for v in p[:3]], "step": int(robot.now().step)}
