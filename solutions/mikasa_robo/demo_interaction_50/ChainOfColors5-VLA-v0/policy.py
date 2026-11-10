"""ChainOfColors5-VLA policy: watch the cue, touch ring cubes in cue order, press the centre button.

Perception: front RGB-D back-projected to world points (front_K / front_extrinsic).
Cubes are saturated blobs 8-75 mm above the table; colour family by channel pattern;
red/dark_red and cyan/teal split by top-face brightness (pairwise in the ring).
Control: closed-loop on measured TCP, P-D on position, small orientation hold.
"""

FAMILIES = ["red", "green", "blue", "yellow", "orange", "cyan", "magenta"]
BRIGHT_SPLIT = 188.0      # top-face max channel separating bright / dark shade (demo: >=207 vs <=169)
TRAVEL_Z = 0.15           # transport height of TCP (expert ~0.16-0.17)
TOUCH_DZ = 0.006          # TCP height above cube top at touch; R009/R031: +22 mm (0.045 m from centre) not accepted
RETRY_DZ = 0.002          # touch height for the recovery round (whole sequence redone if presses gave no success)
RETRY_LAST_STEP = 200     # full recovery chain needs ~200 steps (R031: 177->378); later failures only re-press
BUTTON_DX = 0.010         # press point offset from cap centroid (expert ~ +0.012..0.02 in x)
LIFT_Z = 0.17             # height after button press (expert episode end ~0.16-0.175)
BASE_XY = (-0.615, 0.0)   # arm base (horizontal); J4 margin shrinks as TCP nears it (demo ep28: 0.17 rad at 0.34 m)
KEEP_OUT_R = 0.44         # route transport around this horizontal radius from BASE_XY


def _world_points(frame):
    d = np.asarray(frame.front_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.front_K).astype(np.float64)
    E = np.asarray(frame.front_extrinsic).astype(np.float64)
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R, d > 0


def _blobs(frame, min_pixels=6):
    """Saturated cube blobs in the front view: list of dicts fam, bright, top xyz, n."""
    P, valid = _world_points(frame)
    rgb = np.asarray(frame.front_rgb).astype(np.float64)
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    z = P[..., 2]
    m = valid & (z > 0.008) & (z < 0.075) & (sat > 0.6) & (mx > 60) & (P[..., 0] > -0.5)
    n = rgb / (mx[..., None] + 1e-6)
    r = n[..., 0] > 0.5
    g = n[..., 1] > 0.5
    b = n[..., 2] > 0.5
    fam = np.full(m.shape, -1)
    fam[r & ~g & ~b] = 0
    fam[~r & g & ~b] = 1
    fam[~r & ~g & b] = 2
    fam[r & g & ~b & (n[..., 1] >= 0.87)] = 3
    fam[r & g & ~b & (n[..., 1] < 0.87)] = 4
    fam[~r & g & b] = 5
    fam[r & ~g & b] = 6
    m = m & (fam >= 0)
    H, W = m.shape
    lab = -np.ones((H, W), dtype=int)
    out = []
    for ij in np.argwhere(m):
        i0, j0 = int(ij[0]), int(ij[1])
        if lab[i0, j0] >= 0:
            continue
        f0 = fam[i0, j0]
        idx = len(out)
        lab[i0, j0] = idx
        stack = [(i0, j0)]
        pix = []
        while stack:
            a, c = stack.pop()
            pix.append((a, c))
            for da, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                aa, cc = a + da, c + dc
                if 0 <= aa < H and 0 <= cc < W and m[aa, cc] and lab[aa, cc] < 0 and fam[aa, cc] == f0:
                    lab[aa, cc] = idx
                    stack.append((aa, cc))
        pix = np.array(pix)
        pts = P[pix[:, 0], pix[:, 1]]
        sel = pts[:, 2] > pts[:, 2].max() - 0.005
        top = pts[sel].mean(0)
        toprgb = rgb[pix[sel, 0], pix[sel, 1]].mean(0)
        out.append({"fam": int(f0), "n": int(len(pix)), "top": [float(top[0]), float(top[1]), float(pts[:, 2].max())],
                    "bright": float(toprgb.max())})
    return [o for o in out if o["n"] >= min_pixels]


def _label(fam, bright):
    name = FAMILIES[fam]
    if name == "red" and bright < BRIGHT_SPLIT:
        return "dark_red"
    if name == "cyan" and bright < BRIGHT_SPLIT:
        return "teal"
    return name


def _ring_cubes(blobs):
    """Label ring cubes; red/cyan shades resolved pairwise (brighter = bright shade)."""
    cubes = [b for b in blobs if b["n"] >= 15]
    out = []
    for f in range(len(FAMILIES)):
        group = [b for b in cubes if b["fam"] == f]
        group.sort(key=lambda b: -b["bright"])
        for k, bl in enumerate(group):
            if FAMILIES[f] in ("red", "cyan") and len(group) == 2:
                lab = FAMILIES[f] if k == 0 else ("dark_red" if FAMILIES[f] == "red" else "teal")
            else:
                lab = _label(f, bl["bright"])
            out.append({"label": lab, "pos": bl["top"], "bright": bl["bright"], "n": bl["n"]})
    return out


def _is_ring(blobs):
    return len([b for b in blobs if b["n"] >= 15]) >= 6


def _button(frame):
    """Pink cap of the centre button: top-face centroid [x, y, z] or None."""
    P, valid = _world_points(frame)
    rgb = np.asarray(frame.front_rgb).astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    z = P[..., 2]
    m = valid & (z > 0.045) & (z < 0.075) & (R > 120) & (G < 0.8 * R) & (G > 0.4 * R) & (np.abs(G - B) < 25)
    if m.sum() < 10:
        return None
    pts = P[m]
    top = pts[pts[:, 2] > pts[:, 2].max() - 0.004]
    c = top.mean(0)
    return [float(c[0]), float(c[1]), float(pts[:, 2].max())]


def _rot(rpy):
    r, p, y = rpy
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logrot(M):
    c = max(-1.0, min(1.0, (float(np.trace(M)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def _rot_cmd(tcp, rpy_goal, gain=0.5, limit=0.3):
    if rpy_goal is None:
        return [0.0, 0.0, 0.0]
    w = _logrot(_rot(rpy_goal) @ _rot(tcp[3:6]).T)
    r = -gain * w / 0.1
    nr = float(np.linalg.norm(r))
    if nr > limit:
        r = r * (limit / nr)
    return [float(r[0]), float(r[1]), float(r[2])]


def _goto(robot, target, rpy, grip, tol=0.004, vtol=0.003, max_steps=40, kp=1.2, kd=1.0, xy_first_below=None):
    """Closed-loop move of TCP to target xyz. Returns dict with steps, final error, speed."""
    target = np.array(target, dtype=float)
    prev = _prev_pos(robot)
    steps = 0
    err = target - prev
    v = np.zeros(3)
    while steps < max_steps:
        tcp = np.array(robot.now().tcp, dtype=float)
        pos = tcp[:3]
        v = pos - prev
        prev = pos
        goal = target.copy()
        if xy_first_below is not None and pos[2] < xy_first_below:
            goal[0], goal[1] = pos[0], pos[1]   # rise vertically before travelling
        err = goal - pos
        full = target - pos
        if np.linalg.norm(full) < tol and np.linalg.norm(v) < vtol:
            break
        a = (kp * err - kd * v) / 0.1
        na = float(np.abs(a).max())
        if na > 1.0:
            a = a / na
        rc = _rot_cmd(tcp, rpy)
        robot.step(float(a[0]), float(a[1]), float(a[2]), rc[0], rc[1], rc[2], grip)
        steps += 1
    pos = np.array(robot.now().tcp[:3], dtype=float)
    return {"steps": steps, "err": float(np.linalg.norm(target - pos)), "speed": float(np.linalg.norm(v)),
            "pos": [float(p) for p in pos]}


def _prev_pos(robot):
    """Position one step before now (from this action's frames) so the first D-term uses real velocity."""
    fr = robot.frames()
    if len(fr) >= 2:
        return np.array(fr[-2].tcp[:3], dtype=float)
    return np.array(robot.now().tcp[:3], dtype=float)


def _waypoint(p0, p1):
    """Waypoint xy that keeps the transport segment p0->p1 outside KEEP_OUT_R of the base, or None."""
    b = np.array(BASE_XY, dtype=float)
    a = np.array(p0[:2], dtype=float)
    c = np.array(p1[:2], dtype=float)
    d = c - a
    L2 = float(d @ d)
    s = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, float((b - a) @ d) / L2))
    q = a + s * d
    dist = float(np.linalg.norm(q - b))
    if dist >= KEEP_OUT_R or s <= 0.0 or s >= 1.0:
        return None
    u = (q - b) / dist if dist > 1e-6 else np.array([1.0, 0.0])
    w = b + (KEEP_OUT_R + 0.04) * u
    return [float(w[0]), float(w[1])]


def _travel(robot, x, y, z, rpy, grip):
    """Rise vertically to 0.11, then transport at height z to (x, y), detouring around the base if needed."""
    pos = robot.now().tcp
    w = _waypoint([float(pos[0]), float(pos[1])], [x, y])
    via = None
    if w is not None:
        via = _goto(robot, [w[0], w[1], z], rpy, grip, tol=0.04, vtol=1.0, max_steps=40, xy_first_below=0.11)
    res = _goto(robot, [x, y, z], rpy, grip, tol=0.008, vtol=0.01, max_steps=60, xy_first_below=0.11)
    res["via"] = w
    res["via_steps"] = via["steps"] if via else 0
    return res


def _hold(robot, target, rpy, grip, n):
    target = np.array(target, dtype=float)
    prev = _prev_pos(robot)
    for _ in range(n):
        tcp = np.array(robot.now().tcp, dtype=float)
        v = tcp[:3] - prev
        prev = tcp[:3]
        a = (1.0 * (target - tcp[:3]) - 1.0 * v) / 0.1
        na = float(np.abs(a).max())
        if na > 1.0:
            a = a / na
        rc = _rot_cmd(tcp, rpy)
        robot.step(float(a[0]), float(a[1]), float(a[2]), rc[0], rc[1], rc[2], grip)


def _parse_last(last, key):
    """Extract a boolean/number after '"key": ' in the last-outcome text; None if missing."""
    if not isinstance(last, str):
        return None
    k = '"' + key + '": '
    i = last.find(k)
    if i < 0:
        return None
    s = last[i + len(k):i + len(k) + 20]
    if s.startswith("true"):
        return True
    if s.startswith("false"):
        return False
    num = ""
    for ch in s:
        if ch in "-0123456789.e":
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return None


def perceive_decide(obs, goal, memory, last):
    """Track cue colours, locate ring cubes and button, then sequence touch / press actions."""
    mem = {k: memory.get(k) for k in ("stage", "cue_colors", "targets", "target_labels", "touched",
                                      "button", "home_rpy", "cue_frames", "notes", "press_tries", "round")}
    if mem["stage"] is None:
        mem["stage"] = "cue"
        mem["cue_colors"] = []
        mem["targets"] = []
        mem["target_labels"] = []
        mem["touched"] = 0
        mem["button"] = None
        mem["home_rpy"] = [float(x) for x in obs.current.tcp[3:6]]
        mem["cue_frames"] = 0
        mem["notes"] = ""
        mem["press_tries"] = 0
        mem["round"] = 0
    cue = list(mem["cue_colors"] or [])

    if mem["stage"] == "cue":
        ring_frame = None
        for fr in obs.frames:
            bl = _blobs(fr)
            if _is_ring(bl):
                ring_frame = fr
                break
            big = [b for b in bl if b["n"] >= 15]
            if len(big) == 1:
                lab = _label(big[0]["fam"], big[0]["bright"])
                mem["cue_frames"] = int(mem["cue_frames"] or 0) + 1
                if not cue or cue[-1] != lab:
                    cue.append(lab)
        mem["cue_colors"] = cue
        if ring_frame is None:
            return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 80}}}
        ring = _ring_cubes(_blobs(obs.current))
        if len(ring) < 9:
            alt = _ring_cubes(_blobs(ring_frame))
            if len(alt) > len(ring):
                ring = alt
        targets, labels, missing = [], [], []
        used = []
        for c in cue:
            cand = [i for i, r in enumerate(ring) if r["label"] == c and i not in used]
            if not cand:
                # fallback: same colour family (shade mis-split), not yet used
                fam = {"dark_red": "red", "teal": "cyan"}.get(c, c)
                cand = [i for i, r in enumerate(ring) if i not in used and
                        {"dark_red": "red", "teal": "cyan"}.get(r["label"], r["label"]) == fam]
                if cand:
                    missing.append(c + "~fam")
            if cand:
                used.append(cand[0])
                targets.append(ring[cand[0]]["pos"])
                labels.append(c)
            else:
                missing.append(c)
        btn = _button(obs.current) or _button(ring_frame)
        if btn is None:
            btn = [-0.12, 0.0, 0.06]
        mem["targets"] = targets
        mem["target_labels"] = labels
        mem["button"] = btn
        mem["notes"] = ("cue=" + ",".join(cue) + " ring=" + ";".join(r["label"] + "@%.3f,%.3f" % (r["pos"][0], r["pos"][1]) for r in ring)
                        + " missing=" + ",".join(missing) + " button=%.3f,%.3f" % (btn[0], btn[1]))
        mem["stage"] = "touch" if targets else "button"

    if mem["stage"] == "touch" and isinstance(last, str) and last.startswith("act_touch"):
        if _parse_last(last, "reached") is True:
            mem["touched"] = int(mem["touched"]) + 1
    if mem["stage"] == "touch" and int(mem["touched"]) >= len(mem["targets"]):
        mem["stage"] = "button"

    rnd = int(mem["round"] or 0)
    tries = int(mem["press_tries"] or 0)
    if (mem["stage"] == "button" and tries >= 1 and rnd == 0 and mem["targets"]
            and int(obs.current.step) <= RETRY_LAST_STEP):
        # no native success after the first press: redo the whole chain closer to the cubes, then press again
        mem["round"] = 1
        rnd = 1
        mem["touched"] = 0
        mem["stage"] = "touch"
        mem["notes"] = str(mem["notes"]) + " | recovery round at step %d" % int(obs.current.step)
    if mem["stage"] == "touch":
        t = mem["targets"][int(mem["touched"])]
        dz = TOUCH_DZ if rnd == 0 else RETRY_DZ
        return {"memory": mem, "act": {"tool": "act_touch",
                                       "args": {"x": t[0], "y": t[1], "ztop": t[2], "rpy": mem["home_rpy"], "dz": dz}}}
    b = mem["button"]
    mem["press_tries"] = tries + 1
    return {"memory": mem, "act": {"tool": "act_press_button",
                                   "args": {"x": b[0] + BUTTON_DX, "y": b[1], "ztop": b[2], "rpy": mem["home_rpy"],
                                            "attempt": tries}}}


def act_watch(robot, max_steps=80):
    """Hold still (gripper open) and watch the cue until the ring of cubes appears. Returns steps and ring flag."""
    steps = 0
    seen = _is_ring(_blobs(robot.now()))
    while not seen and steps < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        steps += 1
        seen = _is_ring(_blobs(robot.now()))
    return {"steps": steps, "ring_seen": bool(seen), "frame_step": int(robot.now().step)}


def act_touch(robot, x, y, ztop, rpy=None, dz=TOUCH_DZ):
    """Travel at safe height above cube (x, y), descend to ztop+dz, hold, lift. Gripper open."""
    zt = ztop + dz
    s0 = int(robot.now().step)
    up = _travel(robot, x, y, TRAVEL_Z, rpy, 1.0)
    down = _goto(robot, [x, y, zt], rpy, 1.0, tol=0.004, vtol=0.003, max_steps=30)
    low = robot.now().tcp
    _hold(robot, [x, y, zt], rpy, 1.0, 3)
    held = robot.now().tcp
    dxy = math.hypot(float(held[0]) - x, float(held[1]) - y)
    lift = _goto(robot, [float(held[0]), float(held[1]), TRAVEL_Z - 0.01], rpy, 1.0, tol=0.02, vtol=0.05, max_steps=20)
    reached = bool(down["err"] < 0.012 and dxy < 0.012)
    return {"reached": reached, "travel_steps": up["steps"], "via": up["via"], "via_steps": up["via_steps"],
            "travel_err": up["err"], "down_steps": down["steps"],
            "down_err": down["err"], "low_z": float(low[2]), "hold_xy_err": dxy, "hold_z": float(held[2]),
            "lift_z": lift["pos"][2], "steps": int(robot.now().step) - s0}


def act_press_button(robot, x, y, ztop, rpy=None, attempt=0):
    """Close gripper, move above the button, press down until stall, hold, lift to LIFT_Z and wait."""
    s0 = int(robot.now().step)
    _travel(robot, x, y, TRAVEL_Z - 0.02, rpy, -1.0)
    above = _goto(robot, [x, y, ztop + 0.03], rpy, -1.0, tol=0.004, vtol=0.004, max_steps=20)
    # press: command well below the cap top, stop on stall
    zs = []
    for _ in range(18 + 6 * int(attempt)):
        tcp = robot.now().tcp
        zs.append(float(tcp[2]))
        a = (np.array([x, y, ztop - 0.03 - 0.01 * int(attempt)]) - np.array(tcp[:3], dtype=float)) / 0.1
        a[2] = max(-0.7, min(0.7, a[2]))
        a[0] = max(-0.5, min(0.5, a[0]))
        a[1] = max(-0.5, min(0.5, a[1]))
        rc = _rot_cmd(np.array(tcp, dtype=float), rpy)
        robot.step(float(a[0]), float(a[1]), float(a[2]), rc[0], rc[1], rc[2], -1.0)
        if len(zs) >= 6 and max(zs[-4:]) - min(zs[-4:]) < 0.0015 and zs[-1] < ztop:
            break
    low = float(min(zs + [float(robot.now().tcp[2])]))
    lift = _goto(robot, [x, y, LIFT_Z], rpy, -1.0, tol=0.006, vtol=0.004, max_steps=30)
    _hold(robot, [x, y, LIFT_Z], rpy, -1.0, 8)
    return {"pressed_low_z": low, "cap_top": ztop, "above_err": above["err"], "lift_z": lift["pos"][2],
            "steps": int(robot.now().step) - s0, "attempt": attempt}
