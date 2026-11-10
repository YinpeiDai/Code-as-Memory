"""RotateStrictPos policy: perceive peg (front RGB-D), grasp it across its width at its
centre, rotate the gripper (and peg) about the vertical axis by the instructed positive
angle while holding the centre in place, then release and verify."""

PEG_LEN = 0.24
PEG_W = 0.05
Z_GRASP = 0.024
Z_HOVER = 0.085
ROT_RATE = 0.038          # measured rad per step at saturated yaw command (demos)
MAX_DELTA = 0.35          # rad, max grasp misalignment (only when time is tight)
J7_MIN_YAW_CHANGE = -1.85 # rad of yaw change allowed (J7 upper-limit side)
J7_MAX_YAW_CHANGE = 3.35  # rad of yaw change allowed (J7 lower-limit side)
STEP_LIMIT = 90


def _rotm(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _wrap_half(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


def _heading(tcp):
    R = _rotm(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    return math.atan2(R[1, 0], R[0, 0])


def _logmap(Rm):
    c = max(-1.0, min(1.0, (np.trace(Rm) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3) if ang < 1e-3 else v * 0.0
    return v * (ang / (2 * s))


def _parse_angle(goal):
    """Degrees after the word 'by' in the instruction; None if absent."""
    g = str(goal).lower()
    i = g.find(" by ")
    if i < 0:
        return None
    s = g[i + 4:].strip()
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-+":
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return None


def _peg_points(frame, cam="front"):
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    m = (b > r + 60) & (b > g + 30) & (depth > 0)
    v, u = np.nonzero(m)
    if len(v) == 0:
        return np.zeros((0, 3))
    z = depth[v, u].astype(float) / 1000.0
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    R = np.array(E[:, :3], dtype=float)
    t = np.array(E[:, 3], dtype=float)
    pw = (pc - t) @ R
    return pw[pw[:, 2] < 0.07]


def _peg_pose(frame, wrist=False):
    """Peg top-face pose from the fixed front camera (optionally fused with the wrist camera):
    centre xy, long-axis yaw in [0, pi)."""
    pw = _peg_points(frame, "front")
    if wrist:
        pw = np.concatenate([pw, _peg_points(frame, "wrist")], 0)
    pw = pw[pw[:, 2] > 0.035]
    if len(pw) < 30:
        return None
    q = np.round(pw[:, :2] / 0.003).astype(int)
    key = q[:, 0] * 100000 + q[:, 1]
    _, idx = np.unique(key, return_index=True)
    xy = pw[idx, :2]
    c = xy.mean(0)
    C = np.cov((xy - c).T)
    w, V = np.linalg.eigh(C)
    a_pca = math.atan2(V[1, 1], V[0, 1])
    best = None
    for da in np.radians(np.arange(-15.0, 15.01, 0.25)):
        a = a_pca + da
        n = np.array([-math.sin(a), math.cos(a)])
        p = (xy - c) @ n
        wd = np.percentile(p, 99) - np.percentile(p, 1)
        if best is None or wd < best[0]:
            best = (wd, a)
    a = best[1] % math.pi
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-math.sin(a), math.cos(a)])
    pd = (xy - c) @ d
    pn = (xy - c) @ n
    L = float(pd.max() - pd.min())
    mid = c + n * (np.percentile(pn, 99) + np.percentile(pn, 1)) / 2
    if L > 0.2:
        mid = mid + d * (pd.max() + pd.min()) / 2
    return {"x": float(mid[0]), "y": float(mid[1]), "yaw": float(a), "W": float(best[0]),
            "L": L, "n": int(len(xy)), "ztop": float(np.median(pw[:, 2]))}


def _plan(peg_yaw, theta, y0, steps_left=90, allow_delta=True):
    """Choose grasp heading g0 (TCP x along the peg so fingers close across its width) and the
    nominal final heading. delta>0 (grasp misaligned ahead of the peg) is used only when the
    aligned plan does not fit the step budget; R001 showed closing removes only part of delta."""
    best = None
    for k in (-2, -1, 0, 1, 2):
        a = peg_yaw + k * math.pi
        for dd in np.arange(0.0, MAX_DELTA + 1e-9, 0.01745):
            d = float(dd)
            pre = a + d - y0
            rot = theta - 0.5 * d
            if pre < J7_MIN_YAW_CHANGE or pre > J7_MAX_YAW_CHANGE:
                continue
            if pre + theta > J7_MAX_YAW_CHANGE or pre + theta < J7_MIN_YAW_CHANGE:
                continue
            est = max(11.0, abs(pre) / ROT_RATE + 3.0) + 5.0 + abs(rot) / ROT_RATE + 3.0
            cost = est + (0.0 if d == 0.0 else 2.0 + 5.0 * d)
            if d > 0.0 and (not allow_delta):
                continue
            if best is None or cost < best[0]:
                best = (cost, a + d, a + theta, d, pre, rot, est)
    if best is None:
        return None
    return {"g0": best[1], "gF": best[2], "delta": best[3], "pre": best[4], "rot": best[5],
            "est": best[6]}


def _servo(tcp, target, heading, kp=1.3, kr=1.3, rmax=1.0):
    pos = np.array(tcp[:3], dtype=float)
    dxyz = (np.array(target, dtype=float) - pos) / 0.1 * kp
    dxyz = np.clip(dxyz, -1.0, 1.0)
    R = _rotm(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    h = _heading(tcp)
    Rh = _rotm(math.pi, 0.0, h)
    e = _logmap(Rh @ R.T)          # pure tilt error (world frame), small
    e[2] = e[2] + _wrap(heading - h)  # heading error about world z
    r = -e / 0.1 * kr
    nr = float(np.linalg.norm(r))
    if nr > rmax:
        r = r * (rmax / nr)
    return [float(dxyz[0]), float(dxyz[1]), float(dxyz[2]), float(r[0]), float(r[1]), float(r[2])]


def _state(frame):
    tcp = frame.tcp
    return {"step": int(frame.step), "tcp": [round(float(v), 4) for v in tcp],
            "heading": round(_heading(tcp), 4), "fingers": [round(float(v), 4) for v in frame.fingers]}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: plan -> approach -> grasp -> rotate -> release -> verify/correct."""
    mem = dict(memory)
    cur = obs.current
    step = int(cur.step)
    stage = mem.get("stage") or "start"
    log = mem.get("log") or ""
    lastp = str(last or "")
    if stage == "start":
        theta_deg = _parse_angle(goal)
        if theta_deg is None:
            theta_deg = 0.0
        peg = _peg_pose(cur)
        if peg is None:
            mem.update({"stage": "start", "log": log + "|nopeg@%d" % step, "steps": step,
                        "theta_deg": theta_deg})
            return {"memory": _full(mem), "act": {"tool": "act_wait", "args": {"n": 1}}}
        y0 = _heading(cur.tcp)
        plan = _plan(peg["yaw"], math.radians(theta_deg), y0, allow_delta=False)
        if plan is None or plan["est"] > 72:
            alt = _plan(peg["yaw"], math.radians(theta_deg), y0, allow_delta=True)
            if alt is not None and (plan is None or alt["est"] < plan["est"]):
                plan = alt
        mem.update({"theta_deg": theta_deg, "peg0": [peg["x"], peg["y"]], "peg_yaw0": peg["yaw"],
                    "g_grasp": plan["g0"], "g_final": plan["gF"], "delta": plan["delta"],
                    "cycle": 1, "steps": step,
                    "log": log + "|peg x%.3f y%.3f yaw%.1f W%.3f L%.3f th%.0f g0 %.1f gF %.1f d%.1f" % (
                        peg["x"], peg["y"], math.degrees(peg["yaw"]), peg["W"], peg["L"], theta_deg,
                        math.degrees(plan["g0"]), math.degrees(plan["gF"]), math.degrees(plan["delta"]))})
        mem["stage"] = "approach"
        mem["grasp_xy"] = [peg["x"], peg["y"]]
        return {"memory": _full(mem), "act": {"tool": "act_approach", "args": {
            "x": peg["x"], "y": peg["y"], "heading": plan["g0"], "z_grasp": Z_GRASP}}}
    if stage == "approach":
        mem["stage"] = "grasp"
        mem["steps"] = step
        mem["log"] = log + "|appr@%d" % step
        p = mem.get("grasp_xy") or mem["peg0"]
        return {"memory": _full(mem), "act": {"tool": "act_grasp", "args": {
            "x": p[0], "y": p[1], "z": Z_GRASP, "heading": float(_heading(cur.tcp))}}}
    if stage == "grasp":
        f = float(np.mean(cur.fingers))
        mem["steps"] = step
        p = mem["peg0"]
        if f < 0.012:
            # missed: open and re-plan from a fresh observation after lifting
            mem["stage"] = "recover"
            mem["log"] = log + "|miss f%.3f@%d" % (f, step)
            return {"memory": _full(mem), "act": {"tool": "act_release", "args": {
                "x": float(cur.tcp[0]), "y": float(cur.tcp[1]), "z_up": 0.12}}}
        # actual heading after closing; peg now aligned with fingers
        gnow = _heading(cur.tcp)
        target = mem["g_final"]
        mem["stage"] = "rotate"
        mem["log"] = log + "|grasp f%.3f h%.1f@%d" % (f, math.degrees(gnow), step)
        rho0 = float(mem.get("rho_done") or 0.0)
        return {"memory": _full(mem), "act": {"tool": "act_rotate", "args": {
            "x": mem["peg0"][0], "y": mem["peg0"][1], "z": Z_GRASP + 0.01, "heading": float(target),
            "peg_yaw0": float(mem["peg_yaw0"]), "theta": math.radians(mem["theta_deg"]),
            "rho_start": rho0, "max_steps": max(5, STEP_LIMIT - step - 1)}}}
    if stage == "rotate":
        mem["stage"] = "release"
        mem["steps"] = step
        mem["log"] = log + "|rot h%.1f@%d %s" % (math.degrees(_heading(cur.tcp)), step, lastp[lastp.find("est_deg"):lastp.find("est_deg") + 16])
        p = mem["peg0"]
        return {"memory": _full(mem), "act": {"tool": "act_release", "args": {
            "x": p[0], "y": p[1], "z_up": 0.18, "lower_to": Z_GRASP + 0.003, "x_shift": -0.06}}}
    if stage in ("release", "recover", "verify"):
        peg = _peg_pose(cur)
        mem["steps"] = step
        if peg is None:
            mem["stage"] = "verify"
            mem["log"] = log + "|vnopeg@%d" % step
            return {"memory": _full(mem), "act": {"tool": "act_wait", "args": {"n": 2}}}
        rot = _wrap_half(peg["yaw"] - mem["peg_yaw0"])
        th = math.radians(mem["theta_deg"])
        # resolve the pi ambiguity toward the intended rotation
        while rot < th - math.pi / 2:
            rot += math.pi
        while rot > th + math.pi / 2:
            rot -= math.pi
        rem = th - rot
        disp = math.hypot(peg["x"] - mem["peg0"][0], peg["y"] - mem["peg0"][1])
        mem["rho_done"] = rot
        mem["log"] = log + "|v rot%.1f rem%.1f disp%.3f W%.3f@%d" % (
            math.degrees(rot), math.degrees(rem), disp, peg["W"], step)
        left = STEP_LIMIT - step
        need = 14 + abs(rem) / ROT_RATE
        if abs(rem) > math.radians(2.5) and left > need and (mem.get("cycle") or 1) < 3:
            y0 = _heading(cur.tcp)
            plan = _plan(peg["yaw"], rem, y0, allow_delta=False)
            if plan is not None:
                mem["cycle"] = (mem.get("cycle") or 1) + 1
                mem["g_grasp"] = plan["g0"]
                mem["g_final"] = plan["gF"]
                mem["delta"] = plan["delta"]
                mem["stage"] = "approach"
                mem["grasp_xy"] = [peg["x"], peg["y"]]
                return {"memory": _full(mem), "act": {"tool": "act_approach", "args": {
                    "x": peg["x"], "y": peg["y"], "heading": plan["g0"], "z_grasp": Z_GRASP}}}
        mem["stage"] = "hold"
        return {"memory": _full(mem), "act": {"tool": "act_wait", "args": {"n": 4}}}
    # hold: keep still above the peg
    mem["steps"] = step
    mem["stage"] = "hold"
    return {"memory": _full(mem), "act": {"tool": "act_wait", "args": {"n": 4}}}


def _full(mem):
    keys = ["stage", "theta_deg", "peg0", "peg_yaw0", "grasp_xy", "g_grasp", "g_final", "delta", "rho_done",
            "cycle", "steps", "log"]
    out = {}
    for k in keys:
        out[k] = mem.get(k)
    if out["log"] is not None and len(out["log"]) > 1500:
        out["log"] = out["log"][-1500:]
    return out


def act_wait(robot, n=4):
    """Hold the current pose with zero motion commands for n native steps (gripper unchanged-open)."""
    f = float(np.mean(robot.now().fingers))
    g = 1.0 if f > 0.03 else -1.0
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
    return _state(robot.now())


def act_approach(robot, x, y, heading, z_grasp=0.024, max_steps=60):
    """Open gripper, move above (x, y) at hover height while turning to `heading`; descend to
    z_grasp only when heading error < 15 deg and xy error < 1 cm. Stops when xy<4mm, z<5mm,
    heading<2deg (arrived=True) or after max_steps."""
    hist = []
    arrived = False
    for i in range(int(max_steps)):
        fr = robot.now()
        tcp = fr.tcp
        exy = math.hypot(x - float(tcp[0]), y - float(tcp[1]))
        eh = abs(_wrap(heading - _heading(tcp)))
        low = eh < math.radians(15) and exy < 0.01
        zt = z_grasp if low else max(Z_HOVER, z_grasp)
        if not low and float(tcp[2]) < 0.075 and exy > 0.015:
            zt = Z_HOVER
        ez = abs(zt - float(tcp[2]))
        if low and exy < 0.004 and abs(z_grasp - float(tcp[2])) < 0.005 and eh < math.radians(2.0):
            arrived = True
            break
        cmd = _servo(tcp, [x, y, zt], heading)
        robot.step(*cmd, 1.0)
        if i % 3 == 0:
            hist.append([int(fr.step), round(exy, 4), round(float(tcp[2]), 3), round(math.degrees(eh), 1)])
    out = _state(robot.now())
    out.update({"arrived": arrived, "hist": hist[-12:]})
    return out


def act_grasp(robot, x, y, z, heading, steps=6):
    """Close the gripper while holding the grasp pose; returns finger positions per step."""
    fl = []
    for i in range(int(steps)):
        tcp = robot.now().tcp
        cmd = _servo(tcp, [x, y, z], heading, kp=1.0, kr=1.0)
        robot.step(*cmd, -1.0)
        fl.append(round(float(np.mean(robot.now().fingers)), 4))
        if i >= 1 and abs(fl[-1] - fl[-2]) < 0.0007:
            break
    out = _state(robot.now())
    out.update({"fingers_hist": fl})
    return out


def act_rotate(robot, x, y, z, heading, peg_yaw0, theta, rho_start=0.0, max_steps=80, tol_deg=1.0,
               hold_steps=3):
    """Gripper closed: hold the TCP at (x, y, z) and turn about vertical until the peg rotation
    measured by vision (front+wrist top-face yaw relative to peg_yaw0, unwrapped near the
    prediction) reaches theta (rad). The peg-vs-gripper offset is median-filtered; without a valid
    measurement the nominal final `heading` is used. Stops within tol or when the arm stalls, then
    holds still (gripper closed) for hold_steps."""
    hist = []
    done = False
    stall = 0
    h0 = _heading(robot.now().tcp)
    offs = [float(rho_start)]
    prev = None
    nvalid = 0
    for i in range(int(max_steps)):
        fr = robot.now()
        tcp = fr.tcp
        h = _heading(tcp)
        turned = h - h0
        pred = turned + _median(offs[-3:])
        peg = _peg_pose(fr, wrist=True)
        rho = None
        if peg is not None and 0.042 < peg["W"] < 0.058 and peg["n"] >= 40:
            rho = peg["yaw"] - float(peg_yaw0)
            while rho < pred - math.pi / 2:
                rho += math.pi
            while rho > pred + math.pi / 2:
                rho -= math.pi
            if abs(rho - pred) < math.radians(25) or nvalid == 0:
                offs.append(rho - turned)
                nvalid += 1
        est = turned + _median(offs[-3:])
        rem = float(theta) - est
        if nvalid > 0:
            target = h + rem
            lim = math.radians(30)
            target = max(float(heading) - lim, min(float(heading) + lim, target))
        else:
            target = float(heading)
        if abs(_wrap(target - h)) < math.radians(tol_deg):
            done = True
            break
        if prev is not None and abs(_wrap(h - prev)) < 0.002 and abs(rem) > 0.05:
            stall += 1
            if stall >= 4:
                break
        else:
            stall = 0
        prev = h
        cmd = _servo(tcp, [x, y, z], target, kp=1.3, kr=1.2)
        robot.step(*cmd, -1.0)
        hist.append([int(fr.step), round(math.degrees(est), 1),
                     None if rho is None else round(math.degrees(rho), 1),
                     round(float(np.mean(fr.fingers)), 4)])
    # R002: success fired only once the peg stopped moving -> hold still while grasped
    for _ in range(int(hold_steps)):
        tcp = robot.now().tcp
        robot.step(*_servo(tcp, [x, y, z], _heading(tcp), kp=1.0, kr=0.0), -1.0)
    out = _state(robot.now())
    out.update({"done": done, "stall": stall, "est_deg": round(math.degrees(est), 2),
                "nvalid": nvalid, "hist": hist[-12:]})
    return out


def _median(v):
    w = sorted(v)
    n = len(w)
    if n == 0:
        return 0.0
    return 0.5 * (w[(n - 1) // 2] + w[n // 2])


def act_release(robot, x, y, z_up=0.13, lower_to=None, open_steps=4, x_shift=0.0):
    """Optionally lower to `lower_to`, open the gripper for open_steps, then lift to z_up
    (vertical until 4 cm above the grasp height, then also shifting x by x_shift)."""
    tcp = robot.now().tcp
    h = _heading(tcp)
    if lower_to is not None:
        for _ in range(4):
            tcp = robot.now().tcp
            if abs(float(tcp[2]) - lower_to) < 0.004:
                break
            robot.step(*_servo(tcp, [x, y, lower_to], h, kp=1.0, kr=1.0), -1.0)
    zhold = float(robot.now().tcp[2])
    for _ in range(int(open_steps)):
        tcp = robot.now().tcp
        robot.step(*_servo(tcp, [x, y, zhold], h, kp=1.0, kr=1.0), 1.0)
    for _ in range(12):
        tcp = robot.now().tcp
        if float(tcp[2]) > z_up - 0.015:
            break
        xs = x + (x_shift if float(tcp[2]) > zhold + 0.04 else 0.0)
        robot.step(*_servo(tcp, [xs, y, z_up], h, kp=1.3, kr=1.0), 1.0)
    return _state(robot.now())
