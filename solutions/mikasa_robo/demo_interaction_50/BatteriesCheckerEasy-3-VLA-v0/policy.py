"""BatteriesCheckerEasy-3 policy: perceive layout from the fixed front camera,
then for each of 3 batteries: pick from rack -> insert into socket (lamp is
observed) -> press the confirm button. Memory tracks stage and battery index."""

# Layout offsets calibrated on 50 assigned demos (see task_analysis.md).
GRASP_BIAS = [-0.0079, -0.0006]          # expert grasp TCP - green-top centroid
SOCKET_OFF = [0.1928, 0.1031]            # expert insertion TCP - anchor, +3 mm x to centre battery on the socket disk (R012/R013)
HOLE_TO_TCP = [-0.0028, 0.0004]          # insertion TCP target relative to the wrist-camera socket-disk centre
BUTTON_OFF = [-0.0031, 0.3379]           # expert press TCP - battery anchor
LAMP_OFF = [0.270, 0.098, 0.051]         # lamp white centroid - battery anchor
Z_TRAVEL = 0.20
Z_GRASP = 0.040
Z_RELEASE = 0.045
Z_PRESS = 0.050
Z_SEATED_MAX = 0.052
N_BATT = 3


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotvec(m):
    c = (np.trace(m) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    a = math.acos(c)
    v = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if a < 1e-6:
        return 0.5 * v
    return a / (2.0 * math.sin(a)) * v


def _world_points(frame, mask):
    d = np.asarray(frame.front_depth).astype(float) / 1000.0
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    v, u = np.nonzero(mask & (d > 0))
    z = d[v, u]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _green_points(frame):
    rgb = np.asarray(frame.front_rgb).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return _world_points(frame, (g > r + 30) & (g > b + 20))


def _rack_clusters(frame):
    """Green battery tops resting low (z<0.12): clusters sorted by descending x."""
    W = _green_points(frame)
    if len(W) == 0:
        return []
    W = W[(W[:, 2] > 0.02) & (W[:, 2] < 0.12)]
    if len(W) == 0:
        return []
    o = np.argsort(-W[:, 0])
    W = W[o]
    gaps = np.flatnonzero(np.abs(np.diff(W[:, 0])) > 0.02)
    out = []
    for c in np.split(W, gaps + 1):
        if len(c) >= 5:
            out.append([float(c[:, 0].mean()), float(c[:, 1].mean()), float(c[:, 2].mean()), int(len(c))])
    return out


def _project(frame, p):
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    pc = E[:, :3] @ np.asarray(p, dtype=float) + E[:, 3]
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2] - 0.5
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2] - 0.5
    return int(round(u)), int(round(v))


def _lamp_value(frame, lamp):
    u, v = _project(frame, lamp)
    if u < 2 or v < 2 or u > 253 or v > 253:
        return -1.0
    patch = np.asarray(frame.front_rgb)[v - 2:v + 3, u - 2:u + 3].astype(float)
    return float(patch.mean())


def _wrist_points_img(frame):
    d = np.asarray(frame.wrist_depth).astype(float) / 1000.0
    K = np.asarray(frame.wrist_K, dtype=float)
    E = np.asarray(frame.wrist_extrinsic, dtype=float)
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    x = (uu + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (vv + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    w = (pc - E[:, 3]) @ E[:, :3]
    w = w.reshape(H, W, 3)
    w[d <= 0] = np.nan
    return w, K, E


def _wrist_hole(frame, nominal):
    """Centre of the dark socket disk seen by the wrist camera: dark low-saturation pixels on the grey socket
    top within 3 cm of nominal xy, rays intersected with the socket-top plane. Returns [x, y, z_top, n_dark, n_grey] or None."""
    rgb = np.asarray(frame.wrist_rgb).astype(int)
    mx = rgb.max(2)
    sat = mx - rgb.min(2)
    w, K, E = _wrist_points_img(frame)
    with_d = np.isfinite(w[..., 2])
    wz = np.where(with_d, w[..., 2], 9.0)
    near = with_d & (np.hypot(np.where(with_d, w[..., 0], 9.0) - nominal[0], np.where(with_d, w[..., 1], 9.0) - nominal[1]) < 0.03)
    grey = (sat < 20) & (mx > 110) & near & (wz < 0.15)
    if int(grey.sum()) < 20:
        return None
    ztop = float(np.percentile(wz[grey], 90))
    dark = (sat < 25) & (mx < 115) & (mx > 30) & near & (wz < ztop + 0.01) & (wz > ztop - 0.03)
    if int(dark.sum()) < 5:
        return None
    vv, uu = np.nonzero(dark)
    R = E[:, :3]
    cam = -R.T @ E[:, 3]
    rays = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones(len(uu))], 1) @ R
    sc = (ztop - cam[2]) / rays[:, 2]
    P = cam + rays * sc[:, None]
    return [float(P[:, 0].mean()), float(P[:, 1].mean()), ztop, int(dark.sum()), int(grey.sum())]


def _layout(frame):
    cl = _rack_clusters(frame)
    if len(cl) != N_BATT:
        return None, cl
    xs = [c[0] for c in cl]
    ys = [c[1] for c in cl]
    gaps = [xs[i] - xs[i + 1] for i in range(len(xs) - 1)]
    if min(gaps) < 0.035 or max(gaps) > 0.065 or (max(ys) - min(ys)) > 0.02:
        return None, cl
    ax = sum(xs) / len(xs)
    ay = sum(ys) / len(ys)
    az = sum(c[2] for c in cl) / len(cl)
    lay = {
        "batteries": [[c[0] + GRASP_BIAS[0], c[1] + GRASP_BIAS[1], Z_GRASP] for c in cl],
        "socket": [ax + SOCKET_OFF[0], ay + SOCKET_OFF[1], Z_RELEASE],
        "button": [ax + BUTTON_OFF[0], ay + BUTTON_OFF[1], Z_PRESS],
        "lamp": [ax + LAMP_OFF[0], ay + LAMP_OFF[1], az + LAMP_OFF[2]],
    }
    return lay, cl


def _num_after(text, key):
    """Read the number following '"key": ' in an outcome string; None if absent."""
    if not isinstance(text, str):
        return None
    tag = '"' + key + '": '
    i = text.rfind(tag)
    if i < 0:
        return None
    j = i + len(tag)
    s = ""
    while j < len(text) and (text[j].isdigit() or text[j] in "-.e"):
        s += text[j]
        j += 1
    if text[i + len(tag):i + len(tag) + 4] == "true":
        return 1.0
    if text[i + len(tag):i + len(tag) + 5] == "false":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _goto(robot, target, grip, hold_rpy, tol=0.004, max_steps=40, gain=0.9,
          settle=0.002, stop_z_below=None, rot_gain=0.4, xy_only_tol=None, stall_z=None):
    """Closed-loop move of the TCP to target xyz while holding orientation.
    Returns dict with steps, final error and whether it converged."""
    Rt = _rot(hold_rpy)
    tgt = np.asarray(target, dtype=float)
    steps = 0
    prev = None
    err = None
    converged = False
    for _ in range(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        err = tgt - tcp[:3]
        moved = 1.0 if prev is None else float(np.linalg.norm(tcp[:3] - prev))
        n = float(np.linalg.norm(err))
        if stop_z_below is not None and tcp[2] < stop_z_below:
            converged = True
            break
        if stall_z is not None and tcp[2] < stall_z and moved < 0.0015:
            converged = True
            break
        if xy_only_tol is not None:
            if float(np.linalg.norm(err[:2])) < xy_only_tol and abs(err[2]) < 0.02 and moved < settle * 2:
                converged = True
                break
        elif n < tol and moved < settle:
            converged = True
            break
        prev = tcp[:3]
        cmd = gain * err / 0.1
        m = float(np.max(np.abs(cmd)))
        if m > 1.0:
            cmd = cmd / m
        w = _rotvec(Rt @ _rot(tcp[3:]).T)
        rc = -rot_gain * w / 0.1
        rm = float(np.max(np.abs(rc)))
        if rm > 0.5:
            rc = rc * (0.5 / rm)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                   float(rc[0]), float(rc[1]), float(rc[2]), float(grip))
        steps += 1
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": steps, "converged": converged,
            "err": [round(float(v), 4) for v in (tgt - tcp[:3])]}


def _hold(robot, grip, hold_rpy, n_min=3, n_max=10, until=None):
    """Hold position (zero translation) while the gripper moves; stop once fingers settle."""
    f0 = robot.now()
    anchor = np.asarray(f0.tcp, dtype=float)[:3]
    last_f = float(np.mean(f0.fingers))
    k = 0
    for k in range(1, n_max + 1):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = anchor - tcp[:3]
        cmd = np.clip(0.9 * err / 0.1, -1, 1)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, float(grip))
        fz = float(np.mean(robot.now().fingers))
        if until is not None and until(fz):
            break
        if k >= n_min and abs(fz - last_f) < 0.0005:
            break
        last_f = fz
    return k


def perceive_layout(obs):
    """Detect the 3 green batteries on the rack from the front camera and derive socket/button/lamp positions."""
    lay, cl = _layout(obs.current)
    return {"layout": lay, "clusters": cl}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: init(perceive) -> [pick -> insert -> press] x 3 -> recover if no success."""
    m = dict(memory)
    for k, v in (("stage", None), ("idx", 0), ("attempts", 0), ("batteries", []), ("socket", []),
                 ("button", []), ("lamp", []), ("hold_rpy", []), ("results", []), ("inserted", []),
                 ("queue", []), ("notes", "")):
        if m.get(k) is None:
            m[k] = v
    cur = obs.current
    step_now = int(cur.step)
    stage = m["stage"]
    fingers = float(np.mean(cur.fingers))

    def out(tool, args):
        return {"memory": m, "act": {"tool": tool, "args": args}}

    if stage is None or stage == "init":
        lay, cl = _layout(cur)
        if lay is None:
            m["attempts"] = int(m["attempts"]) + 1
            m["stage"] = "init"
            m["notes"] = "layout detection failed: clusters=" + str(cl)[:300]
            return out("act_wait", {"n": 3})
        m["batteries"] = lay["batteries"]
        m["socket"] = lay["socket"]
        m["button"] = lay["button"]
        m["lamp"] = lay["lamp"]
        tcp = [float(v) for v in cur.tcp]
        m["hold_rpy"] = tcp[3:6]
        m["queue"] = [str(i) for i in range(N_BATT)]
        m["idx"] = 0
        m["attempts"] = 0
        m["stage"] = "pick"
        m["notes"] = "layout ok at step %d" % step_now
        b = m["batteries"][0]
        return out("act_pick", {"x": b[0], "y": b[1], "hold_rpy": m["hold_rpy"]})

    q = list(m["queue"])
    cur_b = int(q[0]) if q else 0
    if stage == "pick":
        grasped = _num_after(last, "grasped")
        if grasped == 1.0 and fingers > 0.003:
            m["stage"] = "insert"
            m["attempts"] = 0
            s = m["socket"]
            return out("act_insert", {"x": s[0], "y": s[1], "hold_rpy": m["hold_rpy"], "lamp": m["lamp"]})
        m["attempts"] = int(m["attempts"]) + 1
        m["notes"] = "grasp failed for battery %d (attempt %d) fingers=%.4f" % (cur_b, m["attempts"], fingers)
        if int(m["attempts"]) >= 4 and len(q) > 1:
            # give up on this battery for now; test the others first
            q = q[1:] + q[:1]
            m["queue"] = q
            m["attempts"] = 0
            cur_b = int(q[0])
        b = m["batteries"][cur_b]
        offs = [[0.0, 0.0], [0.004, 0.0], [-0.004, 0.0], [0.0, 0.004], [0.0, -0.004]]
        o = offs[int(m["attempts"]) % len(offs)]
        return out("act_pick", {"x": b[0] + o[0], "y": b[1] + o[1], "hold_rpy": m["hold_rpy"]})

    if stage == "insert":
        lit = _num_after(last, "lamp_lit")
        back = _num_after(last, "battery_back_in_rack")
        res = list(m["results"])
        res.append("b%d:%s" % (cur_b, "lit" if lit == 1.0 else ("dark" if lit == 0.0 else "unk")))
        m["results"] = res
        if back == 1.0:
            ins = list(m["inserted"])
            if str(cur_b) not in ins:
                ins.append(str(cur_b))
            m["inserted"] = ins
        m["stage"] = "press"
        bt = m["button"]
        return out("act_press", {"x": bt[0], "y": bt[1], "hold_rpy": m["hold_rpy"]})

    if stage == "press":
        q = q[1:]
        if not q:
            # All tested and confirmed but the episode did not end: retest unverified batteries.
            missing = [str(i) for i in range(N_BATT) if str(i) not in m["inserted"]]
            q = missing if missing else [str(i) for i in range(N_BATT)]
            m["notes"] = "no success after full pass at step %d; retest %s" % (step_now, ",".join(q))
            m["idx"] = int(m["idx"]) + 1
        m["queue"] = q
        m["stage"] = "pick"
        m["attempts"] = 0
        b = m["batteries"][int(q[0])]
        return out("act_pick", {"x": b[0], "y": b[1], "hold_rpy": m["hold_rpy"]})

    return out("act_wait", {"n": 5})


def act_wait(robot, n=3):
    """Hold still with open-ish gripper for n steps."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
    return {"waited": int(n)}


def act_pick(robot, x, y, hold_rpy, z_grasp=Z_GRASP, z_travel=Z_TRAVEL):
    """Move above battery (x,y) at travel height with fingers half open, descend, close, lift.
    Returns grasped (fingers not fully closed), per-phase steps and errors."""
    s0 = int(robot.now().step)
    a = _goto(robot, [x, y, z_travel], 0.1, hold_rpy, tol=0.006, max_steps=40, settle=0.004)
    b = _goto(robot, [x, y, z_travel], 0.1, hold_rpy, tol=0.003, max_steps=12, settle=0.003)
    c = _goto(robot, [x, y, z_grasp], 0.1, hold_rpy, tol=0.004, max_steps=30, settle=0.002)
    tcp_g = [round(float(v), 4) for v in robot.now().tcp[:3]]
    k = _hold(robot, -1.0, hold_rpy, n_min=3, n_max=10)
    d = _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.01, max_steps=25, settle=0.01)
    fz = float(np.mean(robot.now().fingers))
    return {"grasped": bool(fz > 0.003), "fingers": round(fz, 4), "tcp_at_grasp": tcp_g,
            "approach": a, "align": b, "descend": c, "close_steps": k, "lift": d,
            "steps": int(robot.now().step) - s0, "step_now": int(robot.now().step)}


def act_insert(robot, x, y, hold_rpy, lamp, z_release=Z_RELEASE, z_travel=Z_TRAVEL):
    """Carry the held battery above the socket (x,y), descend, open, lift and close the gripper.
    Reports lamp brightness during the action and whether the battery returned to the rack."""
    s0 = int(robot.now().step)
    a = _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.006, max_steps=40, settle=0.004)
    b = _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.003, max_steps=12, settle=0.003)
    # Cross-check the front-camera socket target with the wrist camera (disk visible above the held battery).
    h = _wrist_hole(robot.now(), [x, y])
    hole_info = None
    used = "front"
    if h is not None:
        wx, wy = h[0] + HOLE_TO_TCP[0], h[1] + HOLE_TO_TCP[1]
        dev = math.hypot(wx - x, wy - y)
        hole_info = {"wrist_target": [round(wx, 4), round(wy, 4)], "dev": round(dev, 4), "n_dark": h[3], "n_grey": h[4]}
        if h[3] >= 180 and h[4] >= 200 and 0.003 < dev < 0.02:
            x, y = wx, wy
            used = "wrist"
            _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.003, max_steps=12, settle=0.003)
    c = _goto(robot, [x, y, z_release], -1.0, hold_rpy, tol=0.004, max_steps=30, settle=0.002,
              stall_z=0.07)
    # A battery seated in the socket bottoms out at TCP z ~0.047-0.050 (R003-R006). Stalling clearly higher
    # suggests the battery is on the rim: lift slightly and retry with small xy offsets before releasing.
    retries = []
    offs = [[0.003, 0.0], [-0.003, 0.0], [0.0, 0.003], [0.0, -0.003]]
    for o in offs:
        zc = float(robot.now().tcp[2])
        if zc < Z_SEATED_MAX:
            break
        retries.append(round(zc, 4))
        _goto(robot, [x + o[0], y + o[1], zc + 0.02], -1.0, hold_rpy, tol=0.003, max_steps=12, settle=0.003)
        _goto(robot, [x + o[0], y + o[1], z_release], -1.0, hold_rpy, tol=0.004, max_steps=20,
              settle=0.002, stall_z=0.07)
    tcp_r = [round(float(v), 4) for v in robot.now().tcp[:3]]
    k = _hold(robot, 1.0, hold_rpy, n_min=4, n_max=10, until=lambda fz: fz > 0.035)
    d = _goto(robot, [x, y, z_travel], 1.0, hold_rpy, tol=0.01, max_steps=25, settle=0.01)
    frames = robot.frames()
    vals = [_lamp_value(f, lamp) for f in frames]
    lit_frames = [int(f.step) for f, v in zip(frames, vals) if v > 245]
    cl = _rack_clusters(robot.now())
    near_socket = 0
    G = _green_points(robot.now())
    if len(G):
        near_socket = int(np.sum((np.hypot(G[:, 0] - x, G[:, 1] - y) < 0.05) & (G[:, 2] < 0.15)))
    return {"lamp_lit": bool(len(lit_frames) >= 2), "lamp_max": round(max(vals) if vals else -1.0, 1),
            "lit_steps": lit_frames[:12], "battery_back_in_rack": bool(len(cl) == N_BATT),
            "rack_clusters": len(cl), "green_near_socket": near_socket, "tcp_at_release": tcp_r,
            "rim_retries": retries, "hole": hole_info, "target_used": used,
            "approach": a, "align": b, "descend": c, "open_steps": k, "lift": d,
            "steps": int(robot.now().step) - s0, "step_now": int(robot.now().step)}


def act_press(robot, x, y, hold_rpy, z_press=Z_PRESS, z_travel=Z_TRAVEL):
    """Close the gripper, move above the button (x,y), press down to z_press and lift back up."""
    s0 = int(robot.now().step)
    a = _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.006, max_steps=40, settle=0.005)
    b = _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.004, max_steps=10, settle=0.004)
    c = _goto(robot, [x, y, z_press - 0.01], -1.0, hold_rpy, tol=0.004, max_steps=30,
              settle=0.002, stop_z_below=z_press + 0.004, stall_z=0.075)
    zmin = round(float(robot.now().tcp[2]), 4)
    d = _goto(robot, [x, y, z_travel], -1.0, hold_rpy, tol=0.015, max_steps=25, settle=0.01)
    return {"pressed_to_z": zmin, "approach": a, "align": b, "descend": c, "lift": d,
            "steps": int(robot.now().step) - s0, "step_now": int(robot.now().step)}
