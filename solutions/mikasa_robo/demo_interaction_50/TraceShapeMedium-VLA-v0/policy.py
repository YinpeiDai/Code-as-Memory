"""TraceShapeMedium policy: watch the red cube trace a closed shape (while the lamp is red),
then pick up the green cube and move it along the same world-frame path.

Perception: colour masks on the fixed front camera (always unoccluded during the watch phase)
and the wrist camera, deprojected with depth + intrinsics + extrinsics (pixel centres at +0.5).
Control: closed-loop pd_ee_delta_pose steps (translation command = 0.1 m per unit) with a
proportional gain on the measured TCP; orientation held at roll=pi, pitch=0, yaw=0.
"""

STAGES = ("watch", "plan", "grasp", "trace", "hold")
CUBE_TOP_Z = 0.04          # measured top face of the 4 cm cubes (demo depth)
GRASP_Z = 0.033            # TCP height at grasp in demos: 0.030-0.036
CARRY_Z = 0.050            # TCP carry height (demos 0.060); lower = closer to the table-level waypoints
CARRY_Z_LAMP = 0.060       # used when the path passes near the lamp (base top ~0.019 m)
LAMP_CLEAR = 0.075         # xy distance from lamp centre that triggers CARRY_Z_LAMP
HOVER_Z = 0.09             # pre-grasp hover height (wrist refinement); low to keep J4 margin near x=-0.3


# ----------------------------------------------------------------------------- perception
def _deproject(K, E, us, vs, depth_mm):
    z = depth_mm / 1000.0
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


def _mask(rgb, color):
    rgb = rgb.astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "red_cube":      # (216,110,110) top / (138,70,70) side
        return (r > 100) & (np.abs(g - b) < 18) & (r - g > 45) & (g > 40)
    if color == "green_cube":    # (110,216,110) top / (70,138,70) side
        return (g > 100) & (g - r > 45) & (np.abs(r - b) < 18) & (r > 40)
    if color == "lamp_red":      # (255,0..5,0..5)
        return (r > 200) & (g < 40) & (b < 40)
    if color == "lamp_green":
        return (g > 200) & (r < 40) & (b < 40)
    return np.zeros(r.shape, dtype=bool)


def _blob(frame, cam, color, zmin=0.025):
    """Median world position of the top face of a coloured cube seen by camera `cam`."""
    rgb = getattr(frame, cam + "_rgb")
    D = getattr(frame, cam + "_depth")
    m = _mask(rgb, color) & (D > 0)
    vs, us = np.nonzero(m)
    if len(us) < 5:
        return 0, None, False
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    P = _deproject(K, E, us.astype(float), vs.astype(float), D[vs, us].astype(float))
    P = P[P[:, 2] > zmin]
    if len(P) < 5:
        return 0, None, False
    zt = np.percentile(P[:, 2], 90)
    top = P[P[:, 2] > zt - 0.006]
    border = bool(us.min() <= 1 or vs.min() <= 1 or us.max() >= 254 or vs.max() >= 254)
    return int(len(us)), np.median(top, 0), border


def _lamp(frame):
    rgb = frame.front_rgb
    nr = int(_mask(rgb, "lamp_red").sum())
    ng = int(_mask(rgb, "lamp_green").sum())
    if ng > 50 and ng > nr:
        return "green"
    if nr > 50:
        return "red"
    return "off"


def _lamp_xy(frame):
    """World xy of the lamp bulb (pure red/green pixels in the front camera), or None."""
    rgb = frame.front_rgb
    D = frame.front_depth
    m = (_mask(rgb, "lamp_green") | _mask(rgb, "lamp_red")) & (D > 0)
    vs, us = np.nonzero(m)
    if len(us) < 20:
        return None
    P = _deproject(frame.front_K, frame.front_extrinsic, us.astype(float), vs.astype(float),
                   D[vs, us].astype(float))
    return [float(np.median(P[:, 0])), float(np.median(P[:, 1]))]


def _cube_xy(frame, color):
    """Fuse front and wrist estimates of a cube's top-face centre. Returns ([x,y], quality)."""
    nf, pf, bf = _blob(frame, "front", color)
    nw, pw, bw = _blob(frame, "wrist", color)
    okf = pf is not None and nf >= 45 and abs(pf[2] - CUBE_TOP_Z) < 0.012
    okw = pw is not None and nw >= 150 and not bw and abs(pw[2] - CUBE_TOP_Z) < 0.012
    if okf and okw:
        d = float(np.linalg.norm(pf[:2] - pw[:2]))
        if d < 0.008:
            return [float((pf[0] + 2 * pw[0]) / 3), float((pf[1] + 2 * pw[1]) / 3)], "both"
        return [float(pf[0]), float(pf[1])], "front"
    if okf:
        return [float(pf[0]), float(pf[1])], "front"
    if okw:
        return [float(pw[0]), float(pw[1])], "wrist"
    if pf is not None and nf >= 20:
        return [float(pf[0]), float(pf[1])], "front_weak"
    return None, "none"


# ----------------------------------------------------------------------------- path geometry
def _dp(points, eps):
    """Douglas-Peucker simplification of an open polyline (list of [x,y])."""
    P = np.asarray(points, dtype=float)
    if len(P) <= 2:
        return [list(map(float, p)) for p in P]
    keep = np.zeros(len(P), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L = float(np.linalg.norm(ab))
        seg = P[i + 1:j]
        if L < 1e-9:
            d = np.linalg.norm(seg - a, axis=1)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return [[float(p[0]), float(p[1])] for p in P[keep]]


def _plan_path(samples):
    """samples: [[step, x, y, q], ...] red-cube observations before the lamp turned green.
    Returns (vertices, info). Vertices start at the rest position, follow the observed order,
    end at the last observed position and then close back to the start."""
    pts = [[s[1], s[2]] for s in samples if s[3] in ("both", "front", "wrist")]
    if len(pts) < 5:
        pts = [[s[1], s[2]] for s in samples]
    if len(pts) < 2:
        return [], {"n": len(pts)}
    P = np.asarray(pts, dtype=float)
    # drop repeated (stationary) samples
    dd = [P[0]]
    for p in P[1:]:
        if np.linalg.norm(p - dd[-1]) > 0.003:
            dd.append(p)
    P = np.asarray(dd)
    # Only gross single-frame outliers are repaired (>3 cm from the neighbour midpoint, an
    # impossible jump at <=1.6 cm/frame). Smaller deviations are kept so that true corners
    # are not cut; Douglas-Peucker then keeps every sample within its tolerance of the path.
    fixed = 0
    if len(P) >= 3:
        Q = P.copy()
        for i in range(1, len(P) - 1):
            mid = (P[i - 1] + P[i + 1]) / 2
            if np.linalg.norm(P[i] - mid) > 0.03:
                Q[i] = mid
                fixed += 1
        P = Q
    start = P[0]
    verts = _dp(P.tolist(), 0.007)
    if np.linalg.norm(np.asarray(verts[-1]) - start) > 0.004:
        verts.append([float(start[0]), float(start[1])])
    length = float(sum(np.linalg.norm(np.asarray(verts[i + 1]) - np.asarray(verts[i]))
                       for i in range(len(verts) - 1)))
    return verts, {"n": int(len(pts)), "n_unique": int(len(P)), "n_fixed": fixed, "n_vertices": len(verts),
                   "length": round(length, 4)}


# ----------------------------------------------------------------------------- control
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return w / (2 * s) * ang


def _rot_cmd(tcp, gain=0.5, cap=0.3):
    """Rotation arguments that move the measured orientation toward roll=pi, pitch=0, yaw=0.
    CPU IK left-multiplies Rx(-0.1 r0) Ry(-0.1 r1) Rz(-0.1 r2): command = -w/0.1 * gain."""
    R = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_to_R(math.pi, 0.0, 0.0)
    w = _rotvec(Rd @ R.T)
    r = -w / 0.1 * gain
    n = float(np.linalg.norm(r))
    if n > cap:
        r = r * cap / n
    return r, float(np.linalg.norm(w))


def _trans_cmd(err, kp, amax):
    a = np.asarray(err, dtype=float) * kp / 0.1
    n = float(np.linalg.norm(a))
    if n > amax:
        a = a * amax / n
    return a


JLO = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JHI = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


def _jmargin(frame):
    q = np.asarray(frame.joints, dtype=float)
    return float(np.min(np.minimum(q - np.asarray(JLO), np.asarray(JHI) - q)))


def _servo(robot, target, grip, tol=0.005, max_steps=60, kp=1.0, amax=0.7, vtol=0.003):
    """Drive the TCP to target xyz; stop when within tol and moving less than vtol per step."""
    prev = np.asarray(robot.now().tcp[:3], dtype=float)
    steps = 0
    jm = _jmargin(robot.now())
    err = 1.0
    target = np.asarray(target, dtype=float)
    while steps < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        e = target - tcp[:3]
        err = float(np.linalg.norm(e))
        v = float(np.linalg.norm(tcp[:3] - prev))
        if err < tol and (steps > 0 and v < vtol):
            break
        a = _trans_cmd(e, kp, amax)
        r, _ = _rot_cmd(tcp)
        prev = tcp[:3]
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
        jm = min(jm, _jmargin(robot.now()))
    return {"steps": steps, "err": round(err, 4), "jmargin": round(jm, 3)}


def _hold(robot, n, grip):
    for _ in range(n):
        tcp = robot.now().tcp
        r, _ = _rot_cmd(tcp)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), grip)


# ----------------------------------------------------------------------------- decision
def _fresh_memory():
    return {"stage": "watch", "red_samples": [], "lamp_green_step": -1, "green_cube": None,
            "path": [], "grasp_tries": 0, "trace_tries": 0, "carry_z": CARRY_Z_LAMP, "notes": ""}


def perceive_decide(obs, goal, memory, last):
    """Watch phase: log red-cube positions per frame until the lamp turns green; then plan the
    path, grasp the green cube, trace the path and hold."""
    mem = _fresh_memory()
    for k in mem:
        if memory.get(k) is not None:
            mem[k] = memory[k]
    cur = obs.current
    note = ""
    if mem["stage"] == "watch":
        for f in obs.frames:
            lamp = _lamp(f)
            if lamp == "green":
                if mem["lamp_green_step"] < 0:
                    mem["lamp_green_step"] = int(f.step)
                continue
            if mem["lamp_green_step"] >= 0:
                continue
            xy, q = _cube_xy(f, "red_cube")
            if xy is not None:
                if not mem["red_samples"] or mem["red_samples"][-1][0] != int(f.step):
                    mem["red_samples"].append([int(f.step), round(xy[0], 5), round(xy[1], 5), q])
        if mem["lamp_green_step"] >= 0 or int(cur.step) >= 160:
            mem["stage"] = "plan"
        else:
            return {"memory": mem, "act": {"tool": "act_watch", "args": {"max_steps": 160}}}

    if mem["stage"] == "plan":
        path, info = _plan_path(mem["red_samples"])
        mem["path"] = path
        lamp_xy = _lamp_xy(cur)
        zc = CARRY_Z
        if lamp_xy is not None and path:
            dmin = min(math.hypot(p[0] - lamp_xy[0], p[1] - lamp_xy[1]) for p in path)
            if dmin < LAMP_CLEAR:
                zc = CARRY_Z_LAMP
            note += "lamp %s dmin %.3f; " % ([round(v, 3) for v in lamp_xy], dmin)
        else:
            zc = CARRY_Z_LAMP
            note += "lamp not located; "
        mem["carry_z"] = zc
        note += "path %s carry_z %.3f; " % (info, zc)
        mem["stage"] = "grasp"

    if mem["stage"] == "grasp":
        fingers = float(np.mean(cur.fingers))
        held = fingers > 0.012 and float(cur.tcp[2]) > GRASP_Z + 0.008 and mem["grasp_tries"] > 0
        if held:
            mem["stage"] = "trace"
        elif mem["grasp_tries"] >= 3:
            mem["stage"] = "trace"   # proceed anyway (best effort)
            note += "grasp gave up; "
        else:
            xy, q = _cube_xy(cur, "green_cube")
            if xy is None and mem["green_cube"] is not None:
                xy = mem["green_cube"]
            if xy is None:
                xy = [-0.26, 0.0]
                note += "green cube not seen; "
            mem["green_cube"] = [round(xy[0], 5), round(xy[1], 5)]
            mem["grasp_tries"] += 1
            mem["notes"] = (mem["notes"] + note)[-1500:]
            return {"memory": mem, "act": {"tool": "act_grasp", "args": {"x": xy[0], "y": xy[1], "lift_z": mem["carry_z"]}}}

    if mem["stage"] == "trace":
        fingers = float(np.mean(cur.fingers))
        if mem["trace_tries"] > 0 and fingers < 0.010 and mem["grasp_tries"] < 3:
            # a previous trace ended without success and the gripper is empty: re-grasp
            note += "cube lost after trace %d (fingers %.4f); regrasp; " % (mem["trace_tries"], fingers)
            mem["stage"] = "grasp"
            xy, q = _cube_xy(cur, "green_cube")
            if xy is None:
                xy = mem["green_cube"] if mem["green_cube"] is not None else [-0.26, 0.0]
            mem["green_cube"] = [round(xy[0], 5), round(xy[1], 5)]
            mem["grasp_tries"] += 1
            mem["notes"] = (mem["notes"] + note)[-1500:]
            return {"memory": mem, "act": {"tool": "act_grasp", "args": {"x": xy[0], "y": xy[1], "lift_z": mem["carry_z"]}}}
        if mem["trace_tries"] < 3 and len(mem["path"]) >= 2:
            mem["trace_tries"] += 1
            mem["notes"] = (mem["notes"] + note)[-1500:]
            return {"memory": mem, "act": {"tool": "act_trace",
                                            "args": {"path": mem["path"], "z": mem["carry_z"]}}}
        mem["stage"] = "hold"

    mem["notes"] = (mem["notes"] + note)[-1500:]
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


# ----------------------------------------------------------------------------- act tools
def act_watch(robot, max_steps=160):
    """Hold the arm still (gripper open) until the front-camera lamp turns green or max_steps."""
    n = 0
    lamp = _lamp(robot.now())
    while n < max_steps and lamp != "green":
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        lamp = _lamp(robot.now())
    return {"steps": n, "lamp": lamp, "step": int(robot.now().step)}


def act_grasp(robot, x, y, lift_z=0.06):
    """Hover above (x, y), refine the green-cube centre with the wrist camera, descend,
    close the gripper and lift to carry height. Reports measured finger width and offsets."""
    out = {"x0": round(x, 4), "y0": round(y, 4)}
    s1 = _servo(robot, [x, y, HOVER_Z], 1.0, tol=0.008, max_steps=45, amax=0.8)
    nw, pw, bw = _blob(robot.now(), "wrist", "green_cube")
    if pw is not None and nw >= 150 and not bw and abs(pw[2] - CUBE_TOP_Z) < 0.012 \
            and math.hypot(pw[0] - x, pw[1] - y) < 0.04:
        x, y = float(pw[0]), float(pw[1])
        out["refined"] = [round(x, 4), round(y, 4)]
        s1b = _servo(robot, [x, y, HOVER_Z - 0.02], 1.0, tol=0.004, max_steps=20, amax=0.5)
        nw, pw, bw = _blob(robot.now(), "wrist", "green_cube")
        if pw is not None and nw >= 150 and not bw and math.hypot(pw[0] - x, pw[1] - y) < 0.02:
            x, y = float(pw[0]), float(pw[1])
            out["refined2"] = [round(x, 4), round(y, 4)]
    s2 = _servo(robot, [x, y, GRASP_Z + 0.03], 1.0, tol=0.005, max_steps=30, amax=0.6)
    s3 = _servo(robot, [x, y, GRASP_Z], 1.0, tol=0.004, max_steps=25, amax=0.4)
    tcp = robot.now().tcp
    out["tcp_at_grasp"] = [round(float(v), 4) for v in tcp[:3]]
    for _ in range(8):
        r, _ = _rot_cmd(robot.now().tcp)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), -1.0)
    s4 = _servo(robot, [float(tcp[0]), float(tcp[1]), lift_z], -1.0, tol=0.005, max_steps=25, amax=0.5)
    f = robot.now()
    out["fingers"] = [round(float(v), 4) for v in f.fingers]
    out["grasped"] = bool(float(np.mean(f.fingers)) > 0.012)
    out["offset_xy"] = [round(x - float(tcp[0]), 4), round(y - float(tcp[1]), 4)]
    out["servo"] = [s1, s2, s3, s4]
    out["step"] = int(f.step)
    return out


def act_trace(robot, path, z=0.06, lookahead=0.035, vtol=0.009, kp=1.0, amax=0.6):
    """Carry the held cube along the polyline `path` ([[x,y],...]) at TCP height z using a
    look-ahead carrot on the current segment; slows into each vertex. Reports tracking errors."""
    grip = -1.0
    P = [np.asarray(p, dtype=float) for p in path]
    s0 = _servo(robot, [P[0][0], P[0][1], z], grip, tol=0.006, max_steps=50, amax=0.6)
    max_ct = 0.0
    vert_err = []
    steps = 0
    log = []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a
        L = float(np.linalg.norm(ab))
        u = ab / L if L > 1e-9 else np.zeros(2)
        seg_steps = 0
        best = 1.0
        while seg_steps < 80:
            if float(np.mean(robot.now().fingers)) < 0.008:
                f = robot.now()
                return {"dropped": True, "segment": i, "steps": steps, "step": int(f.step),
                        "fingers": [round(float(v), 4) for v in f.fingers]}
            tcp = np.asarray(robot.now().tcp, dtype=float)
            p = tcp[:2]
            db = float(np.linalg.norm(b - p))
            best = min(best, db)
            if db < vtol:
                break
            s = float(np.dot(p - a, u))
            ct = abs(float(u[0] * (p[1] - a[1]) - u[1] * (p[0] - a[0]))) if L > 1e-9 else 0.0
            max_ct = max(max_ct, ct)
            c = a + u * min(L, max(0.0, s) + lookahead)
            e = np.array([c[0] - p[0], c[1] - p[1], z - tcp[2]])
            cmd = _trans_cmd(e, kp, amax)
            r, _ = _rot_cmd(tcp)
            robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(r[0]), float(r[1]), float(r[2]), grip)
            seg_steps += 1
            steps += 1
        vert_err.append(round(best, 4))
        if i % 2 == 0:
            log.append(int(robot.now().step))
    _hold(robot, 3, grip)
    f = robot.now()
    return {"start": s0, "steps": steps, "max_crosstrack": round(max_ct, 4), "vertex_err": vert_err,
            "fingers": [round(float(v), 4) for v in f.fingers], "end_tcp": [round(float(v), 4) for v in f.tcp[:3]],
            "step": int(f.step)}


def act_hold(robot, steps=20):
    """Hold the current pose (gripper closed) for `steps` control steps."""
    _hold(robot, steps, -1.0)
    f = robot.now()
    return {"step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp[:3]]}
