"""Closed-loop Cartesian control on measured TCP for the OSC_POSE delta interface.

Orientation names: 'down' = gripper pointing -z (axis-angle (pi,0,0));
'side' = gripper pointing +y toward the drawer fronts (axis-angle (3pi/2,0,0)),
fingers then close vertically around the horizontal handle bar."""

POS_GAIN = 15.0      # normalized action per metre of error (goal offset = 0.75 * error)
ROT_GAIN = 2.0       # normalized action per radian (matches expert: 2 x error, clipped 0.5)
ROT_CLIP = 0.5


def _skew(k):
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    S = _skew(v / th)
    return np.eye(3) + math.sin(th) * S + (1.0 - math.cos(th)) * (S @ S)


def _mat2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / math.sqrt(max(B[i, i], 1e-12))
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return k * th


def _ori_mat(name):
    if name == "side":
        return _aa2mat([1.5 * math.pi, 0.0, 0.0])
    return _aa2mat([math.pi, 0.0, 0.0])


def _rot_err(tcp, ori):
    Rc = _aa2mat(np.asarray(tcp, dtype=float)[3:6])
    return _mat2aa(_ori_mat(ori) @ Rc.T)


def _command(tcp, p, ori, g, vmax):
    pos = np.asarray(tcp, dtype=float)[:3]
    e = np.asarray(p, dtype=float) - pos
    a = np.clip(POS_GAIN * e, -vmax, vmax)
    r = np.clip(ROT_GAIN * _rot_err(tcp, ori), -ROT_CLIP, ROT_CLIP)
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(g)]


def _fw(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def _run_segment(robot, seg):
    """Execute one segment; returns (reason, steps)."""
    kind = seg.get("kind", "move")
    ori = seg.get("ori", "down")
    g = float(seg.get("g", -1.0))
    vmax = float(seg.get("vmax", 1.0))
    maxn = int(seg.get("max", 250))
    f = robot.now()
    if kind == "grip":
        hold = list(np.asarray(f.tcp, dtype=float)[:3]) if seg.get("p") is None else seg["p"]
        n = int(seg.get("n", 20))
        for i in range(n):
            f = robot.step(*_command(f.tcp, hold, ori, g, 0.3))
        return "done", n
    p = seg["p"]
    tol = float(seg.get("tol", 0.004))
    rtol = float(seg.get("rtol", 0.05))
    settle = bool(seg.get("settle", False))
    hist = []
    for i in range(maxn):
        tcp = np.asarray(f.tcp, dtype=float)
        pe = float(np.linalg.norm(np.asarray(p, dtype=float) - tcp[:3]))
        re = float(np.linalg.norm(_rot_err(tcp, ori)))
        hist.append(tcp[:3])
        speed = float(np.linalg.norm(hist[-1] - hist[-2])) if len(hist) > 1 else 1.0
        if kind == "move" and pe < tol and re < rtol and (not settle or speed < 0.0015):
            return "reached", i
        if kind == "drag":
            if pe < tol:
                return "reached", i
            if i >= 14 and float(np.linalg.norm(hist[-1] - hist[-7])) < 0.002:
                return "stalled", i
        f = robot.step(*_command(tcp, p, ori, g, vmax))
    return "timeout", maxn


def act_sequence(robot, segments):
    """Run a list of motion segments in order with closed-loop TCP control.
    Each segment is a dict:
      {"kind":"move","p":[x,y,z],"ori":"down"|"side","g":-1|1,"tol":m,"settle":bool,"max":n,"vmax":v}
         drive to the pose until within tol (and rotation within rtol);
      {"kind":"drag","p":[x,y,z],"ori":...,"g":1,"max":n} drive toward p and stop when the hand stalls
         (drawer at its stop) or arrives;
      {"kind":"grip","g":1|-1,"n":steps,"ori":...} hold the current pose while closing/opening.
    Returns per-segment reasons/steps, the final measured tcp and finger opening."""
    out = []
    total = 0
    for seg in segments:
        reason, n = _run_segment(robot, seg)
        total += n
        f = robot.now()
        out.append([seg.get("name", seg.get("kind", "move")), reason, n,
                    [round(float(v), 4) for v in np.asarray(f.tcp, dtype=float)[:3]]])
    f = robot.now()
    return {"segments": out, "steps": total,
            "tcp": [round(float(v), 4) for v in np.asarray(f.tcp, dtype=float)],
            "finger_width": round(_fw(f), 4)}


def act_wait(robot, n=100):
    """Hold the current pose with the gripper open for n steps (lets the scorer settle)."""
    f = robot.now()
    hold = list(np.asarray(f.tcp, dtype=float)[:3])
    for i in range(int(n)):
        f = robot.step(*_command(f.tcp, hold, "down", -1.0, 0.3))
    return {"waited": int(n), "tcp": [round(float(v), 4) for v in np.asarray(f.tcp, dtype=float)[:3]]}
