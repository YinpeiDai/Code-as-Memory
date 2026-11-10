"""Motion helpers: Cartesian P-servo on the measured TCP with top-down orientation hold."""

K_POS = 10.0          # action per metre of error (1.0 = 5 cm controller offset); offline sim: no overshoot
K_ROT = 1.0           # rotation action per radian of error
ROT_CLIP = 0.25
R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]   # gripper pointing down, yaw 0 (tcp aa ~ (pi,0,0))


def _aa2mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _mat2aa(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: axis from the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        ax = R[:, i] + np.eye(3)[:, i]
        ax = ax / max(np.linalg.norm(ax), 1e-9)
        return ax * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def _rot_cmd(tcp):
    err = _mat2aa(np.asarray(R_DOWN) @ _aa2mat(tcp[3:6]).T)
    return np.clip(K_ROT * err, -ROT_CLIP, ROT_CLIP)


def _act_vec(tcp, target, vmax=1.0):
    e = np.asarray(target, dtype=float) - np.asarray(tcp[:3], dtype=float)
    return np.clip(K_POS * e, -vmax, vmax)


def _servo(robot, target, grip, tol=0.004, max_steps=200, vmax=1.0, stall_z=False):
    """Drive the TCP to `target` (world xyz) holding gripper command `grip`.

    Stops when within `tol`, after `max_steps`, or (stall_z) when the TCP stops
    descending for 6 steps (contact).  Returns (steps, final error, stalled)."""
    hist = []
    n = 0
    stalled = False
    for n in range(1, max_steps + 1):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        e = np.asarray(target, dtype=float) - tcp[:3]
        if float(np.linalg.norm(e)) < tol:
            break
        hist.append(float(tcp[2]))
        if stall_z and len(hist) > 12 and hist[-7] - hist[-1] < 0.0008:
            stalled = True
            break
        a = _act_vec(tcp, target, vmax)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return n, float(np.linalg.norm(np.asarray(target, dtype=float) - tcp[:3])), stalled


def _hold(robot, grip, steps):
    """Hold the current pose for `steps` steps with gripper command `grip`."""
    f = robot.now()
    tgt = np.asarray(f.tcp, dtype=float)[:3].copy()
    for _ in range(int(steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a = _act_vec(tcp, tgt)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))


def _width(robot):
    fg = np.asarray(robot.now().fingers, dtype=float)
    return float(fg[0] - fg[1])


def _state(robot):
    f = robot.now()
    return {"tcp": [round(float(v), 4) for v in np.asarray(f.tcp)[:3]],
            "width": round(_width(robot), 4), "step": int(f.step)}
