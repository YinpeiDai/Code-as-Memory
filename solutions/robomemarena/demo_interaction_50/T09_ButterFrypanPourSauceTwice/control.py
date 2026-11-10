"""Low-level closed-loop Cartesian control helpers (world frame) for the OSC_POSE delta interface."""

KP_POS = 12.0        # action units per metre of position error (1.0 ~ 1 cm/step)
KP_ROT = 1.6         # action units per radian of orientation error
ROT_MAX = 0.8


def _aa2R(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _R_down(yaw=0.0):
    """Gripper pointing straight down; yaw (rad, about world z) 0 = the demonstrations' working orientation
    (fingers close along world y)."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


def _tilt_deg(frame):
    R = _aa2R(np.asarray(frame.tcp, dtype=float)[3:])
    zc = -float(R[2, 2])
    return math.degrees(math.acos(max(-1.0, min(1.0, zc))))


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _servo(robot, target, Rt, grip, max_steps=200, tol=0.006, rtol=0.05, vmax=1.0, settle=2, ki=0.0):
    """Drive the TCP to `target` (world xyz) with orientation Rt; returns a measured summary.
    ki > 0 adds a bounded integral term (removes the few-mm steady-state offset seen in rollouts)."""
    target = np.asarray(target, dtype=float)
    integ = np.zeros(3)
    good = 0
    n = 0
    e = np.zeros(3)
    r = np.zeros(3)
    for n in range(1, max_steps + 1):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        e = target - tcp[:3]
        r = _R2aa(Rt @ _aa2R(tcp[3:]).T)
        if float(np.linalg.norm(e)) < tol and float(np.linalg.norm(r)) < rtol:
            good += 1
            if good >= settle:
                break
        else:
            good = 0
        if ki > 0 and float(np.linalg.norm(e)) < 0.008:
            integ = np.clip(integ + ki * np.array([e[0], e[1], 0.0]), -0.15, 0.15)
        a = np.clip(KP_POS * e + integ, -vmax, vmax)
        ar = np.clip(KP_ROT * r, -ROT_MAX, ROT_MAX)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), grip)
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    return {"steps": n, "pos_err": float(np.linalg.norm(target - tcp[:3])),
            "rot_err": float(np.linalg.norm(_R2aa(Rt @ _aa2R(tcp[3:]).T))),
            "tcp": [round(float(v), 4) for v in tcp[:3]], "width": round(_width(f), 4),
            "integ": [float(v) for v in integ]}


def _hold(robot, target, Rt, grip, steps, bias=None):
    """Servo in place for a fixed number of steps (e.g. while the fingers move); bias = integral term to keep."""
    target = np.asarray(target, dtype=float)
    b = np.zeros(3) if bias is None else np.asarray(bias, dtype=float)
    for _ in range(int(steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        a = np.clip(KP_POS * (target - tcp[:3]) + b, -1.0, 1.0)
        ar = np.clip(KP_ROT * _R2aa(Rt @ _aa2R(tcp[3:]).T), -ROT_MAX, ROT_MAX)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), grip)
    return _width(robot.now())
