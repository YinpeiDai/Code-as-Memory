"""Closed-loop Cartesian control on measured TCP for the LIBERO OSC_POSE delta interface.

Measured from the demonstrations: one full-scale position command moves the TCP
about 0.012 m per step (fitted slope 0.010-0.013 m/step per unit command), and a
rotation command of 1 turns about 0.035 rad per step in the world frame.
The expert keeps the gripper pointing straight down with yaw 0, i.e. the TCP
axis-angle (pi, 0, 0); fingers then close along world y.
"""

POS_GAIN = 20.0       # command per metre of position error (about 0.24 of the error per step)
ROT_GAIN = 4.0        # command per radian; expert rotation commands = 4.0 x log(R_des R^T) (checked on demo 0)
R_DES = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]  # rotvec (pi, 0, 0)
OPEN, CLOSE = -1.0, 1.0


def _rv2mat(r):
    r = np.asarray(r, dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _mat2rv(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-4:  # near pi: use the diagonal
        k = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(k))
        for j in range(3):
            if j != i and (R[i, j] + R[j, i]) < 0:
                k[j] = -k[j]
        return th * k / np.linalg.norm(k)
    return th * v / (2.0 * s)


def _rot_cmd(rv):
    """World-frame rotation command that turns the current orientation toward R_DES."""
    E = np.array(R_DES) @ _rv2mat(rv).T
    e = _mat2rv(E)
    return np.clip(ROT_GAIN * e, -0.5, 0.5)


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _servo(robot, target, grip, vmax=1.0, tol=0.004, max_steps=300):
    """Drive the TCP to target (world xyz) with a saturated P-law; returns (steps, final error)."""
    target = np.asarray(target, dtype=float)
    n = 0
    err = 1.0
    while n < max_steps:
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        e = target - tcp[:3]
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        a = POS_GAIN * e
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        r = _rot_cmd(tcp[3:])
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)
        n += 1
    return n, err


def _hold(robot, grip, steps, target=None):
    """Hold a position (default: where the TCP is now) for a number of steps with the given gripper command."""
    if target is None:
        target = np.asarray(robot.now().tcp, dtype=float)[:3]
    target = np.asarray(target, dtype=float)
    for _ in range(int(steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a = np.clip(POS_GAIN * (target - tcp[:3]), -0.3, 0.3)
        r = _rot_cmd(tcp[3:])
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)


def _gripper(robot, grip, min_steps=8, max_steps=30, target=None):
    """Command the gripper until the finger gap stops changing; returns the final gap."""
    if target is None:
        target = np.asarray(robot.now().tcp, dtype=float)[:3]
    prev = _gap(robot.now())
    still = 0
    for i in range(int(max_steps)):
        _hold(robot, grip, 1, target)
        g = _gap(robot.now())
        still = still + 1 if abs(g - prev) < 2e-4 else 0
        prev = g
        if i + 1 >= min_steps and still >= 4:
            break
    return prev


def _travel(robot, x, y, z, grip, vmax=1.0):
    """Move to (x, y, z): rise first if needed, cross horizontally, then descend (no low diagonal sweeps)."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    steps = 0
    dist = math.hypot(x - tcp[0], y - tcp[1])
    if dist > 0.03:
        h = max(z, float(tcp[2]))
        if h > tcp[2] + 0.01:
            n, _ = _servo(robot, [tcp[0], tcp[1], h], grip, vmax, 0.01)
            steps += n
        n, _ = _servo(robot, [x, y, h], grip, vmax, 0.006)
        steps += n
    n, err = _servo(robot, [x, y, z], grip, vmax, 0.004)
    return steps + n, err


def act_pick(robot, x, y, grasp_z, pre_z, carry_z):
    """Pick an object at world (x, y): open, travel above it to pre_z, descend to grasp_z,
    close until the fingers stop, then lift to carry_z. Returns holding (finger gap > 1 cm),
    the finger gap and the final TCP."""
    steps = 0
    if _gap(robot.now()) < 0.075:
        _gripper(robot, OPEN, 6, 25)
    n, _ = _travel(robot, x, y, pre_z, OPEN, 1.0)
    steps += n
    n, err = _servo(robot, [x, y, grasp_z], OPEN, 0.5, 0.003, 200)
    steps += n
    _hold(robot, OPEN, 3, [x, y, grasp_z])
    gap_closed = _gripper(robot, CLOSE, 20, 40, [x, y, grasp_z])
    _hold(robot, CLOSE, 5, [x, y, grasp_z])
    tcp0 = np.asarray(robot.now().tcp, dtype=float)
    n, _ = _servo(robot, [tcp0[0], tcp0[1], grasp_z + 0.05], CLOSE, 0.5, 0.006, 100)
    steps += n
    n, _ = _servo(robot, [tcp0[0], tcp0[1], carry_z], CLOSE, 1.0, 0.008, 200)
    steps += n
    f = robot.now()
    gap = _gap(f)
    return {"holding": gap > 0.01, "gap": gap, "gap_at_close": gap_closed,
            "grasp_err": err, "tcp": [float(v) for v in f.tcp[:3]], "steps": steps}


def act_place(robot, x, y, release_z, carry_z):
    """Carry the held object at carry_z to above world (x, y), lower it to release_z, open the
    gripper, wait for the fingers, and rise back to carry_z. Returns the finger gaps and TCP."""
    steps = 0
    gap_before = _gap(robot.now())
    n, _ = _travel(robot, x, y, carry_z, CLOSE, 1.0)
    steps += n
    n, err = _servo(robot, [x, y, release_z], CLOSE, 0.6, 0.004, 200)
    steps += n
    _hold(robot, CLOSE, 5, [x, y, release_z])
    gap_open = _gripper(robot, OPEN, 10, 30, [x, y, release_z])
    _hold(robot, OPEN, 10, [x, y, release_z])
    n, _ = _servo(robot, [x, y, carry_z], OPEN, 1.0, 0.008, 200)
    steps += n
    f = robot.now()
    return {"released": gap_open > 0.07, "gap_before": gap_before, "gap_open": gap_open,
            "place_err": err, "tcp": [float(v) for v in f.tcp[:3]], "steps": steps}


def act_hold(robot, steps=40, grip=-1.0):
    """Hold the current pose for `steps` steps (gripper command grip) so the scorer can settle."""
    _hold(robot, grip, steps)
    f = robot.now()
    return {"tcp": [float(v) for v in f.tcp[:3]], "gap": _gap(f)}


def act_move(robot, x, y, z, grip=-1.0):
    """Travel to world (x, y, z) (rise, cross, descend) with gripper command grip."""
    n, err = _travel(robot, x, y, z, grip, 1.0)
    return {"steps": n, "err": err}
