"""Closed-loop motion helpers and act tools for T07.

Controller facts (verified on demo 0, frames 216-300): a rotation command
equals clip(err/0.5, +-0.8) where err is the world-frame axis-angle from the
measured to the target orientation, i.e. one step sets the target relative to
the *current* pose. Position commands behave the same way (0.05 m per unit).
"""

UPRIGHT = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]  # gripper pointing down
TILT_RAD = math.pi / 4   # expert pour: 45 deg about world -y (tool z leans to +x)
POUR_Z = 0.73            # expert pour TCP height
CARRY_Z = 0.80           # expert lift height
APPROACH_Z = 0.60


def _R_from_aa(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _aa_from_R(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _Ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _tilt_deg(tcp):
    """Angle between tool z axis and world down (deg)."""
    z = _R_from_aa(np.asarray(tcp, dtype=float)[3:6])[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(z[2])))))


def _rot_err(tcp, Rt):
    return _aa_from_R(np.asarray(Rt) @ _R_from_aa(np.asarray(tcp, dtype=float)[3:6]).T)


def _goto(robot, target, Rt=None, grip=-1.0, kp=10.0, vmax=1.0, tol=0.004, rtol=0.03,
          max_steps=200, settle=2):
    """P-control the TCP to `target` (xyz) and orientation Rt. Returns final error info."""
    if Rt is None:
        Rt = UPRIGHT
    target = np.asarray(target, dtype=float)
    ok = 0
    n = 0
    tcp = np.asarray(robot.now().tcp, dtype=float)
    for n in range(1, max_steps + 1):
        e = target - tcp[:3]
        er = _rot_err(tcp, Rt)
        if np.linalg.norm(e) < tol and np.linalg.norm(er) < rtol:
            ok += 1
            if ok >= settle:
                break
        else:
            ok = 0
        a = np.clip(kp * e, -vmax, vmax)
        r = np.clip(er / 0.5, -0.8, 0.8)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        tcp = np.asarray(robot.now().tcp, dtype=float)
    e = target - tcp[:3]
    return {"steps": n, "pos_err": float(np.linalg.norm(e)), "rot_err": float(np.linalg.norm(_rot_err(tcp, Rt))),
            "tcp": [round(float(v), 4) for v in tcp[:3]]}


def _rotate_in_place(robot, Rt, grip, steps, rtol=0.017):
    """Rotate toward Rt with zero translation command (as the expert does) for `steps` steps.
    Returns max tilt seen and the steps at which the error got under rtol."""
    max_tilt = 0.0
    reached = None
    for i in range(steps):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        er = _rot_err(tcp, Rt)
        if reached is None and np.linalg.norm(er) < rtol:
            reached = i
        r = np.clip(er / 0.5, -0.8, 0.8)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), grip)
        max_tilt = max(max_tilt, _tilt_deg(robot.now().tcp))
    return {"max_tilt_deg": round(max_tilt, 1), "reached_step": reached,
            "final_tilt_deg": round(_tilt_deg(robot.now().tcp), 1)}


def _hand_report(robot):
    f = robot.now()
    h = _in_hand(f)
    h["tcp"] = [round(float(v), 4) for v in np.asarray(f.tcp)[:3]]
    h["tilt_deg"] = round(_tilt_deg(f.tcp), 1)
    return h


def act_observe(robot, steps=1):
    """Hold still (gripper command 0) for `steps` native steps and report the hand state."""
    for _ in range(max(1, int(steps))):
        robot.step(0, 0, 0, 0, 0, 0, 0)
    return _hand_report(robot)


def act_raise(robot, z=0.80, grip=0.0):
    """Move the TCP straight up to height z keeping x,y (e.g. to clear the view). grip: -1 open, 1 close, 0 hold."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    r = _goto(robot, [tcp[0], tcp[1], z], grip=grip, tol=0.01, max_steps=150)
    out = _hand_report(robot)
    out["move"] = r
    return out


def act_pick(robot, x, y, z=0.468, approach_z=0.60, lift_z=0.80):
    """Open, move above (x, y), descend to TCP height z, close the gripper, lift to lift_z.
    Returns measured finger width, wrist-depth in-hand test and TCP after each stage."""
    out = {}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if tcp[2] < approach_z - 0.02:
        out["up"] = _goto(robot, [tcp[0], tcp[1], approach_z], grip=-1, tol=0.01, max_steps=80)
    for _ in range(6):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    out["above"] = _goto(robot, [x, y, approach_z], grip=-1, tol=0.004, max_steps=200)
    out["descend"] = _goto(robot, [x, y, z], grip=-1, kp=6.0, vmax=0.6, tol=0.003, max_steps=200, settle=3)
    for _ in range(30):
        robot.step(0, 0, 0, 0, 0, 0, 1)
    out["closed_fingers"] = [round(float(v), 4) for v in robot.now().fingers]
    out["lift"] = _goto(robot, [x, y, lift_z], grip=1, tol=0.01, max_steps=200)
    out["hand"] = _hand_report(robot)
    return out


def act_pour(robot, x, y, z=0.73, hold_steps=60, back_steps=35):
    """Carry the held bottle to TCP (x, y, z) upright, tilt 45 deg about world -y with zero
    translation command for hold_steps steps (expert: 60), then rotate back upright for
    back_steps steps. Returns the measured tilt profile and the in-hand state."""
    out = {}
    out["move"] = _goto(robot, [x, y, z], grip=1, kp=8.0, tol=0.004, max_steps=250)
    Rt = _Ry(-TILT_RAD) @ np.asarray(UPRIGHT)
    out["tilt"] = _rotate_in_place(robot, Rt, 1, int(hold_steps))
    out["back"] = _rotate_in_place(robot, UPRIGHT, 1, int(back_steps))
    out["hand"] = _hand_report(robot)
    return out


def act_place(robot, x, y, release_z=0.66, carry_z=0.80):
    """Carry the held bottle upright at carry_z to above (x, y), lower the TCP to release_z,
    open the gripper and retreat upward. Returns measured TCP / finger states."""
    out = {}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["up"] = _goto(robot, [tcp[0], tcp[1], carry_z], grip=1, tol=0.01, max_steps=120)
    out["over"] = _goto(robot, [x, y, carry_z], grip=1, tol=0.005, max_steps=250)
    out["down"] = _goto(robot, [x, y, release_z], grip=1, kp=6.0, vmax=0.6, tol=0.004, max_steps=200)
    for _ in range(25):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    out["fingers_open"] = [round(float(v), 4) for v in robot.now().fingers]
    out["retreat"] = _goto(robot, [x, y, carry_z], grip=-1, tol=0.01, max_steps=150)
    out["hand"] = _hand_report(robot)
    return out
