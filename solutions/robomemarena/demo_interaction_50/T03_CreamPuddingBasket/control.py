"""Motion primitives on the native OSC_POSE delta interface (world frame)."""

POS_GAIN = 40.0         # action = clip(POS_GAIN * error_m); 1.0 ~ 1 cm/step
ROT_GAIN = 1.0          # action = clip(ROT_GAIN * axis_angle_error / 0.5)
ROT_CLIP = 0.3


def _aa_to_mat(aa):
    a = np.array(aa, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _mat_to_aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
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
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _target_R(yaw=0.0):
    """Gripper pointing straight down; yaw=0 closes the fingers along world y (as in all demos)."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([1.0, -1.0, -1.0])


def _rot_cmd(tcp, yaw=0.0):
    Rc = _aa_to_mat(tcp[3:6])
    err = _mat_to_aa(_target_R(yaw) @ Rc.T)
    return np.clip(ROT_GAIN * err / 0.5, -ROT_CLIP, ROT_CLIP), float(np.linalg.norm(err))


def _move(robot, target, grip, max_steps=250, tol=0.004, speed=1.0, settle=2, stall_steps=12, yaw=0.0):
    """Servo the TCP to world xyz `target` while sending gripper sign `grip`.

    Stops early when the TCP stops progressing (contact): less than 0.6 mm error reduction over
    `stall_steps` steps while still away from the target. Returns final error, steps, stalled flag."""
    target = np.array(target, dtype=float)
    n = 0
    ok_count = 0
    err = None
    hist = []
    stalled = False
    for n in range(1, max_steps + 1):
        f = robot.now()
        tcp = np.array(f.tcp, dtype=float)
        err = target - tcp[:3]
        hist.append(tcp[:3])
        r, rerr = _rot_cmd(tcp, yaw)
        if float(np.linalg.norm(err)) < tol and rerr < 0.06:
            ok_count += 1
            if ok_count >= settle:
                break
        else:
            ok_count = 0
            if (len(hist) > stall_steps and rerr < 0.06
                    and float(np.linalg.norm(target - hist[-1 - stall_steps]))
                    - float(np.linalg.norm(target - hist[-1])) < 0.0006):
                stalled = True
                break
        a = np.clip(POS_GAIN * err, -speed, speed)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    f = robot.now()
    tcp = np.array(f.tcp, dtype=float)
    e = target - tcp[:3]
    return {"steps": n, "err": float(np.linalg.norm(e)), "stalled": stalled, "tcp": [float(v) for v in tcp[:3]]}


def _hold(robot, grip, steps, yaw=0.0):
    """Hold the current position target for `steps` steps with gripper sign `grip`."""
    f = robot.now()
    p = np.array(f.tcp, dtype=float)[:3]
    for _ in range(steps):
        f = robot.now()
        tcp = np.array(f.tcp, dtype=float)
        a = np.clip(POS_GAIN * (p - tcp[:3]), -0.3, 0.3)
        r, _ = _rot_cmd(tcp, yaw)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return [float(v) for v in robot.now().fingers]


def _finger_gap(f):
    fg = np.array(f.fingers, dtype=float)
    return float(abs(fg[0]) + abs(fg[1]))


def act_pick(robot, x, y, z_grasp, z_lift=0.78, approach_dz=0.06, close_steps=25):
    """Open, move above (x, y), descend to z_grasp, close, lift to z_lift.

    If the descent stalls more than 1.5 cm above z_grasp (fingers on an obstacle) the gripper is NOT
    closed; it rises open and reports blocked=True. Returns finger gaps and TCP errors."""
    out = {"blocked": False}
    cur = np.array(robot.now().tcp, dtype=float)[:3]
    z_app = z_grasp + approach_dz
    if cur[2] < z_app:
        out["rise"] = _move(robot, [cur[0], cur[1], z_app], -1, max_steps=120)
    out["above"] = _move(robot, [x, y, z_app], -1, max_steps=250, tol=0.004)
    d = _move(robot, [x, y, z_grasp], -1, max_steps=150, tol=0.003, speed=0.5, settle=3)
    out["descend"] = d
    if d["stalled"] and d["tcp"][2] > z_grasp + 0.015:
        out["blocked"] = True
        out["lift"] = _move(robot, [x, y, z_lift], -1, max_steps=200, tol=0.01)
        out["gap_lifted"] = _finger_gap(robot.now())
        return out
    _hold(robot, 1, close_steps)
    out["gap_closed"] = _finger_gap(robot.now())
    out["lift"] = _move(robot, [x, y, z_lift], 1, max_steps=200, tol=0.01, speed=1.0)
    _hold(robot, 1, 5)
    out["gap_lifted"] = _finger_gap(robot.now())
    return out


def act_push(robot, x_start, y, x_end, z_push, z_safe=0.6):
    """Slide a box along world x with the closed gripper turned 90 deg (hand thin along y).

    Goes to (x_start, y) at z_safe, closes, descends to z_push, moves to x_end, rises and
    turns back to the default yaw. Returns TCP errors of each segment."""
    out = {}
    q = math.pi / 2
    cur = np.array(robot.now().tcp, dtype=float)[:3]
    if cur[2] < z_safe - 0.02:
        out["rise"] = _move(robot, [cur[0], cur[1], z_safe], -1, max_steps=150, tol=0.01)
    out["above"] = _move(robot, [x_start, y, z_safe], 1, max_steps=300, tol=0.005, yaw=q)
    _hold(robot, 1, 10, yaw=q)
    out["down"] = _move(robot, [x_start, y, z_push], 1, max_steps=150, tol=0.004, speed=0.5, yaw=q)
    out["slide"] = _move(robot, [x_end, y, z_push], 1, max_steps=250, tol=0.005, speed=0.3, yaw=q)
    out["up"] = _move(robot, [x_end, y, z_safe], -1, max_steps=150, tol=0.01, yaw=q)
    out["unturn"] = _move(robot, [x_end, y, z_safe], -1, max_steps=100, tol=0.01, yaw=0.0)
    return out


def act_place(robot, x, y, z_release, z_travel=0.78, open_steps=15):
    """Carry the held object at z_travel to above (x, y), descend to z_release, open, rise to z_travel.

    Returns finger gaps before/after opening and TCP errors."""
    out = {}
    cur = np.array(robot.now().tcp, dtype=float)[:3]
    if cur[2] < z_travel - 0.02:
        out["rise"] = _move(robot, [cur[0], cur[1], z_travel], 1, max_steps=150, tol=0.01)
    out["travel"] = _move(robot, [x, y, z_travel], 1, max_steps=300, tol=0.006)
    out["gap_carried"] = _finger_gap(robot.now())
    out["descend"] = _move(robot, [x, y, z_release], 1, max_steps=150, tol=0.004, speed=0.6)
    _hold(robot, -1, open_steps)
    out["gap_open"] = _finger_gap(robot.now())
    out["rise_after"] = _move(robot, [x, y, z_travel], -1, max_steps=150, tol=0.01)
    return out


def act_goto(robot, x, y, z, gripper=-1.0, max_steps=250):
    """Move the TCP to world (x, y, z) holding gripper sign; e.g. return to the observation pose."""
    r = _move(robot, [x, y, z], gripper, max_steps=max_steps, tol=0.005)
    _hold(robot, gripper, 3)
    return r


def act_wait(robot, steps=20, gripper=-1.0):
    """Hold position for `steps` steps (lets the scorer settle)."""
    return {"fingers": _hold(robot, gripper, steps)}
