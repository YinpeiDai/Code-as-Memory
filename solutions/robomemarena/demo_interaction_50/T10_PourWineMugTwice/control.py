"""Pose servo helpers and act tools (world-frame OSC delta control)."""

# Target orientations (rotation matrices, columns = gripper axes in world),
# averaged over the 50 demos (frames ~105 and ~215).
ROT_GRASP = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
ROT_POUR = [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
ROT_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


def _aa2m(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _m2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        w, v = np.linalg.eigh((R + R.T) / 2.0)
        return v[:, int(np.argmax(w))] * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return k * th


def _rot_target(name):
    if name == "grasp":
        return np.array(ROT_GRASP)
    if name == "pour":
        return np.array(ROT_POUR)
    if name == "down":
        return np.array(ROT_DOWN)
    return None


def _errors(frame, target, Rt):
    tcp = np.asarray(frame.tcp, dtype=float)
    ep = np.asarray(target, dtype=float) - tcp[:3]
    if Rt is None:
        er = np.zeros(3)
    else:
        er = _m2aa(Rt @ _aa2m(tcp[3:6]).T)
    return ep, er, tcp


def _scaled(v, k):
    a = k * v
    m = float(np.max(np.abs(a))) if len(a) else 0.0
    if m > 1.0:
        a = a / m
    return a


def act_servo(robot, target, rot="keep", gripper=-1.0, max_steps=150, pos_tol=0.004,
              rot_tol=0.04, hold_steps=0, kp=25.0, kr=3.0, vmax=1.0, kd=50.0, kdr=6.0,
              vel_tol=0.002, stall_ok=False):
    """Servo the TCP to world position `target` [x,y,z] and orientation `rot`
    ('grasp' side grasp, 'pour' bottle horizontal, 'down', or 'keep' = start
    orientation) while sending `gripper` (+1 close, -1 open, 0 hold) each step.
    PD control on measured TCP (per-axis clip to vmax). Arrival = within pos_tol
    (m) / rot_tol (rad) and speed < vel_tol m/step; then `hold_steps` more steps
    are held. Stops after max_steps, or when motion stalls (blocked) for 8 steps
    if stall_ok. Returns measured final pose, errors, stall flag and fingers."""
    f0 = robot.now()
    Rt = _rot_target(rot)
    if Rt is None:
        Rt = _aa2m(np.asarray(f0.tcp, dtype=float)[3:6])
    steps = 0
    held = 0
    reached_at = None
    minf = 1.0
    prev = np.asarray(f0.tcp, dtype=float)
    vel = np.zeros(3)
    wv = np.zeros(3)
    still = 0
    stalled = False
    while steps < int(max_steps):
        ep, er, tcp = _errors(robot.now(), target, Rt)
        if steps > 0:
            vel = tcp[:3] - prev[:3]
            wv = _m2aa(_aa2m(tcp[3:6]) @ _aa2m(prev[3:6]).T)
        prev = tcp
        speed = float(np.linalg.norm(vel))
        ok = (float(np.linalg.norm(ep)) < pos_tol and float(np.linalg.norm(er)) < rot_tol
              and speed < vel_tol)
        if ok and reached_at is None:
            reached_at = steps
        if reached_at is not None:
            if held >= int(hold_steps):
                break
            held += 1
        elif steps > 10 and speed < 0.0005 and float(np.linalg.norm(wv)) < 0.003:
            still += 1
            if stall_ok and still >= 8:
                stalled = True
                break
        else:
            still = 0
        ap = np.clip(kp * ep - kd * vel, -vmax, vmax)
        ar = np.clip(kr * er - kdr * wv, -1.0, 1.0)
        fr = robot.step(float(ap[0]), float(ap[1]), float(ap[2]), float(ar[0]), float(ar[1]), float(ar[2]), float(gripper))
        steps += 1
        minf = min(minf, float(np.sum(np.abs(np.asarray(fr.fingers)))))
    ep, er, tcp = _errors(robot.now(), target, Rt)
    fing = np.asarray(robot.now().fingers, dtype=float)
    return {"steps": steps, "reached": reached_at is not None, "reached_at": reached_at,
            "held": held, "stalled": stalled, "tcp": [float(x) for x in tcp[:3]],
            "pos_err": float(np.linalg.norm(ep)), "rot_err": float(np.linalg.norm(er)),
            "finger_gap": float(np.sum(np.abs(fing))), "min_finger_gap": minf}


def act_gripper(robot, close=True, steps=12, target=None, rot="keep"):
    """Close (close=True) or open the gripper for `steps` steps while holding
    the current (or given `target`) position and orientation. Returns finger gap
    history summary (sum of |finger| in metres; ~0.08 open, ~0.04 on the bottle,
    <0.01 means nothing grasped)."""
    f0 = robot.now()
    tcp0 = np.asarray(f0.tcp, dtype=float)
    tgt = tcp0[:3] if target is None else np.asarray(target, dtype=float)
    Rt = _rot_target(rot)
    if Rt is None:
        Rt = _aa2m(tcp0[3:6])
    g = 1.0 if close else -1.0
    gaps = []
    for _ in range(int(steps)):
        ep, er, tcp = _errors(robot.now(), tgt, Rt)
        ap = _scaled(ep, 10.0)
        ar = _scaled(er, 2.0)
        fr = robot.step(float(ap[0]), float(ap[1]), float(ap[2]), float(ar[0]), float(ar[1]), float(ar[2]), g)
        gaps.append(float(np.sum(np.abs(np.asarray(fr.fingers)))))
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"finger_gap": gaps[-1] if gaps else None, "gap_trace": gaps[::3],
            "tcp": [float(x) for x in tcp[:3]]}
