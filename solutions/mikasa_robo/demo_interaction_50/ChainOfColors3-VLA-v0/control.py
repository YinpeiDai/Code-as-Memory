"""Closed-loop TCP servo helpers on measured tcp (pd_ee_delta_pose, CPU IK)."""

# Gains chosen from a demo-fitted response model v[t+1] = 0.55 v[t] + 0.0235 a[t]
# (scratch/fit.py); g=1, d=1 settles a 0.3 m move in ~13 steps without overshoot in the model.
POS_GAIN = 1.0
VEL_DAMP = 1.0
ROT_GAIN = 0.5
MAX_ROT_CMD = 0.5
# Per-step change limit on the linear command (R001 showed jumps at segment starts: dz 0 -> -1 -> +0.2
# with 13 mm x coupling and 0.035 rad pitch disturbance). Model cost: ~1 extra step per move.
SLEW = 0.3


def _R(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(M):
    c = max(-1.0, min(1.0, (np.trace(M) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    return th / (2 * s) * v


def _rot_err(rpy_target, rpy_cur):
    """Base-frame rotation vector taking the current orientation to the target."""
    return _rotvec(_R(rpy_target) @ _R(rpy_cur).T)


def _tcp(robot):
    return np.asarray(robot.now().tcp, dtype=np.float64)


def _vel(robot, prev):
    cur = _tcp(robot)[:3]
    if prev is None:
        return np.zeros(3)
    return cur - prev


def _new_state():
    return {"prev": None, "a": [0.0, 0.0, 0.0]}


def _cmd(robot, target, rpy0, gripper, st, gain=POS_GAIN, damp=VEL_DAMP, zcmd=None, max_lin=1.0):
    """One servo step. st carries the previous position and command across segments."""
    t = _tcp(robot)
    v = _vel(robot, st["prev"])
    err = np.asarray(target, dtype=np.float64) - t[:3]
    a = (gain * err - damp * v) / 0.1
    m = float(np.abs(a).max())
    if m > max_lin:
        a = a * (max_lin / m)
    if zcmd is not None:
        a[2] = zcmd
    ap = np.asarray(st["a"], dtype=np.float64)
    a = ap + np.clip(a - ap, -SLEW, SLEW)
    st["a"] = [float(a[0]), float(a[1]), float(a[2])]
    st["prev"] = t[:3]
    w = _rot_err(rpy0, t[3:])
    r = -ROT_GAIN * w / 0.1
    rn = float(np.linalg.norm(r))
    if rn > MAX_ROT_CMD:
        r = r * (MAX_ROT_CMD / rn)
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(gripper))
    return t[:3]


def _move_to(robot, target, rpy0, gripper, st, tol=0.004, vtol=0.003, max_steps=40, max_lin=1.0):
    """Servo to target xyz; returns measured stats. Stops when within tol and slow."""
    n = 0
    target = np.asarray(target, dtype=np.float64)
    for n in range(1, max_steps + 1):
        prev = _cmd(robot, target, rpy0, gripper, st, max_lin=max_lin)
        cur = _tcp(robot)[:3]
        sp = float(np.linalg.norm(cur - prev))
        if float(np.linalg.norm(cur - target)) < tol and sp < vtol:
            break
    cur = _tcp(robot)
    return {
        "steps": n,
        "err": float(np.linalg.norm(cur[:3] - target)),
        "pos": [round(float(c), 4) for c in cur[:3]],
        "rot_err": float(np.linalg.norm(_rot_err(rpy0, cur[3:]))),
    }


def _hold(robot, target, rpy0, gripper, steps, st):
    for _ in range(steps):
        _cmd(robot, target, rpy0, gripper, st)
    return [round(float(c), 4) for c in _tcp(robot)[:3]]
