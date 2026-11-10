"""Motion helpers and act tools for T03_CreamPuddingBasket (no imports; np and math are provided).

All motion keeps the expert's fixed top-down gripper orientation (axis-angle [pi, 0, 0]) and uses a
saturated proportional servo on the measured TCP. robot.step gripper: +1 closes, -1 opens."""

DOWN_AA = [math.pi, 0.0, 0.0]
POS_GAIN = 15.0      # action per metre of error (saturates at ~6.7 cm)
ROT_GAIN = 4.0       # action per radian of error (expert used ~4)


def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-4:
        # near pi: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(ax))
        if i == 0:
            ax[1] = math.copysign(ax[1], R[0, 1]); ax[2] = math.copysign(ax[2], R[0, 2])
        elif i == 1:
            ax[0] = math.copysign(ax[0], R[0, 1]); ax[2] = math.copysign(ax[2], R[1, 2])
        else:
            ax[0] = math.copysign(ax[0], R[0, 2]); ax[1] = math.copysign(ax[1], R[1, 2])
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
    return w * th


def _rot_err(tcp):
    """World-frame axis-angle that turns the current hand orientation to the top-down target."""
    return _R2aa(_aa2R(DOWN_AA) @ _aa2R(tcp[3:6]).T)


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0] - f[1]))


def _servo(robot, target, grip, tol=0.004, max_steps=250, vmax=1.0, settle=2):
    """Drive the TCP to target xyz with fixed orientation; returns (reached, steps, final tcp)."""
    target = np.asarray(target, dtype=float)
    ok = 0
    steps = 0
    tcp = np.asarray(robot.now().tcp, dtype=float)
    while steps < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = target - tcp[:3]
        if float(np.max(np.abs(err))) < tol:
            ok += 1
            if ok >= settle:
                return True, steps, tcp
        else:
            ok = 0
        a = np.clip(err * POS_GAIN, -vmax, vmax)
        r = np.clip(_rot_err(tcp) * ROT_GAIN, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        steps += 1
    return False, steps, np.asarray(robot.now().tcp, dtype=float)


def _hold(robot, grip, n):
    """Hold the current target for n steps (orientation still servoed)."""
    tcp0 = np.asarray(robot.now().tcp, dtype=float)
    for _ in range(int(n)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a = np.clip((tcp0[:3] - tcp[:3]) * POS_GAIN, -1, 1)
        r = np.clip(_rot_err(tcp) * ROT_GAIN, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)


def _r(v):
    return [round(float(x), 4) for x in v]


def act_move(robot, x, y, z, gripper=-1.0, lift_first=True, tol=0.004, max_steps=400):
    """Move the top-down gripper to world (x, y, z) metres holding the gripper command (-1 open, +1 closed).

    With lift_first, first rise vertically to z if z is above the current height (avoids sweeping
    through objects). Returns reached flag, steps used, final tcp and finger width."""
    g = 1.0 if gripper > 0 else -1.0
    used = 0
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if lift_first and z > tcp[2] + 0.02:
        _, n, tcp = _servo(robot, [tcp[0], tcp[1], z], g, tol=0.01, max_steps=150, settle=1)
        used += n
    ok, n, tcp = _servo(robot, [x, y, z], g, tol=tol, max_steps=max_steps)
    used += n
    return {"reached": bool(ok), "steps": used, "tcp": _r(tcp[:3]), "width": round(_width(robot.now()), 4)}


def act_grasp(robot, x, y, z_grasp, z_lift=0.78, close_steps=30):
    """Pick from above: open, align over (x, y), descend vertically to z_grasp, close, lift to z_lift.

    Returns the finger width after closing and after lifting (about 0.04-0.05 m when a box is held,
    near 0 when the grasp missed)."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    hover = max(float(tcp[2]), z_grasp + 0.08)
    used = 0
    _, n, tcp = _servo(robot, [x, y, hover], -1.0, tol=0.003, max_steps=200)
    used += n
    # fast descent to 4 cm above the grasp height, then slow final approach
    _, n, tcp = _servo(robot, [x, y, z_grasp + 0.04], -1.0, tol=0.005, max_steps=200, settle=1)
    used += n
    _, n, tcp = _servo(robot, [x, y, z_grasp], -1.0, tol=0.003, max_steps=120, vmax=0.4)
    used += n
    at = _r(np.asarray(robot.now().tcp)[:3])
    for _ in range(int(close_steps)):
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
    used += int(close_steps)
    w_closed = _width(robot.now())
    _, n, tcp = _servo(robot, [x, y, z_lift], 1.0, tol=0.01, max_steps=200, settle=1)
    used += n
    _hold(robot, 1.0, 5)
    used += 5
    return {"grasp_tcp": at, "width_closed": round(w_closed, 4),
            "width_lifted": round(_width(robot.now()), 4), "tcp": _r(np.asarray(robot.now().tcp)[:3]),
            "steps": used}


def act_place(robot, x, y, z_carry=0.80, z_release=0.655, open_steps=20, min_width=0.01):
    """Carry the held object over the basket at (x, y): rise to z_carry, translate, descend to z_release,
    open the gripper, then rise back to z_carry.

    Aborts without opening (and reports dropped) if the finger width falls below min_width while carrying."""
    used = 0
    tcp = np.asarray(robot.now().tcp, dtype=float)
    _, n, tcp = _servo(robot, [tcp[0], tcp[1], z_carry], 1.0, tol=0.01, max_steps=150, settle=1)
    used += n
    _, n, tcp = _servo(robot, [x, y, z_carry], 1.0, tol=0.006, max_steps=300)
    used += n
    _, n, tcp = _servo(robot, [x, y, z_release], 1.0, tol=0.004, max_steps=200)
    used += n
    w = _width(robot.now())
    if w < min_width:
        return {"dropped": True, "width": round(w, 4), "tcp": _r(tcp[:3]), "steps": used}
    for _ in range(int(open_steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1.0)
    used += int(open_steps)
    _, n, tcp = _servo(robot, [x, y, z_carry], -1.0, tol=0.01, max_steps=150, settle=1)
    used += n
    return {"dropped": False, "width_at_release": round(w, 4), "tcp": _r(tcp[:3]), "steps": used}


def act_wait(robot, steps=30):
    """Hold still with the gripper open for `steps` native steps (lets the scorer settle)."""
    _hold(robot, -1.0, int(steps))
    return {"waited": int(steps), "tcp": _r(np.asarray(robot.now().tcp)[:3])}
