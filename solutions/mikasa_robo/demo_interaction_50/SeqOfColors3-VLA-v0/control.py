"""Closed-loop TCP control helpers (pd_ee_delta_pose: 0.1 m / 0.1 rad per unit command)."""

TRAVEL_Z = 0.15
TOUCH_Z = 0.05
TOUCH_X_OFFSET = -0.005   # expert touched ~0.0095 m behind (-x) the measured top-face centre
PRESS_X_OFFSET = 0.008    # expert pressed ~0.015 m in +x of the measured cap centre
LIFT_Z = 0.25             # demos succeed while rising through z ~0.177 after the press
GAIN_P = 1.0
GAIN_D = 1.0
ROT_GAIN = 0.5
ROT_CLIP = 0.3
ROT_DEADBAND = 0.004


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_error(tcp, rot_ref):
    """World-frame rotation vector taking the measured orientation to the reference."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rr = _rpy_to_R(float(rot_ref[0]), float(rot_ref[1]), float(rot_ref[2]))
    return _rotvec(Rr @ Rc.T)


def _command(tcp, prev_tcp, target, rot_ref, vmax=1.0, gain=GAIN_P):
    p = np.asarray(tcp[:3], dtype=np.float64)
    v = p - np.asarray(prev_tcp[:3], dtype=np.float64)
    err = np.asarray(target, dtype=np.float64) - p
    u = (gain * err - GAIN_D * v) / 0.1
    m = float(np.max(np.abs(u)))
    if m > vmax:
        u = u * (vmax / m)
    w = _rot_error(tcp, rot_ref)
    if float(np.linalg.norm(w)) < ROT_DEADBAND:
        r = np.zeros(3)
    else:
        # documented convention: angular offset = -0.1 * r (confirmed in sign by a demo regression)
        r = np.clip(-ROT_GAIN * w / 0.1, -ROT_CLIP, ROT_CLIP)
    return [float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2])], err, v


def _move(robot, target, rot_ref, grip, tol=0.005, vtol=0.003, max_steps=60, min_steps=0, vmax=1.0, ztol=None):
    """Drive the TCP to `target` (world xyz). Returns a measurement summary."""
    prev = np.asarray(robot.now().tcp, dtype=np.float64)
    n = 0
    err = np.asarray(target, dtype=np.float64) - prev[:3]
    v = np.zeros(3)
    zmin = float(prev[2])
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        cmd, err, v = _command(tcp, prev, target, rot_ref, vmax=vmax)
        exy = math.hypot(float(err[0]), float(err[1]))
        ez = abs(float(err[2]))
        zt = tol if ztol is None else ztol
        if n >= min_steps and exy < tol and ez < zt and float(np.linalg.norm(v)) < vtol:
            break
        prev = tcp
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
        n += 1
        zmin = min(zmin, float(robot.now().tcp[2]))
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    e = np.asarray(target, dtype=np.float64) - tcp[:3]
    return {"steps": n, "err": [round(float(x), 4) for x in e], "zmin": round(zmin, 4),
            "rot_err": round(float(np.linalg.norm(_rot_error(tcp, rot_ref))), 4),
            "reached": bool(float(np.linalg.norm(e)) < max(tol, 0.01) * 2)}


def _reach_check(robot, x, y, z, rot_ref):
    try:
        r = robot.dry_run(x, y, z, float(rot_ref[0]), float(rot_ref[1]), float(rot_ref[2]))
        return {"reachable": r.get("reachable"), "margin": r.get("joint_margin")}
    except Exception as exc:
        return {"error": str(exc)[:80]}
