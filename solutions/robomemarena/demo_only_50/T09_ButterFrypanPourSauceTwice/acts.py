"""Act tools. Every tool keeps the expert grasp orientation (gripper down) unless tilting."""


def act_pick(robot, x, y, z_grasp, z_above, z_lift, close_steps=30, approach_v=0.5):
    """Open gripper, move above (x, y, z_above), descend to z_grasp, close for close_steps, lift to z_lift.
    Returns the final tcp, finger width (m) and whether an object seems held (width 0.008-0.072; open ~0.080, butter ~0.039, bottle ~0.062)."""
    R0 = _R_upright()
    n1 = _servo(robot, [x, y, z_above], R0, -1, vmax=1.0, tol=0.003, max_steps=250, settle=3)
    n2 = _servo(robot, [x, y, z_grasp], R0, -1, vmax=approach_v, tol=0.003, max_steps=250, settle=3)
    _hold(robot, [x, y, z_grasp], R0, -1, 4)
    _hold(robot, [x, y, z_grasp], R0, 1, close_steps)
    w_closed = _width(robot.now())
    _servo(robot, [x, y, z_grasp + 0.04], R0, 1, vmax=0.5, tol=0.006, max_steps=60)
    n3 = _servo(robot, [x, y, z_lift], R0, 1, vmax=1.0, tol=0.008, max_steps=200)
    out = _summary(robot)
    out['width_after_close'] = round(w_closed, 4)
    out['held'] = bool(0.008 < out['width'] < 0.072)
    out['approach_err'] = round(n2[1], 4)
    return out


def act_move(robot, x, y, z, grip=1, vmax=1.0, tol=0.006):
    """Move the tcp to (x, y, z) with the gripper pointing down; grip +1 keeps closed, -1 open."""
    n = _servo(robot, [x, y, z], _R_upright(), grip, vmax=vmax, tol=tol, max_steps=300)
    out = _summary(robot)
    out['steps'] = n[0]
    out['err'] = round(n[1], 4)
    return out


def act_release(robot, x, y, z_transit, z_release, z_after, open_steps=30):
    """Carry the held object to (x, y, z_transit), lower to z_release, open, then rise to z_after."""
    R0 = _R_upright()
    _servo(robot, [x, y, z_transit], R0, 1, vmax=1.0, tol=0.008, max_steps=300)
    n = _servo(robot, [x, y, z_release], R0, 1, vmax=0.6, tol=0.004, max_steps=200, settle=3)
    _hold(robot, [x, y, z_release], R0, 1, 3)
    _hold(robot, [x, y, z_release], R0, -1, open_steps)
    _servo(robot, [x, y, z_after], R0, -1, vmax=1.0, tol=0.008, max_steps=200)
    out = _summary(robot)
    out['release_err'] = round(n[1], 4)
    return out


def act_pour(robot, sx, sy, sz, hx, hy, hz, tilt_deg=45.0, hold_steps=60, untilt=True):
    """Pour: go upright to the start (sx, sy, sz), tilt by tilt_deg about world -y with no translation
    command (as the expert does), hold the tilted bottle at (hx, hy, hz) for hold_steps, then optionally
    return upright. Returns the maximum tilt reached and the final state."""
    R0 = _R_upright()
    Rt = _R_tilted(tilt_deg)
    _servo(robot, [sx, sy, sz], R0, 1, vmax=1.0, tol=0.004, max_steps=300, settle=3)
    _hold(robot, [sx, sy, sz], R0, 1, 3)
    rot = _servo(robot, None, Rt, 1, rtol=math.radians(1.5), max_steps=80, settle=2, hold_pos=False)
    max_tilt = _tilt_deg(robot.now())
    for _ in range(int(hold_steps)):
        fr = _step_towards(robot, [hx, hy, hz], Rt, 1, 0.3, True)
        max_tilt = max(max_tilt, _tilt_deg(fr))
    hold_tilt = _tilt_deg(robot.now())
    if untilt:
        _servo(robot, None, R0, 1, rtol=math.radians(1.5), max_steps=80, settle=2, hold_pos=False)
        _hold(robot, None, R0, 1, 5)
    out = _summary(robot)
    out['max_tilt'] = round(max_tilt, 1)
    out['hold_tilt'] = round(hold_tilt, 1)
    out['rot_steps'] = rot[0]
    return out


def act_wait(robot, steps=100, grip=1):
    """Hold the current pose for a number of steps (grip +1 closed, -1 open), letting the scorer settle."""
    fr = robot.now()
    tgt = [float(v) for v in np.asarray(fr.tcp)[:3]]
    R = _aa2R(np.asarray(fr.tcp)[3:6])
    _hold(robot, tgt, R, grip, steps, vmax=0.3)
    return _summary(robot)
