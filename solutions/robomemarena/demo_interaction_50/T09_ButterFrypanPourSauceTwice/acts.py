"""Act tools: pick, place, pour, observe.  Every tool reports measured TCP/finger outcomes."""


def _rise_then_move(robot, x, y, transit_z, grip, vmax=1.0):
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out = []
    if tcp[2] < transit_z - 0.02:
        out.append(_servo(robot, [tcp[0], tcp[1], transit_z], _R_down(), grip, max_steps=120, tol=0.015, rtol=0.1))
    out.append(_servo(robot, [x, y, transit_z], _R_down(), grip, max_steps=200, tol=0.01, rtol=0.06, vmax=vmax))
    return out


def act_pick(robot, x, y, z_grasp, hover_z=0.66, lift_z=0.80, close_steps=28, min_width=0.01, yaw_deg=0.0):
    """Pick an object at world (x, y): open, hover at hover_z, descend to z_grasp, close, lift to lift_z.
    yaw_deg rotates the gripper about world z (0: fingers close along world y).
    Returns finger widths after closing and after lifting and whether the object appears held."""
    Rd = _R_down(math.radians(yaw_deg))
    _rise_then_move(robot, x, y, hover_z, -1.0)
    _servo(robot, [x, y, hover_z], Rd, -1.0, max_steps=60, tol=0.01, rtol=0.05)
    d = _servo(robot, [x, y, z_grasp], Rd, -1.0, max_steps=150, tol=0.004, rtol=0.04, vmax=0.7, settle=3,
               ki=1.0)
    w_closed = _hold(robot, [x, y, z_grasp], Rd, 1.0, close_steps, bias=d["integ"])
    d.pop("integ", None)
    tcp_g = [round(float(v), 4) for v in robot.now().tcp[:3]]
    up = _servo(robot, [x, y, lift_z], Rd, 1.0, max_steps=150, tol=0.01, rtol=0.06, vmax=1.0)
    up.pop("integ", None)
    w_lift = _width(robot.now())
    return {"descend": d, "grasp_tcp": tcp_g, "width_closed": round(w_closed, 4), "width_lifted": round(w_lift, 4),
            "lift": up, "held": bool(min_width < w_lift < 0.074 and min_width < w_closed < 0.074)}


def act_place(robot, x, y, z_release, transit_z=0.80, open_steps=25, retreat=None):
    """Carry the held object at transit_z to above (x, y), descend to z_release, open, rise back to transit_z,
    then optionally move to retreat [x, y, z].  Returns finger widths before/after release."""
    _rise_then_move(robot, x, y, transit_z, 1.0)
    w_before = _width(robot.now())
    d = _servo(robot, [x, y, z_release], _R_down(), 1.0, max_steps=150, tol=0.006, rtol=0.05, vmax=0.8)
    w_opened = _hold(robot, [x, y, z_release], _R_down(), -1.0, open_steps)
    up = _servo(robot, [x, y, transit_z], _R_down(), -1.0, max_steps=120, tol=0.015, rtol=0.1)
    r = None
    if retreat is not None:
        r = _servo(robot, retreat, _R_down(), -1.0, max_steps=200, tol=0.015, rtol=0.1)
    return {"width_before": round(w_before, 4), "release": d, "width_opened": round(w_opened, 4),
            "rise": up, "retreat": r}


def act_pour(robot, x, y, z, tilt_deg=45.0, hold_steps=60, untilt=True, min_width=0.01):
    """With the bottle held, move to (x, y, z) upright, tilt about world -y by tilt_deg (spout over the pan),
    hold hold_steps, then optionally return upright.  Returns tilt reached, steps tilted (>30 deg) and width."""
    _rise_then_move(robot, x, y, max(z, float(robot.now().tcp[2])), 1.0)
    m = _servo(robot, [x, y, z], _R_down(), 1.0, max_steps=120, tol=0.008, rtol=0.05)
    Rt = _rot_y(-math.radians(tilt_deg)) @ _R_down()
    tilt = _servo(robot, [x, y, z], Rt, 1.0, max_steps=80, tol=0.03, rtol=0.06, settle=1)
    tilted = 0
    mx = 0.0
    for _ in range(int(hold_steps)):
        _hold(robot, [x, y, z], Rt, 1.0, 1)
        td = _tilt_deg(robot.now())
        mx = max(mx, td)
        if td > 30.0:
            tilted += 1
    back = None
    if untilt:
        back = _servo(robot, [x, y, z], _R_down(), 1.0, max_steps=80, tol=0.02, rtol=0.05, settle=2)
        _hold(robot, [x, y, z], _R_down(), 1.0, 8)
    w = _width(robot.now())
    return {"move": m, "tilt": tilt, "max_tilt_deg": round(mx, 1), "steps_tilted": tilted,
            "end_tilt_deg": round(_tilt_deg(robot.now()), 1), "untilt": back, "width": round(w, 4),
            "held": bool(min_width < w < 0.074)}


def act_move(robot, x, y, z, grip=-1.0, max_steps=200):
    """Move the TCP (gripper pointing down) to world (x, y, z), holding gripper command grip (+1 close, -1 open)."""
    return _servo(robot, [x, y, z], _R_down(), grip, max_steps=max_steps, tol=0.01, rtol=0.06)


def act_observe(robot, steps=5, grip=0.0):
    """Hold the current pose for a few steps (re-observation)."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    _hold(robot, tcp[:3], _R_down(), grip, steps)
    return {"tcp": [round(float(v), 4) for v in robot.now().tcp[:3]], "width": round(_width(robot.now()), 4)}
