"""Closed-loop TCP servo helpers and act tools (world-frame OSC_POSE delta steps)."""

POS_SCALE = 0.05      # metres of controller target offset per unit command
ROT_SCALE = 0.5       # radians of controller target rotation per unit command
HALF_PI = 1.5707963267948966
BASE_XY = (-0.51, 0.0)  # robot base xy, from FK of joints vs measured TCP (scratch/fk.py, offset constant)
RETRACT_ABOVE = 0.74   # m from the base: picks farther than this retract before lifting
RETRACT_TO = 0.66
VIA_X = 0.06          # transit waypoint x (closer to the robot base, away from full reach)


def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _mat2aa(R):
    R = np.asarray(R, dtype=float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return ax * th


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _target_rot(ori):
    """'down': gripper pointing down, fingers along world y. 'side_py': pointing +y (drawer handle)."""
    if ori == "side_py":
        return _rx(-HALF_PI)
    return _rx(math.pi)


def _state(robot):
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    fing = np.asarray(f.fingers, dtype=float)
    return tcp[:3].copy(), _aa2mat(tcp[3:6]), float(abs(fing[0]) + abs(fing[1]))


def _servo(robot, target, ori="down", grip=-1.0, vmax=1.0, tol=0.004, rot_tol=0.04,
           max_steps=250, gain=20.0, rot_gain=2.0, settle=0, stall_steps=25, unwind=True, ki=2.0):
    """Drive the TCP to `target` (world xyz) with orientation `ori`; returns measured outcome."""
    target = np.asarray(target, dtype=float)
    Rt = _target_rot(ori)
    g = 1.0 if grip > 0 else -1.0
    steps = 0
    hist = []
    rhist = []
    stalled = False
    unwound = False
    inside = 0
    integ = np.zeros(3)
    p, R, gap = _state(robot)
    while steps < max_steps:
        p, R, gap = _state(robot)
        e = target - p
        er = _mat2aa(Rt @ R.T)
        en, ern = float(np.linalg.norm(e)), float(np.linalg.norm(er))
        if en < tol and ern < rot_tol:
            inside += 1
            if inside > settle:
                break
        else:
            inside = 0
        # integral action near the target removes steady-state offsets (R033: lowering stalled 1.6 cm short)
        if tol < en < 0.03:
            integ = integ + e
        else:
            integ = integ * 0.0
        icmd = integ * ki
        ni = float(np.linalg.norm(icmd))
        if ni > 0.5:
            icmd = icmd * (0.5 / ni)
        cmd = e * gain + icmd
        n = float(np.linalg.norm(cmd))
        if n > vmax:
            cmd = cmd * (vmax / n)
        rc = er * rot_gain
        rn = float(np.linalg.norm(rc))
        if rn > 0.6:
            rc = rc * (0.6 / rn)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), g)
        steps += 1
        hist.append(p)
        rhist.append(ern)
        if len(hist) > stall_steps:
            # no progress in position and orientation while not converged (R036: at the reach limit the
            # orientation could not converge; the servo previously ran to max_steps)
            if float(np.linalg.norm(hist[-1] - hist[-1 - stall_steps])) < 0.002 and \
                    abs(rhist[-1 - stall_steps] - rhist[-1]) < 0.01:
                if unwind and not unwound and en > 0.05:
                    # stuck far from the target (R018: arm at full stretch, elbow joint at its limit):
                    # move down and back toward the robot base, then resume once
                    unwound = True
                    _servo(robot, [p[0] - 0.12, p[1] * 0.6, min(p[2], 0.68) - 0.03], ori=ori, grip=grip,
                           vmax=0.6, tol=0.02, rot_tol=0.2, max_steps=80, unwind=False)
                    hist = []
                    rhist = []
                    continue
                stalled = True
                break
    p, R, gap = _state(robot)
    er = _mat2aa(Rt @ R.T)
    return {"pos": [float(v) for v in p], "err": float(np.linalg.norm(target - p)),
            "rot_err": float(np.linalg.norm(er)), "steps": steps, "stalled": stalled, "unwound": unwound,
            "gap": gap}


def _hold(robot, grip, n):
    g = 1.0 if grip > 0 else -1.0
    gaps = []
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
        gaps.append(_state(robot)[2])
    return gaps


def act_move(robot, x, y, z, ori="down", grip=-1.0, vmax=1.0, tol=0.005, max_steps=250):
    """Servo the TCP to world (x, y, z) with orientation 'down' or 'side_py' and gripper sign grip (+1 close, -1 open).
    Returns measured final position, error, steps, stall flag and finger gap."""
    return _servo(robot, [x, y, z], ori=ori, grip=grip, vmax=vmax, tol=tol, max_steps=max_steps)


def act_gripper(robot, close=True, steps=25):
    """Hold the arm still and close (close=True) or open the gripper for `steps` steps; returns finger gaps."""
    gaps = _hold(robot, 1.0 if close else -1.0, steps)
    return {"gap": gaps[-1] if gaps else None, "gap_start": gaps[0] if gaps else None}


def act_open_drawer(robot, hx, hy, hz, pull=0.24, pre=0.10, retreat=0.12, z_offset=0.0, depth_offset=0.0):
    """Open a drawer whose (closed) handle is at world (hx, hy, hz) facing -y: approach from -y with the gripper
    pointing +y, grasp, pull by `pull` metres along -y, release and retreat. Returns measured outcomes per phase."""
    out = {}
    z = hz + z_offset
    out["pre"] = _servo(robot, [hx, hy - pre, z], ori="side_py", grip=-1, vmax=1.0, tol=0.006, rot_tol=0.04,
                        max_steps=220)
    out["approach"] = _servo(robot, [hx, hy + depth_offset, z], ori="side_py", grip=-1, vmax=0.5, tol=0.003,
                             rot_tol=0.03, max_steps=150, settle=3)
    gaps = _hold(robot, 1.0, 30)
    out["grasp_gap"] = gaps[-1]
    y0 = _state(robot)[0][1]
    out["pull"] = _servo(robot, [hx, hy + depth_offset - pull, z], ori="side_py", grip=1, vmax=0.6, tol=0.004,
                         rot_tol=0.05, max_steps=220, stall_steps=20)
    out["pulled"] = float(y0 - out["pull"]["pos"][1])
    out["pull_gap"] = _state(robot)[2]
    _hold(robot, -1.0, 15)
    p = _state(robot)[0]
    out["retreat"] = _servo(robot, [p[0], p[1] - retreat, p[2]], ori="side_py", grip=-1, vmax=1.0, tol=0.01,
                            rot_tol=0.1, max_steps=120)
    return out


def act_pick(robot, x, y, z_grasp=0.436, z_pre=0.52, z_lift=0.75, close_steps=30):
    """Top-down pick at world (x, y): move above, descend to z_grasp, close, lift to z_lift.
    Returns finger gap after closing and after lifting (near 0 means nothing held)."""
    out = {}
    p = _state(robot)[0]
    if p[1] - y > 0.25:
        # coming from the drawer (near full reach): pass close to the robot base at moderate height
        out["via"] = _servo(robot, [VIA_X, 0.5 * (p[1] + y), 0.72], ori="down", grip=-1, vmax=0.7, tol=0.03,
                            rot_tol=0.15, max_steps=150)
    out["pre"] = _servo(robot, [x, y, z_pre], ori="down", grip=-1, vmax=0.8, tol=0.006, rot_tol=0.03, max_steps=250)
    out["descend"] = _servo(robot, [x, y, z_grasp], ori="down", grip=-1, vmax=0.5, tol=0.003, rot_tol=0.03,
                            max_steps=160, settle=2, stall_steps=15)
    gaps = _hold(robot, 1.0, close_steps)
    out["close_gap"] = gaps[-1]
    bx, by = x - BASE_XY[0], y - BASE_XY[1]
    d = math.sqrt(bx * bx + by * by)
    if d > RETRACT_ABOVE:
        # far pick (cookies, ~0.80 m from the base: elbow at its limit while grasping). Lifting straight up kept
        # the arm stretched and twice led into the stretched singularity (R018, R050); instead lift a little and
        # retract radially toward the base at low height, then lift.
        k = RETRACT_TO / d
        rx, ry = BASE_XY[0] + bx * k, BASE_XY[1] + by * k
        out["lift0"] = _servo(robot, [x, y, z_grasp + 0.05], ori="down", grip=1, vmax=0.5, tol=0.01,
                              rot_tol=0.08, max_steps=60)
        out["retract"] = _servo(robot, [rx, ry, 0.58], ori="down", grip=1, vmax=0.6, tol=0.01, rot_tol=0.06,
                                max_steps=150)
        x, y = rx, ry
    out["lift"] = _servo(robot, [x, y, z_lift], ori="down", grip=1, vmax=0.8, tol=0.01, rot_tol=0.06,
                         max_steps=150)
    out["lift_gap"] = _state(robot)[2]
    return out


def act_place(robot, x, y, z_release, z_transit=0.77, open_steps=20):
    """Carry the held object at z_transit (via a waypoint near the robot base) to above world (x, y), lower to
    z_release, open, then rise 4 cm while retracting toward the base. Returns finger gaps and measured poses."""
    out = {}
    p = _state(robot)[0]
    if p[2] < z_transit - 0.02:
        out["rise"] = _servo(robot, [p[0], p[1], z_transit], ori="down", grip=1, vmax=0.8, tol=0.01,
                             rot_tol=0.06, max_steps=100)
    if y - p[1] > 0.25:
        out["via"] = _servo(robot, [VIA_X, 0.5 * (p[1] + y), z_transit], ori="down", grip=1, vmax=0.8, tol=0.03,
                            rot_tol=0.1, max_steps=150)
    out["transit"] = _servo(robot, [x, y, z_transit], ori="down", grip=1, vmax=0.7, tol=0.008, rot_tol=0.04,
                            max_steps=250)
    out["lower"] = _servo(robot, [x, y, z_release], ori="down", grip=1, vmax=0.5, tol=0.004, rot_tol=0.04,
                          max_steps=120, stall_steps=15)
    out["held_gap"] = _state(robot)[2]
    _hold(robot, -1.0, open_steps)
    out["rise_after"] = _servo(robot, [x - 0.04, y - 0.04, z_release + 0.04], ori="down", grip=-1, vmax=0.6,
                               tol=0.01, rot_tol=0.08, max_steps=100)
    return out


def act_wait(robot, steps=30):
    """Hold position with the gripper open for `steps` steps (lets the scene settle / the checker observe)."""
    _hold(robot, -1.0, steps)
    p, R, gap = _state(robot)
    return {"pos": [float(v) for v in p], "gap": gap}


def act_push_drawer(robot, x, y_from, y_to, z, back=0.12):
    """Close an open drawer by pushing its front with the closed gripper (pointing +y): go to (x, y_from, z),
    push along +y to y_to (stops on stall), then back off by `back` along -y. Returns measured poses."""
    out = {}
    p = _state(robot)[0]
    if p[1] > y_from:
        out["clear"] = _servo(robot, [p[0], y_from, p[2]], ori="side_py", grip=1, vmax=1.0, tol=0.01,
                              rot_tol=0.1, max_steps=100)
    out["pre"] = _servo(robot, [x, y_from, z], ori="side_py", grip=1, vmax=1.0, tol=0.006, rot_tol=0.04,
                        max_steps=200)
    out["push"] = _servo(robot, [x, y_to, z], ori="side_py", grip=1, vmax=0.5, tol=0.004, rot_tol=0.06,
                         max_steps=200, stall_steps=15)
    p = _state(robot)[0]
    out["back"] = _servo(robot, [p[0], p[1] - back, p[2]], ori="side_py", grip=-1, vmax=1.0, tol=0.01,
                         rot_tol=0.1, max_steps=100)
    return out
