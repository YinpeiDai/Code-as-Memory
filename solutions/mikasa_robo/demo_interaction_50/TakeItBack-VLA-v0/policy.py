"""TakeItBack-VLA-v0 policy: push the green cube onto the red target until the
target turns magenta, then push the cube back to its recorded original position.

Perception: colour masks in the fixed front RGB-D camera, deprojected to world
coordinates with front_K / front_extrinsic. Control: closed-loop incremental
pose commands on the measured TCP (pd_ee_delta_pose, 0.1 m / 0.1 rad scale).
"""

PUSH_Z = 0.022        # TCP height while pushing (cube is 0.04 tall; demo min z 0.018-0.026)
SAFE_Z = 0.085        # TCP height for travelling over the cube (cube top 0.04)
CONTACT = 0.027       # TCP-to-cube-centre distance during a push (demo wrist estimates 0.025-0.027)
STANDOFF = 0.045      # TCP-to-cube-centre distance before the first push
STANDOFF2 = 0.04      # TCP-to-cube-centre distance before the return push (far side, near reach limit)
MAG_THR = 60          # magenta pixels in the front image => target changed colour
GRIP = -1.0           # gripper closed for pushing
STEP_LIMIT = 60       # native step limit
PITCH_MAX = 0.5       # hand tilt at far reach (model IK reaches x<=0.24 at table height with -0.45, ~0.26 with -0.55)
SLOW_D = 0.085        # push1 lookahead shrinks to d - SLOW_D (cube-target distance d)
SLOW_L = 0.022        # minimum push1 lookahead (~0.009 m/step; slide ~1 cm at <=0.014 m/step, 4.5 cm at 0.02)
K_LAT = 2.0           # push1 lateral steering gain
LOST_LAT = 0.036      # push1 aborts when the cube centre is this far off the TCP push line
D_STOP = 0.05         # push1 stops when cube centre is this close to the target centre (magenta seen at 0.06-0.078)


def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    if ang < math.pi - 1e-3:
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return v / (2 * math.sin(ang)) * ang
    # near pi: axis from the symmetric part
    M = (R + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(M)))
    ax = M[:, i] / math.sqrt(max(M[i, i], 1e-9))
    return ax / np.linalg.norm(ax) * ang


def _rot_cmd(tcp, yaw_t, gain=1.0, pitch_t=0.0):
    """Normalized rotation command that turns the measured orientation toward roll=pi, pitch_t, yaw_t."""
    Rm = _rpy_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_R(math.pi, pitch_t, yaw_t)
    w = _rotvec(Rt @ Rm.T)
    cmd = -w / 0.1 * gain
    n = float(np.linalg.norm(cmd))
    if n > 1.0:
        cmd = cmd / n
    return cmd, float(np.linalg.norm(w))


def _deproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth[v, u].astype(float) / 1000.0
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (pc - t) @ R


def _masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    green = (g > 120) & (r < 90) & (b < 90)
    red = (r > 160) & (g < 100) & (b > g)
    mag = (r > 120) & (b > 150) & (g < 60)
    return green, red, mag


def _cube(frame, cam="front"):
    """Cube centre xy from its top face (world z 0.03-0.05); None if fewer than 6 top-face pixels."""
    rgb = getattr(frame, cam + "_rgb")
    green, _, _ = _masks(rgb)
    if green.sum() < 6:
        return None, 0
    P = _deproject(getattr(frame, cam + "_depth"), getattr(frame, cam + "_K"), getattr(frame, cam + "_extrinsic"), green)
    P = P[(P[:, 2] > 0.03) & (P[:, 2] < 0.05)]
    if len(P) < 6:
        return None, int(len(P))
    return [float(np.median(P[:, 0])), float(np.median(P[:, 1]))], int(len(P))


def _target(frame):
    """Target centre xy from red (or magenta) table-level pixels of the front camera."""
    _, red, mag = _masks(frame.front_rgb)
    m = red if red.sum() >= mag.sum() else mag
    if m.sum() < 20:
        return None
    P = _deproject(frame.front_depth, frame.front_K, frame.front_extrinsic, m)
    P = P[P[:, 2] < 0.012]
    if len(P) < 20:
        return None
    return [float(P[:, 0].mean()), float(P[:, 1].mean())]


def _magenta(frame):
    _, _, mag = _masks(frame.front_rgb)
    return int(mag.sum())


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _wrap_half(a):
    while a > math.pi / 2:
        a -= math.pi
    while a <= -math.pi / 2:
        a += math.pi
    return a


def _pitch_for(x):
    """Hand tilt used at TCP x (the expert tilts to about -0.3 at x>0.15; the arm cannot reach x>~0.19 at
    pitch 0 at table height, model IK + R001 stall at x=0.131 with a 0.1 m command)."""
    return -PITCH_MAX * min(1.0, max(0.0, (x - 0.06) / 0.18))


def _step_to(robot, goal, yaw, kp=1.0, vmax=1.0, pitch=None):
    """One step toward world xyz goal holding orientation (roll=pi, pitch schedule, yaw). The command is
    previewed with robot.dry_run(mode='delta'); if the native IK fails or leaves joint limits the
    translation is halved (up to 3 times). Returns number of halvings (-1 if every preview failed)."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    err = np.asarray(goal, dtype=float) - tcp[:3]
    cmd = err / 0.1 * kp
    m = float(np.max(np.abs(cmd)))
    if m > vmax:
        cmd = cmd * (vmax / m)
    p = _pitch_for(float(goal[0])) if pitch is None else pitch
    rc, _ = _rot_cmd(tcp, yaw, 1.0, p)
    k = 0
    send = None
    while k < 4:
        c6 = [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2])]
        chk = robot.dry_run(*c6, mode="delta")
        if isinstance(chk, dict) and chk.get("within_joint_limits", False) and chk.get("ik_computed", True):
            send = list(chk.get("command", c6))
            break
        cmd = cmd * 0.5
        k += 1
    if send is None:
        send = [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2])]
        k = -1
    robot.step(send[0], send[1], send[2], send[3], send[4], send[5], GRIP)
    return k


def _steps_now(robot):
    return int(robot.now().step)


def _choose_pitch(robot, x, y, z, yaw):
    """Pick the scheduled pitch, or a stronger tilt, for which the absolute IK query reports the pose reachable."""
    base = _pitch_for(x)
    tried = []
    for p in (base, base - 0.1, base - 0.2):
        r = robot.dry_run(x, y, z, math.pi, p, yaw)
        ok = r.get("reachable") if isinstance(r, dict) else None
        tried.append([round(p, 3), ok])
        if ok is None:
            return base, tried
        if ok:
            return p, tried
    return base - 0.2, tried


def act_move(robot, x, y, z, yaw, tol=0.008, gate_xy=0.0, safe_z=0.085, max_steps=20, vmax=1.0, check_reach=False):
    """Closed-loop move of the TCP to world (x, y, z), roll=pi, pitch from the x schedule, given yaw.
    If gate_xy > 0 the TCP stays at or above safe_z until its xy error is below gate_xy.
    Stops when the position error is below tol and the TCP has slowed, or after 3 stalled steps.
    Returns steps, final tcp/error, number of dry-run halvings and stalls."""
    n = 0
    halv = 0
    stalls = 0
    pitch = None
    tried = None
    if check_reach:
        pitch, tried = _choose_pitch(robot, x, y, z, yaw)
    prev = np.asarray(robot.now().tcp, dtype=float)
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        exy = math.hypot(x - tcp[0], y - tcp[1])
        gz = z
        if gate_xy > 0 and exy > gate_xy:
            gz = max(z, safe_z)
        err = math.sqrt(exy ** 2 + (gz - tcp[2]) ** 2)
        speed = float(np.linalg.norm(tcp[:3] - prev[:3]))
        if gz == z and err < tol and (n == 0 or speed < 0.012):
            break
        if n > 0 and speed < 0.003 and err > 2 * tol:
            stalls += 1
            if stalls >= 3:
                break
        prev = tcp
        gx, gy = x, y
        if gate_xy > 0 and exy > gate_xy and tcp[2] < safe_z - 0.02:
            # still low near the cube: rise first, creep in xy (R004 cube moved ~1-2 cm during a diagonal lift)
            gx = tcp[0] + (x - tcp[0]) * 0.25
            gy = tcp[1] + (y - tcp[1]) * 0.25
        k = _step_to(robot, [gx, gy, gz], yaw, vmax=vmax, pitch=pitch)
        halv += max(k, 0) if k >= 0 else 4
        n += 1
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "tcp": tcp.tolist(), "err": float(np.linalg.norm(tcp[:3] - np.array([x, y, z]))),
            "halvings": halv, "stalls": stalls, "pitch": pitch, "reach": tried, "native_step": _steps_now(robot)}


def act_push1(robot, target, cube, yaw, max_steps=25, lookahead=0.07, d_stop=0.05):
    """Push the cube toward the target centre with the closed gripper at PUSH_Z, steering on the
    front-camera cube estimate every step and slowing down near the stop distance. Stops when magenta
    pixels appear (target colour change), when the cube is within D_STOP of the target centre, or when
    the TCP is no longer behind the cube (lost contact)."""
    T = np.asarray(target, dtype=float)
    c = np.asarray(cube, dtype=float)
    n = 0
    seen = 0
    lost = False
    halv = 0
    reason = "max_steps"
    while n < max_steps:
        f = robot.now()
        if _magenta(f) >= MAG_THR:
            reason = "magenta"
            break
        cm, cnt = _cube(f)
        if cm is not None:
            c = np.asarray(cm)
            seen += 1
        d = float(np.linalg.norm(T - c))
        if d < d_stop:
            reason = "near_target"
            break
        u = _unit(T - c)
        tcp = np.asarray(f.tcp, dtype=float)
        rel = tcp[:2] - c
        if float(rel @ u) > -0.012 or abs(float(rel[0] * u[1] - rel[1] * u[0])) > LOST_LAT:
            lost = True
            reason = "lost_contact"
            break
        # decelerate so the cube does not slide on when the push stops (R001/R002: 5-6 cm slide from 0.036 m/step)
        L = min(lookahead, max(SLOW_L, d - SLOW_D))
        # forward advance measured from the TCP itself (front cube estimate can lead the TCP by 0.035-0.045),
        # plus lateral correction onto the cube-target line
        perp = np.array([-u[1], u[0]])
        lat = float((c - tcp[:2]) @ perp)
        # lateral gain K_LAT>1: the cube drifts sideways (away from the x axis) during angled pushes
        # (R008-R037 wrist lateral offsets up to 0.024 m; R038 lost contact at 0.033 with gain 1)
        corr = min(0.03, max(-0.03, K_LAT * lat))
        goal = tcp[:2] + perp * corr + u * L
        k = _step_to(robot, [goal[0], goal[1], PUSH_Z], yaw, vmax=1.0)
        halv += k if k >= 0 else 4
        n += 1
    f = robot.now()
    cm, _ = _cube(f)
    return {"steps": n, "reason": reason, "magenta_px": _magenta(f), "cube": cm, "cube_track": c.tolist(),
            "seen": seen, "lost_contact": lost, "halvings": halv,
            "tcp": np.asarray(f.tcp).tolist(), "native_step": _steps_now(robot)}


def act_lift(robot, dz=0.04, back=0.012, yaw=0.0, max_steps=6):
    """Lift the TCP by dz while backing off `back` m against the push heading (yaw); keep observing until
    the front-camera cube estimate stops moving (< 3 mm/step) or max_steps. Returns the last cube estimate."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    goal = [tcp[0] - back * math.cos(yaw), tcp[1] - back * math.sin(yaw), tcp[2] + dz]
    n = 0
    prevc = None
    still = 0
    while n < max_steps:
        f = robot.now()
        t = np.asarray(f.tcp, dtype=float)
        c, _ = _cube(f)
        if c is not None and prevc is not None and math.hypot(c[0] - prevc[0], c[1] - prevc[1]) < 0.003:
            still += 1
        if c is not None:
            prevc = c
        if t[2] > goal[2] - 0.012 and still >= 1:
            break
        _step_to(robot, goal, yaw)
        n += 1
    c, _ = _cube(robot.now())
    return {"steps": n, "cube": c if c is not None else prevc, "tcp": np.asarray(robot.now().tcp).tolist(),
            "native_step": _steps_now(robot)}


def act_push2(robot, start, goal, yaw, max_steps=20, vmax=1.0, hold=2):
    """Push along the straight line from `start` (TCP xy behind the cube) to `goal` (TCP xy that puts
    the cube centre on its original position) at PUSH_Z, correcting lateral error to the line; then hold
    still for `hold` steps."""
    s = np.asarray(start, dtype=float)
    g = np.asarray(goal, dtype=float)
    v = _unit(g - s)
    n = 0
    halv = 0
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        rem = float((g - tcp[:2]) @ v)
        if rem < 0.004:
            break
        lat = (tcp[:2] - s) - v * float((tcp[:2] - s) @ v)
        ahead = min(0.07, rem)
        aim = tcp[:2] - lat + v * ahead
        k = _step_to(robot, [aim[0], aim[1], PUSH_Z], yaw, vmax=vmax)
        halv += k if k >= 0 else 4
        n += 1
    k = 0
    while k < hold:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        rc, _ = _rot_cmd(tcp, yaw, 0.5, _pitch_for(float(tcp[0])))
        robot.step(0.0, 0.0, 0.0, float(rc[0]), float(rc[1]), float(rc[2]), GRIP)
        k += 1
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "hold": k, "tcp": tcp.tolist(), "remaining": float((g - tcp[:2]) @ v),
            "halvings": halv, "native_step": _steps_now(robot)}


def act_hold(robot, steps=3):
    """Hold the arm still (zero motion, gripper closed) for `steps` native steps."""
    k = 0
    while k < steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP)
        k += 1
    return {"steps": k, "tcp": np.asarray(robot.now().tcp).tolist(), "native_step": _steps_now(robot)}


def _far_point(c1, v):
    """TCP xy behind the cube for the return push; closer stand-off when near the reach limit (x>0.245)."""
    p = np.asarray(c1, dtype=float) - v * STANDOFF2
    if p[0] > 0.245:
        p = np.asarray(c1, dtype=float) - v * 0.032
    return p


def _last_cube(frames):
    for f in reversed(list(frames)):
        c, _ = _cube(f)
        if c is not None:
            return c
    return None


def _mem(memory):
    keys = ["stage", "cube0", "target", "cube1", "yaw_push", "push2_count", "notes", "magenta"]
    out = {}
    for k in keys:
        out[k] = memory.get(k) if isinstance(memory, dict) else None
    return out


def perceive_decide(obs, goal, memory, last):
    """Stage machine: init -> approach -> push1 -> lift -> over -> descend -> push2 -> verify/hold."""
    m = _mem(memory)
    f = obs.current
    tcp = np.asarray(f.tcp, dtype=float)
    stage = m["stage"] or "init"
    notes = m["notes"] or ""
    step = int(f.step)

    def out(st, tool, args, note=None):
        m["stage"] = st
        if note:
            m["notes"] = (notes + " | " + note)[-1500:]
        else:
            m["notes"] = notes
        return {"memory": m, "act": {"tool": tool, "args": args}}

    if any(_magenta(fr) >= MAG_THR for fr in obs.frames):
        m["magenta"] = True

    if stage == "init":
        c0, _ = _cube(f)
        if c0 is None:
            c0, _ = _cube(f, "wrist")
        T = _target(f)
        if c0 is None or T is None:
            return out("init", "act_hold", {"steps": 1}, "s%d no cube/target seen" % step)
        m["cube0"] = c0
        m["target"] = T
        u = _unit(np.asarray(T) - np.asarray(c0))
        yaw = _wrap_half(math.atan2(u[1], u[0]))
        m["yaw_push"] = yaw
        m["push2_count"] = 0
        m["magenta"] = False
        p = np.asarray(c0) - u * STANDOFF
        return out("push1", "act_move", {"x": float(p[0]), "y": float(p[1]), "z": PUSH_Z, "yaw": yaw,
                                          "tol": 0.008, "gate_xy": 0.015, "safe_z": 0.065, "max_steps": 18},
                   "s%d c0=%.3f,%.3f T=%.3f,%.3f yaw=%.2f" % (step, c0[0], c0[1], T[0], T[1], yaw))

    c0 = np.asarray(m["cube0"], dtype=float)
    T = np.asarray(m["target"], dtype=float)
    yaw = float(m["yaw_push"])

    if stage == "push1":
        if m["magenta"]:
            return out("lift", "act_hold", {"steps": 0}, "s%d magenta already" % step)
        c, _ = _cube(f)
        if c is None:
            c = c0.tolist()
        return out("push1_done", "act_push1", {"target": T.tolist(), "cube": c, "yaw": yaw},
                   "s%d push1 from tcp %.3f,%.3f,%.3f cube %.3f,%.3f" % (step, tcp[0], tcp[1], tcp[2], c[0], c[1]))

    if stage == "push1_done" or stage == "lift_check":
        c1 = _last_cube(obs.frames)
        if c1 is not None:
            m["cube1"] = c1
        c = np.asarray(m["cube1"] or c0.tolist(), dtype=float)
        dT = float(np.linalg.norm(T - c))
        if m["magenta"]:
            if stage == "lift_check":
                c1 = np.asarray(m["cube1"] or T.tolist(), dtype=float)
                v = _unit(c0 - c1)
                p = _far_point(c1, v)
                return out("push2", "act_move", {"x": float(p[0]), "y": float(p[1]), "z": PUSH_Z, "yaw": yaw,
                                                  "tol": 0.008, "gate_xy": 0.012, "safe_z": SAFE_Z, "max_steps": 16, "check_reach": True},
                           "s%d magenta after lift; over to %.3f,%.3f (cube1 %.3f,%.3f)" % (step, p[0], p[1], c1[0], c1[1]))
            return out("over", "act_lift", {"dz": 0.045, "back": 0.012, "yaw": yaw},
                       "s%d magenta; cube1=%s dT=%.3f" % (step, str(m["cube1"]), dT))
        if stage == "push1_done":
            # colour change lags about one frame behind the cube entering the target; the cube may also still
            # be sliding (R038: re-approach planned on a stale estimate): lift, wait for it to stop, look
            return out("lift_check", "act_lift", {"dz": 0.045, "back": 0.012, "yaw": yaw},
                       "s%d push1 stopped dT=%.3f, no magenta yet; lift and look" % (step, dT))
        if step > 42:
            return out("lift", "act_hold", {"steps": 0}, "s%d push1 no magenta, giving up" % step)
        u = _unit(T - c)
        p = c - u * STANDOFF
        return out("push1b", "act_move", {"x": float(p[0]), "y": float(p[1]), "z": PUSH_Z, "yaw": yaw,
                                          "tol": 0.008, "gate_xy": 0.012, "safe_z": SAFE_Z, "max_steps": 12},
                   "s%d push1 no magenta (dT=%.3f); re-approach" % (step, dT))

    if stage == "push1b":
        if m["magenta"]:
            return out("over", "act_lift", {"dz": 0.045, "back": 0.012, "yaw": yaw}, "s%d magenta (late)" % step)
        c, _ = _cube(f)
        if c is None:
            c = m["cube1"] or c0.tolist()
        return out("push1_done", "act_push1", {"target": T.tolist(), "cube": c, "yaw": yaw, "d_stop": 0.025},
                   "s%d push1b from tcp %.3f,%.3f,%.3f" % (step, tcp[0], tcp[1], tcp[2]))

    if stage == "lift":
        return out("over", "act_lift", {"dz": 0.04, "back": 0.012, "yaw": yaw}, "s%d lift" % step)

    if stage == "over":
        c1n = _last_cube(obs.frames)
        if c1n is not None:
            m["cube1"] = c1n
        c1 = np.asarray(m["cube1"] or T.tolist(), dtype=float)
        v = _unit(c0 - c1)          # direction of the return push
        p = _far_point(c1, v)        # far side of the cube
        return out("push2", "act_move", {"x": float(p[0]), "y": float(p[1]), "z": PUSH_Z, "yaw": yaw,
                                          "tol": 0.008, "gate_xy": 0.012, "safe_z": SAFE_Z, "max_steps": 16, "check_reach": True},
                   "s%d over to %.3f,%.3f (cube1 %.3f,%.3f)" % (step, p[0], p[1], c1[0], c1[1]))

    if stage == "push2" or stage == "push2_fix":
        c1 = np.asarray(m["cube1"] or T.tolist(), dtype=float)
        v = _unit(c0 - c1)
        g = c0 - v * CONTACT
        along = float((tcp[:2] - c1) @ v)
        if stage == "push2" and (tcp[2] > 0.045 or along > -0.022):
            # far-side move did not finish (stall / reach): descend in place if behind the cube, else retry closer
            if along <= -0.022:
                return out("push2_fix", "act_move", {"x": float(tcp[0]), "y": float(tcp[1]), "z": PUSH_Z, "yaw": yaw,
                                                      "tol": 0.008, "max_steps": 6},
                           "s%d far side high (z %.3f, along %.3f); descend in place" % (step, tcp[2], along))
            p = c1 - v * 0.034
            return out("push2_fix", "act_move", {"x": float(p[0]), "y": float(p[1]), "z": PUSH_Z, "yaw": yaw,
                                                  "tol": 0.008, "gate_xy": 0.012, "safe_z": SAFE_Z, "max_steps": 10,
                                                  "check_reach": True},
                       "s%d not behind cube (along %.3f); retry far side closer" % (step, along))
        m["push2_count"] = int(m["push2_count"] or 0) + 1
        return out("verify", "act_push2", {"start": tcp[:2].tolist(), "goal": g.tolist(), "yaw": yaw},
                   "s%d push2 #%d tcp %.3f,%.3f,%.3f -> %.3f,%.3f" % (step, m["push2_count"], tcp[0], tcp[1], tcp[2], g[0], g[1]))

    if stage == "verify":
        # not yet successful: lift above to let the front camera see the cube
        return out("measure", "act_move", {"x": float(tcp[0]), "y": float(tcp[1]), "z": 0.15, "yaw": yaw,
                                            "tol": 0.02, "max_steps": 5},
                   "s%d no success after push2; lifting to look" % step)

    if stage == "measure":
        c = _last_cube(obs.frames)
        if c is None or step > 52 or int(m["push2_count"] or 0) >= 3:
            return out("hold", "act_hold", {"steps": 2}, "s%d measure cube=%s" % (step, str(c)))
        d = float(np.linalg.norm(np.asarray(c) - c0))
        m["cube1"] = c
        if d < 0.008:
            return out("hold", "act_hold", {"steps": 2}, "s%d cube at %.3f from origin; hold" % (step, d))
        v = _unit(c0 - np.asarray(c))
        p = _far_point(np.asarray(c), v)
        return out("push2", "act_move", {"x": float(p[0]), "y": float(p[1]), "z": PUSH_Z, "yaw": yaw,
                                          "tol": 0.008, "gate_xy": 0.012, "safe_z": SAFE_Z, "max_steps": 12,
                                          "check_reach": True},
                   "s%d cube %.3f,%.3f off by %.3f; re-push" % (step, c[0], c[1], d))

    return out("hold", "act_hold", {"steps": 2}, None)
