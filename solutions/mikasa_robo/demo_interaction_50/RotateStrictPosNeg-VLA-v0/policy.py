"""RotateStrictPosNeg policy: perceive the blue peg (top face) with front+wrist RGB-D,
grasp it across its width at its centre, rotate the wrist by the instructed angle
(positive = counter-clockwise seen from above = yaw increase), hold, then release."""

PEG_TOP_Z = 0.05          # measured in all 50 demos (top-face points 0.044..0.056)
TOP_BAND = (0.044, 0.056)
PRE_Z = 0.10              # pre-grasp TCP height
GRASP_Z = 0.03            # TCP height for grasping (peg is 0.05 tall)
YAW_MIN = -2.00           # rollouts R001-R031: J7 ~= 0.45 - yaw for yaw<0 -> J7 ~2.45 (limit 2.897)
YAW_MAX = 2.80
STEP_LIMIT = 90


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_half(a):
    """Wrap a line orientation into (-pi/2, pi/2]."""
    a = _wrap(a)
    if a > math.pi / 2:
        a -= math.pi
    if a <= -math.pi / 2:
        a += math.pi
    return a


def _parse_angle(goal):
    """Signed degrees from 'Rotate the peg by -14 degrees ...'."""
    text = str(goal)
    i = text.find(" by ")
    if i < 0:
        return None
    rest = text[i + 4:].strip()
    num = ""
    for ch in rest:
        if ch in "+-0123456789.":
            num += ch
        else:
            break
    if num in ("", "+", "-"):
        return None
    return float(num)


def _blue_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (b > 100) & (b > r + 50) & (b > g + 30)


def _cam_points(frame, cam):
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    m = _blue_mask(rgb) & (depth > 0)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs].astype(float) / 1000.0
    X = (xs + 0.5 - K[0, 2]) * z / K[0, 0]
    Y = (ys + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([X, Y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _peg_pose(frame, cams=("front", "wrist"), exclude_xy=None, exclude_r=0.0):
    """Peg top-face estimate: centre (bbox midpoint in peg axes), yaw of long axis
    in (-pi/2, pi/2], length, width, point count."""
    pts = [_cam_points(frame, c) for c in cams]
    P = np.concatenate(pts, 0)
    if len(P) == 0:
        return None
    P = P[(P[:, 2] > TOP_BAND[0]) & (P[:, 2] < TOP_BAND[1])]
    if exclude_xy is not None and len(P):
        d = np.hypot(P[:, 0] - exclude_xy[0], P[:, 1] - exclude_xy[1])
        P = P[d > exclude_r]
    if len(P) < 30:
        return None
    xy = P[:, :2]
    c = xy.mean(0)
    C = np.cov((xy - c).T)
    w, v = np.linalg.eigh(C)
    a = v[:, 1]
    yaw = _wrap_half(math.atan2(a[1], a[0]))
    a = np.array([math.cos(yaw), math.sin(yaw)])
    b = np.array([-a[1], a[0]])
    u = (xy - c) @ a
    s = (xy - c) @ b
    cb = c + a * (u.max() + u.min()) / 2.0 + b * (s.max() + s.min()) / 2.0
    return {"center": [float(cb[0]), float(cb[1])], "yaw": float(yaw),
            "length": float(u.max() - u.min()), "width": float(s.max() - s.min()),
            "n": int(len(P))}


def _rot(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return v * (ang / (2.0 * s))


def _unwrap_to(a, ref):
    """Angle a shifted by multiples of 2pi to be closest to ref."""
    return ref + _wrap(a - ref)


def _command(tcp, yaw_c, target_xyz, target_yaw, kp=1.0, kr=1.0, vmax=1.0):
    """Normalized delta command toward (xyz, gripper-down, yaw). yaw_c is the
    unwrapped current yaw. Rotation uses a world-frame rotation vector: the
    native offset is -0.1*r, so r = -omega/0.1."""
    pos = np.asarray(tcp[:3], dtype=float)
    d = (np.asarray(target_xyz, dtype=float) - pos) / 0.1 * kp
    d = np.clip(d, -vmax, vmax)
    Rc = _rot(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rlevel = _rot(math.pi, 0.0, float(tcp[5]))
    tilt = _rotvec(Rlevel @ Rc.T)
    yaw_err = target_yaw - yaw_c
    omega = np.array([tilt[0], tilt[1], yaw_err])
    r = -omega / 0.1 * kr
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return [float(d[0]), float(d[1]), float(d[2]), float(r[0]), float(r[1]), float(r[2])], yaw_err


def _jvel(robot, prev):
    q = np.asarray(robot.now().joints, dtype=float)
    return q, (0.0 if prev is None else float(np.abs(q - prev).max()))


def perceive_decide(obs, goal, memory, last):
    """Stage machine: plan -> align -> grasp -> rotate -> release -> hold."""
    mem = dict(memory)
    for k in ("stage", "goal_deg", "peg_center", "peg_yaw0", "grasp_yaw", "final_yaw",
              "yaw_ref", "notes"):
        mem.setdefault(k, None)
    cur = obs.current
    stage = mem.get("stage") or "plan"
    notes = mem.get("notes") or ""
    if stage == "plan":
        ang = _parse_angle(goal)
        pose = _peg_pose(cur)
        tcp = cur.tcp
        yaw_now = float(tcp[5])
        if ang is None or pose is None:
            mem["stage"] = "plan"
            mem["notes"] = notes + "|perception/goal failed"
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 1}}}
        th = math.radians(ang)
        phi = pose["yaw"]
        best = None
        cands = []
        for k in (-1, 0, 1):
            psi = phi + k * math.pi
            lo, hi = min(psi, psi + th), max(psi, psi + th)
            ok = lo >= YAW_MIN and hi <= YAW_MAX
            cost = abs(psi - yaw_now) + (0.0 if ok else 100.0)
            cands.append((cost, psi))
        cands.sort()
        psi = cands[0][1]
        cand_list = [float(c[1]) for c in cands if c[0] < 100.0] or [float(psi)]
        mem["goal_deg"] = float(ang)
        mem["peg_center"] = pose["center"]
        mem["peg_yaw0"] = float(phi)
        mem["grasp_yaw"] = float(psi)
        mem["final_yaw"] = float(psi + th)
        mem["yaw_ref"] = yaw_now
        mem["stage"] = "grasp"
        mem["notes"] = "plan: ang=%.1f phi=%.1fdeg c=(%.3f,%.3f) L=%.3f W=%.3f n=%d psi=%.1fdeg" % (
            ang, math.degrees(phi), pose["center"][0], pose["center"][1], pose["length"],
            pose["width"], pose["n"], math.degrees(psi))
        return {"memory": mem, "act": {"tool": "act_align_grasp", "args": {
            "cx": pose["center"][0], "cy": pose["center"][1], "grasp_yaws": cand_list,
            "theta": th, "yaw_ref": yaw_now}}}
    if stage == "grasp":
        mem["stage"] = "rotate"
        mem["notes"] = notes + "|grasp done"
        c = mem["peg_center"]
        return {"memory": mem, "act": {"tool": "act_rotate", "args": {
            "cx": c[0], "cy": c[1], "peg_yaw0": mem["peg_yaw0"],
            "theta": math.radians(mem["goal_deg"]), "yaw_ref": mem["grasp_yaw"],
            "correct": False}}}
    if stage == "rotate":
        mem["stage"] = "release"
        mem["notes"] = notes + "|rotate done"
        return {"memory": mem, "act": {"tool": "act_release", "args": {"yaw_ref": mem["final_yaw"]}}}
    mem["stage"] = "hold"
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 10}}}


def act_wait(robot, n=1):
    """Hold the current pose (zero motion, keep gripper as is) for n steps."""
    f = robot.now()
    g = 1.0 if float(np.mean(f.fingers)) > 0.035 else -1.0
    for _ in range(int(n)):
        robot.step(0, 0, 0, 0, 0, 0, g)
    return {"steps": int(n), "tcp": robot.now().tcp}


def _ik_ok(robot, x, y, z, yaw):
    try:
        r = robot.dry_run(x, y, z, math.pi, 0.0, _wrap(yaw))
    except Exception as e:
        return None, str(e)[:80]
    m = r.get("joint_margin")
    mm = None
    if isinstance(m, (list, tuple)) and len(m):
        mm = float(min(m))
    ok = bool(r.get("reachable")) and (mm is None or mm >= 0.15)
    return ok, mm


def act_align_grasp(robot, cx, cy, grasp_yaws, theta, yaw_ref):
    """Pick the first grasp yaw (ordered by cost) whose grasp and final (yaw+theta)
    poses pass the kinematic dry run; move above the peg centre while turning to it,
    descend to GRASP_Z, close the gripper and wait until the fingers stop."""
    out = {"phases": [], "ik": []}
    grasp_yaw = float(grasp_yaws[0])
    for gy in grasp_yaws:
        ok1, m1 = _ik_ok(robot, cx, cy, GRASP_Z, gy)
        ok2, m2 = _ik_ok(robot, cx, cy, GRASP_Z, gy + theta)
        out["ik"].append([math.degrees(gy), ok1, m1, ok2, m2])
        if ok1 is None or (ok1 and ok2):
            grasp_yaw = float(gy)
            break
    out["grasp_yaw_deg"] = math.degrees(grasp_yaw)
    yaw_c = _unwrap_to(float(robot.now().tcp[5]), yaw_ref)
    n = 0
    # phase A: translate + rotate + descend to PRE_Z
    for i in range(60):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, [cx, cy, PRE_Z], grasp_yaw)
        exy = math.hypot(cx - float(f.tcp[0]), cy - float(f.tcp[1]))
        ez = abs(PRE_Z - float(f.tcp[2]))
        if abs(ye) < math.radians(6) and exy < 0.008 and ez < 0.02:
            break
        robot.step(*cmd, 1.0)
        n += 1
    out["phases"].append({"A_steps": n, "yaw_err_deg": math.degrees(ye), "exy": exy})
    # phase B: descend to grasp height while finishing alignment
    nb = 0
    prev = None
    for i in range(30):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, [cx, cy, GRASP_Z], grasp_yaw)
        exy = math.hypot(cx - float(f.tcp[0]), cy - float(f.tcp[1]))
        ez = abs(GRASP_Z - float(f.tcp[2]))
        q, jv = _jvel(robot, prev)
        prev = q
        if abs(ye) < math.radians(1.5) and exy < 0.004 and ez < 0.005:
            break
        robot.step(*cmd, 1.0)
        nb += 1
    out["phases"].append({"B_steps": nb, "yaw_err_deg": math.degrees(ye), "exy": exy,
                          "z": float(robot.now().tcp[2])})
    # phase C: close while holding pose
    nc = 0
    fprev = None
    stable = 0
    for i in range(15):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, [cx, cy, GRASP_Z], grasp_yaw)
        robot.step(*cmd, -1.0)
        nc += 1
        fw = float(np.sum(robot.now().fingers))
        if fprev is not None and abs(fw - fprev) < 0.0008 and fw < 0.07:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        fprev = fw
    f = robot.now()
    out["phases"].append({"C_steps": nc, "finger_sum": float(np.sum(f.fingers))})
    out["tcp"] = f.tcp
    out["step"] = int(f.step)
    return out


def act_rotate(robot, cx, cy, peg_yaw0, theta, yaw_ref, correct=False):
    """With the peg grasped, turn the gripper by theta relative to the peg's initial
    yaw branch nearest the measured grasp yaw, about the peg centre (TCP xy held at
    the centre, z at GRASP_Z); settle; measure peg yaw from the front camera and
    (if correct) apply up to two bounded corrections."""
    out = {}
    yaw_c = float(robot.now().tcp[5])
    if yaw_ref is not None and abs(yaw_c) > 2.9:
        yaw_c = _unwrap_to(yaw_c, float(yaw_ref))
    branch = yaw_c + _wrap_half(float(peg_yaw0) - yaw_c)
    target = branch + float(theta)
    peg_final_yaw = float(peg_yaw0) + float(theta)
    out["grasp_yaw_meas_deg"] = math.degrees(yaw_c)
    out["target_deg"] = math.degrees(target)
    n = 0
    prev = None
    settled = 0
    corrections = []
    for i in range(45):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, [cx, cy, GRASP_Z], target)
        q, jv = _jvel(robot, prev)
        prev = q
        if abs(ye) < math.radians(0.4) and jv < 0.002:
            settled += 1
        else:
            settled = 0
        if settled >= 2:
            # check peg yaw visually (exclude the gripper region)
            pose = _peg_pose(f, ("front",), exclude_xy=[float(f.tcp[0]), float(f.tcp[1])], exclude_r=0.05)
            if pose is not None and len(corrections) < 2:
                perr = _wrap_half(float(peg_final_yaw) - pose["yaw"])
                corrections.append([math.degrees(perr), pose["n"]])
                if correct and math.radians(1.0) < abs(perr) < math.radians(8.0) and pose["n"] > 80:
                    target += perr
                    settled = 0
                    continue
            break
        robot.step(*cmd, -1.0)
        n += 1
    f = robot.now()
    pose = _peg_pose(f, ("front",), exclude_xy=[float(f.tcp[0]), float(f.tcp[1])], exclude_r=0.05)
    out["steps"] = n
    out["yaw_err_deg"] = math.degrees(target - _unwrap_to(float(f.tcp[5]), yaw_c))
    out["corrections"] = corrections
    out["peg_vis"] = pose
    out["tcp"] = f.tcp
    out["step"] = int(f.step)
    return out


def act_release(robot, yaw_ref):
    """Hold still, open the gripper, then lift 6 cm and hold."""
    f0 = robot.now()
    hold = [float(f0.tcp[0]), float(f0.tcp[1]), float(f0.tcp[2])]
    yaw_c = float(f0.tcp[5])
    if yaw_ref is not None and abs(yaw_c) > 2.9:
        yaw_c = _unwrap_to(yaw_c, float(yaw_ref))
    ytarget = yaw_c
    for i in range(3):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, hold, ytarget)
        robot.step(*cmd, -1.0)
    for i in range(5):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, hold, ytarget)
        robot.step(*cmd, 1.0)
    up = [hold[0], hold[1], hold[2] + 0.06]
    for i in range(30):
        f = robot.now()
        yaw_c = _unwrap_to(float(f.tcp[5]), yaw_c)
        cmd, ye = _command(f.tcp, yaw_c, up, ytarget)
        robot.step(*cmd, 1.0)
    f = robot.now()
    pose = _peg_pose(f)
    return {"tcp": f.tcp, "peg": pose, "step": int(f.step)}
