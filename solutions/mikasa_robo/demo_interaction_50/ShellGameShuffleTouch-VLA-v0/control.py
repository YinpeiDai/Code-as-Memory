"""Motion acts for ShellGameShuffleTouch (pd_ee_delta_pose, 20 Hz, 60-step episode)."""

POS_K = 1.5      # proportional gain on position error (fraction of error per command)
POS_D = 1.0      # damping on measured per-step displacement
ROT_K = 0.4      # fraction of orientation error corrected per command


def _rpy_to_R(r, p, y):
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
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return w * ang / (2.0 * s)


def _rot_cmd(tcp, yaw):
    """Rotation command steering the tool toward pointing straight down with the given yaw."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_to_R(math.pi, 0.0, yaw)
    w = _rotvec(Rd @ Rc.T)
    # native: target rotation = exp(-0.1 r) * R_current  ->  r = -w / 0.1 (scaled)
    r = -ROT_K * w / 0.1
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _pos_cmd(err, vel):
    a = (POS_K * err - POS_D * vel) / 0.1
    return np.clip(a, -1.0, 1.0)


def act_wait(robot, max_steps=8, probe_dz=0.0):
    """Step up to max_steps with a small downward probe command (ignored while the cue phase
    suppresses commands). Stops early when the TCP starts moving (control active) or the
    mug mask has been unchanged for 2 frames after changing. Returns measured TCP motion."""
    f0 = robot.now()
    p0 = np.asarray(f0.tcp[:3], float)
    prev = _mug_mask(f0.front_rgb)
    changed = 0
    stable = 0
    n = 0
    moved = 0.0
    for i in range(int(max_steps)):
        fr = robot.step(0.0, 0.0, float(probe_dz), 0.0, 0.0, 0.0, 1.0)
        n += 1
        m = _mug_mask(fr.front_rgb)
        diff = int((m != prev).sum())
        prev = m
        if diff > 25:
            changed += 1
            stable = 0
        else:
            stable += 1
        moved = float(np.linalg.norm(np.asarray(fr.tcp[:3], float) - p0))
        if moved > 0.002:
            break
        if changed >= 6 and stable >= 2:
            break
    return {"steps": n, "tcp_moved": moved, "mask_changes": changed, "stable": stable,
            "step": int(robot.now().step)}


def act_touch(robot, cx, cy, dx=0.04, dy=0.0, z_pre=0.135, z_low=0.065, close=True, yaw=0.0,
              label="", max_steps=30, z_close=0.085, hold_steps=14, press=0.02):
    """Move the TCP to (cx+dx, cy+dy) above the mug at z_pre, descend to z_low with the gripper
    open; once the TCP is below z_close start closing the fingers on the mug wall (latched). If the
    fingers close completely without the episode ending (missed the wall), reopen and close again
    (regrasp) for up to hold_steps steps. Without close the fingers stay closed (probe). Returns
    measured path, finger log and the gripper cycle count."""
    grip = 1.0 if close else -1.0
    goal = np.array([cx + dx, cy + dy, z_low], float)
    f = robot.now()
    chk = robot.dry_run(float(goal[0]), float(goal[1]), float(z_pre), math.pi, 0.0, float(yaw))
    reach = {"reachable": chk.get("reachable"), "path_clear": chk.get("path_clear"),
             "margin": chk.get("joint_margin")}
    if chk.get("reachable") is False:
        return {"label": label, "skipped": True, "reach": reach, "steps": 0, "step": int(f.step)}
    log = []
    flog = []
    vel = np.zeros(3)
    phase = "approach"
    gstate = "open"
    regrasps = 0
    shut = 0
    n = 0
    settle = 0
    closed_at = -1
    while n < int(max_steps):
        tcp = np.asarray(robot.now().tcp, float)
        fing = float(np.mean(np.asarray(robot.now().fingers, float)))
        p = tcp[:3]
        herr = float(np.linalg.norm(goal[:2] - p[:2]))
        if phase == "approach":
            slow = float(np.linalg.norm(vel[:2])) < 0.004
            ready = herr <= 0.02 or (herr <= 0.03 and slow and n > 2)
            zt = z_low if ready else z_pre
            if ready:
                phase = "descend"
        else:
            zt = z_low
        if gstate != "open":
            zt = z_low - press   # palm presses on the mug top while squeezing, resisting the pop-up
        tgt = np.array([goal[0], goal[1], zt])
        a = _pos_cmd(tgt - p, vel)
        if phase == "descend" and p[2] > z_low + 0.03:
            a[2] = -1.0   # full-speed descent until 3 cm above the hold height, then PD (no hard press)
        r, rerr = _rot_cmd(tcp, yaw)
        if close:
            if gstate == "open" and phase == "descend" and p[2] < z_close:
                gstate = "closing"
                closed_at = int(robot.now().step)
            elif gstate == "closing" and fing < 0.003:
                shut += 1
                if shut >= 2:
                    gstate = "reopen"      # fingers met (2 frames) without success: regrasp
                    regrasps += 1
                    shut = 0
            elif gstate == "reopen" and fing > 0.03:
                gstate = "closing"
        g = -1.0 if gstate == "closing" else grip
        fr = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
        n += 1
        q = np.asarray(fr.tcp[:3], float)
        vel = q - p
        log.append([int(fr.step), round(float(q[0]), 3), round(float(q[1]), 3), round(float(q[2]), 3), round(rerr, 3)])
        flog.append([int(fr.step), gstate, round(float(np.mean(np.asarray(fr.fingers, float))), 4)])
        if closed_at >= 0:
            if int(fr.step) - closed_at >= int(hold_steps) or regrasps > 3:
                break
            continue
        if phase == "descend":
            if float(np.linalg.norm(tgt - q)) < 0.008 and float(np.linalg.norm(vel)) < 0.004 and not close:
                break
            if float(np.linalg.norm(vel)) < 0.002 and q[2] < z_low + 0.04:
                settle += 1
                if settle >= 3 and not close:
                    break
    blocked_z = float(robot.now().tcp[2])
    # lift back to the pre-height before any next attempt
    lift = 0
    for k in range(6):
        tcp = np.asarray(robot.now().tcp, float)
        if tcp[2] > z_pre - 0.015:
            break
        tgt = np.array([tcp[0], tcp[1], z_pre])
        a = _pos_cmd(tgt - tcp[:3], np.zeros(3))
        r, rerr = _rot_cmd(tcp, yaw)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), 1.0)
        lift += 1
        n += 1
    return {"label": label, "steps": n, "reach": reach, "goal": [round(float(v), 3) for v in goal],
            "final_z": round(blocked_z, 3), "closed_at": closed_at, "regrasps": regrasps,
            "path": log[-10:], "fingers": flog[-10:], "lift_steps": lift, "step": int(robot.now().step)}
