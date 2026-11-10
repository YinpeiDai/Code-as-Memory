"""InterceptGrabSlow: track the rolling red ball with the fixed front camera (RGB-D),
hover over it with a vertical open gripper (fingers along world y), descend while
matching its velocity, close, then hold until the native success signal."""

BALL_R = 0.02          # ball radius (m), from finger gap at grasp (~0.038-0.04)
BASE_XY = (-0.615, 0.0)  # robot base position in the world frame (FK vs TCP offset)
MAX_REACH = 0.76       # horizontal TCP radius from base allowed for vertical grasp
HOVER_Z = 0.085        # TCP height while aligning (fingertips above the ball top)
GRASP_Z = 0.02         # TCP height at grasp (demo mean 0.022)
STAGE_Y = -0.45
STAGE_X = -0.2
STAGE_Z = 0.12
V_PER_CMD = 0.04       # steady-state TCP m/step for a unit translation command
CLOSE_GRIP = -1.0
OPEN_GRIP = 1.0


def _red_points(rgb, dep, K, E, zmax):
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    m = (r > 120) & (g < 60) & (b < 60) & ((r - g) > 90) & (dep > 0)
    vs, us = np.nonzero(m)
    if len(vs) == 0:
        return None
    d = dep[vs, us].astype(np.float64) / 1000.0
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    pw = (pc - t) @ R          # R^T (pc - t) for each row
    keep = (pw[:, 2] < zmax) & (pw[:, 2] > -0.02)
    if keep.sum() == 0:
        return None
    return pw[keep], us[keep], vs[keep], -R.T @ t


def _ball_in(frame):
    """Ball centre estimate from one frame: front camera first, wrist as fallback."""
    res = _red_points(frame.front_rgb, frame.front_depth, frame.front_K,
                      frame.front_extrinsic, 0.07)
    if res is not None:
        pts, us, vs, C = res
        border = (us.min() <= 1) or (us.max() >= 254) or (vs.max() >= 254)
        if len(pts) >= 4 and not border:
            p = np.median(pts, 0)
            ray = (p - C) / (np.linalg.norm(p - C) + 1e-9)
            c = p + ray * BALL_R * 0.67
            return [float(c[0]), float(c[1]), float(c[2]), int(len(pts)), "front"]
    res = _red_points(frame.wrist_rgb, frame.wrist_depth, frame.wrist_K,
                      frame.wrist_extrinsic, 0.07)
    if res is not None:
        pts, us, vs, C = res
        if len(pts) >= 60:
            p = np.median(pts, 0)
            ray = (p - C) / (np.linalg.norm(p - C) + 1e-9)
            c = p + ray * BALL_R * 0.67
            return [float(c[0]), float(c[1]), float(c[2]), int(len(pts)), "wrist"]
    return None


def _merge_track(track, step, det):
    track = [list(x) for x in (track or [])]
    if det is None:
        return track
    if track and int(track[-1][0]) >= step:
        return track
    track.append([int(step), round(det[0], 5), round(det[1], 5), round(det[2], 5)])
    return track[-14:]


def _estimate(track, step):
    """Constant-velocity least-squares fit over recent detections -> (pos at step, vel, n)."""
    if not track:
        return None, None, 0
    recent = [t for t in track if t[0] >= track[-1][0] - 10]
    T = np.array([t[0] for t in recent], dtype=np.float64)
    P = np.array([[t[1], t[2]] for t in recent], dtype=np.float64)
    if len(recent) >= 3 and T.max() - T.min() >= 2:
        A = np.stack([np.ones_like(T), T - T.max()], 1)
        coef = np.linalg.lstsq(A, P, rcond=None)[0]
        vel = coef[1]
        pos = coef[0] + vel * (step - T.max())
    else:
        vel = np.array([0.0, 0.013])
        pos = P[-1] + vel * (step - T[-1])
    sp = float(np.hypot(vel[0], vel[1]))
    if sp > 0.03:
        vel = vel * 0.03 / sp
    return [float(pos[0]), float(pos[1])], [float(vel[0]), float(vel[1])], len(recent)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-3:
        return np.zeros(3)
    return v / (2 * s) * ang


def _rot_cmd(tcp, gain=0.5):
    """Normalized rotation command that turns the measured orientation toward vertical-down, yaw 0."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_to_R(math.pi, 0.0, 0.0)
    w = _rotvec(Rd @ Rc.T)
    r = -w * gain / 0.1
    n = np.linalg.norm(r)
    if n > 1:
        r = r / n
    return r, float(np.linalg.norm(w))


def _clamp_reach(x, y):
    dx, dy = x - BASE_XY[0], y - BASE_XY[1]
    r = math.hypot(dx, dy)
    return r


def _aim(pos, vel):
    """Point on the ball's path to aim at: the ball itself if inside reach, else its entry point."""
    for tau in range(0, 45):
        x = pos[0] + vel[0] * tau
        y = pos[1] + vel[1] * tau
        if _clamp_reach(x, y) <= MAX_REACH:
            return [x, y], tau
    # never enters: closest admissible point toward base
    x, y = pos[0], pos[1]
    r = _clamp_reach(x, y)
    s = MAX_REACH / max(r, 1e-6)
    return [BASE_XY[0] + (x - BASE_XY[0]) * s, BASE_XY[1] + (y - BASE_XY[1]) * s], 99


def _step_checked(robot, cmd, grip):
    """One native step; preview the joint target and shrink the command if it leaves joint limits."""
    info = {"margin": None, "scaled": False}
    try:
        chk = robot.dry_run(*[float(c) for c in cmd], mode="delta")
        info["margin"] = chk.get("min_joint_margin_rad")
        if not chk.get("within_joint_limits", True):
            cmd = [c * 0.3 for c in cmd]
            info["scaled"] = True
    except Exception as e:  # preview unavailable; step anyway
        info["err"] = str(e)[:60]
    robot.step(*[float(c) for c in cmd], grip)
    return info


def perceive_decide(obs, goal, memory, last):
    """Update the ball track from every new frame, pick the stage and run the intercept controller."""
    mem = dict(memory or {})
    track = mem.get("ball_track") or []
    for fr in obs.frames:
        track = _merge_track(track, fr.step, _ball_in(fr))
    stage = mem.get("stage") or "search"
    attempts = int(mem.get("attempts") or 0)
    if last and isinstance(last, str):
        for st in ("search", "track", "close", "hold"):
            if ('"stage": "%s"' % st) in last:
                stage = st
        k = last.find('"attempts": ')
        if k >= 0:
            digits = ""
            for ch in last[k + 12:k + 16]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                attempts = int(digits)
    cur = obs.current
    pos, vel, n = _estimate(track, cur.step)
    if stage == "search" and n > 0:
        stage = "track"
    hist = list(mem.get("history") or [])
    hist.append("%d:%s" % (cur.step, stage))
    hist = hist[-40:]
    new_mem = {
        "stage": stage,
        "ball_track": track,
        "ball_estimate": [] if pos is None else [round(pos[0], 4), round(pos[1], 4)],
        "ball_velocity": [] if vel is None else [round(vel[0], 5), round(vel[1], 5)],
        "attempts": attempts,
        "history": hist,
        "goal_ok": ("ball" in str(goal).lower()),
    }
    return {"memory": new_mem,
            "act": {"tool": "act_intercept",
                    "args": {"stage": stage, "track": track, "attempts": attempts,
                             "max_steps": 12}}}


def act_intercept(robot, stage="search", track=None, attempts=0, max_steps=12):
    """Closed-loop ball intercept for up to max_steps native steps (returns early on stage change).
    stage: search (no ball yet) -> track (hover/descend over the moving ball) -> close -> hold.
    Returns the final stage, attempts and per-step diagnostics."""
    track = [list(t) for t in (track or [])]
    log = []
    close_k = 0
    hold_k = 0
    start_stage = stage
    margins = []
    for i in range(int(max_steps)):
        fr = robot.now()
        track = _merge_track(track, fr.step, _ball_in(fr))
        tcp = np.asarray(fr.tcp, dtype=np.float64)
        fing = float(np.mean(fr.fingers))
        pos, vel, n = _estimate(track, fr.step)
        rot, rot_err = _rot_cmd(tcp)
        grip = OPEN_GRIP
        if stage == "search":
            if n > 0:
                stage = "track"
            else:
                tgt = np.array([STAGE_X, STAGE_Y, STAGE_Z])
                u = np.clip((tgt - tcp[:3]) / 0.1, -1, 1)
                info = _step_checked(robot, list(u) + list(rot), grip)
                margins.append(info["margin"])
                log.append([fr.step, "S", round(tcp[0], 3), round(tcp[1], 3), round(tcp[2], 3)])
                continue
        if pos is None:
            pos, vel = [STAGE_X, STAGE_Y], [0.0, 0.0]
        aim, tau = _aim(pos, vel)
        ex = aim[0] - tcp[0]
        ey = aim[1] - tcp[1]
        if stage == "track":
            aligned = tau == 0 and abs(ex) < 0.012 and abs(ey) < 0.012
            if aligned:
                zt = GRASP_Z
            else:
                zt = HOVER_Z
            # stay high when misaligned in y near the ball (fingers straddle along y)
            if tcp[2] < 0.05 and (abs(ey) > 0.018 or abs(ex) > 0.02):
                zt = HOVER_Z
            ff = [vel[0] / V_PER_CMD, vel[1] / V_PER_CMD] if tau == 0 else [0.0, 0.0]
            u = [ex / 0.1 + ff[0], ey / 0.1 + ff[1], (zt - tcp[2]) / 0.1]
            u = list(np.clip(u, -1, 1))
            if tau == 0 and tcp[2] < 0.03 and abs(ex) < 0.012 and abs(ey) < 0.012:
                stage = "close"
                close_k = 0
            info = _step_checked(robot, u + list(rot), grip)
            margins.append(info["margin"])
            log.append([fr.step, "T", round(tcp[0], 3), round(tcp[1], 3), round(tcp[2], 3),
                        round(ex, 3), round(ey, 3), tau])
            if stage != start_stage:
                break
            continue
        if stage == "close":
            grip = CLOSE_GRIP
            close_k += 1
            ffs = 1.0 if close_k <= 2 else 0.0
            u = [ex / 0.1 * ffs + ffs * vel[0] / V_PER_CMD,
                 ey / 0.1 * ffs + ffs * vel[1] / V_PER_CMD,
                 (GRASP_Z - tcp[2]) / 0.1 * ffs]
            u = list(np.clip(u, -1, 1))
            info = _step_checked(robot, u + list(rot), grip)
            margins.append(info["margin"])
            f2 = float(np.mean(robot.now().fingers))
            log.append([fr.step, "C", round(tcp[0], 3), round(tcp[1], 3), round(tcp[2], 3),
                        round(ex, 3), round(ey, 3), round(f2, 4)])
            if close_k >= 4:
                if f2 < 0.009:
                    stage = "track"
                    attempts += 1
                    # reopen while lifting
                    for _ in range(3):
                        robot.step(0.0, 0.0, 0.6, 0.0, 0.0, 0.0, OPEN_GRIP)
                    break
                if f2 < 0.035:
                    stage = "hold"
                    break
            continue
        if stage == "hold":
            grip = CLOSE_GRIP
            hold_k += 1
            dz = 0.0 if hold_k <= 4 else 0.15
            if tcp[2] > 0.07:
                dz = 0.0
            info = _step_checked(robot, [0.0, 0.0, dz] + list(rot * 0.3), grip)
            f2 = float(np.mean(robot.now().fingers))
            log.append([fr.step, "H", round(tcp[0], 3), round(tcp[1], 3), round(tcp[2], 3), round(f2, 4)])
            if f2 < 0.009:
                stage = "track"
                attempts += 1
                for _ in range(3):
                    robot.step(0.0, 0.0, 0.6, 0.0, 0.0, 0.0, OPEN_GRIP)
                break
            continue
    now = robot.now()
    pos, vel, n = _estimate(track, now.step)
    ms = [m for m in margins if m is not None]
    return {"stage": stage, "attempts": attempts,
            "step": int(now.step), "tcp": [round(float(v), 3) for v in now.tcp[:3]],
            "fingers": round(float(np.mean(now.fingers)), 4),
            "ball": None if pos is None else [round(pos[0], 3), round(pos[1], 3)],
            "vel": None if vel is None else [round(vel[0], 4), round(vel[1], 4)],
            "min_joint_margin": None if not ms else round(min(ms), 3),
            "log": log[-12:]}
