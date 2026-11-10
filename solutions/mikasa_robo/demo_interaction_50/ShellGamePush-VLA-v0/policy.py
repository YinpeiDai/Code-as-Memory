"""ShellGamePush policy: remember which cup hides the ball, then press on that cup's top and push it forward (+x).

Perception uses the fixed front camera (RGB-D + calibration). Red pixels are deprojected to world points.
A small red blob (< RED_CUPS_MIN pixels) is the ball cue; three large clusters with tops near z=0.108 m are the
inverted mugs. Control is closed-loop on the measured TCP with a PD law on pd_ee_delta_pose commands.
"""

RED_CUPS_MIN = 300          # red pixel count above which the cups are visible (demos: ball ~40-70 px, cups 670-1250 px)
# cup tops: demos 0.108 m, rollout R001 0.093 m -> top height measured per cup (99th percentile of red z)
PRE_DX = -0.02              # press point relative to cup-top centroid (demo descent x offset -0.017 +- 0.006)
PRE_DY = 0.01               # demo final y offset relative to cup-top centroid: +0.02 +- 0.016
SAFE_Z = 0.165              # transit height above cup tops (0.108)
PRESS_Z = 0.07              # z target while pressing (below cup top so the fingers rest on it, as the demos do)
PUSH_DIST = 0.13            # planned TCP travel in +x from the press point (demo successes at 0.05-0.11 travel)
X_MAX = 0.21                # cap on push end x (arm near full extension beyond; demo max TCP x 0.184)
KP = 1.5                    # R002: real first-step response ~0.3-0.5 of command (faster than demo fit) -> lower gains
KD = 0.3
GRIP = 0.0                  # partially closed fingers (~0.02 m each, like the demos)


def _redmask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r > 60) & (r > 2 * g) & (r > 2 * b)


def _cloud(frame, mask):
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    dep = np.asarray(frame.front_depth)
    vs, us = np.nonzero(mask)
    z = dep[vs, us].astype(np.float64) / 1000.0
    ok = z > 0
    vs, us, z = vs[ok], us[ok], z[ok]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _scene(frame):
    """Return dict with red pixel count, ball xyz (if only a small blob) and cup-top clusters (if cups visible)."""
    m = _redmask(np.asarray(frame.front_rgb))
    n = int(m.sum())
    out = {"red": n, "ball": None, "cups": []}
    if n < 15:
        return out
    P = _cloud(frame, m)
    P = P[P[:, 2] > 0.003]
    if len(P) < 10:
        return out
    if n < RED_CUPS_MIN:
        spread = P.max(0) - P.min(0)
        if spread[0] < 0.08 and spread[1] < 0.08:
            out["ball"] = [float(v) for v in P.mean(0)]
        return out
    body = P[(P[:, 2] > 0.02) & (P[:, 2] < 0.2)]
    if len(body) < 30:
        return out
    body = body[np.argsort(body[:, 1])]
    gaps = np.diff(body[:, 1])
    cuts = [i + 1 for i in range(len(gaps)) if gaps[i] > 0.03]
    cups = []
    for g in np.split(body, cuts):
        if len(g) < 40:
            continue
        zt = float(np.percentile(g[:, 2], 99))
        top = g[g[:, 2] > zt - 0.012]
        if len(top) < 15:
            continue
        c = top.mean(0)
        cups.append([float(c[0]), float(c[1]), zt, int(len(top))])
    out["cups"] = cups
    return out


def _rot(rpy):
    r, p, y = rpy
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
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:
        return np.zeros(3)
    return v * (ang / (2 * s))


def _rot_cmd(tcp):
    """Rotation command steering the gripper to point straight down (roll=pi, pitch=0, yaw=0).

    Native CPU IK: target = Rx(-0.1 r0) Ry(-0.1 r1) Rz(-0.1 r2) * R_meas, so r = -k * rotvec(R_des R_meas^T) / 0.1."""
    Rd = np.diag([1.0, -1.0, -1.0])
    Rm = _rot([float(tcp[3]), float(tcp[4]), float(tcp[5])])
    w = _rotvec(Rd @ Rm.T)
    r = -5.0 * w
    return [float(min(1.0, max(-1.0, v))) for v in r]


def _pd(err, vel, kp=KP, kd=KD, lim=1.0):
    a = kp * err / 0.1 - kd * vel / 0.1
    return float(min(lim, max(-lim, a)))


def _tcp(robot):
    return np.asarray(robot.now().tcp, dtype=np.float64)


def _vel(obs):
    fr = obs.frames
    if len(fr) >= 2:
        a = np.asarray(fr[-1].tcp, dtype=np.float64)
        b = np.asarray(fr[-2].tcp, dtype=np.float64)
        return [float(v) for v in (a[:3] - b[:3])]
    return [0.0, 0.0, 0.0]


def _track(scene, est):
    """Update the target-cup estimate [x, y, ztop] from a scene: the cup cluster nearest est (within 0.05 m) whose
    pixel support is not occluded (>= 75% of the best other cluster). Returns (new_est, reliable)."""
    cups = scene["cups"]
    if est is None or len(cups) == 0:
        return est, False
    d = [math.hypot(c[0] - est[0], c[1] - est[1]) for c in cups]
    k = int(np.argmin(d))
    if d[k] > 0.05:
        return est, False
    others = [c[3] for i, c in enumerate(cups) if i != k]
    if len(others) > 0 and cups[k][3] < 0.75 * max(others):
        return est, False
    c = cups[k]
    return [c[0], c[1], c[2]], True


def _parse_list(text, key):
    """Read a numeric list field (e.g. "est") from the previous act outcome text; None if absent."""
    if not isinstance(text, str) or ('"%s": [' % key) not in text:
        return None
    body = text.split('"%s": [' % key, 1)[1].split("]", 1)[0]
    vals = []
    for part in body.split(","):
        try:
            vals.append(float(part))
        except ValueError:
            return None
    return vals


def _same(a, b):
    return a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) < 0.004 and abs(a[2] - b[2]) < 0.004


def perceive_decide(obs, goal, memory, last):
    """Track the ball cue, choose the cup hiding it, and sequence
    wait_cue -> settle (cups may still slide/settle after appearing) -> approach(+press) -> push -> hold.
    Stage transitions are based on measured TCP and perceived cups, not on commands issued."""
    mem = dict(memory)
    for k in ("ball", "ball_step", "cups", "target", "target_index", "stage", "push_x0", "notes", "cups_step",
              "extra_pushes"):
        mem.setdefault(k, None)
    if mem["stage"] is None:
        mem["stage"] = "wait_cue"
        mem["notes"] = ""
    # 1. record the ball cue from any frame showing only the ball
    for fr in obs.frames:
        if mem["ball"] is None:
            s = _scene(fr)
            if s["ball"] is not None:
                mem["ball"] = s["ball"]
                mem["ball_step"] = int(fr.step)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=np.float64)
    v0 = _vel(obs)
    s = _scene(cur)
    # 2. first frame with cups: choose the cup nearest the remembered ball
    if mem["target"] is None and len(s["cups"]) >= 1:
        cups = s["cups"]
        mem["cups"] = [[c[0], c[1], c[2]] for c in cups]
        mem["cups_step"] = int(cur.step)
        if mem["ball"] is not None:
            b = mem["ball"]
            d = [math.hypot(c[0] - b[0], c[1] - b[1]) for c in cups]
            k = int(np.argmin(d))
            conf = "ball_dist=%.3f" % d[k]
        else:
            k = len(cups) // 2
            conf = "no_ball_seen_fallback_middle"
        mem["target"] = [cups[k][0], cups[k][1], cups[k][2]]
        mem["target_index"] = k
        mem["notes"] += "cups%d@%d tgt%d %s ztop=%.3f; " % (len(cups), cur.step, k, conf, cups[k][2])
        mem["stage"] = "settle"
    elif mem["target"] is None and s["red"] >= RED_CUPS_MIN and mem["ball"] is not None:
        # cups visible but not segmented: fall back to the remembered ball position
        b = mem["ball"]
        mem["target"] = [b[0] - 0.006, b[1] - 0.007, 0.108]
        mem["target_index"] = -1
        mem["notes"] += "cups_unsegmented@%d fallback_ball; " % cur.step
        mem["stage"] = "settle"
    stage = mem["stage"]
    if stage == "settle" and isinstance(last, str) and last.startswith("act_settle"):
        est = _parse_list(last, "est")
        if est is not None and len(est) == 3:
            mem["target"] = est
    # 3. keep the target estimate current while the cup is visible and unoccluded
    if mem["target"] is not None and stage in ("settle", "approach"):
        est, ok = _track(s, mem["target"])
        if ok:
            if stage == "settle" and len(obs.frames) >= 2:
                prev_est, ok2 = _track(_scene(obs.frames[-2]), est)
                if ok2 and _same(prev_est, est):
                    stage = "approach"
                    mem["notes"] += "settled@%d (%.3f,%.3f,%.3f); " % (cur.step, est[0], est[1], est[2])
            mem["target"] = est
    if stage == "settle" and isinstance(last, str) and last.startswith("act_settle") and '"settled": true' in last:
        stage = "approach"
        mem["notes"] += "settled(act)@%d (%.3f,%.3f,%.3f); " % (cur.step, mem["target"][0], mem["target"][1],
                                                               mem["target"][2])
    if stage == "settle" and isinstance(last, str) and last.startswith("act_settle") and '"settled": false' in last:
        stage = "approach"
        mem["notes"] += "settle_timeout@%d; " % cur.step
    t = mem["target"]
    if t is not None:
        px, py = t[0] + PRE_DX, t[1] + PRE_DY
        exy = math.hypot(px - tcp[0], py - tcp[1])
    stalled = isinstance(last, str) and last.startswith("act_approach") and '"contact": true' in last
    if stage == "approach" and exy < 0.04 and (tcp[2] < t[2] + 0.025 or stalled):
        stage = "push"
        mem["push_x0"] = float(tcp[0])
        mem["notes"] += "pressed z=%.3f@%d; " % (tcp[2], cur.step)
    slipped = isinstance(last, str) and last.startswith("act_push") and '"slipped": true' in last
    if stage in ("push", "hold", "extra") and slipped:
        stage = "approach"
        mem["notes"] += "slipped z=%.3f@%d; " % (tcp[2], cur.step)
    if stage == "push" and isinstance(last, str) and last.startswith("act_push") and '"steps": 0' in last:
        stage = "hold"
        mem["notes"] += "push_blocked@%d; " % cur.step
    if stage == "push" and tcp[0] > min(X_MAX, px + PUSH_DIST) - 0.01:
        stage = "hold"
        mem["notes"] += "pushed x=%.3f@%d; " % (tcp[0], cur.step)
    mem["stage"] = stage
    if stage == "wait_cue":
        act = {"tool": "act_wait_cue", "args": {"max_steps": 8}}
    elif stage == "settle":
        act = {"tool": "act_settle", "args": {"x": t[0], "y": t[1], "z_top": t[2], "max_steps": 8}}
    elif stage == "approach":
        act = {"tool": "act_approach", "args": {"x": px, "y": py, "z": SAFE_Z, "z_top": t[2], "v0": v0,
                                                "max_steps": 16}}
    elif stage == "push":
        act = {"tool": "act_push", "args": {"x_end": min(X_MAX, px + PUSH_DIST), "y": py, "z_top": t[2], "v0": v0,
                                            "max_steps": 9}}
    elif stage == "hold":
        mem["stage"] = "extra"
        act = {"tool": "act_hold", "args": {"steps": 2}}
    else:
        # still no native success after the push + hold: push a little further (bounded), then hold again
        n = int(mem.get("extra_pushes") or 0)
        if n < 3 and tcp[0] < X_MAX + 0.01:
            mem["extra_pushes"] = n + 1
            mem["stage"] = "hold"
            mem["notes"] += "extra_push%d x=%.3f@%d; " % (n + 1, tcp[0], cur.step)
            act = {"tool": "act_push", "args": {"x_end": min(X_MAX + 0.02, float(tcp[0]) + 0.04), "y": py,
                                                "z_top": t[2], "v0": v0, "max_steps": 3}}
        else:
            act = {"tool": "act_hold", "args": {"steps": 2}}
    return {"memory": mem, "act": act}


def act_wait_cue(robot, max_steps=8):
    """Send zero arm commands (the task suppresses them while only the ball is shown) until the cups become visible
    in the front camera or max_steps pass. Returns steps used and red pixel counts."""
    counts = []
    for i in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP)
        n = _scene(robot.now())["red"]
        counts.append(n)
        if n >= RED_CUPS_MIN:
            return {"steps": i + 1, "red": counts, "cups_visible": True, "step": int(robot.now().step)}
    return {"steps": int(max_steps), "red": counts, "cups_visible": False}


def act_settle(robot, x, y, z_top, max_steps=8):
    """Cups can still move for a few steps after appearing (R003-R007: tops rising 0.054 -> 0.108 m; R007: all cups
    slid -0.065 m in x over 4 steps). Track the target cup (nearest cluster, unoccluded) each step while moving the
    TCP to a staging pose behind it (x-0.10, y+PRE_DY, SAFE_Z) where the hand cannot occlude it from the front camera.
    Ends when two consecutive estimates agree (<4 mm). Returns the estimate track and settled flag."""
    est = [x, y, z_top]
    prev = _tcp(robot)
    track = []
    for i in range(int(max_steps)):
        p = _tcp(robot)
        v = p[:3] - prev[:3]
        r = _rot_cmd(p)
        gx, gy = est[0] - 0.10, est[1] + PRE_DY
        prev = p
        robot.step(_pd(gx - p[0], v[0]), _pd(gy - p[1], v[1]), _pd(SAFE_Z - p[2], v[2]), r[0], r[1], r[2], GRIP)
        new, ok = _track(_scene(robot.now()), est)
        track.append([round(new[0], 3), round(new[1], 3), round(new[2], 3), ok])
        if ok and _same(new, est):
            return {"est": [float(v) for v in new], "steps": i + 1, "settled": True, "track": track}
        if ok:
            est = new
    return {"est": [float(v) for v in est], "steps": int(max_steps), "settled": False, "track": track}


def act_approach(robot, x, y, z, z_top, v0=None, max_steps=16, near=0.04):
    """PD move of the TCP toward the press point (x, y): first lift to transit height z if below it and far from
    the press point, transit at z while the xy error > near, then descend toward PRESS_Z (below the cup top) so the
    fingers rest on the cup top. Ends when resting (pressing below, z no longer falling, xy error < 0.035).
    (R003: requiring xy error < 0.015 while the fingers pressed on the cup top stalled the arm for 20 steps.)"""
    prev = _tcp(robot)
    if v0 is not None:
        prev = prev.copy()
        prev[:3] = prev[:3] - np.asarray(v0, dtype=np.float64)
    log = []
    used = 0
    contact = False
    zt_prev = z
    for i in range(int(max_steps)):
        p = _tcp(robot)
        v = p[:3] - prev[:3]
        exy = math.hypot(x - p[0], y - p[1])
        # stall while commanding a press target below the TCP = resting on the cup top (independent of z_top)
        if p[2] < z - 0.02 and v[2] > -0.004 and exy < 0.035 and used > 0 and zt_prev < p[2] - 0.01:
            contact = True
            break
        zt = z if exy > near else PRESS_Z
        zt_prev = zt
        r = _rot_cmd(p)
        if exy > near and p[2] < z_top + 0.03:
            # low and away from the press point (e.g. after slipping off): lift before moving sideways
            a = [0.0, 0.0, _pd(z - p[2], v[2])]
        else:
            a = [_pd(x - p[0], v[0]), _pd(y - p[1], v[1]), _pd(zt - p[2], v[2])]
        prev = p
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], GRIP)
        used += 1
        q = _tcp(robot)
        log.append([round(float(q[0]), 3), round(float(q[1]), 3), round(float(q[2]), 3)])
    p = _tcp(robot)
    return {"steps": used, "contact": contact, "final_err": [float(x - p[0]), float(y - p[1])], "z": float(p[2]),
            "path": log}


def act_push(robot, x_end, y, z_top, v0=None, max_steps=9, press=-0.8):
    """Push forward (+x) toward x_end while pressing down (dz=press) on the cup top, holding y and orientation.
    Each command is previewed with robot.dry_run(mode='delta'); if the joint target margin is small the downward
    press is reduced. Stops with slipped=true if the TCP drops 3 cm below the cup top (fingers slid off).
    Returns the measured TCP path and joint margins."""
    prev = _tcp(robot)
    if v0 is not None:
        prev = prev.copy()
        prev[:3] = prev[:3] - np.asarray(v0, dtype=np.float64)
    path = []
    margins = []
    used = 0
    slipped = False
    for i in range(int(max_steps)):
        p = _tcp(robot)
        if p[2] < z_top - 0.03:
            slipped = True
            break
        v = p[:3] - prev[:3]
        ex = x_end - p[0]
        if ex < 0.005:
            break
        r = _rot_cmd(p)
        dx = _pd(ex, v[0], lim=0.8)
        dy = _pd(y - p[1], v[1])
        cmd = [dx, dy, press, r[0], r[1], r[2]]
        chk = robot.dry_run(*cmd, mode="delta")
        m = chk.get("min_joint_margin_rad")
        if (not chk.get("within_joint_limits", True)) or (m is not None and m < 0.08):
            cmd = [dx, dy, 0.3 * press, r[0], r[1], r[2]]
            chk = robot.dry_run(*cmd, mode="delta")
            m = chk.get("min_joint_margin_rad")
            if (not chk.get("within_joint_limits", True)) or (m is not None and m < 0.03):
                margins.append(m)
                break
        margins.append(None if m is None else round(float(m), 3))
        prev = p
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], GRIP)
        used += 1
        q = _tcp(robot)
        path.append([round(float(q[0]), 3), round(float(q[1]), 3), round(float(q[2]), 3)])
    return {"steps": used, "slipped": slipped, "path": path, "margins": margins}


def act_hold(robot, steps=2):
    """Hold position (zero arm motion) for a few steps; returns the TCP."""
    for i in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP)
    p = _tcp(robot)
    return {"tcp": [float(v) for v in p[:3]]}
