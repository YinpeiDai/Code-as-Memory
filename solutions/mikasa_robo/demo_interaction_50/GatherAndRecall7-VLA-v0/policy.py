"""GatherAndRecall7: move 7 cubes onto the disc, watch the lamp flash, press the matching button.

Perception uses the fixed front RGB-D camera (deprojected to world with K and the
world-to-camera extrinsic). Control is closed-loop on the measured TCP with a
proportional pd_ee_delta_pose command. See policy.md and task_analysis.md.
"""

COLORS = ("red", "green", "blue")
Z_TRAVEL = 0.10      # TCP height for transfers (cube bottom ~0.08 > placed cube tops 0.046)
Z_GRASP = 0.022      # TCP height when closing on a 4 cm cube resting on the table
Z_PLACE = 0.036      # TCP height when releasing over the 6 mm disc
Z_BTN_HOVER = 0.12   # hover height above a button (cap tops at ~0.058)
Z_BTN_PRESS = 0.035  # pressing target (descent stops when the episode ends)
RING_R = 0.07        # radius of the six-slot ring on the disc
KP = 1.0             # translation gain: command = KP * error / 0.1 m
KR = 0.5             # rotation gain
RELEASE_GRIP = 0.6   # partial opening at release (avoid pushing neighbours)


# ----------------------------------------------------------------- geometry
def _deproject(depth, K, E):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    pw = (pc - t) @ R
    pw[depth <= 0] = np.nan
    return pw


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _rotvec(Rm):
    c = (np.trace(Rm) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:  # near pi
        d = np.sqrt(np.maximum((np.diag(Rm) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(d))
        axis = d.copy()
        for j in range(3):
            if j != k:
                axis[j] = (Rm[k, j] + Rm[j, k]) / (4.0 * d[k])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * ang
    return v / (2.0 * s) * ang


def _rot_cmd(tcp, yaw_t):
    """Normalized rotation command steering the measured orientation to Rz(yaw_t) Rx(pi)."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(math.pi, 0.0, yaw_t)
    e = _rotvec(Rt @ Rc.T)          # base-frame rotation still needed
    r = -KR * e / 0.1                # step applies -0.1*r
    n = np.linalg.norm(r)
    if n > 1.0:
        r = r / n
    return r


# --------------------------------------------------------------- perception
def _masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    red = (r > 90) & (r * 10 > 25 * g) & (r * 10 > 25 * b)
    green = (g > 90) & (g * 10 > 16 * r) & (g * 10 > 16 * b)
    blue = (b > 90) & (b * 10 > 16 * r) & (b * 10 > 16 * g)
    return {"red": red, "green": green, "blue": blue}


def _cluster(pts, radius=0.03, min_count=8):
    """Greedy clustering of 2-D points; returns [(centre, count)] sorted by count."""
    pts = np.asarray(pts, dtype=np.float64)
    out = []
    left = pts
    while len(left) >= min_count:
        seed = left[0]
        c = seed
        for _ in range(4):
            d = np.linalg.norm(left - c, axis=1)
            sel = d < radius
            c = left[sel].mean(0)
        d = np.linalg.norm(left - c, axis=1)
        sel = d < radius
        if sel.sum() >= min_count:
            out.append((c, int(sel.sum())))
        else:
            sel = np.linalg.norm(left - seed, axis=1) < 1e-9
            sel[0] = True
        left = left[~sel]
    return out


def _scene(frame, known_buttons=None):
    """Detect buttons, disc, lamp and cubes in the front camera."""
    P = _deproject(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgb = frame.front_rgb
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    ok = np.isfinite(Z)
    Zs = np.where(ok, Z, -1.0)
    m = _masks(rgb)
    buttons = {}
    cubes = []
    for col in COLORS:
        cap = m[col] & (Zs > 0.052) & (Zs < 0.07)
        cl = _cluster(np.stack([X[cap], Y[cap]], 1), 0.03, 8) if cap.sum() >= 8 else []
        if cl:
            c = cl[0][0]
            buttons[col] = [float(c[0]), float(c[1]), 0.058]
    for col in COLORS:
        top = m[col] & (Zs > 0.033) & (Zs < 0.05)
        cl = _cluster(np.stack([X[top], Y[top]], 1), 0.03, 8) if top.sum() >= 8 else []
        for c, n in cl:
            bl = list(buttons.values()) + [b for b in (known_buttons or []) if b]
            near_btn = any(math.hypot(c[0] - b[0], c[1] - b[1]) < 0.06 for b in bl)
            if not near_btn:
                cubes.append([float(c[0]), float(c[1]), col, n])
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    grey = (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & (r > 140) & (Zs > 0.0) & (Zs < 0.015) & (X > -0.4)
    disc = None
    if grey.sum() > 200:
        pts = np.stack([X[grey], Y[grey]], 1)
        c = np.median(pts, 0)
        for _ in range(3):
            sel = np.linalg.norm(pts - c, axis=1) < 0.14
            q = pts[sel]
            c = (q.min(0) + q.max(0)) / 2.0
        ext = q.max(0) - q.min(0)
        disc = [float(c[0]), float(c[1]), float(min(ext.max() / 2.0, 0.125))]
    white = (r > 200) & (g > 200) & (b > 200) & (Zs > 0.03) & (Zs < 0.2) & (X > 0.12)
    lamp = None
    if white.sum() > 30:
        vv, uu = np.nonzero(white)
        lamp = [int(uu.min()), int(vv.min()), int(uu.max()) + 1, int(vv.max()) + 1, int(white.sum())]
    return {"buttons": buttons, "cubes": cubes, "disc": disc, "lamp": lamp}


def _flash(frame, lamp):
    u0, v0, u1, v1, n = lamp
    sub = frame.front_rgb[v0:v1, u0:u1]
    m = _masks(sub)
    best, cnt = None, 0
    for col in COLORS:
        k = int(m[col].sum())
        if k > cnt:
            best, cnt = col, k
    if cnt > 0.3 * n:
        return best
    return None


def _slots(disc):
    cx, cy, R = disc
    rr = min(RING_R, R - 0.045)
    out = []
    for k in range(6):
        a = math.radians(60.0 * k)
        out.append([cx + rr * math.cos(a), cy + rr * math.sin(a)])
    out.append([cx, cy])
    return out


# ------------------------------------------------------------------ policy
def perceive_decide(obs, goal, memory, last):
    """Track the lamp flash in every new frame, keep the cube plan and pick the next action."""
    mem = dict(memory)
    cur = obs.current
    if mem.get("stage") is None:
        sc = _scene(cur)
        mem["stage"] = "gather"
        mem["disc"] = sc["disc"]
        mem["buttons"] = [sc["buttons"].get(c) for c in COLORS]
        mem["lamp"] = sc["lamp"]
        mem["cubes"] = [[c[0], c[1], 0.02] for c in sc["cubes"]]
        mem["cube_colors"] = [c[2] for c in sc["cubes"]]
        mem["placed"] = 0
        mem["slot_next"] = 0
        mem["flash_color"] = None
        mem["flash_step"] = None
        mem["flash_votes"] = [0, 0, 0]
        mem["attempts"] = 0
        mem["log"] = "init cubes=%d disc=%s lamp=%s" % (len(sc["cubes"]), sc["disc"], sc["lamp"])
    # --- flash watch over every frame since the previous action
    if mem.get("lamp") is not None:
        votes = list(mem.get("flash_votes") or [0, 0, 0])
        for f in obs.frames:
            col = _flash(f, mem["lamp"])
            if col is not None:
                votes[COLORS.index(col)] += 1
                if mem.get("flash_step") is None:
                    mem["flash_step"] = int(f.step)
        mem["flash_votes"] = votes
        if max(votes) > 0:
            mem["flash_color"] = COLORS[int(np.argmax(votes))]
    if isinstance(last, str) and last.startswith("act_pick_place") and '"finger_open"' in last:
        mem["placed"] = int(mem["placed"]) + 1
        mem["slot_next"] = int(mem["slot_next"]) + 1
    tcp = np.asarray(cur.tcp, dtype=np.float64)
    disc = mem["disc"]
    if mem["stage"] == "gather":
        sc = _scene(cur, mem["buttons"])
        det = [c for c in sc["cubes"] if math.hypot(c[0] - disc[0], c[1] - disc[1]) > disc[2] - 0.005]
        # merge with remembered off-disc cubes that may be occluded now
        prev = [[p[0], p[1], col] for p, col in zip(mem["cubes"], mem["cube_colors"])]
        remaining = [[c[0], c[1], c[2]] for c in det]
        need = 7 - int(mem["placed"])
        if len(remaining) < need:
            for p in prev:
                if math.hypot(p[0] - disc[0], p[1] - disc[1]) < disc[2]:
                    continue
                if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.03 for q in remaining):
                    remaining.append(p)
                if len(remaining) >= need:
                    break
        mem["cubes"] = [[q[0], q[1], 0.02] for q in remaining]
        mem["cube_colors"] = [q[2] for q in remaining]
        if remaining and int(mem["attempts"]) < 14:
            d = [math.hypot(q[0] - tcp[0], q[1] - tcp[1]) for q in remaining]
            i = int(np.argmin(d))
            slots = _slots(disc)
            s = slots[int(mem["slot_next"]) % len(slots)]
            mem["attempts"] = int(mem["attempts"]) + 1
            return {"memory": mem, "act": {"tool": "act_pick_place", "args": {
                "cube": [remaining[i][0], remaining[i][1]], "slot": s}}}
        mem["stage"] = "press"
    if mem["stage"] == "press":
        col = mem.get("flash_color")
        if col is None:
            if int(cur.step) < 700:
                return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}
            col = "green"  # no flash seen: uninformed guess (logged)
            mem["log"] = (mem.get("log") or "") + " | no flash seen, guessing"
        b = mem["buttons"][COLORS.index(col)]
        return {"memory": mem, "act": {"tool": "act_press", "args": {"button": [b[0], b[1]]}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


# -------------------------------------------------------------------- motion
def _go(robot, tgt, grip, max_steps, xy_tol=0.005, z_tol=0.006, settle=True,
        lift_first=None, descend_xy=None, z_hold=None, vmax=1.0, yaw_t=0.0, grip_lift=None):
    """Proportional servo of the measured TCP toward tgt (world xyz).
    lift_first: keep xy until z >= lift_first. descend_xy: keep z at z_hold until xy error < descend_xy.
    Returns (steps, error xyz)."""
    tgt = np.asarray(tgt, dtype=np.float64)
    prev = None
    lifted = lift_first is None
    stall = 0
    for n in range(max_steps):
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        p = tcp[:3]
        err = tgt - p
        if not lifted and p[2] >= lift_first:
            lifted = True
        exy = math.hypot(err[0], err[1])
        vel = 9.0 if prev is None else float(np.linalg.norm(p - prev))
        prev = p
        if exy < xy_tol and abs(err[2]) < z_tol and (not settle or vel < 0.002):
            return n, err
        stall = stall + 1 if vel < 0.0008 else 0
        if stall >= 4 and n > 6 and exy < xy_tol and abs(err[2]) < 0.02:
            return n, err  # blocked (e.g. contact) but laterally on target
        cmd = err.copy()
        if not lifted:
            cmd[0] = 0.0
            cmd[1] = 0.0
            cmd[2] = max(cmd[2], lift_first + 0.01 - p[2])
        elif descend_xy is not None and exy > descend_xy and z_hold is not None:
            cmd[2] = z_hold - p[2]
        a = np.clip(KP * cmd / 0.1, -vmax, vmax)
        r = _rot_cmd(tcp, yaw_t)
        g = grip if (lifted or grip_lift is None) else grip_lift
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], g)
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return max_steps, tgt - tcp[:3]


def _grip_wait(robot, hold, grip, max_steps, done=None, yaw_t=0.0):
    """Hold position while the fingers move; stop when they stop (or done(fingers) is true)."""
    hold = np.asarray(hold, dtype=np.float64)
    last = None
    still = 0
    for n in range(max_steps):
        f = robot.now()
        fing = float(np.mean(f.fingers))
        if done is not None and done(fing) and n >= 2:
            return n, fing
        if last is not None and abs(fing - last) < 0.0006:
            still += 1
            if still >= 2 and n >= 3:
                return n, fing
        else:
            still = 0
        last = fing
        tcp = np.asarray(f.tcp, dtype=np.float64)
        a = np.clip(KP * (hold - tcp[:3]) / 0.1, -1, 1)
        r = _rot_cmd(tcp, yaw_t)
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], grip)
    return max_steps, float(np.mean(robot.now().fingers))


def _released(f):
    return f > 0.026


def act_pick_place(robot, cube, slot):
    """Pick the cube at world xy `cube` (top-down grasp) and release it over disc slot xy `slot`.
    Returns measured stage outcomes (steps, errors, finger widths)."""
    out = {"cube": [round(float(v), 4) for v in cube], "slot": [round(float(v), 4) for v in slot]}
    s0 = int(robot.now().step)
    cx, cy = float(cube[0]), float(cube[1])
    sx, sy = float(slot[0]), float(slot[1])
    chk = robot.dry_run(cx, cy, Z_GRASP, math.pi, 0.0, 0.0, mode="absolute")
    out["dry_grasp_ok"] = bool(chk.get("within_joint_limits", False))
    # up (if low), across at travel height, down onto the cube
    n, e = _go(robot, [cx, cy, Z_GRASP], 1.0, 80, xy_tol=0.004, z_tol=0.006,
               lift_first=0.07, descend_xy=0.012, z_hold=Z_TRAVEL, grip_lift=RELEASE_GRIP)
    out["approach_steps"] = n
    out["grasp_err"] = [round(float(v), 4) for v in e]
    n, fing = _grip_wait(robot, [cx, cy, Z_GRASP], -1.0, 10)
    out["close_steps"] = n
    out["finger_closed"] = round(fing, 4)
    held = fing > 0.012
    out["held"] = bool(held)
    if not held:
        _go(robot, [cx, cy, Z_TRAVEL], 1.0, 20, z_tol=0.02, settle=False)
        out["steps"] = int(robot.now().step) - s0
        return out
    n, e = _go(robot, [sx, sy, Z_PLACE], -1.0, 80, xy_tol=0.006, z_tol=0.006,
               lift_first=0.07, descend_xy=0.012, z_hold=Z_TRAVEL)
    out["transfer_steps"] = n
    out["place_err"] = [round(float(v), 4) for v in e]
    out["finger_carry"] = round(float(np.mean(robot.now().fingers)), 4)
    n, fing = _grip_wait(robot, [sx, sy, Z_PLACE], RELEASE_GRIP, 8, done=_released)
    out["open_steps"] = n
    out["finger_open"] = round(fing, 4)
    out["steps"] = int(robot.now().step) - s0
    out["end_step"] = int(robot.now().step)
    return out


def act_press(robot, button):
    """Close the gripper, hover above the button at world xy `button`, then push straight down."""
    out = {"button": [round(float(v), 4) for v in button]}
    bx, by = float(button[0]), float(button[1])
    s0 = int(robot.now().step)
    n, e = _go(robot, [bx, by, Z_BTN_HOVER], -1.0, 60, xy_tol=0.004, z_tol=0.01, lift_first=0.09,
               grip_lift=RELEASE_GRIP)  # stay open (cube released) until clear, then close
    out["hover_steps"] = n
    out["hover_err"] = [round(float(v), 4) for v in e]
    n, e = _go(robot, [bx, by, Z_BTN_PRESS], -1.0, 40, xy_tol=0.004, z_tol=0.004, vmax=0.6)
    out["press_steps"] = n
    out["press_err"] = [round(float(v), 4) for v in e]
    out["tcp_z"] = round(float(robot.now().tcp[2]), 4)
    _go(robot, [bx, by, Z_BTN_HOVER], -1.0, 20, z_tol=0.02, settle=False)
    out["steps"] = int(robot.now().step) - s0
    out["end_step"] = int(robot.now().step)
    return out


def act_hold(robot, steps=20):
    """Hold the current pose for `steps` control steps (observing)."""
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    g = 1.0 if float(np.mean(robot.now().fingers)) > 0.02 else -1.0
    for _ in range(int(steps)):
        cur = np.asarray(robot.now().tcp, dtype=np.float64)
        a = np.clip(KP * (tcp[:3] - cur[:3]) / 0.1, -1, 1)
        r = _rot_cmd(cur, 0.0)
        robot.step(a[0], a[1], a[2], r[0], r[1], r[2], g)
    return {"held_steps": int(steps), "step": int(robot.now().step)}
