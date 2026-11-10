"""ShellGameTouch policy: remember which mug hides the ball, then touch that mug
by bringing the inner finger pads against its side wall (edge grasp).

Evidence (50 assigned demos, see task_analysis.md):
- frame 0..k (k in 0..4) show the red ball on the table, no mugs; commands are
  zeroed by the env during this cue; mugs (upside-down red YCB mugs, top z~0.108,
  top diameter ~0.092, handle on +y) appear at frame k+1 and stay still.
- the mug covering the ball has its centre ~(-0.0095,-0.007) from the front-camera
  ball-surface centroid (std ~1 mm); neighbours are ~0.197 m apart in y.
- the expert's fingertips resting on the mug top did not end the episode for
  5..23 steps; success came when a finger pad side pressed the mug wall
  (hypothesis: lateral finger contact is required, vertical top contact is not).
"""

MUG_R = 0.046          # top radius of the mugs (m), measured 0.092 diameter
GRASP_D = 0.042        # TCP offset from mug centre along x, fingers straddle the edge chord
Z_SAFE = 0.14          # travel height (fingertips ~0.13 > mug top 0.108)
Z_PRE = 0.128          # pre-descent height while xy still converging
Z_GRASP = 0.08         # TCP height for the edge grasp (fingertips ~0.072)
Z_TOP = 0.108          # mug top height
BALL_TO_MUG = (-0.0095, -0.007)   # mug centre minus front ball-surface centroid
BASE_XY = (-0.615, 0.0)           # robot base in world (from calibration + FK)
REACH_MAX = 0.78       # max horizontal TCP radius used with a vertical gripper
STEP_LIMIT = 30


def _deproject(frame, cam, mask):
    """World points (N,3) of the masked pixels of camera `cam` ('front'/'wrist')."""
    depth = np.asarray(getattr(frame, cam + "_depth"), dtype=np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    vs, us = np.nonzero(mask & (depth > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))
    d = depth[vs, us]
    p = np.stack([(us - K[0, 2]) / K[0, 0] * d, (vs - K[1, 2]) / K[1, 1] * d, d], 1)
    R = E[:, :3]
    t = E[:, 3]
    return (p - t) @ R


def _ball_mask(rgb):
    f = np.asarray(rgb).astype(np.int32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (r > 150) & (g < 70) & (b < 70) & (r >= 3 * g)


def _mug_mask(rgb):
    f = np.asarray(rgb).astype(np.int32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    return (r >= 35) & (r <= 200) & (r >= 2 * g) & (r >= 2 * b)


def _cups_visible(frame):
    return int(_mug_mask(frame.front_rgb).sum()) > 800


def _detect_ball(frame):
    """Ball surface centroid in world from front (preferred) or wrist camera."""
    for cam, minpx in (("front", 8), ("wrist", 20)):
        m = _ball_mask(getattr(frame, cam + "_rgb"))
        if int(m.sum()) < minpx:
            continue
        P = _deproject(frame, cam, m)
        if len(P) < 4:
            continue
        c = np.median(P, axis=0)
        if not (-0.02 < c[2] < 0.08):
            continue
        if cam == "wrist":
            # convert the wrist estimate to the front-camera convention of BALL_TO_MUG
            # (measured over the 50 demos: wrist - front = (-0.0032, +0.0032))
            c = np.array([c[0] + 0.0032, c[1] - 0.0032, c[2]])
        return [float(c[0]), float(c[1]), float(c[2])], cam, int(m.sum())
    return None


def _mug_tops(frame):
    """Centres [x, y, n] of mug top discs seen by the front camera."""
    P = _deproject(frame, "front", _mug_mask(frame.front_rgb))
    if len(P) == 0:
        return []
    P = P[(P[:, 2] > 0.095) & (P[:, 2] < 0.13)]
    if len(P) < 30:
        return []
    cell = 0.01
    ij = np.floor(P[:, :2] / cell).astype(np.int64)
    keys = {}
    for k, (a, b) in enumerate(ij.tolist()):
        keys.setdefault((a, b), []).append(k)
    seen = set()
    out = []
    for start in list(keys.keys()):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        idx = []
        while stack:
            c = stack.pop()
            idx.extend(keys[c])
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nb = (c[0] + da, c[1] + db)
                    if nb in keys and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(idx) < 60:
            continue
        Q = P[np.array(idx)]
        x0, x1 = np.percentile(Q[:, 0], [2, 98])
        y0, y1 = np.percentile(Q[:, 1], [2, 98])
        out.append([float((x0 + x1) / 2), float((y0 + y1) / 2), int(len(idx)),
                    float(x1 - x0), float(y1 - y0)])
    return out


def _choose_grasp(cx, cy):
    """Grasp TCP xy on the +x edge of the mug, or -x edge if +x is out of reach."""
    side = 1.0
    gx = cx + GRASP_D
    if math.hypot(gx - BASE_XY[0], cy - BASE_XY[1]) > REACH_MAX:
        side = -1.0
        gx = cx - GRASP_D
    return gx, cy, side


def _cmd_axis(err, vel, gain=0.75):
    """Normalized command for one axis using the fitted response
    v[t+1] = 0.31*u[t] + 0.23*v[t] (u = 0.1*cmd metres)."""
    u = (gain * err - 0.23 * vel) / 0.31
    return float(max(-1.0, min(1.0, u / 0.1)))


def _tcp(frame):
    return np.asarray(frame.tcp, dtype=np.float64)


def _step_to(robot, target, prev, gripper, gain=0.75, extra=None):
    """One closed-loop step toward target xyz. Returns new tcp."""
    cur = _tcp(robot.now())
    vel = cur[:3] - prev[:3] if prev is not None else np.zeros(3)
    c = [_cmd_axis(target[i] - cur[i], vel[i], gain) for i in range(3)]
    if extra is not None:
        for i in range(3):
            c[i] = float(max(-1.0, min(1.0, c[i] + extra[i])))
    robot.step(c[0], c[1], c[2], 0.0, 0.0, 0.0, gripper)
    return cur


def _frame_step(frame, fallback):
    s = getattr(frame, "step", None)
    if isinstance(s, (int, np.integer)):
        return int(s)
    return fallback


def act_wait_cups(robot, max_steps=8):
    """Hold still (gripper open) until the mugs appear in the front camera, at most
    max_steps native steps. Commands are zeroed by the task during the cue anyway."""
    n = 0
    while n < max_steps and not _cups_visible(robot.now()):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    f = robot.now()
    return {"waited_steps": n, "cups_visible": bool(_cups_visible(f)),
            "step": _frame_step(f, -1)}


def act_touch_mug(robot, gx, gy, side, cx, cy):
    """Move above the edge-grasp point (gx, gy), descend so the open fingers straddle
    the mug edge chord, then close gently and push toward the mug centre so the finger
    pads press the mug side wall. Retries once with a larger offset if the fingertips
    land on the rim. Returns measured stage progress."""
    log = {"stage": "approach"}
    prev = None
    gxx = float(gx)
    n = 0
    # --- approach at safe height, then pre-descent and descent ---
    landed = False
    retries = 0
    while n < 24:
        cur = _tcp(robot.now())
        exy = math.hypot(gxx - cur[0], gy - cur[1])
        if exy > 0.02:
            zt = Z_SAFE
        elif exy > 0.005 and cur[2] > Z_PRE - 0.004:
            zt = Z_PRE
        else:
            zt = Z_GRASP
        if cur[2] < Z_TOP - 0.006 and exy > 0.012:
            zt = cur[2]  # already below the rim: do not climb, just correct xy gently
        # stall on rim: commanded down, above rim height, not descending
        if prev is not None and zt == Z_GRASP and cur[2] > Z_TOP - 0.004 and cur[2] < Z_PRE - 0.008 \
                and (prev[2] - cur[2]) < 0.003:
            retries += 1
            log["rim_stall_%d" % retries] = [round(float(cur[0]), 4), round(float(cur[1]), 4), round(float(cur[2]), 4)]
            if retries > 2:
                break
            gxx = gxx + side * 0.007
            # lift a little before re-centering
            for _ in range(2):
                prev = _step_to(robot, [cur[0], cur[1], Z_PRE + 0.005], prev, 1.0)
                n += 1
            continue
        if zt == Z_GRASP and abs(cur[2] - Z_GRASP) < 0.008 and exy < 0.006:
            landed = True
            break
        prev = _step_to(robot, [gxx, gy, zt], prev, 1.0)
        n += 1
    cur = _tcp(robot.now())
    log.update({"stage": "at_grasp" if landed else "approach_incomplete", "steps": n,
                "tcp": [round(float(v), 4) for v in cur[:3]], "gx_used": round(gxx, 4),
                "fingers": [round(float(v), 4) for v in robot.now().fingers]})
    if not landed and cur[2] > Z_TOP:
        return log
    # --- gentle close: finger target ~0.028 m (contact expected ~0.031) ---
    for _ in range(2):
        prev = _step_to(robot, [gxx, gy, Z_GRASP], prev, 0.4)
        n += 1
    fing = [float(v) for v in robot.now().fingers]
    log["fingers_after_close"] = [round(v, 4) for v in fing]
    # --- push pads against the wall toward the mug centre while squeezing ---
    for k in range(3):
        tx = gxx - side * 0.008 * (k + 1)
        prev = _step_to(robot, [tx, gy, Z_GRASP], prev, 0.2)
        n += 1
    log["fingers_after_push"] = [round(float(v), 4) for v in robot.now().fingers]
    # --- firmer squeeze while pressing the pads inward (expert-like low pose) ---
    for k in range(3):
        tx = gxx - side * 0.025
        prev = _step_to(robot, [tx, gy, Z_GRASP - 0.008], prev, -0.4)
        n += 1
    cur = _tcp(robot.now())
    log.update({"stage": "squeezed", "steps": n,
                "tcp_end": [round(float(v), 4) for v in cur[:3]],
                "fingers_end": [round(float(v), 4) for v in robot.now().fingers]})
    return log


def act_press_fallback(robot, cx, cy, side, max_steps=12):
    """Expert-like fallback (closed loop): lift vertically out of the mug's reach, move
    over the mug top 0.03 m from its centre toward `side`, then press the open fingertips
    down on the top, pull toward the centre and tilt (pitch) while slowly closing the
    fingers, as every demo expert did. Runs until the episode ends or max_steps."""
    prev = None
    n = 0
    # 1) vertical lift (hold xy) so the fingers clear the mug before any xy motion
    while n < 4 and _tcp(robot.now())[2] < Z_PRE - 0.006:
        cur = _tcp(robot.now())
        prev = _step_to(robot, [cur[0], cur[1], Z_PRE + 0.01], prev, 1.0)
        n += 1
    # 2) move over the near-edge point of the top
    tx = cx + side * 0.03
    ty = cy
    while n < 9:
        cur = _tcp(robot.now())
        if math.hypot(tx - cur[0], ty - cur[1]) < 0.006:
            break
        prev = _step_to(robot, [tx, ty, max(Z_PRE, float(cur[2]))], prev, 1.0)
        n += 1
    # 3) press, pull inward (bounded, closed loop on xy), tilt, slowly close
    k = 0
    while n < max_steps:
        cur = _tcp(robot.now())
        vel = cur[:3] - prev[:3] if prev is not None else np.zeros(3)
        ux = _cmd_axis(tx - side * 0.015 - cur[0], vel[0])
        uy = _cmd_axis(ty - cur[1], vel[1])
        ux = max(-0.5, min(0.5, ux))
        uy = max(-0.5, min(0.5, uy))
        uz = -1.0 if cur[2] > Z_TOP - 0.03 else 0.0
        g = max(0.2, 0.9 - 0.06 * k)
        prev = cur
        robot.step(ux, uy, uz, 0.0, -1.0 * side, 0.0, g)
        n += 1
        k += 1
    cur = _tcp(robot.now())
    return {"stage": "pressed", "steps": n, "tcp": [round(float(v), 4) for v in cur[:3]],
            "rpy": [round(float(v), 3) for v in cur[3:]],
            "fingers": [round(float(v), 4) for v in robot.now().fingers]}


def act_hold(robot, steps=1):
    """Hold the current position (zero translation) with the fingers half closed."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return {"held": int(steps), "tcp": [round(float(v), 4) for v in _tcp(robot.now())[:3]]}


def perceive_decide(obs, goal, memory, last):
    """Detect and remember the ball during the cue, locate the mug that covers it once
    the mugs appear, then run the touch, fallback and hold stages in order."""
    mem = {
        "stage": memory.get("stage") or "cue",
        "ball": memory.get("ball"),
        "ball_cam": memory.get("ball_cam"),
        "mug": memory.get("mug"),
        "grasp": memory.get("grasp"),
        "side": memory.get("side"),
        "mugs_seen": memory.get("mugs_seen"),
        "attempts": memory.get("attempts") if memory.get("attempts") is not None else 0,
        "notes": memory.get("notes") or "",
    }
    frames = list(obs.frames)
    cur = obs.current
    # 1) remember the ball as soon as it is visible (it vanishes when mugs appear)
    if mem["ball"] is None:
        for f in frames:
            if _cups_visible(f):
                continue
            r = _detect_ball(f)
            if r is not None:
                mem["ball"] = r[0]
                mem["ball_cam"] = r[1]
                mem["notes"] += "ball@%s via %s px=%d; " % (str(getattr(f, "step", "?")), r[1], r[2])
                break
    # 2) wait for the mugs
    if mem["mug"] is None:
        if not _cups_visible(cur):
            mem["stage"] = "cue"
            return {"memory": mem, "act": {"tool": "act_wait_cups", "args": {"max_steps": 8}}}
        tops = _mug_tops(cur)
        mem["mugs_seen"] = [[round(t[0], 4), round(t[1], 4)] for t in tops]
        target = None
        if mem["ball"] is not None:
            ex = mem["ball"][0] + BALL_TO_MUG[0]
            ey = mem["ball"][1] + BALL_TO_MUG[1]
            best = None
            for t in tops:
                d = math.hypot(t[0] - ex, t[1] - ey)
                if 0.06 < t[3] < 0.12 and 0.06 < t[4] < 0.12 and (best is None or d < best[0]):
                    best = (d, t)
            if best is not None and best[0] < 0.04:
                target = [best[1][0], best[1][1]]
                mem["notes"] += "mug from top disc d=%.3f; " % best[0]
            else:
                target = [ex, ey]
                mem["notes"] += "mug from ball offset (no disc match); "
        elif tops:
            ts = sorted(tops, key=lambda t: t[1])
            mid = ts[len(ts) // 2]
            target = [mid[0], mid[1]]
            mem["notes"] += "NO BALL SEEN: guessing middle mug; "
        else:
            mem["notes"] += "no ball and no mug discs; "
            return {"memory": mem, "act": {"tool": "act_wait_cups", "args": {"max_steps": 1}}}
        gx, gy, side = _choose_grasp(target[0], target[1])
        mem["mug"] = [round(float(target[0]), 4), round(float(target[1]), 4), Z_TOP]
        mem["grasp"] = [round(float(gx), 4), round(float(gy), 4), Z_GRASP]
        mem["side"] = side
        mem["stage"] = "touch"
    if mem["stage"] == "touch":
        mem["stage"] = "fallback"
        mem["attempts"] = int(mem["attempts"]) + 1
        return {"memory": mem, "act": {"tool": "act_touch_mug", "args": {
            "gx": mem["grasp"][0], "gy": mem["grasp"][1], "side": mem["side"],
            "cx": mem["mug"][0], "cy": mem["mug"][1]}}}
    if mem["stage"] == "fallback":
        mem["stage"] = "hold"
        mem["notes"] += "touch stage ended without success; fallback press; "
        return {"memory": mem, "act": {"tool": "act_press_fallback", "args": {
            "cx": mem["mug"][0], "cy": mem["mug"][1], "side": mem["side"], "max_steps": 12}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 2}}}
