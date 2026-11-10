"""ShellGameTouch policy: remember where the ball is during the cue, then touch the cup covering it."""

# ---------------- tunable constants (evidence in task_analysis.md) ----------------
BALL_MIN_PIXELS = 8          # bright-red pixels needed to call the ball visible
GAIN_PER_UNIT = 0.037        # measured TCP metres moved per step per unit command (demo fit)
KP = [0.6, 0.8, 0.8]         # per-axis fraction of remaining error requested per step (x, y, z)
KD = [0.0, 0.5, 0.5]         # per-axis damping on the last measured step displacement
ROT_GAIN = 6.0               # normalized rotation command per radian of error
SAFE_Z = 0.17                # minimum travel height until above the target (cup tops ~0.108 m)
XY_ALIGN_TOL = 0.03          # xy error allowed before descending below SAFE_Z
TARGET_OFFSET = [0.0, 0.0]   # TCP xy offset from estimated cup-top centre
CONTACT_DZ = 0.009           # TCP height above cup top at which closed fingertips touch (R002 stall 0.116 vs top 0.107)
SETTLE_DZ = 0.013            # hold TCP this far above the cup top (closed fingertips ~5 mm above it)
HOLD_STEPS = 5               # steps per hold level before switching settle <-> light contact
HOLD_CMD_MAX = 0.5           # command bound while holding over the cup
HOLD_ROT_SCALE = 0.5         # weaker orientation correction while holding (contact torques)
GRIPPER = -1.0               # finger command during approach/press (-1 closed, +1 open)
TARGET_YAW = 0.0             # desired TCP yaw (roll pi, pitch 0)
HOVER_Z = 0.127              # hover height (TCP) above the cup top (~0.108) before pressing
XY_ARRIVE_TOL = 0.006        # xy error to start pressing
TRAVEL_MAX_STEPS = 16        # bound on travel phase
TRACK_MIN_Z = 0.19           # update the cup-row shift only while the TCP is above this height
SLIDE_THRESH = 0.015         # cup displacement (m) treated as 'cups slid after appearing'
BALL_IN_CUP = 0.03           # max ball offset from a moved cup centre (inner radius - ball radius, assumed; R010 evidence)
BALL_TO_CUP = [-0.006, -0.005]  # cup-top centre minus ball centroid in non-sliding episodes (demos, R001/R004-R006)
UNSETTLED_Z = 0.22           # stay at least this high until the cup row has been still for 2 frames


def _deproject(K, E, us, vs, d):
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (pc - t) @ R


def _ball(frame):
    """Bright red ball in the front camera -> (world xyz list, pixel count) or (None, count)."""
    im = np.asarray(frame.front_rgb).astype(np.int32)
    m = (im[..., 0] > 150) & (im[..., 1] < 50) & (im[..., 2] < 50)
    n = int(m.sum())
    if n < BALL_MIN_PIXELS:
        return None, n
    vs, us = np.nonzero(m)
    d = np.asarray(frame.front_depth)[vs, us].astype(np.float64) / 1000.0
    ok = d > 0
    if ok.sum() < 3:
        return None, n
    w = _deproject(np.asarray(frame.front_K, dtype=np.float64), frame.front_extrinsic, us[ok], vs[ok], d[ok])
    return [float(v) for v in w.mean(0)], n


def _cup_centre(frame, near_xy):
    """Top-face centre of the dark-red cup nearest near_xy from front RGB-D, or None."""
    im = np.asarray(frame.front_rgb).astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 40) & (r - g > 45) & (r - b > 45) & (g < 70) & ~((r > 150) & (g < 50) & (b < 50))
    if m.sum() < 30:
        return None
    vs, us = np.nonzero(m)
    d = np.asarray(frame.front_depth)[vs, us].astype(np.float64) / 1000.0
    ok = d > 0
    w = _deproject(np.asarray(frame.front_K, dtype=np.float64), frame.front_extrinsic, us[ok], vs[ok], d[ok])
    near = w[(np.hypot(w[:, 0] - near_xy[0], w[:, 1] - near_xy[1]) < 0.07) & (w[:, 2] > 0.04) & (w[:, 2] < 0.2)]
    if len(near) < 30:
        return None
    ztop = float(np.percentile(near[:, 2], 97))
    sel = near[near[:, 2] > ztop - 0.012]
    if len(sel) < 20:
        return None
    xs = np.percentile(sel[:, 0], [3, 97])
    ys = np.percentile(sel[:, 1], [3, 97])
    return [float((xs[0] + xs[1]) / 2), float((ys[0] + ys[1]) / 2), ztop, int(len(sel))]


def _cup_tops(frame):
    """All cup top-face centres [x, y, ztop, n] in the front camera, clustered along world y (cups form a row)."""
    im = np.asarray(frame.front_rgb).astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 40) & (r - g > 45) & (r - b > 45) & (g < 70) & ~((r > 150) & (g < 50) & (b < 50))
    if m.sum() < 60:
        return []
    vs, us = np.nonzero(m)
    d = np.asarray(frame.front_depth)[vs, us].astype(np.float64) / 1000.0
    ok = d > 0
    w = _deproject(np.asarray(frame.front_K, dtype=np.float64), frame.front_extrinsic, us[ok], vs[ok], d[ok])
    w = w[(w[:, 2] > 0.04) & (w[:, 2] < 0.2)]
    if len(w) < 60:
        return []
    order = np.argsort(w[:, 1])
    w = w[order]
    gaps = np.flatnonzero(np.diff(w[:, 1]) > 0.03)
    starts = [0] + [int(g) + 1 for g in gaps]
    ends = [int(g) + 1 for g in gaps] + [len(w)]
    out = []
    for a, e in zip(starts, ends):
        c = w[a:e]
        if len(c) < 40:
            continue
        zt = float(np.percentile(c[:, 2], 97))
        t = c[c[:, 2] > zt - 0.012]
        if len(t) < 20:
            continue
        xs = np.percentile(t[:, 0], [3, 97])
        ys = np.percentile(t[:, 1], [3, 97])
        out.append([float((xs[0] + xs[1]) / 2), float((ys[0] + ys[1]) / 2), zt, int(len(t))])
    return out


def _row_shift(ref, now):
    """Median xy displacement of reference cups [x,y,ztop,n] matched (nearest y within 6 cm) in the current detection.
    A match counts only if it looks unoccluded: >= 80% of the reference top points and a cup-like top height (0.088-0.116 m)."""
    ds = []
    for c in ref:
        best = None
        for q in now:
            dy = abs(q[1] - c[1])
            if dy < 0.06 and (best is None or dy < abs(best[1] - c[1])):
                best = q
        if best is None:
            continue
        nref = c[3] if len(c) > 3 else best[3]
        if best[3] >= 0.8 * nref and 0.088 < best[2] < 0.116:
            ds.append([best[0] - c[0], best[1] - c[1]])
    if not ds:
        return None, 0
    ds = np.array(ds)
    return [float(np.median(ds[:, 0])), float(np.median(ds[:, 1]))], len(ds)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return v / (2 * s) * ang


def _rot_cmd(tcp, yaw):
    """Normalized rotation command toward roll=pi, pitch=0, yaw (offset is -0.1*cmd about base axes)."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(math.pi, 0.0, yaw)
    err = _rotvec(Rt @ Rc.T)          # world-frame rotation that takes current to target
    cmd = -ROT_GAIN * err
    return [float(np.clip(c, -1, 1)) for c in cmd], float(np.linalg.norm(err))


def _pos_cmd(err, vel=None):
    """Per-axis PD on measured TCP error; vel = last measured per-step displacement (m/step).
    Gains tuned on a per-axis model fitted to rollout motion (scratch/fitdyn.py, tune.py)."""
    v = vel if vel is not None else [0.0, 0.0, 0.0]
    out = []
    for k in range(3):
        c = (KP[k] * err[k] - KD[k] * v[k]) / GAIN_PER_UNIT
        out.append(float(np.clip(c, -1, 1)))
    return out


def _summ(tcp):
    return [round(float(v), 4) for v in tcp]


def perceive_decide(obs, goal, memory, last):
    """Track the cue ball, then choose approach and press stages."""
    mem = dict(memory)
    for k, v in (("ball", []), ("ball_frames", 0), ("stage", "cue"), ("target", []), ("notes", ""), ("cue_steps", 0), ("cups", [])):
        if k not in mem or mem[k] is None:
            mem[k] = v
    cur = obs.current
    seen = None
    for fr in obs.frames:
        bp, n = _ball(fr)
        if bp is not None:
            seen = bp
            k = int(mem["ball_frames"])
            if mem["ball"]:
                mem["ball"] = [(mem["ball"][i] * k + bp[i]) / (k + 1) for i in range(3)]
            else:
                mem["ball"] = bp
            mem["ball_frames"] = k + 1
    stage = mem["stage"]
    if stage == "cue":
        if seen is not None or (not mem["ball"] and int(mem["cue_steps"]) < 10):
            mem["cue_steps"] = int(mem["cue_steps"]) + 1
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 1}}}
        if not mem["ball"]:
            # fallback (never observed in demos): no ball seen; guess the cup nearest the scene centre line
            mem["ball"] = [0.0, 0.0, 0.035]
            mem["notes"] = "NO BALL SEEN - guessed centre cup. "
        cc = _cup_centre(cur, mem["ball"])
        if cc is not None and math.hypot(cc[0] - mem["ball"][0], cc[1] - mem["ball"][1]) < 0.04:
            centre = cc[:2]
            mem["notes"] = mem["notes"] + "cup top centre from depth n=%d ztop=%.3f" % (cc[3], cc[2])
        else:
            centre = [mem["ball"][0] - 0.008, mem["ball"][1] - 0.006]
            mem["notes"] = mem["notes"] + "cup centre from ball (cup not found)"
        ztop = cc[2] if cc is not None and 0.09 < cc[2] < 0.125 else 0.108
        mem["target"] = [centre[0] + TARGET_OFFSET[0], centre[1] + TARGET_OFFSET[1], max(ztop, 0.105)]
        mem["cups"] = [[round(c[0], 4), round(c[1], 4), round(c[2], 4), int(c[3])] for c in _cup_tops(cur)]
        mem["stage"] = "touch"
    # touch stage: (re)issue the closed-loop touch routine; it runs until the episode ends
    return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": mem["target"][0], "y": mem["target"][1], "ztop": mem["target"][2], "ref_cups": mem["cups"], "ball": mem["ball"][:2]}}}


def act_hold(robot, steps=1):
    """Send zero motion for `steps` native steps (cue phase; commands are suppressed anyway)."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return {"tcp": _summ(robot.now().tcp)}


def act_touch(robot, x, y, ztop=0.108, ref_cups=None, ball=None, max_steps=60):
    """Touch the cup that covers the ball.
    (x, y): cup-top centre at cue end; ref_cups: all cup tops then (for row-shift tracking); ball: cue ball xy.
    Track the cup row while travelling high (cups may slide after appearing), then aim over the ball's modelled place
    (= cup centre unless the row slid), settle the closed fingertips just above / lightly on the cup top and hold still
    (levels alternate every HOLD_STEPS). Returns phase counts, mode, tracking log and a short TCP trace."""
    log = []
    phase = "travel"
    counts = {}
    hold = 0
    track = []
    x0, y0 = x, y
    ref_cups = ref_cups or []
    stable = 0
    prev_tcp = None
    mode = "tracked"
    cup_xy = None
    cx, cy = x, y
    for i in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=np.float64)
        # live row tracking in the front camera (median shift of unoccluded cups), only while travelling high
        if ref_cups and phase == "travel" and tcp[2] > TRACK_MIN_Z and mode == "tracked":
            sh, nm = _row_shift(ref_cups, _cup_tops(fr))
            if sh is not None and math.hypot(sh[0], sh[1]) < 0.15:
                nx, ny = x0 + sh[0], y0 + sh[1]
                moved = math.hypot(nx - cx, ny - cy)
                stable = stable + 1 if moved < 0.002 else 0
                cx, cy = nx, ny
                track.append([int(fr.step), round(cx, 4), round(cy, 4), nm, stable])
        settled = stable >= 2 or not ref_cups
        if settled and phase == "travel" and ball and mode == "tracked":
            # Ball model: the ball stays where it was seen unless the sliding cup's inner wall drags it; aim over it.
            bx, by = ball[0] + BALL_TO_CUP[0], ball[1] + BALL_TO_CUP[1]
            dvx, dvy = bx - cx, by - cy
            dn = math.hypot(dvx, dvy)
            k = 1.0 if dn <= BALL_IN_CUP else BALL_IN_CUP / dn
            cup_xy = [round(cx, 4), round(cy, 4)]
            mode = "ball_model" if dn > SLIDE_THRESH else "static"
            cx, cy = cx + k * dvx, cy + k * dvy
        x, y = cx, cy
        exy = math.hypot(x - tcp[0], y - tcp[1])
        if phase == "travel" and ((settled and exy < XY_ARRIVE_TOL and abs(tcp[2] - HOVER_Z) < 0.012) or counts.get("travel", 0) >= TRAVEL_MAX_STEPS):
            phase = "settle"
            hold = 0
        elif phase in ("settle", "contact"):
            hold += 1
            if hold >= HOLD_STEPS:
                phase = "contact" if phase == "settle" else "settle"
                hold = 0
        if phase == "travel":
            zt = max(SAFE_Z, min(HOVER_Z + 0.6 * exy, 0.28)) if exy > XY_ALIGN_TOL else HOVER_Z
            if not settled:
                zt = max(zt, UNSETTLED_Z)
        elif phase == "settle":
            zt = ztop + SETTLE_DZ
        else:
            zt = ztop + CONTACT_DZ - 0.003
        vel = [0.0, 0.0, 0.0] if prev_tcp is None else [float(tcp[k] - prev_tcp[k]) for k in range(3)]
        p = _pos_cmd([x - tcp[0], y - tcp[1], zt - tcp[2]], vel)
        r, rerr = _rot_cmd(tcp, TARGET_YAW)
        if phase in ("settle", "contact"):
            p = [float(np.clip(v, -HOLD_CMD_MAX, HOLD_CMD_MAX)) for v in p]
            r = [float(np.clip(v * HOLD_ROT_SCALE, -HOLD_CMD_MAX, HOLD_CMD_MAX)) for v in r]
        prev_tcp = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
        counts[phase] = counts.get(phase, 0) + 1
        log.append([int(fr.step), phase[:2]] + [round(float(v), 3) for v in [tcp[0] - cx, tcp[1] - cy, tcp[2], rerr]])
        robot.step(p[0], p[1], p[2], r[0], r[1], r[2], GRIPPER)
    return {"counts": counts, "mode": mode, "cup_xy": cup_xy, "target": [round(cx, 4), round(cy, 4)], "track": track[-6:],
            "settled": bool(stable >= 2), "tcp": _summ(robot.now().tcp), "trace": log[-10:]}
