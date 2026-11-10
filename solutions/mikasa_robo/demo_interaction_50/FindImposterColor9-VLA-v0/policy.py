"""FindImposterColor9: remember the 8 cue colours, wait for the 9-cube test layout,
touch (straddle with open fingers, TCP to cube centre) the cube whose colour was absent."""

GRIP_OPEN = 1.0
GRIP_CLOSED = -1.0
PRESS_Z = 0.045          # closed fingertips (~11 mm below TCP) press onto the 4 cm cube top
ALIGN_Z = 0.066          # TCP height for xy alignment (closed fingertips ~0.8-1.1 cm below TCP, cube tops 0.04)
LAND_GOAL_Z = 0.044      # phase-B descent aims ~4-7 mm below the expected resting TCP height (bounded force)
EXPLORE_HOVER_Z = None   # exploration only (R012 used 0.062): hover static without contact first; None = off
LOW_XY_K = 1.5           # xy P gain below 0.075 m (no 1/A amplification: R012 showed +-8 mm y oscillation)
DESC_K = 0.5             # descent-speed schedule gain (per step, per metre above the resting height)
DESC_VMIN = 0.018        # minimum descent per step until the cube top stops the fingertips
B_ENTRY_XY = 0.010       # start the vertical landing once xy error < 1 cm (cube top is 4 cm wide)
LAND_Z_FORCE = 0.053     # TCP below this => treat as resting on the cube top
LAND_Z_MAX = 0.058       # TCP below this and no longer descending => fingertips rest on the cube
PUSH_A = -0.08           # gentle constant push while resting on the cube (static contact)
CUBE_CENTER_Z = 0.02     # cubes are 4 cm, resting on the table (top faces at ~0.04 m)
SAFE_Z = 0.08            # fingertips stay above cube tops until xy is aligned
LAM = 0.8                # fraction of remaining error requested per step
VMAX = 0.045             # m per step (about the speed reached with a saturated command)
A_GAIN = 0.31            # demo fit (pooled axes): v[t+1] = 0.31*0.1*a[t] + 0.31*v[t]  (act_touch)
B_GAIN = 0.31
AX_A = [0.355, 0.20, 0.21]   # per-axis fit on R001-R005 free motion (x responds fast, y/z carry momentum)
AX_B = [0.13, 0.54, 0.49]


def _names():
    return [p[0] for p in PALETTE]


def _cue_from_frames(frames):
    """Colours over leading frames that show cubes (cue phase). The cue always shows 8 cubes (50 demos +
    50 rollouts), so if more than 8 colours appear keep the 8 with the most complete top-face views."""
    best = {}
    for f in frames:
        d = _detect(f)
        if len(d) == 0:
            break
        for k, v in d.items():
            best[k] = max(best.get(k, 0.0), min(float(v["comp"]), 1.0))
    names = sorted(best, key=lambda k: -best[k])
    if len(names) > 8:
        names = names[:8]
    return sorted(names)


def _choose_target(det, cue):
    """Imposter = test colour not in the cue; fallback = palette colour missing from the cue."""
    new = [k for k in det if k not in cue]
    if len(new) == 1:
        return new[0], "test_minus_cue"
    missing = [k for k in _names() if k not in cue]
    cand = [k for k in missing if k in det]
    if len(cand) >= 1:
        if len(new) > 1:
            cand = [k for k in cand if k in new] or cand
        best = max(cand, key=lambda k: det[k]["n"])
        return best, "palette_minus_cue(%d cand)" % len(cand)
    if new:
        best = max(new, key=lambda k: det[k]["n"])
        return best, "new_ambiguous"
    return None, "none"


def perceive_decide(obs, goal, memory, last):
    """Stage machine: cue -> wait for test layout -> touch imposter -> fallbacks."""
    mem = dict(memory)
    for k in ("stage", "cue_colors", "cue_step", "test_step", "test_colors", "imposter",
              "imposter_xy", "target_rule", "fallback", "log"):
        mem.setdefault(k, None)
    stage = mem["stage"] or "start"
    cur = obs.current
    log = mem["log"] or ""

    if stage == "start":
        cue = _cue_from_frames(obs.frames)
        mem["cue_colors"] = cue
        mem["cue_step"] = int(cur.step)
        mem["stage"] = "wait"
        mem["fallback"] = 0
        mem["log"] = log + "cue@%d:%d colours; " % (cur.step, len(cue))
        return {"memory": mem, "act": {"tool": "act_wait_for_test",
                                       "args": {"cue_colors": cue, "max_steps": 18}}}

    if stage == "wait":
        det = _detect(cur)
        cue = mem["cue_colors"] or []
        if len(det) < 5:
            # test layout not seen yet (should not happen): keep waiting
            mem["log"] = log + "wait-again@%d; " % cur.step
            return {"memory": mem, "act": {"tool": "act_wait_for_test",
                                           "args": {"cue_colors": cue, "max_steps": 10}}}
        name, rule = _choose_target(det, cue)
        mem["test_step"] = int(cur.step)
        mem["test_colors"] = sorted(det)
        mem["imposter"] = name
        mem["target_rule"] = rule
        if name is None:
            # imposter colour not visible (e.g. occluded): stay in 'wait', observe one more step, re-detect
            mem["log"] = log + "no-target@%d; " % cur.step
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 1}}}
        xy = det[name]["xy"]
        mem["imposter_xy"] = [float(xy[0]), float(xy[1])]
        mem["stage"] = "touch"
        mem["log"] = log + "test@%d:%d cubes, imposter=%s (%s) xy=(%.3f,%.3f); " % (
            cur.step, len(det), name, rule, xy[0], xy[1])
        return {"memory": mem, "act": {"tool": "act_press", "args": {
            "x": xy[0], "y": xy[1], "color": name, "max_steps": 18, "settle_steps": 3,
            "hover_z": EXPLORE_HOVER_Z}}}

    if stage in ("touch", "fallback"):
        # Reached/held the target without native success: try the next hypothesis.
        fb = int(mem["fallback"] or 0) + 1
        mem["fallback"] = fb
        mem["stage"] = "fallback"
        xy = mem["imposter_xy"]
        mem["log"] = log + "fallback%d@%d; " % (fb, cur.step)
        if fb == 1:
            # R012: a static no-contact hover over the centre also succeeds -> re-centre at hover height
            # (lifts off a slid/edge contact) and hold still; then press again if steps remain.
            return {"memory": mem, "act": {"tool": "act_press", "args": {
                "x": xy[0], "y": xy[1], "color": "", "max_steps": 10, "settle_steps": 2,
                "hover_z": 0.060, "hover_hold": 4}}}
        elif fb == 2:
            # straddle: TCP to cube-centre height with open fingers (R001 success mode)
            args = {"x": xy[0], "y": xy[1], "z": CUBE_CENTER_Z, "color": "",
                    "gripper": GRIP_OPEN, "max_steps": 8, "hold_steps": 2}
        else:
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 3}}}
        return {"memory": mem, "act": {"tool": "act_touch", "args": args}}

    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 3}}}


def act_wait_for_test(robot, cue_colors, max_steps=18):
    """Send zero motion (commands are suppressed during the cue/blank phases) until the
    cubes vanish and then reappear. Returns per-frame cube counts and colours seen."""
    seen_empty = False
    counts = []
    cue = set(cue_colors)
    extra_cue = []
    for i in range(int(max_steps)):
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP_OPEN)
        d = _detect(f)
        counts.append(len(d))
        if len(d) == 0:
            seen_empty = True
            continue
        if not seen_empty:
            new = [k for k in d if k not in cue]
            if len(d) >= 8 and len(new) == 1 and len(cue) == 8:
                # layout changed without a visible blank frame
                return {"status": "test_seen_no_blank", "step": int(f.step), "counts": counts,
                        "colors": sorted(d)}
            extra_cue += [k for k in d if k not in cue and k not in extra_cue]
            continue
        if len(d) >= 5:
            return {"status": "test_seen", "step": int(f.step), "counts": counts,
                    "colors": sorted(d), "extra_cue": extra_cue}
    return {"status": "timeout_waiting", "step": int(robot.now().step), "counts": counts}


def _limit_check(robot, cmd):
    """Live joint-limit preview of one delta command; shrink it if it would violate limits."""
    scale = 1.0
    for _ in range(3):
        c = [v * scale for v in cmd]
        chk = robot.dry_run(c[0], c[1], c[2], c[3], c[4], c[5], mode="delta")
        if chk.get("within_joint_limits", True):
            return c, scale, chk.get("min_joint_margin_rad")
        scale *= 0.5
    return [0.0] * 6, 0.0, None


def act_touch(robot, x, y, z, color="", gripper=1.0, max_steps=16, hold_steps=2, z_tol=0.006):
    """Closed-loop move of the TCP to (x, y, z) keeping orientation; descend below SAFE_Z
    only once xy is aligned. While high, refine xy from the wrist/front view of `color`.
    Ends early when settled within 6 mm for `hold_steps` steps. Returns the measured trace."""
    tgt = np.array([float(x), float(y), float(z)])
    frames = robot.frames()
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    trace = []
    settled = 0
    refined = 0
    min_margin = 9.0
    for i in range(int(max_steps)):
        fr = robot.now()
        p = np.asarray(fr.tcp[:3], dtype=np.float64)
        v = p - prev if i > 0 else np.zeros(3)
        if i == 0 and len(robot.frames()) >= 2:
            v = p - np.asarray(robot.frames()[-2].tcp[:3], dtype=np.float64)
        prev = p
        if color and p[2] > 0.14 and i > 0:
            d = _detect(fr)
            if color in d and d[color]["comp"] > 0.85:
                nxy = np.array(d[color]["xy"])
                if np.linalg.norm(nxy - tgt[:2]) < 0.03:
                    tgt[:2] = 0.5 * tgt[:2] + 0.5 * nxy
                    refined += 1
        exy = float(np.linalg.norm(tgt[:2] - p[:2]))
        zfloor = tgt[2] + 4.0 * max(0.0, exy - 0.005)
        zgoal = max(tgt[2], min(zfloor, SAFE_Z))
        goal = np.array([tgt[0], tgt[1], zgoal])
        err = goal - p
        dist = float(np.linalg.norm(tgt - p))
        speed = float(np.linalg.norm(v))
        trace.append([int(fr.step), round(float(p[0]), 4), round(float(p[1]), 4),
                      round(float(p[2]), 4), round(dist, 4)])
        if exy < 0.006 and abs(tgt[2] - p[2]) < float(z_tol) and speed < 0.004:
            settled += 1
            if settled > int(hold_steps):
                return {"status": "settled", "dist": dist, "trace": trace, "refined": refined,
                        "target": [float(t) for t in tgt], "min_margin": min_margin}
        else:
            settled = 0
        d = np.clip(LAM * err, -VMAX, VMAX)
        u = (d - B_GAIN * v) / A_GAIN
        a = np.clip(u / 0.1, -1.0, 1.0)
        cmd, scale, mm = _limit_check(robot, [float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0])
        if mm is not None:
            min_margin = min(min_margin, float(mm))
        robot.step(cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, float(gripper))
    p = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    return {"status": "max_steps", "dist": float(np.linalg.norm(tgt - p)), "trace": trace,
            "refined": refined, "target": [float(t) for t in tgt], "min_margin": min_margin}


def act_press(robot, x, y, color="", max_steps=18, settle_steps=3, hover_z=None, hover_hold=4):
    """Closed-gripper touch on top of the cube at (x, y).
    A: xy by a per-axis model-based controller (P-only below 0.075 m, 3 mm deadband); z follows a smooth
       descent-speed schedule v = -clip(DESC_K*(z-0.051), DESC_VMIN, 0.042) so the arm never brakes hard
       (hard braking swung the TCP radially by 1-2 cm in R003-R007); descent pauses (no upward command)
       below 0.085 m while the xy error exceeds 12 mm.
    C: once the fingertips are stopped by the cube top (TCP z < LAND_Z_FORCE, or stopped below
       LAND_Z_MAX) keep a gentle constant push with zero xy so the arm becomes static (success appears to
       need contact/proximity AND a static arm).
    Returns if the episode has not ended after `settle_steps` static pressed steps, or after max_steps."""
    tgt = np.array([float(x), float(y)])
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    prevq = np.asarray(robot.now().joints, dtype=np.float64)
    AA = np.array(AX_A)
    BB = np.array(AX_B)
    phase = "A"
    trace = []
    static = 0
    landed_z = None
    v = np.zeros(3)
    rest_z = 0.051
    hover_n = 0
    hover_info = None
    if hover_z is not None:
        rest_z = float(hover_z)
        phase = "H"
    for i in range(int(max_steps)):
        fr = robot.now()
        p = np.asarray(fr.tcp[:3], dtype=np.float64)
        q = np.asarray(fr.joints, dtype=np.float64)
        if i > 0:
            v = p - prev
        qv = float(np.abs(q - prevq).max() * 20.0) if i > 0 else 0.0
        prev = p
        prevq = q
        if phase == "H" and abs(p[2] - rest_z) < 0.006 and _xy_ok(tgt, p):
            hover_n += 1
            if hover_n > int(hover_hold):
                hover_info = {"step": int(fr.step), "z": float(p[2]), "qv": qv}
                phase = "A"
                rest_z = 0.051
        if phase in ("A", "H") and color and p[2] > 0.14 and i > 0:
            d = _detect(fr)
            if color in d and d[color]["comp"] > 0.85:
                nxy = np.array(d[color]["xy"])
                if np.linalg.norm(nxy - tgt) < 0.03:
                    tgt = 0.5 * tgt + 0.5 * nxy
        exy_v = tgt - p[:2]
        exy = float(np.linalg.norm(exy_v))
        if phase == "A" and i > 0 and (p[2] < LAND_Z_FORCE or (p[2] < LAND_Z_MAX and -v[2] < 0.003)):
            phase = "C"
            landed_z = float(p[2])
        trace.append([int(fr.step), phase, round(float(p[0]), 4), round(float(p[1]), 4),
                      round(float(p[2]), 4), round(qv, 3)])
        if phase in ("A", "H"):
            dxy = np.clip(LAM * exy_v, -VMAX, VMAX)
            if p[2] < 0.075:
                uxy = np.clip(LOW_XY_K * exy_v, -0.03, 0.03)
            else:
                uxy = (dxy - BB[:2] * v[:2]) / AA[:2]
            if exy < 0.003:
                uxy = np.zeros(2)
            vz = -float(np.clip(DESC_K * (p[2] - 0.051), DESC_VMIN, 0.042))
            if phase == "H":
                vz = float(np.clip(0.6 * (rest_z - p[2]), -0.042, 0.02))
            if exy > 0.012 and p[2] < 0.085:
                vz = 0.0
            uz = (vz - BB[2] * v[2]) / AA[2]
            if vz == 0.0:
                uz = min(uz, 0.0)
            a = np.clip(np.array([uxy[0], uxy[1], uz]) / 0.1, -1.0, 1.0)
        else:
            a = np.array([0.0, 0.0, PUSH_A])
            if qv < 0.15:
                static += 1
            else:
                static = 0
            if static >= int(settle_steps):
                return {"status": "pressed_static_no_success", "landed_z": landed_z,
                        "trace": trace, "target": [float(t) for t in tgt]}
        cmd, scale, mm = _limit_check(robot, [float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0])
        robot.step(cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, GRIP_CLOSED)
    return {"status": "max_steps", "phase": phase, "landed_z": landed_z, "trace": trace, "hover": hover_info,
            "target": [float(t) for t in tgt]}


def _xy_ok(tgt, p):
    return float(np.linalg.norm(np.asarray(tgt) - np.asarray(p[:2]))) < 0.008


def act_hold(robot, steps=3):
    """Hold the current pose with open fingers for `steps` native steps."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP_OPEN)
    t = robot.now().tcp
    return {"held": int(steps), "tcp": [float(v) for v in t[:3]]}
