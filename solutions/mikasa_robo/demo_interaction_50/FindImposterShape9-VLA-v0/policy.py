"""FindImposterShape9 policy: remember the first shape set, wait out the blank, find the new shape in the
second set and touch it from above with closed fingers."""

STAGES = ("await_cue", "await_blank", "await_set2", "touch", "retry")
BLUE_ON_WRIST = 250     # wrist blue pixels meaning "a shape set is displayed"
BLUE_ON_FRONT = 150
STEP_GAIN = 0.4         # measured TCP motion per step ~= 0.4 * 0.1 m * command (demo frames 6-15)
APPROACH_GAIN = 0.7     # fraction of remaining error removed per step (first-order, below 1 to avoid overshoot)
PRESS_DEPTH = 0.012     # press target below the measured top surface (m)
HOVER_CLEAR = 0.008     # minimum clearance above the top while xy is still converging (m)
CONE_SLOPE = 1.0        # allowed height above top per metre of xy error (straight-ish diagonal approach)
XY_TOL = 0.012          # xy error to start the final descent (m); leave it again only beyond XY_EXIT
XY_EXIT = 0.025
PD_K = 18.0             # command per metre of error (fit on R003-R013: v' = 0.30 v + 0.028 u per step)
PD_D = 0.4              # damping on last measured per-step displacement; robust for lag a in [0, 0.5]


def _blue_counts(frame):
    w = int(_blue_mask(frame.wrist_rgb).sum())
    f = int(_blue_mask(frame.front_rgb).sum())
    return w, f


def _shown(frame):
    w, f = _blue_counts(frame)
    return (w >= BLUE_ON_WRIST) or (f >= BLUE_ON_FRONT)


def _fmt_scores(sc):
    return ";".join("%s=%.4f" % (k, sc[k]) for k in SHAPE_CLASSES)


def _parse_scores(text):
    out = {}
    for part in str(text).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = float(v)
    return out


def _empty_memory():
    return {"stage": "await_cue", "cue1_step": -1, "set1_classes": [], "miss_scores": "",
            "blank_step": -1, "set2_step": -1, "imposter_class": "", "target": [], "margin": 0.0,
            "set2_classes": [], "touch_calls": 0, "target_cells": 0, "log": ""}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: await_cue -> (record set 1) -> await_blank -> await_set2 -> (infer imposter) -> touch."""
    mem = _empty_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    cur = obs.current
    stage = mem["stage"]
    step = int(cur.step)
    if stage == "await_cue":
        if _shown(cur):
            s1 = _describe(cur, 8)
            sc = _set1_scores([o["cost"] for o in s1])
            mem["miss_scores"] = _fmt_scores(sc)
            mem["set1_classes"] = [o["best"] for o in s1]
            mem["cue1_step"] = step
            mem["stage"] = "await_blank"
            mem["log"] = "set1 %d objects at step %d" % (len(s1), step)
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
    if stage == "await_blank":
        if not _shown(cur):
            mem["blank_step"] = step
            mem["stage"] = "await_set2"
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
    if stage == "await_set2":
        if not _shown(cur):
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
        s2 = _describe(cur, 9)
        sc = _parse_scores(mem["miss_scores"])
        j, c, margin, tot = _infer_imposter(sc, [o["cost"] for o in s2])
        mem["set2_step"] = step
        mem["set2_classes"] = [o["best"] for o in s2]
        if j is None:
            mem["log"] = "no set2 objects segmented"
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
        o = s2[j]
        mem["imposter_class"] = c
        mem["margin"] = margin
        mem["target"] = [o["center"][0], o["center"][1], o["zt"]]
        mem["target_cells"] = int(o["ncell"])
        mem["stage"] = "touch"
        mem["touch_calls"] = 1
        mem["log"] = "imposter %s at %.3f,%.3f top %.3f margin %.3f" % (c, o["center"][0], o["center"][1], o["zt"], margin)
        t = mem["target"]
        return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": t[0], "y": t[1], "ztop": t[2],
                                                                  "ncell": mem["target_cells"]}}}
    # touch / retry: the previous touch returned without native success; press again (slightly deeper)
    t = mem["target"]
    mem["touch_calls"] = int(mem["touch_calls"]) + 1
    mem["stage"] = "retry"
    extra = 0.004 * min(3, int(mem["touch_calls"]) - 1)
    mem["log"] = "retry %d; last=%s" % (mem["touch_calls"], str(last)[:200])
    return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": t[0], "y": t[1], "ztop": t[2] - extra,
                                                              "ncell": 10000}}}


def act_wait(robot, steps=1):
    """Send `steps` zero-motion commands (gripper closing); used while the cue/blank phases play."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    f = robot.now()
    w, fr = _blue_counts(f)
    return {"step": int(f.step), "wrist_blue": w, "front_blue": fr}


def _refine_target(frame, x, y, ncell):
    """Re-segment the current frame; return (x, y, ztop, ncell) of the blob nearest (x, y) if it is seen better."""
    objs, _ = _objects(frame, None)
    best, bd = None, 0.03
    for o in objs:
        d = float(np.hypot(o["cxy"][0] - x, o["cxy"][1] - y))
        if d < bd:
            best, bd = o, d
    if best is None or len(best["ii"]) <= ncell * 1.15:
        return None
    hm, org, zt = _heightmap(best)
    return float(org[0] + 14 * CELL), float(org[1] + 14 * CELL), float(zt), int(len(best["ii"]))


FINGER_Q = 0.04          # open finger joint (m): inner faces at TCP +- q along the hand y axis
FINGER_T = 0.014         # assumed finger thickness incl. margin (m), outward from the inner face
FINGER_HX = 0.013        # assumed half extent of a finger along the hand x axis (m)
MAX_OFFSET = 0.016       # max TCP shift from the target centre along y (demos succeeded with 2-2.6 cm offsets)


def _finger_plan(frame, x, y):
    """Choose a finger opening q and a TCP y-offset so that both fingers clear neighbouring objects while the
    target stays between them (prefers wide opening, zero offset). Uses the current fused height grid.
    Returns (dy, n_blocked_cells, target_inside, q)."""
    objs, _ = _objects(frame, None)
    if not objs:
        return 0.0, 0, True, 0.04
    tgt = min(objs, key=lambda o: float(np.hypot(o["cxy"][0] - x, o["cxy"][1] - y)))
    if float(np.hypot(tgt["cxy"][0] - x, tgt["cxy"][1] - y)) > 0.03:
        tgt = None
    others = [o for o in objs if o is not tgt]
    ox = np.concatenate([GRID_X0 + (o["ii"] + 0.5) * CELL for o in others]) if others else np.zeros(0)
    oy = np.concatenate([GRID_Y0 + (o["jj"] + 0.5) * CELL for o in others]) if others else np.zeros(0)
    near = np.abs(ox - x) < FINGER_HX
    ox, oy = ox[near], oy[near]
    if tgt is not None:
        ty = GRID_Y0 + (tgt["jj"] + 0.5) * CELL
        tlo, thi = min(float(ty.min()), y - 0.018), max(float(ty.max()), y + 0.018)
    else:
        tlo, thi = y - 0.02, y + 0.02
    best = None
    for q in (0.04, 0.035, 0.03, 0.026):
        for k in range(0, 9):
            for sgn in ((1.0,) if k == 0 else (1.0, -1.0)):
                dy = sgn * k * 0.002
                if abs(dy) > MAX_OFFSET:
                    continue
                c = y + dy
                blk = int(((oy > c + q - 0.002) & (oy < c + q + FINGER_T)).sum()
                          + ((oy < c - q + 0.002) & (oy > c - q - FINGER_T)).sum())
                inside = (tlo > c - q + 0.004) and (thi < c + q - 0.004)
                score = blk + (0 if inside else 1000) + abs(dy) * 10.0 + (0.04 - q) * 5.0
                if best is None or score < best[0]:
                    best = (score, dy, blk, inside, q)
    return best[1], best[2], best[3], best[4]


def act_touch(robot, x, y, ztop, ncell=300, max_steps=24, press_steps=4, gripper=1.0, tcp_floor=0.011,
              sink_frac=0.4):
    """Move the gripper (gripper=+1 open, -1 closed) over (x, y) and lower the TCP to
    max(tcp_floor, sink_frac*ztop) (open fingers straddle the object, TCP inside its volume) or, with a
    closed gripper, press onto the top surface at height ztop.
    Diagonal approach limited by a cone above the target; once over the target (xy error < XY_TOL) it stays in
    press mode (hysteresis 2 cm), stops descending when the motion stalls (contact) and keeps pressing for
    `press_steps` steps. While high above, a better-seen view of the target blob may refine (x, y, ztop).
    Returns measured trajectory facts."""
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=np.float64)
    start = tcp[:3].copy()
    x0, y0, z0 = float(x), float(y), float(ztop)
    chk = None
    try:
        chk = robot.dry_run(float(x), float(y), float(ztop) + 0.02, float(tcp[3]), float(tcp[4]), float(tcp[5]),
                            mode="absolute")
    except Exception as e:  # MotionError etc.: preview is diagnostic only
        chk = {"error": str(e)[:120]}
    plan = [0.0, 0, True, 0.04]
    if gripper > 0:
        dy, nb, ins, q = _finger_plan(f, float(x), float(y))
        plan = [round(float(dy), 4), int(nb), bool(ins), q]
        y = float(y) + dy
        gripper = min(float(gripper), q / 0.02 - 1.0)  # absolute finger target: q = 0.02 * (g + 1)
    reach_info = {}
    if isinstance(chk, dict):
        for k in ("within_joint_limits", "min_joint_margin_rad", "error"):
            if k in chk:
                reach_info[k] = chk[k]
    trace = []
    refines = []
    contact_step = -1
    contact_z = None
    pressed = 0
    dxy0 = float(np.hypot(x - tcp[0], y - tcp[1]))
    press_mode = bool(dxy0 < 0.02 and tcp[2] < ztop + 0.05)
    prev_p = tcp[:3].copy()
    prev_z = tcp[2]
    expect_dz = 0.0
    for i in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=np.float64)
        p = tcp[:3]
        dz_meas = p[2] - prev_z
        prev_z = p[2]
        vel = p - prev_p if i > 0 else np.zeros(3)
        prev_p = p.copy()
        if contact_step < 0 and not press_mode and p[2] > ztop + 0.06 and i > 0:
            r = _refine_target(f, x, y, ncell)
            if r is not None:
                x, y, ztop, ncell = r
                refines.append([int(f.step), round(x, 4), round(y, 4), round(ztop, 4), ncell])
        exy = np.array([x - p[0], y - p[1]])
        dxy = float(np.linalg.norm(exy))
        # contact: commanded noticeable descent near the top, but the TCP barely moved
        if contact_step < 0 and i > 0 and expect_dz < -0.003 and dz_meas > 0.3 * expect_dz and p[2] < ztop + 0.03:
            contact_step = int(f.step)
            contact_z = float(p[2])
        if contact_step >= 0:
            pressed += 1
            if gripper < 0 and pressed > press_steps:
                break
        if press_mode and dxy > XY_EXIT:
            press_mode = False
        if not press_mode and dxy <= XY_TOL:
            press_mode = True
        if press_mode or contact_step >= 0:
            zg = ztop - PRESS_DEPTH if gripper < 0 else max(tcp_floor, sink_frac * ztop)
        else:
            zg = ztop + HOVER_CLEAR + CONE_SLOPE * dxy
        ez = zg - p[2]
        k = PD_K
        cx = float(np.clip(PD_K * (exy[0] - PD_D * vel[0]), -1.0, 1.0))
        cy = float(np.clip(PD_K * (exy[1] - PD_D * vel[1]), -1.0, 1.0))
        cz = float(np.clip(PD_K * (ez - PD_D * vel[2]), -1.0, 1.0))
        if p[2] < ztop + 0.03 and cz < -0.4:
            cz = -0.4  # gentle final descent: <= ~1.6 cm per step
        if contact_step >= 0:
            cz = -0.25 if gripper < 0 else float(np.clip(k * ez, -0.5, 0.25))
        expect_dz = STEP_GAIN * 0.1 * cz
        robot.step(cx, cy, cz, 0.0, 0.0, 0.0, float(gripper))
        if len(trace) < 30:
            trace.append([int(f.step), round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4),
                          round(cx, 2), round(cy, 2), round(cz, 2), int(press_mode)])
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=np.float64)
    return {"start": [round(float(v), 4) for v in start], "final_tcp": [round(float(v), 4) for v in tcp],
            "target_in": [round(x0, 4), round(y0, 4), round(z0, 4)],
            "target": [round(float(x), 4), round(float(y), 4), round(float(ztop), 4)], "refines": refines,
            "finger_plan": plan,
            "contact_step": contact_step, "contact_z": contact_z, "pressed_steps": pressed, "reach": reach_info,
            "fingers": [round(float(v), 4) for v in f.fingers], "trace": trace}
