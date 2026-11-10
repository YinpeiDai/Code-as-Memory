"""put_back_block policy: block -> centre of the four mats, press button, block -> original spot.

Memory persists the layout seen at reset (original block spot, centre, button), the
phase and retry counters; every later decision re-perceives the head RGB-D camera to
verify the previous phase before moving on."""

BLOCK_H_DEFAULT = 0.042      # block top above table measured at reset (0.7826 - 0.741)
GRASP_DZ = 0.120             # ee z above block top at grasp (expert, 25/25 demos; held top = ee_z - 0.120)
APPROACH_DZ = 0.080          # pre-grasp height above grasp pose
LIFT_DZ = 0.100              # lift above grasp pose
RELEASE_GAP = 0.010          # block bottom above table when released (expert final release)
PRESS_APPROACH_DZ = 0.184    # left ee above button top before pressing (expert 0.964 - 0.780)
PRESS_DZ = 0.146             # left ee above button top at press (expert 0.926 - 0.780)
PERCEPT_DX = 0.003           # ee x = perceived x + 0.003 (perceived block xy vs ee xy while held, all demos)
PERCEPT_DY = 0.000
QUAT_DOWN = {"right": [0.5, -0.5, 0.5, 0.5], "left": [0.7071068, 0.0, 0.7071068, 0.0]}
MAX_ATTEMPTS = 3

FIELDS = ("phase", "origin", "center", "button", "block_arm", "press_arm", "home_left",
          "home_right", "block_pos", "attempts", "press_extra", "pressed", "history")


def _mem_out(memory, **upd):
    out = {}
    for k in FIELDS:
        out[k] = memory.get(k) if isinstance(memory, dict) else None
    for k, v in upd.items():
        out[k] = v
    return out


def _pose_of(frame, arm):
    p = np.asarray(getattr(frame, arm + "_ee_pose"), dtype=float).tolist()
    return [p[:3], p[3:7]]


def _log(memory, text):
    h = memory.get("history") or ""
    h = (h + " | " + text) if h else text
    return h[-1500:]


def _find_block(frame, near=None, exclude=None, min_top=None, max_top=None, radius=None):
    """Red object (n >= 60 points, not the button) closest to `near`."""
    sc = _scene(frame)
    cands = []
    for r in sc["reds"]:
        if r["n"] < 60 or r["ntop"] < 8:
            continue
        if exclude is not None and _dist2(r["xy"], exclude) < 0.05 and r["top"] < 0.80:
            continue
        if min_top is not None and r["top"] < min_top:
            continue
        if max_top is not None and r["top"] > max_top:
            continue
        cands.append(r)
    if not cands:
        return None
    if near is None:
        return max(cands, key=lambda r: r["n"])
    best = min(cands, key=lambda r: _dist2(r["xy"], near))
    if radius is not None and _dist2(best["xy"], near) > radius:
        return None
    return best


def _parse_last_ok(last):
    if not isinstance(last, str):
        return None
    if '"ok": false' in last or "error" in last or "failed" in last:
        return False
    if '"ok": true' in last:
        return True
    return None


def perceive_decide(obs, goal, memory, last):
    """Phase machine: init -> pick1 -> place_center -> press -> pick2 -> place_back -> done.

    Each phase's act is verified from the head camera (block held / block at target /
    button depressed) before advancing; failures retry from the re-perceived block."""
    memory = dict(memory or {})
    cur = obs.current
    phase = memory.get("phase") or "init"
    attempts = int(memory.get("attempts") or 0)

    if phase == "init":
        lay = _initial_layout(cur)
        if not lay.get("ok"):
            m = _mem_out(memory, phase="init", attempts=attempts + 1,
                         history=_log(memory, "layout not found n_reds=%d n_mats=%d" % (lay.get("n_reds", 0), lay.get("n_mats", 0))))
            return {"memory": m, "act": {"tool": "act_hold", "args": {}}}
        b = lay["block"]
        btn = lay["button"]
        if btn is None:
            btn = {"xy": [-0.252, -0.101], "top": 0.780}   # demo prior, used only if button unseen
        press_arm = "left" if btn["xy"][0] < b["xy"][0] else "right"
        block_arm = "right" if press_arm == "left" else "left"
        origin = [b["xy"][0], b["xy"][1], b["top"]]
        m = _mem_out(memory, phase="pick1", origin=origin, center=list(lay["center"]),
                     button=[btn["xy"][0], btn["xy"][1], btn["top"]], block_arm=block_arm,
                     press_arm=press_arm, home_left=_pose_of(cur, "left"), home_right=_pose_of(cur, "right"),
                     block_pos=origin, attempts=0, press_extra=0.0, pressed=False,
                     history=_log(memory, "layout block=(%.3f,%.3f,%.3f) centre=(%.3f,%.3f) button=(%.3f,%.3f)" % (
                         origin[0], origin[1], origin[2], lay["center"][0], lay["center"][1], btn["xy"][0], btn["xy"][1])))
        return {"memory": m, "act": _pick_act(m)}

    origin = memory["origin"]
    center = memory["center"]
    button = memory["button"]
    arm = memory["block_arm"]
    top0 = origin[2]
    held_min_top = top0 + 0.035

    if phase in ("pick1", "pick2"):
        # last act was a pick; verify block lifted near the gripper
        ee = _pose_of(cur, arm)[0]
        held = _find_block(cur, near=[ee[0] - PERCEPT_DX, ee[1] - PERCEPT_DY], exclude=button[:2],
                           min_top=held_min_top, radius=0.04)
        if held is not None:
            nxt = "place_center" if phase == "pick1" else "place_back"
            m = _mem_out(memory, phase=nxt, attempts=0,
                         history=_log(memory, "%s ok top=%.3f" % (phase, held["top"])))
            tgt = center if nxt == "place_center" else origin
            return {"memory": m, "act": _place_act(m, tgt, final=(nxt == "place_back"))}
        # failed: re-locate block on the table and retry
        ref = memory.get("block_pos") or origin
        blk = _find_block(cur, near=ref[:2], exclude=button[:2], max_top=top0 + 0.02)
        if attempts + 1 >= MAX_ATTEMPTS * 2:
            m = _mem_out(memory, phase="done", history=_log(memory, phase + " gave up"))
            return {"memory": m, "act": {"tool": "act_hold", "args": {}}}
        bp = [blk["xy"][0], blk["xy"][1], blk["top"]] if blk is not None else ref
        m = _mem_out(memory, phase=phase, block_pos=bp, attempts=attempts + 1,
                     history=_log(memory, "%s retry block=(%.3f,%.3f) seen=%s" % (phase, bp[0], bp[1], blk is not None)))
        return {"memory": m, "act": _pick_act(m)}

    if phase in ("place_center", "place_back"):
        tgt = center if phase == "place_center" else origin
        blk = _find_block(cur, near=tgt[:2], exclude=button[:2], max_top=top0 + 0.02)
        ee = _pose_of(cur, arm)[0]
        still_held = _find_block(cur, near=[ee[0] - PERCEPT_DX, ee[1]], exclude=button[:2],
                                 min_top=held_min_top, radius=0.04)
        tol = 0.02 if phase == "place_center" else 0.015
        if still_held is not None and attempts + 1 < MAX_ATTEMPTS:
            m = _mem_out(memory, phase=phase, attempts=attempts + 1,
                         history=_log(memory, phase + " block still held, re-release"))
            return {"memory": m, "act": _place_act(m, tgt, final=(phase == "place_back"))}
        if blk is not None and _dist2(blk["xy"], tgt[:2]) <= tol:
            bp = [blk["xy"][0], blk["xy"][1], blk["top"]]
            if phase == "place_center":
                m = _mem_out(memory, phase="press", block_pos=bp, attempts=0,
                             history=_log(memory, "centre ok (%.3f,%.3f)" % (bp[0], bp[1])))
                return {"memory": m, "act": _press_act(m)}
            m = _mem_out(memory, phase="done", block_pos=bp, attempts=0,
                         history=_log(memory, "back ok (%.3f,%.3f), not ended -> retreat" % (bp[0], bp[1])))
            return {"memory": m, "act": {"tool": "act_home", "args": {"arm": arm, "gripper": 1.0}}}
        if blk is None:
            # could be occluded by the hovering gripper; search anywhere on the table
            blk = _find_block(cur, near=tgt[:2], exclude=button[:2], max_top=top0 + 0.02)
        if blk is None:
            if phase == "place_center":
                m = _mem_out(memory, phase="press", block_pos=[center[0], center[1], top0], attempts=0,
                             history=_log(memory, "centre unseen, assume placed"))
                return {"memory": m, "act": _press_act(m)}
            m = _mem_out(memory, phase="done", history=_log(memory, "back unseen"))
            return {"memory": m, "act": {"tool": "act_home", "args": {"arm": arm, "gripper": 1.0}}}
        # displaced: re-pick and place again
        bp = [blk["xy"][0], blk["xy"][1], blk["top"]]
        nxt = "pick1" if phase == "place_center" else "pick2"
        if attempts + 1 >= MAX_ATTEMPTS:
            if phase == "place_center":
                m = _mem_out(memory, phase="press", block_pos=bp, attempts=0,
                             history=_log(memory, "centre off by %.3f, pressing anyway" % _dist2(bp, tgt)))
                return {"memory": m, "act": _press_act(m)}
            m = _mem_out(memory, phase="done", history=_log(memory, "back off by %.3f" % _dist2(bp, tgt)))
            return {"memory": m, "act": {"tool": "act_home", "args": {"arm": arm, "gripper": 1.0}}}
        m = _mem_out(memory, phase=nxt, block_pos=bp, attempts=attempts + 1,
                     history=_log(memory, "%s displaced (%.3f,%.3f) -> repick" % (phase, bp[0], bp[1])))
        return {"memory": m, "act": _pick_act(m)}

    if phase == "press":
        base = button[2]
        tops = []
        for f in obs.frames:
            sc = _scene(f)
            for r in sc["reds"]:
                if r["n"] >= 60 and _dist2(r["xy"], button[:2]) < 0.03 and r["top"] < base + 0.01:
                    tops.append(r["top"])
        depth = (base - min(tops)) if tops else 0.0
        extra = float(memory.get("press_extra") or 0.0)
        if depth >= 0.003 or attempts + 1 >= MAX_ATTEMPTS:
            blk = _find_block(cur, near=center[:2], exclude=button[:2], max_top=top0 + 0.02)
            bp = [blk["xy"][0], blk["xy"][1], blk["top"]] if blk is not None else (memory.get("block_pos") or [center[0], center[1], top0])
            m = _mem_out(memory, phase="pick2", pressed=depth >= 0.003, block_pos=bp, attempts=0,
                         history=_log(memory, "press depth=%.4f extra=%.3f" % (depth, extra)))
            return {"memory": m, "act": _pick_act(m)}
        m = _mem_out(memory, phase="press", press_extra=extra + 0.008, attempts=attempts + 1,
                     history=_log(memory, "press shallow depth=%.4f, deeper" % depth))
        return {"memory": m, "act": _press_act(m)}

    # done: success normally ends the episode on release. If still running, re-check the
    # block against the original spot and re-place it a limited number of times.
    blk = _find_block(cur, near=origin[:2], exclude=button[:2], max_top=top0 + 0.02)
    if blk is not None and _dist2(blk["xy"], origin[:2]) > 0.015 and attempts < MAX_ATTEMPTS:
        bp = [blk["xy"][0], blk["xy"][1], blk["top"]]
        m = _mem_out(memory, phase="pick2", block_pos=bp, attempts=attempts + 1,
                     history=_log(memory, "done but block off origin (%.3f,%.3f) -> repick" % (bp[0], bp[1])))
        return {"memory": m, "act": _pick_act(m)}
    m = _mem_out(memory, phase="done", attempts=attempts)
    return {"memory": m, "act": {"tool": "act_hold", "args": {}}}


def _pick_act(m):
    bp = m["block_pos"]
    return {"tool": "act_pick", "args": {"arm": m["block_arm"], "x": float(bp[0]) + PERCEPT_DX,
                                         "y": float(bp[1]) + PERCEPT_DY, "top": float(bp[2])}}


def _place_act(m, tgt, final=False):
    top0 = m["origin"][2]
    release_z = top0 + GRASP_DZ + RELEASE_GAP
    return {"tool": "act_place", "args": {"arm": m["block_arm"], "x": float(tgt[0]) + PERCEPT_DX,
                                          "y": float(tgt[1]) + PERCEPT_DY, "release_z": float(release_z),
                                          "retreat_dz": 0.05}}


def _press_act(m):
    b = m["button"]
    home = m["home_left"] if m["press_arm"] == "left" else m["home_right"]
    return {"tool": "act_press", "args": {"arm": m["press_arm"], "x": float(b[0]) + PERCEPT_DX,
                                          "y": float(b[1]) + PERCEPT_DY, "top": float(b[2]),
                                          "extra": float(m.get("press_extra") or 0.0),
                                          "home": [list(home[0]), list(home[1])]}}


# ---------------------------------------------------------------- control

def _step(robot, arm, pos, quat, grip):
    """One native action: `arm` to pos/quat/grip, the other arm holds its current pose."""
    f = robot.now()
    lp = np.asarray(f.left_ee_pose, dtype=float).tolist()
    rp = np.asarray(f.right_ee_pose, dtype=float).tolist()
    lg = float(f.left_gripper)
    rg = float(f.right_gripper)
    tgt = [float(v) for v in pos] + [float(v) for v in quat]
    if arm == "left":
        lp, lg = tgt, float(grip)
    else:
        rp, rg = tgt, float(grip)
    robot.step(*(lp + [lg] + rp + [rg]))


def _try_step(robot, arm, pos, quat, grip, errors, alt_dz=(0.0, 0.02, -0.01, 0.04)):
    """Step with small z fallbacks on planning failure. Returns True if a step succeeded."""
    for dz in alt_dz:
        p = [pos[0], pos[1], pos[2] + dz]
        try:
            _step(robot, arm, p, quat, grip)
            return True
        except Exception as e:  # MotionError (planning failure); EpisodeEnd is not caught
            errors.append("%s z%+.2f: %s" % (arm, dz, str(e)[:80]))
    return False


def _ee(robot, arm):
    return np.asarray(getattr(robot.now(), arm + "_ee_pose"), dtype=float).tolist()


def act_pick(robot, arm, x, y, top):
    """Top-down grasp of the block whose top face centre is at (x, y, top) (ee coordinates):
    approach above with open gripper, descend, close, lift. Returns ok flag and errors."""
    q = QUAT_DOWN[arm]
    gz = top + GRASP_DZ
    errs = []
    ok = _try_step(robot, arm, [x, y, gz + APPROACH_DZ], q, 1.0, errs)
    ok = _try_step(robot, arm, [x, y, gz], q, 1.0, errs, alt_dz=(0.0, 0.005, -0.005)) and ok
    _try_step(robot, arm, [x, y, gz], q, 0.0, errs, alt_dz=(0.0,))
    ok = _try_step(robot, arm, [x, y, gz + LIFT_DZ], q, 0.0, errs) and ok
    return {"ok": bool(ok), "ee": _ee(robot, arm)[:3], "errors": errs[:6]}


def act_place(robot, arm, x, y, release_z, retreat_dz=0.05):
    """Carry the held block above (x, y), lower the ee to release_z, open, and retreat up by retreat_dz."""
    q = QUAT_DOWN[arm]
    errs = []
    cur = _ee(robot, arm)
    hz = max(cur[2], release_z + 0.09)
    ok = _try_step(robot, arm, [x, y, hz], q, 0.0, errs)
    ok = _try_step(robot, arm, [x, y, release_z], q, 0.0, errs, alt_dz=(0.0, 0.005, 0.01)) and ok
    _try_step(robot, arm, [x, y, release_z], q, 1.0, errs, alt_dz=(0.0,))
    _try_step(robot, arm, [x, y, release_z + retreat_dz], q, 1.0, errs)
    return {"ok": bool(ok), "ee": _ee(robot, arm)[:3], "errors": errs[:6]}


def act_press(robot, arm, x, y, top, extra=0.0, home=None):
    """Press the button whose top centre is at (x, y, top) with the closed gripper pointing down,
    then go back up and return the arm to `home` [[x,y,z],[qw,qx,qy,qz]]."""
    q = QUAT_DOWN[arm]
    errs = []
    up = [x, y, top + PRESS_APPROACH_DZ]
    ok = _try_step(robot, arm, up, q, 1.0, errs)
    _try_step(robot, arm, up, q, 0.0, errs, alt_dz=(0.0,))
    ok = _try_step(robot, arm, [x, y, top + PRESS_DZ - extra], q, 0.0, errs, alt_dz=(0.0, -0.004, 0.004)) and ok
    pressed_frame_top = None
    sc = _scene(robot.now())
    for r in sc["reds"]:
        if r["n"] >= 60 and _dist2(r["xy"], [x, y]) < 0.03 and r["top"] < top + 0.01:
            pressed_frame_top = r["top"]
    _try_step(robot, arm, up, q, 0.0, errs)
    if home is not None:
        _try_step(robot, arm, home[0], home[1], 0.0, errs, alt_dz=(0.0, 0.03))
    return {"ok": bool(ok), "button_top_during_press": pressed_frame_top, "errors": errs[:6]}


def act_home(robot, arm, gripper=1.0):
    """Lift `arm` 6 cm (keeping orientation), the episode should already have ended on success."""
    p = _ee(robot, arm)
    errs = []
    _try_step(robot, arm, [p[0], p[1], p[2] + 0.06], p[3:7], gripper, errs)
    return {"ok": not errs, "errors": errs[:4]}


def act_hold(robot):
    """Hold both arms for one native action (observe)."""
    f = robot.now()
    robot.step(*(np.asarray(f.left_ee_pose, dtype=float).tolist() + [float(f.left_gripper)] +
                 np.asarray(f.right_ee_pose, dtype=float).tolist() + [float(f.right_gripper)]))
    return {"observed": True}
