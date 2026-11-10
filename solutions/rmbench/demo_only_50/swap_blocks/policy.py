"""swap_blocks policy: perceive two red blocks in three fixed trays (head RGB-D), swap them
through the empty tray with three single-block pick/place moves, then press the button.

All geometry constants were measured from the 50 assigned expert demonstrations
(see task_analysis.md). Grasp centre = ee position + 0.12 m along the ee local +x axis.
"""

TRAYS = [[-0.138, -0.150], [0.002, -0.148], [0.142, -0.151]]
HOME = {
    "left": [-0.2979, -0.3138, 0.942, 0.7, 0.0, 0.0, 0.7141],
    "right": [0.3064, -0.3128, 0.941, 0.7036, 0.0, 0.0, 0.7106],
}
TCP = 0.12
GRASP_CZ = 0.778    # grasp-centre z when closing (expert ee z 0.898)
PRE_CZ = 0.847      # pre-grasp / post-release centre z (expert ee z ~0.967)
CARRY_CZ = 0.865    # transport centre z (expert ~0.835-0.875)
PLACE_CZ = 0.799    # centre z at release (expert ee z 0.918)
PREF_YAW = {"left": 150.0, "right": 213.5}
BUTTON = [-0.001, -0.279]
BUTTON_EE_Z0 = 0.964
BUTTON_QUAT = [0.70710678, 0.0, 0.70710678, 0.0]
CENTER_BIAS = [0.002, 0.001]   # head-camera top-face centre vs expert grasp centre
TOP_ZMIN = 0.762
TOP_ZMAX = 0.790
OCC_MIN = 40


# ---------------------------------------------------------------- geometry helpers
def _mat2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=float)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def _grasp_frame(yaw_deg, tilt_deg):
    """Top-down gripper: local x (approach) down, local y horizontal at yaw_deg, then the
    approach axis tilted by tilt_deg toward local +z (the expert's far-reach pose)."""
    g = math.radians(yaw_deg)
    a = math.radians(tilt_deg)
    y = np.array([math.cos(g), math.sin(g), 0.0])
    x0 = np.array([0.0, 0.0, -1.0])
    z0 = np.cross(x0, y)
    x = math.cos(a) * x0 + math.sin(a) * z0
    z = np.cross(x, y)
    R = np.stack([x, y, z], 1)
    return R, _mat2quat(R)


def _ee_from_center(center, R):
    c = np.array(center, dtype=float)
    return c - TCP * R[:, 0]


def _arm_yaw(arm, block_yaw):
    """Choose block_yaw + k*90 closest (circularly) to the arm's preferred gripper yaw."""
    best = None
    for k in range(4):
        cand = (block_yaw + 90.0 * k) % 360.0
        d = abs((cand - PREF_YAW[arm] + 180.0) % 360.0 - 180.0)
        if best is None or d < best[0]:
            best = (d, cand)
    return best[1]


# ---------------------------------------------------------------- perception helpers
def _head_points(frame, mask):
    depth = np.asarray(frame.head_depth)
    K = np.asarray(frame.head_K, dtype=float)
    E = np.asarray(frame.head_extrinsic, dtype=float)
    v, u = np.nonzero(mask)
    d = depth[v, u].astype(float) / 1000.0
    ok = d > 0
    u, v, d = u[ok], v[ok], d[ok]
    if len(d) == 0:
        return np.zeros((0, 3))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _red_points(frame):
    rgb = np.asarray(frame.head_rgb).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 120) & (g < 70) & (b < 70)
    P = _head_points(frame, mask)
    if len(P) == 0:
        return P
    keep = (P[:, 1] > -0.23) & (P[:, 1] < 0.10) & (np.abs(P[:, 0]) < 0.40)
    return P[keep]


def _rect_fit(xy):
    """Minimum-area rectangle by brute-force angle search; returns (yaw_deg in [0,90), centre)."""
    c = xy.mean(0)
    X = xy[:, 0] - c[0]
    Y = xy[:, 1] - c[1]
    ang = np.radians(np.arange(0.0, 90.0, 1.0))
    ca = np.cos(ang)[:, None]
    sa = np.sin(ang)[:, None]
    U = X[None, :] * ca + Y[None, :] * sa
    V = -X[None, :] * sa + Y[None, :] * ca
    area = (U.max(1) - U.min(1)) * (V.max(1) - V.min(1))
    i = int(np.argmin(area))
    cu = (U[i].max() + U[i].min()) / 2.0
    cv = (V[i].max() + V[i].min()) / 2.0
    a = float(ang[i])
    cx = c[0] + cu * math.cos(a) - cv * math.sin(a)
    cy = c[1] + cu * math.sin(a) + cv * math.cos(a)
    return float(np.degrees(a)), [float(cx), float(cy)]


def _block_info(pts):
    if len(pts) < 8:
        return None
    yaw, ctr = _rect_fit(pts[:, :2])
    return {"x": ctr[0] + CENTER_BIAS[0], "y": ctr[1] + CENTER_BIAS[1], "yaw": yaw, "n": int(len(pts))}


def _scene(frame):
    """Tray occupancy, block centres/yaws, stray blocks and lifted red points."""
    P = _red_points(frame)
    trays = []
    used = np.zeros(len(P), dtype=bool)
    top = (P[:, 2] > TOP_ZMIN) & (P[:, 2] < TOP_ZMAX) if len(P) else np.zeros(0, dtype=bool)
    for tx, ty in TRAYS:
        if len(P):
            sel = top & (np.abs(P[:, 0] - tx) < 0.065) & (np.abs(P[:, 1] - ty) < 0.06)
        else:
            sel = np.zeros(0, dtype=bool)
        used = used | sel if len(P) else used
        info = _block_info(P[sel]) if len(P) else None
        n = int(sel.sum()) if len(P) else 0
        trays.append({"n": n, "occ": n >= OCC_MIN, "block": info})
    strays = []
    if len(P):
        rest = P[top & (~used)]
        while len(rest) >= OCC_MIN:
            d2 = ((rest[:, None, :2] - rest[None, :, :2]) ** 2).sum(2) if len(rest) < 1500 else None
            if d2 is None:
                seed = rest[0]
            else:
                seed = rest[int(np.argmax((d2 < 0.02 ** 2).sum(1)))]
            near = ((rest[:, :2] - seed[:2]) ** 2).sum(1) < 0.03 ** 2
            if near.sum() >= OCC_MIN:
                info = _block_info(rest[near])
                if info is not None:
                    strays.append(info)
            rest = rest[~near]
            if len(strays) >= 3:
                break
    lifted = P[(P[:, 2] > 0.795) & (P[:, 2] < 0.93)] if len(P) else P
    return {"trays": trays, "strays": strays, "lifted": lifted}


def _wrist_hold(frame, arm):
    """Red pixels in the arm's wrist camera and their median depth (mm). A held block
    fills ~8500 px at ~75 mm in the demonstrations; a block left on the tray is >130 mm away."""
    rgb = np.asarray(getattr(frame, arm + "_wrist_rgb")).astype(int)
    d = np.asarray(getattr(frame, arm + "_wrist_depth"))
    m = (rgb[..., 0] > 120) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70) & (d > 0)
    n = int(m.sum())
    med = float(np.median(d[m])) if n > 0 else -1.0
    return n, med


def _last_number(last, key):
    """Read a numeric field `"key": value` from the previous act result text, else None."""
    if not isinstance(last, str):
        return None
    tag = '"' + key + '": '
    i = last.find(tag)
    if i < 0:
        return None
    j = i + len(tag)
    k = j
    while k < len(last) and last[k] in "0123456789.-eE+":
        k += 1
    try:
        return float(last[j:k])
    except ValueError:
        return None


# ---------------------------------------------------------------- planning helpers
def _plan_for(occupied):
    """Swap via the empty tray e, in the expert's order; arm: right iff tray 2 involved."""
    e = [i for i in range(3) if i not in occupied][0]
    moves = [((e + 1) % 3, e), ((e + 2) % 3, (e + 1) % 3), (e, (e + 2) % 3)]
    return [str(s) + ">" + str(d) for s, d in moves]


def _parse_move(label):
    s, d = label.split(">")
    return int(s), int(d)


def _arm_for(src, dst):
    return "right" if (src == 2 or dst == 2) else "left"


def _tilt_for(arm, x):
    # expert: right arm reaching the left tray used a 5.7 deg tilted approach
    return 5.7 if (arm == "right" and x < -0.07) else (5.7 if (arm == "left" and x > 0.07) else 0.0)


def _pick_action(arm, info):
    x = float(info["x"])
    y = float(info["y"])
    yaw = _arm_yaw(arm, float(info["yaw"]))
    return {"tool": "act_pick", "args": {"arm": arm, "x": x, "y": y, "yaw": yaw, "tilt": _tilt_for(arm, x)}}


def _mem(m):
    keys = ["phase", "plan", "move_idx", "retries", "arm", "grip_yaw", "grip_tilt", "src_xy",
            "decisions", "press_tries", "notes"]
    return {k: m.get(k) for k in keys}


def _decide_pick(mem, sc):
    """Choose the pick for the current move (or advance when it is already done)."""
    plan = mem["plan"]
    while mem["move_idx"] < len(plan):
        src, dst = _parse_move(plan[mem["move_idx"]])
        arm = _arm_for(src, dst)
        if mem["retries"] >= 2 and mem["retries"] % 2 == 0:
            # repeated failures with the expert's arm: try the other arm
            arm = "left" if arm == "right" else "right"
        st = sc["trays"][src]
        if st["occ"] and st["block"] is not None:
            mem["phase"] = "check_pick"
            mem["src_xy"] = [st["block"]["x"], st["block"]["y"], float(src)]
            mem["notes"] = "pick tray %d -> %d with %s arm" % (src, dst, arm)
            act = _pick_action(arm, st["block"])
            mem["arm"] = arm
            mem["grip_yaw"] = act["args"]["yaw"]
            mem["grip_tilt"] = act["args"]["tilt"]
            return act
        if len(sc["strays"]) > 0:
            # a block lying off the trays: it must be the block of this move
            b = sc["strays"][0]
            mem["phase"] = "check_pick"
            mem["src_xy"] = [b["x"], b["y"], -1.0]
            arm = "right" if (b["x"] > 0.07 or dst == 2) else "left"
            mem["notes"] = "pick stray block at %.3f,%.3f with %s arm" % (b["x"], b["y"], arm)
            act = _pick_action(arm, b)
            mem["arm"] = arm
            mem["grip_yaw"] = act["args"]["yaw"]
            mem["grip_tilt"] = act["args"]["tilt"]
            return act
        if sc["trays"][dst]["occ"]:
            mem["move_idx"] = mem["move_idx"] + 1
            mem["retries"] = 0
            continue
        # nothing visible at source nor destination: re-observe a few times then skip
        mem["retries"] = mem["retries"] + 1
        mem["notes"] = "no block found for move %s" % plan[mem["move_idx"]]
        if mem["retries"] > 3:
            mem["move_idx"] = mem["move_idx"] + 1
            mem["retries"] = 0
            continue
        mem["phase"] = "pick"
        return {"tool": "act_reset_arms", "args": {}}
    mem["phase"] = "press"
    mem["notes"] = "all moves done; pressing button"
    return {"tool": "act_press", "args": {"depth_index": 0}}


# ---------------------------------------------------------------- policy
def perceive_decide(obs, goal, memory, last):
    """Swap the two blocks through the empty tray (3 moves), verifying each grasp/place
    with the head RGB-D camera, then press the button with the left arm."""
    try:
        return _decide(obs, memory, last)
    except Exception as e:
        mem = _mem(memory if isinstance(memory, dict) else {})
        mem["notes"] = "decision error: " + repr(e)[:200]
        if mem["phase"] in ("check_pick", "check_place"):
            mem["phase"] = "pick"
        return {"memory": mem, "act": {"tool": "act_reset_arms", "args": {}}}


def _decide(obs, memory, last):
    mem = _mem(memory if isinstance(memory, dict) else {})
    if mem["decisions"] is None:
        mem["decisions"] = 0
    mem["decisions"] = int(mem["decisions"]) + 1
    if mem["retries"] is None:
        mem["retries"] = 0
    if mem["press_tries"] is None:
        mem["press_tries"] = 0
    if mem["move_idx"] is None:
        mem["move_idx"] = 0
    frame = obs.current
    sc = _scene(frame)
    occ_n = [t["n"] for t in sc["trays"]]

    if mem["phase"] is None or mem["plan"] is None:
        ranked = sorted(range(3), key=lambda i: -occ_n[i])
        occupied = [i for i in ranked[:2] if occ_n[i] >= 15]
        if len(occupied) < 2 and mem["retries"] < 2:
            mem["retries"] = mem["retries"] + 1
            mem["phase"] = None
            mem["notes"] = "init: occupancy unclear %s" % occ_n
            return {"memory": mem, "act": {"tool": "act_reset_arms", "args": {}}}
        if len(occupied) < 2:
            occupied = sorted(ranked[:2])
        mem["plan"] = _plan_for(sorted(occupied))
        mem["move_idx"] = 0
        mem["retries"] = 0
        mem["phase"] = "pick"

    if mem["phase"] == "pick":
        act = _decide_pick(mem, sc)
        return {"memory": mem, "act": act}

    if mem["phase"] == "check_pick":
        src, dst = _parse_move(mem["plan"][mem["move_idx"]])
        arm = mem["arm"] if mem["arm"] in ("left", "right") else _arm_for(src, dst)
        sx = mem["src_xy"]
        src_tray = int(round(sx[2])) if sx is not None else src
        L = sc["lifted"]
        near = 0
        if len(L) and sx is not None:
            near = int((((L[:, 0] - sx[0]) ** 2 + (L[:, 1] - sx[1]) ** 2) < 0.08 ** 2).sum())
        still = occ_n[src_tray] >= OCC_MIN if src_tray >= 0 else False
        ok_last = not (isinstance(last, str) and last.startswith("act_pick") and '"ok": false' in last)
        wn, wd = -1, -1.0
        if hasattr(frame, arm + "_wrist_rgb") and hasattr(frame, arm + "_wrist_depth"):
            wn, wd = _wrist_hold(frame, arm)
        held_wrist = wn >= 3000 and 0 < wd < 110
        if ok_last and (not still) and (held_wrist or near >= 10 or wn < 0):
            used_yaw = _last_number(last, "yaw") if last is not None and str(last).startswith("act_pick") else None
            used_tilt = _last_number(last, "tilt") if last is not None and str(last).startswith("act_pick") else None
            if used_yaw is not None:
                mem["grip_yaw"] = used_yaw
            if used_tilt is not None:
                mem["grip_tilt"] = used_tilt
            mem["phase"] = "check_place"
            tx, ty = TRAYS[dst]
            mem["notes"] = "lifted (head %d pts, wrist %d px @%.0f mm); place in tray %d" % (near, wn, wd, dst)
            return {"memory": mem, "act": {"tool": "act_place", "args": {
                "arm": arm, "x": tx, "y": ty, "yaw": float(mem["grip_yaw"]),
                "tilt": float(mem["grip_tilt"])}}}
        mem["retries"] = mem["retries"] + 1
        mem["phase"] = "pick"
        mem["notes"] = "grasp check failed (still=%s, head=%d, wrist=%d@%.0f); retry %d" % (still, near, wn, wd, mem["retries"])
        return {"memory": mem, "act": {"tool": "act_abort", "args": {"arm": arm}}}

    if mem["phase"] == "check_place":
        src, dst = _parse_move(mem["plan"][mem["move_idx"]])
        if sc["trays"][dst]["occ"]:
            mem["move_idx"] = mem["move_idx"] + 1
            mem["retries"] = 0
        else:
            mem["retries"] = mem["retries"] + 1
        mem["phase"] = "pick"
        act = _decide_pick(mem, sc)
        return {"memory": mem, "act": act}

    if mem["phase"] == "press":
        mem["press_tries"] = mem["press_tries"] + 1
        k = mem["press_tries"]
        if k <= 3:
            mem["notes"] = "press retry %d" % k
            return {"memory": mem, "act": {"tool": "act_press", "args": {"depth_index": k}}}
        mem["phase"] = "done"

    mem["notes"] = "done; holding"
    return {"memory": mem, "act": {"tool": "act_reset_arms", "args": {}}}


# ---------------------------------------------------------------- action helpers
def _pose_of(frame, arm):
    p = np.asarray(getattr(frame, arm + "_ee_pose"), dtype=float)
    g = float(np.asarray(getattr(frame, arm + "_gripper")))
    return [float(v) for v in p], g


def _step_arm(robot, arm, pose7, grip):
    """One native step moving `arm` to pose7 with gripper `grip`, holding the other arm.
    Returns True on success, False on a MotionError (unplannable target)."""
    now = robot.now()
    other = "right" if arm == "left" else "left"
    op, og = _pose_of(now, other)
    a = [float(v) for v in pose7] + [float(grip)]
    b = op + [og]
    vals = a + b if arm == "left" else b + a
    try:
        robot.step(*vals)
    except Exception as e:
        if "MotionError" in repr(e):
            return False
        raise
    return True


def _pose_at(center, R, q):
    ee = _ee_from_center(center, R)
    return [float(ee[0]), float(ee[1]), float(ee[2])] + [float(v) for v in q]


def act_pick(robot, arm, x, y, yaw, tilt=0.0):
    """Top-down grasp of a block whose centre is (x, y) with gripper yaw `yaw` (deg) and
    approach tilt `tilt` (deg): pre-grasp, descend, close, lift to carry height.
    Tries alternative orientations when a target cannot be planned."""
    cands = [(yaw, tilt), (yaw, 5.7 if tilt == 0 else 0.0), (yaw + 90.0, tilt), (yaw - 90.0, tilt),
             (yaw, 11.0), (yaw + 90.0, 5.7), (yaw - 90.0, 5.7)]
    log = []
    for gy, tl in cands:
        R, q = _grasp_frame(gy, tl)
        if not _step_arm(robot, arm, _pose_at([x, y, PRE_CZ], R, q), 1.0):
            log.append("pre fail %.0f/%.1f" % (gy, tl))
            continue
        if not _step_arm(robot, arm, _pose_at([x, y, GRASP_CZ], R, q), 1.0):
            log.append("descend fail %.0f/%.1f" % (gy, tl))
            continue
        _step_arm(robot, arm, _pose_at([x, y, GRASP_CZ], R, q), 0.0)
        lifted = _step_arm(robot, arm, _pose_at([x, y, CARRY_CZ], R, q), 0.0)
        if not lifted:
            lifted = _step_arm(robot, arm, _pose_at([x, y, PRE_CZ - 0.02], R, q), 0.0)
        p, g = _pose_of(robot.now(), arm)
        return {"ok": True, "yaw": gy, "tilt": tl, "lifted_cmd": lifted, "ee": p, "log": log}
    return {"ok": False, "log": log}


def act_place(robot, arm, x, y, yaw, tilt=0.0):
    """Carry the held block to a tray centre (x, y), lower, open, lift and return home."""
    log = []
    p, g = _pose_of(robot.now(), arm)
    R, q = _grasp_frame(yaw, tilt)
    cur_c = np.array(p[:3]) + TCP * R[:, 0]
    if abs(cur_c[0] - x) > 0.2:
        mid = [(cur_c[0] + x) / 2.0, (cur_c[1] + y) / 2.0, CARRY_CZ + 0.02]
        if not _step_arm(robot, arm, _pose_at(mid, R, q), 0.0):
            log.append("mid fail")
    if not _step_arm(robot, arm, _pose_at([x, y, CARRY_CZ], R, q), 0.0):
        log.append("above fail")
        alt = None
        for gy, tl in [(yaw, 5.7 if tilt == 0 else 0.0), (yaw, 11.0)]:
            R2, q2 = _grasp_frame(gy, tl)
            if _step_arm(robot, arm, _pose_at([x, y, CARRY_CZ], R2, q2), 0.0):
                alt = (R2, q2)
                break
        if alt is not None:
            R, q = alt
        else:
            log.append("place unreachable")
    if not _step_arm(robot, arm, _pose_at([x, y, PLACE_CZ], R, q), 0.0):
        log.append("lower fail")
    _step_arm(robot, arm, _pose_at([x, y, PLACE_CZ], R, q), 1.0)
    _step_arm(robot, arm, _pose_at([x, y, PRE_CZ + 0.01], R, q), 1.0)
    home_ok = _step_arm(robot, arm, HOME[arm], 1.0)
    return {"ok": True, "home": home_ok, "log": log}


def act_abort(robot, arm):
    """Open the gripper, lift straight up and return the arm home."""
    p, g = _pose_of(robot.now(), arm)
    _step_arm(robot, arm, p, 1.0)
    up = list(p)
    up[2] = max(up[2], PRE_CZ + TCP + 0.02)
    _step_arm(robot, arm, up, 1.0)
    ok = _step_arm(robot, arm, HOME[arm], 1.0)
    return {"ok": ok}


def act_reset_arms(robot):
    """Send both arms to their home poses with open grippers (also used as an observe step)."""
    now = robot.now()
    vals = HOME["left"] + [1.0] + HOME["right"] + [1.0]
    try:
        robot.step(*vals)
    except Exception as e:
        if "MotionError" in repr(e):
            return {"ok": False}
        raise
    return {"ok": True}


def act_press(robot, depth_index=0):
    """Press the button with the closed left gripper pointing down: move above it, then
    descend in small increments (success ends the episode during the press), then lift."""
    above = [BUTTON[0], BUTTON[1], BUTTON_EE_Z0] + BUTTON_QUAT
    ok = _step_arm(robot, "left", above, 0.0)
    extra = 0.004 * max(0, min(3, int(depth_index)))
    zs = [0.945, 0.935, 0.928, 0.922, 0.915, 0.908 - extra, 0.903 - extra]
    for z in zs:
        _step_arm(robot, "left", [BUTTON[0], BUTTON[1], z] + BUTTON_QUAT, 0.0)
    _step_arm(robot, "left", above, 0.0)
    return {"ok": ok, "pressed_to": zs[-1]}
