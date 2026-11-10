"""swap_T policy: perceive both T-blocks (head RGB-D + T-template fit), remember their
start poses, pick both at once, carry home, then place each block at the other's
remembered start pose (position + yaw), verify and correct."""

TABLE_Z_DEFAULT = 0.7407
GRASP_ABOVE_TABLE = 0.0355   # grasp-centre height above table (expert: 0.776 on 0.7407 table)
PREGRASP_DZ = 0.05
LIFT_DZ = 0.10
PLACE_ABOVE_DZ = 0.07
RELEASE_DZ = 0.02            # expert releases ~0.044 above grasp height; we drop lower
RETREAT_DZ = 0.10
TCP = 0.12                   # grasp centre lies 0.12 m along ee +x
CLOSE_G = 0.2                # expert's commanded closed value (identical in all 25 demos)
OPEN_G = 1.0
# T template (top face, metres): total length along stem, crossbar length, crossbar thickness, stem width
T_LT, T_LC, T_TC, T_WS = 0.096, 0.070, 0.026, 0.026
POS_TOL = 0.005
YAW_TOL = math.radians(3.0)
MAX_CORRECTIONS = 3
MAX_REPICKS = 2


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _cloud(frame, cam="head"):
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K"), float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), float)
    d = np.asarray(depth, float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = d > 0
    z = d[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    W = (pc - E[:, 3]) @ E[:, :3]
    return W, np.asarray(rgb)[ok].astype(int)


def _color_mask(C, color):
    r, g, b = C[:, 0], C[:, 1], C[:, 2]
    if color == "red":
        return (r - g > 50) & (r - b > 50)
    return (b - r > 40) & (b - g > 30)


def _table_z(W):
    m = (np.abs(W[:, 0]) < 0.25) & (W[:, 1] > -0.3) & (W[:, 1] < 0.05) & (W[:, 2] > 0.70) & (W[:, 2] < 0.76)
    if m.sum() < 200:
        return TABLE_Z_DEFAULT
    return float(np.median(W[m, 2]))


def _top_points(W, C, color, table_z):
    m = _color_mask(C, color)
    top = (W[:, 2] > table_z + 0.024) & (W[:, 2] < table_z + 0.05) & (np.abs(W[:, 0]) < 0.35) \
        & (W[:, 1] > -0.35) & (W[:, 1] < 0.15)
    P = W[m & top]
    if len(P) < 40:
        return None
    zt = np.percentile(P[:, 2], 80)
    P = P[np.abs(P[:, 2] - zt) < 0.006]
    c = np.median(P[:, :2], 0)
    P = P[np.linalg.norm(P[:, :2] - c, axis=1) < 0.075]
    if len(P) < 40:
        return None
    return P


def _inside_T(s, l):
    bar = (s >= -T_LT / 2) & (s <= -T_LT / 2 + T_TC) & (np.abs(l) <= T_LC / 2)
    stem = (s >= -T_LT / 2) & (s <= T_LT / 2) & (np.abs(l) <= T_WS / 2)
    return bar | stem


def _fit_T(P):
    """Fit the T template to top-face points. Returns (cx, cy, yaw, iou).
    Body frame: origin = centre of the T's bounding box, yaw = direction from
    crossbar toward stem tip."""
    res = 0.002
    g = np.arange(-0.07, 0.07 + 1e-9, res)
    GX, GY = np.meshgrid(g, g)
    GC = np.stack([GX.ravel(), GY.ravel()], 1)
    xy = P[:, :2]
    c0 = xy.mean(0)
    q = xy - c0
    n = GX.shape[0]
    occ = np.zeros((n, n), bool)
    ij = np.round((q + 0.07) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    occ[ij[ok, 1], ij[ok, 0]] = True
    o = occ.copy()
    o[1:] |= occ[:-1]
    o[:-1] |= occ[1:]
    o[:, 1:] |= occ[:, :-1]
    o[:, :-1] |= occ[:, 1:]
    occf = o.ravel()
    a1 = T_TC * T_LC
    a2 = (T_LT - T_TC) * T_WS
    s_c = (a1 * (-T_LT / 2 + T_TC / 2) + a2 * (-T_LT / 2 + T_TC + (T_LT - T_TC) / 2)) / (a1 + a2)

    def score(th, tx, ty):
        c, s = math.cos(th), math.sin(th)
        dx = GC[:, 0] - tx
        dy = GC[:, 1] - ty
        sb = dx * c + dy * s
        lb = -dx * s + dy * c
        T = _inside_T(sb, lb)
        uni = (T | occf).sum()
        return float((T & occf).sum()) / max(1.0, float(uni))

    best = (-1.0, 0.0, 0.0, 0.0)
    for deg in range(-180, 180, 4):
        th = math.radians(deg)
        tx, ty = -s_c * math.cos(th), -s_c * math.sin(th)
        sc = score(th, tx, ty)
        if sc > best[0]:
            best = (sc, th, tx, ty)
    for sa, st in ((2.0, 0.004), (1.0, 0.002), (0.5, 0.001), (0.25, 0.0005)):
        sa = math.radians(sa)
        improved = True
        guard = 0
        while improved and guard < 60:
            improved = False
            guard += 1
            _, th, tx, ty = best
            for dth in (-sa, 0.0, sa):
                for ddx in (-st, 0.0, st):
                    for ddy in (-st, 0.0, st):
                        if dth == 0.0 and ddx == 0.0 and ddy == 0.0:
                            continue
                        sc = score(th + dth, tx + ddx, ty + ddy)
                        if sc > best[0] + 1e-9:
                            best = (sc, th + dth, tx + ddx, ty + ddy)
                            improved = True
    sc, th, tx, ty = best
    return float(c0[0] + tx), float(c0[1] + ty), float(_wrap(th)), float(sc)


def _perceive_blocks(frame):
    """Returns (blocks, table_z): blocks[color] = {x, y, yaw, iou, n} or None."""
    W, C = _cloud(frame, "head")
    tz = _table_z(W)
    out = {}
    for color in ("red", "blue"):
        P = _top_points(W, C, color, tz)
        if P is None:
            out[color] = None
            continue
        x, y, yaw, iou = _fit_T(P)
        out[color] = {"x": x, "y": y, "yaw": yaw, "iou": iou, "n": int(len(P))}
    return out, tz


def _wrist_fraction(frame, arm, color):
    rgb = np.asarray(getattr(frame, arm + "_wrist_rgb")).reshape(-1, 3).astype(int)
    return float(_color_mask(rgb, color).mean())


def _mat2q(R):
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2][1] - R[1][2]) / S, (R[0][2] - R[2][0]) / S, (R[1][0] - R[0][1]) / S]
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        S = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        q = [(R[2][1] - R[1][2]) / S, 0.25 * S, (R[0][1] + R[1][0]) / S, (R[0][2] + R[2][0]) / S]
    elif R[1][1] > R[2][2]:
        S = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        q = [(R[0][2] - R[2][0]) / S, (R[0][1] + R[1][0]) / S, 0.25 * S, (R[1][2] + R[2][1]) / S]
    else:
        S = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        q = [(R[1][0] - R[0][1]) / S, (R[0][2] + R[2][0]) / S, (R[1][2] + R[2][1]) / S, 0.25 * S]
    if q[0] < 0:
        q = [-v for v in q]
    return q


def _down_pose(gx, gy, gz, psi):
    """ee pose (7) whose grasp centre is at (gx,gy,gz), approach axis (+x) pointing
    straight down and finger axis (+y) at world yaw psi."""
    c, s = math.cos(psi), math.sin(psi)
    R = [[0.0, c, s], [0.0, s, -c], [-1.0, 0.0, 0.0]]
    return [float(gx), float(gy), float(gz + TCP)] + [float(v) for v in _mat2q(R)]


def _choose_offset(yaw_pick, yaw_place):
    """Finger-yaw offset relative to block yaw (+-90 deg, fingers across the stem).
    Demonstrations always used finger yaw near 180 deg; pick the offset keeping
    both the grasp and the place finger yaw closest to that."""
    best = None
    for off in (-math.pi / 2, math.pi / 2):
        cost = max(abs(_wrap(yaw_pick + off - math.pi)), abs(_wrap(yaw_place + off - math.pi)))
        if best is None or cost < best[0]:
            best = (cost, off)
    return best[1]


def _wp(left, lg, right, rg):
    return {"left": left, "lg": float(lg), "right": right, "rg": float(rg)}


def _pick_wps(arm, blk, off, grasp_z, home, other_pose, other_g):
    """Waypoints for one arm picking blk and carrying it home; the other arm holds."""
    psi = _wrap(blk["yaw"] + off)
    seq = [
        (_down_pose(blk["x"], blk["y"], grasp_z + PREGRASP_DZ, psi), OPEN_G),
        (_down_pose(blk["x"], blk["y"], grasp_z, psi), OPEN_G),
        (_down_pose(blk["x"], blk["y"], grasp_z, psi), CLOSE_G),
        (_down_pose(blk["x"], blk["y"], grasp_z + LIFT_DZ, psi), CLOSE_G),
        (home, CLOSE_G),
    ]
    out = []
    for pose, g in seq:
        if arm == "left":
            out.append(_wp(pose, g, other_pose, other_g))
        else:
            out.append(_wp(other_pose, other_g, pose, g))
    return out


def _place_wps(arm, tgt, off, grasp_z, home, other_pose, other_g, extra_dz=0.0):
    psi = _wrap(tgt["yaw"] + off)
    rz = grasp_z + RELEASE_DZ + extra_dz
    seq = [
        (_down_pose(tgt["x"], tgt["y"], grasp_z + PLACE_ABOVE_DZ + extra_dz, psi), CLOSE_G),
        (_down_pose(tgt["x"], tgt["y"], rz, psi), CLOSE_G),
        (_down_pose(tgt["x"], tgt["y"], rz, psi), OPEN_G),
        (_down_pose(tgt["x"], tgt["y"], grasp_z + RETREAT_DZ, psi), OPEN_G),
        (home, OPEN_G),
    ]
    out = []
    for pose, g in seq:
        if arm == "left":
            out.append(_wp(pose, g, other_pose, other_g))
        else:
            out.append(_wp(other_pose, other_g, pose, g))
    return out


def _err(blk, tgt):
    if blk is None:
        return 1.0, math.pi
    return math.hypot(blk["x"] - tgt["x"], blk["y"] - tgt["y"]), abs(_wrap(blk["yaw"] - tgt["yaw"]))


FIELDS = ("phase", "red_start", "red_start_yaw", "blue_start", "blue_start_yaw", "table_z",
          "home_left", "home_right", "carry_left", "carry_right", "off_left", "off_right",
          "placing", "place_retries", "repicks", "corrections", "waits", "log")


def _fresh():
    return {"phase": "init", "red_start": None, "red_start_yaw": None, "blue_start": None,
            "blue_start_yaw": None, "table_z": TABLE_Z_DEFAULT, "home_left": None, "home_right": None,
            "carry_left": "none", "carry_right": "none", "off_left": 0.0, "off_right": 0.0,
            "placing": "none", "place_retries": 0, "repicks": 0, "corrections": 0, "waits": 0, "log": ""}


def _ret(mem, tool, args):
    out = {k: mem.get(k) for k in FIELDS}
    if isinstance(out["log"], str) and len(out["log"]) > 1800:
        out["log"] = out["log"][-1800:]
    return {"memory": out, "act": {"tool": tool, "args": args}}


def _target(mem, color):
    """A block's goal = the OTHER block's remembered start pose."""
    if color == "red":
        s, yaw = mem["blue_start"], mem["blue_start_yaw"]
    else:
        s, yaw = mem["red_start"], mem["red_start_yaw"]
    return {"x": s[0], "y": s[1], "yaw": yaw}


def _last_info(last):
    """(motion_error, completed) parsed from the previous act_waypoints result text."""
    s = last if isinstance(last, str) else ""
    err = ('"error": "' in s) or ("-> error:" in s) or ("the tool failed" in s)
    done = None
    k = s.find('"completed": ')
    if k >= 0:
        j = k + len('"completed": ')
        num = ""
        while j < len(s) and s[j].isdigit():
            num += s[j]
            j += 1
        if num:
            done = int(num)
    return err, done


def _home_args(mem, label="hold"):
    lg = CLOSE_G if mem["carry_left"] != "none" else OPEN_G
    rg = CLOSE_G if mem["carry_right"] != "none" else OPEN_G
    return {"waypoints": [_wp(mem["home_left"], lg, mem["home_right"], rg)], "label": label,
            "fallback_home": None}


def _fallback(mem):
    return {"left": mem["home_left"], "right": mem["home_right"]}


def _pick_both(mem, blocks, flip, label):
    """Both arms pick simultaneously (left arm takes the block with smaller x) and carry home."""
    left_c = "red" if blocks["red"]["x"] <= blocks["blue"]["x"] else "blue"
    right_c = "blue" if left_c == "red" else "red"
    mem["off_left"] = _choose_offset(blocks[left_c]["yaw"], _target(mem, left_c)["yaw"])
    mem["off_right"] = _choose_offset(blocks[right_c]["yaw"], _target(mem, right_c)["yaw"])
    if flip:
        mem["off_left"] = _wrap(mem["off_left"] + math.pi)
        mem["off_right"] = _wrap(mem["off_right"] + math.pi)
    gz = float(mem["table_z"]) + GRASP_ABOVE_TABLE
    lw = _pick_wps("left", blocks[left_c], mem["off_left"], gz, mem["home_left"], None, OPEN_G)
    rw = _pick_wps("right", blocks[right_c], mem["off_right"], gz, mem["home_right"], None, OPEN_G)
    wps = [_wp(a["left"], a["lg"], b["right"], b["rg"]) for a, b in zip(lw, rw)]
    mem["carry_left"], mem["carry_right"] = left_c, right_c
    mem["phase"] = "check_pick"
    return _ret(mem, "act_waypoints", {"waypoints": wps, "label": label, "fallback_home": _fallback(mem)})


def _pick_one(mem, arm, blk, color, flip, label):
    other = "right" if arm == "left" else "left"
    off = _choose_offset(blk["yaw"], _target(mem, color)["yaw"])
    if flip:
        off = _wrap(off + math.pi)
    mem["off_" + arm] = off
    og = CLOSE_G if mem["carry_" + other] != "none" else OPEN_G
    gz = float(mem["table_z"]) + GRASP_ABOVE_TABLE
    wps = _pick_wps(arm, blk, off, gz, mem["home_" + arm], mem["home_" + other], og)
    mem["carry_" + arm] = color
    mem["phase"] = "check_pick"
    return _ret(mem, "act_waypoints", {"waypoints": wps, "label": label, "fallback_home": _fallback(mem)})


def _place_step(mem, last_err, last_done):
    """Place phase: first resolve the previous placement (released iff the open
    waypoint, index 2, completed), then place the next carried block
    (left-carried first, as the expert did)."""
    gz = float(mem["table_z"]) + GRASP_ABOVE_TABLE
    extra = 0.0
    if mem["placing"] != "none":
        arm = mem["placing"]
        released = (not last_err) or (last_done is not None and last_done >= 3)
        if released or int(mem["place_retries"]) >= 2:
            mem["carry_" + arm] = "none"
            mem["placing"] = "none"
            mem["place_retries"] = 0
        else:
            mem["place_retries"] = int(mem["place_retries"]) + 1
            extra = 0.02 * int(mem["place_retries"])
            mem["log"] += "place retry %s dz+%.2f; " % (arm, extra)
    order = [mem["placing"]] if mem["placing"] != "none" else ["left", "right"]
    for arm in order:
        c = mem["carry_" + arm]
        if c == "none":
            continue
        other = "right" if arm == "left" else "left"
        og = CLOSE_G if mem["carry_" + other] != "none" else OPEN_G
        tgt = _target(mem, c)
        wps = _place_wps(arm, tgt, float(mem["off_" + arm]), gz, mem["home_" + arm],
                         mem["home_" + other], og, extra)
        mem["placing"] = arm
        mem["phase"] = "place"
        mem["log"] += "place %s with %s at (%.3f,%.3f,%.0fdeg); " % (c, arm, tgt["x"], tgt["y"], math.degrees(tgt["yaw"]))
        return _ret(mem, "act_waypoints", {"waypoints": wps, "label": "place_" + c, "fallback_home": _fallback(mem)})
    mem["phase"] = "verify"
    return None


def perceive_decide(obs, goal, memory, last):
    """swap_T state machine: init (perceive & remember both start poses, pick both) ->
    check_pick (wrist check at home, re-pick failures) -> place (left-carried then
    right-carried block at the other block's remembered start pose) -> verify / correct."""
    mem = _fresh()
    if memory and memory.get("phase"):
        for k in FIELDS:
            if k in memory and memory[k] is not None:
                mem[k] = memory[k]
    frame = obs.current
    last_err, last_done = _last_info(last)

    if mem["phase"] == "init":
        blocks, tz = _perceive_blocks(frame)
        mem["table_z"] = tz
        if mem["home_left"] is None:
            mem["home_left"] = [float(v) for v in frame.left_ee_pose]
            mem["home_right"] = [float(v) for v in frame.right_ee_pose]
        bad = [k for k in ("red", "blue") if blocks[k] is None or blocks[k]["iou"] < 0.5]
        if bad and int(mem["waits"]) < 3:
            mem["waits"] = int(mem["waits"]) + 1
            mem["log"] += "init: unreliable %s; " % (bad,)
            return _ret(mem, "act_waypoints", _home_args(mem))
        if any(blocks[k] is None for k in ("red", "blue")):
            mem["log"] += "init: blocks not found; "
            return _ret(mem, "act_waypoints", _home_args(mem))
        for c in ("red", "blue"):
            b = blocks[c]
            mem[c + "_start"] = [b["x"], b["y"], tz]
            mem[c + "_start_yaw"] = b["yaw"]
        mem["log"] += "start red(%.3f,%.3f,%.0fdeg,iou%.2f) blue(%.3f,%.3f,%.0fdeg,iou%.2f); " % (
            blocks["red"]["x"], blocks["red"]["y"], math.degrees(blocks["red"]["yaw"]), blocks["red"]["iou"],
            blocks["blue"]["x"], blocks["blue"]["y"], math.degrees(blocks["blue"]["yaw"]), blocks["blue"]["iou"])
        return _pick_both(mem, blocks, False, "pick_both")

    if mem["phase"] == "check_pick":
        # act_waypoints ends at home (or returns home on a motion error), so the wrist
        # camera of a holding arm is filled with the block colour (~0.33) and empty otherwise.
        failed = []
        for arm in ("left", "right"):
            c = mem["carry_" + arm]
            if c == "none":
                continue
            f = _wrist_fraction(frame, arm, c)
            mem["log"] += "wrist %s %s %.2f; " % (arm, c, f)
            if f < 0.08:
                failed.append(arm)
        if failed and int(mem["repicks"]) < MAX_REPICKS * 2:
            blocks, _ = _perceive_blocks(frame)
            seen = [k for k in ("red", "blue") if blocks.get(k) is not None and blocks[k]["iou"] >= 0.5]
            if len(failed) == 2 and len(seen) == 2:
                mem["repicks"] = int(mem["repicks"]) + 1
                mem["log"] += "both grasps failed; repick both; "
                return _pick_both(mem, blocks, last_err, "repick_both")
            arm = failed[0]
            c = mem["carry_" + arm]
            mem["repicks"] = int(mem["repicks"]) + 1
            b = blocks.get(c)
            if b is None or b["iou"] < 0.5:
                mem["log"] += "%s not on table; assume %s holds it; " % (c, arm)
                failed = [a for a in failed if a != arm]
            else:
                mem["carry_" + arm] = "none"
                pick_arm = "left" if b["x"] < 0 else "right"
                if mem["carry_" + pick_arm] != "none":
                    pick_arm = arm
                mem["log"] += "repick %s with %s; " % (c, pick_arm)
                return _pick_one(mem, pick_arm, b, c, last_err, "repick_" + c)
        for arm in failed:
            mem["carry_" + arm] = "none"
        mem["phase"] = "place"
        mem["placing"] = "none"
        mem["place_retries"] = 0
        last_err, last_done = False, None

    if mem["phase"] == "place":
        res = _place_step(mem, last_err, last_done)
        if res is not None:
            return res

    # verify / correct (both arms at home, nothing carried)
    blocks, _ = _perceive_blocks(frame)
    worst = None
    for c in ("red", "blue"):
        tgt = _target(mem, c)
        b = blocks.get(c)
        dp, dy = _err(b, tgt)
        mem["log"] += "verify %s dp %.4f dyaw %.1f; " % (c, dp, math.degrees(dy))
        if b is None or b["iou"] < 0.5:
            continue
        score = dp / POS_TOL + dy / YAW_TOL
        if (dp > POS_TOL or dy > YAW_TOL) and (worst is None or score > worst[0]):
            worst = (score, c)
    if worst is None or int(mem["corrections"]) >= MAX_CORRECTIONS:
        mem["waits"] = int(mem["waits"]) + 1
        return _ret(mem, "act_waypoints", _home_args(mem))
    mem["corrections"] = int(mem["corrections"]) + 1
    c = worst[1]
    b = blocks[c]
    o = "blue" if c == "red" else "red"
    ob = blocks.get(o)
    tgt = _target(mem, c)
    if ob is not None:
        occupied = math.hypot(ob["x"] - tgt["x"], ob["y"] - tgt["y"]) < 0.045
        if occupied or (_err(ob, _target(mem, o))[0] > 0.03 and _err(b, tgt)[0] > 0.03):
            mem["log"] += "redo swap with both arms; "
            return _pick_both(mem, blocks, last_err, "redo_pick_both")
    arm = "left" if b["x"] < 0 else "right"
    mem["carry_left"], mem["carry_right"] = "none", "none"
    mem["log"] += "correct %s with %s; " % (c, arm)
    return _pick_one(mem, arm, b, c, last_err, "correct_pick_" + c)


def act_waypoints(robot, waypoints, label="", fallback_home=None):
    """Execute dual-arm waypoints in order. Each waypoint is
    {"left": [x,y,z,qw,qx,qy,qz] or None, "lg": gripper, "right": [...] or None, "rg": gripper};
    None holds that arm at its current measured pose. On the first MotionError the
    sequence stops; if fallback_home={"left": pose7, "right": pose7} is given, both arms
    then return there keeping their current gripper commands. Returns completed count,
    error text and final ee poses/grippers."""
    done = 0
    err = None
    for wp in waypoints:
        now = robot.now()
        lp = wp.get("left") or [float(v) for v in now.left_ee_pose]
        rp = wp.get("right") or [float(v) for v in now.right_ee_pose]
        args = [float(v) for v in lp] + [float(wp.get("lg", 1.0))] + [float(v) for v in rp] + [float(wp.get("rg", 1.0))]
        try:
            robot.step(*args)
        except Exception as e:  # MotionError only; EpisodeEnd is a BaseException and propagates
            err = str(e)[:200]
            break
        done += 1
    recovered = None
    if err is not None and fallback_home:
        now = robot.now()
        args = [float(v) for v in fallback_home["left"]] + [float(now.left_gripper)] + \
            [float(v) for v in fallback_home["right"]] + [float(now.right_gripper)]
        try:
            robot.step(*args)
            recovered = True
        except Exception as e:
            recovered = str(e)[:120]
    now = robot.now()
    return {"label": label, "completed": done, "total": len(waypoints), "error": err,
            "recovered_home": recovered,
            "left_ee": [round(float(v), 4) for v in now.left_ee_pose],
            "right_ee": [round(float(v), 4) for v in now.right_ee_pose],
            "grippers": [round(float(now.left_gripper), 3), round(float(now.right_gripper), 3)]}
