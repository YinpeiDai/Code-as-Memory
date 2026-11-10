"""place_block_mat policy: move each red block (left to right) from its blue mat to the
green mat and straight back to the same blue mat, remembering every original position."""

GRASP_Q = [0.5, -0.5, 0.5, 0.5]   # tool +x points straight down (expert grasp orientation, both arms)
TCP_OFFSET = 0.12                 # ee frame z above the block top face at the expert grasp (0.903 - 0.783)
PLACE_ABOVE_SURFACE = 0.172       # expert release ee z above the support surface (0.913 - 0.741)
CARRY_ABOVE_TOP = 0.217           # expert carry height: ee z 1.0 with block top at 0.783
PRE_ABOVE_GRASP = 0.08            # pre-grasp hover above the grasp pose
GREEN_HALF = 0.05                 # green mat is ~0.078 m square; tolerance for "block on green"
ORIG_TOL = 0.03                   # block counted as on its original mat within this xy distance
MAX_ATTEMPTS = 5
HELD_MIN_PIXELS = 3000          # demos: held >= 12464 px, not held <= 60 px
BLOCK_SIZE = 0.042              # cube top (0.783) minus table/mat surface (0.741)


def _backproject(frame, mask):
    """World xyz (N x 3) of masked head-camera pixels with valid depth."""
    depth = frame.head_depth
    K = np.asarray(frame.head_K, dtype=float)
    E = np.asarray(frame.head_extrinsic, dtype=float)
    v, u = np.nonzero(mask)
    d = depth[v, u].astype(float) / 1000.0
    ok = d > 0
    u = u[ok].astype(float)
    v = v[ok].astype(float)
    d = d[ok]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, d], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _red_blocks(frame):
    """Red cube top faces as [x, y, top_z, n_pixels], sorted by world x (left to right)."""
    rgb = frame.head_rgb.astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    P = _backproject(frame, (r > 120) & (g < 80) & (b < 80))
    if len(P) == 0:
        return []
    P = P[(P[:, 2] > 0.75) & (P[:, 2] < 0.85) & (P[:, 1] > -0.45) & (P[:, 1] < 0.1)]
    if len(P) == 0:
        return []
    P = P[np.argsort(P[:, 0])]
    out = []
    start = 0
    for i in range(1, len(P) + 1):
        if i == len(P) or P[i, 0] - P[i - 1, 0] > 0.02:
            c = P[start:i]
            start = i
            if len(c) < 20:
                continue
            # split clusters that share x but differ in y (e.g. a block on green vs. on a mat)
            c = c[np.argsort(c[:, 1])]
            s2 = 0
            for j in range(1, len(c) + 1):
                if j == len(c) or c[j, 1] - c[j - 1, 1] > 0.02:
                    cc = c[s2:j]
                    s2 = j
                    if len(cc) < 20:
                        continue
                    zt = float(np.percentile(cc[:, 2], 90))
                    top = cc[cc[:, 2] > zt - 0.006]
                    out.append([float(np.median(top[:, 0])), float(np.median(top[:, 1])), zt, int(len(cc))])
    out.sort(key=lambda q: q[0])
    return out


def _green_mat(frame):
    """Green mat centre [x, y, surface_z] or None."""
    rgb = frame.head_rgb.astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    P = _backproject(frame, (g > 130) & (r < 120) & (b < g - 10) & (b > 80))
    if len(P) == 0:
        return None
    P = P[P[:, 2] < 0.76]
    if len(P) < 50:
        return None
    x0, x1 = np.percentile(P[:, 0], [2, 98])
    y0, y1 = np.percentile(P[:, 1], [2, 98])
    return [float((x0 + x1) / 2), float((y0 + y1) / 2), float(np.median(P[:, 2]))]


def _table_z(frame):
    rgb = frame.head_rgb.astype(int)
    m = (rgb[..., 0] > 200) & (rgb[..., 1] > 200) & (rgb[..., 2] > 200)
    P = _backproject(frame, m)
    if len(P) < 100:
        return 0.741
    return float(np.median(P[:, 2]))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _on_green(bl, green):
    return abs(bl[0] - green[0]) < GREEN_HALF and abs(bl[1] - green[1]) < GREEN_HALF


def _nearest(blocks, p, tol):
    best = None
    for bl in blocks:
        d = _dist(bl, p)
        if d < tol and (best is None or d < _dist(best, p)):
            best = bl
    return best


def _held_pixels(frame, arm):
    """Red pixels closer than 11 cm in the given wrist camera (a held cube fills ~12.5k px)."""
    rgb = (frame.left_wrist_rgb if arm == "left" else frame.right_wrist_rgb).astype(int)
    D = frame.left_wrist_depth if arm == "left" else frame.right_wrist_depth
    m = (rgb[..., 0] > 120) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80) & (D > 0) & (D < 110)
    return int(m.sum())


def _blank_memory():
    return {"orig": [], "green": [], "home_left": [], "home_right": [], "idx": 0,
            "phase": "to_green", "attempts": 0, "carrying": False, "arm": "",
            "dst": [], "note": ""}


def perceive_decide(obs, goal, memory, last):
    """Detect red blocks with the head RGB-D camera; remember their original mats; move each,
    left to right, to the green mat and back with the arm on its side. A move is a pick decision
    followed by a place decision; the wrist camera confirms the grasp and the head camera confirms
    every placement before the state machine advances."""
    mem = _blank_memory()
    for k in mem:
        if isinstance(memory, dict) and memory.get(k) is not None:
            mem[k] = memory[k]
    f = obs.current
    if not mem["orig"]:
        f0 = obs.frames[0]
        b0 = _red_blocks(f0)
        mem["orig"] = [[q[0], q[1], q[2]] for q in b0]
        g = _green_mat(f0)
        if g is None:
            g = [0.0, -0.25, _table_z(f0)]
        mem["green"] = g
        mem["home_left"] = [float(v) for v in f0.left_ee_pose]
        mem["home_right"] = [float(v) for v in f0.right_ee_pose]
        mem["idx"] = 0
        mem["phase"] = "to_green"
        mem["attempts"] = 0
        mem["carrying"] = False
    orig = mem["orig"]
    green = mem["green"]
    hold = {"tool": "act_hold", "args": {}}
    if mem["carrying"]:
        arm = mem["arm"]
        home = mem["home_left"] if arm == "left" else mem["home_right"]
        mem["carrying"] = False
        n = _held_pixels(f, arm)
        if n > HELD_MIN_PIXELS:
            mem["note"] = "block %d held (%d px); placing" % (mem["idx"], n)
            return {"memory": mem, "act": {"tool": "act_place", "args": {
                "arm": arm, "dst": mem["dst"], "home": home}}}
        mem["note"] = "block %d grasp missed (%d px); retreat" % (mem["idx"], n)
        return {"memory": mem, "act": {"tool": "act_retreat", "args": {"arm": arm, "home": home}}}
    blocks = _red_blocks(f)
    # advance the state machine from what the head camera sees now
    while mem["idx"] < len(orig):
        o = orig[mem["idx"]]
        on_g = [bl for bl in blocks if _on_green(bl, green)]
        at_o = _nearest(blocks, o, ORIG_TOL)
        if mem["phase"] == "to_green":
            if on_g and mem["attempts"] > 0:
                mem["phase"] = "to_orig"
                mem["attempts"] = 0
                continue
            break
        if at_o is not None and not on_g and mem["attempts"] > 0:
            mem["idx"] += 1
            mem["phase"] = "to_green"
            mem["attempts"] = 0
            continue
        break
    if mem["idx"] < len(orig) and mem["attempts"] >= MAX_ATTEMPTS:
        mem["idx"] += 1
        mem["phase"] = "to_green"
        mem["attempts"] = 0
    if mem["idx"] >= len(orig):
        mem["note"] = "sequence finished"
        return {"memory": mem, "act": hold}
    o = orig[mem["idx"]]
    arm = "left" if o[0] < 0 else "right"
    others = [orig[j] for j in range(len(orig)) if j != mem["idx"]]
    if mem["phase"] == "to_green":
        src = _nearest(blocks, o, ORIG_TOL)
        if src is None:
            # block not where expected: take a red block not on any other original mat
            free = [bl for bl in blocks if _nearest(others, bl, ORIG_TOL) is None]
            src = min(free, key=lambda q: _dist(q, o)) if free else [o[0], o[1], o[2]]
        dst = [green[0], green[1], green[2]]
    else:
        on_g = [bl for bl in blocks if _on_green(bl, green)]
        if on_g:
            src = min(on_g, key=lambda q: _dist(q, green))
        else:
            free = [bl for bl in blocks if _nearest(orig, bl, ORIG_TOL) is None]
            src = min(free, key=lambda q: _dist(q, green)) if free else [green[0], green[1], o[2]]
        dst = [o[0], o[1], o[2] - BLOCK_SIZE]
    mem["attempts"] += 1
    mem["carrying"] = True
    mem["arm"] = arm
    mem["dst"] = [float(dst[0]), float(dst[1]), float(dst[2])]
    mem["note"] = "block %d %s attempt %d: pick %.3f,%.3f" % (mem["idx"], mem["phase"], mem["attempts"], src[0], src[1])
    act = {"tool": "act_pick", "args": {
        "arm": arm, "src": [float(src[0]), float(src[1]), float(src[2])]}}
    return {"memory": mem, "act": act}


def _pose(fr, arm):
    p = fr.left_ee_pose if arm == "left" else fr.right_ee_pose
    return [float(v) for v in p]


def _send(robot, arm, target, grip, hold_pose, hold_grip):
    """One native step: `arm` to target pose (7) with gripper `grip`; the other arm holds."""
    if arm == "left":
        vals = list(target) + [grip] + list(hold_pose) + [hold_grip]
    else:
        vals = list(hold_pose) + [hold_grip] + list(target) + [grip]
    try:
        robot.step(*vals)
        return ""
    except Exception as e:  # MotionError (planning failure); EpisodeEnd is a BaseException and propagates
        return str(e)[:120]


def _hold_of(now, arm):
    other = "right" if arm == "left" else "left"
    g = float(now.right_gripper if arm == "left" else now.left_gripper)
    return _pose(now, other), g


def act_pick(robot, arm, src):
    """With `arm` ('left'/'right') open above src=[x,y,top_z] (block top-face centre), descend to the
    expert grasp height, close, and lift to carry height. The other arm holds. Returns planning errors
    and the final ee pose; the arm stays up holding (or not) the block."""
    now = robot.now()
    hold_pose, hold_grip = _hold_of(now, arm)
    q = list(GRASP_Q)
    sx, sy, top = float(src[0]), float(src[1]), float(src[2])
    gz = top + TCP_OFFSET
    carry = top + CARRY_ABOVE_TOP
    plan = [
        ("pre", [sx, sy, gz + PRE_ABOVE_GRASP], 1.0),
        ("down", [sx, sy, gz], 1.0),
        ("close", [sx, sy, gz], 0.0),
        ("settle", [sx, sy, gz], 0.0),
        ("lift", [sx, sy, carry], 0.0),
    ]
    errors = []
    for name, xyz, g in plan:
        err = _send(robot, arm, xyz + q, g, hold_pose, hold_grip)
        if err:
            errors.append(name + ": " + err)
    return {"arm": arm, "errors": errors, "final_ee": _pose(robot.now(), arm)}


def act_place(robot, arm, dst, home):
    """Carry the held block with `arm` at its current height to above dst=[x,y,surface_z], lower to
    the expert release height, open, lift and return the arm to `home` (7-number pose)."""
    now = robot.now()
    hold_pose, hold_grip = _hold_of(now, arm)
    q = list(GRASP_Q)
    cur = _pose(now, arm)
    dx, dy, surf = float(dst[0]), float(dst[1]), float(dst[2])
    pz = surf + PLACE_ABOVE_SURFACE
    carry = max(cur[2], pz + 0.06)
    plan = [
        ("mid", [(cur[0] + dx) / 2, (cur[1] + dy) / 2, carry], 0.0),
        ("over", [dx, dy, carry], 0.0),
        ("lower", [dx, dy, pz], 0.0),
        ("open", [dx, dy, pz], 1.0),
        ("up", [dx, dy, pz + 0.07], 1.0),
    ]
    errors = []
    for name, xyz, g in plan:
        err = _send(robot, arm, xyz + q, g, hold_pose, hold_grip)
        if err:
            errors.append(name + ": " + err)
    err = _send(robot, arm, [float(v) for v in home], 1.0, hold_pose, hold_grip)
    if err:
        errors.append("home: " + err)
    return {"arm": arm, "errors": errors, "final_ee": _pose(robot.now(), arm)}


def act_retreat(robot, arm, home):
    """Open the gripper of `arm` where it is, then return that arm to `home` (7-number pose)."""
    now = robot.now()
    hold_pose, hold_grip = _hold_of(now, arm)
    errors = []
    err = _send(robot, arm, _pose(now, arm), 1.0, hold_pose, hold_grip)
    if err:
        errors.append("open: " + err)
    err = _send(robot, arm, [float(v) for v in home], 1.0, hold_pose, hold_grip)
    if err:
        errors.append("home: " + err)
    return {"arm": arm, "errors": errors, "final_ee": _pose(robot.now(), arm)}


def act_hold(robot):
    """Hold both arms at their current poses for one native step."""
    n = robot.now()
    robot.step(*_pose(n, "left"), float(n.left_gripper), *_pose(n, "right"), float(n.right_gripper))
    return {"held": True}
