"""rearrange_blocks policy: head-camera RGB-D perception + scripted dual-arm pick/press/pick.

Sequence (from all 50 demonstrations): right arm moves the block lying between the
two blue mats onto the empty mat; left arm presses the red button with a closed
gripper; right arm moves the block that started on a mat to the original
position of the first block (the space between the mats).
"""

# ---- constants measured from the demonstrations (world frame, metres) ----
HOME_L = [-0.29792, -0.31380, 0.94200, 0.70000, 0.0, 0.0, 0.71414]
HOME_R = [0.30643, -0.31280, 0.94100, 0.70356, 0.0, 0.0, 0.71063]
Q_DOWN_R = [0.5, -0.5, 0.5, 0.5]            # right gripper pointing down (expert grasps)
Q_DOWN_L = [0.70710678, 0.0, 0.70710678, 0.0]  # left gripper pointing down (expert press)
TABLE_Z_REF = 0.743      # perceived table/mat surface in demos
PRE_Z = 0.980            # ee z above a block before descending
GRASP_Z = 0.902          # ee z when closing on a 4 cm block resting on the table
LIFT_Z = 1.000
CARRY_Z = 1.020
RELEASE_Z = 0.933        # ee z when opening above the destination
PRESS_UP_Z = 0.963       # ee z above button (gripper closed here)
PRESS_DOWN_Z = 0.925     # ee z at bottom of press
# perception -> expert ee xy offsets (mean over 50 demos, std ~1 mm)
OFF_BLOCK = [0.0045, 0.0022]
OFF_MAT = [0.0045, 0.0030]
OFF_BUTTON = [0.0010, 0.0022]
NEAR = 0.03              # block counts as "at" a location within this xy distance


def _backproject(frame, cam):
    rgb = np.asarray(getattr(frame, cam + "_rgb")).astype(np.int32)
    d = np.asarray(getattr(frame, cam + "_depth")).astype(np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K")).astype(np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic")).astype(np.float64)
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    pw = ((pc - E[:, 3]) @ E[:, :3]).reshape(H, W, 3)
    return rgb, pw, d > 0


def _clusters(P, r, minpts):
    out = []
    rem = P
    guard = 0
    while len(rem) >= minpts and guard < 60:
        guard += 1
        c = rem[0, :2]
        for _ in range(6):
            m = np.linalg.norm(rem[:, :2] - c, axis=1) < r
            c = np.median(rem[m, :2], axis=0)
        m = np.linalg.norm(rem[:, :2] - c, axis=1) < r
        if m.sum() >= minpts:
            out.append(rem[m])
        else:
            m = m.copy()
            m[0] = True
        rem = rem[~m]
    return out


def _scene(frame):
    """Red objects (blocks / button) and blue mats from the head camera."""
    rgb, pw, valid = _backproject(frame, "head")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    z = pw[..., 2]
    red = (r > 140) & (r - g > 80) & (r - b > 80) & valid & (z > 0.75) & (z < 0.90)
    blue = (b > 140) & (b - r > 80) & valid & (z > 0.72) & (z < 0.77)
    B = pw[blue]
    table_z = float(np.median(B[:, 2])) if len(B) > 50 else TABLE_Z_REF
    objs = []
    for C in _clusters(pw[red], 0.045, 30):
        zt = float(np.percentile(C[:, 2], 95))
        top = C[C[:, 2] > zt - 0.006]
        xy = np.median(top[:, :2], axis=0)
        dist = float(np.min(np.linalg.norm(B[:, :2] - xy, axis=1))) if len(B) else 1.0
        objs.append({"xy": [float(xy[0]), float(xy[1])], "ztop": zt, "n": int(len(C)),
                     "dblue": dist, "span": float(np.ptp(C[:, 0]))})
    mats = []
    for C in _clusters(B, 0.08, 40):
        xy = np.median(C[:, :2], axis=0)
        mats.append({"xy": [float(xy[0]), float(xy[1])], "n": int(len(C))})
    return {"objs": objs, "mats": mats, "table_z": table_z}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_initial(sc):
    """Identify button, middle block, block-on-mat and empty mat. Returns dict or None."""
    objs = sorted(sc["objs"], key=lambda o: -o["dblue"])
    if len(objs) < 3 or len(sc["mats"]) < 1:
        return None
    button = objs[0]
    if button["dblue"] < 0.09:
        return None
    blocks = objs[1:3]
    mid = max(blocks, key=lambda o: o["dblue"])
    onmat = min(blocks, key=lambda o: o["dblue"])
    if mid["dblue"] < 0.045 or onmat["dblue"] > 0.045:
        return None
    # empty mat: the blue cluster farthest from both blocks
    best = None
    for m in sc["mats"]:
        dmin = min(_dist(m["xy"], mid["xy"]), _dist(m["xy"], onmat["xy"]))
        if dmin > 0.07 and (best is None or dmin > best[0]):
            best = (dmin, m["xy"])
    if best is None:
        # symmetric layout fallback: mirror the occupied mat through the middle block
        empty = [2 * mid["xy"][0] - onmat["xy"][0], 2 * mid["xy"][1] - onmat["xy"][1]]
    else:
        empty = best[1]
    return {"button": button["xy"], "mid": mid["xy"], "onmat": onmat["xy"], "empty": empty,
            "button_ztop": button["ztop"]}


def _blocks_now(sc, button_xy):
    out = []
    for o in sc["objs"]:
        if button_xy is not None and _dist(o["xy"], button_xy) < 0.05:
            continue
        out.append(o["xy"])
    return out


def _near_any(blocks, xy):
    return any(_dist(b, xy) < NEAR for b in blocks)


def _pick_source(blocks, exclude, prefer):
    """Block to move: not within NEAR of any excluded location, closest to `prefer`."""
    cand = [b for b in blocks if not any(_dist(b, e) < NEAR for e in exclude)]
    if not cand:
        return None
    return min(cand, key=lambda b: _dist(b, prefer))


def _off(xy, off):
    return [xy[0] + off[0], xy[1] + off[1]]


def perceive_decide(obs, goal, memory, last):
    """Perceive the head RGB-D scene, update stage memory and choose the next act."""
    mem = dict(memory)
    for k in ("stage", "mid_xy", "src2_xy", "target1_xy", "button_xy", "dz", "tries",
              "presses", "notes"):
        mem.setdefault(k, None)
    frame = obs.current
    sc = _scene(frame)
    dz = mem["dz"] if mem["dz"] is not None else 0.0
    if mem["stage"] is None:
        p = _parse_initial(sc)
        if p is None:
            mem["tries"] = (mem["tries"] or 0) + 1
            mem["notes"] = "initial parse failed: %d red, %d mats" % (len(sc["objs"]), len(sc["mats"]))
            return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
        dz = max(-0.02, min(0.02, sc["table_z"] - TABLE_Z_REF))
        mem.update({"mid_xy": p["mid"], "src2_xy": p["onmat"], "target1_xy": p["empty"],
                    "button_xy": p["button"], "dz": dz, "tries": 1, "presses": 0,
                    "stage": "pick1", "notes": "init ok"})
        return {"memory": mem, "act": {"tool": "act_pick_place", "args": {
            "src": _off(p["mid"], OFF_BLOCK), "dst": _off(p["empty"], OFF_MAT), "dz": dz}}}

    blocks = _blocks_now(sc, mem["button_xy"])
    t1, src2, mid = mem["target1_xy"], mem["src2_xy"], mem["mid_xy"]
    stage = mem["stage"]
    if stage == "pick1":
        if _near_any(blocks, t1) or mem["tries"] >= 4:
            mem.update({"stage": "press", "presses": 1, "tries": 0,
                        "notes": "block1 on mat: %s" % _near_any(blocks, t1)})
            return {"memory": mem, "act": {"tool": "act_press", "args": {
                "xy": _off(mem["button_xy"], OFF_BUTTON), "dz": dz}}}
        src = _pick_source(blocks, [src2], mid)
        if src is None:
            src = mid
        mem["tries"] = mem["tries"] + 1
        mem["notes"] = "retry pick1 from %s" % [round(v, 3) for v in src]
        return {"memory": mem, "act": {"tool": "act_pick_place", "args": {
            "src": _off(src, OFF_BLOCK), "dst": _off(t1, OFF_MAT), "dz": dz}}}
    if stage == "press" or stage == "pick2":
        done = _near_any(blocks, mid) and _near_any(blocks, t1) and not _near_any(blocks, src2)
        if stage == "pick2" and (done or mem["tries"] >= 4):
            mem.update({"stage": "done", "notes": "final layout ok: %s" % done})
            return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
        src = _pick_source(blocks, [t1, mid], src2)
        if src is None:
            src = src2
        dst_ee = [mid[0] + OFF_BLOCK[0], mid[1] + OFF_BLOCK[1]]
        mem["tries"] = (mem["tries"] or 0) + 1
        mem["stage"] = "pick2"
        mem["notes"] = "pick2 from %s" % [round(v, 3) for v in src]
        return {"memory": mem, "act": {"tool": "act_pick_place", "args": {
            "src": _off(src, OFF_BLOCK), "dst": dst_ee, "dz": dz}}}
    # stage done: success not yet triggered -> re-press the button a limited number of times
    if (mem["presses"] or 0) < 3:
        mem["presses"] = (mem["presses"] or 0) + 1
        mem["notes"] = "re-press %d" % mem["presses"]
        return {"memory": mem, "act": {"tool": "act_press", "args": {
            "xy": _off(mem["button_xy"], OFF_BUTTON), "dz": dz}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}


def _pose(robot, arm):
    f = robot.now()
    p = [float(v) for v in getattr(f, arm + "_ee_pose")]
    g = float(getattr(f, arm + "_gripper"))
    return p, g


def _step(robot, arm, pose, grip, log):
    """Move `arm` to pose/grip while holding the other arm. Returns True on success."""
    other = "left" if arm == "right" else "right"
    op, og = _pose(robot, other)
    if arm == "left":
        vals = list(pose) + [grip] + op + [og]
    else:
        vals = op + [og] + list(pose) + [grip]
    try:
        robot.step(*vals)
        log.append("ok")
        return True
    except Exception as e:  # MotionError: arm stayed still
        log.append("err:" + str(e)[:60])
        return False


def act_pick_place(robot, src, dst, dz=0.0, arm=None):
    """Top-down pick of a block at ee-xy `src` and release above ee-xy `dst`, then return home.
    `dz` shifts all heights by the perceived table-height difference. Arm defaults by x."""
    if arm is None:
        arm = "right" if src[0] > -0.05 else "left"
    q = Q_DOWN_R if arm == "right" else Q_DOWN_L
    home = HOME_R if arm == "right" else HOME_L
    sx, sy = float(src[0]), float(src[1])
    tx, ty = float(dst[0]), float(dst[1])
    log = []
    seq = [
        ([sx, sy, PRE_Z + dz], 1.0, True),
        ([sx, sy, GRASP_Z + dz], 1.0, True),
        ([sx, sy, GRASP_Z + dz], 0.0, False),
        ([sx, sy, LIFT_Z + dz], 0.0, False),
        ([tx, ty, CARRY_Z + dz], 0.0, False),
        ([tx, ty, RELEASE_Z + dz], 0.0, False),
        ([tx, ty, RELEASE_Z + dz], 1.0, False),
        ([tx, ty, LIFT_Z + dz], 1.0, False),
    ]
    for i, (xyz, grip, abortable) in enumerate(seq):
        ok = _step(robot, arm, xyz + q, grip, log)
        if not ok:
            ok = _step(robot, arm, [xyz[0], xyz[1], xyz[2] + 0.01] + q, grip, log)
        if not ok and abortable:
            break
    _step(robot, arm, list(home), 1.0, log)
    p, g = _pose(robot, arm)
    return {"arm": arm, "steps": log, "final_pose": [round(v, 3) for v in p[:3]]}


def act_press(robot, xy, dz=0.0, arm=None):
    """Press a button at ee-xy `xy` top-down with a closed gripper, then return home."""
    if arm is None:
        arm = "left" if xy[0] < 0.05 else "right"
    q = Q_DOWN_L if arm == "left" else Q_DOWN_R
    home = HOME_L if arm == "left" else HOME_R
    x, y = float(xy[0]), float(xy[1])
    log = []
    seq = [([x, y, PRESS_UP_Z + dz], 1.0), ([x, y, PRESS_UP_Z + dz], 0.0),
           ([x, y, PRESS_DOWN_Z + dz], 0.0), ([x, y, PRESS_UP_Z + dz], 0.0)]
    for xyz, grip in seq:
        _step(robot, arm, xyz + q, grip, log)
    _step(robot, arm, list(home), 0.0, log)
    return {"arm": arm, "steps": log}


def act_hold(robot):
    """Hold both arms for one native step (used when nothing else should move)."""
    f = robot.now()
    robot.step(*[float(v) for v in f.left_ee_pose], float(f.left_gripper),
               *[float(v) for v in f.right_ee_pose], float(f.right_gripper))
    return {"held": True}
