"""cover_blocks policy: remember block colours, cover slots left->right, uncover red, green, blue.

Perception uses the fixed head camera (RGB + depth + calibration).  Memory keeps the colour of
the block in each slot (needed once lids hide the blocks), lid/block grasp positions, progress
through the plan, which arm is away from home and retry counters.
"""

SLOT_X = [-0.2, 0.0, 0.2]
LID_Y = -0.05
BLOCK_Y = -0.18
HANDLE_TOP_Z = 0.846
PICK_Z = 0.975
PRE_Z = 1.018
CARRY_Z = 1.045
PLACE_Z = 0.983
RETREAT_Z = 1.06
Q_DOWN = [0.5, -0.5, 0.5, 0.5]
HOME_LEFT = [-0.298, -0.314, 0.942, 0.7, 0.0, 0.0, 0.714]
HOME_RIGHT = [0.306, -0.313, 0.941, 0.704, 0.0, 0.0, 0.711]
COLORS = ["red", "green", "blue"]
UNCOVER_ORDER = ["red", "green", "blue"]
VISIBLE_MIN = 150
COVERED_MAX = 40
MAX_RETRY = 2


def _backproject(frame, cam, mask):
    D = np.asarray(getattr(frame, cam + "_depth")).astype(float) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K")).astype(float)
    E = np.asarray(getattr(frame, cam + "_extrinsic")).astype(float)
    m = mask & (D > 0)
    vs, us = np.nonzero(m)
    if len(us) == 0:
        return np.zeros((0, 3))
    d = D[vs, us]
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    Pc = np.stack([x, y, d], -1)
    return (Pc - E[:, 3]) @ E[:, :3]


def _color_masks(frame):
    im = np.asarray(frame.head_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "red": (r > 150) & (g < 80) & (b < 80),
        "green": (g > 150) & (r < 100) & (b < 100),
        "blue": (b > 150) & (r < 80) & (g < 80),
    }


def _color_stats(frame):
    """Per colour: pixel count and median world point of pixels on the table region."""
    out = {}
    masks = _color_masks(frame)
    for c in COLORS:
        P = _backproject(frame, "head", masks[c])
        if len(P):
            P = P[(P[:, 2] > 0.72) & (P[:, 2] < 0.85) & (np.abs(P[:, 0]) < 0.45) & (P[:, 1] > -0.32) & (P[:, 1] < 0.1)]
        if len(P) >= 30:
            top = P[P[:, 2] > P[:, 2].max() - 0.006]
            out[c] = {"n": int(len(P)), "x": float(np.median(P[:, 0])),
                      "top": [float(np.mean(top[:, 0])), float(np.mean(top[:, 1]))]}
        else:
            out[c] = {"n": int(len(P)), "x": None, "top": None}
    return out


def _handle_clusters(frame):
    """Brown lid-handle points at handle height, clustered in world xy."""
    im = np.asarray(frame.head_rgb).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 120) & (r - b > 40) & (g - b > 15) & (r - g > 10) & (r - g < 70)
    P = _backproject(frame, "head", m)
    if len(P) == 0:
        return []
    P = P[(P[:, 2] > 0.825) & (P[:, 2] < 0.875) & (np.abs(P[:, 0]) < 0.45) & (P[:, 1] > -0.32) & (P[:, 1] < 0.12)]
    out = []
    for _ in range(6):
        if len(P) < 40:
            break
        # densest seed: point with most neighbours on a subsample
        idx = np.arange(0, len(P), max(1, len(P) // 300))
        S = P[idx, :2]
        d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
        seed = S[int(np.argmax((d2 < 0.03 ** 2).sum(1)))]
        near = ((P[:, :2] - seed) ** 2).sum(1) < 0.04 ** 2
        for _k in range(3):
            c = P[near, :2].mean(0)
            near = ((P[:, :2] - c) ** 2).sum(1) < 0.04 ** 2
        if near.sum() >= 40:
            Q = P[near]
            out.append({"x": float(Q[:, 0].mean()), "y": float(Q[:, 1].mean()),
                        "ztop": float(np.percentile(Q[:, 2], 95)), "n": int(near.sum())})
        P = P[~near]
    return out


def _nearest_handle(clusters, xy, radius, min_n=120):
    best, bd = None, radius
    for c in clusters:
        d = math.hypot(c["x"] - xy[0], c["y"] - xy[1])
        if d < bd and c["n"] >= min_n:
            best, bd = c, d
    return best


def _refine(nominal, cluster, deadband=0.015, limit=0.05):
    """Keep the nominal grasp point unless the measured handle is clearly displaced."""
    if cluster is None:
        return [nominal[0], nominal[1], PICK_Z]
    dx, dy = cluster["x"] - nominal[0], cluster["y"] - nominal[1]
    z = PICK_Z + max(-0.015, min(0.015, cluster["ztop"] - HANDLE_TOP_Z))
    if math.hypot(dx, dy) <= deadband or math.hypot(dx, dy) > limit:
        return [nominal[0], nominal[1], z]
    return [cluster["x"], cluster["y"], z]


def _parse_layout(frame):
    """Colour per slot (left->right in the head view = increasing world x)."""
    st = _color_stats(frame)
    found = [(st[c]["x"], c) for c in COLORS if st[c]["x"] is not None]
    slots = [None, None, None]
    for x, c in sorted(found):
        free = [i for i in range(3) if slots[i] is None]
        i = min(free, key=lambda k: abs(SLOT_X[k] - x))
        slots[i] = c
    missing = [c for c in COLORS if c not in slots]
    for i in range(3):
        if slots[i] is None and missing:
            slots[i] = missing.pop(0)
    blocks = []
    for i in range(3):
        nominal = [SLOT_X[i], BLOCK_Y]
        top = st[slots[i]]["top"]
        if top is not None:
            # top-face centroid is biased ~-3 mm in y on the demos; correct only clear displacements
            meas = [top[0], top[1] + 0.003]
            if 0.015 < math.hypot(meas[0] - nominal[0], meas[1] - nominal[1]) < 0.05:
                nominal = meas
        blocks.append([round(nominal[0], 4), round(nominal[1], 4)])
    return slots, blocks, st


def _plan(colors):
    tasks = [["cover", 0], ["cover", 1], ["cover", 2]]
    for c in UNCOVER_ORDER:
        tasks.append(["uncover", colors.index(c)])
    tasks.append(["finish", -1])
    return tasks


def _arm_for(slot, active):
    if SLOT_X[slot] < -0.1:
        return "left"
    if SLOT_X[slot] > 0.1:
        return "right"
    return active if active in ("left", "right") else "left"


def perceive_decide(obs, goal, memory, last):
    """Parse the layout once, verify the previous macro from the head camera, choose the next macro."""
    frame = obs.current
    mem = dict(memory)
    if not mem.get("colors_lr"):
        slots, blocks, st = _parse_layout(obs.frames[0])
        hc = _handle_clusters(obs.frames[0])
        lids = []
        for i in range(3):
            p = _refine([SLOT_X[i], LID_Y], _nearest_handle(hc, [SLOT_X[i], LID_Y], 0.06))
            lids.append([round(p[0], 4), round(p[1], 4)])
        mem = {"colors_lr": slots, "block_xy": blocks, "lid_xy": lids, "task_idx": 0,
               "active_arm": "none", "retries": 0, "home_checked": False,
               "pending": "none", "notes": "layout " + ",".join(slots)}
    colors = mem["colors_lr"]
    tasks = _plan(colors)
    st = _color_stats(frame)
    idx = int(mem["task_idx"])
    note = ""
    # verify the macro that was just executed
    if mem["pending"] != "none" and idx < len(tasks):
        kind, slot = tasks[idx]
        n = st[colors[slot]]["n"] if slot >= 0 else 0
        ok = True
        if kind == "cover":
            ok = n <= COVERED_MAX
        elif kind == "uncover":
            ok = n >= VISIBLE_MIN
        note = "%s slot %d %s count %d -> %s" % (kind, slot, colors[slot] if slot >= 0 else "-", n, "ok" if ok else "FAIL")
        if ok or kind == "finish":
            idx += 1
            mem["retries"] = 0
            mem["home_checked"] = False
        elif kind == "uncover" and not mem["home_checked"] and mem["active_arm"] != "none":
            # the arm may occlude the block: park it and look again before retrying
            mem["home_checked"] = True
            mem["task_idx"] = idx
            mem["pending"] = "home"
            mem["notes"] = note + "; parking arm to re-check"
            arm = mem["active_arm"]
            mem["active_arm"] = "none"
            return {"memory": mem, "act": {"tool": "act_home", "args": {"arm": arm, "both": False}}}
        elif int(mem["retries"]) >= MAX_RETRY:
            idx += 1
            mem["retries"] = 0
            mem["home_checked"] = False
            note += " (giving up after retries)"
        else:
            mem["retries"] = int(mem["retries"]) + 1
            mem["home_checked"] = False
    mem["task_idx"] = idx
    if idx >= len(tasks):
        mem["pending"] = "none"
        mem["notes"] = (note + "; plan complete").strip("; ")
        return {"memory": mem, "act": {"tool": "act_hold", "args": {}}}
    kind, slot = tasks[idx]
    if kind == "finish":
        mem["pending"] = "finish"
        mem["active_arm"] = "none"
        mem["notes"] = (note + "; homing both arms").strip("; ")
        return {"memory": mem, "act": {"tool": "act_home", "args": {"arm": "left", "both": True}}}
    hc = _handle_clusters(frame)
    lid_home = mem["lid_xy"][slot]
    block = mem["block_xy"][slot]
    if kind == "cover":
        nominal_pick = lid_home
        if int(mem["retries"]) > 0:
            # failed cover: find the lid near its home, else any free handle near the slot
            c = _nearest_handle(hc, lid_home, 0.08, 60) or _nearest_handle(hc, block, 0.12, 60)
            pick = _refine(nominal_pick, c, 0.01, 0.12)
        else:
            pick = _refine(nominal_pick, _nearest_handle(hc, nominal_pick, 0.06))
        place = [block[0], block[1]]
    else:
        pick = _refine(block, _nearest_handle(hc, block, 0.06), 0.015, 0.08 if int(mem["retries"]) > 0 else 0.05)
        place = [lid_home[0], lid_home[1]]
    arm = _arm_for(slot, mem["active_arm"])
    park_other = mem["active_arm"] not in ("none", arm)
    mem["active_arm"] = arm
    mem["pending"] = kind
    mem["notes"] = (note + "; %s slot %d (%s) with %s arm, pick %s place %s" % (
        kind, slot, colors[slot], arm, [round(v, 3) for v in pick], [round(v, 3) for v in place])).strip("; ")
    return {"memory": mem, "act": {"tool": "act_transfer", "args": {
        "arm": arm, "pick": [float(v) for v in pick], "place": [float(v) for v in place],
        "park_other": bool(park_other)}}}


def _pose_of(frame, arm):
    return [float(v) for v in getattr(frame, arm + "_ee_pose")], float(getattr(frame, arm + "_gripper"))


def _send(robot, arm, pose, grip, log):
    """Move one arm to pose with gripper grip while holding the other arm; retry once slightly higher."""
    other = "right" if arm == "left" else "left"
    for attempt in range(2):
        now = robot.now()
        op, og = _pose_of(now, other)
        target = list(pose)
        if attempt == 1:
            target[2] += 0.01
        if arm == "left":
            args = target + [grip] + op + [og]
        else:
            args = op + [og] + target + [grip]
        try:
            robot.step(*args)
            return True
        except Exception as e:  # MotionError (planning failure); EpisodeEnd is not an Exception
            log.append("%s step failed (%s): %s" % (arm, attempt, str(e)[:80]))
    return False


def act_transfer(robot, arm, pick, place, park_other=False):
    """Pick a lid at pick=[x,y(,z)] with `arm` and put it down at place=[x,y]; optionally park the other arm first."""
    log = []
    other = "right" if arm == "left" else "left"
    if park_other:
        _send(robot, other, HOME_LEFT if other == "left" else HOME_RIGHT, 1.0, log)
    pz = pick[2] if len(pick) > 2 else PICK_Z
    q = list(Q_DOWN)
    seq = [
        ([pick[0], pick[1], PRE_Z] + q, 1.0),
        ([pick[0], pick[1], pz] + q, 1.0),
        ([pick[0], pick[1], pz] + q, 0.0),
        ([pick[0], pick[1], CARRY_Z] + q, 0.0),
        ([place[0], place[1], CARRY_Z] + q, 0.0),
        ([place[0], place[1], PLACE_Z] + q, 0.0),
        ([place[0], place[1], PLACE_Z] + q, 1.0),
        ([place[0], place[1], RETREAT_Z] + q, 1.0),
    ]
    ok = 0
    for pose, grip in seq:
        if _send(robot, arm, pose, grip, log):
            ok += 1
    end, _g = _pose_of(robot.now(), arm)
    return {"steps_ok": ok, "steps": len(seq), "errors": log, "end_xyz": [round(v, 3) for v in end[:3]]}


def act_home(robot, arm="left", both=False):
    """Return `arm` (or both arms when both=True) to the home pose with the gripper open."""
    log = []
    arms = ["left", "right"] if both else [arm]
    for a in arms:
        _send(robot, a, HOME_LEFT if a == "left" else HOME_RIGHT, 1.0, log)
    return {"homed": arms, "errors": log}


def act_hold(robot):
    """Hold both arms in place for one native step (used once the plan is complete)."""
    now = robot.now()
    lp, lg = _pose_of(now, "left")
    rp, rg = _pose_of(now, "right")
    robot.step(*(lp + [lg] + rp + [rg]))
    return {"held": True}
