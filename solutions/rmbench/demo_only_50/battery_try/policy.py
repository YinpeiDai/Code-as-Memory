"""battery_try policy: place both batteries in the fixed slot, then search the
battery-orientation combinations with a Gray-code flip order (left, right,
left, right, ...) until the dashboard needle rotates (episode ends on success).

Perception (head RGB-D, fixed camera):
  * blue battery pixels -> world points -> per-side centroids (table / slot);
  * red needle pixels near the projected dial pivot -> 'idle' (pointing up)
    or 'rotated' (pointing left).
Control: absolute end-effector waypoints copied in *form* from the expert
(pre-grasp, descend, close, lift, transport, descend, open, retreat, home);
object-dependent targets come from perception plus offsets measured on the
demonstrations (see task_analysis.md).
"""

# ---------------------------------------------------------------- constants
Q_A = [0.70710678, 0.0, 0.70710678, 0.0]    # gripper down, yaw A (left std grasp, right flipped)
Q_B = [0.0, -0.70710678, 0.0, 0.70710678]   # gripper down, yaw B (right std grasp, left flipped)
Q_STD = {"left": Q_A, "right": Q_B}
Q_FLIP = {"left": Q_B, "right": Q_A}

# grasp ee xy = head-camera blue centroid xy + offset (demo mean, 49 episodes)
TABLE_GRASP_OFF = {"left": [-0.0019, 0.0037], "right": [0.0076, 0.0037]}
SLOT_GRASP_OFF = {"left": [0.0008, 0.0035], "right": [0.0041, 0.0050]}
# fixed slot targets (expert ee xy, std < 2 mm over all demos)
SLOT_PLACE = {"left": [-0.0210, -0.1010], "right": [0.0220, -0.0970]}
SLOT_REGRASP_NOM = {"left": [-0.0255, -0.0970], "right": [0.0270, -0.0930]}
FLIP_PLACE = {"left": [-0.0190, -0.1000], "right": [0.0185, -0.1020]}
DEFAULT_TABLE = {"left": [-0.198, -0.150], "right": [0.215, -0.145]}

Z_PRE_TABLE = 0.980
Z_GRASP_TABLE = 0.911
Z_LIFT = 1.010
Z_PRE_SLOT = 0.992
Z_PLACE = 0.915
Z_RETREAT = 0.990
Z_PRE_REGRASP = 0.987
Z_REGRASP = 0.918
Z_FLIP_LIFT = 0.966
Z_FLIP_PRE = 0.990

# dial pivot in world (from head depth, demo 0) used to anchor the needle ROI
NEEDLE_PIVOT = [0.0, 0.0518, 0.805]
DEFAULT_HOME = {"left": [-0.298, -0.314, 0.942, 0.700, 0.0, 0.0, 0.714],
                "right": [0.306, -0.313, 0.941, 0.7036, 0.0, 0.0, 0.7106]}


# ---------------------------------------------------------------- perception
def _blue_world(frame):
    """World xyz of blue (battery) pixels seen by the head camera."""
    rgb = np.asarray(frame.head_rgb).astype(np.int32)
    depth = np.asarray(frame.head_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.head_K, dtype=np.float64)
    E = np.asarray(frame.head_extrinsic, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (b > 120) & (b - r > 60) & (b - g > 20) & (depth > 0)
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.zeros((0, 3))
    z = depth[v, u]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    w = (pc - E[:, 3]) @ E[:, :3]
    keep = (w[:, 2] > 0.74) & (w[:, 2] < 0.86) & (w[:, 1] > -0.40) & (w[:, 1] < 0.05) & (np.abs(w[:, 0]) < 0.45)
    return w[keep]


def _in_slot(w):
    return (np.abs(w[:, 0]) < 0.07) & (w[:, 1] > -0.15) & (w[:, 1] < -0.05)


def _centroid(pts, min_pts):
    if len(pts) < min_pts:
        return None
    return [float(pts[:, 0].mean()), float(pts[:, 1].mean()), float(pts[:, 2].mean()), int(len(pts))]


def _detect_table(frame):
    """Battery centroids on the table outside the slot, split by side (x sign)."""
    w = _blue_world(frame)
    w = w[~_in_slot(w)] if len(w) else w
    out = {}
    for arm, sel in (("left", w[:, 0] < 0), ("right", w[:, 0] >= 0)):
        out[arm] = _centroid(w[sel], 60)
    return out


def _detect_slot(frame):
    """Battery centroids inside the slot compartments (left: x<0, right: x>=0)."""
    w = _blue_world(frame)
    w = w[_in_slot(w)] if len(w) else w
    out = {}
    for arm, sel in (("left", w[:, 0] < 0), ("right", w[:, 0] >= 0)):
        out[arm] = _centroid(w[sel], 60)
    return out


def _project(frame, p):
    K = np.asarray(frame.head_K, dtype=np.float64)
    E = np.asarray(frame.head_extrinsic, dtype=np.float64)
    c = E[:, :3] @ np.asarray(p, dtype=np.float64) + E[:, 3]
    if c[2] <= 0:
        return None
    return [K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]]


def _needle(frame):
    """('rotated'|'idle'|'unknown', counts). Needle points up when idle, left when rotated."""
    uv = _project(frame, NEEDLE_PIVOT)
    if uv is None:
        return "unknown", [0, 0]
    pu, pv = int(round(uv[0])), int(round(uv[1]))
    rgb = np.asarray(frame.head_rgb).astype(np.int32)
    H, W = rgb.shape[0], rgb.shape[1]
    red = (rgb[..., 0] > 140) & (rgb[..., 0] - rgb[..., 1] > 50) & (rgb[..., 0] - rgb[..., 2] > 50)

    def box(v0, v1, u0, u1):
        v0, v1 = max(0, pv + v0), min(H, pv + v1)
        u0, u1 = max(0, pu + u0), min(W, pu + u1)
        if v1 <= v0 or u1 <= u0:
            return 0
        return int(red[v0:v1, u0:u1].sum())

    up = box(-18, -3, -4, 5)
    left = box(-4, 5, -21, -4)
    if left >= 6 and left > up:
        return "rotated", [up, left]
    if up >= 4:
        return "idle", [up, left]
    return "unknown", [up, left]


# ---------------------------------------------------------------- planning
def _pose(xy, z, q):
    return [float(xy[0]), float(xy[1]), float(z)] + [float(v) for v in q]


def _clamp_near(xy, nom, tol):
    if xy is None:
        return [nom[0], nom[1]]
    if abs(xy[0] - nom[0]) > tol or abs(xy[1] - nom[1]) > tol:
        return [nom[0], nom[1]]
    return [xy[0], xy[1]]


def _arm_step(arm, pose, grip):
    if arm == "left":
        return [pose, grip, None, None]
    return [None, None, pose, grip]


def _plan_grasp_both(lxy, rxy):
    ql, qr = Q_STD["left"], Q_STD["right"]
    return [
        [_pose(lxy, Z_PRE_TABLE, ql), 1.0, _pose(rxy, Z_PRE_TABLE, qr), 1.0],
        [_pose(lxy, Z_GRASP_TABLE, ql), 1.0, _pose(rxy, Z_GRASP_TABLE, qr), 1.0],
        [_pose(lxy, Z_GRASP_TABLE, ql), 0.0, _pose(rxy, Z_GRASP_TABLE, qr), 0.0],
        [_pose(lxy, Z_GRASP_TABLE, ql), 0.0, _pose(rxy, Z_GRASP_TABLE, qr), 0.0],
        [_pose(lxy, Z_LIFT, ql), 0.0, _pose(rxy, Z_LIFT, qr), 0.0],
    ]


def _plan_grasp_one(arm, xy):
    q = Q_STD[arm]
    return [
        _arm_step(arm, _pose(xy, Z_PRE_TABLE, q), 1.0),
        _arm_step(arm, _pose(xy, Z_GRASP_TABLE, q), 1.0),
        _arm_step(arm, _pose(xy, Z_GRASP_TABLE, q), 0.0),
        _arm_step(arm, _pose(xy, Z_GRASP_TABLE, q), 0.0),
        _arm_step(arm, _pose(xy, Z_LIFT, q), 0.0),
    ]


def _plan_place(arm, home):
    q = Q_STD[arm]
    xy = SLOT_PLACE[arm]
    return [
        _arm_step(arm, _pose(xy, Z_PRE_SLOT, q), 0.0),
        _arm_step(arm, _pose(xy, Z_PLACE, q), 0.0),
        _arm_step(arm, _pose(xy, Z_PLACE, q), 1.0),
        _arm_step(arm, _pose(xy, Z_RETREAT, q), 1.0),
        _arm_step(arm, list(home), 1.0),
    ]


def _plan_flip(arm, gxy, home):
    qs, qf = Q_STD[arm], Q_FLIP[arm]
    pxy = FLIP_PLACE[arm]
    return [
        _arm_step(arm, _pose(gxy, Z_PRE_REGRASP, qs), 1.0),
        _arm_step(arm, _pose(gxy, Z_REGRASP, qs), 1.0),
        _arm_step(arm, _pose(gxy, Z_REGRASP, qs), 0.0),
        _arm_step(arm, _pose(gxy, Z_REGRASP, qs), 0.0),
        _arm_step(arm, _pose(gxy, Z_FLIP_LIFT, qs), 0.0),
        _arm_step(arm, _pose(pxy, Z_FLIP_PRE, qf), 0.0),
        _arm_step(arm, _pose(pxy, Z_PLACE, qf), 0.0),
        _arm_step(arm, _pose(pxy, Z_PLACE, qf), 1.0),
        _arm_step(arm, _pose(pxy, Z_RETREAT, qf), 1.0),
        _arm_step(arm, list(home), 1.0),
    ]


def _home_of(frame, arm):
    p = [float(v) for v in np.asarray(getattr(frame, arm + "_ee_pose")).tolist()]
    nom = DEFAULT_HOME[arm]
    if len(p) != 7 or abs(p[0] - nom[0]) > 0.05 or abs(p[1] - nom[1]) > 0.05 or abs(p[2] - nom[2]) > 0.05:
        return list(nom)
    return p


def perceive_decide(obs, goal, memory, last):
    """Decide the next stage of the battery search from head-camera perception and memory."""
    m = dict(memory or {})
    fields = ["phase", "home_left", "home_right", "flips", "next_arm", "repairs", "needle", "note"]
    for f in fields:
        m.setdefault(f, None)
    cur = obs.current
    phase = m["phase"] or "init"
    flips = int(m["flips"] or 0)
    repairs = int(m["repairs"] or 0)
    next_arm = m["next_arm"] or "left"
    home = {"left": m["home_left"] or _home_of(obs.frames[0], "left"),
            "right": m["home_right"] or _home_of(obs.frames[0], "right")}
    m["home_left"], m["home_right"] = home["left"], home["right"]
    state, counts = _needle(cur)
    m["needle"] = state
    act = {"tool": "act_hold", "args": {}}

    if phase == "init":
        det = _detect_table(cur)
        lxy = det["left"][:2] if det["left"] else DEFAULT_TABLE["left"]
        rxy = det["right"][:2] if det["right"] else DEFAULT_TABLE["right"]
        lxy = [lxy[0] + TABLE_GRASP_OFF["left"][0], lxy[1] + TABLE_GRASP_OFF["left"][1]]
        rxy = [rxy[0] + TABLE_GRASP_OFF["right"][0], rxy[1] + TABLE_GRASP_OFF["right"][1]]
        act = {"tool": "act_run", "args": {"steps": _plan_grasp_both(lxy, rxy), "label": "grasp_both"}}
        m["phase"] = "grasped"
        m["note"] = "grasp L at %.3f,%.3f R at %.3f,%.3f (det L=%s R=%s)" % (
            lxy[0], lxy[1], rxy[0], rxy[1], det["left"] is not None, det["right"] is not None)
    elif phase == "grasped":
        act = {"tool": "act_run", "args": {"steps": _plan_place("left", home["left"]), "label": "place_left"}}
        m["phase"] = "placed_left"
        m["note"] = "place left battery in slot"
    elif phase == "placed_left":
        act = {"tool": "act_run", "args": {"steps": _plan_place("right", home["right"]), "label": "place_right"}}
        m["phase"] = "search"
        m["note"] = "place right battery in slot"
    elif phase in ("search", "solved"):
        if state == "rotated" or (phase == "solved" and state != "idle"):
            m["phase"] = "solved"
            m["note"] = "needle rotated (%s); holding" % counts
            act = {"tool": "act_hold", "args": {}}
        else:
            m["phase"] = "search"
            arm = next_arm
            slot = _detect_slot(cur)
            if slot[arm] is None and repairs < 2:
                table = _detect_table(cur)
                if table[arm] is not None:
                    xy = [table[arm][0] + TABLE_GRASP_OFF[arm][0], table[arm][1] + TABLE_GRASP_OFF[arm][1]]
                    steps = _plan_grasp_one(arm, xy) + _plan_place(arm, home[arm])
                    act = {"tool": "act_run", "args": {"steps": steps, "label": "repair_" + arm}}
                    m["repairs"] = repairs + 1
                    m["note"] = "%s battery missing from slot; re-pick at %.3f,%.3f" % (arm, xy[0], xy[1])
                    return {"memory": {f: m[f] for f in fields}, "act": act}
            if slot[arm] is not None:
                gxy = [slot[arm][0] + SLOT_GRASP_OFF[arm][0], slot[arm][1] + SLOT_GRASP_OFF[arm][1]]
            else:
                gxy = None
            gxy = _clamp_near(gxy, SLOT_REGRASP_NOM[arm], 0.015)
            act = {"tool": "act_run", "args": {"steps": _plan_flip(arm, gxy, home[arm]), "label": "flip_" + arm}}
            m["flips"] = flips + 1
            m["next_arm"] = "right" if arm == "left" else "left"
            m["note"] = "needle %s %s; flip #%d %s at %.4f,%.4f" % (state, counts, flips + 1, arm, gxy[0], gxy[1])
    m["phase"] = m["phase"] or phase
    if m["flips"] is None:
        m["flips"] = flips
    if m["repairs"] is None:
        m["repairs"] = repairs
    if m["next_arm"] is None:
        m["next_arm"] = next_arm
    return {"memory": {f: m[f] for f in fields}, "act": act}


# ---------------------------------------------------------------- actions
def _full(robot, step):
    now = robot.now()
    lp = step[0] if step[0] is not None else [float(v) for v in np.asarray(now.left_ee_pose).tolist()]
    lg = step[1] if step[1] is not None else float(np.asarray(now.left_gripper))
    rp = step[2] if step[2] is not None else [float(v) for v in np.asarray(now.right_ee_pose).tolist()]
    rg = step[3] if step[3] is not None else float(np.asarray(now.right_gripper))
    return [float(v) for v in lp] + [float(lg)] + [float(v) for v in rp] + [float(rg)]


def act_run(robot, steps, label=""):
    """Execute a list of waypoints. Each item is [left_pose7|None, left_grip|None,
    right_pose7|None, right_grip|None]; None holds that arm/gripper at its current
    value. A waypoint that fails to plan is retried once. Returns errors and final poses."""
    errors = []
    done = 0
    for i, st in enumerate(steps):
        ok = False
        for attempt in range(2):
            try:
                robot.step(*_full(robot, st))
                ok = True
                break
            except Exception as e:  # MotionError; episode end is BaseException and propagates
                errors.append("wp%d try%d: %s" % (i, attempt, str(e)[:120]))
        if ok:
            done += 1
    now = robot.now()
    return {"label": label, "waypoints": len(steps), "completed": done, "errors": errors[:6],
            "left": [round(float(v), 4) for v in np.asarray(now.left_ee_pose).tolist()[:3]],
            "right": [round(float(v), 4) for v in np.asarray(now.right_ee_pose).tolist()[:3]]}


def act_hold(robot):
    """Hold both arms and grippers in place for one native step."""
    robot.step(*_full(robot, [None, None, None, None]))
    return {"held": True}
