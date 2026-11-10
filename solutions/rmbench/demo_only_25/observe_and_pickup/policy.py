"""observe_and_pickup policy: remember the shelf object seen at reset, find its twin on the table, pick it up.

Decision phases (memory.phase):
  observe  -> describe the shelf target from the reset frame, move both grippers outward (clear head view)
  perceive -> segment table objects in head RGB-D, match to remembered target, plan a top-down grasp
  pregrasp -> chosen arm above the object (gripper open)
  descend  -> chosen arm to grasp height
  close    -> close gripper in place
  lift     -> raise 0.10 m
  check    -> verify object left the table; lift more, or retreat and retry
"""

# Calibrated from 22 un-occluded expert demonstrations (shelf minus table feature offsets and spreads).
FEAT_OFFSET = [0.37, -0.26, -0.14, 4.25, -4.31, 5.84]   # h(cm), L(cm), W(cm), R-G, B-G, V
FEAT_SIGMA = [0.15, 0.8, 0.6, 7.0, 9.0, 6.0]
TERM_CAP = 25.0
CENTER_FIX = [0.003, 0.002]          # expert grasp point minus visible top-bbox centre (m)
PRE_H = 0.12                        # pre-grasp grasp-centre height above table
LIFT = 0.10
CLEAR_DX = 0.10
GRASP_OFF = 0.12                    # grasp centre along local +x of ee frame


def _q_down(theta):
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    k = math.sqrt(0.5)
    return [k * c, -k * s, k * c, k * s]


def _norm_yaw(theta):
    """Long-axis yaw is pi-periodic; keep it in [-10deg, 170deg) like the expert grasps."""
    lo = -math.radians(10.0)
    while theta < lo:
        theta += math.pi
    while theta >= lo + math.pi:
        theta -= math.pi
    return theta


def _desc_vec(f):
    r, g, b = f["mean_rgb"]
    return [f["height"] * 100.0, f["length"] * 100.0, f["width"] * 100.0, r - g, b - g, (r + g + b) / 3.0]


def _match_dist(shelf_vec, table_vec):
    d = 0.0
    for i in range(6):
        t = ((shelf_vec[i] - table_vec[i] - FEAT_OFFSET[i]) / FEAT_SIGMA[i]) ** 2
        d += min(t, TERM_CAP)
    return d


def _head(frame):
    P, valid = _cloud(frame.head_depth, frame.head_K, frame.head_extrinsic)
    return P, valid


def _shelf_descriptor(frame):
    P, valid = _head(frame)
    m = _segment_shelf(P, valid)
    if int(m.sum()) < 60:
        return None
    f = _features(frame.head_rgb, P, m, SHELF_Z)
    if f["height"] < 0.008:
        return None
    return _desc_vec(f)


def _top_bbox_center(pts, h):
    z = pts[:, 2] - TABLE_Z
    top = pts[z > 0.6 * h][:, :2]
    if len(top) < 10:
        top = pts[:, :2]
    m = top.mean(0)
    if len(top) < 4:
        return m
    w, V = np.linalg.eigh(np.cov((top - m).T))
    q = (top - m) @ V
    mid = (np.percentile(q, 2, axis=0) + np.percentile(q, 98, axis=0)) / 2.0
    return m + V @ mid


def _table_candidates(frame):
    P, valid = _head(frame)
    objs = _table_objects(P, valid)
    out = []
    for o in objs:
        if o["pix"] > 3000:
            continue
        f = _features(frame.head_rgb, P, o["mask"], TABLE_Z)
        if f["height"] < 0.008:
            continue
        pts = P[o["mask"]]
        c = _top_bbox_center(pts, f["height"])
        out.append({
            "vec": _desc_vec(f),
            "center": [float(c[0]) + CENTER_FIX[0], float(c[1]) + CENTER_FIX[1]],
            "axis": f["axis"],
            "height": f["height"],
            "length": f["length"],
            "width": f["width"],
            "pix": o["pix"],
            "own_near": _count_near_P(P, valid, [float(c[0]) + CENTER_FIX[0], float(c[1]) + CENTER_FIX[1]], o["mask"]),
        })
    return out


def _count_near_P(P, valid, xy, own=None, r=0.035, zlo=0.006, zhi=0.07):
    z = P[..., 2] - TABLE_Z
    d = np.sqrt((P[..., 0] - xy[0]) ** 2 + (P[..., 1] - xy[1]) ** 2)
    m = valid & (d < r) & (z > zlo) & (z < zhi)
    if own is not None:
        m = m & own
    return int(m.sum())


def _count_near(frame, xy, r=0.035, zlo=0.006, zhi=0.07):
    P, valid = _head(frame)
    z = P[..., 2] - TABLE_Z
    d = np.sqrt((P[..., 0] - xy[0]) ** 2 + (P[..., 1] - xy[1]) ** 2)
    return int((valid & (d < r) & (z > zlo) & (z < zhi)).sum())


def _pose_list(p):
    return [float(v) for v in p]


def _blank_memory():
    return {
        "phase": "observe",
        "target_desc": [],
        "target_seen": False,
        "arm": "",
        "grasp": [],
        "home_left": [],
        "home_right": [],
        "attempts": 0,
        "failed": [],
        "near_count": 0,
        "near_other": 0,
        "gz_adjust": 0.0,
        "lift_extra": 0,
        "log": "",
    }


def perceive_decide(obs, goal, memory, last):
    """Phase machine for observe_and_pickup; returns the next act call."""
    mem = _blank_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    cur = obs.current
    L = _pose_list(cur.left_ee_pose)
    R = _pose_list(cur.right_ee_pose)
    lg = float(cur.left_gripper)
    rg = float(cur.right_gripper)
    hold = {"left": L + [lg], "right": R + [rg]}
    phase = mem["phase"]

    # remember the shelf target the first time it is visible (reset frame first)
    if not mem["target_seen"]:
        for fr in list(obs.frames)[:1] + [cur]:
            d = _shelf_descriptor(fr)
            if d is not None:
                mem["target_desc"] = [float(v) for v in d]
                mem["target_seen"] = True
                break

    if phase == "observe":
        mem["home_left"] = L
        mem["home_right"] = R
        mem["phase"] = "perceive"
        mem["log"] = "target " + ("seen" if mem["target_seen"] else "NOT seen")
        left = [L[0] - CLEAR_DX, L[1], L[2]] + L[3:] + [1.0]
        right = [R[0] + CLEAR_DX, R[1], R[2]] + R[3:] + [1.0]
        return {"memory": mem, "act": {"tool": "act_move", "args": {"left": left, "right": right}}}

    if phase == "perceive":
        cands = _table_candidates(cur)
        if not cands:
            mem["log"] = "no table objects found"
            return {"memory": mem, "act": {"tool": "act_move", "args": {"left": hold["left"], "right": hold["right"]}}}
        scores = []
        for c in cands:
            s = _match_dist(mem["target_desc"], c["vec"]) if mem["target_seen"] else 0.0
            # objects that already failed twice at the same spot are penalised
            for fxy in mem["failed"]:
                if math.hypot(c["center"][0] - fxy[0], c["center"][1] - fxy[1]) < 0.03:
                    s += 15.0
            scores.append(s)
        best = int(np.argmin(scores))
        c = cands[best]
        h = c["height"]
        gz = min(max(0.5 * h + 0.027, 0.033), 0.06)
        gz = min(max(gz + float(mem["gz_adjust"]), 0.028), 0.075)
        ax = c["axis"]
        theta = math.atan2(ax[1], ax[0])
        if mem["attempts"] % 2 == 1 and c["length"] < 1.25 * c["width"]:
            theta += math.pi / 2.0
        theta = _norm_yaw(theta)
        arm = "left" if c["center"][0] < 0 else "right"
        mem["arm"] = arm
        mem["grasp"] = [c["center"][0], c["center"][1], gz, theta]
        mem["near_count"] = int(c["own_near"])
        mem["near_other"] = max(0, _count_near(cur, c["center"]) - int(c["own_near"]))
        srt = sorted(scores)
        mem["log"] = "match d=%.1f next=%.1f n=%d h=%.3f L=%.3f W=%.3f" % (
            srt[0], srt[1] if len(srt) > 1 else -1.0, len(cands), h, c["length"], c["width"])
        mem["phase"] = "pregrasp"
        phase = "pregrasp"

    if phase == "close":
        x, y, gz, theta = mem["grasp"]
        ee = L if mem["arm"] == "left" else R
        want = TABLE_Z + gz + GRASP_OFF
        if ee[2] > want + 0.02 or math.hypot(ee[0] - x, ee[1] - y) > 0.03:
            # the descent was not executed (planning failure): retry with a higher grasp
            mem["attempts"] = mem["attempts"] + 1
            mem["gz_adjust"] = float(mem["gz_adjust"]) + 0.01
            mem["log"] = "descent not reached (ee z %.3f want %.3f); retry higher" % (ee[2], want)
            mem["phase"] = "retreat"
            return {"memory": mem, "act": {"tool": "act_move", "args": {"left": hold["left"], "right": hold["right"]}}}

    if phase in ("pregrasp", "descend", "close", "lift", "lift_more"):
        arm = mem["arm"]
        x, y, gz, theta = mem["grasp"]
        q = _q_down(theta)
        if phase == "pregrasp":
            zc, grip, nxt = TABLE_Z + PRE_H, 1.0, "descend"
        elif phase == "descend":
            zc, grip, nxt = TABLE_Z + gz, 1.0, "close"
        elif phase == "close":
            zc, grip, nxt = TABLE_Z + gz, 0.0, "lift"
        elif phase == "lift":
            zc, grip, nxt = TABLE_Z + gz + LIFT, 0.0, "check"
        else:
            zc, grip, nxt = TABLE_Z + gz + LIFT + 0.06, 0.0, "check"
        target = [x, y, zc + GRASP_OFF] + q + [grip]
        mem["phase"] = nxt
        args = {"left": hold["left"], "right": hold["right"]}
        args[arm] = target
        return {"memory": mem, "act": {"tool": "act_move", "args": args}}

    if phase == "check":
        arm = mem["arm"]
        x, y, gz, theta = mem["grasp"]
        n = max(0, _count_near(cur, [x, y]) - mem["near_other"])
        lifted = n < max(20, 0.35 * mem["near_count"])
        if lifted and mem["lift_extra"] < 1:
            mem["lift_extra"] = mem["lift_extra"] + 1
            mem["log"] = "object left table (n=%d of %d); lifting more" % (n, mem["near_count"])
            q = _q_down(theta)
            target = [x, y, TABLE_Z + gz + LIFT + 0.06 + GRASP_OFF] + q + [0.0]
            args = {"left": hold["left"], "right": hold["right"]}
            args[arm] = target
            return {"memory": mem, "act": {"tool": "act_move", "args": args}}
        if lifted:
            mem["log"] = "object lifted but episode continues; holding"
            return {"memory": mem, "act": {"tool": "act_move", "args": {"left": hold["left"], "right": hold["right"]}}}
        # failed: open, retreat the arm outward and re-perceive
        mem["attempts"] = mem["attempts"] + 1
        mem["gz_adjust"] = float(mem["gz_adjust"]) - 0.008
        if mem["attempts"] >= 2:
            mem["failed"] = mem["failed"] + [[x, y]]
        mem["lift_extra"] = 0
        mem["log"] = "grasp failed (n=%d of %d); retry %d" % (n, mem["near_count"], mem["attempts"])
        mem["phase"] = "retreat"
        q = _q_down(theta)
        up = [x, y, TABLE_Z + PRE_H + GRASP_OFF] + q + [1.0]
        args = {"left": hold["left"], "right": hold["right"]}
        args[arm] = up
        return {"memory": mem, "act": {"tool": "act_move", "args": args}}

    if phase == "retreat":
        arm = mem["arm"]
        home = mem["home_left"] if arm == "left" else mem["home_right"]
        dx = -CLEAR_DX if arm == "left" else CLEAR_DX
        tgt = [home[0] + dx, home[1], home[2]] + home[3:] + [1.0]
        args = {"left": hold["left"], "right": hold["right"]}
        args[arm] = tgt
        mem["phase"] = "perceive"
        return {"memory": mem, "act": {"tool": "act_move", "args": args}}

    mem["log"] = "unknown phase " + str(phase)
    mem["phase"] = "perceive"
    return {"memory": mem, "act": {"tool": "act_move", "args": {"left": hold["left"], "right": hold["right"]}}}


def act_move(robot, left, right):
    """Move both arms in one native step. left/right: [x, y, z, qw, qx, qy, qz, gripper] (world, ee frame).
    Returns the reached ee positions and any motion error text."""
    a = [float(v) for v in left] + [float(v) for v in right]
    err = ""
    try:
        robot.step(*a)
    except Exception as e:
        err = str(e)[:200]
    now = robot.now()
    return {
        "left_ee": [round(float(v), 4) for v in now.left_ee_pose],
        "right_ee": [round(float(v), 4) for v in now.right_ee_pose],
        "left_gripper": float(now.left_gripper),
        "right_gripper": float(now.right_gripper),
        "motion_error": err,
    }
