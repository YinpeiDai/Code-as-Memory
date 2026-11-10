"""observe_and_pickup: remember the shelf object at reset, find its twin on the table, grasp top-down, lift."""

SURVEY_Z = 0.99          # ee height for the wrist-camera survey above a candidate (camera ~0.17 m above table)
PRE_DZ = 0.08            # pre-grasp clearance above the grasp pose (expert used ~0.08)
LIFT_DZ = 0.12           # first lift (expert lifted ~0.10)
LIFT2_DZ = 0.06          # extra lift if the episode has not ended
GRASP_OFFSET = 0.12      # grasp centre lies 0.12 m along the ee +x axis (pointing down when top-down)
MAX_TRIES_PER_TARGET = 2
MEM_KEYS = ("stage", "shelf", "cands", "cand_index", "target", "grasp", "arm", "tries", "attempts",
            "home", "refines", "fails", "tried", "outcome", "note")


def _quat_topdown(psi):
    c = math.cos(psi)
    s = math.sin(psi)
    R = [[0.0, c, s], [0.0, s, -c], [-1.0, 0.0, 0.0]]
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


def _psi_from_major(theta, L, W):
    """Finger-closing (ee y) axis across the object's minor axis, kept within [90, 270) deg like the expert."""
    if W <= 1e-6 or L / max(W, 1e-6) < 1.2:
        return math.pi
    psi = theta + math.pi / 2
    while psi < math.pi / 2:
        psi += math.pi
    while psi >= 1.5 * math.pi:
        psi -= math.pi
    return psi


def _grasp_height(h):
    """Grasp-centre height above the table from object height (fit below the expert's choices)."""
    return TABLE_Z + min(0.055, max(0.035, 0.035 + 0.5 * (h - 0.018)))


def _shelf_pack(s):
    return [float(s["h"]), float(s["L"]), float(s["W"])] + [float(v) for v in s["rgb"]] + \
        [float(v) for v in s["hist"]]


def _shelf_unpack(v):
    return {"h": v[0], "L": v[1], "W": v[2], "rgb": [v[3], v[4], v[5]], "hist": np.array(v[6:], np.float64)}


def _rank(objs, s):
    out = []
    for o in objs:
        c = _match_cost(s, o) if s is not None else 0.0
        out.append([float(o["cen"][0]), float(o["cen"][1]), float(c), float(o["h"]), float(o["L"]),
                    float(o["W"]), float(o["yaw"])])
    out.sort(key=lambda r: r[2])
    return out


def _wrist_pick(frame, arm, exp_xy, s, ee_z):
    """Cluster under the wrist camera that is nearest the expected target position."""
    cl = _wrist_clusters(frame, arm, min(TABLE_Z + 0.12, ee_z - 0.15), s)
    best = None
    for c in cl:
        if c["h"] > 0.085:
            continue
        d = math.hypot(c["top_cen"][0] - exp_xy[0], c["top_cen"][1] - exp_xy[1])
        d2 = math.hypot(c["cen"][0] - exp_xy[0], c["cen"][1] - exp_xy[1])
        d = min(d, d2)
        col = 0.0
        if s is not None:
            col = 1.0 - float(np.minimum(np.asarray(s["hist"]), c["hist"]).sum())
        score = d / 0.03 + 1.5 * col + (0.5 if c["n"] < 400 else 0.0)
        if d < 0.07 and (best is None or score < best[0]):
            best = (score, c, d)
    return best


def _last_ok(last):
    if last is None:
        return True
    return ('"ok": true' in last) and ("tool failed" not in last)


def perceive_decide(obs, goal, memory, last):
    """Stage machine: survey -> refine (wrist) -> descend -> close -> lift; on no success, release and retry."""
    m = {k: memory.get(k) for k in MEM_KEYS}
    cur = obs.current
    if m["stage"] is None:
        first = obs.frames[0]
        s = _shelf_object(first)
        objs = _table_objects(first, s)
        cands = _rank(objs, s)
        m.update(stage="start", shelf=_shelf_pack(s) if s is not None else None, cands=cands, cand_index=0,
                 tries=0, attempts=0, refines=0, fails=0, tried=[],
                 home=[float(v) for v in first.left_ee_pose] + [float(v) for v in first.right_ee_pose],
                 note="shelf h=%.3f L=%.3f W=%.3f; %d table candidates" % (
                     s["h"] if s else -1, s["L"] if s else -1, s["W"] if s else -1, len(cands)))
    s = _shelf_unpack(m["shelf"]) if m["shelf"] is not None else None
    stage = m["stage"]
    ok = _last_ok(last)

    def go(tool, args, nxt):
        m["stage"] = nxt
        return {"memory": m, "act": {"tool": tool, "args": args}}

    def start_target():
        if not m["cands"]:
            m["note"] = "no candidates"
            return go("act_hold", {}, "start")
        i = min(m["cand_index"], len(m["cands"]) - 1)
        c = m["cands"][i]
        arm = "left" if c[0] < 0 else "right"
        m["arm"] = arm
        m["target"] = [c[0], c[1] + 0.008]
        m["grasp"] = None
        m["refines"] = 0
        m["fails"] = 0
        return go("act_topdown", {"arm": arm, "x": m["target"][0], "y": m["target"][1], "z": SURVEY_Z,
                                  "psi": math.pi, "grip": 1.0}, "refine")

    if stage == "start":
        return start_target()

    arm = m["arm"]

    def reached(xyz, tol=0.02):
        ee = getattr(cur, arm + "_ee_pose")
        return math.sqrt(sum((float(ee[i]) - xyz[i]) ** 2 for i in range(3))) < tol

    if stage == "refine":
        if not reached([m["target"][0], m["target"][1], float(getattr(cur, arm + "_ee_pose")[2])], 0.03):
            m["fails"] += 1
            if m["fails"] == 1:
                return go("act_topdown", {"arm": arm, "x": m["target"][0], "y": m["target"][1],
                                          "z": SURVEY_Z + 0.04, "psi": math.pi, "grip": 1.0}, "refine")
            if m["fails"] == 2:
                m["arm"] = "right" if arm == "left" else "left"
                return go("act_topdown", {"arm": m["arm"], "x": m["target"][0], "y": m["target"][1],
                                          "z": SURVEY_Z, "psi": math.pi, "grip": 1.0}, "refine")
            pick = None
        else:
            ee = getattr(cur, arm + "_ee_pose")
            pick = _wrist_pick(cur, arm, m["target"], s, float(ee[2]))
        c = m["cands"][min(m["cand_index"], len(m["cands"]) - 1)]
        if pick is not None:
            o = pick[1]
            x, y = o["top_cen"]
            h, L, W, yaw = o["h"], o["L"], o["W"], o["yaw"]
            moved = math.hypot(x - m["target"][0], y - m["target"][1])
            if moved > 0.025 and m["refines"] < 1:
                m["refines"] += 1
                m["target"] = [x, y]
                return go("act_topdown", {"arm": arm, "x": x, "y": y, "z": SURVEY_Z, "psi": math.pi,
                                          "grip": 1.0}, "refine")
            src = "wrist"
        else:
            x, y = m["target"]
            h, L, W, yaw = c[3], c[4], c[5], c[6]
            src = "head"
        gz = max(TABLE_Z + 0.033, _grasp_height(h) - 0.008 * m["tries"])
        psi = _psi_from_major(yaw, L, W)
        m["grasp"] = [x, y, gz, psi]
        m["fails"] = 0
        m["note"] = "grasp from %s at (%.3f, %.3f, %.3f) psi=%.0f h=%.3f arm=%s" % (
            src, x, y, gz, math.degrees(psi), h, arm)
        return go("act_topdown", {"arm": arm, "x": x, "y": y, "z": gz + GRASP_OFFSET + PRE_DZ, "psi": psi,
                                  "grip": 1.0}, "descend")
    g = m["grasp"]
    if stage == "descend":
        if not reached([g[0], g[1], g[2] + GRASP_OFFSET + PRE_DZ]) and m["fails"] < 1:
            m["fails"] += 1
            g[3] = g[3] + math.pi if g[3] < math.pi else g[3] - math.pi
            m["grasp"] = g
            return go("act_topdown", {"arm": arm, "x": g[0], "y": g[1], "z": g[2] + GRASP_OFFSET + PRE_DZ,
                                      "psi": g[3], "grip": 1.0}, "descend")
        m["fails"] = 0
        return go("act_topdown", {"arm": arm, "x": g[0], "y": g[1], "z": g[2] + GRASP_OFFSET, "psi": g[3],
                                  "grip": 1.0}, "close")
    if stage == "close" and not reached([g[0], g[1], g[2] + GRASP_OFFSET], 0.012) and m["fails"] < 1:
        m["fails"] += 1
        g[2] = g[2] + 0.008
        m["grasp"] = g
        return go("act_topdown", {"arm": arm, "x": g[0], "y": g[1], "z": g[2] + GRASP_OFFSET, "psi": g[3],
                                  "grip": 1.0}, "close")
    if stage == "close":
        return go("act_gripper", {"arm": arm, "grip": 0.0}, "lift")
    if stage == "lift":
        return go("act_topdown", {"arm": arm, "x": g[0], "y": g[1], "z": g[2] + GRASP_OFFSET + LIFT_DZ,
                                  "psi": g[3], "grip": 0.0}, "lift2")
    if stage == "lift2":
        return go("act_topdown", {"arm": arm, "x": g[0], "y": g[1],
                                  "z": g[2] + GRASP_OFFSET + LIFT_DZ + LIFT2_DZ, "psi": g[3], "grip": 0.0},
                  "release")
    if stage == "release":
        # still running after the lift: classify the attempt from the wrist view, then clear the scene
        n_left = _table_points_near(cur, arm, g[:2])
        above = reached([g[0], g[1], float(getattr(cur, arm + "_ee_pose")[2])], 0.03)
        m["outcome"] = "missed" if (n_left > 300 or not above) else "held"
        m["attempts"] += 1
        m["tried"] = m["tried"] + [[g[0], g[1]]]
        m["note"] = "attempt %d: %s (%d table points under gripper)" % (m["attempts"], m["outcome"], n_left)
        if m["outcome"] == "missed":
            return go("act_open_home", {"arm": arm, "home": m["home"]}, "reassess")
        return go("act_release_home", {"arm": arm, "x": g[0], "y": g[1], "z": g[2] + GRASP_OFFSET + 0.01,
                                       "psi": g[3], "home": m["home"]}, "reassess")
    if stage == "reassess":
        objs = _table_objects(cur, s)
        cands = _rank(objs, s)
        last_xy = m["tried"][-1] if m["tried"] else None
        same = None
        if last_xy is not None:
            best_d = 0.06
            for i, c in enumerate(cands):
                dd = math.hypot(c[0] - last_xy[0], c[1] - last_xy[1])
                if dd < best_d:
                    best_d = dd
                    same = i
        if m["outcome"] == "missed" and same is not None and m["tries"] + 1 < MAX_TRIES_PER_TARGET:
            m["tries"] += 1
            m["cands"] = [cands[same]] + [c for i, c in enumerate(cands) if i != same]
        else:
            m["tries"] = 0
            rest = [c for c in cands if all(math.hypot(c[0] - t[0], c[1] - t[1]) > 0.04 for t in m["tried"])]
            m["cands"] = rest if rest else cands
        m["cand_index"] = 0
        return start_target()
    return go("act_hold", {}, stage)


def _hold_pose(robot, arm):
    f = robot.now()
    return [float(v) for v in getattr(f, arm + "_ee_pose")], float(getattr(f, arm + "_gripper"))


def _step(robot, left, lg, right, rg):
    try:
        robot.step(*left, lg, *right, rg)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def act_topdown(robot, arm, x, y, z, psi, grip):
    """Move `arm` ('left'/'right') to a top-down pose (ee x-axis down) at world (x, y, z) with finger-axis yaw
    psi (rad) and gripper opening grip (0 closed, 1 open); the other arm holds. Returns ok and the reached pose."""
    q = _quat_topdown(float(psi))
    target = [float(x), float(y), float(z)] + q
    other = "right" if arm == "left" else "left"
    op, og = _hold_pose(robot, other)
    if arm == "left":
        r = _step(robot, target, float(grip), op, og)
    else:
        r = _step(robot, op, og, target, float(grip))
    p = [float(v) for v in getattr(robot.now(), arm + "_ee_pose")]
    r["pose"] = [round(v, 4) for v in p]
    r["pos_err"] = round(math.sqrt(sum((p[i] - target[i]) ** 2 for i in range(3))), 4)
    return r


def act_gripper(robot, arm, grip):
    """Set `arm`'s gripper opening (0 closed, 1 open) while both arms hold their poses."""
    lp, lg = _hold_pose(robot, "left")
    rp, rg = _hold_pose(robot, "right")
    if arm == "left":
        lg = float(grip)
    else:
        rg = float(grip)
    return _step(robot, lp, lg, rp, rg)


def act_release_home(robot, arm, x, y, z, psi, home):
    """Lower `arm` to (x, y, z) top-down, open the gripper, rise, then return both arms to the home poses."""
    out = act_topdown(robot, arm, x, y, z, psi, 0.0)
    act_gripper(robot, arm, 1.0)
    act_topdown(robot, arm, x, y, z + 0.10, psi, 1.0)
    r = _step(robot, [float(v) for v in home[:7]], 1.0, [float(v) for v in home[7:]], 1.0)
    r["lowered"] = out.get("ok", False)
    return r


def act_open_home(robot, arm, home):
    """Open `arm`'s gripper, then return both arms to the home poses (clears the head camera's view)."""
    act_gripper(robot, arm, 1.0)
    return _step(robot, [float(v) for v in home[:7]], 1.0, [float(v) for v in home[7:]], 1.0)


def act_hold(robot):
    """Hold both arms in place for one native action."""
    lp, lg = _hold_pose(robot, "left")
    rp, rg = _hold_pose(robot, "right")
    return _step(robot, lp, lg, rp, rg)
