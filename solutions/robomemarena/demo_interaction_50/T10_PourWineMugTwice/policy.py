"""Pour wine into the mug twice: detect bottle and mug from agentview RGB-D,
side-grasp the bottle, carry it next to the mug, tilt it horizontal over the
mug, return it upright, and tilt again; keep holding until the scorer ends."""

# Offsets learned from 50 demos (expert TCP minus detected object surface median).
GRASP_OFF = [-0.0204, -0.0038]
GRASP_Z = 0.466
APPROACH_Z = 0.80
PREGRASP_Z = 0.53
LIFT_Z = 0.75
POUR_OFFS = [[-0.0125, -0.1178, 0.6553], [-0.0098, -0.1023, 0.6376]]
PRIOR_BOTTLE = [-0.151, 0.237, 0.493]
PRIOR_MUG = [0.041, 0.261, 0.530]
HOME = [-0.06, 0.0, 0.75]
HOLD_GAP = 0.012      # finger gap (m) above which the bottle is considered held
FIRST_HOLD = 45       # steps the first pour is held (demo: ~40 after tilting)
LATER_HOLD = 160      # steps later pours are held while waiting for the scorer
MAX_POURS = 4
MAX_GRASPS = 3
MEM_FIELDS = ("stage", "bottle", "mug", "pours", "grasps", "misses", "note", "detect_ok")

STAGES = ["start", "approach", "pregrasp", "descend", "close", "lift", "transport",
          "tilt", "upright", "retreat", "done"]


def _mem(memory):
    m = dict(memory or {})
    for k in MEM_FIELDS:
        if k not in m:
            m[k] = None
    if m["pours"] is None:
        m["pours"] = 0
    if m["grasps"] is None:
        m["grasps"] = 0
    if m["misses"] is None:
        m["misses"] = 0
    return m


def _grasp_xy(m):
    b = m["bottle"] or PRIOR_BOTTLE
    # after a missed closure, shift the grasp slightly deeper along the approach
    # axis (+y) so a retry is not an identical repeat
    extra = 0.006 * int(m["misses"] or 0)
    return [b[0] + GRASP_OFF[0], b[1] + GRASP_OFF[1] + extra]


def _pour_target(m, k):
    mg = m["mug"] or PRIOR_MUG
    off = POUR_OFFS[min(k, len(POUR_OFFS) - 1)]
    return [mg[0] + off[0], mg[1] + off[1], off[2]]


def _servo(target, rot, gripper, max_steps=150, pos_tol=0.004, rot_tol=0.04, hold_steps=0,
           vmax=1.0, stall_ok=False):
    return {"tool": "act_servo", "args": {"target": [float(x) for x in target], "rot": rot,
            "gripper": float(gripper), "max_steps": int(max_steps), "pos_tol": float(pos_tol),
            "rot_tol": float(rot_tol), "hold_steps": int(hold_steps), "vmax": float(vmax),
            "stall_ok": bool(stall_ok)}}


def _out(m, act):
    return {"memory": {k: m[k] for k in MEM_FIELDS},
            "act": act}


def _detect_into(m, frame):
    det = None
    try:
        det = _detect_objects(frame)
    except (ValueError, IndexError, TypeError, KeyError) as e:
        det = None
    ok = []
    if det is not None and det.get("bottle") is not None:
        b = det["bottle"]
        # sanity: on the table, near the prior work area
        if abs(b[0] - PRIOR_BOTTLE[0]) < 0.15 and abs(b[1] - PRIOR_BOTTLE[1]) < 0.20:
            m["bottle"] = b
            ok.append("bottle")
    if det is not None and det.get("mug") is not None:
        g = det["mug"]
        if abs(g[0] - PRIOR_MUG[0]) < 0.15 and abs(g[1] - PRIOR_MUG[1]) < 0.20:
            if m["mug"] is None or m["stage"] in (None, "start"):
                m["mug"] = g
            ok.append("mug")
    m["detect_ok"] = ",".join(ok) if ok else "none"
    return m


def perceive_decide(obs, goal, memory, last):
    """Stage machine; every stage re-checks measured TCP / finger gap."""
    m = _mem(memory)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    gap = float(np.sum(np.abs(np.asarray(cur.fingers, dtype=float))))
    holding = gap > HOLD_GAP
    st = m["stage"] or "start"

    if st == "start":
        m = _detect_into(m, cur)
        if m["bottle"] is None:
            m["note"] = "bottle not detected; using prior position"
        if m["mug"] is None:
            m["note"] = (m["note"] or "") + " mug not detected; using prior"
        gx, gy = _grasp_xy(m)
        m["stage"] = "approach"
        return _out(m, _servo([gx, gy, APPROACH_Z], "grasp", -1.0, max_steps=200,
                              pos_tol=0.006, rot_tol=0.05))

    if st == "approach":
        gx, gy = _grasp_xy(m)
        m["stage"] = "pregrasp"
        return _out(m, _servo([gx, gy, PREGRASP_Z], "grasp", -1.0, max_steps=150,
                              pos_tol=0.004, rot_tol=0.04))

    if st == "pregrasp":
        gx, gy = _grasp_xy(m)
        m["stage"] = "descend"
        return _out(m, _servo([gx, gy, GRASP_Z], "grasp", -1.0, max_steps=80,
                              pos_tol=0.004, rot_tol=0.04, hold_steps=3, vmax=0.5, stall_ok=True))

    if st == "descend":
        m["stage"] = "close"
        m["grasps"] = int(m["grasps"]) + 1
        return _out(m, {"tool": "act_gripper", "args": {"close": True, "steps": 14, "rot": "grasp"}})

    if st == "close":
        if holding:
            gx, gy = _grasp_xy(m)
            m["stage"] = "lift"
            return _out(m, _servo([gx, gy, LIFT_Z], "grasp", 1.0, max_steps=120,
                                  pos_tol=0.008, rot_tol=0.05))
        m["note"] = "grasp missed (gap %.3f)" % gap
        m["misses"] = int(m["misses"]) + 1
        m["stage"] = "retreat"
        return _out(m, _servo([tcp[0], tcp[1], APPROACH_Z], "grasp", -1.0, max_steps=80,
                              pos_tol=0.01, rot_tol=0.1))

    if st in ("lift", "transport", "tilt", "upright") and not holding:
        m["note"] = "bottle lost during %s (gap %.3f)" % (st, gap)
        m["stage"] = "retreat"
        return _out(m, _servo([tcp[0], tcp[1], max(tcp[2], LIFT_Z)], "grasp", -1.0,
                              max_steps=80, pos_tol=0.01, rot_tol=0.1))

    if st == "lift":
        m["stage"] = "transport"
        return _out(m, _servo(_pour_target(m, 0), "grasp", 1.0, max_steps=200,
                              pos_tol=0.004, rot_tol=0.04))

    if st == "transport" or st == "upright":
        k = int(m["pours"])
        if k >= MAX_POURS:
            m["stage"] = "done"
            return _out(m, _servo(_pour_target(m, k), "grasp", 1.0, max_steps=400,
                                  pos_tol=0.0, rot_tol=0.0))
        m["stage"] = "tilt"
        hold = FIRST_HOLD if k == 0 else LATER_HOLD
        return _out(m, _servo(_pour_target(m, k), "pour", 1.0, max_steps=hold + 120,
                              pos_tol=0.006, rot_tol=0.06, hold_steps=hold))

    if st == "tilt":
        m["pours"] = int(m["pours"]) + 1
        m["stage"] = "upright"
        return _out(m, _servo(_pour_target(m, int(m["pours"])), "grasp", 1.0, max_steps=120,
                              pos_tol=0.006, rot_tol=0.06, hold_steps=8))

    if st == "retreat":
        if int(m["grasps"]) >= MAX_GRASPS:
            m["stage"] = "done"
            m["note"] = (m["note"] or "") + "; grasp attempts exhausted"
            return _out(m, _servo(HOME, "grasp", -1.0, max_steps=200, pos_tol=0.0, rot_tol=0.0))
        m["stage"] = "redetect"
        return _out(m, _servo(HOME, "down", -1.0, max_steps=150, pos_tol=0.01, rot_tol=0.08))

    if st == "redetect":
        m["stage"] = "start"
        m["pours"] = 0
        m = _detect_into(m, cur)
        gx, gy = _grasp_xy(m)
        m["stage"] = "approach"
        return _out(m, _servo([gx, gy, APPROACH_Z], "grasp", -1.0, max_steps=200,
                              pos_tol=0.006, rot_tol=0.05))

    # done / unknown: hold pose
    m["stage"] = "done"
    return _out(m, _servo(tcp[:3], "keep", 1.0 if holding else -1.0, max_steps=200,
                          pos_tol=0.0, rot_tol=0.0))
