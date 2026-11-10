"""Decision logic: sequence sauce -> milk -> orange juice, with survey / pick / place / verify stages."""

FIELDS = ["stage", "idx", "cab1", "cab2", "top", "target", "grasp_tries", "survey_tries",
          "placed", "note", "drops", "verify_tries", "place_dx"]
CAB1_FALLBACK = [0.0, -0.19, 0.663]
CAB2_FALLBACK = [0.0, 0.27, 0.663]


def _mem(memory):
    m = {}
    for k in FIELDS:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["idx"] is None:
        m["idx"] = 0
    for k in ("grasp_tries", "survey_tries", "drops", "verify_tries"):
        if m[k] is None:
            m[k] = 0
    if m["placed"] is None:
        m["placed"] = []
    if m["top"] is None:
        m["top"] = CAB_TOP_DEFAULT
    if m["place_dx"] is None:
        m["place_dx"] = 0.0
    return m


def _survey_pose(cab, k):
    """k-th survey viewpoint above a cabinet (alternates height / offset for re-observation)."""
    poses = [(-0.02, 0.0, SURVEY_Z), (-0.06, 0.0, 1.05), (0.02, 0.03, 1.0), (-0.02, -0.03, 1.0)]
    dx, dy, z = poses[k % len(poses)]
    return {"x": round(cab[0] + dx, 4), "y": round(cab[1] + dy, 4), "z": z}


def _out(m, tool, args):
    return {"memory": {k: m[k] for k in FIELDS}, "act": {"tool": tool, "args": args}}


def _go_survey(m, why):
    m["stage"] = "survey"
    m["note"] = why
    return _out(m, "act_survey", _survey_pose(m["cab1"], m["survey_tries"]))


def _place_args(m, name):
    """Release pose over cabinet2 for `name`, with a transit height that clears the cabinets'
    contents but keeps the arm away from the straight-elbow singularity."""
    off = PLACE_OFF[name]
    t = m["target"] or [0, 0, m["top"] + 0.066]
    grasp_z = _grasp_z(t, m["top"])
    rel = grasp_z - m["top"] + m["cab2"][2] + 0.03
    x = m["cab2"][0] + off[0] + m["place_dx"]
    y = m["cab2"][1] + off[1]
    clear = grasp_z + 0.13
    # pull the target toward the base (within the cabinet top) if it is out of comfortable reach
    while _reach_z(x, y) < clear and x > m["cab2"][0] - 0.08:
        x -= 0.01
    tz = max(clear, min(0.92, _reach_z(x, y)))
    return {"x": round(x, 4), "y": round(y, 4), "release_z": round(rel, 4), "transit_z": round(tz, 4)}


def _grasp_z(t, top):
    """Fingers ~3 cm down the object's side, never closer than 3.6 cm (tcp) to the cabinet top."""
    return max(float(t[2]) - 0.032, top + 0.036)


def perceive_decide(obs, goal, memory, last):
    """Plan the next act from the current frame and episode memory (stage machine)."""
    m = _mem(memory)
    cur = obs.current
    last = last or ""

    # ---- first decision: find both cabinets in the front view
    if m["stage"] is None or m["cab1"] is None or m["cab2"] is None:
        c1, c2, top = _detect_cabinets(cur)
        m["cab1"] = c1 if c1 is not None else CAB1_FALLBACK
        m["cab2"] = c2 if c2 is not None else CAB2_FALLBACK
        m["top"] = top
        m["survey_tries"] = 0
        return _go_survey(m, "cabinets c1=%s c2=%s" % ("ok" if c1 else "fallback", "ok" if c2 else "fallback"))

    if m["idx"] >= len(ORDER):
        # all three released on cabinet2: verify from above cabinet2, then wait for the scorer
        if m["stage"] == "verify":
            dets, _ = _detect_objects(cur, "wrist", m["cab2"], m["cab2"][2])
            missing = [n for n in ORDER if n not in dets]
            m["verify_tries"] += 1
            if missing and m["verify_tries"] <= 2:
                # look for the missing one back on cabinet1
                d1, _ = _detect_objects(cur, "agentview", m["cab1"], m["top"])
                for n in missing:
                    if n in d1:
                        m["idx"] = ORDER.index(n)
                        m["placed"] = [p for p in m["placed"] if p != n]
                        m["survey_tries"] = 0
                        m["grasp_tries"] = 0
                        return _go_survey(m, "verify: %s still on cabinet1" % n)
            m["stage"] = "wait"
            m["note"] = "verify missing=%s" % (missing,)
            return _out(m, "act_wait", {"steps": 40})
        if m["stage"] != "wait":
            m["stage"] = "verify"
            p = _survey_pose(m["cab2"], 0)
            return _out(m, "act_survey", p)
        return _out(m, "act_wait", {"steps": 60})

    name = ORDER[m["idx"]]

    if m["stage"] == "survey":
        dets, counts = _detect_objects(cur, "wrist", m["cab1"], m["top"])
        src = "wrist"
        if name not in dets and m["survey_tries"] >= 1:
            d2, _ = _detect_objects(cur, "agentview", m["cab1"], m["top"])
            if name in d2:
                dets = d2
                src = "agentview"
        if name in dets:
            t = dets[name]
            m["target"] = [round(t[0], 4), round(t[1], 4), round(t[2], 4)]
            gz = _grasp_z(t, m["top"])
            cam = "wrist" if src == "wrist" else "agentview"
            P = _object_points(cur, cam, m["cab1"], m["top"])
            yaws = (0.0, math.pi / 2, math.pi / 4, 3 * math.pi / 4) if name == "sauce" else (0.0, math.pi / 2)
            gx, gy, yaw, cost, c0 = _choose_grasp(P, t, gz, yaws)
            m["stage"] = "pick"
            m["note"] = "%s at %s via %s (n=%d) try %d; grasp (%.3f,%.3f) yaw %.2f finger-hits %d (centre %d)" % (
                name, m["target"], src, t[3], m["grasp_tries"], gx, gy, yaw, cost, c0)
            lift = max(gz + 0.13, min(0.92, _reach_z(gx, gy)))
            return _out(m, "act_pick", {"x": round(gx, 4), "y": round(gy, 4), "grasp_z": round(gz, 4),
                                        "yaw": round(yaw, 4), "lift_z": round(lift, 4)})
        m["survey_tries"] += 1
        if m["survey_tries"] <= 3:
            return _go_survey(m, "%s not seen %s" % (name, counts))
        # not on cabinet1 after several views: check cabinet2 (it may already be there)
        m["stage"] = "check2"
        m["note"] = "%s not on cabinet1 after %d views" % (name, m["survey_tries"])
        return _out(m, "act_survey", _survey_pose(m["cab2"], 0))

    if m["stage"] == "check2":
        dets, _ = _detect_objects(cur, "wrist", m["cab2"], m["cab2"][2])
        if name in dets:
            m["placed"] = m["placed"] + [name]
            m["idx"] += 1
            m["survey_tries"] = 0
            m["grasp_tries"] = 0
            if m["idx"] >= len(ORDER):
                m["stage"] = "verify"
                return _out(m, "act_survey", _survey_pose(m["cab2"], 0))
            return _go_survey(m, "%s found on cabinet2" % name)
        m["survey_tries"] = 0
        return _go_survey(m, "%s not on cabinet2 either; re-survey cabinet1" % name)

    if m["stage"] == "pick":
        held = _held(cur)
        if held:
            m["stage"] = "place"
            m["note"] = "holding %s gap=%.4f" % (name, _gap(cur))
            return _out(m, "act_place", _place_args(m, name))
        m["grasp_tries"] += 1
        m["survey_tries"] = 0
        return _go_survey(m, "grasp of %s missed (gap=%.4f)" % (name, _gap(cur)))

    if m["stage"] == "place":
        if '"limit": true' in last and '"transit"' in last and _held(cur) and m["place_dx"] > -0.08:
            # elbow near its straight limit before arrival: retry closer to the robot base
            m["place_dx"] = m["place_dx"] - 0.04
            m["note"] = "elbow limit in transit; place_dx=%.2f" % m["place_dx"]
            return _out(m, "act_place", _place_args(m, name))
        m["place_dx"] = 0.0
        if '"dropped": true' in last:
            m["drops"] += 1
            m["survey_tries"] = 0
            return _go_survey(m, "%s dropped during transport" % name)
        m["placed"] = m["placed"] + [name]
        m["idx"] += 1
        m["survey_tries"] = 0
        m["grasp_tries"] = 0
        if m["idx"] >= len(ORDER):
            m["stage"] = "verify"
            m["note"] = "all released"
            return _out(m, "act_survey", _survey_pose(m["cab2"], 0))
        return _go_survey(m, "%s released on cabinet2" % name)

    # unknown stage: re-observe cabinet1
    m["survey_tries"] = 0
    return _go_survey(m, "reset from stage %s" % m["stage"])
