"""T15 policy: pick the butter into the frypan, then grasp the milk carton and
pour it over the pan twice (tilt, untilt, tilt). Targets are computed from the
perceived objects with offsets learned from the 50 assigned demonstrations."""

# Offsets learned offline (scratch/reg2.py, scratch/milkpath.py); heights are
# relative to the perceived table plane (~0.431 m).
BUTTER_GRASP_OFF = [-0.0141, 0.0019]     # grasp xy - perceived butter centroid
BUTTER_GRASP_DZ = 0.0023
BUTTER_HOVER_DZ = 0.21
CARRY_DZ = 0.40
RELEASE_OFF = [-0.0008, 0.0141]         # release xy - perceived pan centre
RELEASE_DZ = 0.181
MILK_GRASP_OFF = [-0.0286, -0.0011]      # grasp xy - perceived milk centroid
MILK_GRASP_DZ = 0.0134
MILK_PRE_DX = 0.022                      # pre-grasp: offset along +x, same height
MILK_HOVER_DX = 0.030
MILK_HOVER_DZ = 0.23
MILK_LIFT_DZ = 0.35
POUR_OFF = [-0.0108, -0.0104]            # tcp xy before pouring - pan centre
POUR_DZ = 0.264
POUR_HOLD = 50
FINAL_HOLD = 150
MAX_DECISIONS = 40
# fallbacks (mean perception over the demos) used only when detection fails
DEFAULT = {"table_z": 0.4309, "milk": [-0.0798, -0.1729], "pan": [0.0023, 0.1227],
           "butter": [0.1236, -0.2044]}
HOME = [-0.05, 0.0, 0.67]
BUTTER_GAP_MIN = 0.012   # finger gap (sum, m) when holding butter ~0.039
MILK_GAP_MIN = 0.020     # holding milk ~0.053; empty close ~0


def _perceive_into(mem, frame, which):
    s = _scene(frame, frame.tcp)
    tz = s["table_z"]
    if not (0.38 < tz < 0.48):
        tz = DEFAULT["table_z"]
    mem["table_z"] = tz
    notes = []
    for k in which:
        v = s.get(k)
        ok = v is not None
        if ok and k == "milk":
            ok = s["milk_n"] >= 30
        if ok and k == "pan":
            ok = s["pan_n"] >= 60 and 0.075 < (s["pan_r"] or 0) < 0.13
        if ok and k == "butter":
            ok = s["butter_n"] >= 15
        if ok:
            mem[k] = [float(v[0]), float(v[1])]
            notes.append(k + " seen")
        elif not mem.get(k):
            mem[k] = list(DEFAULT[k])
            notes.append(k + " default")
        else:
            notes.append(k + " kept")
    return ", ".join(notes)


def _move(x, y, z, ori="down", grip=-1.0, tol=0.004, max_steps=300, vmax=1.0, ki=1.0):
    return {"tool": "act_move", "args": {"x": float(x), "y": float(y), "z": float(z), "ori": ori,
                                         "grip": float(grip), "tol": float(tol),
                                         "max_steps": int(max_steps), "vmax": float(vmax),
                                         "ki": float(ki)}}


def _targets(m):
    tz = m["table_z"]
    b = m["butter"]
    pan = m["pan"]
    milk = m["milk"]
    bg = [b[0] + BUTTER_GRASP_OFF[0], b[1] + BUTTER_GRASP_OFF[1]]
    rel = [pan[0] + RELEASE_OFF[0], pan[1] + RELEASE_OFF[1]]
    mg = [milk[0] + MILK_GRASP_OFF[0], milk[1] + MILK_GRASP_OFF[1]]
    pour = [pan[0] + POUR_OFF[0], pan[1] + POUR_OFF[1]]
    return tz, bg, rel, mg, pour


def perceive_decide(obs, goal, memory, last):
    """Stage machine. Memory keeps the perceived object positions (taken once
    at the start, re-taken after a failed grasp), the current stage, retry and
    pour counters. Each decision issues one act_* call for the current stage."""
    m = {
        "stage": memory.get("stage") or "start",
        "table_z": memory.get("table_z") or DEFAULT["table_z"],
        "butter": memory.get("butter") or [],
        "pan": memory.get("pan") or [],
        "milk": memory.get("milk") or [],
        "butter_tries": int(memory.get("butter_tries") or 0),
        "milk_tries": int(memory.get("milk_tries") or 0),
        "pours": int(memory.get("pours") or 0),
        "decisions": int(memory.get("decisions") or 0) + 1,
        "note": "",
    }
    f = obs.current
    gap = _finger_gap(f)
    p, _ = _pose(f)
    st = m["stage"]
    note = ""

    if st == "start" or st == "reperceive_butter" or st == "reperceive_milk":
        which = ["butter", "pan", "milk"] if st == "start" else (["butter"] if st == "reperceive_butter" else ["milk"])
        note = _perceive_into(m, f, which)
        st = "b_above" if st != "reperceive_milk" else "m_above"

    tz, bg, rel, mg, pour = _targets(m)
    hx, hy = mg[0] + MILK_HOVER_DX, mg[1]

    # grasp checks (before using the result of a close)
    if st == "b_lift" and gap < BUTTER_GAP_MIN:
        note += " butter grasp missed (gap %.4f)" % gap
        st = "b_retry"
    if st in ("b_carry", "b_lower") and gap < BUTTER_GAP_MIN:
        note += " butter dropped (gap %.4f)" % gap
        st = "b_retry"
    if st == "m_lift" and gap < MILK_GAP_MIN:
        note += " milk grasp missed (gap %.4f)" % gap
        st = "m_retry"
    if st in ("m_carry", "pour1") and gap < MILK_GAP_MIN:
        note += " milk dropped (gap %.4f)" % gap
        st = "m_retry"
    if st == "b_retry" and m["butter_tries"] >= 2:
        st = "m_above"   # give up on butter; still attempt the pour
        note += " butter retries exhausted"
    if st == "m_retry" and m["milk_tries"] >= 2:
        st = "idle"
        note += " milk retries exhausted"

    if m["decisions"] >= MAX_DECISIONS:
        st = "idle"

    nxt = st
    if st == "b_above":
        act = _move(bg[0], bg[1], tz + BUTTER_HOVER_DZ, "down", -1)
        nxt = "b_down"
    elif st == "b_down":
        act = _move(bg[0], bg[1], tz + BUTTER_GRASP_DZ, "down", -1, tol=0.003, vmax=0.6, ki=0.0)
        nxt = "b_close"
    elif st == "b_close":
        act = {"tool": "act_hold", "args": {"steps": 30, "grip": 1.0}}
        nxt = "b_lift"
    elif st == "b_lift":
        act = _move(bg[0], bg[1], tz + 0.32, "down", 1)
        nxt = "b_carry"
    elif st == "b_carry":
        act = _move(rel[0], rel[1], tz + CARRY_DZ, "down", 1)
        nxt = "b_lower"
    elif st == "b_lower":
        act = _move(rel[0], rel[1], tz + RELEASE_DZ, "down", 1, tol=0.004, vmax=0.8)
        nxt = "b_open"
    elif st == "b_open":
        act = {"tool": "act_hold", "args": {"steps": 25, "grip": -1.0}}
        nxt = "b_up"
    elif st == "b_up":
        act = _move(rel[0], rel[1], tz + 0.35, "down", -1, tol=0.01)
        nxt = "m_above"
    elif st == "b_retry":
        m["butter_tries"] += 1
        act = _move(HOME[0], HOME[1], max(HOME[2], p[2] + 0.05), "down", -1, tol=0.01)
        nxt = "reperceive_butter"
    elif st == "m_above":
        act = _move(hx, hy, tz + CARRY_DZ, "down", -1, tol=0.006)
        nxt = "m_tilt"
    elif st == "m_tilt":
        act = _move(hx, hy, tz + MILK_GRASP_DZ + MILK_HOVER_DZ, "grasp", -1, tol=0.004)
        nxt = "m_down"
    elif st == "m_down":
        act = _move(mg[0] + MILK_PRE_DX, mg[1], tz + MILK_GRASP_DZ, "grasp", -1, tol=0.003, vmax=0.6, ki=0.0)
        nxt = "m_in"
    elif st == "m_in":
        act = _move(mg[0], mg[1], tz + MILK_GRASP_DZ, "grasp", -1, tol=0.003, max_steps=120, vmax=0.4, ki=0.0)
        nxt = "m_close"
    elif st == "m_close":
        act = {"tool": "act_hold", "args": {"steps": 20, "grip": 1.0}}
        nxt = "m_lift"
    elif st == "m_lift":
        act = _move(mg[0], mg[1], tz + MILK_LIFT_DZ, "grasp", 1, tol=0.006, vmax=0.8)
        nxt = "m_carry"
    elif st == "m_carry":
        act = _move(pour[0], pour[1], tz + POUR_DZ, "grasp", 1, tol=0.004, vmax=0.8)
        nxt = "pour1"
    elif st == "pour1":
        act = {"tool": "act_rotate", "args": {"ori": "pour", "grip": 1.0}}
        m["pours"] += 1
        nxt = "hold1"
    elif st == "hold1":
        act = {"tool": "act_hold", "args": {"steps": POUR_HOLD, "grip": 1.0}}
        nxt = "back"
    elif st == "back":
        act = {"tool": "act_rotate", "args": {"ori": "grasp", "grip": 1.0, "ori_tol_deg": 1.5, "settle": 15}}
        nxt = "pour2"
    elif st == "pour2":
        act = {"tool": "act_rotate", "args": {"ori": "pour", "grip": 1.0}}
        m["pours"] += 1
        nxt = "hold2"
    elif st == "hold2":
        act = {"tool": "act_hold", "args": {"steps": FINAL_HOLD, "grip": 1.0}}
        # success normally ends the episode during this hold; if not, pour again
        nxt = "back" if m["pours"] < 4 else "idle"
    elif st == "m_retry":
        m["milk_tries"] += 1
        act = _move(p[0], p[1], tz + CARRY_DZ, "grasp", -1, tol=0.01, max_steps=150)
        nxt = "m_reset"
    elif st == "m_reset":
        act = _move(HOME[0], HOME[1], HOME[2], "down", -1, tol=0.01)
        nxt = "reperceive_milk"
    else:
        hold_grip = 1.0 if gap > MILK_GAP_MIN else -1.0
        act = {"tool": "act_hold", "args": {"steps": 200, "grip": hold_grip}}
        nxt = "idle"
    m["stage"] = nxt
    m["note"] = (st + ": " + note.strip())[:300]
    return {"memory": m, "act": act}
