"""T05_ButterEmptyDrawer policy: check top, middle, bottom drawers (open+close each, in order), then put the
butter into a drawer judged empty. Perception: agentview RGB-D (cabinet top surface -> handle poses; butter blob
near its table location; coloured pixels inside each opened drawer)."""

DRAWERS = ("top", "middle", "bottom")
# Preference among drawers with no visible contents (demos: middle chosen 40/50, bottom 10/50, never top).
EMPTY_PREFERENCE = ("middle", "bottom", "top")
MAX_TRIES = 3
EXPLORE_MISS_DY = 0.0  # exploratory offset for the first top grasp (rollouts 13-15 used -0.05, -0.05, -0.02)
EXPLORE_BUTTER_DY = 0.0  # exploratory offset of the first butter grasp (rollouts 20-21 used 0.045)
EXPLORE_FIRST = None  # exploratory first drawer (rollouts 10, 24, 26, 27, 29 used "top"; 35 used "bottom")
# r10: no high carry any more (0.82-0.88 m above y~0.3-0.4 drove elbow joint 4 to its limit, R027); the butter
# is carried at floor+0.10 and inserted INSERT_DY behind the open drawer's handle.
INSERT_DY = 0.075
CABINET_PRIOR = (0.135, 0.465, 0.713)
# Carry/insert yaw: hand housing spans the drawer width (x) instead of reaching the closed drawer above.
INSERT_YAW = PI / 2


def _num(text, key, default=None):
    """Read a number/bool written as "key": value in an act result string."""
    if not isinstance(text, str):
        return default
    k = '"' + key + '": '
    i = text.find(k)
    if i < 0:
        return default
    s = text[i + len(k):]
    j = 0
    while j < len(s) and s[j] not in ",}]":
        j += 1
    v = s[:j].strip()
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "null":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _mem_default(memory):
    keys = ("stage", "cabinet", "butter", "butter_yaw", "butter_top", "checked", "visible", "tries", "target",
            "tried", "open_y", "holding", "note", "pending")
    m = {k: memory.get(k) for k in keys}
    m["checked"] = list(m["checked"] or [])
    m["visible"] = list(m["visible"] or [])
    m["tried"] = list(m["tried"] or [])
    m["tries"] = int(m["tries"] or 0)
    m["holding"] = bool(m["holding"]) if m["holding"] is not None else False
    return m


def _hz(cab, drawer):
    return cab[2] + HANDLE_DZ[drawer]


def _choose_target(m):
    if EXPLORE_FIRST and not m["tried"]:
        return EXPLORE_FIRST
    for d in EMPTY_PREFERENCE:
        if d in m["visible"] or d in m["tried"]:
            continue
        return d
    for d in EMPTY_PREFERENCE:  # every drawer looked occupied/tried: fall back to untried ones
        if d not in m["tried"]:
            return d
    return EMPTY_PREFERENCE[0]


def perceive_decide(obs, goal, memory, last):
    """Stage machine. Memory: stage, cabinet [hx,hy,topz], butter [x,y,table_z], checked/visible/tried drawer
    lists, tries, target drawer, open_y of the target drawer handle, holding flag, note, pending act name."""
    m = _mem_default(memory)
    cur = obs.current
    tool, args = "act_wait", {"steps": 5}
    pend = m["pending"]
    res = last if isinstance(last, str) else ""
    failed_tool = ("the tool failed" in res) or ("error:" in res[:200])

    # ---------- interpret the previous act ----------
    if pend == "check":
        d = DRAWERS[len(m["checked"])] if len(m["checked"]) < 3 else None
        ok_g = _num(res, "grasped", False)
        travel = _num(res, "travel", 0.0) or 0.0
        gap = _num(res, "close_gap", 1.0)
        colour = _num(res, "colour_px", 0) or 0
        raised = _num(res, "raised_px", 0) or 0
        if d is not None and ok_g and travel > 0.10 and gap is not None and abs(gap) < 0.03:
            m["checked"].append(d)
            if colour >= 5 or raised >= 25:
                m["visible"].append(d)
            m["tries"] = 0
            m["note"] = "checked %s travel=%.3f gap=%.3f colour=%d raised=%d" % (d, travel, gap, colour, raised)
        else:
            m["tries"] += 1
            m["note"] = "check %s failed: grasped=%s travel=%.3f gap=%s %s" % (d, ok_g, travel, gap, res[:120] if failed_tool else "")
            if m["tries"] >= MAX_TRIES and d is not None:
                # cannot verify; record as attempted so the task can proceed
                m["checked"].append(d)
                m["tries"] = 0
    elif pend == "open":
        travel = _num(res, "travel", 0.0) or 0.0
        if _num(res, "grasped", False) and travel > 0.10:
            m["open_y"] = _num(res, "y_open")
            m["stage"] = "pick" if not m["holding"] else "insert"
            m["tries"] = 0
        else:
            m["tries"] += 1
            if m["tries"] >= MAX_TRIES:
                # this drawer would not open: try the next candidate instead of inserting into a closed drawer
                m["tried"].append(m["target"])
                m["target"] = None
                m["tries"] = 0
        m["note"] = "open %s travel=%.3f" % (m["target"], travel)
    elif pend == "pick":
        if _num(res, "holding", False):
            m["holding"] = True
            m["stage"] = "insert"
            m["tries"] = 0
        else:
            m["holding"] = False
            m["tries"] += 1
        m["note"] = "pick width=%s" % _num(res, "width")
    elif pend == "insert":
        # the episode would have ended on success; still running -> this drawer was not accepted
        m["tried"].append(m["target"])
        m["holding"] = bool(_num(res, "holding", False))
        m["stage"] = "retrieve"
        m["tries"] = 0
        m["note"] = "insert into %s gave no success" % m["target"]
    elif pend == "retrieve":
        m["holding"] = bool(_num(res, "holding", False))
        m["stage"] = "close_target"
        m["note"] = "butter returned to table"
    elif pend == "close_target":
        m["stage"] = "open"
        m["target"] = None
        m["open_y"] = None
        m["tries"] = 0

    # ---------- perception ----------
    if m["cabinet"] is None:
        c = _perceive_cabinet(cur)
        if c is not None:
            m["cabinet"] = [c["hx"], c["hy"], c["topz"]]
    if not m["holding"] and (m["butter"] is None or m["stage"] == "pick"):
        # (re-)locate the butter before every pick; keep the old estimate if the new one is implausible
        prior = m["butter"][:2] if m["butter"] is not None else None
        tz = m["butter"][2] if m["butter"] is not None else None
        b = _perceive_butter(cur, prior, 0.12, tz)
        if (b is not None and b["n"] >= 30 and 0.005 < b["len_x"] < 0.12 and 0.005 < b["len_y"] < 0.12
                and 0.38 < b["table_z"] < 0.48):
            m["butter"] = [b["x"], b["y"], b["table_z"]]
            m["butter_yaw"] = b["major_angle"] if b["elong"] > 1.3 else 0.0
            m["butter_top"] = b["top_z"]
    if m["stage"] is None:
        m["stage"] = "check"

    cab = m["cabinet"]
    if cab is None and pend == "look" and m["tries"] >= 2:
        cab = m["cabinet"] = list(CABINET_PRIOR)  # demo-mean pose after repeated detection misses
        m["tries"] = 0
        m["note"] = "cabinet not detected; using prior"
    if cab is None:
        m["tries"] = m["tries"] + 1 if pend == "look" else 0
        m["pending"] = "look"
        return {"memory": m, "act": {"tool": "act_look", "args": {}}}

    # ---------- decide ----------
    st = m["stage"]
    if st == "check":
        if len(m["checked"]) < 3:
            d = DRAWERS[len(m["checked"])]
            k = m["tries"]
            dy = (0.0, 0.008, -0.004)[k % 3]
            dz = 0.0
            if EXPLORE_MISS_DY and d == "top" and k == 0:
                dy = EXPLORE_MISS_DY
            tool, args = "act_check_drawer", {"drawer": d, "hx": cab[0], "hy": cab[1] + dy, "hz": _hz(cab, d) + dz}
            m["pending"] = "check"
        else:
            st = m["stage"] = "open"
    if st == "open":
        if m["target"] is None:
            m["target"] = _choose_target(m)
        d = m["target"]
        k = m["tries"]
        dy = (0.0, 0.008, -0.004)[k % 3]
        dz = 0.0
        tool, args = "act_open_drawer", {"drawer": d, "hx": cab[0], "hy": cab[1] + dy, "hz": _hz(cab, d) + dz}
        m["pending"] = "open"
    elif st == "pick":
        b = m["butter"] or [BUTTER_PRIOR[0], BUTTER_PRIOR[1], 0.43]
        top = m["butter_top"] if m["butter_top"] is not None else b[2] + 0.012
        tool, args = "act_pick_butter", {"bx": b[0], "by": b[1] + (EXPLORE_BUTTER_DY if m["tries"] == 0 else 0.0),
                                         "table_z": b[2], "yaw": m["butter_yaw"] or 0.0, "top_z": top}
        m["pending"] = "pick"
    elif st == "insert":
        d = m["target"]
        oy = m["open_y"] if m["open_y"] is not None else cab[1] - 0.18
        tool, args = "act_insert_butter", {"drawer": d, "hx": cab[0], "y_open": oy, "y_closed": cab[1],
                                           "floor_z": _hz(cab, d) + FLOOR_DZ[d]}
        m["pending"] = "insert"
    elif st == "retrieve":
        b = m["butter"] or [BUTTER_PRIOR[0], BUTTER_PRIOR[1], 0.43]
        d = m["target"]
        oy = m["open_y"] if m["open_y"] is not None else cab[1] - 0.18
        tool, args = "act_return_butter", {"bx": b[0], "by": b[1], "table_z": b[2], "hx": cab[0],
                                           "y_open": oy, "floor_z": _hz(cab, d) + FLOOR_DZ[d],
                                           "holding": m["holding"]}
        m["pending"] = "retrieve"
    elif st == "close_target":
        d = m["tried"][-1] if m["tried"] else m["target"]
        oy = m["open_y"] if m["open_y"] is not None else cab[1] - 0.18
        tool, args = "act_close_drawer", {"drawer": d, "hx": cab[0], "y_open": oy, "y_closed": cab[1], "hz": _hz(cab, d)}
        m["pending"] = "close_target"
    return {"memory": m, "act": {"tool": tool, "args": args}}


def act_look(robot):
    """Hold still for a few steps (re-observe)."""
    _gripper(robot, -1, steps=3)
    return {"looked": True}


def act_wait(robot, steps=5):
    """Hold the current pose with the gripper state unchanged for `steps` steps."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, 0)
    return {"waited": int(steps)}


def _approach_handle(robot, hx, hy, hz):
    """Rotate to the horizontal drawer pose in front of the handle and move in to it (gripper open)."""
    R = _R_drawer()
    p, Rc, _ = _pose(robot)
    pre = np.array([hx, hy - 0.10, hz])
    if p[1] > hy - 0.06:
        _goto(robot, [p[0], hy - 0.10, p[2]], R, -1, tol=0.015, max_steps=60)
    r1 = _goto(robot, pre, R, -1, tol=0.008, rtol=0.05, max_steps=160)
    return r1, r1


def _close_hold(robot, R):
    """Close the gripper in place until the finger width stops changing (10-30 steps); return the width."""
    p, _, _ = _pose(robot)
    ws = []
    for k in range(30):
        q, Rq, f = _pose(robot)
        ws.append(_width(f))
        if k >= 10 and abs(ws[-1] - ws[-4]) < 0.0005 and (ws[-1] < 0.03 or k >= 20):
            break
        _cmd(robot, np.clip((p - q) * 15.0, -0.3, 0.3), np.clip(_m2rv(R @ Rq.T), -0.3, 0.3), 1)
    return _width(robot.now())


def _approach_contact(robot, hx, y, hz, R, extra=0.025, max_steps=60):
    """Slowly move along +y towards y+extra holding x,z; stop at contact (y stalls) or at the target."""
    ys = []
    for n in range(max_steps):
        q, Rq, _ = _pose(robot)
        ys.append(float(q[1]))
        if abs(y + extra - q[1]) < 0.002:
            break
        if n >= 6 and abs(ys[-1] - ys[-5]) < 0.0008:
            break
        a = np.array([(hx - q[0]) * 15.0, (y + extra - q[1]) * 10.0, (hz - q[2]) * 15.0])
        a = np.clip(a, -0.4, 0.4)
        _cmd(robot, a, np.clip(_m2rv(R @ Rq.T) * 2.0, -0.3, 0.3), -1)
    q, _, _ = _pose(robot)
    return float(q[1])


def _grasp_handle(robot, hx, y, hz, R, approach=True):
    """Move in to the handle near (hx,y,hz) until contact and close. Holding = finger width 4-30 mm
    (handle bar ~17 mm). If the fingers stall wide open (> 30 mm, seen once in rollout 24), back off and
    retry with the fingertips backed off 3 mm from contact (then also 3 mm higher)."""
    if approach:
        _goto(robot, [hx, y - 0.06, hz], R, -1, tol=0.008, max_steps=80)
    w = 0.0
    p = None
    for dz, back in ((0.0, 0.0), (0.0, 0.003), (0.003, 0.003)):
        z = hz + dz
        _goto(robot, [hx, y - 0.025, z], R, -1, tol=0.005, rtol=0.05, max_steps=60, vmax=0.6)
        yc = _approach_contact(robot, hx, y, z, R)
        if back > 0:
            # release the contact pressure (fingertips can catch on drawer-front edges) before closing
            _goto(robot, [hx, yc - back, z], R, -1, tol=0.0015, rtol=0.05, max_steps=15, vmax=0.3)
        w = _close_hold(robot, R)
        p, _, _ = _pose(robot)
        if w <= 0.03:
            break
        _gripper(robot, -1, steps=10, R=R)
        _goto(robot, [p[0], p[1] - 0.03, z], R, -1, tol=0.008, max_steps=30)
    ok = 0.004 < w < 0.03
    if not ok:
        _gripper(robot, -1, steps=12, R=R)
        _goto(robot, [p[0], p[1] - 0.06, p[2]], R, -1, tol=0.01, max_steps=40)
    return ok, w, float(p[1])


def _pull_open(robot, hx, hz, R, y_grasp, regrasps=2):
    """Pull the grasped handle along -y until the drawer stops. On finger slip (width < 8 mm) re-grasp the
    handle where it was lost and continue (bounded). Returns y_open, slips, holding."""
    y_target = y_grasp - 0.22
    slips = 0
    holding = True
    res = None
    for _ in range(regrasps + 1):
        res = _slide(robot, y_target, hx, hz, R, 1, max_steps=140, min_width=0.013)
        if not res["slipped"]:
            break
        slips += 1
        y_h = res["y_slip"]
        _gripper(robot, -1, steps=12, R=R)
        q, _, _ = _pose(robot)
        _goto(robot, [hx, min(q[1], y_h) - 0.06, hz], R, -1, tol=0.01, max_steps=40)
        ok, w, y_g = _grasp_handle(robot, hx, y_h, hz, R)
        if not ok:
            holding = False
            break
    p, _, _ = _pose(robot)
    return {"y_open": float(p[1]), "slips": slips, "holding": holding and not res["slipped"]}


def act_check_drawer(robot, drawer, hx, hy, hz):
    """Grasp the drawer handle at (hx,hy,hz), pull it fully open (until y stalls; re-grasp on slip), measure
    visible contents, push it closed (until stall), release and back off. Returns grasp/travel/close data."""
    R = _R_drawer()
    r1, r2 = _approach_handle(robot, hx, hy, hz)
    ok, width, y0 = _grasp_handle(robot, hx, hy, hz, R, approach=False)
    out = {"drawer": drawer, "approach_err": r1["err"], "width": width, "y_grasp": y0}
    if not ok:
        out.update({"grasped": False, "travel": 0.0})
        return out
    pull = _pull_open(robot, hx, hz, R, y0)
    for _ in range(4):
        robot.step(0, 0, 0, 0, 0, 0, 1)
    fr = robot.now()
    y_open = float(fr.tcp[1])
    cont = _drawer_contents(fr, hx, y_open, hy, hz)
    w_open = _width(fr)
    if not pull["holding"]:
        # lost the handle: push the drawer front closed with the closed fingers
        q, _, _ = _pose(robot)
        _goto(robot, [hx, q[1], hz], R, 1, tol=0.006, max_steps=30)
    _slide(robot, y0 + 0.03, hx, hz, R, 1, max_steps=150, gain=8.0)
    p1, _, _ = _pose(robot)
    _gripper(robot, -1, steps=12, R=R)
    _goto(robot, [p1[0], p1[1] - 0.08, p1[2]], R, -1, tol=0.01, max_steps=40)
    out.update({"grasped": True, "travel": float(y0 - y_open), "y_open": y_open, "width_open": w_open,
                "close_gap": float(y0 - p1[1]), "y_closed": float(p1[1]), "slips": pull["slips"],
                "colour_px": cont["colour_px"], "region_px": cont["region_px"], "raised_px": cont["raised_px"]})
    return out


def act_open_drawer(robot, drawer, hx, hy, hz):
    """Grasp the handle, pull the drawer fully open (until stall; re-grasp on slip), release, back off."""
    R = _R_drawer()
    r1, r2 = _approach_handle(robot, hx, hy, hz)
    ok, width, y0 = _grasp_handle(robot, hx, hy, hz, R, approach=False)
    out = {"drawer": drawer, "width": width, "y_grasp": y0}
    if not ok:
        out.update({"grasped": False, "travel": 0.0, "y_open": None})
        return out
    pull = _pull_open(robot, hx, hz, R, y0)
    for _ in range(4):
        robot.step(0, 0, 0, 0, 0, 0, 1)
    p1, _, _ = _pose(robot)
    _gripper(robot, -1, steps=12, R=R)
    _goto(robot, [p1[0], p1[1] - 0.09, p1[2]], R, -1, tol=0.01, max_steps=40)
    out.update({"grasped": True, "travel": float(y0 - p1[1]), "y_open": float(p1[1]), "slips": pull["slips"]})
    return out


def act_close_drawer(robot, drawer, hx, y_open, y_closed, hz):
    """Grasp the handle of an open drawer at y_open, push it closed until stall, release and back off."""
    R = _R_drawer()
    p, Rc, _ = _pose(robot)
    if np.linalg.norm(_m2rv(R @ Rc.T)) > 0.3:
        _goto(robot, [p[0], p[1], max(p[2], 0.75)], Rc, 0, tol=0.01, max_steps=50)
        _goto(robot, [hx, y_open - 0.15, 0.75], R, -1, tol=0.015, max_steps=120)
    _goto(robot, [hx, y_open - 0.10, hz], R, -1, tol=0.008, max_steps=120)
    ok, width, y0 = _grasp_handle(robot, hx, y_open, hz, R)
    if not ok:
        # push the drawer front with closed fingers instead
        _goto(robot, [hx, y_open - 0.03, hz], R, 1, tol=0.008, max_steps=40)
    p0, _, _ = _pose(robot)
    _slide(robot, y_closed + 0.03, hx, hz, R, 1, max_steps=170, gain=8.0)
    p1, _, _ = _pose(robot)
    _gripper(robot, -1, steps=12, R=R)
    _goto(robot, [p1[0], p1[1] - 0.08, p1[2]], R, -1, tol=0.01, max_steps=40)
    return {"drawer": drawer, "width": width, "grasped": ok, "y_start": float(p0[1]), "y_end": float(p1[1]),
            "closed": bool(abs(p1[1] - y_closed) < 0.03)}


def act_pick_butter(robot, bx, by, table_z, yaw=0.0, top_z=None):
    """From anywhere: back away from the cabinet, rise, turn top-down with the fingers across the butter's
    short side (yaw = butter major-axis angle), grasp at mid-height of the butter and lift. Returns holding
    (finger width > 8 mm after lifting; empty fingers read ~1-2 mm)."""
    p, Rc, _ = _pose(robot)
    Rd = _R_down(float(yaw))
    top = float(top_z) if top_z is not None else table_z + 0.012
    gz = max(table_z + 0.007, min((table_z + top) / 2.0, table_z + 0.03))
    if np.linalg.norm(_m2rv(_R_down(0.0) @ Rc.T)) > 0.3 and np.linalg.norm(_m2rv(Rd @ Rc.T)) > 0.3:
        # leave the drawer area with the current (horizontal) orientation first
        _goto(robot, [p[0], p[1] - 0.06, p[2]], Rc, -1, tol=0.01, max_steps=30)
        p, Rc, _ = _pose(robot)
        _goto(robot, [p[0], p[1], max(p[2], 0.75)], Rc, -1, tol=0.012, max_steps=60)
    _goto(robot, [bx, by, table_z + 0.15], Rd, -1, tol=0.008, rtol=0.04, max_steps=200)
    _goto(robot, [bx, by, gz], Rd, -1, tol=0.004, rtol=0.04, max_steps=80, vmax=0.5, settle=4)
    w = _close_hold(robot, Rd)
    _goto(robot, [bx, by, table_z + 0.20], Rd, 1, tol=0.01, max_steps=80, vmax=0.8)
    w2 = _width(robot.now())
    return {"width": w, "width_lifted": w2, "holding": bool(w2 > 0.008), "grasp_z": gz, "yaw": float(yaw)}


def act_insert_butter(robot, drawer, hx, y_open, y_closed, floor_z):
    """Carry the held butter low to the front of the open drawer, slide it over the drawer interior, lower it
    slowly and release it. The episode ends on success; returning means no success was scored for this
    drawer (then the butter is re-grasped and carried back out in front of the drawer)."""
    Rd = _R_down(INSERT_YAW)
    p, Rc, _ = _pose(robot)
    yc = y_open + INSERT_DY
    z_in = floor_z + 0.10
    _goto(robot, [p[0], p[1], max(p[2], z_in)], Rd, 1, tol=0.015, max_steps=60)
    _goto(robot, [hx, y_open - 0.08, z_in], Rd, 1, tol=0.012, max_steps=160)
    _goto(robot, [hx, yc, z_in], Rd, 1, tol=0.008, max_steps=80, vmax=0.5)
    _goto(robot, [hx, yc, floor_z + 0.025], Rd, 1, tol=0.004, max_steps=80, vmax=0.25, settle=10)
    w_before = _width(robot.now())
    _gripper(robot, -1, steps=14)
    for _ in range(25):
        robot.step(0, 0, 0, 0, 0, 0, -1)
    # no success: re-grasp the butter where it lies and carry it back out in front of the drawer
    _goto(robot, [hx, yc, floor_z + 0.012], Rd, -1, tol=0.004, max_steps=40, vmax=0.3, settle=3)
    _close_hold(robot, Rd)
    _goto(robot, [hx, yc, z_in], Rd, 1, tol=0.01, max_steps=60, vmax=0.6)
    _goto(robot, [hx, y_open - 0.08, z_in], Rd, 1, tol=0.012, max_steps=60, vmax=0.6)
    w2 = _width(robot.now())
    return {"drawer": drawer, "width_before_release": w_before, "holding": bool(w2 > 0.008), "width": w2}


def act_return_butter(robot, bx, by, table_z, hx, y_open, floor_z, holding):
    """Put the butter back on the table at (bx,by) (re-grasping it from the drawer first if not held), release
    and rise. All moves near the drawer stay low (z = floor + 0.10) to keep the arm away from its reach limit."""
    Rd = _R_down(INSERT_YAW)
    z_in = floor_z + 0.10
    yc = y_open + INSERT_DY
    if not holding:
        p, Rc, _ = _pose(robot)
        _goto(robot, [p[0], p[1], max(p[2], z_in)], Rd, -1, tol=0.015, max_steps=60)
        _goto(robot, [hx, y_open - 0.08, z_in], Rd, -1, tol=0.012, max_steps=160)
        _goto(robot, [hx, yc, z_in], Rd, -1, tol=0.008, max_steps=80, vmax=0.5)
        _goto(robot, [hx, yc, floor_z + 0.012], Rd, -1, tol=0.004, max_steps=100, vmax=0.4, settle=3)
        _close_hold(robot, Rd)
        _goto(robot, [hx, yc, z_in], Rd, 1, tol=0.01, max_steps=60, vmax=0.6)
        _goto(robot, [hx, y_open - 0.08, z_in], Rd, 1, tol=0.012, max_steps=60, vmax=0.6)
    w = _width(robot.now())
    Rd = _R_down(0.0)
    _goto(robot, [bx, by, table_z + 0.12], Rd, 1, tol=0.01, max_steps=200)
    _goto(robot, [bx, by, table_z + 0.012], Rd, 1, tol=0.005, max_steps=80, vmax=0.4, settle=3)
    _gripper(robot, -1, steps=14, R=Rd)
    _goto(robot, [bx, by, 0.70], Rd, -1, tol=0.012, max_steps=80)
    return {"width_carried": w, "holding": False}
