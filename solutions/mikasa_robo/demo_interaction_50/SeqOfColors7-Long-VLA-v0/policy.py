"""SeqOfColors7-Long policy: watch the cue cubes, wait for the response arc, touch every cued
colour (shortest path), then press the centre button with a closed gripper and lift."""

Z_TOUCH = 0.050      # TCP height for a touch (demo touches bottom out at 0.044-0.063)
Z_TRAVEL = 0.10      # transit height between cubes (cube tops at 0.042)
Z_CLEAR = 0.075      # height reached before leaving a touched cube
CUBE_BIAS = (-0.006, 0.002)    # demo TCP position minus front-camera top-face centre
BUTTON_BIAS = (0.012, 0.002)   # demo press position minus front-camera cap centre
BUTTON_DEFAULT = (-0.112, 0.0)
MIN_CUE_RUN = 3
ORDER_MODE = "cue"   # "cue": touch in the order shown; "short": Held-Karp shortest path


def _f(x):
    return [float(v) for v in x]


def _plan_order(start, pts, end):
    """Held-Karp shortest path start -> all pts -> end (xy). Returns index order."""
    n = len(pts)
    if n == 0:
        return []
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    INF = 1e9
    full = (1 << n) - 1
    dp = {}
    for i in range(n):
        dp[(1 << i, i)] = (d(start, pts[i]), -1)
    for mask in range(1, full + 1):
        for j in range(n):
            if not (mask >> j) & 1 or (mask, j) not in dp:
                continue
            cost = dp[(mask, j)][0]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                c = cost + d(pts[j], pts[k])
                if (nm, k) not in dp or c < dp[(nm, k)][0]:
                    dp[(nm, k)] = (c, j)
    best, bj = INF, -1
    for j in range(n):
        c = dp[(full, j)][0] + d(pts[j], end)
        if c < best:
            best, bj = c, j
    order, mask, j = [], full, bj
    while j != -1:
        order.append(j)
        pj = dp[(mask, j)][1]
        mask &= ~(1 << j)
        j = pj
    return order[::-1]


def _defaults(memory):
    m = dict(memory)
    for k, v in (("stage", None), ("cues", []), ("cue_last", None), ("cue_run", 0),
                 ("cue_steps", []), ("arc_step", -1), ("targets", []), ("target_colours", []),
                 ("done_colours", []), ("pending", None), ("attempts", 0), ("button", None),
                 ("rpy0", None), ("presses", 0), ("round", 0), ("notes", "")):
        if m.get(k) is None:
            m[k] = v
    return m


def perceive_decide(obs, goal, memory, last):
    """Track cue colours, detect the response arc, plan touches, verify touches from measured TCP,
    then press the button. Memory keeps cue observations and per-stage progress."""
    m = _defaults(memory)
    cur = obs.current
    if m["stage"] is None:
        m["stage"] = "watch"
        m["rpy0"] = _f(obs.frames[0].tcp[3:6])
    notes = []

    if m["stage"] == "watch":
        nfr = len(obs.frames)
        for i, fr in enumerate(obs.frames):
            if i >= nfr - 2 and nfr > 2 and len(_arc_cubes(fr)) >= 7:
                break
            c = _cue_colour(fr)
            if c is not None and c == m["cue_last"]:
                m["cue_run"] = int(m["cue_run"]) + 1
            else:
                m["cue_last"] = c
                m["cue_run"] = 1 if c is not None else 0
            if c is not None and m["cue_run"] == MIN_CUE_RUN:
                m["cues"] = list(m["cues"]) + [c]
                m["cue_steps"] = list(m["cue_steps"]) + [int(fr.step) - MIN_CUE_RUN + 1]
        arc = _arc_cubes(cur)
        if len(arc) >= 7:
            m["arc_step"] = int(cur.step)
            seen = {}
            for o in arc:
                if o["colour"] not in seen or o["n"] > seen[o["colour"]]["n"]:
                    seen[o["colour"]] = o
            pts, cols = [], []
            for c in m["cues"]:
                if c in seen and c not in cols:
                    o = seen[c]
                    pts.append([o["x"] + CUBE_BIAS[0], o["y"] + CUBE_BIAS[1], 0.02])
                    cols.append(c)
                else:
                    notes.append("cue colour %s not found in arc" % c)
            b = _button_centre(cur)
            if b is None:
                b = list(BUTTON_DEFAULT)
                notes.append("button default")
            else:
                b = [b[0] + BUTTON_BIAS[0], b[1] + BUTTON_BIAS[1]]
            start = [float(cur.tcp[0]), float(cur.tcp[1])]
            if ORDER_MODE == "short":
                order = _plan_order(start, pts, b)
            else:
                order = list(range(len(pts)))
            m["targets"] = [pts[i] for i in order]
            m["target_colours"] = [cols[i] for i in order]
            m["button"] = [b[0], b[1], 0.05]
            m["stage"] = "touch"
            notes.append("arc at %d with %d cubes; cues %s; plan %s" % (
                cur.step, len(arc), ",".join(m["cues"]), ",".join(m["target_colours"])))
        else:
            m["notes"] = (str(m["notes"]) + " | ".join(notes))[-1500:]
            return {"memory": m, "act": {"tool": "act_watch", "args": {"max_steps": 1190}}}

    if m["stage"] == "touch" and m["pending"] is not None:
        # verify the touch just executed from measured TCP samples
        k = m["target_colours"].index(m["pending"])
        tx, ty = m["targets"][k][0], m["targets"][k][1]
        best = 1.0
        for fr in obs.frames:
            p = fr.tcp
            if math.hypot(float(p[0]) - tx, float(p[1]) - ty) < 0.015:
                best = min(best, float(p[2]))
        if best <= Z_TOUCH + 0.012:
            m["done_colours"] = list(m["done_colours"]) + [m["pending"]]
            m["attempts"] = 0
            notes.append("touched %s minz %.3f" % (m["pending"], best))
        else:
            m["attempts"] = int(m["attempts"]) + 1
            notes.append("touch %s not reached (minz %.3f)" % (m["pending"], best))
            if m["attempts"] >= 3:
                m["done_colours"] = list(m["done_colours"]) + [m["pending"]]
                m["attempts"] = 0
                notes.append("gave up %s" % m["pending"])
        m["pending"] = None

    if m["stage"] == "touch":
        todo = [c for c in m["target_colours"] if c not in m["done_colours"]]
        if todo:
            c = todo[0]
            k = m["target_colours"].index(c)
            t = m["targets"][k]
            m["pending"] = c
            m["notes"] = (str(m["notes"]) + " | " + " | ".join(notes))[-1500:]
            r1 = int(m["round"]) > 0
            return {"memory": m, "act": {"tool": "act_touch", "args": {
                "x": t[0], "y": t[1], "z_touch": Z_TOUCH - (0.004 if r1 else 0.0),
                "z_travel": 0.17 if r1 else Z_TRAVEL,
                "rpy": m["rpy0"], "last": len(todo) == 1}}}
        m["stage"] = "press"

    # press stage: each call presses once; the episode ends by itself on success
    n = int(m["presses"])
    if n >= 3 and int(m["round"]) == 0:
        # fallback: presses not accepted -> re-touch every cued cube in cue order, higher transit
        m["round"] = 1
        cols = [c for c in m["cues"] if c in m["target_colours"]]
        m["targets"] = [m["targets"][m["target_colours"].index(c)] for c in cols]
        m["target_colours"] = cols
        m["done_colours"] = []
        m["stage"] = "touch"
        m["presses"] = 0
        c = cols[0]
        m["pending"] = c
        notes.append("retouch round in cue order")
        m["notes"] = (str(m["notes"]) + " | " + " | ".join(notes))[-1500:]
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": m["targets"][0][0], "y": m["targets"][0][1], "z_touch": Z_TOUCH - 0.004,
            "z_travel": 0.17, "rpy": m["rpy0"], "last": len(cols) == 1}}}
    m["presses"] = n + 1
    hold = 4 if n == 0 else 8
    lift = 0.20 if n == 0 else 0.25
    b = m["button"] if m["button"] is not None else [BUTTON_DEFAULT[0], BUTTON_DEFAULT[1], 0.05]
    notes.append("press %d" % (n + 1))
    m["notes"] = (str(m["notes"]) + " | " + " | ".join(notes))[-1500:]
    return {"memory": m, "act": {"tool": "act_press", "args": {
        "x": b[0], "y": b[1], "rpy": m["rpy0"], "hold": hold, "z_lift": lift, "wait": 20}}}


def act_watch(robot, max_steps=1190):
    """Send zero motion (gripper open) every step until the 9-cube response arc is visible in the
    front camera. Returns steps waited and whether the arc was seen."""
    n = 0
    seen = False
    while n < max_steps:
        if len(_arc_cubes(robot.now())) >= 7:
            seen = True
            break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    return {"arc_seen": seen, "steps": n, "step": int(robot.now().step)}


def act_touch(robot, x, y, z_touch=0.05, z_travel=0.10, rpy=None, last=False, max_steps=90):
    """Move above (x, y), descend until the TCP reaches z_touch, then rise to Z_CLEAR.
    Gripper stays open (closes during the final rise when `last`). Returns measured min z."""
    prev = None
    phase = "approach"
    minz, at_min = 9.0, None
    n = 0
    hold = 0
    start = int(robot.now().step)
    check = None
    while n < max_steps:
        f = robot.now()
        pos = np.asarray(f.tcp[:3], dtype=np.float64)
        dxy = math.hypot(pos[0] - x, pos[1] - y)
        if dxy < 0.02 and pos[2] < minz:
            minz, at_min = float(pos[2]), [float(pos[0]), float(pos[1])]
        grip = 1.0
        if phase == "approach":
            if pos[2] < Z_CLEAR - 0.005 and dxy > 0.03:
                tgt = [pos[0], pos[1], z_travel]
            elif dxy > 0.025:
                tgt = [x, y, z_travel]
            else:
                phase = "descend"
        if phase == "descend":
            tgt = [x, y, z_touch]
            if pos[2] <= z_touch + 0.004:
                hold += 1
                if hold >= 2:
                    phase = "rise"
        if phase == "rise":
            tgt = [x, y, Z_CLEAR + 0.01]
            if last:
                grip = -1.0
            if pos[2] >= Z_CLEAR:
                break
        if check is None:
            t0 = _trans_cmd(pos, tgt, np.zeros(3))
            try:
                check = robot.dry_run(t0[0], t0[1], t0[2], 0.0, 0.0, 0.0, mode="delta")
            except Exception as e:
                check = {"error": str(e)[:100]}
        prev = _step_to(robot, tgt, rpy, grip, prev)
        n += 1
    f = robot.now()
    margin = check.get("min_joint_margin_rad") if isinstance(check, dict) else None
    return {"phase": phase, "steps": n, "start_step": start, "end_step": int(f.step),
            "min_z": minz, "xy_at_min": at_min, "target": [x, y],
            "first_cmd_min_joint_margin": margin}


def act_press(robot, x, y, rpy=None, hold=4, z_lift=0.20, wait=20, max_steps=120):
    """Close the gripper, move above the button, push down until the TCP stalls on the button,
    hold `hold` steps, lift to z_lift and wait there (success ends the episode)."""
    prev = None
    phase = "approach"
    n = 0
    held = 0
    waited = 0
    minz = 9.0
    stall_z = None
    start = int(robot.now().step)
    while n < max_steps:
        f = robot.now()
        pos = np.asarray(f.tcp[:3], dtype=np.float64)
        vel = pos - prev if prev is not None else np.zeros(3)
        dxy = math.hypot(pos[0] - x, pos[1] - y)
        if dxy < 0.02:
            minz = min(minz, float(pos[2]))
        if phase == "approach":
            if pos[2] < Z_CLEAR - 0.005 and dxy > 0.03:
                tgt = [pos[0], pos[1], Z_TRAVEL]
            elif dxy > 0.015:
                tgt = [x, y, Z_TRAVEL]
            else:
                phase = "push"
        if phase == "push":
            tgt = [x, y, 0.035]
            if pos[2] < 0.062 and abs(vel[2]) < 0.0015:
                held += 1
                if stall_z is None:
                    stall_z = float(pos[2])
                if held >= hold:
                    phase = "lift"
        if phase == "lift":
            tgt = [x, y, z_lift]
            if pos[2] >= z_lift - 0.01:
                waited += 1
                if waited >= wait:
                    break
        prev = _step_to(robot, tgt, rpy, -1.0, prev)
        n += 1
    return {"phase": phase, "steps": n, "start_step": start, "end_step": int(robot.now().step),
            "min_z": minz, "stall_z": stall_z, "held": held, "final_z": float(robot.now().tcp[2])}
