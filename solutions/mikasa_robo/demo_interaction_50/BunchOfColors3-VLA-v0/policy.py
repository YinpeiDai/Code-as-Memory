"""BunchOfColors3 policy: remember the cue colours, touch the matching ring cubes, press the button.

Stages (memory.stage):
  cue    - read cue cubes (<=5 saturated cubes on the table) from the earliest frames
  wait   - cue recorded; wait (zero motion) until the ring of >=7 cubes appears
  touch  - touch each target cube in a short planned order (open gripper, TCP ~5 cm high)
  button - close gripper, press the centre button, lift and hold; re-press if no success
"""

TOUCH_OFFSET = (-0.0083, 0.0026)   # expert TCP minus detected top-face centroid (50 demos, std ~1.5 mm)
Z_TOUCH = 0.048                    # expert touch TCP 0.044-0.062 ok; R008: 0.075 NOT registered
Z_TRAVEL = 0.13
BUTTON_DEFAULT = (-0.125, 0.0)     # button centre observed in all 50 demos (std ~3 mm)
MAX_PRESSES = 3

FIELDS = ("stage", "cue_colours", "cue_frame", "ring_colours", "ring_xy", "plan", "done",
          "button_xy", "home_rpy", "presses", "touch_tries", "rounds", "notes")


def _mem(memory):
    m = {k: memory.get(k) for k in FIELDS}
    if m["stage"] is None:
        m["stage"] = "cue"
    for k in ("cue_colours", "ring_colours", "ring_xy", "plan", "done"):
        if m[k] is None:
            m[k] = []
    for k in ("presses", "touch_tries", "cue_frame", "rounds"):
        if m[k] is None:
            m[k] = 0 if k != "cue_frame" else -1
    if m["notes"] is None:
        m["notes"] = ""
    return m


def _plan_order(start_xy, targets, button_xy):
    """Order targets (list of (colour, xy)) to minimise path start->targets->button."""
    best = None
    idx = list(range(len(targets)))

    def perms(a):
        if len(a) <= 1:
            return [a]
        out = []
        for i in range(len(a)):
            for rest in perms(a[:i] + a[i + 1:]):
                out.append([a[i]] + rest)
        return out

    for p in perms(idx):
        cur = np.asarray(start_xy, dtype=np.float64)
        L = 0.0
        for i in p:
            q = np.asarray(targets[i][1], dtype=np.float64)
            L += float(np.linalg.norm(q - cur))
            cur = q
        L += float(np.linalg.norm(np.asarray(button_xy) - cur))
        if best is None or L < best[0]:
            best = (L, p)
    return [targets[i][0] for i in best[1]]


def perceive_decide(obs, goal, memory, last):
    """Track cue -> ring -> touches -> button press, choosing the next act."""
    m = _mem(memory)
    cur = obs.current
    if m["home_rpy"] is None:
        m["home_rpy"] = [float(cur.tcp[3]), float(cur.tcp[4]), float(cur.tcp[5])]

    if m["stage"] in ("cue", "wait"):
        frames = list(obs.frames)
        ring = None
        for f in frames:
            cubes = _scene(f)
            n = len(cubes)
            if m["stage"] == "cue" and 1 <= n <= 5:
                m["cue_colours"] = sorted(set(c["colour"] for c in cubes))
                m["cue_frame"] = int(f.step)
                m["stage"] = "wait"
                m["notes"] += "cue@%d:%s; " % (f.step, ",".join(m["cue_colours"]))
            elif n >= 7 and m["stage"] in ("wait", "cue"):
                # keep the frame that explains the most cue colours, then the most cubes
                cols = set(c["colour"] for c in cubes)
                score = (len(cols & set(m["cue_colours"])), len(cols))
                if ring is None or score >= ring[2]:
                    ring = (f, cubes, score)
        if ring is not None and m["stage"] == "wait" and ring[2][0] < len(m["cue_colours"]) \
                and int(cur.step) < 40:
            # a cue colour is not visible in the ring yet (occlusion/merge): look again
            m["notes"] += "ring@%d lacks cue colour; " % ring[0].step
            return {"memory": m, "act": {"tool": "act_wait", "args": {"max_steps": 2, "min_steps": 2}}}
        if ring is not None and m["stage"] == "wait":
            f, cubes, _ = ring
            m["ring_colours"] = [c["colour"] for c in cubes]
            m["ring_xy"] = [[round(c["xy"][0], 4), round(c["xy"][1], 4)] for c in cubes]
            b = _button(f.front_rgb, f.front_depth, f.front_K, f.front_extrinsic)
            if b is not None and abs(b[0] - BUTTON_DEFAULT[0]) < 0.06 and abs(b[1] - BUTTON_DEFAULT[1]) < 0.06:
                m["button_xy"] = [round(b[0], 4), round(b[1], 4)]
            else:
                m["button_xy"] = [BUTTON_DEFAULT[0], BUTTON_DEFAULT[1]]
            targets = []
            missing = []
            for col in m["cue_colours"]:
                js = [j for j, c in enumerate(m["ring_colours"]) if c == col]
                if js:
                    targets.append((col, m["ring_xy"][js[0]]))
                else:
                    missing.append(col)
            m["plan"] = _plan_order([float(cur.tcp[0]), float(cur.tcp[1])], targets, m["button_xy"])
            m["notes"] += "ring@%d n=%d missing=%s plan=%s; " % (f.step, len(cubes), ",".join(missing), ",".join(m["plan"]))
            m["stage"] = "touch"
        elif m["stage"] == "cue" and int(cur.step) > 40:
            m["notes"] += "no cue seen; "
            m["stage"] = "wait"
        if m["stage"] in ("cue", "wait"):
            return {"memory": m, "act": {"tool": "act_wait", "args": {"max_steps": 30}}}

    if m["stage"] == "touch":
        # record outcome of the previous touch
        if isinstance(last, str) and last.startswith("act_touch") and '"reached": true' in last:
            col = last.split('"colour": "')[1].split('"')[0] if '"colour": "' in last else None
            if col and col not in m["done"]:
                m["done"].append(col)
                m["touch_tries"] = 0
        remaining = [c for c in m["plan"] if c not in m["done"]]
        if remaining and m["touch_tries"] < 3:
            col = remaining[0]
            j = m["ring_colours"].index(col)
            xy = m["ring_xy"][j]
            m["touch_tries"] += 1
            return {"memory": m, "act": {"tool": "act_touch", "args": {
                "x": xy[0] + TOUCH_OFFSET[0], "y": xy[1] + TOUCH_OFFSET[1], "colour": col,
                "rpy": m["home_rpy"], "z_touch": Z_TOUCH - 0.004 * min(1, m["rounds"])}}}
        m["stage"] = "button"

    # button stage
    if m["presses"] >= 1 and m["rounds"] < 1 and isinstance(last, str) and last.startswith("act_press"):
        # a full press+lift+hold gave no success: assume a touch was not counted, redo all touches lower
        m["rounds"] += 1
        m["done"] = []
        m["touch_tries"] = 0
        m["stage"] = "touch"
        m["notes"] += "no success after press %d: retouch round %d; " % (m["presses"], m["rounds"])
        col = m["plan"][0]
        xy = m["ring_xy"][m["ring_colours"].index(col)]
        m["touch_tries"] = 1
        return {"memory": m, "act": {"tool": "act_touch", "args": {
            "x": xy[0] + TOUCH_OFFSET[0], "y": xy[1] + TOUCH_OFFSET[1], "colour": col,
            "rpy": m["home_rpy"], "z_touch": Z_TOUCH - 0.004}}}
    m["presses"] += 1
    if m["presses"] > MAX_PRESSES:
        return {"memory": m, "act": {"tool": "act_wait", "args": {"max_steps": 20}}}
    bx, by = m["button_xy"]
    args = {"x": bx, "y": by, "rpy": m["home_rpy"]}
    return {"memory": m, "act": {"tool": "act_press", "args": args}}


def act_wait(robot, max_steps=30, min_steps=0):
    """Hold still with open gripper until the ring of >=7 cubes is visible (or max_steps);
    always takes at least min_steps steps."""
    n = 0
    seen = 0
    while n < int(max_steps):
        f = robot.now()
        seen = len(_scene(f))
        if seen >= 7 and n >= int(min_steps):
            break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    return {"steps": n, "cubes_seen": seen, "step": int(robot.now().step)}


def act_touch(robot, x, y, colour, rpy, z_touch=Z_TOUCH, z_travel=Z_TRAVEL, hold=3):
    """Touch the cube at (x, y): travel at z_travel with open gripper, descend to z_touch,
    hold, and rise back to z_travel. Returns measured approach results."""
    out = {"colour": colour, "target": [round(x, 4), round(y, 4)]}
    reach = None
    try:
        chk = robot.dry_run(x, y, z_touch, rpy[0], rpy[1], rpy[2])
        reach = bool(chk.get("reachable", True)) if isinstance(chk, dict) else None
    except Exception as e:  # dry run unavailable: proceed, record
        reach = "n/a"
    out["reachable"] = reach
    p = robot.now().tcp
    if float(p[2]) < z_travel - 0.03:
        out["rise0"] = _servo(robot, [float(p[0]), float(p[1]), z_travel], rpy, 1.0, tol=0.015, vtol=0.02, max_steps=25)
    out["travel"] = _servo(robot, [x, y, z_travel], rpy, 1.0, tol=0.01, vtol=0.008, max_steps=60)
    out["descend"] = _servo(robot, [x, y, z_touch], rpy, 1.0, tol=0.005, vtol=0.004, max_steps=40)
    f = robot.now()
    zmin = float(f.tcp[2])
    zs = _hold(robot, rpy, 1.0, hold)
    zmin = min([zmin] + zs)
    f = robot.now()
    out["xy_err"] = round(float(math.hypot(float(f.tcp[0]) - x, float(f.tcp[1]) - y)), 4)
    out["zmin"] = round(zmin, 4)
    out["reached"] = bool(out["xy_err"] < 0.012 and zmin < z_touch + 0.012)
    out["step_low"] = int(f.step)
    out["rise"] = _servo(robot, [x, y, z_travel], rpy, 1.0, tol=0.015, vtol=0.02, max_steps=25)
    return out


def act_press(robot, x, y, rpy, z_above=0.085, press_steps=8, press_dz=-0.6, z_lift=0.24, lift_hold=10):
    """Press the button at (x, y) with a closed gripper: travel at z=0.13, descend to z_above,
    push down for press_steps, then lift to z_lift and hold (success is expected after release)."""
    out = {}
    p = robot.now().tcp
    if float(p[2]) < 0.10:
        out["rise0"] = _servo(robot, [float(p[0]), float(p[1]), Z_TRAVEL], rpy, -1.0, tol=0.015, vtol=0.02, max_steps=25)
    out["travel"] = _servo(robot, [x, y, Z_TRAVEL], rpy, -1.0, tol=0.01, vtol=0.008, max_steps=60, min_steps=4)
    out["above"] = _servo(robot, [x, y, z_above], rpy, -1.0, tol=0.006, vtol=0.006, max_steps=30)
    zs = []
    for _ in range(int(press_steps)):
        f = robot.now()
        ex = (x - float(f.tcp[0])) / 0.1
        ey = (y - float(f.tcp[1])) / 0.1
        r, _ = _rot_cmd(f.tcp, rpy)
        robot.step(max(-0.3, min(0.3, 0.8 * ex)), max(-0.3, min(0.3, 0.8 * ey)), press_dz,
                   float(r[0]), float(r[1]), float(r[2]), -1.0)
        zs.append(round(float(robot.now().tcp[2]), 4))
    out["press_z"] = zs
    out["press_xy"] = [round(float(robot.now().tcp[0]), 4), round(float(robot.now().tcp[1]), 4)]
    out["lift"] = _servo(robot, [x, y, z_lift], rpy, -1.0, tol=0.01, vtol=0.006, max_steps=40)
    out["lift_hold"] = _hold(robot, rpy, -1.0, lift_hold)
    out["end_step"] = int(robot.now().step)
    return out
