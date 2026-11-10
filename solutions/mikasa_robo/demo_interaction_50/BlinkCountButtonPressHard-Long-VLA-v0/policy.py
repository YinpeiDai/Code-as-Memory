"""BlinkCountButtonPress policy: perceive layout, count blinks, press N times, submit.

Stages (memory.stage): None -> 'count' -> 'press' -> 'submit' -> 'done'.
Counting is done in perceive_decide from every observed frame (obs.frames), so the
count is an observation, not an act claim.  Presses are also counted from the
measured TCP trajectory (descend below the cap top near the button, then rise).
"""

COLORS = {
    "red": (230, 20, 20), "green": (20, 230, 20), "blue": (20, 20, 230),
    "black": (60, 60, 60), "white": (215, 215, 215), "yellow": (230, 230, 20),
}
CUE_MAX_STEPS = 700          # hard cap on waiting for the status lamp
PRESS_CHUNK = 8              # presses per act call (keeps each call short)
LIFT_ABOVE_TOP = 0.13        # lift height above cap top between presses (m)
PRESS_BELOW_TOP = 0.020      # commanded press depth below cap top (m)
BOTTOM_BELOW_TOP = 0.005     # measured TCP depth below cap top that counts as bottomed
COUNT_DOWN_BELOW_TOP = 0.003 # perceive: TCP this far below cap top = a press
COUNT_UP_ABOVE_TOP = 0.095   # perceive: TCP this far above cap top = released


# ---------------------------------------------------------------- geometry
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        # near pi: use diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return d * ang
    return w / (2.0 * s) * ang


def _world_map(frame):
    D = np.asarray(frame.front_depth).astype(float) / 1000.0
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    R, t = E[:, :3], E[:, 3]
    v, u = np.mgrid[0:256, 0:256]
    x = (u - K[0, 2]) * D / K[0, 0]
    y = (v - K[1, 2]) * D / K[1, 1]
    P = np.stack([x, y, D], -1).reshape(-1, 3)
    W = ((P - t) @ R).reshape(256, 256, 3)
    return W, D


def _clusters(mask):
    """8-connected components of a boolean mask (pure Python BFS over set pixels)."""
    ys, xs = np.nonzero(mask)
    todo = set(zip(ys.tolist(), xs.tolist()))
    out = []
    while todo:
        seed = todo.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            a, b = stack.pop()
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    q = (a + da, b + db)
                    if q in todo:
                        todo.discard(q)
                        comp.append(q)
                        stack.append(q)
        out.append(comp)
    return out


def _color_name(rgb, names=None):
    """Hue-based colour name: red/green/blue/yellow for saturated colours, else black/white."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 40:
        return "black" if (r + g + b) / 3.0 < 130 else "white"
    if r >= g and r >= b:
        return "yellow" if g > 0.7 * r and b < 0.5 * r else "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


# ---------------------------------------------------------------- goal parsing
def _parse_goal(goal):
    g = (goal or "").lower().replace(",", " ").replace(".", " ")
    w = g.split()
    res = {"count_lamp": "blue", "status_lamp": "red", "go_color": "green",
           "press_button": "red", "submit_button": "black"}
    for i in range(len(w) - 2):
        if w[i + 1] == "lamp" and w[i + 2].startswith("blink"):
            res["count_lamp"] = w[i]
        if w[i + 1] == "lamp" and w[i + 2] == "turns" and i + 3 < len(w):
            res["status_lamp"] = w[i]
            res["go_color"] = w[i + 3]
    for i in range(len(w) - 1):
        if w[i + 1] == "button":
            nxt = " ".join(w[i + 2:i + 6])
            if "exactly" in nxt or "that many" in nxt:
                res["press_button"] = w[i]
            if "submit" in nxt:
                res["submit_button"] = w[i]
    return res


# ---------------------------------------------------------------- perception
def _segment(frame, spec):
    """Find the two buttons and two lamps from front RGB-D.  Returns dict or error."""
    W, D = _world_map(frame)
    rgb = np.asarray(frame.front_rgb).astype(float)
    z = W[:, :, 2]
    mask = (z > 0.005) & (z < 0.2) & (D > 0) & (W[:, :, 0] > -0.42)
    buttons, lamps = [], []
    for comp in _clusters(mask):
        if len(comp) < 30:
            continue
        ys = np.array([c[0] for c in comp])
        xs = np.array([c[1] for c in comp])
        w = W[ys, xs]
        zmax = float(w[:, 2].max())
        top = w[:, 2] > zmax - 0.008
        c = rgb[ys[top], xs[top]].mean(0)
        item = {"n": len(comp), "xyz": [float(w[top, 0].mean()), float(w[top, 1].mean()), zmax],
                "rgb": [float(c[0]), float(c[1]), float(c[2])],
                "box": [int(xs[top].min()), int(ys[top].min()), int(xs[top].max()), int(ys[top].max())]}
        if 0.04 < zmax < 0.085:
            buttons.append(item)
        elif 0.085 <= zmax < 0.15:
            lamps.append(item)
    out = {"buttons": buttons, "lamps": lamps}
    names = list(COLORS)
    pb = [b for b in buttons if _color_name(b["rgb"], names) == spec["press_button"]]
    sb = [b for b in buttons if _color_name(b["rgb"], names) == spec["submit_button"]]
    if len(buttons) == 2 and (len(pb) != 1 or len(sb) != 1 or pb[0] is sb[0]):
        # fall back: brighter / redder one is the press button
        bs = sorted(buttons, key=lambda b: b["rgb"][0] - 0.5 * (b["rgb"][1] + b["rgb"][2]) + 0.2 * b["rgb"][0])
        sb, pb = [bs[0]], [bs[1]]
    if len(pb) >= 1 and len(sb) >= 1:
        out["press"] = max(pb, key=lambda b: b["n"])
        out["submit"] = max(sb, key=lambda b: b["n"])
    # status lamp: the lamp currently showing the status colour (red at reset)
    st = [l for l in lamps if _color_name(l["rgb"], names) == spec["status_lamp"]]
    if len(lamps) == 2:
        if len(st) != 1:
            st = [max(lamps, key=lambda l: l["rgb"][0] - l["rgb"][1])]
        out["status"] = st[0]
        out["count"] = lamps[0] if lamps[1] is st[0] else lamps[1]
    return out


def _lamp_state(frame, box):
    """Dominant saturated colour in the lamp box: 'red','green','blue', or 'off'."""
    u0, v0, u1, v1 = box
    patch = np.asarray(frame.front_rgb)[v0:v1 + 1, u0:u1 + 1].reshape(-1, 3).astype(int)
    if len(patch) == 0:
        return "off"
    r, g, b = patch[:, 0], patch[:, 1], patch[:, 2]
    n = float(len(patch))
    fr = np.sum((r > 150) & (g < 100) & (b < 100)) / n
    fg = np.sum((g > 150) & (r < 100) & (b < 100)) / n
    fb = np.sum((b > 150) & (r < 100) & (g < 100)) / n
    best = max((fr, "red"), (fg, "green"), (fb, "blue"))
    return best[1] if best[0] > 0.25 else "off"


# ---------------------------------------------------------------- perceive/decide
def _blank_memory():
    return {"stage": None, "spec": "", "press_button": None, "submit_button": None,
            "count_lamp_box": None, "status_lamp_box": None, "blink_count": 0,
            "lamp_on": False, "blink_steps": [], "green_step": None, "last_step": -1,
            "presses_done": 0, "press_down": False, "press_log": [], "submit_attempts": 0,
            "rot_sign": 0.0, "home_rpy": None, "notes": ""}


def _int_after(text, key):
    if not text:
        return None
    k = text.find('"' + key + '"')
    if k < 0:
        return None
    s = text[k + len(key) + 3:].lstrip()
    num = ""
    for ch in s:
        if ch in "-0123456789.":
            num += ch
        else:
            break
    if num in ("", "-", "."):
        return None
    return float(num)


def perceive_decide(obs, goal, memory, last):
    """Count blinks of the count lamp, wait for the go colour, press the press button
    that many times (verified from measured TCP), then press the submit button."""
    mem = _blank_memory()
    for k in mem:
        if memory and memory.get(k) is not None:
            mem[k] = memory[k]
    spec = _parse_goal(goal)
    cur = obs.current
    notes = []

    if mem["stage"] is None:
        seg = _segment(cur, spec)
        if "press" not in seg or "count" not in seg:
            mem["notes"] = "segmentation failed: %d buttons %d lamps" % (len(seg["buttons"]), len(seg["lamps"]))
            mem["stage"] = None
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
        mem["spec"] = str(spec)
        mem["press_button"] = seg["press"]["xyz"]
        mem["submit_button"] = seg["submit"]["xyz"]
        mem["count_lamp_box"] = seg["count"]["box"]
        mem["status_lamp_box"] = seg["status"]["box"]
        mem["home_rpy"] = [float(cur.tcp[3]), float(cur.tcp[4]), float(cur.tcp[5])]
        mem["stage"] = "count"
        notes.append("layout press=%s submit=%s" % (str([round(v, 3) for v in mem["press_button"]]),
                                                    str([round(v, 3) for v in mem["submit_button"]])))

    # ---- update observation-based counters from every new frame
    pb = mem["press_button"]
    for f in obs.frames:
        if f.step <= mem["last_step"]:
            continue
        mem["last_step"] = f.step
        if mem["green_step"] is None:
            st = _lamp_state(f, mem["count_lamp_box"])
            on = st == spec["count_lamp"]
            if on and not mem["lamp_on"]:
                mem["blink_count"] += 1
                mem["blink_steps"] = list(mem["blink_steps"]) + [int(f.step)]
            mem["lamp_on"] = bool(on)
            if _lamp_state(f, mem["status_lamp_box"]) == spec["go_color"]:
                mem["green_step"] = int(f.step)
        else:
            p = f.tcp
            near = math.hypot(float(p[0]) - pb[0], float(p[1]) - pb[1]) < 0.03
            if not mem["press_down"] and near and float(p[2]) < pb[2] - COUNT_DOWN_BELOW_TOP:
                mem["press_down"] = True
            elif mem["press_down"] and float(p[2]) > pb[2] + COUNT_UP_ABOVE_TOP:
                mem["press_down"] = False
                mem["presses_done"] += 1
                mem["press_log"] = list(mem["press_log"]) + [int(f.step)]

    rs = _int_after(last, "rot_sign")
    if rs is not None and rs != 0:
        mem["rot_sign"] = rs

    if mem["stage"] == "count":
        if mem["green_step"] is None:
            mem["notes"] = "; ".join(notes + ["counting: %d blinks so far" % mem["blink_count"]])
            return {"memory": mem, "act": {"tool": "act_watch_cue",
                                           "args": {"status_box": mem["status_lamp_box"],
                                                    "go_color": spec["go_color"],
                                                    "max_steps": CUE_MAX_STEPS}}}
        mem["stage"] = "press"

    if mem["stage"] == "press":
        remaining = int(mem["blink_count"]) - int(mem["presses_done"])
        if remaining > 0:
            n = min(PRESS_CHUNK, remaining)
            mem["notes"] = "; ".join(notes + ["count=%d done=%d pressing %d" % (mem["blink_count"], mem["presses_done"], n)])
            return {"memory": mem, "act": {"tool": "act_press_button",
                                           "args": {"button": pb, "presses": n,
                                                    "lift": LIFT_ABOVE_TOP, "rot_sign": mem["rot_sign"],
                                                    "home_rpy": mem["home_rpy"]}}}
        mem["stage"] = "submit"

    if mem["stage"] == "submit":
        mem["submit_attempts"] += 1
        mem["notes"] = "; ".join(notes + ["count=%d done=%d submit attempt %d" % (mem["blink_count"], mem["presses_done"], mem["submit_attempts"])])
        if mem["submit_attempts"] <= 2:
            return {"memory": mem, "act": {"tool": "act_press_button",
                                           "args": {"button": mem["submit_button"], "presses": 1,
                                                    "lift": LIFT_ABOVE_TOP, "rot_sign": mem["rot_sign"],
                                                    "home_rpy": mem["home_rpy"],
                                                    "extra_depth": 0.004 * (mem["submit_attempts"] - 1)}}}
        mem["stage"] = "done"
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 20}}}


# ---------------------------------------------------------------- acts
def act_wait(robot, steps=1):
    """Hold still (zero motion, gripper open) for `steps` native steps."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"waited": int(steps), "step": int(robot.now().step)}


def act_watch_cue(robot, status_box, go_color="green", max_steps=700):
    """Hold still while the cue plays; stop on the first frame whose status lamp shows go_color."""
    n = 0
    seen = None
    while n < int(max_steps):
        if _lamp_state(robot.now(), status_box) == go_color:
            seen = int(robot.now().step)
            break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    return {"steps": n, "go_seen_step": seen, "step": int(robot.now().step)}


def _calib_rot_sign(robot, rpy):
    """Ask the live controller which sign maps an orientation error to a rotation command."""
    f = robot.now()
    p = f.tcp
    Rm = _rpy_to_R(float(p[3]), float(p[4]), float(p[5]))
    Rt = _rpy_to_R(float(p[3]), float(p[4]) + 0.05, float(p[5]))
    mine = _rotvec(Rt @ Rm.T) / 0.1
    try:
        chk = robot.dry_run(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]) + 0.05,
                            float(p[5]), mode="absolute")
        cmd = np.array(chk["command"][3:6], dtype=float)
    except Exception:
        return 0.0, "dry_run failed"
    d = float(np.dot(cmd, mine))
    if abs(d) < 1e-6:
        return 0.0, "ambiguous %s" % str(cmd.round(3).tolist())
    return (1.0 if d > 0 else -1.0), "cmd %s mine %s" % (str(cmd.round(3).tolist()), str(mine.round(3).tolist()))


def _step_to(robot, target, rpy, rot_sign, grip, gain=1.6, rot_gain=0.4):
    f = robot.now()
    p = np.asarray(f.tcp, dtype=float)
    e = np.asarray(target, dtype=float) - p[:3]
    cmd = np.clip(e * gain / 0.1, -1.0, 1.0)
    rc = np.zeros(3)
    if rot_sign != 0 and rpy is not None:
        Rm = _rpy_to_R(p[3], p[4], p[5])
        Rd = _rpy_to_R(rpy[0], rpy[1], rpy[2])
        rv = _rotvec(Rd @ Rm.T)
        rc = rot_sign * rot_gain * rv / 0.1
        nr = float(np.linalg.norm(rc))
        if nr > 0.5:
            rc = rc * 0.5 / nr
    robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(rc[0]), float(rc[1]), float(rc[2]), grip)
    return robot.now()


def _move(robot, target, rpy, rot_sign, tol=0.004, max_steps=60, grip=-1.0, settle=0.003):
    """Closed-loop move; stops when within tol AND the last measured step moved < settle (m)."""
    n = 0
    prev = None
    while n < max_steps:
        p = np.asarray(robot.now().tcp, dtype=float)
        moved = 1.0 if prev is None else float(np.linalg.norm(p[:3] - prev))
        if np.linalg.norm(np.asarray(target) - p[:3]) < tol and (settle is None or moved < settle):
            break
        prev = p[:3].copy()
        _step_to(robot, target, rpy, rot_sign, grip)
        n += 1
    p = np.asarray(robot.now().tcp, dtype=float)
    return n, float(np.linalg.norm(np.asarray(target) - p[:3]))


def act_press_button(robot, button, presses=1, lift=0.09, rot_sign=0.0, home_rpy=None, extra_depth=0.0):
    """Close the gripper, hover above button [x,y,ztop], press it `presses` times
    (descend until bottomed/stalled below the cap top, then lift `lift` above the top).
    Returns per-press measured bottom z and step counts."""
    bx, by, bz = float(button[0]), float(button[1]), float(button[2])
    info = {}
    if rot_sign == 0:
        rot_sign, why = _calib_rot_sign(robot, home_rpy)
        info["rot_calib"] = why
    info["rot_sign"] = rot_sign
    hover = [bx, by, bz + lift]
    n_h, err_h = _move(robot, hover, home_rpy, rot_sign, tol=0.010, max_steps=80)
    info["hover_steps"], info["hover_err"] = n_h, round(err_h, 4)
    log = []
    for k in range(int(presses)):
        s0 = int(robot.now().step)
        target = [bx, by, bz - PRESS_BELOW_TOP - extra_depth]
        prev = float(robot.now().tcp[2])
        n = 0
        bottom = False
        stall = 0
        while n < 40:
            f = _step_to(robot, target, home_rpy, rot_sign, -1.0)
            n += 1
            z = float(f.tcp[2])
            if z < bz - BOTTOM_BELOW_TOP - extra_depth:
                bottom = True
            if z < bz and abs(z - prev) < 0.0008:
                stall += 1
            prev = z
            if bottom or stall >= 2:
                break
        zmin = float(robot.now().tcp[2])
        for _ in range(2):  # brief hold at the bottom
            f = _step_to(robot, target, home_rpy, rot_sign, -1.0)
            zmin = min(zmin, float(f.tcp[2]))
        n_up, _ = _move(robot, [bx, by, bz + lift], home_rpy, rot_sign, tol=0.008, max_steps=40, settle=None)
        p = robot.now().tcp
        log.append([k, s0, n, round(zmin, 4), bool(bottom), stall, n_up,
                    round(float(p[0]) - bx, 4), round(float(p[1]) - by, 4)])
    info["presses"] = log
    info["presses_completed"] = len(log)
    info["step"] = int(robot.now().step)
    return info
