"""TimedTransferHard policy: pick the cube from the source disc, carry it above the
target disc, wait for the lamp-timed count and lower it into the target zone so it
arrives on the requested step.  All perception uses the front/wrist RGB-D cameras."""

# ---------------------------------------------------------------- constants
CUBE_HALF = 0.02          # cube half size (top at ~0.047 when resting on a 0.007 disc)
GRASP_ABOVE_CENTER = 0.006  # TCP z above cube centre when grasping (demo median ~0.005)
APPROACH_ABOVE = 0.07     # pre-grasp TCP height above cube top
CARRY_CENTER_Z = 0.135    # cube centre height while transporting
HOVER_CENTER_Z = 0.105    # cube centre height while waiting above the target disc
ZONE_CENTER_Z = 0.068     # demo-estimated cube-centre height at which success fires
LAND_CENTER_Z = 0.030     # cube centre height when resting on the target disc
DESCENT_LEAD = 2          # descent starts at green+count-2 -> zone entry at green+count (window measured 190..210)
LAMP_MIN_PIXELS = 20

COLORS = {
    "red": (207, 87, 87),
    "green": (87, 197, 87),
}


def _clip(v, lo=-1.0, hi=1.0):
    return float(min(hi, max(lo, v)))


def _parse_goal(goal):
    g = (goal or "").lower()
    words = g.replace(",", " ").replace(".", " ").split()
    count = 200
    for i, w in enumerate(words[:-1]):
        if w == "step" and words[i + 1].isdigit():
            count = int(words[i + 1])
    src, dst = "green", "red"
    if " from the " in g and " to the " in g:
        a = g.split(" from the ", 1)[1]
        src = a.split()[0]
        b = a.split(" to the ", 1)
        if len(b) > 1:
            dst = b[1].split()[0]
    obj = "blue"
    if "move the " in g:
        obj = g.split("move the ", 1)[1].split()[0]
    lamp_on = "green"
    if "turns " in g:
        lamp_on = g.split("turns ", 1)[1].split()[0].strip(",.")
    return {"count": count, "src": src, "dst": dst, "obj": obj, "lamp": lamp_on}


def _lamp_pixels(frame):
    im = frame.front_rgb
    r = im[..., 0]
    g = im[..., 1]
    b = im[..., 2]
    return int(((g > 220) & (r < 60) & (b < 60)).sum())


def _backproject(depth, K, E, mask):
    ys, xs = np.nonzero(mask)
    d = depth[ys, xs].astype(float) / 1000.0
    ok = d > 0
    ys, xs, d = ys[ok], xs[ok], d[ok]
    if len(d) == 0:
        return np.zeros((0, 3))
    x = (xs - K[0, 2]) / K[0, 0] * d
    y = (ys - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (pc - t) @ R


def _color_mask(im, name):
    im = im.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if name == "blue":
        return (b > r + 40) & (b > g + 40)
    if name in COLORS:
        c = COLORS[name]
        return (np.abs(im - np.array(c)).max(-1) <= 6)
    if name == "red":
        return (r > g + 80) & (r > b + 80)
    return np.zeros(im.shape[:2], dtype=bool)


def _scene(frame, gp):
    """Front-camera estimates of cube centre, source and target disc centres."""
    out = {}
    for key, name in (("cube", gp["obj"]), ("src", gp["src"]), ("dst", gp["dst"])):
        m = _color_mask(frame.front_rgb, name)
        p = _backproject(frame.front_depth, frame.front_K, frame.front_extrinsic, m)
        if len(p) < 5:
            out[key] = None
            out[key + "_px"] = int(len(p))
            continue
        out[key + "_px"] = int(len(p))
        if key == "cube":
            top = float(np.percentile(p[:, 2], 90))
            sel = p[p[:, 2] > top - 0.006]
            if len(sel) < 3:
                sel = p
            out[key] = [float(np.mean(sel[:, 0])), float(np.mean(sel[:, 1])), top - CUBE_HALF]
        else:
            out[key] = [float(np.mean(p[:, 0])), float(np.mean(p[:, 1])), float(np.median(p[:, 2]))]
    return out


def _wrist_cube(frame, obj="blue"):
    """Cube top-face world xy and z seen by the wrist camera (cube held or below)."""
    m = _color_mask(frame.wrist_rgb, obj)
    p = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic, m)
    if len(p) < 30:
        return None
    top = float(np.percentile(p[:, 2], 60))
    sel = p[np.abs(p[:, 2] - top) < 0.006]
    if len(sel) < 10:
        sel = p
    return {"x": float(np.mean(sel[:, 0])), "y": float(np.mean(sel[:, 1])),
            "top": float(np.median(sel[:, 2])), "px": int(len(p))}


def _watch(frame, st):
    """Update lamp onset record from one frame."""
    if st["green_step"] < 0:
        if _lamp_pixels(frame) >= LAMP_MIN_PIXELS:
            st["green_step"] = int(frame.step)
            st["green_first_seen_frame"] = st["green_step"] == st.get("first_step", -999)


def _step(robot, st, a, grip):
    f = robot.step(_clip(a[0]), _clip(a[1]), _clip(a[2]), 0.0, 0.0, 0.0, grip)
    st["n"] += 1
    _watch(f, st)
    return f


def _servo(robot, st, target, grip, tol=0.004, max_steps=60, gain=1.2, amax=1.0, settle_v=0.004):
    """P-control of the TCP position toward target (world xyz) with zero rotation."""
    f = robot.now()
    prev = np.asarray(f.tcp[:3], dtype=float)
    k = 0
    err = np.asarray(target, dtype=float) - prev
    for k in range(max_steps):
        p = np.asarray(f.tcp[:3], dtype=float)
        err = np.asarray(target, dtype=float) - p
        v = float(np.linalg.norm(p - prev))
        if float(np.linalg.norm(err)) < tol and v < settle_v and k > 0:
            break
        a = np.clip(gain * err / 0.1, -amax, amax)
        prev = p
        f = _step(robot, st, a, grip)
    p = np.asarray(f.tcp[:3], dtype=float)
    return {"steps": k, "err": float(np.linalg.norm(np.asarray(target) - p))}


def _new_state(robot, green_step):
    f = robot.now()
    st = {"n": 0, "green_step": int(green_step), "first_step": int(f.step)}
    _watch(f, st)
    return st


def _last_number(last, key):
    """Read a numeric field from the previous action's result text ('tool -> {json}')."""
    if last is None:
        return None
    text = str(last)
    tag = '"' + key + '": '
    i = text.find(tag)
    if i < 0:
        return None
    j = i + len(tag)
    k = j
    while k < len(text) and (text[k].isdigit() or text[k] in "-."):
        k += 1
    try:
        return float(text[j:k])
    except ValueError:
        return None


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Stage machine: pick -> transport -> timed place. Tracks the lamp onset step."""
    gp = _parse_goal(goal)
    mem = {
        "stage": memory.get("stage") or "pick",
        "green_step": memory.get("green_step", -1),
        "cube": memory.get("cube") or [],
        "target": memory.get("target") or [],
        "attempts": memory.get("attempts", 0) or 0,
        "log": memory.get("log") or "",
    }
    if mem["green_step"] is None:
        mem["green_step"] = -1
    # lamp onset from every frame since the previous action
    if mem["green_step"] < 0:
        for fr in obs.frames:
            if _lamp_pixels(fr) >= LAMP_MIN_PIXELS:
                mem["green_step"] = int(fr.step)
                break
    gs = _last_number(last, "green_step")
    if gs is not None and gs >= 0 and (mem["green_step"] < 0 or gs < mem["green_step"]):
        mem["green_step"] = int(gs)
    cur = obs.current
    if mem["stage"] == "pick":
        sc = _scene(cur, gp)
        if sc["cube"] is not None:
            mem["cube"] = sc["cube"]
        if sc["dst"] is not None and not mem["target"]:
            mem["target"] = sc["dst"][:2]
    if not mem["cube"] or not mem["target"]:
        mem["log"] = (mem["log"] + " | perception missing").strip()[-600:]
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1, "green_step": int(mem["green_step"])}}}
    fingers = float(np.mean(cur.fingers))
    if mem["stage"] == "pick":
        mem["attempts"] += 1
        mem["stage"] = "check_grasp"
        return {"memory": mem, "act": {"tool": "act_pick", "args": {
            "x": mem["cube"][0], "y": mem["cube"][1], "zc": mem["cube"][2],
            "green_step": int(mem["green_step"])}}}
    if mem["stage"] == "check_grasp":
        # holding the cube leaves the fingers part-closed (~0.018 m each); 0 = missed, 0.04 = open
        ok = 0.008 < fingers < 0.035
        if not ok and mem["attempts"] < 3:
            mem["stage"] = "pick"
            mem["log"] = (mem["log"] + " | grasp failed fingers=%.4f" % fingers)[-600:]
            return {"memory": mem, "act": {"tool": "act_release_lift", "args": {"green_step": int(mem["green_step"])}}}
        mem["stage"] = "place"
        return {"memory": mem, "act": {"tool": "act_transport", "args": {
            "tx": mem["target"][0], "ty": mem["target"][1], "green_step": int(mem["green_step"])}}}
    # place (repeated if it returns without the episode ending)
    mem["stage"] = "place"
    return {"memory": mem, "act": {"tool": "act_timed_place", "args": {
        "tx": mem["target"][0], "ty": mem["target"][1], "green_step": int(mem["green_step"]),
        "count": int(gp["count"])}}}


# ---------------------------------------------------------------- actions
def act_wait(robot, steps=1, green_step=-1):
    """Hold position for a few steps (gripper unchanged-open)."""
    st = _new_state(robot, green_step)
    g = 1.0 if float(np.mean(robot.now().fingers)) > 0.035 else -1.0
    for _ in range(int(steps)):
        _step(robot, st, [0, 0, 0], g)
    return {"green_step": st["green_step"], "step": int(robot.now().step)}


def act_pick(robot, x, y, zc, green_step=-1):
    """Approach above the cube, refine with the wrist camera, descend, close and lift."""
    st = _new_state(robot, green_step)
    top = zc + CUBE_HALF
    r1 = _servo(robot, st, [x, y, top + APPROACH_ABOVE], 1.0, tol=0.006, max_steps=40)
    w = _wrist_cube(robot.now())
    rx, ry = x, y
    if w is not None and abs(w["x"] - x) < 0.03 and abs(w["y"] - y) < 0.03:
        rx, ry = 0.5 * (x + w["x"]), 0.5 * (y + w["y"])
    r2 = _servo(robot, st, [rx, ry, top + 0.03], 1.0, tol=0.005, max_steps=25)
    gz = zc + GRASP_ABOVE_CENTER
    r3 = _servo(robot, st, [rx, ry, gz], 1.0, tol=0.003, max_steps=30, gain=1.0)
    for _ in range(6):
        _step(robot, st, [0, 0, 0], -1.0)
    fing = float(np.mean(robot.now().fingers))
    grasped = fing > 0.008
    r4 = _servo(robot, st, [rx, ry, CARRY_CENTER_Z + 0.01], -1.0, tol=0.01, max_steps=25)
    fing2 = float(np.mean(robot.now().fingers))
    return {"grasped": bool(grasped and fing2 > 0.008), "fingers": fing2,
            "pick_xy": [rx, ry], "wrist": w, "servo": [r1, r2, r3, r4],
            "green_step": st["green_step"], "step": int(robot.now().step), "n": st["n"]}


def act_release_lift(robot, green_step=-1):
    """Open the gripper and lift after a failed grasp."""
    st = _new_state(robot, green_step)
    p = robot.now().tcp
    for _ in range(4):
        _step(robot, st, [0, 0, 0], 1.0)
    _servo(robot, st, [float(p[0]), float(p[1]), 0.15], 1.0, tol=0.01, max_steps=20)
    return {"green_step": st["green_step"], "step": int(robot.now().step)}


def act_transport(robot, tx, ty, green_step=-1):
    """Carry the held cube to the carry height above the target disc centre."""
    st = _new_state(robot, green_step)
    f = robot.now()
    w = _wrist_cube(f)
    p = f.tcp
    off = [0.0, 0.0, -0.011]
    if w is not None:
        off = [w["x"] - float(p[0]), w["y"] - float(p[1]), w["top"] - CUBE_HALF - float(p[2])]
    r = _servo(robot, st, [tx - off[0], ty - off[1], CARRY_CENTER_Z - off[2]], -1.0,
               tol=0.006, max_steps=60)
    return {"offset": off, "servo": r, "fingers": float(np.mean(robot.now().fingers)),
            "green_step": st["green_step"], "step": int(robot.now().step), "n": st["n"]}


def act_timed_place(robot, tx, ty, green_step, count=200):
    """Hover above the target, then descend so the cube enters the target zone on
    step green_step+count; keep lowering to rest and hold."""
    st = _new_state(robot, green_step)
    f = robot.now()
    w = _wrist_cube(f)
    p = f.tcp
    off = [0.0, 0.0, -0.011]
    if w is not None:
        off = [w["x"] - float(p[0]), w["y"] - float(p[1]), w["top"] - CUBE_HALF - float(p[2])]
    hover = [tx - off[0], ty - off[1], HOVER_CENTER_Z - off[2]]
    log = []
    # hover until the descent time (lamp may still be unseen: hover meanwhile)
    hold_steps = 0
    while True:
        f = robot.now()
        gs = st["green_step"]
        if gs >= 0 and int(f.step) >= gs + count - DESCENT_LEAD:
            break
        if int(f.step) > 290:
            break
        pp = np.asarray(f.tcp[:3], dtype=float)
        err = np.asarray(hover) - pp
        a = np.clip(1.0 * err / 0.1, -0.6, 0.6)
        a = np.where(np.abs(err) < 0.0015, 0.0, a)
        _step(robot, st, a, -1.0)
        hold_steps += 1
    f = robot.now()
    w2 = _wrist_cube(f)
    if w2 is not None:
        off = [w2["x"] - float(f.tcp[0]), w2["y"] - float(f.tcp[1]), w2["top"] - CUBE_HALF - float(f.tcp[2])]
    start = {"step": int(f.step), "tcp": [float(v) for v in f.tcp[:3]], "cube_z": float(f.tcp[2]) + off[2]}
    land = [tx - off[0], ty - off[1], LAND_CENTER_Z - off[2]]
    # fast descent until cube centre is below the zone height, then land gently
    for i in range(40):
        f = robot.now()
        pp = np.asarray(f.tcp[:3], dtype=float)
        cz = float(pp[2]) + off[2]
        log.append([int(f.step), round(cz, 4)])
        exy = np.asarray(land[:2]) - pp[:2]
        axy = np.clip(1.0 * exy / 0.1, -0.3, 0.3)
        if cz > ZONE_CENTER_Z - 0.02:
            az = -1.0
        else:
            az = _clip(1.0 * (land[2] - pp[2]) / 0.1, -0.5, 0.5)
        _step(robot, st, [axy[0], axy[1], az], -1.0)
    # not ended: report (episode continues with next decision)
    f = robot.now()
    return {"hold_steps": hold_steps, "start": start, "descent_log": log[:12],
            "final_tcp": [float(v) for v in f.tcp[:3]], "offset": off,
            "green_step": st["green_step"], "step": int(f.step), "n": st["n"]}
