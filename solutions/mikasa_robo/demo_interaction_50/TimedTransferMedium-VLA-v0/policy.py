"""TimedTransferMedium policy: pick the blue cube from the green disc, hover it above the red disc,
and lower it onto the red disc so that it arrives at step ~150 counted from the frame where the white
lamp first appears green.  All perception uses RGB-D + calibration; no hidden state."""

# ---------------- constants (from demo analysis, see task_analysis.md) ----------------
LAMP_MIN_PIXELS = 20          # front-camera lamp-green pixels; demos: 0 before onset, >=178 after
TARGET_COUNT = 150            # instruction: step 150 of the count
DESCENT_LEAD = 4              # descent starts at onset + 150 - DESCENT_LEAD (arrival ~onset+150; probes: 142..156 ok; 139 fail; fail at 158 if not placed)
FALLBACK_AFTER = 175          # steps after onset: if still no success, set the cube down and release
# expert-derived calibration: expert TCP relative to camera-median estimates
FRONT_CUBE_OFF = (-0.0188, -0.0008)
WRIST_CUBE_OFF = (-0.0092, -0.0041)
FRONT_RED_OFF = (-0.0100, 0.0006)
WRIST_RED_OFF = (-0.0059, 0.0007)
GRASP_Z = 0.030               # TCP height for grasp (expert 0.027-0.044)
APPROACH_Z = 0.10
HOVER_Z = 0.19                # expert hover 0.188-0.199
PLACE_ABOVE_GRASP = 0.012     # final hold: TCP this far above grasp height (cube just above disc)


def _masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    lamp = (g > 215) & (r < 90) & (b < 90)
    return {
        "red": (r > 150) & (np.abs(g - b) < 25) & (r - g > 70),
        "blue": (b > r + 50) & (b > g + 50),
        "lamp": lamp,
        "gdisc": (g > r + 60) & (g > b + 60) & ~lamp,
    }


def _backproject(depth, K, E, mask):
    ys, xs = np.nonzero(mask & (depth > 0))
    if len(ys) == 0:
        return None
    z = depth[ys, xs].astype(float) / 1000.0
    xc = (xs + 0.5 - K[0, 2]) * z / K[0, 0]
    yc = (ys + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([xc, yc, z], 1)
    R = np.asarray(E)[:, :3].astype(float)
    t = np.asarray(E)[:, 3].astype(float)
    return (pc - t) @ R


def _lamp_pixels(frame):
    return int(_masks(frame.front_rgb)["lamp"].sum())


def _locate(frame):
    """Cube grasp xy and red-disc place xy (expert-calibrated) from front and wrist cameras."""
    out = {"n": {}}
    ests = {"cube": [], "red": []}
    for cam, offs in (("front", (FRONT_CUBE_OFF, FRONT_RED_OFF)), ("wrist", (WRIST_CUBE_OFF, WRIST_RED_OFF))):
        rgb = getattr(frame, cam + "_rgb")
        dep = getattr(frame, cam + "_depth")
        K = getattr(frame, cam + "_K")
        E = getattr(frame, cam + "_extrinsic")
        m = _masks(rgb)
        for key, mk, off, nmin in (("cube", "blue", offs[0], 25), ("red", "red", offs[1], 80)):
            P = _backproject(dep, K, E, m[mk])
            n = 0 if P is None else len(P)
            out["n"][cam + "_" + key] = n
            if n >= nmin:
                med = np.median(P, 0)
                ests[key].append([float(med[0] + off[0]), float(med[1] + off[1]), float(med[2])])
    for key in ("cube", "red"):
        if ests[key]:
            a = np.array(ests[key])
            out[key] = [float(v) for v in a.mean(0)]
            out[key + "_spread"] = float(np.max(np.abs(a[:, :2] - a[:, :2].mean(0)))) if len(a) > 1 else 0.0
        else:
            out[key] = None
    return out


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return w * ang


def _command(tcp, target, rpy_target, vmax=1.0, rot_gain=0.5):
    """Normalized 6-D command toward target xyz and orientation (CPU IK left-multiplies Rx(-0.1r)...)."""
    tcp = np.asarray(tcp, dtype=float)
    d = (np.asarray(target, dtype=float) - tcp[:3]) / 0.1
    n = np.max(np.abs(d))
    if n > vmax:
        d = d * (vmax / n)
    Rm = _rpy_to_R(tcp[3], tcp[4], tcp[5])
    Rt = _rpy_to_R(rpy_target[0], rpy_target[1], rpy_target[2])
    w = _rotvec(Rt @ Rm.T)
    r = -rot_gain * w / 0.1
    rn = np.linalg.norm(r)
    if rn > 0.5:
        r = r * (0.5 / rn)
    return [float(d[0]), float(d[1]), float(d[2]), float(r[0]), float(r[1]), float(r[2])]


def _watch(robot, st):
    """Update lamp onset bookkeeping from the latest frame."""
    f = robot.now()
    if st["onset"] is None and _lamp_pixels(f) >= LAMP_MIN_PIXELS:
        st["onset"] = int(f.step)


def _servo(robot, st, target, grip, max_steps, tol, vmax=1.0, settle_steps=0):
    """Closed-loop move to target; returns steps used and final error."""
    used = 0
    calm = 0
    err = 1.0
    for _ in range(max_steps):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        err = float(np.max(np.abs(tcp[:3] - np.asarray(target))))
        if err < tol:
            calm += 1
            if calm > settle_steps:
                break
        else:
            calm = 0
        cmd = _command(tcp, target, st["rpy"], vmax=vmax)
        robot.step(*cmd, grip)
        used += 1
        _watch(robot, st)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    err = float(np.max(np.abs(tcp[:3] - np.asarray(target))))
    return used, err


def perceive_decide(obs, goal, memory, last):
    """Track stage, lamp onset (first green-lamp frame step) and object locations; choose next act."""
    mem = dict(memory)
    for k, v in (("stage", "init"), ("onset_step", -1), ("cube_xy", []), ("red_xy", []), ("rpy0", []),
                 ("grasp_z", GRASP_Z), ("attempts", 0), ("log", "")):
        if k not in mem or mem[k] is None:
            mem[k] = v
    # lamp onset from every frame since the previous action
    if mem["onset_step"] < 0:
        for f in obs.frames:
            if _lamp_pixels(f) >= LAMP_MIN_PIXELS:
                mem["onset_step"] = int(f.step)
                break
    if isinstance(last, dict):
        res = last
    else:
        res = None
    cur = obs.current
    if not mem["rpy0"]:
        t = [float(v) for v in cur.tcp]
        mem["rpy0"] = [math.pi if abs(t[3]) > 2.0 else t[3], 0.0, t[5]]
    if mem["stage"] == "init" or mem["stage"] == "regrasp":
        loc = _locate(cur)
        if loc["cube"] is not None:
            mem["cube_xy"] = [loc["cube"][0], loc["cube"][1]]
        if loc["red"] is not None and (not mem["red_xy"] or mem["stage"] == "init"):
            mem["red_xy"] = [loc["red"][0], loc["red"][1]]
        mem["log"] = (mem["log"] + " | loc@%d n=%s cube=%s red=%s" % (
            int(cur.step), loc["n"], mem["cube_xy"], mem["red_xy"]))[-1500:]
        if not mem["cube_xy"] or not mem["red_xy"]:
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 2, "onset_step": mem["onset_step"]}}}
        mem["stage"] = "picking"
        mem["attempts"] = int(mem["attempts"]) + 1
        return {"memory": mem, "act": {"tool": "act_pick", "args": {
            "cube_xy": mem["cube_xy"], "rpy": mem["rpy0"], "onset_step": mem["onset_step"]}}}
    if mem["stage"] == "picking":
        # outcome of act_pick is in `last` text; re-check grasp from fingers
        fing = [float(v) for v in cur.fingers]
        if mem["onset_step"] < 0 and isinstance(last, str) and '"onset_step": ' in last:
            pass
        held = 0.008 < (fing[0] + fing[1]) / 2.0 < 0.03
        mem["log"] = (mem["log"] + " | pick@%d fingers=%.4f held=%s" % (int(cur.step), (fing[0] + fing[1]) / 2, held))[-1500:]
        if not held:
            mem["stage"] = "regrasp" if int(mem["attempts"]) < 3 else "transport"
            if mem["stage"] == "regrasp":
                return {"memory": mem, "act": {"tool": "act_release_up", "args": {"rpy": mem["rpy0"]}}}
        mem["grasp_z"] = float(mem["grasp_z"])
        mem["stage"] = "transport"
    if mem["stage"] == "transport":
        mem["stage"] = "placing"
        return {"memory": mem, "act": {"tool": "act_transport_and_place", "args": {
            "red_xy": mem["red_xy"], "rpy": mem["rpy0"], "onset_step": mem["onset_step"],
            "grasp_z": mem["grasp_z"]}}}
    # placing returned without episode end: keep holding low
    return {"memory": mem, "act": {"tool": "act_transport_and_place", "args": {
        "red_xy": mem["red_xy"], "rpy": mem["rpy0"], "onset_step": mem["onset_step"],
        "grasp_z": mem["grasp_z"]}}}


def act_hold(robot, steps=2, onset_step=-1):
    """Hold the current pose with an open gripper for a few steps (used when perception failed)."""
    st = {"onset": None if onset_step < 0 else onset_step, "rpy": None}
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
        _watch(robot, st)
    return {"step": int(robot.now().step), "onset_step": st["onset"]}


def act_pick(robot, cube_xy, rpy, onset_step=-1):
    """Approach above the cube, descend to grasp height, close the gripper and lift to hover height.
    Returns measured finger opening (cube held ~0.018-0.019 m), TCP and the lamp onset step if seen."""
    st = {"onset": None if onset_step < 0 else int(onset_step), "rpy": rpy}
    _watch(robot, st)
    out = {"start_step": int(robot.now().step)}
    cx, cy = float(cube_xy[0]), float(cube_xy[1])
    u1, e1 = _servo(robot, st, [cx, cy, APPROACH_Z], 1.0, 30, 0.012)
    u2, e2 = _servo(robot, st, [cx, cy, GRASP_Z], 1.0, 30, 0.004, vmax=0.8, settle_steps=1)
    out["approach"] = [u1, round(e1, 4)]
    out["descend"] = [u2, round(e2, 4)]
    out["tcp_at_grasp"] = [round(float(v), 4) for v in robot.now().tcp]
    tcp = np.asarray(robot.now().tcp, dtype=float)
    for _ in range(4):
        cmd = _command(tcp, [cx, cy, float(tcp[2])], rpy, vmax=0.3)
        robot.step(0, 0, 0, cmd[3] * 0.0, cmd[4] * 0.0, cmd[5] * 0.0, -1.0)
        _watch(robot, st)
    out["fingers_closed"] = [round(float(v), 4) for v in robot.now().fingers]
    u3, e3 = _servo(robot, st, [cx, cy, HOVER_Z], -1.0, 30, 0.01)
    out["lift"] = [u3, round(e3, 4)]
    out["fingers_lifted"] = [round(float(v), 4) for v in robot.now().fingers]
    out["onset_step"] = -1 if st["onset"] is None else st["onset"]
    out["end_step"] = int(robot.now().step)
    return out


def act_release_up(robot, rpy):
    """Open the gripper and rise to approach height (recovery after a missed grasp)."""
    st = {"onset": None, "rpy": rpy}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    for _ in range(3):
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
    u, e = _servo(robot, st, [float(tcp[0]), float(tcp[1]), max(APPROACH_Z, float(tcp[2]))], 1.0, 20, 0.01)
    return {"steps": u, "err": round(e, 4), "step": int(robot.now().step)}


def act_transport_and_place(robot, red_xy, rpy, onset_step, grasp_z=GRASP_Z):
    """Move the held cube above the red disc at hover height, hold there until step
    onset+150-DESCENT_LEAD, then descend under P-control (about 4 steps) and keep the cube just above
    the disc.  The episode is expected to end with success during the low hold; a late fallback
    (onset+FALLBACK_AFTER) sets the cube down on the disc and opens the gripper."""
    st = {"onset": None if onset_step is None or onset_step < 0 else int(onset_step), "rpy": rpy}
    _watch(robot, st)
    rx, ry = float(red_xy[0]), float(red_xy[1])
    out = {"start_step": int(robot.now().step)}
    hover = [rx, ry, HOVER_Z]
    low_z = float(grasp_z) + PLACE_ABOVE_GRASP
    u, e = _servo(robot, st, hover, -1.0, 40, 0.006)
    out["transport"] = [u, round(e, 4), int(robot.now().step)]
    held = 0
    while True:
        f = robot.now()
        if st["onset"] is not None and int(f.step) >= st["onset"] + TARGET_COUNT - DESCENT_LEAD:
            break
        if int(f.step) > 235:
            break
        cmd = _command(np.asarray(f.tcp, dtype=float), hover, rpy)
        robot.step(*cmd, -1.0)
        held += 1
        _watch(robot, st)
    out["hover_steps"] = held
    out["onset_step"] = -1 if st["onset"] is None else st["onset"]
    out["descent_start"] = int(robot.now().step)
    out["hover_tcp"] = [round(float(v), 4) for v in robot.now().tcp]
    out["fingers_hover"] = [round(float(v), 4) for v in robot.now().fingers]
    zs = []
    onset = st["onset"] if st["onset"] is not None else 0
    released = False
    for _ in range(300):
        f = robot.now()
        s = int(f.step)
        if s >= 249:
            break
        if s - onset >= FALLBACK_AFTER:
            target, grip = [rx, ry, float(grasp_z) + 0.004], -1.0
            if s - onset >= FALLBACK_AFTER + 6:
                grip = 1.0
                released = True
        else:
            target, grip = [rx, ry, low_z], -1.0
        cmd = _command(np.asarray(f.tcp, dtype=float), target, rpy)
        robot.step(*cmd, grip)
        if len(zs) < 25:
            zs.append([int(robot.now().step) - onset, round(float(robot.now().tcp[2]), 4)])
    out["descent_rel_z"] = zs
    out["released"] = released
    out["end_step"] = int(robot.now().step)
    out["end_tcp"] = [round(float(v), 4) for v in robot.now().tcp]
    out["fingers"] = [round(float(v), 4) for v in robot.now().fingers]
    return out
