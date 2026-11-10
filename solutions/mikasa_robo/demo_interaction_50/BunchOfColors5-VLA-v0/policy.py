"""BunchOfColors5: remember the cued cube colours, touch those cubes in the arc, press the centre button.

Observed task structure (from 50 demos): frame 0 shows a cue of N (=5) coloured cubes at fixed slots in front of the
button; the cue vanishes (1-5 frames), the table is empty (1-5 frames), then an arc of 9 cubes, one per colour, appears.
The expert hovers the open gripper TCP ~1-2 cm above each cued cube (TCP z 0.045-0.062), lifts, then presses the pink
button cap with a closed gripper (TCP z stalls ~0.053) and lifts; native success fires while lifting (TCP z ~0.17).
"""

PALETTE = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "orange": (1.0, 0.72, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
}
# brightness split inside the two families that contain a bright and a dark member
DARK_NAME = {"red": "dark_red", "cyan": "teal"}
CUE_SPLIT = {"red": 178.0, "cyan": 186.0}     # cue slots: red 200 vs 156, cyan 223 vs 150
ARC_SPLIT = {"red": 165.0, "cyan": 165.0}     # arc: red 178-208 vs 140-152, cyan 179-208 vs 131-152

CUBE_X_OFFSET = -0.010   # expert touch xy - top-face centroid (front camera sees front faces): mean -0.010 m
BUTTON_X_OFFSET = 0.008  # press point relative to pink cap top centroid (expert +0.008..+0.022)
TRANSIT_Z = 0.13
TOUCH_Z = 0.048          # R011: hover at 0.066 did NOT count; 0.050-0.051 always counted
RETOUCH_Z = 0.045        # lower second pass (expert minimum) used only if two presses fail
PRESS_Z = 0.040          # commanded; the cap stops the TCP near 0.053
LIFT_Z = 0.24
GAIN = 0.8               # command = GAIN * error / 0.1, clipped to [-1, 1]
ROT_GAIN = 0.4


# ---------------------------------------------------------------- perception helpers
def _world_points(frame):
    D = np.asarray(frame.front_depth).astype(np.float64) / 1000.0
    K = np.asarray(frame.front_K).astype(np.float64)
    E = np.asarray(frame.front_extrinsic).astype(np.float64)
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * D
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * D
    Pc = np.stack([x, y, D], -1)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    return Pw, D > 0


def _components(mask, min_size):
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    ys, xs = np.nonzero(mask)
    comps = []
    n = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        stack = [(y0, x0)]
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        if len(pix) >= min_size:
            comps.append(np.array(pix))
    return comps


def _family(c):
    m = float(max(c))
    if m < 60:
        return None, m
    r = np.array(c, dtype=np.float64) / m
    best, bd = None, 9.0
    for name, ref in PALETTE.items():
        d = float(np.sum((r - np.array(ref)) ** 2))
        if d < bd:
            best, bd = name, d
    return best, m


def _scene_cubes(frame):
    """Saturated blobs standing on the table (world z 0.008-0.10) in the front camera."""
    P, ok = _world_points(frame)
    I = np.asarray(frame.front_rgb).astype(np.int32)
    sat = I.max(-1) - I.min(-1)
    z = P[..., 2]
    mask = ok & (z > 0.008) & (z < 0.10) & (sat > 80) & (P[..., 0] > -0.5) & (P[..., 0] < 0.4) & (np.abs(P[..., 1]) < 0.6)
    cubes = []
    for pix in _components(mask, 15):
        yy, xx = pix[:, 0], pix[:, 1]
        pts = P[yy, xx]
        col = I[yy, xx].mean(0)
        fam, bright = _family(col)
        top = pts[pts[:, 2] > 0.032]
        if len(top) >= 4:
            cx, cy = float(top[:, 0].mean()) + CUBE_X_OFFSET, float(top[:, 1].mean())
        else:
            cx, cy = float(pts[:, 0].mean()) - 0.013, float(pts[:, 1].mean())
        cubes.append({"fam": fam, "bright": float(bright), "x": cx, "y": cy, "n": int(len(pix)),
                      "rgb": [int(v) for v in col]})
    return cubes


def _name_cubes(cubes, split, pairwise):
    fams = {}
    for c in cubes:
        fams.setdefault(c["fam"], []).append(c)
    for fam, members in fams.items():
        for c in members:
            c["color"] = fam
        if fam in DARK_NAME:
            if pairwise and len(members) == 2:
                lo = min(members, key=lambda c: c["bright"])
                lo["color"] = DARK_NAME[fam]
            else:
                for c in members:
                    if c["bright"] < split[fam]:
                        c["color"] = DARK_NAME[fam]
    return cubes


def _button(frame):
    """Pink cap top centroid of the centre button."""
    P, ok = _world_points(frame)
    I = np.asarray(frame.front_rgb).astype(np.int32)
    pink = ok & (I[..., 0] > 150) & (I[..., 0] - I[..., 1] > 40) & (I[..., 1] > 70) & \
        (np.abs(I[..., 1] - I[..., 2]) < 30) & (P[..., 2] > 0.045) & (P[..., 2] < 0.09)
    if pink.sum() < 10:
        return None
    pts = P[pink]
    zt = pts[:, 2].max()
    top = pts[pts[:, 2] > zt - 0.004]
    return [float(top[:, 0].mean()) + BUTTON_X_OFFSET, float(top[:, 1].mean()), float(zt)]


def _classify_frame(frame):
    """Returns ('cue'|'arc'|'empty', cubes)."""
    cubes = _scene_cubes(frame)
    if len(cubes) >= 8:
        return "arc", _name_cubes(cubes, ARC_SPLIT, True)
    if len(cubes) >= 1:
        return "cue", _name_cubes(cubes, CUE_SPLIT, False)
    return "empty", cubes


# ---------------------------------------------------------------- control helpers
def _Rm(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _rotvec(Rm):
    c = max(-1.0, min(1.0, (float(np.trace(Rm)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    return w / (2 * s) * th


def _target_R(yaw):
    return _Rm([math.pi, 0.0, yaw])


def _cmd(tcp, goal, yaw, xy_gate_z=None, gain=GAIN, vmax=1.0):
    e = np.array(goal, dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64)
    u = np.clip(gain * e / 0.1, -vmax, vmax)
    if xy_gate_z is not None and float(tcp[2]) < xy_gate_z:
        u[0] = 0.0
        u[1] = 0.0
    ew = _rotvec(_target_R(yaw) @ _Rm(tcp[3:6]).T)
    r = -ROT_GAIN * ew / 0.1
    nr = float(np.linalg.norm(r))
    if nr > 0.5:
        r = r * 0.5 / nr
    return [float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2])]


def _move(robot, goal, grip, yaw, tol_xy, tol_z, max_steps, xy_gate_z=None, stop_fn=None):
    """Closed-loop move of the TCP toward goal; returns (steps, final tcp, reached)."""
    n = 0
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    while n < max_steps:
        e = np.array(goal) - tcp[:3]
        if abs(e[0]) < tol_xy and abs(e[1]) < tol_xy and abs(e[2]) < tol_z:
            return n, tcp, True
        if stop_fn is not None and stop_fn(tcp):
            return n, tcp, True
        robot.step(*_cmd(tcp, goal, yaw, xy_gate_z), grip)
        n += 1
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    return n, tcp, False



BASE_XY = (-0.615, 0.0)   # robot base (J1 axis) in world xy; R001-R027: J4 margin ~0.17 rad at 0.34 m, +0.03/cm
VIA_MIN_R = 0.44          # transit segments passing closer than this to the base go through a via point
VIA_R = 0.47


def _via(a, b):
    """Via point keeping a straight xy transit from a to b at least VIA_R from the base, or None."""
    ax, ay = float(a[0]) - BASE_XY[0], float(a[1]) - BASE_XY[1]
    bx, by = float(b[0]) - BASE_XY[0], float(b[1]) - BASE_XY[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    r = math.hypot(cx, cy)
    if r >= VIA_MIN_R or t <= 0.0 or t >= 1.0:
        return None
    if r < 1e-6:
        cx, cy, r = 1.0, 0.0, 1.0
    return [BASE_XY[0] + cx / r * VIA_R, BASE_XY[1] + cy / r * VIA_R]


def _path_len(a, b):
    v = _via(a, b)
    if v is None:
        return math.hypot(b[0] - a[0], b[1] - a[1])
    return math.hypot(v[0] - a[0], v[1] - a[1]) + math.hypot(b[0] - v[0], b[1] - v[1])


def _transit(robot, x, y, grip, yaw, tol_xy):
    """Rise to TRANSIT_Z and move above (x, y), via a point away from the base when needed."""
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    steps, used_via = 0, False
    v = _via(tcp[:2], [x, y])
    if v is not None:
        used_via = True
        n, tcp, _ = _move(robot, [v[0], v[1], TRANSIT_Z], grip, yaw, 0.03, 0.03, 40, xy_gate_z=0.085)
        steps += n
    n, tcp, ok = _move(robot, [x, y, TRANSIT_Z], grip, yaw, tol_xy, 0.03, 60, xy_gate_z=0.085)
    return steps + n, tcp, ok, used_via


# ---------------------------------------------------------------- planning
def _perms(items):
    if len(items) <= 1:
        return [list(items)]
    out = []
    for i in range(len(items)):
        for rest in _perms(items[:i] + items[i + 1:]):
            out.append([items[i]] + rest)
    return out


def _plan(start, targets, end):
    idx = list(range(len(targets)))
    if len(idx) > 7:
        return idx
    best, bc = idx, 1e9
    for p in _perms(idx):
        c, cur = 0.0, start
        for i in p:
            c += _path_len(cur, targets[i])
            cur = targets[i]
        if end is not None:
            c += _path_len(cur, end)
        if c < bc:
            best, bc = p, c
    return best


# ---------------------------------------------------------------- policy
def _scan_frames(frames, mem):
    """Update cue colours from the first cue frame; return (arc_cubes, arc_frame) from the last arc frame."""
    arc, arc_fr = None, None
    for fr in frames:
        kind, cubes = _classify_frame(fr)
        if kind == "cue" and not mem["cue_colors"] and len(cubes) >= 1:
            mem["cue_colors"] = sorted(set(c["color"] for c in cubes))
            mem["cue_step"] = int(fr.step)
            mem["log"] += "cue@%d:%s;" % (fr.step, ",".join(mem["cue_colors"]))
        if kind == "arc":
            arc, arc_fr = cubes, fr
    return arc, arc_fr


def _plan_targets(mem, arc, fr):
    want = set(mem["cue_colors"])
    tg, tc = [], []
    for c in arc:
        if c["color"] in want:
            tg.append([round(c["x"], 4), round(c["y"], 4)])
            tc.append(c["color"])
    btn = _button(fr)
    if btn is None:
        btn = [-0.112, 0.0, 0.059]
        mem["log"] += "button_default;"
    tcp = fr.tcp
    order = _plan([float(tcp[0]), float(tcp[1])], tg, btn)
    mem["targets"] = [tg[i] for i in order]
    mem["target_colors"] = [tc[i] for i in order]
    mem["button"] = [round(float(v), 4) for v in btn]
    names = [c["color"] for c in arc]
    mem["log"] += "arc@%d n=%d uniq=%d targets=%s;" % (fr.step, len(arc), len(set(names)), ",".join(mem["target_colors"]))


def perceive_decide(obs, goal, memory, last):
    """Stage machine: cue -> wait for arc -> touch each cued cube -> press button -> lift and wait.
    Progress is verified from the measured TCP in the frames of the previous action."""
    mem = {
        "stage": memory.get("stage") or "cue",
        "cue_colors": memory.get("cue_colors") or [],
        "cue_step": memory.get("cue_step") if memory.get("cue_step") is not None else -1,
        "targets": memory.get("targets") or [],
        "target_colors": memory.get("target_colors") or [],
        "touched": memory.get("touched") or [],
        "button": memory.get("button") or [],
        "yaw": memory.get("yaw") if memory.get("yaw") is not None else float(obs.current.tcp[5]),
        "press_tries": memory.get("press_tries") or 0,
        "touch_round": memory.get("touch_round") or 0,
        "log": memory.get("log") or "",
    }
    if mem["stage"] in ("cue", "wait_arc"):
        arc, fr = _scan_frames(obs.frames, mem)
        if arc is None:
            mem["stage"] = "wait_arc"
            if isinstance(last, str) and last.startswith("act_wait_for_arc"):
                mem["log"] += "noarc;"
            return {"memory": mem, "act": {"tool": "act_wait_for_arc", "args": {}}}
        _plan_targets(mem, arc, fr)
        mem["stage"] = "touch"
        mem["touched"] = []
        last = None

    if mem["stage"] == "touch":
        i = len(mem["touched"])
        if isinstance(last, str) and last.startswith("act_touch") and i < len(mem["targets"]):
            x, y = mem["targets"][i]
            minz, err = 9.0, 9.0
            for fr in obs.frames:
                t = fr.tcp
                d = math.hypot(float(t[0]) - x, float(t[1]) - y)
                if d < 0.02 and float(t[2]) < minz:
                    minz, err = float(t[2]), d
            col = mem["target_colors"][i]
            mem["log"] += "t:%s z%.3f e%.3f;" % (col, minz, err)
            retried = ("retry:%s;" % col) in mem["log"]
            if minz < 0.060 or retried:
                mem["touched"] = mem["touched"] + [col]
            else:
                mem["log"] += "retry:%s;" % col
            i = len(mem["touched"])
        if i < len(mem["targets"]):
            x, y = mem["targets"][i]
            lower = mem["touch_round"] > 0 or (("retry:%s;" % mem["target_colors"][i]) in mem["log"])
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": x, "y": y, "yaw": mem["yaw"],
                "touch_z": RETOUCH_Z if lower else TOUCH_Z}}}
        mem["stage"] = "press"
        last = None

    if mem["stage"] == "press":
        if isinstance(last, str) and last.startswith("act_press"):
            zs = [float(fr.tcp[2]) for fr in obs.frames]
            mem["log"] += "press%d minz%.3f maxz%.3f;" % (mem["press_tries"], min(zs), max(zs))
            # the episode is still running, so the press did not produce success
            if mem["press_tries"] == 2 and mem["touch_round"] == 0 and float(obs.current.step) < 250:
                # two presses failed: assume a touch did not register; redo all touches lower, then press again
                mem["touch_round"] = 1
                mem["touched"] = []
                mem["stage"] = "touch"
                mem["log"] += "retouch;"
                # reversed planned order starts next to the button, where the arm is now
                mem["targets"] = list(reversed(mem["targets"]))
                mem["target_colors"] = list(reversed(mem["target_colors"]))
                x, y = mem["targets"][0]
                return {"memory": mem, "act": {"tool": "act_touch", "args": {
                    "x": x, "y": y, "yaw": mem["yaw"], "touch_z": RETOUCH_Z}}}
        mem["press_tries"] = mem["press_tries"] + 1
        b = mem["button"]
        offs = [0.0, -0.006, 0.0, 0.006, -0.006]
        dx = offs[min(mem["press_tries"] - 1, len(offs) - 1)]
        return {"memory": mem, "act": {"tool": "act_press", "args": {
            "x": b[0] + dx, "y": b[1], "yaw": mem["yaw"],
            "lift_z": LIFT_Z}}}

    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


def act_wait_for_arc(robot, max_steps=60):
    """Hold still (zero motion, gripper open) until the 9-cube arc is visible in the front camera or max_steps
    pass. Returns the frame-kind sequence (c=cue, e=empty, a=arc) and steps waited."""
    fr = robot.now()
    n = 0
    log = []
    while True:
        kind, cubes = _classify_frame(fr)
        log.append(kind[0])
        if kind == "arc":
            return {"arc_step": int(fr.step), "n_cubes": len(cubes), "waited": n, "seq": "".join(log)}
        if n >= max_steps:
            return {"arc_step": -1, "waited": n, "seq": "".join(log)}
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        fr = robot.now()


def act_touch(robot, x, y, yaw=0.0, touch_z=TOUCH_Z):
    """Hover-touch one cube: rise to transit height, move above (x, y), descend open gripper to TOUCH_Z,
    pause briefly, rise back. Returns measured minimum z, xy error at the bottom and steps used."""
    steps = 0
    tcp = np.asarray(robot.now().tcp, dtype=np.float64)
    # 1. transit (xy motion gated until clear of the cube tops)
    n, tcp, ok1, via = _transit(robot, x, y, 1.0, yaw, 0.012)
    steps += n
    # 2. descend
    n, tcp, ok2 = _move(robot, [x, y, touch_z], 1.0, yaw, 0.006, 0.004, 30)
    steps += n
    minz, err = float(tcp[2]), float(math.hypot(tcp[0] - x, tcp[1] - y))
    # 3. brief dwell at the bottom
    for _ in range(2):
        robot.step(*_cmd(tcp, [x, y, touch_z], yaw), 1.0)
        steps += 1
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        if tcp[2] < minz:
            minz, err = float(tcp[2]), float(math.hypot(tcp[0] - x, tcp[1] - y))
    # 4. rise
    n, tcp, ok3 = _move(robot, [x, y, TRANSIT_Z - 0.02], 1.0, yaw, 1.0, 0.02, 20)
    steps += n
    return {"via": via, "min_z": minz, "xy_err": err,
            "steps": steps, "reached_above": ok1, "reached_bottom": ok2, "end_tcp": [round(float(v), 4) for v in tcp[:3]]}


def act_press(robot, x, y, yaw=0.0, lift_z=LIFT_Z):
    """Press the button cap with a closed gripper: transit above (x, y) closing the gripper, descend until the
    cap stops the TCP, hold, then lift to LIFT_Z and wait there (native success is expected during the lift)."""
    steps = 0
    n, tcp, ok1, via = _transit(robot, x, y, -1.0, yaw, 0.008)
    steps += n
    # make sure the fingers are closed before descending
    k = 0
    while float(np.sum(robot.now().fingers)) > 0.004 and k < 10:
        robot.step(*_cmd(robot.now().tcp, [x, y, TRANSIT_Z], yaw), -1.0)
        k += 1
    steps += k
    # descend until stalled on the cap
    minz = 9.0
    prev = float(robot.now().tcp[2])
    stall = 0
    k = 0
    while k < 30:
        tcp = np.asarray(robot.now().tcp, dtype=np.float64)
        robot.step(*_cmd(tcp, [x, y, PRESS_Z], yaw, vmax=0.8), -1.0)
        k += 1
        z = float(robot.now().tcp[2])
        minz = min(minz, z)
        if z < 0.062 and abs(z - prev) < 0.0015:
            stall += 1
        prev = z
        if stall >= 5:
            break
    steps += k
    bottom_xy = [round(float(v), 4) for v in robot.now().tcp[:2]]
    # lift and wait
    n, tcp, ok3 = _move(robot, [x, y, lift_z], -1.0, yaw, 1.0, 0.01, 30)
    steps += n
    for _ in range(15):
        robot.step(*_cmd(robot.now().tcp, [x, y, lift_z], yaw), -1.0)
        steps += 1
    return {"pressed": True, "min_z": minz, "stall": stall, "bottom_xy": bottom_xy, "steps": steps,
            "end_z": float(robot.now().tcp[2])}


def act_hold(robot, steps=20):
    """Hold the current pose with the gripper closed for a number of steps."""
    tcp = robot.now().tcp
    goal = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
    for _ in range(steps):
        robot.step(*_cmd(robot.now().tcp, goal, float(tcp[5])), -1.0)
    return {"held": steps}
