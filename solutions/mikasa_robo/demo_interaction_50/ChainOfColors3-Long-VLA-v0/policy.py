"""ChainOfColors3-Long policy: watch the cue cubes, wait for the arc of cubes,
touch the cued cubes in order with an open gripper, then press the button with a
closed gripper.  All perception uses the fixed front RGB-D camera."""

ROI_ROWS = (85, 200)       # front-image rows covering the table area with cubes/button
ROI_COLS = (10, 246)
CUE_ROWS = (132, 175)      # front-image crop around the cue spot (world ~ (0.005, 0))
CUE_COLS = (105, 151)
CUE_XY = (0.005, 0.0)
SAT_MIN = 100              # cube pixels are strongly saturated colours
TOUCH_OFF = (-0.0066, 0.0025)   # demo: TCP touch point minus cube top-face centroid
BUTTON_OFF = (0.0195, 0.0036)   # demo: TCP press point minus button-cap centroid
TOUCH_Z = 0.054            # demo touches 0.057-0.061 accepted; R020: rest at 0.0655 NOT accepted -> extra margin
HOVER_Z = 0.10
PRESS_Z = 0.053            # demo: TCP bottoms out at 0.053 on the button
LIFT_Z = 0.21              # R001-R007: success registered exactly when TCP z crossed ~0.170 after the press


def _world_points(frame, r0, r1, c0, c1):
    K = frame.front_K
    E = frame.front_extrinsic
    d = frame.front_depth[r0:r1, c0:c1].astype(np.float64) / 1000.0
    vv, uu = np.mgrid[r0:r1, c0:c1]
    x = (uu - K[0, 2]) * d / K[0, 0]
    y = (vv - K[1, 2]) * d / K[1, 1]
    pc = np.stack([x, y, d], -1)
    R = E[:, :3].astype(np.float64)
    t = E[:, 3].astype(np.float64)
    pw = (pc - t) @ R
    return pw, d > 0


def _cube_pixels(frame, r0, r1, c0, c1):
    pw, ok = _world_points(frame, r0, r1, c0, c1)
    rgb = frame.front_rgb[r0:r1, c0:c1].astype(np.int32)
    sat = rgb.max(2) - rgb.min(2)
    z = pw[..., 2]
    m = ok & (z > 0.008) & (z < 0.05) & (sat > SAT_MIN) & (pw[..., 0] > -0.45)
    return pw[m], rgb[m]


def _clusters(pts, rgb, radius=0.035, min_n=15):
    out = []
    if len(pts) == 0:
        return out
    xy = pts[:, :2]
    left = np.ones(len(pts), dtype=bool)
    guard = 0
    while left.any() and guard < 40:
        guard += 1
        i = int(np.flatnonzero(left)[0])
        c = xy[i]
        sel = left & (np.linalg.norm(xy - c, axis=1) < radius)
        for _ in range(3):
            c = xy[sel].mean(0)
            sel = left & (np.linalg.norm(xy - c, axis=1) < radius)
        if not sel.any():
            left[i] = False
            continue
        left = left & ~sel
        if sel.sum() < min_n:
            continue
        z = pts[sel, 2]
        top = sel & (pts[:, 2] > z.max() - 0.006)
        out.append({
            "n": int(sel.sum()),
            "xy": [float(v) for v in xy[top].mean(0)],
            "zmax": float(z.max()),
            "rgb": [float(v) for v in rgb[top].mean(0)],
        })
    return out


def _detect_arc(frame):
    pts, rgb = _cube_pixels(frame, ROI_ROWS[0], ROI_ROWS[1], ROI_COLS[0], ROI_COLS[1])
    cubes = [c for c in _clusters(pts, rgb) if 0.03 < c["zmax"] < 0.05]
    return cubes


def _detect_button(frame):
    pw, ok = _world_points(frame, ROI_ROWS[0], ROI_ROWS[1], ROI_COLS[0], ROI_COLS[1])
    z = pw[..., 2]
    m = ok & (z > 0.052) & (z < 0.07) & (pw[..., 0] > -0.3) & (pw[..., 0] < 0.05) & (np.abs(pw[..., 1]) < 0.15)
    if m.sum() < 20:
        return None
    xy = pw[m][:, :2].mean(0)
    return [float(xy[0]), float(xy[1]), int(m.sum())]


def _cue_color(frame):
    pts, rgb = _cube_pixels(frame, CUE_ROWS[0], CUE_ROWS[1], CUE_COLS[0], CUE_COLS[1])
    if len(pts) < 20:
        return None
    near = np.linalg.norm(pts[:, :2] - np.array(CUE_XY), axis=1) < 0.04
    if near.sum() < 20:
        return None
    p = pts[near]
    top = p[:, 2] > p[:, 2].max() - 0.006
    if top.sum() < 8:
        return None
    return [float(v) for v in rgb[near][top].mean(0)]


def _arc_present(frame):
    """Cheap test: many saturated, elevated pixels away from the cue spot."""
    pts, rgb = _cube_pixels(frame, ROI_ROWS[0], ROI_ROWS[1], ROI_COLS[0], ROI_COLS[1])
    if len(pts) == 0:
        return False
    far = np.linalg.norm(pts[:, :2] - np.array(CUE_XY), axis=1) > 0.06
    return int(far.sum()) > 250


def _cue_sequence(frames, prior):
    """Collapse per-frame cue colours into runs; returns list of [r,g,b,start,len]."""
    runs = [list(r) for r in prior]
    for f in frames:
        c = _cue_color(f)
        if c is None:
            continue
        st = int(f.step)
        if runs:
            last = runs[-1]
            same = np.linalg.norm(np.array(last[:3]) - np.array(c)) < 25
            contiguous = st - (last[3] + last[4]) <= 2
            if same and contiguous:
                last[4] = st - last[3] + 1
                continue
        runs.append([c[0], c[1], c[2], st, 1])
    return runs


def _match(cues, cubes):
    """Greedy distinct assignment of cue colours to arc cubes by RGB distance."""
    A = np.array([c["rgb"] for c in cubes])
    used = []
    out = []
    for cue in cues:
        d = np.linalg.norm(A - np.array(cue[:3]), axis=1)
        for u in used:
            d[u] = 1e9
        order = np.argsort(d)
        k = int(order[0])
        used.append(k)
        second = float(d[order[1]]) if len(order) > 1 else 999.0
        out.append({"idx": k, "dist": float(d[k]), "second": second})
    return out


def perceive_decide(obs, goal, memory, last):
    """Track cue colours, detect arc cubes/button, then sequence touch and press acts."""
    mem = {
        "stage": memory.get("stage") or "wait",
        "cue_runs": memory.get("cue_runs") or [],
        "targets": memory.get("targets") or [],
        "button": memory.get("button") or [],
        "touched": int(memory.get("touched") or 0),
        "presses": int(memory.get("presses") or 0),
        "notes": memory.get("notes") or "",
    }
    frames = list(obs.frames)
    cur = obs.current
    if mem["stage"] == "wait":
        mem["cue_runs"] = _cue_sequence(frames, mem["cue_runs"])
        cubes = _detect_arc(cur)
        if len(cubes) < 6 or len(mem["cue_runs"]) == 0:
            mem["notes"] = "waiting step %d cubes=%d cues=%d" % (int(cur.step), len(cubes), len(mem["cue_runs"]))
            return {"memory": mem, "act": {"tool": "act_wait_arc", "args": {"max_steps": 790}}}
        cues = [r for r in mem["cue_runs"] if r[4] >= 3]
        m = _match(cues, cubes)
        targets = []
        for k in m:
            c = cubes[k["idx"]]
            targets.append([c["xy"][0] + TOUCH_OFF[0], c["xy"][1] + TOUCH_OFF[1]])
        btn = _detect_button(cur)
        if btn is None:
            bxy = [-0.125 + BUTTON_OFF[0], 0.0 + BUTTON_OFF[1]]
        else:
            bxy = [btn[0] + BUTTON_OFF[0], btn[1] + BUTTON_OFF[1]]
        mem["targets"] = targets
        mem["button"] = bxy
        mem["stage"] = "touch"
        mem["notes"] = "arc at step %d: %d cubes, cues %s, match %s, button %s" % (
            int(cur.step), len(cubes),
            [[int(v) for v in r[:3]] + [int(r[3]), int(r[4])] for r in cues],
            [[k["idx"], int(k["dist"]), int(k["second"])] for k in m], btn)
    if mem["stage"] == "touch":
        i = mem["touched"]
        if i < len(mem["targets"]):
            x, y = mem["targets"][i]
            mem["touched"] = i + 1
            if mem["touched"] >= len(mem["targets"]):
                mem["stage"] = "press"
            return {"memory": mem, "act": {"tool": "act_touch", "args": {"x": x, "y": y}}}
        mem["stage"] = "press"
    if mem["stage"] == "press":
        mem["presses"] = mem["presses"] + 1
        if mem["presses"] > 1:
            mem["stage"] = "repress"
        bx, by = mem["button"]
        return {"memory": mem, "act": {"tool": "act_press", "args": {"x": bx, "y": by}}}
    if mem["stage"] == "repress" and mem["presses"] < 4:
        mem["presses"] = mem["presses"] + 1
        bx, by = mem["button"]
        return {"memory": mem, "act": {"tool": "act_press", "args": {"x": bx, "y": by, "hold": 8}}}
    mem["stage"] = "done"
    return {"memory": mem, "act": {"tool": "act_idle", "args": {"steps": 40}}}


# ---------------------------------------------------------------- motion helpers

def _tcp(robot):
    return np.array(robot.now().tcp[:3], dtype=np.float64)


def _cmd(err, gain, vmax):
    o = gain * np.asarray(err, dtype=np.float64)
    n = np.abs(o).max()
    if n > vmax:
        o = o * (vmax / n)
    return o / 0.1


def _move(robot, goal, grip, tol, max_steps, gain=1.1, vmax=0.1, settle=False, log=None):
    """Closed-loop translation toward goal; returns (steps, final error norm)."""
    goal = np.array(goal, dtype=np.float64)
    prev = _tcp(robot)
    for s in range(max_steps):
        p = _tcp(robot)
        e = goal - p
        vel = np.linalg.norm(p - prev)
        if np.linalg.norm(e) < tol and (not settle or vel < 0.002):
            return s, float(np.linalg.norm(e))
        d = _cmd(e, gain, vmax)
        prev = p
        robot.step(float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, 0.0, grip)
    return max_steps, float(np.linalg.norm(goal - _tcp(robot)))


def _reach(robot, x, y, z):
    """Kinematic check of the target pose at the current orientation (no motion)."""
    rpy = robot.now().tcp[3:]
    try:
        r = robot.dry_run(float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2]))
    except Exception as ex:
        return {"error": str(ex)[:80]}
    if not isinstance(r, dict):
        return {"raw": str(r)[:80]}
    return {"reachable": r.get("reachable"), "margin": r.get("joint_margin")}


def _press_down(robot, x, y, z_goal, max_steps):
    """Descend with closed gripper until the TCP stalls on the button (low and not moving)."""
    goal = np.array([x, y, z_goal])
    prev = _tcp(robot)
    still = 0
    for s in range(max_steps):
        p = _tcp(robot)
        if p[2] < 0.062 and abs(p[2] - prev[2]) < 0.0015:
            still += 1
        else:
            still = 0
        if still >= 2 or p[2] < z_goal + 0.002:
            return s
        prev = p
        d = _cmd(goal - p, 1.1, 0.1)
        robot.step(float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, 0.0, -1.0)
    return max_steps


def act_wait_arc(robot, max_steps=790):
    """Hold still (zero motion, gripper open) until the arc of cubes appears; returns steps waited."""
    n = 0
    for n in range(max_steps):
        if _arc_present(robot.now()):
            return {"arc_seen": True, "steps": n, "frame_step": int(robot.now().step)}
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"arc_seen": False, "steps": max_steps, "frame_step": int(robot.now().step)}


def act_touch(robot, x, y, z_touch=TOUCH_Z, hover=HOVER_Z, hold=2):
    """Lift to hover, move above (x,y), descend to z_touch with open gripper, hold, rise to hover."""
    s0 = int(robot.now().step)
    reach = _reach(robot, x, y, z_touch)
    p = _tcp(robot)
    n_lift = 0
    if p[2] < hover - 0.02:
        n_lift, _ = _move(robot, [p[0], p[1], hover], 1.0, 0.02, 30)
    n_tr, e_tr = _move(robot, [x, y, hover], 1.0, 0.012, 60)
    n_dn, e_dn = _move(robot, [x, y, z_touch], 1.0, 0.004, 30, settle=True)
    p = _tcp(robot)
    mins = p.copy()
    for _ in range(hold):
        e = np.array([x, y, z_touch]) - _tcp(robot)
        d = _cmd(e, 1.0, 0.1)
        robot.step(float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, 0.0, 1.0)
        q = _tcp(robot)
        if q[2] < mins[2]:
            mins = q
    n_up, _ = _move(robot, [x, y, hover], 1.0, 0.02, 30)
    return {"target": [x, y], "touch_tcp": [float(v) for v in mins], "xy_err": float(np.linalg.norm(mins[:2] - np.array([x, y]))),
            "steps": [n_lift, n_tr, n_dn, hold, n_up], "start": s0, "end": int(robot.now().step),
            "rpy": [float(v) for v in robot.now().tcp[3:]], "reach": reach}


def act_press(robot, x, y, z_press=PRESS_Z, hold=6, lift=LIFT_Z, wait=25):
    """Close gripper, move above button (x,y), push down to z_press, hold, lift to `lift`, wait."""
    s0 = int(robot.now().step)
    reach = _reach(robot, x, y, z_press)
    p = _tcp(robot)
    n_lift = 0
    if p[2] < HOVER_Z - 0.02:
        n_lift, _ = _move(robot, [p[0], p[1], HOVER_Z], -1.0, 0.02, 30)
    n_tr, _ = _move(robot, [x, y, HOVER_Z], -1.0, 0.012, 60)
    n_dn = _press_down(robot, x, y, z_press - 0.008, 30)
    zs = []
    for _ in range(hold):
        e = np.array([x, y, z_press - 0.008]) - _tcp(robot)
        d = _cmd(e, 1.0, 0.1)
        robot.step(float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, 0.0, -1.0)
        zs.append(float(_tcp(robot)[2]))
    n_up, _ = _move(robot, [x, y, lift], -1.0, 0.01, 40)
    for _ in range(wait):
        e = np.array([x, y, lift]) - _tcp(robot)
        d = _cmd(e, 1.0, 0.1)
        robot.step(float(d[0]), float(d[1]), float(d[2]), 0.0, 0.0, 0.0, -1.0)
    return {"target": [x, y], "hold_z": zs, "fingers": [float(v) for v in robot.now().fingers],
            "steps": [n_lift, n_tr, n_dn, hold, n_up, wait], "start": s0, "end": int(robot.now().step),
            "reach": reach}


def act_idle(robot, steps=40):
    """Hold position for a number of steps."""
    for _ in range(steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"end": int(robot.now().step)}
