"""SeqOfColors5-VLA policy: memorise the cue colours, touch those cubes, press the centre button.

Perception uses the fixed front RGB-D camera (deprojected to world coordinates with the
per-frame intrinsics/extrinsics). Cube top faces render with exact palette colours, so each
object is classified by the nearest palette colour of its top-face pixels.
"""

PALETTE = {
    "magenta": (231, 5, 231),
    "blue": (5, 5, 231),
    "red": (231, 5, 5),
    "yellow": (231, 231, 5),
    "teal": (5, 169, 169),
    "darkred": (169, 5, 5),
    "cyan": (5, 231, 231),
    "green": (5, 231, 5),
    "orange": (231, 169, 5),
    "button": (211, 136, 136),
}
CUE_XY = (0.01, 0.0)        # where cue cubes appear (measured in all 50 demos)
CUE_HALF = 0.03
TOUCH_Z = 0.05              # TCP height over a cube (cube top is 0.04; demos 0.045-0.062)
TRAVEL_Z = 0.15             # transit height between cubes
TOUCH_OFFSET = (-0.005, 0.0)  # demos: TCP ~0.005-0.01 m on -x of the perceived top centre
PRESS_OFFSET = (0.015, 0.0)   # demos: TCP ~0.015-0.02 m on +x of the perceived cap centre
PRESS_Z = 0.042             # commanded TCP height when pressing (contact stops it near 0.053)
LIFT_Z = 0.20               # demos reached success while lifting to ~0.17-0.18
MAX_PRESSES = 4
REDO_TOUCHES = False  # R012: redoing touches after a wrong-order pass did not recover success


# ---------------------------------------------------------------- perception helpers
def _world(frame):
    """Per-pixel world XYZ (H, W, 3) from the front depth; NaN where depth is unknown."""
    d = np.asarray(frame.front_depth).astype(float) / 1000.0
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1)
    W = (P - E[:, 3]) @ E[:, :3]
    W[d <= 0] = np.nan
    return W


def _object_mask(frame, W):
    im = np.asarray(frame.front_rgb).astype(int)
    sat = im.max(2) - im.min(2)
    z = W[..., 2]
    with np.errstate(invalid="ignore"):
        m = (sat > 50) & (z > 0.004) & (z < 0.09)
    return im, m


def _classify(rgb):
    best, bd = None, 1e9
    for name, c in PALETTE.items():
        d = float(np.abs(np.asarray(rgb, float) - np.asarray(c, float)).max())
        if d < bd:
            best, bd = name, d
    return best, bd


def _components(mask):
    H, Wd = mask.shape
    lab = -np.ones((H, Wd), int)
    comps = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if lab[y0, x0] >= 0:
            continue
        k = len(comps)
        st = [(y0, x0)]
        lab[y0, x0] = k
        pix = []
        while st:
            y, x = st.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < Wd and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = k
                    st.append((yy, xx))
        comps.append(np.array(pix))
    return comps


def _objects(frame):
    """Coloured objects above the table: list of dicts(color, dist, xyz top centre, npix)."""
    W = _world(frame)
    im, m = _object_mask(frame, W)
    out = []
    for c in _components(m):
        if len(c) < 8:
            continue
        P = W[c[:, 0], c[:, 1]]
        zt = float(np.nanmax(P[:, 2]))
        top = P[:, 2] > zt - 0.004
        rgb = np.median(im[c[top, 0], c[top, 1]], 0)
        name, dist = _classify(rgb)
        xyz = np.nanmedian(P[top], 0)
        out.append({"color": name, "dist": dist, "xyz": [float(v) for v in xyz],
                    "npix": int(len(c)), "ztop": zt})
    return out


def _cue_color(frame):
    """Colour of the cue cube at the cue location in this frame, or None."""
    W = _world(frame)
    im, m = _object_mask(frame, W)
    with np.errstate(invalid="ignore"):
        reg = (np.abs(W[..., 0] - CUE_XY[0]) < CUE_HALF) & (np.abs(W[..., 1] - CUE_XY[1]) < CUE_HALF)
        top = W[..., 2] > 0.032
    sel = m & reg & top
    if int(sel.sum()) < 10:
        return None
    rgb = np.median(im[sel], 0)
    name, dist = _classify(rgb)
    if name == "button" or dist > 45:
        return None
    return name


def _arc_count(frame):
    """Number of coloured pixels away from the cue spot and button (arc cubes visible)."""
    W = _world(frame)
    im, m = _object_mask(frame, W)
    with np.errstate(invalid="ignore"):
        cue = (np.abs(W[..., 0] - CUE_XY[0]) < CUE_HALF) & (np.abs(W[..., 1] - CUE_XY[1]) < CUE_HALF)
        btn = (np.abs(W[..., 0] + 0.125) < 0.06) & (np.abs(W[..., 1]) < 0.06)
    return int((m & ~cue & ~btn).sum())


def _button_pressed_pixels(frame):
    """Pink cap pixels lower than 0.052 m (the cap renders at ~0.044 while pressed)."""
    W = _world(frame)
    im = np.asarray(frame.front_rgb).astype(int)
    pink = np.abs(im - np.array([211, 136, 136])).max(2) < 30
    with np.errstate(invalid="ignore"):
        near = (np.abs(W[..., 0] + 0.125) < 0.07) & (np.abs(W[..., 1]) < 0.07)
        low = (W[..., 2] < 0.052) & (W[..., 2] > 0.03)
        high = (W[..., 2] >= 0.052) & (W[..., 2] < 0.07)
    return int((pink & near & low).sum()), int((pink & near & high).sum())


# ---------------------------------------------------------------- control helpers
def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def _rot_cmd(tcp, rpy_target, gain=4.0, limit=0.3):
    """Rotation command holding the target orientation (offset = -0.1*r, world axes)."""
    Rt = _rpy2R(*rpy_target)
    Rm = _rpy2R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _rotvec(Rt @ Rm.T)
    r = -gain * w
    n = float(np.linalg.norm(r))
    if n > limit:
        r = r * (limit / n)
    return r, float(np.linalg.norm(w))


def _move(robot, target, rpy, grip, tol=0.004, max_steps=60, gain=1.0, vmax=1.0, settle=0.003):
    """Closed-loop move of the TCP to `target` (world xyz). Returns measured summary."""
    target = np.asarray(target, float)
    prev = np.asarray(robot.now().tcp, float)
    n = 0
    zmin = float(prev[2])
    for n in range(1, max_steps + 1):
        tcp = np.asarray(robot.now().tcp, float)
        err = target - tcp[:3]
        u = gain * err / 0.1
        nu = float(np.linalg.norm(u))
        if nu > vmax:
            u = u * (vmax / nu)
        r, _ = _rot_cmd(tcp, rpy)
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        now = np.asarray(robot.now().tcp, float)
        zmin = min(zmin, float(now[2]))
        speed = float(np.linalg.norm(now[:3] - prev[:3]))
        prev = now
        if float(np.linalg.norm(target - now[:3])) < tol and speed < settle:
            break
    tcp = np.asarray(robot.now().tcp, float)
    return {"steps": n, "err": float(np.linalg.norm(target - tcp[:3])), "tcp": [round(float(v), 4) for v in tcp[:3]],
            "zmin": round(zmin, 4)}


def _press_down(robot, x, y, z_press, rpy, max_steps=25):
    """Descend with closed gripper until the TCP stops on the button (or reaches z_press)."""
    prev = np.asarray(robot.now().tcp, float)
    still = 0
    n = 0
    for n in range(1, max_steps + 1):
        tcp = np.asarray(robot.now().tcp, float)
        err = np.array([x, y, z_press]) - tcp[:3]
        u = err / 0.1
        nu = float(np.linalg.norm(u))
        if nu > 0.7:
            u = u * (0.7 / nu)
        r, _ = _rot_cmd(tcp, rpy)
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2]), -1.0)
        now = np.asarray(robot.now().tcp, float)
        dz = float(prev[2] - now[2])
        prev = now
        if now[2] < 0.062 and dz < 0.0015:
            still += 1
        else:
            still = 0
        if still >= 2 or now[2] <= z_press + 0.002:
            break
    tcp = np.asarray(robot.now().tcp, float)
    return {"steps": n, "tcp": [round(float(v), 4) for v in tcp[:3]]}


def _hold(robot, rpy, grip, steps):
    zs = []
    for _ in range(steps):
        tcp = np.asarray(robot.now().tcp, float)
        r, _ = _rot_cmd(tcp, rpy)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), grip)
        zs.append(float(robot.now().tcp[2]))
    return zs


# ---------------------------------------------------------------- decision
def _nearest_palette_targets(objs, cue):
    """Pick, for each cue colour, the detected cube of that colour (largest blob)."""
    found = {}
    for o in objs:
        if o["color"] == "button" or o["dist"] > 45:
            continue
        c = o["color"]
        if c not in found or o["npix"] > found[c]["npix"]:
            found[c] = o
    return found


def perceive_decide(obs, goal, memory, last):
    """Parse cue frames, locate cubes/button once the arc appears, then touch each cue cube and press."""
    mem = dict(memory)
    defaults = {"stage": "cue", "cue_colors": [], "cue_log": "", "targets": [], "target_colors": [],
                "touched": [], "button": None, "home_rpy": None, "press_count": 0, "redo_count": 0, "log": "",
                "all_cubes": ""}
    for k, v in defaults.items():
        if mem.get(k) is None:
            mem[k] = v
    cur = obs.current
    if mem["home_rpy"] is None:
        t = cur.tcp
        mem["home_rpy"] = [float(t[3]), float(t[4]), float(t[5])]

    if mem["stage"] == "cue":
        cue = list(mem["cue_colors"])
        log = mem["cue_log"]
        for f in obs.frames:
            if _arc_count(f) > 300:
                continue
            c = _cue_color(f)
            if c is not None and (not cue or cue[-1] != c) and c not in cue:
                cue.append(c)
                log += "%s@%s " % (c, f.step)
        mem["cue_colors"] = cue
        mem["cue_log"] = log
        if _arc_count(cur) > 300:
            objs = _objects(cur)
            found = _nearest_palette_targets(objs, cue)
            btn = [o for o in objs if o["color"] == "button"]
            mem["all_cubes"] = "; ".join("%s(%.3f,%.3f)" % (c, o["xyz"][0], o["xyz"][1]) for c, o in found.items())
            if btn:
                b = max(btn, key=lambda o: o["npix"])
                mem["button"] = [b["xyz"][0], b["xyz"][1], b["xyz"][2]]
            else:
                mem["button"] = [-0.125, 0.0, 0.058]
            tc, tp = [], []
            for c in cue:
                if c in found:
                    tc.append(c)
                    tp.append([found[c]["xyz"][0], found[c]["xyz"][1], found[c]["xyz"][2]])
            missing = [c for c in cue if c not in found]
            if len(cue) >= 1 and not missing:
                mem["target_colors"] = tc
                mem["targets"] = tp
                mem["stage"] = "touch"
                mem["log"] = "arc at step %s; cue=%s" % (cur.step, ",".join(cue))
            else:
                mem["log"] = "arc visible but missing %s (cue=%s)" % (",".join(missing), ",".join(cue))
                return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}
        else:
            return {"memory": mem, "act": {"tool": "act_wait_for_arc", "args": {"max_steps": 80}}}

    if mem["stage"] == "touch":
        remaining = [(c, p) for c, p in zip(mem["target_colors"], mem["targets"]) if c not in mem["touched"]]
        if remaining:
            # Cue order (all 50 demos touch in cue order; R001 nearest-first order failed).
            c, p = remaining[0]
            mem["touched"] = list(mem["touched"]) + [c]
            mem["log"] = "touching %s at (%.3f,%.3f)" % (c, p[0], p[1])
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": p[0] + TOUCH_OFFSET[0], "y": p[1] + TOUCH_OFFSET[1], "rpy": mem["home_rpy"]}}}
        mem["stage"] = "press"

    if mem["stage"] == "press":
        b = mem["button"]
        k = int(mem["press_count"])
        if REDO_TOUCHES and k == 1 and int(mem["redo_count"]) == 0:
            # A press without the episode ending: assume a touch was not accepted and
            # repeat the whole touch sequence in cue order once, then press again.
            mem["redo_count"] = 1
            mem["touched"] = []
            mem["stage"] = "touch"
            c, p = mem["target_colors"][0], mem["targets"][0]
            mem["touched"] = [c]
            mem["log"] = "redo touches; touching %s" % c
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": p[0] + TOUCH_OFFSET[0], "y": p[1] + TOUCH_OFFSET[1], "rpy": mem["home_rpy"]}}}
        if k >= MAX_PRESSES:
            mem["stage"] = "done"
        else:
            mem["press_count"] = k + 1
            off = PRESS_OFFSET if (k < 2 or k % 2 == 0) else (0.0, 0.0)
            mem["log"] = "press %d at (%.3f,%.3f)" % (k + 1, b[0] + off[0], b[1] + off[1])
            return {"memory": mem, "act": {"tool": "act_press", "args": {
                "x": b[0] + off[0], "y": b[1] + off[1], "rpy": mem["home_rpy"]}}}

    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 10}}}


# ---------------------------------------------------------------- actions
def act_wait_for_arc(robot, max_steps=80):
    """Hold still (gripper open) until the arc of candidate cubes is visible; returns cue colours seen."""
    seen = []
    n = 0
    for n in range(1, max_steps + 1):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        f = robot.now()
        if _arc_count(f) > 300:
            break
        c = _cue_color(f)
        if c is not None and (not seen or seen[-1] != c):
            seen.append(c)
    return {"steps": n, "cue_seen": seen, "arc_pixels": _arc_count(robot.now()), "step": robot.now().step}


def act_wait(robot, steps=1):
    """Hold position for `steps` native steps (gripper unchanged-open)."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"steps": int(steps), "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_touch(robot, x, y, rpy, z_touch=TOUCH_Z, travel_z=TRAVEL_Z, hold=2):
    """Move above (x, y) at travel height, descend to z_touch with the gripper open, hold, rise again."""
    out = {}
    tcp = np.asarray(robot.now().tcp, float)
    chk = robot.dry_run(x, y, z_touch, rpy[0], rpy[1], rpy[2], mode="absolute")
    out["dry_run_ok"] = bool(chk.get("within_joint_limits", True))
    out["dry_margin"] = chk.get("min_joint_margin_rad")
    if tcp[2] < travel_z - 0.02:
        out["rise"] = _move(robot, [tcp[0], tcp[1], travel_z], rpy, 1.0, tol=0.02, settle=1.0, max_steps=20)
    out["transit"] = _move(robot, [x, y, travel_z], rpy, 1.0, tol=0.008, settle=0.006, max_steps=60)
    out["descend"] = _move(robot, [x, y, z_touch], rpy, 1.0, tol=0.004, settle=0.003, max_steps=30)
    zs = _hold(robot, rpy, 1.0, hold)
    bottom = np.asarray(robot.now().tcp, float)
    out["bottom_xy_err"] = [round(float(bottom[0] - x), 4), round(float(bottom[1] - y), 4)]
    out["hold_z"] = [round(v, 4) for v in zs]
    out["ascend"] = _move(robot, [x, y, travel_z], rpy, 1.0, tol=0.02, settle=1.0, max_steps=20)
    out["step"] = robot.now().step
    return out


def act_press(robot, x, y, rpy, z_press=PRESS_Z, lift_z=LIFT_Z, hold=4, wait=8):
    """Close the gripper, press the button at (x, y), hold, lift to lift_z and wait for the task to finish."""
    out = {}
    tcp = np.asarray(robot.now().tcp, float)
    out["approach"] = _move(robot, [x, y, 0.12], rpy, -1.0, tol=0.006, settle=0.005, max_steps=60)
    fing = robot.now().fingers
    if float(fing[0]) > 0.005:
        _hold(robot, rpy, -1.0, 4)
    out["fingers"] = round(float(robot.now().fingers[0]), 4)
    out["descend"] = _press_down(robot, x, y, z_press, rpy)
    low, high = _button_pressed_pixels(robot.now())
    out["cap_px_low_high"] = [low, high]
    zs = _hold(robot, rpy, -1.0, hold)
    out["hold_z"] = [round(v, 4) for v in zs]
    out["lift"] = _move(robot, [x, y, lift_z], rpy, -1.0, tol=0.005, settle=0.003, max_steps=30)
    zs = _hold(robot, rpy, -1.0, wait)
    out["wait_z"] = [round(v, 4) for v in zs[::3]]
    out["step"] = robot.now().step
    return out
