"""SeqOfColors5-Long-VLA policy: memorize 5 cue cube colours, touch those cubes, press the centre button."""

FAMILIES = {"R": (1.0, 0.0, 0.0), "O": (1.0, 0.73, 0.0), "Y": (1.0, 1.0, 0.0),
            "G": (0.0, 1.0, 0.0), "C": (0.0, 1.0, 1.0), "B": (0.0, 0.0, 1.0),
            "M": (1.0, 0.0, 1.0)}
SINGLE_NAME = {"O": "orange", "Y": "yellow", "G": "green", "B": "blue", "M": "magenta"}
PAIR_NAME = {"R": ("red", "darkred"), "C": ("cyan", "teal")}
BRIGHT_SPLIT = 200.0          # top-face max channel: 231 bright, 169 dark (demo evidence)
CUE_UV = (128.0, 152.0)       # cue cube centroid in the front image (all 50 demos)
CUBE_XY_OFFSET = (-0.010, 0.0)    # expert touch TCP minus perceived top-face centre (demos)
BUTTON_XY_OFFSET = (0.0, 0.0)    # press the pink cap centre (cap spans x -0.151..-0.097 => radius ~2.7 cm)
BUTTON_DEFAULT = (-0.127, 0.0)    # perceived cap centre, 50 demos: mean (-0.127, 0.000), std 0.003
TRAVEL_Z = 0.10
TOUCH_Z = 0.050


# ---------------------------------------------------------------- perception
def _pure_mask(img):
    im = img.astype(np.int32)
    return (im.min(axis=2) < 30) & (im.max(axis=2) > 90)


def _components(mask, min_size=8):
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while pts:
        s = pts.pop()
        stack = [s]
        comp = [s]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.remove(q)
                        stack.append(q)
                        comp.append(q)
        if len(comp) >= min_size:
            comps.append(np.array(comp))
    return comps


def _family_labels(img):
    im = img.astype(np.float64)
    mx = np.maximum(im.max(axis=2), 1.0)
    nrm = im / mx[..., None]
    keys = list(FAMILIES)
    fv = np.array([FAMILIES[k] for k in keys])
    d2 = ((nrm[:, :, None, :] - fv[None, None, :, :]) ** 2).sum(axis=-1)
    return keys, np.argmin(d2, axis=-1)


def _deproject(frame, ys, xs):
    depth = frame.front_depth
    K = frame.front_K
    E = frame.front_extrinsic
    d = depth[ys, xs].astype(np.float64) / 1000.0
    ok = d > 0
    u = xs[ok] + 0.5
    v = ys[ok] + 0.5
    z = d[ok]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (P - t) @ R


def _cue_color(frame):
    """Colour name of the single cue cube shown in front of the button, or None."""
    img = frame.front_rgb
    im = img.astype(np.int32)
    m = (im.min(axis=2) < 25) & (im.max(axis=2) > 120)
    n = int(m.sum())
    if n < 15 or n > 300:
        return None
    ys, xs = np.nonzero(m)
    if abs(xs.mean() - CUE_UV[0]) > 25 or abs(ys.mean() - CUE_UV[1]) > 25:
        return None
    px = im[m].astype(np.float64)
    col = px.max(axis=0)
    bright = float(col.max())
    nrm = col / max(bright, 1.0)
    fam = min(FAMILIES, key=lambda k: float(((np.array(FAMILIES[k]) - nrm) ** 2).sum()))
    if fam in PAIR_NAME:
        return PAIR_NAME[fam][0] if bright >= BRIGHT_SPLIT else PAIR_NAME[fam][1]
    return SINGLE_NAME[fam]


def _ring_present(frame):
    m = _pure_mask(frame.front_rgb)
    n = int(m.sum())
    if n < 300:
        return False
    ys, xs = np.nonzero(m)
    return float(xs.std()) > 25.0


def _detect_cubes(frame):
    """All visible cubes: [{'name','fam','bright','xy','n'}] using colour families and depth."""
    img = frame.front_rgb
    mask = _pure_mask(img)
    keys, lab = _family_labels(img)
    im = img.astype(np.float64)
    found = []
    for fi, fam in enumerate(keys):
        comps = _components(mask & (lab == fi))
        expected = 2 if fam in PAIR_NAME else 1
        # split an over-large merged blob when fewer blobs than expected were found
        if len(comps) < expected and comps:
            big = max(comps, key=len)
            if len(big) > 140:
                comps.remove(big)
                comps.extend(_split2(big))
        for c in comps:
            px = im[c[:, 0], c[:, 1]]
            pmx = px.max(axis=1)
            W = _deproject(frame, c[:, 0], c[:, 1])
            if len(W) < 5:
                continue
            top = W[:, 2] > W[:, 2].max() - 0.006
            xy = np.median(W[top, :2], axis=0)
            found.append({"fam": fam, "n": int(len(c)), "bright": float(np.percentile(pmx, 90)),
                          "xy": [float(xy[0]), float(xy[1])], "ztop": float(np.median(W[top, 2]))})
    # name the cubes: pairs are distinguished by relative brightness
    for fam in keys:
        group = [c for c in found if c["fam"] == fam]
        if fam in PAIR_NAME:
            group.sort(key=lambda c: -c["bright"])
            if len(group) >= 2 and group[0]["bright"] - group[-1]["bright"] > 25:
                group[0]["name"] = PAIR_NAME[fam][0]
                for c in group[1:]:
                    c["name"] = PAIR_NAME[fam][1]
            else:
                for c in group:
                    c["name"] = PAIR_NAME[fam][0] if c["bright"] >= BRIGHT_SPLIT else PAIR_NAME[fam][1]
        else:
            for c in group:
                c["name"] = SINGLE_NAME[fam]
    return found


def _split2(c):
    pts = c.astype(np.float64)
    a = pts[np.argmin(pts[:, 1])]
    b = pts[np.argmax(pts[:, 1])]
    for _ in range(10):
        da = ((pts - a) ** 2).sum(axis=1)
        db = ((pts - b) ** 2).sum(axis=1)
        sel = da < db
        if sel.all() or (~sel).all():
            break
        a = pts[sel].mean(axis=0)
        b = pts[~sel].mean(axis=0)
    return [c[sel], c[~sel]]


def _detect_button(frame):
    im = frame.front_rgb.astype(np.float64)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r - g > 45) & (np.abs(g - b) < 20) & (g > 90)
    ys, xs = np.nonzero(m)
    if len(ys) < 20:
        return None
    W = _deproject(frame, ys, xs)
    if len(W) < 10:
        return None
    top = W[W[:, 2] > W[:, 2].max() - 0.003]
    # centre = midpoint of the cap-top extent (robust to the visible side of the cylinder)
    xy = [(np.percentile(top[:, 0], 2) + np.percentile(top[:, 0], 98)) / 2.0,
          (np.percentile(top[:, 1], 2) + np.percentile(top[:, 1], 98)) / 2.0]
    bx = float(xy[0]) + BUTTON_XY_OFFSET[0]
    by = float(xy[1]) + BUTTON_XY_OFFSET[1]
    if abs(bx - BUTTON_DEFAULT[0]) > 0.04 or abs(by - BUTTON_DEFAULT[1]) > 0.04:
        return None
    return [bx, by]


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Track the cue colours, locate the cubes when the response ring appears, then touch each cue
    cube (in cue order) and press the centre button."""
    mem = dict(memory)
    for k, v in (("stage", "cue"), ("cue_colors", []), ("cue_last", ""), ("cue_frames", 0),
                 ("ring_step", -1), ("targets", []), ("target_names", []), ("touched", 0),
                 ("button", list(BUTTON_DEFAULT)), ("home_rpy", None), ("presses", 0), ("retries", 0), ("log", "")):
        if mem.get(k) is None:
            mem[k] = v
    cur = obs.current
    if mem["home_rpy"] is None:
        mem["home_rpy"] = [float(a) for a in cur.tcp[3:6]]
    log = mem["log"]

    if mem["stage"] == "cue":
        colors = list(mem["cue_colors"])
        lastc = mem["cue_last"]
        nf = int(mem["cue_frames"])
        for f in obs.frames:
            c = _cue_color(f)
            if c is None:
                lastc = ""
                continue
            nf += 1
            if c != lastc and c not in colors:
                colors.append(c)
            lastc = c
        mem["cue_colors"], mem["cue_last"], mem["cue_frames"] = colors, lastc, nf
        if _ring_present(cur):
            cubes = _detect_cubes(cur)
            mem["ring_step"] = int(cur.step)
            targets, names = [], []
            for name in colors:
                cand = [c for c in cubes if c["name"] == name]
                if not cand:
                    log += " missing:" + name
                    continue
                c = max(cand, key=lambda c: c["n"])
                targets.append([c["xy"][0] + CUBE_XY_OFFSET[0], c["xy"][1] + CUBE_XY_OFFSET[1]])
                names.append(name)
            btn = _detect_button(cur)
            if btn is not None:
                mem["button"] = btn
            mem["targets"], mem["target_names"] = targets, names
            mem["stage"] = "touch"
            log += " ring@%d cue=%s cubes=%d" % (int(cur.step), ",".join(colors), len(cubes))
        else:
            mem["log"] = log
            return {"memory": mem, "act": {"tool": "act_wait_for_ring", "args": {"max_steps": 300}}}

    rpy = mem["home_rpy"]
    if mem["stage"] == "touch":
        i = int(mem["touched"])
        # retry the previous touch once if it did not complete (timeout inside the act or a tool fault)
        if (i > 0 and isinstance(last, str) and last.startswith("act_touch")
                and '"done": true' not in last and int(mem["retries"]) < 3):
            mem["retries"] = int(mem["retries"]) + 1
            i -= 1
            log += " retry%d" % i
        if i < len(mem["targets"]):
            x, y = mem["targets"][i]
            mem["touched"] = i + 1
            log += " touch%d:%s" % (i, mem["target_names"][i])
            mem["log"] = log
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": x, "y": y, "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}}}
        mem["stage"] = "press"

    if mem["stage"] == "press":
        mem["presses"] = int(mem["presses"]) + 1
        log += " press%d" % mem["presses"]
        mem["log"] = log
        bx, by = mem["button"]
        return {"memory": mem, "act": {"tool": "act_press_button", "args": {
            "x": bx, "y": by, "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}}}

    mem["log"] = log
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 20}}}


# ---------------------------------------------------------------- control
def _rotm(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-6:
        return np.zeros(3)
    return v / (2.0 * s) * ang


def _command(tcp, target, Rref, gain=1.0, rot_gain=0.5, vmax=1.0):
    p = np.asarray(tcp[:3], dtype=np.float64)
    e = (np.asarray(target, dtype=np.float64) - p) / 0.1 * gain
    e = np.clip(e, -vmax, vmax)
    w = _rotvec(Rref @ _rotm(tcp[3:6]).T)
    r = np.clip(-w / 0.1 * rot_gain, -0.5, 0.5)
    return [float(e[0]), float(e[1]), float(e[2]), float(r[0]), float(r[1]), float(r[2])]


def act_wait_for_ring(robot, max_steps=300):
    """Send zero commands (gripper open) until the response ring of cubes appears or max_steps elapse."""
    n = 0
    seen = False
    while n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        if _ring_present(robot.now()):
            seen = True
            break
    return {"ring_seen": seen, "steps": n, "step": int(robot.now().step)}


def act_touch(robot, x, y, roll, pitch, yaw, touch_z=TOUCH_Z, travel_z=TRAVEL_Z, hold=2, max_steps=90):
    """Travel at travel_z above (x, y), descend to touch_z with the gripper open, hold briefly and rise."""
    Rref = _rotm([roll, pitch, yaw])
    stage = "travel"
    held = 0
    n = 0
    zmin = 9.0
    xy_at_min = None
    while n < max_steps:
        f = robot.now()
        p = f.tcp
        dxy = math.hypot(x - float(p[0]), y - float(p[1]))
        if stage == "travel" and dxy < 0.02:
            stage = "descend"
        if stage == "descend" and float(p[2]) <= touch_z + 0.006 and dxy < 0.015:
            held += 1
            if held >= hold:
                stage = "ascend"
        if stage == "ascend" and float(p[2]) >= travel_z - 0.02:
            break
        zt = touch_z if stage == "descend" else travel_z
        if float(p[2]) < zmin:
            zmin = float(p[2])
            xy_at_min = [float(p[0]), float(p[1])]
        cmd = _command(p, [x, y, zt], Rref)
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], 1.0)
        n += 1
    p = robot.now().tcp
    return {"done": stage == "ascend", "stage": stage, "steps": n, "zmin": zmin, "xy_at_min": xy_at_min,
            "target": [x, y], "final": [float(p[0]), float(p[1]), float(p[2])], "step": int(robot.now().step)}


def act_press_button(robot, x, y, roll, pitch, yaw, press_steps=8, push=-0.4, lift_z=0.17, max_steps=120):
    """Close the gripper, move above the button cap centre, push down gently (holding xy and orientation)
    until the TCP has been at cap-depressed height for press_steps, then lift and wait. If the TCP sinks
    below the cap (slipped off onto the plate) the press is aborted and the arm lifts."""
    Rref = _rotm([roll, pitch, yaw])
    stage = "travel"
    n = 0
    pressed = 0
    zmin = 9.0
    lifted_at = -1
    slipped = False
    press_xy = []
    while n < max_steps:
        f = robot.now()
        p = f.tcp
        dxy = math.hypot(x - float(p[0]), y - float(p[1]))
        if stage == "travel" and dxy < 0.008 and float(f.fingers.mean()) < 0.01:
            stage = "descend"
        if stage == "descend" and float(p[2]) < 0.064:
            stage = "press"
        if stage == "press":
            if float(p[2]) < 0.046:
                slipped = True
                stage = "lift"
                lifted_at = n
            elif pressed >= press_steps:
                stage = "lift"
                lifted_at = n
        zmin = min(zmin, float(p[2]))
        if stage == "travel":
            cmd = _command(p, [x, y, TRAVEL_Z], Rref)
        elif stage == "descend":
            cmd = _command(p, [x, y, 0.03], Rref, vmax=0.6)
        elif stage == "press":
            cmd = _command(p, [x, y, 0.0], Rref, gain=2.0, rot_gain=1.0)
            cmd[2] = push
            if float(p[2]) < 0.057:
                pressed += 1
                press_xy.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
        else:
            cmd = _command(p, [x, y, lift_z], Rref)
            if lifted_at >= 0 and n - lifted_at > 40:
                break
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], -1.0)
        n += 1
    p = robot.now().tcp
    return {"stage": stage, "steps": n, "zmin": zmin, "pressed_steps": pressed, "slipped": slipped,
            "press_track": press_xy[:3] + press_xy[-2:],
            "final": [float(p[0]), float(p[1]), float(p[2])], "step": int(robot.now().step)}


def act_hold(robot, steps=20):
    """Hold position with zero motion (gripper closed) for a number of steps."""
    for _ in range(steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"held": steps, "step": int(robot.now().step)}
