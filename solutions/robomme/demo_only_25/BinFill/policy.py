"""BinFill policy: parse the goal's colour counts, put exactly that many cubes of each
colour into the bin (front RGB-D for coarse localisation, wrist RGB-D for the final
centre/yaw), count placements in memory, then press the button."""

COLORS = ("red", "green", "blue")
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "a": 1, "an": 1, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
# measured TCP sits ~5 mm in -x of the perceived object centre in every expert grasp
TCP_OFFSET_X = -0.005
BUTTON_OFFSET_X = -0.008
HOVER_Z = 0.15
BIN_Z = 0.20
CUBE_HALF = 0.02
HOME = (-0.06, 0.0, 0.40)
ROLL = math.pi


def _parse_goal(goal):
    text = str(goal).lower().replace(",", " ").replace(".", " ")
    words = text.split()
    want = []
    for i, w in enumerate(words):
        base = w[:-1] if w.endswith("s") else w
        if base == "cube" and i >= 2 and words[i - 1] in COLORS and words[i - 2] in NUMBERS:
            want.extend([words[i - 1]] * NUMBERS[words[i - 2]])
    return want


def _color_masks(rgb):
    im = rgb.astype(float)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {"red": (r > 50) & (g < 0.35 * r) & (b < 0.35 * r),
            "green": (g > 50) & (g > 1.5 * r) & (g > 1.5 * b),
            "blue": (b > 50) & (b > 1.5 * r) & (b > 1.3 * g)}


def _world_points(K, E, depth):
    """HxWx3 world points from optical-axis depth (mm); invalid depth gives NaN."""
    K = np.asarray(K, float)
    E = np.asarray(E, float)
    H, W = depth.shape
    d = depth.astype(float) / 1000.0
    v, u = np.mgrid[0:H, 0:W].astype(float)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1) - E[:, 3]
    P = pc @ E[:, :3]
    P[d <= 0] = np.nan
    return P


def _label(mask):
    """4-connected component labels of a boolean mask (0 = background)."""
    H, W = mask.shape
    lab = np.where(mask, np.arange(1, H * W + 1).reshape(H, W), 0)
    for _ in range(400):
        m = lab.copy()
        m[1:, :] = np.maximum(m[1:, :], lab[:-1, :])
        m[:-1, :] = np.maximum(m[:-1, :], lab[1:, :])
        m[:, 1:] = np.maximum(m[:, 1:], lab[:, :-1])
        m[:, :-1] = np.maximum(m[:, :-1], lab[:, 1:])
        m = np.where(mask, m, 0)
        if np.array_equal(m, lab):
            break
        lab = m
    return lab


def _components(mask, minpix):
    lab = _label(mask)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    return [lab == i for i, c in zip(ids, counts) if c >= minpix]


def _rect_angle(xy):
    """Edge angle (rad, in [-pi/4, pi/4)) of a square from its top-face points."""
    c = xy.mean(0)
    best = None
    for deg in range(-45, 45, 2):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        p = (xy - c) @ np.array([[ca, -sa], [sa, ca]])
        area = float(np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
        if best is None or area < best[0]:
            best = (area, a)
    a0 = best[1]
    for deg in (-1.0, 1.0):
        a = a0 + math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        p = (xy - c) @ np.array([[ca, -sa], [sa, ca]])
        area = float(np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
        if area < best[0]:
            best = (area, a)
    return best[1]


def _detect_bin_button(frame):
    """Bin centre (dark interior) and button centre (white plate) from the front camera."""
    P = _world_points(frame.front_K, frame.front_extrinsic, frame.front_depth)
    im = frame.front_rgb.astype(float)
    mx, mn = im.max(2), im.min(2)
    z = P[..., 2]
    zf = np.nan_to_num(z, nan=9.0)
    dark = (mx < 40) & (zf > 0.0) & (zf < 0.045)
    bin_xy = None
    comps = _components(dark, 30)
    if comps:
        big = max(comps, key=lambda m: int(m.sum()))
        pts = P[big]
        bin_xy = [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))]
    button_xy = None
    xs = np.nan_to_num(P[..., 0], nan=-9.0)
    white = (mn > 150) & (mx - mn < 30) & (zf < 0.05) & (xs > -0.4)
    if bin_xy is not None:
        dist = np.hypot(np.nan_to_num(P[..., 0], nan=9.0) - bin_xy[0], np.nan_to_num(P[..., 1], nan=9.0) - bin_xy[1])
        white = white & (dist > 0.07)
    comps = _components(white, 15)
    if comps:
        big = max(comps, key=lambda m: int(m.sum()))
        c = np.median(P[big], 0)
        grey = (mn > 60) & (mx - mn < 30) & (zf > 0.002) & (zf < 0.05)
        dist = np.hypot(np.nan_to_num(P[..., 0], nan=9.0) - c[0], np.nan_to_num(P[..., 1], nan=9.0) - c[1])
        pts = P[grey & (dist < 0.05)]
        if len(pts) >= 10:
            lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
            button_xy = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)]
        else:
            button_xy = [float(c[0]), float(c[1])]
    return bin_xy, button_xy


def _detect_cubes(rgb, depth, K, E, colors, zlo=0.028, zhi=0.06, split=False):
    """Cube top faces of the given colours: list of dicts with colour, xy, top z, points."""
    P = _world_points(K, E, depth)
    zf = np.nan_to_num(P[..., 2], nan=9.0)
    top = (zf > zlo) & (zf < zhi)
    out = []
    masks = _color_masks(rgb)
    for c in colors:
        comps = _components(masks[c] & top, 6)
        if split:
            comps = [s for comp in comps for s in _split(comp, P)]
        for comp in comps:
            pts = P[comp]
            out.append({"color": c, "xy": [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))],
                        "z": float(np.median(pts[:, 2])), "n": int(len(pts)), "pts": pts[:, :2]})
    return out


def _extent(xy):
    a = _rect_angle(xy)
    ca, sa = math.cos(a), math.sin(a)
    p = (xy - xy.mean(0)) @ np.array([[ca, -sa], [sa, ca]])
    return max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))


def _erode(mask):
    m = mask.copy()
    m[1:, :] &= mask[:-1, :]
    m[:-1, :] &= mask[1:, :]
    m[:, 1:] &= mask[:, :-1]
    m[:, :-1] &= mask[:, 1:]
    m[0, :] = False
    m[-1, :] = False
    m[:, 0] = False
    m[:, -1] = False
    return m


def _split(comp, P):
    """Separate touching same-colour top faces by erosion when the blob is wider than one cube."""
    pts = P[comp][:, :2]
    if len(pts) < 20 or _extent(pts) < 0.052:
        return [comp]
    m = comp
    for _ in range(12):
        m = _erode(m)
        parts = _components(m, 15)
        if len(parts) >= 2:
            return parts
        if not parts:
            break
    return [comp]


def _all_colored_points(rgb, depth, K, E):
    P = _world_points(K, E, depth)
    zf = np.nan_to_num(P[..., 2], nan=9.0)
    m = _color_masks(rgb)
    anyc = (m["red"] | m["green"] | m["blue"]) & (zf > 0.004) & (zf < 0.12)
    return P[anyc][:, :2]


def _choose_target(cubes, others_xy, bin_xy, avoid):
    best = None
    for cb in cubes:
        xy = np.array(cb["xy"])
        if bin_xy is not None and np.hypot(*(xy - np.array(bin_xy))) < 0.075:
            continue
        if xy[0] < -0.33 or xy[0] > 0.2 or abs(xy[1]) > 0.3:
            continue
        if any(np.hypot(*(xy - np.array(a))) < 0.02 for a in (avoid or [])):
            continue
        if len(others_xy):
            d = np.hypot(others_xy[:, 0] - xy[0], others_xy[:, 1] - xy[1])
            d = d[d > 0.035]
            clear = float(d.min()) if len(d) else 1.0
        else:
            clear = 1.0
        score = min(clear, 0.1) + 0.0001 * cb["n"]
        if best is None or score > best[0]:
            best = (score, cb)
    return None if best is None else best[1]


def perceive_decide(obs, goal, memory, last):
    """Parse the goal once, locate bin/button once, then pick-place each remaining cube
    colour in turn (counting only verified releases), and finally press the button."""
    f = obs.current
    mem = dict(memory)
    if mem.get("remaining") is None:
        mem["remaining"] = _parse_goal(goal)
        mem["placed"] = []
        mem["failures"] = 0
        mem["avoid"] = []
        mem["status"] = "start"
    if mem.get("bin") is None or mem.get("button") is None:
        b, k = _detect_bin_button(f)
        if mem.get("bin") is None and b is not None:
            mem["bin"] = b
        if mem.get("button") is None and k is not None:
            mem["button"] = k
    remaining = list(mem["remaining"])
    last = _last_text(last)
    if "placed_color" in last:
        for c in COLORS:
            if ('"placed_color": "%s"' % c) in last and c in remaining:
                remaining.remove(c)
                mem["placed"] = list(mem["placed"]) + [c]
                mem["avoid"] = []
                break
    elif "grasp_failed_at" in last:
        mem["failures"] = int(mem["failures"] or 0) + 1
        try:
            i = last.index("[", last.index("grasp_failed_at"))
            j = last.index("]", i)
            xy = [float(v) for v in last[i + 1:j].split(",")[:2]]
            mem["avoid"] = (list(mem["avoid"] or []) + [xy])[-3:]
        except (ValueError, IndexError):
            pass
    mem["remaining"] = remaining

    def out(tool, args, status):
        mem["status"] = status
        return {"memory": {"remaining": mem["remaining"], "placed": mem["placed"], "bin": mem.get("bin"),
                           "button": mem.get("button"), "failures": mem["failures"],
                           "avoid": mem["avoid"], "status": status},
                "act": {"tool": tool, "args": args}}

    if mem.get("bin") is None or mem.get("button") is None:
        return out("act_view_home", {"wait": 5}, "looking for bin/button")
    if not remaining:
        return out("act_press_button", {"button_xy": mem["button"]}, "pressing button")
    colors = sorted(set(remaining))
    cubes = _detect_cubes(f.front_rgb, f.front_depth, f.front_K, f.front_extrinsic, colors)
    others = _all_colored_points(f.front_rgb, f.front_depth, f.front_K, f.front_extrinsic)
    big = [c for c in cubes if c["n"] >= 10]
    tgt = _choose_target(big, others, mem["bin"], mem["avoid"])
    if tgt is None:
        tgt = _choose_target(cubes, others, mem["bin"], mem["avoid"])
    if tgt is None:
        if "viewed_home" in last:
            return out("act_view_home", {"wait": 25}, "waiting for target cubes")
        return out("act_view_home", {"wait": 3}, "no target visible; viewing from home")
    return out("act_pick_place", {"color": tgt["color"], "xy": tgt["xy"], "bin_xy": mem["bin"],
                                  "button_xy": mem["button"]},
               "picking %s" % tgt["color"])


def _cmd(robot, x, y, z, yaw, grip):
    return robot.step(float(x), float(y), float(z), ROLL, 0.0, float(yaw), float(grip))


def _move(robot, target, yaw, grip, speed=0.014, tol=0.004, settle=12):
    """Carrot-interpolated straight-line move of the TCP to target with fixed gripper command."""
    f = robot.now()
    start = np.array(f.tcp[:3], float)
    yaw0 = float(f.tcp[5])
    goal = np.array(target, float)
    dist = float(np.linalg.norm(goal - start))
    dyaw = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))
    n = max(1, int(math.ceil(max(dist / speed, abs(dyaw) / 0.08))))
    for i in range(1, n + 1):
        s = i / n
        p = start + (goal - start) * s
        _cmd(robot, p[0], p[1], p[2], yaw0 + dyaw * s, grip)
    for _ in range(settle):
        f = robot.now()
        if np.linalg.norm(np.array(f.tcp[:3]) - goal) < tol:
            break
        _cmd(robot, goal[0], goal[1], goal[2], yaw0 + dyaw, grip)
    return robot.now()


def _gripper(robot, grip, n):
    f = robot.now()
    p = f.tcp
    for _ in range(n):
        _cmd(robot, p[0], p[1], p[2], p[5], grip)
    return robot.now()


def _choose_yaw(center, angle, others, cur_yaw):
    """Pick the grasp yaw (cube edge angle mod 90 deg) whose finger sweep has most clearance."""
    best = None
    for k in (-1, 0, 1):
        yaw = angle + k * math.pi / 2
        if abs(yaw) > math.radians(100):
            continue
        d = np.array([math.sin(yaw), -math.cos(yaw)])
        nrm = np.array([math.cos(yaw), math.sin(yaw)])
        conflicts = 0
        if len(others):
            rel = others - np.array(center)
            far = np.hypot(rel[:, 0], rel[:, 1]) > 0.032
            along = np.abs(rel @ d)
            perp = np.abs(rel @ nrm)
            conflicts = int(np.sum(far & (along < 0.075) & (perp < 0.022)))
        cost = conflicts * 1.0 + 0.01 * abs(yaw)
        if best is None or cost < best[0]:
            best = (cost, yaw, conflicts)
    return best[1], best[2]


def _cam_offset(frame, yaw):
    """World xy offset of the wrist camera from the TCP if the TCP had this yaw."""
    E = np.asarray(frame.wrist_extrinsic, float)
    cam = -E[:, :3].T @ E[:, 3]
    d = cam[:2] - np.asarray(frame.tcp[:2], float)
    a = yaw - float(frame.tcp[5])
    ca, sa = math.cos(a), math.sin(a)
    return [ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]]


def _last_text(last):
    if last is None:
        return ""
    if isinstance(last, dict):
        parts = []
        for k, v in last.items():
            parts.append('"%s": %s' % (k, ('"%s"' % v) if isinstance(v, str) else str(v)))
        return "{" + ", ".join(parts) + "}"
    return str(last)


def act_view_home(robot, wait=5):
    """Move the arm high near the base (clear front view) with the gripper open and wait `wait` steps."""
    f = robot.now()
    _move(robot, HOME, 0.0, 1.0, speed=0.018, settle=6)
    _gripper(robot, 1.0, int(max(1, min(60, wait))))
    return {"viewed_home": True}


def act_pick_place(robot, color, xy, bin_xy, button_xy=None):
    """Hover over the front-camera cube estimate `xy`, refine centre/yaw with the wrist camera,
    grasp, verify by finger width, carry to above `bin_xy` and release.
    Returns placed_color on a verified release, otherwise grasp_failed_at [x, y]."""
    f = robot.now()
    yaw_now = float(f.tcp[5])
    if f.tcp[2] < 0.12:
        _move(robot, [f.tcp[0], f.tcp[1], HOVER_Z], yaw_now, 1.0, speed=0.012, settle=3)
    off = _cam_offset(f, 0.0)
    hover = [xy[0] - off[0], xy[1] - off[1], HOVER_Z]
    f = _move(robot, hover, 0.0, 1.0, speed=0.016, tol=0.005, settle=10)
    center, angle, topz = list(xy), 0.0, 0.04
    found = False
    others = np.zeros((0, 2))
    for attempt in range(2):
        f = _gripper(robot, 1.0, 2)
        cubes = _detect_cubes(f.wrist_rgb, f.wrist_depth, f.wrist_K, f.wrist_extrinsic, [color], zlo=0.028, zhi=0.07, split=True)
        cand = [c for c in cubes if c["n"] >= 25 and np.hypot(c["xy"][0] - center[0], c["xy"][1] - center[1]) < 0.045]
        if not cand:
            break
        c = min(cand, key=lambda c: np.hypot(c["xy"][0] - center[0], c["xy"][1] - center[1]))
        center, topz = c["xy"], c["z"]
        angle = _rect_angle(c["pts"])
        others = _all_colored_points(f.wrist_rgb, f.wrist_depth, f.wrist_K, f.wrist_extrinsic)
        found = True
        off = _cam_offset(f, 0.0)
        if attempt == 0 and np.hypot(center[0] - off[0] - f.tcp[0], center[1] - off[1] - f.tcp[1]) > 0.025:
            f = _move(robot, [center[0] - off[0], center[1] - off[1], HOVER_Z], 0.0, 1.0, speed=0.012, settle=6)
            continue
        break
    if not found:
        return {"grasp_failed_at": [float(xy[0]), float(xy[1])], "reason": "cube not seen by wrist camera"}
    obst = [others]
    for c0, half in ((bin_xy, 0.055), (button_xy, 0.038)):
        if c0 is not None:
            s = np.linspace(-half, half, 12)
            ring = [[c0[0] + a, c0[1] + b] for a in s for b in (-half, half)] + \
                   [[c0[0] + b, c0[1] + a] for a in s for b in (-half, half)]
            obst.append(np.array(ring))
    others = np.concatenate(obst, 0)
    yaw, conflicts = _choose_yaw(center, angle, others, yaw_now)
    gx, gy = center[0] + TCP_OFFSET_X, center[1]
    gz = max(0.015, min(0.06, topz - CUBE_HALF - 0.001))
    _move(robot, [gx, gy, HOVER_Z], yaw, 1.0, speed=0.012, settle=4)
    _move(robot, [gx, gy, gz + 0.04], yaw, 1.0, speed=0.014, settle=2)
    _move(robot, [gx, gy, gz], yaw, 1.0, speed=0.008, tol=0.003, settle=8)
    f = _gripper(robot, -1.0, 8)
    f = _move(robot, [gx, gy, HOVER_Z], yaw, -1.0, speed=0.012, settle=3)
    width = float(np.mean(f.fingers))
    if width < 0.008:
        _gripper(robot, 1.0, 5)
        return {"grasp_failed_at": [float(center[0]), float(center[1])], "reason": "empty gripper",
                "finger": width, "conflicts": conflicts}
    target = [bin_xy[0] + TCP_OFFSET_X, bin_xy[1], BIN_Z]
    f = _move(robot, target, yaw, -1.0, speed=0.016, tol=0.006, settle=10)
    width = float(np.mean(f.fingers))
    if width < 0.008:
        _gripper(robot, 1.0, 5)
        return {"grasp_failed_at": [float(center[0]), float(center[1])], "reason": "cube dropped in transit"}
    f = _gripper(robot, 1.0, 8)
    return {"placed_color": color, "grasp_xy": [float(gx), float(gy)], "yaw": float(yaw),
            "finger_before_release": width}


def act_press_button(robot, button_xy):
    """Close the gripper above the button, push its cap down to z~0.02 and come back up."""
    f = robot.now()
    bx, by = button_xy[0] + BUTTON_OFFSET_X, button_xy[1]
    if f.tcp[2] < 0.12:
        _move(robot, [f.tcp[0], f.tcp[1], 0.15], float(f.tcp[5]), 1.0, speed=0.012, settle=3)
    _move(robot, [bx, by, 0.15], 0.0, -1.0, speed=0.016, tol=0.005, settle=10)
    _gripper(robot, -1.0, 4)
    _move(robot, [bx, by, 0.06], 0.0, -1.0, speed=0.012, settle=2)
    _move(robot, [bx, by, 0.022], 0.0, -1.0, speed=0.005, tol=0.003, settle=10)
    _gripper(robot, -1.0, 6)
    _move(robot, [bx, by, 0.15], 0.0, -1.0, speed=0.012, settle=3)
    return {"pressed": True}
