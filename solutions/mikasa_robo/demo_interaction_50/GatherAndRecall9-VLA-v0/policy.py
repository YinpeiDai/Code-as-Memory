"""GatherAndRecall9 policy: move 9 cubes onto the disc, watch the lamp flash, press matching button.

Perception uses the fixed front camera (RGB + depth back-projected to world).
Control is closed-loop on measured TCP with proportional delta-pose commands.
"""

TRANSIT_Z = 0.13      # carry / travel height (m), well above button caps (0.06) and lamp path
GRASP_Z = 0.025       # TCP height for grasping a cube (cube centre ~0.02)
PLACE_Z = 0.045       # TCP height at which a carried cube is released over the disc
GRID_DX = 0.05        # slot pitch along x (perpendicular to the finger axis)
GRID_DY = 0.08        # slot pitch along y (finger axis): leaves room for the finger between cubes
RELEASE_G = 1.0       # release opening (R002: 0.5 still opened fully to 0.04, so the gripper acts binary)
OUTLIER_R = 0.094     # cubes measured farther than this from the disc centre are re-placed
                      # (R001 failed with a cube at 0.1004, R002 succeeded with max 0.098; top-face measure)
PRESS_Z = 0.035       # TCP target while pressing (cap top ~0.06); success ends episode first
COLORS = ("red", "green", "blue")
HOLD_OFFSET = (0.002, -0.0047)  # measured cube-centre minus release-TCP offset (R002, 9 placements, front camera)


# ----------------------------------------------------------------- geometry
def _world_points(depth, K, E):
    D = depth.astype(np.float64) / 1000.0
    v, u = np.mgrid[0:256, 0:256]
    x = (u - K[0, 2]) * D / K[0, 0]
    y = (v - K[1, 2]) * D / K[1, 1]
    pc = np.stack([x, y, D], -1).reshape(-1, 3)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    W = (pc - t) @ R
    W = W.reshape(256, 256, 3)
    W[D <= 0] = np.nan
    return W


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    return th / (2 * s) * v


def _color_masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (
        (r > 120) & (g < 80) & (b < 80),
        (g > 120) & (r < 80) & (b < 80),
        (b > 120) & (r < 80) & (g < 80),
    )


def _cluster(pts, rad):
    clusters = []
    for p in pts:
        best = -1
        bd = rad
        for i, c in enumerate(clusters):
            d = math.hypot(p[0] - c[0] / c[2], p[1] - c[1] / c[2])
            if d < bd:
                bd = d
                best = i
        if best < 0:
            clusters.append([p[0], p[1], 1])
        else:
            clusters[best][0] += p[0]
            clusters[best][1] += p[1]
            clusters[best][2] += 1
    out = [[c[0] / c[2], c[1] / c[2], c[2]] for c in clusters]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if math.hypot(out[i][0] - out[j][0], out[i][1] - out[j][1]) < rad:
                    n = out[i][2] + out[j][2]
                    out[i] = [(out[i][0] * out[i][2] + out[j][0] * out[j][2]) / n,
                              (out[i][1] * out[i][2] + out[j][1] * out[j][2]) / n, n]
                    out.pop(j)
                    merged = True
                    break
            if merged:
                break
    return out


def _detect_disc(W, rgb):
    mx = rgb.max(2).astype(np.int32)
    mn = rgb.min(2).astype(np.int32)
    z = W[..., 2]
    m = ((mx - mn) < 20) & (mn > 120) & (z > 0.002) & (z < 0.02)
    p = W[m][:, :2]
    if len(p) < 50:
        return None, 0
    c = np.median(p, 0)
    q = p
    for _ in range(3):
        q = p[np.linalg.norm(p - c, axis=1) < 0.15]
        if len(q) < 30:
            return None, 0
        c = (q.min(0) + q.max(0)) / 2.0
    rad = float(np.mean((q.max(0) - q.min(0)) / 2.0))
    return [float(c[0]), float(c[1])], rad


def _detect_cubes(W, rgb, disc_c, disc_r, buttons=None, zlo=0.036):
    """Cube top faces: saturated R/G/B pixels whose world z is zlo-0.052 (cube tops at ~0.04).
    Points on the disc and near button caps are excluded."""
    masks = _color_masks(rgb)
    z = W[..., 2]
    out = []
    for ci in range(3):
        m = masks[ci] & (z > zlo) & (z < 0.052)
        p = W[m]
        if len(p) == 0:
            continue
        if disc_c is not None:
            dd = np.hypot(p[:, 0] - disc_c[0], p[:, 1] - disc_c[1])
            p = p[dd > disc_r + 0.01]
        for b in (buttons or []):
            if b is not None and len(p):
                p = p[np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) > 0.045]
        for c in _cluster(p, 0.04):
            if c[2] >= 4:
                out.append([round(c[0], 4), round(c[1], 4), 0.02, ci, c[2]])
    return out


def _detect_buttons(W, rgb):
    masks = _color_masks(rgb)
    z = W[..., 2]
    out = []
    for ci in range(3):
        m = masks[ci] & (z > 0.052) & (z < 0.08)
        p = W[m]
        if len(p) < 5:
            out.append(None)
            continue
        c = np.median(p, 0)
        q = p[np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < 0.03]
        if len(q) < 5:
            q = p
        out.append([float(q[:, 0].mean()), float(q[:, 1].mean()), float(np.median(q[:, 2]))])
    return out


def _detect_lamp(W, rgb):
    """White, low-saturation blob above the table in front of the disc (x > 0.15)."""
    mx = rgb.max(2).astype(np.int32)
    mn = rgb.min(2).astype(np.int32)
    z = W[..., 2]
    m = ((mx - mn) < 35) & (mn > 140) & (z > 0.03) & (z < 0.15) & (W[..., 0] > 0.15)
    vs, us = np.nonzero(m)
    if len(vs) < 20:
        return None
    cv, cu = int(np.median(vs)), int(np.median(us))
    return [max(0, cv - 6), min(256, cv + 7), max(0, cu - 6), min(256, cu + 7)]


def _lamp_color(rgb, roi):
    """Returns (color index or -1, mean rgb) of the lamp ROI."""
    if roi is None:
        return -1, None
    patch = rgb[roi[0]:roi[1], roi[2]:roi[3]].reshape(-1, 3).astype(np.float64)
    m = patch.mean(0)
    if m.max() - m.min() > 60:
        return int(np.argmax(m)), m
    return -1, m


def _scan_flash(frames, roi):
    """Scan frames for a saturated lamp colour. Returns (color_idx, step, count) of the majority colour."""
    counts = [0, 0, 0]
    first = [None, None, None]
    for f in frames:
        ci, _ = _lamp_color(f.front_rgb, roi)
        if ci >= 0:
            counts[ci] += 1
            if first[ci] is None:
                first[ci] = int(f.step)
    best = int(np.argmax(counts))
    if counts[best] == 0:
        return -1, None, 0
    return best, first[best], counts[best]


def _grid_slots(c):
    """10 slots in a brick pattern: rows at x offsets -0.075,-0.025,+0.025,+0.075 (pitch GRID_DX);
    inner rows hold 3 cubes at y = -0.08, 0, +0.08, outer rows 2 cubes at y = +-0.04 (pitch GRID_DY).
    Any two slots differ by >= GRID_DX in x or >= GRID_DY in y (room for the fingers along y).
    Max radius 0.085 m. The 10th slot is a spare used for repairs."""
    rel = [(-0.025, 0.08), (0.025, 0.08), (-0.075, 0.04), (0.075, 0.04),
           (-0.025, 0.0), (0.025, 0.0),
           (-0.075, -0.04), (-0.025, -0.08), (0.025, -0.08), (0.075, -0.04)]
    return [[round(c[0] + dx, 4), round(c[1] + dy, 4), PLACE_Z] for dx, dy in rel]


def _disc_cubes(W, rgb, disc_c, disc_r, lamp_roi=None):
    """Cubes resting on the disc (top faces z 0.044-0.058, disc top ~0.008) and cubes elsewhere on the
    table (top faces z 0.034-0.046). Only top-face pixels are used so the centroid is not pulled
    toward the camera by side faces. The lamp (which may be flashing) is masked out.
    Returns (on_disc, off_disc) lists of [x, y, dist_to_centre, n_pixels]."""
    masks = _color_masks(rgb)
    z = W[..., 2]
    keep = np.ones(z.shape, dtype=bool)
    if lamp_roi is not None:
        keep[max(0, lamp_roi[0] - 25):lamp_roi[1] + 25, max(0, lamp_roi[2] - 25):lamp_roi[3] + 25] = False
    d_all = np.hypot(W[..., 0] - disc_c[0], W[..., 1] - disc_c[1])
    inside = d_all < disc_r + 0.015
    on, off = [], []
    for ci in range(3):
        for which, zl, zh in (("on", 0.044, 0.058), ("off", 0.034, 0.046)):
            region = inside if which == "on" else ~inside
            m = masks[ci] & keep & region & (z > zl) & (z < zh)
            p = W[m]
            if len(p) == 0:
                continue
            for c in _cluster(p, 0.03):
                if c[2] < 6:
                    continue
                d = math.hypot(c[0] - disc_c[0], c[1] - disc_c[1])
                rec = [round(c[0], 4), round(c[1], 4), round(d, 4), int(c[2])]
                (on if which == "on" else off).append(rec)
    return on, off


def _free_spot(disc_c, cubes, skip=None):
    """Position on the disc (radius <= 0.088) with finger clearance from every measured cube.
    Clearance score per cube = max(|dx|/GRID_DX, |dy|/GRID_DY); spots need min score >= 0.9
    (measured positions carry a few mm of noise). Smallest radius wins; else best score >= 0.75."""
    best = None
    alt = None
    for ix in range(-9, 10):
        for iy in range(-9, 10):
            x = disc_c[0] + 0.01 * ix
            y = disc_c[1] + 0.01 * iy
            r = math.hypot(x - disc_c[0], y - disc_c[1])
            if r > 0.088:
                continue
            sc = 9.0
            for c in cubes:
                if skip is not None and math.hypot(c[0] - skip[0], c[1] - skip[1]) < 0.01:
                    continue
                sc = min(sc, max(abs(c[0] - x) / GRID_DX, abs(c[1] - y) / GRID_DY))
            if sc >= 0.9 and (best is None or r < best[2]):
                best = [round(x, 4), round(y, 4), r]
            if alt is None or sc > alt[2]:
                alt = [round(x, 4), round(y, 4), sc]
    if best is None:
        if alt is None or alt[2] < 0.75:
            return None
        best = alt
    return [best[0], best[1], PLACE_Z]


# ----------------------------------------------------------------- control
def _cmd(tcp, target, R_t, kp=10.0, kr=5.0, vmax=1.0):
    p = np.asarray(tcp[:3], dtype=np.float64)
    e = np.asarray(target, dtype=np.float64) - p
    a = kp * e
    m = np.max(np.abs(a))
    if m > vmax:
        a = a * (vmax / m)
    Rm = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    w = _rotvec(R_t @ Rm.T)
    r = -kr * w
    rn = np.max(np.abs(r))
    if rn > 0.5:
        r = r * (0.5 / rn)
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])], float(np.linalg.norm(e))


def _flash_check(robot, roi, flash):
    """Update flash record [color_idx, step, count] from the latest frame."""
    f = robot.now()
    ci, _ = _lamp_color(f.front_rgb, roi)
    if ci >= 0:
        if flash[0] < 0:
            flash[0] = ci
            flash[1] = int(f.step)
            flash[2] = 1
        elif flash[0] == ci:
            flash[2] += 1
        else:
            flash[3] += 1  # conflicting colour frames
    return flash


def _move(robot, target, R_t, grip, roi, flash, tol=0.005, vtol=0.004, max_steps=60, vmax=1.0, kp=10.0):
    """Step toward target until within tol and moving slower than vtol (m/step)."""
    n = 0
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    err = 1.0
    while n < max_steps:
        tcp = robot.now().tcp
        a, err = _cmd(tcp, target, R_t, kp=kp, vmax=vmax)
        cur = np.asarray(tcp[:3], dtype=np.float64)
        speed = float(np.linalg.norm(cur - prev))
        if err < tol and (vtol is None or speed < vtol) and n > 0:
            break
        prev = cur
        robot.step(a[0], a[1], a[2], a[3], a[4], a[5], grip)
        _flash_check(robot, roi, flash)
        n += 1
    tcp = robot.now().tcp
    return n, float(np.linalg.norm(np.asarray(tcp[:3]) - np.asarray(target)))


def _gripper(robot, target_tcp, R_t, grip, roi, flash, n_max=8, settle=0.0015):
    """Hold position while the fingers move; stop when finger motion is small."""
    n = 0
    prevf = float(robot.now().fingers[0])
    while n < n_max:
        a, _ = _cmd(robot.now().tcp, target_tcp, R_t)
        robot.step(a[0], a[1], a[2], a[3], a[4], a[5], grip)
        _flash_check(robot, roi, flash)
        n += 1
        f = float(robot.now().fingers[0])
        if n >= 2 and abs(f - prevf) < settle:
            break
        prevf = f
    return n, float(robot.now().fingers[0])


# ----------------------------------------------------------------- acts
def act_pick_place(robot, cube, slot, rpy, lamp_roi, transit_z=TRANSIT_Z, grasp_z=GRASP_Z):
    """Pick the cube at `cube` [x,y,z] and release it over `slot` [x,y,z] on the disc.
    Watches the lamp ROI every step. Returns step counts, finger width while holding,
    final error and any flash seen ([color_idx, first_step, count, conflicts])."""
    R_t = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    flash = [-1, None, 0, 0]
    log = {}
    s0 = int(robot.now().step)
    cx, cy = float(cube[0]), float(cube[1])
    sx, sy, sz = float(slot[0]) - HOLD_OFFSET[0], float(slot[1]) - HOLD_OFFSET[1], float(slot[2])
    # 1. above cube (blend: do not wait for full settle)
    tcp = robot.now().tcp
    hz = max(float(tcp[2]), transit_z) if math.hypot(tcp[0] - cx, tcp[1] - cy) > 0.05 else float(tcp[2])
    if float(tcp[2]) < transit_z - 0.03 and math.hypot(tcp[0] - cx, tcp[1] - cy) > 0.03:
        _move(robot, [float(tcp[0]), float(tcp[1]), transit_z], R_t, 1.0, lamp_roi, flash, tol=0.03, vtol=None, max_steps=20)
    n1, e1 = _move(robot, [cx, cy, transit_z], R_t, 1.0, lamp_roi, flash, tol=0.008, vtol=0.006, max_steps=60)
    # 2. descend to grasp height
    n2, e2 = _move(robot, [cx, cy, grasp_z], R_t, 1.0, lamp_roi, flash, tol=0.004, vtol=0.003, max_steps=40)
    # 3. close
    n3, fw = _gripper(robot, [cx, cy, grasp_z], R_t, -1.0, lamp_roi, flash, n_max=8)
    log["grasp_xy_err"] = [round(float(robot.now().tcp[0]) - cx, 4), round(float(robot.now().tcp[1]) - cy, 4)]
    log["finger_closed"] = round(fw, 4)
    holding = fw > 0.008
    # 4. lift
    n4, _ = _move(robot, [cx, cy, transit_z], R_t, -1.0, lamp_roi, flash, tol=0.03, vtol=None, max_steps=30)
    fw2 = float(robot.now().fingers[0])
    log["finger_lifted"] = round(fw2, 4)
    if fw2 < 0.008:
        log["result"] = "grasp_failed"
        log["steps"] = int(robot.now().step) - s0
        log["flash"] = flash
        return log
    # 5. transport above slot
    n5, e5 = _move(robot, [sx, sy, transit_z], R_t, -1.0, lamp_roi, flash, tol=0.008, vtol=0.006, max_steps=60)
    # 6. descend
    n6, e6 = _move(robot, [sx, sy, sz], R_t, -1.0, lamp_roi, flash, tol=0.004, vtol=0.003, max_steps=40)
    fw3 = float(robot.now().fingers[0])
    log["finger_before_release"] = round(fw3, 4)
    # 7. release
    n7, fo = _gripper(robot, [sx, sy, sz], R_t, RELEASE_G, lamp_roi, flash, n_max=8)
    log["finger_released"] = round(fo, 4)
    t = robot.now().tcp
    log["release_tcp"] = [round(float(t[0]), 4), round(float(t[1]), 4), round(float(t[2]), 4)]
    # 8. retreat up
    n8, _ = _move(robot, [sx, sy, transit_z], R_t, RELEASE_G, lamp_roi, flash, tol=0.03, vtol=None, max_steps=30)
    log["result"] = "placed" if fw3 > 0.008 else "dropped_in_transit"
    log["seg_steps"] = [n1, n2, n3, n4, n5, n6, n7, n8]
    log["steps"] = int(robot.now().step) - s0
    log["flash"] = flash
    return log


def act_wait(robot, n, rpy, lamp_roi):
    """Hold the current pose for up to n steps while watching the lamp; stops early once a flash is seen."""
    R_t = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    flash = [-1, None, 0, 0]
    t0 = robot.now().tcp
    hold = [float(t0[0]), float(t0[1]), float(t0[2])]
    k = 0
    while k < n:
        a, _ = _cmd(robot.now().tcp, hold, R_t)
        robot.step(a[0], a[1], a[2], a[3], a[4], a[5], 1.0)
        _flash_check(robot, lamp_roi, flash)
        k += 1
        if flash[0] >= 0 and flash[2] >= 3:
            break
    return {"steps": k, "flash": flash}


def act_press(robot, button, rpy, lamp_roi, approach_z=0.12, press_z=PRESS_Z, hold_steps=15):
    """Close the gripper, move above `button` [x,y,z cap], descend onto it and hold.
    Success ends the episode inside this call. Returns the lowest TCP z reached."""
    R_t = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    flash = [-1, None, 0, 0]
    bx, by = float(button[0]), float(button[1])
    tcp = robot.now().tcp
    if float(tcp[2]) < approach_z - 0.02:
        _move(robot, [float(tcp[0]), float(tcp[1]), approach_z], R_t, -1.0, lamp_roi, flash, tol=0.03, vtol=None, max_steps=20)
    n1, e1 = _move(robot, [bx, by, approach_z], R_t, -1.0, lamp_roi, flash, tol=0.005, vtol=0.004, max_steps=60)
    n2, e2 = _move(robot, [bx, by, press_z], R_t, -1.0, lamp_roi, flash, tol=0.003, vtol=0.002, max_steps=40, vmax=0.6)
    zmin = float(robot.now().tcp[2])
    k = 0
    while k < hold_steps:
        a, _ = _cmd(robot.now().tcp, [bx, by, press_z], R_t)
        robot.step(a[0], a[1], a[2], a[3], a[4], a[5], -1.0)
        zmin = min(zmin, float(robot.now().tcp[2]))
        k += 1
    # lift back (in case success requires release)
    n3, _ = _move(robot, [bx, by, approach_z], R_t, -1.0, lamp_roi, flash, tol=0.01, vtol=None, max_steps=20)
    t = robot.now().tcp
    return {"seg_steps": [n1, n2, k, n3], "zmin": round(zmin, 4), "xy_err": [round(float(t[0]) - bx, 4), round(float(t[1]) - by, 4)]}


# ----------------------------------------------------------------- decision
def act_inspect(robot, view, rpy, lamp_roi):
    """Move (closed loop) to the view pose `view` [x,y,z] away from the disc so the front camera sees
    every cube; watches the lamp. Returns the reached pose."""
    R_t = _rpy_to_R(rpy[0], rpy[1], rpy[2])
    flash = [-1, None, 0, 0]
    tcp = robot.now().tcp
    if float(tcp[2]) < float(view[2]) - 0.03:
        _move(robot, [float(tcp[0]), float(tcp[1]), float(view[2])], R_t, 1.0, lamp_roi, flash, tol=0.03, vtol=None, max_steps=20)
    n, e = _move(robot, [float(view[0]), float(view[1]), float(view[2])], R_t, 1.0, lamp_roi, flash, tol=0.01, vtol=0.006, max_steps=50)
    return {"steps": n, "err": round(e, 4), "flash": flash}


def _init_scene(f):
    W = _world_points(f.front_depth, f.front_K, f.front_extrinsic)
    rgb = f.front_rgb
    disc_c, disc_r = _detect_disc(W, rgb)
    buttons = _detect_buttons(W, rgb)
    cubes = _detect_cubes(W, rgb, disc_c, disc_r if disc_c is not None else 0.12, buttons)
    roi = _detect_lamp(W, rgb)
    return disc_c, disc_r, cubes, buttons, roi


def _parse_last(last):
    """Split `last` ('tool -> result json text') into tool name and result text."""
    if not last or "->" not in last:
        return None, ""
    tool, _, res = last.partition("->")
    return tool.strip(), res.strip()


def _view_pose(mem):
    b = None
    if mem.get("flash_color") in COLORS:
        b = mem["buttons"][COLORS.index(mem["flash_color"])]
    if b is None:
        bs = [x for x in (mem.get("buttons") or []) if x is not None]
        b = bs[len(bs) // 2] if bs else [-0.2, mem["disc_center"][1], 0.06]
    return [round(float(b[0]), 4), round(float(b[1]), 4), 0.15]


def perceive_decide(obs, goal, memory, last):
    """Plan-and-track policy for GatherAndRecall9: init -> gather -> inspect (-> repair) -> await_flash -> press."""
    mem = dict(memory or {})
    f = obs.current
    stage = mem.get("stage") or "init"
    log = mem.get("log") or ""
    if stage == "init":
        disc_c, disc_r, cubes, buttons, roi = _init_scene(f)
        t = f.tcp
        mem["rpy"] = [float(t[3]), float(t[4]), float(t[5])]
        mem["disc_center"] = disc_c
        mem["disc_radius"] = disc_r
        mem["cubes"] = [[c[0], c[1], c[2]] for c in cubes]
        mem["buttons"] = buttons
        mem["lamp_roi"] = roi
        mem["slots"] = _grid_slots(disc_c) if disc_c is not None else []
        mem["placed"] = 0
        mem["flash_color"] = None
        mem["flash_step"] = None
        mem["flash_votes"] = [0, 0, 0]
        mem["attempts"] = 0
        mem["waits"] = 0
        mem["repairs"] = 0
        mem["inspections"] = 0
        mem["presses"] = 0
        log = "init: %d cubes, disc %s r=%.3f, buttons %s, lamp %s" % (len(cubes), disc_c, disc_r, buttons, roi)
        stage = "gather"
    roi = mem.get("lamp_roi")
    # flash scan over every frame delivered since the last decision
    ci, st, cnt = _scan_flash(obs.frames, roi)
    votes = list(mem.get("flash_votes") or [0, 0, 0])
    if ci >= 0:
        votes[ci] += cnt
        if mem.get("flash_step") is None:
            mem["flash_step"] = st
    if max(votes) > 0:
        mem["flash_color"] = COLORS[int(np.argmax(votes))]
    mem["flash_votes"] = votes
    tool, res = _parse_last(last)
    disc_c = mem["disc_center"]
    disc_r = mem["disc_radius"]
    if tool == "act_pick_place":
        if '"result": "placed"' in res:
            if stage != "repair":
                mem["placed"] = int(mem["placed"]) + 1
            if stage != "repair":
                mem["cubes"] = mem["cubes"][1:]
            mem["attempts"] = 0
        elif stage != "repair":
            # grasp failed or cube dropped: look again from the unoccluded view pose before retrying
            mem["attempts"] = int(mem["attempts"]) + 1
            stage = "relocate"
        log = (log + " | pp:" + res[:220])[-2500:]
    elif tool:
        log = (log + " | " + tool + ":" + res[:160])[-2500:]
    act = None
    if stage == "repair" and tool == "act_pick_place":
        stage = "inspect"
    if stage == "relocate":
        if tool == "act_inspect":
            W = _world_points(f.front_depth, f.front_K, f.front_extrinsic)
            on, off = _disc_cubes(W, f.front_rgb, disc_c, disc_r, roi)
            off = [o for o in off if all(b is None or math.hypot(o[0] - b[0], o[1] - b[1]) > 0.045 for b in mem["buttons"])]
            log = (log + " | relocate: off=%s" % [o[:2] for o in off])[-2500:]
            mem["cubes"] = [[o[0], o[1], 0.02] for o in off]
            if int(mem["attempts"]) >= 6:
                mem["cubes"] = []
            stage = "gather"
        else:
            act = {"tool": "act_inspect", "args": {"view": _view_pose(mem), "rpy": mem["rpy"], "lamp_roi": roi}}
    if act is None and stage == "gather":
        if mem["cubes"] and int(mem["placed"]) < 9:
            t = f.tcp
            cubes = mem["cubes"]
            dist = [math.hypot(c[0] - t[0], c[1] - t[1]) for c in cubes]
            if int(mem["attempts"]) >= 3:
                k = max(range(len(cubes)), key=lambda i: dist[i])  # repeated failures: try a different cube
            else:
                k = min(range(len(cubes)), key=lambda i: dist[i])
            if True:
                cubes = [cubes[k]] + cubes[:k] + cubes[k + 1:]
                mem["cubes"] = cubes
            slot = mem["slots"][min(int(mem["placed"]), len(mem["slots"]) - 1)]
            act = {"tool": "act_pick_place", "args": {"cube": cubes[0], "slot": slot, "rpy": mem["rpy"], "lamp_roi": roi}}
        else:
            stage = "inspect"
    if act is None and stage == "inspect":
        if tool == "act_inspect":
            W = _world_points(f.front_depth, f.front_K, f.front_extrinsic)
            on, off = _disc_cubes(W, f.front_rgb, disc_c, disc_r, roi)
            off = [o for o in off if all(b is None or math.hypot(o[0] - b[0], o[1] - b[1]) > 0.045 for b in mem["buttons"])]
            bad = [o for o in on if o[2] > OUTLIER_R] + off
            log = (log + " | inspect: on=%d off=%d bad=%s" % (len(on), len(off), [b[:3] for b in bad]))[-2500:]
            if bad and int(mem["repairs"]) < 3:
                b = max(bad, key=lambda q: q[2])
                spot = _free_spot(disc_c, on, skip=b)
                if spot is not None:
                    mem["repairs"] = int(mem["repairs"]) + 1
                    stage = "repair"
                    act = {"tool": "act_pick_place", "args": {"cube": [b[0], b[1], 0.02], "slot": spot, "rpy": mem["rpy"], "lamp_roi": roi}}
            if act is None:
                stage = "await_flash"
        elif int(mem["inspections"]) < 4:
            mem["inspections"] = int(mem["inspections"]) + 1
            act = {"tool": "act_inspect", "args": {"view": _view_pose(mem), "rpy": mem["rpy"], "lamp_roi": roi}}
        else:
            stage = "await_flash"
    if act is None and stage == "await_flash":
        if mem.get("flash_color") is None and int(mem["waits"]) < 4:
            mem["waits"] = int(mem["waits"]) + 1
            act = {"tool": "act_wait", "args": {"n": 40, "rpy": mem["rpy"], "lamp_roi": roi}}
        else:
            stage = "press"
    if act is None and stage == "press":
        color = mem.get("flash_color") or "green"
        b = mem["buttons"][COLORS.index(color)]
        if b is None:
            b = [bb for bb in mem["buttons"] if bb is not None][0]
        mem["presses"] = int(mem["presses"]) + 1
        act = {"tool": "act_press", "args": {"button": b, "rpy": mem["rpy"], "lamp_roi": roi}}
    mem["stage"] = stage
    mem["log"] = log
    out = {}
    for k in _FIELDS:
        out[k] = mem.get(k)
    return {"memory": out, "act": act}


_FIELDS = ("stage", "rpy", "disc_center", "disc_radius", "cubes", "buttons", "lamp_roi", "slots",
           "placed", "flash_color", "flash_step", "flash_votes", "attempts", "waits", "repairs",
           "inspections", "presses", "log")
