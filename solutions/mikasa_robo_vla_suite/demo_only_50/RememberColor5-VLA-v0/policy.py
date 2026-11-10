"""RememberColor5: remember the cue cube colour, then touch the same-coloured cube.

Phases observed in all 50 demonstrations (front camera, fixed):
  cue    : exactly one saturated cube on the table (1-5 frames from reset)
  delay  : empty table (1-5 frames); robot commands are replaced by zero
  answer : five cubes (red, yellow, green, blue, magenta); commands act from the
           first frame on which the cubes are visible
Policy: record the cue colour while one cube is visible, localise the cube of
that colour in 3D (front RGB-D, wrist RGB-D fallback) on the first answer
frame, then servo the closed gripper onto the cube top and keep pressing.
Fallback (no success after pressing): open gripper, straddle the cube and push
a finger against its side, as the experts did.
"""

HUES = (("red", 0.0), ("yellow", 60.0), ("green", 120.0), ("cyan", 180.0),
        ("blue", 240.0), ("magenta", 300.0), ("red", 360.0))
SAT_MIN = 0.85
VAL_MIN = 0.25
MIN_PIX = 6
STEP_LIMIT = 25
PRESS_DEPTH = 0.012     # TCP goal below the cube top while pressing (m); ~0.3 command at contact
HOVER_CLEAR = 0.02      # TCP clearance above the cube top before final descent
ALIGN_TOL = 0.006       # xy error (m) below which the final descent starts
GAIN = 15.0             # command units per metre of error
ROT_GAIN = 5.0
PRESS_PATIENCE = 3      # blocked press steps before trying the straddle fallback


def _hue_name(h):
    best, bd = "red", 1e9
    for name, c in HUES:
        d = abs(h - c)
        if d < bd:
            best, bd = name, d
    return best


def _color_pixels(rgb):
    """Saturated-colour pixels grouped by colour name: {name: (ys, xs)}."""
    f = np.asarray(rgb).astype(np.float32) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    s = (mx - mn) / (mx + 1e-6)
    mask = (s > SAT_MIN) & (mx > VAL_MIN)
    ys, xs = np.nonzero(mask)
    out = {}
    if len(ys) == 0:
        return out
    p = f[ys, xs]
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    pmx = mx[ys, xs]
    d = pmx - mn[ys, xs] + 1e-6
    h = np.where(pmx == r, np.mod((g - b) / d, 6.0),
                 np.where(pmx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    names = [_hue_name(float(v)) for v in h]
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(n, []).append(i)
    for n, idx in groups.items():
        idx = np.array(idx)
        if len(idx) < MIN_PIX:
            continue
        cy, cx = np.median(ys[idx]), np.median(xs[idx])
        keep = idx[(np.abs(ys[idx] - cy) < 25) & (np.abs(xs[idx] - cx) < 25)]
        if len(keep) >= MIN_PIX:
            out[n] = (ys[keep], xs[keep])
    return out


def _deproject(K, E, xs, ys, depth_mm):
    z = depth_mm.astype(np.float64) / 1000.0
    K = np.asarray(K, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    x = (xs + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (ys + 0.5 - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], axis=-1)
    return (pc - E[:, 3]) @ E[:, :3]


def _cube_xyz(frame, cam, ys, xs):
    """World [x, y, z_top] of a cube from its pixels; None when depth is missing."""
    dep = np.asarray(getattr(frame, cam + "_depth"))
    d = dep[ys, xs]
    ok = d > 0
    if ok.sum() < 3:
        return None
    P = _deproject(getattr(frame, cam + "_K"), getattr(frame, cam + "_extrinsic"),
                   xs[ok], ys[ok], d[ok])
    zt = float(P[:, 2].max())
    top = P[P[:, 2] > zt - 0.005]
    return [float(top[:, 0].mean()), float(top[:, 1].mean()), float(np.median(top[:, 2]))]


def _rotm(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    a = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-6:
        return 0.5 * v
    sa = math.sin(a)
    if abs(sa) < 1e-6:
        return np.zeros(3)
    return v * (a / (2.0 * sa))


def _rot_cmd(rpy_now, rpy_home):
    """Rotation command that drifts the gripper back to its initial orientation."""
    if rpy_home is None:
        return [0.0, 0.0, 0.0]
    e = _rotvec(_rotm(rpy_home) @ _rotm(rpy_now).T)
    r = -ROT_GAIN * e
    n = float(np.linalg.norm(r))
    if n > 0.5:
        r = r * (0.5 / n)
    return [float(v) for v in r]


def _clip(v, lo=-1.0, hi=1.0):
    return float(max(lo, min(hi, v)))


def _fresh_memory(memory):
    keys = ("stage", "cue_color", "cue_step", "cue_pixels", "target", "target_source",
            "home_rpy", "press_steps", "blocked_steps", "fallback_steps", "last_z",
            "colors_seen", "note")
    m = {}
    for k in keys:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "wait"
    for k in ("press_steps", "blocked_steps", "fallback_steps"):
        if m[k] is None:
            m[k] = 0
    if m["colors_seen"] is None:
        m["colors_seen"] = []
    return m


def _observe_frames(obs, m):
    """Update cue / answer-cube memory from every frame since the last action.

    Front camera first; the wrist camera is used only when the front view shows
    no coloured cube at all (e.g. occlusion)."""
    answer = None
    for fr in obs.frames:
        cols = _color_pixels(fr.front_rgb)
        cam = "front"
        if len(cols) == 0:
            wc = _color_pixels(fr.wrist_rgb)
            if len(wc) > 0:
                cols, cam = wc, "wrist"
        names = sorted(cols.keys())
        if len(names) == 1 and m["target"] is None:
            n = names[0]
            npx = int(len(cols[n][0]))
            # the cue is the single cube shown before the answer cubes
            if m["cue_color"] is not None and m["cue_color"] != n:
                m["note"] = "cue colour changed %s->%s" % (m["cue_color"], n)
            if m["cue_color"] != n or npx >= (m["cue_pixels"] or 0) or cam == "front":
                m["cue_color"] = n
                m["cue_pixels"] = npx
                m["cue_step"] = int(fr.step)
        if len(names) >= 3:
            answer = (fr, cols, cam)
            if m["target"] is None:
                break  # first answer frame: arm still clear of the cubes
    return answer


def _locate_target(answer, m):
    fr, cols, cam = answer
    m["colors_seen"] = sorted(cols.keys())
    want = m["cue_color"]
    if want is None:
        want = sorted(cols.keys(), key=lambda n: -len(cols[n][0]))[0]
        m["note"] = "cue never seen; guessing %s" % want
    for c in ("front", "wrist"):
        cc = cols if c == cam else _color_pixels(getattr(fr, c + "_rgb"))
        if want in cc:
            p = _cube_xyz(fr, c, cc[want][0], cc[want][1])
            if p is not None:
                m["target"] = p
                m["target_source"] = c
                return
    m["note"] = "target colour %s not localised" % want


def perceive_decide(obs, goal, memory, last):
    """Track cue colour and phase, localise the matching cube, and choose one native step."""
    m = _fresh_memory(memory if isinstance(memory, dict) else {})
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=np.float64)
    if m["home_rpy"] is None:
        m["home_rpy"] = [float(v) for v in tcp[3:6]]
    answer = _observe_frames(obs, m)
    if m["target"] is None and answer is not None:
        _locate_target(answer, m)
        if m["target"] is not None:
            m["stage"] = "reach"
    rot = _rot_cmd(tcp[3:6], m["home_rpy"])
    if m["target"] is None:
        # cue / delay phase: commands are zeroed by the task anyway
        if answer is None:
            m["note"] = "waiting (cue=%s)" % m["cue_color"]
        return {"memory": m, "act": {"tool": "act_step", "args": {
            "dx": 0.0, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0,
            "gripper": -1.0}}}

    tx, ty, ztop = float(m["target"][0]), float(m["target"][1]), float(m["target"][2])
    z = float(tcp[2])
    ex, ey = tx - float(tcp[0]), ty - float(tcp[1])
    dxy = math.hypot(ex, ey)
    last_z = m["last_z"]
    grip = -1.0

    if m["stage"] in ("reach", "press"):
        if m["stage"] == "reach" and dxy < ALIGN_TOL:
            m["stage"] = "press"
        if m["stage"] == "press" and dxy > 0.015:
            m["stage"] = "reach"
        if m["stage"] == "press":
            zg = ztop - PRESS_DEPTH
            m["press_steps"] = int(m["press_steps"]) + 1
            # stalled while pressing: on the cube top (expected) or, if it
            # missed, hovering at the press goal; both count toward the fallback
            if last_z is not None and abs(z - float(last_z)) < 0.002:
                m["blocked_steps"] = int(m["blocked_steps"]) + 1
            if int(m["blocked_steps"]) >= PRESS_PATIENCE:
                m["stage"] = "straddle"
        else:
            # descend along a cone: TCP stays >= HOVER_CLEAR + dxy above the top
            zg = ztop + HOVER_CLEAR + 1.0 * dxy
            if zg > z and z > ztop + HOVER_CLEAR:
                zg = z  # never climb while approaching from above
    if m["stage"] == "straddle":
        # expert-like: open fingers around the cube, low TCP, then sweep along
        # the finger axis so one finger pushes against the cube side
        grip = 1.0
        n = int(m["fallback_steps"])
        m["fallback_steps"] = n + 1
        Rm = _rotm(tcp[3:6])
        fy = Rm[:2, 1]
        nrm = float(np.linalg.norm(fy)) or 1.0
        fy = fy / nrm
        if n < 2:
            zg = ztop + 0.04
            off = 0.0
        else:
            zg = 0.015
            off = 0.03 if (n // 3) % 2 == 0 else -0.03
            if z > ztop + 0.005:
                off = 0.0
        ex = tx + off * float(fy[0]) - float(tcp[0])
        ey = ty + off * float(fy[1]) - float(tcp[1])
    ux = _clip(GAIN * ex)
    uy = _clip(GAIN * ey)
    uz = _clip(GAIN * (zg - z))
    m["last_z"] = z
    m["note"] = "%s dxy=%.4f z=%.3f zg=%.3f step=%d" % (m["stage"], dxy, z, zg, int(cur.step))
    return {"memory": m, "act": {"tool": "act_step", "args": {
        "dx": ux, "dy": uy, "dz": uz, "drx": rot[0], "dry": rot[1], "drz": rot[2],
        "gripper": grip}}}


def act_step(robot, dx, dy, dz, drx, dry, drz, gripper):
    """Send one native pd_ee_delta_pose step; returns the measured TCP before/after."""
    before = [float(v) for v in robot.now().tcp]
    robot.step(float(dx), float(dy), float(dz), float(drx), float(dry), float(drz), float(gripper))
    now = robot.now()
    after = [float(v) for v in now.tcp]
    return {"step": int(now.step), "tcp_before": before[:3], "tcp_after": after[:3],
            "moved_m": float(np.linalg.norm(np.array(after[:3]) - np.array(before[:3]))),
            "fingers": [float(v) for v in now.fingers]}
