"""ShellGameTouch policy: remember the ball seen during the cue, then touch the cup that covers it.

Stages (memory.stage): 'cue' -> 'approach' -> 'press'.  All perception is from the fixed front
RGB-D camera deprojected with its calibration; no hidden state is used.
"""

CUP_TOP_DEFAULT = 0.108      # measured top height of every cup in all 25 demos (front depth)
CONTACT_Z_MARGIN = 0.017     # TCP below top+margin counts as having landed on the cup
CUP_MIN_PIXELS = 400         # cups: ~1300-1900 dark-red pixels; ball alone: 40-70
BALL_MIN_PIXELS = 8
MAX_CUE_FRAMES = 7          # demos: cups visible by frame 5 (frames_seen 6); then trust ball memory

# Linear state-feedback laws fitted (least squares) to the expert commands of all 25 assigned demos.
# Feature f = [tcp_x-bx, tcp_y-by, tcp_z-top, roll_dev, pitch, yaw, 1] with (bx,by)=remembered ball xy,
# roll_dev = (roll mod 2pi) - pi.  Rows give the 7 command components (dx..drz, gripper).
APPROACH_ROT = [
    [-0.312, 2.058, -0.887, -0.24, -0.138, -0.345, 0.076],
    [-4.608, 0.184, -2.807, 1.663, -2.499, 0.519, -0.03],
    [-0.098, -0.079, -0.691, 1.445, -1.271, -1.319, 0.288],
]
PRESS_LAW = [
    [-9.066, -0.677, 3.421, 0.174, -0.541, 0.525, -0.316],
    [3.206, -7.789, 0.759, 0.074, 0.728, -0.096, 0.053],
    [8.914, -1.567, -6.307, 1.31, -0.67, 0.674, -1.126],
    [-7.432, 4.849, 2.558, 1.401, -0.329, 0.42, 0.186],
    [-10.198, -3.379, 7.293, -0.179, 0.27, 0.54, -0.645],
    [-1.907, -2.07, 2.291, -0.546, -0.105, 0.115, 0.176],
    [-0.323, 0.238, 6.764, -0.934, 0.576, -0.5, 0.973],
]


def _masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    red = (r > 50) & (g < 50) & (b < 50) & (r > 2 * g + 20)
    ball = (r > 140) & (g < 15) & (b < 15)
    cup = red & np.logical_not(ball)
    return cup, ball


def _deproject(frame, mask):
    """World XYZ of the masked front-camera pixels with valid depth (N x 3)."""
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return np.zeros((0, 3))
    z = np.asarray(frame.front_depth, dtype=float)[vs, us] / 1000.0
    ok = z > 0
    vs, us, z = vs[ok], us[ok], z[ok]
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    p = np.stack([x, y, z], axis=1)
    return (p - E[:, 3]) @ E[:, :3]


def _scene(frame):
    """Ball and cup measurements from one front RGB-D frame."""
    cup_m, ball_m = _masks(np.asarray(frame.front_rgb))
    out = {"ball": None, "ball_px": int(ball_m.sum()), "cups": [], "cup_px": int(cup_m.sum()),
           "top": None}
    if out["ball_px"] >= BALL_MIN_PIXELS:
        w = _deproject(frame, ball_m)
        w = w[(w[:, 2] > -0.02) & (w[:, 2] < 0.12)] if len(w) else w
        if len(w) >= BALL_MIN_PIXELS // 2:
            out["ball"] = [float(np.median(w[:, 0])), float(np.median(w[:, 1]))]
    if out["cup_px"] >= CUP_MIN_PIXELS:
        w = _deproject(frame, cup_m)
        w = w[(w[:, 2] > 0.01) & (w[:, 2] < 0.2)] if len(w) else w
        if len(w) > 50:
            order = np.argsort(w[:, 1])
            w = w[order]
            gaps = np.nonzero(np.diff(w[:, 1]) > 0.04)[0]
            starts = [0] + [int(g) + 1 for g in gaps]
            ends = [int(g) + 1 for g in gaps] + [len(w)]
            cups = []
            tops = []
            for s, e in zip(starts, ends):
                c = w[s:e]
                if len(c) < 60:
                    continue
                zt = float(np.percentile(c[:, 2], 98))
                top = c[c[:, 2] > zt - 0.012]
                cups.append([float(np.median(top[:, 0])), float(np.median(top[:, 1])), zt])
                tops.append(zt)
            out["cups"] = cups
            if tops:
                out["top"] = float(np.median(tops))
    return out


def _feature(tcp, bx, by, top):
    roll_dev = (float(tcp[3]) % (2 * math.pi)) - math.pi
    return np.array([float(tcp[0]) - bx, float(tcp[1]) - by, float(tcp[2]) - top,
                     roll_dev, float(tcp[4]), float(tcp[5]), 1.0])


def _clip(v):
    return [float(min(1.0, max(-1.0, x))) for x in v]


def perceive_decide(obs, goal, memory, last):
    """Track cue -> approach -> press using the front camera; remember ball xy from the cue."""
    m = {}
    for k in ("stage", "ball_xy", "ball_obs", "cups", "target_xy", "cup_top", "frames_seen",
              "cue_frames", "contact_step", "notes"):
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "cue"
    if m["ball_obs"] is None:
        m["ball_obs"] = 0
    if m["frames_seen"] is None:
        m["frames_seen"] = 0
    if m["cue_frames"] is None:
        m["cue_frames"] = 0
    if m["notes"] is None:
        m["notes"] = ""

    frames = list(obs.frames)
    m["frames_seen"] = int(m["frames_seen"]) + len(frames)
    cur = frames[-1]
    # Update ball memory from every new frame where the ball is visible (cue phase).
    if m["stage"] == "cue":
        bsum = [0.0, 0.0]
        n = int(m["ball_obs"])
        if m["ball_xy"] is not None and n > 0:
            bsum = [m["ball_xy"][0] * n, m["ball_xy"][1] * n]
        for fr in frames:
            s = _scene(fr)
            if s["ball"] is not None and s["cup_px"] < CUP_MIN_PIXELS:
                bsum = [bsum[0] + s["ball"][0], bsum[1] + s["ball"][1]]
                n += 1
                m["cue_frames"] = int(m["cue_frames"]) + 1
        if n > 0:
            m["ball_xy"] = [bsum[0] / n, bsum[1] / n]
            m["ball_obs"] = n
        sc = _scene(cur)
        # Demos: cups appear after 1-5 cue frames; past MAX_CUE_FRAMES proceed on the ball memory alone.
        if sc["cup_px"] < CUP_MIN_PIXELS and (m["frames_seen"] < MAX_CUE_FRAMES or m["ball_xy"] is None):
            wait = max(1, MAX_CUE_FRAMES - int(m["frames_seen"])) if m["ball_xy"] is not None else 3
            return {"memory": m, "act": {"tool": "act_wait_for_cups", "args": {"max_steps": int(wait)}}}
        # Cups now visible: commit to the target cup.
        m["cups"] = [[c[0], c[1]] for c in sc["cups"]]
        top = sc["top"] if sc["top"] is not None else CUP_TOP_DEFAULT
        if abs(top - CUP_TOP_DEFAULT) > 0.02:
            top = CUP_TOP_DEFAULT
        m["cup_top"] = float(top)
        if m["ball_xy"] is not None:
            bx, by = m["ball_xy"]
            if sc["cups"]:
                k = int(np.argmin([math.hypot(c[0] - bx, c[1] - by) for c in sc["cups"]]))
                dist = math.hypot(sc["cups"][k][0] - bx, sc["cups"][k][1] - by)
                m["notes"] = "ball-cup dist %.3f, %d cups" % (dist, len(sc["cups"]))
                if dist > 0.06:
                    # Ball memory disagrees with every cup: trust the nearest cup centre instead.
                    bx, by = sc["cups"][k][0] - 0.005, sc["cups"][k][1] - 0.007
            m["target_xy"] = [float(bx), float(by)]
        elif sc["cups"]:
            cs = sorted(sc["cups"], key=lambda c: c[1])
            c = cs[len(cs) // 2]
            m["target_xy"] = [c[0] - 0.005, c[1] - 0.007]
            m["notes"] = "ball never seen; guessing middle cup"
        else:
            m["target_xy"] = [0.0, 0.0]
            m["notes"] = "no ball and no cup clusters; blind guess"
        m["stage"] = "approach"

    if isinstance(last, str) and last.startswith("act_approach") and '"landed": true' in last:
        if m["stage"] == "approach":
            m["stage"] = "press"
            m["contact_step"] = int(m["frames_seen"])

    tx, ty = m["target_xy"]
    top = m["cup_top"] if m["cup_top"] is not None else CUP_TOP_DEFAULT
    tcp = np.asarray(cur.tcp, dtype=float)
    if m["stage"] == "approach":
        if tcp[2] < top + CONTACT_Z_MARGIN and math.hypot(tcp[0] - tx, tcp[1] - ty) < 0.05:
            m["stage"] = "press"
            m["contact_step"] = int(m["frames_seen"])
    if m["stage"] == "approach":
        return {"memory": m, "act": {"tool": "act_approach",
                                     "args": {"bx": tx, "by": ty, "top": top, "max_steps": 14}}}
    return {"memory": m, "act": {"tool": "act_press",
                                 "args": {"bx": tx, "by": ty, "top": top, "max_steps": 8}}}


def act_wait_for_cups(robot, max_steps=10):
    """Send zero commands (the task suppresses commands during the cue) until cups appear."""
    n = 0
    seen = False
    for _ in range(int(max_steps)):
        sc = _scene(robot.now())
        if sc["cup_px"] >= CUP_MIN_PIXELS:
            seen = True
            break
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    if not seen:
        seen = _scene(robot.now())["cup_px"] >= CUP_MIN_PIXELS
    return {"steps": n, "cups_visible": bool(seen)}


def act_approach(robot, bx, by, top, max_steps=14):
    """Move above the remembered ball xy and descend until the TCP rests on the cup top."""
    bx, by, top = float(bx), float(by), float(top)
    gx, gy = bx + 0.012, by          # expert landings: TCP ~1-2 cm on the camera (+x) side
    steps = 0
    landed = False
    prev_z = None
    for _ in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        exy = math.hypot(gx - tcp[0], gy - tcp[1])
        if tcp[2] < top + CONTACT_Z_MARGIN and exy < 0.05:
            landed = True
            break
        if prev_z is not None and tcp[2] < top + 0.03 and abs(prev_z - tcp[2]) < 0.003 and exy < 0.05:
            landed = True
            break
        dx = 12.0 * (gx - tcp[0])
        dy = 12.0 * (gy - tcp[1])
        # Descend at full speed unless low and still far from the cup axis (avoid hitting the rim).
        height = tcp[2] - top
        # (expert landings had up to 3.6 cm xy error and then slid on the top; cup radius ~4 cm)
        if exy > 0.045 and height < 0.05:
            dz = 0.0
        elif exy > 0.08 and height < 0.10:
            dz = -0.3
        else:
            dz = -1.0
        f = _feature(tcp, bx, by, top)
        rot = [float(np.dot(row, f)) for row in APPROACH_ROT]
        cmd = _clip([dx, dy, dz] + rot)
        prev_z = float(tcp[2])
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], 1.0)
        steps += 1
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": steps, "landed": bool(landed or tcp[2] < top + CONTACT_Z_MARGIN),
            "tcp": [round(float(v), 4) for v in tcp[:3]],
            "rel": [round(float(tcp[0] - bx), 4), round(float(tcp[1] - by), 4), round(float(tcp[2] - top), 4)]}


def act_press(robot, bx, by, top, max_steps=8):
    """Keep the fingers on the target cup top with the expert-fitted press law (push, tilt, hold open)."""
    bx, by, top = float(bx), float(by), float(top)
    steps = 0
    log = []
    recover = False
    recoveries = 0
    for _ in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        f = _feature(tcp, bx, by, top)
        cmd = [float(np.dot(row, f)) for row in PRESS_LAW]
        if abs(f[0]) > 0.06 or abs(f[1]) > 0.06 or f[2] < -0.045:
            # Off the cup (far from its axis, or slid >4.5 cm below the top; demos never exceeded 3.6 cm):
            # rise, recentre above the cup, then resume pressing.
            recover = True
        if recover:
            gx, gy = bx + 0.012, by
            if f[2] < 0.02:
                cmd = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]     # rise first, so the cup is not swept
                recoveries += 1
            elif math.hypot(gx - tcp[0], gy - tcp[1]) > 0.015:
                cmd = [12.0 * (gx - tcp[0]), 12.0 * (gy - tcp[1]), 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                recover = False
        if tcp[4] < -0.6 and cmd[4] < 0:
            cmd[4] = 0.0               # do not tilt beyond the range seen in demos (pitch >= -0.51)
        cmd = _clip(cmd)
        robot.step(*cmd)
        steps += 1
        if len(log) < 8:
            log.append([round(float(v), 3) for v in f[:3]])
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": steps, "rel_log": log, "recovery_steps": recoveries,
            "rel": [round(float(tcp[0] - bx), 4), round(float(tcp[1] - by), 4), round(float(tcp[2] - top), 4)],
            "pitch": round(float(tcp[4]), 3)}
