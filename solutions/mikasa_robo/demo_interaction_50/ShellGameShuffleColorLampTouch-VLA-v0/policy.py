"""ShellGameShuffleColorLampTouch: remember ball colours under 3 cups, track the
cups through the shuffle with the fixed front RGB-D camera, read the lamp colour
when it lights up (end of the cue phase), then touch the cup hiding the ball of
the lamp's colour."""

COLOR_NAMES = ["red", "green", "blue"]
CUP_TOP_Z = 0.098      # front-depth points above this belong to a resting cup top (top ~0.105-0.108 m)
SAFE_Z = 0.16          # transit height above cup tops
TOUCH_Z = 0.04         # descent goal for the TCP over the target cup top centre
GAIN = 2.0             # command = GAIN * error / 0.1 m, clipped to [-1, 1]


def _world(frame, cam="front"):
    """Back-project a depth image to world XYZ (H x W x 3)."""
    depth = getattr(frame, cam + "_depth").astype(float) / 1000.0
    K = getattr(frame, cam + "_K")
    E = getattr(frame, cam + "_extrinsic")
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * depth
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (P - t) @ R


def _color_masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return [
        (r > 150) & (g < 90) & (b < 90),
        (g > 120) & (r < 90) & (b < 90),
        (b > 150) & (r < 90) & (g < 90),
    ]


def _cup_mask(rgb, Wp):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (r > 40) & (r > 2 * g) & (r > 2 * b) & (r < 200) & (Wp[..., 2] > 0.004) & (Wp[..., 2] < 0.25)


def _scene_initial(frame):
    """Balls (colour, world xy) and the lamp (world xyz, pixel bbox) in the reset frame."""
    rgb = frame.front_rgb
    Wp = _world(frame)
    balls = []
    for k, m in enumerate(_color_masks(rgb)):
        m = m & (Wp[..., 2] > 0.01) & (Wp[..., 2] < 0.08)
        n = int(m.sum())
        if n >= 8:
            c = Wp[m].mean(0)
            balls.append({"color": COLOR_NAMES[k], "xy": [float(c[0]), float(c[1])], "n": n})
    balls = sorted(balls, key=lambda d: d["xy"][1])
    lamp = None
    if balls:
        bx = float(np.mean([d["xy"][0] for d in balls]))
        white = (rgb.min(-1) > 200) & (Wp[..., 2] > 0.08) & (Wp[..., 2] < 0.3) & (Wp[..., 0] > bx + 0.12)
        if int(white.sum()) >= 10:
            ys, xs = np.nonzero(white)
            c = Wp[white].mean(0)
            lamp = {"xyz": [float(c[0]), float(c[1]), float(c[2])],
                    "bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]}
    return balls, lamp


def _lamp_color(rgb, bbox):
    """Return (colour or None, counts) of saturated lamp pixels inside the lamp bbox."""
    y0, y1, x0, x1 = bbox
    y0 = max(0, y0 - 4)
    x0 = max(0, x0 - 4)
    y1 = min(255, y1 + 4)
    x1 = min(255, x1 + 4)
    sub = rgb[y0:y1 + 1, x0:x1 + 1]
    counts = [int(m.sum()) for m in _color_masks(sub)]
    k = int(np.argmax(counts))
    if counts[k] >= 20:
        return COLOR_NAMES[k], counts
    return None, counts


def _track(frames, init_xy):
    """Track the three cups (top-down xy) through frames with velocity-predicted k-means."""
    hist = [np.array(init_xy, dtype=float), np.array(init_xy, dtype=float)]
    minpts = []
    for fr in frames:
        Wp = _world(fr)
        P = Wp[_cup_mask(fr.front_rgb, Wp)][:, :2]
        c = hist[-1] + (hist[-1] - hist[-2])
        if len(P) < 30:
            hist.append(hist[-1].copy())
            minpts.append(0)
            continue
        a = None
        for _ in range(6):
            d = ((P[:, None, :] - c[None]) ** 2).sum(-1)
            a = d.argmin(1)
            c = np.array([P[a == k].mean(0) if (a == k).sum() > 5 else c[k] for k in range(3)])
        hist.append(c)
        minpts.append(int(min((a == k).sum() for k in range(3))))
    return hist[-1], minpts


def _cup_tops(frame, centres):
    """Refine each tracked cup centre to the centre of its flat top (depth points above CUP_TOP_Z)."""
    Wp = _world(frame)
    P = Wp[_cup_mask(frame.front_rgb, Wp)]
    tops = []
    for k in range(len(centres)):
        near = np.linalg.norm(P[:, :2] - np.array(centres[k])[None], axis=1) < 0.07
        Q = P[near & (P[:, 2] > CUP_TOP_Z) & (P[:, 2] < 0.13)]
        if len(Q) >= 15:
            tops.append([float(Q[:, 0].mean()), float(Q[:, 1].mean()), float(np.median(Q[:, 2]))])
        else:
            tops.append([float(centres[k][0]), float(centres[k][1]), 0.106])
    return tops


def _goal_color(goal):
    g = (goal or "").lower()
    for c in COLOR_NAMES:
        if c in g:
            return c
    return ""


def _mem(memory, **upd):
    base = {
        "stage": "init",
        "ball_colors": [],
        "ball_xy": [],
        "lamp_xyz": [],
        "lamp_bbox": [],
        "lamp_color": "",
        "cup_xy": [],
        "target_xy": [],
        "target_index": -1,
        "steps_used": 0,
        "touch_tries": 0,
        "top_z": 0.106,
        "notes": "",
    }
    for k in base:
        if k in memory and memory[k] is not None:
            base[k] = memory[k]
    base.update(upd)
    return base


def perceive_decide(obs, goal, memory, last):
    """Stage machine: init -> wait for lamp -> track cups & pick target -> touch."""
    m = _mem(memory)
    frames = list(obs.frames)
    cur = obs.current
    stage = m["stage"]
    if stage == "init":
        balls, lamp = _scene_initial(frames[0])
        if len(balls) == 3:
            colors = [b["color"] for b in balls]
            xy = [b["xy"] for b in balls]
            note = "balls seen at reset"
        else:
            # fallback: cluster cups by y order; demos always show red, green, blue left->right
            Wp = _world(cur)
            P = Wp[_cup_mask(cur.front_rgb, Wp)][:, :2]
            if len(P) >= 30:
                qs = np.quantile(P[:, 1], [1 / 6.0, 0.5, 5 / 6.0])
                xy = [[float(np.median(P[:, 0])), float(q)] for q in qs]
            else:
                xy = [[0.0, -0.2], [0.0, 0.0], [0.0, 0.2]]
            colors = list(COLOR_NAMES)
            note = "balls not all seen (%d); fallback order" % len(balls)
        if lamp is None:
            bx = float(np.mean([p[0] for p in xy]))
            by = float(np.mean([p[1] for p in xy]))
            lamp_xyz = [bx + 0.235, by, 0.16]
            bbox = []
            note += "; lamp not found"
        else:
            lamp_xyz = lamp["xyz"]
            bbox = lamp["bbox"]
        m = _mem(m, stage="waiting", ball_colors=colors, ball_xy=xy, lamp_xyz=lamp_xyz,
                 lamp_bbox=bbox, steps_used=int(cur.step), notes=note)
        return {"memory": m, "act": {"tool": "act_wait_for_lamp", "args": {"lamp_bbox": bbox, "max_steps": 50}}}

    if stage == "waiting":
        color = ""
        if m["lamp_bbox"]:
            c, counts = _lamp_color(cur.front_rgb, m["lamp_bbox"])
            color = c or ""
        if not color:
            color = _goal_color(goal)
        if not color:
            # lamp not seen yet: keep waiting
            m = _mem(m, steps_used=int(cur.step), notes=m["notes"] + "; lamp still off")
            return {"memory": m, "act": {"tool": "act_wait_for_lamp", "args": {"lamp_bbox": m["lamp_bbox"], "max_steps": 50}}}
        # track over all frames observed during the wait (all but the last one, cups are at rest by then)
        track_frames = frames[:-1] if len(frames) > 1 else frames
        final, minpts = _track(track_frames, m["ball_xy"])
        tops = _cup_tops(cur, [list(map(float, p)) for p in final])
        idx = m["ball_colors"].index(color) if color in m["ball_colors"] else 0
        tgt = tops[idx][:2]
        top_z = tops[idx][2]
        note = m["notes"] + "; lamp %s at step %d; min cluster pts %d" % (color, int(cur.step), min(minpts) if minpts else -1)
        m = _mem(m, stage="touch", lamp_color=color, cup_xy=[t[:2] for t in tops], target_xy=tgt, target_index=idx,
                 top_z=top_z,
                 steps_used=int(cur.step), notes=note)
        return {"memory": m, "act": {"tool": "act_touch", "args": {"x": tgt[0], "y": tgt[1], "top_z": top_z, "max_steps": 60}}}

    # touch stage repeated (previous touch ended without the episode ending)
    tries = int(m["touch_tries"]) + 1
    tgt = m["target_xy"] or [float(cur.tcp[0]), float(cur.tcp[1])]
    m = _mem(m, touch_tries=tries, steps_used=int(cur.step), notes=m["notes"] + "; retouch %d" % tries)
    return {"memory": m, "act": {"tool": "act_touch", "args": {"x": tgt[0], "y": tgt[1], "top_z": float(m["top_z"]), "max_steps": 60, "press_dz": -0.2}}}


def act_wait_for_lamp(robot, lamp_bbox, max_steps=50):
    """Send zero arm commands (gripper closed) one native step at a time until the
    lamp in lamp_bbox lights up (red/green/blue) or max_steps pass. Returns the step
    count, lamp colour and pixel counts."""
    n = 0
    color = None
    counts = []
    while n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
        fr = robot.now()
        if lamp_bbox:
            color, counts = _lamp_color(fr.front_rgb, lamp_bbox)
            if color:
                break
    return {"waited_steps": n, "lamp_color": color, "counts": counts, "step": int(robot.now().step)}


def _pd_cmd(err, vel, kp=2.0, kd=0.5):
    return [max(-1.0, min(1.0, (kp * e - kd * v) / 0.1)) for e, v in zip(err, vel)]


def act_touch(robot, x, y, top_z=0.106, z_goal=0.09, max_steps=60, lift_first=False, hover_z=0.15,
              press_dz=-0.1, press_steps=8, hold_steps=8, descend_speed=0.6):
    """Touch the cup whose top centre is (x, y) and then keep the arm still.
    top_z is the perceived cup-top height (contact expected just above it).
    1) [lift] -> hover above (x, y, hover_z) (PD on measured TCP, tol 18 mm xy / 25 mm z);
    2) descend with bounded speed (xy still corrected) until contact (TCP z stops falling
       near top_z, or TCP within 13 mm of top_z = fingertips on the top) or
       z_goal is reached;
    3) press: zero xy/rotation, small constant downward offset press_dz for press_steps
       (arm blocked by the cup -> static while touching);
    4) hold: all-zero command for hold_steps.
    Hypothesis under test: success = touching the target cup while the arm is static.
    Returns events, joint-speed trace and TCP trace relative to (x, y)."""
    events = []
    trace = []
    qtrace = []
    n = 0
    phase = "lift" if lift_first else "hover"
    prev_q = np.array(robot.now().joints, dtype=float)
    vel = np.zeros(3)
    contact_n = 0
    n_desc = 0
    k_phase = 0
    while n < max_steps:
        fr = robot.now()
        p = np.array(fr.tcp[:3], dtype=float)
        if phase == "lift":
            err = np.array([0.0, 0.0, hover_z + 0.02 - p[2]])
            if err[2] < 0.01:
                phase = "hover"
                events.append(["hover", n])
                continue
            cmd = _pd_cmd(err, vel)
        elif phase == "hover":
            err = np.array([x - p[0], y - p[1], hover_z - p[2]])
            if math.hypot(err[0], err[1]) < 0.018 and abs(err[2]) < 0.025:
                phase = "descend"
                events.append(["descend", n, [round(float(v), 3) for v in p]])
                continue
            if p[2] < hover_z - 0.03 and math.hypot(err[0], err[1]) > 0.04:
                err[0] *= 0.3
                err[1] *= 0.3
            cmd = _pd_cmd(err, vel)
        elif phase == "descend":
            err = np.array([x - p[0], y - p[1], z_goal - p[2]])
            cmd = _pd_cmd(err, vel)
            cmd[2] = max(-descend_speed, cmd[2])
            n_desc += 1
        elif phase == "press":
            cmd = [0.0, 0.0, press_dz]
            k_phase += 1
            if k_phase > press_steps:
                phase = "hold"
                k_phase = 0
                events.append(["hold", n])
                continue
        else:
            cmd = [0.0, 0.0, 0.0]
            k_phase += 1
            if k_phase > hold_steps:
                break
        robot.step(cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, -1.0)
        n += 1
        now = robot.now()
        q = np.array(now.tcp[:3], dtype=float)
        jq = np.array(now.joints, dtype=float)
        vel = q - p
        qv = float(np.abs(jq - prev_q).max())
        prev_q = jq
        trace.append([round(float(q[0] - x), 3), round(float(q[1] - y), 3), round(float(q[2]), 3)])
        qtrace.append(round(qv, 4))
        if phase == "descend":
            if q[2] <= z_goal + 0.005:
                phase = "press"
                k_phase = 0
                events.append(["press(z_goal)", n, round(float(q[2]), 3)])
            elif (vel[2] > -0.003 and q[2] < top_z + 0.025 and n_desc >= 2) or q[2] < top_z + 0.013:
                contact_n += 1
                if contact_n >= 1:
                    phase = "press"
                    k_phase = 0
                    events.append(["press(contact)", n, round(float(q[2]), 3)])
            else:
                contact_n = 0
    end = robot.now()
    return {"steps": n, "phase": phase, "events": events, "end_tcp": [round(float(v), 4) for v in end.tcp],
            "trace_rel": trace[-16:], "joint_step": qtrace[-16:], "step": int(end.step)}
