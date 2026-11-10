"""RememberColor9: remember the cue cube colour, wait for the 9-cube ring, then
move the open gripper onto the cube of the same colour and hold still.

Perception uses only the fixed front RGB-D camera (back-projected with K and
extrinsic). Control is one native step per decision with a velocity-aware
per-axis P controller fitted on the demonstrations.
"""

# Colour prototypes measured on the top faces of cubes in the demos (front cam).
COLORS = {
    "red": (220, 4, 4), "darkred": (161, 4, 4), "green": (4, 220, 4),
    "blue": (4, 4, 220), "yellow": (220, 220, 4), "orange": (220, 161, 4),
    "magenta": (220, 4, 220), "cyan": (4, 220, 220), "teal": (4, 161, 161),
}
CUBE_TOP_Z = 0.040          # measured top-face height of every cube (m)
CUBE_CENTER_Z = 0.020
FRONT_BIAS_XY = (0.002, 0.003)  # front-vs-wrist centroid difference in demos
MIN_PIX = 15                # pixels needed to accept a colour blob
# Measured one-step response (demos + R001): d_next ~= 0.28*0.1*a + 0.25*d_prev per axis.
# R001 showed that a high-gain velocity-compensating controller oscillates in x/y,
# so use a moderate P gain (K=1 -> target offset equals the error) with mild lead.
KP = 1.0
KP_XY = 1.3
KV_LEAD = 0.25
Z_SAFE = 0.055              # TCP height keeping fingertips above cube tops (0.040)
XY_DESCEND = 0.020          # xy error below which the TCP may go below Z_SAFE
XY_TOL = 0.010
Z_TOL = 0.012
Z_TOL_TOP = 0.005           # top touch: hold only once the fingertips reach the cube top
Z_AIM_BELOW = 0.015        # aim slightly below the goal height so the tail is not slow
Z_MIN = 0.015               # never aim the TCP lower (fingertips ~0.01 below TCP)
DRIFT_EXIT = 0.025          # leave hold only if pushed this far from the goal
STATIC_JV = 0.10            # rad/s max joint speed considered settled
# Touch stages: (name, dx, dy, z goal, settled steps before moving on, gripper)
# dy == "auto": neighbour-aware shift along the finger axis (world y) for open-finger straddle.
STAGES = [
    ("top", 0.0, 0.0, 0.049, 2, -1.0),          # closed fingertips resting on the cube top
    ("straddle", 0.0, "auto", 0.025, 3, 1.0),   # open fingers around the cube, TCP near centre
    ("nudge_py", 0.0, 0.020, 0.020, 3, 1.0),    # -y finger pressed on the cube face
    ("low", 0.0, 0.0, 0.015, 3, 1.0),
]
FINGER_OUTER = 0.056        # finger outer face from TCP along the finger axis (wrist depth)
FINGER_HALF_W = 0.012       # finger half width across the finger axis
CUBE_HALF = 0.020
MAX_SHIFT = 0.012           # keeps >=5 mm between inner pads (0.037) and target faces


def _straddle_shift(target, neighbors):
    """Shift along world y giving the open fingers clearance to neighbouring cubes.

    Returns (shift, clearance). Prefers the smallest shift with >= 12 mm clearance,
    otherwise the shift with the largest clearance (bounded by MAX_SHIFT)."""
    cands = []
    for k in range(-6, 7):
        d = 0.002 * k
        c = 1.0
        for nb in neighbors:
            u = nb[1] - target[1]
            w = nb[0] - target[0]
            if abs(w) > CUBE_HALF + FINGER_HALF_W + 0.01 or abs(u) > 0.15:
                continue
            if u >= 0:
                c = min(c, (u - CUBE_HALF) - (d + FINGER_OUTER))
            else:
                c = min(c, (-u - CUBE_HALF) - (FINGER_OUTER - d))
        cands.append((d, c))
    good = [x for x in cands if x[1] >= 0.012]
    if good:
        return min(good, key=lambda x: abs(x[0]))
    return max(cands, key=lambda x: (x[1], -abs(x[0])))


def _front_points(frame):
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    d = np.asarray(frame.front_depth, dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    world = (pc - E[:, 3]) @ E[:, :3]
    return world, d


def _detect(frame):
    """Colour -> [pixel count, x, y] of cube top faces seen by the front camera."""
    world, d = _front_points(frame)
    rgb = np.asarray(frame.front_rgb, dtype=float)
    sat = rgb.max(2) - rgb.min(2)
    z = world[..., 2]
    m = (d > 0) & (z > CUBE_TOP_Z - 0.006) & (z < CUBE_TOP_Z + 0.01) & (sat > 60)
    px = rgb[m]
    pw = world[m]
    out = {}
    if len(px) == 0:
        return out
    names = list(COLORS)
    P = np.array([COLORS[n] for n in names], dtype=float)
    dist = np.sqrt(((px[:, None, :] - P[None, :, :]) ** 2).sum(2))
    lab = dist.argmin(1)
    ok = dist.min(1) < 50
    for k in range(len(names)):
        s = ok & (lab == k)
        n = int(s.sum())
        if n >= MIN_PIX:
            xy = np.median(pw[s][:, :2], 0)
            out[names[k]] = [n, float(xy[0]) + FRONT_BIAS_XY[0], float(xy[1]) + FRONT_BIAS_XY[1]]
    return out


def _axis_cmd(err, vel, k):
    a = k * (err - KV_LEAD * vel) / 0.1
    return max(-1.0, min(1.0, a))


def perceive_decide(obs, goal, memory, last):
    """Track cue colour, cube appearance and touch stage; return one native step."""
    mem = dict(memory) if memory else {}
    cue_votes = dict(mem.get("cue_votes") or {})
    cue_color = mem.get("cue_color") or ""
    target = mem.get("target") or []
    target_n = int(mem.get("target_samples") or 0)
    appear_step = mem.get("appear_step")
    appear_step = -1 if appear_step is None else int(appear_step)
    stage = int(mem.get("stage") or 0)
    hold = int(mem.get("hold") or 0)
    stage_steps = int(mem.get("stage_steps") or 0)
    events = list(mem.get("events") or [])
    prev_tcp = mem.get("prev_tcp") or []
    phase = mem.get("phase") or "cue"
    neighbors = list(mem.get("neighbors") or [])

    frames = list(obs.frames)
    cur = frames[-1]
    step = int(cur.step)
    tcp = np.asarray(cur.tcp, dtype=float)

    # ---- perception over every new frame (cue may last a single frame) ----
    for fr in frames:
        if fr.phase != "exec" and fr.phase != "demo":
            continue
        det = _detect(fr)
        n_col = len(det)
        if n_col >= 5:
            if appear_step < 0:
                appear_step = int(fr.step)
                events.append("cubes@%d n=%d" % (int(fr.step), n_col))
            if cue_color and float(np.asarray(fr.tcp)[2]) > 0.2:
                neighbors = [[det[k][1], det[k][2]] for k in det if k != cue_color]
            if cue_color and cue_color in det and float(np.asarray(fr.tcp)[2]) > 0.2:
                c = det[cue_color]
                if target_n == 0:
                    target = [c[1], c[2], CUBE_CENTER_Z]
                else:
                    target = [(target[0] * target_n + c[1]) / (target_n + 1),
                              (target[1] * target_n + c[2]) / (target_n + 1), CUBE_CENTER_Z]
                target_n += 1
        elif n_col >= 1 and appear_step < 0:
            name = max(det, key=lambda k: det[k][0])
            c = det[name]
            if abs(c[1]) < 0.06 and abs(c[2]) < 0.06:
                cue_votes[name] = int(cue_votes.get(name, 0)) + 1
                cue_color = max(cue_votes, key=lambda k: cue_votes[k])
        if appear_step >= 0 and not cue_color and phase != "nocue":
            events.append("no cue seen before cubes")
            phase = "nocue"

    if appear_step >= 0 and not target and cue_color and phase != "lost":
        events.append("cue colour %s not found among cubes" % cue_color)
        phase = "lost"

    # ---- velocity estimate from measured TCP ----
    if len(frames) >= 2:
        vel = tcp[:3] - np.asarray(frames[-2].tcp, dtype=float)[:3]
    elif len(prev_tcp) == 3:
        vel = tcp[:3] - np.asarray(prev_tcp, dtype=float)
    else:
        vel = np.zeros(3)

    # joint speed (rad/s at 20 Hz) from consecutive measured joint samples
    joints = np.asarray(cur.joints, dtype=float)
    prev_j = mem.get("prev_joints") or []
    if len(frames) >= 2:
        jv = float(np.abs(joints - np.asarray(frames[-2].joints, dtype=float)).max() * 20.0)
    elif len(prev_j) == 7:
        jv = float(np.abs(joints - np.asarray(prev_j, dtype=float)).max() * 20.0)
    else:
        jv = 0.0
    settled = int(mem.get("settled") or 0)

    cmd = [0.0, 0.0, 0.0]
    grip = STAGES[min(stage, len(STAGES) - 1)][5]
    if not target:
        phase = phase if phase in ("nocue", "lost") else ("cue" if appear_step < 0 else "search")
    else:
        stage = min(stage, len(STAGES) - 1)
        name, ox, oy, gz, nhold, grip = STAGES[stage]
        if oy == "auto":
            oy, clear = _straddle_shift(target, neighbors)
        goal_p = np.array([target[0] + ox, target[1] + oy, gz])
        fingers = float(np.asarray(cur.fingers, dtype=float).mean())
        err = goal_p - tcp[:3]
        exy = float(np.hypot(err[0], err[1]))
        ezg = float(err[2])
        in_hold = phase.startswith("hold_")
        ztol = Z_TOL_TOP if name == "top" else Z_TOL
        stalled = exy < XY_TOL and abs(float(vel[2])) < 0.002 and float(tcp[2]) < Z_SAFE + 0.01 and stage_steps > 2
        near = exy < XY_TOL and (abs(ezg) < ztol or (name == "top" and stalled))
        if in_hold and (exy > DRIFT_EXIT or abs(ezg) > DRIFT_EXIT):
            in_hold = False
            events.append("drift exit %s @%d" % (name, step))
        if grip > 0 and fingers < 0.034 and float(tcp[2]) < Z_SAFE:
            phase = "open_" + name
            cmd = [0.0, 0.0, 0.0]
        elif near or in_hold:
            # stop commanding; let the arm come to rest (success seems to need a static arm)
            phase = "hold_" + name
            cmd = [0.0, 0.0, 0.0]
            hold += 1
            settled = settled + 1 if jv < STATIC_JV else 0
            if settled >= nhold and stage < len(STAGES) - 1:
                events.append("stage %s settled %d @%d no success off=(%.3f,%.3f,%.3f)" % (
                    name, settled, step, -err[0], -err[1], -err[2]))
                stage += 1
                hold = 0
                settled = 0
                stage_steps = 0
                phase = "move_next"
        else:
            phase = "move_" + name
            stage_steps += 1
            settled = 0
            # stay above neighbouring cubes until roughly over the target
            zgoal = max(gz - Z_AIM_BELOW, Z_MIN)
            if exy > XY_DESCEND:
                # do not descend below the safe height until over the goal; never lift
                zgoal = max(zgoal, min(Z_SAFE, float(tcp[2])))
            ez = zgoal - float(tcp[2])
            cmd = [_axis_cmd(err[0], vel[0], KP_XY), _axis_cmd(err[1], vel[1], KP_XY),
                   _axis_cmd(ez, vel[2], KP)]
            if stage_steps > 12 and stage < len(STAGES) - 1:
                events.append("stage %s timeout @%d err=%.3f" % (name, step, float(np.linalg.norm(err))))
                stage += 1
                hold = 0
                stage_steps = 0
    events = events[-12:]
    memory_out = {
        "phase": phase,
        "cue_color": cue_color,
        "cue_votes": cue_votes,
        "appear_step": appear_step,
        "target": target,
        "target_samples": target_n,
        "stage": stage,
        "hold": hold,
        "stage_steps": stage_steps,
        "prev_tcp": [float(v) for v in tcp[:3]],
        "prev_joints": [float(v) for v in joints],
        "settled": settled,
        "neighbors": neighbors,
        "joint_speed": jv,
        "events": events,
        "step": step,
    }
    return {"memory": memory_out,
            "act": {"tool": "act_step", "args": {"dx": cmd[0], "dy": cmd[1], "dz": cmd[2], "gripper": grip}}}


def act_step(robot, dx=0.0, dy=0.0, dz=0.0, gripper=1.0):
    """Send one native pd_ee_delta_pose step (no rotation); report measured TCP before/after."""
    before = np.asarray(robot.now().tcp, dtype=float)
    robot.step(float(dx), float(dy), float(dz), 0.0, 0.0, 0.0, float(gripper))
    after = np.asarray(robot.now().tcp, dtype=float)
    return {"cmd": [round(float(dx), 3), round(float(dy), 3), round(float(dz), 3)],
            "tcp_before": [round(float(v), 4) for v in before[:3]],
            "tcp_after": [round(float(v), 4) for v in after[:3]],
            "moved": round(float(np.linalg.norm(after[:3] - before[:3])), 4),
            "step": int(robot.now().step)}
