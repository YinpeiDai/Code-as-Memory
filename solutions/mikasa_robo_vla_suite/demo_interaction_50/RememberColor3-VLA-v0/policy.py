"""RememberColor3: observe the cue cube's colour, wait, then touch the cube of the same colour.

Perception: front RGB-D -> world points (front_K, front_extrinsic); saturated pixels
above the table (z in 6..70 mm) are clustered into cubes; each cube = top-face xy and
median RGB.  Phases are inferred from the number of cubes visible (1 = cue, 0 = delay,
>=2 = choice).  Control: one native step per decision, momentum-aware Cartesian
controller toward the chosen cube, gripper open, orientation held pointing down.
"""

EXPLORE_OFFSET = None   # diagnostic offset hold (R005-R012); disabled for the final policy
EXPLORE_HOLD = 3
EXPLORE_MAX_APPEAR = 7
CUBE_CENTER_Z = 0.02
FALLBACK_AFTER = 3     # settle decisions before lowering the touch pose
FALLBACK_DZ = 0.01
FALLBACK_TOL = 0.006
SETTLE_XY = 0.018      # lateral part of the settle gate
TABLE_Z_MIN = 0.006
TABLE_Z_MAX = 0.07
SAT_MIN = 0.7
CLUSTER_R = 0.035
CUBE_HALF = 0.02
FUNNEL_K = 1.5         # extra height (m) per metre of lateral error
FUNNEL_FREE = 0.015    # lateral error allowed at the touch height
FUNNEL_MAX = 0.04
NEAR_DIST = 0.03       # within this 3-D error use the gentler gains below
ALPHA_NEAR = 0.5
CMD_NEAR = 0.6         # max |command| per axis when near the goal
FINAL_Z = 0.04         # TCP height at the touch pose: 2 cm above cube centre (R005: static 3.7 cm above succeeded)
MOM = 0.35             # fitted per-step velocity persistence (demos)
GAIN_B = 0.025         # fitted metres of motion per unit command per step
VMAX = 0.045           # m/step cruise request (saturated motion is ~0.04 m/step)
DEC = 0.012            # m/step^2 planned deceleration (R002/R003: abrupt brakes swing TCP x by ~1.3 cm)
ALPHA = 0.65           # desired fraction of remaining error per step
SETTLE_TOL = 0.018     # within this 3-D error: send zero arm command (brake, let joints come to rest)


def _world_points(depth, K, E):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = np.asarray(E[:, :3], float)
    t = np.asarray(E[:, 3], float)
    pw = (pc - t) @ R
    return pw.reshape(h, w, 3), z > 0


def _cubes(frame):
    """Saturated blobs above the table in the front camera: list of dicts."""
    P, valid = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    im = frame.front_rgb.astype(float)
    mx = im.max(-1)
    mn = im.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    m = (valid & (P[..., 2] > TABLE_Z_MIN) & (P[..., 2] < TABLE_Z_MAX)
         & (np.abs(P[..., 0]) < 0.6) & (np.abs(P[..., 1]) < 0.6)
         & (sat > SAT_MIN) & (mx > 80))
    pts = P[m]
    cols = im[m]
    out = []
    remaining = np.ones(len(pts), bool)
    guard = 0
    while remaining.any() and guard < 20:
        guard += 1
        i = int(np.flatnonzero(remaining)[0])
        c = pts[i, :2]
        for _ in range(5):
            sel = remaining & (np.linalg.norm(pts[:, :2] - c, axis=1) < CLUSTER_R)
            if not sel.any():
                break
            c = pts[sel, :2].mean(0)
        sel = remaining & (np.linalg.norm(pts[:, :2] - c, axis=1) < CLUSTER_R)
        sel[i] = True
        remaining &= ~sel
        if sel.sum() < 8:
            continue
        zt = pts[sel, 2].max()
        top = sel & (pts[:, 2] > zt - 0.008)
        xy = pts[top, :2].mean(0)
        out.append({"xy": [float(xy[0]), float(xy[1])], "ztop": float(zt),
                    "rgb": [float(v) for v in np.median(cols[sel], 0)],
                    "n": int(sel.sum())})
    out.sort(key=lambda b: b["xy"][1])
    return out


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if s < 1e-4:
        # near pi: use diagonal
        a = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        return a * ang
    return w / (2 * s) * ang


def _rot_cmd(rpy, gain=0.5, cap=0.4):
    """Normalized rotation command holding the gripper pointing straight down, yaw 0."""
    Rd = _rot([math.pi, 0.0, 0.0])
    Rc = _rot(rpy)
    w = _rotvec(Rd @ Rc.T)          # base-frame rotation from current to desired
    r = -gain * w / 0.1             # step applies exp(-0.1 r)
    n = float(np.linalg.norm(r))
    if n > cap:
        r = r * cap / n
    return r, float(np.linalg.norm(w))


def _color_dist(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    an = a / max(np.linalg.norm(a), 1.0)
    bn = b / max(np.linalg.norm(b), 1.0)
    return float(np.linalg.norm(an - bn)) + 0.002 * float(np.linalg.norm(a - b))


FIELDS = ["cue_rgb", "cue_xy", "cue_frames", "cue_step", "delay_frames", "cubes",
          "target_xy", "target_rgb", "target_conf", "appear_step", "stage", "stage_steps",
          "prev_tcp", "n_act", "settle_total", "log"]


def _mem(memory):
    m = {}
    defaults = {"cue_rgb": [], "cue_xy": [], "cue_frames": 0, "cue_step": -1, "delay_frames": 0,
                "cubes": [], "target_xy": [], "target_rgb": [], "target_conf": 0.0,
                "appear_step": -1, "stage": "observe", "stage_steps": 0, "prev_tcp": [],
                "n_act": 0, "settle_total": 0, "log": ""}
    for k in FIELDS:
        m[k] = memory.get(k, defaults[k]) if memory else defaults[k]
        if m[k] is None:
            m[k] = defaults[k]
    return m


def perceive_decide(obs, goal, memory, last):
    """Track cue / delay / choice phases from the front camera, then drive to the matching cube."""
    m = _mem(memory)
    frames = list(obs.frames) if obs is not None else []
    # ---- perception over every new frame (cue may last only one frame) ----
    for fr in frames:
        if m["target_xy"]:
            break
        cubes = _cubes(fr)
        if not m["cue_rgb"]:
            if len(cubes) == 1:
                m["cue_rgb"] = cubes[0]["rgb"]
                m["cue_xy"] = cubes[0]["xy"]
                m["cue_step"] = int(fr.step)
                m["cue_frames"] = 1
            elif len(cubes) >= 2:
                m["log"] = (m["log"] + " cubes_before_cue@%d" % fr.step)[-400:]
        else:
            if len(cubes) == 1 and m["delay_frames"] == 0:
                m["cue_frames"] = int(m["cue_frames"]) + 1
            elif len(cubes) == 0:
                m["delay_frames"] = int(m["delay_frames"]) + 1
            elif len(cubes) >= 2:
                d = [_color_dist(c["rgb"], m["cue_rgb"]) for c in cubes]
                order = np.argsort(d)
                j = int(order[0])
                conf = (d[int(order[1])] - d[j]) if len(d) > 1 else 0.0
                m["cubes"] = [[round(c["xy"][0], 4), round(c["xy"][1], 4)] + [int(v) for v in c["rgb"]] for c in cubes]
                m["target_xy"] = cubes[j]["xy"]
                m["target_rgb"] = cubes[j]["rgb"]
                m["target_conf"] = float(conf)
                m["appear_step"] = int(fr.step)
                m["stage"] = "approach"
                m["stage_steps"] = 0
    cur = obs.current
    tcp = [float(v) for v in cur.tcp]
    vel = [0.0, 0.0, 0.0]
    if m["prev_tcp"] and m["stage"] != "observe" and m["stage_steps"] > 0:
        vel = [tcp[i] - float(m["prev_tcp"][i]) for i in range(3)]
    m["prev_tcp"] = tcp
    if not m["target_xy"]:
        act = {"tool": "act_wait", "args": {}}
    else:
        tx, ty = float(m["target_xy"][0]), float(m["target_xy"][1])
        exy = math.hypot(tx - tcp[0], ty - tcp[1])
        st = m["stage"]
        # fallback: if the arm has been held static at the touch pose for a while without the episode ending,
        # lower the touch pose step by step toward the cube centre (the success radius may be smaller here)
        level = int(m["settle_total"]) // FALLBACK_AFTER
        fz = max(CUBE_CENTER_Z, FINAL_Z - FALLBACK_DZ * level)
        tol = SETTLE_TOL if level == 0 else FALLBACK_TOL
        explore = (EXPLORE_OFFSET is not None and int(m["appear_step"]) <= EXPLORE_MAX_APPEAR
                   and "x_done" not in str(m["log"]))
        if explore and st == "x_settle" and int(m["stage_steps"]) >= EXPLORE_HOLD:
            m["log"] = (str(m["log"]) + " x_done@%d" % int(cur.step))[-400:]
            explore = False
        if explore:
            tx += EXPLORE_OFFSET[0]
            ty += EXPLORE_OFFSET[1]
            fz += EXPLORE_OFFSET[2]
            exy = math.hypot(tx - tcp[0], ty - tcp[1])
        err3 = math.sqrt(exy * exy + (fz - tcp[2]) ** 2)
        if explore:
            new = "x_settle" if ((err3 < SETTLE_TOL and exy < SETTLE_XY) or st == "x_settle") else "approach"
        else:
            new = "settle" if (err3 < tol and exy < SETTLE_XY) else "approach"
        if new != st:
            m["stage_steps"] = 0
        st = new
        m["stage"] = st
        # funnel: allowed height above the touch pose grows with lateral error, so open fingers
        # clear the cube top without a discontinuous hover target (R002: abrupt brake -> x swing)
        tz = fz + min(FUNNEL_MAX, FUNNEL_K * max(0.0, exy - FUNNEL_FREE))
        goal_p = [tx, ty, tz]
        m["stage_steps"] = int(m["stage_steps"]) + 1
        if st == "settle":
            m["settle_total"] = int(m["settle_total"]) + 1
        if st in ("settle", "x_settle"):
            act = {"tool": "act_settle", "args": {"target": goal_p}}
        else:
            act = {"tool": "act_reach", "args": {"target": goal_p, "vel": vel, "gripper": 1.0}}
    m["n_act"] = int(m["n_act"]) + 1
    return {"memory": m, "act": act}


def act_wait(robot):
    """Send one zero-motion step with the gripper open (cue/delay phases; native commands are suppressed then)."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp]}


def act_reach(robot, target, vel, gripper=1.0):
    """One momentum-aware Cartesian step toward world target [x,y,z]; holds gripper pointing down.

    vel is the last measured per-step TCP displacement (m).  Returns measured TCP and error."""
    f = robot.now()
    tcp = np.asarray(f.tcp, float)
    e = np.asarray(target, float) - tcp[:3]
    v = np.asarray(vel, float)
    dist = float(np.linalg.norm(e))
    near = dist < NEAR_DIST
    # speed profile: cruise VMAX, deceleration-limited (sqrt(2*DEC*d)), and at most ALPHA*d per step
    sdes = min(VMAX, math.sqrt(2.0 * DEC * dist), (ALPHA_NEAR if near else ALPHA) * dist)
    a = (sdes * e / max(dist, 1e-9) - MOM * v) / GAIN_B
    cap = CMD_NEAR if near else 1.0
    s = float(np.abs(a).max())
    if s > cap:
        a = a * cap / s
    r, rerr = _rot_cmd(tcp[3:])
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(gripper))
    g = robot.now()
    t2 = np.asarray(g.tcp, float)
    return {"step": int(g.step), "cmd": [round(float(x), 3) for x in a] + [round(float(x), 3) for x in r],
            "tcp": [round(float(x), 4) for x in t2], "err_before": [round(float(x), 4) for x in e],
            "err_after": [round(float(x), 4) for x in (np.asarray(target, float) - t2[:3])],
            "rot_err": round(rerr, 4)}


def act_settle(robot, target):
    """Zero arm command (controller target = measured pose) with gripper open, so the arm comes to rest at the cube.

    Success appears to need the arm nearly static (max joint speed ~<0.2 rad/s) while the TCP is at the cube."""
    f = robot.now()
    q0 = np.asarray(f.joints, float)
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    g = robot.now()
    q1 = np.asarray(g.joints, float)
    t2 = np.asarray(g.tcp, float)
    return {"step": int(g.step), "tcp": [round(float(x), 4) for x in t2],
            "err_after": [round(float(x), 4) for x in (np.asarray(target, float) - t2[:3])],
            "qspeed_est": round(float(np.abs(q1 - q0).max() * 20.0), 3)}
