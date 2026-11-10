"""TimedTransferEasy-Long: pick the blue cube from the green disc, hold it above the
red disc, and bring it onto the red disc ~300 steps after the white lamp turns green.

Timing constants come from 50 expert demos (see task_analysis.md):
  * lamp onset (first saturated-green lamp frame) at frame 1..3;
  * experts start the final descent at onset+289 (46/50) or onset+286 (4/50);
  * native success fires at onset+293..297, when the cube is ~2-4 cm above its rest
    pose on the red disc (cube-to-goal distance threshold ~0.04 m, hypothesis).
Rollouts R001-R008: entering the goal region at onset+265/277/283 -> immediate fail; at
onset+287/296/304/314 -> success; hovering until onset+316 -> fail. Target: onset+300.
"""

# ---- task constants (evidence in task_analysis.md) ----
DESCENT_START = 295      # descent start after lamp onset; cube enters goal at onset+300 (window ~285..315, R001-R008)
HOLD_Z = 0.18            # TCP height while carrying / waiting above the red disc
CUBE_HALF = 0.02         # cube half size (top face at 0.0467, green disc top ~0.0065)
GRASP_BELOW_TOP = 0.018  # TCP this far below cube top when closing
PLACE_GAP = 0.004        # leave this gap between cube bottom and red disc surface
POS_GAIN = 1.0
ROT_GAIN = 0.6
VMAX = 1.0
DESCENT_VMAX = 0.8


def _masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return {
        "cube": (b > 150) & (r < 130) & (g < 130) & (np.abs(r - g) < 25),
        "red": (r > 180) & (g < 110) & (b < 110) & (np.abs(g - b) < 20),
        "green": (g > 160) & (g < 235) & (r < 120) & (b < 120) & (np.abs(r - b) < 20),
        "lamp": (g > 220) & (r < 60) & (b < 60),
    }


def _backproject(mask, depth, K, E):
    v, u = np.nonzero(mask & (depth > 0))
    if len(u) == 0:
        return None
    d = depth[v, u].astype(float) / 1000.0
    xc = (u + 0.5 - K[0, 2]) * d / K[0, 0]
    yc = (v + 0.5 - K[1, 2]) * d / K[1, 1]
    pc = np.stack([xc, yc, d], 1)
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return (pc - t) @ R


def _lamp_count(frame):
    n = 0
    for cam in ("front_rgb", "wrist_rgb"):
        img = getattr(frame, cam, None)
        if img is None:
            continue
        g = img[..., 1].astype(int)
        m = (g > 220) & (img[..., 0].astype(int) < 60) & (img[..., 2].astype(int) < 60)
        n = max(n, int(m.sum()))
    return n


def _lamp_on(frame):
    return _lamp_count(frame) > 20


def _cube_from(frame, cam):
    rgb = getattr(frame, cam + "_rgb")
    m = _masks(rgb)["cube"]
    P = _backproject(m, getattr(frame, cam + "_depth"), getattr(frame, cam + "_K"),
                     getattr(frame, cam + "_extrinsic"))
    if P is None or len(P) < 15:
        return None
    zt = float(np.percentile(P[:, 2], 98))
    top = P[P[:, 2] > zt - 0.006]
    if len(top) < 5:
        return None
    lo = np.percentile(top[:, :2], 3, axis=0)
    hi = np.percentile(top[:, :2], 97, axis=0)
    c = (lo + hi) / 2.0
    return [float(c[0]), float(c[1]), zt, int(len(P))]


def _disc_from(frame, cam, name):
    rgb = getattr(frame, cam + "_rgb")
    m = _masks(rgb)[name]
    P = _backproject(m, getattr(frame, cam + "_depth"), getattr(frame, cam + "_K"),
                     getattr(frame, cam + "_extrinsic"))
    if P is None or len(P) < 30:
        return None
    P = P[P[:, 2] < 0.02]
    if len(P) < 30:
        return None
    lo = np.percentile(P[:, :2], 1, axis=0)
    hi = np.percentile(P[:, :2], 99, axis=0)
    c = (lo + hi) / 2.0
    ext = hi - lo
    return [float(c[0]), float(c[1]), float(np.median(P[:, 2])), float(ext[0]), float(ext[1]), int(len(P))]


def _scene(frame):
    """Cube (top-face centre, top z) and red/green disc centres from one frame."""
    cw = _cube_from(frame, "wrist")
    cf = _cube_from(frame, "front")
    cube = cw if (cw is not None and cw[3] >= 80) else (cf if cf is not None else cw)
    red = _disc_from(frame, "front", "red")
    if red is None or red[3] < 0.08 or red[4] < 0.08:
        rw = _disc_from(frame, "wrist", "red")
        red = rw if rw is not None else red
    green = _disc_from(frame, "front", "green")
    return {"cube": cube, "red": red, "green": green}


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(M):
    c = (np.trace(M) - 1.0) / 2.0
    th = math.acos(max(-1.0, min(1.0, c)))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


def _down_R(yaw):
    return _rot([math.pi, 0.0, yaw])


def _cmd(tcp, target, Rt, vmax=VMAX, kp=POS_GAIN):
    err = np.asarray(target, dtype=float) - np.asarray(tcp[:3], dtype=float)
    a = np.clip(kp * err / 0.1, -vmax, vmax)
    e = _rotvec(Rt @ _rot(tcp[3:6]).T)
    r = np.clip(-ROT_GAIN * e / 0.1, -1.0, 1.0)
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2])], err


def _step(robot, cmd, grip, st):
    f = robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
    st["n"] = st["n"] + 1
    if st["on"] < 0 and _lamp_on(f):
        st["on"] = int(f.step)
    return f


def _servo(robot, target, Rt, grip, st, tol=0.004, max_steps=60, vmax=VMAX, settle=2):
    ok = 0
    f = robot.now()
    err = np.zeros(3)
    for _ in range(max_steps):
        cmd, err = _cmd(f.tcp, target, Rt, vmax)
        if float(np.linalg.norm(err)) < tol:
            ok += 1
            if ok >= settle:
                break
        else:
            ok = 0
        f = _step(robot, cmd, grip, st)
    return f, float(np.linalg.norm(np.asarray(target) - np.asarray(robot.now().tcp[:3])))


def _hold(robot, target, Rt, grip, st, n):
    f = robot.now()
    for _ in range(n):
        cmd, _e = _cmd(f.tcp, target, Rt)
        f = _step(robot, cmd, grip, st)
    return f


def _num(text, key, default=None):
    if not isinstance(text, str):
        return default
    k = '"' + key + '": '
    i = text.find(k)
    if i < 0:
        return default
    j = i + len(k)
    s = ""
    while j < len(text) and text[j] in "-0123456789.eE":
        s += text[j]
        j += 1
    if text[j:j + 4] == "true":
        return True
    if text[j:j + 5] == "false":
        return False
    if s == "":
        return default
    return float(s)


def _scan_lamp(frames, on):
    if on is not None and on >= 0:
        return on
    for fr in frames:
        if _lamp_on(fr):
            return int(fr.step)
    return -1


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive scene -> act_pick -> act_timed_place -> act_hold.
    Memory keeps scene estimates, lamp onset step and stage progress."""
    mem = dict(memory)
    for k in ("stage", "cube_pos", "red_pos", "green_pos", "lamp_on_step", "grasp_offset",
              "pick_attempts", "notes"):
        if k not in mem:
            mem[k] = None
    stage = mem["stage"] or "start"
    on = mem["lamp_on_step"] if mem["lamp_on_step"] is not None else -1
    on = int(on)
    if on < 0 and last:
        v = _num(last, "lamp_on_step", None)
        if v is not None and v is not True and v is not False and v >= 0:
            on = int(v)
    on = _scan_lamp(obs.frames, on)
    mem["lamp_on_step"] = on
    cur = obs.current
    notes = mem["notes"] or ""
    attempts = int(mem["pick_attempts"] or 0)

    if stage == "picking":
        grasped = _num(last, "grasped", None)
        if grasped is None:
            grasped = float(min(cur.fingers)) > 0.006
        if grasped:
            ox = _num(last, "off_x", 0.0)
            oy = _num(last, "off_y", 0.0)
            oz = _num(last, "off_z", 0.0)
            mem["grasp_offset"] = [float(ox), float(oy), float(oz)]
            stage = "carry"
        else:
            notes += " pick%d_failed@%d" % (attempts, int(cur.step))
            stage = "start" if attempts < 3 else "give_up"

    if stage == "start":
        sc = _scene(cur)
        if sc["cube"] is not None:
            mem["cube_pos"] = [sc["cube"][0], sc["cube"][1], sc["cube"][2]]
        if sc["red"] is not None and mem["red_pos"] is None:
            mem["red_pos"] = [sc["red"][0], sc["red"][1], sc["red"][2]]
        if sc["green"] is not None and mem["green_pos"] is None:
            mem["green_pos"] = [sc["green"][0], sc["green"][1], sc["green"][2]]
        if mem["red_pos"] is None and mem["green_pos"] is not None:
            g = mem["green_pos"]
            mem["red_pos"] = [g[0] - 0.008, g[1] + 0.216, g[2]]  # demo-measured layout offset
            notes += " red_from_green"
        if mem["cube_pos"] is None:
            notes += " no_cube@%d" % int(cur.step)
            mem["stage"] = "start"
            mem["notes"] = notes[-900:]
            mem["pick_attempts"] = attempts
            return {"memory": mem, "act": {"tool": "act_hold_still", "args": {"n": 5, "lamp_on_step": on}}}
        c = mem["cube_pos"]
        mem["stage"] = "picking"
        mem["pick_attempts"] = attempts + 1
        notes += " cube=(%.3f,%.3f,%.3f)" % (c[0], c[1], c[2])
        if mem["red_pos"] is not None:
            notes += " red=(%.3f,%.3f)" % (mem["red_pos"][0], mem["red_pos"][1])
        mem["notes"] = notes[-900:]
        return {"memory": mem, "act": {"tool": "act_pick", "args": {
            "cx": c[0], "cy": c[1], "ctop": c[2], "lamp_on_step": on}}}

    if stage == "carry":
        r = mem["red_pos"]
        o = mem["grasp_offset"] or [0.0, 0.0, 0.0]
        mem["stage"] = "placed"
        mem["notes"] = notes[-900:]
        mem["pick_attempts"] = attempts
        return {"memory": mem, "act": {"tool": "act_timed_place", "args": {
            "rx": r[0], "ry": r[1], "rz": r[2], "off_x": o[0], "off_y": o[1], "off_z": o[2],
            "lamp_on_step": on, "descent_start": DESCENT_START}}}

    # placed / give_up: keep holding the current pose
    mem["stage"] = stage
    mem["notes"] = notes[-900:]
    mem["pick_attempts"] = attempts
    return {"memory": mem, "act": {"tool": "act_hold_still", "args": {"n": 100, "lamp_on_step": on}}}


def act_pick(robot, cx, cy, ctop, lamp_on_step=-1):
    """Grasp the cube whose top-face centre is (cx, cy) at height ctop and lift it to
    HOLD_Z. Monitors the lamp every step. Returns lamp_on_step, grasped, finger width,
    and the cube-centre offset relative to the TCP at grasp (off_x/off_y/off_z)."""
    st = {"on": int(lamp_on_step), "n": 0}
    f0 = robot.now()
    if st["on"] < 0 and _lamp_on(f0):
        st["on"] = int(f0.step)
    yaw = float(f0.tcp[5])
    Rt = _down_R(yaw)
    gz = max(0.018, ctop - GRASP_BELOW_TOP)
    pre = [cx, cy, ctop + 0.07]
    f, e1 = _servo(robot, pre, Rt, 1.0, st, tol=0.01, max_steps=50, settle=1)
    f, e2 = _servo(robot, [cx, cy, gz], Rt, 1.0, st, tol=0.003, max_steps=40, vmax=0.7, settle=2)
    tg = np.array(f.tcp[:3], dtype=float)
    # close and wait for the fingers to stop
    prev = 1.0
    for _ in range(12):
        f = _step(robot, [0.0] * 6, -1.0, st)
        w = float(min(f.fingers))
        if abs(w - prev) < 2e-4:
            break
        prev = w
    f = _hold(robot, [cx, cy, gz], Rt, -1.0, st, 2)
    w = float(min(f.fingers))
    f, e3 = _servo(robot, [cx, cy, HOLD_Z], Rt, -1.0, st, tol=0.01, max_steps=50, settle=1)
    w2 = float(min(robot.now().fingers))
    cube_c = np.array([cx, cy, ctop - CUBE_HALF])
    off = cube_c - tg
    return {"lamp_on_step": st["on"], "grasped": bool(w2 > 0.006), "finger": w2, "finger_at_close": w,
            "off_x": float(off[0]), "off_y": float(off[1]), "off_z": float(off[2]),
            "tcp_grasp": tg.tolist(), "err_pre": e1, "err_grasp": e2, "err_lift": e3,
            "step": int(robot.now().step), "n": st["n"]}


def act_timed_place(robot, rx, ry, rz, off_x=0.0, off_y=0.0, off_z=0.0, lamp_on_step=-1,
                    descent_start=289, end_hold=40):
    """Carry the grasped cube above the red disc centre (rx, ry), wait until
    lamp_on_step + descent_start, descend so the cube rests on the red disc, and keep
    holding. Records when the estimated cube-goal distance first drops below 0.04 m."""
    st = {"on": int(lamp_on_step), "n": 0}
    f = robot.now()
    if st["on"] < 0 and _lamp_on(f):
        st["on"] = int(f.step)
    yaw = float(f.tcp[5])
    Rt = _down_R(yaw)
    tx, ty = rx - off_x, ry - off_y
    cube_rest_z = rz + CUBE_HALF + PLACE_GAP
    tz = max(0.012, cube_rest_z - off_z)
    above = [tx, ty, HOLD_Z]
    reach = None
    try:
        dr = robot.dry_run(tx, ty, tz, math.pi, 0.0, yaw, mode="absolute")
        reach = {"ok": bool(dr.get("within_joint_limits")), "min_margin": dr.get("min_joint_margin_rad")}
    except Exception as ex:
        reach = {"err": str(ex)[:80]}
    f, e1 = _servo(robot, above, Rt, -1.0, st, tol=0.004, max_steps=80, settle=3)
    arrive = int(f.step)
    # wait for the lamp and the descent time, holding closed-loop
    waited = 0
    while True:
        f = robot.now()
        if st["on"] >= 0 and int(f.step) >= st["on"] + descent_start:
            break
        if int(f.step) >= 590:
            break
        cmd, _e = _cmd(f.tcp, above, Rt)
        f = _step(robot, cmd, -1.0, st)
        waited += 1
    t0 = int(robot.now().step)
    goal = np.array([rx, ry, rz + CUBE_HALF])
    cross = -1
    zs = []
    dist_min = 9.0
    for k in range(200):
        f = robot.now()
        cube = np.array(f.tcp[:3], dtype=float) + np.array([off_x, off_y, off_z])
        d = float(np.linalg.norm(cube - goal))
        dist_min = min(dist_min, d)
        if cross < 0 and d < 0.04:
            cross = int(f.step)
        if len(zs) < 40:
            zs.append(round(float(f.tcp[2]), 4))
        if k > 15 and cross >= 0 and int(f.step) >= cross + end_hold:
            break
        cmd, _e = _cmd(f.tcp, [tx, ty, tz], Rt, DESCENT_VMAX)
        f = _step(robot, cmd, -1.0, st)
    fin = robot.now()
    return {"lamp_on_step": st["on"], "arrive_above_step": arrive, "err_above": e1, "waited": waited,
            "descent_t0": t0, "cross_step": cross, "cross_rel": (cross - st["on"]) if cross >= 0 else None,
            "dist_min": dist_min, "z_profile": zs, "final_tcp": [float(v) for v in fin.tcp[:3]],
            "finger": float(min(fin.fingers)), "step": int(fin.step), "reach": reach}


def act_hold_still(robot, n=50, lamp_on_step=-1):
    """Hold the current TCP pose and gripper state for n steps (closed-loop)."""
    st = {"on": int(lamp_on_step), "n": 0}
    f = robot.now()
    grip = -1.0 if float(min(f.fingers)) < 0.035 else 1.0
    target = [float(v) for v in f.tcp[:3]]
    Rt = _rot(f.tcp[3:6])
    _hold(robot, target, Rt, grip, st, int(n))
    return {"lamp_on_step": st["on"], "step": int(robot.now().step)}
