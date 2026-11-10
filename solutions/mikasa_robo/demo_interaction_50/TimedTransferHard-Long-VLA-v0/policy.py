"""TimedTransferHard-Long policy: pick the blue cube from the green disc, hover above the
red disc, and lower it onto the red disc at a fixed step offset after the lamp turns green.

Evidence (50 assigned demos): lamp turns green at frame 1-3; expert grasps by ~frame 20,
hovers over the red disc, starts descending at frame green+954 (47/50 demos; 3 at +951)
and native success fires 5-7 steps later (frame green+957..961) with the cube still held.
Rollouts R001-R006 measured the window (count = frame - first green frame): zone entry at count
945 -> immediate fail; 957/977/1020/1048 -> success; hovering past 1050 -> fail. The policy
therefore enters the zone at count ~1000 (window centre). Zone: entered when TCP z drops from
0.089 to 0.069 above the red centre (hover at 0.12 is safely outside).
"""

LAMP_MIN_PIXELS = 20
DOWN_R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
DESCEND_OFFSET = 997      # descent command start; zone entry ~3 steps later = count 1000 (window ~950..1050 measured R001-R006)
HOVER_Z = 0.12            # TCP z while waiting above the red disc
PLACE_Z = 0.035           # TCP z with cube resting on the disc (cube centre ~0.027)
GRASP_Z = 0.030
APPROACH_Z = 0.12
LIFT_Z = 0.15
POST_PLACE_STEPS = 90


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, float(c)))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return d * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _lamp_pixels(frame):
    rgb = frame.front_rgb
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return int(((g > 200) & (r < 60) & (b < 60)).sum())


def _lamp_on(frame):
    return _lamp_pixels(frame) >= LAMP_MIN_PIXELS


def _backproject(K, E, depth, mask):
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64) / 1000.0
    ok = z > 0
    v, u, z = v[ok], u[ok], z[ok]
    if len(z) == 0:
        return np.zeros((0, 3))
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    R = np.asarray(E)[:, :3].astype(np.float64)
    t = np.asarray(E)[:, 3].astype(np.float64)
    return (pc - t) @ R


def _color_masks(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    red = (r > 150) & (r - g > 70) & (np.abs(g - b) < 25)
    green = (g > 150) & (g - r > 60) & (np.abs(r - b) < 25) & (r > 40)
    blue = (b > 120) & (b - r > 50) & (b - g > 50)
    return red, green, blue


def _layout_from_camera(frame, cam):
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K")).astype(np.float64)
    E = getattr(frame, cam + "_extrinsic")
    red, green, blue = _color_masks(rgb)
    out = {}
    for name, m in (("red", red), ("green", green)):
        P = _backproject(K, E, depth, m)
        P = P[(P[:, 2] > -0.02) & (P[:, 2] < 0.03)] if len(P) else P
        out[name] = [float(P[:, 0].mean()), float(P[:, 1].mean())] if len(P) >= 30 else None
        out[name + "_n"] = int(len(P))
    P = _backproject(K, E, depth, blue)
    if len(P) >= 10:
        top = P[P[:, 2] > P[:, 2].max() - 0.006]
        out["cube"] = [float(top[:, 0].mean()), float(top[:, 1].mean()), float(P[:, 2].max())]
        out["cube_n"] = int(len(top))
    else:
        out["cube"] = None
        out["cube_n"] = int(len(P))
    return out


def _layout(frame):
    """Red/green disc centres and cube centre (world xy). Wrist camera preferred (more pixels,
    no cube occlusion of the discs); front camera as fallback with its measured +x bias removed."""
    w = _layout_from_camera(frame, "wrist")
    f = _layout_from_camera(frame, "front")
    res = {"wrist": w, "front": f}
    red = w["red"] if (w["red"] is not None and w["red_n"] > 400) else None
    if red is None and f["red"] is not None:
        red = [f["red"][0] - 0.0035, f["red"][1]]
    cube = w["cube"][:2] if (w["cube"] is not None and w["cube_n"] > 60) else None
    if cube is None and f["cube"] is not None:
        cube = [f["cube"][0] - 0.002, f["cube"][1]]
    green = w["green"] if w["green"] is not None else f["green"]
    if cube is None and green is not None:
        cube = list(green)
    if red is None and green is not None:
        red = [green[0], green[1] + 0.219]
    res["red_xy"] = red
    res["cube_xy"] = cube
    res["green_xy"] = green
    return res


def _servo_cmd(tcp, target, kp=1.0, vmax=0.6, kr=1.0, rmax=0.5):
    tcp = np.asarray(tcp, dtype=np.float64)
    e = np.asarray(target, dtype=np.float64) - tcp[:3]
    d = kp * e / 0.1
    n = float(np.linalg.norm(d))
    if n > vmax:
        d = d * (vmax / n)
    Rm = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    er = _rotvec(np.array(DOWN_R) @ Rm.T)
    rc = -kr * er / 0.1
    nr = float(np.linalg.norm(rc))
    if nr > rmax:
        rc = rc * (rmax / nr)
    return [float(d[0]), float(d[1]), float(d[2]), float(rc[0]), float(rc[1]), float(rc[2])], float(np.linalg.norm(e)), float(np.linalg.norm(er))


def _step_to(robot, target, grip, kp=1.0, vmax=0.6):
    cmd, _, _ = _servo_cmd(robot.now().tcp, target, kp=kp, vmax=vmax)
    robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], grip)
    return robot.now()


def _move(robot, target, grip, max_steps, tol=0.004, vtol=0.002, kp=1.0, vmax=0.6, watch=None):
    """Closed-loop move; stops when within tol and per-step motion below vtol."""
    prev = np.asarray(robot.now().tcp[:3], dtype=np.float64)
    n = 0
    err = 1.0
    for n in range(1, max_steps + 1):
        fr = _step_to(robot, target, grip, kp=kp, vmax=vmax)
        if watch is not None:
            watch(fr)
        p = np.asarray(fr.tcp[:3], dtype=np.float64)
        err = float(np.linalg.norm(p - np.asarray(target)))
        sp = float(np.linalg.norm(p - prev))
        prev = p
        if err < tol and sp < vtol:
            break
    return {"steps": n, "err": round(err, 4)}


def _first_green(frames):
    for fr in frames:
        if _lamp_on(fr):
            return int(fr.step)
    return None


def perceive_decide(obs, goal, memory, last):
    """Track lamp onset, layout and stage; choose the next act."""
    m = dict(memory)
    cur = obs.current
    if m.get("stage") is None:
        lay = _layout(cur)
        m["stage"] = "pick"
        m["red_xy"] = lay["red_xy"]
        m["cube_xy"] = lay["cube_xy"]
        m["pick_tries"] = 0
        m["green_step"] = None
        m["log"] = "layout red=%s cube=%s green=%s wn=%s/%s fn=%s/%s" % (
            lay["red_xy"], lay["cube_xy"], lay["green_xy"], lay["wrist"]["red_n"], lay["wrist"]["cube_n"],
            lay["front"]["red_n"], lay["front"]["cube_n"])
        m["place_done"] = False
    if m.get("green_step") is None:
        g = _first_green(obs.frames)
        if g is not None:
            m["green_step"] = g
    m["last_step"] = int(cur.step)
    fingers = np.asarray(cur.fingers, dtype=np.float64)
    holding = bool(fingers.min() > 0.008 and fingers.max() < 0.035)
    if last:
        m["log"] = str(last)[:600]

    stage = m["stage"]
    if stage == "pick":
        if m["pick_tries"] > 0 and holding:
            stage = "hold"
        elif m["pick_tries"] >= 3:
            stage = "hold"
    elif stage == "hold" and m["place_done"]:
        stage = "after"
    m["stage"] = stage

    if stage == "pick":
        if m["pick_tries"] > 0:
            lay = _layout(cur)
            if lay["cube_xy"] is not None:
                m["cube_xy"] = lay["cube_xy"]
        m["pick_tries"] = m["pick_tries"] + 1
        cx, cy = m["cube_xy"]
        act = {"tool": "act_pick", "args": {"cube_x": cx, "cube_y": cy}}
    elif stage == "hold":
        m["place_done"] = True
        rx, ry = m["red_xy"]
        act = {"tool": "act_hover_and_place", "args": {
            "red_x": rx, "red_y": ry, "green_step": m["green_step"],
            "descend_offset": DESCEND_OFFSET, "hover_z": HOVER_Z, "place_z": PLACE_Z}}
    else:
        rx, ry = m["red_xy"]
        act = {"tool": "act_hold_pose", "args": {"x": rx, "y": ry, "z": PLACE_Z, "steps": 100, "grip": -1.0}}
    return {"memory": m, "act": act}


def act_pick(robot, cube_x, cube_y, approach_z=APPROACH_Z, grasp_z=GRASP_Z, lift_z=LIFT_Z):
    """Approach above the cube, descend, close the gripper and lift. Returns measured phases."""
    out = {"start_step": int(robot.now().step)}
    seen = {"g": None}

    def watch(fr):
        if seen["g"] is None and _lamp_on(fr):
            seen["g"] = int(fr.step)

    if _lamp_on(robot.now()):
        seen["g"] = int(robot.now().step)
    out["approach"] = _move(robot, [cube_x, cube_y, approach_z], 1.0, 60, tol=0.006, watch=watch)
    out["descend"] = _move(robot, [cube_x, cube_y, grasp_z], 1.0, 40, tol=0.004, vmax=0.5, watch=watch)
    for _ in range(8):
        fr = _step_to(robot, [cube_x, cube_y, grasp_z], -1.0)
        watch(fr)
    out["fingers_closed"] = [round(float(v), 4) for v in robot.now().fingers]
    out["lift"] = _move(robot, [cube_x, cube_y, lift_z], -1.0, 40, tol=0.008, vmax=0.5, watch=watch)
    fr = robot.now()
    out["end_step"] = int(fr.step)
    out["tcp"] = [round(float(v), 4) for v in fr.tcp]
    out["fingers"] = [round(float(v), 4) for v in fr.fingers]
    out["lamp_green_step"] = seen["g"]
    return out


def act_hover_and_place(robot, red_x, red_y, green_step=None, descend_offset=DESCEND_OFFSET,
                        hover_z=HOVER_Z, place_z=PLACE_Z, post_steps=POST_PLACE_STEPS):
    """Carry the held cube above the red disc, actively hold there until command index
    green_step+descend_offset (detecting the lamp onset here if still unknown), then lower the
    cube onto the disc and keep holding. Returns measured hold drift and timing."""
    out = {"start_step": int(robot.now().step), "green_step_in": green_step}
    g = {"v": green_step}

    def watch(fr):
        if g["v"] is None and _lamp_on(fr):
            g["v"] = int(fr.step)

    hover = [red_x, red_y, hover_z]
    try:
        chk = robot.dry_run(red_x, red_y, hover_z, math.pi, 0.0, 0.0)
        out["dry_hover"] = {"reachable": chk.get("reachable"), "margin": chk.get("joint_margin")}
    except Exception as e:
        out["dry_hover"] = str(e)[:100]
    out["carry"] = _move(robot, hover, -1.0, 80, tol=0.004, vmax=0.6, watch=watch)
    max_err = 0.0
    n_hold = 0
    while True:
        fr = robot.now()
        watch(fr)
        k = int(fr.step)
        gs = g["v"] if g["v"] is not None else 2
        if k >= gs + descend_offset:
            break
        if k > 1150:
            break
        fr = _step_to(robot, hover, -1.0, kp=1.0, vmax=0.3)
        n_hold += 1
        e = float(np.linalg.norm(np.asarray(fr.tcp[:3]) - np.asarray(hover)))
        if n_hold > 20:
            max_err = max(max_err, e)
    out["hold_steps"] = n_hold
    out["hold_max_err"] = round(max_err, 4)
    out["green_step"] = g["v"]
    out["descend_start_step"] = int(robot.now().step)
    out["tcp_before_descend"] = [round(float(v), 4) for v in robot.now().tcp]
    traj = []
    target = [red_x, red_y, place_z]
    for i in range(post_steps):
        fr = _step_to(robot, target, -1.0, kp=1.0, vmax=0.5)
        if i < 15:
            traj.append(round(float(fr.tcp[2]), 3))
    out["descent_z"] = traj
    out["end_step"] = int(robot.now().step)
    out["tcp_end"] = [round(float(v), 4) for v in robot.now().tcp]
    out["fingers_end"] = [round(float(v), 4) for v in robot.now().fingers]
    return out


def act_hold_pose(robot, x, y, z, steps=100, grip=-1.0):
    """Hold a TCP position (gripper down) for a number of steps; fallback while waiting."""
    for _ in range(int(steps)):
        _step_to(robot, [x, y, z], grip, kp=1.0, vmax=0.4)
    fr = robot.now()
    return {"end_step": int(fr.step), "tcp": [round(float(v), 4) for v in fr.tcp]}
