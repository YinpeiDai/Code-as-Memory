"""T12: open the middle drawer, put cookies in it, then chocolate in the same drawer.

All geometry is in world metres.  Perception uses the agentview RGB-D point cloud;
control is closed-loop servoing of the measured TCP pose with OSC delta steps.
Offsets below were calibrated offline against the 50 assigned expert recordings
(see task_analysis.md); they are scene-relative, never per-episode lookups.
"""

# ---------------------------------------------------------------- constants
HANDLE_FROM_TOP_XMAX = 0.1285   # handle x = 99th pct x of cabinet top - this
HANDLE_FROM_TOP_YMAX = 0.204    # handle y = 99th pct y of cabinet top - this
HANDLE_Z = 0.607                # middle-drawer handle grasp height (const in all demos)
HANDLE_FROM_TOP_Z = 0.106       # handle z = cabinet top z - this (cross-check)
DEFAULT_HANDLE = [0.137, 0.454, 0.607]
DEFAULT_OBJ = {"cookies": [0.2193, -0.324], "chocolate": [-0.2171, -0.3284]}
OBJ_BOX = {"cookies": [0.14, 0.32, -0.42, -0.27], "chocolate": [-0.36, -0.12, -0.45, -0.24]}
# grasp = top-surface centroid - offset (offsets of symmetric expert grasps)
GRASP_OFFSET = {"cookies": [0.0065, -0.001], "chocolate": [0.0117, -0.0088]}
GRASP_Z = 0.434
PULL_DY = 0.24                  # expert pull distance (drawer stops ~0.165 and slips)
# release points relative to the detected handle (closed-drawer handle position)
SLOT = {"cookies": [0.0, -0.100, 0.120], "chocolate": [-0.065, -0.045, 0.120]}
CARRY_Z = 0.88
TABLE_Z = 0.430
MIN_HOLD_APERTURE = 0.008       # finger gap (m) above which something is held
DRAWER_LEVELS = ("top", "middle", "bottom")


# ---------------------------------------------------------------- math helpers
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _r_down(yaw=0.0):
    """Gripper pointing down, fingers closing along world y (expert pick pose)."""
    return _rz(yaw) @ _rx(math.pi)


def _r_handle():
    """Gripper pointing +y (into the cabinet face), fingers closing vertically."""
    return _rx(1.5 * math.pi)


def _aperture(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


# ---------------------------------------------------------------- perception helpers
def _cloud(frame):
    """World XYZ for every agentview pixel (H, W, 3) and a validity mask."""
    depth = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], -1)
    Wp = (P - E[:, 3]) @ E[:, :3]
    return Wp, depth > 0.05


def _find_cabinet(Wp, valid, rgb):
    dark = rgb.astype(int).sum(-1) < 200
    m = valid & dark & (Wp[..., 1] > 0.2) & (Wp[..., 2] > 0.68) & (Wp[..., 2] < 0.76) \
        & (Wp[..., 0] > -0.3) & (Wp[..., 0] < 0.5)
    P = Wp[m]
    if len(P) < 300:
        return None
    xmin, xmax = np.percentile(P[:, 0], 1), np.percentile(P[:, 0], 99)
    ymin, ymax = np.percentile(P[:, 1], 1), np.percentile(P[:, 1], 99)
    ztop = float(np.median(P[:, 2]))
    hx = float(xmax - HANDLE_FROM_TOP_XMAX)
    hy = float(ymax - HANDLE_FROM_TOP_YMAX)
    hx2 = float((xmin + xmax) / 2 - 0.0045)
    hy2 = float(ymin - 0.016)
    hz = HANDLE_Z if abs(ztop - HANDLE_Z - HANDLE_FROM_TOP_Z) < 0.03 else ztop - HANDLE_FROM_TOP_Z
    return {"handle": [hx, hy, float(hz)], "check": [hx2, hy2], "n": int(len(P)),
            "top_z": ztop, "width": float(xmax - xmin), "depth": float(ymax - ymin)}


def _find_object(Wp, valid, name):
    x0, x1, y0, y1 = OBJ_BOX[name]
    m = valid & (Wp[..., 0] > x0) & (Wp[..., 0] < x1) & (Wp[..., 1] > y0) & (Wp[..., 1] < y1) \
        & (Wp[..., 2] > TABLE_Z + 0.007) & (Wp[..., 2] < 0.50)
    P = Wp[m]
    if len(P) < 25:
        return None
    zt = float(np.percentile(P[:, 2], 90))
    Q = P[P[:, 2] > zt - 0.006]
    c = Q[:, :2].mean(0)
    off = GRASP_OFFSET[name]
    return {"grasp": [float(c[0] - off[0]), float(c[1] - off[1])], "top": zt, "n": int(len(Q))}


def _drawer_state(frame, handle):
    """Front-panel position of the drawer below the cabinet face (y relative to the handle)."""
    Wp, valid = _cloud(frame)
    hx, hy, hz = handle
    m = valid & (Wp[..., 1] > hy - 0.24) & (Wp[..., 1] < hy - 0.03) & (Wp[..., 2] > hz - 0.035) \
        & (Wp[..., 2] < hz + 0.04) & (np.abs(Wp[..., 0] - hx) < 0.10)
    P = Wp[m]
    n = int(len(P))
    if n < 60:
        return {"n": n, "front_rel": 0.0, "open": False}
    front = float(np.percentile(P[:, 1], 5) - hy)
    return {"n": n, "front_rel": front, "open": bool(n > 150 and -0.21 < front < -0.12)}


def _object_on_table(frame, name, grasp_xy):
    Wp, valid = _cloud(frame)
    off = GRASP_OFFSET[name]
    cx, cy = grasp_xy[0] + off[0], grasp_xy[1] + off[1]
    m = valid & (np.abs(Wp[..., 0] - cx) < 0.035) & (np.abs(Wp[..., 1] - cy) < 0.02) \
        & (Wp[..., 2] > TABLE_Z + 0.007) & (Wp[..., 2] < 0.47)
    return int(m.sum())


def _parse_goal(goal):
    g = (goal or "").lower()
    objs = []
    for name in ("cookies", "chocolate"):
        i = g.find(name)
        if i >= 0:
            objs.append((i, name))
    order = [n for _, n in sorted(objs)] or ["cookies", "chocolate"]
    level = "middle"
    for lv in DRAWER_LEVELS:
        if lv + " drawer" in g:
            level = lv
            break
    return order, level


# ---------------------------------------------------------------- motion helpers
def _servo(robot, pos, rot, grip, tol=0.004, rtol=0.04, max_steps=200, gain=12.0,
           vmax=1.0, rgain=1.5, rmax=0.5, settle=0):
    """Drive the measured TCP to pos (world m) and rot (3x3) with P-control on OSC deltas."""
    pos = np.asarray(pos, dtype=float)
    fr = robot.now()
    n = 0
    stall = 0
    best = 1e9
    reached = False
    for n in range(1, max_steps + 1):
        tcp = np.asarray(fr.tcp, dtype=float)
        e = pos - tcp[:3]
        en = float(np.linalg.norm(e))
        if rot is not None:
            w = _mat2aa(rot @ _aa2mat(tcp[3:6]).T)
        else:
            w = np.zeros(3)
        wn = float(np.linalg.norm(w))
        if en < tol and wn < rtol:
            reached = True
            break
        d = e * gain
        dn = float(np.linalg.norm(d))
        if dn > vmax:
            d = d * (vmax / dn)
        r = w * rgain
        rn = float(np.linalg.norm(r))
        if rn > rmax:
            r = r * (rmax / rn)
        fr = robot.step(float(d[0]), float(d[1]), float(d[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        if en < best - 0.0005:
            best = en
            stall = 0
        else:
            stall += 1
    for _ in range(settle):
        fr = robot.step(0, 0, 0, 0, 0, 0, grip)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"reached": reached, "steps": n, "err": float(np.linalg.norm(pos - tcp[:3])),
            "tcp": [float(v) for v in tcp[:3]]}


def _gripper(robot, grip, steps):
    for _ in range(steps):
        robot.step(0, 0, 0, 0, 0, 0, grip)
    return _aperture(robot.now())


def _pull(robot, target, rot, max_steps=90):
    """Pull with the gripper closed until target, a stall, or the handle slips out."""
    target = np.asarray(target, dtype=float)
    fr = robot.now()
    hist = []
    n = 0
    for n in range(1, max_steps + 1):
        tcp = np.asarray(fr.tcp, dtype=float)
        e = target - tcp[:3]
        if float(np.linalg.norm(e)) < 0.005:
            break
        d = e * 10.0
        dn = float(np.linalg.norm(d))
        if dn > 1.0:
            d = d / dn
        w = _mat2aa(rot @ _aa2mat(tcp[3:6]).T) * 1.5
        fr = robot.step(float(d[0]), float(d[1]), float(d[2]), float(w[0]), float(w[1]), float(w[2]), 1.0)
        hist.append(float(fr.tcp[1]))
        if len(hist) > 25 and abs(hist[-1] - hist[-21]) < 0.003:
            break
    return {"steps": n, "y": float(robot.now().tcp[1]), "aperture": _aperture(robot.now())}


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Plan the next sub-task from the current agentview RGB-D and episode memory."""
    mem = dict(memory)
    fr = obs.current
    order, level = _parse_goal(goal)
    if mem.get("order") is None:
        mem["order"] = order
    if mem.get("drawer_level") is None:
        mem["drawer_level"] = level
    if mem.get("placed") is None:
        mem["placed"] = []
    if mem.get("attempts") is None:
        mem["attempts"] = 0
    if mem.get("notes") is None:
        mem["notes"] = ""
    if mem.get("decisions") is None:
        mem["decisions"] = 0
    mem["decisions"] = int(mem["decisions"]) + 1

    # ---- initial scene memory: cabinet handle and object grasp points
    if mem.get("handle") is None:
        Wp, valid = _cloud(fr)
        cab = _find_cabinet(Wp, valid, np.asarray(fr.agentview_rgb))
        if cab is None:
            mem["handle"] = list(DEFAULT_HANDLE)
            mem["notes"] += "cabinet not found, default handle; "
        else:
            mem["handle"] = cab["handle"]
            dx = cab["handle"][0] - cab["check"][0]
            dy = cab["handle"][1] - cab["check"][1]
            mem["notes"] += "cabinet n=%d check d=(%.3f,%.3f); " % (cab["n"], dx, dy)
        if mem["drawer_level"] != "middle":
            mem["notes"] += "drawer level %s not calibrated, using middle; " % mem["drawer_level"]
        for name in ("cookies", "chocolate"):
            ob = _find_object(Wp, valid, name)
            key = name + "_grasp"
            if ob is None:
                off = GRASP_OFFSET[name]
                mem[key] = [DEFAULT_OBJ[name][0] - off[0], DEFAULT_OBJ[name][1] - off[1], GRASP_Z]
                mem["notes"] += "%s not found, prior; " % name
            else:
                mem[key] = [ob["grasp"][0], ob["grasp"][1], GRASP_Z]
        mem["phase"] = "open_drawer"
        mem["drawer_open"] = False

    hx, hy, hz = mem["handle"]
    phase = mem.get("phase") or "open_drawer"

    # ---- evaluate the outcome of the previous act
    if last and isinstance(last, str):
        if last.startswith("act_open_drawer"):
            st = _drawer_state(fr, mem["handle"])
            mem["notes"] += "drawer n=%d front=%.3f; " % (st["n"], st["front_rel"])
            if st["n"] >= 60:
                mem["drawer_front"] = float(st["front_rel"])
            if st["open"]:
                mem["drawer_open"] = True
                phase = "pick"
                mem["attempts"] = 0
            else:
                mem["attempts"] = int(mem["attempts"]) + 1
                if st["n"] >= 60 and st["front_rel"] < -0.02:
                    # partially open: the handle moved with the front panel
                    mem["handle_now_dy"] = float(min(0.0, st["front_rel"] + 0.005))
                if mem["attempts"] >= 3:
                    mem["drawer_open"] = True   # proceed anyway; best effort
                    phase = "pick"
                    mem["attempts"] = 0
        elif last.startswith("act_pick"):
            held = '"held": true' in last
            if held:
                phase = "place"
            else:
                mem["attempts"] = int(mem["attempts"]) + 1
                name = mem["order"][len(mem["placed"])] if len(mem["placed"]) < len(mem["order"]) else None
                if name is not None:
                    Wp, valid = _cloud(fr)
                    ob = _find_object(Wp, valid, name)
                    if ob is not None:
                        mem[name + "_grasp"] = [ob["grasp"][0], ob["grasp"][1], GRASP_Z]
                        mem["notes"] += "re-seen %s; " % name
                    else:
                        mem["notes"] += "%s not re-seen; " % name
                phase = "pick"
                if int(mem["attempts"]) >= 4:
                    # give up on this object; keep the episode moving
                    mem["notes"] += "gave up %s; " % name
                    mem["placed"] = list(mem["placed"]) + [name]
                    mem["attempts"] = 0
                    phase = "pick" if len(mem["placed"]) < len(mem["order"]) else "wait"
        elif last.startswith("act_place"):
            name = mem["order"][len(mem["placed"])]
            if '"dropped": true' in last:
                # lost on the way: look for it again near its start spot and re-pick
                mem["attempts"] = int(mem["attempts"]) + 1
                Wp, valid = _cloud(fr)
                ob = _find_object(Wp, valid, name)
                mem["notes"] += "dropped %s; " % name
                if ob is not None and int(mem["attempts"]) < 4:
                    mem[name + "_grasp"] = [ob["grasp"][0], ob["grasp"][1], GRASP_Z]
                    phase = "pick"
                else:
                    mem["placed"] = list(mem["placed"]) + [name]
                    mem["attempts"] = 0
                    phase = "pick" if len(mem["placed"]) < len(mem["order"]) else "verify"
            else:
                mem["placed"] = list(mem["placed"]) + [name]
                mem["attempts"] = 0
                phase = "pick" if len(mem["placed"]) < len(mem["order"]) else "verify"
        elif last.startswith("act_wait"):
            # still running: look whether an object is still on the table at its start spot
            phase = "verify"

    if phase == "verify":
        redo = None
        for name in mem["order"]:
            g = mem[name + "_grasp"]
            if _object_on_table(fr, name, g) > 15:
                redo = name
                break
        if redo is not None and int(mem["attempts"]) < 3:
            mem["placed"] = [n for n in mem["placed"] if n != redo]
            # keep order: objects before redo stay placed
            mem["placed"] = [n for n in mem["order"][:mem["order"].index(redo)] if n in mem["placed"]]
            mem["attempts"] = int(mem["attempts"]) + 1
            phase = "pick"
        else:
            phase = "wait"

    mem["phase"] = phase
    out = {k: mem.get(k) for k in ("phase", "handle", "cookies_grasp", "chocolate_grasp", "drawer_open",
                                   "placed", "order", "attempts", "notes", "drawer_level", "decisions",
                                   "handle_now_dy", "drawer_front")}
    if len(out["notes"] or "") > 1500:
        out["notes"] = out["notes"][-1500:]

    if phase == "open_drawer":
        dy = float(mem.get("handle_now_dy") or 0.0)
        act = {"tool": "act_open_drawer", "args": {"handle": [hx, hy + dy, hz]}}
    elif phase == "pick":
        name = mem["order"][len(mem["placed"])]
        g = list(mem[name + "_grasp"])
        jitter = (0.0, 0.007, -0.007, 0.0)[int(mem["attempts"] or 0) % 4]
        g[1] = float(g[1]) + jitter
        act = {"tool": "act_pick", "args": {"grasp": g, "name": name}}
    elif phase == "place":
        name = mem["order"][len(mem["placed"])]
        s = list(SLOT.get(name, SLOT["cookies"]))
        front = mem.get("drawer_front")
        if front is not None and -0.21 < float(front) < -0.02:
            # keep the drop point >= 4.5 cm behind the measured front panel
            s[1] = max(s[1], float(front) + 0.045)
        act = {"tool": "act_place", "args": {"target": [hx + s[0], hy + s[1], hz + s[2]]}}
    else:
        act = {"tool": "act_wait", "args": {"steps": 60}}
    return {"memory": out, "act": act}


# ---------------------------------------------------------------- acts
def act_open_drawer(robot, handle):
    """Approach the drawer handle [x,y,z] with a horizontal gripper, grasp, pull it open, release, back off."""
    hx, hy, hz = [float(v) for v in handle]
    Rh = _r_handle()
    res = {}
    # lift slightly and reorient on the way to a pre-grasp point in front of the handle
    res["pre"] = _servo(robot, [hx, hy - 0.11, hz], Rh, -1.0, tol=0.006, rtol=0.05, max_steps=160)
    res["in"] = _servo(robot, [hx, hy, hz], Rh, -1.0, tol=0.003, rtol=0.04, max_steps=90, gain=10.0, vmax=0.6)
    res["grip"] = _gripper(robot, 1.0, 22)
    res["pull"] = _pull(robot, [hx, hy - PULL_DY, hz], Rh)
    res["release"] = _gripper(robot, -1.0, 15)
    y_now = float(robot.now().tcp[1])
    res["back"] = _servo(robot, [hx, min(y_now, hy - 0.2) - 0.13, hz + 0.01], Rh, -1.0, tol=0.01,
                         rtol=0.1, max_steps=60)
    return res


def act_pick(robot, grasp, name="object"):
    """Top-down grasp of an object at grasp [x,y,z]; lifts it and reports whether it is held."""
    gx, gy, gz = [float(v) for v in grasp]
    Rd = _r_down(0.0)
    res = {"held": None, "aperture": None, "name": name}
    cur = np.asarray(robot.now().tcp, dtype=float)
    # stay high while far away (e.g. coming from the drawer or carrying height)
    if cur[2] > 0.70 or float(np.linalg.norm(cur[:2] - np.array([gx, gy]))) > 0.15:
        res["hover"] = _servo(robot, [gx, gy, max(0.63, min(cur[2], 0.70))], Rd, -1.0, tol=0.008,
                              rtol=0.05, max_steps=220)
    res["above"] = _servo(robot, [gx, gy, 0.53], Rd, -1.0, tol=0.004, rtol=0.03, max_steps=120)
    res["down"] = _servo(robot, [gx, gy, gz], Rd, -1.0, tol=0.003, rtol=0.04, max_steps=110, gain=10.0, vmax=0.6)
    res["aperture_closed"] = _gripper(robot, 1.0, 22)
    res["lift"] = _servo(robot, [gx, gy, 0.75], Rd, 1.0, tol=0.01, rtol=0.06, max_steps=90)
    ap = _aperture(robot.now())
    res["aperture"] = ap
    res["held"] = bool(ap > MIN_HOLD_APERTURE)
    if not res["held"]:
        _gripper(robot, -1.0, 10)
        # clear the camera's view of the table so the object can be re-measured
        res["clear"] = _servo(robot, [0.5 * gx, -0.05, 0.80], Rd, -1.0, tol=0.02, rtol=0.1, max_steps=80)
    return res


def act_place(robot, target):
    """Carry the held object high, descend to target [x,y,z] above the open drawer, release, retreat up."""
    tx, ty, tz = [float(v) for v in target]
    Rd = _r_down(0.0)
    res = {"dropped": False}
    cur = np.asarray(robot.now().tcp, dtype=float)
    res["up"] = _servo(robot, [cur[0], cur[1], CARRY_Z], Rd, 1.0, tol=0.02, rtol=0.06, max_steps=60)
    res["over"] = _servo(robot, [tx, ty, CARRY_Z], Rd, 1.0, tol=0.006, rtol=0.04, max_steps=220)
    res["down"] = _servo(robot, [tx, ty, tz], Rd, 1.0, tol=0.004, rtol=0.04, max_steps=120, gain=10.0, vmax=0.7)
    res["held_before_release"] = _aperture(robot.now())
    res["dropped"] = bool(res["held_before_release"] < MIN_HOLD_APERTURE)
    res["release"] = _gripper(robot, -1.0, 20)
    res["retreat"] = _servo(robot, [tx, ty, CARRY_Z], Rd, -1.0, tol=0.01, rtol=0.08, max_steps=80)
    return res


def act_wait(robot, steps=60):
    """Hold still with the gripper open so the checker can settle; returns the final TCP."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, -1.0)
    return {"waited": int(steps), "tcp": [float(v) for v in robot.now().tcp[:3]]}
