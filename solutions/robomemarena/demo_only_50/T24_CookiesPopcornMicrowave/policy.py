"""Code-as-Memory policy for T24_CookiesPopcornMicrowave (LIBERO / RoboMemArena).

Task: open the microwave door, put the first-named object (cookies) into the
microwave, then put the second-named object (popcorn) where the first was
placed (the microwave).  Structure and constants were derived from the 50
assigned expert recordings (see task_analysis.md):
  * the expert is a waypoint controller: pos = clip(6*(target-tcp), -1, 1),
    rot = clip(2*axis_angle(R_target R^T), -0.5, 0.5), fixed segment lengths;
  * only two gripper orientations are used: DOWN (top grasp) and SIDE
    (gripper axis pointing +y, into the microwave);
  * waypoints are fixed offsets from three perceived anchors: the microwave
    handle point M, and the two object grasp points.
"""

# ---------------------------------------------------------------- constants
MW_CX = -0.12257      # handle-target x minus 99.5th pct x of microwave points
MW_CY = 0.02933       # handle-target y minus 0.5th pct y of microwave points
MW_Z = 0.582          # handle-target z (constant in all recordings)
OBJ_DX = -0.0039      # grasp x minus centroid x of visible object points
OBJ_DY = -0.0008      # grasp y minus centroid y
NOMINAL = {"microwave": [0.048, 0.428, 0.582], "cookies": [0.168, -0.095],
           "popcorn": [0.166, -0.196]}
# colour signature of each known object as seen in agentview (red-blue)
REDNESS_SIGN = {"cookies": 1.0, "popcorn": -1.0}
KP_POS = 6.0
KP_ROT = 2.0
R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
DOOR_OPEN_MIN = 150   # door-region dark points meaning "door open"
HOLD_MIN = 0.008      # finger opening (m) above which a closed gripper holds something
MEMORY_KEYS = ["phase", "first_obj", "second_obj", "microwave", "first_xy",
               "second_xy", "cookies_placed_at", "retries", "notes"]


# ---------------------------------------------------------------- geometry
def _aa2R(v):
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(th)
    if s < 1e-6:  # near pi: use the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return ax * th
    return w / (2 * s) * th


def _backproject(frame):
    d = np.asarray(frame.agentview_depth, float) / 1000.0
    K = np.asarray(frame.agentview_K, float)
    E = np.asarray(frame.agentview_extrinsic, float)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    Pc = np.stack([x, y, d], -1)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    return Pw, d > 0


# ---------------------------------------------------------------- perception
def _perceive_microwave(frame):
    """Handle grasp point from the closed microwave's front face and door-side face."""
    Pw, ok = _backproject(frame)
    im = np.asarray(frame.agentview_rgb).astype(int)
    m = ok & (Pw[..., 2] > 0.45) & (Pw[..., 2] < 0.8) & (Pw[..., 1] > 0.15) \
        & (Pw[..., 0] > -0.4) & (Pw[..., 0] < 0.5) & (im.max(-1) < 90)
    n = int(m.sum())
    if n < 300:
        return list(NOMINAL["microwave"]), n
    P = Pw[m]
    x = float(np.percentile(P[:, 0], 99.5)) + MW_CX
    y = float(np.percentile(P[:, 1], 0.5)) + MW_CY
    return [x, y, MW_Z], n


def _perceive_objects(frame):
    """Two small boxes lying on the table: returns list of (centroid_xy, redness, npts)."""
    Pw, ok = _backproject(frame)
    im = np.asarray(frame.agentview_rgb).astype(int)
    m = ok & (Pw[..., 2] > 0.437) & (Pw[..., 2] < 0.5) & (Pw[..., 1] > -0.45) \
        & (Pw[..., 1] < 0.12) & (Pw[..., 0] > -0.1) & (Pw[..., 0] < 0.45)
    P = Pw[m]
    C = im[m]
    if len(P) < 10:
        return []
    groups = []
    for axis in (1, 0):
        order = np.argsort(P[:, axis])
        vals = P[order, axis]
        gaps = np.diff(vals)
        if len(gaps) and float(gaps.max()) > 0.02:
            cut = int(np.argmax(gaps)) + 1
            groups = [order[:cut], order[cut:]]
            break
    if not groups:
        groups = [np.arange(len(P))]
    out = []
    for g in groups:
        if len(g) < 5:
            continue
        xy = P[g, :2].mean(0)
        red = float((C[g, 0] - C[g, 2]).mean())
        out.append(([float(xy[0]) + OBJ_DX, float(xy[1]) + OBJ_DY], red, int(len(g))))
    return out


def _door_open_count(frame, M):
    """Dark points where the opened door stands (about 0.4 m behind the closed
    microwave's front face, beside its door side). 0 in every recording before the
    pull; 319-339 after it."""
    Pw, ok = _backproject(frame)
    im = np.asarray(frame.agentview_rgb).astype(int)
    xf = M[0] - MW_CX
    yl = M[1] - MW_CY
    m = ok & (Pw[..., 2] > 0.47) & (Pw[..., 2] < 0.66) & (im.max(-1) < 90) \
        & (Pw[..., 0] > xf - 0.46) & (Pw[..., 0] < xf - 0.33) \
        & (Pw[..., 1] > yl - 0.21) & (Pw[..., 1] < yl - 0.02)
    return int(m.sum())


def _assign(objs, first, second, prev):
    """Map perceived clusters to object names by colour (cookies red, popcorn blue)."""
    res = {first: None, second: None}
    if len(objs) >= 2:
        objs = sorted(objs, key=lambda o: -o[2])[:2]
        s1 = REDNESS_SIGN.get(first, 1.0)
        a, b = objs[0], objs[1]
        if (a[1] - b[1]) * s1 >= 0:
            res[first], res[second] = a[0], b[0]
        else:
            res[first], res[second] = b[0], a[0]
    elif len(objs) == 1:
        # one cluster: give it to whichever name its colour matches best
        o = objs[0]
        name = first if o[1] * REDNESS_SIGN.get(first, 1.0) > 0 else second
        res[name] = o[0]
    for k in res:
        if res[k] is None:
            res[k] = (prev or {}).get(k) or list(NOMINAL.get(k, [0.167, -0.145]))
    return res


def _parse_goal(goal):
    g = (goal or "").lower()
    names = ["cookies", "popcorn"]
    found = sorted([(g.find(n), n) for n in names if g.find(n) >= 0])
    if len(found) == 2:
        return found[0][1], found[1][1]
    return "cookies", "popcorn"


def _nearest(objs, ref):
    if not objs:
        return None
    best = min(objs, key=lambda o: (o[0][0] - ref[0]) ** 2 + (o[0][1] - ref[1]) ** 2)
    if (best[0][0] - ref[0]) ** 2 + (best[0][1] - ref[1]) ** 2 > 0.06 ** 2:
        return None
    return best[0]


# ---------------------------------------------------------------- plans
def _seg(tgt, rot, grip, n, name):
    return {"t": tgt, "r": rot, "g": grip, "n": n, "name": name}


def _plan_door(M):
    x, y, z = M
    return [_seg([x, y - 0.15, z], "side", -1, 30, "pre_handle"),
            _seg([x, y, z], "side", -1, 30, "to_handle"),
            _seg(None, None, 1, 20, "grip_handle"),
            _seg([x - 0.5, y - 0.2, z], "side", 1, 40, "pull_door"),
            _seg(None, None, -1, 10, "release_handle")]


def _plan_pick(xy):
    x, y = xy
    return [_seg([x, y, 0.6834], "down", -1, 40, "above_obj"),
            _seg([x, y, 0.4134], "down", -1, 30, "descend"),
            _seg(None, None, 1, 20, "close")]


def _plan_place_first(xy, M):
    x, y = xy
    mx, my, mz = M
    return [_seg([x, y, 0.5634], "down", 1, 40, "lift"),
            _seg([x - 0.1513, y + 0.1995, 0.5581], "down", 1, 40, "carry"),
            _seg([x - 0.0013, y - 0.0005, 0.5581], "side", 1, 30, "turn_side"),
            _seg([mx - 0.05, my + 0.03, 0.602], "side", 1, 75, "mw_front"),
            _seg([mx - 0.0005, my + 0.08, 0.572], "side", 1, 25, "insert"),
            _seg(None, None, -1, 20, "release")]


def _plan_retreat(M):
    mx, my, mz = M
    return [_seg([mx - 0.0007, my - 0.1204, 0.5719], "side", -1, 30, "back_out"),
            _seg([mx - 0.0007, my - 0.1204, 0.7219], "side", -1, 30, "raise")]


def _plan_place_second(xy, goal_pt):
    x, y = xy
    return [_seg([x, y, 0.5637], "down", 1, 40, "lift"),
            _seg([x - 0.2014, y + 0.1997, 0.5584], "down", 1, 40, "carry"),
            _seg([x - 0.2014, y + 0.1997, 0.5584], "side", 1, 40, "turn_side"),
            _seg(list(goal_pt), "side", 1, 90, "insert")]


def _out(mem, tool, args):
    full = {k: mem.get(k) for k in MEMORY_KEYS}
    return {"memory": full, "act": {"tool": tool, "args": args}}


# ---------------------------------------------------------------- decide
def perceive_decide(obs, goal, memory, last):
    """Phase machine. Memory keeps perceived anchors (microwave handle, object grasp
    points), where the first object was released, the phase and retry counts."""
    mem = dict(memory or {})
    for k in MEMORY_KEYS:
        mem.setdefault(k, None)
    f = obs.current
    phase = mem["phase"] or "start"
    retries = int(mem["retries"] or 0)
    fing = float(np.abs(np.asarray(f.fingers, float)).mean())

    if phase == "start":
        first, second = _parse_goal(goal)
        M, n = _perceive_microwave(f)
        objs = _perceive_objects(f)
        asg = _assign(objs, first, second, None)
        mem.update(first_obj=first, second_obj=second, microwave=[round(v, 5) for v in M],
                   first_xy=asg[first], second_xy=asg[second], retries=0,
                   notes="mw_pts=%d objs=%d" % (n, len(objs)))
        mem["phase"] = "door"
        return _out(mem, "act_segments", {"segs": _plan_door(M), "tag": "door"})

    M = mem["microwave"] or NOMINAL["microwave"]
    if phase == "door":
        n_open = _door_open_count(f, M)
        if n_open < DOOR_OPEN_MIN and retries < 2:
            # door did not swing open: go back to the handle and pull again
            mem["retries"] = retries + 1
            mem["notes"] = "door not open (%d pts), retry %d" % (n_open, retries + 1)
            return _out(mem, "act_segments", {"segs": _plan_door(M), "tag": "door_retry"})
        objs = _perceive_objects(f)
        for key in ("first_xy", "second_xy"):
            xy = _nearest(objs, mem[key])
            if xy is not None:
                mem[key] = xy
        mem["notes"] = "door_pts=%d objs=%d" % (n_open, len(objs))
        mem["retries"] = 0
        mem["phase"] = "pick_first"
        return _out(mem, "act_segments", {"segs": _plan_pick(mem["first_xy"]), "tag": "pick_first"})

    if phase in ("pick_first", "pick_second"):
        which = "first_xy" if phase == "pick_first" else "second_xy"
        if fing > HOLD_MIN or retries >= 2:
            if phase == "pick_first":
                mem["phase"] = "place_first"
                return _out(mem, "act_segments",
                            {"segs": _plan_place_first(mem["first_xy"], M), "tag": "place_first"})
            goal_pt = _second_goal(mem)
            mem["phase"] = "place_second"
            return _out(mem, "act_segments",
                        {"segs": _plan_place_second(mem["second_xy"], goal_pt), "tag": "place_second"})
        # missed grasp: open, lift and re-perceive the object, then try again
        mem["retries"] = retries + 1
        mem["phase"] = "regrasp_" + phase
        xy = mem[which]
        segs = [_seg(None, None, -1, 15, "open"),
                _seg([xy[0], xy[1], 0.70], "down", -1, 30, "lift_clear")]
        return _out(mem, "act_segments", {"segs": segs, "tag": "lift_for_regrasp"})

    if phase.startswith("regrasp_"):
        base = phase[len("regrasp_"):]
        which = "first_xy" if base == "pick_first" else "second_xy"
        objs = _perceive_objects(f)
        xy = _nearest(objs, mem[which])
        if xy is not None:
            mem[which] = xy
        mem["notes"] = "regrasp %s at %s" % (base, mem[which])
        mem["phase"] = base
        return _out(mem, "act_segments", {"segs": _plan_pick(mem[which]), "tag": base + "_retry"})

    if phase == "place_first":
        tcp = np.asarray(f.tcp, float)[:3]
        mem["cookies_placed_at"] = [round(float(v), 5) for v in tcp]
        mem["phase"] = "retreat"
        return _out(mem, "act_segments", {"segs": _plan_retreat(M), "tag": "retreat"})

    if phase == "retreat":
        mem["phase"] = "pick_second"
        mem["retries"] = 0
        return _out(mem, "act_segments", {"segs": _plan_pick(mem["second_xy"]), "tag": "pick_second"})

    if phase == "place_second":
        # not yet scored: push a little deeper, then let go and back out
        g = _second_goal(mem)
        mem["phase"] = "release_second"
        segs = [_seg([g[0], g[1] + 0.03, g[2]], "side", 1, 30, "deeper"),
                _seg(None, None, -1, 25, "open")]
        return _out(mem, "act_segments", {"segs": segs, "tag": "deeper_release"})

    if phase == "release_second":
        g = _second_goal(mem)
        mem["phase"] = "done"
        segs = [_seg([g[0], g[1] - 0.15, g[2]], "side", -1, 40, "back_out")]
        return _out(mem, "act_segments", {"segs": segs, "tag": "back_out"})

    mem["phase"] = "done"
    return _out(mem, "act_segments", {"segs": [_seg(None, None, 0, 40, "wait")], "tag": "wait"})


def _second_goal(mem):
    """Where the second object goes: the place where the first one was released."""
    M = mem.get("microwave") or NOMINAL["microwave"]
    rel = mem.get("cookies_placed_at")
    if rel is not None and abs(rel[0] - (M[0] - 0.0299)) < 0.03 and abs(rel[1] - (M[1] + 0.0397)) < 0.03:
        return [rel[0] - 0.0211, rel[1] - 0.0101, 0.6016]
    return [M[0] - 0.051, M[1] + 0.0296, 0.6016]


# ---------------------------------------------------------------- act
def _command(tcp, seg):
    if seg["t"] is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p = np.asarray(tcp[:3], float)
    dp = np.clip(KP_POS * (np.asarray(seg["t"], float) - p), -1.0, 1.0)
    Rt = np.array(R_SIDE if seg["r"] == "side" else R_DOWN, float)
    R = _aa2R(tcp[3:6])
    dr = np.clip(KP_ROT * _R2aa(Rt @ R.T), -0.5, 0.5)
    return [float(v) for v in dp] + [float(v) for v in dr]


def act_segments(robot, segs, tag=""):
    """Run waypoint segments with the expert's P-law. Each seg is
    {"t": [x,y,z] or None (hold still), "r": "down"|"side", "g": -1 open/+1 close/0,
    "n": steps, "name": str}. Returns final tcp, finger opening and step count."""
    steps = 0
    for seg in segs:
        for _ in range(int(seg["n"])):
            f = robot.now()
            cmd = _command(np.asarray(f.tcp, float), seg)
            robot.step(*cmd, float(seg["g"]))
            steps += 1
    f = robot.now()
    return {"tag": tag, "steps": steps,
            "tcp": [round(float(v), 4) for v in np.asarray(f.tcp, float)[:3]],
            "fingers": [round(float(v), 4) for v in np.asarray(f.fingers, float)]}
