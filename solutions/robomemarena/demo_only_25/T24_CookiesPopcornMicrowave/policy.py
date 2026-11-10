"""T24 CookiesPopcornMicrowave: open the microwave, put the cookies in, then put the popcorn where the
cookies were placed (inside the microwave). Waypoints are expressed relative to perceived anchors
(handle from the microwave point cloud, grasp points of the two boxes) and executed with the
expert-style P-controller (control.py). Memory keeps the anchors, the cookie placement and progress."""

HOME = [-0.058, 0.0, 0.671]
# Waypoint offsets fitted from the expert's implied targets (scratch/targets.py), relative to anchors.
H_PRE = [0.0015, -0.136, 0.0]          # in front of the handle, fingers pointing +y
H_IN = [0.0015, 0.0139, 0.0]           # handle between the fingers
H_PULL = [-0.4985, -0.1861, 0.0]       # pull target (x saturates; the door arc limits travel)
OBJ_ABOVE = [0.0045, -0.0012, 0.2452]  # above a box (relative to grasp point)
OBJ_DOWN = [0.0045, -0.0012, -0.0248]  # press-down target (lands at the grasp point after 30 steps)
OBJ_LIFT = [0.0045, -0.0012, 0.1252]
C_STAGE = [-0.1467, 0.1983, 0.1199]    # cookie carry staging point
C_ROT = [0.0033, -0.0017, 0.1199]      # rotate to side orientation while moving here
P_STAGE = [-0.1992, 0.1958, 0.1206]    # popcorn staging (rotation in place)
INSERT = [-0.0485, 0.044, 0.020]       # insertion waypoint inside the microwave (relative to handle)
PUSH = [0.001, 0.094, -0.010]          # push the box down/deeper before releasing
RETREAT = [0.0005, -0.1065, -0.0101]
RISE = [0.0005, -0.1065, 0.1399]
GAP_EMPTY = 0.008    # finger gap (m) below which the gripper closed on nothing
DOOR_MIN_POINTS = 150

FIELDS = ("phase", "handle", "cookies", "popcorn", "cookie_place", "table_z", "tries", "door_tries", "log")


def _add(a, b):
    return [float(a[0]) + b[0], float(a[1]) + b[1], float(a[2]) + b[2]]


def _seg(name, target, rot, grip, steps):
    return {"name": name, "target": None if target is None else [float(v) for v in target],
            "rot": rot, "grip": grip, "steps": int(steps)}


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _door_segments(H):
    return [_seg("handle_pre", _add(H, H_PRE), "side", -1, 30),
            _seg("handle_in", _add(H, H_IN), "side", -1, 30),
            _seg("handle_close", None, None, 1, 20)]


def _out(mem, tool, args):
    m = {k: mem.get(k) for k in FIELDS}
    return {"memory": m, "act": {"tool": tool, "args": args}}


def perceive_decide(obs, goal, memory, last):
    """Phase machine. First call perceives the scene; later calls verify grasps / door state and advance."""
    mem = {k: (memory.get(k) if isinstance(memory, dict) else None) for k in FIELDS}
    cur = obs.current
    phase = mem["phase"]
    if mem["tries"] is None:
        mem["tries"] = 0
    if mem["door_tries"] is None:
        mem["door_tries"] = 0
    if mem["log"] is None:
        mem["log"] = ""

    if phase is None:
        sc = _scene(cur)
        mem["table_z"] = float(sc["table_z"])
        notes = []
        if sc["handle"] is None:
            mem["handle"] = list(DEFAULT_HANDLE)
            notes.append("microwave not found: default handle")
        else:
            mem["handle"] = [float(v) for v in sc["handle"]]
        b = sc["boxes"]
        if b is None:
            mem["cookies"] = list(DEFAULT_COOKIES)
            mem["popcorn"] = list(DEFAULT_POPCORN)
            notes.append("boxes not found: defaults")
        else:
            mem["cookies"] = [float(v) for v in b["cookies"]]
            mem["popcorn"] = [float(v) for v in b["popcorn"]]
            notes.append("boxes by " + b["how"])
        mem["log"] = "; ".join(notes)
        mem["phase"] = "door_grasped"
        mem["tries"] = 0
        return _out(mem, "act_run", {"segments": _door_segments(mem["handle"])})

    H = mem["handle"]
    C = mem["cookies"]
    P = mem["popcorn"]

    if phase == "door_grasped":
        if _gap(cur) < GAP_EMPTY and mem["tries"] < 2:
            mem["tries"] = mem["tries"] + 1
            mem["log"] = mem["log"] + "; handle missed, retry"
            segs = [_seg("open", None, None, -1, 10)] + _door_segments(H)
            return _out(mem, "act_run", {"segments": segs})
        mem["phase"] = "door_check"
        mem["tries"] = 0
        segs = [_seg("pull", _add(H, H_PULL), "side", 1, 40),
                _seg("release", None, None, -1, 10),
                _seg("above_cookies", _add(C, OBJ_ABOVE), "down", -1, 40)]
        return _out(mem, "act_run", {"segments": segs})

    if phase == "door_check":
        n = _door_open_points(cur, H)
        if n < DOOR_MIN_POINTS and mem["door_tries"] < 1:
            mem["door_tries"] = mem["door_tries"] + 1
            mem["tries"] = 0
            mem["log"] = mem["log"] + "; door not open (%d pts), retry" % n
            mem["phase"] = "door_grasped"
            return _out(mem, "act_run", {"segments": _door_segments(H)})
        mem["phase"] = "cookie_grasped"
        mem["tries"] = 0
        segs = [_seg("cookie_down", _add(C, OBJ_DOWN), "down", -1, 30),
                _seg("cookie_close", None, None, 1, 20)]
        return _out(mem, "act_run", {"segments": segs})

    if phase in ("cookie_grasped", "popcorn_grasped"):
        obj = C if phase == "cookie_grasped" else P
        name = "cookie" if phase == "cookie_grasped" else "popcorn"
        if _gap(cur) < GAP_EMPTY and mem["tries"] < 2:
            mem["tries"] = mem["tries"] + 1
            mem["log"] = mem["log"] + "; %s grasp missed, re-perceive" % name
            mem["phase"] = name + "_reperceive"
            segs = [_seg("open", None, None, -1, 10),
                    _seg("lift", _add(obj, OBJ_ABOVE), "down", -1, 30),
                    _seg("home", HOME, "down", -1, 40)]
            return _out(mem, "act_run", {"segments": segs})
        mem["phase"] = name + "_lifted"
        return _out(mem, "act_run", {"segments": [_seg(name + "_lift", _add(obj, OBJ_LIFT), "down", 1, 40)]})

    if phase in ("cookie_reperceive", "popcorn_reperceive"):
        name = "cookie" if phase == "cookie_reperceive" else "popcorn"
        key = "cookies" if name == "cookie" else "popcorn"
        if name == "cookie":
            b = _scene(cur)["boxes"]
            if b is not None:
                mem["cookies"] = [float(v) for v in b["cookies"]]
                mem["popcorn"] = [float(v) for v in b["popcorn"]]
            else:
                mem["log"] = mem["log"] + "; re-perception failed"
        else:
            r = _single_box(cur, mem["popcorn"], mem["table_z"], 0.08)
            if r is not None:
                mem["popcorn"] = r
            else:
                mem["log"] = mem["log"] + "; re-perception failed"
        obj = mem[key]
        mem["phase"] = name + "_grasped"
        segs = [_seg(name + "_above", _add(obj, OBJ_ABOVE), "down", -1, 40),
                _seg(name + "_down", _add(obj, OBJ_DOWN), "down", -1, 30),
                _seg(name + "_close", None, None, 1, 20)]
        return _out(mem, "act_run", {"segments": segs})

    if phase == "cookie_lifted":
        if _gap(cur) < GAP_EMPTY and mem["tries"] < 2:
            # dropped while lifting: look again from home and regrasp
            mem["tries"] = mem["tries"] + 1
            mem["log"] = mem["log"] + "; cookies dropped, re-perceive"
            mem["phase"] = "cookie_reperceive"
            return _out(mem, "act_run", {"segments": [_seg("home", HOME, "down", -1, 40)]})
        place = _add(H, INSERT)
        mem["cookie_place"] = place          # remembered: where the cookies are put
        mem["phase"] = "popcorn_check"
        mem["tries"] = 0
        segs = [_seg("cookie_stage", _add(C, C_STAGE), "down", 1, 40),
                _seg("cookie_rotate", _add(C, C_ROT), "side", 1, 30),
                _seg("cookie_insert", place, "side", 1, 75),
                _seg("cookie_push", _add(H, PUSH), "side", 1, 25),
                _seg("cookie_release", None, None, -1, 20),
                _seg("retreat", _add(H, RETREAT), "side", -1, 30),
                _seg("rise", _add(H, RISE), "side", -1, 30)]
        return _out(mem, "act_run", {"segments": segs})

    if phase == "popcorn_check":
        r = _single_box(cur, P, mem["table_z"])
        if r is not None and abs(r[0] - P[0]) < 0.04 and abs(r[1] - P[1]) < 0.04:
            mem["popcorn"] = r
            P = r
        else:
            mem["log"] = mem["log"] + "; popcorn refine failed"
        mem["phase"] = "popcorn_grasped"
        mem["tries"] = 0
        segs = [_seg("popcorn_above", _add(P, OBJ_ABOVE), "down", -1, 40),
                _seg("popcorn_down", _add(P, OBJ_DOWN), "down", -1, 30),
                _seg("popcorn_close", None, None, 1, 20)]
        return _out(mem, "act_run", {"segments": segs})

    if phase == "popcorn_lifted":
        if _gap(cur) < GAP_EMPTY and mem["tries"] < 2:
            mem["tries"] = mem["tries"] + 1
            mem["log"] = mem["log"] + "; popcorn dropped, re-perceive"
            mem["phase"] = "popcorn_reperceive"
            return _out(mem, "act_run", {"segments": [_seg("home", HOME, "down", -1, 40)]})
        place = mem["cookie_place"]
        if place is None:
            place = _add(H, INSERT)
        mem["phase"] = "popcorn_release"
        segs = [_seg("popcorn_stage", _add(P, P_STAGE), "down", 1, 40),
                _seg("popcorn_rotate", _add(P, P_STAGE), "side", 1, 40),
                _seg("popcorn_insert", place, "side", 1, 80)]
        return _out(mem, "act_run", {"segments": segs})

    if phase == "popcorn_release":
        # success normally fires during the insertion; otherwise let go inside and back out
        mem["phase"] = "done"
        segs = [_seg("popcorn_release", None, None, -1, 20),
                _seg("retreat", _add(H, RETREAT), "side", -1, 30),
                _seg("rise", _add(H, RISE), "side", -1, 30)]
        return _out(mem, "act_run", {"segments": segs})

    # done: keep still while the checker settles
    mem["phase"] = "done"
    return _out(mem, "act_hold", {"steps": 100, "grip": -1})
