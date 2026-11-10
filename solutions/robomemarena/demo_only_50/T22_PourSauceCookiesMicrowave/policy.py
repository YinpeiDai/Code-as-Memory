"""T22: pour tomato sauce over the cookies twice, then put the cookies into the microwave.

Stage machine over declared memory. Geometry is expressed relative to objects perceived from the
agentview RGB-D frame at the start of the episode (table height, sauce-can centroid, cookie-box
centroid, microwave front/handle edges); the offsets were measured from the expert recordings."""

_PI = math.pi
_RDOWN = [math.pi, 0.0, 0.0]                      # gripper pointing down, fingers along world y
_RDOOR = [2.4183992, -2.4183992, -2.4183992]      # approach +y, fingers along world x (240 deg about (1,-1,-1))
_TILT = 0.80                                      # pour tilt about world +x (rad), as the expert
_SAUCE_OFF = (-0.0120, -0.0103, 0.038)            # grasp TCP - sauce visible centroid (x, y), z above table
_POUR_OFF = (-0.0006, -0.0112, 0.254)             # pour TCP - cookie centroid, z above table
_COOKIE_OFF = (-0.0054, -0.0022, 0.002)           # cookie grasp TCP - cookie centroid, z above table
_HANDLE_OFF = (-0.1243, 0.0144, 0.1515)           # handle TCP - (mw front x, mw front-most y), z above table
_HINGE_OFF = (-0.3822, 0.0978)                    # door hinge (fitted arc centre) - (mw x, mw y)
_HINGE_R = 0.2845
_DOOR_END_DEG = -112.8
_RETREAT_OFF = (-0.047, -0.100, -0.018)        # retreat TCP - handle release TCP
_REORIENT_OFF = (0.008, -0.073, 0.022)          # re-orient-down TCP - handle release TCP
_INSERT_X = -0.175                                 # insertion TCP x - mw front x
_INSERT_Y = -0.005                                 # insertion TCP y - mw front-most y (success seen at -0.045)
_INSERT_Z = 0.166
_CAVITY_X = -0.20                                 # cavity opening centre x - mw front x (frame 900 depth)
_CAVITY_Z = 0.13                                  # cavity opening centre z above table
_MAX_RETRY = 2

_FIELDS = ("stage", "table_z", "sauce", "cookie", "mw_ref", "retries", "pours", "door_angle", "log")


def _pour_count(goal):
    """Number of pours requested by the instruction ("twice" in every assigned recording)."""
    g = (goal or "").lower()
    for word, n in (("three times", 3), ("thrice", 3), ("twice", 2), ("two times", 2), ("once", 1)):
        if word in g:
            return n
    return 2


def _mem(memory):
    m = {k: memory.get(k) for k in _FIELDS} if memory else {k: None for k in _FIELDS}
    if m["retries"] is None:
        m["retries"] = 0
    if m["pours"] is None:
        m["pours"] = 0
    if m["log"] is None:
        m["log"] = ""
    return m


def _note(m, s):
    m["log"] = (m["log"] + "; " + s)[-1500:]


def _rz_rv(base_rv, ang):
    c = math.cos(ang)
    s = math.sin(ang)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return [float(v) for v in _mat2rv(Rz @ _rv2mat(base_rv))]


def _tilt_rv():
    c = math.cos(_TILT)
    s = math.sin(_TILT)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return [float(v) for v in _mat2rv(Rx @ _rv2mat(_RDOWN))]


def _scene_into(m, frame, which):
    sc = _perceive_scene(frame)
    if which in ("all",):
        m["table_z"] = sc["table_z"] if sc["table_ok"] else 0.43
        _note(m, "table %.3f" % m["table_z"])
    if which in ("all", "sauce"):
        if sc["sauce"] is not None:
            m["sauce"] = sc["sauce"]
            _note(m, "sauce %.3f,%.3f n%d" % (sc["sauce"][0], sc["sauce"][1], sc["sauce_n"]))
        elif m["sauce"] is None:
            m["sauce"] = [-0.02, -0.195, m["table_z"] + 0.075]
            _note(m, "sauce fallback")
    if which in ("all", "cookie"):
        if sc["cookie"] is not None:
            m["cookie"] = sc["cookie"]
            _note(m, "cookie %.3f,%.3f n%d" % (sc["cookie"][0], sc["cookie"][1], sc["cookie_n"]))
        elif m["cookie"] is None:
            m["cookie"] = [0.172, -0.194, m["table_z"] + 0.013]
            _note(m, "cookie fallback")
    if which == "all":
        if sc["mw_lo"] is not None:
            m["mw_ref"] = [sc["mw_hi"][0], sc["mw_lo"][1], m["table_z"]]
            _note(m, "mw %.3f,%.3f n%d" % (m["mw_ref"][0], m["mw_ref"][1], sc["mw_n"]))
        else:
            m["mw_ref"] = [0.17, 0.39, m["table_z"]]
            _note(m, "mw fallback")


def _seg(name, pos, rot, grip, **kw):
    d = {"name": name, "pos": None if pos is None else [float(v) for v in pos], "rot": rot, "grip": float(grip)}
    d.update(kw)
    return d


def _sauce_grasp(m):
    s = m["sauce"]
    T = m["table_z"]
    return [s[0] + _SAUCE_OFF[0], s[1] + _SAUCE_OFF[1], T + _SAUCE_OFF[2]]


def _cookie_grasp(m):
    c = m["cookie"]
    T = m["table_z"]
    return [c[0] + _COOKIE_OFF[0], c[1] + _COOKIE_OFF[1], T + _COOKIE_OFF[2]]


def _handle(m):
    r = m["mw_ref"]
    return [r[0] + _HANDLE_OFF[0], r[1] + _HANDLE_OFF[1], r[2] + _HANDLE_OFF[2]]


def _hinge(m):
    r = m["mw_ref"]
    return [r[0] + _HINGE_OFF[0], r[1] + _HINGE_OFF[1]]


def _door_angle_of(m, tcp):
    h = _hinge(m)
    return math.degrees(math.atan2(tcp[1] - h[1], tcp[0] - h[0]))


def _act_for(stage, m, frame, goal=""):
    T = m["table_z"]
    tcp = [float(v) for v in np.asarray(frame.tcp)]
    if stage == "to_sauce":
        g = _sauce_grasp(m)
        return "act_sequence", {"segments": [
            _seg("above", [g[0], g[1], T + 0.24], _RDOWN, -1, tol=0.005, rot_tol=0.04, max_steps=150),
            _seg("down", g, _RDOWN, -1, tol=0.003, rot_tol=0.04, max_steps=150, vmax=0.7)]}
    if stage == "grasp_sauce":
        return "act_move", {"pos": _sauce_grasp(m), "rot": _RDOWN, "grip": 1.0, "tol": 0.004,
                            "rot_tol": 0.1, "max_steps": 60, "hold": 30}
    if stage == "pour":
        g = _sauce_grasp(m)
        c = m["cookie"]
        p = [c[0] + _POUR_OFF[0], c[1] + _POUR_OFF[1], T + _POUR_OFF[2]]
        tilt = _tilt_rv()
        segs = [_seg("lift", [g[0], g[1], T + 0.38], _RDOWN, 1, tol=0.01, rot_tol=0.06, max_steps=120, need_gap=0.02),
                _seg("over_cookies", p, _RDOWN, 1, tol=0.004, rot_tol=0.04, max_steps=150, need_gap=0.02)]
        for i in range(max(0, _pour_count(goal) - int(m["pours"]))):
            segs.append(_seg("tilt%d" % (i + 1), p, tilt, 1, tol=0.012, rot_tol=0.04, max_steps=120, hold=40, need_gap=0.02))
            segs.append(_seg("untilt%d" % (i + 1), p, _RDOWN, 1, tol=0.012, rot_tol=0.04, max_steps=120, hold=25, need_gap=0.02))
        return "act_sequence", {"segments": segs}
    if stage == "place_sauce":
        g = _sauce_grasp(m)
        return "act_sequence", {"segments": [
            _seg("above_place", [g[0], g[1], T + 0.20], _RDOWN, 1, tol=0.006, rot_tol=0.04, max_steps=150),
            _seg("lower", [g[0], g[1], T + 0.040], _RDOWN, 1, tol=0.003, rot_tol=0.05, max_steps=150, vmax=0.4, stall_stop=True, stall_err=0.07),
            _seg("open", None, None, -1, tol=0.05, rot_tol=1.0, max_steps=40, hold=20),
            _seg("lift", [g[0], g[1], T + 0.26], _RDOWN, -1, tol=0.01, rot_tol=0.06, max_steps=120)]}
    if stage == "to_handle":
        h = _handle(m)
        return "act_sequence", {"segments": [
            _seg("pre_handle", [h[0], h[1] - 0.19, h[2] + 0.002], _RDOOR, -1, tol=0.005, rot_tol=0.04, max_steps=220),
            _seg("handle", h, _RDOOR, -1, tol=0.003, rot_tol=0.04, max_steps=150, vmax=0.5)]}
    if stage == "regrasp_handle":
        # door partially open: approach the handle on the arc with a yaw-rotated gripper
        hc = _hinge(m)
        a = math.radians(m["door_angle"])
        a0 = math.radians(-18.0)
        dz = a - a0
        hx = hc[0] + (_HINGE_R - 0.012) * math.cos(a)
        hy = hc[1] + (_HINGE_R - 0.012) * math.sin(a)
        nx, ny = -math.sin(dz), math.cos(dz)
        rot = _rz_rv(_RDOOR, dz)
        z = _handle(m)[2]
        return "act_sequence", {"segments": [
            _seg("back", [tcp[0] - 0.12 * nx, tcp[1] - 0.12 * ny, z], None, -1, tol=0.01, max_steps=80),
            _seg("pre", [hx - 0.15 * nx, hy - 0.15 * ny, z], rot, -1, tol=0.006, rot_tol=0.05, max_steps=200),
            _seg("in", [hx, hy, z], rot, -1, tol=0.004, rot_tol=0.05, max_steps=150, vmax=0.4)]}
    if stage == "grasp_handle":
        tgt = _handle(m) if m["retries"] == 0 else tcp[:3]
        return "act_move", {"pos": tgt, "rot": None, "grip": 1.0, "tol": 0.05, "rot_tol": 1.0,
                            "max_steps": 40, "hold": 22}
    if stage == "pull_door":
        hc = _hinge(m)
        a = _door_angle_of(m, tcp)
        z = _handle(m)[2]
        pts = []
        ang = a - 8.0
        while ang > _DOOR_END_DEG:
            r = math.radians(ang)
            pts.append([hc[0] + _HINGE_R * math.cos(r), hc[1] + _HINGE_R * math.sin(r), z])
            ang -= 8.0
        r = math.radians(_DOOR_END_DEG)
        pts.append([hc[0] + _HINGE_R * math.cos(r), hc[1] + _HINGE_R * math.sin(r), z])
        # orientation is left free (zero rotation command), as the expert did while pulling
        return "act_path", {"points": pts, "rot": None, "grip": 1.0, "pass_tol": 0.02, "tol": 0.012,
                            "max_steps": 400, "vmax": 0.8, "min_gap": 0.006}
    if stage == "leave_door":
        # release the handle, slide the fingers off it (-y, along the opened door face), back away
        # from the door and turn the gripper down again (offsets from the expert, relative to release)
        x, y, z = tcp[0], tcp[1], tcp[2]
        return "act_sequence", {"segments": [
            _seg("open", None, None, -1, tol=0.03, rot_tol=1.0, max_steps=30, hold=15),
            _seg("slide_off", [x - 0.01, y - 0.07, z], None, -1, tol=0.008, max_steps=80, vmax=0.6),
            _seg("retreat", [x + _RETREAT_OFF[0], y + _RETREAT_OFF[1], z + _RETREAT_OFF[2]], None, -1, tol=0.008, max_steps=100, vmax=0.6),
            _seg("reorient", [x + _REORIENT_OFF[0], y + _REORIENT_OFF[1], z + _REORIENT_OFF[2]], _RDOWN, -1, tol=0.01, rot_tol=0.05, max_steps=150)]}
    if stage == "to_cookie":
        g = _cookie_grasp(m)
        return "act_sequence", {"segments": [
            _seg("above", [g[0], g[1], T + 0.206], _RDOWN, -1, tol=0.005, rot_tol=0.04, max_steps=200),
            _seg("down", g, _RDOWN, -1, tol=0.003, rot_tol=0.04, max_steps=150, vmax=0.7, stall_stop=True)]}
    if stage == "grasp_cookie":
        return "act_move", {"pos": tcp[:3], "rot": _RDOWN, "grip": 1.0, "tol": 0.05, "rot_tol": 0.2,
                            "max_steps": 50, "hold": 30}
    if stage == "insert":
        g = _cookie_grasp(m)
        r = m["mw_ref"]
        x = r[0] + _INSERT_X
        return "act_sequence", {"segments": [
            _seg("lift", [g[0], g[1], T + 0.296], _RDOWN, 1, tol=0.01, rot_tol=0.06, max_steps=120, need_gap=0.03),
            _seg("carry", [g[0] - 0.146, g[1] + 0.193, T + 0.294], _RDOWN, 1, tol=0.008, rot_tol=0.06, max_steps=150, need_gap=0.03),
            _seg("turn", [g[0] - 0.146, g[1] + 0.193, T + 0.294], _RDOOR, 1, tol=0.01, rot_tol=0.04, max_steps=150, need_gap=0.03),
            _seg("pre_insert", [x, r[1] - 0.16, T + 0.18], _RDOOR, 1, tol=0.006, rot_tol=0.04, max_steps=150, need_gap=0.03),
            _seg("insert", [x, r[1] + _INSERT_Y, T + _INSERT_Z], _RDOOR, 1, tol=0.004, rot_tol=0.04, max_steps=150, vmax=0.5, hold=40, need_gap=0.03)]}
    if stage == "deeper":
        # fallback when holding at the opening did not end the episode: centre the box on the
        # cavity opening seen in the recordings (x ref-0.30..-0.12, z table+0.07..+0.19), push it
        # in, release it inside and back off
        r = m["mw_ref"]
        x = r[0] + _CAVITY_X
        z = T + _CAVITY_Z
        return "act_sequence", {"segments": [
            _seg("align", [x, r[1] - 0.01, z], _RDOOR, 1, tol=0.005, rot_tol=0.05, max_steps=120, vmax=0.4),
            _seg("push", [x, r[1] + 0.12, z], _RDOOR, 1, tol=0.005, rot_tol=0.05, max_steps=150, vmax=0.3, hold=30, stall_stop=True, stall_err=0.2),
            _seg("release", None, None, -1, tol=0.05, rot_tol=1.0, max_steps=40, hold=25),
            _seg("back_off", [x, r[1] - 0.15, z + 0.01], _RDOOR, -1, tol=0.01, rot_tol=0.1, max_steps=150, vmax=0.4)]}
    if stage == "reopen":
        # failed grasp: open and move up out of the way of the agentview camera before re-perceiving
        return "act_sequence", {"segments": [
            _seg("open", None, None, -1, tol=0.03, rot_tol=1.0, max_steps=25, hold=15),
            _seg("up", [tcp[0], tcp[1], T + 0.26], None, -1, tol=0.01, max_steps=100),
            _seg("clear", [-0.25, 0.0, T + 0.30], _RDOWN, -1, tol=0.02, rot_tol=0.1, max_steps=150)]}
    return "act_observe", {"steps": 40, "grip": 0.0}


def _gap_of(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _next_stage(m, obs, goal=""):
    """Check the outcome of the stage just executed (from the frames it produced) and pick the next."""
    st = m["stage"]
    f = obs.current
    gap = _gap_of(f)
    gaps = [_gap_of(x) for x in obs.frames]
    tcp = [float(v) for v in np.asarray(f.tcp)]
    if st == "to_sauce":
        return "grasp_sauce"
    if st == "grasp_sauce":
        if gap < 0.02 and m["retries"] < _MAX_RETRY:
            m["retries"] += 1
            m["log_target"] = "sauce"
            _note(m, "sauce grasp missed gap %.3f" % gap)
            return "reopen:sauce"
        return "pour"
    if st == "pour":
        # count completed tilts: a tilt counts when the gap stayed > 0.02 through it
        if gap < 0.02 and m["retries"] < _MAX_RETRY:
            m["retries"] += 1
            _note(m, "sauce lost during pour gap %.3f" % gap)
            return "reopen:sauce"
        m["pours"] = _pour_count(goal)  # all requested tilts executed (sequence not aborted)
        return "place_sauce"
    if st == "place_sauce":
        return "to_handle"
    if st in ("to_handle", "regrasp_handle"):
        return "grasp_handle"
    if st == "grasp_handle":
        if gap < 0.006 and m["retries"] < _MAX_RETRY + 1:
            m["retries"] += 1
            _note(m, "handle grasp missed gap %.3f" % gap)
            m["door_angle"] = _door_angle_of(m, tcp)
            return "regrasp_handle" if m["door_angle"] < -25.0 else "to_handle"
        return "pull_door"
    if st == "pull_door":
        ang = _door_angle_of(m, tcp)
        m["door_angle"] = ang
        lost = gap < 0.006 or (len(gaps) > 4 and min(gaps[3:]) < 0.006)
        _note(m, "pull end angle %.1f lost %s" % (ang, lost))
        if lost and ang > _DOOR_END_DEG + 15.0 and m["retries"] < _MAX_RETRY + 1:
            m["retries"] += 1
            return "regrasp_handle" if ang < -25.0 else "to_handle"
        if (not lost) and ang > -85.0 and m["retries"] < _MAX_RETRY + 1:
            # still holding but the door is not open far enough for the insertion path: pull again
            m["retries"] += 1
            return "pull_door"
        return "leave_door"
    if st == "leave_door":
        return "to_cookie"
    if st == "to_cookie":
        return "grasp_cookie"
    if st == "grasp_cookie":
        if gap < 0.03 and m["retries"] < _MAX_RETRY + 2:
            m["retries"] += 1
            _note(m, "cookie grasp missed gap %.3f" % gap)
            return "reopen:cookie"
        return "insert"
    if st == "insert":
        _note(m, "insert done gap %.3f y %.3f" % (gap, tcp[1]))
        if gap < 0.03 and m["retries"] < _MAX_RETRY + 2:
            m["retries"] += 1
            return "reopen:cookie"
        return "deeper"
    if st and st.startswith("reopen"):
        return "reperceive:" + st.split(":")[1]
    return "idle"


def perceive_decide(obs, goal, memory, last):
    """Perceive the scene on the first call, then run the pour/door/insert stage machine.
    Each decision checks the finger gap / TCP from the frames of the previous action."""
    m = _mem(memory)
    f = obs.current
    if m["stage"] is None:
        _scene_into(m, f, "all")
        stage = "to_sauce"
    else:
        stage = _next_stage(m, obs, goal)
        if stage.startswith("reperceive:"):
            which = stage.split(":")[1]
            _scene_into(m, f, which)
            stage = "to_sauce" if which == "sauce" else "to_cookie"
            if which == "sauce":
                m["pours"] = 0
        elif stage.startswith("reopen:"):
            pass
    m.pop("log_target", None)
    m["stage"] = stage
    tool, args = _act_for(stage.split(":")[0], m, f, goal)
    return {"memory": {k: m[k] for k in _FIELDS}, "act": {"tool": tool, "args": args}}
