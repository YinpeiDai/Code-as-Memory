"""Task policy: put butter in the frypan, pour the sauce bottle over it twice, then place the bottle in the drainer.

Stage machine kept in declared memory; every transition re-checks measured evidence (finger width, TCP, agentview
RGB-D) rather than trusting that a commanded action succeeded."""

DEFAULT_TABLE_Z = 0.4308
BUTTER_PRIOR = [0.109, -0.203]       # only used if the butter is never detected (demo observation: fixed)
PAN_PRIOR = [0.0, 0.248]
DRAINER_PRIOR = [-0.055, 0.54]
NEUTRAL = [-0.05, 0.0, 0.70]
MAX_PICK_TRIES = 3
BUTTER_RELEASE_DZ = 0.212            # demo: butter released 21 cm above the table over the pan centre
BOTTLE_POUR_DZ = 0.30                # demo: pour TCP height ~0.72-0.75 (table 0.431)
POUR_DX = 0.01
DIAG_BUTTER_PLACE_DY = 0.0           # diagnostic only: y offset of the FIRST butter release (0.0 deployed)
DIAG_FIRST_TRY_DX = 0.0              # diagnostic only: x offset on the first pick tries (0.0 in deployed code)


def _orange_near(frame, c, radius, tz):
    W = _cloud(frame, "agentview")
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    o = (rgb[..., 0] > rgb[..., 1] + 25) & (rgb[..., 0] > rgb[..., 2] + 40) & (rgb[..., 0] > 70)
    P = W[o]
    P = P[np.isfinite(P[:, 2])]
    P = P[(P[:, 2] > tz + 0.005) & (P[:, 2] < tz + 0.2)]
    if len(P) == 0:
        return 0
    d = np.sqrt((P[:, 0] - c[0]) ** 2 + (P[:, 1] - c[1]) ** 2)
    return int((d < radius).sum())


def _mem_defaults(memory):
    m = {}
    keys = ["stage", "table_z", "butter_xy", "bottle_xy", "bottle_h", "pan_xy", "drainer_xy",
            "butter_yaw", "butter_tries", "bottle_tries", "pours_done", "butter_in_pan", "bottle_held", "decisions", "log"]
    for k in keys:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "detect"
    for k in ("butter_yaw", "butter_tries", "bottle_tries", "pours_done", "decisions"):
        if m[k] is None:
            m[k] = 0
    for k in ("butter_in_pan", "bottle_held"):
        if m[k] is None:
            m[k] = False
    if m["log"] is None:
        m["log"] = ""
    return m


def _note(m, s):
    m["log"] = (m["log"] + " | " + s)[-1500:]


def _update_scene(m, frame, keep=True):
    """Detect objects; overwrite memory entries only with fresh detections (missing -> keep previous)."""
    r = _detect(frame)
    if m["table_z"] is None or abs(r["table_z"] - DEFAULT_TABLE_Z) < 0.05:
        m["table_z"] = r["table_z"]
    found = []
    if r["butter"] is not None:
        m["butter_xy"] = r["butter"]["grasp"]
        m["butter_yaw"] = _grasp_yaw(r["butter"])
        found.append("butter")
    if r["bottle"] is not None:
        m["bottle_xy"] = r["bottle"]["grasp"]
        m["bottle_h"] = r["bottle"]["h"]
        found.append("bottle")
    if r["pan"] is not None:
        m["pan_xy"] = r["pan"]["c"]
        found.append("pan")
    if r["drainer"] is not None and (m["drainer_xy"] is None or _dist(m["drainer_xy"], r["drainer"]["c"]) < 0.08):
        m["drainer_xy"] = r["drainer"]["c"]
        found.append("drainer")
    return found


def _dist(a, b):
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _act(tool, **args):
    return {"tool": tool, "args": args}


def _grasp_yaw(b):
    """Gripper yaw (deg) putting the fingers across the butter's short side; 0 (demo pose) unless clearly rotated."""
    yaw = float(b.get("yaw", 0.0))
    if b.get("elong", 1.0) < 1.3 or abs(yaw) < 12.0:
        return 0.0
    return yaw


def _pick_butter(m):
    tz = m["table_z"] or DEFAULT_TABLE_Z
    xy = m["butter_xy"] or BUTTER_PRIOR
    m["butter_tries"] += 1
    m["stage"] = "butter_picked"
    dx = DIAG_FIRST_TRY_DX if m["butter_tries"] == 1 else 0.0
    return _act("act_pick", x=float(xy[0] + dx), y=float(xy[1]), z_grasp=float(tz + 0.002), hover_z=float(tz + 0.21),
                lift_z=float(tz + 0.37), yaw_deg=float(m["butter_yaw"] or 0.0))


def _pick_bottle(m):
    tz = m["table_z"] or DEFAULT_TABLE_Z
    xy = m["bottle_xy"]
    h = m["bottle_h"] or 0.074
    zg = tz + min(0.06, max(0.02, 0.5 * h))
    m["bottle_tries"] += 1
    m["stage"] = "bottle_picked"
    dx = DIAG_FIRST_TRY_DX if m["bottle_tries"] == 1 else 0.0
    return _act("act_pick", x=float(xy[0] + dx), y=float(xy[1]), z_grasp=float(zg), hover_z=float(tz + 0.23),
                lift_z=float(tz + 0.36))


def _pour(m, hold, untilt=True):
    tz = m["table_z"] or DEFAULT_TABLE_Z
    pan = m["pan_xy"] or PAN_PRIOR
    return _act("act_pour", x=float(pan[0] + POUR_DX), y=float(pan[1]), z=float(tz + BOTTLE_POUR_DZ),
                tilt_deg=45.0, hold_steps=int(hold), untilt=bool(untilt))


def _held(w, last):
    """Grasp evidence: fingers stopped between empty-closed (~0) and fully open (~0.078), and the act
    tool's own measured verdict when available."""
    ok = 0.01 < w < 0.074
    if isinstance(last, str) and '"held": false' in last:
        ok = False
    return ok


def _decide(obs, goal, m, last=None):
    f = obs.current
    tz = m["table_z"] or DEFAULT_TABLE_Z
    w = _width(f)
    tcp = np.asarray(f.tcp, dtype=float)
    st = m["stage"]

    if st == "detect":
        found = _update_scene(m, f)
        _note(m, "detect:" + ",".join(found))
        if m["bottle_xy"] is None or m["pan_xy"] is None:
            if m["decisions"] < 3:
                return _act("act_move", x=NEUTRAL[0], y=NEUTRAL[1], z=NEUTRAL[2], grip=-1.0, max_steps=80)
            if m["pan_xy"] is None:
                m["pan_xy"] = PAN_PRIOR
                _note(m, "pan prior used")
        if m["butter_xy"] is None:
            _note(m, "butter prior used")
        return _pick_butter(m)

    if st == "butter_picked":
        if _held(w, last) and tcp[2] > tz + 0.2:
            pan = m["pan_xy"] or PAN_PRIOR
            b = m["bottle_xy"] or [pan[0] - 0.2, pan[1] - 0.45]
            m["stage"] = "butter_placed"
            _note(m, "butter held w=%.3f" % w)
            dy = DIAG_BUTTER_PLACE_DY if m["butter_tries"] == 1 else 0.0
            return _act("act_place", x=float(pan[0]), y=float(pan[1] + dy), z_release=float(tz + BUTTER_RELEASE_DZ),
                        transit_z=float(tz + 0.37), retreat=[float(b[0]), float(b[1]), float(tz + 0.27)])
        _note(m, "butter grasp miss w=%.3f" % w)
        m["stage"] = "butter_redetect"
        return _act("act_move", x=NEUTRAL[0], y=NEUTRAL[1], z=NEUTRAL[2], grip=-1.0, max_steps=120)

    if st == "butter_redetect":
        _update_scene(m, f)
        if m["butter_tries"] < MAX_PICK_TRIES:
            return _pick_butter(m)
        _note(m, "butter tries exhausted")
        m["stage"] = "bottle_pick"
        st = "bottle_pick"

    if st == "butter_placed":
        pan = m["pan_xy"] or PAN_PRIOR
        n_in = _orange_near(f, pan, 0.085, tz)
        _update_scene(m, f)
        if n_in >= 8:
            m["butter_in_pan"] = True
            _note(m, "butter in pan (%d px)" % n_in)
        else:
            m["butter_in_pan"] = False
            _note(m, "butter NOT seen in pan (%d px)" % n_in)
            r = _detect(f)
            if r["butter"] is not None and m["butter_tries"] < MAX_PICK_TRIES:
                m["butter_xy"] = r["butter"]["grasp"]
                m["butter_yaw"] = _grasp_yaw(r["butter"])
                _note(m, "butter seen at (%.3f,%.3f) yaw %.0f" % (m["butter_xy"][0], m["butter_xy"][1], m["butter_yaw"]))
                return _pick_butter(m)
        m["stage"] = "bottle_pick"
        st = "bottle_pick"

    if st == "bottle_pick":
        if m["bottle_xy"] is None:
            _update_scene(m, f)
        if m["bottle_xy"] is None:
            _note(m, "no bottle detection")
            return _act("act_move", x=NEUTRAL[0], y=NEUTRAL[1], z=NEUTRAL[2], grip=-1.0, max_steps=120)
        return _pick_bottle(m)

    if st == "bottle_picked":
        if _held(w, last) and tcp[2] > tz + 0.2:
            m["bottle_held"] = True
            m["stage"] = "pour1"
            _note(m, "bottle held w=%.3f" % w)
            return _pour(m, 60, True)
        m["bottle_held"] = False
        _note(m, "bottle grasp miss w=%.3f" % w)
        if m["bottle_tries"] >= MAX_PICK_TRIES:
            m["stage"] = "done"
            return _act("act_move", x=NEUTRAL[0], y=NEUTRAL[1], z=NEUTRAL[2], grip=-1.0, max_steps=120)
        m["stage"] = "bottle_redetect"
        return _act("act_move", x=NEUTRAL[0], y=NEUTRAL[1], z=NEUTRAL[2], grip=-1.0, max_steps=120)

    if st == "bottle_redetect":
        _update_scene(m, f)
        return _pick_bottle(m)

    if st in ("pour1", "pour2"):
        if _held(w, last):
            m["pours_done"] += 1
            _note(m, "%s done, held w=%.3f" % (st, w))
            if m["pours_done"] < 2:
                m["stage"] = "pour2"
                return _pour(m, 150, True)
            m["stage"] = "drainer"
            d = m["drainer_xy"] or DRAINER_PRIOR
            return _act("act_place", x=float(d[0]), y=float(d[1]), z_release=float(tz + 0.15),
                        transit_z=float(tz + 0.36), retreat=[NEUTRAL[0], NEUTRAL[1], NEUTRAL[2]])
        m["bottle_held"] = False
        _note(m, "bottle lost during %s w=%.3f" % (st, w))
        m["stage"] = "bottle_redetect" if m["bottle_tries"] < MAX_PICK_TRIES else "done"
        return _act("act_move", x=NEUTRAL[0], y=NEUTRAL[1], z=NEUTRAL[2], grip=-1.0, max_steps=120)

    m["stage"] = "done"
    return _act("act_observe", steps=20, grip=0.0)


def perceive_decide(obs, goal, memory, last):
    """Stage machine for: butter -> frypan, pour sauce twice over it, sauce bottle -> bowl drainer."""
    m = _mem_defaults(memory)
    m["decisions"] += 1
    act = _decide(obs, goal, m, last)
    return {"memory": m, "act": act}
