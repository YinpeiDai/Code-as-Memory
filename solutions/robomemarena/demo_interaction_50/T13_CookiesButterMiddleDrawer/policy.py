"""Task policy: open the middle drawer, put cookies in it, then put butter in the same drawer."""

FIELDS = ("stage", "handle", "cab_box", "cookies", "butter", "drawer_open", "cookies_placed", "butter_placed",
          "drawer_attempts", "cookies_attempts", "butter_attempts", "notes")
OPEN_POINTS_MIN = 150      # middle-drawer body points (z 0.50-0.625); R001 ~300 open, 0 closed
TOP_OPEN_POINTS_MIN = 100  # top-drawer body points (z 0.655-0.73); R007 ~200 when the top drawer was opened
TOP_DRAWER_PUSH_Z = 0.675
HOLD_GAP_MIN = 0.008
HOLD_GAP_MAX = 0.072   # fully open is 0.078-0.080; held cookies 0.062, butter 0.039 (all rollouts)
MAX_DRAWER_ATTEMPTS = 4
MAX_PICK_ATTEMPTS = 5
# drop points relative to the closed-handle position (measured from 50 demos: cookies -0.038/-0.036, butter
# -0.001/-0.045; z 0.739 / 0.727)
# r4: cookies dropped nearer the drawer centre / robot base (demo drop point is at full arm reach: elbow joint at
# its limit in every demo and rollout; R018 got stuck leaving it); butter kept near the demo point, offset in x.
# r5: butter also pulled in toward the base (R030, far cabinet: butter release sagged to z 0.717 at full reach);
# footprints (cookies 8x6 cm, butter 7x4 cm, long axis x) stay disjoint: cookies y [hy-0.145, hy-0.085],
# butter y [hy-0.085, hy-0.045]; drawer interior spans about y [hy-0.21, hy], x [hx-0.105, hx+0.105].
# r6: depth profile of the open drawer (R001 frame 200, scratch/drawer_profile) shows the floor (z 0.58) only for
# y in [hy-0.12, hy-0.02] (front-wall rim 0.641 at hy-0.13) and x within about hx +- 0.10, so the cookies are
# placed at the -x side, the butter at the +x side; footprints disjoint, butter reach equal to the demo drop.
# r7: butter x reduced again (R036: butter drop at hx+0.02 reached the arm's limit); footprints overlap only in a
# ~1.5 x 2 cm corner (demos overlapped ~4 cm in x with full y overlap and always succeeded).
COOKIES_DROP = (-0.055, -0.090, 0.725)
BUTTER_DROP = (0.005, -0.060, 0.720)
DRAWER_RETRY_VARIANTS = ((0.0, 0.0), (-0.006, 0.004), (0.006, 0.004), (0.0, 0.008))
PICK_Z = 0.434


def _mem(memory):
    m = {}
    for k in FIELDS:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    for k in ("drawer_attempts", "cookies_attempts", "butter_attempts"):
        if m[k] is None:
            m[k] = 0
    for k in ("drawer_open", "cookies_placed", "butter_placed"):
        if m[k] is None:
            m[k] = False
    if m["notes"] is None:
        m["notes"] = ""
    return m


def _note(m, s):
    n = (m["notes"] + " | " + s) if m["notes"] else s
    m["notes"] = n[-600:]


def _cab_from_box(m):
    b = m["cab_box"]
    if not b:
        return None
    return {"xmin": b[0][0], "ymin": b[0][1], "xmax": b[1][0], "ymax": b[1][1]}


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(abs(f[0]) + abs(f[1]))


def _locate_scene(frame, m):
    cab = _cabinet(frame)
    if cab is not None:
        m["handle"] = [round(v, 4) for v in cab["handle"]]
        m["cab_box"] = [[cab["xmin"], cab["ymin"]], [cab["xmax"], cab["ymax"]]]
        _note(m, "cab n=%d handle=(%.3f,%.3f)" % (cab["n"], cab["handle"][0], cab["handle"][1]))
    elif m["handle"] is None:
        h = CABINET_NOMINAL_HANDLE
        m["handle"] = [h[0], h[1], h[2]]
        m["cab_box"] = [[h[0] - 0.125, h[1] + 0.017], [h[0] + 0.125, h[1] + 0.207]]
        _note(m, "cabinet not detected; nominal handle")
    for name, nom in (("cookies", COOKIES_NOMINAL), ("butter", BUTTER_NOMINAL)):
        if m[name] is None:
            d = _object_on_table(frame, nom)
            m[name] = d["xy"] if d is not None else [nom[0], nom[1]]
            if d is None:
                _note(m, name + " not detected; nominal")


def _redetect(frame, m, name, nom):
    """Re-find an object on the table near its last known/nominal position; returns xy or None."""
    for centre, r in ((m[name], 0.05), (nom, 0.06), (nom, 0.10)):
        if centre is None:
            continue
        d = _object_on_table(frame, centre, radius=r)
        if d is not None:
            return d["xy"]
    return None


def _act(tool, **args):
    return {"tool": tool, "args": args}


def _open_drawer_act(m):
    h = m["handle"] or list(CABINET_NOMINAL_HANDLE)
    i = min(int(m["drawer_attempts"]), len(DRAWER_RETRY_VARIANTS) - 1)
    zo, do = DRAWER_RETRY_VARIANTS[i]
    return _act("act_open_drawer", hx=h[0], hy=h[1], hz=h[2], z_offset=zo, depth_offset=do)


def _pick_act(m, name):
    xy = m[name]
    if not xy:
        nom = COOKIES_NOMINAL if name == "cookies" else BUTTER_NOMINAL
        xy = [nom[0], nom[1]]
        m[name] = xy
    return _act("act_pick", x=xy[0], y=xy[1], z_grasp=PICK_Z)


def _place_act(m, name):
    h = m["handle"] or list(CABINET_NOMINAL_HANDLE)
    off = COOKIES_DROP if name == "cookies" else BUTTER_DROP
    return _act("act_place", x=h[0] + off[0], y=h[1] + off[1], z_release=off[2])


def perceive_decide(obs, goal, memory, last):
    """Stage machine: locate cabinet/cookies/butter -> open middle drawer -> pick+place cookies -> pick+place
    butter -> wait. Every stage transition is gated on a measured outcome (open-drawer depth points, finger gap,
    object absence from the table)."""
    m = _mem(memory)
    f = obs.current
    stage = m["stage"]
    act = None

    if stage is None:
        _locate_scene(f, m)
        m["stage"] = "open"
        m["drawer_attempts"] = 0
        act = _open_drawer_act(m)

    elif stage in ("open", "close_top"):
        cab = _cab_from_box(m)
        lv = _drawer_levels(f, cab) if cab is not None else {"middle": 0, "top": 0}
        pts = lv["middle"]
        if lv["top"] >= TOP_OPEN_POINTS_MIN and int(m["drawer_attempts"]) < MAX_DRAWER_ATTEMPTS + 2:
            # a wrong (upper) drawer is open: push it closed before (re)opening the middle one
            m["drawer_attempts"] = int(m["drawer_attempts"]) + 1
            _note(m, "top drawer open pts=%d -> push closed" % lv["top"])
            m["stage"] = "close_top"
            act = _act("act_push_drawer", x=cab["xmin"] + 0.07, y_from=cab["ymin"] - 0.33,
                       y_to=cab["ymin"] + 0.004, z=TOP_DRAWER_PUSH_Z)
        elif stage == "close_top":
            _note(m, "after push: top=%d middle=%d" % (lv["top"], lv["middle"]))
            m["stage"] = "open"
            act = _open_drawer_act(m)
        elif pts >= OPEN_POINTS_MIN:
            m["drawer_open"] = True
            _note(m, "middle drawer open pts=%d top=%d" % (pts, lv["top"]))
            m["stage"] = "pick_cookies"
            act = _pick_act(m, "cookies")
        else:
            m["drawer_attempts"] = int(m["drawer_attempts"]) + 1
            _note(m, "middle drawer not open pts=%d top=%d try=%d" % (pts, lv["top"], m["drawer_attempts"]))
            if m["drawer_attempts"] < MAX_DRAWER_ATTEMPTS:
                act = _open_drawer_act(m)
            else:
                # proceed anyway: further pulls have not helped; the pick may still succeed if partly open
                m["stage"] = "pick_cookies"
                act = _pick_act(m, "cookies")

    elif stage in ("pick_cookies", "pick_butter"):
        name = stage[5:]
        key = name + "_attempts"
        if HOLD_GAP_MIN < _gap(f) < HOLD_GAP_MAX:
            _note(m, "%s held gap=%.3f" % (name, _gap(f)))
            m["stage"] = "place_" + name
            act = _place_act(m, name)
        else:
            m[key] = int(m[key]) + 1
            nom = COOKIES_NOMINAL if name == "cookies" else BUTTER_NOMINAL
            xy = _redetect(f, m, name, nom)
            if xy is not None:
                m[name] = xy
            _note(m, "%s missed try=%d redetect=%s" % (name, m[key], "yes" if xy is not None else "no"))
            if m[key] < MAX_PICK_ATTEMPTS:
                act = _pick_act(m, name)
            else:
                m["stage"] = "pick_butter" if name == "cookies" else "done"
                act = _pick_act(m, "butter") if name == "cookies" else _act("act_wait", steps=40)

    elif stage in ("place_cookies", "place_butter"):
        name = stage[6:]
        nom = COOKIES_NOMINAL if name == "cookies" else BUTTER_NOMINAL
        on_table = _object_on_table(f, m[name] if m[name] is not None else nom, radius=0.05)
        if on_table is not None and int(m[name + "_attempts"]) < MAX_PICK_ATTEMPTS:
            m[name + "_attempts"] = int(m[name + "_attempts"]) + 1
            m[name] = on_table["xy"]
            _note(m, "%s still on table after place; re-pick" % name)
            m["stage"] = "pick_" + name
            act = _pick_act(m, name)
        else:
            m[name + "_placed"] = True
            _note(m, "%s placed" % name)
            if name == "cookies":
                m["stage"] = "pick_butter"
                act = _pick_act(m, "butter")
            else:
                m["stage"] = "done"
                act = _act("act_wait", steps=40)

    else:  # done: the episode should already have been scored; keep observing, re-pick if butter reappears
        b = _object_on_table(f, m["butter"] if m["butter"] is not None else BUTTER_NOMINAL, radius=0.05)
        c = _object_on_table(f, m["cookies"] if m["cookies"] is not None else COOKIES_NOMINAL, radius=0.05)
        if c is not None and int(m["cookies_attempts"]) < MAX_PICK_ATTEMPTS + 2:
            m["cookies"] = c["xy"]
            m["stage"] = "pick_cookies"
            act = _pick_act(m, "cookies")
        elif b is not None and int(m["butter_attempts"]) < MAX_PICK_ATTEMPTS + 2:
            m["butter"] = b["xy"]
            m["stage"] = "pick_butter"
            act = _pick_act(m, "butter")
        else:
            act = _act("act_wait", steps=60)

    return {"memory": {k: m[k] for k in FIELDS}, "act": act}
