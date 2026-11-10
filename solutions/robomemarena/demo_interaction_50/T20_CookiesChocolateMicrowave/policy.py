"""Task program: open the microwave, put cookies inside, then put chocolate where the cookies went (the microwave).

Stages (memory.stage):
  start -> handle_check -> door_check -> cookies_check -> cookies_release -> choc_check -> choc_place -> done
Geometry offsets are relative to the perceived microwave centroid M (agentview depth, points above the table
with y > 0.2) and to the perceived object centroids; they were measured over the 50 demonstrations."""

# nominal scene (mean over demos) used only when perception fails
NOMINAL_MW = [0.066, 0.530]
NOMINAL_COOKIES = [0.172, -0.095, 0.444]
NOMINAL_CHOC = [0.174, -0.193, 0.452]

HANDLE_OFF = [-0.019, -0.115]       # handle grasp TCP relative to M
HANDLE_Z = 0.582
PULL_PATH = [[0.013, -0.008], [-0.014, -0.060], [-0.050, -0.110], [-0.088, -0.145], [-0.125, -0.171],
             [-0.163, -0.190], [-0.205, -0.197], [-0.252, -0.198], [-0.301, -0.191], [-0.350, -0.188],
             [-0.372, -0.186]]       # TCP path relative to handle grasp while the door swings open
GRASP_OFF = [-0.0085, 0.0015]       # grasp TCP relative to small-object centroid
COOKIE_Z = 0.438
CHOC_Z = 0.441
PLACE_OFF = [-0.048, -0.059]        # cookies release point relative to M
PLACE_Z = 0.577
CHOC_IN_OFF = [-0.078, -0.10]       # chocolate insertion target relative to M (success seen at -0.18)
CHOC_IN_Z = 0.600
# Exploratory-rollout switch: names in this list force one deliberate failure of that step on its first try
# ("handle": stop 4 cm short of the handle; "door": release without pulling; "cookies"/"choc": grasp 5 cm off in x).
# Empty in the deployed policy.
DEBUG_PERTURB = []
FIELDS = ["stage", "mw", "cookies", "chocolate", "tries", "door_open", "cookies_in", "notes"]


def _mem(memory):
    m = {}
    for k in FIELDS:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "start"
    if m["tries"] is None:
        m["tries"] = 0
    if m["notes"] is None:
        m["notes"] = ""
    return m


def _note(m, s):
    t = (m["notes"] + " | " + s) if m["notes"] else s
    m["notes"] = t[-4000:]


def _out(m, segs):
    return {"memory": m, "act": {"tool": "act_run", "args": {"segments": segs}}}


def _approach_handle(m, k):
    mw = m["mw"]
    hx = mw[0] + HANDLE_OFF[0]
    hy = mw[1] + HANDLE_OFF[1] + 0.006 * k
    if k == 0 and "handle" in DEBUG_PERTURB and "dbg_handle" not in m["notes"]:
        hy -= 0.04
        _note(m, "dbg_handle")
    return [
        {"name": "pre_handle", "pos": [hx, hy - 0.17, HANDLE_Z], "ori": "side", "grip": -1, "tol": 0.01, "max": 160},
        {"name": "near_handle", "pos": [hx, hy - 0.06, HANDLE_Z], "ori": "side", "grip": -1, "tol": 0.006, "max": 80},
        {"name": "at_handle", "pos": [hx, hy, HANDLE_Z], "ori": "side", "grip": -1, "tol": 0.003, "max": 60,
         "vmax": 0.6},
        {"name": "close_handle", "wait": 20, "grip": 1},
    ]


def _pull(m):
    mw = m["mw"]
    hx = mw[0] + HANDLE_OFF[0]
    hy = mw[1] + HANDLE_OFF[1]
    segs = []
    for i, (ox, oy) in enumerate(PULL_PATH):
        last = i == len(PULL_PATH) - 1
        segs.append({"name": "pull%d" % i, "pos": [hx + ox, hy + oy, HANDLE_Z], "ori": "side", "grip": 1,
                     "tol": 0.01 if last else 0.015, "max": 40, "gain": 25.0})
    segs.append({"name": "release_handle", "wait": 12, "grip": -1})
    segs.append({"name": "clear", "pos": [0.02, -0.05, 0.67], "ori": "down", "grip": -1, "tol": 0.01, "max": 150})
    return segs


def _pick(obj, z, above_z, dx=0.0):
    gx = obj[0] + GRASP_OFF[0] + dx
    gy = obj[1] + GRASP_OFF[1]
    return [
        {"name": "above", "pos": [gx, gy, above_z], "ori": "down", "grip": -1, "tol": 0.006, "max": 150},
        {"name": "descend", "pos": [gx, gy, z], "ori": "down", "grip": -1, "tol": 0.003, "max": 120},
        {"name": "close", "wait": 20, "grip": 1},
        {"name": "lift", "pos": [gx, gy, z + 0.12], "ori": "down", "grip": 1, "tol": 0.01, "max": 80},
    ]


def _rescan(obs, m, which):
    s = _scene(obs.current)
    if s.get(which) is not None:
        m[which] = s[which]
        _note(m, "rescan %s %s" % (which, [round(v, 3) for v in s[which]]))
        return True
    return False


def perceive_decide(obs, goal, memory, last):
    """Stage machine for 'put cookies into the microwave, then chocolate where the cookies were placed'.

    Reads agentview RGB-D and finger gaps, keeps scene estimates and progress in memory, and returns one
    act_run call (list of servo segments) per decision."""
    m = _mem(memory)
    fr = obs.current
    gap = float(fr.fingers[0] - fr.fingers[1])
    st = m["stage"]

    if st == "start":
        s = _scene(fr)
        if not s["ok"] and m["tries"] < 2:
            m["tries"] += 1
            _note(m, "scene not found n_mw=%d n_obj=%s" % (s["n_mw"], s["n_obj"]))
            return _out(m, [{"name": "observe", "wait": 3, "grip": -1}])
        m["mw"] = s["microwave"] if s["microwave"] is not None else NOMINAL_MW
        m["cookies"] = s["cookies"] if s["cookies"] is not None else NOMINAL_COOKIES
        m["chocolate"] = s["chocolate"] if s["chocolate"] is not None else NOMINAL_CHOC
        _note(m, "scene mw=%s ck=%s ch=%s by %s" % ([round(v, 3) for v in m["mw"]],
                                                   [round(v, 3) for v in m["cookies"]],
                                                   [round(v, 3) for v in m["chocolate"]], s["assign"]))
        m["tries"] = 0
        m["door_open"] = False
        m["cookies_in"] = False
        m["stage"] = "handle_check"
        return _out(m, _approach_handle(m, 0))

    if st == "handle_check":
        if gap > 0.006:
            _note(m, "handle grasped gap=%.4f" % gap)
            m["stage"] = "door_check"
            if "door" in DEBUG_PERTURB and "dbg_door" not in m["notes"]:
                _note(m, "dbg_door")
                return _out(m, [{"name": "release_handle", "wait": 12, "grip": -1},
                                {"name": "clear", "pos": [0.02, -0.05, 0.67], "ori": "down", "grip": -1,
                                 "tol": 0.01, "max": 150}])
            if "door_partial" in DEBUG_PERTURB and "dbg_partial" not in m["notes"]:
                _note(m, "dbg_partial")
                segs = _pull(m)
                return _out(m, segs[:4] + segs[-2:])
            return _out(m, _pull(m))
        m["tries"] += 1
        _note(m, "handle missed gap=%.4f try=%d" % (gap, m["tries"]))
        if m["tries"] > 3:
            m["stage"] = "door_check"
            return _out(m, _pull(m))
        mw = m["mw"]
        back = {"name": "back_off", "pos": [mw[0] + HANDLE_OFF[0], mw[1] + HANDLE_OFF[1] - 0.12, HANDLE_Z],
                "ori": "side", "grip": -1, "tol": 0.01, "max": 80}
        return _out(m, [{"name": "open", "wait": 10, "grip": -1}, back] + _approach_handle(m, m["tries"]))

    if st == "door_check":
        ds = _door_state(fr, m["mw"])
        opened = ds["leaf_pts"] > 60 and ds["handle_pts"] < 15
        _note(m, "door %s" % ds)
        if not opened and m["tries"] < 3:
            m["tries"] += 1
            m["stage"] = "handle_check"
            return _out(m, _approach_handle(m, 0))
        m["door_open"] = bool(opened)
        m["tries"] = 0
        m["stage"] = "cookies_check"
        dx = 0.0
        if "cookies" in DEBUG_PERTURB and "dbg_cookies" not in m["notes"]:
            dx = 0.09
            _note(m, "dbg_cookies")
        return _out(m, _pick(m["cookies"], COOKIE_Z, 0.60, dx))

    if st == "cookies_check":
        if gap > 0.03:
            _note(m, "cookies held gap=%.4f" % gap)
            m["stage"] = "cookies_release"
            ck = m["cookies"]
            mw = m["mw"]
            gx, gy = ck[0] + GRASP_OFF[0], ck[1] + GRASP_OFF[1]
            px, py = mw[0] + PLACE_OFF[0], mw[1] + PLACE_OFF[1]
            return _out(m, [
                {"name": "staging", "pos": [gx - 0.145, gy + 0.19, 0.557], "ori": "down", "grip": 1, "tol": 0.01,
                 "max": 120},
                {"name": "turn_side", "pos": [gx - 0.145, gy + 0.19, 0.557], "ori": "side", "grip": 1, "tol": 0.01,
                 "rtol": 0.04, "max": 80},
                {"name": "pre_place", "pos": [px, mw[1] - 0.30, PLACE_Z], "ori": "side", "grip": 1, "tol": 0.01,
                 "max": 100},
                {"name": "place", "pos": [px, py, PLACE_Z], "ori": "side", "grip": 1, "tol": 0.005, "max": 120,
                 "vmax": 0.6},
            ])
        m["tries"] += 1
        _note(m, "cookies grasp missed gap=%.4f try=%d" % (gap, m["tries"]))
        _rescan(obs, m, "cookies")
        if m["tries"] > 3:
            m["stage"] = "choc_check"
            m["tries"] = 0
            return _out(m, [{"name": "open", "wait": 10, "grip": -1}] + _pick(m["chocolate"], CHOC_Z, 0.70))
        return _out(m, [{"name": "open", "wait": 10, "grip": -1}] + _pick(m["cookies"], COOKIE_Z, 0.60))

    if st == "cookies_release":
        mw = m["mw"]
        _note(m, "at place gap=%.4f tcp=%s" % (gap, [round(float(v), 3) for v in fr.tcp[:3]]))
        m["cookies_in"] = gap > 0.02
        m["stage"] = "choc_check"
        m["tries"] = 0
        dx = 0.0
        if "choc" in DEBUG_PERTURB and "dbg_choc" not in m["notes"]:
            dx = 0.09
            _note(m, "dbg_choc")
        segs = [
            {"name": "release_cookies", "wait": 20, "grip": -1},
            {"name": "retreat", "pos": [mw[0] - 0.021, mw[1] - 0.208, 0.572], "ori": "side", "grip": -1,
             "tol": 0.01, "max": 80},
            {"name": "rise", "pos": [mw[0] - 0.019, mw[1] - 0.219, 0.711], "ori": "side", "grip": -1, "tol": 0.01,
             "max": 80},
        ]
        return _out(m, segs + _pick(m["chocolate"], CHOC_Z, 0.70, dx))

    if st == "choc_check":
        if gap > 0.025:
            _note(m, "chocolate held gap=%.4f" % gap)
            m["stage"] = "choc_place"
            ch = m["chocolate"]
            mw = m["mw"]
            gx, gy = ch[0] + GRASP_OFF[0], ch[1] + GRASP_OFF[1]
            ix, iy = mw[0] + CHOC_IN_OFF[0], mw[1] + CHOC_IN_OFF[1]
            return _out(m, [
                {"name": "staging", "pos": [gx - 0.196, gy + 0.193, 0.56], "ori": "down", "grip": 1, "tol": 0.01,
                 "max": 120},
                {"name": "turn_side", "pos": [gx - 0.196, gy + 0.193, 0.56], "ori": "side", "grip": 1, "tol": 0.01,
                 "rtol": 0.04, "max": 80},
                {"name": "pre_insert", "pos": [ix, mw[1] - 0.33, CHOC_IN_Z], "ori": "side", "grip": 1, "tol": 0.01,
                 "max": 100},
                {"name": "insert", "pos": [ix, iy, CHOC_IN_Z], "ori": "side", "grip": 1, "tol": 0.005, "max": 120,
                 "vmax": 0.6},
            ])
        m["tries"] += 1
        _note(m, "chocolate grasp missed gap=%.4f try=%d" % (gap, m["tries"]))
        _rescan(obs, m, "chocolate")
        if m["tries"] > 3:
            m["stage"] = "done"
            return _out(m, [{"name": "open", "wait": 10, "grip": -1}])
        return _out(m, [{"name": "open", "wait": 10, "grip": -1}] + _pick(m["chocolate"], CHOC_Z, 0.70))

    if st == "choc_place":
        mw = m["mw"]
        _note(m, "chocolate at target without success gap=%.4f tcp=%s" % (gap, [round(float(v), 3) for v in fr.tcp[:3]]))
        m["stage"] = "done"
        ix = mw[0] + CHOC_IN_OFF[0]
        return _out(m, [
            {"name": "release_choc", "wait": 20, "grip": -1},
            {"name": "retreat", "pos": [ix, mw[1] - 0.30, 0.62], "ori": "side", "grip": -1, "tol": 0.01, "max": 80},
            {"name": "monitor", "wait": 60, "grip": -1},
        ])

    _note(m, "done stage; holding")
    return _out(m, [{"name": "monitor", "wait": 50, "grip": -1}])
