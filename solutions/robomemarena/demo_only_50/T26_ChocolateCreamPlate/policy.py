"""T26_ChocolateCreamPlate policy: move chocolate pudding, then cream cheese, from plate1 to plate2.

Evidence (50 assigned demos): the expert always picks chocolate first, then cream cheese,
grasping both with the top-down gripper yawed -45 deg, and releases both at the plate2
centre at TCP z ~0.523 with yaw 0 (cream lands on top of chocolate).
"""

# Offsets learned offline: expert grasp TCP minus agentview detection (mean over 50 demos).
CHOC_OFF = (-0.020, -0.0036, -0.019)   # dx, dy, dz-from-top
CREAM_OFF = (0.001, 0.0, -0.023)
CHOC_Z = (0.442, 0.466)
CREAM_Z = (0.432, 0.455)
RELEASE_Z = 0.523
GRASP_YAW = -45.0
PLACE_YAW = 0.0
PLATE2_DEFAULT = (0.180, 0.125)
ON_PLATE_R = 0.05


def _grasp_point(obj, kind):
    off = CHOC_OFF if kind == "choc" else CREAM_OFF
    zr = CHOC_Z if kind == "choc" else CREAM_Z
    z = min(zr[1], max(zr[0], obj["top"] + off[2]))
    return [obj["x"] + off[0], obj["y"] + off[1], z]


def _obj_xy(obj, kind):
    off = CHOC_OFF if kind == "choc" else CREAM_OFF
    return [obj["x"] + off[0], obj["y"] + off[1]]


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def perceive_decide(obs, goal, memory, last):
    """Detect objects/plate2 in agentview RGB-D, update progress memory and choose the next act."""
    f = obs.current
    mem = dict(memory)
    for k, v in (("stage", "start"), ("holding", ""), ("picks", 0), ("waits", 0), ("note", "")):
        if mem.get(k) is None:
            mem[k] = v
    det = _detect(f)
    choc, cream = det["choc"], det["cream"]
    if choc is not None:
        mem["choc"] = [choc["x"], choc["y"], choc["top"]]
    if cream is not None:
        mem["cream"] = [cream["x"], cream["y"], cream["top"]]
    if mem.get("plate2") is None:
        avoid = [o for o in (mem.get("choc"), mem.get("cream")) if o is not None]
        p2 = _plate2(det["white"], avoid)
        mem["plate2"] = p2 if p2 is not None else list(PLATE2_DEFAULT)
    p2 = mem["plate2"]
    gap = _gap(f)
    holding = mem["holding"] if (mem["holding"] and gap > 0.010) else ""
    mem["holding"] = holding

    def objd(kind):
        v = mem.get(kind)
        if v is None:
            return None
        return {"x": v[0], "y": v[1], "top": v[2]}

    if holding:
        mem["stage"] = "place_" + holding
        act = {"tool": "act_place", "args": {"x": p2[0], "y": p2[1], "z": RELEASE_Z,
                                              "hold_steps": 12 if holding == "cream" else 0}}
        # expected result; overwritten by the next detection when the object is visible
        off = CHOC_OFF if holding == "choc" else CREAM_OFF
        prev = mem.get(holding) or [0.0, 0.0, 0.47]
        mem[holding] = [p2[0] - off[0], p2[1] - off[1], prev[2]]
        mem["holding"] = ""
        mem["note"] = "placing %s at plate2 %.3f %.3f" % (holding, p2[0], p2[1])
    else:
        c, k = objd("choc"), objd("cream")
        choc_on = c is not None and _dist(_obj_xy(c, "choc"), p2) < ON_PLATE_R
        cream_on = k is not None and _dist(_obj_xy(k, "cream"), p2) < ON_PLATE_R
        if (not choc_on) and c is not None and mem["picks"] < 8:
            g = _grasp_point(c, "choc")
            mem["stage"] = "pick_choc"
            mem["holding"] = "choc"
            mem["picks"] = mem["picks"] + 1
            act = {"tool": "act_pick", "args": {"x": g[0], "y": g[1], "z": g[2], "yaw": GRASP_YAW}}
            mem["note"] = "pick choc at %.3f %.3f %.3f" % tuple(g)
        elif (not cream_on) and k is not None and mem["picks"] < 8:
            g = _grasp_point(k, "cream")
            mem["stage"] = "pick_cream"
            mem["holding"] = "cream"
            mem["picks"] = mem["picks"] + 1
            act = {"tool": "act_pick", "args": {"x": g[0], "y": g[1], "z": g[2], "yaw": GRASP_YAW}}
            mem["note"] = "pick cream at %.3f %.3f %.3f" % tuple(g)
        else:
            mem["stage"] = "done"
            mem["waits"] = mem["waits"] + 1
            act = {"tool": "act_wait", "args": {"steps": 40}}
            mem["note"] = "both on plate2 (choc_on=%s cream_on=%s); waiting" % (choc_on, cream_on)
    out = {}
    for name in ("stage", "holding", "picks", "waits", "plate2", "choc", "cream", "note"):
        out[name] = mem.get(name)
    return {"memory": out, "act": act}


def act_pick(robot, x, y, z, yaw=-45.0):
    """Top-down grasp at world (x, y, z) with gripper yaw (deg): approach at high z, descend, close 30 steps, lift.
    Returns the finger gap after lifting (> 0.01 m means something is held)."""
    s1 = _goto(robot, [x, y, HIGH_Z], yaw, -1, tol=0.01, rtol=0.05, max_steps=250)
    s2 = _goto(robot, [x, y, z + 0.05], yaw, -1, tol=0.006, rtol=0.03, max_steps=120)
    s3 = _goto(robot, [x, y, z], yaw, -1, tol=0.004, rtol=0.03, max_steps=100, vmax=0.6)
    f = _hold(robot, 30, yaw, 1)
    s4 = _goto(robot, [x, y, HIGH_Z], yaw, 1, tol=0.01, rtol=0.08, max_steps=150)
    f = robot.now()
    return {"gap": _gap(f), "tcp": [float(v) for v in np.asarray(f.tcp)[:3]],
            "segments": [list(s1), list(s2), list(s3), list(s4)]}


def act_place(robot, x, y, z, hold_steps=0):
    """Carry the held object above (x, y) at high z with yaw 0, descend to z, hold, open and lift."""
    s1 = _goto(robot, [x, y, HIGH_Z], PLACE_YAW, 1, tol=0.01, rtol=0.05, max_steps=250)
    s2 = _goto(robot, [x, y, z], PLACE_YAW, 1, tol=0.005, rtol=0.03, max_steps=150, vmax=0.7)
    if hold_steps > 0:
        _hold(robot, hold_steps, PLACE_YAW, 1)
    _hold(robot, 20, PLACE_YAW, -1)
    s3 = _goto(robot, [x, y, HIGH_Z], PLACE_YAW, -1, tol=0.01, rtol=0.08, max_steps=120)
    f = robot.now()
    return {"gap": _gap(f), "tcp": [float(v) for v in np.asarray(f.tcp)[:3]],
            "segments": [list(s1), list(s2), list(s3)]}


def act_wait(robot, steps=40):
    """Hold still with the gripper open for `steps` steps (lets the scorer settle)."""
    f = _hold(robot, steps, PLACE_YAW, -1)
    return {"tcp": [float(v) for v in np.asarray(f.tcp)[:3]], "gap": _gap(f)}
