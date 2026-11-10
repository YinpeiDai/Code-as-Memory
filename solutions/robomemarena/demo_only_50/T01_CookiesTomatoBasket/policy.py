"""Decision program for T01: cookies into a basket, then tomato sauce into the SAME basket.

Memory holds the scene measured at the first decision (object and basket positions) and
the stage. The chosen basket is written once and reused for the tomato sauce, which is
the 'same basket' requirement of the instruction.
"""

CARRY_Z_ABOVE_RIM = 0.255   # carry height above basket rim (expert carried at ~0.82 m, rim 0.566 m)
RELEASE_ABOVE_RIM = 0.090   # release height above rim (expert released at 0.656 m)
PRE_ABOVE_GRASP = 0.10      # pre-grasp height above the grasp height
COOKIE_GRASP_BELOW_TOP = 0.012  # expert grasp tcp 0.433 m vs measured cookie top 0.445 m
TOMATO_GRASP_BELOW_TOP = 0.006  # expert grasp tcp 0.500 m vs measured can top 0.505 m
MAX_PICK_ATTEMPTS = 4
FIELDS = ("stage", "cookies", "tomato", "basket", "other_basket", "table_z", "attempts",
          "cookies_done", "decisions", "log")


def _goal_objects(goal):
    """Parse the instruction: first and second objects and the container word."""
    g = (goal or "").lower()
    first, second = "cookies", "tomato"
    i_c, i_t = g.find("cookie"), g.find("tomato")
    if i_c >= 0 and i_t >= 0 and i_t < i_c:
        first, second = "tomato", "cookies"
    return first, second


def _near(a, b, r):
    return a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) < r


def _grasp_args(obj, pos, table_z, rim_z):
    below = COOKIE_GRASP_BELOW_TOP if obj == "cookies" else TOMATO_GRASP_BELOW_TOP
    gz = max(table_z + 0.002, pos[2] - below)
    return {"x": pos[0], "y": pos[1], "grasp_z": gz, "pre_z": gz + PRE_ABOVE_GRASP,
            "carry_z": rim_z + CARRY_Z_ABOVE_RIM}


def _place_args(basket):
    return {"x": basket[0], "y": basket[1], "release_z": basket[2] + RELEASE_ABOVE_RIM,
            "carry_z": basket[2] + CARRY_Z_ABOVE_RIM}


def _pick_act(mem, obj):
    return {"tool": "act_pick", "args": _grasp_args(obj, mem[obj], mem["table_z"], mem["basket"][2])}


def _refresh(mem, obj, scene, radius):
    """Update a remembered object position from a fresh scene if it is plausibly the same object."""
    new = scene.get(obj)
    if new is not None and (mem.get(obj) is None or _near(new, mem[obj], radius)):
        mem[obj] = [float(v) for v in new]
        return True
    return False


def perceive_decide(obs, goal, memory, last):
    """Stage machine: init -> pick first -> place -> pick second -> place (same basket) -> done."""
    mem = {k: memory.get(k) for k in FIELDS}
    mem["decisions"] = int(mem["decisions"] or 0) + 1
    stage = mem["stage"] or "init"
    first, second = _goal_objects(goal)
    f = obs.current
    tcp = [float(v) for v in f.tcp[:3]]
    gap = _gap(f)
    act = {"tool": "act_hold", "args": {"steps": 40, "grip": -1.0}}
    note = ""

    if stage == "init":
        sc = _scene(f, exclude_xyz=tcp)
        mem["table_z"] = float(sc["table_z"])
        mem["cookies"] = sc["cookies"] or list(NOMINAL_COOKIES)
        mem["tomato"] = sc["tomato"] or list(NOMINAL_TOMATO)
        bs = sc["baskets"]
        # Expert always used the basket with the smaller world x (the one farther from the
        # agentview camera, nearer the robot base); both baskets look identical.
        mem["basket"] = [float(v) for v in bs[0]] if bs else list(NOMINAL_BASKET)
        mem["other_basket"] = [float(v) for v in bs[1]] if len(bs) > 1 else None
        mem["attempts"] = 0
        mem["cookies_done"] = False
        stage = "pick_" + first
        act = _pick_act(mem, first)
        note = "scene found=%s n_baskets=%d" % (sc["found"], len(bs))

    elif stage in ("pick_cookies", "pick_tomato"):
        obj = stage[5:]
        if gap > 0.01 and tcp[2] > mem["table_z"] + 0.12:
            stage = "place_" + obj
            act = {"tool": "act_place", "args": _place_args(mem["basket"])}
            note = "holding %s gap=%.4f" % (obj, gap)
        else:
            mem["attempts"] = int(mem["attempts"] or 0) + 1
            sc = _scene(f, exclude_xyz=tcp)
            _refresh(mem, obj, sc, 0.12)
            if mem["attempts"] <= MAX_PICK_ATTEMPTS:
                act = _pick_act(mem, obj)
                note = "grasp of %s failed (gap=%.4f); retry %d" % (obj, gap, mem["attempts"])
            else:
                note = "giving up on %s" % obj
                stage = "pick_" + second if obj == first else "done"
                mem["attempts"] = 0
                if stage != "done":
                    act = _pick_act(mem, second)

    elif stage in ("place_cookies", "place_tomato"):
        obj = stage[6:]
        sc = _scene(f, exclude_xyz=tcp)
        left = sc.get(obj)
        # an object of this kind still on the table near where it was taken = it fell out
        if left is not None and mem["attempts"] < MAX_PICK_ATTEMPTS and gap > 0.07:
            mem[obj] = [float(v) for v in left]
            mem["attempts"] = int(mem["attempts"] or 0) + 1
            stage = "pick_" + obj
            act = _pick_act(mem, obj)
            note = "%s still on table after place; re-pick" % obj
        elif obj == first:
            mem["cookies_done"] = True
            mem["attempts"] = 0
            _refresh(mem, second, sc, 0.06)
            stage = "pick_" + second
            act = _pick_act(mem, second)
            note = "%s placed; picking %s" % (obj, second)
        else:
            stage = "done"
            note = "both placed; holding"

    else:  # done: the episode has not ended; look for an object left on the table
        sc = _scene(f, exclude_xyz=tcp)
        redo = None
        for obj in (first, second):
            if sc.get(obj) is not None and int(mem["attempts"] or 0) < MAX_PICK_ATTEMPTS:
                redo = obj
                break
        if redo is not None and gap > 0.07:
            mem[redo] = [float(v) for v in sc[redo]]
            mem["attempts"] = int(mem["attempts"] or 0) + 1
            stage = "pick_" + redo
            act = _pick_act(mem, redo)
            note = "%s seen on table while done; re-pick" % redo
        else:
            act = {"tool": "act_hold", "args": {"steps": 60, "grip": -1.0}}
            note = "waiting"

    mem["stage"] = stage
    mem["log"] = note
    return {"memory": mem, "act": act}
