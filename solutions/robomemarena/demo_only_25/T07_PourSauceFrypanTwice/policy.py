"""Decision layer: a phase machine over declared memory.

Sequence (from the 25 expert demos): grasp bottle -> lift -> pre-pour pose over the
frypan -> tilt 45 deg and hold (pour 1) -> upright retreat -> pre-pour -> tilt and
hold (pour 2) -> lift -> place into the bowl drainer -> idle.
"""

GRASP_Z = 0.0375          # TCP height above table at grasp (demo mean, std < 1 mm)
HOVER_Z = 0.17            # hover height above table before the descent
LIFT_Z = 0.79             # absolute lift height after grasp (expert ~0.79)
PRE_OFF = [-0.0037, -0.0349, 0.729]    # pre-tilt pose rel. pan centre xy, abs z
POUR_OFF = [0.0165, -0.0197, 0.757]    # held tilted pose rel. pan centre xy, abs z
RETREAT_OFF = [-0.0545, -0.060, 0.627] # upright retreat between pours
POUR_TILT = 45.0
POUR_HOLD = [80, 110]     # min servo steps for pour 1 and pour 2 (expert ~55 / >=37 tilted)
CARRY_Z = 0.77            # carry height toward the drainer (walls ~0.163 above table)
RELEASE_DZ = 0.185        # release TCP height above table inside the drainer slot
MIN_GAP = 0.02            # finger gap that indicates a held bottle (held ~0.06, empty ~0)


def _num(v, d):
    return d if v is None else v


def _act(tool, **args):
    return {"tool": tool, "args": args}


def _move(p, tilt=0.0, grip=-1.0, max_steps=300, min_steps=0, tol=0.004, rtol=2.5, vmax=1.0):
    return _act("act_move", x=float(p[0]), y=float(p[1]), z=float(p[2]), tilt_deg=float(tilt),
                gripper=float(grip), max_steps=int(max_steps), min_steps=int(min_steps),
                pos_tol=float(tol), rot_tol_deg=float(rtol), vmax=float(vmax))


def perceive_decide(obs, goal, memory, last):
    """Phase machine for 'pour sauce over the frypan twice, then bottle into the bowl drainer'."""
    m = dict(memory)
    for k in ("phase", "table_z", "bottle", "pan", "drainer", "pours_done", "grasp_tries", "idle_count", "note"):
        if k not in m:
            m[k] = None
    cur = obs.current
    tcp = [float(v) for v in np.asarray(cur.tcp, dtype=float)[:3]]
    f = np.asarray(cur.fingers, dtype=float)
    gap = float(f[0] - f[1])
    phase = m["phase"]
    pours = int(_num(m["pours_done"], 0))
    tries = int(_num(m["grasp_tries"], 0))
    idle = int(_num(m["idle_count"], 0))

    if phase is None or m["bottle"] is None or m["pan"] is None or m["drainer"] is None or m["table_z"] is None:
        s = _locate(cur)
        m["table_z"] = float(s["table_z"])
        m["bottle"] = [float(s["bottle"][0]), float(s["bottle"][1])]
        m["pan"] = [float(s["pan"][0]), float(s["pan"][1])]
        m["drainer"] = [float(s["drainer"][0]), float(s["drainer"][1])]
        m["note"] = "scene: " + ("; ".join(s["notes"]) if s["notes"] else "ok")
        phase = None
    tz = float(m["table_z"])
    bx, by = m["bottle"]
    px, py = m["pan"]
    dx, dy = m["drainer"]
    pre = [px + PRE_OFF[0], py + PRE_OFF[1], PRE_OFF[2]]
    pour = [px + POUR_OFF[0], py + POUR_OFF[1], POUR_OFF[2]]
    ret = [px + RETREAT_OFF[0], py + RETREAT_OFF[1], RETREAT_OFF[2]]
    holding = gap > MIN_GAP

    if phase is None or phase == "retry_up":
        if phase == "retry_up":
            s = _locate(cur)
            m["bottle"] = [float(s["bottle"][0]), float(s["bottle"][1])]
            bx, by = m["bottle"]
        nxt, act = "hover", _move([bx, by, tz + HOVER_Z], grip=-1.0, max_steps=250, tol=0.004)
    elif phase == "hover":
        nxt, act = "descend", _move([bx, by, tz + GRASP_Z], grip=-1.0, max_steps=250, tol=0.003, vmax=0.6)
    elif phase == "descend":
        nxt, act = "close", _act("act_gripper", close=True, steps=30)
    elif phase == "close":
        if holding or tries >= 2:
            nxt, act = "lift", _move([bx, by, LIFT_Z], grip=1.0, max_steps=200, tol=0.01, rtol=4.0)
        else:
            tries += 1
            nxt, act = "release_retry", _act("act_gripper", close=False, steps=20)
    elif phase == "release_retry":
        nxt, act = "retry_up", _move([tcp[0], tcp[1], tz + 0.25], grip=-1.0, max_steps=200, tol=0.01)
    elif phase == "lift":
        if not holding and tries < 2:
            tries += 1
            nxt, act = "release_retry", _act("act_gripper", close=False, steps=20)
        else:
            nxt, act = "pre_pour", _move(pre, grip=1.0, max_steps=250, tol=0.006, rtol=3.0)
    elif phase == "pre_pour":
        hold = POUR_HOLD[min(pours, 1)]
        nxt, act = "pour", _move(pour, tilt=POUR_TILT, grip=1.0, max_steps=hold + 60, min_steps=hold,
                                 tol=0.01, rtol=4.0, vmax=0.6)
    elif phase == "pour":
        pours += 1
        if pours < 2:
            nxt, act = "retreat", _move(ret, grip=1.0, max_steps=200, tol=0.01, rtol=4.0, vmax=0.6)
        else:
            nxt, act = "carry_up", _move([tcp[0], tcp[1], CARRY_Z], grip=1.0, max_steps=200, tol=0.01, rtol=4.0, vmax=0.6)
    elif phase == "retreat":
        nxt, act = "pre_pour", _move(pre, grip=1.0, max_steps=250, tol=0.006, rtol=3.0)
    elif phase == "carry_up":
        nxt, act = "carry", _move([dx, dy, CARRY_Z], grip=1.0, max_steps=300, tol=0.006, rtol=3.0)
    elif phase == "carry":
        nxt, act = "lower", _move([dx, dy, tz + RELEASE_DZ], grip=1.0, max_steps=250, tol=0.005, rtol=3.0, vmax=0.5)
    elif phase == "lower":
        nxt, act = "release", _act("act_gripper", close=False, steps=25)
    elif phase == "release":
        nxt, act = "done", _move([dx, dy, CARRY_Z], grip=-1.0, max_steps=200, tol=0.01, rtol=5.0, vmax=0.6)
    else:
        idle += 1
        nxt, act = "done", _act("act_idle", steps=200, gripper=-1.0)

    m["phase"] = nxt
    m["pours_done"] = int(pours)
    m["grasp_tries"] = int(tries)
    m["idle_count"] = int(idle)
    return {"memory": m, "act": act}
