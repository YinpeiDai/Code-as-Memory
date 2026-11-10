"""T16 PourMilkMugTwice: perceive milk / mug / drainer once from the agentview RGB-D frame,
then run a phase machine that mirrors the structure of the expert recordings:
approach -> descend -> close -> lift -> move over mug -> tilt -pi/3 about world y -> hold ->
untilt -> move over mug -> tilt -> hold -> (fallback) untilt -> carry to drainer -> lower -> release.
"""

# ---------------------------------------------------------------- constants (from demo analysis)
TABLE_Z_DEFAULT = 0.43
MILK_PRIOR = [-0.0765, -0.1754]      # mean perceived milk median xy over 25 demos
MUG_PRIOR = [0.1889, -0.0284]        # mean perceived mug median xy over 25 demos
DRAINER_PRIOR = [-0.155, 0.52, 0.42, 0.62]   # center x, center y, y_min, y_max
GRASP_OFFSET = [-0.0255, -0.0023]    # expert grasp target - perceived milk median (std < 1 mm)
POUR_OFFSET = [-0.0137, -0.0136]     # expert pour target - perceived mug median (std ~1.5 mm)
APPROACH_Z = 0.69
GRASP_Z = 0.51
LIFT_Z = 0.839
POUR_Z = 0.676
DROP_Z = 0.61                        # fallback release height (tcp); carton bottom ~0.09 m below tcp
POUR_ANGLE = -1.0471975511965976     # -pi/3 about world y relative to gripper-down
PHASES = ["approach", "descend", "close", "lift", "move1", "tilt1", "hold1", "untilt1",
          "move2", "tilt2", "hold2", "untilt2", "carry", "lower", "release", "retreat", "idle"]


# ---------------------------------------------------------------- geometry helpers
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        return V[:, int(np.argmax(w))] * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return k * th


def _down():
    return np.diag([1.0, -1.0, -1.0])


def _tilt_target(angle):
    c, s = math.cos(angle), math.sin(angle)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return Ry @ _down()


def _rot_err(Rt, tcp):
    return _mat2aa(Rt @ _aa2mat(tcp[3:6]).T)


def _clip(v, lim):
    return [max(-lim, min(lim, float(x))) for x in v]


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R, z > 0


def _perceive_scene(frame):
    """Milk / mug / drainer estimates from one agentview RGB-D frame (world metres)."""
    out = {"ok": False, "milk": list(MILK_PRIOR), "mug": list(MUG_PRIOR),
           "drainer": list(DRAINER_PRIOR), "table_z": TABLE_Z_DEFAULT,
           "n_milk": 0, "n_mug": 0, "n_drainer": 0, "notes": ""}
    names = frame.sensors
    if not ("agentview_depth" in names and "agentview_K" in names and "agentview_extrinsic" in names):
        out["notes"] = "missing agentview depth/calibration; using priors"
        return out
    pw, ok = _backproject(np.asarray(frame.agentview_depth), np.asarray(frame.agentview_K, dtype=float),
                          np.asarray(frame.agentview_extrinsic, dtype=float))
    pts = pw[ok]
    # table height: dominant z among points over the table area
    tab = pts[(np.abs(pts[:, 1]) < 0.6) & (pts[:, 0] > -0.4) & (pts[:, 0] < 0.4)
              & (pts[:, 2] > 0.3) & (pts[:, 2] < 0.6)]
    table_z = TABLE_Z_DEFAULT
    if len(tab) > 500:
        hist, edges = np.histogram(tab[:, 2], bins=60, range=(0.3, 0.6))
        i = int(np.argmax(hist))
        cand = 0.5 * (edges[i] + edges[i + 1])
        if abs(cand - TABLE_Z_DEFAULT) < 0.05:
            sel = tab[np.abs(tab[:, 2] - cand) < 0.01, 2]
            table_z = float(np.median(sel)) if len(sel) else float(cand)
    out["table_z"] = table_z
    lo = table_z + 0.015
    tcp = np.asarray(frame.tcp, dtype=float) if "tcp" in names else np.array([9.0, 9.0, 9.0])
    m = ((pts[:, 2] > lo) & (pts[:, 2] < 0.65) & (np.abs(pts[:, 1]) < 0.7)
         & (pts[:, 0] > -0.4) & (pts[:, 0] < 0.4))
    near_hand = (np.hypot(pts[:, 0] - tcp[0], pts[:, 1] - tcp[1]) < 0.05) & (pts[:, 2] > tcp[2] - 0.08)
    obj = pts[m & ~near_hand]
    milk = obj[(obj[:, 1] > -0.35) & (obj[:, 1] < -0.1) & (obj[:, 0] < 0.05) & (obj[:, 2] < 0.62)]
    mug = obj[(obj[:, 0] > 0.05) & (np.abs(obj[:, 1]) < 0.15) & (obj[:, 2] < 0.62)]
    dr = obj[(obj[:, 1] > 0.3)]
    out["n_milk"], out["n_mug"], out["n_drainer"] = int(len(milk)), int(len(mug)), int(len(dr))
    notes = []
    good = True
    if len(milk) >= 30:
        mk = np.median(milk[:, :2], 0)
        if np.hypot(mk[0] - MILK_PRIOR[0], mk[1] - MILK_PRIOR[1]) < 0.12:
            out["milk"] = [float(mk[0]), float(mk[1])]
        else:
            good = False
            notes.append("milk far from prior")
    else:
        good = False
        notes.append("milk not segmented")
    if len(mug) >= 30:
        mg = np.median(mug[:, :2], 0)
        if np.hypot(mg[0] - MUG_PRIOR[0], mg[1] - MUG_PRIOR[1]) < 0.12:
            out["mug"] = [float(mg[0]), float(mg[1])]
        else:
            good = False
            notes.append("mug far from prior")
    else:
        good = False
        notes.append("mug not segmented")
    if len(dr) >= 100:
        top = dr[dr[:, 2] > np.max(dr[:, 2]) - 0.02]
        x0, x1 = float(np.percentile(top[:, 0], 2)), float(np.percentile(top[:, 0], 98))
        y0, y1 = float(np.percentile(top[:, 1], 2)), float(np.percentile(top[:, 1], 98))
        if 0.08 < x1 - x0 < 0.4 and 0.08 < y1 - y0 < 0.4:
            out["drainer"] = [0.5 * (x0 + x1), 0.5 * (y0 + y1), y0, y1]
        else:
            notes.append("drainer bbox implausible")
    else:
        notes.append("drainer not segmented")
    out["ok"] = good
    out["notes"] = "; ".join(notes)
    return out


def _plan(scene):
    """Phase targets derived from the perceived scene."""
    mk, mg, dr = scene["milk"], scene["mug"], scene["drainer"]
    gx, gy = mk[0] + GRASP_OFFSET[0], mk[1] + GRASP_OFFSET[1]
    px, py = mg[0] + POUR_OFFSET[0], mg[1] + POUR_OFFSET[1]
    # near-robot half of the drainer (compartment between its near wall and its centre divider)
    dy = min(dr[2] + 0.05, dr[1])
    return {"grasp": [gx, gy], "pour": [px, py], "drop": [dr[0], dy]}


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """Phase machine for 'pick milk, pour into mug twice, place milk into bowl drainer'.

    Memory: phase (current phase label), milk_xy / mug_xy / drainer (perceived once on the first
    decision), pours_done (completed tilt-and-hold cycles), decisions (counter), log (short text).
    Each decision runs one whole phase through one act_* call."""
    frame = obs.current
    phase = memory.get("phase")
    milk = memory.get("milk_xy")
    mug = memory.get("mug_xy")
    drainer = memory.get("drainer")
    pours = memory.get("pours_done")
    decisions = memory.get("decisions")
    log = memory.get("log")
    pours = 0 if pours is None else int(pours)
    decisions = 0 if decisions is None else int(decisions)
    log = "" if log is None else str(log)

    if phase is None or milk is None or mug is None or drainer is None:
        scene = _perceive_scene(frame)
        milk, mug, drainer = scene["milk"], scene["mug"], scene["drainer"]
        phase = "approach"
        log = ("scene ok=%s milk=(%.3f,%.3f) mug=(%.3f,%.3f) drainer=(%.3f,%.3f) n=%d/%d/%d %s" %
               (scene["ok"], milk[0], milk[1], mug[0], mug[1], drainer[0], drainer[1],
                scene["n_milk"], scene["n_mug"], scene["n_drainer"], scene["notes"]))
    else:
        # advance after the previous act finished
        if phase in PHASES:
            i = PHASES.index(phase)
            if phase in ("hold1", "hold2"):
                pours += 1
            phase = PHASES[min(i + 1, len(PHASES) - 1)]
        else:
            phase = "idle"

    plan = _plan({"milk": milk, "mug": mug, "drainer": drainer})
    g, p, d = plan["grasp"], plan["pour"], plan["drop"]
    fingers_closed = 1.0

    if phase == "approach":
        act = {"tool": "act_move", "args": {"target": [g[0], g[1], APPROACH_Z], "gain": 6.0, "tol": 0.02,
                                            "max_steps": 120, "gripper": -1.0, "rot_gain": 4.0, "fixed": False}}
    elif phase == "descend":
        act = {"tool": "act_move", "args": {"target": [g[0], g[1], GRASP_Z], "gain": 2.5, "tol": 0.0,
                                            "max_steps": 100, "gripper": -1.0, "rot_gain": 4.0, "fixed": True}}
    elif phase == "close":
        act = {"tool": "act_hold", "args": {"steps": 30, "gripper": fingers_closed}}
    elif phase == "lift":
        act = {"tool": "act_move", "args": {"target": [g[0], g[1], LIFT_Z], "gain": 5.0, "tol": 0.02,
                                            "max_steps": 150, "gripper": fingers_closed, "rot_gain": 0.0,
                                            "fixed": False}}
    elif phase in ("move1", "move2"):
        act = {"tool": "act_move", "args": {"target": [p[0], p[1], POUR_Z], "gain": 6.0, "tol": 0.03,
                                            "max_steps": 150, "gripper": fingers_closed, "rot_gain": 0.0,
                                            "fixed": False}}
    elif phase in ("tilt1", "tilt2"):
        act = {"tool": "act_rotate", "args": {"angle_y": POUR_ANGLE, "steps": 30, "gripper": fingers_closed}}
    elif phase in ("hold1", "hold2"):
        act = {"tool": "act_hold", "args": {"steps": 30 if phase == "hold1" else 40, "gripper": fingers_closed}}
    elif phase in ("untilt1", "untilt2"):
        act = {"tool": "act_rotate", "args": {"angle_y": 0.0, "steps": 30, "gripper": fingers_closed}}
    elif phase == "carry":
        act = {"tool": "act_carry", "args": {"target": [d[0], d[1], LIFT_Z], "gripper": fingers_closed,
                                             "max_steps": 300}}
    elif phase == "lower":
        act = {"tool": "act_move", "args": {"target": [d[0], d[1], DROP_Z], "gain": 4.0, "tol": 0.01,
                                            "max_steps": 120, "gripper": fingers_closed, "rot_gain": 2.0,
                                            "fixed": False}}
    elif phase == "release":
        act = {"tool": "act_hold", "args": {"steps": 30, "gripper": -1.0}}
    elif phase == "retreat":
        act = {"tool": "act_move", "args": {"target": [d[0], d[1], LIFT_Z], "gain": 4.0, "tol": 0.01,
                                            "max_steps": 200, "gripper": -1.0, "rot_gain": 2.0,
                                            "fixed": False}}
    else:
        act = {"tool": "act_hold", "args": {"steps": 200, "gripper": -1.0}}
    decisions += 1
    memory_out = {"phase": phase, "milk_xy": [float(milk[0]), float(milk[1])],
                  "mug_xy": [float(mug[0]), float(mug[1])],
                  "drainer": [float(x) for x in drainer], "pours_done": pours,
                  "decisions": decisions, "log": log[-400:]}
    return {"memory": memory_out, "act": act}


# ---------------------------------------------------------------- actions
def act_move(robot, target, gain=6.0, tol=0.02, max_steps=100, gripper=-1.0, rot_gain=0.0, fixed=False):
    """Proportional Cartesian move toward target [x,y,z] (world m): action = clip(gain*(target-tcp), +-1).
    If rot_gain>0 also servo orientation to gripper-down (clip 0.8). Stops when |error|<tol (checked
    before each step) unless fixed=True, in which case exactly max_steps steps run.
    Returns steps, final tcp xyz and final error."""
    tg = np.asarray(target, dtype=float)
    n = 0
    tcp = np.asarray(robot.now().tcp, dtype=float)
    err = float(np.linalg.norm(tg - tcp[:3]))
    max_steps = int(max_steps)
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        e = tg - tcp[:3]
        err = float(np.linalg.norm(e))
        if (not fixed) and err < tol:
            break
        pos = _clip(gain * e, 1.0)
        if rot_gain > 0:
            rot = _clip(rot_gain * _rot_err(_down(), tcp), 0.8)
        else:
            rot = [0.0, 0.0, 0.0]
        robot.step(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], float(gripper))
        n += 1
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "tcp": [float(x) for x in tcp[:3]],
            "err": float(np.linalg.norm(tg - tcp[:3])),
            "fingers": [float(x) for x in np.asarray(robot.now().fingers)]}


def act_rotate(robot, angle_y=0.0, steps=30, gripper=1.0):
    """Rotate in place (zero translation command) toward gripper-down rotated by angle_y rad about world y,
    rotation action = clip(2*axis-angle error, +-0.8), for a fixed number of steps.
    Returns final tcp and remaining orientation error (rad)."""
    Rt = _tilt_target(float(angle_y))
    for _ in range(int(steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        rot = _clip(2.0 * _rot_err(Rt, tcp), 0.8)
        robot.step(0.0, 0.0, 0.0, rot[0], rot[1], rot[2], float(gripper))
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"tcp": [float(x) for x in tcp[:3]],
            "rot_err": float(np.linalg.norm(_rot_err(Rt, tcp))),
            "fingers": [float(x) for x in np.asarray(robot.now().fingers)]}


def act_hold(robot, steps=30, gripper=0.0):
    """Send zero motion with the given gripper command for a number of steps (close/hold/release)."""
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"tcp": [float(x) for x in tcp[:3]],
            "fingers": [float(x) for x in np.asarray(robot.now().fingers)]}


def act_carry(robot, target, gripper=1.0, max_steps=300):
    """Carry the held object: first rise to target z at the current xy, then move to target xy,
    keeping gripper-down orientation. Returns final tcp and error."""
    tg = np.asarray(target, dtype=float)
    n = 0
    max_steps = int(max_steps)
    stage = 0
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, dtype=float)
        if stage == 0:
            goal = np.array([tcp[0], tcp[1], tg[2]])
            if abs(tg[2] - tcp[2]) < 0.02:
                stage = 1
                continue
        else:
            goal = tg
            if float(np.linalg.norm(tg - tcp[:3])) < 0.015:
                break
        pos = _clip(5.0 * (goal - tcp[:3]), 1.0)
        rot = _clip(2.0 * _rot_err(_down(), tcp), 0.8)
        robot.step(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], float(gripper))
        n += 1
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"steps": n, "tcp": [float(x) for x in tcp[:3]], "err": float(np.linalg.norm(tg - tcp[:3]))}
