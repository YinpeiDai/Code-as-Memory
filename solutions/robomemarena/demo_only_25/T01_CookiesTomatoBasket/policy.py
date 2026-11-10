"""T01_CookiesTomatoBasket policy: perceive objects in agentview RGB-D, pick cookies into the
basket nearest the robot, remember that basket, then pick tomato sauce into the same basket."""

# ---- constants derived from the assigned demonstrations (see task_analysis.md) ----
_COOKIE_GRASP_OFF = [-0.002, -0.001]      # grasp xy minus perceived cookie-blob centroid
_COOKIE_GRASP_DZ = 0.003                  # grasp tcp z above table plane (expert: +0.002)
_TOMATO_GRASP_OFF = [-0.012, 0.003]       # grasp xy minus perceived (front-surface) can centroid
_TOMATO_GRASP_DZ = 0.069                  # grasp tcp z above table plane (expert: +0.069)
_BASKET_REL_OFF = [-0.002, 0.0]           # release xy minus basket bbox centre
_BASKET_REL_DZ = 0.088                    # release tcp z above basket rim (expert: +0.088)
_HOVER_DZ = 0.11                          # pre-grasp hover above table
_CARRY_DZ = 0.37                          # transport height above table (expert ~0.38)
_DEFAULT_TABLE_Z = 0.431
_MAX_RETRIES = 3                          # failed grasps per object before moving on
_HOLD_MIN, _HOLD_MAX = 0.01, 0.072        # finger gap (m) meaning "object held" (open 0.078-0.080, held 0.062)


def _aa2R(a):
    a = np.asarray(a, dtype=float)
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.eye(3)
    k = a / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * Kx @ Kx


def _logR(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-4:
        # near pi: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
    return w * th


def _rot_cmd(aa):
    """World-frame rotation command driving the gripper to point straight down, yaw 0."""
    Rd = np.diag([1.0, -1.0, -1.0])
    w = _logR(Rd @ _aa2R(aa).T)
    return [float(v) for v in np.clip(w * 4.0, -0.3, 0.3)]


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _move(robot, target, grip, vmax=1.0, tol=0.004, max_steps=300, gain=10.0):
    """P-control the TCP to world `target` holding gripper command `grip`. Returns final error (m)."""
    target = np.asarray(target, dtype=float)
    best = 1e9
    stall = 0
    err = 1e9
    for _ in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        e = target - tcp[:3]
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        if err < best - 0.0005:
            best = err
            stall = 0
        else:
            stall += 1
            if stall > 40:
                break
        a = np.clip(e * gain, -vmax, vmax)
        r = _rot_cmd(tcp[3:])
        robot.step(float(a[0]), float(a[1]), float(a[2]), r[0], r[1], r[2], grip)
    return err


def _gripper(robot, grip, min_steps=12, max_steps=45):
    """Hold position while the fingers open (-1) or close (+1) until they stop moving."""
    hold = np.asarray(robot.now().tcp, dtype=float)[:3]
    prev = _gap(robot.now())
    still = 0
    for i in range(int(max_steps)):
        fr = robot.now()
        tcp = np.asarray(fr.tcp, dtype=float)
        a = np.clip((hold - tcp[:3]) * 10.0, -0.3, 0.3)
        r = _rot_cmd(tcp[3:])
        robot.step(float(a[0]), float(a[1]), float(a[2]), r[0], r[1], r[2], grip)
        g = _gap(robot.now())
        still = still + 1 if abs(g - prev) < 0.0003 else 0
        prev = g
        if i + 1 >= min_steps and still >= 6:
            break
    return _gap(robot.now())


def _held(gap):
    return _HOLD_MIN < gap < _HOLD_MAX


def _pos3(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def _scene_targets(frame, table_z):
    """Grasp/release targets from one agentview frame; entries None when not detected."""
    sc = _detect_scene(frame)
    tz = sc["table_z"] if sc["table_z"] is not None else table_z
    if tz is None or not (0.2 < tz < 0.8):
        tz = _DEFAULT_TABLE_Z
    out = {"table_z": float(tz), "cookies": None, "tomato": None, "baskets": []}
    c = sc["cookies"]
    if c is not None:
        out["cookies"] = [c["xyz"][0] + _COOKIE_GRASP_OFF[0], c["xyz"][1] + _COOKIE_GRASP_OFF[1],
                          tz + _COOKIE_GRASP_DZ]
    t = sc["tomato"]
    if t is not None:
        out["tomato"] = [t["xyz"][0] + _TOMATO_GRASP_OFF[0], t["xyz"][1] + _TOMATO_GRASP_OFF[1],
                         tz + _TOMATO_GRASP_DZ]
    for b in sc["baskets"]:
        rim = tz + b["top"]
        out["baskets"].append([b["center"][0] + _BASKET_REL_OFF[0], b["center"][1] + _BASKET_REL_OFF[1],
                               rim + _BASKET_REL_DZ])
    return out


def _choose_basket(baskets):
    """Basket for the first object: the one nearest the robot (smallest world x), as in every demo."""
    if not baskets:
        return None
    return sorted(baskets, key=lambda b: b[0])[0]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: init -> pick_cookies -> place_cookies -> pick_tomato -> place_tomato -> done.

    Memory keeps object grasp targets, the basket chosen for the cookies (reused for the tomato
    sauce: 'the same basket'), the table height, the stage and a retry counter.
    """
    fr = obs.current
    mem = {
        "stage": memory.get("stage"),
        "table_z": memory.get("table_z"),
        "cookies_pos": memory.get("cookies_pos"),
        "tomato_pos": memory.get("tomato_pos"),
        "basket_pos": memory.get("basket_pos"),
        "placed": memory.get("placed"),
        "retries": memory.get("retries"),
        "repicked": memory.get("repicked"),
        "note": memory.get("note"),
    }
    if mem["placed"] is None:
        mem["placed"] = []
    if mem["repicked"] is None:
        mem["repicked"] = []
    if mem["retries"] is None:
        mem["retries"] = 0
    first_obj = "cookies"
    second_obj = "tomato"
    gl = str(goal).lower()
    if "tomato" in gl and "cookie" in gl and gl.find("tomato") < gl.find("cookie"):
        first_obj, second_obj = "tomato", "cookies"
    gap = _gap(fr)
    sc = _scene_targets(fr, mem["table_z"])
    if mem["table_z"] is None:
        mem["table_z"] = sc["table_z"]
    tz = float(mem["table_z"])
    stage = mem["stage"]

    nxt = {first_obj: "pick_" + second_obj, second_obj: "done"}
    if stage is None:
        # first decision: remember everything seen
        mem["cookies_pos"] = sc["cookies"]
        mem["tomato_pos"] = sc["tomato"]
        mem["basket_pos"] = _choose_basket(sc["baskets"])
        stage = "pick_" + first_obj
    elif stage.startswith("pick_"):
        obj = stage[5:]
        if _held(gap):
            stage = "place_" + obj
        else:
            mem["retries"] = int(mem["retries"]) + 1
            if sc[obj] is not None:
                mem[obj + "_pos"] = sc[obj]
            if mem["retries"] > _MAX_RETRIES:
                # give up on this object rather than loop forever; try the next one
                mem["retries"] = 0
                stage = nxt[obj]
    elif stage.startswith("place_"):
        obj = stage[6:]
        if obj not in mem["placed"]:
            mem["placed"] = list(mem["placed"]) + [obj]
        if sc[obj] is not None and obj not in mem["repicked"]:
            # object visible on the table outside the baskets after release: one re-pick,
            # with the retry budget nearly spent so a failed (empty) re-pick moves on
            mem[obj + "_pos"] = sc[obj]
            mem["repicked"] = list(mem["repicked"]) + [obj]
            mem["retries"] = _MAX_RETRIES
            stage = "pick_" + obj
        else:
            mem["retries"] = 0
            stage = nxt[obj]
    if stage == "pick_" + second_obj and mem["stage"] != stage and sc[second_obj] is not None:
        mem[second_obj + "_pos"] = sc[second_obj]   # fresh view of the next object
    if mem["basket_pos"] is None:
        mem["basket_pos"] = _choose_basket(sc["baskets"])

    mem["stage"] = stage
    carry = tz + _CARRY_DZ
    if stage.startswith("pick_"):
        obj = stage[5:]
        if mem[obj + "_pos"] is None and sc[obj] is not None:
            mem[obj + "_pos"] = sc[obj]
        tgt = mem[obj + "_pos"]
        if tgt is None:
            mem["note"] = "no target for " + obj
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 40}}}
        mem["note"] = "pick " + obj
        return {"memory": mem, "act": {"tool": "act_pick", "args": {
            "target": _pos3(tgt), "hover_z": tz + _HOVER_DZ + (0.06 if obj == "tomato" else 0.0),
            "carry_z": carry}}}
    if stage.startswith("place_"):
        tgt = mem["basket_pos"]
        if tgt is None:
            mem["note"] = "no basket seen"
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 40}}}
        mem["note"] = "place " + stage[6:]
        return {"memory": mem, "act": {"tool": "act_place", "args": {
            "target": _pos3(tgt), "carry_z": carry}}}
    mem["note"] = "done; waiting for the checker"
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 60}}}


def act_pick(robot, target, hover_z, carry_z):
    """Open, move above `target` at `hover_z`, descend, close, lift to `carry_z`.
    Returns finger gap, holding flag and final tcp. If empty, parks away from the camera view."""
    target = [float(v) for v in target]
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if _gap(robot.now()) < 0.07:
        _gripper(robot, -1.0)
    z0 = max(float(tcp[2]), float(hover_z))
    _move(robot, [tcp[0], tcp[1], z0], -1.0, tol=0.01, max_steps=150)
    _move(robot, [target[0], target[1], z0], -1.0, tol=0.008, max_steps=300)
    _move(robot, [target[0], target[1], hover_z], -1.0, tol=0.004, max_steps=200)
    e = _move(robot, target, -1.0, vmax=0.5, tol=0.003, max_steps=250)
    g_close = _gripper(robot, 1.0, min_steps=15)
    cur = np.asarray(robot.now().tcp, dtype=float)
    _move(robot, [cur[0], cur[1], cur[2] + 0.05], 1.0, vmax=0.6, tol=0.01, max_steps=60)
    _move(robot, [cur[0], cur[1], carry_z], 1.0, tol=0.01, max_steps=200)
    g = _gap(robot.now())
    held = _held(g)
    if not held:
        _gripper(robot, -1.0)
        cur = np.asarray(robot.now().tcp, dtype=float)
        _move(robot, [cur[0] - 0.15, cur[1], carry_z], -1.0, tol=0.01, max_steps=150)
    end = np.asarray(robot.now().tcp, dtype=float)
    return {"descend_err": e, "gap_after_close": g_close, "gap": g, "holding": held,
            "tcp": [float(v) for v in end[:3]]}


def act_place(robot, target, carry_z):
    """Carry at `carry_z` to above `target`, descend to it, open, wait, lift back to `carry_z`."""
    target = [float(v) for v in target]
    tcp = np.asarray(robot.now().tcp, dtype=float)
    _move(robot, [tcp[0], tcp[1], carry_z], 1.0, tol=0.01, max_steps=150)
    _move(robot, [target[0], target[1], carry_z], 1.0, tol=0.006, max_steps=300)
    e = _move(robot, target, 1.0, vmax=0.6, tol=0.004, max_steps=250)
    g_before = _gap(robot.now())
    _gripper(robot, -1.0, min_steps=20)
    cur = np.asarray(robot.now().tcp, dtype=float)
    _move(robot, [cur[0], cur[1], carry_z], -1.0, tol=0.01, max_steps=200)
    end = np.asarray(robot.now().tcp, dtype=float)
    return {"descend_err": e, "gap_before_release": g_before, "gap": _gap(robot.now()),
            "tcp": [float(v) for v in end[:3]]}


def act_wait(robot, steps=40):
    """Hold the current pose (gripper open) for `steps` steps, e.g. while the checker settles."""
    hold = np.asarray(robot.now().tcp, dtype=float)[:3]
    for _ in range(int(max(1, min(steps, 400)))):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        a = np.clip((hold - tcp[:3]) * 10.0, -0.3, 0.3)
        r = _rot_cmd(tcp[3:])
        robot.step(float(a[0]), float(a[1]), float(a[2]), r[0], r[1], r[2], -1.0)
    return {"waited": int(steps), "gap": _gap(robot.now())}
