"""rearrange_blocks policy: perceive blocks/mats/button from the head RGB-D camera, then
pick block-between-mats -> empty mat, press button, pick block-that-started-on-mat -> midpoint."""

# Offsets from perceived centres to the expert's commanded grasp-centre targets
# (mean over 25 demos: blocks/mats +4.1 mm x, +1.0 mm y; button +1.9 mm x, +1.1 mm y).
XY_BIAS = (0.004, 0.001)
BUTTON_BIAS = (0.002, 0.001)
# Expert grasp-centre heights (metres) relative to the table surface (0.741) / object tops.
GRASP_ABOVE_TOP = 0.001      # grasp centre ~1 mm above block top (0.782 for top 0.781)
PREGRASP_ABOVE_GRASP = 0.078  # 0.860
LIFT_ABOVE_TABLE = 0.138      # 0.879
TRANSFER_ABOVE_TABLE = 0.162  # 0.903
RELEASE_ABOVE_TABLE = 0.072   # 0.813
PREPRESS_ABOVE_TABLE = 0.102  # 0.843
PRESS_ABOVE_TABLE = 0.064     # 0.805
# Expert tool orientations (w, x, y, z): gripper pointing straight down.
RIGHT_DOWN_Q = (0.5, -0.5, 0.5, 0.5)
LEFT_DOWN_Q = (0.70710678, 0.0, 0.70710678, 0.0)
TCP_OFFSET = 0.12
ON_TOL = 0.022   # block counted as on the target if its centre is within this distance
MAX_RETRIES = 2


def _qmat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _ee_from_tcp(tcp, q):
    """End-effector pose (7) whose grasp centre (0.12 m along local +x) is at `tcp`."""
    ax = _qmat(q)[:, 0]
    p = np.array(tcp, dtype=float) - TCP_OFFSET * ax
    return [float(p[0]), float(p[1]), float(p[2])] + [float(v) for v in q]


def _tcp_of(pose):
    pose = [float(v) for v in pose]
    p = np.array(pose[:3]) + TCP_OFFSET * _qmat(pose[3:])[:, 0]
    return [float(p[0]), float(p[1]), float(p[2])]


def _send(robot, arm, pose, grip):
    """One native step moving `arm` to `pose`/`grip` while holding the other arm."""
    f = robot.now()
    lp = [float(v) for v in f.left_ee_pose]
    rp = [float(v) for v in f.right_ee_pose]
    lg = float(f.left_gripper)
    rg = float(f.right_gripper)
    if arm == "left":
        lp, lg = list(pose), float(grip)
    else:
        rp, rg = list(pose), float(grip)
    robot.step(*(lp + [lg] + rp + [rg]))


def _move(robot, arm, pose, grip, log):
    """Step with up to two fallbacks (slightly raised targets) when planning fails."""
    tries = [0.0, 0.01, 0.025]
    for k, dz in enumerate(tries):
        p = list(pose)
        p[2] = p[2] + dz
        try:
            _send(robot, arm, p, grip)
            if dz:
                log.append("raised %.3f" % dz)
            return True
        except Exception as e:  # MotionError (planning failure); EpisodeEnd is not an Exception
            log.append("motion error: %s" % str(e)[:80])
    return False


def _state(robot):
    f = robot.now()
    return {
        "left_tcp": _tcp_of(f.left_ee_pose),
        "right_tcp": _tcp_of(f.right_ee_pose),
        "left_gripper": float(f.left_gripper),
        "right_gripper": float(f.right_gripper),
    }


def _down_q(arm):
    return LEFT_DOWN_Q if arm == "left" else RIGHT_DOWN_Q


def act_observe(robot):
    """Hold both arms for one native step and report the arm state."""
    f = robot.now()
    robot.step(*([float(v) for v in f.left_ee_pose] + [float(f.left_gripper)] +
                 [float(v) for v in f.right_ee_pose] + [float(f.right_gripper)]))
    return _state(robot)


def act_pick(robot, arm, x, y, grasp_z, table_z):
    """Top-down pick with `arm` ('left'/'right') at grasp centre (x, y, grasp_z):
    open approach above, descend, close, lift to a safe height. Returns arm state and log."""
    q = _down_q(arm)
    log = []
    ok = _move(robot, arm, _ee_from_tcp([x, y, grasp_z + PREGRASP_ABOVE_GRASP], q), 1.0, log)
    ok = _move(robot, arm, _ee_from_tcp([x, y, grasp_z], q), 1.0, log) and ok
    pose = [float(v) for v in robot.now().left_ee_pose] if arm == "left" else [float(v) for v in robot.now().right_ee_pose]
    _move(robot, arm, pose, 0.0, log)
    ok = _move(robot, arm, _ee_from_tcp([x, y, table_z + LIFT_ABOVE_TABLE], q), 0.0, log) and ok
    out = _state(robot)
    out["ok"] = bool(ok)
    out["log"] = log
    return out


def act_place(robot, arm, x, y, table_z, home_pose):
    """Carry the held block to above (x, y), lower to the release height, open, lift and
    return `arm` to `home_pose` (7 values). Returns arm state and log."""
    q = _down_q(arm)
    log = []
    ok = _move(robot, arm, _ee_from_tcp([x, y, table_z + TRANSFER_ABOVE_TABLE], q), 0.0, log)
    ok = _move(robot, arm, _ee_from_tcp([x, y, table_z + RELEASE_ABOVE_TABLE], q), 0.0, log) and ok
    pose = [float(v) for v in robot.now().left_ee_pose] if arm == "left" else [float(v) for v in robot.now().right_ee_pose]
    _move(robot, arm, pose, 1.0, log)
    _move(robot, arm, _ee_from_tcp([x, y, table_z + LIFT_ABOVE_TABLE], q), 1.0, log)
    if home_pose:
        _move(robot, arm, [float(v) for v in home_pose], 1.0, log)
    out = _state(robot)
    out["ok"] = bool(ok)
    out["log"] = log
    return out


def act_press(robot, arm, x, y, table_z, home_pose):
    """Press a button at (x, y) top-down with a closed gripper: move above, close, press to
    the expert's press height, rise, return to `home_pose`."""
    q = _down_q(arm)
    log = []
    above = _ee_from_tcp([x, y, table_z + PREPRESS_ABOVE_TABLE], q)
    ok = _move(robot, arm, above, 1.0, log)
    _move(robot, arm, above, 0.0, log)
    ok = _move(robot, arm, _ee_from_tcp([x, y, table_z + PRESS_ABOVE_TABLE], q), 0.0, log) and ok
    _move(robot, arm, above, 0.0, log)
    if home_pose:
        _move(robot, arm, [float(v) for v in home_pose], 0.0, log)
    out = _state(robot)
    out["ok"] = bool(ok)
    out["log"] = log
    return out


def _arm_for(x):
    return "right" if x > -0.05 else "left"


def _nearest_red(scene, xy, exclude, max_d):
    """Red cluster nearest to xy whose centre is not within 0.03 of any excluded xy."""
    best, bd = None, max_d
    for rd in scene["reds"]:
        if any(_dist(rd["xy"], e) < 0.03 for e in exclude if e):
            continue
        d = _dist(rd["xy"], xy)
        if d < bd:
            best, bd = rd, d
    return best


def _blank_memory():
    return {"phase": "start", "table_z": None, "block_first": None, "block_second": None,
            "empty_mat": None, "occupied_mat": None, "mid_target": None, "button": None,
            "home": None, "retries": 0, "log": ""}


def perceive_decide(obs, goal, memory, last):
    """State machine: start -> picked1 -> placed1 -> pressed -> picked2 -> placed2 -> done."""
    mem = _blank_memory()
    for k in mem:
        if memory.get(k) is not None:
            mem[k] = memory[k]
    f = obs.current
    hold = {"tool": "act_observe", "args": {}}
    ph = mem["phase"]
    if mem["home"] is None:
        mem["home"] = [[float(v) for v in f.left_ee_pose], [float(v) for v in f.right_ee_pose]]
    home = {"left": mem["home"][0], "right": mem["home"][1]}

    if ph == "start":
        it = _interpret(_scene(f))
        if not it["ok"] or it["button"] is None:
            mem["retries"] = int(mem["retries"]) + 1
            mem["log"] = "perception failed: %s" % it.get("why", "no button")
            return {"memory": mem, "act": hold}
        tz = it["table_z"]
        mem["table_z"] = tz
        mem["block_first"] = [it["block_mid"][0], it["block_mid"][1], it["block_mid_top"]]
        mem["block_second"] = [it["block_mat"][0], it["block_mat"][1], it["block_mat_top"]]
        mem["empty_mat"] = list(it["empty_mat"])
        mem["occupied_mat"] = list(it["occupied_mat"])
        mem["mid_target"] = list(it["mid"])
        mem["button"] = list(it["button"])
        mem["retries"] = 0
        ph = "need_pick1"

    tz = float(mem["table_z"]) if mem["table_z"] is not None else TABLE_Z_DEFAULT

    def pick(block):
        x = block[0] + XY_BIAS[0]
        y = block[1] + XY_BIAS[1]
        gz = min(max(block[2] + GRASP_ABOVE_TOP, tz + 0.03), tz + 0.05)
        return {"tool": "act_pick", "args": {"arm": _arm_for(x), "x": x, "y": y, "grasp_z": gz, "table_z": tz}}

    def place(target, x_block):
        arm = _arm_for(x_block)
        return {"tool": "act_place", "args": {"arm": arm, "x": target[0] + XY_BIAS[0], "y": target[1] + XY_BIAS[1],
                                              "table_z": tz, "home_pose": home[arm]}}

    if ph == "need_pick1":
        mem["phase"] = "picked1"
        mem["log"] = "pick first block"
        return {"memory": mem, "act": pick(mem["block_first"])}
    if ph == "picked1":
        mem["phase"] = "placed1"
        mem["log"] = "place first block on empty mat"
        return {"memory": mem, "act": place(mem["empty_mat"], mem["block_first"][0])}
    if ph == "placed1":
        sc = _scene(f)
        on = _nearest_red(sc, mem["empty_mat"], [mem["button"], mem["block_second"]], ON_TOL)
        if on is not None or int(mem["retries"]) >= MAX_RETRIES:
            mem["phase"] = "pressed"
            mem["retries"] = 0
            mem["log"] = "first block on mat" if on is not None else "first block unverified; continue"
            bx = mem["button"][0] + BUTTON_BIAS[0]
            by = mem["button"][1] + BUTTON_BIAS[1]
            arm = "left" if bx < 0.05 else "right"
            return {"memory": mem, "act": {"tool": "act_press", "args": {"arm": arm, "x": bx, "y": by,
                                                                          "table_z": tz, "home_pose": home[arm]}}}
        # retry: find the first block wherever it is now (not the button, not the second block)
        rd = _nearest_red(sc, mem["empty_mat"], [mem["button"], mem["block_second"]], 0.25)
        mem["retries"] = int(mem["retries"]) + 1
        if rd is None:
            mem["log"] = "first block not found; retry original spot"
            mem["phase"] = "picked1"
            return {"memory": mem, "act": pick(mem["block_first"])}
        mem["block_first"] = [rd["xy"][0], rd["xy"][1], rd["top"]]
        mem["phase"] = "picked1"
        mem["log"] = "retry first block at %.3f %.3f" % (rd["xy"][0], rd["xy"][1])
        return {"memory": mem, "act": pick(mem["block_first"])}
    if ph == "pressed":
        sc = _scene(f)
        rd = _nearest_red(sc, mem["block_second"], [mem["button"], mem["empty_mat"]], 0.03)
        if rd is not None:
            mem["block_second"] = [rd["xy"][0], rd["xy"][1], rd["top"]]
        mem["phase"] = "picked2"
        mem["log"] = "pick second block"
        return {"memory": mem, "act": pick(mem["block_second"])}
    if ph == "picked2":
        mem["phase"] = "placed2"
        mem["log"] = "place second block between mats"
        return {"memory": mem, "act": place(mem["mid_target"], mem["block_second"][0])}
    if ph == "placed2":
        sc = _scene(f)
        on = _nearest_red(sc, mem["mid_target"], [mem["button"], mem["empty_mat"]], ON_TOL)
        if on is not None or int(mem["retries"]) >= MAX_RETRIES:
            mem["phase"] = "done"
            mem["log"] = "second block at midpoint" if on is not None else "second block unverified"
            return {"memory": mem, "act": hold}
        rd = _nearest_red(sc, mem["mid_target"], [mem["button"], mem["empty_mat"]], 0.25)
        mem["retries"] = int(mem["retries"]) + 1
        if rd is not None:
            mem["block_second"] = [rd["xy"][0], rd["xy"][1], rd["top"]]
        mem["phase"] = "picked2"
        mem["log"] = "retry second block"
        return {"memory": mem, "act": pick(mem["block_second"])}
    mem["phase"] = "done"
    return {"memory": mem, "act": hold}
