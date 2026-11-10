"""blocks_ranking_try policy (Code-as-Memory, python runtime).

Task: three coloured cubes stand in a row (3 slots); a button stands at the left.
The hidden target ordering is unknown and pressing the button gives no visible
feedback, so the program enumerates the orderings: press the button, swap two
cubes (right arm), press again (left arm), ... using the expert's swap cycle
(slot pairs 2-3, 1-3, 1-2, 2-3, 1-3[, 1-2 back to start]) that visits all 6
permutations. The episode ends by itself when the arrangement is correct.
"""

# ---------------------------------------------------------------- constants
# Swap cycle observed in all 49 demonstrations (0-indexed slots). Followed by
# (0, 1) the cycle returns to the initial order, so it can be repeated.
SWAP_CYCLE = [[1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1]]
RIGHT_DOWN_Q = [0.5, -0.5, 0.5, 0.5]       # right gripper pointing down (demo ~0.507,-0.493,0.494,0.506)
RIGHT_DOWN_Q_ALT = [0.507, -0.493, 0.494, 0.506]
LEFT_DOWN_Q = [0.7071068, 0.0, 0.7071068, 0.0]  # left gripper pointing down (button press)
NOMINAL_SLOTS = [[0.042, -0.100], [0.161, -0.100], [0.280, -0.100]]  # fallback only
NOMINAL_BUTTON = [-0.200, -0.100, 0.780]  # fallback only (xy, top z)
TOP_BIAS_XY = [0.0047, 0.0005]   # expert grasp xy minus head-camera top-face centroid (demo fit)
GRASP_ABOVE_TOP = 0.116          # ee z at grasp = cube top + 0.116 (demo: top 0.780, ee 0.896)
PRE_ABOVE_GRASP = 0.034          # pre-grasp height (demo 0.930)
PLACE_ABOVE_GRASP = 0.020        # release height (demo 0.916)
LIFT_Z = 0.996                   # transport height (demo ~0.995)
RETREAT_Z = 1.014                # after release (demo ~1.014)
TEMP_DY = -0.098                 # temporary spot: same x, y - 0.098 (demo -0.198)
BTN_HOVER = 0.184                # left ee z = button top + 0.184 (demo 0.964)
BTN_PRESS = 0.146                # left ee z = button top + 0.146 (demo 0.926)
TABLE_Z = 0.741
MAX_MOVES_PER_ACT = 4


# ---------------------------------------------------------------- perception
def _cloud(frame, cam):
    d = np.asarray(getattr(frame, cam + "_depth"), dtype=np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    pw = (pc - E[:, 3]) @ E[:, :3]
    rgb = np.asarray(getattr(frame, cam + "_rgb")).reshape(-1, 3).astype(np.float64)
    return pw, rgb, d.reshape(-1) > 0


def _chroma(rgb):
    s = rgb.sum(-1, keepdims=True) + 1e-6
    return rgb / s


def _colored_points(frame):
    """Saturated points standing above the table in the cube work area."""
    pw, rgb, valid = _cloud(frame, "head")
    sat = rgb.max(-1) - rgb.min(-1)
    m = (valid & (pw[:, 2] > TABLE_Z + 0.008) & (pw[:, 2] < TABLE_Z + 0.08)
         & (pw[:, 0] > -0.09) & (pw[:, 0] < 0.42) & (pw[:, 1] > -0.32) & (pw[:, 1] < 0.06)
         & (sat > 70))
    return pw[m], rgb[m]


def _blob(pts):
    """Top-face centroid (bias-corrected to the expert grasp point) and top z."""
    zt = float(np.percentile(pts[:, 2], 95))
    top = pts[pts[:, 2] > zt - 0.006]
    c = top[:, :2].mean(0)
    return [float(c[0] + TOP_BIAS_XY[0]), float(c[1] + TOP_BIAS_XY[1])], zt


def _initial_scene(frame):
    """Cluster the three cubes by x into slots; returns slots, colours, top z."""
    pts, rgb = _colored_points(frame)
    out = {"ok": False, "slots": [list(s) for s in NOMINAL_SLOTS], "colors": [], "top_z": TABLE_Z + 0.039}
    if len(pts) < 60:
        return out
    xs = np.sort(pts[:, 0])
    gaps = np.diff(xs)
    if len(gaps) < 3:
        return out
    cut = np.sort(np.argsort(gaps)[-2:])
    if gaps[cut[0]] < 0.03 or gaps[cut[1]] < 0.03:
        return out
    bounds = [xs[0] - 1, (xs[cut[0]] + xs[cut[0] + 1]) / 2, (xs[cut[1]] + xs[cut[1] + 1]) / 2, xs[-1] + 1]
    slots, cols, tops = [], [], []
    for k in range(3):
        s = (pts[:, 0] >= bounds[k]) & (pts[:, 0] < bounds[k + 1])
        if s.sum() < 20:
            return out
        c, zt = _blob(pts[s])
        slots.append(c)
        tops.append(zt)
        cols.append([float(v) for v in rgb[s].mean(0)])
    out.update(ok=True, slots=slots, colors=cols, top_z=float(np.median(tops)))
    return out


def _locate(frame, ref_colors):
    """Locate each reference-coloured cube anywhere in the work area."""
    pts, rgb = _colored_points(frame)
    res = [None, None, None]
    if len(pts) < 20 or len(ref_colors) != 3:
        return res
    ch = _chroma(rgb)
    ref = _chroma(np.asarray(ref_colors, dtype=np.float64))
    dist = np.linalg.norm(ch[:, None, :] - ref[None, :, :], axis=-1)
    lab = np.argmin(dist, 1)
    good = dist.min(1) < 0.08
    for k in range(3):
        s = good & (lab == k)
        if s.sum() < 15:
            continue
        q = pts[s]
        # keep the largest spatial group (reject stray pixels)
        med = np.median(q[:, :2], 0)
        q = q[np.linalg.norm(q[:, :2] - med, axis=1) < 0.04]
        if len(q) < 15:
            continue
        c, zt = _blob(q)
        res[k] = {"xy": c, "top": zt, "n": int(len(q))}
    return res


def _slot_of(xy, slots, tol=0.03):
    for i, s in enumerate(slots):
        if math.hypot(xy[0] - s[0], xy[1] - s[1]) < tol:
            return i
    return -1


def _plan_moves(cur, target_slot_of_color, slots, tol=0.03):
    """Pick/place moves (right arm) bringing colour k from cur[k] to slots[target[k]].
    A cube whose target is blocked goes to a temporary spot (same x, y+TEMP_DY)."""
    cur = [list(c) for c in cur]
    moves = []
    for _ in range(8):
        mis = [k for k in range(3)
               if math.hypot(cur[k][0] - slots[target_slot_of_color[k]][0],
                             cur[k][1] - slots[target_slot_of_color[k]][1]) > tol]
        if not mis:
            break
        done = False
        for k in mis:
            t = slots[target_slot_of_color[k]]
            blocked = any(j != k and math.hypot(cur[j][0] - t[0], cur[j][1] - t[1]) < 0.035 for j in range(3))
            if not blocked:
                moves.append([cur[k][0], cur[k][1], t[0], t[1], k])
                cur[k] = [t[0], t[1]]
                done = True
                break
        if done:
            continue
        # every target blocked (a cycle): park the misplaced cube with lowest slot index
        k = sorted(mis, key=lambda j: (max(_slot_of(cur[j], slots), 0), cur[j][0]))[0]
        tmp = [cur[k][0], cur[k][1] + TEMP_DY]
        moves.append([cur[k][0], cur[k][1], tmp[0], tmp[1], k])
        cur[k] = tmp
    return moves


# ---------------------------------------------------------------- decision
def _blank_memory():
    return {"stage": "start", "swaps_done": 0, "presses": 0, "retries": 0,
            "slots": [], "button": [], "top_z": 0.0, "ref_colors": [],
            "order": [], "park_pose": [], "log": ""}


def perceive_decide(obs, goal, memory, last):
    """Enumerate cube orderings: press button, swap per SWAP_CYCLE, verify, press, ..."""
    mem = _blank_memory()
    for k in mem:
        if isinstance(memory, dict) and k in memory and memory[k] is not None:
            mem[k] = memory[k]
    frame = obs.current
    if mem["stage"] == "start":
        first = obs.frames[0]
        scene = _initial_scene(first)
        mem["slots"] = scene["slots"]
        mem["top_z"] = scene["top_z"]
        mem["ref_colors"] = scene["colors"] if scene["ok"] else []
        mem["order"] = [0, 1, 2]          # order[slot] = colour index standing there
        mem["button"] = _find_button(first)
        mem["park_pose"] = [float(v) for v in first.right_ee_pose]
        mem["log"] = "scene ok=%s slots=%s button=%s" % (scene["ok"], _r(mem["slots"]), _r(mem["button"]))
        mem["stage"] = "press"
    if mem["stage"] == "press":
        mem["presses"] = int(mem["presses"]) + 1
        mem["stage"] = "swap"
        mem["retries"] = 0
        b = mem["button"]
        return {"memory": mem, "act": {"tool": "act_press_button",
                "args": {"bx": b[0], "by": b[1], "top_z": b[2]}}}
    # stage swap / verify: work out the target order and the current positions
    pair = SWAP_CYCLE[int(mem["swaps_done"]) % len(SWAP_CYCLE)]
    order = list(mem["order"])
    target = list(order)
    if mem["stage"] == "swap":
        target[pair[0]], target[pair[1]] = order[pair[1]], order[pair[0]]
    else:  # verify: target already stored in order
        target = order
    slots = mem["slots"]
    tsc = [0, 0, 0]
    for s in range(3):
        tsc[target[s]] = s
    if len(mem["ref_colors"]) == 3:
        loc = _locate(frame, mem["ref_colors"])
    else:
        loc = [None, None, None]
    if mem["stage"] == "swap":
        # positions believed from memory order; perception refines them
        cur = [None, None, None]
        for s in range(3):
            cur[order[s]] = list(slots[s])
        for k in range(3):
            if loc[k] is not None and math.hypot(loc[k]["xy"][0] - cur[k][0], loc[k]["xy"][1] - cur[k][1]) < 0.05:
                cur[k] = loc[k]["xy"]
            elif loc[k] is not None and int(mem["retries"]) > 0:
                cur[k] = loc[k]["xy"]
        moves = _plan_moves(cur, tsc, slots)
        mem["order"] = target
        mem["stage"] = "verify"
        mem["log"] = "swap %s -> order %s moves %s" % (pair, target, _r(moves))
        return {"memory": mem, "act": {"tool": "act_arrange",
                "args": {"moves": moves[:MAX_MOVES_PER_ACT], "top_z": mem["top_z"],
                         "park": mem["park_pose"], "ref_colors": mem["ref_colors"]}}}
    # verify the arrangement before pressing
    ok = True
    cur = [None, None, None]
    for k in range(3):
        if loc[k] is None:
            cur[k] = list(slots[tsc[k]])   # unseen: assume in place
            continue
        cur[k] = loc[k]["xy"]
        t = slots[tsc[k]]
        if math.hypot(cur[k][0] - t[0], cur[k][1] - t[1]) > 0.02:
            ok = False
    missing = [k for k in range(3) if loc[k] is None]
    moves = [] if ok else _plan_moves(cur, tsc, slots, 0.012)
    if ok or len(moves) == 0 or int(mem["retries"]) >= 3:
        mem["swaps_done"] = int(mem["swaps_done"]) + 1
        mem["log"] = "verify ok=%s missing=%s retries=%s" % (ok, missing, mem["retries"])
        mem["presses"] = int(mem["presses"]) + 1
        mem["stage"] = "swap"
        mem["retries"] = 0
        b = mem["button"]
        return {"memory": mem, "act": {"tool": "act_press_button",
                "args": {"bx": b[0], "by": b[1], "top_z": b[2]}}}
    mem["retries"] = int(mem["retries"]) + 1
    mem["log"] = "repair missing=%s moves %s" % (missing, _r(moves))
    return {"memory": mem, "act": {"tool": "act_arrange",
            "args": {"moves": moves[:MAX_MOVES_PER_ACT], "top_z": mem["top_z"],
                     "park": mem["park_pose"], "ref_colors": mem["ref_colors"]}}}


def _r(x):
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    if isinstance(x, float):
        return round(x, 3)
    return x


def _find_button(frame):
    """Red button: saturated red blob left of the cubes; returns [x, y, top_z]."""
    pw, rgb, valid = _cloud(frame, "head")
    sat = rgb.max(-1) - rgb.min(-1)
    m = (valid & (pw[:, 2] > TABLE_Z + 0.008) & (pw[:, 2] < TABLE_Z + 0.08)
         & (pw[:, 0] < -0.08) & (pw[:, 0] > -0.42) & (pw[:, 1] > -0.32) & (pw[:, 1] < 0.08)
         & (sat > 80) & (rgb[:, 0] > 150) & (rgb[:, 1] < 0.6 * rgb[:, 0]) & (rgb[:, 2] < 0.6 * rgb[:, 0]))
    q = pw[m]
    if len(q) < 40:
        return list(NOMINAL_BUTTON)
    zt = float(np.percentile(q[:, 2], 95))
    top = q[q[:, 2] > zt - 0.006]
    c = top[:, :2].mean(0)
    return [float(c[0]), float(c[1]), zt]


# ---------------------------------------------------------------- acting
def _pose(arm_now):
    return [float(v) for v in arm_now]


def _step(robot, left, lg, right, rg, log):
    """One native step; on a planning failure retry same target, then +-4 mm in z
    for the arm(s) that actually move. Returns True when a step succeeded."""
    now = robot.now()
    lmov = math.dist([float(v) for v in now.left_ee_pose[:3]], left[:3]) > 0.002
    rmov = math.dist([float(v) for v in now.right_ee_pose[:3]], right[:3]) > 0.002
    for i, dz in enumerate([0.0, 0.0, 0.004, -0.004]):
        l = [float(v) for v in left]
        r = [float(v) for v in right]
        if lmov:
            l[2] += dz
        if rmov:
            r[2] += dz
        try:
            robot.step(*l[:7], float(lg), *r[:7], float(rg))
            return True
        except Exception as e:  # MotionError: arm stayed still; EpisodeEnd propagates
            log.append("step fail %d: %s" % (i, str(e)[:80]))
    return False


def act_press_button(robot, bx, by, top_z):
    """Move left arm above the button (closed gripper), press down and release."""
    log = []
    now = robot.now()
    right = _pose(now.right_ee_pose)
    rg = float(now.right_gripper)
    hover = [bx, by, top_z + BTN_HOVER] + LEFT_DOWN_Q
    press = [bx, by, top_z + BTN_PRESS] + LEFT_DOWN_Q
    lpos = now.left_ee_pose
    near = math.hypot(lpos[0] - bx, lpos[1] - by) < 0.01 and abs(lpos[2] - hover[2]) < 0.02
    if not near:
        _step(robot, hover, float(now.left_gripper), right, rg, log)
    if float(robot.now().left_gripper) > 0.1:
        _step(robot, hover, 0.0, right, rg, log)
    _step(robot, press, 0.0, right, rg, log)
    _step(robot, press, 0.0, right, rg, log)  # dwell at the bottom
    ok = _step(robot, hover, 0.0, right, rg, log)
    z = float(robot.now().left_ee_pose[2])
    return {"pressed": True, "final_left_z": z, "back_up": ok, "log": log[-5:]}


def _still_there(frame, ref_colors, k, xy):
    """True when colour k is still seen standing near xy (grasp failed)."""
    if k < 0 or len(ref_colors) != 3 or not hasattr(frame, "head_depth"):
        return False, None
    loc = _locate(frame, ref_colors)
    if loc[k] is None:
        return False, None
    d = math.hypot(loc[k]["xy"][0] - xy[0], loc[k]["xy"][1] - xy[1])
    return d < 0.03, loc[k]["xy"]


def act_arrange(robot, moves, top_z, park, ref_colors=None):
    """Right arm pick/place moves [[px,py,qx,qy,colour],...] then park at the start pose.
    After each lift the head camera checks the cube left its spot; one re-grasp, else abort."""
    log = []
    refs = ref_colors or []
    gz = min(max(top_z + GRASP_ABOVE_TOP, 0.886), 0.906)
    now = robot.now()
    left = _pose(now.left_ee_pose)
    lg = float(now.left_gripper)
    q = RIGHT_DOWN_Q
    done = 0
    aborted = False

    def go(p, g):
        okk = _step(robot, left, lg, p + q, g, log)
        if not okk:
            okk = _step(robot, left, lg, p + RIGHT_DOWN_Q_ALT, g, log)
        return okk

    for mv in moves:
        px, py, qx, qy = [float(v) for v in mv[:4]]
        k = int(mv[4]) if len(mv) > 4 else -1
        grasped = False
        for attempt in range(2):
            go([px, py, LIFT_Z], 1.0)
            go([px, py, gz + PRE_ABOVE_GRASP], 1.0)
            go([px, py, gz], 1.0)
            go([px, py, gz], 0.0)
            go([px, py, LIFT_Z], 0.0)
            failed, seen = _still_there(robot.now(), refs, k, [px, py])
            if not failed:
                grasped = True
                break
            log.append("grasp miss colour %d at %s" % (k, _r([px, py])))
            go([px, py, LIFT_Z], 1.0)
            if seen is not None:
                px, py = seen[0], seen[1]
        if not grasped:
            aborted = True
            break
        go([qx, qy, LIFT_Z], 0.0)
        go([qx, qy, gz + PLACE_ABOVE_GRASP], 0.0)
        go([qx, qy, gz + PLACE_ABOVE_GRASP], 1.0)
        go([qx, qy, RETREAT_Z], 1.0)
        done += 1
    if len(park) == 7:
        _step(robot, left, lg, list(park), 1.0, log)
    return {"moves": len(moves), "moves_done": done, "aborted": aborted, "log": log[-6:]}
