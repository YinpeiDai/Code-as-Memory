"""blocks_ranking_try policy: try cube permutations (one slot swap at a time) and press the button after each.

The correct order is hidden; pressing the button with the correct order ends the episode with success.
Memory keeps the slot geometry, the believed arrangement of cube ids, the set of tried arrangements and
the current swap stage, so no arrangement is pressed twice.
"""

TABLE_Z = 0.741
CUBE_TOP_Z = 0.781
NOMINAL_SLOTS = [[0.04, -0.10], [0.16, -0.10], [0.28, -0.10]]
NOMINAL_BUTTON = [-0.20, -0.10]
TEMP_DY = -0.10            # temporary spot is 10 cm towards the robot from the vacated slot (expert used y=-0.2)
Q_RIGHT_DOWN = [0.5, -0.5, 0.5, 0.5]     # expert right-arm grasp orientation (gripper +x points down)
Q_LEFT_DOWN = [0.7071, 0.0, 0.7071, 0.0]  # expert left-arm press orientation
RIGHT_HOME = [0.306, -0.313, 0.941, 0.704, 0.0, 0.0, 0.711]
Z_ABOVE = 0.996
Z_GRASP = 0.896
Z_RELEASE = 0.916
BTN_Z_ABOVE = 0.964
BTN_Z_PRESS = 0.926
# Expert's cyclic transposition order over slot indices (0-based); visits all 6 permutations.
SWAP_CYCLE = [[1, 2], [0, 2], [0, 1]]
MATCH_TOL = 0.035
GRASP_DX = 0.0035          # expert grasped ~3.5 mm +x of the measured cube centre in all 25 demos
MAX_PRESS_RETRY = 2


def _world_points(frame, cam):
    depth = np.asarray(getattr(frame, cam + "_depth"), dtype=np.float64) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    ext = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1).reshape(-1, 3)
    pw = (pc - ext[:, 3]) @ ext[:, :3]
    return pw, depth.reshape(-1) > 0


def _clusters(pts, cell=0.01, min_pts=25):
    """Connected components of a 1 cm xy occupancy grid; returns list of (centre_xy, count, member_mask)."""
    out = []
    if len(pts) == 0:
        return out
    x0, y0 = -0.46, -0.41
    ix = np.floor((pts[:, 0] - x0) / cell).astype(int)
    iy = np.floor((pts[:, 1] - y0) / cell).astype(int)
    nx, ny = int(ix.max()) + 2, int(iy.max()) + 2
    grid = np.zeros((nx, ny), dtype=int)
    np.add.at(grid, (ix, iy), 1)
    occ = grid >= 2
    label = -np.ones((nx, ny), dtype=int)
    comps = []
    for sx, sy in zip(*np.nonzero(occ)):
        if label[sx, sy] >= 0:
            continue
        k = len(comps)
        stack = [(int(sx), int(sy))]
        label[sx, sy] = k
        cells = []
        while stack:
            cx, cy = stack.pop()
            cells.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ax, ay = cx + dx, cy + dy
                    if 0 <= ax < nx and 0 <= ay < ny and occ[ax, ay] and label[ax, ay] < 0:
                        label[ax, ay] = k
                        stack.append((ax, ay))
        comps.append(cells)
    pl = label[ix, iy]
    for k in range(len(comps)):
        mem = pl == k
        n = int(mem.sum())
        if n >= min_pts:
            out.append((np.median(pts[mem, :2], axis=0), n, mem))
    return out


def _detect(frame):
    """Detect cubes (any colour) and the button from the head camera depth: objects 1-5 cm above the table."""
    pw, valid = _world_points(frame, "head")
    rgb = np.asarray(frame.head_rgb).reshape(-1, 3).astype(np.int32)
    z = pw[:, 2]
    m = valid & (z > TABLE_Z + 0.008) & (z < TABLE_Z + 0.07)
    m &= (pw[:, 0] > -0.45) & (pw[:, 0] < 0.50) & (pw[:, 1] > -0.40) & (pw[:, 1] < 0.15)
    pts = pw[m]
    cols = rgb[m]
    cubes, buttons = [], []
    for c, n, mem in _clusters(pts):
        zmax = float(np.max(pts[mem, 2]))
        top = mem & (pts[:, 2] > zmax - 0.004)
        if top.sum() >= 10:
            tp = pts[top, :2]
            ctr = (tp.min(axis=0) + tp.max(axis=0)) / 2.0  # top-face extent midpoint (checked: matches cube extents)
        else:
            ctr = c
        col = np.median(cols[mem], axis=0)
        item = {"xy": [float(ctr[0]), float(ctr[1])], "n": n, "ztop": zmax,
                "rgb": [int(col[0]), int(col[1]), int(col[2])]}
        if ctr[0] < -0.10:
            buttons.append(item)
        else:
            cubes.append(item)
    cubes.sort(key=lambda it: -it["n"])
    return cubes, buttons


def _near(cubes, xy, tol=MATCH_TOL):
    best, bd = None, tol
    for i, c in enumerate(cubes):
        d = math.hypot(c["xy"][0] - xy[0], c["xy"][1] - xy[1])
        if d < bd:
            best, bd = i, d
    return best


def _perm_str(arr):
    return "".join(arr)


FIELDS = ["slots", "button", "arrangement", "temp_cube", "tried", "swap", "stage", "press_fail",
          "stall", "decisions", "note"]


def _default_memory():
    return {"slots": [], "button": [], "arrangement": [], "temp_cube": "", "tried": [],
            "swap": [], "stage": "init", "press_fail": 0, "stall": 0, "decisions": 0, "note": ""}


def _choose_swap(arr, tried):
    """Next transposition (slot indices) leading to an untried arrangement; expert cyclic order first."""
    k = max(len(tried) - 1, 0)
    for j in range(3):
        a, b = SWAP_CYCLE[(k + j) % 3]
        cand = list(arr)
        cand[a], cand[b] = cand[b], cand[a]
        if _perm_str(cand) not in tried:
            return [a, b]
    ids = sorted(arr)
    perms = []
    for p in ids:
        for q in ids:
            for r in ids:
                if len({p, q, r}) == 3:
                    perms.append(p + q + r)
    for p in perms:
        if p not in tried:
            for a in range(3):
                if arr[a] != p[a]:
                    return [a, arr.index(p[a])]
    return None


def perceive_decide(obs, goal, memory, last):
    """Perceive cubes/button from head RGB-D, update the trial memory, and pick the next action.

    Stages: init -> press -> pressed -> choose -> move1 (slot a -> temp) -> move2 (slot b -> slot a)
    -> move3 (temp -> slot b) -> press ... Each move is verified from the next head-camera view."""
    mem = _default_memory()
    for k in mem:
        if isinstance(memory, dict) and memory.get(k) is not None:
            mem[k] = memory[k]
    mem["decisions"] = int(mem["decisions"]) + 1
    cubes, buttons = _detect(obs.current)

    def press(note):
        mem["stage"] = "pressed"
        mem["note"] = note
        return {"memory": mem, "act": {"tool": "act_press_button",
                                       "args": {"bx": float(mem["button"][0]), "by": float(mem["button"][1])}}}

    if mem["stage"] == "init":
        sl = [list(p) for p in NOMINAL_SLOTS]
        if len(cubes) >= 3:
            cand = sorted([c["xy"] for c in cubes[:3]], key=lambda p: p[0])
            if all(math.hypot(cand[i][0] - NOMINAL_SLOTS[i][0], cand[i][1] - NOMINAL_SLOTS[i][1]) < 0.08
                   for i in range(3)):
                sl = cand
        mem["slots"] = [[float(p[0]), float(p[1])] for p in sl]
        b = list(NOMINAL_BUTTON)
        if buttons:
            bb = max(buttons, key=lambda it: it["n"])["xy"]
            if math.hypot(bb[0] - b[0], bb[1] - b[1]) < 0.04:
                b = bb
        mem["button"] = [float(b[0]), float(b[1])]
        mem["arrangement"] = ["A", "B", "C"]
        mem["temp_cube"] = ""
        mem["tried"] = []
        mem["swap"] = []
        return press("press initial arrangement ABC")

    arr = list(mem["arrangement"])
    if mem["stage"] == "pressed":
        reached = isinstance(last, str) and '"reached": true' in last
        if reached or int(mem["press_fail"]) >= MAX_PRESS_RETRY:
            if _perm_str(arr) not in mem["tried"]:
                mem["tried"] = list(mem["tried"]) + [_perm_str(arr)]
            mem["press_fail"] = 0
            mem["stage"] = "choose"
        else:
            mem["press_fail"] = int(mem["press_fail"]) + 1
            return press("press not confirmed; press again " + _perm_str(arr))

    if mem["stage"] == "choose":
        choice = _choose_swap(arr, list(mem["tried"]))
        if choice is None:
            mem["tried"] = []
            return press("all arrangements tried; press again")
        mem["swap"] = [str(choice[0]), str(choice[1])]
        mem["stage"] = "move1"

    slots = mem["slots"]
    a, b = int(mem["swap"][0]), int(mem["swap"][1])
    sa, sb = slots[a], slots[b]
    temp = [sa[0], sa[1] + TEMP_DY]

    # Verify the previous move from the current view and advance (no-op when the move failed).
    # Advancing needs all three cubes visible (guards against occlusion), unless we already stalled twice.
    stage_before = mem["stage"]
    enough = len(cubes) >= 3 or int(mem["stall"]) >= 2
    for _ in range(3 if enough else 0):
        st = mem["stage"]
        if st == "move1" and _near(cubes, sa) is None and _near(cubes, temp) is not None:
            mem["temp_cube"] = arr[a]
            arr[a] = "_"
            mem["stage"] = "move2"
        elif st == "move2" and _near(cubes, sb) is None and _near(cubes, sa) is not None:
            arr[a] = arr[b]
            arr[b] = "_"
            mem["stage"] = "move3"
        elif st == "move3" and _near(cubes, temp) is None and _near(cubes, sb) is not None:
            arr[b] = mem["temp_cube"]
            mem["temp_cube"] = ""
            mem["stage"] = "press"
    mem["arrangement"] = arr
    if mem["stage"] == stage_before and stage_before in ("move1", "move2", "move3") and last is not None \
            and isinstance(last, str) and last.startswith("act_move_cube"):
        mem["stall"] = int(mem["stall"]) + 1
    else:
        mem["stall"] = 0

    if mem["stage"] == "press":
        return press("press arrangement " + _perm_str(arr))

    st = mem["stage"]
    if st == "move1":
        src_nom, dst = sa, temp
    elif st == "move2":
        src_nom, dst = sb, sa
    else:
        src_nom, dst = temp, sb
    i = _near(cubes, src_nom, 0.06)
    if i is not None:
        src = cubes[i]["xy"]
    else:
        # the cube is not where expected (e.g. dropped): take a detection at no known spot, else nominal
        spots = [slots[0], slots[1], slots[2], temp]
        stray = [c for c in cubes
                 if all(math.hypot(c["xy"][0] - p[0], c["xy"][1] - p[1]) > MATCH_TOL for p in spots)
                 and -0.12 < c["xy"][0] < 0.42 and -0.30 < c["xy"][1] < 0.05]
        src = stray[0]["xy"] if stray else src_nom
    mem["note"] = "%s swap %d<->%d: (%.3f,%.3f)->(%.3f,%.3f) seen %d cubes" % (
        st, a, b, src[0], src[1], dst[0], dst[1], len(cubes))
    return {"memory": mem, "act": {"tool": "act_move_cube",
                                   "args": {"px": float(src[0]), "py": float(src[1]),
                                            "qx": float(dst[0]), "qy": float(dst[1])}}}


def _hold_left(robot):
    f = robot.now()
    return [float(v) for v in f.left_ee_pose] + [float(f.left_gripper)]


def _hold_right(robot):
    f = robot.now()
    return [float(v) for v in f.right_ee_pose] + [float(f.right_gripper)]


def _step(robot, left, right, log):
    try:
        robot.step(*(list(left) + list(right)))
        return True
    except Exception as e:  # MotionError: the arm stayed still; episode end is BaseException and propagates
        log.append(str(e)[:120])
        return False


def act_move_cube(robot, px, py, qx, qy):
    """Right arm: pick the cube centred at (px,py) on the table and place it at (qx,qy); then park at home.
    Returns whether every motion succeeded and the final gripper/ee state."""
    log = []
    L = _hold_left(robot)
    q = Q_RIGHT_DOWN
    px = px + GRASP_DX
    qx = qx + GRASP_DX
    seq = [
        [px, py, Z_ABOVE] + q + [1.0],
        [px, py, Z_GRASP] + q + [1.0],
        [px, py, Z_GRASP] + q + [0.0],
        [px, py, Z_ABOVE] + q + [0.0],
        [qx, qy, Z_ABOVE] + q + [0.0],
        [qx, qy, Z_RELEASE] + q + [0.0],
        [qx, qy, Z_RELEASE] + q + [1.0],
        [qx, qy, Z_ABOVE] + q + [1.0],
    ]
    ok = True
    for i, r in enumerate(seq):
        if not _step(robot, L, r, log):
            ok = False
            if i < 2:
                break  # could not reach the cube: skip this move; perception will request it again
    _step(robot, L, list(RIGHT_HOME) + [1.0], log)
    f = robot.now()
    return {"ok": ok, "errors": log, "right_ee": [round(float(v), 3) for v in f.right_ee_pose[:3]]}


def act_press_button(robot, bx, by):
    """Left arm: move above the button at (bx,by) with a closed gripper, press down and come back up.
    Returns reached=true when the measured end-effector got to the press depth."""
    log = []
    R = _hold_right(robot)
    q = Q_LEFT_DOWN
    f = robot.now()
    lp = f.left_ee_pose
    near = math.hypot(float(lp[0]) - bx, float(lp[1]) - by) < 0.01 and abs(float(lp[2]) - BTN_Z_ABOVE) < 0.02
    if not near:
        _step(robot, [bx, by, BTN_Z_ABOVE] + q + [1.0], R, log)
        _step(robot, [bx, by, BTN_Z_ABOVE] + q + [0.0], R, log)
    good = _step(robot, [bx, by, BTN_Z_PRESS] + q + [0.0], R, log)
    zmin = float(robot.now().left_ee_pose[2])
    _step(robot, [bx, by, BTN_Z_ABOVE] + q + [0.0], R, log)
    reached = bool(good and zmin < BTN_Z_PRESS + 0.01)
    return {"reached": reached, "z_press": round(zmin, 4), "errors": log}
