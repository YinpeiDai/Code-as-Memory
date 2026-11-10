"""swap_blocks policy: perceive two red blocks on three fixed trays (head RGB-D),
rotate them cyclically (M->L, R->M, L->R starting with the move whose destination is
empty), then press the red button with the closed left gripper."""

TRAY_X = [-0.138, 0.002, 0.144]       # left, middle, right tray centres (world x)
TRAY_Y = [-0.150, -0.148, -0.150]     # tray centres (world y), from expert place poses
TRAY_NAMES = ["L", "M", "R"]
BUTTON_XY = [-0.001, -0.279]          # expert press location (identical in all demos)
GRASP_Z = 0.778                       # grasp-centre height when closing on a block
PLACE_Z = 0.799                       # grasp-centre height when releasing a block
APPROACH_Z = 0.846                    # grasp-centre height above block/tray
CARRY_Z = 0.870                       # grasp-centre height while carrying
TCP = 0.12                            # grasp centre along local +x of ee frame
LEFT_HOME = [-0.298, -0.314, 0.942, 0.7, 0.0, 0.0, 0.714]
RIGHT_HOME = [0.306, -0.313, 0.941, 0.704, 0.0, 0.0, 0.711]
YAW_REF = {"left": math.radians(153.0), "right": math.radians(-147.0)}
RIGHT_FAR_TILT = -0.1                 # rad about local y when right arm works at left tray
# cyclic move order used by the expert: (src, dst, arm)
CYCLE = [(1, 0, "left"), (2, 1, "right"), (0, 2, "right")]


def _world_points(frame, cam):
    d = np.asarray(getattr(frame, cam + "_depth")).astype(float) / 1000.0
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    Pc = np.stack([x, y, d], -1).reshape(-1, 3)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    return Pw.reshape(H, W, 3), d > 0


def _red_points(frame):
    rgb = np.asarray(frame.head_rgb).astype(int)
    Pw, valid = _world_points(frame, "head")
    red = (rgb[..., 0] > 140) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90) & valid
    red = red & (rgb[..., 0] > rgb[..., 1] + 70)
    return Pw[red]


def _detect_blocks(frame):
    """Red clusters on the tray row (y > -0.23 m, above table). Returns list of dicts."""
    P = _red_points(frame)
    if len(P) == 0:
        return []
    P = P[(P[:, 1] > -0.23) & (P[:, 1] < -0.05) & (P[:, 2] > 0.745) & (P[:, 2] < 0.80)
          & (P[:, 0] > -0.30) & (P[:, 0] < 0.30)]
    if len(P) < 20:
        return []
    order = np.argsort(P[:, 0])
    P = P[order]
    gaps = np.flatnonzero(np.diff(P[:, 0]) > 0.02)
    starts = [0] + [int(g) + 1 for g in gaps]
    ends = [int(g) + 1 for g in gaps] + [len(P)]
    out = []
    for s, e in zip(starts, ends):
        Q = P[s:e]
        if len(Q) < 30:
            continue
        zmax = float(np.percentile(Q[:, 2], 98))
        top = Q[Q[:, 2] > zmax - 0.006]
        if len(top) < 10:
            top = Q
        mu = top[:, :2].mean(0)
        dx = top[:, :2] - mu
        ang = np.arctan2(dx[:, 1], dx[:, 0])
        r2 = dx[:, 0] ** 2 + dx[:, 1] ** 2
        c = np.sum(r2 * np.exp(4j * ang))
        # 4-fold moment points at the square's corners; edges are 45 deg away
        edge_yaw = float(np.angle(c) / 4.0 + math.pi / 4.0)
        tray = int(np.argmin([abs(mu[0] - tx) for tx in TRAY_X]))
        out.append({"x": float(mu[0]), "y": float(mu[1]), "top": zmax,
                    "yaw": edge_yaw, "n": int(len(Q)), "tray": tray,
                    "tray_err": float(abs(mu[0] - TRAY_X[tray]))})
    return out


def _detect_button(frame):
    P = _red_points(frame)
    if len(P) == 0:
        return None
    B = P[(P[:, 1] < -0.22) & (P[:, 1] > -0.40) & (np.abs(P[:, 0]) < 0.15) & (P[:, 2] > 0.745)]
    if len(B) < 20:
        return None
    top = B[B[:, 2] > B[:, 2].max() - 0.004]
    mid = (top[:, :2].min(0) + top[:, :2].max(0)) / 2.0
    return [float(mid[0]), float(mid[1]), float(B[:, 2].max())]


def _occupancy(blocks):
    occ = [None, None, None]
    for b in blocks:
        if b["tray_err"] < 0.05 and (occ[b["tray"]] is None or b["n"] > occ[b["tray"]]["n"]):
            occ[b["tray"]] = b
    return occ


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _choose_yaw(edge_yaw, arm):
    ref = YAW_REF[arm]
    best = None
    for k in range(4):
        cand = _wrap(edge_yaw + k * math.pi / 2)
        dist = abs(_wrap(cand - ref))
        if best is None or dist < best[0]:
            best = (dist, cand)
    return best[1]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _down_rot(yaw, tilt=0.0):
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.cross(x, y)
    R = np.stack([x, y, z], 1)
    if tilt:
        R = R @ _rot_y(tilt)
    return R


def _mat2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def _ee_pose(centre, R):
    p = np.asarray(centre, dtype=float) - TCP * R[:, 0]
    return [float(p[0]), float(p[1]), float(p[2])] + _mat2quat(R)


def _pose_of(frame, arm):
    return [float(v) for v in np.asarray(getattr(frame, arm + "_ee_pose"), dtype=float)]


def _send(robot, arm, pose, grip):
    """One native step: `arm` goes to `pose` with gripper `grip`; the other arm holds."""
    cur = robot.now()
    other = "right" if arm == "left" else "left"
    hold = _pose_of(cur, other)
    hold_g = float(np.asarray(getattr(cur, other + "_gripper")))
    if arm == "left":
        args = list(pose) + [grip] + hold + [hold_g]
    else:
        args = hold + [hold_g] + list(pose) + [grip]
    robot.step(*args)


def _orientation_candidates(arm, yaw, tilt):
    out = []
    for dk in (0, 1, -1, 2):
        for tl in ([tilt, 0.0, 2 * tilt] if tilt else [0.0, RIGHT_FAR_TILT if arm == "right" else 0.0]):
            key = (dk, round(tl, 3))
            if key in [(o[2], round(o[3], 3)) for o in out]:
                continue
            y = _wrap(yaw + dk * math.pi / 2)
            out.append((y, _down_rot(y, tl), dk, tl))
    return out


def act_move_block(robot, arm, src_x, src_y, yaw, tilt, dst_x, dst_y):
    """Pick the block whose top-face centre is (src_x, src_y) with a top-down grasp
    (gripper finger axis yaw `yaw` rad, extra tilt `tilt` rad about the gripper's local y),
    carry it and release it over (dst_x, dst_y); then return that arm home. The other arm
    holds. Returns which orientation was planned and any motion errors."""
    home = LEFT_HOME if arm == "left" else RIGHT_HOME
    errors = []
    used = None
    for (y, R, dk, tl) in _orientation_candidates(arm, float(yaw), float(tilt)):
        try:
            _send(robot, arm, _ee_pose([src_x, src_y, APPROACH_Z], R), 1.0)
            _send(robot, arm, _ee_pose([src_x, src_y, GRASP_Z], R), 1.0)
        except Exception as e:
            errors.append("pick dk=%d tilt=%.2f: %s" % (dk, tl, str(e)[:80]))
            continue
        used = (y, R, dk, tl)
        break
    if used is None:
        try:
            _send(robot, arm, home, 1.0)
        except Exception as e:
            errors.append("home: " + str(e)[:80])
        return {"picked": False, "errors": errors}
    y, R, dk, tl = used
    _send(robot, arm, _ee_pose([src_x, src_y, GRASP_Z], R), 0.0)
    lifted = True
    try:
        _send(robot, arm, _ee_pose([src_x, src_y, CARRY_Z], R), 0.0)
    except Exception as e:
        errors.append("lift: " + str(e)[:80])
        lifted = False
    placed_pose = None
    for (z_above, z_place) in [(CARRY_Z, PLACE_Z), (APPROACH_Z, PLACE_Z + 0.01)]:
        try:
            _send(robot, arm, _ee_pose([dst_x, dst_y, z_above], R), 0.0)
            _send(robot, arm, _ee_pose([dst_x, dst_y, z_place], R), 0.0)
            placed_pose = z_place
            break
        except Exception as e:
            errors.append("place: " + str(e)[:80])
    _send(robot, arm, _pose_of(robot.now(), arm), 1.0)  # open gripper in place
    try:
        _send(robot, arm, _ee_pose([dst_x, dst_y, APPROACH_Z], R) if placed_pose else _pose_of(robot.now(), arm), 1.0)
    except Exception as e:
        errors.append("retreat: " + str(e)[:80])
    try:
        _send(robot, arm, home, 1.0)
    except Exception as e:
        errors.append("home: " + str(e)[:80])
    return {"picked": True, "lifted": lifted, "placed_at_z": placed_pose, "yaw_deg": round(math.degrees(y), 1),
            "dk": dk, "tilt": tl, "errors": errors,
            "ee_end": [round(v, 3) for v in _pose_of(robot.now(), arm)[:3]]}


def act_press_button(robot, x, y, press_z):
    """Close the left gripper above the button at (x, y), push straight down until the grasp
    centre reaches `press_z` (world z, m), hold, then lift and return home. Success ends the
    episode during the push; a normal return means the episode did not end."""
    R = _down_rot(math.pi / 2)
    errors = []
    _send(robot, "left", _ee_pose([x, y, 0.844], R), 1.0)
    _send(robot, "left", _ee_pose([x, y, 0.844], R), 0.0)
    for z in [0.825, press_z]:
        try:
            _send(robot, "left", _ee_pose([x, y, z], R), 0.0)
        except Exception as e:
            errors.append(str(e)[:80])
    _send(robot, "left", _ee_pose([x, y, press_z], R), 0.0)
    _send(robot, "left", _ee_pose([x, y, 0.860], R), 0.0)
    try:
        _send(robot, "left", LEFT_HOME, 1.0)
    except Exception as e:
        errors.append("home: " + str(e)[:80])
    return {"pressed_to": press_z, "errors": errors}


def act_home(robot):
    """Return both arms to their start poses with open grippers (clears the head-camera view)."""
    robot.step(*(LEFT_HOME + [1.0] + RIGHT_HOME + [1.0]))
    return {"home": True}


def _plan_from_empty(empty):
    j = [c[1] for c in CYCLE].index(empty)
    return [CYCLE[(j + k) % 3] for k in range(3)]


def _move_label(m):
    return TRAY_NAMES[m[0]] + ">" + TRAY_NAMES[m[1]]


def _parse_move(label):
    s, d = label.split(">")
    return TRAY_NAMES.index(s), TRAY_NAMES.index(d)


def perceive_decide(obs, goal, memory, last):
    """Swap the two blocks via the empty tray (3 cyclic moves M>L, R>M, L>R starting with the
    move into the empty tray), verifying each move with the head RGB-D camera, then press
    the button with the closed left gripper."""
    mem = {
        "phase": memory.get("phase") or "init",
        "plan": list(memory.get("plan") or []),
        "move_idx": int(memory.get("move_idx") or 0),
        "retries": int(memory.get("retries") or 0),
        "press_attempts": int(memory.get("press_attempts") or 0),
        "initial_occupancy": list(memory.get("initial_occupancy") or []),
        "occupancy": [],
        "block_positions": [],
        "button": list(memory.get("button") or BUTTON_XY),
        "note": "",
    }
    frame = obs.current
    blocks = _detect_blocks(frame)
    occ = _occupancy(blocks)
    mem["occupancy"] = [TRAY_NAMES[i] for i in range(3) if occ[i] is not None]
    mem["block_positions"] = [[round(b["x"], 4), round(b["y"], 4), round(b["top"], 4)] for b in blocks]
    notes = []

    if mem["phase"] == "init":
        if len(mem["occupancy"]) != 2:
            mem["retries"] += 1
            mem["note"] = "expected 2 occupied trays, saw %s" % mem["occupancy"]
            return {"memory": mem, "act": {"tool": "act_home", "args": {}}}
        empty = [i for i in range(3) if occ[i] is None][0]
        mem["plan"] = [_move_label(m) for m in _plan_from_empty(empty)]
        mem["initial_occupancy"] = list(mem["occupancy"])
        btn = _detect_button(frame)
        if btn is not None and math.hypot(btn[0] - BUTTON_XY[0], btn[1] - BUTTON_XY[1]) > 0.025:
            mem["button"] = [btn[0], btn[1]]
        mem["phase"] = "moving"
        mem["move_idx"] = 0
        mem["retries"] = 0
        notes.append("plan %s" % mem["plan"])
    elif mem["phase"] == "moving":
        src, dst = _parse_move(mem["plan"][mem["move_idx"]])
        if occ[dst] is not None and occ[src] is None:
            notes.append("move %s verified" % mem["plan"][mem["move_idx"]])
            mem["move_idx"] += 1
            mem["retries"] = 0
        else:
            mem["retries"] += 1
            notes.append("move %s not verified (occupancy %s)" % (mem["plan"][mem["move_idx"]], mem["occupancy"]))
        if mem["move_idx"] >= 3:
            mem["phase"] = "press"

    if mem["phase"] == "moving":
        src, dst = _parse_move(mem["plan"][mem["move_idx"]])
        b = occ[src]
        if b is None:
            # source tray empty but destination not filled: look for a stray red block
            used = [o for o in occ if o is not None]
            strays = [c for c in blocks if not any(o is c for o in used)]
            if strays:
                b = min(strays, key=lambda c: abs(c["x"] - TRAY_X[src]))
                notes.append("stray block at %.3f,%.3f" % (b["x"], b["y"]))
            else:
                notes.append("no block found for move")
                mem["note"] = "; ".join(notes)
                return {"memory": mem, "act": {"tool": "act_home", "args": {}}}
        arm = "left" if dst == 0 else "right"
        tilt = RIGHT_FAR_TILT if (arm == "right" and b["x"] < -0.07) else 0.0
        yaw = _choose_yaw(b["yaw"], arm)
        notes.append("%s by %s arm" % (mem["plan"][mem["move_idx"]], arm))
        mem["note"] = "; ".join(notes)
        return {"memory": mem, "act": {"tool": "act_move_block", "args": {
            "arm": arm, "src_x": b["x"], "src_y": b["y"], "yaw": yaw, "tilt": tilt,
            "dst_x": TRAY_X[dst], "dst_y": TRAY_Y[dst]}}}

    # press phase (reached only after all three moves were verified)
    k = mem["press_attempts"]
    mem["press_attempts"] = k + 1
    press_z = [0.806, 0.796, 0.788, 0.806, 0.796][k % 5]
    bx, by = mem["button"][0], mem["button"][1]
    if k % 5 >= 3:
        by = by - 0.008 if k % 5 == 3 else by + 0.008
    notes.append("press attempt %d to z=%.3f" % (k + 1, press_z))
    mem["note"] = "; ".join(notes)
    return {"memory": mem, "act": {"tool": "act_press_button", "args": {"x": bx, "y": by, "press_z": press_z}}}
