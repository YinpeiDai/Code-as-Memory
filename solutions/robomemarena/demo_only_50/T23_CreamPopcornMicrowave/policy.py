"""T23 CreamPopcornMicrowave: open the microwave door, put the cream inside, then put the
popcorn where the cream was placed (inside the microwave).

Perception (first decision): microwave handle H from agentview RGB-D (dark microwave body),
cream / popcorn grasp points from wrist RGB-D (blobs above the table, classified by colour).
Control: closed-loop waypoint servo on measured TCP, waypoints anchored to perceived objects.
All numeric offsets were measured on the 50 assigned demonstrations (see task_analysis.md).
"""

# ---------------------------------------------------------------- constants (from demos)
Q_DOWN = [3.14159, 0.0, 0.0]            # gripper pointing down (grasp from top)
Q_SIDE = [2.415, -2.415, -2.42]          # gripper horizontal, pointing +y (into microwave)
Q_HANDLE = [2.404, -2.40, -2.434]        # mean orientation at handle grasp
H_Z = 0.583                              # handle grasp height (constant in all demos)
GRASP_Z = 0.438                          # TCP height for grasping the flat boxes
LIFT_Z = 0.557                           # carry height after lifting an object
ABOVE_Z = 0.68                           # hover height above an object before descending
# microwave handle from agentview microwave cloud: H_x = q98(x) + MW_DX, H_y = mid(y) + MW_DY
MW_DX = -0.0748
MW_DY = -0.1212
MW_DY2 = -0.0006                          # H_y = q2(y) + MW_DY2
# grasp point = wrist blob centroid + offset
CREAM_OFF = [-0.0047, 0.0013]
POP_OFF = [-0.0028, 0.0031]
CREAM_OFF_AV = [-0.0074, 0.0015]
POP_OFF_AV = [-0.0049, 0.0035]
# demo means (fallback only)
H_MEAN = [0.096, 0.411, 0.583]
C_MEAN = [0.165, -0.093, 0.438]
P_MEAN = [0.164, -0.191, 0.438]
# handle-relative approach and door-pull path (mean measured TCP path of demos)
H_PRE = [[-0.059, -0.162, 0.011], [-0.055, -0.062, 0.0], [-0.030, -0.028, 0.0]]
PULL = [[-0.023, -0.007], [-0.054, -0.057], [-0.091, -0.106], [-0.128, -0.142],
        [-0.165, -0.168], [-0.200, -0.188], [-0.237, -0.202], [-0.278, -0.209],
        [-0.322, -0.205], [-0.359, -0.199], [-0.381, -0.195]]
# cream release relative to handle; carry waypoints relative to handle
CREAM_PRE_PATH = [[-0.094, -0.157, 0.017], [-0.100, -0.040, 0.020], [-0.098, 0.035, 0.018]]
CREAM_PLACE = [-0.078, 0.059, -0.004]
CREAM_RETREAT = [[-0.051, -0.089, -0.010], [-0.049, -0.100, 0.127]]
# popcorn: rotation spot relative to its grasp point; entry relative to the cream release spot
POP_ROT_SPOT = [-0.194, 0.188, 0.117]
POP_ENTRY = [[-0.030, -0.220, 0.015], [-0.030, -0.120, 0.021]]   # rel. cream release
POP_INSIDE = [[-0.030, -0.100, 0.021], [-0.028, -0.055, 0.021]]  # rel. cream release
KP = 20.0      # position gain: action = KP*err (KP=20 puts the OSC goal on the target)
KR = 2.0       # rotation gain: action = KR*rotvec_err (0.5 rad per unit)


# ---------------------------------------------------------------- geometry helpers
def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        S = (R + R.T) / 2.0
        w, V = np.linalg.eigh(S)
        k = V[:, int(np.argmax(w))]
        v2 = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(np.dot(k, v2)) < 0:
            k = -k
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return v * th


def _rot_err(q_target, q_now):
    """World-frame rotation vector taking the current orientation to the target."""
    return _R2aa(_aa2R(q_target) @ _aa2R(q_now).T)


def _cloud(depth, K, E):
    z = np.asarray(depth, dtype=float) / 1000.0
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0][2]) * z / K[0][0]
    y = (v - K[1][2]) * z / K[1][1]
    pc = np.stack([x, y, z], -1)
    E = np.asarray(E, dtype=float)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, z > 0


def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


# ---------------------------------------------------------------- perception
def _perceive_microwave(frame):
    """Handle grasp point from the agentview microwave cloud, or None."""
    pw, ok = _cloud(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = (rgb.max(-1) < 90) & ((rgb.max(-1) - rgb.min(-1)) < 20)
    m = ok & dark & (pw[..., 2] > 0.46) & (pw[..., 2] < 0.8) & (pw[..., 1] > 0.15) \
        & (pw[..., 1] < 0.9) & (pw[..., 0] > -0.45) & (pw[..., 0] < 0.45)
    n = int(m.sum())
    if n < 500:
        return None, n
    P = pw[m]
    q98x = float(np.percentile(P[:, 0], 98))
    q2y = float(np.percentile(P[:, 1], 2))
    q98y = float(np.percentile(P[:, 1], 98))
    hx = q98x + MW_DX
    hy = 0.5 * ((q2y + q98y) / 2.0 + MW_DY + q2y + MW_DY2)
    return [hx, hy, H_Z], n


def _door_open_count(frame):
    """Dark points where the opened door panel lies (0 when closed, >150 when open)."""
    pw, ok = _cloud(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = (rgb.max(-1) < 90) & ((rgb.max(-1) - rgb.min(-1)) < 20)
    m = ok & dark & (pw[..., 0] > -0.28) & (pw[..., 0] < -0.12) & (pw[..., 1] > 0.12) \
        & (pw[..., 1] < 0.38) & (pw[..., 2] > 0.47) & (pw[..., 2] < 0.65)
    return int(m.sum())


def _two_blobs(P, cols):
    """Split table-top points into two clusters (2-means on xy, seeded at extreme y)."""
    xy = P[:, :2]
    c = np.array([xy[int(np.argmax(xy[:, 1]))], xy[int(np.argmin(xy[:, 1]))]])
    lab = np.zeros(len(xy), dtype=int)
    for _ in range(15):
        d0 = ((xy - c[0]) ** 2).sum(1)
        d1 = ((xy - c[1]) ** 2).sum(1)
        lab = (d1 < d0).astype(int)
        if lab.min() == lab.max():
            break
        c = np.array([xy[lab == 0].mean(0), xy[lab == 1].mean(0)])
    out = []
    for k in (0, 1):
        sel = lab == k
        if sel.sum() < 15:
            out.append(None)
            continue
        cc = cols[sel]
        out.append({"xy": xy[sel].mean(0), "n": int(sel.sum()),
                    "rb": float((cc[:, 0] - cc[:, 2]).mean()), "sel": sel})
    return out


def _perceive_objects_cam(frame, cam):
    pw, ok = _cloud(getattr(frame, cam + "_depth"), getattr(frame, cam + "_K"),
                    getattr(frame, cam + "_extrinsic"))
    rgb = np.asarray(getattr(frame, cam + "_rgb")).astype(int)
    m = ok & (pw[..., 2] > 0.437) & (pw[..., 2] < 0.5) & (pw[..., 0] > -0.15) \
        & (pw[..., 0] < 0.40) & (pw[..., 1] > -0.40) & (pw[..., 1] < 0.10)
    if int(m.sum()) < 30:
        return None
    rows, colsix = np.nonzero(m)
    blobs = _two_blobs(pw[m], rgb[m])
    if blobs[0] is None or blobs[1] is None:
        return None
    res = []
    for b in blobs:
        r = rows[b["sel"]]
        cc = colsix[b["sel"]]
        edge = bool(r.min() <= 1 or r.max() >= 254 or cc.min() <= 1 or cc.max() >= 254)
        res.append({"xy": [float(b["xy"][0]), float(b["xy"][1])], "n": b["n"], "rb": b["rb"], "edge": edge})
    # cream is blue/white (R-B strongly negative), popcorn is warm; tie-break: cream at larger y
    if abs(res[0]["rb"] - res[1]["rb"]) > 10:
        cream, pop = (res[0], res[1]) if res[0]["rb"] < res[1]["rb"] else (res[1], res[0])
    else:
        cream, pop = (res[0], res[1]) if res[0]["xy"][1] > res[1]["xy"][1] else (res[1], res[0])
    return {"cream": cream, "pop": pop}


def _perceive_objects(frame):
    w = _perceive_objects_cam(frame, "wrist")
    a = _perceive_objects_cam(frame, "agentview")
    out = {}
    for name, off, offa, mean in (("cream", CREAM_OFF, CREAM_OFF_AV, C_MEAN),
                                  ("pop", POP_OFF, POP_OFF_AV, P_MEAN)):
        est = None
        src = "mean"
        if w is not None and not w[name]["edge"]:
            est = [w[name]["xy"][0] + off[0], w[name]["xy"][1] + off[1]]
            src = "wrist"
        if a is not None:
            ea = [a[name]["xy"][0] + offa[0], a[name]["xy"][1] + offa[1]]
            if est is None:
                est, src = ea, "agentview"
            elif math.hypot(est[0] - ea[0], est[1] - ea[1]) > 0.03:
                src = "wrist(agentview disagrees)"
        if est is None or math.hypot(est[0] - mean[0], est[1] - mean[1]) > 0.15:
            est, src = [mean[0], mean[1]], "mean"
        out[name] = [float(est[0]), float(est[1]), GRASP_Z]
        out[name + "_src"] = src
    return out


def _perceive_single(frame, guess, other=None):
    """Re-locate one box near `guess` from the wrist camera (for grasp retries). Points closer
    to `other` (the other box, if still on the table) than to `guess` are ignored."""
    pw, ok = _cloud(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    m = ok & (pw[..., 2] > 0.437) & (pw[..., 2] < 0.5) \
        & (np.abs(pw[..., 0] - guess[0]) < 0.07) & (np.abs(pw[..., 1] - guess[1]) < 0.07)
    if other is not None:
        dg = (pw[..., 0] - guess[0]) ** 2 + (pw[..., 1] - guess[1]) ** 2
        do = (pw[..., 0] - other[0]) ** 2 + (pw[..., 1] - other[1]) ** 2
        m = m & (dg < do)
    if int(m.sum()) < 30:
        return None
    P = pw[m]
    return [float(P[:, 0].mean()), float(P[:, 1].mean()), GRASP_Z]


# ---------------------------------------------------------------- decision
def _rel(base, off):
    return [base[0] + off[0], base[1] + off[1], base[2] + off[2]]


def _mv(points, ori, grip, tol=0.004, vmax=1.0, max_steps=400, pass_tol=0.015):
    return {"tool": "act_move", "args": {"waypoints": points, "ori": ori, "gripper": grip,
                                         "tol": tol, "vmax": vmax, "max_steps": max_steps,
                                         "pass_tol": pass_tol}}


def _grip(close, steps):
    return {"tool": "act_gripper", "args": {"close": close, "steps": steps}}


def perceive_decide(obs, goal, memory, last):
    """Phase machine for: open microwave, cream into microwave, popcorn into the cream's spot.

    Memory: phase, handle/cream/popcorn grasp points, cream release spot (the location the
    popcorn must go to), retry counters and a short note."""
    mem = {k: memory.get(k) for k in ("phase", "handle", "cream", "popcorn", "cream_spot",
                                      "retries", "door_open", "note")}
    cur = obs.current
    phase = mem["phase"] or "start"
    if mem["retries"] is None:
        mem["retries"] = 0
    tcp = [float(v) for v in np.asarray(cur.tcp)[:3]]
    width = _width(cur)
    act = None

    if phase == "start":
        h, n = _perceive_microwave(cur)
        if h is None or math.hypot(h[0] - H_MEAN[0], h[1] - H_MEAN[1]) > 0.12:
            h = list(H_MEAN)
        objs = _perceive_objects(cur)
        mem["handle"] = h
        mem["cream"] = objs["cream"]
        mem["popcorn"] = objs["pop"]
        mem["note"] = "mw pts %d; cream %s; pop %s" % (n, objs["cream_src"], objs["pop_src"])
        mem["door_open"] = False
        phase = "approach_handle"

    H = mem["handle"]
    C = mem["cream"]
    P = mem["popcorn"]

    if phase == "approach_handle":
        act = _mv([_rel(H, o) for o in H_PRE] + [list(H)], Q_HANDLE, -1, tol=0.003, vmax=1.0)
        nxt = "push_handle"
    elif phase == "push_handle":
        act = {"tool": "act_push", "args": {"target": list(H), "direction": [0.95, 0.30, 0.0],
                                            "ori": Q_HANDLE, "max_steps": 15, "limit": 0.01}}
        nxt = "close_handle"
    elif phase == "close_handle":
        act = _grip(True, 20)
        nxt = "pull_door"
    elif phase == "pull_door":
        if width < 0.004 and mem["retries"] < 1:
            # fingers closed on nothing: release and re-approach once
            mem["retries"] += 1
            mem["note"] = "handle grasp missed; retry"
            act = _grip(False, 12)
            nxt = "approach_handle"
        else:
            pts = [[H[0] + p[0], H[1] + p[1], H[2] + 0.0005] for p in PULL]
            act = _mv(pts, Q_HANDLE, 1, tol=0.006, vmax=1.0, max_steps=200, pass_tol=0.03)
            nxt = "release_handle"
    elif phase == "release_handle":
        act = _grip(False, 12)
        nxt = "check_door"
    elif phase == "check_door":
        cnt = _door_open_count(cur)
        mem["door_open"] = bool(cnt > 80)
        mem["note"] = "door count %d" % cnt
        # back straight out of the handle (gripper points +y) before turning
        act = _mv([[tcp[0], tcp[1] - 0.05, tcp[2] + 0.01]], Q_HANDLE, -1, tol=0.008, max_steps=40)
        nxt = "above_cream"
    elif phase == "above_cream":
        act = _mv([[C[0], C[1], ABOVE_Z]], Q_DOWN, -1, tol=0.006, max_steps=300)
        nxt = "descend_cream"
    elif phase == "descend_cream":
        act = _mv([[C[0], C[1], GRASP_Z + 0.03], [C[0], C[1], GRASP_Z]], Q_DOWN, -1,
                  tol=0.003, vmax=0.6, max_steps=200, pass_tol=0.006)
        nxt = "close_cream"
    elif phase == "close_cream":
        act = _grip(True, 20)
        nxt = "lift_cream"
    elif phase == "lift_cream":
        act = _mv([[C[0], C[1], LIFT_Z]], Q_DOWN, 1, tol=0.006, max_steps=150)
        nxt = "check_cream"
    elif phase == "check_cream":
        if width < 0.01 and mem["retries"] < 3:
            mem["retries"] += 1
            mem["note"] = "cream grasp missed (w=%.3f); retry" % width
            act = _grip(False, 12)
            nxt = "relocate_cream"
        else:
            act = _mv([[C[0], C[1], LIFT_Z - 0.009]], Q_SIDE, 1, tol=0.006, max_steps=150)
            nxt = "carry_cream"
    elif phase == "relocate_cream":
        act = _mv([[C[0], C[1], ABOVE_Z]], Q_DOWN, -1, tol=0.006, max_steps=150)
        nxt = "relocate_cream2"
    elif phase == "relocate_cream2":
        p = _perceive_single(cur, [C[0] - CREAM_OFF[0], C[1] - CREAM_OFF[1]], P)
        if p is not None:
            mem["cream"] = [p[0] + CREAM_OFF[0], p[1] + CREAM_OFF[1], GRASP_Z]
            C = mem["cream"]
        act = _mv([[C[0], C[1], ABOVE_Z]], Q_DOWN, -1, tol=0.004, max_steps=100)
        nxt = "descend_cream"
    elif phase == "carry_cream":
        act = _mv([_rel(H, o) for o in CREAM_PRE_PATH], Q_SIDE, 1, tol=0.005, max_steps=300,
                  pass_tol=0.02)
        nxt = "place_cream"
    elif phase == "place_cream":
        act = _mv([_rel(H, CREAM_PLACE)], Q_SIDE, 1, tol=0.004, vmax=0.5, max_steps=120)
        nxt = "release_cream"
    elif phase == "release_cream":
        mem["cream_spot"] = tcp
        act = _grip(False, 20)
        nxt = "retreat_cream"
    elif phase == "retreat_cream":
        act = _mv([_rel(H, o) for o in CREAM_RETREAT], Q_SIDE, -1, tol=0.008, max_steps=200,
                  pass_tol=0.015)
        nxt = "above_pop"
    elif phase == "above_pop":
        act = _mv([[P[0], P[1], ABOVE_Z]], Q_DOWN, -1, tol=0.006, max_steps=300)
        nxt = "descend_pop"
    elif phase == "descend_pop":
        act = _mv([[P[0], P[1], GRASP_Z + 0.03], [P[0], P[1], GRASP_Z]], Q_DOWN, -1,
                  tol=0.003, vmax=0.6, max_steps=200, pass_tol=0.006)
        nxt = "close_pop"
    elif phase == "close_pop":
        act = _grip(True, 20)
        nxt = "lift_pop"
    elif phase == "lift_pop":
        act = _mv([[P[0], P[1], LIFT_Z]], Q_DOWN, 1, tol=0.006, max_steps=150)
        nxt = "check_pop"
    elif phase == "check_pop":
        if width < 0.01 and mem["retries"] < 4:
            mem["retries"] += 1
            mem["note"] = "popcorn grasp missed (w=%.3f); retry" % width
            act = _grip(False, 12)
            nxt = "relocate_pop"
        else:
            act = _mv([_rel(P, POP_ROT_SPOT)], Q_DOWN, 1, tol=0.006, max_steps=200)
            nxt = "rotate_pop"
    elif phase == "relocate_pop":
        act = _mv([[P[0], P[1], ABOVE_Z]], Q_DOWN, -1, tol=0.006, max_steps=150)
        nxt = "relocate_pop2"
    elif phase == "relocate_pop2":
        p = _perceive_single(cur, [P[0] - POP_OFF[0], P[1] - POP_OFF[1]])
        if p is not None:
            mem["popcorn"] = [p[0] + POP_OFF[0], p[1] + POP_OFF[1], GRASP_Z]
            P = mem["popcorn"]
        act = _mv([[P[0], P[1], ABOVE_Z]], Q_DOWN, -1, tol=0.004, max_steps=100)
        nxt = "descend_pop"
    elif phase == "rotate_pop":
        act = _mv([_rel(P, POP_ROT_SPOT)], Q_SIDE, 1, tol=0.008, max_steps=150)
        nxt = "carry_pop"
    elif phase == "carry_pop":
        spot = mem["cream_spot"] or _rel(H, CREAM_PLACE)
        act = _mv([_rel(spot, o) for o in POP_ENTRY + POP_INSIDE[:1]], Q_SIDE, 1, tol=0.005,
                  max_steps=300, pass_tol=0.02)
        nxt = "insert_pop"
    elif phase == "insert_pop":
        spot = mem["cream_spot"] or _rel(H, CREAM_PLACE)
        act = _mv([_rel(spot, POP_INSIDE[1])], Q_SIDE, 1, tol=0.004, vmax=0.4, max_steps=60)
        nxt = "settle_pop"
    elif phase == "settle_pop":
        act = _grip(True, 30)
        nxt = "release_pop"
    elif phase == "release_pop":
        act = _grip(False, 25)
        nxt = "back_off"
    elif phase == "back_off":
        act = _mv([[tcp[0], tcp[1] - 0.10, tcp[2] + 0.02]], Q_SIDE, -1, tol=0.008, max_steps=120)
        nxt = "wait"
    else:
        act = _grip(False, 50)
        nxt = "wait"

    mem["phase"] = nxt
    return {"memory": mem, "act": act}


# ---------------------------------------------------------------- actions
def _servo_action(frame, target, ori, grip, vmax):
    tcp = np.asarray(frame.tcp, dtype=float)
    e = np.asarray(target, dtype=float) - tcp[:3]
    a = np.clip(KP * e, -vmax, vmax)
    r = np.clip(KR * _rot_err(ori, tcp[3:6]), -0.5, 0.5)
    return [float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip)], e


def act_move(robot, waypoints, ori, gripper=-1, tol=0.004, vmax=1.0, max_steps=400, pass_tol=0.015):
    """Servo the TCP through `waypoints` ([[x,y,z],...], world m) holding orientation `ori`
    (axis-angle) and gripper sign (+1 close, -1 open). Intermediate points are passed within
    `pass_tol`, the last within `tol` (and rotation within ~0.05 rad) or after `max_steps`.
    Returns the final TCP, error and steps used."""
    steps = 0
    idx = 0
    n = len(waypoints)
    frame = robot.now()
    err = 1.0
    rerr = 1.0
    best = [9.0, 9.0]
    since = 0
    stalled = False
    while steps < max_steps:
        frame = robot.now()
        tgt = waypoints[idx]
        cmd, e = _servo_action(frame, tgt, ori, gripper, vmax)
        err = float(np.linalg.norm(e))
        rerr = float(np.linalg.norm(_rot_err(ori, np.asarray(frame.tcp, dtype=float)[3:6])))
        if idx < n - 1:
            if err < pass_tol and rerr < 0.35:
                idx += 1
                best = [9.0, 9.0]
                since = 0
                continue
        elif err < tol and rerr < 0.06:
            break
        # stall guard: no progress on the current waypoint for 25 steps (contact / limits)
        if err < best[0] - 0.0005 or rerr < best[1] - 0.005:
            best = [min(best[0], err), min(best[1], rerr)]
            since = 0
        else:
            since += 1
            if since >= 25 and (idx == n - 1 or err < 3.0 * pass_tol):
                if idx < n - 1:
                    idx += 1
                    best = [9.0, 9.0]
                    since = 0
                    continue
                stalled = True
                break
        robot.step(*cmd)
        steps += 1
    frame = robot.now()
    tcp = np.asarray(frame.tcp, dtype=float)
    return {"reached": bool(idx == n - 1 and err < max(tol, 0.006)), "waypoint": idx,
            "stalled": stalled,
            "steps": steps, "pos_err": round(err, 4), "rot_err": round(rerr, 3),
            "tcp": [round(float(v), 4) for v in tcp[:3]], "width": round(_width(frame), 4)}


def act_push(robot, target, direction, ori, max_steps=15, limit=0.01):
    """Press gently past `target` along `direction` (like the expert's handle contact) until
    the TCP stalls, moves `limit` m beyond the target, or `max_steps`. Gripper stays open."""
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    tgt = np.asarray(target, dtype=float)
    goal = tgt + 0.05 * d
    steps = 0
    still = 0
    prev = np.asarray(robot.now().tcp, dtype=float)[:3]
    while steps < max_steps:
        frame = robot.now()
        p = np.asarray(frame.tcp, dtype=float)[:3]
        if float(np.dot(p - tgt, d)) > limit:
            break
        e = goal - p
        a = np.clip(6.3 * e, -0.35, 0.35)
        r = np.clip(KR * _rot_err(ori, np.asarray(frame.tcp, dtype=float)[3:6]), -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), -1.0)
        steps += 1
        p2 = np.asarray(robot.now().tcp, dtype=float)[:3]
        still = still + 1 if float(np.linalg.norm(p2 - prev)) < 0.0004 else 0
        prev = p2
        if still >= 4:
            break
    p = np.asarray(robot.now().tcp, dtype=float)[:3]
    return {"steps": steps, "stalled": bool(still >= 4),
            "beyond_target": round(float(np.dot(p - tgt, d)), 4),
            "tcp": [round(float(v), 4) for v in p]}


def act_gripper(robot, close, steps=20):
    """Hold the arm (zero motion command) and close (True) or open (False) the gripper for
    `steps` native steps. Returns the final finger width."""
    g = 1.0 if close else -1.0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
    frame = robot.now()
    return {"width": round(_width(frame), 4),
            "tcp": [round(float(v), 4) for v in np.asarray(frame.tcp, dtype=float)[:3]]}
