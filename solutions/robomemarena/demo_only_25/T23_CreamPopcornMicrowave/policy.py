"""T23 CreamPopcornMicrowave: open microwave, put cream inside, remember where the
cream was placed, then put the popcorn at that remembered location.

Geometry constants below were measured from the 25 assigned expert recordings
(see task_analysis.md); all targets are expressed relative to objects that are
perceived at run time (microwave handle from agentview depth, cream/popcorn from
wrist depth), never per-seed lookups."""

# ---------------- constants learned from demonstrations ----------------
R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]   # gripper pointing down
R_DOOR = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]   # gripper pointing +y (horizontal)
TABLE_Z_DEFAULT = 0.431
MW_TOP_DEFAULT = 0.645
HANDLE_X_FROM_TOPXMAX = -0.074      # handle_x = top_xmax + this (LOO rms 0.8 mm)
HANDLE_Y_A = 0.8423                  # handle_y = A * top_ymin + B (fit in scratch/fit_constants.py, LOO rms 0.4 mm)
HANDLE_Y_B = 0.0424
HANDLE_Z_FROM_TOP = -0.062          # grasp height relative to microwave top
HINGE_REL = [-0.304, 0.080]         # door hinge relative to handle (xy)
DOOR_RADIUS = 0.287
DOOR_END_DEG = -107.0
OBJ_GRASP_OFFSET = [-0.0025, 0.0015]  # expert grasp minus visible-point centroid
GRASP_ABOVE_TABLE = 0.008
HOVER_Z = 0.66
CARRY_Z = 0.59                      # carry/staging height before insertion
CREAM_INSERT_A = [-0.098, 0.038, 0.585]  # relative to handle xy, absolute z (interior ~0.475..0.62)
CREAM_RELEASE = [-0.078, 0.059, 0.578]
STAGE_Y_REL = -0.31
POP_REL_CREAM = [-0.055, -0.020]    # popcorn release rel. to remembered cream placement: beside it, toward interior centre
POP_Z = 0.585                       # insertion height (box top must clear the ~0.62 ceiling)
POP_RELEASE_Z = 0.578               # release height (cream was released at 0.579)


# ---------------- small math helpers ----------------
def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    s = math.sin(th)
    if s > 1e-3:
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
        return w * th
    # near pi
    d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
    i = int(np.argmax(d))
    ax = np.zeros(3)
    ax[i] = d[i]
    for j in range(3):
        if j != i:
            ax[j] = (R[i, j] + R[j, i]) / (4.0 * d[i])
    ax = ax / max(np.linalg.norm(ax), 1e-9)
    return ax * th


def _rot_named(name):
    if name == "door":
        return np.array(R_DOOR, dtype=float)
    return np.array(R_DOWN, dtype=float)


def _backproject(depth, K, E):
    z = np.asarray(depth, dtype=float) / 1000.0
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    Pc = np.stack([x, y, z], -1).reshape(-1, 3)
    R = E[:, :3]
    t = E[:, 3]
    Pw = (Pc - t) @ R
    return Pw, z.reshape(-1) > 0


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _tcp(frame):
    return [float(v) for v in np.asarray(frame.tcp, dtype=float)[:3]]


# ---------------- perception ----------------
def _detect_microwave(frame):
    """Handle grasp point from agentview depth: top surface of the tallest box at y>0.2."""
    P, valid = _backproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    m = valid & (P[:, 2] > 0.46) & (P[:, 2] < 0.72) & (P[:, 1] > 0.2) & (P[:, 1] < 0.9) \
        & (P[:, 0] > -0.35) & (P[:, 0] < 0.5)
    Q = P[m]
    if len(Q) < 50:
        return None
    zt = float(np.percentile(Q[:, 2], 99.5))
    top = Q[Q[:, 2] > zt - 0.01]
    if len(top) < 20:
        return None
    xmax = float(top[:, 0].max())
    ymin = float(top[:, 1].min())
    hx = xmax + HANDLE_X_FROM_TOPXMAX
    hy = HANDLE_Y_A * ymin + HANDLE_Y_B
    hz = zt + HANDLE_Z_FROM_TOP
    return {"handle": [hx, hy, hz], "top_z": zt, "npts": int(len(Q))}


def _table_z(P, valid):
    m = valid & (P[:, 2] > 0.36) & (P[:, 2] < 0.50)
    if m.sum() < 100:
        return TABLE_Z_DEFAULT
    z = P[m, 2]
    h, edges = np.histogram(z, bins=70, range=(0.36, 0.50))
    i = int(np.argmax(h))
    zz = z[(z >= edges[i] - 0.004) & (z <= edges[i + 1] + 0.004)]
    tz = float(np.median(zz))
    if abs(tz - TABLE_Z_DEFAULT) > 0.03:
        return TABLE_Z_DEFAULT
    return tz


def _detect_boxes(frame, table_z=None):
    """Cream and popcorn (flat boxes) from wrist depth, labelled by colour (cream is bluish)."""
    P, valid = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    if table_z is None:
        table_z = _table_z(P, valid)
    rgb = np.asarray(frame.wrist_rgb, dtype=float).reshape(-1, 3)
    m = valid & (P[:, 2] > table_z + 0.006) & (P[:, 2] < table_z + 0.06) \
        & (P[:, 0] > -0.05) & (P[:, 0] < 0.40) & (P[:, 1] > -0.40) & (P[:, 1] < 0.10)
    Q = P[m]
    C = rgb[m]
    out = {"table_z": table_z, "cream": None, "popcorn": None, "npts": int(len(Q))}
    if len(Q) < 40:
        return out
    order = np.argsort(Q[:, 1])
    ys = Q[order, 1]
    gaps = np.diff(ys)
    i = int(np.argmax(gaps)) if len(gaps) else 0
    if len(gaps) == 0 or gaps[i] < 0.015:
        return out
    thr = (ys[i] + ys[i + 1]) / 2.0
    hi = Q[:, 1] > thr
    groups = []
    for g in (hi, ~hi):
        if g.sum() < 20:
            return out
        c = Q[g].mean(0)
        col = C[g].mean(0)
        groups.append({"c": c, "blue": float(col[2] - col[0]), "top": float(np.percentile(Q[g, 2], 90))})
    # cream is the bluish package; fall back on y-order (cream had larger y in all demos)
    if abs(groups[0]["blue"] - groups[1]["blue"]) > 8.0:
        ci = 0 if groups[0]["blue"] > groups[1]["blue"] else 1
    else:
        ci = 0
    for name, g in (("cream", groups[ci]), ("popcorn", groups[1 - ci])):
        out[name] = [float(g["c"][0] + OBJ_GRASP_OFFSET[0]), float(g["c"][1] + OBJ_GRASP_OFFSET[1]), g["top"]]
    return out


def _refine_box(frame, guess, table_z):
    """Re-localise one box near `guess` from a hover view (wrist depth). None if unsure."""
    P, valid = _backproject(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    d = np.hypot(P[:, 0] - guess[0], P[:, 1] - guess[1])
    m = valid & (P[:, 2] > table_z + 0.006) & (P[:, 2] < table_z + 0.06) & (d < 0.06)
    Q = P[m]
    if len(Q) < 60:
        return None
    c = Q.mean(0)
    if math.hypot(c[0] - guess[0], c[1] - guess[1]) > 0.03:
        return None
    return [float(c[0] + OBJ_GRASP_OFFSET[0]), float(c[1] + OBJ_GRASP_OFFSET[1]), float(np.percentile(Q[:, 2], 90))]


# ---------------- decision ----------------
FIELDS = ("stage", "order", "handle", "cream", "popcorn", "table_z", "cream_placed", "retries", "event")


def _wp(x, y, z, rot, tol=0.008):
    return [float(x), float(y), float(z), rot, float(tol)]


def _mv(mem, wps, grip, label, nxt, max_steps=150):
    mem["stage"] = nxt
    mem["event"] = label
    return {"tool": "act_move", "args": {"waypoints": wps, "gripper": float(grip),
                                         "max_steps_per_waypoint": int(max_steps)}}


def _gr(mem, close, steps, label, nxt):
    mem["stage"] = nxt
    mem["event"] = label
    return {"tool": "act_gripper", "args": {"close": bool(close), "steps": int(steps)}}


def _handle_approach(H):
    return [_wp(H[0], H[1] - 0.08, H[2], "door", 0.012), _wp(H[0], H[1] - 0.03, H[2], "door", 0.006),
            _wp(H[0], H[1], H[2], "door", 0.003)]


def _cream_place_default(H):
    return [H[0] + CREAM_RELEASE[0], H[1] + CREAM_RELEASE[1], CREAM_RELEASE[2]]


def _parse_order(goal):
    """Which package goes into the microwave first (the instruction names it first)."""
    g = str(goal or "").lower()
    ic, ip = g.find("cream"), g.find("popcorn")
    if ic >= 0 and ip >= 0 and ip < ic:
        return ["popcorn", "cream"]
    return ["cream", "popcorn"]


def _decide(obs, mem, goal):
    if mem["order"] is None or len(mem["order"]) != 2:
        mem["order"] = _parse_order(goal)
    first, second = mem["order"][0], mem["order"][1]
    if mem["stage"] is None:
        mem["stage"] = 0
    if mem["retries"] is None:
        mem["retries"] = 0
    f = obs.current
    tcp = _tcp(f)
    gap = _gap(f)
    st = int(mem["stage"])

    if st == 0 or mem["handle"] is None:
        mw = _detect_microwave(f)
        bx = _detect_boxes(f)
        H = mw["handle"] if mw is not None else [0.097, 0.411, 0.583]
        mem["handle"] = [float(v) for v in H]
        mem["table_z"] = float(bx["table_z"])
        tz = mem["table_z"]
        mem["cream"] = bx["cream"] if bx["cream"] is not None else [0.165, -0.091, tz + 0.014]
        mem["popcorn"] = bx["popcorn"] if bx["popcorn"] is not None else [0.164, -0.192, tz + 0.014]
        return _mv(mem, _handle_approach(H), -1.0, "approach handle", 1)

    H = mem["handle"]
    tz = mem["table_z"] if mem["table_z"] is not None else TABLE_Z_DEFAULT

    if st == 1:
        return _gr(mem, True, 22, "close on handle", 2)

    if st == 2:
        if (gap < 0.006 or gap > 0.07) and mem["retries"] < 2:
            # missed the handle (fingers closed on nothing): back off and retry with a
            # small lateral shift relative to the detected handle (memory keeps the detection)
            mem["retries"] = int(mem["retries"]) + 1
            dx = 0.012 if mem["retries"] == 1 else -0.012
            H2 = [H[0] + dx, H[1], H[2]]
            wps = [_wp(tcp[0], tcp[1] - 0.08, tcp[2], "door", 0.012)] + _handle_approach(H2)
            return _mv(mem, wps, -1.0, "handle grasp missed, retry", 1)
        hx, hy = H[0] + HINGE_REL[0], H[1] + HINGE_REL[1]
        a0 = math.degrees(math.atan2(tcp[1] - hy, tcp[0] - hx))
        r = math.hypot(tcp[0] - hx, tcp[1] - hy)
        if a0 < -60.0 or a0 > 20.0:
            a0 = -17.0   # TCP not where the closed door handle should be: use the demo start angle
        r = min(max(r, DOOR_RADIUS - 0.02), DOOR_RADIUS + 0.02)
        wps = []
        n = max(2, int(abs(DOOR_END_DEG - a0) / 6.0) + 1)
        for i in range(1, n + 1):
            a = math.radians(a0 + (DOOR_END_DEG - a0) * i / n)
            rr = r + (DOOR_RADIUS - r) * i / n
            wps.append(_wp(hx + rr * math.cos(a), hy + rr * math.sin(a), H[2], "door",
                           0.015 if i < n else 0.006))
        mem["retries"] = 0
        return _mv(mem, wps, 1.0, "pull door open", 3)

    if st == 3:
        return _gr(mem, False, 15, "release handle", 4)

    if st == 4:
        c = mem[first]
        wps = [_wp(tcp[0], tcp[1] - 0.08, tcp[2] + 0.02, "door", 0.015),
               _wp(c[0], c[1], HOVER_Z, "down", 0.006)]
        mem["retries"] = 0
        return _mv(mem, wps, -1.0, "retreat from door, hover over " + first, 5)

    if st in (5, 12):
        key = first if st == 5 else second
        g = mem[key]
        rf = _refine_box(f, g, tz)
        if rf is not None:
            mem[key] = rf
            g = rf
        gz = tz + GRASP_ABOVE_TABLE
        wps = [_wp(g[0], g[1], gz + 0.06, "down", 0.006), _wp(g[0], g[1], gz, "down", 0.003)]
        return _mv(mem, wps, -1.0, "descend to " + key + (" (refined)" if rf is not None else ""), st + 1)

    if st in (6, 13):
        return _gr(mem, True, 22, "grasp " + (first if st == 6 else second), st + 1)

    if st in (7, 14):
        key = first if st == 7 else second
        g = mem[key]
        if (gap < 0.008 or gap > 0.07) and mem["retries"] < 3:
            mem["retries"] = int(mem["retries"]) + 1
            return _mv(mem, [_wp(g[0], g[1], HOVER_Z, "down", 0.006)], -1.0, key + " grasp missed, retry", st - 2)
        if st == 7:
            sx = H[0] + CREAM_INSERT_A[0]
        else:
            cp = mem["cream_placed"] if mem["cream_placed"] is not None else _cream_place_default(H)
            sx = cp[0] + POP_REL_CREAM[0]
        sy = H[1] + STAGE_Y_REL
        wps = [_wp(g[0], g[1], CARRY_Z, "down", 0.012), _wp(sx, sy, CARRY_Z, "down", 0.01),
               _wp(sx, sy, CARRY_Z, "door", 0.006)]
        return _mv(mem, wps, 1.0, "lift " + key + ", stage and rotate", st + 1)

    if st == 8:
        wps = [_wp(H[0] + CREAM_INSERT_A[0], H[1] + CREAM_INSERT_A[1], CREAM_INSERT_A[2], "door", 0.008),
               _wp(H[0] + CREAM_RELEASE[0], H[1] + CREAM_RELEASE[1], CREAM_RELEASE[2], "door", 0.004)]
        return _mv(mem, wps, 1.0, "insert " + first, 9)

    if st == 9:
        return _gr(mem, False, 18, "release " + first, 10)

    if st == 10:
        mem["cream_placed"] = [float(v) for v in tcp]   # remembered placement location
        mem["retries"] = 0
        wps = [_wp(tcp[0], H[1] - 0.10, tcp[2], "door", 0.012), _wp(tcp[0], H[1] - 0.10, 0.70, "door", 0.012)]
        return _mv(mem, wps, -1.0, "remember " + first + " placement, retreat", 11)

    if st == 11:
        p = mem[second]
        return _mv(mem, [_wp(p[0], p[1], HOVER_Z, "down", 0.006)], -1.0, "hover over " + second, 12)

    cp = mem["cream_placed"] if mem["cream_placed"] is not None else _cream_place_default(H)
    tx, ty = cp[0] + POP_REL_CREAM[0], cp[1] + POP_REL_CREAM[1]

    if st == 15:
        wps = [_wp(tx, ty - 0.035, POP_Z, "door", 0.008), _wp(tx, ty, POP_RELEASE_Z, "door", 0.004)]
        return _mv(mem, wps, 1.0, "carry " + second + " to where " + first + " was placed", 16)

    if st == 16:
        return _gr(mem, False, 18, "release " + second, 17)

    if st == 17:
        return _mv(mem, [_wp(tcp[0], H[1] - 0.12, tcp[2], "door", 0.012)], -1.0, "retreat after popcorn", 18)

    mem["stage"] = 18
    mem["event"] = "done; waiting for scorer"
    return {"tool": "act_wait", "args": {"steps": 40}}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive microwave handle and boxes at the first decision, open the
    door, put cream in, remember where it was placed, then bring popcorn to that place."""
    memory = dict(memory) if isinstance(memory, dict) else {}
    mem = {k: memory.get(k) for k in FIELDS}
    act = _decide(obs, mem, goal)
    return {"memory": mem, "act": act}
