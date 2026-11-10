"""T04 ButterNonEmptyDrawer: check top, middle, bottom drawers (open + close each), remember which one
contains an object, then reopen it and put the butter inside.

All geometry is world-frame metres. Perception uses the agentview RGB-D point cloud (calibrated K/extrinsic).
"""

ORDER = ["top", "middle", "bottom"]
# Offsets relative to the perceived cabinet top surface (z) and its front edge (y); from 32 demos (std ~1 mm).
HANDLE_DZ = {"top": -0.018, "middle": -0.104, "bottom": -0.183}
HANDLE_DY = {"top": -0.020, "middle": -0.026, "bottom": -0.026}
FLOOR_DZ = {"top": -0.063, "middle": -0.133, "bottom": -0.203}
HANDLE_DX = -0.128          # handle x relative to cabinet near (+x) edge
PULL = 0.17                 # drawer pull distance (demo 0.165-0.177)
DEFAULT_CAB = [0.265, 0.468, 0.713]   # [near-edge x, front y, top z] demo means (fallback only)
DEFAULT_BUTTER = [0.111, -0.202, 0.437]
ROT_H = [4.71238898, 0.0, 0.0]        # gripper pointing +y (towards drawer fronts)
ROT_D = [3.14159265, 0.0, 0.0]        # gripper pointing down
SAFE_Z = 0.84               # carry height (0.93 was out of reach over the far drawer, R002)
MEMORY_FIELDS = ["stage", "cabinet", "butter", "idx", "scores", "open_ok", "grasp_y", "retries",
                 "target", "pending", "log"]


# ---------------------------------------------------------------- math helpers
def _aa2mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(float(M[i, i]), 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _rot_err(tcp, rot_aa):
    return _mat2aa(_aa2mat(rot_aa) @ _aa2mat(tcp[3:6]).T)


# ---------------------------------------------------------------- perception helpers
def _cloud(frame):
    d = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    R, t = E[:, :3], E[:, 3]
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    Pc = np.stack([x, y, d], -1)
    Pw = (Pc - t) @ R
    return Pw, d > 0


def _perceive_cabinet(frame):
    """[near-edge x, front y, top z] of the dark cabinet from the agentview cloud, or None."""
    P, m = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    br = rgb.max(-1)
    sel = (m & (br < 60) & (P[..., 2] > 0.45) & (P[..., 1] > 0.15) & (P[..., 1] < 0.9)
           & (P[..., 0] > -0.2) & (P[..., 0] < 0.5))
    Q = P[sel]
    if len(Q) < 500:
        return None
    ztop = float(np.percentile(Q[:, 2], 99.5))
    top = Q[Q[:, 2] > ztop - 0.008]
    if len(top) < 100:
        return None
    x98 = float(np.percentile(Q[:, 0], 98))
    ty2 = float(np.percentile(top[:, 1], 2))
    ztop = float(np.median(top[:, 2]))
    return [x98, ty2, ztop]


def _perceive_butter(frame):
    """Grasp point of the butter (saturated small box on the table near the expected place), or None."""
    P, m = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    sat = rgb.max(-1) - rgb.min(-1)
    sel = (m & (sat > 50) & (P[..., 2] > 0.425) & (P[..., 2] < 0.5) & (P[..., 0] > -0.1)
           & (P[..., 0] < 0.35) & (P[..., 1] > -0.45) & (P[..., 1] < 0.1))
    Q = P[sel]
    if len(Q) < 10:
        return None
    d = np.linalg.norm(Q[:, :2] - np.array(DEFAULT_BUTTER[:2]), axis=1)
    near = Q[d < 0.08]
    if len(near) < 10:
        return None
    c = near.mean(0)
    # visible points are biased toward the camera (+x); demo offset centroid -> grasp: (-0.014, +0.004)
    return [float(c[0] - 0.014), float(c[1] + 0.004), DEFAULT_BUTTER[2]]


def _handle(cab, level):
    return [cab[0] + HANDLE_DX, cab[1] + HANDLE_DY[level], cab[2] + HANDLE_DZ[level]]


def _drawer_view(frame, cab, level, open_y):
    """Counts inside the (opened) drawer: floor points (drawer really open) and object-like pixels."""
    P, m = _cloud(frame)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    sat = rgb.max(-1) - rgb.min(-1)
    br = rgb.max(-1)
    hx = cab[0] + HANDLE_DX
    floor = cab[2] + FLOOR_DZ[level]
    box = m & (np.abs(P[..., 0] - hx) < 0.09) & (P[..., 1] > open_y + 0.025) & (P[..., 1] < cab[1] - 0.02)
    floor_n = int((box & (np.abs(P[..., 2] - floor) < 0.012)).sum())
    above = box & (P[..., 2] > floor + 0.008) & (P[..., 2] < floor + 0.09)
    colored = int((above & ((sat > 25) | (br > 90))).sum())
    geo = int((box & (P[..., 2] > floor + 0.015) & (P[..., 2] < floor + 0.09)).sum())
    return {"floor": floor_n, "colored": colored, "geo": geo, "box": int(box.sum())}


# ---------------------------------------------------------------- motion helpers
def _hold(robot, n, grip):
    for _ in range(int(n)):
        robot.step(0, 0, 0, 0, 0, 0, grip)
    return robot.now()


def _move(robot, pos, rot, grip, tol=0.004, rtol=0.03, max_steps=150, k=10.0, vmax=1.0, stall=0):
    """Closed-loop OSC delta move to world pos / axis-angle rot. stall>0: stop when progress stalls."""
    pos = np.asarray(pos, dtype=float)
    hist = []
    n = 0
    reason = "max_steps"
    for n in range(int(max_steps)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        e = pos - tcp[:3]
        er = _rot_err(tcp, rot)
        if np.linalg.norm(e) < tol and np.linalg.norm(er) < rtol:
            reason = "reached"
            break
        hist.append(tcp[:3].copy())
        if stall and len(hist) > stall + 5:
            if np.linalg.norm(hist[-1] - hist[-1 - stall]) < 0.0015:
                reason = "stalled"
                break
        a = np.clip(k * e, -vmax, vmax)
        ar = np.clip(2.0 * er, -0.5, 0.5)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(ar[0]), float(ar[1]), float(ar[2]), grip)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"reason": reason, "steps": n, "err": float(np.linalg.norm(pos - tcp[:3])),
            "tcp": [round(float(v), 4) for v in tcp[:3]]}


def _close_gripper(robot, n_max=25):
    prev = None
    for i in range(n_max):
        f = robot.step(0, 0, 0, 0, 0, 0, 1)
        fw = float(np.asarray(f.fingers)[0])
        if i >= 8 and prev is not None and abs(fw - prev) < 0.0003:
            break
        prev = fw
    _hold(robot, 3, 1)
    return float(np.asarray(robot.now().fingers)[0])


def _open_gripper(robot, n=12):
    _hold(robot, n, -1)
    return float(np.asarray(robot.now().fingers)[0])


# ---------------------------------------------------------------- act tools
def act_open_drawer(robot, handle, extra_dy=0.0, extra_dz=0.0):
    """Rotate the gripper horizontal (+y), approach the drawer handle [x,y,z], grasp it and pull the drawer
    open by PULL metres, then hold still. Returns grasp/pull measurements (fingers, tcp, pull distance)."""
    hx, hy, hz = float(handle[0]), float(handle[1]) + extra_dy, float(handle[2]) + extra_dz
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out = {}
    if np.linalg.norm(_rot_err(tcp, ROT_H)) > 0.4 and tcp[1] > hy - 0.12:
        # Not yet oriented and too close to cabinet: back away first.
        _move(robot, [tcp[0], hy - 0.14, tcp[2]], ROT_H, -1, tol=0.01, rtol=0.2,
              max_steps=60)
    out["pre"] = _move(robot, [hx, hy - 0.08, hz], ROT_H, -1, tol=0.006, rtol=0.04, max_steps=150)
    out["approach"] = _move(robot, [hx, hy, hz], ROT_H, -1, tol=0.003, rtol=0.03, max_steps=60, vmax=0.5,
                            stall=6)
    gy = float(np.asarray(robot.now().tcp)[1])
    out["grasp_y"] = gy
    out["fingers_grasp"] = _close_gripper(robot)
    out["pull"] = _move(robot, [hx, gy - PULL, hz], ROT_H, 1, tol=0.004, rtol=0.05, max_steps=90, stall=8)
    f = _hold(robot, 4, 1)
    out["fingers_pulled"] = float(np.asarray(f.fingers)[0])
    out["pulled"] = float(gy - float(np.asarray(f.tcp)[1]))
    return out


def act_close_drawer(robot, handle, grasp_y):
    """Push the held drawer back to its closed position (grasp_y), release the handle and back away 7 cm
    (-y). Returns the push stop position and fingers."""
    hx, hz = float(handle[0]), float(handle[2])
    out = {}
    out["push"] = _move(robot, [hx, float(grasp_y) + 0.004, hz], ROT_H, 1, tol=0.004, rtol=0.05,
                        max_steps=90, stall=8)
    _hold(robot, 3, 1)
    out["closed_y"] = float(np.asarray(robot.now().tcp)[1])
    out["fingers_open"] = _open_gripper(robot, 12)
    y = float(np.asarray(robot.now().tcp)[1])
    out["retreat"] = _move(robot, [hx, y - 0.07, hz], ROT_H, -1, tol=0.008, rtol=0.06, max_steps=50)
    return out


def act_release_open(robot):
    """Release a held (open) drawer handle, back away 8 cm (-y) and lift to z=0.80, keeping it open."""
    out = {"fingers_open": _open_gripper(robot, 12)}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["back"] = _move(robot, [tcp[0], tcp[1] - 0.08, tcp[2]], ROT_H, -1, tol=0.008, rtol=0.06, max_steps=50)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["lift"] = _move(robot, [tcp[0], tcp[1], 0.80], ROT_H, -1, tol=0.01, rtol=0.06, max_steps=60)
    return out


def act_pick_butter(robot, butter, dz=0.0):
    """Rotate gripper down, move above the butter [x,y,z], descend, grasp and lift to z=0.80.
    Returns fingers after grasp and after lift (about 0 means missed)."""
    bx, by, bz = float(butter[0]), float(butter[1]), float(butter[2]) + dz
    out = {}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if tcp[2] < 0.75:
        _move(robot, [tcp[0], tcp[1], 0.80], [float(v) for v in tcp[3:6]], -1, tol=0.015, rtol=1.0, max_steps=40)
    out["above"] = _move(robot, [bx, by, 0.62], ROT_D, -1, tol=0.006, rtol=0.03, max_steps=180)
    out["pre"] = _move(robot, [bx, by, bz + 0.05], ROT_D, -1, tol=0.004, rtol=0.03, max_steps=60)
    out["descend"] = _move(robot, [bx, by, bz], ROT_D, -1, tol=0.003, rtol=0.03, max_steps=60, vmax=0.4,
                           stall=6)
    out["fingers_grasp"] = _close_gripper(robot)
    out["lift"] = _move(robot, [bx, by, 0.80], ROT_D, 1, tol=0.008, rtol=0.05, max_steps=100)
    out["fingers_lift"] = float(np.asarray(robot.now().fingers)[0])
    return out


def act_place_in_drawer(robot, place, release_z):
    """Carry the held butter high (z=SAFE_Z) to above place [x,y], descend to release_z, open the gripper,
    then lift away and wait. Returns positions and fingers."""
    px, py = float(place[0]), float(place[1])
    out = {}
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["up"] = _move(robot, [tcp[0], tcp[1], SAFE_Z], ROT_D, 1, tol=0.01, rtol=0.05, max_steps=80)
    out["over"] = _move(robot, [px, py, SAFE_Z], ROT_D, 1, tol=0.006, rtol=0.04, max_steps=150, stall=10)
    out["down"] = _move(robot, [px, py, float(release_z)], ROT_D, 1, tol=0.004, rtol=0.04, max_steps=120,
                        vmax=0.4, stall=8)
    out["fingers_held"] = float(np.asarray(robot.now().fingers)[0])
    out["fingers_open"] = _open_gripper(robot, 15)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    out["lift"] = _move(robot, [tcp[0], tcp[1], SAFE_Z], ROT_D, -1, tol=0.01, rtol=0.05, max_steps=80)
    _hold(robot, 20, -1)
    return out


def act_wait(robot, steps=20):
    """Hold still with the gripper open for `steps` native steps (lets the scorer settle)."""
    _hold(robot, steps, -1)
    return {"waited": int(steps)}


# ---------------------------------------------------------------- decision
def _mem(memory):
    m = {k: memory.get(k) if isinstance(memory, dict) else None for k in MEMORY_FIELDS}
    if m["scores"] is None:
        m["scores"] = [None, None, None]
    if m["open_ok"] is None:
        m["open_ok"] = [None, None, None]
    if m["idx"] is None:
        m["idx"] = 0
    if m["retries"] is None:
        m["retries"] = 0
    if m["log"] is None:
        m["log"] = ""
    return m


def _note(m, s):
    m["log"] = (m["log"] + " | " + s)[-1500:]


def _last_num(last, key):
    """Read a numeric field '"key": value' from the previous act result text, or None."""
    if not isinstance(last, str):
        return None
    tag = '"' + key + '": '
    j = last.find(tag)
    if j < 0:
        return None
    j += len(tag)
    k = j
    while k < len(last) and last[k] in "-+.0123456789eE":
        k += 1
    try:
        return float(last[j:k])
    except ValueError:
        return None


DIAG_FIRST_DY = 0.0   # diagnostic hook (used only in exploratory attempt 21); keep 0.0
FORCE_TARGET = None   # diagnostic override hook (used only in exploratory attempts 2-3); keep None


def _choose_target(scores):
    if FORCE_TARGET in ORDER:
        return FORCE_TARGET, "forced_diagnostic"
    best, bi = -1, None
    for i, s in enumerate(scores):
        if s is not None and s[0] > best:
            best, bi = s[0], i
    if bi is not None and best >= 8:
        return ORDER[bi], "colored"
    # fallback: largest geometric fill ratio
    best, bi = -1.0, 0
    for i, s in enumerate(scores):
        if s is not None and s[2] > 0 and s[1] / float(s[2]) > best:
            best, bi = s[1] / float(s[2]), i
    return ORDER[bi], "geometric_fallback"


def _place_for(cab, level):
    floor = cab[2] + FLOOR_DZ[level]
    rz = floor + 0.10 if level == "top" else max(floor + 0.10, cab[2] - 0.01)
    return [cab[0] + HANDLE_DX - 0.01, cab[1] - 0.085], rz


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive cabinet/butter, check drawers top->middle->bottom (open, look, close),
    reopen the drawer containing an object, pick the butter and place it there."""
    m = _mem(memory)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    fing = float(np.asarray(cur.fingers)[0])

    def ret(tool, args):
        out = {k: m[k] for k in MEMORY_FIELDS}
        return {"memory": out, "act": {"tool": tool, "args": args}}

    if m["stage"] is None:
        cab = _perceive_cabinet(obs.frames[0]) or _perceive_cabinet(cur)
        if cab is None:
            cab = list(DEFAULT_CAB)
            _note(m, "cabinet not detected: default")
        m["cabinet"] = [round(float(v), 4) for v in cab]
        b = _perceive_butter(obs.frames[0]) or _perceive_butter(cur)
        if b is None:
            b = list(DEFAULT_BUTTER)
            _note(m, "butter not detected: default")
        m["butter"] = [round(float(v), 4) for v in b]
        m["stage"] = "check"
        m["idx"] = 0
        m["pending"] = "open"
        m["retries"] = 0
        args = {"handle": _handle(m["cabinet"], ORDER[0])}
        if DIAG_FIRST_DY:
            args["extra_dy"] = DIAG_FIRST_DY
        return ret("act_open_drawer", args)

    cab = m["cabinet"]
    if m["stage"] in ("check", "target"):
        i = int(m["idx"]) if m["stage"] == "check" else ORDER.index(m["target"])
        level = ORDER[i]
        h = _handle(cab, level)
        if m["pending"] == "open":
            # find grasp y from the frames: tcp y at the moment the fingers started closing is approx.
            # we use the furthest +y reached in this action's frames near handle height
            gy = _last_num(last, "grasp_y")
            if gy is None:
                ys = [float(np.asarray(f.tcp)[1]) for f in obs.frames]
                gy = max(ys) if ys else h[1]
            m["grasp_y"] = round(gy, 4)
            pulled = gy - tcp[1]
            view = _drawer_view(cur, cab, level, tcp[1])
            held = fing > 0.004
            ok = held and pulled > 0.12 and view["floor"] >= 90
            _note(m, "%s open: fing=%.4f pulled=%.3f floor=%d col=%d" % (level, fing, pulled, view["floor"],
                                                                        view["colored"]))
            if not ok:
                if m["retries"] < 2:
                    m["retries"] += 1
                    dy = 0.008 * m["retries"]
                    dz = [0.0, -0.006, 0.006][m["retries"]]
                    m["pending"] = "open_retry"
                    # release whatever we hold, back away, then retry with adjusted approach
                    return ret("act_close_drawer", {"handle": h, "grasp_y": gy})
                _note(m, "%s open failed after retries" % level)
            if m["stage"] == "check":
                m["open_ok"][i] = bool(ok)
                m["scores"][i] = [view["colored"], view["geo"], view["box"]]
                m["pending"] = "close"
                return ret("act_close_drawer", {"handle": h, "grasp_y": gy})
            m["pending"] = "release"
            return ret("act_release_open", {})
        if m["pending"] == "open_retry":
            m["pending"] = "open"
            r = int(m["retries"])
            return ret("act_open_drawer", {"handle": h, "extra_dy": 0.008 * r,
                                           "extra_dz": [0.0, -0.006, 0.006][r]})
        if m["pending"] == "close":
            gy = m["grasp_y"] if m["grasp_y"] is not None else h[1]
            ys = [float(np.asarray(f.tcp)[1]) for f in obs.frames]
            _note(m, "%s closed: maxy=%.3f vs grasp %.3f" % (level, max(ys) if ys else -1, gy))
            m["idx"] = i + 1
            m["retries"] = 0
            if m["idx"] < 3:
                m["pending"] = "open"
                return ret("act_open_drawer", {"handle": _handle(cab, ORDER[m["idx"]])})
            tgt, how = _choose_target(m["scores"])
            m["target"] = tgt
            _note(m, "target=%s (%s) scores=%s" % (tgt, how, m["scores"]))
            m["stage"] = "target"
            m["pending"] = "open"
            return ret("act_open_drawer", {"handle": _handle(cab, tgt)})
        if m["pending"] == "release":
            m["stage"] = "pick"
            m["retries"] = 0
            m["pending"] = "pick"
            return ret("act_pick_butter", {"butter": m["butter"]})

    if m["stage"] == "pick":
        held = fing > 0.004 and tcp[2] > 0.7
        _note(m, "pick: fing=%.4f z=%.3f" % (fing, tcp[2]))
        if held:
            m["stage"] = "place"
            place, rz = _place_for(cab, m["target"])
            return ret("act_place_in_drawer", {"place": place, "release_z": rz})
        if m["retries"] < 2:
            m["retries"] += 1
            b = _perceive_butter(cur)
            if b is not None:
                m["butter"] = [round(float(v), 4) for v in b]
            dz = [0.0, -0.008, 0.008][m["retries"]]
            return ret("act_pick_butter", {"butter": m["butter"], "dz": dz})
        m["stage"] = "done"
        return ret("act_wait", {"steps": 20})

    if m["stage"] == "place":
        # Episode still running after the place: check whether new colored pixels appeared in the target drawer.
        lv = m["target"]
        ti = ORDER.index(lv)
        gy = m["grasp_y"] if m["grasp_y"] is not None else _handle(cab, lv)[1]
        view = _drawer_view(cur, cab, lv, gy - PULL)
        before = m["scores"][ti][0] if m["scores"][ti] is not None else 0
        _note(m, "after place: %s colored=%d (before %d) fing=%.4f" % (lv, view["colored"], before, fing))
        if view["colored"] >= before + 15 or m["retries"] >= 3:
            m["stage"] = "done"
            return ret("act_wait", {"steps": 40})
        # butter not seen in the drawer: look for it near its last known place and retry once more
        m["retries"] = 3
        b = _perceive_butter(cur)
        if b is not None:
            m["butter"] = [round(float(v), 4) for v in b]
        m["stage"] = "pick"
        return ret("act_pick_butter", {"butter": m["butter"]})

    return ret("act_wait", {"steps": 40})
