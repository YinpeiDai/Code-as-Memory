"""T21 Butter -> microwave, then chocolate -> the place where the butter was put.

Pipeline (one act per phase, progress kept in declared memory):
  start        perceive microwave handle, butter and chocolate from the first
               agentview RGB-D frame; open the microwave door by its handle.
  butter_pick  top-down grasp of the butter, lift.
  butter_place carry, turn the hand sideways, insert into the microwave, release;
               the measured release pose is stored as `butter_spot` (the memory
               the second instruction clause refers to).
  choc_pick    top-down grasp of the chocolate, lift.
  choc_place   carry, turn sideways, insert beside `butter_spot`, release.
  done         hold still (the checker settles the episode).
"""

_AV_K = [[231.76450199, 0.0, 127.5], [0.0, 309.01933598, 127.5], [0.0, 0.0, 1.0]]
_R_DOWN = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
_R_SIDE = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
_ZTAB = 0.430
_BUTTER_DEFAULT = [0.163, -0.094, 0.437]
_CHOC_DEFAULT = [0.164, -0.193, 0.440]
_HANDLE_DEFAULT = [0.095, 0.412, 0.583]
_HOLD_GAP = 0.008
_OPEN_GAP = 0.065

# ---------------------------------------------------------------- geometry

def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2quat(R):
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _rot_err(R_target, R_cur):
    """World-frame axis-angle vector rotating R_cur onto R_target (shortest)."""
    Re = np.asarray(R_target, dtype=float) @ np.asarray(R_cur, dtype=float).T
    q = _R2quat(Re)
    if q[0] < 0:
        q = -q
    vn = float(np.linalg.norm(q[1:]))
    if vn < 1e-9:
        return np.zeros(3)
    ang = 2.0 * math.atan2(vn, q[0])
    return q[1:] / vn * ang


def _ori(name):
    return np.array(_R_SIDE if name == "side" else _R_DOWN, dtype=float)


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _pos(frame):
    return np.asarray(frame.tcp, dtype=float)[:3]


def _backproject(depth, K, E):
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    Pc = np.stack([x, y, z], -1)
    return (Pc - E[:, 3]) @ E[:, :3]

# ---------------------------------------------------------------- perception

def _perceive_scene(frame):
    """Handle, butter and chocolate grasp points (world) from agentview RGB-D."""
    info = {"notes": []}
    dep = getattr(frame, "agentview_depth", None)
    if dep is None:
        return {"notes": ["no agentview depth; defaults"], "ztab": _ZTAB, "handle": list(_HANDLE_DEFAULT),
                "butter": None, "chocolate": None, "n_objects": 0}
    depth = np.asarray(dep)
    rgb = np.asarray(frame.agentview_rgb).astype(float)
    K = getattr(frame, "agentview_K", None)
    if K is None:
        K = _AV_K
    P = _backproject(depth, K, frame.agentview_extrinsic)
    valid = depth > 0
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    tab = valid & (np.abs(X - 0.15) < 0.2) & (Y > -0.45) & (Y < 0.25)
    ztab = _ZTAB
    if int(tab.sum()) > 500:
        zt = float(np.median(Z[tab]))
        if abs(zt - _ZTAB) < 0.05:
            ztab = zt
    info["ztab"] = ztab
    # microwave: everything raised in the right part of the table
    reg = valid & (X > -0.25) & (X < 0.4) & (Y > 0.15) & (Y < 0.75) & (Z > ztab + 0.02) & (Z < ztab + 0.45)
    handle = list(_HANDLE_DEFAULT)
    if int(reg.sum()) > 300:
        pm = P[reg]
        hx = float(np.max(pm[:, 0])) - 0.075
        hy = float(np.percentile(pm[:, 1], 2)) + 0.002
        if abs(hx - _HANDLE_DEFAULT[0]) < 0.12 and abs(hy - _HANDLE_DEFAULT[1]) < 0.12:
            handle = [hx, hy, ztab + 0.153]
        else:
            info["notes"].append("microwave estimate implausible; default handle")
    else:
        info["notes"].append("microwave not seen; default handle")
    info["handle"] = handle
    # small objects on the table, split by gaps along y
    obj = tab & (Z > ztab + 0.006) & (Z < ztab + 0.15)
    vs, us = np.nonzero(obj)
    groups = []
    if len(vs) >= 10:
        pts = P[vs, us]
        col = rgb[vs, us]
        order = np.argsort(pts[:, 1])
        ys = pts[order, 1]
        cuts = np.nonzero(np.diff(ys) > 0.015)[0]
        for g in np.split(order, cuts + 1):
            if len(g) < 10:
                continue
            p = pts[g]
            c = col[g]
            top = float(np.percentile(p[:, 2], 90))
            h = max(0.0, top - ztab)
            xy = np.median(p[:, :2], 0)
            groups.append({
                "x": float(xy[0]) - 0.5 * h,   # agentview sees the front face: shift back by half height
                "y": float(xy[1]) + 0.001,
                "z": max(ztab + 0.008, top - 0.012),
                "sat": float(np.mean(np.max(c, 1) - np.min(c, 1))),
                "n": int(len(g)),
            })
    info["n_objects"] = len(groups)
    butter = None
    choc = None
    if len(groups) >= 2:
        groups = sorted(groups, key=lambda g: -g["n"])[:2]
        gb = groups[0] if groups[0]["sat"] > groups[1]["sat"] else groups[1]
        gc = groups[1] if gb is groups[0] else groups[0]
        butter = [gb["x"], gb["y"], gb["z"]]
        choc = [gc["x"], gc["y"], gc["z"]]
    elif len(groups) == 1:
        g = groups[0]
        p = [g["x"], g["y"], g["z"]]
        if g["sat"] > 45:
            butter = p
        else:
            choc = p
        info["notes"].append("only one table object found")
    info["butter"] = butter
    info["chocolate"] = choc
    return info

def _door_open_count(frame, H):
    """Agentview points where the swung-open door panel stands (in front of and
    left of the microwave). 0 in all 25 closed-door first frames, 315-351 once
    open with the arm clear, ~100 when the arm partly occludes."""
    dep = getattr(frame, "agentview_depth", None)
    if dep is None or H is None:
        return 999
    depth = np.asarray(dep)
    K = getattr(frame, "agentview_K", None)
    if K is None:
        K = _AV_K
    P = _backproject(depth, K, frame.agentview_extrinsic) - np.asarray(H, dtype=float)
    m = (depth > 0) & (P[..., 1] > -0.25) & (P[..., 1] < -0.05) & (P[..., 0] > -0.45) & \
        (P[..., 0] < -0.2) & (P[..., 2] > -0.13) & (P[..., 2] < 0.08)
    return int(m.sum())

# ---------------------------------------------------------------- control

def _servo(robot, waypoints, ori, grip, pass_tol=0.02, tol=0.004, rot_tol=0.05,
           vmax=1.0, max_steps=200, stall_steps=12, kp=10.0, kr=1.0):
    """Drive the TCP through world waypoints (last one converged) holding a
    target orientation; gripper command held. Returns measured summary."""
    Rt = _ori(ori)
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    i = 0
    steps = 0
    best = 1e9
    since_best = 0
    fr = robot.now()
    while steps < max_steps:
        p = _pos(fr)
        Rc = _aa2R(np.asarray(fr.tcp, dtype=float)[3:6])
        er = _rot_err(Rt, Rc)
        last_wp = i == len(wps) - 1
        e = wps[i] - p
        d = float(np.linalg.norm(e))
        if not last_wp and d < pass_tol:
            i += 1
            continue
        if last_wp:
            if d < tol and float(np.linalg.norm(er)) < rot_tol:
                break
            if d < best - 0.0005:
                best = d
                since_best = 0
            else:
                since_best += 1
            if since_best > stall_steps and float(np.linalg.norm(er)) < 3 * rot_tol:
                break
            a = kp * e
        else:
            a = e / max(d, 1e-9) * vmax
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a / m * vmax
        ar = np.clip(kr * er, -0.5, 0.5)
        fr = robot.step(float(a[0]), float(a[1]), float(a[2]),
                        float(ar[0]), float(ar[1]), float(ar[2]), float(grip))
        steps += 1
    p = _pos(fr)
    return {"steps": steps, "pos": [float(v) for v in p],
            "err": float(np.linalg.norm(wps[-1] - p)), "gap": _gap(fr)}


def _hold(robot, n, grip):
    fr = robot.now()
    for _ in range(int(n)):
        fr = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
    return fr


def _holding(g):
    """Closed fingers stopped by an object: butter 0.038, chocolate 0.046 in demos."""
    return _HOLD_GAP < g < _OPEN_GAP


def _add(a, b):
    return [float(a[0]) + b[0], float(a[1]) + b[1], float(a[2]) + b[2]]

# ---------------------------------------------------------------- policy

def perceive_decide(obs, goal, memory, last):
    """Phase machine for: butter into the microwave, then chocolate to the same place."""
    mem = dict(memory)
    for k in ("phase", "handle", "butter", "chocolate", "butter_spot", "ztab", "retries", "log"):
        if k not in mem:
            mem[k] = None
    if mem["phase"] is None:
        mem["phase"] = "start"
    if mem["retries"] is None:
        mem["retries"] = 0
    if mem["log"] is None:
        mem["log"] = ""
    cur = obs.current
    gap = _gap(cur)
    ok_last = isinstance(last, str) and '"ok": true' in last
    phase = mem["phase"]

    def done(next_phase, note):
        mem["phase"] = next_phase
        mem["retries"] = 0
        mem["log"] = (mem["log"] + " | " + note)[-600:]

    # ----- update progress from the previous act's outcome
    if phase == "start" and isinstance(last, str) and last.startswith("act_open_door"):
        cnt = _door_open_count(cur, mem["handle"])
        if cnt >= 60 or int(mem["retries"]) >= 2:
            done("butter_pick", "door pts=%d grasp=%s" % (cnt, "ok" if ok_last else "missed"))
        else:
            mem["retries"] = int(mem["retries"]) + 1
            mem["log"] = (mem["log"] + " | door closed? pts=%d retry" % cnt)[-600:]
    elif phase == "butter_pick" and isinstance(last, str) and last.startswith("act_pick"):
        if _holding(gap):
            done("butter_place", "butter held gap=%.3f" % gap)
        else:
            mem["retries"] = int(mem["retries"]) + 1
            mem["phase"] = "butter_view"
    elif phase == "butter_view" and isinstance(last, str) and last.startswith("act_clear_view"):
        mem["phase"] = "butter_pick"
    elif phase == "butter_place" and isinstance(last, str) and last.startswith("act_insert"):
        sp = _parse_list(last, "release")
        if '"lost": true' in last:
            mem["retries"] = int(mem["retries"]) + 1
            mem["phase"] = "butter_view"
        else:
            if sp is not None:
                mem["butter_spot"] = sp
            done("choc_pick", "butter placed")
    elif phase == "choc_pick" and isinstance(last, str) and last.startswith("act_pick"):
        if _holding(gap):
            done("choc_place", "chocolate held gap=%.3f" % gap)
        else:
            mem["retries"] = int(mem["retries"]) + 1
            mem["phase"] = "choc_view"
    elif phase == "choc_view" and isinstance(last, str) and last.startswith("act_clear_view"):
        mem["phase"] = "choc_pick"
    elif phase == "choc_place" and isinstance(last, str) and last.startswith("act_insert"):
        if '"lost": true' in last:
            mem["retries"] = int(mem["retries"]) + 1
            mem["phase"] = "choc_view"
        else:
            done("done", "chocolate placed")
    phase = mem["phase"]

    # ----- perception that must persist (first frame shows everything unoccluded)
    if mem["handle"] is None:
        info = _perceive_scene(cur)
        mem["handle"] = info["handle"]
        mem["ztab"] = info["ztab"]
        mem["butter"] = info["butter"] if info["butter"] is not None else list(_BUTTER_DEFAULT)
        mem["chocolate"] = info["chocolate"] if info["chocolate"] is not None else list(_CHOC_DEFAULT)
        mem["log"] = ("seen objs=%d %s" % (info["n_objects"], ";".join(info["notes"])))[-600:]
    elif phase in ("butter_pick", "choc_pick") and int(mem["retries"]) > 0:
        # re-measure after the arm moved clear of the table
        info = _perceive_scene(cur)
        key = "butter" if phase == "butter_pick" else "chocolate"
        new = info[key]
        if new is not None and float(np.linalg.norm(np.array(new[:2]) - np.array(mem[key][:2]))) < 0.15:
            mem[key] = new
    H = mem["handle"]
    zt = float(mem["ztab"]) if mem["ztab"] is not None else _ZTAB

    if phase == "start":
        act = {"tool": "act_open_door", "args": {"handle": H, "extra_push": 0.008 * int(mem["retries"])}}
    elif phase in ("butter_pick", "choc_pick"):
        key = "butter" if phase == "butter_pick" else "chocolate"
        if int(mem["retries"]) > 2:
            # give up on this object and move on
            done("butter_place" if key == "butter" else "choc_place", key + " grasp failed")
            phase = mem["phase"]
            act = _place_act(mem, H, phase)
        else:
            act = {"tool": "act_pick", "args": {"target": mem[key], "lift_z": zt + 0.128}}
    elif phase in ("butter_view", "choc_view"):
        act = {"tool": "act_clear_view", "args": {}}
    elif phase in ("butter_place", "choc_place"):
        act = _place_act(mem, H, phase)
    else:
        act = {"tool": "act_idle", "args": {"steps": 60}}
    return {"memory": mem, "act": act}


def _place_act(mem, H, phase):
    if phase == "butter_place":
        stage = _add(H, [-0.075, -0.315, -0.027])
        path = [_add(H, [-0.092, -0.142, 0.003]), _add(H, [-0.100, -0.030, 0.016]),
                _add(H, [-0.097, 0.039, 0.015]), _add(H, [-0.082, 0.060, -0.002])]
        back = [_add(H, [-0.068, -0.029, -0.005]), _add(H, [-0.052, -0.090, -0.010]),
                _add(H, [-0.050, -0.100, 0.128])]
        return {"tool": "act_insert", "args": {"stage": stage, "path": path, "back": back}}
    spot = mem["butter_spot"]
    if spot is None:
        spot = _add(H, [-0.082, 0.060, -0.002])
    stage = _add(spot, [-0.043, -0.475, -0.022])
    path = [_add(spot, [-0.027, -0.300, 0.005]), _add(spot, [-0.027, -0.121, 0.019]),
            _add(spot, [-0.027, -0.035, 0.013])]
    back = [_add(spot, [-0.027, -0.250, 0.020]), _add(spot, [-0.027, -0.300, 0.100])]
    return {"tool": "act_insert", "args": {"stage": stage, "path": path, "back": back}}


def _parse_list(text, key):
    """Read '"key": [a, b, c]' from an outcome string; None if absent."""
    tag = '"' + key + '": ['
    i = text.find(tag)
    if i < 0:
        return None
    j = text.find("]", i)
    if j < 0:
        return None
    parts = text[i + len(tag):j].split(",")
    vals = []
    for s in parts:
        s = s.strip()
        if not s:
            return None
        ok = all(ch in "0123456789.-+e" for ch in s)
        if not ok:
            return None
        vals.append(float(s))
    if len(vals) != 3:
        return None
    return vals

# ---------------------------------------------------------------- act tools

def act_open_door(robot, handle, extra_push=0.0):
    """Turn the hand sideways, close on the microwave handle and swing the door
    open along the arc the expert used; release and rise clear with the hand
    turned back down. Returns grasp gap and final pose."""
    H = [float(v) for v in handle]
    out = {"tries": 0}
    grabbed = False
    push = float(extra_push)
    for t in range(3):
        out["tries"] = t + 1
        _servo(robot, [_add(H, [-0.112, -0.281, 0.027]), _add(H, [-0.059, -0.164, 0.011]),
                       _add(H, [-0.055, -0.063 + push, -0.002])],
               "side", -1, pass_tol=0.03, tol=0.006, max_steps=120)
        # press gently past the measured handle point: the expert stopped on contact there
        r = _servo(robot, [_add(H, [-0.030, -0.028 + push, 0.0]), _add(H, [0.020, 0.006 + push, 0.0])],
                   "side", -1, pass_tol=0.015, tol=0.003, max_steps=60, stall_steps=8)
        out["at_handle"] = r["pos"]
        fr = _hold(robot, 22, 1)
        g = _gap(fr)
        out["grasp_gap"] = g
        if _holding(g):
            grabbed = True
            break
        # missed: open, back off and try a little deeper
        _hold(robot, 10, -1)
        _servo(robot, [_add(H, [-0.055, -0.07, 0.0])], "side", -1, tol=0.008, max_steps=40)
        push += 0.01
    G = _pos(robot.now())
    rel = [[-0.040, -0.064, 0.0], [-0.087, -0.118, 0.0], [-0.133, -0.155, 0.001],
           [-0.177, -0.181, 0.001], [-0.223, -0.198, 0.001], [-0.277, -0.200, 0.001],
           [-0.328, -0.193, 0.0], [-0.364, -0.186, 0.001]]
    path = [[G[0] + d[0], G[1] + d[1], G[2] + d[2]] for d in rel]
    r = _servo(robot, path, "side", 1, pass_tol=0.02, tol=0.006, max_steps=120, stall_steps=10)
    out["pull_end"] = r["pos"]
    out["gap_after_pull"] = r["gap"]
    _hold(robot, 10, -1)
    _servo(robot, [_add(H, [-0.354, -0.250, 0.022]), _add(H, [-0.295, -0.314, 0.057]),
                   _add(H, [-0.233, -0.374, 0.077]), _add(H, [-0.162, -0.421, 0.090])],
           "down", -1, pass_tol=0.03, tol=0.01, rot_tol=0.08, max_steps=120)
    out["ok"] = bool(grabbed)
    return out


def act_pick(robot, target, lift_z):
    """Top-down grasp at world `target` [x, y, z]: hover 12 cm above, descend,
    close, lift to `lift_z`. Returns the finger gap while lifted."""
    T = [float(v) for v in target]
    p = _pos(robot.now())
    above = [T[0], T[1], T[2] + 0.12]
    wps = []
    if p[2] < T[2] + 0.08:
        wps.append([p[0], p[1], T[2] + 0.12])
    wps.append(above)
    _servo(robot, wps, "down", -1, pass_tol=0.03, tol=0.006, rot_tol=0.04, max_steps=200)
    r = _servo(robot, [[T[0], T[1], T[2] + 0.04], T], "down", -1, pass_tol=0.01, tol=0.003,
               rot_tol=0.04, vmax=0.5, max_steps=80, stall_steps=8)
    fr = _hold(robot, 20, 1)
    g0 = _gap(fr)
    r2 = _servo(robot, [[T[0], T[1], float(lift_z)]], "down", 1, tol=0.005, max_steps=80)
    fr = _hold(robot, 3, 1)
    g = _gap(fr)
    return {"ok": bool(_holding(g)), "gap_closed": g0, "gap_lifted": g,
            "grasp_at": r["pos"], "lifted_to": r2["pos"]}


def act_insert(robot, stage, path, back):
    """Carry the held object to `stage`, turn the hand sideways, follow `path`
    into the microwave, release, then retreat along `back`. Returns the
    measured release pose as `release`."""
    p = _pos(robot.now())
    S = [float(v) for v in stage]
    _servo(robot, [[S[0], S[1], max(S[2], p[2])], S], "down", 1, pass_tol=0.03, tol=0.008,
           max_steps=150)
    _servo(robot, [S], "side", 1, tol=0.008, rot_tol=0.04, max_steps=80)
    if not _holding(_gap(robot.now())):
        # the object slipped out on the way: open and report, the policy re-picks
        _hold(robot, 8, -1)
        _servo(robot, [[S[0], S[1], S[2] + 0.12]], "down", -1, tol=0.01, max_steps=80)
        return {"ok": False, "lost": True, "release": None, "end": [float(v) for v in _pos(robot.now())]}
    r = _servo(robot, path, "side", 1, pass_tol=0.02, tol=0.004, rot_tol=0.05, max_steps=150)
    rel = r["pos"]
    held = r["gap"]
    _hold(robot, 15, -1)
    b = _servo(robot, back, "side", -1, pass_tol=0.03, tol=0.01, max_steps=120)
    return {"ok": True, "lost": False, "release": rel, "gap_before_release": held, "end": b["pos"]}


def act_clear_view(robot):
    """Open the gripper and raise the arm to the start pose so agentview sees the table."""
    _hold(robot, 8, -1)
    p = _pos(robot.now())
    r = _servo(robot, [[p[0], p[1], 0.67], [-0.058, 0.0, 0.671]], "down", -1, pass_tol=0.03,
               tol=0.01, max_steps=150)
    return {"ok": True, "pos": r["pos"]}


def act_idle(robot, steps=60):
    """Hold still with the gripper open for `steps` native steps."""
    fr = _hold(robot, steps, -1)
    return {"ok": True, "pos": [float(v) for v in _pos(fr)]}
