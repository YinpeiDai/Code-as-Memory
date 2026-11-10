"""T18 ChocolateButterCabinet: move chocolate pudding and butter from the top of
cabinet1 (the cabinet holding the two boxes, at -y) to the top of cabinet2.

Perception: agentview RGB-D (calibrated) finds the two dark cabinet tops at
frame 0; a top-down wrist RGB-D scan above cabinet1 finds the boxes (points
above the top surface), classified by colour (butter = bright saturated red,
chocolate pudding = dark red-brown).  Control: proportional OSC_POSE delta
steps toward world targets with fixed top-down orientation."""

TOP_Z_DEFAULT = 0.663
HOVER_Z = 0.95
CARRY_Z = 1.0
DOWN_QUAT_AA = [math.pi, 0.0, 0.0]
CHOC_OFFSET_Y = 0.067
BUTTER_OFFSET_Y = -0.070
HELD_MIN_WIDTH = 0.008


# ---------------------------------------------------------------- geometry
def _cloud(depth, K, E):
    d = np.asarray(depth, dtype=float) / 1000.0
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    K = np.asarray(K, dtype=float)
    E = np.asarray(E, dtype=float)
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    Pc = np.stack([x, y, d], -1)
    Pw = (Pc - E[:, 3]) @ E[:, :3]
    Pw[d <= 0] = np.nan
    return Pw


def _aa_to_mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * (Kx @ Kx)


def _mat_to_aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        # near pi: use diagonal
        k = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(k))
        if i == 0:
            k = np.array([k[0], R[0, 1] / (2 * k[0]), R[0, 2] / (2 * k[0])])
        elif i == 1:
            k = np.array([R[0, 1] / (2 * k[1]), k[1], R[1, 2] / (2 * k[1])])
        else:
            k = np.array([R[0, 2] / (2 * k[2]), R[1, 2] / (2 * k[2]), k[2]])
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(th)) * th


def _rot_cmd(tcp, gain=1.0):
    Rc = _aa_to_mat(tcp[3:6])
    Rd = _aa_to_mat(DOWN_QUAT_AA)
    err = _mat_to_aa(Rd @ Rc.T)
    return np.clip(err / 0.5 * gain, -0.5, 0.5)


# -------------------------------------------------------------- perception
def _cabinets(frame):
    """Two cabinet tops from the agentview RGB-D frame."""
    P = _cloud(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    rgb = np.asarray(frame.agentview_rgb).astype(int)
    dark = rgb.sum(-1) < 200
    z = P[..., 2]
    sel = dark & (z > 0.5) & (z < 0.9) & (np.abs(P[..., 0]) < 0.4) & (np.abs(P[..., 1]) < 0.7)
    zz = z[sel]
    if len(zz) < 100:
        return None
    h, b = np.histogram(zz, bins=np.arange(0.5, 0.9, 0.004))
    top = float(b[int(np.argmax(h))] + 0.002)
    m = sel & (np.abs(z - top) < 0.012)
    pts = P[m]
    if len(pts) < 100:
        return None
    ys = np.sort(pts[:, 1])
    gaps = np.diff(ys)
    k = int(np.argmax(gaps))
    split = (ys[k] + ys[k + 1]) / 2.0
    cabs = []
    for s in (pts[:, 1] < split, pts[:, 1] >= split):
        q = pts[s]
        if len(q) < 50:
            return None
        lo = np.percentile(q, 2, 0)
        hi = np.percentile(q, 98, 0)
        zt = float(np.median(q[:, 2]))
        cabs.append([float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), zt,
                     float(hi[0] - lo[0]), float(hi[1] - lo[1]), int(len(q))])
    return {"top": top, "cabs": cabs, "gap": float(gaps[k])}


def _classify(rgb):
    r = rgb[..., 0].astype(float)
    g = rgb[..., 1].astype(float)
    b = rgb[..., 2].astype(float)
    butter = (r >= 125) & (b <= 0.3 * r)
    choc = (r >= 30) & (r <= 95) & (g <= 0.5 * r) & (b <= 0.5 * r)
    return butter, choc


def _components(mask, min_size):
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=int)
    comps = []
    vs, us = np.nonzero(mask)
    cur = 0
    for v0, u0 in zip(vs.tolist(), us.tolist()):
        if lab[v0, u0]:
            continue
        cur += 1
        stack = [(v0, u0)]
        lab[v0, u0] = cur
        members = []
        while stack:
            v, u = stack.pop()
            members.append((v, u))
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, c = v + dv, u + du
                if 0 <= a < H and 0 <= c < W and mask[a, c] and not lab[a, c]:
                    lab[a, c] = cur
                    stack.append((a, c))
        if len(members) >= min_size:
            comps.append(np.array(members))
    return comps


def _objects_on(frame, cab):
    """Boxes standing on cabinet `cab` ([x, y, top_z]) seen by the wrist RGB-D."""
    P = _cloud(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    top = float(cab[2])
    z = P[..., 2]
    m = (z > top + 0.008) & (z < top + 0.09)
    m &= (np.abs(P[..., 0] - cab[0]) < 0.15) & (np.abs(P[..., 1] - cab[1]) < 0.115)
    rgb = np.asarray(frame.wrist_rgb)
    butter, choc = _classify(rgb)
    comps = _components(m, 40)
    cands = []
    for c in comps:
        pts = P[c[:, 0], c[:, 1]]
        nb = butter[c[:, 0], c[:, 1]]
        nc = choc[c[:, 0], c[:, 1]]
        fb, fc = float(nb.mean()), float(nc.mean())
        if nb.sum() > 30 and nc.sum() > 30 and fb > 0.07 and fc > 0.07:
            C = np.array([np.median(pts[nb, :2], 0), np.median(pts[nc, :2], 0)])
            a = np.zeros(len(pts), dtype=int)
            for _ in range(10):
                D = ((pts[:, None, :2] - C[None]) ** 2).sum(-1)
                a = np.argmin(D, 1)
                for k in range(2):
                    if (a == k).sum() > 5:
                        C[k] = pts[a == k, :2].mean(0)
            cands.append(("butter", pts[a == 0], fb))
            cands.append(("chocolate", pts[a == 1], fc))
        elif fb > 0.02 or fc > 0.02:
            cands.append(("butter" if fb > fc else "chocolate", pts, max(fb, fc)))
    out = {}
    for name, pts, s in sorted(cands, key=lambda c: -len(c[1])):
        if name in out or len(pts) < 40:
            continue
        out[name] = {"xy": [float(v) for v in pts[:, :2].mean(0)],
                     "top": float(np.percentile(pts[:, 2], 90)), "n": int(len(pts)),
                     "frac": round(float(s), 3)}
    return out


# ------------------------------------------------------------------ control
def _width(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _goto(robot, target, grip, tol=0.006, max_steps=250, vmax=1.0, settle=2):
    """Proportional move of the TCP to world `target` with top-down orientation."""
    target = np.asarray(target, dtype=float)
    good = 0
    n = 0
    for n in range(1, max_steps + 1):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        err = target - tcp[:3]
        if float(np.linalg.norm(err)) < tol:
            good += 1
            if good >= settle:
                break
        else:
            good = 0
        a = np.clip(err * 25.0, -vmax, vmax)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"reached": bool(np.linalg.norm(target - tcp[:3]) < 2 * tol), "steps": n,
            "tcp": [round(float(v), 4) for v in tcp[:3]]}


def _descend(robot, xy, z_stop, grip, max_steps=150):
    """Move down at xy until z <= z_stop or the TCP stalls (contact)."""
    hist = []
    n = 0
    for n in range(1, max_steps + 1):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        hist.append(float(tcp[2]))
        if tcp[2] <= z_stop:
            break
        if len(hist) > 8 and hist[-8] - hist[-1] < 0.0015:
            break
        ex = (xy[0] - tcp[0]) * 25.0
        ey = (xy[1] - tcp[1]) * 25.0
        dz = (z_stop - 0.008 - tcp[2]) * 12.0
        a = np.clip([ex, ey, dz], -0.6, 0.6)
        r = _rot_cmd(tcp)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    return {"z": round(float(tcp[2]), 4), "steps": n}


def _hold(robot, grip, n):
    for _ in range(int(n)):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        r = _rot_cmd(tcp)
        robot.step(0.0, 0.0, 0.0, float(r[0]), float(r[1]), float(r[2]), grip)


# ------------------------------------------------------------------- acts
def act_hover(robot, x, y, z=0.95, grip=-1.0):
    """Rise vertically (if low) then move the TCP to (x, y, z) top-down; returns final TCP and gripper width."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if tcp[2] < z - 0.05:
        _goto(robot, [tcp[0], tcp[1], z], grip, tol=0.02, settle=1)
    res = _goto(robot, [x, y, z], grip, tol=0.006)
    _hold(robot, grip, 6)
    res["width"] = round(_width(robot.now()), 4)
    return res


def act_pick(robot, x, y, top_z=0.663, lift_z=1.0):
    """Open, go above (x, y), descend to the cabinet top (contact or top_z+0.012), close, lift to lift_z.
    Returns finger width after the lift (held when > 0.008)."""
    _goto(robot, [x, y, top_z + 0.14], -1.0, tol=0.006, max_steps=250)
    _hold(robot, -1.0, 4)
    d = _descend(robot, [x, y], top_z + 0.012, -1.0)
    _hold(robot, 1.0, 25)
    w_closed = _width(robot.now())
    tcp = np.asarray(robot.now().tcp, dtype=float)
    _goto(robot, [tcp[0], tcp[1], lift_z], 1.0, tol=0.01, max_steps=200, settle=1)
    _hold(robot, 1.0, 5)
    w = _width(robot.now())
    return {"grasp_z": d["z"], "width_closed": round(w_closed, 4), "width": round(w, 4),
            "held": bool(w > HELD_MIN_WIDTH), "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_place(robot, x, y, top_z=0.663, carry_z=1.0):
    """Carry the held object at carry_z to (x, y), lower until contact with the surface at top_z, open, rise."""
    tcp = np.asarray(robot.now().tcp, dtype=float)
    if tcp[2] < carry_z - 0.03:
        _goto(robot, [tcp[0], tcp[1], carry_z], 1.0, tol=0.01, settle=1)
    _goto(robot, [x, y, carry_z], 1.0, tol=0.008)
    _goto(robot, [x, y, top_z + 0.10], 1.0, tol=0.006)
    w_before = _width(robot.now())
    d = _descend(robot, [x, y], top_z + 0.012, 1.0)
    _hold(robot, -1.0, 15)
    tcp = np.asarray(robot.now().tcp, dtype=float)
    _goto(robot, [tcp[0], tcp[1], top_z + 0.2], -1.0, tol=0.01, max_steps=150, settle=1)
    return {"release_z": d["z"], "width_before": round(w_before, 4),
            "tcp": [round(float(v), 4) for v in robot.now().tcp[:3]]}


def act_wait(robot, n=50):
    """Hold the current pose with the gripper open for n steps."""
    _hold(robot, -1.0, n)
    return {"waited": int(n)}


# ------------------------------------------------------------------ policy
def _mem(memory):
    keys = ["phase", "cab1", "cab2", "chocolate", "butter", "target", "tries", "placed", "notes"]
    out = {k: memory.get(k) for k in keys}
    if out["phase"] is None:
        out["phase"] = "start"
    if out["tries"] is None:
        out["tries"] = 0
    if out["placed"] is None:
        out["placed"] = []
    if out["notes"] is None:
        out["notes"] = ""
    return out


def _held_from(last):
    if not last:
        return None
    s = str(last)
    if '"held": true' in s:
        return True
    if '"held": false' in s:
        return False
    return None


def perceive_decide(obs, goal, memory, last):
    """Phase machine: locate cabinets -> scan cabinet1 -> pick chocolate -> place on
    cabinet2 (+y side) -> rescan -> pick butter -> place on cabinet2 (-y side) -> verify."""
    m = _mem(memory)
    cur = obs.current
    ph = m["phase"]

    if ph == "start" or m["cab1"] is None:
        info = _cabinets(obs.frames[0]) or _cabinets(cur)
        if info is None:
            m["notes"] = "cabinets not found; waiting"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"n": 5}}}
        cabs = sorted(info["cabs"], key=lambda c: c[1])
        # cabinet1 holds the boxes; in every demonstration it is the -y one
        m["cab1"] = [cabs[0][0], cabs[0][1], cabs[0][2]]
        m["cab2"] = [cabs[1][0], cabs[1][1], cabs[1][2]]
        m["phase"] = "scan"
        m["target"] = "chocolate"
        c1 = m["cab1"]
        return {"memory": m, "act": {"tool": "act_hover",
                "args": {"x": c1[0] - 0.04, "y": c1[1], "z": HOVER_Z}}}

    c1, c2 = m["cab1"], m["cab2"]
    order = ["chocolate", "butter"]

    if ph == "scan":
        objs = _objects_on(cur, c1)
        m["chocolate"] = objs["chocolate"]["xy"] + [objs["chocolate"]["top"]] if "chocolate" in objs else None
        m["butter"] = objs["butter"]["xy"] + [objs["butter"]["top"]] if "butter" in objs else None
        todo = [n for n in order if n not in m["placed"]]
        visible = [n for n in todo if m[n] is not None]
        m["notes"] = "scan " + str({k: v.get("n") for k, v in objs.items()})
        if not todo:
            m["phase"] = "done"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"n": 40}}}
        if not visible:
            m["tries"] = int(m["tries"]) + 1
            if m["tries"] > 6:
                m["phase"] = "done"
                return {"memory": m, "act": {"tool": "act_wait", "args": {"n": 40}}}
            # look again from a slightly different, higher viewpoint
            dx = [0.0, 0.03, -0.03, 0.0, 0.0, 0.0][int(m["tries"]) % 6]
            dy = [0.0, 0.0, 0.0, 0.03, -0.03, 0.0][int(m["tries"]) % 6]
            return {"memory": m, "act": {"tool": "act_hover",
                    "args": {"x": c1[0] - 0.04 + dx, "y": c1[1] + dy, "z": HOVER_Z + 0.03}}}
        # after repeated misses on the preferred object, try the other one first
        tgt = visible[0]
        if int(m["tries"]) >= 3 and len(visible) > 1 and m["target"] == visible[0]:
            tgt = visible[1]
            m["tries"] = 0
        m["target"] = tgt
        m["phase"] = "picked"
        p = m[tgt]
        # after a miss, perturb the grasp across the finger-closing axis (world y)
        jit = [0.0, 0.008, -0.008, 0.0][int(m["tries"]) % 4]
        return {"memory": m, "act": {"tool": "act_pick",
                "args": {"x": p[0], "y": p[1] + jit, "top_z": c1[2], "lift_z": CARRY_Z}}}

    if ph == "picked":
        held = _held_from(last)
        if held is None:
            held = _width(cur) > HELD_MIN_WIDTH
        tgt = m["target"]
        if held:
            m["phase"] = "placed"
            dy = CHOC_OFFSET_Y if tgt == "chocolate" else BUTTER_OFFSET_Y
            return {"memory": m, "act": {"tool": "act_place",
                    "args": {"x": c2[0], "y": c2[1] + dy, "top_z": c2[2], "carry_z": CARRY_Z}}}
        m["tries"] = int(m["tries"]) + 1
        m["phase"] = "scan"
        m["notes"] = "grasp of " + str(tgt) + " missed"
        return {"memory": m, "act": {"tool": "act_hover",
                "args": {"x": c1[0] - 0.04, "y": c1[1], "z": HOVER_Z}}}

    if ph == "placed":
        tgt = m["target"]
        if tgt not in m["placed"]:
            m["placed"] = list(m["placed"]) + [tgt]
        m["phase"] = "scan"
        m["tries"] = 0
        return {"memory": m, "act": {"tool": "act_hover",
                "args": {"x": c1[0] - 0.04, "y": c1[1], "z": HOVER_Z}}}

    # done: idle while the checker settles
    return {"memory": m, "act": {"tool": "act_wait", "args": {"n": 60}}}
