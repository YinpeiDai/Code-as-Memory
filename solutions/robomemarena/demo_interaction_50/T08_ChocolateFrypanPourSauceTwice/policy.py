"""Policy for: pick chocolate into frypan, pour tomato sauce twice, put sauce bottle into bowl drainer.

Perception: agentview depth back-projected to world points (calibrated K and extrinsics).
Control: closed-loop P servo on measured TCP position/orientation (world frame OSC deltas).
"""

TABLE_Z = 0.430
# nominal object locations seen in all 50 demos (used only as search priors / fallbacks)
NOM_CHOC = [-0.119, -0.174]
NOM_SAUCE = [0.087, -0.175]
NOM_PAN = [0.005, 0.128]
NOM_DRAINER = [-0.11, 0.54]
# learned offsets: expert grasp/drop TCP minus perceived feature (mean over 50 demos)
OFF_CHOC = [-0.004, 0.000]
OFF_SAUCE = [-0.007, -0.012]
OFF_PAN_DROP = [-0.006, -0.016]
OFF_POUR = [-0.006, -0.041]  # pour start relative to fitted pan centre
Z_CHOC_GRASP = 0.438
Z_SAUCE_GRASP = 0.468
Z_HOVER = 0.670
Z_CHOC_LIFT = 0.790
Z_TRANSIT = 0.830
Z_CHOC_RELEASE = 0.643
Z_AFTER_RELEASE = 0.817
Z_SAUCE_LIFT = 0.786
Z_POUR = 0.724
TILT_DEG = 45.0


# ---------------------------------------------------------------- math helpers
def _aa2R(v):
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _R2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _R2aa(R):
    q = _R2quat(R)
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-9:
        return np.zeros(3)
    th = 2 * math.atan2(s, q[0])
    return q[1:] / s * th


def _R_down():
    return np.diag([1.0, -1.0, -1.0])


def _R_tilt(deg):
    th = math.radians(-deg)
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    return Ry @ _R_down()


def _ori_err(tcp, Rt):
    Rc = _aa2R(tcp[3:6])
    return _R2aa(Rt @ Rc.T)


def _gap(frame):
    f = frame.fingers
    return float(f[0] - f[1])


# ---------------------------------------------------------------- perception
def _cloud(frame):
    K = frame.agentview_K
    E = frame.agentview_extrinsic
    d = frame.agentview_depth.astype(float) / 1000.0
    v, u = np.mgrid[0:256, 0:256]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)
    w = (pc - E[:, 3]) @ E[:, :3]
    return w, d


def _components(mask):
    lab = np.zeros(mask.shape, int)
    comps = []
    rs, cs = np.nonzero(mask)
    H, W = mask.shape
    for r0, c0 in zip(rs.tolist(), cs.tolist()):
        if lab[r0, c0]:
            continue
        n = len(comps) + 1
        lab[r0, c0] = n
        st = [(r0, c0)]
        pix = []
        while st:
            r, c = st.pop()
            pix.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    st.append((rr, cc))
        comps.append(pix)
    return comps


def _circfit(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    c = s[:2]
    r2 = s[2] + c @ c
    return c, math.sqrt(max(r2, 1e-9))


def _find_pan(w, d, rgb):
    z = w[..., 2]
    m = (rgb.max(2) < 90) & (d > 0) & (z > 0.43) & (z < 0.5) & (w[..., 0] > -0.25) & (w[..., 0] < 0.25) \
        & (w[..., 1] > -0.1) & (w[..., 1] < 0.4)
    xy = w[m][:, :2]
    if len(xy) < 40:
        return None
    keep = np.ones(len(xy), bool)
    c, r = None, None
    for _ in range(8):
        if keep.sum() < 20:
            return None
        c, r = _circfit(xy[keep])
        res = np.abs(np.linalg.norm(xy - c, axis=1) - r)
        keep = res < max(0.012, 2.5 * float(np.median(res[keep])))
    if not (0.07 < r < 0.12) or keep.sum() < 60:
        return None
    return [float(c[0]), float(c[1]), float(r)]


def _find_blob(w, d, prior, zmin, zmax, ztop_min, ztop_max, radius=0.10, exclude=None):
    """Connected above-table blob nearest a prior xy; returns [x, y, ztop, npts] or None."""
    z = w[..., 2]
    m = (d > 0) & (z > zmin) & (z < zmax) & (np.abs(w[..., 0] - prior[0]) < radius) \
        & (np.abs(w[..., 1] - prior[1]) < radius)
    if exclude is not None:
        m = m & (np.hypot(w[..., 0] - exclude[0], w[..., 1] - exclude[1]) > exclude[2])
    best = None
    for pix in _components(m):
        if len(pix) < 15:
            continue
        rr = np.array([p[0] for p in pix])
        cc = np.array([p[1] for p in pix])
        p = w[rr, cc]
        top = float(np.percentile(p[:, 2], 95))
        if not (ztop_min <= top <= ztop_max):
            continue
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        if ext.max() > 0.12:
            continue
        mx, my = float(p[:, 0].mean()), float(p[:, 1].mean())
        dist = math.hypot(mx - prior[0], my - prior[1])
        if best is None or dist < best[0]:
            best = (dist, [mx, my, top, len(pix)])
    return None if best is None else best[1]


def _find_drainer(w, d):
    z = w[..., 2]
    m = (d > 0) & (z > 0.55) & (z < 0.65) & (w[..., 0] > -0.4) & (w[..., 0] < 0.2) \
        & (w[..., 1] > 0.3) & (w[..., 1] < 0.8)
    p = w[m]
    if len(p) < 50:
        return None
    lo = np.percentile(p[:, :2], 3, axis=0)
    hi = np.percentile(p[:, :2], 97, axis=0)
    return [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(hi[0] - lo[0]), float(hi[1] - lo[1]),
            float(np.percentile(p[:, 2], 95))]


def _scene(frame, pan_hint=None):
    w, d = _cloud(frame)
    rgb = frame.agentview_rgb
    tcp = frame.tcp
    # exclude the gripper/arm vicinity
    excl = [float(tcp[0]), float(tcp[1]), 0.07]
    pan = _find_pan(w, d, rgb)
    choc = _find_blob(w, d, NOM_CHOC, 0.437, 0.6, 0.443, 0.49, 0.10, excl)
    sauce = _find_blob(w, d, NOM_SAUCE, 0.437, 0.6, 0.485, 0.56, 0.10, excl)
    drainer = _find_drainer(w, d)
    pc = pan if pan is not None else pan_hint
    inpan = None
    if pc is not None:
        rr = np.hypot(w[..., 0] - pc[0], w[..., 1] - pc[1])
        mm = (rr < 0.06) & (d > 0) & (w[..., 2] > 0.452) & (w[..., 2] < 0.52)
        inpan = int(mm.sum())
    return {"pan": pan, "choc": choc, "sauce": sauce, "drainer": drainer, "inpan": inpan}


# ---------------------------------------------------------------- control helpers
def _servo_cmd(tcp, target, Rt, kp=10.0, vmax=1.0, kr=2.0, rmax=0.8):
    e = np.asarray(target, float) - tcp[:3]
    a = np.clip(e * kp, -vmax, vmax)
    er = _ori_err(tcp, Rt)
    r = np.clip(er * kr, -rmax, rmax)
    return a, r, float(np.linalg.norm(e)), float(np.linalg.norm(er))


def _move(robot, target, grip, Rt=None, tol=0.006, rtol=0.05, max_steps=200, vmax=1.0, kp=10.0, settle=0):
    if Rt is None:
        Rt = _R_down()
    n = 0
    dist, rerr = 1.0, 1.0
    calm = 0
    while n < max_steps:
        tcp = robot.now().tcp
        a, r, dist, rerr = _servo_cmd(tcp, target, Rt, kp, vmax)
        if dist < tol and rerr < rtol:
            calm += 1
            if calm > settle:
                break
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    return {"steps": n, "dist": round(dist, 4), "rot_err": round(rerr, 3)}


def _gripper(robot, grip, min_steps=10, max_steps=35):
    n = 0
    prev = _gap(robot.now())
    still = 0
    while n < max_steps:
        tcp = robot.now().tcp
        a, r, _, _ = _servo_cmd(tcp, tcp[:3], _R_down(), kp=0.0)
        robot.step(0, 0, 0, float(r[0]) * 0.0, 0, 0, grip)
        n += 1
        g = _gap(robot.now())
        still = still + 1 if abs(g - prev) < 0.0005 else 0
        prev = g
        if n >= min_steps and still >= 4:
            break
    return n


def _rotate(robot, Rt, grip, rtol=0.03, max_steps=45, rmax=0.8):
    n = 0
    rerr = 1.0
    while n < max_steps:
        tcp = robot.now().tcp
        er = _ori_err(tcp, Rt)
        rerr = float(np.linalg.norm(er))
        if rerr < rtol:
            break
        r = np.clip(er * 2.0, -rmax, rmax)
        robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    return n, rerr


# ---------------------------------------------------------------- act tools
def act_grasp(robot, x, y, z_grasp, z_hover, z_lift):
    """Open gripper, move above (x,y) at z_hover, descend to z_grasp, close, lift to z_lift.
    Returns measured finger gap before/after and whether something is held after the lift."""
    out = {}
    out["hover"] = _move(robot, [x, y, z_hover], -1, tol=0.01, max_steps=200)
    out["descend"] = _move(robot, [x, y, z_grasp], -1, tol=0.004, max_steps=200, vmax=0.6, settle=3)
    out["gap_open"] = round(_gap(robot.now()), 4)
    out["close_steps"] = _gripper(robot, 1, min_steps=15, max_steps=40)
    out["gap_closed"] = round(_gap(robot.now()), 4)
    out["lift"] = _move(robot, [x, y, z_lift], 1, tol=0.01, max_steps=200, vmax=1.0)
    g = _gap(robot.now())
    out["gap_lifted"] = round(g, 4)
    out["held"] = bool(0.008 < g < 0.07)
    out["tcp"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    return out


def act_place(robot, x, y, z_transit, z_release, z_after):
    """Carry the held object at z_transit to above (x,y), descend to z_release, open, lift to z_after.
    Aborts (dropped=True) if the finger gap collapses during transport."""
    out = {"dropped": False}
    tcp = robot.now().tcp
    if tcp[2] < z_transit - 0.02:
        out["up"] = _move(robot, [float(tcp[0]), float(tcp[1]), z_transit], 1, tol=0.015, max_steps=120)
    if _gap(robot.now()) < 0.006:
        out["dropped"] = True
        return out
    out["carry"] = _move(robot, [x, y, z_transit], 1, tol=0.01, max_steps=200)
    out["descend"] = _move(robot, [x, y, z_release], 1, tol=0.006, max_steps=150, vmax=0.8, settle=2)
    out["gap_before_open"] = round(_gap(robot.now()), 4)
    if out["gap_before_open"] < 0.006:
        out["dropped"] = True
    out["open_steps"] = _gripper(robot, -1, min_steps=12, max_steps=30)
    out["lift"] = _move(robot, [x, y, z_after], -1, tol=0.012, max_steps=120)
    out["tcp"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    return out


def act_carry(robot, x, y, z, grip=1):
    """Move the TCP (gripper down) to (x,y,z) holding the gripper command; reports gap."""
    out = _move(robot, [x, y, z], grip, tol=0.008, max_steps=250)
    out["gap"] = round(_gap(robot.now()), 4)
    out["tcp"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    return out


def act_pour(robot, x, y, z, hold_steps=35, tilt_deg=45.0, back=True):
    """Pour once: go to pour start (x,y,z) upright, tilt the held bottle by tilt_deg about world -y
    (zero translation command, as in the demos), hold, then return upright if back."""
    out = {}
    out["approach"] = _move(robot, [x, y, z], 1, tol=0.012, max_steps=200)
    n, e = _rotate(robot, _R_tilt(tilt_deg), 1, rtol=0.03, max_steps=45)
    out["tilt_steps"], out["tilt_err"] = n, round(e, 3)
    held = 0
    for _ in range(hold_steps):
        robot.step(0, 0, 0, 0, 0, 0, 1)
        held += 1
    out["held"] = held
    out["gap_tilted"] = round(_gap(robot.now()), 4)
    if back:
        n, e = _rotate(robot, _R_down(), 1, rtol=0.05, max_steps=45)
        out["back_steps"], out["back_err"] = n, round(e, 3)
    out["gap"] = round(_gap(robot.now()), 4)
    out["tcp"] = [round(float(v), 4) for v in robot.now().tcp[:6]]
    return out


def act_wait(robot, steps=20, grip=0):
    """Hold position (zero motion) for a number of steps with a gripper command."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, grip)
    return {"waited": int(steps), "gap": round(_gap(robot.now()), 4)}


# ---------------------------------------------------------------- decision
def _xy(v, default):
    if v is None:
        return list(default)
    return [float(v[0]), float(v[1])]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive scene -> grasp chocolate -> place in frypan -> grasp sauce bottle ->
    pour twice over the pan -> place bottle in bowl drainer -> wait."""
    mem = dict(memory)
    for k in ("stage", "pan", "choc", "sauce", "drainer", "pours", "tries", "note"):
        mem.setdefault(k, None)
    cur = obs.current
    gap = _gap(cur)
    tcp = cur.tcp
    stage = mem["stage"]
    pours = int(mem["pours"] or 0)
    tries = int(mem["tries"] or 0)
    note = ""

    if stage is None:
        sc = _scene(cur)
        p = sc["pan"]
        mem["pan"] = [p[0], p[1]] if p is not None else list(NOM_PAN)
        mem["choc"] = [sc["choc"][0], sc["choc"][1]] if sc["choc"] is not None else list(NOM_CHOC)
        mem["sauce"] = [sc["sauce"][0], sc["sauce"][1]] if sc["sauce"] is not None else list(NOM_SAUCE)
        dr = sc["drainer"]
        mem["drainer"] = [dr[0], dr[1]] if dr is not None else list(NOM_DRAINER)
        note = "scene pan=%s choc=%s sauce=%s drainer=%s" % (p is not None, sc["choc"] is not None,
                                                             sc["sauce"] is not None, dr is not None)
        stage = "grasp_choc"
        tries = 0
    pan = _xy(mem["pan"], NOM_PAN)
    choc = _xy(mem["choc"], NOM_CHOC)
    sauce = _xy(mem["sauce"], NOM_SAUCE)
    drainer = _xy(mem["drainer"], NOM_DRAINER)

    # ---- outcome checks for the previous stage (measured from the latest frame)
    held = 0.008 < gap < 0.07  # closed on an object (fully open is ~0.08, empty closed ~0)
    if stage == "verify_choc_grasp":
        if held:
            stage = "place_choc"
        else:
            tries += 1
            sc = _scene(cur)
            if sc["choc"] is not None:
                choc = [sc["choc"][0], sc["choc"][1]]
            note = "choc grasp missed (gap %.4f), retry %d" % (gap, tries)
            stage = "grasp_choc" if tries < 3 else "grasp_sauce"
    elif stage == "verify_choc_place":
        sc = _scene(cur, pan + [0.09])
        ok_pan = sc["inpan"] is not None and sc["inpan"] > 20
        if ok_pan or tries >= 2:
            stage = "grasp_sauce"
            tries = 0
            note = "choc in pan pts=%s" % sc["inpan"]
        else:
            if sc["choc"] is not None:
                choc = [sc["choc"][0], sc["choc"][1]]
                tries += 1
                stage = "grasp_choc"
                note = "choc not in pan (pts=%s); re-grasp" % sc["inpan"]
            else:
                stage = "grasp_sauce"
                tries = 0
                note = "choc not seen in pan (pts=%s) nor at origin; continue" % sc["inpan"]
    elif stage == "verify_sauce_grasp":
        if held:
            stage = "pour"
            tries = 0
        else:
            tries += 1
            sc = _scene(cur)
            if sc["sauce"] is not None:
                sauce = [sc["sauce"][0], sc["sauce"][1]]
            note = "sauce grasp missed (gap %.4f), retry %d" % (gap, tries)
            stage = "grasp_sauce"
    elif stage == "verify_pour":
        if held:
            pours += 1
            stage = "pour" if pours < 2 else "to_drainer"
        else:
            note = "bottle dropped during pour"
            tries += 1
            stage = "grasp_sauce_again"

    # ---- choose the action
    if stage == "grasp_choc":
        act = {"tool": "act_grasp", "args": {"x": choc[0] + OFF_CHOC[0], "y": choc[1] + OFF_CHOC[1],
                                             "z_grasp": Z_CHOC_GRASP, "z_hover": Z_HOVER, "z_lift": Z_CHOC_LIFT}}
        nxt = "verify_choc_grasp"
    elif stage == "place_choc":
        act = {"tool": "act_place", "args": {"x": pan[0] + OFF_PAN_DROP[0], "y": pan[1] + OFF_PAN_DROP[1],
                                             "z_transit": Z_TRANSIT, "z_release": Z_CHOC_RELEASE,
                                             "z_after": Z_AFTER_RELEASE}}
        nxt = "verify_choc_place"
    elif stage in ("grasp_sauce", "grasp_sauce_again"):
        if stage == "grasp_sauce_again":
            sc = _scene(cur)
            if sc["sauce"] is not None:
                sauce = [sc["sauce"][0], sc["sauce"][1]]
        if tries >= 4:
            act = {"tool": "act_wait", "args": {"steps": 50, "grip": -1}}
            nxt = "give_up"
        else:
            act = {"tool": "act_grasp", "args": {"x": sauce[0] + OFF_SAUCE[0], "y": sauce[1] + OFF_SAUCE[1],
                                                 "z_grasp": Z_SAUCE_GRASP, "z_hover": Z_HOVER,
                                                 "z_lift": Z_SAUCE_LIFT}}
            nxt = "verify_sauce_grasp"
    elif stage == "pour":
        last_pour = pours >= 1
        act = {"tool": "act_pour", "args": {"x": pan[0] + OFF_POUR[0], "y": pan[1] + OFF_POUR[1], "z": Z_POUR,
                                            "hold_steps": 80 if last_pour else 42, "tilt_deg": TILT_DEG,
                                            "back": True}}
        nxt = "verify_pour"
    elif stage == "to_drainer":
        act = {"tool": "act_place", "args": {"x": drainer[0], "y": drainer[1] - 0.06, "z_transit": 0.80,
                                             "z_release": 0.66, "z_after": 0.80}}
        nxt = "done"
    else:
        act = {"tool": "act_wait", "args": {"steps": 40, "grip": -1}}
        nxt = stage if stage in ("done", "give_up") else "done"

    mem["stage"] = nxt
    mem["pan"] = pan
    mem["choc"] = choc
    mem["sauce"] = sauce
    mem["drainer"] = drainer
    mem["pours"] = pours
    mem["tries"] = tries
    mem["note"] = note
    return {"memory": mem, "act": act}
