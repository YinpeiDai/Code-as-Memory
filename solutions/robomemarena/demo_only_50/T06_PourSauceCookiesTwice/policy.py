"""T06 Pour tomato sauce over cookies twice, then place the bottle in the bowl drainer.

Perception: agentview depth + calibration -> world point cloud -> objects standing on the
table plane (connected components) classified by height: cookies (~1.5 cm), sauce bottle
(~7.5 cm), bowl drainer (~16 cm).  Control: closed-loop world-frame pose servo on the
measured TCP.  Poses below were measured from all 50 assigned expert demonstrations.
"""

# Fixed start orientation (identical in all 50 demos) as robosuite axis-angle.
R0_AA = [3.1403, 0.0008, -0.0892]
# Pour orientation = rotation of R0 about world y by this world axis-angle (demo hold mean).
POUR_ROT_W = [0.001, -0.841, 0.0]
# Grasp TCP = visible bottle centroid + offset; grasp height (world z). Demo residual <=2.5 mm.
GRASP_OFF = [-0.0076, -0.0114]
GRASP_Z = 0.471
# Pour waypoints relative to visible cookies centroid (x, y) with absolute z.
PRE_POUR = [-0.001, 0.0095, 0.773]
HOLD_POUR = [0.039, -0.0545, 0.879]
UPRIGHT = [-0.026, 0.003, 0.712]
# Defaults (demo means) used only if perception fails.
DEF_BOTTLE = [-0.03, -0.08]
DEF_COOKIES = [0.223, -0.328]
DEF_DRAINER = [-0.131, 0.546]
SAFE_Z = 0.78


def _rotm(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _logm(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-8:
        return np.zeros(3)
    s = math.sin(th)
    if s < 1e-6:
        # near pi: fall back to eigen-like axis from diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
    return w * th


def _r_down():
    return _rotm(R0_AA)


def _r_pour():
    return _rotm(POUR_ROT_W) @ _r_down()


def _tilt_deg(frame):
    R = _rotm(np.asarray(frame.tcp, dtype=float)[3:6])
    c = float(R[:, 2] @ _r_down()[:, 2])
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _holding(frame):
    # closed on the bottle: demos show gap ~0.062 m; fully open ~0.08, closed on nothing ~0
    g = _gap(frame)
    return 0.035 < g < 0.072


def _cloud(frame):
    depth = np.asarray(frame.agentview_depth, dtype=float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], -1).reshape(-1, 3)
    Wp = (P - E[:, 3]) @ E[:, :3]
    return Wp.reshape(H, W, 3), depth > 0


def _components(mask):
    H, W = mask.shape
    lab = -np.ones((H, W), dtype=int)
    comps = []
    vs, us = np.nonzero(mask)
    for i in range(len(vs)):
        v0, u0 = int(vs[i]), int(us[i])
        if lab[v0, u0] >= 0:
            continue
        cid = len(comps)
        lab[v0, u0] = cid
        stack = [(v0, u0)]
        members = []
        while stack:
            v, u = stack.pop()
            members.append((v, u))
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = v + dv, u + du
                if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] < 0:
                    lab[a, b] = cid
                    stack.append((a, b))
        comps.append(members)
    return comps


def _scene(frame):
    """Locate bottle, cookies and drainer (world xy) from the agentview depth image."""
    Wp, valid = _cloud(frame)
    z = Wp[..., 2]
    tab = valid & (np.abs(Wp[..., 0]) < 0.6) & (np.abs(Wp[..., 1]) < 0.7) & (z > 0.35) & (z < 0.5)
    tz = float(np.median(z[tab])) if tab.sum() > 100 else 0.431
    m = (valid & (z > tz + 0.008) & (z < tz + 0.3) & (Wp[..., 0] > -0.35)
         & (Wp[..., 0] < 0.6) & (np.abs(Wp[..., 1]) < 0.75))
    objs = []
    for members in _components(m):
        if len(members) < 15:
            continue
        idx = np.array(members)
        P = Wp[idx[:, 0], idx[:, 1]]
        objs.append({"n": len(members), "x": float(P[:, 0].mean()), "y": float(P[:, 1].mean()),
                     "h": float(P[:, 2].max() - tz)})
    tcp = np.asarray(frame.tcp, dtype=float)
    bottles = [o for o in objs if 0.04 < o["h"] < 0.12 and o["y"] < 0.36]
    cookies = [o for o in objs if o["h"] < 0.04 and o["n"] >= 30]
    drainers = [o for o in objs if 0.12 <= o["h"] < 0.24 and o["n"] >= 300]
    out = {"table_z": tz, "n_objects": len(objs)}
    if bottles:
        b = max(bottles, key=lambda o: o["n"])
        out["bottle"] = [b["x"], b["y"]]
    if cookies:
        c = min(cookies, key=lambda o: (o["x"] - DEF_COOKIES[0]) ** 2 + (o["y"] - DEF_COOKIES[1]) ** 2)
        if (c["x"] - DEF_COOKIES[0]) ** 2 + (c["y"] - DEF_COOKIES[1]) ** 2 < 0.25 ** 2:
            out["cookies"] = [c["x"], c["y"]]
    if drainers:
        d = max(drainers, key=lambda o: o["n"])
        out["drainer"] = [d["x"], d["y"]]
    return out


def _servo(robot, p_t, R_t, grip, max_steps, tol_p=0.004, tol_r=0.03, vmax=1.0, rmax=0.8,
           kp=14.0, kr=3.0):
    """Closed-loop pose servo; returns (steps, final position error, final rotation error)."""
    p_t = np.asarray(p_t, dtype=float)
    integ = np.zeros(3)
    n = 0
    ep = np.zeros(3)
    er = np.zeros(3)
    for n in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        ep = p_t - tcp[:3]
        er = _logm(R_t @ _rotm(tcp[3:6]).T)
        if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
            break
        if np.linalg.norm(ep) < 0.02:
            integ = np.clip(integ + ep * 0.6, -0.15, 0.15)
        a = np.clip(ep * kp + integ, -vmax, vmax)
        r = np.clip(er * kr, -rmax, rmax)
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
    return n, float(np.linalg.norm(ep)), float(np.linalg.norm(er))


def _hold(robot, p_t, R_t, grip, steps):
    for _ in range(int(steps)):
        _servo(robot, p_t, R_t, grip, 1, tol_p=0.0, tol_r=0.0)


def _pos(robot):
    return [float(v) for v in np.asarray(robot.now().tcp, dtype=float)[:3]]


def perceive_decide(obs, goal, memory, last):
    """Phase machine: perceive scene -> grasp bottle -> pour over cookies (repeat) -> drainer -> hold."""
    cur = obs.current
    mem = dict(memory)
    phase = mem.get("phase") or "start"
    attempts = int(mem.get("grasp_attempts") or 0)
    pours = int(mem.get("pours_done") or 0)
    notes = mem.get("notes") or ""
    cookies = mem.get("cookies") or DEF_COOKIES
    drainer = mem.get("drainer") or DEF_DRAINER

    def out(tool, args, **upd):
        m = {"phase": phase, "bottle": mem.get("bottle"), "cookies": cookies, "drainer": drainer,
             "grasp_attempts": attempts, "pours_done": pours, "notes": notes[-600:]}
        m.update(upd)
        return {"memory": m, "act": {"tool": tool, "args": args}}

    if phase == "start" or (phase == "grasp" and not _holding(cur)):
        if phase == "grasp" and attempts >= 6:
            return out("act_hold", {"steps": 100, "grip": -1.0}, phase="grasp")
        sc = _scene(cur)
        bottle = sc.get("bottle") or mem.get("bottle") or DEF_BOTTLE
        if phase == "start":
            cookies = sc.get("cookies") or DEF_COOKIES
            drainer = sc.get("drainer") or DEF_DRAINER
        notes += " | scene%d b=%s c=%s d=%s" % (attempts, [round(v, 3) for v in bottle],
                                                 [round(v, 3) for v in cookies], [round(v, 3) for v in drainer])
        attempts += 1
        phase = "grasp"
        return out("act_grasp", {"x": bottle[0] + GRASP_OFF[0], "y": bottle[1] + GRASP_OFF[1]},
                   phase="grasp", bottle=bottle, grasp_attempts=attempts, notes=notes)

    if phase in ("grasp", "pour"):
        if not _holding(cur):
            # dropped during a pour: re-perceive and regrasp
            phase = "grasp"
            notes += " | dropped after %d pours" % pours
            sc = _scene(cur)
            bottle = sc.get("bottle") or mem.get("bottle") or DEF_BOTTLE
            attempts += 1
            return out("act_grasp", {"x": bottle[0] + GRASP_OFF[0], "y": bottle[1] + GRASP_OFF[1]},
                       phase="grasp", bottle=bottle, grasp_attempts=attempts, notes=notes)
        if pours < 3:
            hold = 50 if pours == 0 else 110
            pours += 1
            return out("act_pour", {"cx": cookies[0], "cy": cookies[1], "hold_steps": hold,
                                    "upright_steps": 30}, phase="pour", pours_done=pours)
        phase = "place"
        return out("act_place", {"x": drainer[0], "y": drainer[1]}, phase="done")

    return out("act_hold", {"steps": 200, "grip": -1.0}, phase="done")


def act_grasp(robot, x, y):
    """Open, move above (x, y), descend to the grasp height, close, lift to safe height.
    Returns finger gap and whether the bottle is held; retreats out of view if not held."""
    Rd = _r_down()
    p = _pos(robot)
    if p[2] < 0.6:
        _servo(robot, [p[0], p[1], 0.66], Rd, -1.0, 60, tol_p=0.01)
    _servo(robot, [x, y, 0.62], Rd, -1.0, 150, tol_p=0.005)
    _servo(robot, [x, y, GRASP_Z], Rd, -1.0, 150, tol_p=0.003, vmax=0.5)
    for _ in range(30):
        _servo(robot, [x, y, GRASP_Z], Rd, 1.0, 1, tol_p=0.0, tol_r=0.0)
    p = _pos(robot)
    _servo(robot, [p[0], p[1], SAFE_Z], Rd, 1.0, 120, tol_p=0.01, vmax=0.8)
    held = _holding(robot.now())
    if not held:
        _servo(robot, [x, y, SAFE_Z], Rd, -1.0, 15, tol_p=0.01)
        _servo(robot, [-0.2, 0.0, SAFE_Z], Rd, -1.0, 120, tol_p=0.01)
    return {"holding": held, "gap": _gap(robot.now()), "tcp": _pos(robot)}


def act_pour(robot, cx, cy, hold_steps=50, upright_steps=30):
    """One pour over the cookies at (cx, cy): pre-pour pose, tilt ~48 deg about world y to the
    demonstrated hold pose, hold, return upright and rest. Returns tilt statistics."""
    Rd, Rp = _r_down(), _r_pour()
    pre = [cx + PRE_POUR[0], cy + PRE_POUR[1], PRE_POUR[2]]
    hold = [cx + HOLD_POUR[0], cy + HOLD_POUR[1], HOLD_POUR[2]]
    up = [cx + UPRIGHT[0], cy + UPRIGHT[1], UPRIGHT[2]]
    p = _pos(robot)
    if p[2] < SAFE_Z - 0.03 and math.hypot(p[0] - pre[0], p[1] - pre[1]) > 0.05:
        _servo(robot, [p[0], p[1], SAFE_Z], Rd, 1.0, 80, tol_p=0.01)
    _servo(robot, pre, Rd, 1.0, 150, tol_p=0.006)
    n_tilt, e_p, e_r = _servo(robot, hold, Rp, 1.0, 150, tol_p=0.006, tol_r=0.03, rmax=0.8)
    max_tilt = _tilt_deg(robot.now())
    _hold(robot, hold, Rp, 1.0, hold_steps)
    max_tilt = max(max_tilt, _tilt_deg(robot.now()))
    _servo(robot, up, Rd, 1.0, 150, tol_p=0.006, tol_r=0.03)
    _hold(robot, up, Rd, 1.0, upright_steps)
    return {"holding": _holding(robot.now()), "tilt_steps": n_tilt, "hold_err_p": e_p,
            "hold_err_r": e_r, "max_tilt_deg": max_tilt, "end_tilt_deg": _tilt_deg(robot.now())}


def act_place(robot, x, y):
    """Carry the bottle above the drainer centre (x, y), lower it just above the rim, release, lift."""
    Rd = _r_down()
    p = _pos(robot)
    _servo(robot, [p[0], p[1], SAFE_Z], Rd, 1.0, 80, tol_p=0.01)
    tx, ty = x - 0.005, y - 0.05
    _servo(robot, [tx, ty, SAFE_Z], Rd, 1.0, 200, tol_p=0.006)
    _servo(robot, [tx, ty, 0.655], Rd, 1.0, 150, tol_p=0.004, vmax=0.4)
    _hold(robot, [tx, ty, 0.655], Rd, -1.0, 25)
    _servo(robot, [tx, ty, SAFE_Z + 0.05], Rd, -1.0, 100, tol_p=0.01)
    return {"released_at": [tx, ty], "gap": _gap(robot.now()), "tcp": _pos(robot)}


def act_hold(robot, steps=100, grip=-1.0):
    """Stay at the current pose for `steps` steps (monitoring period); gripper sign as given."""
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    _hold(robot, tcp[:3], _rotm(tcp[3:6]), grip, steps)
    return {"tcp": _pos(robot), "gap": _gap(robot.now())}
