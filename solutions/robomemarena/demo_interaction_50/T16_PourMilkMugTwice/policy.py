"""T16_PourMilkMugTwice policy: perceive milk / mug / drainer from agentview RGB-D,
grasp the milk, pour it over the mug twice (tilt ~60 deg about world -y, hold, untilt),
then place the milk in the bowl drainer if the episode has not already ended."""

TABLE_Z_MIN = 0.445          # points above this (world z, m) are object candidates
MILK_PRIOR = [-0.083, -0.174]  # demo mean of the visible milk centroid (fallback / assignment prior)
MUG_PRIOR = [0.181, -0.031]
DRAINER_PRIOR = [-0.132, 0.544]
GRASP_OFF = [-0.0135, -0.0008]   # grasp tcp - visible milk centroid (demos: std 1 mm)
GRASP_BELOW_TOP = 0.035          # grasp tcp z = milk top(95 pct) - this
POUR1_OFF = [-0.031, -0.021, 0.687]  # pour-1 start tcp xy = mug centroid + off, absolute z
POUR2_OFF = [0.002, -0.026, 0.698]   # pour-2 start tcp
POUR_TILT = 1.056                # rad (60.5 deg) about world -y
LIFT_Z = 0.82
HOME = [-0.06, 0.0, 0.68]


def _aa2mat(aa):
    aa = np.asarray(aa, dtype=float)
    th = float(np.linalg.norm(aa))
    if th < 1e-9:
        return np.eye(3)
    k = aa / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the diagonal
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(d))
        ax = np.zeros(3)
        ax[i] = d[i]
        for j in range(3):
            if j != i:
                ax[j] = (R[i, j] + R[j, i]) / (4.0 * d[i])
        return ax / np.linalg.norm(ax) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _down():
    return np.diag([1.0, -1.0, -1.0])


def _pour_target():
    return _rot_y(-POUR_TILT) @ _down()


def _tilt_deg(aa):
    R = _aa2mat(aa)
    return math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))


def _ori_err(frame, Rt):
    R = _aa2mat(frame.tcp[3:6])
    return _mat2aa(Rt @ R.T)


def _width(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def _backproject(depth, K, ext):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], -1)
    R = ext[:, :3]
    t = ext[:, 3]
    return (pc - t) @ R, depth > 0


def _components(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=int)
    n = 0
    rs, cs = np.nonzero(mask)
    for r0, c0 in zip(rs.tolist(), cs.tolist()):
        if lab[r0, c0]:
            continue
        n += 1
        stack = [(r0, c0)]
        lab[r0, c0] = n
        while stack:
            r, c = stack.pop()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    stack.append((rr, cc))
    return lab, n


def _clusters(frame, gripper_xyz=None):
    w, ok = _backproject(frame.agentview_depth, frame.agentview_K, frame.agentview_extrinsic)
    z = w[..., 2]
    m = ok & (z > TABLE_Z_MIN) & (z < 0.65) & (np.abs(w[..., 0]) < 0.45) & (w[..., 1] > -0.5) & (w[..., 1] < 0.75)
    if gripper_xyz is not None:
        g = np.asarray(gripper_xyz, dtype=float)
        # remove the hand/arm: points near or above the gripper, within a column around it
        dxy = np.hypot(w[..., 0] - g[0], w[..., 1] - g[1])
        m = m & ~((dxy < 0.09) & (z > g[2] - 0.12))
    lab, n = _components(m)
    out = []
    for i in range(1, n + 1):
        s = lab == i
        cnt = int(s.sum())
        if cnt < 20:
            continue
        p = w[s]
        top = float(np.percentile(p[:, 2], 95))
        rim = p[p[:, 2] > top - 0.012]
        out.append({"n": cnt, "cen": p.mean(0), "top": top,
                    "lo": p[:, :2].min(0), "hi": p[:, :2].max(0),
                    "rim_lo": rim[:, :2].min(0), "rim_hi": rim[:, :2].max(0)})
    return out


def _detect(frame, gripper_xyz=None):
    """Assign clusters to drainer / milk / mug using size and the demo layout prior."""
    cl = _clusters(frame, gripper_xyz)
    res = {"milk": None, "mug": None, "drainer": None, "n_clusters": len(cl)}
    big = [c for c in cl if c["n"] > 700]
    if big:
        d = min(big, key=lambda c: np.hypot(c["cen"][0] - DRAINER_PRIOR[0], c["cen"][1] - DRAINER_PRIOR[1]))
        if np.hypot(d["cen"][0] - DRAINER_PRIOR[0], d["cen"][1] - DRAINER_PRIOR[1]) < 0.2:
            res["drainer"] = d
    rest = [c for c in cl if c is not res["drainer"] and c["n"] < 700]
    # milk: small (<250 px) cluster closest to the milk prior; mug: larger (>200 px) closest to mug prior
    mk = [c for c in rest if c["n"] <= 260 and c["top"] < 0.60 and np.hypot(c["cen"][0] - MILK_PRIOR[0], c["cen"][1] - MILK_PRIOR[1]) < 0.15]
    if mk:
        res["milk"] = min(mk, key=lambda c: np.hypot(c["cen"][0] - MILK_PRIOR[0], c["cen"][1] - MILK_PRIOR[1]))
    mg = [c for c in rest if c["n"] >= 200 and c is not res["milk"] and np.hypot(c["cen"][0] - MUG_PRIOR[0], c["cen"][1] - MUG_PRIOR[1]) < 0.15]
    if mg:
        res["mug"] = min(mg, key=lambda c: np.hypot(c["cen"][0] - MUG_PRIOR[0], c["cen"][1] - MUG_PRIOR[1]))
    return res


RETRY_ADJ = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.006, 0.0, 0.002], [-0.006, 0.0, 0.002]]


def _grasp_target(m, dz):
    """Grasp tcp from the milk estimate; later retries shift the grasp (lower / along x)."""
    milk = m["milk"]
    k = min(int(m["grasp_tries"] or 0), len(RETRY_ADJ) - 1)
    adj = RETRY_ADJ[k]
    gz = max(0.455, min(0.55, milk[2] - GRASP_BELOW_TOP)) + adj[2]
    return [milk[0] + GRASP_OFF[0] + adj[0], milk[1] + GRASP_OFF[1] + adj[1], gz + dz]


DRAINER_SLOT_PRIOR = [-0.16, 0.478]


def _drainer_slot(c):
    """Centre of the drainer compartment nearer the robot (-y side) from the top-rim box, and rim z.
    Rollout 12: rim box x[-0.26,-0.06] y[0.42,0.65], central divider at y~0.535, slots ~8 cm wide."""
    lo, hi = c["rim_lo"], c["rim_hi"]
    cx = 0.5 * (float(lo[0]) + float(hi[0]))
    cy = float(lo[1]) + 0.25 * (float(hi[1]) - float(lo[1]))
    if abs(cx - DRAINER_SLOT_PRIOR[0]) > 0.08 or abs(cy - DRAINER_SLOT_PRIOR[1]) > 0.08:
        cx, cy = DRAINER_SLOT_PRIOR[0], DRAINER_SLOT_PRIOR[1]
    return [round(cx, 4), round(cy, 4), round(float(c["top"]), 4)]


def _p(c):
    return [round(float(c["cen"][0]), 4), round(float(c["cen"][1]), 4), round(float(c["top"]), 4)]


MEMORY_FIELDS = ("stage", "milk", "mug", "drainer", "grasp_tries", "pours_done",
                 "pour_hold_steps", "perception_ok", "notes")


def _mem(memory):
    m = {}
    for k in MEMORY_FIELDS:
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["grasp_tries"] is None:
        m["grasp_tries"] = 0
    if m["pours_done"] is None:
        m["pours_done"] = 0
    if m["notes"] is None:
        m["notes"] = ""
    return m


def _note(m, s):
    m["notes"] = (str(m["notes"]) + " | " + s)[-600:]


def _out(m, tool, args):
    return {"memory": m, "act": {"tool": tool, "args": args}}


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive -> approach -> descend -> close -> lift -> pour1 (move, tilt, hold, untilt)
    -> pour2 -> drainer placement. Checks finger width for grasp / hold, tcp for motion results."""
    m = _mem(memory)
    f = obs.current
    tcp = [float(v) for v in f.tcp[:3]]
    width = _width(f)
    stage = m["stage"]

    if stage is None or stage == "perceive":
        det = _detect(f, tcp if stage == "perceive" else tcp)
        ok = det["milk"] is not None and det["mug"] is not None
        m["perception_ok"] = bool(ok)
        if det["milk"] is not None:
            m["milk"] = _p(det["milk"])
        elif m["milk"] is None:
            m["milk"] = [MILK_PRIOR[0], MILK_PRIOR[1], 0.557]
            _note(m, "milk not detected; using prior")
        if det["mug"] is not None:
            m["mug"] = _p(det["mug"])
        elif m["mug"] is None:
            m["mug"] = [MUG_PRIOR[0], MUG_PRIOR[1], 0.56]
            _note(m, "mug not detected; using prior")
        if det["drainer"] is not None:
            m["drainer"] = _drainer_slot(det["drainer"])
        elif m["drainer"] is None:
            m["drainer"] = [DRAINER_SLOT_PRIOR[0], DRAINER_SLOT_PRIOR[1], 0.593]
        _note(m, "det milk=%s mug=%s n=%d" % (m["milk"], m["mug"], det["n_clusters"]))
        m["stage"] = "approach"
        stage = "approach"

    milk, mug, dr = m["milk"], m["mug"], m["drainer"]

    if stage == "approach":
        m["stage"] = "descend"
        return _out(m, "act_move", {"target": _grasp_target(m, 0.10), "gripper": -1.0, "max_steps": 120, "tol": 0.004})

    if stage == "descend":
        m["stage"] = "close"
        return _out(m, "act_move", {"target": _grasp_target(m, 0.0), "gripper": -1.0, "max_steps": 100, "tol": 0.003, "vmax": 0.6})

    if stage == "close":
        m["stage"] = "check_grasp"
        return _out(m, "act_gripper", {"close": True, "steps": 28})

    if stage == "check_grasp":
        if width > 0.01:
            m["stage"] = "lift"
            _note(m, "grasp width %.4f" % width)
            stage = "lift"
        else:
            m["grasp_tries"] = int(m["grasp_tries"]) + 1
            _note(m, "grasp missed (width %.4f) try %d" % (width, m["grasp_tries"]))
            m["stage"] = "regrasp_retreat"
            return _out(m, "act_move", {"target": [tcp[0], tcp[1], min(0.85, tcp[2] + 0.12)], "gripper": -1.0, "max_steps": 60, "tol": 0.01})

    if stage == "regrasp_retreat":
        m["stage"] = "regrasp_home"
        if tcp[2] < 0.62:
            m["stage"] = "regrasp_retreat"
            return _out(m, "act_move", {"target": [tcp[0], tcp[1], 0.68], "gripper": -1.0, "max_steps": 60, "tol": 0.01})
        return _out(m, "act_move", {"target": HOME, "gripper": -1.0, "max_steps": 120, "tol": 0.01})

    if stage == "regrasp_home":
        det = _detect(f, tcp)
        if det["milk"] is not None:
            m["milk"] = _p(det["milk"])
            _note(m, "re-detected milk %s" % m["milk"])
        else:
            _note(m, "milk not re-detected; reuse last estimate")
        if int(m["grasp_tries"]) >= 4:
            m["stage"] = "give_up"
            _note(m, "grasp retries exhausted (task NOT complete)")
            return _out(m, "act_hold", {"steps": 20, "gripper": -1.0})
        m["stage"] = "descend"
        return _out(m, "act_move", {"target": _grasp_target(m, 0.10), "gripper": -1.0, "max_steps": 120, "tol": 0.004})

    if stage == "lift":
        m["stage"] = "to_pour"
        return _out(m, "act_move", {"target": [tcp[0], tcp[1], LIFT_Z], "gripper": 1.0, "max_steps": 80, "tol": 0.01})

    if stage == "to_pour":
        if width < 0.01:
            _note(m, "milk lost after lift (width %.4f)" % width)
            m["stage"] = "regrasp_retreat"
            m["grasp_tries"] = int(m["grasp_tries"]) + 1
            return _out(m, "act_gripper", {"close": False, "steps": 10})
        n = int(m["pours_done"])
        # pour index 2 is a fallback re-pour (only reached when no success after pour 2): use the pour-1 pose
        off = POUR2_OFF if n == 1 else POUR1_OFF
        m["stage"] = "tilt"
        return _out(m, "act_move", {"target": [mug[0] + off[0], mug[1] + off[1], off[2]], "gripper": 1.0, "max_steps": 150, "tol": 0.004})

    if stage == "tilt":
        if width < 0.01:
            _note(m, "milk lost before pour (width %.4f)" % width)
            m["stage"] = "regrasp_retreat"
            m["grasp_tries"] = int(m["grasp_tries"]) + 1
            return _out(m, "act_gripper", {"close": False, "steps": 10})
        n = int(m["pours_done"])
        hold = 52 if n == 0 else 110
        m["stage"] = "untilt"
        return _out(m, "act_pour", {"hold_steps": hold, "tilt_rad": POUR_TILT})

    if stage == "untilt":
        m["pours_done"] = int(m["pours_done"]) + 1
        _note(m, "pour %d done, tilt %.1f" % (m["pours_done"], _tilt_deg(f.tcp[3:6])))
        if int(m["pours_done"]) == 2:
            _note(m, "no success after 2 pours: one fallback re-pour, then drainer")
        m["stage"] = "to_pour" if int(m["pours_done"]) < 3 else "to_drainer"
        return _out(m, "act_untilt", {})

    if stage == "to_drainer":
        m["stage"] = "drainer_over"
        return _out(m, "act_move", {"target": [tcp[0], tcp[1], 0.85], "gripper": 1.0, "max_steps": 60, "tol": 0.01})

    if stage == "drainer_over":
        m["stage"] = "drainer_down"
        return _out(m, "act_move", {"target": [dr[0], dr[1], 0.85], "gripper": 1.0, "max_steps": 200, "tol": 0.006})

    if stage == "drainer_down":
        m["stage"] = "release"
        # milk bottom is ~0.093 m below the tcp: tcp at rim + 0.07 puts it ~2 cm inside the slot
        return _out(m, "act_move", {"target": [dr[0], dr[1], dr[2] + 0.07], "gripper": 1.0, "max_steps": 120, "tol": 0.005, "vmax": 0.5})

    if stage == "release":
        m["stage"] = "retreat"
        return _out(m, "act_gripper", {"close": False, "steps": 25})

    if stage == "retreat":
        m["stage"] = "wait"
        return _out(m, "act_move", {"target": [tcp[0], tcp[1], 0.85], "gripper": -1.0, "max_steps": 60, "tol": 0.01})

    # wait / give_up: idle while the checker settles
    if stage != "give_up":
        m["stage"] = "wait"
    return _out(m, "act_hold", {"steps": 60, "gripper": -1.0})


def _rot_cmd(frame, Rt, gain=1.9, cap=0.8):
    e = _ori_err(frame, Rt)
    r = gain * e
    nrm = float(np.max(np.abs(r)))
    if nrm > cap:
        r = r * (cap / nrm)
    return r, float(np.linalg.norm(e))


def act_move(robot, target, gripper=1.0, max_steps=150, tol=0.004, vmax=1.0, gain=30.0):
    """Move the TCP to world `target` [x,y,z] (m) with a straight-line P controller, holding the
    gripper-down orientation. gripper: +1 close / -1 open. Returns measured final tcp, error, steps."""
    tgt = np.asarray(target, dtype=float)
    Rd = _down()
    steps = 0
    stalled = False
    hist = []
    f = robot.now()
    for i in range(int(max_steps)):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        err = tgt - p
        d = float(np.linalg.norm(err))
        if d < tol and i > 0:
            break
        hist.append(p)
        # stall: commanded motion but < 1.5 mm progress over the last 12 steps (contact / limit)
        if len(hist) > 12 and float(np.linalg.norm(hist[-1] - hist[-13])) < 0.0015 and d < 0.03:
            stalled = True
            break
        v = gain * err
        s = float(np.max(np.abs(v)))
        if s > vmax:
            v = v * (vmax / s)
        r, _ = _rot_cmd(f, Rd, gain=1.5, cap=0.3)
        robot.step(float(v[0]), float(v[1]), float(v[2]), float(r[0]), float(r[1]), float(r[2]), float(gripper))
        steps += 1
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=float)
    return {"tcp": p.round(4).tolist(), "err_m": round(float(np.linalg.norm(tgt - p)), 4), "steps": steps, "stalled": stalled,
            "width": round(_width(f), 4), "tilt_deg": round(_tilt_deg(f.tcp[3:6]), 1)}


def act_gripper(robot, close=True, steps=28):
    """Close (+1) or open (-1) the gripper for `steps` native steps with zero motion; returns finger width."""
    g = 1.0 if close else -1.0
    w0 = _width(robot.now())
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, g)
    f = robot.now()
    return {"width_before": round(w0, 4), "width": round(_width(f), 4), "tcp": np.asarray(f.tcp[:3]).round(4).tolist()}


def act_pour(robot, hold_steps=52, tilt_rad=1.056, max_tilt_steps=45):
    """Tilt the held milk about world -y to `tilt_rad` (zero translation command, gripper closed), then hold
    until `hold_steps` frames with tilt > 45 deg have elapsed. Returns measured tilt and tcp."""
    Rt = _rot_y(-float(tilt_rad)) @ _down()
    n45 = 0
    k = 0
    for k in range(int(max_tilt_steps)):
        f = robot.now()
        r, e = _rot_cmd(f, Rt)
        if e < math.radians(1.0):
            break
        robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), 1.0)
        if _tilt_deg(robot.now().tcp[3:6]) > 45:
            n45 += 1
    peak = _tilt_deg(robot.now().tcp[3:6])
    h = 0
    while n45 < int(hold_steps) and h < int(hold_steps) + 40:
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
        h += 1
        if _tilt_deg(robot.now().tcp[3:6]) > 45:
            n45 += 1
    f = robot.now()
    return {"tilt_steps": k, "peak_tilt_deg": round(peak, 1), "hold_steps": h, "frames_over45": n45,
            "tilt_deg": round(_tilt_deg(f.tcp[3:6]), 1), "tcp": np.asarray(f.tcp[:3]).round(4).tolist(),
            "width": round(_width(f), 4)}


def act_untilt(robot, max_steps=45, settle=8):
    """Rotate back to the gripper-down orientation (zero translation, gripper closed), then settle."""
    Rd = _down()
    k = 0
    for k in range(int(max_steps)):
        f = robot.now()
        r, e = _rot_cmd(f, Rd)
        if e < math.radians(1.5):
            break
        robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), 1.0)
    for _ in range(int(settle)):
        f = robot.now()
        r, e = _rot_cmd(f, Rd, gain=1.5, cap=0.3)
        robot.step(0, 0, 0, float(r[0]), float(r[1]), float(r[2]), 1.0)
    f = robot.now()
    return {"steps": k, "tilt_deg": round(_tilt_deg(f.tcp[3:6]), 1), "tcp": np.asarray(f.tcp[:3]).round(4).tolist(),
            "width": round(_width(f), 4)}


def act_hold(robot, steps=60, gripper=1.0):
    """Hold still for `steps` native steps with the given gripper command; returns tcp and width."""
    for _ in range(int(steps)):
        robot.step(0, 0, 0, 0, 0, 0, float(gripper))
    f = robot.now()
    return {"tcp": np.asarray(f.tcp[:3]).round(4).tolist(), "width": round(_width(f), 4)}
