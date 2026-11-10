"""Code-as-Memory policy for T14_CookiesChocolateDrawers.

Instruction: "Put cookies into the top drawer and put chocolate into another drawer."
Phase machine (memory.phase): open_top -> pick_cookies -> place_cookies -> close_top
-> open_mid -> pick_choc -> place_choc -> done.  perceive_decide localises the cabinet
handle and the two objects from the first agentview RGB-D frame (calibrated back-projection),
stores them in memory and emits one act_segments call per phase.  act_segments runs a closed-loop
Cartesian P-controller (OSC_POSE delta actions) through the given waypoints and grip commands.
"""

PI = 3.141592653589793

# ---- cabinet geometry, calibrated on the 50 assigned demonstrations (see task_analysis.md) ----
CAB_TOP_Z_LO = 0.66            # search window for the cabinet top plane height
CAB_TOP_Z_HI = 0.78
HANDLE_DX = -0.1293            # top-drawer handle x  = max x of cabinet top plane + HANDLE_DX
HANDLE_DY = -0.0095            # handle y = 1st-percentile y of cabinet top plane + HANDLE_DY
TOP_TO_H1 = -0.0402            # top-drawer handle z  = cabinet top z + TOP_TO_H1   (0.6734)
TOP_TO_H2 = -0.1070            # second-drawer handle z = cabinet top z + TOP_TO_H2 (0.6066)
DEFAULT_TOP = [0.13, 0.46, 0.7136]   # fallback (median over demos) if perception fails

# ---- objects: prior table locations (demo range is about +-1-2 cm) and grasp offsets ----
COOKIE_NOM = [0.218, -0.328]
CHOC_NOM = [-0.225, -0.322]
COOKIE_OFF = [-0.0036, 0.0004]       # grasp = perceived mid-range + offset (expert grasp points)
CHOC_OFF = [-0.0100, 0.0055]
GRASP_Z = 0.433                      # expert grasp height for both objects (table ~0.430)

KP = 12.0      # position gain: action = KP * error[m]
KR = 2.0       # rotation gain: action = KR * axis-angle error[rad]
ROT_MAX = 0.5


def _aa2R(w):
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _R2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > PI - 1e-3:
        M = (R + np.eye(3)) / 2.0
        d = np.clip(np.diag(M), 0.0, None)
        i = int(np.argmax(d))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        k = k / max(np.linalg.norm(k), 1e-12)
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2.0 * math.sin(th)))


def _goal_R(name):
    if name == "side":
        return _aa2R([1.5 * PI, 0.0, 0.0])   # gripper axis +y (towards the drawer fronts)
    return _aa2R([PI, 0.0, 0.0])             # gripper pointing down, fingers close along y


def _cloud(frame):
    D = np.asarray(frame.agentview_depth).astype(float) / 1000.0
    K = np.asarray(frame.agentview_K, dtype=float)
    E = np.asarray(frame.agentview_extrinsic, dtype=float)
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    x = (u - K[0, 2]) / K[0, 0] * D
    y = (v - K[1, 2]) / K[1, 1] * D
    pc = np.stack([x, y, D], -1)
    w = (pc - E[:, 3]) @ E[:, :3]
    return w, D > 0


def _perceive_cabinet(w, ok):
    X, Y, Z = w[..., 0], w[..., 1], w[..., 2]
    reg = ok & (Y > 0.25) & (Y < 0.85) & (X > -0.2) & (X < 0.5) & (Z > CAB_TOP_Z_LO) & (Z < CAB_TOP_Z_HI)
    if reg.sum() < 200:
        return None
    zs = Z[reg]
    bins = np.arange(CAB_TOP_Z_LO, CAB_TOP_Z_HI + 0.004, 0.004)
    h, b = np.histogram(zs, bins=bins)
    i = int(np.argmax(h))
    near = zs[np.abs(zs - (b[i] + 0.002)) < 0.006]
    ztop = float(np.median(near))
    top = reg & (np.abs(Z - ztop) < 0.004)
    if top.sum() < 200:
        return None
    xs, ys = X[top], Y[top]
    return [float(xs.max()), float(np.percentile(ys, 1)), ztop, int(top.sum())]


def _perceive_object(w, ok, nom, radius=0.07):
    X, Y, Z = w[..., 0], w[..., 1], w[..., 2]
    tm = ok & (X > 0.0) & (X < 0.35) & (Y > -0.1) & (Y < 0.25)
    tz = float(np.median(Z[tm])) if tm.sum() > 50 else 0.43
    if abs(tz - 0.43) > 0.03:
        tz = 0.43
    m = ok & (np.hypot(X - nom[0], Y - nom[1]) < radius) & (Z > tz + 0.006) & (Z < tz + 0.1)
    n = int(m.sum())
    if n < 15:
        return None
    P = w[m]
    lo = np.percentile(P[:, :2], 3, axis=0)
    hi = np.percentile(P[:, :2], 97, axis=0)
    mid = (lo + hi) / 2.0
    return [float(mid[0]), float(mid[1]), float(P[:, 2].max()), tz, n]


def _scene(frame):
    """Cabinet handle and object grasp points from one agentview RGB-D frame."""
    notes = []
    try:
        w, ok = _cloud(frame)
    except (ValueError, TypeError, IndexError, KeyError, ArithmeticError) as exc:
        w, ok = np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=bool)
        notes.append("no depth: %s" % str(exc)[:60])
    cab = _perceive_cabinet(w, ok)
    if cab is None:
        cab = [DEFAULT_TOP[0], DEFAULT_TOP[1], DEFAULT_TOP[2], 0]
        notes.append("cabinet fallback")
    handle = [cab[0] + HANDLE_DX, cab[1] + HANDLE_DY, cab[2] + TOP_TO_H1, cab[2] + TOP_TO_H2]
    ck = _perceive_object(w, ok, COOKIE_NOM)
    if ck is None:
        ck = [COOKIE_NOM[0] - COOKIE_OFF[0], COOKIE_NOM[1] - COOKIE_OFF[1], 0.445, 0.43, 0]
        notes.append("cookies fallback")
    ch = _perceive_object(w, ok, CHOC_NOM)
    if ch is None:
        ch = [CHOC_NOM[0] - CHOC_OFF[0], CHOC_NOM[1] - CHOC_OFF[1], 0.454, 0.43, 0]
        notes.append("chocolate fallback")
    cookies = [ck[0] + COOKIE_OFF[0], ck[1] + COOKIE_OFF[1], GRASP_Z]
    choc = [ch[0] + CHOC_OFF[0], ch[1] + CHOC_OFF[1], GRASP_Z]
    return {"handle": handle, "cookies": cookies, "chocolate": choc, "cab": cab,
            "ck": ck, "ch": ch, "notes": notes}


def _mv(pos, ori, grip, tol=0.004, rtol=0.04, vmax=1.0, max_steps=200, stall=False, settle=1):
    return {"kind": "move", "pos": [float(p) for p in pos], "ori": ori, "grip": float(grip),
            "tol": tol, "rtol": rtol, "vmax": vmax, "max_steps": int(max_steps),
            "stall": bool(stall), "settle": int(settle)}


def _gr(grip, steps):
    return {"kind": "grip", "grip": float(grip), "steps": int(steps)}


def _gap(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] - f[1])


def _plan(phase, m, cur):
    """Waypoint list for one phase; m is memory (handle, objects, measured drawer positions)."""
    hx, hy, hz1 = m["handle"][0], m["handle"][1], m["handle"][2]
    hz2 = m["mid_handle_z"]
    cx, cy, cz = m["cookies"]
    px, py, pz = m["chocolate"]
    if phase == "open_top":
        return [
            _mv([hx, hy - 0.10, hz1], "side", -1, tol=0.012, rtol=0.08, max_steps=160),
            _mv([hx, hy - 0.10, hz1], "side", -1, tol=0.004, rtol=0.03, max_steps=60),
            _mv([hx, hy + 0.001, hz1], "side", -1, tol=0.003, rtol=0.03, vmax=0.5, max_steps=120, settle=3),
            _gr(1, 30),
            _mv([hx, hy - 0.21, hz1 + 0.004], "side", 1, tol=0.003, vmax=0.6, max_steps=220, stall=True),
        ]
    if phase == "pick_cookies":
        seq = []
        sx, sy, sz = cur
        if m.get("holding_handle"):
            sx, sy, sz = hx, m["top_pull"][1] - 0.113, m["top_pull"][2]
            seq += [_gr(-1, 15), _mv([sx, sy, sz], "side", -1, tol=0.012, max_steps=120)]
        seq += [
            _mv([sx, sy, max(sz, 0.80)], "keep", -1, tol=0.02, rtol=1.0, max_steps=100),
            _mv([cx, cy, 0.62], "down", -1, tol=0.012, rtol=0.08, max_steps=300),
            _mv([cx, cy, 0.62], "down", -1, tol=0.004, rtol=0.03, max_steps=60),
            _mv([cx, cy, cz], "down", -1, tol=0.003, rtol=0.03, vmax=0.6, max_steps=160, settle=3),
            _gr(1, 30),
            _mv([cx, cy, 0.53], "down", 1, tol=0.01, max_steps=100),
        ]
        return seq
    if phase == "place_cookies":
        return [
            _mv([cur[0], cur[1], 0.80], "down", 1, tol=0.02, max_steps=100),
            _mv([hx + 0.004, hy - 0.055, 0.83], "down", 1, tol=0.015, rtol=0.06, max_steps=300),
            _mv([hx + 0.004, hy - 0.055, 0.80], "down", 1, tol=0.005, rtol=0.04, max_steps=80),
            _gr(-1, 20),
            _mv([hx + 0.004, hy - 0.06, 0.86], "down", -1, tol=0.012, max_steps=80),
        ]
    if phase == "close_top":
        ty, tz = m["top_pull"][1], hz1 + 0.003   # drawer slides along y: handle keeps its x and z
        return [
            _mv([hx, ty - 0.11, 0.76], "side", -1, tol=0.02, rtol=0.1, max_steps=250),
            _mv([hx, ty - 0.11, tz], "side", -1, tol=0.005, rtol=0.03, max_steps=120),
            _mv([hx, ty + 0.001, tz], "side", -1, tol=0.003, rtol=0.03, vmax=0.5, max_steps=120, settle=3),
            _gr(1, 30),
            _mv([hx, hy + 0.012, hz1], "side", 1, tol=0.002, vmax=0.6, max_steps=220, stall=True),
        ]
    if phase == "open_mid":
        seq = []
        if m.get("holding_handle"):
            seq += [_gr(-1, 15)]
        cy_now = cur[1]
        seq += [
            _mv([hx, min(cy_now, hy) - 0.10, hz1 + 0.004], "side", -1, tol=0.01, max_steps=100),
            _mv([hx, hy - 0.035, hz2 + 0.004], "side", -1, tol=0.005, rtol=0.03, max_steps=150),
            _mv([hx, hy - 0.005, hz2], "side", -1, tol=0.003, rtol=0.03, vmax=0.5, max_steps=120, settle=3),
            _gr(1, 30),
            _mv([hx, hy - 0.22, hz2 + 0.001], "side", 1, tol=0.003, vmax=0.6, max_steps=220, stall=True),
        ]
        return seq
    if phase == "pick_choc":
        seq = []
        sx, sy, sz = cur
        if m.get("holding_handle"):
            my = m["mid_pull"][1] if m.get("mid_pull") else cur[1]
            sx, sy, sz = hx, my - 0.15, cur[2]
            seq += [_gr(-1, 15), _mv([sx, sy, sz], "side", -1, tol=0.012, max_steps=120)]
        seq += [
            _mv([sx, sy, max(sz, 0.70)], "keep", -1, tol=0.02, rtol=1.0, max_steps=100),
            _mv([px, py, 0.70], "down", -1, tol=0.012, rtol=0.08, max_steps=300),
            _mv([px, py, 0.66], "down", -1, tol=0.004, rtol=0.03, max_steps=60),
            _mv([px, py, pz], "down", -1, tol=0.003, rtol=0.03, vmax=0.6, max_steps=160, settle=3),
            _gr(1, 30),
            _mv([px, py, 0.53], "down", 1, tol=0.01, max_steps=100),
        ]
        return seq
    if phase == "place_choc":
        return [
            _mv([cur[0], cur[1], 0.80], "down", 1, tol=0.02, max_steps=100),
            _mv([hx - 0.005, hy - 0.06, 0.80], "down", 1, tol=0.015, rtol=0.06, max_steps=300),
            _mv([hx - 0.005, hy - 0.06, 0.725], "down", 1, tol=0.005, rtol=0.04, max_steps=100),
            _gr(-1, 20),
            _mv([hx - 0.005, hy - 0.06, 0.80], "down", -1, tol=0.012, max_steps=80),
        ]
    if phase == "clear":
        return [
            _gr(-1, 10),
            _mv([cur[0], cur[1], max(cur[2], 0.75)], "keep", -1, tol=0.02, rtol=1.0, max_steps=100),
            _mv([-0.06, 0.0, 0.72], "down", -1, tol=0.02, rtol=0.1, max_steps=250),
        ]
    return [_gr(0, 40)]


FIELDS = ["phase", "handle", "mid_handle_z", "cookies", "chocolate", "top_pull", "mid_pull", "holding_handle",
          "attempts", "decisions", "log"]


def perceive_decide(obs, goal, memory, last):
    """Phase machine: localise cabinet/objects once, then choose the next manipulation phase.

    Returns one act_segments call (waypoints) per decision; memory keeps the scene estimate,
    the measured pulled-drawer poses and the progress through the task.
    """
    m = {k: memory.get(k) for k in FIELDS} if isinstance(memory, dict) else {k: None for k in FIELDS}
    fr = obs.current
    tcp = np.asarray(fr.tcp, dtype=float)
    cur = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
    gap = _gap(fr)
    log = m["log"] or ""
    m["decisions"] = int(m["decisions"] or 0) + 1
    m["attempts"] = int(m["attempts"] or 0)
    phase = m["phase"]
    note = ""
    if phase is None:
        sc = _scene(fr)
        m["handle"] = [round(v, 5) for v in sc["handle"][:3]]
        m["mid_handle_z"] = round(sc["handle"][3], 5)
        m["cookies"] = [round(v, 5) for v in sc["cookies"]]
        m["chocolate"] = [round(v, 5) for v in sc["chocolate"]]
        m["holding_handle"] = False
        note = "scene H=%s C=%s P=%s %s" % (m["handle"], m["cookies"], m["chocolate"], ",".join(sc["notes"]))
        phase = "open_top"
    else:
        hy = m["handle"][1]
        if phase == "open_top":
            pulled = hy - cur[1]
            if gap > 0.006 and 0.10 < pulled < 0.205:
                m["top_pull"] = [round(v, 5) for v in cur]
                m["holding_handle"] = True
                phase, m["attempts"] = "pick_cookies", 0
                note = "top drawer pulled %.3f" % pulled
            elif m["attempts"] < 2:
                m["attempts"] += 1
                m["holding_handle"] = False
                note = "top pull failed (pulled %.3f gap %.3f); retry" % (pulled, gap)
                phase = "retry_open_top"
            else:
                m["top_pull"] = [m["handle"][0], hy - 0.1667, m["handle"][2] + 0.007]
                m["holding_handle"] = gap > 0.006
                phase, m["attempts"] = "pick_cookies", 0
                note = "top pull uncertain (%.3f); continue" % pulled
        elif phase == "pick_cookies":
            m["holding_handle"] = False
            if gap > 0.025:
                phase, m["attempts"] = "place_cookies", 0
                note = "cookies grasped gap %.3f" % gap
            elif m["attempts"] < 2:
                m["attempts"] += 1
                phase = "clear_cookies"
                note = "cookie grasp missed gap %.3f" % gap
            else:
                phase, m["attempts"] = "close_top", 0
                note = "cookie grasp failed; skipping"
        elif phase == "clear_cookies":
            sc = _scene(fr)
            if "cookies fallback" not in sc["notes"]:
                m["cookies"] = [round(v, 5) for v in sc["cookies"]]
            note = "re-perceived cookies %s" % m["cookies"]
            phase = "pick_cookies"
        elif phase == "place_cookies":
            phase, m["attempts"] = "close_top", 0
            note = "cookies released"
        elif phase == "close_top":
            note = "top drawer push stalled at dy=%.3f" % (cur[1] - hy)
            m["holding_handle"] = gap > 0.006
            phase, m["attempts"] = "open_mid", 0
        elif phase == "open_mid":
            pulled = hy - cur[1]
            if gap > 0.006 and 0.10 < pulled < 0.215:
                m["mid_pull"] = [round(v, 5) for v in cur]
                m["holding_handle"] = True
                phase, m["attempts"] = "pick_choc", 0
                note = "mid drawer pulled %.3f" % pulled
            elif m["attempts"] < 2:
                m["attempts"] += 1
                m["holding_handle"] = False
                phase = "retry_open_mid"
                note = "mid pull failed (pulled %.3f gap %.3f); retry" % (pulled, gap)
            else:
                m["holding_handle"] = gap > 0.006
                phase, m["attempts"] = "pick_choc", 0
                note = "mid pull uncertain; continue"
        elif phase == "pick_choc":
            m["holding_handle"] = False
            if gap > 0.02:
                phase, m["attempts"] = "place_choc", 0
                note = "chocolate grasped gap %.3f" % gap
            elif m["attempts"] < 2:
                m["attempts"] += 1
                phase = "clear_choc"
                note = "chocolate grasp missed gap %.3f" % gap
            else:
                phase = "done"
                note = "chocolate grasp failed"
        elif phase == "clear_choc":
            sc = _scene(fr)
            if "chocolate fallback" not in sc["notes"]:
                m["chocolate"] = [round(v, 5) for v in sc["chocolate"]]
            note = "re-perceived chocolate %s" % m["chocolate"]
            phase = "pick_choc"
        elif phase in ("place_choc", "done"):
            if phase == "place_choc" and m["attempts"] < 1:
                w, ok = _cloud(fr)
                left = _perceive_object(w, ok, m["chocolate"][:2], radius=0.05)
                if left is not None and left[4] >= 25 and left[2] > left[3] + 0.012:
                    m["attempts"] += 1
                    phase = "pick_choc"
                    m["chocolate"] = [round(left[0] + CHOC_OFF[0], 5), round(left[1] + CHOC_OFF[1], 5), GRASP_Z]
                    note = "chocolate still on table (n=%d); retry" % left[4]
                else:
                    phase = "done"
                    note = "chocolate placed; waiting for scorer"
            else:
                phase = "done"
                note = ""
        elif phase == "retry_open_top":
            phase = "open_top"
        elif phase == "retry_open_mid":
            phase = "open_mid"
    m["phase"] = phase
    # build the action for the chosen phase
    if phase in ("clear_cookies", "clear_choc"):
        segs = _plan("clear", m, cur)
    elif phase == "retry_open_top":
        segs = [_gr(-1, 15), _mv([cur[0], cur[1] - 0.06, cur[2]], "side", -1, tol=0.01, max_steps=80)]
        segs += _plan("open_top", m, cur)
        phase = "open_top"
        m["phase"] = phase
    elif phase == "retry_open_mid":
        segs = [_gr(-1, 15), _mv([cur[0], cur[1] - 0.06, cur[2]], "side", -1, tol=0.01, max_steps=80)]
        segs += _plan("open_mid", m, cur)[1:] if m.get("holding_handle") else _plan("open_mid", m, cur)
        phase = "open_mid"
        m["phase"] = phase
    elif phase == "done":
        segs = [_gr(-1, 60)]
    else:
        segs = _plan(phase, m, cur)
    if note:
        log = (log + " | " if log else "") + "d%d %s" % (m["decisions"], note)
    m["log"] = log[-1500:]
    return {"memory": m, "act": {"tool": "act_segments", "args": {"segments": segs, "label": phase}}}


def act_segments(robot, segments, label=""):
    """Run waypoint segments with a closed-loop Cartesian controller.

    segments: list of {"kind":"move","pos":[x,y,z],"ori":"down"|"side"|"keep","grip":+1 close/-1 open,
    "tol","rtol","vmax","max_steps","stall","settle"} or {"kind":"grip","grip":g,"steps":n}.
    A move ends when position/orientation errors are within tolerance (for `settle` steps), when
    progress stalls (drawer at its limit / contact) or after max_steps.  Returns per-segment outcomes.
    """
    out = []
    fr = robot.now()
    keepR = _aa2R(np.asarray(fr.tcp, dtype=float)[3:6])
    for seg in segments:
        if seg.get("kind") == "grip":
            g = float(seg.get("grip", 0.0))
            for _ in range(int(seg.get("steps", 10))):
                fr = robot.step(0, 0, 0, 0, 0, 0, g)
            out.append({"grip": g, "gap": round(_gap(fr), 4)})
            continue
        goal = np.asarray(seg["pos"], dtype=float)
        ori = seg.get("ori", "keep")
        Rg = keepR if ori == "keep" else _goal_R(ori)
        g = float(seg.get("grip", -1.0))
        tol, rtol = float(seg.get("tol", 0.004)), float(seg.get("rtol", 0.04))
        vmax = float(seg.get("vmax", 1.0))
        stall = bool(seg.get("stall", False))
        settle = int(seg.get("settle", 1))
        hist = []
        ok_count = 0
        status = "max_steps"
        n = 0
        for n in range(1, int(seg.get("max_steps", 200)) + 1):
            tcp = np.asarray(fr.tcp, dtype=float)
            p = tcp[:3]
            Rc = _aa2R(tcp[3:6])
            err = goal - p
            rerr = _R2aa(Rg @ Rc.T)
            hist.append(p.copy())
            dist = float(np.linalg.norm(err))
            rd = float(np.linalg.norm(rerr))
            if dist < tol and rd < rtol:
                ok_count += 1
                if ok_count >= settle:
                    status = "reached"
                    break
            else:
                ok_count = 0
            if len(hist) > 12:
                moved = float(np.linalg.norm(hist[-1] - hist[-11]))
                if stall and n > 15 and moved < 0.0015:
                    status = "stalled"
                    break
                if (not stall) and n > 60 and len(hist) > 40 and float(np.linalg.norm(hist[-1] - hist[-40])) < 0.002 and rd < 0.1:
                    status = "stuck"
                    break
            a = np.clip(KP * err, -vmax, vmax)
            r = np.clip(KR * rerr, -ROT_MAX, ROT_MAX)
            fr = robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), g)
        tcp = np.asarray(fr.tcp, dtype=float)
        if ori != "keep":
            keepR = Rg
        out.append({"status": status, "steps": n, "pos": [round(float(v), 4) for v in tcp[:3]],
                    "err": round(float(np.linalg.norm(goal - tcp[:3])), 4), "gap": round(_gap(fr), 4)})
    return {"label": label, "segments": out}
