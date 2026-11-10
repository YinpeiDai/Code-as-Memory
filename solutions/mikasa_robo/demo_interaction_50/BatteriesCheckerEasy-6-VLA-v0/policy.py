"""BatteriesCheckerEasy-6 policy: test all six batteries in the socket, pressing the button after each.

Layout (from 50 demos, rigid translation only): six green batteries in a 2x3 grid on a tray,
socket (grey block with a dark hole, lamp behind it) and a pink button. Per battery the expert
does: pick from its slot -> lower into the socket and release (battery vanishes, lamp may light,
battery reappears in its tray slot) -> close gripper and press the button.  Success fires during
the 6th button press.
"""

# ---- layout constants measured in demos (metres, world frame) ----
SLOT_OFFSETS = [[0.0497, -0.0259], [0.0001, -0.0260], [-0.0498, -0.0256],
                [0.0494, 0.0257], [0.0000, 0.0258], [-0.0494, 0.0260]]
SOCKET_FROM_TRAY = [0.1971, 0.0780]
BUTTON_FROM_TRAY = [0.0044, 0.3132]
LAMP_FROM_SOCKET = [0.0746, -0.0037, 0.106]
GREEN_BIAS = [0.0055, -0.0020]   # front-camera green-pixel mean minus expert pick centre
HOLE_BIAS = [0.0051, -0.0024]    # front-camera dark-hole mean minus expert insert point
PINK_BIAS = [0.0040, -0.0021]    # front-camera pink mean minus expert press point
WRIST_GREEN_BIAS = [0.0055, 0.0014]  # wrist-camera green mean at hover minus expert pick (48 picks, 8 demos)
Z_HOVER = 0.20
Z_TRANSIT = 0.21
Z_GRASP = 0.040
Z_RELEASE = 0.045
Z_PRESS = 0.045       # commanded; the button bottoms out near TCP z = 0.053 in demos
G_PREGRASP = 0.1      # half-open (fingers ~0.0175 m each)
G_OPEN = 1.0
G_CLOSE = -1.0
N_BATT = 6
ALWAYS_REFINE = False  # R041 ran refine on every pick live: success, wrist and front agree within ~1 mm


# ---------------- geometry helpers ----------------
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return v / 2.0
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return v * (ang / (2.0 * s))


def _rot_cmd(tcp, rpy_t, gain=1.0, cap=0.6):
    """Command (drx,dry,drz) turning measured orientation toward rpy_t.

    Native CPU IK applies Rx(-0.1 r0)Ry(-0.1 r1)Rz(-0.1 r2) left-multiplied, so r = -w/0.1."""
    Rc = _rpy_to_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rt = _rpy_to_R(rpy_t[0], rpy_t[1], rpy_t[2])
    w = _rotvec(Rt @ Rc.T)
    r = -gain * w / 0.1
    n = float(np.linalg.norm(r))
    if n > cap:
        r = r * (cap / n)
    return r


def _deproject(depth, K, E, mask):
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(float) / 1000.0
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    c = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return (c - E[:, 3]) @ E[:, :3]


def _project(K, E, p):
    c = E[:, :3] @ np.asarray(p, dtype=float) + E[:, 3]
    if c[2] <= 1e-6:
        return None
    u = K @ c
    return [float(u[0] / u[2]), float(u[1] / u[2])]


def _masks(rgb):
    F = rgb.astype(int)
    r, g, b = F[..., 0], F[..., 1], F[..., 2]
    green = (g > r + 40) & (g > b + 20)
    pink = (r > g + 25) & (np.abs(g - b) < 10) & (r > 120)
    dark = (r < 105) & (r > 70) & (np.abs(r - g) < 6) & (b >= r) & (b - r < 12)
    return green, pink, dark


def _layout(frame):
    """Estimate tray centre, slot centres, socket and button from the fixed front camera."""
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    D = frame.front_depth
    green, pink, dark = _masks(frame.front_rgb)
    G = _deproject(D, K, E, green)
    if len(G):
        G = G[(G[:, 2] > 0.04) & (G[:, 2] < 0.08)]
    if len(G) < 30:
        return {"ok": False, "why": "green pixels %d" % len(G)}
    offs = np.array(SLOT_OFFSETS)
    c = G[:, :2].mean(0)
    # robust grid fit: assign points to nearest slot, average per-slot centres
    for _ in range(3):
        slots = c + offs
        dist = np.linalg.norm(G[:, None, :2] - slots[None], axis=2)
        lab = np.argmin(dist, 1)
        near = dist[np.arange(len(G)), lab] < 0.02
        ests = []
        for s in range(N_BATT):
            m = near & (lab == s)
            if m.sum() >= 5:
                ests.append(G[m, :2].mean(0) - offs[s])
        if not ests:
            break
        c = np.mean(np.array(ests), 0)
    nslots = len(ests)
    tc = c - np.array(GREEN_BIAS)
    sock_pred = tc + np.array(SOCKET_FROM_TRAY)
    H = _deproject(D, K, E, dark)
    sock = sock_pred.copy()
    nh = 0
    if len(H):
        H = H[(np.linalg.norm(H[:, :2] - sock_pred, axis=1) < 0.03) & (H[:, 2] < 0.05)]
        nh = len(H)
        if nh >= 15:
            hole = H[:, :2].mean(0) - np.array(HOLE_BIAS)
            if np.linalg.norm(hole - sock_pred) < 0.012:
                sock = 0.5 * (hole + sock_pred)
    btn_pred = tc + np.array(BUTTON_FROM_TRAY)
    P = _deproject(D, K, E, pink)
    btn = btn_pred.copy()
    npk = 0
    if len(P):
        P = P[(np.linalg.norm(P[:, :2] - btn_pred, axis=1) < 0.05) & (P[:, 2] > 0.02)]
        npk = len(P)
        if npk >= 20:
            pb = P[:, :2].mean(0) - np.array(PINK_BIAS)
            if np.linalg.norm(pb - btn_pred) < 0.015:
                btn = 0.5 * (pb + btn_pred)
    slots = [[float(tc[0] + o[0]), float(tc[1] + o[1])] for o in SLOT_OFFSETS]
    return {"ok": True, "tray": [float(tc[0]), float(tc[1])], "slots": slots,
            "socket": [float(sock[0]), float(sock[1])], "button": [float(btn[0]), float(btn[1])],
            "n_green": int(len(G)), "n_slots_seen": int(nslots), "n_hole": int(nh), "n_pink": int(npk)}


def _lamp_brightness(frame, socket):
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    p = [socket[0] + LAMP_FROM_SOCKET[0], socket[1] + LAMP_FROM_SOCKET[1], LAMP_FROM_SOCKET[2]]
    uv = _project(K, E, p)
    if uv is None:
        return -1.0
    u, v = int(round(uv[0])), int(round(uv[1]))
    if u < 3 or v < 3 or u > 252 or v > 252:
        return -1.0
    patch = frame.front_rgb[v - 3:v + 4, u - 3:u + 4].astype(float)
    return float(patch.mean())


# ---------------- motion helpers ----------------
def _goto(robot, tgt, grip, rpy, tol=0.004, max_steps=60, vmax=0.08, settle=0.003, stall_tol=0.012):
    """Closed-loop move of the TCP to tgt (world xyz) while holding orientation rpy.

    Ends when within tol and not moving, or when stalled (moved < 0.6 mm over 3 steps)
    within stall_tol of the target with xy already within 4 mm (blocked vertically, e.g. the
    battery bottoming out in the socket)."""
    tgt = np.asarray(tgt, dtype=float)
    hist = []
    n = 0
    stalled = False
    for n in range(1, max_steps + 1):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        e = tgt - p
        en = float(np.linalg.norm(e))
        moving = len(hist) > 0 and float(np.linalg.norm(p - hist[-1])) > settle
        if en < tol and not moving:
            n -= 1
            break
        exy = float(np.linalg.norm(e[:2]))
        if (len(hist) >= 3 and en < stall_tol and exy < 0.004
                and float(np.linalg.norm(p - hist[-3])) < 0.0006):
            stalled = True
            n -= 1
            break
        hist.append(p)
        d = e.copy()
        if en > vmax:
            d = d * (vmax / en)
        r = _rot_cmd(f.tcp, rpy)
        robot.step(d[0] / 0.1, d[1] / 0.1, d[2] / 0.1, r[0], r[1], r[2], grip)
    p = np.asarray(robot.now().tcp[:3], dtype=float)
    return {"steps": n, "err": round(float(np.linalg.norm(tgt - p)), 4), "pos": [round(float(x), 4) for x in p],
            "stalled": stalled}


def _hold(robot, grip, rpy, n, tgt=None):
    """Hold position (or tgt) for n steps with the given gripper command."""
    for _ in range(n):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        e = (np.asarray(tgt, dtype=float) - p) if tgt is not None else np.zeros(3)
        r = _rot_cmd(f.tcp, rpy)
        robot.step(e[0] / 0.1, e[1] / 0.1, e[2] / 0.1, r[0], r[1], r[2], grip)


def _fingers(robot):
    return float(np.mean(robot.now().fingers))


def _wait_fingers(robot, grip, rpy, tgt, min_steps, max_steps, stop):
    """Hold at tgt while the gripper moves; stop(width, prev) ends the wait."""
    prev = _fingers(robot)
    n = 0
    for n in range(1, max_steps + 1):
        _hold(robot, grip, rpy, 1, tgt)
        w = _fingers(robot)
        if n >= min_steps and stop(w, prev):
            break
        prev = w
    return n


# ---------------- policy ----------------
def _parse_flag(last, key):
    if not last:
        return None
    t = '"%s": true' % key
    f = '"%s": false' % key
    if t in last:
        return True
    if f in last:
        return False
    return None


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive layout once, then for each of 6 batteries pick -> insert -> press."""
    m = dict(memory)
    print("[decide] step=%s phase=%s battery=%s last=%s" % (obs.current.step, memory.get("phase"), memory.get("battery"), (last or "")[:1500]))
    for k, v in (("battery", 0), ("phase", "init"), ("results", []), ("retries", 0), ("log", "")):
        if m.get(k) is None:
            m[k] = v
    cur = obs.current
    if m.get("hold_rpy") is None:
        m["hold_rpy"] = [float(cur.tcp[3]), float(cur.tcp[4]), float(cur.tcp[5])]
    if m.get("tray") is None:
        lay = _layout(cur)
        if not lay["ok"]:
            m["retries"] = int(m["retries"]) + 1
            m["log"] = "layout failed: " + lay.get("why", "")
            m["phase"] = "init"
            return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 3}}}
        m["tray"] = lay["tray"]
        m["slots"] = lay["slots"]
        m["socket"] = lay["socket"]
        m["button"] = lay["button"]
        m["log"] = "layout green=%d slots=%d hole=%d pink=%d" % (lay["n_green"], lay["n_slots_seen"], lay["n_hole"], lay["n_pink"])
        m["phase"] = "pick"
    else:
        ph = m["phase"]
        if ph == "pick_issued":
            w = float(np.mean(cur.fingers))
            if 0.004 < w < 0.016:
                m["phase"] = "insert"
                m["retries"] = 0
            else:
                m["retries"] = int(m["retries"]) + 1
                m["phase"] = "pick"
            m["log"] = "after pick fingers=%.4f" % w
        elif ph == "insert_issued":
            lit = _parse_flag(last, "lamp_lit")
            res = list(m["results"])
            res.append("lit" if lit else ("dark" if lit is False else "unknown"))
            m["results"] = res
            m["phase"] = "press"
            m["log"] = "after insert: " + (last or "")[:200]
        elif ph == "press_issued":
            m["battery"] = int(m["battery"]) + 1
            m["phase"] = "pick" if int(m["battery"]) < N_BATT else "extra"
            m["log"] = "after press: " + (last or "")[:200]
    ph = m["phase"]
    rpy = m["hold_rpy"]
    i = int(m["battery"])
    if ph == "pick":
        if int(m["retries"]) > 2:
            # give up on this battery's grasp; still press so the sequence advances
            m["phase"] = "press_issued"
            return {"memory": m, "act": {"tool": "act_press", "args": {"x": m["button"][0], "y": m["button"][1], "rpy": rpy}}}
        s = m["slots"][i % N_BATT]
        m["phase"] = "pick_issued"
        return {"memory": m, "act": {"tool": "act_pick", "args": {"x": s[0], "y": s[1], "rpy": rpy, "refine": bool(ALWAYS_REFINE or int(m["retries"]) > 0)}}}
    if ph == "insert":
        m["phase"] = "insert_issued"
        return {"memory": m, "act": {"tool": "act_insert", "args": {"x": m["socket"][0], "y": m["socket"][1], "rpy": rpy}}}
    if ph == "press":
        m["phase"] = "press_issued"
        return {"memory": m, "act": {"tool": "act_press", "args": {"x": m["button"][0], "y": m["button"][1], "rpy": rpy}}}
    # all six done but no success yet: press the button again
    m["phase"] = "press_issued"
    return {"memory": m, "act": {"tool": "act_press", "args": {"x": m["button"][0], "y": m["button"][1], "rpy": rpy}}}


def _wrist_refine(frame, x, y, radius=0.02):
    """Refine a battery centre from the wrist camera (green top-surface pixels near (x, y))."""
    K = np.asarray(frame.wrist_K, dtype=float)
    E = np.asarray(frame.wrist_extrinsic, dtype=float)
    green, _, _ = _masks(frame.wrist_rgb)
    G = _deproject(frame.wrist_depth, K, E, green)
    if len(G) == 0:
        return None, 0
    G = G[(G[:, 2] > 0.055) & (G[:, 2] < 0.07)]
    G = G[np.linalg.norm(G[:, :2] - np.array([x, y]), axis=1) < radius]
    if len(G) < 20:
        return None, int(len(G))
    c = G[:, :2].mean(0) - np.array(WRIST_GREEN_BIAS)
    if float(np.linalg.norm(c - np.array([x, y]))) > 0.008:
        return None, int(len(G))
    return [float(c[0]), float(c[1])], int(len(G))


def act_wait(robot, steps=3):
    """Hold still with the gripper open for a few steps (used when perception fails)."""
    rpy = [float(v) for v in robot.now().tcp[3:]]
    _hold(robot, G_OPEN, rpy, int(steps))
    return {"waited": int(steps)}


def act_pick(robot, x, y, rpy, refine=False):
    """Pick the battery at slot (x, y): hover, descend half-open, close, lift. Returns finger width."""
    out = {"target": [round(x, 4), round(y, 4)]}
    p = np.asarray(robot.now().tcp[:3], dtype=float)
    if p[2] < Z_HOVER - 0.03:
        _goto(robot, [p[0], p[1], Z_TRANSIT], G_PREGRASP, rpy, tol=0.02, max_steps=20)
    out["hover"] = _goto(robot, [x, y, Z_HOVER], G_PREGRASP, rpy, tol=0.004, max_steps=50)
    if refine:
        c, n = _wrist_refine(robot.now(), x, y)
        out["wrist_refine"] = [c, n]
        if c is not None:
            x, y = c[0], c[1]
            out["hover2"] = _goto(robot, [x, y, Z_HOVER], G_PREGRASP, rpy, tol=0.003, max_steps=20)
    out["descend"] = _goto(robot, [x, y, Z_GRASP], G_PREGRASP, rpy, tol=0.004, max_steps=40)
    out["close_steps"] = _wait_fingers(robot, G_CLOSE, rpy, [x, y, Z_GRASP], 3, 12,
                                       lambda w, pw: abs(w - pw) < 0.0004)
    out["fingers_closed"] = round(_fingers(robot), 4)
    out["lift"] = _goto(robot, [x, y, Z_TRANSIT], G_CLOSE, rpy, tol=0.01, max_steps=30)
    w = _fingers(robot)
    out["fingers"] = round(w, 4)
    out["holding"] = bool(0.004 < w < 0.016)
    return out


def act_insert(robot, x, y, rpy):
    """Carry the held battery over the socket (x, y), lower, release, record lamp, lift and close."""
    out = {"target": [round(x, 4), round(y, 4)]}
    sock = [x, y]
    out["base_lamp"] = round(_lamp_brightness(robot.now(), sock), 1)
    out["hover"] = _goto(robot, [x, y, Z_HOVER], G_CLOSE, rpy, tol=0.003, max_steps=50)
    out["descend1"] = _goto(robot, [x, y, Z_RELEASE + 0.04], G_CLOSE, rpy, tol=0.006, max_steps=25)
    out["descend"] = _goto(robot, [x, y, Z_RELEASE], G_CLOSE, rpy, tol=0.004, max_steps=25, vmax=0.03)
    start = len(robot.frames())
    out["open_steps"] = _wait_fingers(robot, G_OPEN, rpy, [x, y, Z_RELEASE], 4, 10,
                                      lambda w, pw: w > 0.03)
    _hold(robot, G_OPEN, rpy, 2, [x, y, Z_RELEASE])
    out["lift"] = _goto(robot, [x, y, Z_TRANSIT], G_OPEN, rpy, tol=0.015, max_steps=30)
    fr = robot.frames()
    lamps = [_lamp_brightness(f, sock) for f in fr[max(0, start - 3):]]
    out["lamp_max"] = round(max(lamps), 1) if lamps else -1
    out["lamp_lit"] = bool(lamps and max(lamps) > out["base_lamp"] + 15)
    c, n = _wrist_refine(robot.now(), x, y, radius=0.05)
    out["green_near_socket_wrist"] = n
    _wait_fingers(robot, G_CLOSE, rpy, None, 3, 8, lambda w, pw: w < 0.004)
    out["fingers"] = round(_fingers(robot), 4)
    return out


def act_press(robot, x, y, rpy):
    """Press the button at (x, y) with the closed gripper, then lift back to transit height."""
    out = {"target": [round(x, 4), round(y, 4)]}
    p = np.asarray(robot.now().tcp[:3], dtype=float)
    if p[2] < Z_HOVER - 0.03:
        _goto(robot, [p[0], p[1], Z_TRANSIT], G_CLOSE, rpy, tol=0.02, max_steps=20)
    out["hover"] = _goto(robot, [x, y, Z_HOVER], G_CLOSE, rpy, tol=0.004, max_steps=50)
    # descend until the button bottoms out (TCP stalls) or a step cap is reached
    zs = []
    n = 0
    for n in range(1, 31):
        f = robot.now()
        pz = float(f.tcp[2])
        zs.append(pz)
        if len(zs) >= 3 and pz < 0.065 and abs(zs[-1] - zs[-3]) < 0.0015:
            break
        e = np.array([x, y, Z_PRESS]) - np.asarray(f.tcp[:3], dtype=float)
        dn = float(np.linalg.norm(e))
        if dn > 0.08:
            e = e * (0.08 / dn)
        r = _rot_cmd(f.tcp, rpy)
        robot.step(e[0] / 0.1, e[1] / 0.1, e[2] / 0.1, r[0], r[1], r[2], G_CLOSE)
    out["press_steps"] = n
    out["z_min"] = round(min(zs), 4)
    out["lift"] = _goto(robot, [x, y, Z_TRANSIT], G_CLOSE, rpy, tol=0.015, max_steps=30)
    return out
