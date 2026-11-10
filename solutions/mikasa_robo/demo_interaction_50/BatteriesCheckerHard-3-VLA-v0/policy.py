"""BatteriesCheckerHard-3 policy: perceive 3 batteries in the front camera, then for each battery
pick -> insert into socket -> release and observe lamp -> regrasp -> return to slot -> press button."""

# ---------------- constants learned from the 50 assigned demonstrations ----------------
# Detected green-top centroid (front camera, frame 0) minus expert grasp TCP xy (mean over 150 batteries).
DET_BIAS = [0.0058, -0.0019]
# Offsets from the mean of the 3 raw battery-top detections (rigid layout, std < 2 mm).
SOCKET_OFF = [0.1955, 0.1051]     # where the battery sits in the socket (expert regrasp xy)
BUTTON_OFF = [-0.0101, 0.3430]    # expert button press xy
LAMP_OFF = [0.084, -0.003, 0.105]  # visible lamp-bulb surface relative to socket xy (world z absolute)
Z_SAFE = 0.21        # transit height (expert 0.20-0.23)
Z_GRASP = 0.042      # TCP height when grasping in slot (expert 0.040-0.046)
Z_SOCK_RELEASE = 0.045  # release height in socket (expert 0.041-0.044)
Z_REGRASP = 0.045    # regrasp height in socket (expert 0.039-0.050)
Z_SLOT_RELEASE = 0.048  # release height in slot (expert 0.043-0.049)
Z_OBSERVE = 0.18     # lift height while lamp is observed (expert 0.17-0.18)
Z_PRESS = 0.054      # press depth on button (expert 0.053-0.062)
G_OPEN, G_PRE, G_CLOSE, G_REGRASP, G_RELEASE = 1.0, 0.1, -1.0, 0.22, 0.28


def _R_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logm(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: use the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return ax * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


R_GOAL = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # roll=pi, pitch=0, yaw=0 (gripper down)


def _rot_err(tcp):
    R = _R_rpy(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    return _logm(R_GOAL @ R.T)


def _unproject(us, vs, zs, K, E):
    xc = (us - K[0, 2]) * zs / K[0, 0]
    yc = (vs - K[1, 2]) * zs / K[1, 1]
    pc = np.stack([xc, yc, zs], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (pc - t) @ R


def _project(P, K, E):
    pc = E[:, :3] @ np.array(P, dtype=float) + E[:, 3]
    return [float(K[0, 0] * pc[0] / pc[2] + K[0, 2]), float(K[1, 1] * pc[1] / pc[2] + K[1, 2])]


def _components(mask):
    ys, xs = np.nonzero(mask)
    S = set(zip(ys.tolist(), xs.tolist()))
    seen = set()
    comps = []
    for p in list(S):
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        comp = []
        while stack:
            q = stack.pop()
            comp.append(q)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    r = (q[0] + dy, q[1] + dx)
                    if r in S and r not in seen:
                        seen.add(r)
                        stack.append(r)
        comps.append(np.array(comp))
    return comps


def _detect_batteries(frame):
    """Green battery tops in the fixed front camera -> world xyz centroids, sorted by x descending."""
    img = np.asarray(frame.front_rgb).astype(int)
    dep = np.asarray(frame.front_depth).astype(float) / 1000.0
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    g = (img[:, :, 1] > 120) & (img[:, :, 1] - img[:, :, 0] > 40) & (img[:, :, 1] - img[:, :, 2] > 25)
    out = []
    for c in _components(g):
        if len(c) < 4:
            continue
        z = dep[c[:, 0], c[:, 1]]
        ok = z > 0
        if ok.sum() < 3:
            continue
        w = _unproject(c[ok, 1].astype(float), c[ok, 0].astype(float), z[ok], K, E).mean(0)
        if -0.4 < w[0] < 0.4 and -0.4 < w[1] < 0.4 and 0.02 < w[2] < 0.12:
            out.append([float(w[0]), float(w[1]), float(w[2]), int(len(c))])
    out.sort(key=lambda p: -p[0])
    return out


def _lamp_count(frame, uv):
    img = np.asarray(frame.front_rgb)
    u, v = int(round(uv[0])), int(round(uv[1]))
    sub = img[max(0, v - 10):v + 11, max(0, u - 10):u + 11]
    if sub.size == 0:
        return 0
    return int((sub.min(axis=2) >= 248).sum())


def _fing(frame):
    f = np.asarray(frame.fingers, dtype=float)
    return float(f[0] + f[1])


def _holding(frame):
    """Closed gripper with an object between the fingers: sum ~0.018 (battery), ~0 when empty."""
    s = _fing(frame)
    return 0.008 < s < 0.03


# ---------------- motion primitives (used inside act_* only) ----------------

def _goto(robot, goal, grip, vmax=1.0, tol=0.004, max_steps=60, settle=0.0015, rot_gain=0.3, log=None, stall_stop=False):
    goal = np.array(goal, dtype=float)
    prev = None
    n = 0
    still = 0
    stalled = False
    for n in range(max_steps):
        f = robot.now()
        p = np.asarray(f.tcp[:3], dtype=float)
        e = goal - p
        spd = 1.0 if prev is None else float(np.linalg.norm(p - prev))
        if float(np.linalg.norm(e)) < tol and spd < settle:
            break
        still = still + 1 if spd < 0.0006 else 0
        if stall_stop and still >= 4:
            stalled = True  # blocked (e.g. battery bottomed out in the socket): stop pushing
            break
        a = e / 0.1
        m = float(np.max(np.abs(a)))
        if m > vmax:
            a = a * (vmax / m)
        w = _rot_err(f.tcp)
        r = -10.0 * rot_gain * w
        rn = float(np.linalg.norm(r))
        if rn > 0.3:
            r = r * (0.3 / rn)
        if float(np.linalg.norm(w)) > 0.35:
            r = np.zeros(3)  # guard: do not chase a large orientation error
        prev = p
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), float(grip))
    f = robot.now()
    p = np.asarray(f.tcp[:3], dtype=float)
    res = {"goal": [round(float(x), 4) for x in goal], "reached": [round(float(x), 4) for x in p],
           "err": round(float(np.linalg.norm(goal - p)), 4), "steps": n,
           "rot_err": round(float(np.linalg.norm(_rot_err(f.tcp))), 3), "stalled": stalled}
    if log is not None:
        log.append(res)
    return res


def _hold(robot, grip, n, log=None, until_fingers=None):
    """Hold the measured pose (zero arm command) while the gripper moves; stop early when the
    finger sum stops changing (if until_fingers) after at least 3 steps."""
    prev = _fing(robot.now())
    k = 0
    for k in range(n):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(grip))
        cur = _fing(robot.now())
        if until_fingers and k >= 3 and abs(cur - prev) < 0.0005:
            break
        prev = cur
    if log is not None:
        log.append({"hold": k + 1, "fingers": round(_fing(robot.now()), 4)})
    return k + 1


def _z_up(robot, grip, z, log):
    p = np.asarray(robot.now().tcp[:3], dtype=float)
    if p[2] < z - 0.02:
        _goto(robot, [p[0], p[1], z], grip, vmax=1.0, tol=0.01, settle=0.01, max_steps=30, log=log)


# ---------------- decision ----------------

def perceive_decide(obs, goal, memory, last):
    """Stage machine over 3 batteries: pick -> insert(observe lamp) -> regrasp -> return -> press."""
    mem = dict(memory)
    cur = obs.current
    print("decide step=%s stage=%s battery=%s lamp=%s/%s tcp=%s fingers=%.4f nframes=%d last=%s" % (
        cur.step, memory.get("stage"), memory.get("battery"), memory.get("lamp_results"), memory.get("lamp_counts"),
        [round(float(v), 3) for v in cur.tcp],
        _fing(cur), len(obs.frames), (last or "")[:600]))
    for k in ("stage", "battery", "slots", "socket", "button", "lamp_uv", "lamp_results", "lamp_counts",
              "retries", "notes", "decisions"):
        mem.setdefault(k, None)
    mem["decisions"] = (mem["decisions"] or 0) + 1
    notes = mem["notes"] or ""

    if mem["stage"] is None:
        dets = _detect_batteries(obs.frames[0])
        if len(dets) > 3:
            dets = sorted(dets, key=lambda d: -d[3])[:3]
            dets.sort(key=lambda d: -d[0])
        if len(dets) == 0:
            mem["notes"] = notes + "no batteries detected;"
            mem["stage"] = None
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 2}}}
        if len(dets) < 3:
            # complete a 3-battery column from the rigid 0.0498 m spacing
            base = dets[0]
            full = [[base[0] - 0.0498 * i, base[1], base[2], 0] for i in range(3)]
            for d in dets:
                i = int(round((base[0] - d[0]) / 0.0498))
                if 0 <= i < 3:
                    full[i] = d
            notes += "only %d detections;" % len(dets)
            dets = full
        m = np.array([[d[0], d[1]] for d in dets]).mean(0)
        slots = [[round(d[0] - DET_BIAS[0], 4), round(d[1] - DET_BIAS[1], 4)] for d in dets]
        sock = [round(float(m[0] + SOCKET_OFF[0]), 4), round(float(m[1] + SOCKET_OFF[1]), 4)]
        btn = [round(float(m[0] + BUTTON_OFF[0]), 4), round(float(m[1] + BUTTON_OFF[1]), 4)]
        K = np.asarray(cur.front_K, dtype=float)
        E = np.asarray(cur.front_extrinsic, dtype=float)
        uv = _project([sock[0] + LAMP_OFF[0], sock[1] + LAMP_OFF[1], LAMP_OFF[2]], K, E)
        base_cnt = _lamp_count(obs.frames[0], uv)
        mem.update(stage="pick", battery=0, slots=slots, socket=sock, button=btn,
                   lamp_uv=[round(uv[0], 2), round(uv[1], 2)], lamp_results=[], lamp_counts=[base_cnt],
                   retries=0)
        notes += "dets=%s;" % [[round(d[0], 3), round(d[1], 3), d[3]] for d in dets]
        mem["notes"] = notes[-1500:]
        s = slots[0]
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": s[0], "y": s[1]}}}

    stage = mem["stage"]
    i = int(mem["battery"])
    slots, sock, btn = mem["slots"], mem["socket"], mem["button"]
    retries = int(mem["retries"] or 0)
    hold = _holding(cur)
    z = float(cur.tcp[2])

    if stage == "pick":
        if hold:
            mem.update(stage="insert", retries=0)
            mem["notes"] = (notes + "b%d picked;" % i)[-1500:]
            return {"memory": mem, "act": {"tool": "act_insert", "args": {"x": sock[0], "y": sock[1]}}}
        retries += 1
        mem["retries"] = retries
        mem["notes"] = (notes + "b%d pick miss f=%.4f;" % (i, _fing(cur)))[-1500:]
        s = slots[i]
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": s[0], "y": s[1]}}}

    if stage == "insert":
        # observe lamp over all frames of the insert action
        uv = mem["lamp_uv"]
        cnt = max([_lamp_count(f, uv) for f in obs.frames] + [0])
        lit = cnt >= 40
        res = list(mem["lamp_results"] or [])
        res.append("lit" if lit else "unlit")
        cnts = list(mem["lamp_counts"] or [])
        cnts.append(cnt)
        mem.update(stage="regrasp", lamp_results=res, lamp_counts=cnts, retries=0)
        mem["notes"] = (notes + "b%d inserted lamp=%d;" % (i, cnt))[-1500:]
        return {"memory": mem, "act": {"tool": "act_regrasp", "args": {"x": sock[0], "y": sock[1]}}}

    if stage == "regrasp":
        if hold:
            mem.update(stage="return", retries=0)
            mem["notes"] = (notes + "b%d regrasped;" % i)[-1500:]
            s = slots[i]
            return {"memory": mem, "act": {"tool": "act_return", "args": {"x": s[0], "y": s[1]}}}
        retries += 1
        mem["retries"] = retries
        mem["notes"] = (notes + "b%d regrasp miss f=%.4f;" % (i, _fing(cur)))[-1500:]
        return {"memory": mem, "act": {"tool": "act_regrasp", "args": {"x": sock[0], "y": sock[1]}}}

    if stage == "return":
        mem.update(stage="press", retries=0)
        mem["notes"] = (notes + "b%d returned;" % i)[-1500:]
        return {"memory": mem, "act": {"tool": "act_press", "args": {"x": btn[0], "y": btn[1], "depth": Z_PRESS}}}

    if stage == "press":
        i += 1
        mem["battery"] = i
        if i >= 3:
            # all checked; episode should have ended at the final press. Press again a bit deeper.
            mem.update(stage="press", battery=3, retries=retries + 1)
            mem["notes"] = (notes + "extra press;")[-1500:]
            return {"memory": mem, "act": {"tool": "act_press",
                                           "args": {"x": btn[0], "y": btn[1], "depth": Z_PRESS - 0.005 * min(retries + 1, 2)}}}
        mem.update(stage="pick", retries=0)
        mem["notes"] = (notes + "pressed;")[-1500:]
        s = slots[i]
        return {"memory": mem, "act": {"tool": "act_pick", "args": {"x": s[0], "y": s[1]}}}

    mem["notes"] = (notes + "unknown stage;")[-1500:]
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 2}}}


# ---------------- actions ----------------

def act_wait(robot, n=2):
    """Hold still for n steps (gripper unchanged-open)."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"waited": int(n)}


def act_pick(robot, x, y):
    """Move above slot (x,y), descend with pre-closed gripper, close on the battery, lift to transit height."""
    log = []
    _z_up(robot, G_OPEN if _fing(robot.now()) > 0.03 else G_PRE, Z_SAFE - 0.02, log)
    _goto(robot, [x, y, Z_SAFE], G_PRE, vmax=1.0, tol=0.006, settle=0.004, max_steps=45, log=log)
    _goto(robot, [x, y, 0.10], G_PRE, vmax=1.0, tol=0.004, settle=0.003, max_steps=30, log=log)
    _goto(robot, [x, y, Z_GRASP], G_PRE, vmax=0.5, tol=0.003, settle=0.0015, max_steps=30, log=log, stall_stop=True)
    _hold(robot, G_CLOSE, 10, log=log, until_fingers=True)
    _goto(robot, [x, y, Z_SAFE], G_CLOSE, vmax=1.0, tol=0.008, settle=0.006, max_steps=30, log=log)
    f = robot.now()
    return {"fingers": round(_fing(f), 4), "holding": _holding(f), "log": log}


def act_insert(robot, x, y):
    """Carry the held battery above the socket (x,y), lower it in, open, lift and wait (lamp observation)."""
    log = []
    _z_up(robot, G_CLOSE, Z_SAFE, log)
    _goto(robot, [x, y, Z_SAFE], G_CLOSE, vmax=0.8, tol=0.005, settle=0.003, max_steps=45, log=log)
    _goto(robot, [x, y, 0.10], G_CLOSE, vmax=0.8, tol=0.003, settle=0.002, max_steps=30, log=log)
    _goto(robot, [x, y, Z_SOCK_RELEASE], G_CLOSE, vmax=0.4, tol=0.003, settle=0.0015, max_steps=30, log=log, stall_stop=True)
    _hold(robot, G_OPEN, 10, log=log, until_fingers=True)
    _goto(robot, [x, y, Z_OBSERVE], G_OPEN, vmax=1.0, tol=0.006, settle=0.004, max_steps=25, log=log)
    _hold(robot, G_REGRASP, 10, log=log)
    f = robot.now()
    return {"fingers": round(_fing(f), 4), "log": log}


def act_regrasp(robot, x, y):
    """Descend on the battery in the socket with a half-open gripper, close, lift to transit height."""
    log = []
    _goto(robot, [x, y, 0.10], G_REGRASP, vmax=1.0, tol=0.004, settle=0.003, max_steps=30, log=log)
    _goto(robot, [x, y, Z_REGRASP], G_REGRASP, vmax=0.5, tol=0.003, settle=0.0015, max_steps=30, log=log, stall_stop=True)
    _hold(robot, G_CLOSE, 10, log=log, until_fingers=True)
    _goto(robot, [x, y, Z_SAFE], G_CLOSE, vmax=0.8, tol=0.008, settle=0.006, max_steps=30, log=log)
    f = robot.now()
    return {"fingers": round(_fing(f), 4), "holding": _holding(f), "log": log}


def act_return(robot, x, y):
    """Carry the battery back above its slot (x,y), lower it, release, lift."""
    log = []
    _z_up(robot, G_CLOSE, Z_SAFE, log)
    _goto(robot, [x, y, Z_SAFE], G_CLOSE, vmax=0.8, tol=0.005, settle=0.003, max_steps=45, log=log)
    _goto(robot, [x, y, 0.10], G_CLOSE, vmax=0.8, tol=0.003, settle=0.002, max_steps=30, log=log)
    _goto(robot, [x, y, Z_SLOT_RELEASE], G_CLOSE, vmax=0.4, tol=0.003, settle=0.0015, max_steps=30, log=log, stall_stop=True)
    _hold(robot, G_RELEASE, 8, log=log, until_fingers=True)
    _goto(robot, [x, y, Z_SAFE], G_RELEASE, vmax=1.0, tol=0.008, settle=0.006, max_steps=30, log=log)
    f = robot.now()
    return {"fingers": round(_fing(f), 4), "log": log}


def act_press(robot, x, y, depth=0.054):
    """Close the gripper fully, move above the button (x,y), press down to TCP z=depth, lift back up."""
    log = []
    _hold(robot, G_CLOSE, 6, log=log, until_fingers=True)
    _goto(robot, [x, y, Z_SAFE], G_CLOSE, vmax=1.0, tol=0.005, settle=0.003, max_steps=45, log=log)
    _goto(robot, [x, y, float(depth)], G_CLOSE, vmax=1.0, tol=0.004, settle=0.002, max_steps=30, log=log, stall_stop=True)
    _goto(robot, [x, y, Z_SAFE], G_CLOSE, vmax=1.0, tol=0.008, settle=0.006, max_steps=30, log=log)
    return {"log": log}
