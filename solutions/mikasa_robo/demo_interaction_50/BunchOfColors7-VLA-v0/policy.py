"""BunchOfColors7-VLA policy: remember the cued cube colours, touch every cube of
those colours in the response ring (TCP hover above the cube centre), then press
the centre button with a closed gripper and lift clearly above it.

Perception uses the fixed front RGB-D camera only: depth is back-projected to
world points, pixels 8-70 mm above the table are grouped into blobs, small blobs
are cubes, the colour of a cube is the median RGB of its top face matched to a
palette of nine colours seen in all 50 demonstrations, and the button is the
pink cap near the scene centre.
"""

PALETTE = {
    "red": [231, 5, 5], "darkred": [169, 4, 4], "green": [5, 231, 5],
    "blue": [4, 4, 231], "cyan": [4, 231, 231], "teal": [4, 169, 169],
    "magenta": [231, 4, 231], "orange": [231, 169, 4], "yellow": [231, 231, 4],
}
# empirical offset between the front-camera top-face centre estimate and the
# expert TCP touch position (50 demos, 350 touches: mean -6.5 mm x, +2.5 mm y)
CUBE_OFFSET = [-0.0065, 0.0025]
# press point relative to pink-cap bounding-box centre (expert presses were
# 0.0-0.03 m in +x of the cap centre; cap radius ~0.027 m)
BUTTON_OFFSET = [0.010, 0.003]
Z_TRAVEL = 0.15
Z_TOUCH = 0.045
Z_PRESS = 0.047
Z_LIFT = 0.24
STEP_LIMIT = 400


def _world_points(depth, K, E):
    d = np.asarray(depth, dtype=float) / 1000.0
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    C = np.stack([x, y, d], -1)
    R = np.asarray(E)[:, :3]
    t = np.asarray(E)[:, 3]
    return (C - t) @ R


def _blobs(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return []
    W = mask.shape[1]
    keys = set((pts[:, 0] * W + pts[:, 1]).tolist())
    seen = set()
    out = []
    for k in (pts[:, 0] * W + pts[:, 1]).tolist():
        if k in seen:
            continue
        stack = [k]
        seen.add(k)
        comp = []
        while stack:
            c = stack.pop()
            comp.append(c)
            r, q = divmod(c, W)
            for n in (c - W, c + W, c - 1 if q > 0 else -1, c + 1 if q < W - 1 else -1):
                if n >= 0 and n in keys and n not in seen:
                    seen.add(n)
                    stack.append(n)
        comp = np.array(comp)
        out.append(np.stack([comp // W, comp % W], 1))
    return out


def _classify(rgb):
    best, bd = None, 1e9
    for name, ref in PALETTE.items():
        d = float(np.sqrt(np.sum((np.asarray(rgb, float) - np.asarray(ref, float)) ** 2)))
        if d < bd:
            best, bd = name, d
    if bd > 60:
        lev = ["H" if c > 200 else ("M" if c > 120 else "L") for c in rgb]
        best = "unk_" + "".join(lev)
    return best, bd


def _detect(frame):
    """Cubes (list of dicts with colour and xy) and button xy in one front frame."""
    P = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgb = np.asarray(frame.front_rgb).astype(int)
    z = P[..., 2]
    valid = np.asarray(frame.front_depth) > 0
    m = valid & (z > 0.008) & (z < 0.07) & (P[..., 0] > -0.45) & (P[..., 0] < 0.35) & (np.abs(P[..., 1]) < 0.5)
    cubes = []
    button = None
    for b in _blobs(m):
        n = len(b)
        if n < 15:
            continue
        pts = P[b[:, 0], b[:, 1]]
        col = rgb[b[:, 0], b[:, 1]]
        if n > 300:
            pink = (col[:, 0] > col[:, 2] + 40) & (col[:, 1] < col[:, 0] - 40)
            if pink.sum() >= 10:
                pp = pts[pink]
                top = pp[pp[:, 2] > pp[:, 2].max() - 0.003]
                c = (top.min(0) + top.max(0)) / 2
                button = [float(c[0]), float(c[1]), float(pp[:, 2].max())]
            continue
        zmax = pts[:, 2].max()
        top = pts[:, 2] > zmax - 0.004
        med = np.median(col[top], 0)
        name, dist = _classify(med)
        c = pts[top][:, :2].mean(0)
        cubes.append({"color": name, "dist": round(dist, 1), "xy": [float(c[0]), float(c[1])],
                      "n": int(n), "zmax": float(zmax)})
    return cubes, button


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_err(rpy_target, rpy_cur):
    """Rotation vector w (base frame) with R_target = exp(w) R_cur."""
    M = _rpy_to_R(*rpy_target) @ _rpy_to_R(*rpy_cur).T
    c = max(-1.0, min(1.0, (np.trace(M) - 1) / 2))
    th = math.acos(c)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / 2
    if th < 1e-6:
        return v
    return v * th / math.sin(th)


def _servo(robot, target, grip, rpy0, tol=0.004, vtol=0.004, max_steps=40,
           gate_z=None, k=1.0, min_steps=0, rate=0.35, zcap_low=0.6, xy_gate=None):
    """Close the loop on measured TCP toward target xyz.
    - gate_z: while z < gate_z and rising, hold xy at the start xy (rise first).
    - xy_gate=(z_below, xy_tol): do not descend below z_below while xy error > xy_tol.
    - commands are acceleration-limited (|increase| <= rate per step) to avoid the
      lateral transients seen with step changes; deceleration is not limited.
    - below z=0.10 the vertical command is capped at zcap_low.
    Returns (steps, final tcp, reached)."""
    n = 0
    prev = np.asarray(robot.now().tcp, float)
    start_xy = prev[:2].copy()
    a_prev = np.zeros(3)
    reached = False
    tgt = np.asarray(target, float)
    while n < max_steps:
        tcp = np.asarray(robot.now().tcp, float)
        err = tgt - tcp[:3]
        speed = float(np.linalg.norm(tcp[:3] - prev[:3]))
        if n >= min_steps and np.all(np.abs(err) < tol) and speed < vtol:
            reached = True
            break
        a_des = np.clip(k * err / 0.1, -1.0, 1.0)
        gated = gate_z is not None and tcp[2] < gate_z and tgt[2] > tcp[2]
        if gated:
            a_des[0] = float(np.clip(k * (start_xy[0] - tcp[0]) / 0.1, -0.3, 0.3))
            a_des[1] = float(np.clip(k * (start_xy[1] - tcp[1]) / 0.1, -0.3, 0.3))
        if xy_gate is not None and tcp[2] < xy_gate[0] + 0.01 and float(np.hypot(err[0], err[1])) > xy_gate[1]:
            a_des[2] = max(a_des[2], float(np.clip(k * (xy_gate[0] - tcp[2]) / 0.1, -1.0, 1.0)))
        if tcp[2] < 0.10:
            a_des[2] = float(np.clip(a_des[2], -zcap_low, zcap_low))
        a = np.zeros(3)
        for j in range(3):
            if a_des[j] * a_prev[j] >= 0 and abs(a_des[j]) <= abs(a_prev[j]):
                a[j] = a_des[j]
            else:
                a[j] = a_prev[j] + float(np.clip(a_des[j] - a_prev[j], -rate, rate))
        r = np.zeros(3)
        if rpy0 is not None and not gated:
            w = _rot_err(rpy0, tcp[3:6])
            if np.linalg.norm(w) > 0.03:
                r = np.clip(-3.0 * w, -0.3, 0.3)
        prev = tcp
        a_prev = a
        robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), grip)
        n += 1
    return n, np.asarray(robot.now().tcp, float), reached


def _hold(robot, grip, steps):
    for _ in range(steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip)


def _plan(start_xy, cubes, end_xy):
    """Order cubes for the shortest start -> cubes -> end path (exact for <= 8)."""
    n = len(cubes)
    if n == 0:
        return []
    pts = [np.asarray(c, float) for c in cubes]
    s = np.asarray(start_xy, float)
    e = np.asarray(end_xy, float)
    if n > 8:
        order, cur, left = [], s, list(range(n))
        while left:
            j = min(left, key=lambda i: float(np.linalg.norm(pts[i] - cur)))
            order.append(j)
            left.remove(j)
            cur = pts[j]
        return order
    D = np.array([[float(np.linalg.norm(a - b)) for b in pts] for a in pts])
    ds = [float(np.linalg.norm(p - s)) for p in pts]
    de = [float(np.linalg.norm(p - e)) for p in pts]
    # Held-Karp DP over subsets
    INF = 1e9
    full = (1 << n)
    dp = [[INF] * n for _ in range(full)]
    par = [[-1] * n for _ in range(full)]
    for i in range(n):
        dp[1 << i][i] = ds[i]
    for S in range(full):
        for i in range(n):
            v = dp[S][i]
            if v >= INF or not (S >> i) & 1:
                continue
            for j in range(n):
                if (S >> j) & 1:
                    continue
                T = S | (1 << j)
                c = v + D[i][j]
                if c < dp[T][j]:
                    dp[T][j] = c
                    par[T][j] = i
    S = full - 1
    last = min(range(n), key=lambda i: dp[S][i] + de[i])
    order = []
    while last >= 0:
        order.append(last)
        prev = par[S][last]
        S ^= (1 << last)
        last = prev
    return order[::-1]


def _new_memory():
    return {
        "stage": "cue", "scene_phase": "pre", "cue_colors": [], "cue_frames": 0,
        "cue_step": -1, "resp_step": -1, "resp_frames": 0, "resp_colors": [],
        "targets": [], "target_colors": [], "distractors": [], "button": [],
        "touched": [], "touch_attempts": 0, "press_count": 0, "rpy0": [],
        "notes": "",
    }


def _process_frames(mem, frames):
    """Run the cue/gap/response state machine on new frames."""
    for f in frames:
        if mem["scene_phase"] == "resp_done":
            break
        cubes, button = _detect(f)
        n = len(cubes)
        ph = mem["scene_phase"]
        if ph == "pre":
            if n > 0:
                mem["scene_phase"] = "cue"
                ph = "cue"
            else:
                continue
        if ph == "cue":
            if n > 0:
                cols = sorted(set(mem["cue_colors"]) | set(c["color"] for c in cubes))
                mem["cue_colors"] = cols
                mem["cue_frames"] += 1
                if mem["cue_step"] < 0:
                    mem["cue_step"] = int(f.step)
            else:
                mem["scene_phase"] = "gap"
            continue
        if ph == "gap":
            if n > 0:
                mem["scene_phase"] = "resp"
                mem["resp_frames"] = 0
                ph = "resp"
            else:
                continue
        if ph == "resp":
            if n == 0:
                continue
            if mem["resp_frames"] > 0 and n == len(mem["resp_colors"]):
                mem["resp_frames"] += 1
            else:
                mem["resp_frames"] = 1
            mem["resp_colors"] = [c["color"] for c in cubes]
            mem["resp_step"] = int(f.step)
            tg, tc, ds = [], [], []
            for c in cubes:
                xy = [c["xy"][0] + CUBE_OFFSET[0], c["xy"][1] + CUBE_OFFSET[1]]
                if c["color"] in mem["cue_colors"]:
                    tg.append(xy)
                    tc.append(c["color"])
                else:
                    ds.append(xy)
            mem["targets"], mem["target_colors"], mem["distractors"] = tg, tc, ds
            if button is not None:
                mem["button"] = [button[0] + BUTTON_OFFSET[0], button[1] + BUTTON_OFFSET[1]]
            if mem["resp_frames"] >= 2:
                mem["scene_phase"] = "resp_done"
    return mem


def perceive_decide(obs, goal, memory, last):
    """Stage machine: cue -> wait for response ring -> touch each cued cube -> press button."""
    mem = dict(memory) if memory and memory.get("stage") else _new_memory()
    cur = obs.current
    tcp = [float(v) for v in cur.tcp]
    if not mem["rpy0"]:
        mem["rpy0"] = tcp[3:6]
    step = int(cur.step)
    if mem["stage"] == "cue":
        mem = _process_frames(mem, list(obs.frames))
        if mem["scene_phase"] == "resp_done":
            # plan touch order from current TCP to button
            btn = mem["button"] if mem["button"] else [-0.115, 0.0]
            order = _plan(tcp[:2], mem["targets"], btn)
            mem["targets"] = [mem["targets"][i] for i in order]
            mem["target_colors"] = [mem["target_colors"][i] for i in order]
            if not mem["button"]:
                mem["button"] = btn
                mem["notes"] += "button not detected, default used; "
            mem["stage"] = "touch"
            mem["notes"] += "cue=%s resp=%s; " % (",".join(mem["cue_colors"]), ",".join(mem["resp_colors"]))
        else:
            return {"memory": mem, "act": {"tool": "act_wait", "args": {
                "max_steps": 8, "seen_gap": mem["scene_phase"] in ("gap", "resp")}}}
    if mem["stage"] == "touch":
        # record outcome of previous touch
        if last and last.startswith("act_touch") and '"reached_touch": true' in last:
            if len(mem["touched"]) < len(mem["targets"]):
                mem["touched"] = mem["touched"] + [mem["target_colors"][len(mem["touched"])]]
            mem["touch_attempts"] = 0
        elif last and last.startswith("act_touch"):
            mem["touch_attempts"] += 1
            if mem["touch_attempts"] >= 2:
                mem["touched"] = mem["touched"] + ["skip_" + mem["target_colors"][len(mem["touched"])]]
                mem["touch_attempts"] = 0
        i = len(mem["touched"])
        if i < len(mem["targets"]):
            x, y = mem["targets"][i]
            retouch = "retouch" in mem["notes"]
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": x, "y": y, "z_touch": Z_TOUCH - (0.004 if retouch else 0.0),
                "z_travel": Z_TRAVEL, "hold": 3 if retouch else 2,
                "rpy0": mem["rpy0"]}}}
        mem["stage"] = "press"
    if mem["stage"] == "press":
        if last and last.startswith("act_press"):
            mem["press_count"] += 1
        # fallback: one press without native success -> some touch may not have
        # registered; redo every target slightly lower once, if the budget allows
        if mem["press_count"] == 1 and "retouch" not in mem["notes"] and step < STEP_LIMIT - 175:
            mem["notes"] += "retouch at step %d; " % step
            mem["touched"] = []
            mem["touch_attempts"] = 0
            mem["stage"] = "touch"
            x, y = mem["targets"][0]
            return {"memory": mem, "act": {"tool": "act_touch", "args": {
                "x": x, "y": y, "z_touch": Z_TOUCH - 0.004, "z_travel": Z_TRAVEL, "hold": 3,
                "rpy0": mem["rpy0"]}}}
        bx, by = mem["button"]
        # later presses: shift slightly toward the expert's typical press point
        dx = [0.0, 0.006, -0.006, 0.0][mem["press_count"] % 4]
        return {"memory": mem, "act": {"tool": "act_press", "args": {
            "x": bx + dx, "y": by, "z_press": Z_PRESS, "z_travel": Z_TRAVEL - 0.02,
            "z_lift": Z_LIFT, "hold": 5, "rpy0": mem["rpy0"]}}}
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"max_steps": 5}}}


def act_wait(robot, max_steps=8, seen_gap=False):
    """Hold still (gripper open) for up to max_steps steps; stop early once an
    empty frame (gap after the cue) has been followed by two frames with the same
    non-zero cube count (the response ring). Returns per-step cube counts."""
    counts = []
    for k in range(max_steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        cubes, _ = _detect(robot.now())
        counts.append(len(cubes))
        if len(cubes) == 0:
            seen_gap = True
        if seen_gap and len(counts) >= 2 and counts[-1] > 0 and counts[-1] == counts[-2]:
            break
    return {"steps": len(counts), "counts": counts, "seen_gap": seen_gap,
            "step": int(robot.now().step)}


def act_touch(robot, x, y, z_touch=0.048, z_travel=0.15, hold=2, rpy0=None):
    """Rise to z_travel, move above (x, y), descend to z_touch with the gripper
    open, hold `hold` steps. Returns measured bottom pose and step counts."""
    s0 = int(robot.now().step)
    tcp = np.asarray(robot.now().tcp, float)
    n1 = 0
    if np.hypot(tcp[0] - x, tcp[1] - y) > 0.02:
        n1, tcp, _ = _servo(robot, [x, y, z_travel], 1.0, rpy0, tol=0.012, vtol=1.0,
                            max_steps=40, gate_z=min(0.10, z_travel - 0.02))
    n2, tcp, ok = _servo(robot, [x, y, z_touch], 1.0, rpy0, tol=0.004, vtol=0.004, max_steps=30,
                         xy_gate=(0.09, 0.005))
    _hold(robot, 1.0, hold)
    tcp = np.asarray(robot.now().tcp, float)
    errxy = float(np.hypot(tcp[0] - x, tcp[1] - y))
    good = bool(errxy < 0.012 and tcp[2] < z_touch + 0.012)
    return {"reached_touch": good, "tcp": tcp[:3].round(4).tolist(), "xy_err": round(errxy, 4),
            "steps_travel": n1, "steps_descend": n2, "step": int(robot.now().step),
            "used": int(robot.now().step) - s0}


def act_press(robot, x, y, z_press=0.047, z_travel=0.13, z_lift=0.24, hold=4, rpy0=None):
    """Close the gripper, move above the button at (x, y), press down to z_press,
    hold, then lift to z_lift and wait there. The episode ends on native success."""
    s0 = int(robot.now().step)
    tcp = np.asarray(robot.now().tcp, float)
    n0 = 0
    if tcp[2] < 0.10:
        # leave the last cube vertically with the gripper still OPEN (closing next to
        # a cube pushed it in R021)
        n0, tcp, _ = _servo(robot, [tcp[0], tcp[1], 0.11], 1.0, None, tol=0.01, vtol=1.0,
                            max_steps=12, gate_z=0.12)
    n1 = 0
    if np.hypot(tcp[0] - x, tcp[1] - y) > 0.01:
        n1, tcp, _ = _servo(robot, [x, y, z_travel], -1.0, rpy0, tol=0.008, vtol=1.0,
                            max_steps=40, gate_z=0.10)
    fing = float(np.mean(robot.now().fingers))
    k = 0
    while fing > 0.005 and k < 6:
        _hold(robot, -1.0, 1)
        fing = float(np.mean(robot.now().fingers))
        k += 1
    # descend; stop when the cap bottoms out (z stalls below 0.062) or target reached
    n2 = 0
    zs = []
    while n2 < 25:
        tcp = np.asarray(robot.now().tcp, float)
        zs.append(float(tcp[2]))
        if tcp[2] < z_press + 0.003:
            break
        if len(zs) >= 3 and zs[-1] < 0.062 and zs[-3] - zs[-1] < 0.0015:
            break
        a = np.clip((np.array([x, y, z_press]) - tcp[:3]) / 0.1, -1.0, 1.0)
        a[2] = max(float(a[2]), -0.7)
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, -1.0)
        n2 += 1
    zmin = float(robot.now().tcp[2])
    for _ in range(hold):
        tcp = np.asarray(robot.now().tcp, float)
        a = np.clip((np.array([x, y, z_press]) - tcp[:3]) / 0.1, -1.0, 1.0)
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, -1.0)
        zmin = min(zmin, float(robot.now().tcp[2]))
    bottom = np.asarray(robot.now().tcp, float)[:3].round(4).tolist()
    n3, tcp, _ = _servo(robot, [x, y, z_lift], -1.0, rpy0, tol=0.006, vtol=0.004, max_steps=25)
    _hold(robot, -1.0, 6)
    return {"pressed_bottom": bottom, "zmin": round(zmin, 4), "lift_tcp": np.asarray(robot.now().tcp, float)[:3].round(4).tolist(),
            "steps": [n0, n1, k, n2, n3], "step": int(robot.now().step), "used": int(robot.now().step) - s0}
