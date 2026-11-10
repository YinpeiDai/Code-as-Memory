"""StopCube policy: watch the oscillating cube, hover over the button, press it as the cube
reaches the target disc for the N-th time (N parsed from the goal)."""

ORDINALS = {"first": 1, "once": 1, "second": 2, "twice": 2, "third": 3, "thrice": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10}

HOVER_Z = 0.153
# Commanded TCP z around the button trigger, derived from the mean of the 25 expert
# demonstrations (offset -24 .. +7 frames relative to the frame the cube stopped).
PRESS_PROFILE = [0.154, 0.153, 0.153, 0.152, 0.151, 0.149, 0.146, 0.142, 0.137, 0.132, 0.126, 0.12,
                 0.112, 0.104, 0.096, 0.087, 0.077, 0.067, 0.058, 0.05, 0.043, 0.036, 0.03, 0.024,
                 0.019, 0.015, 0.012, 0.009, 0.007, 0.006, 0.005, 0.005]
PROFILE_ZERO = 24          # index of the trigger frame inside PRESS_PROFILE
PRESS_XY_OFFSET = (-0.002, 0.003)   # expert press waypoint minus perceived button-top centroid
KNOWN_PERIODS = (120.0, 160.0, 240.0)
APPROACH_STEPS = 45


def _parse_count(goal):
    g = str(goal).lower().replace("-", " ")
    words = [w.strip(".,;:!?") for w in g.split()]
    for i, w in enumerate(words):
        if w == "time" or w == "times":
            if i > 0:
                p = words[i - 1]
                if p in ORDINALS:
                    return ORDINALS[p]
                digits = "".join(ch for ch in p if ch.isdigit())
                if digits:
                    return int(digits)
    for w in words:
        if w in ORDINALS and w not in ("one",):
            return ORDINALS[w]
    for w in words:
        digits = "".join(ch for ch in w if ch.isdigit())
        if digits and len(digits) == len(w.rstrip("stndrdth")) and int(digits) > 0:
            return int(digits)
    return 1


def _fit_motion(ts, xs, ys, target_xy):
    """Fit the cube's oscillation along its principal axis: p(s) = a sin ws + b cos ws + c.

    Returns dict(period, a, b, c, axis, rms) or None when there is too little data.
    """
    ts = np.asarray(ts, dtype=float)
    P = np.stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)], 1)
    ok = np.isfinite(P).all(1)
    ts = ts[ok]
    P = P[ok]
    if len(ts) < 12:
        return None
    C = P - P.mean(0)
    if float(np.abs(C).max()) < 0.03:
        return None
    ax = np.linalg.svd(C, full_matrices=False)[2][0]
    if ax[1] < 0 or (abs(ax[1]) < 1e-6 and ax[0] < 0):
        ax = -ax
    p = (P - np.asarray(target_xy, dtype=float)) @ ax
    keep = np.ones(len(p), dtype=bool)
    best = None
    for _ in range(2):
        best = None
        for T in np.arange(90.0, 330.0, 0.5):
            w = 2 * math.pi / T
            M = np.stack([np.sin(w * ts), np.cos(w * ts), np.ones_like(ts)], 1)
            co = np.linalg.lstsq(M[keep], p[keep], rcond=None)[0]
            r = M @ co - p
            rms = float(np.sqrt(np.mean(r[keep] ** 2)))
            if best is None or rms < best[0]:
                best = (rms, T, co, r)
        keep = np.abs(best[3]) < max(0.02, 3 * best[0])
        if keep.sum() < 12:
            break
    rms, T, co, r = best
    for K in KNOWN_PERIODS:
        if abs(T - K) / K < 0.04:
            w = 2 * math.pi / K
            M = np.stack([np.sin(w * ts), np.cos(w * ts), np.ones_like(ts)], 1)
            co2 = np.linalg.lstsq(M[keep], p[keep], rcond=None)[0]
            rms2 = float(np.sqrt(np.mean((M @ co2 - p)[keep] ** 2)))
            if rms2 < rms * 1.5 + 0.002:
                T, co, rms = K, co2, rms2
            break
    return {"period": float(T), "a": float(co[0]), "b": float(co[1]), "c": float(co[2]),
            "axis": [float(ax[0]), float(ax[1])], "rms": float(rms), "n": int(keep.sum())}


def _crossings(fit, s0=0.0, s1=1400.0):
    """Times (in frame-step units) at which the fitted cube passes the target (p = 0)."""
    w = 2 * math.pi / fit["period"]
    s = np.arange(s0, s1, 0.25)
    p = fit["a"] * np.sin(w * s) + fit["b"] * np.cos(w * s)
    out = []
    for i in range(1, len(s)):
        if p[i - 1] == 0.0 or (p[i - 1] < 0) != (p[i] < 0):
            f = p[i - 1] / (p[i - 1] - p[i]) if p[i - 1] != p[i] else 0.0
            out.append(float(s[i - 1] + f * (s[i] - s[i - 1])))
    return out


def _min_jerk(a, b, n):
    out = []
    for k in range(1, n + 1):
        u = k / float(n)
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        out.append([a[i] + (b[i] - a[i]) * s for i in range(len(a))])
    return out


def _stamp(frame, counter):
    s = getattr(frame, "step", None)
    try:
        v = float(s)
    except (TypeError, ValueError):
        return float(counter)
    if not math.isfinite(v):
        return float(counter)
    return v


def perceive_decide(obs, goal, memory, last):
    """Single-shot policy: run act_stop_cube once with N parsed from the goal, then hold."""
    n = _parse_count(goal)
    done = bool(memory.get("pressed")) if isinstance(memory, dict) else False
    if done and isinstance(last, str) and '"error"' in last:
        done = False          # our tool reported a perception failure before moving to press; retry
    mem = {"count": n, "pressed": True}
    if done:
        return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 60}}}
    return {"memory": mem, "act": {"tool": "act_stop_cube", "args": {"count": n}}}


def act_hold(robot, steps=60):
    """Hold the current TCP pose with the gripper closed for `steps` control steps."""
    f = robot.now()
    tcp = [float(v) for v in f.tcp]
    for _ in range(int(steps)):
        robot.step(tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5], -1.0)
    return {"held": int(steps)}


def act_stop_cube(robot, count=1):
    """Locate target disc and button (front RGB-D), hover over the button while tracking the
    cube, fit its oscillation, and press the button timed to the count-th target pass."""
    f0 = robot.now()
    sc = _scene(f0.front_rgb, f0.front_depth, f0.front_K, f0.front_extrinsic)
    if sc["target"] is None or sc["button"] is None:
        return {"error": "target or button not found", "target": sc["target"], "button": sc["button"]}
    target = [float(sc["target"][0]), float(sc["target"][1])]
    button = [float(sc["button"][0]), float(sc["button"][1])]
    scene = {"target": target, "button": button}
    press = [button[0] + PRESS_XY_OFFSET[0], button[1] + PRESS_XY_OFFSET[1]]
    tcp0 = [float(v) for v in f0.tcp]
    rpy = tcp0[3:6]
    ts, xs, ys = [], [], []
    counter = [0]

    def record(frames):
        for fr in frames:
            xy, n = _cube_xy(fr.front_rgb, fr.front_depth, fr.front_K, fr.front_extrinsic, target, button)
            ts.append(_stamp(fr, counter[0]))
            xs.append(xy[0] if xy is not None else float("nan"))
            ys.append(xy[1] if xy is not None else float("nan"))

    record([f0])
    seen = [0]

    def do_step(x, y, z, g):
        robot.step(x, y, z, rpy[0], rpy[1], rpy[2], g)
        counter[0] += 1
        fr = robot.frames()
        new = fr[seen[0]:]
        seen[0] = len(fr)
        if not new:
            new = [robot.now()]
        record(new)

    # 1. approach the hover pose above the button
    start = tcp0[:3]
    goal_pose = [press[0], press[1], HOVER_Z]
    for p in _min_jerk(start, goal_pose, APPROACH_STEPS):
        do_step(p[0], p[1], p[2], 1.0)
    for _ in range(4):
        do_step(press[0], press[1], HOVER_Z, -1.0)
    # 2. hover, refit and wait for the planned trigger
    fit = None
    trigger = None
    steps_taken = 0
    while steps_taken < 1250:
        now = ts[-1]
        if fit is None or steps_taken % 5 == 0:
            fit = _fit_motion(ts, xs, ys, target) or fit
            if fit is not None:
                cr = _crossings(fit, 0.0, 1400.0)
                if len(cr) >= count:
                    trigger = cr[count - 1]
        if trigger is not None and now >= trigger - PROFILE_ZERO:
            break
        do_step(press[0], press[1], HOVER_Z, -1.0)
        steps_taken += 1
    if trigger is None:
        return {"error": "could not fit cube motion", "n_obs": len(ts)}
    # 3. press, following the expert-derived z profile aligned to the trigger frame
    late = 0
    prev = -1
    lag0 = int(round(ts[-1] - trigger)) + PROFILE_ZERO
    while True:
        k = int(round(ts[-1] - trigger)) + PROFILE_ZERO
        k = min(max(k, prev + 1), prev + 2)
        if k >= len(PRESS_PROFILE):
            break
        k = max(k, 0)
        prev = k
        do_step(press[0], press[1], PRESS_PROFILE[k], -1.0)
        late += 1
        if late > 60:
            break
    zlow = float(robot.now().tcp[2])
    # 4. rise back above the button and hold
    cur = [press[0], press[1], PRESS_PROFILE[-1]]
    for p in _min_jerk(cur, [press[0], press[1], HOVER_Z], 30):
        do_step(p[0], p[1], p[2], -1.0)
    for _ in range(20):
        do_step(press[0], press[1], HOVER_Z, -1.0)
    return {"target": target, "button": button, "period": fit["period"], "fit_rms": fit["rms"],
            "trigger_step": trigger, "count": count, "start_lag": lag0, "z_at_press_end": zlow}
