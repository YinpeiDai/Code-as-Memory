"""Closed-loop Cartesian servo tools for the OSC_POSE delta controller."""

SIDE_R = [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
DOWN_R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


def _aa2R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _R2aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    th = math.acos(max(-1.0, min(1.0, c)))
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        k = V[:, int(np.argmax(w))]
        return k * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def _target_R(ori):
    if ori == "side":
        return np.array(SIDE_R)
    if ori == "down":
        return np.array(DOWN_R)
    if ori is None:
        return None
    return _aa2R(ori)


def _gap(frame):
    f = frame.fingers
    return float(f[0] - f[1])


def act_run(robot, segments):
    """Execute a list of motion segments with closed-loop servoing on the measured TCP.

    Each segment is a dict:
      {"pos": [x, y, z] or null, "ori": "side" | "down" | [ax, ay, az] | null, "grip": -1 open / 1 close / 0 hold,
       "tol": metres (default 0.004), "rtol": rad (default 0.05), "max": steps (default 150),
       "gain": position gain per metre (default 10), "vmax": max normalized speed (default 1.0), "name": label}
    or a wait segment {"wait": n_steps, "grip": g, "name": label} which holds position.
    Returns per-segment measured outcomes (reached flag, steps, final position/rotation error, finger gap)
    and the final tcp and finger gap."""
    report = []
    total = 0
    for seg in segments:
        g = float(seg.get("grip", 0))
        name = seg.get("name", "")
        if "wait" in seg:
            n = int(seg["wait"])
            for _ in range(n):
                robot.step(0, 0, 0, 0, 0, 0, g)
            total += n
            fr = robot.now()
            report.append({"name": name, "steps": n, "gap": round(_gap(fr), 4)})
            continue
        tgt = None if seg.get("pos") is None else np.array(seg["pos"], dtype=float)
        Rt = _target_R(seg.get("ori"))
        tol = float(seg.get("tol", 0.004))
        rtol = float(seg.get("rtol", 0.05))
        mx = int(seg.get("max", 150))
        gain = float(seg.get("gain", 10.0))
        vmax = float(seg.get("vmax", 1.0))
        reached = False
        n = 0
        perr = 0.0
        rerr = 0.0
        min_gap = 1.0
        while n < mx:
            fr = robot.now()
            tcp = np.array(fr.tcp, dtype=float)
            a = np.zeros(6)
            perr = 0.0
            rerr = 0.0
            if tgt is not None:
                e = tgt - tcp[:3]
                perr = float(np.linalg.norm(e))
                v = gain * e
                nv = float(np.max(np.abs(v)))
                if nv > vmax:
                    v = v * (vmax / nv)
                a[:3] = v
            if Rt is not None:
                Rc = _aa2R(tcp[3:6])
                er = _R2aa(Rt @ Rc.T)
                rerr = float(np.linalg.norm(er))
                w = 2.0 * er
                nw = float(np.max(np.abs(w)))
                if nw > 0.5:
                    w = w * (0.5 / nw)
                a[3:] = w
            if perr < tol and rerr < rtol:
                reached = True
                break
            robot.step(a[0], a[1], a[2], a[3], a[4], a[5], g)
            n += 1
            min_gap = min(min_gap, _gap(robot.now()))
        total += n
        fr = robot.now()
        report.append({"name": name, "reached": reached, "steps": n, "perr": round(perr, 4),
                       "rerr": round(rerr, 3), "gap": round(_gap(fr), 4), "min_gap": round(min_gap, 4)})
    fr = robot.now()
    return {"segments": report, "total_steps": total,
            "tcp": [round(float(v), 4) for v in fr.tcp[:3]], "gap": round(_gap(fr), 4)}
