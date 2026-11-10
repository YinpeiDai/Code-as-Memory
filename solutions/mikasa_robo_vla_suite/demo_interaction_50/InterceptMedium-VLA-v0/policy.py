"""InterceptMedium policy: track the rolling red ball with the front/wrist RGB-D
cameras, hover above it, match its velocity, descend with open fingers straddling
it (closing axis ~ world y), grasp, carry it on the table to the bullseye target
centre, release and retreat.  All perception is colour masks + depth
back-projection with the published intrinsics/extrinsics."""

BALL_R = 0.02          # ball radius measured from demos (sphere fit, z=0.0202)
Z_HOVER = 0.068        # TCP height while tracking (fingertips ~0.01 below TCP)
Z_GRASP = 0.021        # TCP height for grasping (ball equator 0.02)
G_CMD = 0.048          # steady TCP metres per step per unit command (demo fit)
ALPHA = 0.6            # first-order velocity response per step (demo fit)
VMAX = 0.042           # max desired TCP speed per axis, m/step
VZMAX = 0.036
KP = 0.5               # relative-position gain (fraction of error per step)
CARRY_V = 0.042        # carry speed (m/step); ball stayed locked to TCP within 0.5 mm at 0.031 m/step in R002-R032
WAIT_POINT = [-0.2, -0.35]
XY_LO = [-0.45, -0.62]
XY_HI = [0.25, 0.35]


def _cam_points(depth, K, E, ys, xs):
    z = depth[ys, xs].astype(float) / 1000.0
    rx = (xs + 0.5 - K[0, 2]) / K[0, 0]
    ry = (ys + 0.5 - K[1, 2]) / K[1, 1]
    return np.stack([rx * z, ry * z, z], 1), np.stack([rx, ry, np.ones_like(rx)], 1)


def _ball_cam(rgb, depth, K, E, minpix=4):
    """Sphere fit of red-ball pixels; returns (world xyz list, npix) or (None, n)."""
    im = rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = (r > 70) & (g < 45) & (b < 45) & (r > 3 * g + 20) & (depth > 0)
    ys, xs = np.nonzero(m)
    n = len(ys)
    if n < minpix:
        return None, n
    P, d = _cam_points(depth, K, E, ys, xs)
    z = P[:, 2]
    zmin = np.percentile(z, 10)
    ok = z < zmin + 2.5 * BALL_R
    P, d = P[ok], d[ok]
    if len(P) < minpix:
        return None, n
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    c = P.mean(0) + (2.0 / 3.0) * BALL_R * d.mean(0)
    for _ in range(6):
        diff = P - c
        nn = np.linalg.norm(diff, axis=1)
        nn = np.maximum(nn, 1e-6)
        res = nn - BALL_R
        J = -diff / nn[:, None]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        c = c + step
    R = E[:, :3]
    t = E[:, 3]
    w = R.T @ (c - t)
    if not np.all(np.isfinite(w)):
        return None, n
    return [float(w[0]), float(w[1]), float(w[2])], int(len(P))


def _ball_obs(frame, pred=None, gate=0.05):
    """Fused ball estimate from front and wrist cameras, gated by prediction."""
    ests = []
    for cam in ("front", "wrist"):
        rgb = getattr(frame, cam + "_rgb")
        dep = getattr(frame, cam + "_depth")
        K = getattr(frame, cam + "_K")
        E = getattr(frame, cam + "_extrinsic")
        p, n = _ball_cam(rgb, dep, K, E)
        if p is None:
            continue
        if p[2] < -0.01 or p[2] > 0.25:
            continue
        if pred is not None:
            if math.hypot(p[0] - pred[0], p[1] - pred[1]) > gate:
                continue
        wgt = float(n) if cam == "front" else float(n) * 0.2
        ests.append((p, wgt, cam, n))
    if not ests:
        return None
    W = sum(e[1] for e in ests)
    p = [sum(e[0][i] * e[1] for e in ests) / W for i in range(3)]
    return {"p": p, "src": "+".join(e[2] for e in ests), "n": [e[3] for e in ests]}


def _target_obs(frame):
    """Bullseye centre on the table from the front camera (red ring colour)."""
    im = frame.front_rgb.astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    dep = frame.front_depth
    m = (np.abs(r - 204) < 22) & (np.abs(g - 71) < 18) & (np.abs(b - 76) < 18) & (dep > 0)
    ys, xs = np.nonzero(m)
    if len(ys) < 30:
        return None
    P, _ = _cam_points(dep, frame.front_K, frame.front_extrinsic, ys, xs)
    E = frame.front_extrinsic
    W = (P - E[:, 3]) @ E[:, :3]
    W = W[np.abs(W[:, 2]) < 0.02]
    if len(W) < 30:
        return None
    c = np.median(W, 0)
    W = W[np.hypot(W[:, 0] - c[0], W[:, 1] - c[1]) < 0.15]
    c = W.mean(0)
    rad = float(np.percentile(np.hypot(W[:, 0] - c[0], W[:, 1] - c[1]), 98))
    return {"xy": [float(c[0]), float(c[1])], "npix": int(len(W)), "radius": rad}


def _fit_vel(track, now_step, nmax=4):
    """Least-squares position/velocity of the ball at now_step from recent samples."""
    pts = [p for p in track if p[3] < 0.032][-nmax:]
    if len(pts) == 0:
        return None, None
    if len(pts) == 1:
        p = pts[-1]
        return [p[1], p[2]], [0.0, 0.0]
    ts = np.array([p[0] for p in pts], dtype=float)
    X = np.array([[p[1], p[2]] for p in pts])
    A = np.stack([np.ones_like(ts), ts - now_step], 1)
    coef = np.linalg.lstsq(A, X, rcond=None)[0]
    return [float(coef[0, 0]), float(coef[0, 1])], [float(coef[1, 0]), float(coef[1, 1])]


def _axis_cmd(u, vg):
    """Command for desired next-step displacement u given current per-step velocity vg."""
    c = (vg + (u - vg) / ALPHA) / G_CMD
    return max(-1.0, min(1.0, c))


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def perceive_decide(obs, goal, memory, last):
    """Stage machine: intercept (track+grasp) -> carry+release -> hold."""
    f = obs.current
    stage = memory.get("stage") or "start"
    target = memory.get("target")
    log = memory.get("log") or ""
    attempts = memory.get("grasp_attempts") or 0
    if target is None or len(target) < 2:
        t = _target_obs(f)
        if t is not None:
            target = t["xy"]
            log += "target %.3f,%.3f npix %d r %.3f; " % (target[0], target[1], t["npix"], t["radius"])
        else:
            target = [0.0, 0.0]
            log += "target not found; "
    if isinstance(last, dict):
        grasped = bool(last.get("grasped"))
    else:
        txt = str(last) if last is not None else ""
        grasped = ('"grasped": true' in txt) or ("'grasped': True" in txt)
    if stage in ("start", "intercept"):
        if stage == "intercept":
            if grasped:
                stage = "carry"
            else:
                attempts += 1
        if stage != "carry":
            stage = "intercept"
    elif stage == "carry":
        stage = "hold"
    if stage == "intercept" and attempts >= 4:
        stage = "hold"
    if isinstance(last, str):
        log += "last:" + last[:200] + "; "
    else:
        log += "last:" + str(last)[:200] + "; "
    log = log[-1500:]
    mem = {"stage": stage, "target": target, "grasp_attempts": attempts, "log": log}
    vel0 = [0.0, 0.0, 0.0]
    if len(obs.frames) >= 2:
        a = obs.frames[-2].tcp
        vel0 = [float(f.tcp[j]) - float(a[j]) for j in range(3)]
    if stage == "intercept":
        act = {"tool": "act_intercept", "args": {"target_x": target[0], "target_y": target[1], "vel0": vel0}}
    elif stage == "carry":
        act = {"tool": "act_carry", "args": {"target_x": target[0], "target_y": target[1], "vel0": vel0}}
    else:
        act = {"tool": "act_hold", "args": {"steps": 60}}
    return {"memory": mem, "act": act}


def act_intercept(robot, target_x, target_y, vel0=None, max_steps=60):
    """Track the rolling ball at hover height, match its velocity, descend with open
    fingers around it and close. Returns grasped flag and a per-stage log."""
    track = []           # [step, x, y, z]
    stage = "track"
    aligned = 0
    close_k = 0
    miss = 0
    prev_tcp = None
    events = []
    f0 = robot.now()
    k0 = int(f0.step)
    for i in range(max_steps):
        f = robot.now()
        k = int(f.step)
        tcp = [float(v) for v in f.tcp]
        fing = [float(v) for v in f.fingers]
        if prev_tcp is None:
            vg = [float(v) for v in vel0] if vel0 else [0.0, 0.0, 0.0]
        else:
            vg = [tcp[j] - prev_tcp[j] for j in range(3)]
        prev_tcp = tcp
        pb, vb = _fit_vel(track, k)
        pred = None
        if pb is not None:
            pred = pb
        ob = _ball_obs(f, pred, gate=(0.06 if miss < 3 else 1.0))
        if ob is not None:
            track.append([k, ob["p"][0], ob["p"][1], ob["p"][2]])
            track = track[-8:]
            miss = 0
            pb, vb = _fit_vel(track, k)
        else:
            miss += 1
        have = pb is not None and len(track) >= 2
        # desired xy
        if have:
            bx, by = pb[0], pb[1]
            vbx, vby = vb[0], vb[1]
            ux = vbx + KP * (bx - tcp[0])
            uy = vby + KP * (by - tcp[1])
            relx, rely = tcp[0] - bx, tcp[1] - by
            rvx, rvy = vg[0] - vbx, vg[1] - vby
        else:
            ux = KP * (WAIT_POINT[0] - tcp[0])
            uy = KP * (WAIT_POINT[1] - tcp[1])
            relx = rely = rvx = rvy = 9.0
        # workspace clamp of the next position
        nx = _clip(tcp[0] + ux, XY_LO[0], XY_HI[0])
        ny = _clip(tcp[1] + uy, XY_LO[1], XY_HI[1])
        ux = _clip(nx - tcp[0], -VMAX, VMAX)
        uy = _clip(ny - tcp[1], -VMAX, VMAX)
        grip = 1.0
        if stage == "track":
            zdes = Z_HOVER
            if have and abs(relx) < 0.007 and abs(rely) < 0.010 and abs(rvx) < 0.006 and abs(rvy) < 0.006 and tcp[2] < Z_HOVER + 0.02:
                aligned += 1
            else:
                aligned = 0
            if aligned >= 1:
                stage = "descend"
                events.append(["descend", k, round(relx, 4), round(rely, 4), round(rvx, 4), round(rvy, 4)])
        if stage == "descend":
            zdes = Z_GRASP
            if have and (abs(relx) > 0.03 or abs(rely) > 0.019) and tcp[2] > 0.035:
                stage = "track"
                zdes = Z_HOVER
                events.append(["abort", k, round(relx, 4), round(rely, 4)])
            elif tcp[2] < Z_GRASP + 0.008:
                stage = "close"
                close_k = 0
                events.append(["close", k, round(relx, 4), round(rely, 4), round(tcp[2], 4)])
        if stage == "close":
            zdes = Z_GRASP
            grip = -1.0
            close_k += 1
            if close_k >= 3:
                mf = min(fing)
                if mf > 0.010:
                    events.append(["grasped", k, [round(v, 4) for v in fing]])
                    return {"grasped": True, "steps": k - k0, "step": k, "events": events,
                            "ball": [round(v, 4) for v in (track[-1][1:] if track else [])],
                            "tcp": [round(v, 4) for v in tcp]}
                events.append(["missed", k, [round(v, 4) for v in fing]])
                return {"grasped": False, "steps": k - k0, "step": k, "events": events,
                        "ball": [round(v, 4) for v in (track[-1][1:] if track else [])],
                        "tcp": [round(v, 4) for v in tcp]}
        uz = _clip(0.6 * (zdes - tcp[2]), -VZMAX, VZMAX)
        cx = _axis_cmd(ux, vg[0])
        cy = _axis_cmd(uy, vg[1])
        cz = _axis_cmd(uz, vg[2])
        robot.step(cx, cy, cz, 0.0, 0.0, 0.0, grip)
    return {"grasped": False, "steps": max_steps, "events": events, "timeout_in_act": True}


def act_carry(robot, target_x, target_y, vel0=None, max_steps=40):
    """Carry the grasped ball along the table to the target centre, release, retreat."""
    prev_tcp = None
    events = []
    phase = "carry"
    rel_k = 0
    off = [0.0, 0.0]
    for i in range(max_steps):
        f = robot.now()
        k = int(f.step)
        tcp = [float(v) for v in f.tcp]
        if prev_tcp is None:
            vg = [float(v) for v in vel0] if vel0 else [0.0, 0.0, 0.0]
        else:
            vg = [tcp[j] - prev_tcp[j] for j in range(3)]
        prev_tcp = tcp
        ob = _ball_obs(f, [tcp[0], tcp[1]], gate=0.05)
        if ob is not None and phase == "carry":
            off = [0.7 * off[0] + 0.3 * (ob["p"][0] - tcp[0]), 0.7 * off[1] + 0.3 * (ob["p"][1] - tcp[1])]
        gx = target_x - off[0]
        gy = target_y - off[1]
        grip = -1.0
        if phase == "carry":
            dx, dy = gx - tcp[0], gy - tcp[1]
            dist = math.hypot(dx, dy)
            sp = min(CARRY_V, 0.45 * dist)
            ux = sp * dx / dist if dist > 1e-6 else 0.0
            uy = sp * dy / dist if dist > 1e-6 else 0.0
            zdes = Z_GRASP
            if dist < 0.012 and math.hypot(vg[0], vg[1]) < 0.006:
                phase = "release"
                events.append(["release", k, round(dist, 4), [round(v, 4) for v in tcp]])
        if phase == "release":
            ux = uy = 0.0
            zdes = Z_GRASP
            grip = 1.0
            rel_k += 1
            if rel_k > 3:
                phase = "retreat"
        if phase == "retreat":
            ux = uy = 0.0
            grip = 1.0
            zdes = 0.12
        uz = _clip(0.6 * (zdes - tcp[2]), -VZMAX, VZMAX)
        robot.step(_axis_cmd(ux, vg[0]), _axis_cmd(uy, vg[1]), _axis_cmd(uz, vg[2]), 0.0, 0.0, 0.0, grip)
    return {"done": True, "events": events}


def act_hold(robot, steps=60):
    """Hold position with the gripper open (waits for the episode to end)."""
    for i in range(steps):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": steps}
