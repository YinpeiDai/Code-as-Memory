"""InterceptFast policy: perceive rolling ball + target with the front RGB-D camera,
place an angled finger 'wall' on the ball line upstream of the target, let the ball
deflect toward the target, hold, and (fallback) push a resting ball into the target."""

BALL_R = 0.02          # ball radius (m), measured from demo depth (top of ball at z~0.04)
GOAL_R = 0.10          # visual target radius (m); success hypothesis: ball centre within ~0.1
WALL_YAW = -0.85       # gripper yaw (rad) with roll=pi, pitch=0 -> finger +x_h face normal at -49 deg
FACE_OFF = (0.032, 0.040)  # ball centre at contact in hand frame (x_h, y_h) relative to TCP (from demos)
AIM_DY = 0.09          # fallback contact offset
AIM_BIAS = 0.0         # added to modelled contact y
E_REST = 0.2           # restitution in the wall-deflection model (R001-R003 headings 21-48 deg at yaw -0.85)
YAW_RATE = 0.04        # achievable yaw change per step (rad), from demos
PHI_MIN = -1.0         # x face: yaw range (= wall normal angle)
PHI_MAX = -0.4
Y_PSI_MIN = 0.25       # y face: yaw range (= outgoing heading)
Y_PSI_MAX = 0.56       # cap: non-slide (bounce/partial) contacts by yaw: <=26deg 0/7, 27-34deg 1/10, 35-37deg 2/7, 38-41deg 6/13 (R007-R045)
FACES = ("x", "y")     # x: side face of the upstream finger; y: its outer face (cleaner slide, R007/R008)
X_FACE_COST = 10.0     # penalty (steps) for the x face: ball can be caught/kicked by the 2nd finger (R002/3/6/9)
OUTER_OFF = (-0.003, 0.075)  # ball centre at contact, outer face (hand frame); x_h aim -3 mm: successful impacts at x_h 0..0.003, R012 bounced at 0.008
SLIDE_K = 0.34         # y face: slide speed = SLIDE_K * vin * sin(yaw) (R007 0.35, R008 0.33)
X_SPEED_K = 0.25       # x face: out speed ~0.2-0.37 vin
BASE_XY = (-0.615, 0.0)
REACH_MIN = 0.28       # horizontal TCP distance from the base needed for a vertical gripper at z=0.02 (offline IK)
T_MARGIN = 2.0         # extra steps required before contact
LOCK_T = 5.0           # freeze wall angle when contact is this many steps away
XY_RATE = 0.05         # achievable horizontal TCP speed (m/step), demos ~0.04 per axis
Z_RATE = 0.035         # achievable descent speed (m/step)
STRIKE_Z = 0.020       # TCP height during the strike (ball centre z=0.02)
SAFE_Z = 0.075         # TCP height while travelling (fingertips above ball top 0.04)
PRIOR_DX = 0.205       # demo regularity: target x - ball start x
PRIOR_VX = 0.0012      # demo regularity: ball x drift per step
STEP_LIMIT = 60


def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    if ang > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(M)))
        ax = M[:, k] / math.sqrt(max(M[k, k], 1e-9))
        return ax / np.linalg.norm(ax) * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return w * ang


def _cam_points(frame, mask, prefix="front"):
    depth = getattr(frame, prefix + "_depth")
    K = getattr(frame, prefix + "_K")
    E = getattr(frame, prefix + "_extrinsic").astype(np.float64)
    vs, us = np.nonzero(mask & (depth > 0))
    if len(vs) == 0:
        return np.zeros((0, 3)), E
    z = depth[vs, us].astype(np.float64) / 1000.0
    x = (us + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (vs + 0.5 - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1)
    W = (P - E[:, 3]) @ E[:, :3]
    return W, E


def _detect_ball(frame):
    """Ball centre (world) from the front camera: saturated red pixels above the table."""
    rgb = frame.front_rgb
    r = rgb[..., 0].astype(np.int32); g = rgb[..., 1].astype(np.int32); b = rgb[..., 2].astype(np.int32)
    mask = (r > 90) & (g < 60) & (b < 60) & (r > 3 * g)
    W, E = _cam_points(frame, mask)
    if len(W) < 3:
        return None
    W = W[(W[:, 2] > 0.003) & (W[:, 2] < 0.09)]
    if len(W) < 3:
        return None
    m = W.mean(0)
    cam = -E[:, :3].T @ E[:, 3]
    ray = m - cam
    c = m + ray / np.linalg.norm(ray) * BALL_R * 2.0 / 3.0
    return [float(c[0]), float(c[1]), float(c[2]), int(len(W))]


def _detect_target(frame):
    """Target centre (world xy) from red/white ring pixels lying on the table."""
    rgb = frame.front_rgb
    r = rgb[..., 0].astype(np.int32); g = rgb[..., 1].astype(np.int32); b = rgb[..., 2].astype(np.int32)
    red = (r > 120) & (b >= g - 3) & (r - g > 70) & (g > 35)
    white = (r > 200) & (g > 200) & (b > 200) & (np.abs(r - b) < 12)
    W, E = _cam_points(frame, red | white)
    if len(W) < 50:
        return None
    W = W[np.abs(W[:, 2]) < 0.006]
    if len(W) < 50:
        return None
    m = W.mean(0)
    return [float(m[0]), float(m[1]), int(len(W))]


def _fit_line(track, upto_y=None):
    """Linear fits x(t), y(t) from on-table ball detections [t,x,y,z,n]."""
    pts = [p for p in track if 0.010 < p[3] < 0.032 and p[4] >= 12 and (upto_y is None or p[2] < upto_y)]
    if len(pts) < 3:
        return None
    # drop partially occluded / image-border detections (pixel count well below the typical count)
    med = float(np.median([p[4] for p in pts]))
    good = [p for p in pts if p[4] >= 0.8 * med]
    if len(good) >= 3:
        pts = good
    pts = pts[-12:]
    T = np.array([p[0] for p in pts], float)
    X = np.array([p[1] for p in pts], float)
    Y = np.array([p[2] for p in pts], float)
    A = np.stack([np.ones_like(T), T], 1)
    cx = np.linalg.lstsq(A, X, rcond=None)[0]
    cy = np.linalg.lstsq(A, Y, rcond=None)[0]
    vx = float(min(0.004, max(-0.002, cx[1])))
    if len(pts) < 5:
        vx = PRIOR_VX
        cx = [float(np.mean(X - vx * T)), vx]
    return {"x0": float(cx[0]), "vx": vx, "y0": float(cy[0]), "vy": float(cy[1]), "n": len(pts)}


def _x_at_y(line, y):
    if line["vy"] < 0.005:
        return None
    t = (y - line["y0"]) / line["vy"]
    return line["x0"] + line["vx"] * t, t


def _pose_cmd(tcp, target_xyz, R_des, kp=1.5, kr=1.5):
    p = np.array(tcp[:3], float)
    err = np.array(target_xyz, float) - p
    d = np.clip(kp * err / 0.1, -1.0, 1.0)
    Rm = _rpy_R(tcp[3], tcp[4], tcp[5])
    w = _rotvec(R_des @ Rm.T)
    rcmd = -kr * w / 0.1
    nr = float(np.linalg.norm(rcmd))
    if nr > 1.0:
        rcmd = rcmd / nr
    return [float(d[0]), float(d[1]), float(d[2]), float(rcmd[0]), float(rcmd[1]), float(rcmd[2])], float(np.linalg.norm(err[:2])), float(np.linalg.norm(w))


def _strike_tcp(ball_xy_contact, yaw, face="x"):
    """TCP xy so that the ball centre meets the chosen finger face.
    face 'x': +x_h side face of the +y_h finger (expert-proven), ball centre at FACE_OFF (hand frame).
    face 'y': outer (+y_h) face of the +y_h finger, ball centre at OUTER_OFF (hand frame)."""
    xh = np.array([math.cos(yaw), math.sin(yaw)])
    yh = np.array([math.sin(yaw), -math.cos(yaw)])   # hand y axis with roll = pi
    o = FACE_OFF if face == "x" else OUTER_OFF
    off = o[0] * xh + o[1] * yh
    return [ball_xy_contact[0] - off[0], ball_xy_contact[1] - off[1]]


def _face_yaw(phi, face):
    """Gripper yaw giving wall normal angle phi with the chosen face."""
    return phi if face == "x" else phi + math.pi / 2


def perceive_decide(obs, goal, memory, last):
    """Detect target/ball from the front camera, keep stage memory and choose the next act."""
    stage = memory.get("stage") or "start"
    target = memory.get("target")
    track = list(memory.get("ball_track") or [])
    for f in obs.frames:
        b = _detect_ball(f)
        if b is not None and not any(p[0] == f.step for p in track):
            track.append([int(f.step), b[0], b[1], b[2], b[3]])
    if target is None:
        tg = _detect_target(obs.current)
        if tg is not None:
            target = [tg[0], tg[1]]
    track = track[-40:]
    log = memory.get("log") or ""
    if last:
        log = (log + " | " + str(last)[:600])[-2500:]
    step = int(obs.current.step)
    if target is None:
        act = {"tool": "act_wait", "args": {"n": 1}}
        new_stage = stage
    elif stage == "start":
        act = {"tool": "act_intercept", "args": {"target": target, "track": track}}
        new_stage = "struck"
    else:
        ball = track[-1] if track else None
        if ball is not None and step < STEP_LIMIT - 8:
            act = {"tool": "act_push", "args": {"target": target, "ball": ball[1:3]}}
            new_stage = "pushing"
        else:
            act = {"tool": "act_wait", "args": {"n": 2}}
            new_stage = stage
    mem = {"stage": new_stage, "target": target, "ball_track": track, "log": log}
    return {"memory": mem, "act": act}


def act_wait(robot, n=1):
    """Hold the arm still (zero delta, gripper open) for n steps; returns the last step."""
    for _ in range(int(n)):
        robot.step(0, 0, 0, 0, 0, 0, 1.0)
    return {"step": int(robot.now().step)}


def _theta_out(phi):
    """Model: outgoing ball heading (rad from +x toward +y) after hitting a wall with
    horizontal normal angle phi while rolling along +y (restitution E_REST)."""
    s, c = math.sin(phi), math.cos(phi)
    k = 1.0 + E_REST
    return math.atan2(1.0 - k * s * s, -k * s * c)


def _heading(psi, face):
    """Outgoing ball heading (rad from +x toward +y) for gripper yaw psi.
    'y' face: the ball slides along the outer finger face, i.e. along +x_h -> heading = psi (R007/R008).
    'x' face: wall-reflection model with normal angle phi = psi (R001-R006, noisy)."""
    if face == "y":
        return psi
    return _theta_out(psi)


def _out_speed(psi, face, vin):
    if face == "y":
        return SLIDE_K * vin * max(0.05, math.sin(psi))
    return X_SPEED_K * vin


def _plan_one(line, target, tcp, step, ball_y_now, psi, face):
    yaw_now = tcp[5]
    vin = line["vy"] if line is not None and line["vy"] > 0.005 else 0.031
    th = _heading(psi, face)
    t_rem = 20.0
    xc = target[0] - PRIOR_DX
    y_c = target[1]
    dx_goal = PRIOR_DX
    for _ in range(3):
        if line is not None:
            xc = line["x0"] + line["vx"] * (step + t_rem)
        else:
            xc = target[0] - PRIOR_DX + PRIOR_VX * (step + t_rem)
        dx_goal = max(0.05, target[0] - xc)
        y_c = target[1] - dx_goal * math.tan(th) + AIM_BIAS
        if line is not None:
            t_rem = (y_c - ball_y_now) / vin
    dpsi = psi - yaw_now
    lim = YAW_RATE * max(0.0, t_rem - 0.5)
    psi_c = yaw_now + max(-lim, min(lim, dpsi))      # yaw actually reached at contact
    txy = _strike_tcp([xc, y_c], psi_c, face)
    t_need = max(abs(dpsi) / YAW_RATE,
                 math.hypot(txy[0] - tcp[0], txy[1] - tcp[1]) / XY_RATE,
                 max(0.0, tcp[2] - STRIKE_Z) / Z_RATE) + T_MARGIN
    travel = max(0.0, dx_goal / max(0.2, math.cos(th)) - GOAL_R)
    t_done = step + t_rem + travel / max(1e-4, _out_speed(psi, face, vin))
    reach = math.hypot(txy[0] - BASE_XY[0], txy[1] - BASE_XY[1])
    return {"psi": psi, "face": face, "psi_c": psi_c, "y_c": y_c, "x_c": xc, "t_rem": t_rem,
            "t_need": t_need, "tcp_xy": txy, "t_done": t_done, "reach": reach}


def _plan(line, target, tcp, step, ball_y_now, face=None):
    """Pick finger face and gripper yaw: feasible (arrive/rotate before the ball, reachable),
    then minimal predicted success step (+ penalty for the less predictable x face)."""
    cands = []
    faces = [face] if face else list(FACES)
    for f in faces:
        lo, hi = (Y_PSI_MIN, Y_PSI_MAX) if f == "y" else (PHI_MIN, PHI_MAX)
        for i in range(17):
            psi = lo + (hi - lo) * i / 16.0
            c = _plan_one(line, target, tcp, step, ball_y_now, psi, f)
            c["slack"] = c["t_rem"] - c["t_need"]
            if c["reach"] < REACH_MIN:
                c["slack"] -= 100.0
            c["cost"] = c["t_done"] + (X_FACE_COST if f == "x" else 0.0)
            cands.append(c)
    feas = [c for c in cands if c["slack"] >= 0]
    if feas:
        return min(feas, key=lambda c: c["cost"])
    return max(cands, key=lambda c: c["slack"])


def act_intercept(robot, target, track):
    """Closed-loop strike: track the ball line, place the angled finger wall at the
    contact point upstream of the target, wait for contact, hold while the ball rolls on.
    Returns measured stage progress (contact step, ball positions, distances)."""
    track = [list(p) for p in track]
    contact_step = None
    face = None
    phi_lock = None
    lowered = False
    plan = None
    plans = []
    hist = []
    stopped = 0
    min_d = 9.0
    reason = "budget"
    ready_step = None
    while True:
        f = robot.now()
        step = int(f.step)
        if step >= STEP_LIMIT - 1:
            reason = "step_limit"
            break
        b = _detect_ball(f)
        if b is not None and not any(p[0] == step for p in track):
            track.append([step, b[0], b[1], b[2], b[3]])
        tcp = [float(v) for v in f.tcp]
        if contact_step is None:
            line = _fit_line(track, upto_y=tcp[1] - 0.12)
            ball_y = track[-1][2] if track else -0.8
            if line is not None and track:
                ball_y = line["y0"] + line["vy"] * step
            if phi_lock is None:
                plan = _plan(line, target, tcp, step, ball_y, face)
                if face is None and line is not None and line["n"] >= 3:
                    face = plan["face"]          # commit the finger face once the ball line is known
                    phi_lock = plan["psi"]       # and keep the wall angle while it stays feasible
            else:
                plan = _plan_one(line, target, tcp, step, ball_y, phi_lock, face)
                if plan["t_rem"] - plan["t_need"] < -1.0 and plan["t_rem"] > LOCK_T:
                    alt = _plan(line, target, tcp, step, ball_y, face)
                    if alt["t_rem"] - alt["t_need"] >= 0.0:
                        plan = alt
                        phi_lock = alt["psi"]
            print("plan", step, face, round(plan["psi"], 2), round(plan["psi_c"], 2), round(plan["y_c"], 3), round(plan["t_done"], 1),
                  round(plan["t_rem"], 1), round(plan["t_need"], 1), [round(v, 3) for v in plan["tcp_xy"]],
                  "tcp", [round(v, 3) for v in tcp[:3]], round(tcp[5], 2), "ball", track[-1][1:3] if track else None)
            plans.append([step, plan["face"], round(plan["tcp_xy"][0], 4), round(plan["tcp_xy"][1], 4), round(plan["psi"], 3),
                          round(plan["y_c"], 3), round(plan["t_rem"], 1), line["n"] if line else 0])
            rec = [p for p in track if p[0] >= step - 3]
            if len(rec) >= 2 and line is not None:
                dt = max(1, rec[-1][0] - rec[0][0])
                vx_now = (rec[-1][1] - rec[0][1]) / dt
                vy_now = (rec[-1][2] - rec[0][2]) / dt
                near = abs(rec[-1][2] - plan["y_c"]) < 0.08
                if near and (vx_now > 0.004 or vy_now < 0.5 * line["vy"]):
                    contact_step = step
        if b is not None:
            dgoal = math.hypot(b[0] - target[0], b[1] - target[1])
            min_d = min(min_d, dgoal)
            hist.append([step, round(b[0], 3), round(b[1], 3), round(dgoal, 3)])
        goal_xy = plan["tcp_xy"]
        R_des = _rpy_R(math.pi, 0.0, plan["psi"])
        cmd, exy, erot = _pose_cmd(tcp, [goal_xy[0], goal_xy[1], STRIKE_Z], R_des)
        lowered = lowered or tcp[2] < SAFE_Z - 0.015
        if exy > 0.03 and contact_step is None and plan["t_rem"] > 3.0 and not (lowered and plan["t_rem"] < 8.0):
            cmd, exy, erot = _pose_cmd(tcp, [goal_xy[0], goal_xy[1], SAFE_Z], R_des)
        elif ready_step is None and exy < 0.01 and abs(tcp[2] - STRIKE_Z) < 0.01:
            ready_step = step
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], 1.0)
        if contact_step is not None:
            post = [p for p in track if p[0] >= contact_step + 2]
            if step - contact_step >= 5 and len(post) >= 3:
                p0, p1 = post[-3], post[-1]
                v = np.array([p1[1] - p0[1], p1[2] - p0[2]]) / max(1, p1[0] - p0[0])
                rel = np.array([target[0] - p1[1], target[1] - p1[2]])
                sp = float(np.linalg.norm(v))
                if sp > 1e-4:
                    along = float(rel @ v) / sp
                    perp = abs(float(rel[0] * v[1] - rel[1] * v[0])) / sp
                    if along < 0.0 or perp > 0.095:
                        reason = "miss_predicted"
                        break
                    to_rim = along - math.sqrt(max(0.0, GOAL_R ** 2 - perp ** 2))
                    if step + to_rim / sp > STEP_LIMIT - 2:
                        reason = "too_slow"
                        break
            if step - contact_step > 40:
                reason = "post_contact_timeout"
                break
            if len(track) >= 3:
                a, c = track[-3], track[-1]
                sp = math.hypot(c[1] - a[1], c[2] - a[2]) / max(1, c[0] - a[0])
                stopped = stopped + 1 if sp < 0.002 else 0
                if stopped >= 3:
                    reason = "ball_stopped"
                    break
        elif track and track[-1][2] > target[1] + 0.25:
            reason = "ball_passed"
            break
    fin = robot.now()
    return {"reason": reason, "contact_step": contact_step, "ready_step": ready_step,
            "plan_last": plans[-1] if plans else None, "plans": plans[::4][-6:],
            "final_tcp": [round(float(v), 3) for v in fin.tcp], "min_ball_goal_dist": round(min_d, 3),
            "ball_hist": hist[-10:], "step": int(fin.step)}


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(1e-9, ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(a + ab * t - p))


def act_push(robot, target, ball):
    """Fallback: closed-loop push of the (possibly rolling) ball toward the target with closed fingers.
    Every step: detect the ball, estimate its velocity, aim at the pre-push point behind the predicted
    ball position; travel low unless the path would cross the ball (then at SAFE_Z); push along
    ball->target. Returns stage log and minimum ball-target distance."""
    T = np.array(target[:2], float)
    b = np.array(ball[:2], float)
    hist = []
    stage = "approach"
    log = []
    yaw0 = float(robot.now().tcp[5])
    R_des = _rpy_R(math.pi, 0.0, yaw0)
    dmin = 9.0
    while True:
        f = robot.now()
        step = int(f.step)
        if step >= STEP_LIMIT - 1:
            break
        det = _detect_ball(f)
        if det is not None:
            hist.append([step, det[0], det[1]])
            hist = hist[-4:]
            b = np.array(det[:2], float)
        vel = np.zeros(2)
        if len(hist) >= 2:
            dt = max(1, hist[-1][0] - hist[0][0])
            vel = np.array([hist[-1][1] - hist[0][1], hist[-1][2] - hist[0][2]]) / dt
        tcp = [float(v) for v in f.tcp]
        p = np.array(tcp[:2])
        dist = float(np.linalg.norm(T - b))
        dmin = min(dmin, dist)
        if stage == "push":
            u = (T - b) / max(dist, 1e-6)
            goal = b + u * 0.04
            cmd, exy, erot = _pose_cmd(tcp, [goal[0], goal[1], STRIKE_Z], R_des, kp=0.8)
            rel = b - p
            if abs(rel[0] * u[1] - rel[1] * u[0]) > 0.035 or rel @ u < -0.01:
                stage = "approach"
        if stage == "approach":
            k = 0.0
            bk = b
            for _ in range(3):
                bk = b + vel * k
                uk = (T - bk) / max(1e-6, float(np.linalg.norm(T - bk)))
                behind = bk - uk * (BALL_R + 0.025)
                k = min(8.0, float(np.linalg.norm(behind - p)) / 0.035)
            high = _seg_dist(b, p, behind) < 0.035 or _seg_dist(bk, p, behind) < 0.035
            d_b = float(np.linalg.norm(behind - p))
            z = SAFE_Z if (high and d_b > 0.02) else STRIKE_Z
            cmd, exy, erot = _pose_cmd(tcp, [behind[0], behind[1], z], R_des)
            if d_b < 0.015 and tcp[2] < STRIKE_Z + 0.012:
                stage = "push"
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], -1.0)
        log.append([step, stage, round(dist, 3)])
        print("push", step, stage, [round(v, 3) for v in b], [round(v, 4) for v in vel], round(dist, 3), [round(v, 3) for v in tcp[:3]])
    return {"stages": log[-5:], "dmin": round(dmin, 3), "step": int(robot.now().step)}
