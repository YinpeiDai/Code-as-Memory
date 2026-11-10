"""Closed-loop motion tools: P control of measured TCP pose toward waypoints."""

POS_GAIN = 10.0        # action per metre of error (expert used ~10)
ROT_GAIN = 2.0         # action per radian (action 1 = 0.5 rad target offset)
ROT_CLIP = 0.5
ROT_TOL = 0.06         # rad


def _pose_action(frame, target, R_t):
    tcp = np.asarray(frame.tcp, dtype=float)
    ep = np.asarray(target, dtype=float) - tcp[:3]
    R_c = _aa2mat(tcp[3:6])
    er = _mat2aa(R_t @ R_c.T)
    # uniform scaling keeps straight-line translation and geodesic rotation paths
    a_p = POS_GAIN * ep
    mp = float(np.max(np.abs(a_p)))
    if mp > 1.0:
        a_p = a_p / mp
    a_r = ROT_GAIN * er
    mr = float(np.max(np.abs(a_r)))
    if mr > ROT_CLIP:
        a_r = a_r * (ROT_CLIP / mr)
    return a_p, a_r, float(np.linalg.norm(ep)), float(np.linalg.norm(er))


def act_move(robot, waypoints, gripper=-1.0, max_steps_per_waypoint=150):
    """Follow waypoints [x, y, z, 'down'|'door', pos_tol] with P control on the measured TCP,
    holding the gripper command (+1 close, -1 open) throughout. Returns reached flags and pose."""
    reached = []
    steps = 0
    f = robot.now()
    for wp in waypoints:
        target = [float(wp[0]), float(wp[1]), float(wp[2])]
        R_t = _rot_named(wp[3] if len(wp) > 3 else "down")
        tol = float(wp[4]) if len(wp) > 4 else 0.008
        ok = False
        settle = 0
        best = 1e9
        since_best = 0
        for k in range(int(max_steps_per_waypoint)):
            a_p, a_r, dp, dr = _pose_action(f, target, R_t)
            if dp < tol and dr < ROT_TOL:
                settle += 1
                if settle >= 2:
                    ok = True
                    break
            score = dp + 0.05 * dr
            if score < best - 0.0005:
                best = score
                since_best = 0
            else:
                since_best += 1
                if since_best >= 25:   # stalled (contact or constraint): give up on this waypoint
                    break
            f = robot.step(float(a_p[0]), float(a_p[1]), float(a_p[2]),
                           float(a_r[0]), float(a_r[1]), float(a_r[2]), float(gripper))
            steps += 1
        reached.append(ok)
    tcp = np.asarray(f.tcp, dtype=float)
    return {"reached": reached, "steps": steps, "tcp": [round(float(v), 4) for v in tcp[:3]],
            "finger_gap": round(_gap(f), 4)}


def act_gripper(robot, close=True, steps=20):
    """Close (+1) or open (-1) the gripper for `steps` steps with zero arm motion command."""
    f = robot.now()
    g = 1.0 if close else -1.0
    for k in range(int(steps)):
        # zero motion (controller holds the current pose), as the expert did while actuating
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
    return {"finger_gap": round(_gap(f), 4), "tcp": [round(float(v), 4) for v in np.asarray(f.tcp)[:3]]}


def act_wait(robot, steps=40):
    """Hold still with the gripper open for `steps` steps (lets the scorer settle)."""
    f = robot.now()
    for k in range(int(steps)):
        f = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"waited": int(steps), "finger_gap": round(_gap(f), 4)}
