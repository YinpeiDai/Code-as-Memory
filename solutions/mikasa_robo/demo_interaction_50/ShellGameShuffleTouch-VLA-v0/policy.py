"""ShellGameShuffleTouch policy: remember the ball's cup, track the three mugs through the
shuffle with the front RGB-D camera, then touch the tracked mug by straddling its rim wall
with the open gripper and closing the fingers on the wall."""

TOUCH_PLAN = [
    # name, TCP offset (dx, dy) from the mug-top centre, pre-height, hold height, close?
    # The mugs are INVERTED (solid top at z~0.106, top disk radius ~0.036, side radius ~0.051), so the open
    # fingers (along world y, inner pads at +-0.038, tips 9 mm below the TCP) straddle the side wall
    # tangentially: TCP over the +x wall, the palm rests on the mug top at TCP z~0.07, and closing squeezes
    # the wall chord between the two fingers (grasp-like contact, which is what triggers success).
    {"name": "wall_straddle_px", "dx": 0.047, "dy": 0.0, "z_pre": 0.125, "z_low": 0.068, "close": True},
    {"name": "wall_straddle_nx", "dx": -0.047, "dy": 0.0, "z_pre": 0.125, "z_low": 0.068, "close": True},
    {"name": "outer_low", "dx": 0.058, "dy": 0.0, "z_pre": 0.125, "z_low": 0.025, "close": True},
]


def _get(memory, key, default):
    v = memory.get(key) if isinstance(memory, dict) else None
    return default if v is None else v


def perceive_decide(obs, goal, memory, last):
    """Track ball cup through the shuffle from all frames since the last action; then
    choose the next touch attempt. Returns all declared memory fields and one act."""
    stage = _get(memory, "stage", "observe")
    ball = list(_get(memory, "ball_xy", []))
    cups = [list(c) for c in _get(memory, "cups", [])]
    ball_cup = int(_get(memory, "ball_cup", -1))
    moved = int(_get(memory, "moved_frames", 0))
    still = int(_get(memory, "still_frames", 0))
    tcp0 = list(_get(memory, "tcp0", []))
    target = list(_get(memory, "target", []))
    attempt = int(_get(memory, "attempt", 0))
    log = str(_get(memory, "log", ""))
    cur = obs.current
    if not tcp0:
        tcp0 = [float(v) for v in cur.tcp]
    tcp_moved = float(np.linalg.norm(np.asarray(cur.tcp[:3]) - np.asarray(tcp0[:3])))

    if stage == "observe":
        last_P = None
        last_a = None
        for fr in obs.frames:
            P = _mug_points(fr)
            if len(P) < MUG_MIN_POINTS:
                if not cups:
                    b = _ball_point(fr)
                    if b is not None:
                        ball = [float(b[0]), float(b[1]), float(b[2])]
                continue
            if not cups:
                C, a = _init_cups(P)
                cups = C.tolist()
                if ball:
                    d = [abs(c[1] - ball[1]) + abs(c[0] - ball[0]) for c in cups]
                    ball_cup = int(np.argmin(d))
                log += "cups@%s ball_cup=%d ball=%s; " % (fr.step, ball_cup, [round(v, 3) for v in ball])
                last_P, last_a = P, a
                continue
            C, a = _kmeans(P, np.array(cups), 5)
            m = float(np.abs(C - np.array(cups)).max())
            cups = C.tolist()
            if m > STILL_EPS:
                moved += 1
                still = 0
            else:
                still += 1
            last_P, last_a = P, a
        if cups and ball_cup < 0 and moved >= 8:
            ball_cup = 1
            log += "ball unseen: default middle cup; "
        done = cups and ball_cup >= 0 and moved >= 8 and still >= 1 and (tcp_moved > 0.002 or still >= 2)
        if not done:
            probe = -0.3 if tcp_moved < 0.002 else 0.0
            mem = _mem("observe", ball, cups, ball_cup, moved, still, tcp0, target, attempt, log)
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"max_steps": 8, "probe_dz": probe}}}
        if last_P is None:
            last_P = _mug_points(cur)
            C, last_a = _kmeans(last_P, np.array(cups), 3)
        cx, cy, rad, n = _rim_center(last_P, last_a, ball_cup)
        target = [cx, cy, rad]
        log += "shuffle done@%s moved=%d still=%d target=(%.3f,%.3f) R=%.3f n=%d; " % (
            cur.step, moved, still, cx, cy, rad, n)
        stage = "touch"

    if stage == "touch" and attempt < len(TOUCH_PLAN):
        plan = TOUCH_PLAN[attempt]
        mem = _mem("touch", ball, cups, ball_cup, moved, still, tcp0, target, attempt + 1, log)
        args = {"cx": target[0], "cy": target[1], "dx": plan["dx"], "dy": plan["dy"],
                "z_pre": plan["z_pre"], "z_low": plan["z_low"], "close": plan["close"],
                "yaw": float(tcp0[5]), "label": plan["name"]}
        return {"memory": mem, "act": {"tool": "act_touch", "args": args}}
    mem = _mem("hold", ball, cups, ball_cup, moved, still, tcp0, target, attempt, log)
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"max_steps": 10, "probe_dz": 0.0}}}


def _mem(stage, ball, cups, ball_cup, moved, still, tcp0, target, attempt, log):
    return {"stage": stage, "ball_xy": ball, "cups": cups, "ball_cup": int(ball_cup),
            "moved_frames": int(moved), "still_frames": int(still), "tcp0": tcp0,
            "target": target, "attempt": int(attempt), "log": log[-1500:]}
