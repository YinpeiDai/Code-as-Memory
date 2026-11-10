"""InsertPeg policy: imitate the task-provided demonstration video.

Evidence (25 assigned recordings): the execution scene is reset to exactly the
same configuration as the instruction video, and the expert execution repeats
the demo trajectory (TCP deviation <= 5.6 cm max, typically < 1 mm).  The
policy therefore reconstructs the demonstrated TCP path (from the demo frames'
tcp, or from the wrist-camera extrinsic through a fixed hand-eye transform)
and the grasp moment (finger widths), then tracks that path with a small
velocity lead and a final push along the demonstrated insertion direction.
"""

# Hand-eye: wrist camera pose expressed in the TCP frame (derived from recordings,
# std 0 over all 25 episodes).  cam_R_in_tcp maps camera axes to TCP axes.
_CAM_OFF = [0.0465, -0.0200, -0.0674]
_CAM_R = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
_OPEN_W = 0.04          # finger width when open
_CLOSE_THR = 0.03       # finger width below which the demo gripper is closed
_LEAD = 0.5             # velocity lead fitted on expert commands vs measured tcp
_PUSH_EXTRA = 0.08     # extra push beyond the demonstrated end (m)
_PUSH_STEP = 0.0015     # push increment per control step (m)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _R_to_rpy(R):
    p = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    r = math.atan2(float(R[2, 1]), float(R[2, 2]))
    y = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return [r, p, y]


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _tcp_from_wrist_extrinsic(E):
    """World TCP pose [x,y,z,r,p,y] from a 3x4 world-to-wrist-camera matrix."""
    E = np.asarray(E, dtype=float)
    Rw, tw = E[:, :3], E[:, 3]
    Rc = Rw.T
    pc = -Rw.T @ tw
    Rt = Rc @ np.array(_CAM_R).T
    pt = pc - Rt @ np.array(_CAM_OFF)
    return [float(pt[0]), float(pt[1]), float(pt[2])] + _R_to_rpy(Rt)


def _frame_pose(f, source):
    if source == "tcp":
        v = getattr(f, "tcp", None)
        if v is None:
            return None
        v = np.asarray(v, dtype=float).ravel()
        if v.size < 6 or not np.isfinite(v[:6]).all():
            return None
        return [float(x) for x in v[:6]]
    E = getattr(f, "wrist_extrinsic", None)
    if E is None:
        return None
    E = np.asarray(E, dtype=float)
    if E.shape != (3, 4) or not np.isfinite(E).all():
        return None
    return _tcp_from_wrist_extrinsic(E)


def _finger(f):
    v = getattr(f, "fingers", None)
    if v is None:
        return None
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0 or not np.isfinite(v).all():
        return None
    return float(np.mean(v))


def _unwrap_rows(P):
    P = np.array(P, dtype=float)
    for k in (3, 4, 5):
        for i in range(1, len(P)):
            P[i, k] = P[i - 1, k] + _wrap(P[i, k] - P[i - 1, k])
    return P


def _build_reference(demo_frames):
    """Per-control-step reference [x,y,z,r,p,y,closed] and close index from demo frames."""
    for source in ("tcp", "wrist"):
        rows = []
        for f in demo_frames:
            pose = _frame_pose(f, source)
            if pose is None:
                continue
            rows.append((int(f.step), pose, _finger(f)))
        if len(rows) < 10:
            continue
        rows.sort(key=lambda r: r[0])
        steps, poses, fing = [], [], []
        for s, p, w in rows:
            if steps and s == steps[-1]:
                continue
            steps.append(s); poses.append(p); fing.append(w)
        P = _unwrap_rows(poses)
        travel = float(np.sum(np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1)))
        if travel < 0.05:
            continue                    # no informative motion in this source
        st = np.array(steps, dtype=float)
        grid = np.arange(st[0], st[-1] + 1.0)
        Q = np.stack([np.interp(grid, st, P[:, k]) for k in range(6)], 1)
        if all(w is not None for w in fing):
            W = np.interp(grid, st, np.array(fing, dtype=float))
            idx = [i for i in range(len(W)) if W[i] < _CLOSE_THR]
            close = idx[0] if idx else -1
        else:
            close = -1
        if close < 0:
            # no finger evidence: grasp at the first lowest-z dwell in the first half
            half = max(2, len(Q) // 2)
            close = int(np.argmin(Q[:half, 2])) + 3
        ref = []
        for i in range(len(Q)):
            ref.append([float(v) for v in Q[i]] + [1.0 if i >= close else 0.0])
        return {"source": source, "ref": ref, "close": int(close), "travel": travel}
    return None


def _push_plan(ref):
    """Final insertion push: (unit xy direction, low-segment start index, insertion z).

    The demonstrated insertion happens after the last transport (z > 0.1 m) and
    descent; the peg slides along the gripper's yaw axis, the sign is taken from
    the demonstrated low-height displacement."""
    P = _unwrap_rows([r[:6] for r in ref])
    n = len(P)
    hi = [i for i in range(n) if P[i, 2] > 0.1]
    k = hi[-1] if hi else 0
    low = P[k:, 2]
    zmin = float(np.min(low))
    j = k
    while j < n - 1 and P[j, 2] > zmin + 0.006:
        j += 1
    zins = float(np.median(P[j:min(n, j + 10), 2]))
    yaw = float(P[j, 5])
    u = np.array([math.cos(yaw), math.sin(yaw)])
    disp = P[-1, :2] - P[j, :2]
    if float(np.linalg.norm(disp)) < 0.01:
        return None, j, zins
    sgn = 1.0 if float(disp @ u) >= 0 else -1.0
    return [float(sgn * u[0]), float(sgn * u[1])], int(j), zins


def _cmd(robot, pose, grip, stats):
    try:
        return robot.step(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], grip)
    except Exception as e:     # IK/motion rejection; EpisodeEnd is not an Exception
        stats["rejected"] += 1
        stats["last_error"] = str(e)[:120]
        if "1500" in str(e):
            raise
        return robot.now()


def _pos_err(f, pose):
    return float(np.linalg.norm(np.asarray(f.tcp, dtype=float)[:3] - np.asarray(pose[:3])))


def perceive_decide(obs, goal, memory, last):
    """Extract the demonstrated TCP path from the instruction video on the first
    decision, then track it with act_follow_demo; afterwards finish with act_hold."""
    mem = dict(memory)
    for k in ("ref", "close_index", "source", "stage"):
        mem.setdefault(k, None)
    if mem["stage"] is None:
        demo = [f for f in obs.frames if getattr(f, "phase", "") == "demo"]
        built = _build_reference(demo) if demo else None
        if built is None:
            mem["stage"] = "no_demo"
            return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 30}}}
        mem["ref"] = built["ref"]
        mem["close_index"] = built["close"]
        mem["source"] = built["source"]
        mem["stage"] = "follow"
        return {"memory": mem, "act": {"tool": "act_follow_demo",
                                       "args": {"ref": built["ref"], "close_index": built["close"]}}}
    if mem["stage"] == "follow" and mem["ref"]:
        mem["stage"] = "push"
        return {"memory": mem, "act": {"tool": "act_push", "args": {"ref": mem["ref"], "extra": _PUSH_EXTRA}}}
    mem["stage"] = "hold"
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 40}}}


def act_follow_demo(robot, ref, close_index):
    """Track the demonstrated per-step TCP reference [x,y,z,r,p,y,closed] with a
    velocity lead; waits at the grasp pose until reached before closing."""
    stats = {"rejected": 0, "last_error": None, "steps": 0, "grasp_wait": 0}
    R = _unwrap_rows([r[:6] for r in ref])
    closed = [r[6] > 0.5 for r in ref]
    n = len(R)
    # approach the first reference pose if the robot starts elsewhere
    f = robot.now()
    cur = np.array([float(v) for v in f.tcp[:6]])
    gap = _pos_err(f, R[0])
    nap = int(min(200, max(0, math.ceil(gap / 0.004))))
    for j in range(nap):
        a = (j + 1.0) / nap
        p = cur.copy()
        p[:3] = (1 - a) * cur[:3] + a * R[0, :3]
        for k in (3, 4, 5):
            p[k] = cur[k] + a * _wrap(R[0, k] - cur[k])
        _cmd(robot, p, 1.0, stats); stats["steps"] += 1
    for i in range(n - 1):
        nxt = R[i + 1]
        tgt = nxt + _LEAD * (nxt - R[i])
        grip = -1.0 if closed[i + 1] else 1.0
        if closed[i + 1] and not closed[i]:
            # make sure we are at the grasp pose before closing
            for w in range(40):
                if _pos_err(robot.now(), R[i + 1]) < 0.003:
                    break
                _cmd(robot, R[i + 1], 1.0, stats); stats["grasp_wait"] += 1; stats["steps"] += 1
        _cmd(robot, tgt, grip, stats); stats["steps"] += 1
    f = robot.now()
    stats["final_err"] = _pos_err(f, R[-1])
    stats["fingers"] = _finger(f)
    return stats


def act_push(robot, ref, extra=0.08):
    """Continue the demonstrated final insertion push along the peg axis by
    `extra` metres beyond the demonstrated end (gripper closed), then hold."""
    stats = {"rejected": 0, "last_error": None, "steps": 0}
    R = _unwrap_rows([r[:6] for r in ref])
    d, j, zins = _push_plan(ref)
    last = R[-1].copy()
    ori = R[j, 3:6]
    if d is None:
        for i in range(30):
            _cmd(robot, last, -1.0, stats); stats["steps"] += 1
        stats["pushed"] = 0.0
        return stats
    nstep = int(max(1, math.ceil(extra / _PUSH_STEP)))
    p = list(last)
    for i in range(nstep):
        s = (i + 1.0) * _PUSH_STEP
        p = [last[0] + d[0] * s, last[1] + d[1] * s, zins, ori[0], ori[1], ori[2]]
        _cmd(robot, p, -1.0, stats); stats["steps"] += 1
    for i in range(40):
        _cmd(robot, p, -1.0, stats); stats["steps"] += 1
    stats["pushed"] = extra
    stats["dir"] = d
    return stats


def act_hold(robot, steps=30):
    """Hold the current TCP pose (gripper state from finger width) for `steps` steps."""
    f = robot.now()
    w = _finger(f)
    grip = -1.0 if (w is not None and w < _CLOSE_THR) else 1.0
    p = [float(v) for v in f.tcp[:6]]
    stats = {"rejected": 0, "last_error": None, "steps": 0}
    for j in range(int(max(1, min(200, steps)))):
        _cmd(robot, p, grip, stats); stats["steps"] += 1
    return stats
