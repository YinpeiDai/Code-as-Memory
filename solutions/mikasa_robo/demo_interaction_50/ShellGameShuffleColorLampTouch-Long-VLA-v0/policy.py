"""ShellGameShuffleColorLampTouch policy (Code-as-Memory).

Stages: watch (cue balls -> cups appear -> shuffle tracking -> lamp colour),
then touch (move above the tracked cup whose ball colour matches the lamp,
descend vertically until the native success ends the episode).
All perception uses the fixed front camera RGB-D deprojected to world XYZ.
"""

COLORS = ("red", "green", "blue")
HOVER_Z = 0.21
MIN_CUP_PIX = 800


def _world_points(frame):
    d = frame.front_depth.astype(np.float32) / 1000.0
    K = frame.front_K
    E = frame.front_extrinsic
    R = E[:, :3]
    t = E[:, 3]
    v, u = np.mgrid[0:256, 0:256]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1)
    pw = (pc - t) @ R
    return pw, d > 0


def _channels(frame):
    rgb = frame.front_rgb.astype(np.int32)
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def _cup_points(frame, P, valid):
    r, g, b = _channels(frame)
    m = (r > 50) & (r < 200) & (g < 70) & (b < 75) & (r > 2 * g) & (P[..., 2] > 0.01) & valid
    return P[m]


def _balls(frame, P, valid):
    """Cue balls on the table: colour -> [x, y] world (z about 0.033)."""
    r, g, b = _channels(frame)
    reg = valid & (P[..., 2] > 0.01) & (P[..., 2] < 0.07) & (P[..., 0] < 0.15)
    masks = {
        "red": (r > 150) & (g < 80) & (b < 80),
        "green": (g > 120) & (r < 100) & (b < 100),
        "blue": (b > 120) & (r < 80) & (g < 80),
    }
    out = {}
    for c in COLORS:
        m = masks[c] & reg
        if int(m.sum()) >= 10:
            p = P[m]
            out[c] = [float(p[:, 0].mean()), float(p[:, 1].mean()), int(m.sum())]
    return out


def _lamp(frame, P, valid):
    """Pixel counts of a saturated lamp colour in the lamp region (front of the table, raised)."""
    r, g, b = _channels(frame)
    reg = valid & (P[..., 0] > 0.15) & (P[..., 2] > 0.05)
    cnt = {
        "red": int(((r > 200) & (g < 90) & (b < 90) & reg).sum()),
        "green": int(((g > 200) & (r < 90) & (b < 90) & reg).sum()),
        "blue": int(((b > 200) & (r < 90) & (g < 90) & reg).sum()),
    }
    best = max(COLORS, key=lambda c: cnt[c])
    return (best if cnt[best] >= 50 else None), cnt


def _kmeans(pts, init, iters=5):
    c = np.array(init, dtype=float).reshape(-1, 2)
    cnt = np.zeros(len(c))
    for _ in range(iters):
        dd = ((pts[:, None, :2] - c[None, :, :]) ** 2).sum(-1)
        lab = dd.argmin(1)
        newc = c.copy()
        for k in range(len(c)):
            s = pts[lab == k]
            cnt[k] = len(s)
            if len(s) >= 30:
                newc[k] = s[:, :2].mean(0)
        c = newc
    return c, cnt


def _track_update(st, frame):
    """One frame of the watch state machine. st is a plain dict (JSON-able)."""
    P, valid = _world_points(frame)
    lamp, lcnt = _lamp(frame, P, valid)
    cups = _cup_points(frame, P, valid)
    st["frames"] = st.get("frames", 0) + 1
    if not st.get("tracks"):
        b = _balls(frame, P, valid)
        if len(b) == 3:
            st["balls"] = {c: b[c][:2] for c in COLORS}
            st["ball_step"] = int(frame.step)
        if len(cups) >= MIN_CUP_PIX and st.get("balls"):
            order = sorted(COLORS, key=lambda c: st["balls"][c][1])
            init = [st["balls"][c] for c in order]
            c, cnt = _kmeans(cups, init, 10)
            st["order"] = order
            st["tracks"] = c.tolist()
            st["vel"] = [[0.0, 0.0] for _ in order]
            st["cup_step"] = int(frame.step)
            st["lost"] = 0
            st["maxdisp"] = 0.0
        return st, lamp
    tr = np.array(st["tracks"])
    vel = np.array(st["vel"])
    pred = tr + vel
    if len(cups) >= 90:
        c, cnt = _kmeans(cups, pred, 5)
        for k in range(len(c)):
            if cnt[k] < 30:
                c[k] = pred[k]
                st["lost"] = st.get("lost", 0) + 1
    else:
        c = pred
        st["lost"] = st.get("lost", 0) + 1
    disp = float(np.abs(c - tr).max())
    st["maxdisp"] = max(st.get("maxdisp", 0.0), disp)
    st["vel"] = (0.8 * (c - tr)).tolist()
    st["tracks"] = c.tolist()
    return st, lamp


def _target_from(st, color):
    order = st["order"]
    k = order.index(color)
    xy = list(st["tracks"][k])
    # cups come to rest on the original ball slots: snap to nearest slot if close
    slots = [st["balls"][c] for c in COLORS]
    dmin, best = 1e9, None
    for s in slots:
        dd = math.hypot(s[0] - xy[0], s[1] - xy[1])
        if dd < dmin:
            dmin, best = dd, s
    snapped = dmin < 0.05
    if snapped:
        xy = [best[0], best[1]]
    return xy, snapped, dmin



def _enc(st):
    """Encode watch state as 'key=v1,v2,...;' text (no json module available)."""
    parts = []
    for k in sorted(st):
        v = st[k]
        if isinstance(v, dict):
            continue
        if isinstance(v, (list, tuple)):
            flat = []
            for e in v:
                if isinstance(e, (list, tuple)):
                    flat.extend(e)
                else:
                    flat.append(e)
            vals = []
            for e in flat:
                vals.append(e if isinstance(e, str) else ("%.5f" % float(e)))
            parts.append(k + "=" + ",".join(vals))
        elif isinstance(v, str):
            parts.append(k + "=" + v)
        else:
            parts.append(k + "=" + ("%.5f" % float(v)))
    if st.get("balls"):
        b = st["balls"]
        parts.append("balls=" + ",".join("%.5f" % float(q) for c in COLORS for q in b[c]))
    return ";".join(parts)


def _dec(text):
    st = {}
    if not text:
        return st
    for part in text.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        vals = v.split(",") if v else []
        if k == "order":
            st[k] = vals
        elif k == "balls":
            f = [float(q) for q in vals]
            st[k] = {c: [f[2 * j], f[2 * j + 1]] for j, c in enumerate(COLORS)}
        elif k in ("tracks", "vel"):
            f = [float(q) for q in vals]
            st[k] = [[f[2 * j], f[2 * j + 1]] for j in range(len(f) // 2)]
        elif k in ("lamp",):
            st[k] = v
        else:
            st[k] = float(vals[0]) if len(vals) == 1 else [float(q) for q in vals]
    return st


def _memo_get(last, key):
    """Read 'key=<value>' from the MEMO[...] block of the previous act result string."""
    if not isinstance(last, str) or "MEMO[" not in last:
        return None
    body = last.split("MEMO[", 1)[1].split("]MEMO", 1)[0]
    for part in body.split("|"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


def _rpy_matrix(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_log(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def _servo_cmd(tcp, goal_xyz, R_des, kp=1.0, vmax=0.6, rot_gain=10.0, rmax=0.5):
    """P command toward goal_xyz (normalized units: 1.0 = 0.1 m) and orientation hold toward R_des."""
    err = np.array(goal_xyz, dtype=float) - np.array(tcp[:3], dtype=float)
    cmd = kp * err / 0.1
    n = float(np.abs(cmd).max())
    if n > vmax:
        cmd = cmd * (vmax / n)
    e = _rot_log(R_des @ _rpy_matrix(tcp[3:6]).T)
    r = -rot_gain * e
    rn = float(np.linalg.norm(r))
    if rn > rmax:
        r = r * (rmax / rn)
    return [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(r[0]), float(r[1]), float(r[2])], float(np.linalg.norm(err))


def perceive_decide(obs, goal, memory, last):
    """Stage machine: 'watch' (track cue colours under cups through the shuffle until the lamp lights),
    then 'touch' (move above the matching cup and descend onto it); retries touch with logged outcomes."""
    mem = {
        "stage": memory.get("stage") or "watch",
        "watch_state": memory.get("watch_state") or "",
        "lamp_color": memory.get("lamp_color") or "",
        "target": memory.get("target") or [],
        "touch_tries": int(memory.get("touch_tries") or 0),
        "notes": memory.get("notes") or "",
    }
    if isinstance(last, str):
        mem["notes"] = (mem["notes"] + " || " + last[:400])[-2500:]
        st = _memo_get(last, "state")
        if st is not None:
            mem["watch_state"] = st
        col = _memo_get(last, "lamp")
        if col:
            mem["lamp_color"] = col
        tx = _memo_get(last, "tx")
        ty = _memo_get(last, "ty")
        if tx and ty:
            mem["target"] = [float(tx), float(ty)]
        nxt = _memo_get(last, "next")
        if nxt:
            mem["stage"] = nxt
    if mem["stage"] == "watch":
        return {"memory": mem, "act": {"tool": "act_watch", "args": {"state_text": mem["watch_state"]}}}
    if mem["stage"] == "touch" and len(mem["target"]) == 2:
        mem["touch_tries"] = mem["touch_tries"] + 1
        k = mem["touch_tries"]
        # later tries press a little deeper
        depth = 0.085 - 0.01 * min(k - 1, 3)
        return {"memory": mem, "act": {"tool": "act_touch",
                                        "args": {"x": mem["target"][0], "y": mem["target"][1], "press_z": depth}}}
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"n": 20}}}


def act_watch(robot, state_text="", max_steps=650):
    """Hold still (zero motion, gripper open) and process every frame: record cue ball colours/positions,
    initialise three cup tracks when cups appear, track them through the shuffle, and stop when the lamp
    shows a colour. Returns a MEMO block with lamp colour and target cup xy (world, metres)."""
    st = _dec(state_text)
    frame = robot.now()
    lamp = None
    lamp_seen = 0
    steps = 0
    lamp_step = -1
    while True:
        st, lamp = _track_update(st, frame)
        if lamp is not None and st.get("tracks"):
            lamp_seen += 1
            if lamp_step < 0:
                lamp_step = int(frame.step)
            if lamp_seen >= 2:
                break
        else:
            lamp_seen = 0
        if steps >= max_steps:
            break
        frame = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        steps += 1
    st_txt = _enc(st)
    if lamp is None or not st.get("tracks"):
        return {"memo": "MEMO[next=watch|state=" + st_txt + "]MEMO",
                "lamp_seen": False, "steps": steps, "has_tracks": bool(st.get("tracks"))}
    xy, snapped, dsnap = _target_from(st, lamp)
    memo = "MEMO[next=touch|lamp=%s|tx=%.5f|ty=%.5f|state=%s]MEMO" % (lamp, xy[0], xy[1], st_txt)
    return {"memo": memo, "lamp_seen": True, "lamp": lamp, "lamp_step": lamp_step, "steps": steps,
            "cue_step": st.get("ball_step"), "cup_step": st.get("cup_step"),
            "order": st.get("order"), "tracks": st.get("tracks"), "balls": st.get("balls"),
            "snapped": snapped, "snap_dist": dsnap, "maxdisp": st.get("maxdisp"), "lost": st.get("lost")}


def act_touch(robot, x, y, press_z=0.085, hover_z=HOVER_Z, max_steps=160):
    """Move the open gripper to (x, y, hover_z) keeping orientation, then descend vertically toward
    press_z over the cup, holding when the TCP stalls on the cup top. The native success ends the
    episode; otherwise returns the measured approach/descent record."""
    f = robot.now()
    tcp0 = np.array(f.tcp, dtype=float)
    R_des = _rpy_matrix(tcp0[3:6])
    log = []
    phase = "lift" if tcp0[2] < hover_z - 0.02 else "travel"
    pre = robot.dry_run(x, y, hover_z, tcp0[3], tcp0[4], tcp0[5], mode="absolute")
    log.append(["dry_run_hover", bool(pre.get("within_joint_limits")), round(float(pre.get("min_joint_margin_rad") or 0.0), 3)])
    steps = 0
    zs = []
    stall_steps = 0
    while steps < max_steps:
        tcp = np.array(robot.now().tcp, dtype=float)
        if phase == "lift":
            goal = [tcp[0], tcp[1], hover_z]
            cmd, err = _servo_cmd(tcp, goal, R_des)
            if abs(tcp[2] - hover_z) < 0.02:
                phase = "travel"
        if phase == "travel":
            goal = [x, y, hover_z]
            cmd, err = _servo_cmd(tcp, goal, R_des)
            if math.hypot(tcp[0] - x, tcp[1] - y) < 0.006 and abs(tcp[2] - hover_z) < 0.02:
                phase = "descend"
                log.append(["at_hover", steps] + [round(float(q), 4) for q in tcp[:3]])
        if phase == "descend":
            goal = [x, y, press_z]
            cmd, err = _servo_cmd(tcp, goal, R_des, kp=1.0, vmax=0.5)
            zs.append(float(tcp[2]))
            if len(zs) > 4 and zs[-5] - zs[-1] < 0.003 and tcp[2] < 0.16:
                stall_steps += 1
            if stall_steps > 25:
                log.append(["stalled", steps] + [round(float(q), 4) for q in tcp[:3]])
                break
        if steps % 5 == 0:
            log.append([phase, steps] + [round(float(q), 4) for q in tcp[:3]])
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], 1.0)
        steps += 1
    tcp = np.array(robot.now().tcp, dtype=float)
    # back off upward so a retry starts from clear space
    for _ in range(12):
        t2 = np.array(robot.now().tcp, dtype=float)
        cmd, err = _servo_cmd(t2, [x, y, hover_z], R_des)
        robot.step(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5], 1.0)
    return {"memo": "MEMO[next=touch]MEMO", "phase": phase, "steps": steps, "final_tcp": [round(float(q), 4) for q in tcp[:6]],
            "min_z": round(min(zs), 4) if zs else None, "log": log[-25:]}


def act_hold(robot, n=20):
    """Hold still for n steps with gripper open."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": int(n)}
