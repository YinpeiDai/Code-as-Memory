"""RememberColor3-VLA policy: remember the cue cube colour, then touch the same-coloured cube.

Observed task structure (assigned demos): frame 0 shows one cue cube (1-5 frames), then an empty
table (1-5 frames), then three cubes (red/green/blue) appear. Robot commands are replaced by zero
until the three cubes appear. Native episode limit is 25 steps, so the approach must be fast.
"""

TABLE_HUE = (5.0, 45.0)      # orange wooden table hue band (degrees), excluded from cube masks
MIN_BLOB = 20                # min pixels of a colour cluster in the front image
HUE_PROTOS = (("red", 0.0), ("yellow", 60.0), ("green", 120.0), ("cyan", 180.0),
              ("blue", 240.0), ("magenta", 300.0))
Z_LOW = 0.004                # commanded straddle depth (below demo min TCP z 0.009 so the tail does not creep; table blocks fingertips)
X_OFFSET = 0.005             # expert final TCP sits ~+0.01 m in x from the top-face centre; halved to stay closer
HOLD_XY = 0.015              # bottom zone: xy error below this ...
HOLD_DZ = 0.018              # ... and TCP z below z_low + this -> send zero commands (let the arm settle)
HOLD_STEPS = 4               # zero-command steps before giving up on the straddle and squeezing
SLOPE = 3.0                  # z gate: z_goal = z_low + SLOPE*max(exy-0.004, 0) (open fingers straddle the cube)
Z_CRUISE = 0.08              # travel height: fingertips 4 cm above cube tops (z 0.042) while xy error is large
GAIN = 2.0
DAMP = 0.5


def _hue_sat(img):
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    hue = np.degrees(np.arctan2(np.sqrt(3.0) * (g - b), 2.0 * r - g - b)) % 360.0
    return hue, sat, mx


def _detect(frame):
    """Colour clusters in the fixed front camera with world positions of their top faces."""
    img = frame.front_rgb
    hue, sat, mx = _hue_sat(img)
    table = (hue > TABLE_HUE[0]) & (hue < TABLE_HUE[1])
    m = (sat > 0.6) & (mx > 50) & (~table)
    out = []
    if int(m.sum()) < MIN_BLOB:
        return out
    depth = frame.front_depth.astype(np.float64) / 1000.0
    K = np.asarray(frame.front_K, dtype=np.float64)
    E = np.asarray(frame.front_extrinsic, dtype=np.float64)
    R = E[:, :3]
    t = E[:, 3]
    for name, h0 in HUE_PROTOS:
        dh = np.abs(((hue - h0 + 180.0) % 360.0) - 180.0)
        mk = m & (dh < 30.0)
        ys, xs = np.nonzero(mk)
        if len(ys) < MIN_BLOB:
            continue
        # keep the dominant spatial cluster (median-centred window) to reject stray pixels
        cy, cx = np.median(ys), np.median(xs)
        keep = (np.abs(ys - cy) < 25) & (np.abs(xs - cx) < 25)
        ys, xs = ys[keep], xs[keep]
        if len(ys) < MIN_BLOB:
            continue
        z = depth[ys, xs]
        ok = z > 0
        rgb = img[ys, xs].astype(np.float64).mean(axis=0)
        item = {"color": name, "pixels": int(len(ys)), "rgb": [float(v) for v in rgb],
                "u": float(xs.mean()), "v": float(ys.mean()), "xyz": None}
        if int(ok.sum()) >= 5:
            zz = z[ok]
            px = (xs[ok] - K[0, 2]) * zz / K[0, 0]
            py = (ys[ok] - K[1, 2]) * zz / K[1, 1]
            pc = np.stack([px, py, zz], axis=1)
            W = (pc - t) @ R
            top = W[W[:, 2] > W[:, 2].max() - 0.004]
            item["xyz"] = [float(top[:, 0].mean()), float(top[:, 1].mean()), float(W[:, 2].max())]
        out.append(item)
    return out


def _rot(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logrot(R):
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _rot_cmd(R_goal, rpy_now, gain=1.0):
    """Rotation command that drives the measured orientation toward R_goal.
    Demo regression: world rotation increment ~ -0.035 * r per step, so r = -w_err/0.1*gain."""
    w = _logrot(R_goal @ _rot(rpy_now).T)
    r = -w / 0.1 * gain
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _match(cands, cue_name, cue_rgb):
    same = [c for c in cands if c["color"] == cue_name and c["xyz"] is not None]
    if same:
        return max(same, key=lambda c: c["pixels"]), 1.0
    if any(c["color"] == cue_name for c in cands):
        return None, 0.0  # cue-coloured cube seen but no depth yet: wait a frame rather than pick another colour
    # fallback: nearest chromaticity to the cue
    best, bd = None, 1e9
    q = np.array(cue_rgb, dtype=np.float64)
    q = q / (q.sum() + 1e-6)
    for c in cands:
        if c["xyz"] is None:
            continue
        v = np.array(c["rgb"], dtype=np.float64)
        v = v / (v.sum() + 1e-6)
        d = float(np.linalg.norm(v - q))
        if d < bd:
            best, bd = c, d
    return best, max(0.0, 1.0 - bd)


def perceive_decide(obs, goal, memory, last):
    """Track cue colour, cube appearance and stage; choose the next act tool."""
    mem = {
        "stage": memory.get("stage") or "observe_cue",
        "cue_color": memory.get("cue_color") or "",
        "cue_rgb": memory.get("cue_rgb") or "",
        "cue_step": memory.get("cue_step", -1) if memory.get("cue_step") is not None else -1,
        "cue_frames": memory.get("cue_frames") or 0,
        "cubes_step": memory.get("cubes_step", -1) if memory.get("cubes_step") is not None else -1,
        "target": memory.get("target") or [],
        "target_conf": memory.get("target_conf") or 0.0,
        "candidates": memory.get("candidates") or "",
        "frames_seen": memory.get("frames_seen") or 0,
        "last_step": memory.get("last_step", -1) if memory.get("last_step") is not None else -1,
        "log": memory.get("log") or "",
    }
    for f in obs.frames:
        if f.step <= mem["last_step"]:
            continue
        mem["last_step"] = int(f.step)
        mem["frames_seen"] = int(mem["frames_seen"]) + 1
        if mem["target"]:
            continue
        cands = _detect(f)
        if not mem["cue_color"]:
            if len(cands) == 1:
                c = cands[0]
                mem["cue_color"] = c["color"]
                mem["cue_rgb"] = ",".join("%.1f" % v for v in c["rgb"])
                mem["cue_step"] = int(f.step)
                mem["cue_frames"] = 1
                mem["stage"] = "wait_cubes"
            continue
        if len(cands) == 1 and cands[0]["color"] == mem["cue_color"] and mem["cubes_step"] < 0:
            mem["cue_frames"] = int(mem["cue_frames"]) + 1
            continue
        if len(cands) >= 2:
            rgb = [float(v) for v in mem["cue_rgb"].split(",")] if mem["cue_rgb"] else [0, 0, 0]
            best, conf = _match(cands, mem["cue_color"], rgb)
            mem["candidates"] = ";".join(
                "%s:%s" % (c["color"], ("%.3f,%.3f" % (c["xyz"][0], c["xyz"][1])) if c["xyz"] else "none")
                for c in cands)
            if best is not None:
                mem["cubes_step"] = int(f.step)
                mem["target"] = [best["xyz"][0], best["xyz"][1], 0.02]
                mem["target_conf"] = float(conf)
                mem["stage"] = "approach"
    if last is not None:
        mem["log"] = str(last)[:400]
    st = mem["stage"]
    if not mem["target"]:
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 1}}}
    tx, ty = float(mem["target"][0]), float(mem["target"][1])
    if st == "approach":
        mem["stage"] = "squeeze"
        return {"memory": mem, "act": {"tool": "act_touch", "args": {
            "x": tx + X_OFFSET, "y": ty, "z_low": Z_LOW, "gripper": 1.0}}}
    if st == "squeeze":
        mem["stage"] = "hold"
        return {"memory": mem, "act": {"tool": "act_squeeze", "args": {"x": tx + X_OFFSET, "y": ty, "z": Z_LOW, "n": 6}}}
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 1}}}


def act_wait(robot, n=1):
    """Send n zero-motion steps with the gripper open (cue/delay phases; commands are zeroed anyway)."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"step": int(f.step), "tcp": [float(v) for v in f.tcp[:3]]}


def act_touch(robot, x, y, z_low=0.012, gripper=1.0, max_steps=22, lift_first=False):
    """Closed-loop move of the TCP to (x, y) and down to z_low with the given gripper target, holding the
    start orientation. Descends only as xy error shrinks. Returns measured trace if the episode
    has not ended (native success raises and ends the episode)."""
    f0 = robot.now()
    R_goal = _rot(f0.tcp[3:6])
    prev = np.array(f0.tcp[:3], dtype=np.float64)
    vel = np.zeros(3)
    trace = []
    settled = 0
    lifted = not lift_first
    for k in range(int(max_steps)):
        f = robot.now()
        p = np.array(f.tcp[:3], dtype=np.float64)
        if k > 0:
            vel = p - prev
        prev = p
        exy = math.hypot(x - p[0], y - p[1])
        if not lifted:
            zg = 0.07
            if p[2] > 0.06:
                lifted = True
        else:
            zg = min(max(z_low + SLOPE * max(exy - 0.004, 0.0), z_low), Z_CRUISE)
        goal = np.array([x, y, zg])
        err = goal - p
        u = np.clip(err / 0.1 * GAIN - DAMP * vel / 0.1, -1.0, 1.0)
        r, rot_err = _rot_cmd(R_goal, f.tcp[3:6], 1.0)
        trace.append([int(f.step), round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4),
                      round(exy, 4), round(rot_err, 3)])
        if lifted and exy < HOLD_XY and p[2] < z_low + HOLD_DZ:
            # at the bottom of the straddle: stop commanding motion so the arm comes to rest
            settled += 1
            if settled > HOLD_STEPS:
                return {"arrived": True, "held": settled - 1, "trace": trace[-8:],
                        "fingers": [float(v) for v in f.fingers]}
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))
            continue
        settled = 0
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2]), float(gripper))
    f = robot.now()
    return {"arrived": False, "trace": trace[-8:], "fingers": [float(v) for v in f.fingers]}


def act_squeeze(robot, x, y, z=0.012, n=4):
    """Hold the TCP at (x, y, z) and close the gripper for n steps so the fingers contact the cube."""
    f0 = robot.now()
    R_goal = _rot(f0.tcp[3:6])
    out = []
    for _ in range(int(n)):
        f = robot.now()
        p = np.array(f.tcp[:3], dtype=np.float64)
        u = np.clip((np.array([x, y, z]) - p) / 0.1 * 1.5, -1.0, 1.0)
        r, _ = _rot_cmd(R_goal, f.tcp[3:6], 1.0)
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(r[0]), float(r[1]), float(r[2]), -1.0)
        g = robot.now()
        out.append([int(g.step), round(float(g.fingers[0]), 4), round(float(g.fingers[1]), 4)])
    return {"squeezed": True, "fingers_trace": out}
