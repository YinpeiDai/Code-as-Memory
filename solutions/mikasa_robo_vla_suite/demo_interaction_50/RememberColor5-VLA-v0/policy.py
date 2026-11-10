"""RememberColor5: observe the lone cue cube, wait for the five candidates, touch the cube of the same colour.

Perception uses the fixed front RGB-D camera: saturated non-table pixels are grouped by their primary-colour code
(which of R, G, B are bright), back-projected with depth/intrinsics/extrinsic, and the top face (highest 4 mm of
points) gives the cube centre in xy. Control is a per-axis velocity-compensated servo of the measured TCP toward a
point inside the cube (fingers open, straddling it), followed by closing the gripper on the cube.
"""

CUBE_MIN_PIX = 12          # blob must have at least this many pixels in the front image
TOUCH_Z = 0.016            # TCP height for the final straddle (cube is 4 cm, centre z ~0.02)
CONE_SLOPE = 1.0           # allowed height above TOUCH_Z per metre of xy error (avoid side-swiping cubes)
SERVO_GAIN = 0.8           # desired next-step displacement as a fraction of the error
RESP_A = 0.029             # fitted from demos: v[t+1] = RESP_A*u[t] + RESP_B*v[t]  (m per step per unit command)
RESP_B = 0.27
ROT_GAIN = 0.5             # fraction of orientation error corrected per command
HOLD_XY = 0.010            # switch to zero-command hold when xy error below this and the predicted stop height
HOLD_DZ = 0.006            # is below TOUCH_Z + HOLD_DZ
NATIVE_LIMIT = 25


def _rpy_matrix(rpy):
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if ang < 1e-6:
        return 0.5 * v
    if s < 1e-4:
        # near pi: use diagonal
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return axis * ang
    return v * (ang / (2.0 * s))


def _code_name(code):
    names = {(1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue", (1, 1, 0): "yellow",
             (1, 0, 1): "magenta", (0, 1, 1): "cyan"}
    return names.get(tuple(code), "c%d%d%d" % tuple(code))


def _blobs(frame):
    """Coloured cube blobs in the front camera: list of dicts (name, rgb, n, xy, ztop, spread)."""
    im = np.asarray(frame.front_rgb).astype(float)
    depth = np.asarray(frame.front_depth).astype(float) / 1000.0
    K = np.asarray(frame.front_K, dtype=float)
    E = np.asarray(frame.front_extrinsic, dtype=float)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    table = (r >= g) & (g >= b) & (g >= 0.45 * r) & (b >= 0.2 * r) & (g <= 0.8 * r)
    mask = (sat > 0.55) & (mx > 50) & (~table) & (depth > 0)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    px = im[ys, xs]
    bright = px > 0.5 * px.max(axis=1, keepdims=True)
    codes = bright[:, 0].astype(int) * 4 + bright[:, 1].astype(int) * 2 + bright[:, 2].astype(int)
    z = depth[ys, xs]
    u = xs + 0.5
    v = ys + 0.5
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    out = []
    for cval in sorted(set(codes.tolist())):
        if cval == 7 or cval == 0:
            continue
        sel = codes == cval
        n = int(sel.sum())
        if n < CUBE_MIN_PIX:
            continue
        p = pw[sel]
        # reject far-away / floating clusters (keep points near the dominant cluster)
        med = np.median(p, axis=0)
        keep = np.linalg.norm(p - med, axis=1) < 0.06
        p = p[keep]
        if len(p) < CUBE_MIN_PIX:
            continue
        ztop = float(p[:, 2].max())
        top = p[p[:, 2] > ztop - 0.004]
        code = (cval // 4, (cval // 2) % 2, cval % 2)
        rgb = px[sel].mean(axis=0)
        rgb = rgb / (np.linalg.norm(rgb) + 1e-6)
        out.append({"name": _code_name(code), "rgb": [float(x) for x in rgb], "n": int(len(p)),
                    "xy": [float(top[:, 0].mean()), float(top[:, 1].mean())], "ztop": ztop,
                    "spread": float(np.ptp(p[:, 0]) + np.ptp(p[:, 1]))})
    return out


def _empty_memory():
    return {"stage": "wait_cue", "cue_color": "", "cue_rgb": [], "cue_frames": 0, "cue_last_step": -1,
            "cand_step": -1, "n_candidates": 0, "candidates": "", "target": [], "attempts": 0,
            "last_step": 0, "notes": ""}


def perceive_decide(obs, goal, memory, last):
    """Track the cue colour, detect the candidate cubes, pick the matching one and choose the next action."""
    mem = _empty_memory()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    for fr in obs.frames:
        mem["last_step"] = int(fr.step)
        if mem["stage"] not in ("wait_cue", "cue", "delay"):
            continue
        bl = _blobs(fr)
        if len(bl) == 1:
            if mem["stage"] in ("wait_cue", "cue"):
                b0 = bl[0]
                if mem["cue_frames"] > 0 and mem["cue_color"] != b0["name"]:
                    mem["notes"] += "cue colour changed %s->%s at %d; " % (mem["cue_color"], b0["name"], fr.step)
                mem["cue_color"] = b0["name"]
                mem["cue_rgb"] = b0["rgb"]
                mem["cue_frames"] = int(mem["cue_frames"]) + 1
                mem["cue_last_step"] = int(fr.step)
                mem["stage"] = "cue"
        elif len(bl) == 0:
            if mem["stage"] == "cue":
                mem["stage"] = "delay"
        else:
            if mem["cue_frames"] == 0:
                mem["notes"] += "candidates seen before any cue at %d; " % fr.step
                continue
            names = [b0["name"] for b0 in bl]
            if mem["cue_color"] not in names and len(bl) < 5 and int(mem["n_candidates"]) > -2:
                # matching cube possibly occluded: wait for up to two more frames before falling back
                mem["n_candidates"] = int(mem["n_candidates"]) - 1
                mem["notes"] += "cue colour not among %d candidates at %d; " % (len(bl), fr.step)
                continue
            cue = np.array(mem["cue_rgb"], dtype=float)
            best, bestd = None, 9.0
            for b0 in bl:
                d = float(np.linalg.norm(np.array(b0["rgb"]) - cue))
                if b0["name"] == mem["cue_color"]:
                    d -= 1.0
                if d < bestd:
                    best, bestd = b0, d
            mem["cand_step"] = int(fr.step)
            mem["n_candidates"] = len(bl)
            mem["candidates"] = "; ".join("%s(%.3f,%.3f,n=%d)" % (b0["name"], b0["xy"][0], b0["xy"][1], b0["n"])
                                          for b0 in bl)
            mem["target"] = [best["xy"][0], best["xy"][1], 0.02]
            mem["stage"] = "reach"
            if len(bl) != 5:
                mem["notes"] += "saw %d candidates at %d; " % (len(bl), fr.step)
    if mem["stage"] in ("wait_cue", "cue", "delay"):
        act = {"tool": "act_wait", "args": {}}
    elif mem["stage"] == "reach":
        act = {"tool": "act_touch", "args": {"x": mem["target"][0], "y": mem["target"][1]}}
        mem["stage"] = "touched"
        mem["attempts"] = int(mem["attempts"]) + 1
    else:
        # touch routine returned without native success: retry with a fresh open/descend/close cycle
        mem["attempts"] = int(mem["attempts"]) + 1
        act = {"tool": "act_touch", "args": {"x": mem["target"][0], "y": mem["target"][1], "reopen": True}}
    return {"memory": mem, "act": act}


def act_wait(robot):
    """Send one zero command (gripper open) while the cue / delay phase runs; returns the observed step."""
    robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    f = robot.now()
    return {"step": int(f.step), "fingers": [float(v) for v in f.fingers]}


def act_touch(robot, x, y, reopen=False, max_steps=24, hold_steps=4, close_after_hold=True):
    """Servo the TCP (orientation held) over cube (x, y) and descend between the open fingers to TOUCH_Z, then send
    zero commands so the arm comes to rest (success appears to need a static arm). If no native success after
    hold_steps, close the gripper on the cube while holding. Returns a measured per-step log."""
    f0 = robot.now()
    R_ref = _rpy_matrix([math.pi, 0.0, float(f0.tcp[5])])
    tgt = np.array([float(x), float(y), TOUCH_Z])
    prev = np.asarray(f0.tcp[:3], dtype=float)
    prevq = np.asarray(f0.joints, dtype=float)
    vel = np.zeros(3)
    log = []
    mode = "reopen" if reopen else "approach"
    reopen_left = 3
    held = 0
    closed = 0
    for k in range(int(max_steps)):
        f = robot.now()
        if int(f.step) >= NATIVE_LIMIT:
            break
        p = np.asarray(f.tcp[:3], dtype=float)
        q = np.asarray(f.joints, dtype=float)
        if k > 0:
            vel = p - prev
        qv = float(np.abs(q - prevq).max() * 20.0) if k > 0 else -1.0
        prev = p
        prevq = q
        exy = float(np.linalg.norm(tgt[:2] - p[:2]))
        grip = 1.0
        if mode == "reopen":
            goal = np.array([tgt[0], tgt[1], max(p[2], 0.07)])
            reopen_left -= 1
            if reopen_left <= 0:
                mode = "approach"
        elif mode == "approach":
            zstop = p[2] + 0.37 * vel[2]
            if exy < HOLD_XY and zstop < TOUCH_Z + HOLD_DZ:
                mode = "hold"
        if mode == "approach" or mode == "reopen":
            if mode == "approach":
                zg = TOUCH_Z + max(0.0, CONE_SLOPE * (exy - 0.01))
                goal = np.array([tgt[0], tgt[1], zg])
            e = goal - p
            u = np.clip((SERVO_GAIN * e - RESP_B * vel) / RESP_A, -1.0, 1.0)
            Rc = _rpy_matrix(f.tcp[3:6])
            w = _rotvec(R_ref @ Rc.T)
            rcmd = np.clip(-ROT_GAIN * w / 0.1, -1.0, 1.0)
        else:
            u = np.zeros(3)
            rcmd = np.zeros(3)
            held += 1
            if held > int(hold_steps) and close_after_hold:
                grip = -1.0
                closed += 1
        log.append([int(f.step), mode[0]] + [round(float(v), 3) for v in p] + [round(qv, 2)] +
                   [round(float(v), 2) for v in u] + [grip])
        robot.step(float(u[0]), float(u[1]), float(u[2]), float(rcmd[0]), float(rcmd[1]), float(rcmd[2]), grip)
        if closed > 6:
            break
    f = robot.now()
    return {"final_tcp": [round(float(v), 4) for v in f.tcp], "fingers": [round(float(v), 4) for v in f.fingers],
            "step": int(f.step), "mode": mode, "held": held, "closed": closed, "log": log}
