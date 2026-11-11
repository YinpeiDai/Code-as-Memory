"""RememberColor5-VLA-v0 policy: remember the cue cube colour, wait for the five
candidate cubes, then move the TCP onto the centre of the cube with the same colour.

Evidence (250 assigned demos, see task_analysis.md):
  * frame 0 always shows exactly one saturated cube (the cue) for 1-5 frames,
    then an empty table for 1-5 frames, then 5 cubes (red, green, blue, yellow,
    magenta) appear at frame 3..10. Commands are zeroed until that frame.
  * cube ~0.04 m, resting on the table (z=0); expert ends with the TCP at the
    cube centre height (z~0.02) with the gripper open, 8-11 steps after the
    candidates appear. Native step limit is 25.
"""

SAT_MIN = 90          # max(rgb)-min(rgb) for a cube pixel
DARK_MAX = 25         # min(rgb) for a cube pixel (table min is ~38+)
MIN_PIX = 12          # minimum pixels for a cube blob
STEP_GAIN = 0.038     # measured steady TCP displacement per step for |a|=1 (m)
KP = 1.0              # fraction of the error commanded per step
KD = 0.5              # velocity damping (per-step velocity units)
CONE = 0.5            # stay this much higher per metre of xy error
XY_ALIGNED = 0.01     # xy error under which full descent is allowed
HALF_CUBE = 0.02      # cube half size (m)
TCP_ABOVE_CENTRE = 0.003  # aim slightly above the cube centre (table clearance)
REACH_TOL = 0.008     # m: arrived
SETTLE_SPEED = 0.005  # m/step: considered at rest
HOLD_OPEN_STEPS = 2   # steps at rest with open gripper before closing on cube
LATE_CLOSE_STEP = 20  # native step after which we close if near the cube
NATIVE_LIMIT = 25


def _codes(rgb):
    """Per-pixel colour code string array ('' for non-cube pixels)."""
    im = rgb.astype(np.int32)
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    mask = (mx - mn > SAT_MIN) & (mn < DARK_MAX)
    return im, mx, mask


def _blobs(rgb):
    """Saturated primary-colour blobs: list of dicts code, ys, xs, n, rgb."""
    im, mx, mask = _codes(rgb)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    pix = im[ys, xs]
    half = 0.5 * mx[ys, xs][:, None]
    bits = (pix > half).astype(np.int32)
    keys = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    out = []
    for k in sorted(set(keys.tolist())):
        if k == 0 or k == 7:
            continue
        sel = keys == k
        y = ys[sel]
        x = xs[sel]
        if len(y) < MIN_PIX:
            continue
        my = np.median(y)
        mxx = np.median(x)
        keep = (np.abs(y - my) <= 15) & (np.abs(x - mxx) <= 15)
        y = y[keep]
        x = x[keep]
        if len(y) < MIN_PIX:
            continue
        code = str((k >> 2) & 1) + str((k >> 1) & 1) + str(k & 1)
        out.append({"code": code, "ys": y, "xs": x, "n": int(len(y)),
                    "rgb": [float(v) for v in im[y, x].mean(axis=0)]})
    return out


def _deproject(K, E, ys, xs, depth_mm):
    z = depth_mm.astype(np.float64) / 1000.0
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], axis=1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (pc - t) @ R


def _blob_top(frame, cam, blob, border=0):
    """World estimate of a cube: top-face centre xy and top z, or None."""
    ys, xs = blob["ys"], blob["xs"]
    if border > 0:
        if ys.min() < border or xs.min() < border or ys.max() > 255 - border or xs.max() > 255 - border:
            return None
    dep = getattr(frame, cam + "_depth")
    d = dep[ys, xs]
    ok = d > 0
    if ok.sum() < MIN_PIX:
        return None
    P = _deproject(getattr(frame, cam + "_K"), getattr(frame, cam + "_extrinsic"), ys[ok], xs[ok], d[ok])
    zmax = float(P[:, 2].max())
    top = P[P[:, 2] > zmax - 0.006]
    if len(top) < 4:
        return None
    c = top.mean(axis=0)
    return [float(c[0]), float(c[1]), float(zmax)]


def _detect(frame, cam):
    """List of {code, rgb, n, pos=[x, y, top_z]} for one camera of one frame."""
    out = []
    for b in _blobs(getattr(frame, cam + "_rgb")):
        p = _blob_top(frame, cam, b)
        out.append({"code": b["code"], "rgb": [round(v, 1) for v in b["rgb"]], "n": b["n"], "pos": p})
    return out


def _choose(cands, cue_code, cue_rgb):
    """Pick the candidate matching the cue colour; returns (cand, confidence)."""
    good = [c for c in cands if c["pos"] is not None]
    if not good:
        return None, 0.0
    same = [c for c in good if c["code"] == cue_code]
    if len(same) >= 1:
        best = max(same, key=lambda c: c["n"])
        return best, (1.0 if len(same) == 1 else 0.5)
    if cue_rgb is not None:
        ref = np.array(cue_rgb, dtype=float)
        best = min(good, key=lambda c: float(np.linalg.norm(np.array(c["rgb"]) - ref)))
        return best, 0.3
    return good[0], 0.0


def _fresh(memory):
    m = {}
    for k in ("cue_code", "cue_rgb", "cue_step", "cand_step", "candidates", "target_code",
              "target_pos", "target_conf", "stage", "steps_seen", "last_result"):
        m[k] = memory.get(k) if isinstance(memory, dict) else None
    if m["stage"] is None:
        m["stage"] = "wait_cue"
    if m["steps_seen"] is None:
        m["steps_seen"] = 0
    return m


def perceive_decide(obs, goal, memory, last):
    """Record the cue colour (single cube seen before the candidates), detect the
    candidates when several cubes appear, choose the same-colour cube, then touch it."""
    m = _fresh(memory)
    if last is not None:
        m["last_result"] = str(last)[:400]
    for f in obs.frames:
        st = int(f.step) if f.step is not None else 0
        m["steps_seen"] = max(int(m["steps_seen"]), st)
        if m["candidates"] is not None:
            continue
        blobs = _detect(f, "front")
        if len(blobs) == 1 and m["cue_code"] is None:
            m["cue_code"] = blobs[0]["code"]
            m["cue_rgb"] = blobs[0]["rgb"]
            m["cue_step"] = st
            m["stage"] = "wait_candidates"
        elif len(blobs) >= 2:
            cands = blobs
            if m["cue_code"] is not None and not any(c["code"] == m["cue_code"] and c["pos"] is not None for c in cands):
                wr = _detect(f, "wrist")
                cands = cands + [c for c in wr if c["code"] == m["cue_code"] and c["pos"] is not None]
            m["candidates"] = [{"code": c["code"], "pos": c["pos"], "n": c["n"]} for c in cands]
            m["cand_step"] = st
            best, conf = _choose(cands, m["cue_code"], m["cue_rgb"])
            if best is not None:
                m["target_code"] = best["code"]
                m["target_pos"] = best["pos"]
                m["target_conf"] = conf
                m["stage"] = "touch"
            else:
                m["stage"] = "no_target"
    if m["stage"] == "touch" and m["target_pos"] is not None:
        p = m["target_pos"]
        act = {"tool": "act_touch", "args": {"tx": float(p[0]), "ty": float(p[1]), "top_z": float(p[2]),
                                            "code": str(m["target_code"]), "start_step": int(m["steps_seen"])}}
    elif m["stage"] == "no_target":
        # candidates appeared but none could be localised: re-detect on the next frame
        m["candidates"] = None
        m["stage"] = "wait_candidates" if m["cue_code"] is not None else "wait_cue"
        act = {"tool": "act_wait", "args": {"max_steps": 1}}
    else:
        act = {"tool": "act_wait", "args": {"max_steps": NATIVE_LIMIT}}
    return {"memory": m, "act": act}


def act_wait(robot, max_steps=25):
    """Send zero motion (gripper open) until several cubes are visible in the front
    camera or max_steps steps pass. Returns steps used and what was seen."""
    n = 0
    seen = []
    for i in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        f = robot.now()
        k = len(_blobs(f.front_rgb))
        seen.append(k)
        if k >= 2:
            return {"steps": n, "candidates_visible": True, "blob_counts": seen}
    return {"steps": n, "candidates_visible": False, "blob_counts": seen}


def act_touch(robot, tx, ty, top_z, code, start_step=0, max_steps=30):
    """Closed-loop move of the TCP to the centre of the chosen cube (xy from the
    top face, z = top - 0.018) with the gripper open, staying above the cube until
    xy is aligned; hold still there, then close the gripper on the cube and hold.
    Wrist-camera detections of the same colour refine the xy target."""
    front_xy = np.array([float(tx), float(ty)])
    txy = front_xy.copy()
    tz = max(0.012, float(top_z) - HALF_CUBE) + TCP_ABOVE_CENTRE
    f = robot.now()
    prev = np.array(f.tcp[:3], dtype=float)
    vel = np.zeros(3)
    settled = 0
    closing = False
    refined = 0
    log = []
    step_no = int(start_step)
    for i in range(int(max_steps)):
        f = robot.now()
        p = np.array(f.tcp[:3], dtype=float)
        if i > 0:
            vel = p - prev
        prev = p
        if not closing:
            wb = [b for b in _blobs(f.wrist_rgb) if b["code"] == code and b["n"] >= 40]
            if len(wb) == 1:
                est = _blob_top(f, "wrist", wb[0], border=3)
                if est is not None and abs(est[2] - float(top_z)) < 0.012:
                    exy = np.array(est[:2])
                    if np.linalg.norm(exy - front_xy) < 0.025:
                        txy = 0.5 * (front_xy + exy)
                        refined += 1
        g = np.array([txy[0], txy[1], tz])
        exy_err = float(np.linalg.norm(g[:2] - p[:2]))
        g[2] = max(g[2], tz + CONE * max(0.0, exy_err - XY_ALIGNED))
        e = g - p
        a = np.clip(KP * e / STEP_GAIN - KD * vel / STEP_GAIN, -1.0, 1.0)
        dist = float(np.linalg.norm(np.array([txy[0], txy[1], tz]) - p))
        speed = float(np.linalg.norm(vel))
        if dist < REACH_TOL and speed < SETTLE_SPEED:
            settled += 1
        if not closing and (settled >= HOLD_OPEN_STEPS or (step_no >= LATE_CLOSE_STEP and dist < 0.015)):
            closing = True
        grip = -1.0 if closing else 1.0
        log.append([step_no, round(dist, 4), round(speed, 4), int(closing)])
        robot.step(float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, grip)
        step_no += 1
    f = robot.now()
    return {"final_tcp": [float(v) for v in f.tcp[:3]], "target": [float(txy[0]), float(txy[1]), tz],
            "wrist_refinements": refined, "closed": closing, "log": log[-8:]}
