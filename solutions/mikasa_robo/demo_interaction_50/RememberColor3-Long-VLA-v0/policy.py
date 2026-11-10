"""RememberColor3-Long: observe the cue cube colour, wait through the delay, touch the same-coloured cube.

Perception: saturated "pure colour" pixels (min channel < 0.15 * max channel, max > 60) in the fixed front
camera are grouped into connected blobs; each blob gets a hue. The cue phase shows one blob, the test phase
three blobs. The target is the test blob whose hue is closest to the remembered cue hue; its 3D centre comes
from front depth + calibration (top-face pixels).
Control: delta-pose servo on measured TCP to a hover point above the target, then descend so the TCP is
~1.5 cm above the cube top (~3.5 cm from cube centre) and hold there (success ends the episode).
"""

MIN_BLOB_PX = 15
HOVER_Z = 0.10
TOUCH_ABOVE_TOP = 0.010
CUBE_DEFAULT_TOP = 0.041
PROBE_ABOVE = 0.0   # exploratory: >0 -> settle this far above cube top, then descend slowly
PROBE_RATE = 0.0015


def _pure_mask(img):
    im = img.astype(np.int32)
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    return (mx > 60) & (mn * 100 < 15 * mx)


def _hue_of(rgb):
    r, g, b = [float(v) for v in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx - mn < 1e-6:
        return 0.0
    if mx == r:
        h = (g - b) / (mx - mn) % 6.0
    elif mx == g:
        h = (b - r) / (mx - mn) + 2.0
    else:
        h = (r - g) / (mx - mn) + 4.0
    return (60.0 * h) % 360.0


def _hue_name(h):
    names = [(0.0, "red"), (60.0, "yellow"), (120.0, "green"), (180.0, "cyan"), (240.0, "blue"),
             (300.0, "magenta"), (360.0, "red")]
    best = min(names, key=lambda p: abs(p[0] - h))
    return best[1]


def _hue_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _blobs(img):
    """Connected components (8-neighbour) of pure-colour pixels, split by hue. Returns list of dicts."""
    mask = _pure_mask(img)
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return []
    im = img.astype(np.float64)
    hues = {}
    for v, u in zip(vs.tolist(), us.tolist()):
        hues[(v, u)] = _hue_of(im[v, u])
    seen = set()
    out = []
    for start in hues:
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        comp = []
        h0 = hues[start]
        while stack:
            p = stack.pop()
            comp.append(p)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    q = (p[0] + dv, p[1] + du)
                    if q in hues and q not in seen and _hue_diff(hues[q], h0) < 40.0:
                        seen.add(q)
                        stack.append(q)
        if len(comp) < MIN_BLOB_PX:
            continue
        cv = np.array([c[0] for c in comp])
        cu = np.array([c[1] for c in comp])
        mean_rgb = im[cv, cu].mean(axis=0)
        h = _hue_of(mean_rgb)
        out.append({"n": len(comp), "v": cv, "u": cu, "hue": h, "name": _hue_name(h),
                    "rgb": [float(x) for x in mean_rgb], "uc": float(cu.mean()), "vc": float(cv.mean())})
    # merge blobs with the same hue name (occlusion can split one cube); keep largest as representative
    merged = {}
    for b in out:
        k = b["name"]
        if k not in merged:
            merged[k] = b
        else:
            a = merged[k]
            if abs(a["uc"] - b["uc"]) < 25 and abs(a["vc"] - b["vc"]) < 25:
                v = np.concatenate([a["v"], b["v"]])
                u = np.concatenate([a["u"], b["u"]])
                merged[k] = {"n": len(v), "v": v, "u": u, "hue": a["hue"], "name": k, "rgb": a["rgb"],
                             "uc": float(u.mean()), "vc": float(v.mean())}
            else:
                if b["n"] > a["n"]:
                    merged[k + "_2"] = a
                    merged[k] = b
                else:
                    merged[k + "_2"] = b
    return sorted(merged.values(), key=lambda b: b["uc"])


def _distinct_hue_count(blobs):
    names = set(b["name"] for b in blobs)
    return len(names)


def _backproject(frame, blob):
    depth = frame.front_depth
    K = np.array(frame.front_K, dtype=np.float64)
    E = np.array(frame.front_extrinsic, dtype=np.float64)
    v = blob["v"]
    u = blob["u"]
    z = depth[v, u].astype(np.float64) / 1000.0
    ok = z > 0
    if ok.sum() < 3:
        return None
    u = u[ok].astype(np.float64)
    v = v[ok].astype(np.float64)
    z = z[ok]
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], axis=1)
    R = E[:, :3]
    t = E[:, 3]
    pw = (pc - t) @ R
    zt = pw[:, 2].max()
    top = pw[pw[:, 2] > zt - 0.006]
    return [float(top[:, 0].mean()), float(top[:, 1].mean()), float(zt)]


def _summ(blobs):
    return [{"name": b["name"], "hue": round(b["hue"], 1), "n": b["n"], "uc": round(b["uc"], 1),
             "vc": round(b["vc"], 1)} for b in blobs]


def _mem_defaults(memory):
    m = {
        "stage": "cue",
        "cue_hue": -1.0,
        "cue_color": "",
        "cue_step": -1,
        "cubes_step": -1,
        "target": [],
        "target_color": "",
        "target_hue_diff": -1.0,
        "touch_calls": 0,
        "decisions": 0,
        "log": "",
    }
    for k in m:
        if k in memory and memory[k] is not None:
            m[k] = memory[k]
    return m


def perceive_decide(obs, goal, memory, last):
    """Track cue colour, wait for the three test cubes, then touch the cube of the cue colour."""
    m = _mem_defaults(memory)
    m["decisions"] = int(m["decisions"]) + 1
    frames = list(obs.frames)
    note = ""
    # 1. cue colour: first frame with exactly one coloured blob (only before the test cubes are found)
    if float(m["cue_hue"]) < 0 and m["stage"] in ("cue", "wait"):
        for f in frames:
            bl = _blobs(f.front_rgb)
            if len(bl) == 1:
                m["cue_hue"] = float(bl[0]["hue"])
                m["cue_color"] = bl[0]["name"]
                m["cue_step"] = int(f.step)
                note += "cue %s hue %.1f at step %d; " % (bl[0]["name"], bl[0]["hue"], f.step)
                break
            if len(bl) >= 3:
                break
    cur = obs.current
    if m["stage"] in ("cue", "wait"):
        bl = _blobs(cur.front_rgb)
        partial_ok = False
        if float(m["cue_hue"]) >= 0 and len(frames) >= 4:
            partial_ok = all(_test_cubes_ready(_blobs(fq.front_rgb), float(m["cue_hue"])) for fq in frames[-4:])
        if float(m["cue_hue"]) >= 0 and (_distinct_hue_count(bl) >= 3 or
                                         (partial_ok and _test_cubes_ready(bl, float(m["cue_hue"])))):
            best = min(bl, key=lambda b: _hue_diff(b["hue"], float(m["cue_hue"])))
            pos = _backproject(cur, best)
            if pos is None:
                note += "target depth missing; "
            else:
                m["target"] = pos
                m["target_color"] = best["name"]
                m["target_hue_diff"] = float(_hue_diff(best["hue"], float(m["cue_hue"])))
                m["cubes_step"] = int(cur.step)
                m["stage"] = "touch"
                note += "cubes %s at step %d -> target %s %s; " % (
                    str(_summ(bl)), cur.step, best["name"], str([round(p, 3) for p in pos]))
        elif _distinct_hue_count(bl) >= 3:
            note += "3 cubes but no cue known; "
        if m["stage"] != "touch":
            m["stage"] = "wait" if float(m["cue_hue"]) >= 0 else "cue"
            m["log"] = (str(m["log"]) + note)[-1500:]
            return {"memory": m, "act": {"tool": "act_wait_for_cubes",
                                         "args": {"max_steps": 650, "cue_hue": float(m["cue_hue"])}}}
    # touch stage
    if not m["target"]:
        m["log"] = (str(m["log"]) + "no target; ")[-1500:]
        return {"memory": m, "act": {"tool": "act_wait_for_cubes", "args": {"max_steps": 5}}}
    m["touch_calls"] = int(m["touch_calls"]) + 1
    t = m["target"]
    extra = 0.01 * (int(m["touch_calls"]) - 1)
    above = max(0.0, TOUCH_ABOVE_TOP - extra)
    m["stage"] = "touch"
    m["log"] = (str(m["log"]) + note + "touch call %d above %.3f; " % (m["touch_calls"], above))[-1500:]
    probe = 0.0
    if PROBE_ABOVE > 0 and int(m["touch_calls"]) == 1:
        above = PROBE_ABOVE
        probe = PROBE_RATE
    return {"memory": m, "act": {"tool": "act_touch", "args": {
        "x": float(t[0]), "y": float(t[1]), "z_top": float(t[2]), "above": above, "max_steps": 60,
        "probe_rate": probe}}}


def _test_cubes_ready(bl, cue_hue):
    """True when the test cubes are visible: >=3 distinct hues, or (fallback for partial occlusion) >=2
    distinct hues one of which matches the remembered cue hue within 30 deg."""
    k = _distinct_hue_count(bl)
    if k >= 3:
        return True
    if k >= 2 and cue_hue is not None and cue_hue >= 0:
        return min(_hue_diff(b["hue"], cue_hue) for b in bl) < 30.0
    return False


def act_wait_for_cubes(robot, max_steps=650, cue_hue=-1.0):
    """Hold the arm still (zero delta, gripper open) until three differently coloured cubes are visible in
    the front camera for two consecutive frames (or two hues incl. the cue hue for 4 frames), or max_steps
    elapse. Returns step counts and blob summary."""
    n = 0
    consec = 0
    consec2 = 0
    first_cue = None
    last_bl = []
    start_step = int(robot.now().step)
    while n < max_steps:
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
        f = robot.now()
        bl = _blobs(f.front_rgb)
        if len(bl) == 1 and first_cue is None:
            first_cue = {"name": bl[0]["name"], "hue": round(bl[0]["hue"], 1), "step": int(f.step)}
        if _distinct_hue_count(bl) >= 3:
            consec += 1
            last_bl = bl
            if consec >= 2:
                break
        else:
            consec = 0
        if _test_cubes_ready(bl, float(cue_hue)):
            consec2 += 1
            last_bl = bl
            if consec2 >= 4:
                break
        else:
            consec2 = 0
    f = robot.now()
    return {"steps": n, "start_step": start_step, "end_step": int(f.step),
            "cubes_seen": consec >= 2 or consec2 >= 4, "partial": consec < 2 and consec2 >= 4,
            "blobs": _summ(last_bl), "cue_seen_in_wait": first_cue, "tcp": [float(x) for x in f.tcp]}


def act_touch(robot, x, y, z_top, above=0.015, max_steps=120, probe_rate=0.0):
    """Servo the TCP (delta pose, gripper open) to hover above (x, y) at HOVER_Z, then descend to
    z_top + above and hold. Returns the measured approach trace summary if the episode has not ended."""
    f0 = robot.now()
    tcp0 = [float(v) for v in f0.tcp]
    z_touch = float(z_top) + float(above)
    centre = np.array([x, y, float(z_top) - 0.02])
    check = None
    try:
        check = robot.dry_run(float(x), float(y), z_touch, tcp0[3], tcp0[4], tcp0[5], mode="absolute")
        check = {"within": check.get("within_joint_limits"), "min_margin": check.get("min_joint_margin_rad")}
    except Exception as e:
        check = {"error": str(e)[:200]}
    trace = []
    min_d = 1e9
    reached_step = -1
    n = 0
    z_ref = None
    while n < max_steps:
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        exy = math.hypot(x - tcp[0], y - tcp[1])
        tz = HOVER_Z if exy > 0.02 else z_touch
        if probe_rate > 0 and exy <= 0.02:
            # exploratory slow descent: settle at z_touch, then lower a reference by probe_rate per step
            if z_ref is None and abs(tcp[2] - z_touch) < 0.003 and exy < 0.006:
                z_ref = z_touch
            if z_ref is not None:
                z_ref = max(float(z_top) - 0.01, z_ref - float(probe_rate))
                tz = z_ref
        if exy > 0.02 and tcp[2] < z_touch:
            tz = max(tz, tcp[2])
        err = np.array([x - tcp[0], y - tcp[1], tz - tcp[2]])
        cmd = np.clip(err / 0.1, -1.0, 1.0)
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]), 0.0, 0.0, 0.0, 1.0)
        n += 1
        tcp = np.array(robot.now().tcp, dtype=np.float64)
        d = float(np.linalg.norm(tcp[:3] - centre))
        min_d = min(min_d, d)
        if reached_step < 0 and abs(tcp[2] - z_touch) < 0.006 and exy < 0.01:
            reached_step = n
        if n % 5 == 0 or n < 3 or z_ref is not None:
            trace.append([n] + [round(float(v), 4) for v in tcp[:3]] + [round(d, 4)])
    f = robot.now()
    return {"steps": n, "end_step": int(f.step), "z_touch": z_touch, "dry_run": check, "min_dist_centre": min_d,
            "reached_step": reached_step, "final_tcp": [float(v) for v in f.tcp], "trace": trace[-40:]}
