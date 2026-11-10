"""T21 ButterChocolateMicrowave policy: open the microwave door, put the butter in the microwave, then put the
chocolate in the same place (the microwave), with perception-based targets and observation checks."""

# Door pull path relative to the handle grasp point (mean of 50 demos, frames 84..120): the door arc.
PULL_REL = [[-0.054, -0.057, 0.0], [-0.091, -0.106, 0.0], [-0.128, -0.142, 0.0], [-0.165, -0.168, 0.0],
            [-0.200, -0.188, 0.001], [-0.237, -0.202, 0.001], [-0.278, -0.208, 0.0], [-0.322, -0.205, 0.0],
            [-0.359, -0.199, 0.0], [-0.392, -0.192, 0.001]]
# Retreat after releasing the handle (relative to the handle; demo frames 135..145).
RETREAT_REL = [[-0.354, -0.251, 0.022], [-0.295, -0.314, 0.057], [-0.232, -0.374, 0.077]]
HOVER_Z = 0.62
DOOR_OPEN_X = -0.245      # door rel-x above this = not open enough (see _door_open_x)
LIFT_Z = 0.558


def _seg_move(pos, ori, grip, tol=0.006, mx=120, vmax=1.0, otol=0.05, stall=False, pass_tol=None, hold=False,
              block=False, gain=10.0):
    return {"kind": "move", "pos": [float(v) for v in pos], "ori": ori, "grip": grip, "tol": tol,
            "max": mx, "vmax": vmax, "otol": otol, "stall": stall, "pass_tol": pass_tol, "need_hold": hold,
            "block": block, "gain": gain}


def _seg_grip(grip, steps):
    return {"kind": "grip", "grip": grip, "steps": steps}


def _num(text, key, default=None):
    """Read `"key": number` from the previous act result text."""
    if not isinstance(text, str):
        return default
    i = text.rfind('"' + key + '": ')
    if i < 0:
        return default
    j = i + len(key) + 4
    k = j
    while k < len(text) and text[k] in "-+.0123456789e":
        k += 1
    try:
        return float(text[j:k])
    except ValueError:
        return default


def perceive_decide(obs, goal, memory, last):
    """Phase machine. Perceives microwave/objects at the start, verifies each stage from sensors
    (finger width, door pixels, wrist detections, butter colour inside microwave) and picks the next act_script."""
    mem = {k: memory.get(k) for k in ("phase", "handle", "butter", "choc", "table_z", "tries", "door_tries",
                                      "push_tries", "butter_placed", "log")}
    if mem["log"] is None:
        mem["log"] = []
    if mem["tries"] is None:
        mem["tries"] = 0
    if mem["door_tries"] is None:
        mem["door_tries"] = 0
    if mem["push_tries"] is None:
        mem["push_tries"] = 0
    if mem["butter_placed"] is None:
        mem["butter_placed"] = False
    phase = mem["phase"] or "start"
    cur = obs.current
    tcp = np.asarray(cur.tcp)
    width = _width(cur)
    log = list(mem["log"])[-30:]

    def out(next_phase, segments, note):
        mem["phase"] = next_phase
        log.append(note)
        mem["log"] = log[-30:]
        return {"memory": mem, "act": {"tool": "act_script", "args": {"segments": segments}}}

    if phase == "start":
        mw = _detect_microwave(cur)
        handle = mw["handle"] if mw else list(HANDLE_DEFAULT)
        y_max = (mw["y_lo"] - 0.03) if mw else 0.3
        blobs, tz = _detect_objects_agentview(cur, y_max)
        b, c = _classify(blobs)
        mem["handle"] = [float(v) for v in handle]
        mem["table_z"] = float(tz)
        mem["butter"] = [b["center"][0], b["center"][1], b["top"]] if b else list(BUTTER_DEFAULT)
        mem["choc"] = [c["center"][0], c["center"][1], c["top"]] if c else list(CHOC_DEFAULT)
        log.append("detect mw=%s butter=%s choc=%s" % (bool(mw), bool(b), bool(c)))
        return out("door_view", _door_view_segments(mem["handle"], None), "door_view")

    h = mem["handle"] or list(HANDLE_DEFAULT)
    tz = mem["table_z"] or TABLE_Z_DEFAULT

    if phase == "door_view":
        # Wrist look at the handle bar from the pre-grasp pose; trust the agentview estimate when they agree.
        bar = _detect_bar(cur)
        note = "bar=None"
        if bar is not None:
            hw = [bar["bar_x"] + BAR_TO_GRASP[0], bar["bar_y"] + BAR_TO_GRASP[1], h[2]]
            dev = math.hypot(hw[0] - h[0], hw[1] - h[1])
            note = "bar dev=%.3f" % dev
            if dev > 0.01:
                h = hw
                mem["handle"] = [float(v) for v in hw]
                note += " use_wrist"
        k = mem["door_tries"]
        off = [[0.0, 0.0], [0.0, 0.006], [0.006, 0.0], [-0.006, 0.004], [0.0, -0.004]][k % 5]
        return out("door_grasp", _door_grasp_segments(h, off), "door_grasp try=%d %s" % (k, note))

    if phase == "door_grasp":
        # Handle in the hand: fingers stop on the bar (demo width ~0.023); fully closed means a miss.
        if width > 0.006 or mem["door_tries"] > 3:
            return out("door_pull", _door_pull_segments(h, mem["butter"]), "door_pull w=%.3f" % width)
        mem["door_tries"] += 1
        segs = [_seg_grip(-1, 12), _seg_move([tcp[0], tcp[1] - 0.05, tcp[2]], SIDE_AA, -1, tol=0.01)]
        return out("door_view", segs + _door_view_segments(h, None)[1:], "door_miss w=%.3f" % width)

    if phase in ("door_pull", "door_push"):
        cnt = _door_open_count(cur, h)
        ox = _door_open_x(cur, h) if cnt >= 40 else None
        if ox is not None and ox > DOOR_OPEN_X and mem["push_tries"] < 2:
            # Door only partly open (handle slipped during the pull): push it open from the inner side.
            mem["push_tries"] += 1
            py = [-0.14, -0.10][(mem["push_tries"] - 1) % 2]
            mx = _door_max_x(cur, h, py - 0.11, py + 0.11)
            sx = (mx if mx is not None else ox + 0.03) + 0.05
            return out("door_push", _door_push_segments(h, sx, py, mem["butter"]),
                       "door_push ox=%.3f sx=%.3f" % (ox, sx))
        if cnt >= 40 or mem["door_tries"] > 4:
            # >=120 clearly open in all demos (~400); 40..120 treated as partly open/occluded (uncertain).
            return _pick_decide(mem, out, cur, "butter", tz, "door_open cnt=%d" % cnt)
        # Door not open: re-detect the microwave (door probably swung back) and grasp again.
        # Keep the handle estimate (agentview re-detection with the door ajar is biased, R007); look again with the
        # wrist from the pre-grasp pose and grasp with a different small offset.
        mem["door_tries"] += 1
        segs = [_seg_move([tcp[0], tcp[1], 0.66], None, -1, tol=0.01)]
        return out("door_view", segs + _door_view_segments(h, None)[1:], "door_retry cnt=%d" % cnt)

    if phase == "hover_butter":
        return _pick_decide(mem, out, cur, "butter", tz, "hover_butter")

    if phase == "pick_butter":
        if width > 0.012 and tcp[2] > LIFT_Z - 0.03:
            mem["tries"] = 0
            return out("place_butter", _place_butter_segments(h), "place_butter w=%.3f" % width)
        mem["tries"] += 1
        return _hover(mem, out, "butter", "pick_butter_miss w=%.3f" % width)

    if phase in ("place_butter", "place_choc") and '"blocked"' in (last or "") and mem["tries"] < 2:
        # Insertion stalled short of the target (door edge / obstacle): back out holding the object and retry
        # on a line shifted +x (away from the door side); not released outside the microwave.
        mem["tries"] += 1
        dx = 0.025 * mem["tries"]
        back = [_seg_move([tcp[0], h[1] - 0.15, tcp[2]], SIDE_AA, 1, tol=0.01, hold=True)]
        segs = (_place_butter_segments(h, dx, False) if phase == "place_butter"
                else _place_choc_segments(h, dx, False))
        return out(phase, back + segs, "%s_blocked retry dx=%.3f" % (phase, dx))

    if phase == "place_butter":
        # Wrist looks into the microwave after retreating: the butter print is visible if it is inside.
        cnt = _red_count(cur)
        dropped = '"dropped"' in (last or "")
        mem["butter_placed"] = bool(cnt > 60 and width > 0.06 and not dropped)
        if (dropped or cnt <= 60) and mem["tries"] < 2:
            mem["tries"] += 1
            return out("redetect_butter", _viewpoint_segments(tcp, width),
                       "butter_lost red=%d dropped=%s" % (cnt, dropped))
        segs = [_seg_move([tcp[0], tcp[1], 0.71], SIDE_AA, -1, tol=0.01)]
        c = mem["choc"]
        segs.append(_seg_move([c[0], c[1], HOVER_Z + 0.06], DOWN_AA, -1, tol=0.012, otol=0.08, pass_tol=0.05))
        segs.append(_seg_move([c[0], c[1], HOVER_Z], DOWN_AA, -1, tol=0.006))
        mem["tries"] = 0
        return out("hover_choc", segs, "butter_check red=%d placed=%s" % (cnt, mem["butter_placed"]))

    if phase in ("redetect_butter", "redetect_choc"):
        name = "butter" if phase == "redetect_butter" else "choc"
        mw_y = (h[1] - 0.04)
        blobs, tz2 = _detect_objects_agentview(cur, mw_y)
        b, c = _classify(blobs)
        found = b if name == "butter" else c
        if found is None and len(blobs) == 1 and name == "choc":
            found = blobs[0]
        if found is not None:
            mem[name] = [found["center"][0], found["center"][1], found["top"]]
            segs = [_seg_move([mem[name][0], mem[name][1], HOVER_Z], DOWN_AA, -1, tol=0.008, otol=0.08)]
            return out("hover_" + name, segs, "redetect_%s found" % name)
        if name == "butter":
            # Not on the table: assume it went into the microwave (uncertain) and continue with the chocolate.
            c = mem["choc"]
            segs = [_seg_move([c[0], c[1], HOVER_Z], DOWN_AA, -1, tol=0.008, otol=0.08)]
            mem["tries"] = 0
            return out("hover_choc", segs, "redetect_butter none")
        return out("final", [_seg_grip(-1, 20)], "redetect_choc none")

    if phase == "hover_choc":
        return _pick_decide(mem, out, cur, "choc", tz, "hover_choc")

    if phase == "pick_choc":
        if width > 0.012 and tcp[2] > LIFT_Z - 0.03:
            mem["tries"] = 0
            return out("place_choc", _place_choc_segments(h), "place_choc w=%.3f" % width)
        mem["tries"] += 1
        return _hover(mem, out, "choc", "pick_choc_miss w=%.3f" % width)

    if phase == "place_choc" and '"dropped"' in (last or "") and mem["tries"] < 2:
        mem["tries"] += 1
        return out("redetect_choc", _viewpoint_segments(tcp, width), "choc_dropped")

    if phase == "place_choc":
        # Still running: chocolate was inserted (and released) but success was not reported yet.
        segs = [_seg_move([h[0] - 0.105, h[1] + 0.02, 0.60], SIDE_AA, -1, tol=0.006, mx=60, vmax=0.4, stall=True),
                _seg_grip(-1, 30)]
        return out("final", segs, "place_choc_deeper")

    # final / unknown: hold still and observe (settling period), then back off once.
    mem["tries"] += 1
    if mem["tries"] == 3:
        return out("final", [_seg_move([tcp[0], tcp[1] - 0.12, tcp[2] + 0.05], None, -1, tol=0.01),
                             _seg_grip(-1, 40)], "retreat_wait")
    return out("final", [_seg_grip(-1, 40)], "wait")


def _viewpoint_segments(tcp, width):
    """Open, back out of the microwave (-y) if inside, rise, and park the arm away from the table objects."""
    segs = [_seg_grip(-1, 12)]
    if tcp[1] > 0.2:
        segs.append(_seg_move([tcp[0], 0.2, tcp[2]], None, -1, tol=0.012))
    segs.append(_seg_move([tcp[0], min(tcp[1], 0.2), 0.70], None, -1, tol=0.012))
    segs.append(_seg_move([-0.10, 0.05, 0.70], DOWN_AA, -1, tol=0.015, otol=0.08))
    return segs


def _red_count(frame):
    rgb = np.asarray(frame.wrist_rgb).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return int(((r - g > 60) & (r - b > 60)).sum())


def _hover(mem, out, name, note):
    """Back to a top-down viewpoint above the object's last known position for a fresh wrist look."""
    o = mem[name]
    if mem["tries"] > 3:
        # Bounded: give up on this object's re-grasp loop and go on with what we have.
        nxt = "place_butter" if name == "butter" else "place_choc"
        h = mem["handle"]
        segs = _place_butter_segments(h) if name == "butter" else _place_choc_segments(h)
        return out(nxt, segs, note + " giveup")
    segs = [_seg_grip(-1, 12), _seg_move([o[0], o[1], HOVER_Z], DOWN_AA, -1, tol=0.008)]
    return out("hover_" + name, segs, note)


def _pick_decide(mem, out, cur, name, tz, note):
    """At a top-down hover: refine the object position with the wrist depth camera, then grasp and lift."""
    o = mem[name]
    ref = _wrist_refine(cur, o, tz)
    if ref is not None:
        o = [ref[0], ref[1], ref[2]]
        mem[name] = [float(v) for v in o]
    gx = o[0] - 0.004
    gy = o[1]
    zg = max(tz + 0.006, o[2] - 0.014)
    off = [0.0, 0.0]
    if mem["tries"] and ref is None:
        # Without a fresh detection vary the grasp point slightly instead of repeating the same miss.
        k = mem["tries"]
        off = [0.008 * ((k % 2) * 2 - 1), 0.0]
    segs = [_seg_grip(-1, 1),
            _seg_move([gx + off[0], gy + off[1], HOVER_Z], DOWN_AA, -1, tol=0.006, otol=0.04),
            _seg_move([gx + off[0], gy + off[1], zg + 0.06], DOWN_AA, -1, tol=0.006, otol=0.03),
            _seg_move([gx + off[0], gy + off[1], zg], DOWN_AA, -1, tol=0.004, otol=0.03, mx=80, vmax=0.5,
                      stall=True),
            _seg_grip(1, 20),
            _seg_move([gx + off[0], gy + off[1], LIFT_Z], DOWN_AA, 1, tol=0.008, mx=80)]
    return out("pick_" + name, segs, note + " ref=%s" % (ref is not None))


def _door_view_segments(h, unused):
    """Sideways gripper to the pre-grasp viewpoint handle+(-0.055,-0.07) (demo frames 24-30)."""
    hx, hy, hz = h
    return [_seg_grip(-1, 1),
            _seg_move([hx - 0.056, hy - 0.11, hz], SIDE_AA, -1, tol=0.01, otol=0.04, mx=150),
            _seg_move([hx - 0.055, hy - 0.07, hz], SIDE_AA, -1, tol=0.004, otol=0.02, mx=60)]


def _door_grasp_segments(h, off):
    hx, hy, hz = h
    hx = hx + off[0]
    hy = hy + off[1]
    return [_seg_move([hx - 0.055, hy - 0.06, hz], SIDE_AA, -1, tol=0.005, otol=0.03),
            _seg_move([hx + 0.012, hy + 0.004, hz], SIDE_AA, -1, tol=0.003, otol=0.03, mx=60, vmax=0.4,
                      stall=True),
            _seg_grip(1, 20)]


def _door_pull_segments(h, butter):
    hx, hy, hz = h
    segs = []
    for i, r in enumerate(PULL_REL):
        last = i == len(PULL_REL) - 1
        # Saturated (expert-like ~1 cm/step) through the via-points: slow pulls let the handle slide out of the
        # fingers and left the door ~70 deg open for handle y >= 0.424 (R022, R027, R032-R034).
        segs.append(_seg_move([hx + r[0], hy + r[1], hz + r[2]], SIDE_AA, 1, tol=0.006, mx=40,
                              pass_tol=None if last else 0.02, gain=10.0 if last else 40.0))
        segs[-1]["keep_dir"] = True
    segs.append(_seg_grip(-1, 12))
    for r in RETREAT_REL:
        segs.append(_seg_move([hx + r[0], hy + r[1], hz + r[2]], None, -1, tol=0.01, pass_tol=0.025))
    b = butter or BUTTER_DEFAULT
    segs.append(_seg_move([b[0], b[1], HOVER_Z + 0.05], DOWN_AA, -1, tol=0.012, otol=0.08, pass_tol=0.04))
    segs.append(_seg_move([b[0], b[1], HOVER_Z], DOWN_AA, -1, tol=0.006, otol=0.04))
    return segs


def _door_push_segments(h, sx, py, butter):
    """Gripper down, fingers closed, enter between microwave and door at rel (sx, py) from above the door top
    (door z 0.465-0.645), push along -x to rel x -0.31 (door ~100 deg), lift and return over the butter."""
    hx, hy, hz = h
    x0 = hx + sx
    y0 = hy + py
    b = butter or BUTTER_DEFAULT
    return [_seg_move([x0, y0, 0.74], DOWN_AA, -1, tol=0.012, otol=0.05),
            _seg_grip(1, 10),
            _seg_move([x0, y0, 0.56], DOWN_AA, 1, tol=0.006, vmax=0.5, stall=True),
            _seg_move([hx - 0.31, y0, 0.56], DOWN_AA, 1, tol=0.008, mx=100, vmax=0.4, stall=True),
            _seg_move([hx - 0.25, y0, 0.56], DOWN_AA, 1, tol=0.01, mx=40),
            _seg_move([hx - 0.25, y0, 0.74], DOWN_AA, 1, tol=0.012),
            _seg_grip(-1, 10),
            _seg_move([b[0], b[1], HOVER_Z + 0.05], DOWN_AA, -1, tol=0.012, otol=0.08, pass_tol=0.04),
            _seg_move([b[0], b[1], HOVER_Z], DOWN_AA, -1, tol=0.006, otol=0.04)]


def _place_butter_segments(h, dx=0.0, stage=True):
    """Staging (demo world point), rotate sideways, push +y into the microwave (blocked-stall aborts), release,
    retreat. dx shifts the insertion line in +x (away from the door side) on retries."""
    hx, hy, hz = h
    segs = []
    if stage:
        segs += [_seg_move([hx - 0.076, 0.096, 0.556], DOWN_AA, 1, tol=0.01, pass_tol=None, hold=True),
                 _seg_move([hx - 0.076, 0.092, 0.552], SIDE_AA, 1, tol=0.01, otol=0.03, hold=True)]
    return segs + [
            _seg_move([hx - 0.097 + dx, hy - 0.10, 0.590], SIDE_AA, 1, tol=0.01, pass_tol=0.03, hold=True,
                      stall=True, block=True),
            _seg_move([hx - 0.097 + dx, hy + 0.039, 0.598], SIDE_AA, 1, tol=0.006, mx=100, stall=True, block=True),
            _seg_move([hx - 0.081 + dx, hy + 0.058, 0.582], SIDE_AA, 1, tol=0.005, mx=40, vmax=0.4, stall=True),
            _seg_grip(-1, 20),
            _seg_move([hx - 0.052 + dx, hy - 0.089, 0.573], SIDE_AA, -1, tol=0.008)]


def _place_choc_segments(h, dx=0.0, stage=True):
    hx, hy, hz = h
    segs = []
    if stage:
        segs += [_seg_move([-0.03, -0.003, 0.559], DOWN_AA, 1, tol=0.01, hold=True),
                 _seg_move([-0.03, -0.003, 0.555], SIDE_AA, 1, tol=0.01, otol=0.03, hold=True)]
    return segs + [
            _seg_move([hx - 0.11 + dx, hy - 0.16, 0.595], SIDE_AA, 1, tol=0.01, pass_tol=0.03, hold=True,
                      stall=True, block=True),
            _seg_move([hx - 0.11 + dx, hy - 0.03, 0.60], SIDE_AA, 1, tol=0.006, mx=80, vmax=0.5, stall=True,
                      block=True),
            _seg_grip(1, 20),
            _seg_move([hx - 0.11 + dx, hy - 0.005, 0.60], SIDE_AA, 1, tol=0.006, mx=60, vmax=0.4, stall=True),
            _seg_grip(-1, 20),
            _seg_move([hx - 0.08 + dx, hy - 0.12, 0.60], SIDE_AA, -1, tol=0.01)]


def act_script(robot, segments):
    """Run a list of segments. move: P-control the TCP to pos (world m) and optional orientation `ori`
    (robosuite axis-angle) with gripper sign `grip`; stops at tol, at pass_tol (via-point), on stall or max steps.
    grip: hold pose for `steps` steps with gripper sign. Returns per-segment measured TCP/fingers/steps."""
    report = []
    prev = None
    for s in segments:
        if s.get("if_prev") is not None and prev != s.get("if_prev"):
            report.append({"k": "skip"})
            continue
        if s["kind"] == "grip":
            for _ in range(int(s["steps"])):
                robot.step(0, 0, 0, 0, 0, 0, s["grip"])
            f = robot.now()
            report.append({"k": "g", "w": round(_width(f), 4)})
            continue
        target = np.asarray(s["pos"], dtype=float)
        ori = s.get("ori")
        tol = s.get("tol", 0.006)
        ptol = s.get("pass_tol")
        otol = s.get("otol", 0.05)
        vmax = s.get("vmax", 1.0)
        hist = []
        n = 0
        status = "max"
        for n in range(int(s.get("max", 120))):
            f = robot.now()
            tcp = np.asarray(f.tcp, dtype=float)
            err = target - tcp[:3]
            d = float(np.linalg.norm(err))
            oe = np.zeros(3) if ori is None else _ori_err(ori, tcp[3:6])
            on = float(np.linalg.norm(oe))
            if on < otol and (d < tol or (ptol is not None and d < ptol)):
                status = "ok"
                break
            if s.get("need_hold") and _width(f) < 0.008:
                status = "dropped"
                break
            hist.append(tcp[:3])
            if s.get("stall") and len(hist) > 12 and float(np.linalg.norm(hist[-1] - hist[-9])) < 0.0015:
                status = "stall"
                break
            a = err * s.get("gain", 10.0)
            if s.get("keep_dir"):
                # scale uniformly so the commanded direction stays on the via-point polyline
                m = float(np.max(np.abs(a)))
                if m > vmax:
                    a = a * (vmax / m)
            a = np.clip(a, -vmax, vmax)
            r = np.clip(oe * 2.0, -0.5, 0.5)
            robot.step(a[0], a[1], a[2], r[0], r[1], r[2], s["grip"])
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        if s.get("block") and status == "stall" and float(np.linalg.norm(target - tcp[:3])) > 0.03:
            status = "blocked"
        report.append({"k": "m", "st": status, "n": n, "e": round(float(np.linalg.norm(target - tcp[:3])), 4),
                       "w": round(_width(f), 4)})
        prev = status
        if status in ("dropped", "blocked"):
            break
    f = robot.now()
    return {"segments": report, "tcp": [round(float(v), 4) for v in f.tcp[:3]], "width": round(_width(f), 4)}
