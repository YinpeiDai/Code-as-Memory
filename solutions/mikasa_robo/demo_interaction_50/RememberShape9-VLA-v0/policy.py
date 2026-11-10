"""RememberShape9 policy: remember the cue shape's top-down height map, find the
candidate with the same height map once the 9 candidates appear, and touch it.

Perception (pure numpy):
  * blue pixels (b > r+40 and b > g+40) in front and wrist RGB are back-projected
    with depth, K and extrinsic into world points;
  * points with z > 2 mm are rasterised into a 3 mm world-XY grid holding the max
    height per cell (then a 3x3 max filter fills sampling holes);
  * the cue's height map (31x31 cells around its centroid) is stored in memory;
  * when candidates appear, the template is slid over the scene height map and the
    mean |dz| over the template support dilated by 2 cells is minimised.
  * the touch point is the cue's reference point (world origin, where every demo
    cue was placed) moved by the matched shift (= the object's own origin, the
    point the expert demonstrations approach).
"""

RES = 0.003
X0 = -0.40
Y0 = -0.40
NX = 200
NY = 270
HW = 15
CUE_REF = [0.0, 0.0]
WRIST_CAND_PX = 1000
FRONT_CAND_PX = 350
CUE_MIN_PX = 20
Z_SAFE = 0.06
Z_GOAL = 0.012
Z_PRESS = -0.01


def _blue(rgb):
    im = rgb.astype(np.int32)
    r = im[..., 0]
    g = im[..., 1]
    b = im[..., 2]
    return (b > r + 40) & (b > g + 40)


def _points(frame, cam):
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    m = _blue(rgb) & (depth > 0)
    vs, us = np.nonzero(m)
    if len(vs) == 0:
        return np.zeros((0, 3))
    d = depth[vs, us].astype(float) / 1000.0
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _scene_points(frame):
    return np.concatenate([_points(frame, "wrist"), _points(frame, "front")], 0)


def _hmap(P):
    H = np.zeros((NX, NY))
    if len(P) == 0:
        return H
    P = P[P[:, 2] > 0.002]
    i = np.floor((P[:, 0] - X0) / RES).astype(int)
    j = np.floor((P[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < NX) & (j >= 0) & (j < NY)
    np.maximum.at(H, (i[ok], j[ok]), P[ok, 2])
    Hp = np.pad(H, 1)
    out = H.copy()
    for di in range(3):
        for dj in range(3):
            out = np.maximum(out, Hp[di:di + NX, dj:dj + NY])
    return out


def _dilate(M, n):
    for _ in range(n):
        P = np.pad(M, 1)
        M = P[1:-1, 1:-1] | P[:-2, 1:-1] | P[2:, 1:-1] | P[1:-1, :-2] | P[1:-1, 2:]
    return M


def _cell_xy(i, j):
    return [X0 + (i + 0.5) * RES, Y0 + (j + 0.5) * RES]


def _blue_counts(frame):
    return int(_blue(frame.wrist_rgb).sum()), int(_blue(frame.front_rgb).sum())


def _is_candidates(frame):
    w, f = _blue_counts(frame)
    return w > WRIST_CAND_PX or f > FRONT_CAND_PX


def _cue_template(frame):
    H = _hmap(_scene_points(frame))
    occ = np.argwhere(H > 0)
    if len(occ) < 8:
        return None
    ci, cj = np.round(occ.mean(0)).astype(int)
    Hp = np.pad(H, HW)
    T = Hp[ci:ci + 2 * HW + 1, cj:cj + 2 * HW + 1]
    return {"template": np.round(T, 4).tolist(), "center": _cell_xy(ci, cj),
            "cells": int(len(occ)), "zmax": float(H.max())}


ANGLES = [-45.0, -37.5, -30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0]


def _rot_template(Tm, deg):
    """Rotate a square height template about its centre (nearest-neighbour)."""
    n = Tm.shape[0]
    c = (n - 1) / 2.0
    th = math.radians(deg)
    ct = math.cos(th)
    st = math.sin(th)
    ii, jj = np.mgrid[0:n, 0:n]
    di = ii - c
    dj = jj - c
    si = np.round(ct * di + st * dj + c).astype(int)
    sj = np.round(-st * di + ct * dj + c).astype(int)
    ok = (si >= 0) & (si < n) & (sj >= 0) & (sj < n)
    out = np.zeros_like(Tm)
    out[ok] = Tm[si[ok], sj[ok]]
    return out


def _match(template, center, frame, H=None):
    """Slide (and rotate, +-45 deg) the cue height template over the scene height map.
    Score = mean |dz| over the template support dilated by 2 cells; lower is better."""
    Tm = np.asarray(template, dtype=float)
    if H is None:
        H = _hmap(_scene_points(frame))
    Hs = np.pad(H, HW)
    occ = np.argwhere(Hs > 0)
    if len(occ) == 0 or (Tm > 0).sum() == 0:
        return None
    Ts = []
    for a in ANGLES:
        Tr = _rot_template(Tm, a)
        M = _dilate(Tr > 0, 2)
        Ts.append((a, Tr, M, Tr[M]))
    coarse = [t for t in Ts if t[0] in (-45.0, -22.5, 0.0, 22.5, 45.0)]

    def win(i, j):
        W = Hs[i - HW:i + HW + 1, j - HW:j + HW + 1]
        return W if W.shape == Tm.shape else None

    sc = {}
    for i, j in occ:
        key = (int(i) // 2 * 2, int(j) // 2 * 2)
        if key in sc:
            continue
        W = win(key[0], key[1])
        if W is None:
            continue
        sc[key] = min(float(np.abs(W[M] - tv).mean()) for a, Tr, M, tv in coarse)
    ranked = sorted(sc, key=lambda k: sc[k])
    seeds = []
    for p in ranked:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) * RES > 0.03 for q in seeds):
            seeds.append(p)
        if len(seeds) >= 8:
            break
    res = []
    for i0, j0 in seeds:
        best = (1e9, None, None)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                W = win(i0 + di, j0 + dj)
                if W is None:
                    continue
                for a, Tr, M, tv in Ts:
                    v = float(np.abs(W[M] - tv).mean())
                    if v < best[0]:
                        best = (v, (i0 + di, j0 + dj), a)
        if best[1] is not None:
            res.append(best)
    if not res:
        return None
    res = sorted(res, key=lambda r: r[0])
    best, (bi, bj), ang = res[0]
    second = res[1][0] if len(res) > 1 else None
    xy = _cell_xy(bi - HW, bj - HW)
    th = math.radians(ang)
    ox = CUE_REF[0] - center[0]
    oy = CUE_REF[1] - center[1]
    # template rotation by +ang maps cue offset (ox, oy) to R(ang) (ox, oy) in (i, j) = (x, y)
    rx = math.cos(th) * ox - math.sin(th) * oy
    ry = math.sin(th) * ox + math.cos(th) * oy
    target = [xy[0] + rx, xy[1] + ry]
    Tr = [t for t in Ts if t[0] == ang][0]
    touch, ztop = _touch_point(Hs, Tr[1], Tr[2], bi, bj)
    others = []
    for v, (i, j), a in res[1:5]:
        c = _cell_xy(i - HW, j - HW)
        others.append([round(c[0], 3), round(c[1], 3), round(v, 4), a])
    return {"target": target, "best": best, "second": second, "touch": touch, "ztop": ztop,
            "angle": ang, "matched_centroid": xy, "others": others}


def _touch_point(Hs, Tm, M, bi, bj):
    """A solid point on the matched object's top surface: among window cells (inside the
    dilated template support) within 70 % of the top height, the one with the most such
    cells in its 5x5 neighbourhood, ties broken by distance to their centroid."""
    W = Hs[bi - HW:bi + HW + 1, bj - HW:bj + HW + 1] * M
    if W.max() <= 0:
        return _cell_xy(bi - HW, bj - HW), 0.02
    top = float(np.percentile(W[W > 0], 95))
    high = W >= 0.7 * top
    P = np.pad(high.astype(float), 2)
    n = W.shape[0]
    solid = np.zeros(W.shape)
    for di in range(5):
        for dj in range(5):
            solid += P[di:di + n, dj:dj + n]
    solid = solid * high
    hc = np.argwhere(high).mean(0)
    ii, jj = np.mgrid[0:n, 0:n]
    dist = np.hypot(ii - hc[0], jj - hc[1])
    sc = solid - 0.05 * dist
    k = int(np.argmax(np.where(high, sc, -1e9)))
    ti, tj = k // n, k % n
    xy = _cell_xy(bi - HW + ti - HW, bj - HW + tj - HW)
    return xy, top


def _fields(memory):
    keys = ["stage", "cue_seen", "cue_step", "cue_template", "cue_center", "cue_info",
            "target", "touch", "ztop", "match_best", "match_second", "match_others", "steps_seen", "log"]
    out = {}
    for k in keys:
        out[k] = memory.get(k)
    if out["stage"] is None:
        out["stage"] = "start"
    if out["cue_seen"] is None:
        out["cue_seen"] = False
    if out["log"] is None:
        out["log"] = ""
    if out["steps_seen"] is None:
        out["steps_seen"] = 0
    return out


def perceive_decide(obs, goal, memory, last):
    """Stage machine: record cue -> wait for candidates -> match -> reach/touch."""
    mem = _fields(memory)
    frames = obs.frames
    cur = obs.current
    mem["steps_seen"] = int(cur.step)
    # record the cue as soon as a small single blue object is seen
    if not mem["cue_seen"]:
        for fr in frames:
            w, f = _blue_counts(fr)
            if (w > CUE_MIN_PX or f > CUE_MIN_PX // 2) and not _is_candidates(fr):
                cue = _cue_template(fr)
                if cue is not None:
                    mem["cue_seen"] = True
                    mem["cue_step"] = int(fr.step)
                    mem["cue_template"] = cue["template"]
                    mem["cue_center"] = cue["center"]
                    mem["cue_info"] = "cells=%d zmax=%.4f" % (cue["cells"], cue["zmax"])
                    mem["log"] += "cue@%d %s; " % (fr.step, mem["cue_info"])
                    break
    stage = mem["stage"]
    if stage in ("start", "wait"):
        if _is_candidates(cur):
            mem["stage"] = "settle"
            mem["log"] += "cand@%d; " % (cur.step,)
            return {"memory": mem, "act": {"tool": "act_settle", "args": {"max_steps": 2}}}
        mem["stage"] = "wait"
        return {"memory": mem, "act": {"tool": "act_wait_candidates", "args": {"max_steps": 30}}}
    if stage == "settle":
        mem["log"] += "settle: %s; " % (str(last)[:120],)
        if _is_candidates(cur):
            if not mem["cue_seen"]:
                mem["log"] += "candidates without cue; "
                mem["stage"] = "nocue"
                return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 30}}}
            res = _match(mem["cue_template"], mem["cue_center"], cur)
            if res is None:
                mem["stage"] = "nomatch"
                return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 30}}}
            mem["target"] = [round(res["target"][0], 4), round(res["target"][1], 4)]
            mem["touch"] = [round(res["touch"][0], 4), round(res["touch"][1], 4)]
            mem["ztop"] = round(res["ztop"], 4)
            mem["match_best"] = round(res["best"], 5)
            mem["match_second"] = None if res["second"] is None else round(res["second"], 5)
            mem["match_others"] = res["others"]
            mem["log"] += "match@%d target=%s best=%.4f 2nd=%s angle=%s; " % (
                cur.step, mem["target"], res["best"], mem["match_second"], res["angle"])
            mem["stage"] = "reach"
            return {"memory": mem, "act": {"tool": "act_reach", "args": {
                "x": mem["touch"][0], "y": mem["touch"][1], "ztop": mem["ztop"]}}}
        mem["stage"] = "wait"
        return {"memory": mem, "act": {"tool": "act_wait_candidates", "args": {"max_steps": 30}}}
    if stage == "reach":
        mem["stage"] = "press"
        mem["log"] += "reach result: %s; " % (str(last)[:200],)
        return {"memory": mem, "act": {"tool": "act_press", "args": {
            "x": mem["touch"][0], "y": mem["touch"][1], "ztop": mem["ztop"]}}}
    mem["stage"] = "hold"
    return {"memory": mem, "act": {"tool": "act_hold", "args": {"steps": 30}}}


def _front_change(fa, fb):
    """Pixels whose front blue mask differs or whose blue depth moved > 3 mm."""
    ma = _blue(fa.front_rgb)
    mb = _blue(fb.front_rgb)
    both = ma & mb
    dd = np.abs(fa.front_depth.astype(np.int32) - fb.front_depth.astype(np.int32))
    return int((ma ^ mb).sum() + (both & (dd > 3)).sum())


def act_settle(robot, max_steps=3, gripper=-1.0, tol=8):
    """Candidates may still be falling/settling on their first visible frame. Descend
    straight down at full speed (useful travel anyway) one step at a time until two
    consecutive front-camera frames agree (blue mask/depth change < tol pixels).
    Returns the per-step change counts and whether the scene was judged settled."""
    prev = robot.now()
    diffs = []
    settled = False
    for k in range(max_steps):
        fr = robot.step(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, gripper)
        d = _front_change(prev, fr)
        diffs.append(d)
        prev = fr
        if d < tol:
            settled = True
            break
    return {"steps": len(diffs), "changes": diffs, "settled": settled, "step": int(prev.step)}


def act_wait_candidates(robot, max_steps=30):
    """Send zero arm commands (gripper open) until the candidate objects appear.
    Returns the number of steps waited and blue pixel counts at the end."""
    n = 0
    fr = robot.now()
    while not _is_candidates(fr) and n < max_steps:
        fr = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        n += 1
    w, f = _blue_counts(fr)
    return {"waited": n, "candidates": bool(_is_candidates(fr)), "wrist_px": w, "front_px": f,
            "step": int(fr.step)}


def _cmd_to(tcp, x, y, z):
    ex = x - float(tcp[0])
    ey = y - float(tcp[1])
    ez = z - float(tcp[2])
    return [max(-1.0, min(1.0, ex / 0.1)), max(-1.0, min(1.0, ey / 0.1)),
            max(-1.0, min(1.0, ez / 0.1))]


def act_reach(robot, x, y, ztop, max_steps=18, gripper=-1.0, hold_push=0.03):
    """Close the gripper and bring the fingertips onto the object's top surface at world
    (x, y); ztop is the object's top height. Full-speed approach; the TCP stays at least
    4 cm above ztop until the XY error is under 1.5 cm, then descends toward ztop - 1 cm.
    Once the descent stalls (contact), XY commands are zeroed and only a small constant
    downward offset (hold_push, in command units) is sent so the arm rests still while in
    contact. Returns the measured trace and the step at which contact was detected."""
    trace = []
    fr = robot.now()
    margin = None
    contact_at = None
    for k in range(max_steps):
        tcp = fr.tcp
        z0 = float(tcp[2])
        exy = math.hypot(x - float(tcp[0]), y - float(tcp[1]))
        if contact_at is None:
            zg = ztop - 0.01 if exy < 0.015 else ztop + 0.04
            c = _cmd_to(tcp, x, y, zg)
            if exy < 0.015 and z0 < ztop + 0.05:
                c[2] = max(c[2], -0.5)
        else:
            c = [0.0, 0.0, -abs(hold_push)]
        if k == 0:
            chk = robot.dry_run(c[0], c[1], c[2], 0.0, 0.0, 0.0, mode="delta")
            margin = chk.get("min_joint_margin_rad")
        fr = robot.step(c[0], c[1], c[2], 0.0, 0.0, 0.0, gripper)
        z1 = float(fr.tcp[2])
        trace.append([round(float(v), 3) for v in fr.tcp[:3]] + [round(float(fr.fingers[0]), 3)])
        if contact_at is None and exy < 0.02 and z1 < ztop + 0.03 and z0 - z1 < 0.003:
            contact_at = int(fr.step)
    t = fr.tcp
    return {"steps": len(trace), "tcp": [round(float(v), 4) for v in t[:3]],
            "contact_at": contact_at,
            "xy_err": round(math.hypot(x - float(t[0]), y - float(t[1])), 4),
            "fingers": [round(float(v), 4) for v in fr.fingers], "first_margin": margin,
            "trace": trace}


def act_press(robot, x, y, ztop, steps=6, gripper=-1.0):
    """Fallback after a stalled reach: small XY circle (1 cm) around (x, y) while pushing
    toward ztop - 3 cm, to make contact with the target. Returns the trace."""
    fr = robot.now()
    trace = []
    for k in range(steps):
        a = 2.0 * math.pi * k / max(1, steps)
        c = _cmd_to(fr.tcp, x + 0.01 * math.cos(a), y + 0.01 * math.sin(a), ztop - 0.03)
        fr = robot.step(c[0], c[1], c[2], 0.0, 0.0, 0.0, gripper)
        trace.append([round(float(v), 3) for v in fr.tcp[:3]])
    return {"tcp": [round(float(v), 4) for v in fr.tcp[:3]], "trace": trace,
            "fingers": [round(float(v), 4) for v in fr.fingers]}


def act_hold(robot, steps=30):
    """Hold the arm still with zero commands for `steps` native steps."""
    fr = robot.now()
    for k in range(steps):
        fr = robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    return {"held": steps, "tcp": [round(float(v), 4) for v in fr.tcp[:3]]}
