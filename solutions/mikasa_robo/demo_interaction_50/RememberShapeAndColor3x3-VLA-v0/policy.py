"""RememberShapeAndColor3x3: remember the cue (colour + shape), wait, touch the matching object.

Perception: saturated red/green/blue pixel masks in the fixed front camera and the wrist
camera, 8-connected components, back-projection with depth + calibration into world
coordinates (table plane z = 0). Shape features: top height (T-shape ~0.016 m, sphere and
cube ~0.040 m) and metric visible area (cube clearly larger than sphere).
Control: closed-loop incremental TCP motion with a fitted first-order lag model.
"""

COLORS = ("red", "green", "blue")
T_HEIGHT_MAX = 0.026      # p95 height below this -> T-shape (T top 0.016, others 0.040)
FRONT_CUBE_AREA = 16.0    # cm^2 normalised visible area thresholds (demos: sphere<=13.4, cube>=18.8 front)
WRIST_CUBE_AREA = 15.5    # wrist: sphere<=14.7, cube>=16 (unoccluded)
MIN_PIX = 4
HOVER_ABOVE_TOP = 0.014   # TCP height above the object top for the primary touch (fingertips ~8 mm above)
# fitted response of one control step (demos): v' = A_GAIN*cmd_m + B_MOM*v
A_GAIN = 0.29
B_MOM = 0.375


def _masks(rgb):
    im = rgb.astype(np.int32)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    red = (r > 80) & (g * 100 < 35 * r) & (b * 100 < 35 * r)
    gre = (g > 60) & (g * 10 > r * 16) & (g * 10 > b * 16)
    blu = (b > 60) & (b * 10 > r * 16) & (b * 10 > g * 16)
    return {"red": red, "green": gre, "blue": blu}


def _components(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    if len(ys) > 20000:
        return []
    seen = {}
    pix = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    for p in zip(ys.tolist(), xs.tolist()):
        if p in seen:
            continue
        stack = [p]
        seen[p] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pix and q not in seen:
                        seen[q] = True
                        stack.append(q)
        comps.append(np.array(pts, dtype=np.int64))
    return comps


def _backproject(pts, depth, K, E):
    d = depth[pts[:, 0], pts[:, 1]].astype(np.float64) / 1000.0
    ok = d > 0
    pts = pts[ok]
    d = d[ok]
    if len(d) == 0:
        return np.zeros((0, 3)), d
    u = pts[:, 1] + 0.5
    v = pts[:, 0] + 0.5
    xc = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], 1)
    R = np.asarray(E, dtype=np.float64)[:, :3]
    t = np.asarray(E, dtype=np.float64)[:, 3]
    return (xc - t) @ R, d


def _cluster_tall(xy, idx, z):
    """Greedy 3 cm clustering of tall points idx; fragments closer than 4 cm are merged."""
    centres = []
    rem = list(idx)
    while rem and len(centres) < 4:
        seed = max(rem, key=lambda i: z[i])
        d = np.hypot(xy[rem, 0] - xy[seed, 0], xy[rem, 1] - xy[seed, 1])
        member = [rem[j] for j in range(len(rem)) if d[j] < 0.03]
        c = np.median(xy[member], 0)
        d2 = np.hypot(xy[rem, 0] - c[0], xy[rem, 1] - c[1])
        member = set(rem[j] for j in range(len(rem)) if d2[j] < 0.03) | set(member)
        centres.append([c, len(member)])
        rem = [i for i in rem if i not in member]
    changed = True
    while changed and len(centres) > 1:
        changed = False
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                ci, cj = centres[i], centres[j]
                if math.hypot(ci[0][0] - cj[0][0], ci[0][1] - cj[0][1]) < 0.04:
                    w = ci[1] + cj[1]
                    centres[i] = [(ci[0] * ci[1] + cj[0] * cj[1]) / w, w]
                    centres.pop(j)
                    changed = True
                    break
            if changed:
                break
    tot = float(sum(c[1] for c in centres))
    return [c[0] for c in centres if c[1] >= 0.2 * tot]


def _split_blob(X, area_per_pt):
    """Split one colour blob (world points) into objects; same-colour objects may touch.

    T-shape top points (0.011 < z < 0.0195) farther than 3 cm from every tall top (z > 0.033,
    cube/sphere) form a separate T. Tall tops are split into several objects only when the tall
    part is larger than one cube can look (> 32 cm^2). Returns a list of (centre_xy, point_mask)."""
    z = X[:, 2]
    xy = X[:, :2]
    tall = np.nonzero(z > 0.033)[0]
    centres = []
    if len(tall) >= 3:
        centres = [np.median(xy[tall], 0)]
        low = (z > 0.011) & (z < 0.0195)
        dmin = np.hypot(xy[:, 0] - centres[0][0], xy[:, 1] - centres[0][1])
        t_far = low & (dmin > 0.03)
        n_tall_part = len(z) - int(t_far.sum())
        if n_tall_part * area_per_pt > 32.0:
            cs = _cluster_tall(xy, tall, z)
            if len(cs) >= 2:
                centres = cs
        dmin = np.min(np.stack([np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) for c in centres], 0), 0)
        far = low & (dmin > 0.03)
        if far.sum() >= 6 and far.sum() * area_per_pt >= 1.5:
            centres.append(np.median(xy[far], 0))
    if len(centres) <= 1:
        return [(None, np.ones(len(z), dtype=bool))]
    D = np.stack([np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) for c in centres], 0)
    lab = np.argmin(D, 0)
    return [(centres[k], lab == k) for k in range(len(centres))]


def _detect(frame, cam):
    """Coloured objects seen by one camera: list of dicts with world features."""
    rgb = getattr(frame, cam + "_rgb")
    depth = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K"), dtype=np.float64)
    E = getattr(frame, cam + "_extrinsic")
    out = []
    fx = K[0, 0]
    for col, m in _masks(rgb).items():
        for pts in _components(m):
            if len(pts) < MIN_PIX:
                continue
            Xall, dall = _backproject(pts, depth, K, E)
            if len(Xall) < MIN_PIX:
                continue
            app = float(np.median((dall / fx) ** 2)) * 1e4
            parts = _split_blob(Xall, app)
            for _c, sel in parts:
                X = Xall[sel]
                d = dall[sel]
                if len(X) < MIN_PIX:
                    continue
                z = X[:, 2]
                zm = float(np.percentile(z, 95))
                top = X[z > zm - 0.006]
                area = float(len(X) * (float(np.median(d)) / fx) ** 2 * 1e4)
                xy = np.median(top[:, :2], 0)
                # reject things that are not on the table (e.g. far background)
                if zm > 0.10 or zm < 0.004:
                    continue
                out.append({"color": col, "cam": cam, "n": int(len(X)), "area": area,
                            "zm": zm, "xy": [float(xy[0]), float(xy[1])], "split": len(parts)})
    return out


def _fuse(front, wrist):
    """Merge per-camera detections of the same object (same colour, < 3 cm apart)."""
    objs = []
    for o in front:
        objs.append({"color": o["color"], "xy": list(o["xy"]), "zm": o["zm"],
                     "score": o["area"] / FRONT_CUBE_AREA, "views": ["front"], "nf": o["n"], "nw": 0})
    for o in wrist:
        best, bd = None, 0.03
        for q in objs:
            if q["color"] != o["color"] or "wrist" in q["views"]:
                continue
            dd = math.hypot(q["xy"][0] - o["xy"][0], q["xy"][1] - o["xy"][1])
            if dd < bd:
                best, bd = q, dd
        s = o["area"] / WRIST_CUBE_AREA
        if best is None:
            objs.append({"color": o["color"], "xy": list(o["xy"]), "zm": o["zm"], "score": s,
                         "views": ["wrist"], "nf": 0, "nw": o["n"]})
        else:
            best["xy"] = [(best["xy"][0] + o["xy"][0]) / 2.0, (best["xy"][1] + o["xy"][1]) / 2.0]
            best["zm"] = max(best["zm"], o["zm"])
            best["score"] = max(best["score"], s)   # occlusion only shrinks area
            best["views"].append("wrist")
            best["nw"] = o["n"]
    return objs


def _label_shapes(objs):
    """Assign shapes; each colour normally has exactly one T, one sphere and one cube."""
    for o in objs:
        o["shape"] = "T" if o["zm"] < T_HEIGHT_MAX else ("cube" if o["score"] > 1.0 else "sphere")
        o["how"] = "threshold"
    for col in COLORS:
        group = [o for o in objs if o["color"] == col]
        if len(group) != 3:
            continue
        group.sort(key=lambda o: o["zm"])
        if group[0]["zm"] >= T_HEIGHT_MAX or group[1]["zm"] < T_HEIGHT_MAX:
            continue
        rest = sorted(group[1:], key=lambda o: o["score"])
        group[0]["shape"], rest[0]["shape"], rest[1]["shape"] = "T", "sphere", "cube"
        for o in group:
            o["how"] = "triplet"
    return objs


def _cue_shape(o, cube_thr):
    if o["zm"] < T_HEIGHT_MAX:
        return "T"
    return "cube" if o["area"] > cube_thr else "sphere"


def _parse_goal(goal):
    """Optional explicit colour/shape words in the instruction (normally absent: the cue is visual)."""
    g = (goal or "").lower()
    col = None
    for c in COLORS:
        if c in g:
            col = c
    shp = None
    for word, s in (("cube", "cube"), ("sphere", "sphere"), ("ball", "sphere"), ("t-shape", "T"), ("t shape", "T")):
        if word in g:
            shp = s
    return col, shp


def perceive_decide(obs, goal, memory, last):
    """Track cue -> delay -> choice phases; record the cue; then touch the matching object."""
    mem = dict(memory)
    defaults = {"phase": "wait_cue", "cue_color": "", "cue_shape": "", "cue_step": -1,
                "cue_votes": "", "target": [], "target_info": "", "objects_step": -1,
                "decisions": 0, "touch_attempts": 0, "log": ""}
    for k, v in defaults.items():
        if mem.get(k) is None:
            mem[k] = v
    mem["decisions"] = int(mem["decisions"]) + 1
    if last:
        print("[policy] step %s last: %s" % (obs.current.step, str(last)[:1500]))
    cur = obs.current
    # 1. cue: examine every new frame (the cue can vanish quickly)
    votes = mem["cue_votes"]
    for fr in obs.frames:
        if mem["phase"] not in ("wait_cue", "cue_seen"):
            break
        front = _detect(fr, "front")
        if len(front) == 1:
            o = front[0]
            s = _cue_shape(o, FRONT_CUBE_AREA)
            w = _detect(fr, "wrist")
            ws = _cue_shape(w[0], WRIST_CUBE_AREA) if len(w) == 1 else "?"
            votes = (votes + ";" if votes else "") + "%d:%s/%s/%s(a%.1f,z%.3f)" % (
                fr.step, o["color"], s, ws, o["area"], o["zm"])
            if mem["phase"] == "wait_cue":
                mem["cue_color"], mem["cue_shape"], mem["cue_step"] = o["color"], s, int(fr.step)
                mem["phase"] = "cue_seen"
    mem["cue_votes"] = votes[-600:]
    # 2. choice phase: several objects visible
    front = _detect(cur, "front")
    wrist = _detect(cur, "wrist")
    n_vis = max(len(front), len(wrist))
    if mem["phase"] in ("wait_cue", "cue_seen") and n_vis >= 5:
        gc, gs = _parse_goal(goal)
        col = mem["cue_color"] or gc or "red"
        shp = mem["cue_shape"] or gs or "cube"
        objs = _label_shapes(_fuse(front, wrist))
        cands = [o for o in objs if o["color"] == col and o["shape"] == shp]
        if not cands:
            same = [o for o in objs if o["color"] == col] or objs
            if shp == "T":
                cands = sorted(same, key=lambda o: o["zm"])
            else:
                tall = [o for o in same if o["zm"] >= T_HEIGHT_MAX] or same
                cands = sorted(tall, key=lambda o: -o["score"] if shp == "cube" else abs(o["score"] - 0.8))
        if len(cands) > 1:
            if shp == "cube":
                cands.sort(key=lambda o: -o["score"])
            elif shp == "T":
                cands.sort(key=lambda o: o["zm"])
            else:
                cands.sort(key=lambda o: abs(o["score"] - 0.8))
        tg = cands[0]
        mem["target"] = [tg["xy"][0], tg["xy"][1], tg["zm"]]
        mem["target_info"] = "%s %s %s views=%s score=%.2f zm=%.3f nobj=%d" % (
            tg["color"], tg["shape"], tg["how"], "+".join(tg["views"]), tg["score"], tg["zm"], len(objs))
        mem["objects_step"] = int(cur.step)
        mem["phase"] = "touch"
        mem["log"] = (mem["log"] + " | objects@%d target %s" % (cur.step, mem["target_info"]))[-800:]
    if mem["phase"] == "touch":
        mem["touch_attempts"] = int(mem["touch_attempts"]) + 1
        x, y, zt = mem["target"]
        k = mem["touch_attempts"]
        # open gripper so the object fits between the fingers; bring the TCP down to (slightly
        # below) the object's centre height. Online: success came at TCP z 0.027 (cube) and
        # 0.020 (sphere), never with closed fingertips resting on top (R001).
        # Online evidence: success = arm at rest close to the object (R016: static 3.5 cm above a
        # sphere centre succeeded). Primary: closed gripper hovering just above the object top,
        # so no finger can come down on a neighbouring object. Fallback: open gripper, centre height.
        if k == 1:
            args = {"x": x, "y": y, "z_top": zt, "z_final": float(zt + HOVER_ABOVE_TOP),
                    "gripper": -1.0, "hold": 3}
        else:
            # the object may have been displaced: re-localise it near the old estimate
            objs = _label_shapes(_fuse(front, wrist))
            same = [o for o in objs if o["color"] == mem["cue_color"]
                    and ((o["zm"] < T_HEIGHT_MAX) == (mem["cue_shape"] == "T"))
                    and math.hypot(o["xy"][0] - x, o["xy"][1] - y) < 0.10]
            if same:
                match = [o for o in same if o["shape"] == mem["cue_shape"]] or same
                nb = min(match, key=lambda o: math.hypot(o["xy"][0] - x, o["xy"][1] - y))
                if math.hypot(nb["xy"][0] - x, nb["xy"][1] - y) > 0.01:
                    x, y = nb["xy"][0], nb["xy"][1]
                    mem["target"] = [x, y, zt]
                    mem["log"] = (mem["log"] + " | relocalised %.3f,%.3f" % (x, y))[-800:]
            zf = max(0.012, 0.5 * zt)
            if k >= 3:
                zf = max(0.009, zf - 0.005)
            args = {"x": x, "y": y, "z_top": zt, "z_final": float(zf), "gripper": 1.0, "hold": 4}
        mem["log"] = (mem["log"] + " | touch#%d@%d" % (k, cur.step))[-800:]
        return {"memory": mem, "act": {"tool": "act_touch", "args": args}}
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 1}}}


def act_wait(robot, steps=1):
    """Hold still for `steps` control steps (commands are suppressed during the cue/delay phase)."""
    n = 0
    for _ in range(int(steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
        n += 1
    f = robot.now()
    return {"waited": n, "step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp[:3]]}


def _rot(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    """Rotation vector (world frame) of rotation matrix R."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * w
    return w * (ang / (2.0 * math.sin(ang)))


def act_touch(robot, x, y, z_top, z_final, gripper=1.0, hold=6, max_steps=22,
              alpha=0.7, xy_tol=0.012, rot_gain=0.5, hold_tol=0.006, xy_gain_near=0.4, hold_xy_tol=0.010):
    """Bring the TCP over (x, y) and down to z_final (open gripper straddles the object), then hold.

    Descent follows a cone: while the horizontal error exy > xy_tol the TCP stays above
    z_top + 0.015 + 1.5*exy (capped at z_top + 0.06), so the fingers never come down beside the
    object prematurely. Lag-compensated P control on measured TCP (v' = 0.29 cmd + 0.375 v);
    orientation is held at the start orientation with a small rotation correction during the
    approach (step convention: offset = -0.1*r rad). Once within hold_tol of the final point the
    arm gets zero offsets so it comes to rest (online evidence: success registers only when the
    arm is nearly static). Returns the measured trajectory summary."""
    traj = []
    f0 = robot.now()
    t0 = np.asarray(f0.tcp, dtype=np.float64)
    R0 = _rot(t0[3], t0[4], t0[5])
    prev = t0[:3].copy()
    v = np.zeros(3)
    held = 0
    min_margin = 9.0
    stage = "approach"
    for k in range(int(max_steps)):
        f = robot.now()
        tc = np.asarray(f.tcp, dtype=np.float64)
        p = tc[:3]
        if k > 0:
            v = p - prev
        prev = p.copy()
        exy = math.hypot(x - p[0], y - p[1])
        if stage == "approach" and exy < xy_tol:
            stage = "descend"
        if stage == "approach":
            tz = max(z_final, z_top + min(0.06, 0.015 + 1.5 * exy))
        else:
            tz = z_final
        tgt = np.array([x, y, tz])
        err = tgt - p
        if stage == "descend" and exy < hold_xy_tol and abs(err[2]) < hold_tol:
            stage = "hold"
        if stage == "hold" and (exy > 2.0 * hold_xy_tol or abs(err[2]) > 2.0 * hold_tol):
            stage = "descend"
        if stage == "hold":
            # zero offset = joint target at the measured pose: lets the arm come to rest
            cmd = np.zeros(3)
            rcmd = np.zeros(3)
            held += 1
        else:
            c = (alpha * err - B_MOM * v) / A_GAIN
            if stage == "descend":
                # lateral: plain P with lower gain (momentum term caused +-1-2 cm x limit cycles)
                c[0] = xy_gain_near * err[0] / A_GAIN
                c[1] = xy_gain_near * err[1] / A_GAIN
            cmd = np.clip(c / 0.1, -1.0, 1.0)
            if stage == "descend" and err[2] < -0.003 and cmd[2] > -0.15:
                cmd[2] = -0.15
            w = _rotvec(R0 @ _rot(tc[3], tc[4], tc[5]).T)
            rcmd = np.clip(-rot_gain * w / 0.1, -0.5, 0.5) if stage == "approach" else np.zeros(3)
            chk = robot.dry_run(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                                float(rcmd[0]), float(rcmd[1]), float(rcmd[2]), mode="delta")
            mm = chk.get("min_joint_margin_rad")
            if mm is not None:
                min_margin = min(min_margin, float(mm))
            if not chk.get("within_joint_limits", True):
                cmd = cmd * 0.3
                rcmd = rcmd * 0.0
        traj.append([int(f.step), stage[0]] + [round(float(q), 3) for q in p] +
                    [round(float(q), 2) for q in cmd])
        if held > hold:
            break
        robot.step(float(cmd[0]), float(cmd[1]), float(cmd[2]),
                   float(rcmd[0]), float(rcmd[1]), float(rcmd[2]), float(gripper))
    f = robot.now()
    return {"stage": stage, "held": held, "final_tcp": [round(float(q), 4) for q in f.tcp],
            "fingers": [round(float(q), 4) for q in f.fingers], "min_joint_margin": round(min_margin, 3),
            "traj": traj[-14:]}
