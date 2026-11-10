"""battery_try policy: place both batteries in the fixed slot tray, then try orientation
combinations with a cyclic Gray code (flip left, flip right, flip left, ...) until the
simulator reports success (the episode ends at once) or the dashboard needle is seen rotated."""

# ---------------------------------------------------------------- constants (from demos)
# Default tray centroid (world, m); re-measured from the head camera at the first decision.
TRAY_DEFAULT = [-0.0007, -0.1024]
# Slot targets for the ee frame, relative to the tray centroid (expert means over 25 demos).
SLOT_REL = {
    "left": {"place": [-0.0198, 0.0014], "regrasp": [-0.0248, 0.0054], "release": [-0.0183, 0.0024]},
    "right": {"place": [0.0227, 0.0044], "regrasp": [0.0277, 0.0094], "release": [0.0187, 0.0004]},
}
# ee xy = visible-blue-centroid xy + offset (perspective bias of the head camera; demo fit, <1 mm residual)
GRASP_OFFSET = {"left": [-0.0015, 0.0035], "right": [0.0080, 0.0037]}
# Top-down grasp quaternions (w, x, y, z) used by the expert; right = Rz(pi) * left.
Q_DOWN = {"left": [0.70710678, 0.0, 0.70710678, 0.0], "right": [0.0, -0.70710678, 0.0, 0.70710678]}
Z_TABLE_GRASP = 0.911
Z_SLOT_PLACE = 0.913
Z_SLOT_GRASP = 0.918
Z_PRE = 1.0
Z_LIFT = 1.01
Z_FLIP_LIFT = 0.97
Z_FLIP_ROT = 0.99
FLIP_ORDER = ["left", "right"]  # cyclic: L, R, L, R ... visits all 4 combinations in 3 flips
NEEDLE_PIVOT = [-0.003, 0.054, 0.804]


# ---------------------------------------------------------------- perception helpers
def _cam_points(frame, cam, mask):
    """World xyz (N x 3) of masked pixels with valid depth for camera `cam`."""
    dep = getattr(frame, cam + "_depth")
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    v, u = np.nonzero(mask)
    z = dep[v, u].astype(float) / 1000.0
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - E[:, 3]) @ E[:, :3]


def _blue_mask(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (b > 150) & (b - r > 60) & (b - g > 20)


def _blue_points(frame):
    P = _cam_points(frame, "head", _blue_mask(frame.head_rgb))
    return P[(P[:, 2] > 0.74) & (P[:, 2] < 0.83)]


def _tray_center(frame):
    rgb = frame.head_rgb
    dark = rgb.max(axis=-1) < 110
    P = _cam_points(frame, "head", dark)
    sel = (np.abs(P[:, 0]) < 0.1) & (P[:, 1] < -0.04) & (P[:, 1] > -0.2) & (P[:, 2] > 0.74) & (P[:, 2] < 0.77)
    Q = P[sel]
    if len(Q) < 200:
        return list(TRAY_DEFAULT), "default"
    c = [float(Q[:, 0].mean()), float(Q[:, 1].mean())]
    if abs(c[0] - TRAY_DEFAULT[0]) > 0.03 or abs(c[1] - TRAY_DEFAULT[1]) > 0.03:
        return list(TRAY_DEFAULT), "default(outlier)"
    return c, "measured"


def _scene(frame, tray):
    """Batteries lying on the table (outside the tray) per side, and tray-half occupancy."""
    P = _blue_points(frame)
    tx, ty = tray
    in_tray = (np.abs(P[:, 0] - tx) < 0.06) & (np.abs(P[:, 1] - ty) < 0.05)
    table = P[(~in_tray) & (P[:, 2] < 0.80)]
    out = {"slot_left": int((in_tray & (P[:, 0] < tx)).sum()), "slot_right": int((in_tray & (P[:, 0] >= tx)).sum())}
    for side, sel in (("left", table[:, 0] < tx), ("right", table[:, 0] >= tx)):
        Q = table[sel]
        if len(Q) >= 60:
            # keep the dominant cluster around the median
            med = np.median(Q[:, :2], axis=0)
            d = np.sqrt(((Q[:, :2] - med) ** 2).sum(1))
            Q = Q[d < 0.05]
        if len(Q) >= 60:
            c = Q[:, :2].mean(0)
            off = GRASP_OFFSET[side]
            out[side] = [float(c[0] + off[0]), float(c[1] + off[1])]
            out[side + "_n"] = int(len(Q))
        else:
            out[side] = None
            out[side + "_n"] = int(len(Q))
    return out


def _needle(frame):
    """'up' (initial), 'rotated' (correct combination) or 'unknown' (occluded)."""
    rgb = frame.head_rgb
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    red = (r > 140) & (r - g > 60) & (r - b > 60)
    P = _cam_points(frame, "head", red)
    px, py, pz = NEEDLE_PIVOT
    sel = (np.abs(P[:, 0] - px) < 0.06) & (P[:, 1] > py - 0.05) & (P[:, 1] < py + 0.06) & (np.abs(P[:, 2] - pz) < 0.015)
    Q = P[sel]
    if len(Q) < 5:
        return "unknown", int(len(Q))
    dx = float(Q[:, 0].mean() - px)
    dy = float(Q[:, 1].mean() - py)
    if dy > 0.008 and abs(dx) < 0.008:
        return "up", int(len(Q))
    if abs(dx) > 0.008 or dy < 0.004:
        return "rotated", int(len(Q))
    return "unknown", int(len(Q))


# ---------------------------------------------------------------- control helpers
def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _flipped(q):
    """q rotated by 180 degrees about world z."""
    return _qmul([0.0, 0.0, 0.0, 1.0], q)


def _pose(xyz, q):
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])] + [float(v) for v in q]


def _cur(robot, arm):
    f = robot.now()
    return [float(v) for v in getattr(f, arm + "_ee_pose")], float(getattr(f, arm + "_gripper"))


def _step(robot, log, left=None, right=None, tries=2):
    """One native action. left/right = (pose7, gripper) or None to hold. Returns True if no MotionError."""
    for k in range(tries):
        lp, lg = left if left is not None else _cur(robot, "left")
        rp, rg = right if right is not None else _cur(robot, "right")
        try:
            robot.step(*(list(lp) + [lg] + list(rp) + [rg]))
            return True
        except Exception as e:  # MotionError: that arm stayed still; retry once
            log.append("motion_error(try %d): %s" % (k, str(e)[:120]))
    return False


def _arm_seq(robot, arm, seq, log):
    """Run a list of (pose7, gripper) targets for `arm` while the other arm holds."""
    ok = True
    for pose, g in seq:
        if arm == "left":
            ok = _step(robot, log, left=(pose, g)) and ok
        else:
            ok = _step(robot, log, right=(pose, g)) and ok
    return ok


def _report(robot, log, ok):
    f = robot.now()
    return {"ok": bool(ok), "left_ee": [round(float(v), 4) for v in f.left_ee_pose],
            "right_ee": [round(float(v), 4) for v in f.right_ee_pose],
            "left_gripper": round(float(f.left_gripper), 3), "right_gripper": round(float(f.right_gripper), 3),
            "log": log[-6:]}


# ---------------------------------------------------------------- act tools
def act_pick(robot, left_xy=None, right_xy=None):
    """Top-down grasp of table batteries at ee xy targets (None = that arm holds), then lift to Z_LIFT.
    Both arms move together when both targets are given."""
    log = []
    ok = True
    for z, g in ((Z_PRE, 1.0), (Z_TABLE_GRASP, 1.0), (Z_TABLE_GRASP, 0.0), (Z_LIFT, 0.0)):
        L = (_pose([left_xy[0], left_xy[1], z], Q_DOWN["left"]), g) if left_xy is not None else None
        R = (_pose([right_xy[0], right_xy[1], z], Q_DOWN["right"]), g) if right_xy is not None else None
        ok = _step(robot, log, left=L, right=R) and ok
    return _report(robot, log, ok)


def act_place(robot, arm, place_xy, home):
    """Carry the held battery of `arm` above its tray half, lower, release, lift and return to `home` (7-pose)."""
    log = []
    q = Q_DOWN[arm]
    x, y = place_xy
    seq = [(_pose([x, y, Z_LIFT], q), 0.0), (_pose([x, y, Z_SLOT_PLACE], q), 0.0),
           (_pose([x, y, Z_SLOT_PLACE], q), 1.0), (_pose([x, y, Z_LIFT], q), 1.0), (list(home), 1.0)]
    ok = _arm_seq(robot, arm, seq, log)
    return _report(robot, log, ok)


def act_flip(robot, arm, grasp_xy, release_xy, home):
    """Re-grasp `arm`'s battery in the tray, lift, rotate it 180 deg about vertical, put it back, go home."""
    log = []
    q0 = Q_DOWN[arm]
    q1 = _flipped(q0)
    gx, gy = grasp_xy
    rx, ry = release_xy
    seq = [(_pose([gx, gy, Z_PRE], q0), 1.0), (_pose([gx, gy, Z_SLOT_GRASP], q0), 1.0),
           (_pose([gx, gy, Z_SLOT_GRASP], q0), 0.0), (_pose([gx, gy, Z_FLIP_LIFT], q0), 0.0)]
    ok = _arm_seq(robot, arm, seq, log)
    # rotate 180 deg about z while held; fall back to two 90 deg steps if the direct target fails
    if not _arm_seq(robot, arm, [(_pose([rx, ry, Z_FLIP_ROT], q1), 0.0)], log):
        qh = _qmul([0.70710678, 0.0, 0.0, 0.70710678], q0)
        ok = _arm_seq(robot, arm, [(_pose([rx, ry, Z_FLIP_ROT], qh), 0.0), (_pose([rx, ry, Z_FLIP_ROT], q1), 0.0)], log)
    seq = [(_pose([rx, ry, Z_SLOT_PLACE], q1), 0.0), (_pose([rx, ry, Z_SLOT_PLACE], q1), 1.0),
           (_pose([rx, ry, Z_LIFT], q1), 1.0), (list(home), 1.0)]
    ok = _arm_seq(robot, arm, seq, log) and ok
    return _report(robot, log, ok)


def act_home(robot, left_home, right_home, open_grippers=True):
    """Move both arms to their home poses (grippers open if open_grippers)."""
    log = []
    g = 1.0 if open_grippers else None
    lg = g if g is not None else _cur(robot, "left")[1]
    rg = g if g is not None else _cur(robot, "right")[1]
    ok = _step(robot, log, left=(list(left_home), lg), right=(list(right_home), rg))
    return _report(robot, log, ok)


def act_hold(robot, steps=1):
    """Hold both arms in place for `steps` native actions (used after the needle shows success)."""
    log = []
    ok = True
    for _ in range(int(steps)):
        ok = _step(robot, log) and ok
    return _report(robot, log, ok)


# ---------------------------------------------------------------- decision
def perceive_decide(obs, goal, memory, last):
    """State machine: pick both -> place left -> place right -> Gray-code flips until success."""
    m = dict(memory)
    f = obs.current
    if m.get("stage") is None:
        m["stage"] = "pick"
        m["home"] = [[float(v) for v in f.left_ee_pose], [float(v) for v in f.right_ee_pose]]
        tray, src = _tray_center(f)
        m["tray"] = tray
        m["placed"] = []
        m["flips_done"] = 0
        m["pick_attempts"] = 0
        m["history"] = "tray %s (%s)" % ([round(v, 4) for v in tray], src)
    home_l, home_r = m["home"]
    tray = m["tray"]
    needle, nn = _needle(f)
    m["needle"] = needle
    sc = _scene(f, tray)
    m["decisions"] = int(m.get("decisions") or 0) + 1

    def ret(tool, args, note):
        m["last_action"] = tool + ":" + note
        m["history"] = (str(m.get("history") or "") + " | " + note)[-1500:]
        return {"memory": {k: m.get(k) for k in FIELDS}, "act": {"tool": tool, "args": args}}

    def slot(arm, key):
        r = SLOT_REL[arm][key]
        return [tray[0] + r[0], tray[1] + r[1]]

    if needle == "rotated":
        m["stage"] = "done"
        return ret("act_hold", {"steps": 2}, "needle rotated -> hold")

    if m["stage"] == "pick":
        m["pick_attempts"] += 1
        lt = sc["left"] if "left" not in m["placed"] else None
        rt = sc["right"] if "right" not in m["placed"] else None
        m["stage"] = "place"
        m["holding"] = [a for a, t in (("left", lt), ("right", rt)) if t is not None]
        if lt is None and rt is None:
            m["stage"] = "flip"
            return ret("act_home", {"left_home": home_l, "right_home": home_r}, "no table battery seen -> flip")
        return ret("act_pick", {"left_xy": lt, "right_xy": rt},
                   "pick L=%s R=%s" % (None if lt is None else [round(v, 3) for v in lt],
                                       None if rt is None else [round(v, 3) for v in rt]))

    if m["stage"] == "place":
        holding = list(m.get("holding") or [])
        if holding:
            arm = holding[0]
            m["holding"] = holding[1:]
            m["placed"] = sorted(set(list(m["placed"]) + [arm]))
            return ret("act_place", {"arm": arm, "place_xy": slot(arm, "place"),
                                     "home": home_l if arm == "left" else home_r}, "place " + arm)
        # all carried batteries released: verify nothing left on the table (failed grasp -> retry)
        missing = [a for a in ("left", "right") if sc[a] is not None]
        if missing and m["pick_attempts"] < 3:
            m["pick_attempts"] += 1
            m["placed"] = [a for a in m["placed"] if a not in missing]
            m["holding"] = missing
            lt = sc["left"] if "left" in missing else None
            rt = sc["right"] if "right" in missing else None
            return ret("act_pick", {"left_xy": lt, "right_xy": rt}, "retry pick " + ",".join(missing))
        m["stage"] = "flip"

    if m["stage"] == "flip":
        stray = [a for a in ("left", "right") if sc[a] is not None]
        if stray and m["pick_attempts"] < 5:
            # a battery is lying on the table (dropped): put it back into its tray half first
            m["pick_attempts"] += 1
            m["placed"] = [a for a in m["placed"] if a not in stray]
            m["holding"] = stray
            m["stage"] = "place"
            lt = sc["left"] if "left" in stray else None
            rt = sc["right"] if "right" in stray else None
            return ret("act_pick", {"left_xy": lt, "right_xy": rt}, "stray battery -> re-pick " + ",".join(stray))
        k = int(m["flips_done"])
        arm = FLIP_ORDER[k % 2]
        m["flips_done"] = k + 1
        return ret("act_flip", {"arm": arm, "grasp_xy": slot(arm, "regrasp"), "release_xy": slot(arm, "release"),
                                "home": home_l if arm == "left" else home_r},
                   "flip#%d %s (needle %s, slots %d/%d)" % (k + 1, arm, needle, sc["slot_left"], sc["slot_right"]))

    return ret("act_hold", {"steps": 1}, "idle")


FIELDS = ["stage", "home", "tray", "placed", "holding", "flips_done", "pick_attempts", "needle",
          "last_action", "decisions", "history"]
