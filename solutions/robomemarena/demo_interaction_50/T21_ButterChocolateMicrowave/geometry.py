"""Geometry, perception and control helpers shared by the policy tools (no imports; np and math in scope)."""

DOWN_AA = [3.14159265, 0.0, 0.0]          # gripper pointing down, fingers close along world y
SIDE_AA = [2.415, -2.415, -2.415]         # gripper pointing +y (into microwave), fingers close along world x
TABLE_Z_DEFAULT = 0.431
HANDLE_DEFAULT = [0.096, 0.411, 0.583]
BUTTER_DEFAULT = [0.167, -0.095, 0.443]
CHOC_DEFAULT = [0.171, -0.191, 0.453]


def _skew(k):
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def _aa2mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3)
    K = _skew(v / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def _mat2aa(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-3:
        M = (R + np.eye(3)) / 2.0
        d = np.sqrt(np.clip(np.diag(M), 0.0, None))
        i = int(np.argmax(d))
        k = M[:, i] / max(d[i], 1e-9)
        k = k / max(np.linalg.norm(k), 1e-9)
        return k * th
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return k * th


def _ori_err(target_aa, cur_aa):
    """World-frame axis-angle that rotates the current orientation onto the target."""
    return _mat2aa(_aa2mat(target_aa) @ _aa2mat(cur_aa).T)


def _backproject(depth, K, E):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float) / 1000.0
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    Pc = np.stack([x, y, z], -1)
    R = E[:, :3]
    t = E[:, 3]
    return (Pc - t) @ R


def _components(mask, min_size=1):
    """4-connected components of a boolean image; list of (rows, cols) arrays."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    pts = np.argwhere(mask)
    for p in pts:
        r0, c0 = int(p[0]), int(p[1])
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        rr = []
        cc = []
        while stack:
            r, c = stack.pop()
            rr.append(r)
            cc.append(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = r + dr, c + dc
                if 0 <= a < H and 0 <= b < W and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        if len(rr) >= min_size:
            out.append((np.array(rr), np.array(cc)))
    return out


def _frame_points(frame, cam):
    depth = np.asarray(getattr(frame, cam + "_depth"))
    K = np.asarray(getattr(frame, cam + "_K"))
    E = np.asarray(getattr(frame, cam + "_extrinsic"))
    P = _backproject(depth, K, E)
    return P, depth > 0, np.asarray(getattr(frame, cam + "_rgb")).astype(int)


def _dark_mask(rgb):
    return (rgb.max(-1) < 70) & (np.abs(rgb[..., 0] - rgb[..., 2]) < 12)


def _detect_microwave(frame):
    """Microwave body from agentview dark pixels -> handle grasp estimate (calibrated on 50 demos)."""
    P, valid, rgb = _frame_points(frame, "agentview")
    m = (_dark_mask(rgb) & valid & (P[..., 2] > 0.45) & (P[..., 2] < 0.9)
         & (P[..., 0] > -0.4) & (P[..., 0] < 0.5) & (P[..., 1] > 0.1) & (P[..., 1] < 0.8))
    Q = P[m]
    if len(Q) < 300:
        return None
    front_x = float(np.percentile(Q[:, 0], 99.5))
    y_lo = float(np.percentile(Q[:, 1], 2))
    y_hi = float(np.percentile(Q[:, 1], 98))
    hx = front_x + HANDLE_X_OFFSET
    hy = HANDLE_Y_SLOPE * 0.5 * (y_lo + y_hi) + HANDLE_Y_OFFSET
    return {"handle": [hx, hy, HANDLE_DEFAULT[2]], "front_x": front_x, "y_lo": y_lo,
            "y_hi": y_hi, "n": int(len(Q))}


HANDLE_X_OFFSET = -0.0749
HANDLE_Y_SLOPE = 0.8455
HANDLE_Y_OFFSET = -0.0345


def _table_z(P, valid, region):
    z = P[..., 2][valid & region]
    z = z[(z > 0.38) & (z < 0.48)]
    if len(z) < 50:
        return TABLE_Z_DEFAULT
    return float(np.median(z))


def _detect_objects_agentview(frame, y_max):
    """Elevated blobs on the table (left of the microwave) seen by agentview."""
    P, valid, rgb = _frame_points(frame, "agentview")
    region = (P[..., 0] > -0.15) & (P[..., 0] < 0.35) & (P[..., 1] > -0.45) & (P[..., 1] < y_max)
    tz = _table_z(P, valid, region)
    m = valid & region & (P[..., 2] > tz + 0.005) & (P[..., 2] < tz + 0.08)
    return _blobs(P, rgb, m, tz, 12)


def _blobs(P, rgb, m, tz, min_size):
    out = []
    for rr, cc in _components(m, min_size):
        q = P[rr, cc]
        col = rgb[rr, cc].mean(0)
        lo = q.min(0)
        hi = q.max(0)
        out.append({"center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
                    "mean": [float(v) for v in q.mean(0)],
                    "top": float(np.percentile(q[:, 2], 95)), "n": int(len(rr)),
                    "ext": [float(hi[0] - lo[0]), float(hi[1] - lo[1])],
                    "rgb": [float(v) for v in col], "bright": float(col.mean()),
                    "orange": float(np.mean((rgb[rr, cc, 0] > 150) & (rgb[rr, cc, 2] < 110)))})
    return out, tz


def _classify(blobs):
    """Butter: brighter / more orange box. Chocolate: darker brown box. Returns (butter, choc)."""
    cand = [b for b in blobs if 0.02 < b["ext"][0] < 0.14 and 0.015 < b["ext"][1] < 0.1]
    if len(cand) < 2:
        return (cand[0], None) if (len(cand) == 1 and cand[0]["bright"] + 100 * cand[0]["orange"] > 61) else (None, cand[0] if cand else None)
    cand = sorted(cand, key=lambda b: -b["n"])[:2]
    a, b = cand
    score_a = a["bright"] + 100 * a["orange"]
    score_b = b["bright"] + 100 * b["orange"]
    return (a, b) if score_a > score_b else (b, a)


def _wrist_refine(frame, guess, tz):
    """Top-down wrist view: blob nearest the guess; returns [x, y, top_z] or None."""
    P, valid, rgb = _frame_points(frame, "wrist")
    m = valid & (P[..., 2] > tz + 0.005) & (P[..., 2] < tz + 0.08)
    m = m & (np.abs(P[..., 0] - guess[0]) < 0.09) & (np.abs(P[..., 1] - guess[1]) < 0.08)
    blobs, _ = _blobs(P, rgb, m, tz, 40)
    best = None
    for b in blobs:
        if not (0.03 < b["ext"][0] < 0.13 and 0.02 < b["ext"][1] < 0.08):
            continue
        d = math.hypot(b["center"][0] - guess[0], b["center"][1] - guess[1])
        if d < 0.05 and (best is None or d < best[0]):
            best = (d, b)
    if best is None:
        return None
    b = best[1]
    return [b["center"][0], b["center"][1], b["top"], b["ext"][0], b["ext"][1], b["n"]]


def _door_open_count(frame, handle):
    """Dark agentview points where the opened door swings (in front of / left of the closed door)."""
    P, valid, rgb = _frame_points(frame, "agentview")
    hx, hy = handle[0], handle[1]
    r = (_dark_mask(rgb) & valid & (P[..., 0] > hx - 0.45) & (P[..., 0] < hx - 0.08)
         & (P[..., 1] > hy - 0.3) & (P[..., 1] < hy) & (P[..., 2] > 0.45) & (P[..., 2] < 0.8))
    return int(r.sum())


def _butter_color_count(frame):
    """Wrist pixels with the butter box's saturated red/orange/yellow print."""
    rgb = np.asarray(frame.wrist_rgb).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    m = (r > 150) & (b < 90) & (g > 40) & (r - b > 90)
    return int(m.sum())


def _width(frame):
    f = np.asarray(frame.fingers)
    return float(f[0] - f[1])


def _detect_bar(frame):
    """Wrist view from the pre-grasp pose (gripper pointing +y at the door): the handle bar is the dark structure
    2-7 cm in front of the door plane. Returns bar centre x, bar front y, door plane y (world m) or None."""
    P, valid, rgb = _frame_points(frame, "wrist")
    m = (valid & _dark_mask(rgb) & (P[..., 2] > 0.50) & (P[..., 2] < 0.66)
         & (P[..., 1] > 0.3) & (P[..., 1] < 0.75))
    Q = P[m]
    if len(Q) < 200:
        return None
    door_y = float(np.percentile(Q[:, 1], 60))
    near = Q[(Q[:, 1] < door_y - 0.02) & (Q[:, 1] > door_y - 0.07)]
    if len(near) < 30:
        return None
    x5 = float(np.percentile(near[:, 0], 5))
    x95 = float(np.percentile(near[:, 0], 95))
    if x95 - x5 > 0.04:
        return None
    return {"bar_x": float(np.median(near[:, 0])), "bar_y": float(np.percentile(near[:, 1], 5)),
            "door_y": door_y, "n": int(len(near)), "width": x95 - x5}


BAR_TO_GRASP = [0.035, 0.014]   # grasp TCP minus (bar centre x, bar front y), demo frames 27-30


def _door_points(frame, handle):
    """Opened-door dark points (world minus handle; x, y) seen by agentview, or an empty array."""
    P, valid, rgb = _frame_points(frame, "agentview")
    hx, hy = handle[0], handle[1]
    r = (_dark_mask(rgb) & valid & (P[..., 0] > hx - 0.5) & (P[..., 0] < hx - 0.08)
         & (P[..., 1] > hy - 0.35) & (P[..., 1] < hy) & (P[..., 2] > 0.45) & (P[..., 2] < 0.8))
    Q = P[r]
    return Q[:, :2] - np.array([hx, hy])


def _door_open_x(frame, handle):
    """Median handle-relative x of the opened door for rel y in [-0.2,-0.05]. Fully open (~90 deg) doors gave
    -0.288..-0.259 in 19 successful rollouts; a door left ~70 deg open (R022) gave -0.193 and blocked the insertion."""
    R = _door_points(frame, handle)
    if len(R) == 0:
        return None
    s = (R[:, 1] > -0.2) & (R[:, 1] < -0.05)
    if int(s.sum()) < 20:
        return None
    return float(np.median(R[s, 0]))


def _door_max_x(frame, handle, y_lo, y_hi):
    R = _door_points(frame, handle)
    if len(R) == 0:
        return None
    s = (R[:, 1] > y_lo) & (R[:, 1] < y_hi)
    if int(s.sum()) < 10:
        return None
    return float(np.percentile(R[s, 0], 95))
