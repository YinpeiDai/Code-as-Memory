"""Perception helpers for rearrange_blocks (head RGB-D, fixed calibration read from each frame)."""

TABLE_Z_DEFAULT = 0.741


def _head_cloud(frame):
    """World points (N,3) and RGB (N,3 int) of valid head-depth pixels."""
    depth = np.asarray(frame.head_depth).astype(np.float64)
    K = np.asarray(frame.head_K, dtype=np.float64)
    E = np.asarray(frame.head_extrinsic, dtype=np.float64)
    rgb = np.asarray(frame.head_rgb).astype(np.int32)
    v, u = np.nonzero(depth > 0)
    z = depth[v, u] / 1000.0
    xc = (u - K[0, 2]) * z / K[0, 0]
    yc = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([xc, yc, z], 1)
    pw = (pc - E[:, 3]) @ E[:, :3]
    return pw, rgb[v, u]


def _clusters(pts, cell=0.01, min_pts=30):
    """Connected components of points on an xy grid (8-neighbour). Returns list of index arrays."""
    if len(pts) == 0:
        return []
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    cells = {}
    for k in range(len(ij)):
        key = (int(ij[k, 0]), int(ij[k, 1]))
        if key in cells:
            cells[key].append(k)
        else:
            cells[key] = [k]
    seen = set()
    out = []
    for start in cells:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            c = stack.pop()
            members.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(members) >= min_pts:
            out.append(np.array(members))
    return out


def _scene(frame):
    """Detect table height, blue mats and red objects (blocks + button) in world frame."""
    pw, rgb = _head_cloud(frame)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    work = (pw[:, 2] > 0.70) & (pw[:, 2] < 0.86) & (pw[:, 1] > -0.45) & (pw[:, 1] < 0.25) & (np.abs(pw[:, 0]) < 0.5)
    near_table = work & (pw[:, 2] < 0.75)
    table_z = float(np.median(pw[near_table, 2])) if near_table.sum() > 100 else TABLE_Z_DEFAULT
    if abs(table_z - TABLE_Z_DEFAULT) > 0.03:
        table_z = TABLE_Z_DEFAULT
    red = work & (r > 120) & (r - g > 60) & (r - b > 60)
    blue = work & (b > 140) & (b - r > 60) & (pw[:, 2] < table_z + 0.012)
    mats = []
    P = pw[blue]
    for idx in _clusters(P, 0.01, 40):
        q = P[idx]
        lo = np.percentile(q[:, :2], 2, axis=0)
        hi = np.percentile(q[:, :2], 98, axis=0)
        c = (lo + hi) / 2.0
        mats.append({"xy": [float(c[0]), float(c[1])], "size": [float(hi[0] - lo[0]), float(hi[1] - lo[1])], "n": int(len(idx))})
    reds = []
    P = pw[red]
    for idx in _clusters(P, 0.01, 30):
        q = P[idx]
        top = float(np.percentile(q[:, 2], 95))
        topm = q[:, 2] > top - 0.006
        qt = q[topm] if topm.sum() >= 10 else q
        lo = np.percentile(qt[:, :2], 3, axis=0)
        hi = np.percentile(qt[:, :2], 97, axis=0)
        c = (lo + hi) / 2.0
        reds.append({"xy": [float(c[0]), float(c[1])], "top": top, "size": [float(hi[0] - lo[0]), float(hi[1] - lo[1])], "n": int(len(idx))})
    mats.sort(key=lambda m: -m["n"])
    reds.sort(key=lambda m: -m["n"])
    return {"table_z": table_z, "mats": mats, "reds": reds}


def _dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _interpret(scene):
    """Assign roles: two mats, block between mats, block on a mat, empty mat, button, midpoint."""
    mats = scene["mats"][:2]
    if len(mats) < 2:
        return {"ok": False, "why": "found %d mats" % len(scene["mats"])}
    mats = sorted(mats, key=lambda m: m["xy"][0])
    m0, m1 = mats[0]["xy"], mats[1]["xy"]
    mid = [(m0[0] + m1[0]) / 2.0, (m0[1] + m1[1]) / 2.0]
    span = _dist(m0, m1)
    reds = scene["reds"]
    # blocks: red clusters near the mat row (within the segment between mats, extended a little)
    ax = np.array([m1[0] - m0[0], m1[1] - m0[1]]) / max(span, 1e-6)
    cand = []
    for rd in reds:
        d = np.array([rd["xy"][0] - m0[0], rd["xy"][1] - m0[1]])
        s = float(d @ ax)
        perp = float(abs(d[0] * ax[1] - d[1] * ax[0]))
        cand.append((perp + max(0.0, -s) + max(0.0, s - span), rd))
    cand.sort(key=lambda c: c[0])
    blocks = [c[1] for c in cand if c[0] < 0.06][:2]
    others = [c[1] for c in cand if all(c[1] is not b for b in blocks)]
    if len(blocks) < 2:
        return {"ok": False, "why": "found %d blocks near mats" % len(blocks)}
    button = None
    if others:
        others.sort(key=lambda r: -r["n"])
        button = others[0]["xy"]
    dm = [[_dist(bk["xy"], m) for m in (m0, m1)] for bk in blocks]
    on = [min(d) for d in dm]
    i_on = 0 if on[0] < on[1] else 1
    i_mid = 1 - i_on
    occupied = 0 if dm[i_on][0] < dm[i_on][1] else 1
    empty = m1 if occupied == 0 else m0
    return {
        "ok": True,
        "block_mid": blocks[i_mid]["xy"],
        "block_mat": blocks[i_on]["xy"],
        "block_mid_top": blocks[i_mid]["top"],
        "block_mat_top": blocks[i_on]["top"],
        "empty_mat": empty,
        "occupied_mat": m0 if occupied == 0 else m1,
        "mid": mid,
        "button": button,
        "table_z": scene["table_z"],
        "on_dist": on[i_on],
        "mid_dist": on[i_mid],
    }
