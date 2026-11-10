"""Peg perception helpers (blue 0.24 x 0.05 x 0.05 m bar lying on the table).

Both RGB-D cameras are back-projected into the world frame; blue pixels whose
world height lies near the peg top surface (z ~ 0.05 m) are voxelised in xy
(4 mm cells, equalising front/wrist density) and a principal-axis fit gives the
peg axis yaw (mod pi), centre and visible extents.
"""

PEG_LEN = 0.24
PEG_WID = 0.05
PEG_TOP = 0.05


def _blue_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (b > 100) & (b > r + 50) & (b > g + 30)


def _cam_points(rgb, dep, K, E, step=1):
    m = _blue_mask(rgb) & (dep > 0)
    v, u = np.nonzero(m)
    if len(v) == 0:
        return np.zeros((0, 3))
    d = dep[v, u].astype(np.float64) / 1000.0
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * d
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    return (pc - t) @ R


def _peg_top_points(frame, use_wrist=True):
    cams = ["front", "wrist"] if use_wrist else ["front"]
    out = []
    for cam in cams:
        P = _cam_points(getattr(frame, cam + "_rgb"), getattr(frame, cam + "_depth"),
                        getattr(frame, cam + "_K"), getattr(frame, cam + "_extrinsic"))
        if len(P):
            P = P[(P[:, 2] > PEG_TOP - 0.012) & (P[:, 2] < PEG_TOP + 0.012)]
            out.append(P)
    if not out:
        return np.zeros((0, 3))
    return np.concatenate(out, 0)


def _voxel_xy(P, cell=0.004):
    if len(P) == 0:
        return np.zeros((0, 2))
    k = np.floor(P[:, :2] / cell).astype(np.int64)
    key = (k[:, 0] + 100000) * 200003 + (k[:, 1] + 100000)
    _, idx = np.unique(key, return_index=True)
    return (k[idx].astype(np.float64) + 0.5) * cell


def _min_width_yaw(Q, yaw0, span=0.35, step=0.005):
    """Yaw (near yaw0) minimising the robust perpendicular extent of cells Q."""
    best = None
    D = Q - Q.mean(0)
    k = int(span / step)
    for i in range(-k, k + 1):
        y = yaw0 + i * step
        n = np.array([-math.sin(y), math.cos(y)])
        pn = D @ n
        w = float(np.percentile(pn, 99.0) - np.percentile(pn, 1.0))
        if best is None or w < best[0] - 1e-9:
            best = (w, y)
    # refine
    w0, y0 = best
    for i in range(-10, 11):
        y = y0 + i * step / 10.0
        n = np.array([-math.sin(y), math.cos(y)])
        pn = D @ n
        w = float(np.percentile(pn, 99.0) - np.percentile(pn, 1.0))
        if w < best[0] - 1e-9:
            best = (w, y)
    return best[1], best[0]


def _peg_estimate(frame, prev_yaw=None, use_wrist=True):
    """Return dict(ok, cx, cy, yaw, ext_a, ext_n, cells, ...) or ok=False.

    yaw is the peg axis angle (mod pi), unwrapped to be near prev_yaw if given.
    """
    P = _peg_top_points(frame, use_wrist)
    Q = _voxel_xy(P)
    if len(Q) < 30:
        return {"ok": False, "cells": int(len(Q))}
    m = Q.mean(0)
    D = Q - m
    C = D.T @ D / len(D)
    w, V = np.linalg.eigh(C)
    a = V[:, 1]
    yaw_pca = math.atan2(a[1], a[0])
    ref = yaw_pca if prev_yaw is None else prev_yaw + _wrap_half(yaw_pca - prev_yaw)
    yaw, wid = _min_width_yaw(Q, ref)
    if prev_yaw is not None:
        yaw = prev_yaw + _wrap_half(yaw - prev_yaw)
    a = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-a[1], a[0]])
    pa = D @ a
    pn = D @ n
    lo = float(np.percentile(pa, 0.5))
    hi = float(np.percentile(pa, 99.5))
    nlo = float(np.percentile(pn, 1.0))
    nhi = float(np.percentile(pn, 99.0))
    c = m + a * (0.5 * (lo + hi)) + n * (0.5 * (nlo + nhi))
    return {"ok": True, "cx": float(c[0]), "cy": float(c[1]), "yaw": float(yaw),
            "yaw_pca": float(ref), "ext_a": float(hi - lo), "ext_n": float(nhi - nlo),
            "lo": lo, "hi": hi, "mx": float(m[0]), "my": float(m[1]),
            "cells": int(len(Q))}


def _wrap_half(x):
    """Wrap an axis angle difference to [-pi/2, pi/2)."""
    return (x + math.pi / 2) % math.pi - math.pi / 2


def _wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _parse_goal_deg(goal):
    """Parse 'Rotate the peg by -14 degrees ...' -> -14.0 (None if absent)."""
    s = str(goal).lower()
    i = s.find(" by ")
    cands = []
    if i >= 0:
        cands.append(s[i + 4:])
    cands.append(s)
    for seg in cands:
        num = ""
        started = False
        for ch in seg:
            if ch in "+-" and not started:
                num = ch
                started = True
            elif ch.isdigit() or (ch == "." and started):
                num += ch
                started = True
            elif started:
                if num not in ("+", "-", ""):
                    break
                num = ""
                started = False
        if num not in ("", "+", "-"):
            val = float(num)
            rest = seg[seg.find(num) + len(num):]
            if "rad" in rest[:8] and "deg" not in rest[:8]:
                val = val * 180.0 / math.pi
            if "clockwise" in s and "counter" not in s and "anti" not in s:
                val = -abs(val)
            return val
    return None
