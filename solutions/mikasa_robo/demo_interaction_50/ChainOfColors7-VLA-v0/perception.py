"""Perception helpers for ChainOfColors: colour blobs, cue sequence, layout and button (front camera)."""

_LEVEL_HI = 200.0
_LEVEL_MID_LO = 120.0
_LEVEL_ZERO = 70.0
_COLOR_NAMES = {
    (2, 0, 0): "red", (1, 0, 0): "darkred", (0, 2, 0): "green", (0, 0, 2): "blue",
    (2, 2, 0): "yellow", (2, 0, 2): "magenta", (0, 2, 2): "cyan", (0, 1, 1): "teal",
    (2, 1, 0): "orange", (0, 1, 0): "darkgreen", (0, 0, 1): "navy", (1, 1, 0): "olive",
    (1, 0, 1): "purple", (2, 2, 2): "white",
}


def _level(v):
    if v >= _LEVEL_HI:
        return 2
    if v >= _LEVEL_MID_LO:
        return 1
    if v < _LEVEL_ZERO:
        return 0
    return -1


def _color_name(c):
    key = (_level(c[0]), _level(c[1]), _level(c[2]))
    if -1 in key:
        return "unk%d_%d_%d" % (int(c[0]), int(c[1]), int(c[2]))
    return _COLOR_NAMES.get(key, "c%d%d%d" % key)


def _components(mask):
    """4-connected components of a boolean mask; returns list of (ys, xs) arrays."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return []
    idx = {}
    for i in range(n):
        idx[(int(ys[i]), int(xs[i]))] = i
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        members = []
        while stack:
            k = stack.pop()
            members.append(k)
            y, x = int(ys[k]), int(xs[k])
            for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                j = idx.get(q)
                if j is not None and not seen[j]:
                    seen[j] = True
                    stack.append(j)
        m = np.array(members)
        comps.append((ys[m], xs[m]))
    return comps


def _backproject(depth, K, E, ys, xs):
    z = depth[ys, xs].astype(float) / 1000.0
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    if len(z) == 0:
        return np.zeros((0, 3))
    xc = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    yc = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([xc, yc, z], 1)
    R = np.asarray(E[:, :3], dtype=float)
    t = np.asarray(E[:, 3], dtype=float)
    return (pc - t) @ R


def _hue_key(px, mx):
    n = px / np.maximum(mx, 1.0)[:, None]
    key = np.full(n.shape, -1, dtype=int)
    key[n < 0.35] = 0
    key[(n > 0.55) & (n < 0.85)] = 1
    key[n >= 0.85] = 2
    return key[:, 0] * 9 + key[:, 1] * 3 + key[:, 2], (key >= 0).all(1)


def _split2(ys, xs):
    """Split pixel set into two spatial clusters (2-means along the principal axis)."""
    P = np.stack([ys, xs], 1).astype(float)
    c = P.mean(0)
    Q = P - c
    w, V = np.linalg.eigh(Q.T @ Q)
    ax = V[:, -1]
    proj = Q @ ax
    lab = proj > 0
    for _ in range(10):
        if lab.all() or (~lab).all():
            break
        c1, c0 = P[lab].mean(0), P[~lab].mean(0)
        new = ((P - c1) ** 2).sum(1) < ((P - c0) ** 2).sum(1)
        if (new == lab).all():
            break
        lab = new
    return [(ys[lab], xs[lab]), (ys[~lab], xs[~lab])]


def _blobs(frame, min_px=15, with_3d=False, split_px=150):
    """Saturated colour blobs (per hue) in the front image with top-face colour and name."""
    im = np.asarray(frame.front_rgb).astype(float)
    mx = im.max(2)
    mn = im.min(2)
    mask = (mx > 70) & ((mx - mn) > 0.8 * mx)
    ys0, xs0 = np.nonzero(mask)
    out = []
    if len(ys0) == 0:
        return out
    hk, ok = _hue_key(im[ys0, xs0], mx[ys0, xs0])
    groups = []
    for h in sorted(set(hk[ok].tolist())):
        m = np.zeros(mask.shape, dtype=bool)
        sel = ok & (hk == h)
        m[ys0[sel], xs0[sel]] = True
        for ys, xs in _components(m):
            if len(ys) < min_px:
                continue
            if len(ys) > split_px:
                groups.extend(_split2(ys, xs))
            else:
                groups.append((ys, xs))
    for ys, xs in groups:
        if len(ys) < min_px:
            continue
        px = im[ys, xs]
        v = px.max(1)
        thr = np.percentile(v, 90) * 0.95
        top = v >= thr
        c = px[top].mean(0)
        b = {"n": int(len(ys)), "v": float(ys.mean()), "u": float(xs.mean()),
             "rgb": [float(c[0]), float(c[1]), float(c[2])], "color": _color_name(c)}
        if with_3d:
            P = _backproject(np.asarray(frame.front_depth), np.asarray(frame.front_K),
                             np.asarray(frame.front_extrinsic), ys[top], xs[top])
            if len(P) >= 3:
                b["xyz"] = [float(P[:, 0].mean()), float(P[:, 1].mean()), float(P[:, 2].max())]
                b["ntop"] = int(len(P))
        out.append(b)
    return out


def _button(frame):
    """Pink button cap: centre xy and top height from front RGB-D, or None."""
    im = np.asarray(frame.front_rgb).astype(float)
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    mask = (r > 185) & (np.abs(g - b) < 14) & (g > 115) & (g < 165) & (r - g > 50)
    comps = [c for c in _components(mask) if len(c[0]) >= 6]
    if not comps:
        return None
    ys, xs = max(comps, key=lambda c: len(c[0]))
    P = _backproject(np.asarray(frame.front_depth), np.asarray(frame.front_K),
                     np.asarray(frame.front_extrinsic), ys, xs)
    if len(P) < 3:
        return None
    return {"xyz": [float(np.median(P[:, 0])), float(np.median(P[:, 1])), float(np.median(P[:, 2]))],
            "n": int(len(ys)), "v": float(ys.mean()), "u": float(xs.mean())}


def _is_cue_frame(blobs):
    return len(blobs) == 1


def _scan_frames(frames, state):
    """Update cue/layout state from a list of frames.

    state: dict with keys cue (ordered list of colour names), cue_steps, layout (list of
    {color, xyz}), layout_step, button.  Cue = frames with exactly one colour blob before the
    layout (>=7 blobs) appears; consecutive distinct colours are appended in order.
    """
    for fr in frames:
        if state.get("layout"):
            break
        bl = _blobs(fr)
        if len(bl) == 1:
            name = bl[0]["color"]
            if not state["cue"] or state["cue"][-1] != name or state.get("gap", False):
                if name not in state["cue"]:
                    state["cue"].append(name)
                    state["cue_steps"].append(int(fr.step))
            state["gap"] = False
        elif len(bl) == 0:
            state["gap"] = True
        elif len(bl) >= 7:
            full = _blobs(fr, with_3d=True)
            state["layout"] = [{"color": b["color"], "xyz": b.get("xyz"), "n": b["n"], "rgb": b["rgb"]} for b in full if b.get("xyz")]
            state["layout_step"] = int(fr.step)
            btn = _button(fr)
            state["button"] = btn["xyz"] if btn else None
    return state
