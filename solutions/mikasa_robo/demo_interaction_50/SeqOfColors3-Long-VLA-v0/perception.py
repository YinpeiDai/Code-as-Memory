"""Perception helpers for SeqOfColors3: colour classification, depth back-projection,
cue reading and cube/button localisation. Helpers only (names start with _)."""

COLOR_NAMES = ["red", "darkred", "orange", "yellow", "green", "cyan", "teal", "blue", "magenta"]
# Calibrated from 150 expert touches / 50 presses in the demos: expert TCP xy minus
# the perceived top-face centroid (std ~2 mm for cubes, ~5 mm for the button).
CUBE_XY_OFFSET = (-0.007, 0.0024)
BUTTON_XY_OFFSET = (0.016, 0.0025)


def _world_points(frame, cam):
    """Back-project a camera's depth image to world XYZ (H x W x 3) and depth in metres."""
    K = np.asarray(getattr(frame, cam + "_K"), dtype=float)
    E = np.asarray(getattr(frame, cam + "_extrinsic"), dtype=float)
    d = np.asarray(getattr(frame, cam + "_depth"), dtype=float) / 1000.0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1).reshape(-1, 3)
    W = (P - E[:, 3]) @ E[:, :3]
    return W.reshape(h, w, 3), d


def _classify(rgb):
    """Per-pixel colour label index into COLOR_NAMES, -1 for unsaturated pixels."""
    rgb = np.asarray(rgb)
    r = rgb[..., 0].astype(float)
    g = rgb[..., 1].astype(float)
    b = rgb[..., 2].astype(float)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    ok = ((mx - mn) > 100) & (mn < 60)
    R = r > 0.5 * mx
    G = g > 0.5 * mx
    B = b > 0.5 * mx
    lab = np.full(r.shape, -1)
    redf = ok & R & ~G & ~B
    yel = ok & R & G & ~B
    cy = ok & ~R & G & B
    gr = g / np.maximum(r, 1.0)
    lab[redf & (mx >= 190)] = 0
    lab[redf & (mx < 190)] = 1
    lab[yel & (gr < 0.86)] = 2
    lab[yel & (gr >= 0.86)] = 3
    lab[ok & ~R & G & ~B] = 4
    lab[cy & (mx >= 190)] = 5
    lab[cy & (mx < 190)] = 6
    lab[ok & ~R & ~G & B] = 7
    lab[ok & R & ~G & B] = 8
    return lab


def _top_face_cubes(frame, cam="front", zlo=0.036, zhi=0.05, min_px=6):
    """Colour -> [x, y, pixel_count] of cube top faces visible in one frame.
    Top faces are pixels whose back-projected height lies in [zlo, zhi] m (cube top ~0.041 m).
    The centroid is a robust mean around the median, so stray side-face pixels of a
    similar-hued cube do not shift it."""
    W, d = _world_points(frame, cam)
    lab = _classify(getattr(frame, cam + "_rgb"))
    top = (W[..., 2] > zlo) & (W[..., 2] < zhi) & (d > 0)
    out = {}
    for k in range(len(COLOR_NAMES)):
        m = top & (lab == k)
        n = int(m.sum())
        if n < min_px:
            continue
        xy = W[m][:, :2]
        med = np.median(xy, axis=0)
        near = np.linalg.norm(xy - med, axis=1) < 0.03
        if int(near.sum()) < min_px:
            continue
        c = xy[near].mean(axis=0)
        out[COLOR_NAMES[k]] = [float(c[0]), float(c[1]), int(near.sum())]
    return out


def _button_cap(frame, cam="front"):
    """[x, y, z, pixels] of the pink button cap top (pixels above 0.052 m), or None."""
    W, d = _world_points(frame, cam)
    rgb = np.asarray(getattr(frame, cam + "_rgb")).astype(int)
    m = ((W[..., 2] > 0.052) & (W[..., 2] < 0.07) & (d > 0)
         & (rgb[..., 0] > rgb[..., 1] + 40) & (np.abs(rgb[..., 1] - rgb[..., 2]) < 20))
    n = int(m.sum())
    if n < 10:
        return None
    p = W[m]
    med = np.median(p[:, :2], axis=0)
    near = np.linalg.norm(p[:, :2] - med, axis=1) < 0.05
    c = p[near].mean(axis=0)
    return [float(c[0]), float(c[1]), float(c[2]), int(near.sum())]


def _scene_colors(frame):
    """{colour: top-face pixel count} for cube top faces in the front camera (depth-gated,
    so the orange-brown table is excluded)."""
    c = _top_face_cubes(frame, "front", min_px=8)
    return {k: v[2] for k, v in c.items()}


def _cue_color(counts):
    """The single dominant cue colour of one frame, or None. Edge pixels of a bright cube
    can be labelled with its dark twin (red/darkred, cyan/teal), so require dominance."""
    if not counts:
        return None
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    if items[0][1] < 20:
        return None
    if len(items) > 1 and items[1][1] * 3 > items[0][1]:
        return None
    return items[0][0]


def _layout_visible(counts):
    """True once the response layout (several cubes at once) is on the table."""
    return len([k for k in counts if counts[k] >= 15]) >= 5
