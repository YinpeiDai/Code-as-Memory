"""Rotation math and the waypoint controller (reverse-engineered from the expert: pos gain 6, rot gain 2, rot clip 0.5)."""

KP_POS = 6.0
KP_ROT = 2.0
ROT_CLIP = 0.5
# 120 deg about (-1,1,1)/sqrt(3): gripper turned sideways (used for handle / microwave interior)
SIDE_AA = [-1.2091995761561452, 1.2091995761561452, 1.2091995761561452]
# 180 deg about world x: gripper pointing straight down
TOP_AA = [math.pi, 0.0, 0.0]


def _aa_to_R(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def _R_to_aa(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _orient_aa(name):
    return SIDE_AA if name == "side" else TOP_AA


def _control(tcp, target, orient):
    """One expert-style action [dx,dy,dz,drx,dry,drz] toward target xyz and named orientation."""
    tcp = np.asarray(tcp, dtype=float)
    p = np.clip(KP_POS * (np.asarray(target, dtype=float) - tcp[:3]), -1.0, 1.0)
    Rt = _aa_to_R(_orient_aa(orient))
    Rc = _aa_to_R(tcp[3:6])
    w = _R_to_aa(Rt @ Rc.T)
    r = np.clip(KP_ROT * w, -ROT_CLIP, ROT_CLIP)
    return [float(p[0]), float(p[1]), float(p[2]), float(r[0]), float(r[1]), float(r[2])]
