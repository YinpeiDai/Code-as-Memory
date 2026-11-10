"""Motion helpers: orientation math, a first-order tracking law fitted to the
expert recordings (one step moves ~0.035*a m plus ~0.11 of the previous step's
motion), ball-trajectory estimation from per-step detections."""

STEP_GAIN = 0.035      # m of TCP motion per unit translation command (fit on demos)
STEP_MOM = 0.11        # carry-over of the previous step's displacement (fit on demos)
ROT_GAIN = 0.035       # rad per unit rotation command (0.1 rad * ~0.35 response)


def _rpy_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    if ang > math.pi - 1e-3:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        ax = M[:, i] / math.sqrt(max(M[i, i], 1e-9))
        return ax * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * math.sin(ang)))


def _orient_cmd(tcp, yaw_des, k=0.45):
    """Rotation command (drx, dry, drz) that turns the measured TCP toward a
    top-down grasp orientation Rz(yaw_des) Rx(pi). The native convention is
    angular offset = -0.1*r about the base axes, so r = -offset/0.1."""
    Rc = _rpy_R(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _rpy_R(math.pi, 0.0, yaw_des)
    w = _rotvec(Rd @ Rc.T)
    off = k * w
    # achieved rotation per step ~= ROT_GAIN * (-r)  =>  r = -off / ROT_GAIN
    r = -off / ROT_GAIN
    n = float(np.linalg.norm(r))
    if n > 1.0:
        r = r / n
    return r, float(np.linalg.norm(w))


def _track_cmd(p, dp_prev, target, target_vel, k=0.6, vmax=0.039, vzmax=0.039):
    """Translation command (dx, dy, dz) so the next measured TCP moves toward
    `target` (position now) which itself moves `target_vel` per step."""
    e = np.asarray(target, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    want = k * e + np.asarray(target_vel, dtype=np.float64)
    if abs(want[2]) > vzmax:
        want[2] = vzmax if want[2] > 0 else -vzmax
    n = float(np.linalg.norm(want))
    if n > vmax:
        want = want * (vmax / n)
    a = (want - STEP_MOM * np.asarray(dp_prev, dtype=np.float64)) / STEP_GAIN
    return np.clip(a, -1.0, 1.0)


def _fit_ball(hist, t_now):
    """hist: list of [t, x, y, z]. Returns (pos_at_t_now[3], vel[3], n_used, resid)."""
    if not hist:
        return None
    H = np.array(hist[-10:], dtype=np.float64)
    if len(H) == 1:
        return H[0, 1:4].copy(), np.zeros(3), 1, 0.0
    def fit(Hs):
        ts = Hs[:, 0]
        A = np.stack([ts - t_now, np.ones_like(ts)], 1)
        coef = np.linalg.lstsq(A, Hs[:, 1:4], rcond=None)[0]
        res = Hs[:, 1:4] - A @ coef
        return coef[1], coef[0], res
    pos, vel, res = fit(H)
    last_res = float(np.linalg.norm(res[-1, :2]))
    if len(H) >= 4 and last_res > 0.005:
        # trajectory changed (e.g. contact): trust only the latest samples
        H2 = H[-3:]
        pos, vel, res = fit(H2)
        n_used = 3
    else:
        n_used = len(H)
    vel = np.array(vel)
    vel[2] = 0.0
    sp = float(np.linalg.norm(vel[:2]))
    if sp > 0.04:
        vel = vel * (0.04 / sp)
    return np.array(pos), vel, n_used, last_res
