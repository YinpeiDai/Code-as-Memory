"""RotateLenientPos policy: perceive the blue peg (front + wrist RGB-D), then rotate it
counter-clockwise (about world +z) by the instructed angle by pushing one end sideways with
the closed gripper, closed-loop on the measured peg yaw, then stop and let the arm settle."""

PEG_L = 0.24          # measured peg length (m), demos: 0.238-0.240
PEG_W = 0.05          # measured peg width/height (m)
FINGER_R = 0.012      # assumed half thickness of the closed finger tips (m)
LEVER = 0.085         # push contact point distance from peg centre along its axis (m)
Z_PUSH = 0.03         # TCP height while pushing (peg top is at 0.05)
Z_HOVER = 0.09        # TCP height when travelling above the table
STEP_GAIN = 0.39      # measured fraction of a commanded offset achieved in one step
GRIP = -1.0           # gripper closed for pushing
BASE_XY = (-0.615, 0.0)


def _wrap90(a):
    return (a + 90.0) % 180.0 - 90.0


def _parse_target(goal):
    """Angle in degrees from 'Rotate the peg by N degrees ...'. Sign kept if present."""
    words = str(goal).replace(',', ' ').split()
    for i, w in enumerate(words):
        if w.lower().startswith('degree') and i > 0:
            try:
                return float(words[i - 1])
            except ValueError:
                pass
    num = ''
    for ch in str(goal):
        if ch.isdigit() or ch in '.-':
            num += ch
        elif num:
            break
    try:
        return float(num)
    except ValueError:
        return 0.0


def _cam_points(frame, cam):
    rgb = getattr(frame, cam + '_rgb')
    depth = getattr(frame, cam + '_depth')
    K = getattr(frame, cam + '_K')
    E = getattr(frame, cam + '_extrinsic')
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    m = (b > 100) & (b - r > 60) & (b - g > 30) & (depth > 0)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs].astype(np.float64) / 1000.0
    xc = (xs - K[0, 2]) / K[0, 0] * z
    yc = (ys - K[1, 2]) / K[1, 1] * z
    P = np.stack([xc, yc, z], 1)
    R = np.asarray(E[:, :3], dtype=np.float64)
    t = np.asarray(E[:, 3], dtype=np.float64)
    W = (P - t) @ R
    keep = (W[:, 2] < 0.065) & (W[:, 2] > -0.01)
    return W[keep]


def _fit_axis(xy):
    """Yaw (rad, in [-pi/2, pi/2)) of the long axis by minimising the robust width."""
    if len(xy) > 2500:
        xy = xy[np.linspace(0, len(xy) - 1, 2500).astype(int)]
    c = xy.mean(0)
    def width(a):
        p = (xy - c) @ np.array([-math.sin(a), math.cos(a)])
        return np.percentile(p, 99) - np.percentile(p, 1)
    angs = np.radians(np.arange(-90.0, 90.0, 3.0))
    ws = [width(a) for a in angs]
    a0 = angs[int(np.argmin(ws))]
    best = min(ws)
    for st in (1.0, 0.25):
        cand = a0 + np.radians(np.arange(-3.0, 3.01, st))
        ws = [width(a) for a in cand]
        a0 = cand[int(np.argmin(ws))]
        best = min(ws)
    a0 = (a0 + math.pi / 2) % math.pi - math.pi / 2
    return a0, best


def _cam_fit(P, min_ext=0.09):
    """Axis fit of one camera's points; valid only if the visible blob looks like a peg footprint."""
    if len(P) < 40:
        return None
    xy = P[:, :2]
    a, wid = _fit_axis(xy)
    u = np.array([math.cos(a), math.sin(a)])
    pu = (xy - xy.mean(0)) @ u
    ext = float(np.percentile(pu, 99.5) - np.percentile(pu, 0.5))
    ok = (0.035 <= wid <= 0.07) and ext >= min_ext
    return {'a': a, 'w': float(wid), 'ext': ext, 'ok': ok, 'xy': xy}


def _interval_pick(lo, hi, half, prior):
    """Centre coordinate consistent with a visible extent [lo, hi] of a part of known half size."""
    cmin, cmax = hi - half, lo + half
    if cmin > cmax:
        return 0.5 * (lo + hi)
    if prior is None:
        return 0.5 * (cmin + cmax)
    return max(cmin, min(cmax, prior))


def _peg(frame, tcp_xy=None, prior_c=None):
    """Peg estimate from one frame: yaw_deg (mod 180), centre xy, visible extent, pixel counts.
    Each camera is fitted separately (a close wrist view may show only a side face); valid fits
    are fused. The centre is the point consistent with the visible extents (known 0.24 x 0.05
    footprint) closest to prior_c (previous estimate) when part of the peg is occluded."""
    fits = []
    counts = []
    for cam in ('front', 'wrist'):
        P = _cam_points(frame, cam)
        counts.append(int(len(P)))
        ft = _cam_fit(P)
        if ft is not None and ft['ok']:
            fits.append(ft)
    if not fits:
        # weaker acceptance: short visible piece with a clean width
        for cam in ('front', 'wrist'):
            ft = _cam_fit(_cam_points(frame, cam), min_ext=0.08)
            if ft is not None and ft['ok']:
                fits.append(ft)
        if not fits:
            return None
    if len(fits) == 2:
        da = _wrap90(math.degrees(fits[1]['a'] - fits[0]['a']))
        if abs(da) <= 4.0:
            a = fits[0]['a'] + math.radians(da) / 2
        else:
            a = max(fits, key=lambda q: q['ext'])['a']
    else:
        a = fits[0]['a']
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    xy = np.concatenate([q['xy'] for q in fits], 0)
    u = np.array([math.cos(a), math.sin(a)])
    n = np.array([-u[1], u[0]])
    pu = xy @ u
    pn = xy @ n
    lo, hi = float(np.percentile(pu, 0.5)), float(np.percentile(pu, 99.5))
    nlo, nhi = float(np.percentile(pn, 1)), float(np.percentile(pn, 99))
    pc = None if prior_c is None else np.asarray(prior_c, dtype=float)
    cu = _interval_pick(lo, hi, PEG_L / 2, None if pc is None else float(pc @ u))
    cn = _interval_pick(nlo, nhi, PEG_W / 2, None if pc is None else float(pc @ n))
    c = cu * u + cn * n
    return {'yaw': math.degrees(a), 'c': [float(c[0]), float(c[1])], 'ext': float(hi - lo),
            'width': float(min(q['w'] for q in fits)), 'n': int(len(xy)), 'n_front': counts[0],
            'n_wrist': counts[1], 'cams': len(fits),
            'strong': bool(max(q['ext'] for q in fits) >= 0.15)}


def _Rrpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _logR(M):
    c = max(-1.0, min(1.0, (np.trace(M) - 1) / 2))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    s = math.sin(th)
    if s < 1e-6:
        return np.zeros(3)
    return w / (2 * s) * th


def _rot_cmd(tcp, yaw_des, gain=1.2):
    """Rotation arguments that move the gripper toward pointing down with yaw yaw_des."""
    Rm = _Rrpy(float(tcp[3]), float(tcp[4]), float(tcp[5]))
    Rd = _Rrpy(math.pi, 0.0, yaw_des)
    e = _logR(Rd @ Rm.T)                    # world-frame rotation still needed
    r = -gain * e / (0.1 * STEP_GAIN)       # native offset is -0.1*r per step (partial)
    nr = float(np.linalg.norm(r))
    if nr > 1.0:
        r = r / nr
    return r


def _trans_cmd(tcp, p_des, gain=1.0, vmax=1.0):
    """Translation arguments aimed to cover `gain` of the error this step (bounded)."""
    e = np.asarray(p_des, dtype=float) - np.asarray(tcp[:3], dtype=float)
    a = gain * e / (0.1 * STEP_GAIN)
    m = float(np.max(np.abs(a)))
    if m > vmax:
        a = a * (vmax / m)
    return a


def _step(robot, a, r):
    robot.step(float(a[0]), float(a[1]), float(a[2]), float(r[0]), float(r[1]), float(r[2]), GRIP)
    return robot.now()


def _unwrap(yaw, ref, prev_rot):
    return prev_rot + _wrap90(yaw - (ref + prev_rot))


def _accept(peg, ref, rot):
    """A weak (short visible piece) estimate is used only if close to the tracked rotation."""
    if peg is None:
        return False
    if peg['strong'] or ref is None:
        return True
    return abs(_wrap90(peg['yaw'] - (ref + rot))) <= 12.0


def _rot2(v, phi):
    cph, sph = math.cos(phi), math.sin(phi)
    return np.array([cph * v[0] - sph * v[1], sph * v[0] + cph * v[1]])


def _push_path(c, u, n, s, direction, rem_rad):
    """Predicted TCP contact path while pushing end s: the peg turns by rem_rad and (measured in
    rollouts R001-R009) its centre drifts ~0.14 m per rad along the push direction."""
    d0 = direction * s * n
    o0 = s * LEVER * u - d0 * (PEG_W / 2 + FINGER_R)
    pts = []
    for k in range(7):
        phi = direction * abs(rem_rad) * k / 6.0
        cc = c + 0.14 * abs(phi) * _rot2(d0, phi / 2)
        pts.append(cc + _rot2(o0, phi))
    return pts


def _plan_end(peg, tcp, direction, rem_deg=45.0, avoid_xy=None):
    """Choose the peg end and push direction. direction=+1 rotates CCW (+z).
    Score: approach distance, the predicted push path must stay within the arm's low reach
    (radius 0.36-0.75 m from the base; R010 stalled at 0.78, R022 reached 0.776; the path model
    matched measured TCP radii within 0.03 in R011-R025), and strongly prefer the end nearer the
    robot when the peg points at the robot: otherwise the hand hides the peg body from the front
    camera and the wrist camera (looking toward +x) cannot see it either (R003, R024)."""
    a = math.radians(peg['yaw'])
    u = np.array([math.cos(a), math.sin(a)])
    n = np.array([-u[1], u[0]])
    c = np.array(peg['c'])
    base = np.array(BASE_XY)
    best = None
    for s in (1.0, -1.0):
        d = direction * s * n
        pre = c + s * LEVER * u - d * (PEG_W / 2 + FINGER_R + 0.02)
        dist = float(np.linalg.norm(pre - np.asarray(tcp[:2])))
        path = [pre] + _push_path(c, u, n, s, direction, math.radians(rem_deg))
        rads = [float(np.linalg.norm(p - base)) for p in path]
        over_start = max(0.0, rads[0] - 0.76)
        over = max(0.0, max(rads) - 0.76) + max(0.0, 0.36 - min(rads))
        sx = s * float(u[0])             # > 0: the peg body lies between the hand and the robot
        vis = 0.5 * LEVER * sx + 0.4 * max(0.0, sx) ** 2
        # peg lost from view in every far-end push with sx >= 0.81 (R003/R024/R036/R038/R046);
        # sx 0.41-0.76 worked (R002/R015)
        vis += 3.0 * min(1.0, max(0.0, (sx - 0.7) / 0.15))
        # an over-reaching path only ends the push early (reach guard / stall) and is re-planned,
        # so its penalty is capped; an unreachable start is not recoverable
        prad = rads[1:]
        reach_ok = len(prad) - 1
        for k in range(len(prad)):
            if prad[k] > 0.78:
                reach_ok = max(0, k - 1)
                break
        lost = 1.0 - reach_ok / float(len(prad) - 1)   # fraction of the turn beyond reach
        score = dist + vis + 40.0 * over_start + min(1.5, 20.0 * over) * (0.3 + 0.7 * lost) + 1.5 * lost
        if avoid_xy is not None and float(np.linalg.norm(pre - np.asarray(avoid_xy))) < 0.08:
            score += 0.6                 # this contact failed (stall / lost) last time
        if best is None or score < best[0]:
            best = (score, s, d, pre, max(rads))
    return best[1], best[3], best[4]


def _track(frames, ref, rot, prior_c):
    """Unwrapped rotation through a frame sequence; returns (ref, rot, last peg estimate, centre)."""
    peg = None
    for fr in frames:
        p = _peg(fr, np.asarray(fr.tcp, dtype=float)[:2], prior_c)
        if not _accept(p, ref, rot):
            continue
        peg = p
        prior_c = p['c']
        if ref is None:
            ref = p['yaw']
            rot = 0.0
        rot = _unwrap(p['yaw'], ref, rot)
    return ref, rot, peg, prior_c


def _parse_fail(last):
    """xy of the pre-contact point of a failed push, from the previous act outcome text."""
    if not last:
        return None
    t = str(last)
    k = t.find('"fail_pre": [')
    if k < 0:
        return None
    j = t.find(']', k)
    try:
        v = [float(x) for x in t[k + len('"fail_pre": ['):j].split(',')]
    except ValueError:
        return None
    return v if len(v) == 2 else None


def perceive_decide(obs, goal, memory, last):
    """Parse the angle, track the peg rotation through all new frames, and choose
    rotate (push), hold (settle) or search (lift to view the peg)."""
    mem = dict(memory) if memory else {}
    target = mem.get('target_deg')
    if target is None:
        target = _parse_target(goal)
    ref = mem.get('ref_yaw_deg')
    rot = mem.get('rotation_deg')
    if rot is None:
        rot = 0.0
    prior_c = mem.get('peg_center')
    ref, rot, peg, prior_c = _track(obs.frames, ref, rot, prior_c)
    n_dec = int(mem.get('n_decisions') or 0) + 1
    err = target - rot
    pushes = int(mem.get('pushes') or 0)
    holds = int(mem.get('holds') or 0)
    thresh = 3.0 if holds < 2 else 1.5
    if peg is None or ref is None:
        act = {'tool': 'act_search', 'args': {}}
        stage = 'search'
    elif abs(err) > thresh and pushes < 6:
        act = {'tool': 'act_rotate_peg', 'args': {'ref_yaw_deg': float(ref), 'start_rot_deg': float(rot),
                                                  'target_deg': float(target), 'prior_c': prior_c,
                                                  'avoid_xy': _parse_fail(last)}}
        stage = 'rotate'
        pushes += 1
    else:
        act = {'tool': 'act_hold', 'args': {'steps': 4}}
        stage = 'hold'
        holds += 1
    out = {
        'target_deg': float(target),
        'ref_yaw_deg': None if ref is None else float(ref),
        'rotation_deg': float(rot),
        'peg_center': prior_c,
        'stage': stage,
        'pushes': pushes,
        'holds': holds,
        'n_decisions': n_dec,
        'last_result': str(last)[:600] if last is not None else '',
    }
    return {'memory': out, 'act': act}


def act_hold(robot, steps=4):
    """Hold still (zero motion, gripper closed) for `steps` native steps; report joint motion and step."""
    moves = []
    for _ in range(int(steps)):
        q0 = np.asarray(robot.now().joints, dtype=float)
        f = _step(robot, np.zeros(3), np.zeros(3))
        moves.append(round(float(np.max(np.abs(np.asarray(f.joints, dtype=float) - q0))), 4))
    return {'held': int(steps), 'max_dq_per_step': moves, 'step': int(robot.now().step)}


def act_search(robot, max_steps=8):
    """The peg is not visible: lift the gripper and move it back toward the robot (above
    x=-0.12, z=0.25) until a peg estimate is available again; report steps and whether it was found."""
    found = False
    for k in range(int(max_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        if k > 0 and _peg(f, tcp[:2]) is not None:
            found = True
            break
        target = [min(tcp[0], -0.12) if tcp[2] > 0.15 else tcp[0], tcp[1] * 0.5, 0.25]
        _step(robot, _trans_cmd(tcp, target, gain=0.8), _rot_cmd(tcp, 0.0))
    return {'found': found, 'step': int(robot.now().step)}


def act_rotate_peg(robot, ref_yaw_deg, start_rot_deg, target_deg, prior_c=None, avoid_xy=None,
                   lead_deg=0.8, max_push_steps=30):
    """Approach one end of the peg beside its side face, push it so it turns toward target_deg
    (rotation about +z relative to ref_yaw_deg, unwrapped from start_rot_deg), stop, back off
    and settle. Returns the measured rotation, stage log and step counts."""
    log = []
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    yaw_g = 0.0
    peg = _peg(f, tcp[:2], prior_c)
    if not _accept(peg, ref_yaw_deg, start_rot_deg):
        return {'status': 'no_peg', 'step': int(f.step)}
    prior_c = peg['c']
    rot = _unwrap(peg['yaw'], ref_yaw_deg, start_rot_deg)
    err = target_deg - rot
    direction = 1.0 if err > 0 else -1.0
    s, pre, rmax = _plan_end(peg, tcp, direction, abs(err), avoid_xy)
    plan_yaw = peg['yaw']
    log.append('plan rot=%.1f err=%.1f s=%d pre=(%.3f,%.3f) c=(%.3f,%.3f) yaw=%.1f rmax=%.3f' % (
        rot, err, int(s), pre[0], pre[1], peg['c'][0], peg['c'][1], plan_yaw, rmax))
    # --- approach: travel at hover height, then descend beside the peg ---
    t0 = int(f.step)
    for k in range(25):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        exy = float(np.linalg.norm(pre - tcp[:2]))
        g = 0.85
        if exy > 0.02 and tcp[2] < Z_HOVER - 0.012:
            target = [tcp[0], tcp[1], Z_HOVER]          # rise straight up before travelling
        elif exy > 0.02:
            zt = min(max(Z_HOVER, tcp[2] - 0.06), tcp[2]) if tcp[2] > Z_HOVER else Z_HOVER
            target = [pre[0], pre[1], zt]
        else:
            target = [pre[0], pre[1], Z_PUSH]
            if tcp[2] < 0.12:
                g = 0.5                  # arm momentum overshot to z 0.013-0.017 with 0.85 (R005, R038)
        if exy < 0.012 and abs(tcp[2] - Z_PUSH) < 0.008:
            break
        a_cmd = _trans_cmd(tcp, target, gain=g)
        _step(robot, a_cmd, _rot_cmd(tcp, yaw_g))
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    log.append('approach steps=%d tcp=(%.3f,%.3f,%.3f)' % (int(f.step) - t0, tcp[0], tcp[1], tcp[2]))
    # --- push ---
    t1 = int(f.step)
    a_p = math.radians(plan_yaw)
    u_prev = np.array([math.cos(a_p), math.sin(a_p)])
    status = 'push_limit'
    rot_start = None
    bad = 0
    stall = 0
    last_xy = None
    last_cmd = 0.0
    d = np.zeros(2)
    for k in range(int(max_push_steps)):
        f = robot.now()
        tcp = np.asarray(f.tcp, dtype=float)
        peg = _peg(f, tcp[:2], prior_c)
        if not _accept(peg, ref_yaw_deg, rot):
            bad += 1
            if bad >= 2:
                status = 'lost_peg'
                break
            _step(robot, _trans_cmd(tcp, [tcp[0], tcp[1], Z_PUSH], gain=0.5), _rot_cmd(tcp, yaw_g))
            continue
        prior_c = peg['c']
        rot_prev = rot
        rot = _unwrap(peg['yaw'], ref_yaw_deg, rot)
        if rot_start is None:
            rot_start = rot
            rot_prev = rot
        rate = max(0.0, direction * (rot - rot_prev))
        rem = target_deg - rot
        if direction * rem <= lead_deg + 0.7 * rate:
            status = 'reached'
            break
        a = math.radians(peg['yaw'])
        u = np.array([math.cos(a), math.sin(a)])
        if float(u @ u_prev) < 0:
            u = -u
        u_prev = u
        n = np.array([-u[1], u[0]])
        c = np.array(peg['c'])
        d = direction * s * n
        rel = tcp[:2] - c
        along = float(rel @ (s * u))
        h = float(rel @ d)               # signed distance along push dir (contact at -(W/2+R))
        if along < 0.0 or along > PEG_L / 2 + 0.02 or h > 0.0:
            bad += 1
            if bad >= 2:
                status = 'lost_contact along=%.3f h=%.3f' % (along, h)
                break
            _step(robot, _trans_cmd(tcp, [tcp[0], tcp[1], Z_PUSH], gain=0.5), _rot_cmd(tcp, yaw_g))
            continue
        bad = 0
        v = min(0.02, max(0.004, 0.0025 + abs(math.radians(rem)) * 0.05))
        h_goal = -(PEG_W / 2 + FINGER_R)
        progressed = abs(rot - rot_start) > 1.5
        if not progressed or h < h_goal - 0.02:
            dv = min(0.012, v + 0.5 * max(0.0, h_goal - h))
        else:
            dv = v
        corr = max(-0.006, min(0.006, 0.5 * (LEVER - along)))
        p_des = tcp[:2] + d * dv + s * u * corr
        a_cmd = _trans_cmd(tcp, [p_des[0], p_des[1], Z_PUSH], gain=1.0)
        if float(np.linalg.norm(tcp[:2] - np.array(BASE_XY))) > 0.785:
            status = 'reach_limit'
            break
        if last_xy is not None and float(np.linalg.norm(tcp[:2] - last_xy)) < 0.0015 and last_cmd > 0.15:
            stall += 1
            if stall >= 2:
                status = 'stalled at r=%.3f' % float(np.linalg.norm(tcp[:2] - np.array(BASE_XY)))
                break
        else:
            stall = 0
        last_xy = tcp[:2].copy()
        last_cmd = float(np.max(np.abs(a_cmd[:2])))
        if len(log) < 40:
            log.append('k%d rot=%.1f al=%.3f h=%.3f dv=%.3f' % (k, rot, along, h, dv))
        _step(robot, a_cmd, _rot_cmd(tcp, yaw_g))
    t2 = int(robot.now().step)
    log.append('push end %s rot=%.1f' % (status, rot))
    # --- back off from the side face (and up when the peg was lost) and settle ---
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    up = 0.01 if status == 'reached' else 0.06
    back = tcp[:3] + np.array([-d[0] * 0.02, -d[1] * 0.02, up])
    for k in range(2):
        tcp = np.asarray(robot.now().tcp, dtype=float)
        _step(robot, _trans_cmd(tcp, back, gain=0.7), _rot_cmd(tcp, yaw_g, gain=0.5))
    dqs = []
    for k in range(8):
        q0 = np.asarray(robot.now().joints, dtype=float)
        f = _step(robot, np.zeros(3), np.zeros(3))
        dq = float(np.max(np.abs(np.asarray(f.joints, dtype=float) - q0)))
        dqs.append(round(dq, 4))
        if k >= 2 and dq < 0.004:
            break
    f = robot.now()
    tcp = np.asarray(f.tcp, dtype=float)
    peg = _peg(f, tcp[:2], prior_c)
    if _accept(peg, ref_yaw_deg, rot):
        rot = _unwrap(peg['yaw'], ref_yaw_deg, rot)
    fail = None if status == 'reached' else [round(float(pre[0]), 4), round(float(pre[1]), 4)]
    return {'status': status, 'fail_pre': fail, 'rotation_deg': round(rot, 2), 'err_deg': round(target_deg - rot, 2),
            'push_steps': t2 - t1, 'step': int(f.step), 'settle_dq': dqs, 'log': log[-16:]}
