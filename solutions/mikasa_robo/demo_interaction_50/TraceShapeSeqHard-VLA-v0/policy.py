"""TraceShapeSeqHard policy: watch the red cue cube, fit its shapes, grasp the green cube, retrace the
shapes in order at carrying height, release the cube and press the button."""

TRACE_Z = 0.06       # TCP height while tracing (expert median 0.060)
GRASP_Z = 0.032      # TCP height at gripper close (expert 0.030-0.040)
MAX_GRASP_TRIES = 3
MAX_PRESS_TRIES = 4

FIELDS = ('stage', 'start_step', 'green', 'button', 'red_samples', 'green_step', 'shapes',
          'shape_idx', 'grasp_tries', 'press_tries', 'notes')


def _holding(frame):
    f = np.array(frame.fingers, float)
    return bool(f.min() > 0.008 and f.max() < 0.03)


def _note(mem, text):
    notes = mem.get('notes') or ''
    notes = (notes + ' | ' + text)[-1500:]
    mem['notes'] = notes


def _collect_red(frames, samples):
    """Append red-cube samples (both cameras) until the lamp turns green; return lamp-green step or None."""
    for fr in frames:
        if _lamp_state(fr) == 'green':
            return int(fr.step)
        for cam in ('wrist', 'front'):
            p, n = _locate_top(fr, cam, 'red')
            if p is not None and 0.02 < p[2] < 0.07:
                samples.append([int(fr.step), round(float(p[0]), 5), round(float(p[1]), 5)])
    return None


def _green_from(frame):
    pf, nf = _locate_top(frame, 'front', 'green')
    pw, nw = _locate_top(frame, 'wrist', 'green')
    if pf is not None and nf >= 8 and pf[2] < 0.06:
        return [float(pf[0]), float(pf[1]), float(pf[2])]
    if pw is not None and pw[2] < 0.06:
        return [float(pw[0]), float(pw[1]), float(pw[2])]
    return None


def _button_from(frame):
    p = _locate_both(frame, 'button')
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: watch -> grasp -> trace each shape -> release -> press."""
    mem = {}
    for k in FIELDS:
        mem[k] = memory.get(k) if isinstance(memory, dict) else None
    cur = obs.current
    stage = mem['stage']

    if stage is None:
        mem['start_step'] = int(obs.frames[0].step)
        mem['red_samples'] = []
        mem['shape_idx'] = 0
        mem['grasp_tries'] = 0
        mem['press_tries'] = 0
        mem['shapes'] = []
        mem['green'] = _green_from(cur)
        mem['button'] = _button_from(cur)
        samples = []
        G = _collect_red(obs.frames, samples)
        mem['red_samples'] = samples
        mem['stage'] = 'watch'
        _note(mem, 'init green=%s button=%s' % (mem['green'], mem['button']))
        return {'memory': mem, 'act': {'tool': 'act_watch_cue', 'args': {'max_steps': 900}}}

    if stage == 'watch':
        samples = list(mem['red_samples'] or [])
        G = _collect_red(obs.frames, samples)
        if mem['green'] is None:
            mem['green'] = _green_from(obs.frames[0])
        if mem['button'] is None:
            mem['button'] = _button_from(obs.frames[0])
        if G is None:
            mem['red_samples'] = samples
            _note(mem, 'no lamp green yet at step %d' % int(cur.step))
            return {'memory': mem, 'act': {'tool': 'act_watch_cue', 'args': {'max_steps': 900}}}
        shapes = _fit_shapes(samples, G, int(mem['start_step'] or 0))
        mem['green_step'] = G
        mem['shapes'] = shapes
        mem['red_samples'] = []
        mem['stage'] = 'grasp'
        mem['grasp_tries'] = 1
        _note(mem, 'G=%d shapes=%s' % (G, [[s['type'], s['res'], s['n']] for s in shapes]))
        g = mem['green'] or _green_from(cur)
        return {'memory': mem, 'act': {'tool': 'act_grasp', 'args': {'x': g[0], 'y': g[1]}}}

    if stage == 'grasp' or stage == 'trace':
        held = _holding(cur) and float(cur.tcp[2]) > 0.04
        if stage == 'trace' and held:
            mem['shape_idx'] = int(mem['shape_idx'] or 0) + 1
        if not held:
            tries = int(mem['grasp_tries'] or 0)
            if tries < MAX_GRASP_TRIES + int(mem['shape_idx'] or 0):
                g = _green_from(cur) or mem['green']
                mem['green'] = g
                mem['grasp_tries'] = tries + 1
                mem['stage'] = 'grasp'
                _note(mem, 'not holding after %s (fingers %s); regrasp at %s' % (
                    stage, [round(float(v), 4) for v in cur.fingers], [round(v, 3) for v in g]))
                return {'memory': mem, 'act': {'tool': 'act_grasp', 'args': {'x': g[0], 'y': g[1]}}}
            _note(mem, 'grasp failed repeatedly; tracing without cube')
        shapes = mem['shapes'] or []
        idx = int(mem['shape_idx'] or 0)
        while idx < len(shapes) and not shapes[idx]['pts']:
            idx += 1
        mem['shape_idx'] = idx
        if idx < len(shapes):
            mem['stage'] = 'trace'
            s = shapes[idx]
            _note(mem, 'trace %d %s' % (idx, s['type']))
            return {'memory': mem, 'act': {'tool': 'act_trace_shape',
                                           'args': {'pts': s['pts'], 'corner': s['corner'], 'z': TRACE_Z}}}
        mem['stage'] = 'release'
        return {'memory': mem, 'act': {'tool': 'act_release', 'args': {}}}

    if stage == 'release' or stage == 'press':
        tries = int(mem['press_tries'] or 0)
        b = mem['button']
        if tries > 0:
            nb = _button_from(cur)
            if nb is not None and b is not None and math.hypot(nb[0] - b[0], nb[1] - b[1]) < 0.03:
                b = [(nb[0] + b[0]) / 2, (nb[1] + b[1]) / 2, b[2]]
        if b is None:
            b = _button_from(cur)
        mem['button'] = b
        if b is None or tries >= MAX_PRESS_TRIES:
            mem['stage'] = 'done'
            return {'memory': mem, 'act': {'tool': 'act_hold', 'args': {'n': 50}}}
        mem['press_tries'] = tries + 1
        mem['stage'] = 'press'
        _note(mem, 'press try %d at %s' % (tries + 1, [round(v, 3) for v in b]))
        return {'memory': mem, 'act': {'tool': 'act_press', 'args': {'x': b[0], 'y': b[1], 'ztop': b[2],
                                                                     'depth': 0.01 * tries}}}

    return {'memory': mem, 'act': {'tool': 'act_hold', 'args': {'n': 50}}}


def act_watch_cue(robot, max_steps=900):
    """Hold still with the gripper open while the red cube traces its shapes; stop on the first frame
    where the lamp is green. Returns steps waited and lamp observations."""
    seen_red = False
    for k in range(int(max_steps)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        st = _lamp_state(robot.now())
        if st == 'red':
            seen_red = True
        if st == 'green':
            return {'steps': k + 1, 'lamp': 'green', 'step': int(robot.now().step), 'seen_red': seen_red}
    return {'steps': int(max_steps), 'lamp': 'not green', 'seen_red': seen_red}


def act_grasp(robot, x, y, z_above=0.12):
    """Move above the green cube at (x, y), descend with open gripper, close, lift to tracing height."""
    R = _down(0.0)
    out = {}
    out['above'] = _move_to(robot, [x, y, z_above], R, 1.0, tol=0.006, max_steps=150, vmax=1.0)
    out['pre'] = _move_to(robot, [x, y, GRASP_Z + 0.03], R, 1.0, tol=0.004, max_steps=80, vmax=0.5)
    out['down'] = _move_to(robot, [x, y, GRASP_Z], R, 1.0, tol=0.003, max_steps=60, vmax=0.3, settle=3)
    _hold(robot, [x, y, GRASP_Z], R, -1.0, 12)
    out['lift'] = _move_to(robot, [x, y, TRACE_Z], R, -1.0, tol=0.004, max_steps=80, vmax=0.4)
    f = robot.now()
    out['fingers'] = [round(float(v), 4) for v in f.fingers]
    out['tcp'] = [round(float(v), 4) for v in f.tcp[:3]]
    out['holding'] = _holding(f)
    return out


def act_trace_shape(robot, pts, corner, z=0.06, speed=0.008, vmax=0.6):
    """Carry the held cube along the closed waypoint path `pts` ([x, y] list, first == last) at height z.
    Corner points are reached within 4 mm before continuing. Reports tracking and coverage errors."""
    R = _down(0.0)
    P = [np.array(p, float) for p in pts]
    out = {'n_pts': len(P)}
    out['transit'] = _move_to(robot, [P[0][0], P[0][1], z], R, -1.0, tol=0.004, max_steps=150, vmax=vmax)
    path, flags = _densify(pts, corner, speed)
    visited = []
    i = 0
    wait = 0
    steps = 0
    max_lag = 0.0
    while i < len(path) and steps < 600:
        tgt = path[i]
        nxt = path[min(i + 1, len(path) - 1)]
        d = nxt - tgt
        nd = float(np.linalg.norm(d))
        ff = None
        if nd > 1e-9 and not flags[i]:
            v = d / nd * speed / STEP_GAIN * 0.8
            ff = [v[0], v[1], 0.0]
        _cmd(robot, [tgt[0], tgt[1], z], R, -1.0, kp=1.2, vmax=vmax, ff=ff)
        steps += 1
        p = np.array(robot.now().tcp[:2], float)
        visited.append(p)
        lag = float(np.linalg.norm(p - tgt))
        max_lag = max(max_lag, lag)
        if flags[i]:
            wait += 1
            if lag < 0.004 or wait > 25:
                i += 1
                wait = 0
        elif lag < 0.015:
            i += 1
    for k in range(3):
        _cmd(robot, [path[-1][0], path[-1][1], z], R, -1.0, kp=1.2, vmax=vmax)
        visited.append(np.array(robot.now().tcp[:2], float))
    V = np.array(visited)
    cover = [float(np.min(np.linalg.norm(V - q, axis=1))) for q in path]
    xt = []
    for p in V:
        best = 1.0
        for j in range(len(P) - 1):
            best = min(best, _seg_dist(p, P[j], P[j + 1]))
        xt.append(best)
    f = robot.now()
    out.update({'steps': steps, 'max_lag': round(max_lag, 4), 'cover_max': round(max(cover), 4),
                'cover_mean': round(float(np.mean(cover)), 4), 'xtrack_max': round(max(xt), 4),
                'end_err': round(float(np.linalg.norm(V[-1] - P[-1])), 4),
                'fingers': [round(float(v), 4) for v in f.fingers], 'holding': _holding(f)})
    return out


def act_release(robot, z_drop=0.10):
    """Lift the cube a little at the final point, open to drop it, then close the gripper and rise."""
    R = _down(0.0)
    p = np.array(robot.now().tcp[:3], float)
    out = {}
    out['lift'] = _move_to(robot, [p[0], p[1], z_drop], R, -1.0, tol=0.005, max_steps=60, vmax=0.5)
    _hold(robot, [p[0], p[1], z_drop], R, 1.0, 10)
    out['up'] = _move_to(robot, [p[0], p[1], 0.13], R, -1.0, tol=0.01, max_steps=40, vmax=0.8)
    f = robot.now()
    out['fingers'] = [round(float(v), 4) for v in f.fingers]
    return out


def act_press(robot, x, y, ztop=0.058, depth=0.0):
    """Press the button cap at (x, y) with closed fingers: approach low (hover 6 cm above the cap, where
    the arm reaches further), descend 3 mm/step to 1 cm below the cap top (+depth), hold, rise."""
    zhover = ztop + 0.06
    R, ang, info = _press_orientation(robot, x, y, ztop + 0.01)
    out = {'tilt': round(ang, 3), 'reach': info}
    p = np.array(robot.now().tcp[:3], float)
    if p[2] < zhover:
        out['rise'] = _move_to(robot, [p[0], p[1], zhover], R, -1.0, tol=0.01, max_steps=30, vmax=0.8)
    out['hover'] = _move_to(robot, [x, y, zhover], R, -1.0, tol=0.005, max_steps=150, vmax=0.7)
    out['near'] = _move_to(robot, [x, y, ztop + 0.035], R, -1.0, tol=0.004, max_steps=60, vmax=0.5)
    zt = ztop + 0.035
    zmin = ztop - 0.01 - depth
    trace = []
    for k in range(70):
        zt = max(zmin, zt - 0.003)
        _cmd(robot, [x, y, zt], R, -1.0, kp=1.0, vmax=0.4)
        if k % 5 == 0:
            trace.append(round(float(robot.now().tcp[2]), 4))
    out['z_trace'] = trace
    out['up'] = _move_to(robot, [x, y, zhover], R, -1.0, tol=0.01, max_steps=40, vmax=0.8)
    out['tcp'] = [round(float(v), 4) for v in robot.now().tcp]
    return out


def act_hold(robot, n=50):
    """Keep the current pose for n steps (fallback when nothing else is left to do)."""
    f = robot.now()
    p = np.array(f.tcp[:3], float)
    R = _rpy2R(f.tcp[3:])
    for k in range(int(n)):
        _cmd(robot, p, R, -1.0, kp=1.0, vmax=0.3)
    return {'held': int(n)}
