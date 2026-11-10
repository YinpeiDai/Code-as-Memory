"""Top-level decision logic for SwingXtimes."""

HOVER_Z = 0.15
CARRY_Z = 0.10


def _holding(frame):
    f = float(np.sum(np.asarray(frame.fingers, dtype=float)))
    return 0.012 < f < 0.065


def _scan_front(frame, color):
    w, ok = _world_points(frame.front_depth, frame.front_K, frame.front_extrinsic)
    rgb = frame.front_rgb
    cube = _find_cube(w, ok, rgb, color)
    fx = _find_fixtures(w, ok, rgb)
    return cube, fx


def _scan_wrist(frame, color, near):
    w, ok = _world_points(frame.wrist_depth, frame.wrist_K, frame.wrist_extrinsic)
    region = None if near is None else (near[0], near[1], 0.07)
    return _find_cube(w, ok, frame.wrist_rgb, color, region)


def _grip_yaw(yaw):
    return float((yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4)


def perceive_decide(obs, goal, memory, last):
    """SwingXtimes policy: parse cube colour and repetition count from the goal, locate the cube, the two
    targets and the button with front RGB-D, refine the cube with the wrist camera, grasp it, alternate
    right (-y) / left (+y) target visits `count` times, put the cube down on the left target, press the button."""
    m = dict(memory)
    for k in ('stage', 'color', 'count', 'cube', 'cube_yaw', 'target_r', 'target_l', 'button', 'legs', 'tries',
              'home', 'note'):
        if k not in m:
            m[k] = None
    f = obs.current
    if m['color'] is None or m['count'] is None:
        g = _parse_goal(goal)
        m['color'], m['count'] = g['color'], g['count']
    if m['legs'] is None:
        m['legs'] = 0
    if m['tries'] is None:
        m['tries'] = 0
    if m['home'] is None:
        m['home'] = [float(v) for v in np.asarray(f.tcp)[:3]]
    if m['stage'] is None:
        m['stage'] = 'scan'
    if getattr(f, 'phase', 'exec') == 'demo':
        return {'memory': m, 'act': {'tool': 'act_wait', 'args': {'steps': 1}}}

    st = m['stage']
    if st == 'scan':
        cube, fx = _scan_front(f, m['color'])
        ts = sorted(fx['targets'], key=lambda p: p[1])
        if len(ts) >= 2 and m['target_r'] is None:
            m['target_r'], m['target_l'] = ts[0], ts[-1]
        if fx['button'] is not None and m['button'] is None:
            m['button'] = fx['button']
        if cube is None:
            m['tries'] += 1
            m['note'] = 'cube not seen in front view'
            if m['tries'] > 4:
                m['stage'] = 'press'
            return {'memory': m, 'act': {'tool': 'act_wait', 'args': {'steps': 5}}}
        m['cube'] = cube['xy']
        m['cube_yaw'] = _grip_yaw(cube['yaw'])
        m['stage'] = 'refine'
        return {'memory': m, 'act': {'tool': 'act_move', 'args': {
            'x': m['cube'][0], 'y': m['cube'][1], 'z': HOVER_Z, 'yaw': m['cube_yaw'], 'gripper': 1.0,
            'lift_first': True}}}

    if st == 'refine':
        c = _scan_wrist(f, m['color'], m['cube'])
        if c is not None and c['n'] >= 30:
            m['cube'] = c['xy']
            m['cube_yaw'] = _grip_yaw(c['yaw'])
            m['note'] = 'wrist refined'
        m['stage'] = 'grasped'
        return {'memory': m, 'act': {'tool': 'act_grasp', 'args': {
            'x': m['cube'][0], 'y': m['cube'][1], 'yaw': m['cube_yaw']}}}

    if st in ('grasped', 'swing'):
        if not _holding(f):
            if st == 'swing' and m['legs'] > 0:
                m['legs'] -= 1
            m['tries'] += 1
            m['note'] = 'not holding cube'
            if m['tries'] > 4:
                m['stage'] = 'press'
                return {'memory': m, 'act': {'tool': 'act_move', 'args': {
                    'x': f.tcp[0], 'y': f.tcp[1], 'z': HOVER_Z, 'yaw': 0.0, 'gripper': 1.0}}}
            m['stage'] = 'scan'
            h = m['home']
            return {'memory': m, 'act': {'tool': 'act_move', 'args': {
                'x': h[0], 'y': h[1], 'z': h[2], 'yaw': 0.0, 'gripper': 1.0, 'lift_first': True}}}
        if m['target_r'] is None or m['target_l'] is None:
            m['stage'] = 'press'
            return {'memory': m, 'act': {'tool': 'act_wait', 'args': {'steps': 1}}}
        m['stage'] = 'swing'
        if m['legs'] < 2 * m['count']:
            t = m['target_r'] if m['legs'] % 2 == 0 else m['target_l']
            m['legs'] += 1
            return {'memory': m, 'act': {'tool': 'act_carry', 'args': {
                'x': t[0], 'y': t[1], 'z': CARRY_Z, 'yaw': m['cube_yaw']}}}
        m['stage'] = 'press'
        t = m['target_l']
        return {'memory': m, 'act': {'tool': 'act_place', 'args': {'x': t[0], 'y': t[1], 'yaw': m['cube_yaw']}}}

    if st == 'press':
        if m['button'] is None:
            m['note'] = 'button unknown'
            m['stage'] = 'done'
            return {'memory': m, 'act': {'tool': 'act_wait', 'args': {'steps': 5}}}
        m['stage'] = 'done'
        b = m['button']
        return {'memory': m, 'act': {'tool': 'act_press', 'args': {'x': b[0], 'y': b[1]}}}

    # done: press again once if the episode is still running, then idle
    if m['note'] != 'pressed twice' and m['button'] is not None:
        m['note'] = 'pressed twice'
        b = m['button']
        return {'memory': m, 'act': {'tool': 'act_press', 'args': {'x': b[0], 'y': b[1]}}}
    return {'memory': m, 'act': {'tool': 'act_wait', 'args': {'steps': 20}}}
