"""Decision entry point for VideoPlaceButton."""


def _match_exec_target(frame, xy):
    """Snap a tracked target location to the nearest target ring seen in the current frame."""
    if xy is None:
        return None
    try:
        pts = _target_points(frame, 1)
    except (ValueError, TypeError, IndexError):
        return xy
    dets = _cluster(pts, CLUSTER_R, 15) if len(pts) else []
    best, bd = None, 1e9
    for c, n in dets:
        d = float(np.hypot(c[0] - xy[0], c[1] - xy[1]))
        if d < bd:
            best, bd = c, d
    if best is None or bd > 0.05:
        return list(xy)
    c, _ = _ring_fit(pts, best, 0.01)
    return [float(c[0]), float(c[1])]


def perceive_decide(obs, goal, memory, last):
    """Watch the instruction video once (goal target, following hard-mode target swaps),
    then pick the goal-coloured cube and place it on that target; verify and retry."""
    mem = dict(memory)
    for k in ('target', 'analysed', 'attempts', 'stage', 'info', 'color'):
        mem.setdefault(k, None)
    cur = obs.current
    color, when = _parse_goal(goal)
    mem['color'] = color
    if not mem['analysed']:
        demo = [f for f in obs.frames if getattr(f, 'phase', '') == 'demo']
        res = None
        if len(demo) >= 10:
            stride = 1 if len(demo) <= 1500 else int(math.ceil(len(demo) / 1500.0))
            res = _analyse_video(demo + [cur], goal, stride)
        xy = None
        info = 'no video'
        if res is not None:
            xy = res.get('target_xy')
            info = 'method=%s when=%s site0=%s final=%s swaps=%s' % (
                res.get('method'), when, res.get('site0'), res.get('final_site'), res.get('swaps'))
        if xy is None:
            # no usable video evidence: fall back to the nearest visible target to the cube
            dets = _detect_targets(cur)
            cube = _detect_cube(cur, color)
            if dets:
                if cube is not None:
                    dets = sorted(dets, key=lambda t: (t[0] - cube[0]) ** 2 + (t[1] - cube[1]) ** 2)
                xy = dets[0]
                info += ' fallback-target'
        xy = _match_exec_target(cur, xy)
        mem['target'] = xy
        mem['analysed'] = True
        mem['attempts'] = 0
        mem['stage'] = 'place'
        mem['info'] = info
    tgt = mem['target']
    if tgt is None:
        return {'memory': mem, 'act': {'tool': 'act_wait', 'args': {'steps': 20}}}
    cube = _detect_cube(cur, color)
    on_target = False
    if cube is not None:
        on_target = float(np.hypot(cube[0] - CUBE_X_BIAS - tgt[0], cube[1] - tgt[1])) < 0.03
    attempts = int(mem['attempts'] or 0)
    if attempts > 0 and cube is None:
        # cube not visible after an attempt (occluded or held): do not disturb it blindly
        return {'memory': mem, 'act': {'tool': 'act_wait', 'args': {'steps': 20}}}
    if mem['stage'] == 'done' or (attempts > 0 and on_target):
        mem['stage'] = 'done'
        return {'memory': mem, 'act': {'tool': 'act_wait', 'args': {'steps': 20}}}
    if attempts >= 4:
        mem['stage'] = 'done'
        return {'memory': mem, 'act': {'tool': 'act_wait', 'args': {'steps': 20}}}
    mem['attempts'] = attempts + 1
    args = {'color': color, 'target_x': float(tgt[0]), 'target_y': float(tgt[1])}
    if cube is not None:
        args['cube_x'] = float(cube[0] - CUBE_X_BIAS)
        args['cube_y'] = float(cube[1])
    return {'memory': mem, 'act': {'tool': 'act_pick_place', 'args': args}}
