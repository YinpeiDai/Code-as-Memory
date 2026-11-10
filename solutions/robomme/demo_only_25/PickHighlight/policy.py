"""Decision logic for PickHighlight: observe highlights, press button, pick each highlighted cube."""

RING_MIN = 40          # white table pixels in the annulus around a cube that mark a highlight
RING_COVER = 8         # of 12 angular sectors around the cube containing white table pixels
MATCH_R = 0.025        # metres; same cube across frames
MAX_OBSERVE = 4
MAX_TRIES = 3


def _scan_frames(frames, min_step=10, max_frames=6):
    """Cluster wrist-camera cube detections over frames; return clusters with highlight votes."""
    use = [f for f in frames if f.phase == 'exec' and f.step >= min_step and getattr(f, 'wrist_depth', None) is not None]
    if len(use) > max_frames:
        idx = np.linspace(0, len(use) - 1, max_frames).round().astype(int)
        use = [use[i] for i in sorted(set(idx.tolist()))]
    clusters = []
    buttons = []
    for f in use:
        cubes = _detect_cubes(f.wrist_rgb, f.wrist_depth, f.wrist_K, f.wrist_extrinsic)
        b = _detect_button(f.wrist_rgb, f.wrist_depth, f.wrist_K, f.wrist_extrinsic)
        if b is not None:
            buttons.append(b)
        for c in cubes:
            hit = None
            for k in clusters:
                if k['color'] == c['color'] and math.hypot(k['x'] - c['xyz'][0], k['y'] - c['xyz'][1]) < MATCH_R:
                    hit = k
                    break
            if hit is None:
                hit = {'color': c['color'], 'x': c['xyz'][0], 'y': c['xyz'][1], 'n': 0, 'hl': 0}
                clusters.append(hit)
            hit['n'] += 1
            hit['hl'] += 1 if (c['ring'] >= RING_MIN and c['cover'] >= RING_COVER) else 0
    button = None
    if buttons:
        button = np.median(np.array(buttons), 0).tolist()
    return clusters, button, len(use)


def perceive_decide(obs, goal, memory, last):
    """State machine: hold & watch highlights -> press button -> pick (and replace) each highlighted cube."""
    m = dict(memory)
    stage = m.get('stage') or 'start'
    targets = m.get('targets') or []
    colors = m.get('target_colors') or []
    done = m.get('done') or 0
    tries = m.get('tries') or 0
    nobs = m.get('observe_count') or 0
    button = m.get('button')
    pressed = bool(m.get('pressed'))

    def out(tool, args):
        return {"memory": {"stage": stage, "targets": targets, "target_colors": colors, "done": done,
                           "tries": tries, "observe_count": nobs, "button": button, "pressed": pressed},
                "act": {"tool": tool, "args": args}}

    if stage == 'start':
        stage = 'observe'
        nobs = 1
        return out('act_hold', {"steps": 28})

    if stage == 'observe' or stage == 'pressing':
        clusters, b, nused = _scan_frames(obs.frames)
        if b is not None and button is None:
            button = b
        hl = [k for k in clusters if k['hl'] * 2 >= max(k['n'], 1) and k['hl'] >= 1]
        if hl and not targets:
            hl = sorted(hl, key=lambda k: math.hypot(k['x'] - (button[0] if button else 0), k['y'] - (button[1] if button else 0)))
            targets = [[float(k['x']), float(k['y']), 0.02] for k in hl]
            colors = [k['color'] for k in hl]
        if stage == 'pressing':
            pressed = True
        if not pressed:
            if not targets and nobs < MAX_OBSERVE:
                nobs += 1
                return out('act_hold', {"steps": 15})
            if button is None and nobs < MAX_OBSERVE + 2:
                nobs += 1
                return out('act_hold', {"steps": 5})
            if button is not None:
                stage = 'pressing'
                return out('act_press_button', {"x": float(button[0]), "y": float(button[1])})
            pressed = True
        if not targets and nobs < MAX_OBSERVE + 4:
            # highlights may follow the press; watch from the start pose
            stage = 'observe'
            nobs += 1
            return out('act_hold', {"steps": 15})
        stage = 'pick'
        done = 0
        tries = 0

    if stage == 'pick':
        res = _parse_last(last)
        if res is not None and done < len(targets):
            if res.get('x') is not None and res.get('y') is not None:
                targets[done][0] = float(res['x'])
                targets[done][1] = float(res['y'])
            if res.get('held'):
                done += 1
                tries = 0
            else:
                tries += 1
                if tries >= MAX_TRIES:
                    done += 1
                    tries = 0
        if done < len(targets):
            t = targets[done]
            last_one = done == len(targets) - 1
            return out('act_pick', {"x": float(t[0]), "y": float(t[1]), "color": colors[done],
                                    "place_back": not last_one})
        stage = 'finished'
    return out('act_hold', {"steps": 20})


def _parse_last(last):
    """Extract held/x/y from the previous act_pick outcome (dict or JSON text); None otherwise."""
    if isinstance(last, dict):
        return last if 'held' in last else None
    if not isinstance(last, str) or '"held"' not in last:
        return None
    res = {'held': '"held": true' in last}
    for key in ('x', 'y'):
        tag = '"' + key + '": '
        if tag in last:
            txt = last.split(tag)[1]
            num = ''
            for ch in txt:
                if ch in '0123456789.-eE+':
                    num += ch
                else:
                    break
            try:
                res[key] = float(num)
            except ValueError:
                res[key] = None
    return res
