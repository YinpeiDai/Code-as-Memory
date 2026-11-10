"""Instruction-video analysis for VideoPlaceButton (no imports)."""

OPEN_W = 0.032      # per-finger width above which the gripper is open
EMPTY_W = 0.008     # closed with nothing between the fingers (button press)


def _parse_goal(goal):
    """Return (colour, 'before'|'after') from the goal text."""
    g = (goal or '').lower()
    color = 'red'
    best = 10 ** 9
    for c in ('red', 'green', 'blue'):
        k = g.find(c + ' cube')
        if k < 0:
            k = g.find(c)
        if 0 <= k < best:
            best, color = k, c
    when = 'before' if 'before' in g else 'after'
    return color, when


def _finger_w(frame):
    f = getattr(frame, 'fingers', None)
    if f is None:
        return None
    f = np.asarray(f, float).ravel()
    if len(f) == 0 or not np.all(np.isfinite(f)):
        return None
    return float(f.mean())


def _robot_events(frames):
    """Placements and press from proprioception. Returns dict or None when unavailable."""
    ws, ps = [], []
    for fr in frames:
        w = _finger_w(fr)
        p = getattr(fr, 'tcp', None)
        if w is None or p is None:
            return None
        ws.append(w)
        ps.append(np.asarray(p, float)[:3])
    ws = np.asarray(ws)
    ps = np.asarray(ps)
    if len(ws) < 10 or ws.max() - ws.min() < 0.01:
        return None
    releases = []          # (index, xy) where a held cube is let go
    press = None
    holding = False
    empty_closed = False
    for i in range(len(ws)):
        w = ws[i]
        if EMPTY_W <= w < OPEN_W - 0.004 and ps[i][2] < 0.08:
            holding = True
        if w < EMPTY_W and press is None:
            empty_closed = True
        if empty_closed and press is None and w < EMPTY_W and ps[i][2] < 0.08:
            press = i
        if w >= OPEN_W:
            if holding:
                j = i
                while j > 0 and ws[j] >= OPEN_W - 0.004:
                    j -= 1
                releases.append((i, ps[j][:2].tolist(), float(ps[j][2])))
            holding = False
            empty_closed = False
    if press is None:
        return None
    return {'releases': releases, 'press': press}


def _vision_rests(cubes, min_run=4):
    """Rest locations (index, xy) of the cube on the table from per-frame detections.

    A rest is a run of at least min_run samples with the cube still (< 6 mm)."""
    rests = []
    run = []
    for i, c in enumerate(cubes):
        ok = c is not None and c[2] < 0.05
        if ok and run and np.hypot(c[0] - cubes[run[-1]][0], c[1] - cubes[run[-1]][1]) < 0.006:
            run.append(i)
        else:
            if len(run) >= min_run:
                rests.append(run)
            run = [i] if ok else []
    if len(run) >= min_run:
        rests.append(run)
    out = []
    for r in rests:
        xy = np.mean([cubes[k][:2] for k in r], 0)
        if out and np.hypot(*(xy - np.asarray(out[-1][1]))) < 0.03:
            continue
        out.append((r[0], xy.tolist()))
    return out


def _swap_windows(off_flags, max_gap=6, min_len=3):
    """Group frame indices with off-site purple blobs into windows [(start, end)]."""
    wins = []
    start, last = None, None
    for k, f in enumerate(off_flags):
        if f:
            if start is None:
                start = k
            elif k - last > max_gap:
                wins.append((start, last))
                start = k
            last = k
    if start is not None:
        wins.append((start, last))
    return [w for w in wins if w[1] - w[0] + 1 >= min_len]


def _analyse_video(frames, goal, stride=1):
    """Find the goal target from the instruction video, following target swaps.

    Sites are the ring centres in the first video frame. A swap shows up as purple
    blobs away from every site while two sites are vacant; those two sites exchange
    their target identities. Returns a dict with target_xy (final site of the goal
    target) and diagnostics."""
    color, when = _parse_goal(goal)
    frames = list(frames)
    if not frames:
        return None
    sel = list(range(0, len(frames), max(1, stride)))
    if sel[-1] != len(frames) - 1:
        sel.append(len(frames) - 1)
    calib = None
    for fr in reversed(frames):
        K, E = _cam(fr)
        if K is not None and E is not None:
            calib = (K, E)
            break
    if calib is None:
        return None
    p0 = _target_points(frames[0], 2, calib)
    sites = [_ring_fit(p0, s, 0.01)[0].tolist() for s in _detect_targets(frames[0], calib)]
    if not sites:
        return None
    ref, _ = _site_support(p0, sites)
    ref = np.maximum(ref, 10)
    off_flags, vac = [], []
    for k in sel:
        pts = _target_points(frames[k], 2, calib)
        sup, off = _site_support(pts, sites)
        off_flags.append(len(off) >= 8)
        vac.append(sup < 0.35 * ref)
    vac = np.asarray(vac)
    wins = _swap_windows(off_flags, 6, 3)
    ident_at_site = list(range(len(sites)))     # ident_at_site[s] = identity now at site s
    swaps = []
    for a, b in wins:
        frac = vac[a:b + 1].mean(0)
        order = [int(i) for i in np.argsort(-frac)]
        if len(order) >= 2 and frac[order[1]] > 0.25:
            s1, s2 = order[0], order[1]
            ident_at_site[s1], ident_at_site[s2] = ident_at_site[s2], ident_at_site[s1]
            swaps.append([sel[a], sel[b], s1, s2, [round(float(x), 2) for x in frac]])

    ev = _robot_events(frames)
    chosen_idx, chosen_xy, method = None, None, None
    if ev is not None:
        rel = ev['releases']
        before = [r for r in rel if r[0] < ev['press']]
        after = [r for r in rel if r[0] > ev['press']]
        r = before[-1] if (when == 'before' and before) else (after[0] if after else None)
        if r is not None:
            chosen_idx, chosen_xy, method = r[0], r[1], 'proprioception'
    if chosen_xy is None:
        cubes = []
        for k in sel:
            cubes.append(_detect_cube(frames[k], color, calib))
        steps = [getattr(fr, 'step', None) for fr in frames[:50]]
        gaps = [b - a for a, b in zip(steps[:-1], steps[1:])
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > a]
        native = float(np.median(gaps)) if gaps else 1.0
        rests = _vision_rests(cubes, max(3, int(round(20.0 / (max(1, stride) * max(1.0, native))))))
        S0 = np.asarray(sites)
        on_site = []
        for k, xy in rests[1:]:
            dd = np.hypot(S0[:, 0] - xy[0], S0[:, 1] - xy[1])
            j = int(np.argmin(dd))
            if dd[j] < 0.04 and (not on_site or on_site[-1][2] != j):
                on_site.append((k, xy, j))
        want = 0 if when == 'before' else 1
        if len(on_site) > want:
            chosen_idx, chosen_xy, method = sel[on_site[want][0]], on_site[want][1], 'vision'
    out = {'color': color, 'when': when, 'sites': sites, 'swaps': swaps, 'n_targets': len(sites)}
    if chosen_xy is None:
        out.update({'target_xy': None, 'method': 'none'})
        return out
    S = np.asarray(sites)
    d = np.hypot(S[:, 0] - chosen_xy[0], S[:, 1] - chosen_xy[1])
    site0 = int(np.argmin(d))
    # identity of the chosen target is the one at site0 at the time of placement (swaps happen later,
    # but apply only swaps that end after the placement)
    ident = list(range(len(sites)))
    for sw in swaps:
        if sw[1] < chosen_idx:
            ident[sw[2]], ident[sw[3]] = ident[sw[3]], ident[sw[2]]
    target_ident = ident[site0]
    final_site = None
    cur = list(range(len(sites)))
    for sw in swaps:
        cur[sw[2]], cur[sw[3]] = cur[sw[3]], cur[sw[2]]
    for s_i, idn in enumerate(cur):
        if idn == target_ident:
            final_site = s_i
    out.update({'target_xy': sites[final_site], 'place_xy': chosen_xy, 'site0': site0,
                'final_site': final_site, 'match_err': float(d[site0]), 'method': method})
    return out
