"""Shape knowledge for RememberShape9: canonical-pose feature prototypes of the
nine shapes (medians measured on assigned demonstrations), a per-object cost,
a one-to-one assignment of detected objects to shapes, and target selection."""

SHAPE_PROTO = {
    'C': {'hmax': 0.0066, 'area': 15.28, 'rmax': 3.3258, 'circ': 0.44, 'solid': 0.7466, 'f2': 0.0844, 'f3': 0.0494, 'f4': 0.0075, 'f5': 0.0174, 'top': 0.974, 'hrel': 0.9386, 'ext_major': 6.0849, 'ext_minor': 4.1335, 'hole': 0.0},
    'T': {'hmax': 0.0159, 'area': 10.32, 'rmax': 2.6549, 'circ': 0.467, 'solid': 0.7548, 'f2': 0.0584, 'f3': 0.2034, 'f4': 0.0355, 'f5': 0.0206, 'top': 0.9498, 'hrel': 0.9562, 'ext_major': 4.2912, 'ext_minor': 4.2307, 'hole': 0.0},
    'cross': {'hmax': 0.0159, 'area': 16.84, 'rmax': 3.2339, 'circ': 0.5104, 'solid': 0.6251, 'f2': 0.0146, 'f3': 0.0159, 'f4': 0.2321, 'f5': 0.0136, 'top': 0.9431, 'hrel': 0.9504, 'ext_major': 6.2213, 'ext_minor': 6.1139, 'hole': 0.0},
    'cube': {'hmax': 0.0409, 'area': 16.8, 'rmax': 2.9362, 'circ': 0.6222, 'solid': 0.9675, 'f2': 0.0138, 'f3': 0.006, 'f4': 0.0749, 'f5': 0.005, 'top': 0.9697, 'hrel': 0.9772, 'ext_major': 5.2365, 'ext_minor': 5.089, 'hole': 0.0},
    'cylinder': {'hmax': 0.0409, 'area': 13.16, 'rmax': 2.2248, 'circ': 0.8412, 'solid': 0.9309, 'f2': 0.01, 'f3': 0.0059, 'f4': 0.0067, 'f5': 0.0064, 'top': 0.9626, 'hrel': 0.9717, 'ext_major': 4.3001, 'ext_minor': 4.1848, 'hole': 0.0},
    'pyramid': {'hmax': 0.0307, 'area': 15.48, 'rmax': 2.9239, 'circ': 0.5867, 'solid': 0.9143, 'f2': 0.0165, 'f3': 0.0118, 'f4': 0.0745, 'f5': 0.0142, 'top': 0.2297, 'hrel': 0.5977, 'ext_major': 5.3955, 'ext_minor': 5.0045, 'hole': 0.0},
    'sphere': {'hmax': 0.0398, 'area': 11.68, 'rmax': 2.1833, 'circ': 0.7735, 'solid': 0.9083, 'f2': 0.0235, 'f3': 0.0097, 'f4': 0.008, 'f5': 0.0078, 'top': 0.7577, 'hrel': 0.87, 'ext_major': 4.1694, 'ext_minor': 3.862, 'hole': 0.0},
    'star': {'hmax': 0.0159, 'area': 19.76, 'rmax': 3.2484, 'circ': 0.5956, 'solid': 0.6809, 'f2': 0.0085, 'f3': 0.0072, 'f4': 0.0096, 'f5': 0.156, 'top': 0.951, 'hrel': 0.9552, 'ext_major': 6.219, 'ext_minor': 6.0769, 'hole': 0.0},
    'torus': {'hmax': 0.0162, 'area': 21.36, 'rmax': 3.0362, 'circ': 0.7357, 'solid': 0.835, 'f2': 0.0062, 'f3': 0.0053, 'f4': 0.005, 'f5': 0.0046, 'top': 0.9577, 'hrel': 0.9599, 'ext_major': 5.8091, 'ext_minor': 5.7302, 'hole': 0.9},
}

# feature scale used in the squared-distance cost (roughly 2-3x the observed spread)
SHAPE_SCALE = {
    "hmax": 0.003, "area": 2.0, "rmax": 0.25, "circ": 0.08, "solid": 0.06,
    "f2": 0.05, "f3": 0.04, "f4": 0.04, "f5": 0.035, "top": 0.08, "hrel": 0.04,
    "hole": 0.3,
}
TERM_CAP = 9.0          # robust cap per feature term
UNASSIGNED_COST = 45.0  # cost of leaving a detected object without a shape
CONFIDENT_COST = 12.0   # assignments above this cost are re-examined during elimination
ELIM_UNMATCHED = 8.0    # orientation-free cost of an object left unmatched in elimination


def _shape_cost(f, cls):
    p = SHAPE_PROTO[cls]
    c = 0.0
    for k, s in SHAPE_SCALE.items():
        v = min(f["hole"], 60) / 60.0 if k == "hole" else f[k]
        t = ((v - p[k]) / s) ** 2
        c += min(t, TERM_CAP)
    return c


def _pose_free_cost(f, cls):
    """Loose orientation-free comparison (largest and smallest object dimension, cm)."""
    p = SHAPE_PROTO[cls]
    obs = sorted([f["ext_major"], f["ext_minor"], f["hmax"] * 100.0])
    ref = sorted([p["ext_major"], p["ext_minor"], p["hmax"] * 100.0])
    return abs(obs[2] - ref[2]) / 0.6 + abs(obs[0] - ref[0]) / 0.5


def _classify_cue(f):
    costs = {c: _shape_cost(f, c) for c in SHAPE_PROTO}
    best = min(costs, key=costs.get)
    return best, costs


def _assign(objs):
    """Minimum-cost one-to-one assignment of objects to shapes (DP over class subsets).
    Returns a list of class names or None (unassigned) and the cost matrix."""
    classes = sorted(SHAPE_PROTO)
    nc = len(classes)
    C = [[_shape_cost(o, c) for c in classes] for o in objs]
    INF = 1e18
    # dp[mask] = (cost, choices)
    dp = {0: (0.0, [])}
    for i in range(len(objs)):
        nd = {}
        for mask, (cost, ch) in dp.items():
            cand = [(cost + UNASSIGNED_COST, mask, ch + [None])]
            for j in range(nc):
                if not mask & (1 << j):
                    cand.append((cost + C[i][j], mask | (1 << j), ch + [classes[j]]))
            for cst, m, chs in cand:
                if m not in nd or nd[m][0] > cst:
                    nd[m] = (cst, chs)
        dp = nd
    best = min(dp.values(), key=lambda t: t[0])
    return best[1], C, classes


def _select_target(cue_feat, objs):
    """Pick the arc object with the cue's shape. Returns (index, info dict)."""
    cue_cls, cue_costs = _classify_cue(cue_feat)
    labels, C, classes = _assign(objs)
    info = {"cue_class": cue_cls, "cue_cost": round(cue_costs[cue_cls], 2), "labels": labels}
    if cue_cls in labels:
        i = labels.index(cue_cls)
        info["how"] = "assigned"
        info["cost"] = round(C[i][classes.index(cue_cls)], 2)
        return i, info
    # elimination: the cue's shape was not found in canonical pose. Jointly re-assign the poorly
    # explained objects (unassigned or cost > CONFIDENT_COST) to the shapes they left over,
    # using the orientation-free size comparison (tilted objects).
    probs = [i for i, l in enumerate(labels)
             if l is None or C[i][classes.index(l)] > CONFIDENT_COST]
    if not probs:
        probs = list(range(len(objs)))
    taken = set(labels[i] for i in range(len(objs)) if i not in probs and labels[i] is not None)
    shapes = [c for c in classes if c not in taken]
    best = None
    if len(probs) > 5:   # keep the joint search small; fall back to the per-object comparison
        i = min(probs, key=lambda k: _pose_free_cost(objs[k], cue_cls))
        info["how"] = "elimination"
        info["cost"] = round(_pose_free_cost(objs[i], cue_cls), 2)
        info["problem_objects"] = probs
        return i, info
    for i in probs:
        others = [k for k in probs if k != i]
        rest = [c for c in shapes if c != cue_cls]
        cost = _pose_free_cost(objs[i], cue_cls) + _min_match(others, rest, objs)
        if best is None or cost < best[0]:
            best = (cost, i)
    i = best[1]
    info["how"] = "elimination"
    info["cost"] = round(_pose_free_cost(objs[i], cue_cls), 2)
    info["problem_objects"] = probs
    return i, info


def _min_match(obj_ids, shapes, objs):
    """Minimum total orientation-free cost matching objects to distinct shapes (objects may stay
    unmatched at ELIM_UNMATCHED each when shapes run out). Small sets only."""
    if not obj_ids:
        return 0.0
    first, rest = obj_ids[0], obj_ids[1:]
    best = ELIM_UNMATCHED + _min_match(rest, shapes, objs)
    for c in shapes:
        v = _pose_free_cost(objs[first], c) + _min_match(rest, [s for s in shapes if s != c], objs)
        if v < best:
            best = v
    return best


def _forced_costs(objs, cls):
    """For each object k: minimum total assignment cost when only object k may take `cls`
    and some object must take it (used to rank fallback candidates)."""
    classes = sorted(SHAPE_PROTO)
    nc = len(classes)
    j = classes.index(cls)
    C = [[_shape_cost(o, c) for c in classes] for o in objs]
    out = []
    for k in range(len(objs)):
        dp = {0: 0.0}
        for i in range(len(objs)):
            nd = {}
            for mask, cost in dp.items():
                opts = [(cost + UNASSIGNED_COST, mask)]
                for jj in range(nc):
                    if mask & (1 << jj):
                        continue
                    if jj == j and i != k:
                        continue
                    opts.append((cost + C[i][jj], mask | (1 << jj)))
                for c2, m2 in opts:
                    if m2 not in nd or nd[m2] > c2:
                        nd[m2] = c2
            dp = nd
        out.append(min(v for m, v in dp.items() if m & (1 << j)))
    return out


def _rank_candidates(cue_feat, objs, first):
    """Selected target first, then the remaining objects by forced-assignment cost."""
    cls, _ = _classify_cue(cue_feat)
    fc = _forced_costs(objs, cls)
    rest = sorted([k for k in range(len(objs)) if k != first], key=lambda k: fc[k])
    best = fc[first]
    return [first] + rest, [round(fc[k] - best, 1) for k in [first] + rest]
