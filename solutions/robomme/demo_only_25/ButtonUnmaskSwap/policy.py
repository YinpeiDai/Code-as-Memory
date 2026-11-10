"""Decision logic for ButtonUnmaskSwap.

Scene: three coloured cubes are visible at the start, white containers (3 or 4; extras are empty) drop
over them, then containers swap places while the robot presses the two buttons. The policy tracks every
container through all observed front RGB-D frames, keeps the colour label of the cube it covered, presses
both buttons, waits for the containers to settle and then picks the containers hiding the goal colours
in goal order (non-final picks are put back down).
"""

TOP_BIAS = (-0.006, 0.001)     # expert grasp TCP minus detected container-top centre (mean over 10 episodes)
BUTTON_BIAS_X = -0.023         # expert press TCP minus visible button-blob centroid
BUTTON_BIAS_Y_POS = 0.005      # for the +y button
BUTTON_BIAS_Y_NEG = -0.0013    # for the -y button
SETTLE_FRAMES = 215            # swaps observed only before recording frame ~211
STATIC_FRAMES = 25


def _parse_goal(goal):
    words = str(goal).lower().replace(",", " ").split()
    out = []
    for i, w in enumerate(words):
        if w == "hiding" and i + 3 < len(words) and words[i + 3].startswith("cube"):
            c = words[i + 2]
            if c in COLORS:
                out.append(c)
    if not out:
        for w in words:
            if w in COLORS and w not in out:
                out.append(w)
    return out


def _empty_memory():
    return {"stage": "start", "targets": [], "target_index": 0, "cube_colors": [], "cube_positions": [],
            "containers": [], "container_labels": [], "buttons": [], "button_index": 0, "frames_seen": 0,
            "static_frames": 0, "pick_attempts": 0, "last_pick": [], "tracking": True}


def _label_containers(C, colors, cpos):
    labels = ["none"] * len(C)
    used = set()
    pairs = []
    for i, c in enumerate(C):
        for j, p in enumerate(cpos):
            pairs.append((math.hypot(c[0] - p[0], c[1] - p[1]), i, j))
    pairs.sort()
    for d, i, j in pairs:
        if d < 0.06 and labels[i] == "none" and j not in used:
            labels[i] = colors[j]
            used.add(j)
    return labels


def _update_tracking(frames, mem):
    """Consume new frames: record cubes, initialise containers once they appear, then track them."""
    for f in frames:
        if getattr(f, "phase", "exec") == "demo":
            continue
        mem["frames_seen"] += 1
        if not mem["tracking"]:
            continue
        P = _container_points(f)
        if not mem["containers"]:
            cubes = _cubes(f)
            if len(cubes) >= len(mem["cube_colors"]) and len(cubes) > 0:
                mem["cube_colors"] = [c for c in COLORS if c in cubes]
                mem["cube_positions"] = [cubes[c][:2] for c in mem["cube_colors"]]
            comps = _components(P)
            big = [c for c in comps if c[2] > 200]
            if len(big) >= 3 and len(big) == len(comps) and mem["cube_colors"] and len(cubes) == 0:
                C = [[c[0], c[1]] for c in big]
                mem["containers"] = C
                mem["container_labels"] = _label_containers(C, mem["cube_colors"], mem["cube_positions"])
                mem["static_frames"] = 0
            continue
        C0 = np.array(mem["containers"], dtype=np.float64)
        C1 = _kmeans_update(P, C0)
        moved = float(np.abs(C1 - C0).max()) if len(C0) else 0.0
        mem["containers"] = [[float(a), float(b)] for a, b in C1]
        if moved > 0.0015:
            mem["static_frames"] = 0
        else:
            mem["static_frames"] += 1


def _yaw_choice(a):
    a = a % (math.pi / 2)
    if a > math.pi / 4:
        a -= math.pi / 2
    return a


def _target_pose(f, mem, color):
    if color not in mem["container_labels"]:
        return None
    c = mem["containers"][mem["container_labels"].index(color)]
    top = _container_top(f, c)
    if top is not None and math.hypot(top["x"] - c[0], top["y"] - c[1]) < 0.035 and top["n"] >= 15:
        return [top["x"] + TOP_BIAS[0], top["y"] + TOP_BIAS[1], _yaw_choice(top["yaw"])]
    # fall back on the tracked visible-face centroid (biased ~1.5 cm towards the camera)
    return [c[0] - 0.015, c[1], 0.0]


def perceive_decide(obs, goal, memory, last):
    """Track containers in every new frame, then choose the next act: press buttons, hold until the
    swaps end, pick the goal containers in order (putting non-final ones back)."""
    mem = _empty_memory()
    for k in mem:
        if memory.get(k) is not None:
            mem[k] = memory[k]
    if not mem["targets"]:
        mem["targets"] = _parse_goal(goal)
    _update_tracking(obs.frames, mem)
    f = obs.current
    stage = mem["stage"]
    act = {"tool": "act_hold", "args": {"steps": 10}}

    if stage == "start":
        if not mem["buttons"]:
            bs = [b for b in _buttons(f) if b[3] >= 60]
            bs = bs[:2]
            out = []
            for b in bs:
                dy = BUTTON_BIAS_Y_POS if b[1] > 0 else BUTTON_BIAS_Y_NEG
                out.append([b[0] + BUTTON_BIAS_X, b[1] + dy])
            mem["buttons"] = out
        stage = "press"

    if stage == "press":
        if mem["button_index"] < len(mem["buttons"]):
            b = mem["buttons"][mem["button_index"]]
            mem["button_index"] += 1
            act = {"tool": "act_press_button", "args": {"x": b[0], "y": b[1]}}
        else:
            stage = "wait"

    if stage == "wait" and act["tool"] == "act_hold":
        ready = (mem["frames_seen"] >= SETTLE_FRAMES and mem["static_frames"] >= STATIC_FRAMES
                 and len(mem["containers"]) > 0)
        if ready:
            mem["tracking"] = False
            stage = "pick"
        else:
            act = {"tool": "act_hold", "args": {"steps": 15}}

    if stage == "picked" and last is not None:
        txt = str(last).lower().replace("'", '"')
        held = '"held": true' in txt
        idx = mem["target_index"]
        if not held:
            if '"held"' in txt:
                mem["pick_attempts"] += 1
            stage = "pick"
        elif idx + 1 < len(mem["targets"]):
            p = mem["last_pick"]
            act = {"tool": "act_place", "args": {"x": p[0], "y": p[1], "yaw": p[2]}}
            mem["target_index"] = idx + 1
            mem["pick_attempts"] = 0
            stage = "placed"
        else:
            stage = "done"
    elif stage == "placed":
        stage = "pick"

    if stage == "pick":
        idx = mem["target_index"]
        if idx >= len(mem["targets"]) or mem["pick_attempts"] >= 4:
            stage = "done"
        else:
            pose = _target_pose(f, mem, mem["targets"][idx])
            if pose is None:
                stage = "done"
            else:
                mem["last_pick"] = pose
                act = {"tool": "act_pick", "args": {"x": pose[0], "y": pose[1], "yaw": pose[2]}}
                stage = "picked"

    if stage == "done":
        act = {"tool": "act_hold", "args": {"steps": 30}}
    mem["stage"] = stage
    return {"memory": mem, "act": act}
