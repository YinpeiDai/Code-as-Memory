"""GatherAndRecall3 policy: put the three cubes on the disc, remember the lamp flash colour,
then press the button of that colour."""

MAX_GRASP_TRIES = 3
FLASH_WAIT_LIMIT = 60     # native steps to wait for a flash after all cubes are placed


def _scan_flash(frames, box, mem):
    """Scan every frame since the last decision for a coloured lamp."""
    if box is None:
        return
    for f in frames:
        col, n = _lamp_color(f.front_rgb, box)
        if col is not None:
            if mem["flash_color"] is None:
                mem["flash_color"] = col
                mem["flash_step"] = int(f.step)
            if col == mem["flash_color"]:
                mem["flash_frames"] = int(mem["flash_frames"] or 0) + 1
            else:
                mem["notes"] = (mem["notes"] or "") + f" conflicting_flash:{col}@{int(f.step)}"


def _on_disc(p, disc, r):
    return math.hypot(p[0] - disc[0], p[1] - disc[1]) < r - 0.02


def perceive_decide(obs, goal, memory, last):
    """Perceive the scene (front camera), record any lamp flash in memory, and choose the next stage:
    pick-place the next cube onto a free disc slot, then press the button matching the flash colour."""
    mem = dict(memory)
    for k in ("stage", "cube_colors", "cube_pos", "disc", "disc_r", "button_colors", "button_pos",
              "lamp_box", "flash_color", "flash_step", "flash_frames", "placed", "tries",
              "press_tries", "wait_steps", "step", "notes"):
        mem.setdefault(k, None)
    cur = obs.current
    if mem["stage"] is None:
        s = _scene(obs.frames[0])
        cols = [c for c in COLORS if c in s["cubes"]]
        # pick order: closest to the robot base first (most negative x), as in demos
        cols.sort(key=lambda c: s["cubes"][c][0])
        mem["cube_colors"] = cols
        mem["cube_pos"] = [s["cubes"][c][:3] for c in cols]
        mem["disc"] = s["disc"]
        mem["disc_r"] = s["disc_r"]
        bc = [c for c in COLORS if c in s["buttons"]]
        mem["button_colors"] = bc
        mem["button_pos"] = [s["buttons"][c] for c in bc]
        mem["lamp_box"] = s["lamp_box"]
        mem["flash_frames"] = 0
        mem["placed"] = []
        mem["tries"] = 0
        mem["press_tries"] = 0
        mem["wait_steps"] = 0
        mem["notes"] = f"init cubes={cols} buttons={bc} disc={s['disc']} lamp={s['lamp_box']}"
        mem["stage"] = "gather"
    _scan_flash(obs.frames, mem["lamp_box"], mem)
    mem["step"] = int(cur.step)

    # current view: update cube positions / placed status
    s = _scene(cur)
    disc, dr = mem["disc"], mem["disc_r"] or 0.125
    on_disc = []
    for i, c in enumerate(mem["cube_colors"]):
        if c in s["cubes"]:
            p = s["cubes"][c]
            if disc is not None and _on_disc(p, disc, dr):
                on_disc.append(c)
            else:
                mem["cube_pos"][i] = p[:3]
    placed = list(mem["placed"] or [])
    if isinstance(last, str) and last.startswith("act_pick_place") and '"placed": true' in last:
        c = last.split('"color": "')[1].split('"')[0] if '"color": "' in last else None
        if c and c not in placed:
            placed.append(c)
    for c in on_disc:
        if c not in placed:
            placed.append(c)
    # a cube clearly visible off the disc is not placed, whatever the act reported
    for c in list(placed):
        if c in s["cubes"] and c not in on_disc and s["cubes"][c][3] >= 20:
            placed.remove(c)
            mem["notes"] = (mem["notes"] or "") + f" off_disc:{c}@{int(cur.step)}"
    mem["placed"] = placed

    if mem["stage"] == "gather":
        remaining = [c for c in mem["cube_colors"] if c not in placed]
        if remaining and int(mem["tries"]) < MAX_GRASP_TRIES * len(mem["cube_colors"]):
            c = remaining[0]
            i = mem["cube_colors"].index(c)
            p = mem["cube_pos"][i]
            # choose the first slot not occupied by a detected cube
            occupied = [s["cubes"][o][:2] for o in on_disc]
            slot = None
            for k in range(len(SLOTS)):
                q = [disc[0] + SLOTS[k][0], disc[1] + SLOTS[k][1]]
                if all(math.hypot(q[0] - o[0], q[1] - o[1]) > 0.06 for o in occupied):
                    slot = q
                    break
            if slot is None:
                slot = [disc[0], disc[1]]
            mem["tries"] = int(mem["tries"]) + 1
            grasp = [p[0] + GRASP_OFFSET[0], p[1] + GRASP_OFFSET[1]]
            return {"memory": mem, "act": {"tool": "act_pick_place",
                    "args": {"color": c, "grasp_xy": grasp, "place_xy": slot}}}
        mem["stage"] = "press"

    if mem["stage"] == "press":
        col = mem["flash_color"]
        if col is None and int(mem["wait_steps"]) < FLASH_WAIT_LIMIT:
            mem["wait_steps"] = int(mem["wait_steps"]) + 10
            return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 10}}}
        if col is None or col not in mem["button_colors"]:
            col = mem["button_colors"][1] if len(mem["button_colors"]) > 1 else mem["button_colors"][0]
            mem["notes"] = (mem["notes"] or "") + f" no_flash_guess:{col}"
        b = mem["button_pos"][mem["button_colors"].index(col)]
        k = int(mem["press_tries"])
        mem["press_tries"] = k + 1
        depth = [0.045, 0.035, 0.03, 0.03][min(k, 3)]
        return {"memory": mem, "act": {"tool": "act_press",
                "args": {"color": col, "button_xy": [b[0] + BUTTON_OFFSET[0], b[1] + BUTTON_OFFSET[1]],
                         "press_z": depth}}}
    return {"memory": mem, "act": {"tool": "act_wait", "args": {"steps": 5}}}


def act_pick_place(robot, color, grasp_xy, place_xy, grasp_z=0.022):
    """Pick the cube at grasp_xy (TCP target, world metres) and place it at place_xy on the disc.
    Returns per-phase step counts, finger widths and whether the cube was released at the slot."""
    out = {"color": color, "placed": False}
    gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    px, py = float(place_xy[0]), float(place_xy[1])
    s0 = int(robot.now().step)
    # approach above the cube (open gripper)
    r = _goto(robot, [gx, gy, 0.085], 1.0, tol=0.012, max_steps=40)
    if float(np.sum(robot.now().fingers)) < 0.07:
        out["reopen"] = _gripper(robot, 1.0, max_steps=10, min_steps=3)
    out["approach"] = r
    r = _goto(robot, [gx, gy, float(grasp_z)], 1.0, tol=0.004, max_steps=30, vmax=0.7, settle=0.003)
    out["descend"] = r
    out["tcp_grasp"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    g = _gripper(robot, -1.0, max_steps=10, min_steps=3)
    out["close"] = g
    if g["fingers"] < 0.01:
        _goto(robot, [gx, gy, 0.09], -1.0, tol=0.02, max_steps=15)
        out["error"] = "grasp_missed"
        out["steps"] = int(robot.now().step) - s0
        return out
    r = _goto(robot, [gx, gy, 0.07], -1.0, tol=0.02, max_steps=20)
    r = _goto(robot, [px, py, CARRY_Z], -1.0, tol=0.012, max_steps=50)
    out["transport"] = r
    held = float(np.sum(robot.now().fingers))
    out["fingers_transport"] = round(held, 4)
    if held < 0.01:
        out["error"] = "dropped"
        out["steps"] = int(robot.now().step) - s0
        return out
    r = _goto(robot, [px, py, RELEASE_Z], -1.0, tol=0.005, max_steps=30, vmax=0.7, settle=0.003)
    out["lower"] = r
    out["tcp_release"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    g = _gripper(robot, 1.0, max_steps=10, min_steps=3)
    out["open"] = g
    r = _goto(robot, [px, py, 0.09], 1.0, tol=0.02, max_steps=15)
    out["placed"] = True
    out["steps"] = int(robot.now().step) - s0
    out["end_step"] = int(robot.now().step)
    return out


def act_press(robot, color, button_xy, press_z=0.045):
    """Press the button at button_xy (TCP target) with a closed gripper down to press_z, hold, lift."""
    out = {"color": color}
    bx, by = float(button_xy[0]), float(button_xy[1])
    s0 = int(robot.now().step)
    out["above"] = _goto(robot, [bx, by, 0.12], -1.0, tol=0.008, max_steps=50)
    out["down"] = _goto(robot, [bx, by, press_z], -1.0, tol=0.004, max_steps=30, vmax=0.6)
    out["tcp_bottom"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    _goto(robot, [bx, by, press_z], -1.0, tol=0.0, max_steps=6, vmax=0.6)
    out["tcp_hold"] = [round(float(v), 4) for v in robot.now().tcp[:3]]
    out["up"] = _goto(robot, [bx, by, 0.12], -1.0, tol=0.015, max_steps=20)
    out["steps"] = int(robot.now().step) - s0
    return out


def act_wait(robot, steps=10):
    """Hold the current pose for `steps` native steps (gripper keeps its last open/closed state)."""
    f = robot.now()
    grip = 1.0 if float(np.sum(f.fingers)) > 0.06 else -1.0
    hold = np.asarray(f.tcp, dtype=np.float64)[:3]
    r = _goto(robot, hold, grip, tol=0.0, max_steps=int(steps))
    return {"waited": r["steps"], "step": int(robot.now().step)}
