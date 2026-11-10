"""GatherAndRecall5 policy: move all five cubes onto the disc while watching the lamp, then press
the button whose colour matches the lamp flash.  Perception uses the fixed front RGB-D camera."""

N_CUBES = 5
SLOT_R = 0.05   # diagonal offset (m, per axis) of the four outer placement slots from the disc centre


def _flash_in_frames(frames, box):
    """Return (colour, step) of the first saturated lamp frame in `frames`, else (None, None)."""
    if not box:
        return None, None
    v0, v1, u0, u1 = [int(b) for b in box]
    for fr in frames:
        patch = np.asarray(fr.front_rgb)[v0:v1 + 1, u0:u1 + 1].astype(np.int32)
        if patch.size == 0:
            continue
        flat = patch.reshape(-1, 3)
        for ci, name in enumerate(COLORS):
            others = [k for k in range(3) if k != ci]
            m = (flat[:, ci] > 180) & (flat[:, others].max(1) < 90)
            if m.sum() >= max(8, 0.25 * len(flat) * 0.5):
                return name, int(fr.step)
    return None, None


def _slots(disc):
    cx, cy = disc[0], disc[1]
    s = SLOT_R
    return [[cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s], [cx - s, cy - s], [cx, cy]]


def _free_slot(disc, scene, used):
    """First placement slot (diagonals, then centre) not used by an earlier successful place and with no
    detected on-disc cube within 4.5 cm; falls back to ignoring `used`, then to the emptiest slot."""
    on = [c for c in scene["cubes"] if math.hypot(c["x"] - disc[0], c["y"] - disc[1]) <= disc[2]]
    slots = _slots(disc)
    used = used or []

    def clear(s):
        return min([math.hypot(c["x"] - s[0], c["y"] - s[1]) for c in on] or [1.0])

    for s in slots:
        if clear(s) > 0.045 and all(math.hypot(u[0] - s[0], u[1] - s[1]) > 0.02 for u in used):
            return s
    for s in slots:
        if clear(s) > 0.045:
            return s
    return max(slots, key=clear)


def _off_disc_cubes(scene, disc):
    out = []
    for c in scene["cubes"]:
        if disc is None or math.hypot(c["x"] - disc[0], c["y"] - disc[1]) > disc[2] + 0.005:
            out.append([round(c["x"], 4), round(c["y"], 4)])
    return out


def _on_disc_count(scene, disc):
    if disc is None:
        return 0
    return sum(1 for c in scene["cubes"] if math.hypot(c["x"] - disc[0], c["y"] - disc[1]) <= disc[2])


def perceive_decide(obs, goal, memory, last):
    """Perceive the scene/lamp, update episode memory and choose the next act."""
    mem = dict(memory)
    frame = obs.current
    hist = mem.get("history") or ""
    if mem.get("stage") is None:
        sc = _scene(frame)
        mem["stage"] = "gather"
        mem["disc"] = [round(v, 4) for v in sc["disc"]] if sc["disc"] else None
        mem["buttons"] = {k: [round(x, 4) for x in v] for k, v in sc["buttons"].items()}
        mem["lamp_box"] = sc["lamp"]
        mem["cubes"] = _off_disc_cubes(sc, mem["disc"])
        mem["placed"] = 0
        mem["flash_color"] = None
        mem["flash_step"] = None
        mem["press_tries"] = 0
        mem["fails"] = 0
        hist = "init: %d cubes, disc=%s, buttons=%s, lamp=%s;" % (
            len(mem["cubes"]), mem["disc"], sorted(mem["buttons"]), mem["lamp_box"])
    # lamp flash watch over every frame since the previous action
    if mem.get("flash_color") is None:
        col, st = _flash_in_frames(obs.frames, mem.get("lamp_box"))
        if col is not None:
            mem["flash_color"] = col
            mem["flash_step"] = st
            hist += " flash %s@%d;" % (col, st)
    disc = mem["disc"]
    # outcome of the previous pick/place
    res_ok = last is not None and '"placed": true' in last
    if last is not None and last.startswith("act_pick_place"):
        if res_ok:
            mem["placed"] = int(mem["placed"]) + 1
            tgt = mem.get("last_target")
            rem = mem.get("cubes") or []
            if tgt and rem:
                j = min(range(len(rem)), key=lambda i: math.hypot(rem[i][0] - tgt[0], rem[i][1] - tgt[1]))
                rem = rem[:j] + rem[j + 1:]
            mem["cubes"] = rem
            if mem.get("last_slot"):
                mem["slots_used"] = (mem.get("slots_used") or []) + [mem["last_slot"]]
        else:
            mem["fails"] = int(mem["fails"]) + 1
        hist += " pp:%s;" % ("ok" if res_ok else "fail")
    act = None
    if mem["stage"] == "gather":
        sc = _scene(frame, mem.get("buttons"))
        seen = _off_disc_cubes(sc, disc)
        if mem["placed"] >= N_CUBES and not seen:
            mem["stage"] = "press"
        else:
            cands = list(seen)
            expected = N_CUBES - int(mem["placed"])
            if len(cands) < expected:
                # possibly occluded by the arm: add remembered cubes not matched by a detection
                for r in (mem.get("cubes") or []):
                    if all(math.hypot(r[0] - s[0], r[1] - s[1]) > 0.03 for s in cands):
                        cands.append(r)
            if not cands:
                mem["stage"] = "press"
            else:
                tcp = np.asarray(frame.tcp)
                cands = sorted(cands, key=lambda c: math.hypot(c[0] - tcp[0], c[1] - tcp[1]))
                c = cands[0]
                slot = _free_slot(disc, sc, mem.get("slots_used"))
                mem["last_slot"] = [float(slot[0]), float(slot[1])]
                mem["last_target"] = [float(c[0]), float(c[1])]
                if seen:
                    # refresh remembered off-disc cubes with the current detections
                    mem["cubes"] = [list(s) for s in cands]
                act = {"tool": "act_pick_place",
                       "args": {"cube_x": float(c[0]), "cube_y": float(c[1]),
                                "place_x": float(slot[0]), "place_y": float(slot[1]),
                                "disc_z": float(disc[3])}}
    if act is None and mem["stage"] == "press":
        col = mem.get("flash_color")
        if col is None:
            col = "green"   # fallback guess; recorded in history
            hist += " no flash seen, guessing green;"
        b = mem["buttons"].get(col)
        if b is None:
            act = {"tool": "act_wait", "args": {"n": 5}}
        else:
            depth = 0.0 + 0.01 * int(mem["press_tries"])
            mem["press_tries"] = int(mem["press_tries"]) + 1
            act = {"tool": "act_press", "args": {"bx": float(b[0]), "by": float(b[1]), "bz": float(b[2]),
                                                 "extra_depth": float(depth)}}
    mem["history"] = hist[-1500:]
    return {"memory": {k: mem.get(k) for k in MEMORY_FIELDS}, "act": act}


MEMORY_FIELDS = ["stage", "disc", "buttons", "lamp_box", "cubes", "last_target", "last_slot", "slots_used", "placed", "flash_color", "flash_step",
                 "press_tries", "fails", "history"]


def act_pick_place(robot, cube_x, cube_y, place_x, place_y, disc_z, yaw=0.0):
    """Pick the cube centred at (cube_x, cube_y) on the table and release it at (place_x, place_y) on the disc
    whose top surface is at disc_z. Returns measured stage outcomes; 'placed' is true when the cube was held
    until release."""
    out = {"start_step": int(robot.now().step)}
    gx, gy = cube_x + GRASP_OFFSET[0], cube_y + GRASP_OFFSET[1]
    px, py = place_x + GRASP_OFFSET[0], place_y + GRASP_OFFSET[1]
    z0 = float(robot.now().tcp[2])
    if z0 < TRANSIT_Z - 0.03:
        here = robot.now().tcp
        out["rise"] = _move(robot, [here[0], here[1], TRANSIT_Z], yaw, 1.0, tol=0.02, vtol=1.0, max_steps=15)
    out["approach"] = _move(robot, [gx, gy, TRANSIT_Z], yaw, 1.0, tol=0.005, vtol=0.004, max_steps=60)
    out["descend"] = _move(robot, [gx, gy, GRASP_Z], yaw, 1.0, tol=0.005, vtol=0.004, max_steps=30)
    out["close"] = _gripper(robot, -1.0, yaw, n_min=4, n_max=10)
    out["lift"] = _move(robot, [gx, gy, TRANSIT_Z], yaw, -1.0, tol=0.02, vtol=1.0, max_steps=20)
    held = float(np.mean(robot.now().fingers))
    out["held_fingers"] = round(held, 4)
    if held < 0.006:
        out["placed"] = False
        out["reason"] = "grasp missed"
        # move clear of the front camera's view of the cube area before the next perception
        out["clear"] = _move(robot, [-0.05, 0.0, 0.18], yaw, 1.0, tol=0.02, vtol=0.01, max_steps=25)
        out["end_step"] = int(robot.now().step)
        return out
    out["transit"] = _move(robot, [px, py, TRANSIT_Z], yaw, -1.0, tol=0.006, vtol=0.005, max_steps=60)
    out["lower"] = _move(robot, [px, py, disc_z + RELEASE_DZ], yaw, -1.0, tol=0.006, vtol=0.005, max_steps=25)
    held2 = float(np.mean(robot.now().fingers))
    out["held_before_release"] = round(held2, 4)
    out["open"] = _gripper(robot, 1.0, yaw, n_min=3, n_max=8)
    out["retreat"] = _move(robot, [px, py, TRANSIT_Z], yaw, 1.0, tol=0.02, vtol=1.0, max_steps=15)
    out["placed"] = held2 > 0.006
    out["end_step"] = int(robot.now().step)
    return out


def act_press(robot, bx, by, bz, extra_depth=0.0):
    """Close the gripper, move above the button cap (bx, by, top bz) and push down onto it; then lift."""
    out = {"start_step": int(robot.now().step)}
    tx, ty = bx, by
    z0 = float(robot.now().tcp[2])
    if z0 < TRANSIT_Z - 0.03:
        here = robot.now().tcp
        out["rise"] = _move(robot, [here[0], here[1], TRANSIT_Z], 0.0, -1.0, tol=0.02, vtol=1.0, max_steps=15)
    out["approach"] = _move(robot, [tx, ty, bz + 0.06], 0.0, -1.0, tol=0.005, vtol=0.004, max_steps=60)
    out["fingers"] = round(float(np.mean(robot.now().fingers)), 4)
    out["push"] = _move(robot, [tx, ty, bz + 0.005 - extra_depth], 0.0, -1.0, tol=0.003, vtol=0.002,
                        max_steps=25, gain=0.6)
    out["hold"] = _move(robot, [tx, ty, bz + 0.005 - extra_depth], 0.0, -1.0, tol=0.0, vtol=0.0,
                        max_steps=10, gain=0.6)
    out["lift"] = _move(robot, [tx, ty, bz + 0.06], 0.0, -1.0, tol=0.01, vtol=1.0, max_steps=15)
    out["end_step"] = int(robot.now().step)
    return out


def act_wait(robot, n=5):
    """Hold still with the gripper closed for n steps."""
    for _ in range(int(n)):
        robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    return {"waited": int(n), "step": int(robot.now().step)}
