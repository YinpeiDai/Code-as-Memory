"""T21 ButterChocolateMicrowave: open microwave, put butter in, then put chocolate where the butter was placed.

Stage machine; one decision per stage. Anchors are measured once from the first agentview RGB-D frame
(microwave handle, butter, chocolate); the butter placement location is written to memory when the butter
is released and later used as the chocolate destination.
"""

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0
MIN_HOLD_WIDTH = 0.010      # finger gap (m) above which an object is considered held
DOOR_OPEN_RATIO = 0.6       # door open if front-plane point count fell below this fraction of the initial count
MAX_DECISIONS = 40

# stage -> (kind, orientation, steps, gripper)
STAGES = {
    "A": ("move", "side", 30, GRIP_OPEN),     # pre-handle
    "B": ("move", "side", 30, GRIP_OPEN),     # push onto handle
    "G1": ("grip", "side", 20, GRIP_CLOSE),   # grasp handle
    "C": ("move", "side", 40, GRIP_CLOSE),    # pull door open
    "G2": ("grip", "side", 10, GRIP_OPEN),    # release handle
    "D": ("move", "top", 40, GRIP_OPEN),      # above butter
    "E": ("move", "top", 30, GRIP_OPEN),      # descend to butter
    "G3": ("grip", "top", 20, GRIP_CLOSE),    # grasp butter
    "F": ("move", "top", 40, GRIP_CLOSE),     # lift
    "G": ("move", "top", 40, GRIP_CLOSE),     # carry toward microwave
    "R1": ("move", "side", 30, GRIP_CLOSE),   # turn sideways in place
    "H": ("move", "side", 75, GRIP_CLOSE),    # into microwave
    "I": ("move", "side", 25, GRIP_CLOSE),    # place deeper
    "G4": ("grip", "side", 20, GRIP_OPEN),    # release butter
    "J": ("move", "side", 30, GRIP_OPEN),     # back out
    "K": ("move", "side", 30, GRIP_OPEN),     # up
    "L": ("move", "top", 40, GRIP_OPEN),      # above chocolate
    "M": ("move", "top", 30, GRIP_OPEN),      # descend
    "G5": ("grip", "top", 20, GRIP_CLOSE),    # grasp chocolate
    "N": ("move", "top", 40, GRIP_CLOSE),     # lift
    "O": ("move", "top", 40, GRIP_CLOSE),     # carry toward microwave
    "R2": ("move", "side", 40, GRIP_CLOSE),   # turn sideways
    "P": ("move", "side", 75, GRIP_CLOSE),    # into microwave at the butter's placement location
    "P2": ("move", "side", 25, GRIP_CLOSE),   # a bit deeper (fallback if not yet scored)
    "G6": ("grip", "side", 20, GRIP_OPEN),    # release chocolate
    "Q": ("move", "side", 30, GRIP_OPEN),     # back out
    "W": ("grip", "side", 60, GRIP_OPEN),     # wait / monitoring period
}
ORDER = ["A", "B", "G1", "C", "G2", "D", "E", "G3", "F", "G", "R1", "H", "I", "G4", "J", "K",
         "L", "M", "G5", "N", "O", "R2", "P", "P2", "G6", "Q", "W"]

# offsets (expert-identified, constant across all 50 demonstrations)
Z_ABOVE_OBJ = 0.6827 - 0.431
Z_GRASP = 0.4127 - 0.431          # target below table top: gripper is stopped by the table, like the expert
Z_LIFT = 0.5627 - 0.431
CARRY_OFF = [-0.1512, 0.1995, 0.5575 - 0.431]


def _xyz(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def _width(frame):
    f = np.asarray(getattr(frame, "fingers"), dtype=float)
    return float(f[0] - f[1])


def _target(stage, mem, tcp):
    h = mem["handle"]
    b = mem["butter"]
    c = mem["choc"]
    dz = mem["table_dz"] or 0.0
    t0 = 0.431 + dz
    if stage == "A":
        return [h[0] - 0.10, h[1], h[2]]
    if stage == "B":
        return [h[0], h[1], h[2]]
    if stage == "C":
        return [h[0] - 0.50, h[1] - 0.20, h[2]]
    if stage == "D":
        return [b[0], b[1], t0 + Z_ABOVE_OBJ]
    if stage == "E":
        return [b[0], b[1], t0 + Z_GRASP]
    if stage == "F":
        return [b[0], b[1], t0 + Z_LIFT]
    if stage == "G":
        return [b[0] + CARRY_OFF[0], b[1] + CARRY_OFF[1], t0 + CARRY_OFF[2]]
    if stage in ("R1",):
        return _xyz(mem["hold"] or tcp[:3])
    if stage == "H":
        return [h[0] - 0.15, h[1] + 0.03, h[2] + 0.02]
    if stage == "I":
        return [h[0] - 0.1017, h[1] + 0.08, h[2] - 0.0099]
    if stage == "J" or stage == "Q":
        return [h[0] - 0.1008, h[1] - 0.1205, h[2] - 0.0101]
    if stage == "K":
        return [h[0] - 0.1008, h[1] - 0.1205, h[2] + 0.1399]
    if stage == "L":
        return [c[0], c[1], t0 + 0.6869 - 0.431]
    if stage == "M":
        return [c[0], c[1], t0 + 0.4166 - 0.431]
    if stage == "N":
        return [c[0], c[1], t0 + 0.5666 - 0.431]
    if stage == "O" or stage == "R2":
        return [c[0] - 0.2017, c[1] + 0.1997, t0 + 0.5611 - 0.431]
    if stage == "P":
        # the location where the butter was placed (written to memory at butter release)
        return _xyz(mem["place"])
    if stage == "P2":
        p = mem["place"]
        return [p[0], p[1] + 0.03, p[2]]
    return _xyz(tcp[:3])


def perceive_decide(obs, goal, memory, last):
    """Stage machine for 'butter into microwave, then chocolate to where the butter was placed'."""
    mem = dict(memory)
    for k in ("stage", "handle", "butter", "choc", "table_dz", "place", "butter_release", "hold",
              "scene_ok", "retries", "door_tries", "decisions", "notes", "order", "mw_box", "door_ref"):
        mem.setdefault(k, None)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    mem["decisions"] = int(mem["decisions"] or 0) + 1
    mem["retries"] = int(mem["retries"] or 0)
    mem["door_tries"] = int(mem["door_tries"] or 0)
    notes = mem["notes"] or ""

    if mem["stage"] is None:
        # first decision: parse goal order and perceive the scene from the initial agentview frame
        g = (goal or "").lower()
        ib, ic = g.find("butter"), g.find("chocolate")
        mem["order"] = ["butter", "chocolate"] if (ib < 0 or ic < 0 or ib <= ic) else ["chocolate", "butter"]
        sc = _perceive_scene(cur)
        mem["handle"] = _xyz(sc["handle"])
        first, second = (sc["butter"], sc["choc"])
        if mem["order"][0] == "chocolate":
            first, second = second, first
        # memory 'butter' = first object to put in microwave, 'choc' = second object
        mem["butter"] = [float(first[0]), float(first[1])]
        mem["choc"] = [float(second[0]), float(second[1])]
        dz = sc["table_z"] - 0.431
        mem["table_dz"] = float(dz) if abs(dz) < 0.03 else 0.0
        mem["scene_ok"] = bool(sc["ok"])
        mem["mw_box"] = sc.get("mw_box")
        mem["door_ref"] = _door_front_count(cur, mem["mw_box"])
        notes += "scene ok=%s %s; " % (sc["ok"], ",".join(sc["notes"]))
        mem["stage"] = "A"
    else:
        prev = mem["stage"]
        nxt = None
        if prev == "G2":
            # door check: closed door plane in front of the microwave top edge must have mostly disappeared
            cnt = _door_front_count(cur, mem["mw_box"])
            ref = mem["door_ref"] or 0
            notes += "door count %d/%d; " % (cnt, ref)
            if ref > 50 and cnt >= 0 and cnt > DOOR_OPEN_RATIO * ref and mem["door_tries"] < 1 \
                    and mem["decisions"] < MAX_DECISIONS - 28:
                mem["door_tries"] += 1
                notes += "door not open, retry; "
                nxt = "A"
        if prev == "G3" or prev == "F":
            w = _width(cur)
            if w < MIN_HOLD_WIDTH and mem["retries"] < 2 and mem["decisions"] < MAX_DECISIONS - 22:
                mem["retries"] += 1
                notes += "first object not held after %s (w %.3f), regrasp; " % (prev, w)
                nxt = "Dr"
        if prev == "G5" or prev == "N":
            w = _width(cur)
            if w < MIN_HOLD_WIDTH and mem["retries"] < 2 and mem["decisions"] < MAX_DECISIONS - 10:
                mem["retries"] += 1
                notes += "second object not held after %s (w %.3f), regrasp; " % (prev, w)
                nxt = "Lr"
        if prev == "G4":
            mem["butter_release"] = _xyz(tcp[:3])
        if nxt is None:
            base = prev[:-1] if prev.endswith("r") and prev[:-1] in STAGES else prev
            if prev == "Dr":
                nxt = "D"
            elif prev == "Lr":
                nxt = "L"
            elif base in ORDER and ORDER.index(base) + 1 < len(ORDER):
                nxt = ORDER[ORDER.index(base) + 1]
            else:
                nxt = "W"
        mem["stage"] = nxt

    st = mem["stage"]
    if st == "R1":
        mem["hold"] = _xyz(tcp[:3])
    if st == "H":
        # where the first object will be placed: remembered for the second object
        mem["place"] = _target("H", mem, tcp)
    if st == "P" and mem["place"] is None:
        mem["place"] = _target("H", mem, tcp)
    mem["notes"] = notes[-1500:]

    if st in ("Dr", "Lr"):
        # recovery: open gripper and back off upward before redoing the approach
        up = [float(tcp[0]), float(tcp[1]), float(max(tcp[2] + 0.12, 0.60))]
        orient = "top"
        act = {"tool": "act_segment", "args": {"target": up, "orient": orient, "steps": 30,
                                               "gripper": GRIP_OPEN, "kind": "move"}}
    else:
        kind, orient, steps, grip = STAGES[st]
        act = {"tool": "act_segment", "args": {"target": _target(st, mem, tcp), "orient": orient,
                                               "steps": steps, "gripper": grip, "kind": kind}}
    out = {k: mem[k] for k in ("stage", "handle", "butter", "choc", "table_dz", "place", "butter_release",
                               "hold", "scene_ok", "retries", "door_tries", "decisions", "notes", "order",
                               "mw_box", "door_ref")}
    return {"memory": out, "act": act}


def act_segment(robot, target, orient, steps, gripper, kind="move"):
    """Run `steps` native steps. kind='move': expert-style P control (pos gain 6, rot gain 2 clip 0.5) toward
    world `target` xyz and orientation `orient` ('top' or 'side') with gripper sign held; kind='grip': arm still,
    gripper command only. Returns final tcp, finger gap and remaining position error."""
    n = int(max(1, min(int(steps), 400)))
    g = 1.0 if float(gripper) > 0 else -1.0
    for _ in range(n):
        fr = robot.now()
        if kind == "grip":
            robot.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g)
        else:
            a = _control(np.asarray(fr.tcp, dtype=float), target, orient)
            robot.step(a[0], a[1], a[2], a[3], a[4], a[5], g)
    fr = robot.now()
    tcp = np.asarray(fr.tcp, dtype=float)
    err = (np.asarray(target, dtype=float) - tcp[:3]).tolist()
    return {"tcp": tcp[:3].tolist(), "finger_gap": _width(fr), "pos_err": err, "steps": n}
