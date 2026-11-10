"""RememberColor5: remember the cue cube colour, wait, then touch the cube of that colour.

Perception uses the fixed front RGB-D camera: pixels above the table (world z 12-70 mm) with strong
colour saturation are clustered in world xy; each cluster is a cube with a top-face colour signature.
Phases are inferred from what is visible: one cube = cue, none = delay, >=3 cubes = selection.
"""

PRESS_DEPTH = 0.004  # fingertip goal below the cube top (gentle: R009 knocked the cube away at 0.033 m/step)
TIP = 0.009          # closed fingertip extent below the TCP
CLEAR_Z = 0.015      # fingertip clearance above cube tops while the TCP is not yet over the target
XY_DESCEND = 0.010   # xy error (m) below which the final descent onto the top face starts
XY_GAIN = 1.2        # proportional gains on the normalized command (1.0 = target the goal exactly)
Z_GAIN = 1.5
FLOOR_SLOPE = 0.5    # hover-floor rise per metre of remaining xy error
PRESS_PUSH = 0.05     # normalized downward command while resting on the cube (5 mm target offset)


def _world_points(depth, K, E):
    z = depth.astype(np.float64) / 1000.0
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    E = np.asarray(E, dtype=np.float64)
    return (pc - E[:, 3]) @ E[:, :3]


def _detect_cubes(frame):
    """Coloured cubes seen by the front camera: list of dicts with xy, ztop, rgb, n."""
    rgb = np.asarray(frame.front_rgb).astype(np.float64)
    depth = np.asarray(frame.front_depth)
    pw = _world_points(depth, np.asarray(frame.front_K, dtype=np.float64), frame.front_extrinsic)
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    m = (depth > 0) & (pw[..., 2] > 0.012) & (pw[..., 2] < 0.07)
    m = m & (np.abs(pw[..., 1]) < 0.6) & (pw[..., 0] > -0.6) & (pw[..., 0] < 0.4)
    m = m & (sat > 0.6) & (mx >= 70)
    P = pw[m]
    C = rgb[m]
    out = []
    left = np.ones(len(P), dtype=bool)
    for _ in range(12):
        idx = np.flatnonzero(left)
        if len(idx) == 0:
            break
        c = P[idx[0], :2]
        for _k in range(3):
            d = np.hypot(P[idx, 0] - c[0], P[idx, 1] - c[1])
            mem = idx[d < 0.03]
            c = np.median(P[mem, :2], 0)
        d = np.hypot(P[idx, 0] - c[0], P[idx, 1] - c[1])
        mem = idx[d < 0.035]
        if len(mem) == 0:
            mem = idx[:1]
        left[mem] = False
        if len(mem) < 8:
            continue
        Q = P[mem]
        zt = Q[:, 2].max()
        top = Q[:, 2] > zt - 0.006
        col = np.median(C[mem][top], 0)
        out.append({"xy": [float(v) for v in np.median(Q[top, :2], 0)],
                    "ztop": float(np.median(Q[top, 2])), "rgb": [float(v) for v in col],
                    "n": int(len(mem))})
    return out


def _chroma(rgb):
    a = np.asarray(rgb, dtype=np.float64)
    return a / max(a.max(), 1.0)


def _color_name(rgb):
    ch = _chroma(rgb)
    bits = [int(v > 0.5) for v in ch]
    names = {(1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue", (1, 1, 0): "yellow",
             (1, 0, 1): "magenta", (0, 1, 1): "cyan"}
    return names.get(tuple(bits), "rgb%d_%d_%d" % tuple(int(v) for v in rgb))


def _match(cue_rgb, cubes):
    """Index of the cube whose top colour best matches the cue, plus best and runner-up distances."""
    cc = _chroma(cue_rgb)
    ds = [float(np.linalg.norm(_chroma(c["rgb"]) - cc) + 0.002 * abs(max(c["rgb"]) - max(cue_rgb))) for c in cubes]
    order = np.argsort(ds)
    second = ds[order[1]] if len(ds) > 1 else 9.0
    return int(order[0]), ds[order[0]], second


def _fresh():
    return {"stage": "observe", "cue_rgb": [], "cue_color": "", "cue_step": -1, "cue_frames": 0,
            "sel_step": -1, "target": [], "target_n": 0, "retarget": 0, "match": "", "last_z": None, "grip": 0, "steps": 0, "events": ""}


def perceive_decide(obs, goal, memory, last):
    """Track cue / delay / selection phases from the front camera, remember the cue colour,
    lock the matching cube position and drive the TCP onto it (one native step per decision)."""
    mem = _fresh()
    for k in mem:
        if k in memory and memory[k] is not None:
            mem[k] = memory[k]
    cur = obs.current
    mem["steps"] = int(cur.step)
    ev = []
    # ---- perception: process every frame since the last action (cue may be brief)
    if not mem["target"]:
        for fr in obs.frames:
            cubes = _detect_cubes(fr)
            if len(cubes) == 1 and mem["sel_step"] < 0:
                c = cubes[0]
                if mem["cue_frames"] == 0:
                    mem["cue_rgb"] = c["rgb"]
                    mem["cue_step"] = int(fr.step)
                    mem["cue_color"] = _color_name(c["rgb"])
                    ev.append("cue %s at step %d" % (mem["cue_color"], fr.step))
                mem["cue_frames"] = int(mem["cue_frames"]) + 1
            elif len(cubes) >= 3 and not mem["target"]:
                mem["sel_step"] = int(fr.step)
                if mem["cue_rgb"]:
                    i, d1, d2 = _match(mem["cue_rgb"], cubes)
                else:
                    i, d1, d2 = 0, 9.0, 9.0  # no cue seen: cannot know; take first cube
                t = cubes[i]
                mem["target"] = [t["xy"][0], t["xy"][1], t["ztop"]]
                mem["target_n"] = int(t["n"])
                mem["match"] = "%s d=%.3f next=%.3f n=%d of %d cubes" % (_color_name(t["rgb"]), d1, d2, t["n"], len(cubes))
                mem["stage"] = "approach"
                ev.append("selection at step %d: %s" % (fr.step, mem["match"]))
    elif mem["cue_rgb"]:
        # re-track the target: a knocked cube (R009) must be followed; accept only well-visible matches
        cubes = _detect_cubes(cur)
        if cubes:
            i, d1, d2 = _match(mem["cue_rgb"], cubes)
            t = cubes[i]
            moved = math.hypot(t["xy"][0] - mem["target"][0], t["xy"][1] - mem["target"][1])
            if d1 < 0.15 and t["n"] >= 0.8 * float(mem["target_n"]) and moved > 0.02:
                mem["target"] = [t["xy"][0], t["xy"][1], t["ztop"]]
                mem["retarget"] = int(mem["retarget"]) + 1
                mem["stage"] = "approach"
                mem["grip"] = 0
                ev.append("target moved %.3f m at step %d -> re-approach" % (moved, cur.step))
    # ---- control: closed gripper, fingertips pressed onto the target cube's top face
    tcp = np.asarray(cur.tcp, dtype=np.float64)
    gripper = -1.0
    cmd = [0.0] * 6
    if mem["target"]:
        tx, ty, tz = mem["target"]
        exy = math.hypot(tx - tcp[0], ty - tcp[1])
        st = mem["stage"]
        lz = float(mem["last_z"]) if mem["last_z"] is not None else 9.0
        zpress = tz + TIP - PRESS_DEPTH
        if exy < XY_DESCEND:
            gz = zpress
        else:
            gz = max(zpress, tz + TIP + CLEAR_Z + FLOOR_SLOPE * (exy - XY_DESCEND))
            if int(mem["retarget"]) == 0:
                gz = min(gz, tcp[2])  # never climb during the first approach (R004 climbed 3 steps)
        on_top = tcp[2] < tz + TIP + 0.006 and (lz - tcp[2]) < 0.004 and exy < 0.02
        if st == "approach" and on_top:
            st = "press"
        if st == "approach" and tcp[2] < tz - 0.004 and "below-top" not in str(mem["events"]):
            ev.append("below-top at step %d (exy=%.3f z=%.3f): fingertips passed the cube top" % (cur.step, exy, tcp[2]))
        if st == "press" and tcp[2] < tz + TIP - 0.012:
            # fingertips well below the top face: not resting on the cube (missed / cube gone) -> re-approach
            st = "approach"
            mem["retarget"] = int(mem["retarget"]) + 1
            mem["grip"] = 0
            ev.append("missed top at step %d (z=%.3f) -> re-approach" % (cur.step, tcp[2]))
            gz = max(zpress, tz + TIP + CLEAR_Z)
        if st == "press":
            mem["grip"] = int(mem["grip"]) + 1
        if st != mem["stage"]:
            ev.append("%s at step %d (exy=%.3f z=%.3f dz=%.4f)" % (st, cur.step, exy, tcp[2], lz - tcp[2]))
        mem["stage"] = st
        err = np.array([tx - tcp[0], ty - tcp[1], gz - tcp[2]])
        a = np.clip(err * np.array([XY_GAIN, XY_GAIN, Z_GAIN]) / 0.1, -1.0, 1.0)
        if st == "press":
            a = np.array([0.0, 0.0, -PRESS_PUSH])  # hold still on the top face: no lateral push (R003 drift)
        cmd = [float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0]
        mem["last_z"] = float(tcp[2])
    if ev:
        mem["events"] = (str(mem["events"]) + "; " + "; ".join(ev))[-900:]
    return {"memory": mem, "act": {"tool": "act_step", "args": {"cmd": cmd, "gripper": gripper}}}


def act_step(robot, cmd, gripper=1.0):
    """Send one native pd_ee_delta_pose step: cmd = [dx,dy,dz,drx,dry,drz] normalized, gripper +1 open/-1 closed.
    Returns the measured TCP and fingers after the step."""
    c = [float(v) for v in cmd][:6]
    f = robot.step(c[0], c[1], c[2], c[3], c[4], c[5], float(gripper))
    return {"step": int(f.step), "tcp": [round(float(v), 4) for v in f.tcp],
            "fingers": [round(float(v), 4) for v in f.fingers], "cmd": [round(v, 3) for v in c]}
