"""T14 Cookies/Chocolate/Drawers policy.

Goal: put cookies into the top drawer, then chocolate into another (middle) drawer.
Plan (derived from all 25 assigned demonstrations):
  open top drawer -> cookies into top drawer -> close top drawer ->
  open middle drawer -> chocolate into middle drawer -> hold.
Memory keeps the perceived geometry (from the first frame, before the arm moves
into view) and the stage; stage checks use the measured tcp/fingers."""

# geometry relative to the perceived top handle (fitted on demonstrations)
PRE_DY = -0.08            # pre-grasp offset in front of a handle
TOP_PULL_DY = -0.20       # pull target (drawer stops near -0.167)
MID_PULL_DY = -0.21       # pull target (middle drawer stops near -0.183)
PUSH_DY = 0.012           # push target for closing (stops near -0.005)
COOKIE_DROP = [-0.003, -0.055, 0.12]
CHOC_DROP = [-0.005, -0.058, 0.050]
TRANSIT_Z = 0.90
SIDE_HIGH_Z = 0.83       # expert rotated down->side at z 0.83..0.87 (demo steps ~710-720)
HOLD_MIN = 0.006          # finger width (sum of |q|) above which something is held
MAX_TRIES = 2


def _goal_ok(goal):
    g = str(goal).lower()
    return ("cookie" in g) and ("chocolate" in g) and ("drawer" in g)


def _mv(p, ori, g, tol=0.004, settle=False, mx=250, name="move", vmax=1.0):
    return {"kind": "move", "p": [float(v) for v in p], "ori": ori, "g": g, "tol": tol,
            "settle": settle, "max": mx, "name": name, "vmax": vmax}


def _grip(g, n, ori, name="grip"):
    return {"kind": "grip", "g": g, "n": n, "ori": ori, "name": name}


def _drag(p, ori, mx=160, name="drag"):
    return {"kind": "drag", "p": [float(v) for v in p], "ori": ori, "g": 1.0, "tol": 0.004,
            "max": mx, "name": name}


def _approach_handle(h, ori_from_down):
    """Segments to bring an open side-gripper onto handle point h, approaching along +y."""
    segs = []
    pre = [h[0], h[1] + PRE_DY, h[2]]
    segs.append(_mv(pre, "side", -1.0, tol=0.01, name="pre_handle"))
    segs.append(_mv(pre, "side", -1.0, tol=0.004, settle=True, mx=80, name="pre_handle_fine"))
    segs.append(_mv(h, "side", -1.0, tol=0.003, settle=True, mx=150, name="to_handle"))
    return segs


def _pick(obj, name):
    above = [obj[0], obj[1], 0.63]
    return [_mv(above, "down", -1.0, tol=0.006, name="above_" + name),
            _mv(above, "down", -1.0, tol=0.003, settle=True, mx=80, name="above_fine"),
            _mv(obj, "down", -1.0, tol=0.003, settle=True, mx=200, name="descend_" + name)]


def perceive_decide(obs, goal, memory, last):
    """Stage machine: perceive geometry once, then issue one act_sequence per sub-task."""
    mem = dict(memory)
    for k in ("stage", "handle", "cookie", "choc", "top_open_y", "mid_open_y", "tries", "note"):
        if k not in mem:
            mem[k] = None
    if mem["tries"] is None:
        mem["tries"] = 0
    f = obs.current
    tcp = np.asarray(f.tcp, dtype=float)
    fw = _fw(f)
    stage = mem["stage"]

    if stage is None or mem["handle"] is None:
        scene = _perceive_scene(f)
        mem["handle"] = [round(float(v), 5) for v in scene["handle"]]
        mem["cookie"] = [round(float(v), 5) for v in scene["cookie"]]
        mem["choc"] = [round(float(v), 5) for v in scene["choc"]]
        mem["note"] = "perceived ok=%s n=%s goal_ok=%s" % (scene["ok"], scene["n"], _goal_ok(goal))
        stage = "start"

    H = mem["handle"]
    C = mem["cookie"]
    K = mem["choc"]
    M = [H[0], H[1] + MID_HANDLE_DY, H[2] + MID_HANDLE_DZ]
    segs = None
    nxt = stage

    if stage == "start":
        segs = _approach_handle(H, True)
        nxt = "at_top"
    elif stage == "at_top":
        segs = [_grip(1.0, 25, "side", "close_top"),
                _drag([H[0], H[1] + TOP_PULL_DY, H[2]], "side", name="pull_top")]
        nxt = "pulled_top"
    elif stage == "pulled_top":
        opened = tcp[1] < H[1] - 0.12
        held = fw > HOLD_MIN
        if (opened and held) or mem["tries"] >= MAX_TRIES:
            mem["top_open_y"] = float(min(tcp[1], H[1] - 0.12))
            mem["tries"] = 0
            oy = mem["top_open_y"]
            segs = [_grip(-1.0, 15, "side", "release_top"),
                    _mv([H[0], oy - 0.10, H[2] + 0.005], "side", -1.0, tol=0.01, name="retreat"),
                    _mv([H[0], oy - 0.10, 0.81], "side", -1.0, tol=0.015, name="lift")]
            segs += _pick(C, "cookie")
            nxt = "at_cookie"
        else:
            mem["tries"] = mem["tries"] + 1
            mem["note"] = "top pull failed (y=%.3f fw=%.4f); retry" % (tcp[1], fw)
            segs = [_grip(-1.0, 15, "side", "reopen"),
                    _mv([tcp[0], tcp[1] - 0.06, tcp[2]], "side", -1.0, tol=0.01, name="back_off")]
            segs += _approach_handle(H, False)
            nxt = "at_top"
    elif stage == "at_cookie":
        segs = [_grip(1.0, 25, "down", "close_cookie"),
                _mv([C[0], C[1], 0.75], "down", 1.0, tol=0.01, name="lift_cookie")]
        nxt = "lifted_cookie"
    elif stage == "lifted_cookie":
        if fw > HOLD_MIN or mem["tries"] >= MAX_TRIES:
            mem["tries"] = 0
            oy = mem["top_open_y"]
            drop = [H[0] + COOKIE_DROP[0], H[1] + COOKIE_DROP[1], H[2] + COOKIE_DROP[2]]
            segs = [_mv([drop[0], drop[1], TRANSIT_Z], "down", 1.0, tol=0.012, name="transit_cookie"),
                    _mv(drop, "down", 1.0, tol=0.005, settle=True, mx=150, name="lower_cookie"),
                    _grip(-1.0, 15, "down", "release_cookie"),
                    _mv([drop[0], drop[1], TRANSIT_Z], "down", -1.0, tol=0.015, name="up"),
                    _mv([H[0], oy + PRE_DY, SIDE_HIGH_Z], "side", -1.0, tol=0.015, name="side_high")]
            nxt = "placed_cookie"
        else:
            mem["tries"] = mem["tries"] + 1
            mem["note"] = "cookie grasp failed fw=%.4f; retry" % fw
            segs = [_grip(-1.0, 15, "down", "reopen")] + _pick(C, "cookie")
            nxt = "at_cookie"
    elif stage == "placed_cookie":
        oy = mem["top_open_y"]
        hopen = [H[0], oy, H[2] + 0.003]
        segs = [_mv([H[0], oy + PRE_DY, hopen[2]], "side", -1.0, tol=0.006, name="down_to_handle"),
                _mv([H[0], oy + PRE_DY, hopen[2]], "side", -1.0, tol=0.004, settle=True, mx=80, name="fine"),
                _mv(hopen, "side", -1.0, tol=0.003, settle=True, mx=150, name="to_open_handle"),
                _grip(1.0, 25, "side", "close_top2"),
                _drag([H[0], H[1] + PUSH_DY, hopen[2]], "side", name="push_top")]
        nxt = "pushed_top"
    elif stage == "pushed_top":
        closed = tcp[1] > H[1] - 0.03
        if closed or mem["tries"] >= MAX_TRIES:
            mem["tries"] = 0
            segs = [_grip(-1.0, 15, "side", "release_top2"),
                    _mv([H[0], H[1] - 0.12, H[2]], "side", -1.0, tol=0.008, name="retreat2"),
                    _mv([M[0], M[1] - 0.12, M[2]], "side", -1.0, tol=0.008, name="down_mid")]
            segs += _approach_handle(M, False)[1:]
            nxt = "at_mid"
        else:
            mem["tries"] = mem["tries"] + 1
            mem["note"] = "top not closed (y=%.3f); push with closed hand" % tcp[1]
            segs = [_grip(-1.0, 15, "side", "reopen"),
                    _mv([tcp[0], tcp[1] - 0.05, H[2]], "side", -1.0, tol=0.008, name="back"),
                    _grip(1.0, 20, "side", "fist"),
                    _drag([H[0], H[1] + PUSH_DY, H[2]], "side", name="push_fist")]
            nxt = "pushed_top"
    elif stage == "at_mid":
        segs = [_grip(1.0, 25, "side", "close_mid"),
                _drag([M[0], M[1] + MID_PULL_DY - MID_HANDLE_DY, M[2]], "side", name="pull_mid")]
        nxt = "pulled_mid"
    elif stage == "pulled_mid":
        opened = tcp[1] < M[1] - 0.12
        held = fw > HOLD_MIN
        if (opened and held) or mem["tries"] >= MAX_TRIES:
            mem["tries"] = 0
            mem["mid_open_y"] = float(min(tcp[1], M[1] - 0.12))
            oy = mem["mid_open_y"]
            segs = [_grip(-1.0, 15, "side", "release_mid"),
                    _mv([M[0], oy - 0.12, M[2]], "side", -1.0, tol=0.01, name="retreat3"),
                    _mv([M[0], oy - 0.12, 0.70], "side", -1.0, tol=0.015, name="lift3"),
                    _mv([K[0], K[1], 0.70], "down", -1.0, tol=0.015, name="transit_choc")]
            segs += _pick(K, "choc")
            nxt = "at_choc"
        else:
            mem["tries"] = mem["tries"] + 1
            mem["note"] = "mid pull failed (y=%.3f fw=%.4f); retry" % (tcp[1], fw)
            segs = [_grip(-1.0, 15, "side", "reopen"),
                    _mv([tcp[0], tcp[1] - 0.06, tcp[2]], "side", -1.0, tol=0.01, name="back_off")]
            segs += _approach_handle(M, False)
            nxt = "at_mid"
    elif stage == "at_choc":
        segs = [_grip(1.0, 25, "down", "close_choc"),
                _mv([K[0], K[1], 0.80], "down", 1.0, tol=0.01, name="lift_choc")]
        nxt = "lifted_choc"
    elif stage == "lifted_choc":
        if fw > HOLD_MIN or mem["tries"] >= MAX_TRIES:
            mem["tries"] = 0
            drop = [H[0] + CHOC_DROP[0], H[1] + CHOC_DROP[1], H[2] + CHOC_DROP[2]]
            segs = [_mv([drop[0], drop[1], TRANSIT_Z], "down", 1.0, tol=0.012, name="transit_choc2"),
                    _mv(drop, "down", 1.0, tol=0.005, settle=True, mx=150, name="lower_choc"),
                    _grip(-1.0, 20, "down", "release_choc"),
                    _mv([drop[0], drop[1], 0.85], "down", -1.0, tol=0.015, name="up_final")]
            nxt = "done"
        else:
            mem["tries"] = mem["tries"] + 1
            mem["note"] = "chocolate grasp failed fw=%.4f; retry" % fw
            segs = [_grip(-1.0, 15, "down", "reopen")] + _pick(K, "choc")
            nxt = "at_choc"

    mem["stage"] = nxt
    if segs is None:
        return {"memory": mem, "act": {"tool": "act_wait", "args": {"n": 100}}}
    return {"memory": mem, "act": {"tool": "act_sequence", "args": {"segments": segs}}}
