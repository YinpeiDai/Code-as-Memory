"""Task program: put the butter into the frypan, then pour milk over it twice (two tilt cycles)."""

MILK_TILT = 49.0          # gripper tilt about world +y used to grasp the upright milk (expert 49.2 deg)
POUR_TILT = 0.0           # gripper straight down while pouring -> milk tilted ~49 deg
MILK_PRE_DX = 0.036       # tcp +x offset from milk grasp point before descending (expert ~0.036)
MILK_DOWN_DX = 0.021      # tcp +x offset at the bottom of the descent, then push to the grasp point
TRANSIT_DZ = 0.37         # transit height above table
POUR_DZ = 0.265           # tcp height above table while pouring (expert 0.678..0.70 with table 0.43)
HOLD_FINGER = 0.006       # mean |finger| above this means an object is held (butter 0.019, milk 0.026)
OPEN_FINGER = 0.036       # mean |finger| above this means the gripper did not close (open = 0.039..0.040)
MAX_RETRIES = 2

FIELDS = ("phase", "table_z", "butter", "pan", "pan_r", "milk", "retries", "pours", "log")


def _f(v, d):
    return d if v is None else v


def _finger_width(frame):
    f = getattr(frame, "fingers", None)
    if f is None:
        return 0.0
    f = np.asarray(f, dtype=float)
    return float(np.mean(np.abs(f)))


def _failed(last):
    return isinstance(last, str) and ("the tool failed" in last or " -> error:" in last)


def _move(x, y, z, tilt=0.0, grip=-1.0, **kw):
    args = {"x": float(x), "y": float(y), "z": float(z), "tilt": float(tilt), "grip": float(grip)}
    for k, v in kw.items():
        args[k] = v
    return {"tool": "act_move", "args": args}


def _apply_detection(mem, det, which):
    if "table" in which:
        mem["table_z"] = float(det["table_z"])
    if "butter" in which and det.get("butter") is not None:
        mem["butter"] = [float(det["butter"][0]), float(det["butter"][1])]
    if "pan" in which and det.get("pan") is not None:
        mem["pan"] = [float(det["pan"][0]), float(det["pan"][1])]
        mem["pan_r"] = float(det["pan"][2])
    if "milk" in which and det.get("milk") is not None:
        mem["milk"] = [float(det["milk"][0]), float(det["milk"][1]), float(det["milk"][2])]


def _act_for(phase, mem, frame):
    h = mem["table_z"]
    bx = mem["butter"][0] + BUTTER_GRASP_OFF[0]
    by = mem["butter"][1] + BUTTER_GRASP_OFF[1]
    px, py = mem["pan"][0], mem["pan"][1]
    lx, ly = px + PAN_PLACE_OFF[0], py + PAN_PLACE_OFF[1]
    ox, oy = px + PAN_POUR_OFF[0], py + PAN_POUR_OFF[1]
    mx = mem["milk"][0] + MILK_GRASP_OFF[0]
    my = mem["milk"][1] + MILK_GRASP_OFF[1]
    tcp = np.asarray(frame.tcp, dtype=float)
    if phase == "b_open":
        return {"tool": "act_grip", "args": {"close": False, "steps": 15}}
    if phase == "b_above":
        return _move(bx, by, h + 0.19, 0.0, -1.0, tol=0.006, max_steps=250)
    if phase == "b_down":
        return _move(bx, by, h + BUTTER_GRASP_DZ, 0.0, -1.0, tol=0.003, max_steps=150,
                     max_speed=0.6, stall_ok=True)
    if phase == "b_close":
        return {"tool": "act_grip", "args": {"close": True, "steps": 30}}
    if phase == "b_lift":
        return _move(tcp[0], tcp[1], h + 0.32, 0.0, 1.0, tol=0.01, max_steps=150)
    if phase == "p_above":
        return _move(lx, ly, h + TRANSIT_DZ, 0.0, 1.0, tol=0.006, max_steps=250)
    if phase == "p_down":
        return _move(lx, ly, h + 0.18, 0.0, 1.0, tol=0.005, max_steps=150)
    if phase == "p_open":
        return {"tool": "act_grip", "args": {"close": False, "steps": 25}}
    if phase == "p_lift":
        return _move(lx, ly, h + TRANSIT_DZ, 0.0, -1.0, tol=0.01, max_steps=150)
    if phase == "m_open":
        return {"tool": "act_grip", "args": {"close": False, "steps": 15, "tilt": MILK_TILT}}
    if phase == "m_up":
        return _move(tcp[0], tcp[1], h + TRANSIT_DZ, MILK_TILT, -1.0, tol=0.01, max_steps=150)
    if phase == "m_above":
        return _move(mx + MILK_PRE_DX, my, h + TRANSIT_DZ, 0.0, -1.0, tol=0.006, max_steps=250)
    if phase == "m_tilt":
        return _move(mx + MILK_PRE_DX, my, h + 0.30, MILK_TILT, -1.0, tol=0.006,
                     rot_tol_deg=1.5, max_steps=150)
    if phase == "m_pre":
        return _move(mx + MILK_PRE_DX, my, h + 0.25, MILK_TILT, -1.0, tol=0.004,
                     rot_tol_deg=1.5, max_steps=120)
    if phase == "m_down":
        return _move(mx + MILK_DOWN_DX, my, h + MILK_GRASP_DZ, MILK_TILT, -1.0, tol=0.003,
                     rot_tol_deg=2.0, max_steps=200, max_speed=0.6, stall_ok=True)
    if phase == "m_push":
        return _move(mx, my, h + MILK_GRASP_DZ, MILK_TILT, -1.0, tol=0.003, rot_tol_deg=2.0,
                     max_steps=120, max_speed=0.3, stall_ok=True)
    if phase == "m_close":
        return {"tool": "act_grip", "args": {"close": True, "steps": 30, "tilt": MILK_TILT}}
    if phase == "m_lift":
        return _move(tcp[0], tcp[1], h + 0.35, MILK_TILT, 1.0, tol=0.01, rot_tol_deg=3.0,
                     max_steps=150, max_speed=0.6)
    if phase == "o_pre":
        return _move(ox, oy, h + POUR_DZ, MILK_TILT, 1.0, tol=0.005, rot_tol_deg=2.0,
                     max_steps=250, max_speed=0.6)
    if phase == "pour":
        # first pour: tilt the milk and hold; later pours (after the first) hold much longer
        hold = 75 if _f(mem["pours"], 0) == 0 else 300
        return _move(ox, oy, h + POUR_DZ, POUR_TILT, 1.0, tol=0.006, rot_tol_deg=2.0,
                     min_steps=hold, max_steps=hold + 40)
    if phase == "unpour":
        return _move(ox, oy, h + POUR_DZ - 0.02, MILK_TILT, 1.0, tol=0.006, rot_tol_deg=2.0,
                     min_steps=25, max_steps=80)
    # fallback / end: keep the milk tilted over the pan
    return _move(ox, oy, h + POUR_DZ, POUR_TILT, 1.0, tol=0.006, min_steps=200, max_steps=400)


def _next_phase(phase, mem, frame, last):
    """Transition after the act of `phase` finished, using measured fingers / tool errors."""
    w = _finger_width(frame)
    r = _f(mem["retries"], 0)
    if _failed(last) and r < 6:
        mem["retries"] = r + 1
        return phase
    order = ["b_above", "b_down", "b_close", "b_lift", "p_above", "p_down", "p_open", "p_lift",
             "m_above", "m_tilt", "m_pre", "m_down", "m_push", "m_close", "m_lift", "o_pre"]
    if phase == "b_open":
        return "b_above"
    if phase == "m_open":
        return "m_up"
    if phase == "m_up":
        return "m_above"
    empty = w < HOLD_FINGER or w > OPEN_FINGER
    if phase in ("b_lift", "p_above") and empty and r < MAX_RETRIES:
        mem["retries"] = r + 1
        mem["_redetect"] = "butter"
        return "b_open"
    if phase in ("m_lift", "o_pre") and empty and r < 2 * MAX_RETRIES:
        mem["retries"] = r + 1
        mem["_redetect"] = "milk"
        return "m_open"
    if phase in order and phase != "o_pre":
        return order[order.index(phase) + 1]
    if phase == "o_pre":
        return "pour"
    if phase == "pour":
        mem["pours"] = _f(mem["pours"], 0) + 1
        return "unpour"
    if phase == "unpour":
        return "pour"
    return "pour"


def perceive_decide(obs, goal, memory, last):
    """Detect objects at the first decision, then run the butter -> pan -> milk -> pour x2 phases."""
    mem = {k: memory.get(k) for k in FIELDS}
    frame = obs.current
    if mem["phase"] is None:
        det = _detect_scene(frame)
        mem["table_z"] = DEFAULT_TABLE_Z
        mem["butter"] = [DEFAULT_BUTTER[0], DEFAULT_BUTTER[1]]
        mem["pan"] = [DEFAULT_PAN[0], DEFAULT_PAN[1]]
        mem["pan_r"] = DEFAULT_PAN[2]
        mem["milk"] = [DEFAULT_MILK_TOP[0], DEFAULT_MILK_TOP[1], DEFAULT_TABLE_Z + 0.134]
        _apply_detection(mem, det, ("table", "butter", "pan", "milk"))
        missing = [k for k in ("butter", "pan", "milk") if det.get(k) is None]
        mem["retries"] = 0
        mem["pours"] = 0
        mem["log"] = "init det missing=" + ",".join(missing)
        phase = "b_above"
    else:
        for k, d in (("table_z", DEFAULT_TABLE_Z), ("retries", 0), ("pours", 0)):
            mem[k] = _f(mem[k], d)
        mem["_redetect"] = None
        phase = _next_phase(mem["phase"], mem, frame, last)
        red = mem.pop("_redetect", None)
        if red is not None:
            det = _detect_scene(frame)
            _apply_detection(mem, det, (red,))
        mem["log"] = (str(mem["phase"]) + "->" + phase + " w=" + str(round(_finger_width(frame), 4))
                      + " pours=" + str(mem["pours"]))
    mem.pop("_redetect", None)
    act = _act_for(phase, mem, frame)
    mem["phase"] = phase
    return {"memory": {k: mem[k] for k in FIELDS}, "act": act}
