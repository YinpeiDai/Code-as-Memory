"""Task policy: pick the goal items in the stated order and drop each into the same basket."""

MAX_PICK_ATTEMPTS = 4
FIELDS = ("stage", "order", "idx", "table_z", "basket", "basket_boxes", "item_locs", "attempts",
          "pending", "placed", "waits", "log")


def _mem(memory):
    m = {k: memory.get(k) if isinstance(memory, dict) else None for k in FIELDS}
    if m["placed"] is None:
        m["placed"] = []
    if m["attempts"] is None:
        m["attempts"] = 0
    if m["waits"] is None:
        m["waits"] = 0
    if m["idx"] is None:
        m["idx"] = 0
    if m["log"] is None:
        m["log"] = ""
    return m


def _note(m, text):
    m["log"] = (m["log"] + " | " + text)[-600:]


def _locate(frame, name, table_z, boxes, near=None):
    """Item xy from the wrist view, falling back to the agentview."""
    for cam in ("wrist", "agentview"):
        items = _detect_items(frame, table_z, boxes, cam=cam)
        it = _match_item(items, name, near=near, radius=0.15) if near is not None else None
        if it is None:
            it = _match_item(items, name)
        if it is not None:
            return it["xy"], cam, it
    return None, None, None


def _pick_act(m, name, xy, det=None):
    yaw, ov = _grasp_yaw(det, float(xy[0]), float(xy[1]), m["basket_boxes"] or [])
    return {"tool": "act_pick", "args": {"item": name, "x": float(xy[0]), "y": float(xy[1]),
                                         "table_z": float(m["table_z"]), "yaw": float(yaw),
                                         "basket_boxes": m["basket_boxes"] or []}}


def _refresh_basket(m, frame):
    """Re-measure the remembered basket (it may have been nudged); accept only a detection of the same size
    near the remembered one, so arm-merged or occluded clusters are ignored."""
    old = _basket_box(m)
    if old is None or m["basket"] is None:
        return False
    det = _detect_baskets(frame, m["table_z"])
    w0, h0 = old[1] - old[0], old[3] - old[2]
    for b in det["baskets"]:
        bx = b["box"]
        if abs((bx[1] - bx[0]) - w0) > 0.015 or abs((bx[3] - bx[2]) - h0) > 0.015:
            continue
        c = b["center"]
        if math.hypot(c[0] - m["basket"][0], c[1] - m["basket"][1]) > 0.06:
            continue
        boxes = list(m["basket_boxes"] or [])
        for k in range(len(boxes)):
            if boxes[k] == old:
                boxes[k] = [float(v) for v in bx]
        m["basket_boxes"] = boxes
        m["basket"] = [float(c[0]), float(c[1]), float(m["basket"][2])]
        return True
    return False


def _place_act(m):
    b = m["basket"]
    return {"tool": "act_place", "args": {"x": float(b[0]), "y": float(b[1]),
                                          "release_z": float(b[2]) + 0.09}}


def perceive_decide(obs, goal, memory, last):
    """Detect baskets/items with RGB-D, track stage in memory, and choose the next pick, place or wait."""
    m = _mem(memory)
    cur = obs.current
    tcp = np.asarray(cur.tcp, dtype=float)
    fing = float(np.mean(np.abs(np.asarray(cur.fingers, dtype=float))))
    # held state: trust the measured outcome of the last act tool when available (a lengthwise grasp of a long
    # box leaves the fingers at ~0.038, close to fully open 0.039-0.040); otherwise use the finger reading
    lt = last if isinstance(last, str) else ""
    if lt.startswith("act_pick") and '"held": ' in lt:
        holding = '"held": true' in lt and fing > HELD_MIN
    elif lt.startswith("act_place") and ('"released": true' in lt or '"lost_in_transit": true' in lt):
        holding = False
    else:
        holding = fing > HELD_MIN and fing < 0.0386

    if m["stage"] is None or m["stage"] == "init":
        m["order"] = _parse_order(goal)
        bdet = _detect_baskets(cur, m["table_z"])
        m["table_z"] = bdet["table_z"]
        boxes = [b["box"] for b in bdet["baskets"]]
        m["basket_boxes"] = boxes
        locs = []
        dets = []
        for name in m["order"]:
            xy, cam, det = _locate(cur, name, m["table_z"], boxes)
            locs.append(xy)
            dets.append(det)
        m["item_locs"] = locs
        ref = locs[0] if locs and locs[0] is not None else [float(tcp[0]), float(tcp[1])]
        b = _choose_basket(bdet["baskets"], ref)
        if b is None or locs[0] is None:
            m["attempts"] += 1
            _note(m, "init: missing basket or item, re-observe")
            m["stage"] = "init"
            m["pending"] = "observe"
            # higher, centred viewpoint to see more of the table
            z = 0.80 if m["attempts"] % 2 else 0.70
            return {"memory": m, "act": {"tool": "act_goto", "args": {"x": -0.05, "y": 0.0, "z": z}}}
        m["basket"] = [float(v) for v in b["center"]]
        m["stage"] = "pick"
        m["attempts"] = 1
        m["pending"] = "pick"
        _note(m, "init ok basket=%s" % [round(v, 3) for v in m["basket"]])
        return {"memory": m, "act": _pick_act(m, m["order"][0], locs[0], dets[0])}

    order = m["order"] or ["butter", "popcorn"]
    boxes = m["basket_boxes"] or []
    if m["idx"] >= len(order):
        m["stage"] = "done"

    if m["stage"] == "pick":
        name = order[m["idx"]]
        if m["pending"] == "pick" and holding and tcp[2] > m["table_z"] + 0.15:
            m["stage"] = "place"
            m["pending"] = "place"
            ok = _refresh_basket(m, cur)
            _note(m, "%s held f=%.4f basket%s=%s" % (name, fing, "" if ok else "(mem)",
                                                  [round(v, 3) for v in m["basket"][:2]]))
            return {"memory": m, "act": _place_act(m)}
        # grasp missed (or not tried yet): re-detect from the current view
        if m["pending"] == "pick":
            _note(m, "%s grasp miss f=%.4f" % (name, fing))
        prev = m["item_locs"][m["idx"]] if m["item_locs"] else None
        xy, cam, det = _locate(cur, name, m["table_z"], boxes, near=prev)
        if xy is None and prev is not None and m["pending"] == "observe":
            xy = prev  # still unseen after re-observing: try the last known location
        if xy is None:
            # nothing seen: re-observe from a high pose over the last known location (or the table centre)
            m["pending"] = "observe"
            m["attempts"] += 1
            gx, gy = (prev[0], prev[1]) if prev is not None else (-0.05, 0.0)
            _note(m, "%s not seen, re-observe" % name)
            return {"memory": m, "act": {"tool": "act_goto", "args": {"x": float(gx) - 0.05, "y": float(gy), "z": 0.80}}}
        m["item_locs"][m["idx"]] = [float(xy[0]), float(xy[1])]
        m["attempts"] += 1
        m["pending"] = "pick"
        return {"memory": m, "act": _pick_act(m, name, xy, det)}

    if m["stage"] == "place":
        name = order[m["idx"]]
        if m["pending"] == "place" and not holding and '"lost_in_transit": true' in lt:
            # the fingers closed on nothing while carrying: the item fell; find it with either camera
            xy, cam, det = _locate(cur, name, m["table_z"], boxes)
            m["stage"] = "pick"
            m["attempts"] = 1
            _note(m, "%s lost in transit, seen=%s" % (name, cam))
            if xy is None:
                m["pending"] = "observe"
                return {"memory": m, "act": {"tool": "act_goto", "args": {"x": float(tcp[0]) - 0.05,
                                                                          "y": float(tcp[1]), "z": 0.80}}}
            m["item_locs"][m["idx"]] = [float(xy[0]), float(xy[1])]
            m["pending"] = "pick"
            return {"memory": m, "act": _pick_act(m, name, xy, det)}
        if m["pending"] == "place" and not holding:
            # released; check whether the item lies on the table outside the basket
            xy, cam, det = _locate(cur, name, m["table_z"], boxes)
            if xy is not None and tcp[2] > m["table_z"] + 0.15 and math.hypot(
                    xy[0] - m["basket"][0], xy[1] - m["basket"][1]) > 0.12 and cam == "wrist":
                m["stage"] = "pick"
                m["attempts"] = 1
                m["item_locs"][m["idx"]] = [float(xy[0]), float(xy[1])]
                m["pending"] = "pick"
                _note(m, "%s seen on table after place, re-pick" % name)
                return {"memory": m, "act": _pick_act(m, name, xy, det)}
            seen = _basket_contents(cur, _basket_box(m), m["table_z"])
            m["placed"] = list(m["placed"]) + [name]
            _note(m, "%s placed basket_px=%s" % (name, seen))
            m["idx"] += 1
            m["attempts"] = 0
            if m["idx"] >= len(order):
                m["stage"] = "done"
            else:
                m["stage"] = "pick"
                nxt = order[m["idx"]]
                prev = m["item_locs"][m["idx"]]
                xy, cam, det = _locate(cur, nxt, m["table_z"], boxes, near=prev)
                if prev is not None and (xy is None or cam != "wrist"):
                    xy, det = prev, None
                if xy is None:
                    m["pending"] = "observe"
                    return {"memory": m, "act": {"tool": "act_goto", "args": {"x": -0.05, "y": 0.0, "z": 0.80}}}
                m["item_locs"][m["idx"]] = [float(xy[0]), float(xy[1])]
                m["attempts"] = 1
                m["pending"] = "pick"
                return {"memory": m, "act": _pick_act(m, nxt, xy, det)}
        else:
            if holding:
                m["pending"] = "place"
                return {"memory": m, "act": _place_act(m)}
            # not holding and no place attempted: go back to picking
            m["stage"] = "pick"
            m["pending"] = "pick"
            prev = m["item_locs"][m["idx"]]
            xy, cam, det = _locate(cur, name, m["table_z"], boxes, near=prev)
            if xy is None:
                xy, det = prev, None
            m["attempts"] += 1
            return {"memory": m, "act": _pick_act(m, name, xy, det)}

    # done: all items released over the basket; give the scorer time, then re-check the table
    m["stage"] = "done"
    m["waits"] += 1
    if m["waits"] >= 3 and m["waits"] % 3 == 0:
        for k, name in enumerate(order):
            xy, cam, det = _locate(cur, name, m["table_z"], boxes)
            if xy is not None and math.hypot(xy[0] - m["basket"][0], xy[1] - m["basket"][1]) > 0.12:
                m["idx"] = k
                m["stage"] = "pick"
                m["attempts"] = 1
                m["item_locs"][k] = [float(xy[0]), float(xy[1])]
                m["pending"] = "pick"
                _note(m, "%s still on table at done, re-pick" % name)
                return {"memory": m, "act": _pick_act(m, name, xy, det)}
    m["pending"] = "wait"
    return {"memory": m, "act": {"tool": "act_wait", "args": {"steps": 20, "gripper": -1.0}}}


def _basket_box(m):
    b = m["basket"]
    for box in m["basket_boxes"] or []:
        if box[0] <= b[0] <= box[1] and box[2] <= b[1] <= box[3]:
            return box
    return None
