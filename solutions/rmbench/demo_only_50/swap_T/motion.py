"""Motion helpers for swap_T (top-down grasps with the aloha-agilex arms)."""

# ee-frame z (world) at grasp: grasp centre (0.12 m along ee +x, pointing down)
# sits 0.035 m above the table, as in every expert grasp (0.7757 m).
EE_GRASP_Z = 0.896
PRE_DZ = 0.07       # pre-grasp / pre-place height above grasp height
LIFT_DZ = 0.10      # lift height after grasping
PLACE_DZ = 0.02
ROUTE_DZ = 0.16     # high transit waypoint above grasp height (block clears the other T)     # release height above grasp height (block bottom ~2 cm above table)
POS_TOL = 0.0025    # re-target when measured ee position misses by more than this
HOME_LEFT = [-0.2979, -0.3138, 0.942, 0.7, 0.0, 0.0, 0.7141]
HOME_RIGHT = [0.3064, -0.3128, 0.941, 0.7036, 0.0, 0.0, 0.7106]


def _quat_down(yaw):
    """Quaternion (w,x,y,z) with ee +x pointing down and ee +z along world yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[0.0, -s, c], [0.0, c, s], [-1.0, 0.0, 0.0]])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    if q[0] < 0:
        q = [-v for v in q]
    return [float(v) for v in q]


def _ee_yaw(pose):
    """World yaw of the ee +z axis for a w,x,y,z pose (top-down grasps)."""
    w, x, y, z = [float(v) for v in pose[3:7]]
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    return math.atan2(zy, zx)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _gripper_yaw(block_yaw, flip):
    return _wrap(block_yaw + (math.pi if flip else 0.0))


def _default_flip(block_yaw):
    """Choose gripper yaw (block yaw or +pi) closest to +90 deg (ee z pointing away
    from the robot), which covers every expert grasp (block yaw 48..132 deg)."""
    d0 = abs(_wrap(block_yaw - math.pi / 2))
    d1 = abs(_wrap(block_yaw + math.pi - math.pi / 2))
    return 0 if d0 <= d1 else 1


def _pose(x, y, z, yaw):
    return [float(x), float(y), float(z)] + _quat_down(yaw)


def _cur(robot):
    f = robot.now()
    return (list(np.asarray(f.left_ee_pose, dtype=float)), float(f.left_gripper),
            list(np.asarray(f.right_ee_pose, dtype=float)), float(f.right_gripper))


def _send(robot, L, lg, R, rg):
    """One native step; returns error text or None (MotionError is an Exception)."""
    try:
        robot.step(*[float(v) for v in L], float(lg), *[float(v) for v in R], float(rg))
    except Exception as e:  # MotionError: an arm could not be planned
        return "MotionError: " + str(e)[:200]
    return None


def _pos_err(pose, target):
    return float(np.linalg.norm(np.asarray(pose[:3], dtype=float) - np.asarray(target[:3], dtype=float)))


def _move(robot, L=None, lg=None, R=None, rg=None, correct=True, log=None):
    """Move one or both arms (None = hold current pose / gripper) in one native
    step, then re-target small measured position misses (<= 5 cm) up to twice.
    Returns the remaining position errors (m) of the commanded arms."""
    cl, clg, cr, crg = _cur(robot)
    tl = L if L is not None else cl
    tr = R if R is not None else cr
    glv = clg if lg is None else lg
    grv = crg if rg is None else rg
    err = _send(robot, tl, glv, tr, grv)
    if err is not None:
        if log is not None:
            log.append("plan error: " + err)
    if correct:
        for _ in range(2):
            cl2, _, cr2, _ = _cur(robot)
            el = _pos_err(cl2, tl) if L is not None else 0.0
            er = _pos_err(cr2, tr) if R is not None else 0.0
            if max(el, er) <= POS_TOL or max(el, er) > 0.05:
                break
            nl, nr = cl2, cr2
            if L is not None and POS_TOL < el <= 0.05:
                d = np.asarray(tl[:3]) - np.asarray(cl2[:3])
                nl = list(np.asarray(tl[:3]) + d) + list(tl[3:])
            if R is not None and POS_TOL < er <= 0.05:
                d = np.asarray(tr[:3]) - np.asarray(cr2[:3])
                nr = list(np.asarray(tr[:3]) + d) + list(tr[3:])
            if log is not None:
                log.append("retarget el=%.4f er=%.4f" % (el, er))
            e3 = _send(robot, nl, glv, nr, grv)
            if e3 is not None:
                if log is not None:
                    log.append("retarget plan error: " + e3)
                break
    cl2, _, cr2, _ = _cur(robot)
    return {"left_err": _pos_err(cl2, tl) if L is not None else 0.0,
            "right_err": _pos_err(cr2, tr) if R is not None else 0.0}
