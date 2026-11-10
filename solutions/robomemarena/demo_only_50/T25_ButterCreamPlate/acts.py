"""Act tools: hover over an object, grasp it, place it above a plate, wait."""


def act_hover(robot, x, y, z=0.65):
    """Open the gripper and move the TCP above (x, y) at height z in the expert grasp orientation
    (gripper yawed -45 deg). The wrist camera then looks down on the object for refinement."""
    rep = _move(robot, [x, y, z], GRASP_ROTVEC, -1.0, pos_tol=0.004, rot_tol=0.03, max_steps=250)
    _hold(robot, [x, y, z], GRASP_ROTVEC, -1.0, 3)
    return rep


def act_grasp(robot, x, y, z_grasp=0.438, lift_z=0.69):
    """Align above (x, y) at the current height, descend to z_grasp (stops early on contact stall),
    close the gripper for 30 steps, lift to lift_z. Returns the finger width (m) and motion reports."""
    rep = {}
    z0 = max(float(robot.now().tcp[2]), 0.6)
    rep["align"] = _move(robot, [x, y, z0], GRASP_ROTVEC, -1.0, pos_tol=0.003, rot_tol=0.03, max_steps=80,
                         vmax=0.6)
    rep["descend"] = _move(robot, [x, y, z_grasp], GRASP_ROTVEC, -1.0, pos_tol=0.003, rot_tol=0.04,
                           max_steps=120, vmax=0.5, stall_steps=8, stall_err=0.06)
    tgt = [x, y, float(robot.now().tcp[2])]
    _hold(robot, tgt, GRASP_ROTVEC, 1.0, 30)
    rep["lift"] = _move(robot, [x, y, lift_z], GRASP_ROTVEC, 1.0, pos_tol=0.01, rot_tol=0.08, max_steps=120,
                        vmax=0.7)
    fing = robot.now().fingers
    rep["finger_width"] = float(abs(fing[0]) + abs(fing[1]))
    return rep


def act_place(robot, x, y, z_release=0.525, carry_z=0.70):
    """With the gripper closed, carry to (x, y) at carry_z (gripper pointing straight down), descend to
    z_release, open for 15 steps and lift back to carry_z. Returns motion reports."""
    rep = {}
    rep["carry"] = _move(robot, [x, y, carry_z], PLACE_ROTVEC, 1.0, pos_tol=0.005, rot_tol=0.05, max_steps=250)
    rep["descend"] = _move(robot, [x, y, z_release], PLACE_ROTVEC, 1.0, pos_tol=0.004, rot_tol=0.05,
                           max_steps=120, vmax=0.6, stall_steps=8, stall_err=0.06)
    tgt = [x, y, float(robot.now().tcp[2])]
    _hold(robot, tgt, PLACE_ROTVEC, -1.0, 15)
    rep["lift"] = _move(robot, [x, y, carry_z], PLACE_ROTVEC, -1.0, pos_tol=0.01, rot_tol=0.08, max_steps=120)
    return rep


def act_wait(robot, n=40):
    """Hold the current pose with the gripper open for n steps (lets the checker settle)."""
    f = robot.now()
    tgt = [float(v) for v in f.tcp[:3]]
    _hold(robot, tgt, [float(v) for v in f.tcp[3:6]], -1.0, int(n))
    return {"waited": int(n)}
