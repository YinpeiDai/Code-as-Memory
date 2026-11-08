"""Maintain host-only historical MIKASA success-test diagnostics.

These summaries are never added to new learner prompts. Preserve existing run
exports; this script only maintains ``metadata/mikasa_robo/success_tests.json``.

Each entry summarizes, in physical terms, the success and failure predicates of
one env class's ``evaluate()`` (plus the first ``step()`` override in its class
chain), as read from the pinned source. ``{EXPR}`` placeholders are arithmetic
over class constants and are resolved per task, leaf class first, so subclasses
that change a threshold get their own value. The SHA-256 of the summarized code
is recorded; the host diagnostic refuses a summary whose code changed.

Rules for the text: success/failure predicates and thresholds only; nothing
that identifies per-episode hidden state (which object is the target, counts,
seeds) and no implementation identifiers.

    .venv/bin/python -I scripts/pin_mikasa_robo_success_tests.py
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STATIC = "the arm is static (every arm joint speed at most 0.2 rad/s)"
PRESS = (
    "A press registers when the TCP is within {BUTTON_PRESS_XY_RADIUS} m horizontally of the "
    "button's centre and no more than {BUTTON_PRESS_Z_MARGIN - BUTTON_CAP_TRAVEL * BUTTON_PRESS_EVENT_RATIO} m "
    "above the top of its unpressed cap; the button is released when the TCP is again at least "
    "{BUTTON_PRESS_Z_MARGIN - BUTTON_CAP_TRAVEL * BUTTON_RELEASE_READY_RATIO} m above the cap top or "
    "leaves that radius."
)
LIFT = (
    "only after the button has been released and the TCP has risen at least "
    "{REQUIRED_LIFT_HEIGHT - LIFT_CONFIRM_TOL} m above its height when the press registered"
)
HEAD = "Official success test (from the environment's evaluation code): "

TESTS = {
    "BlinkCountButtonPressVLABaseEnv": HEAD
    + "button contacts are judged from the TCP position and count only once the phase lamp has "
    "turned green. "
    + PRESS
    + " A red press is counted "
    + LIFT
    + "; no new red press registers until the previous one is counted. Registering more red "
    "presses than there were blinks fails the episode at once. The first black press ends the "
    "episode: success if the counted red presses equal the number of blinks, failure otherwise.",
    "RememberColorVLABaseEnv": HEAD
    + "success at the first step where the TCP is within {GOAL_THRESH} m of the centre of the "
    "cube whose colour matches the cue (cube centres rest {CUBE_HALFSIZE} m above the table) "
    "while " + STATIC + ". There is no failure condition; otherwise the episode times out.",
    "FindImposterColorVLABaseEnv": HEAD
    + "after all cubes reappear, success at the first step where the TCP is within {GOAL_THRESH} m "
    "of the centre of the cube whose colour was absent from the cue (cube centres rest "
    "{CUBE_HALFSIZE} m above the table) while " + STATIC + ". There is no failure condition; "
    "otherwise the episode times out.",
    "ShellGameTouchVLABaseEnv": HEAD
    + "after the cups are lowered, success at the first step where the TCP is within "
    "{GOAL_THRESH} m of the reference point of the cup hiding the ball (about 0.053 m above the "
    "table) while " + STATIC + " and no cup has moved more than "
    "{MUG_DISPLACEMENT_SUCCESS_THRESH} m horizontally since the cups were lowered. There is no "
    "failure condition; otherwise the episode times out.",
    "BunchOfColorsVLABaseEnv": HEAD
    + "checked only after all cubes have reappeared. A cube is marked touched when the TCP comes "
    "within {CUBE_HALFSIZE} m of the point {CUBE_HALFSIZE + 0.005} m above the cube's centre "
    "while " + STATIC + "; marks are permanent. The button responds only while every cue cube "
    "and no other cube is marked and no cube is more than {MAX_ALLOWED_CUBE_DISPLACEMENT} m "
    "horizontally from where it reappeared. "
    + PRESS
    + " The press counts "
    + LIFT
    + ". Success at the first step with a counted press, exactly the cue cubes marked, no cube "
    "displaced beyond that limit and a static arm. Touching a non-cue cube makes success "
    "impossible; there is no early failure, so the episode then runs to its step limit.",
}
NO_FAIL = " There is no failure condition; otherwise the episode times out."
TOUCH_REACH = (
    "success at the first step where the TCP is within {GOAL_THRESH} m of the reference point "
    "of the {target} while " + STATIC + "." + NO_FAIL
)
ORDERED = (
    "checked only after all cubes have reappeared. A cube is marked touched when the TCP comes "
    "within {CUBE_HALFSIZE} m of the point {CUBE_HALFSIZE + 0.005} m above the cube's centre "
    "while " + STATIC + "; marks are permanent and their order is recorded. The button responds "
    "only while exactly the cue cubes are marked, in the order they were shown in the cue, and no "
    "cube is more than {MAX_ALLOWED_CUBE_DISPLACEMENT} m horizontally from where it reappeared. "
    + PRESS
    + " The press counts "
    + LIFT
    + ". Success at the first step with a counted press, those marks, no cube displaced beyond "
    "that limit and a static arm. Touching a non-cue cube, or cue cubes out of cue order, makes "
    "success impossible; there is no early failure, so the episode then runs to its step limit."
)
BATTERY_PRESS = (
    "the button is pressed: the TCP is within {BUTTON_PRESS_XY_RADIUS} m horizontally of the "
    "button's centre and no more than {BUTTON_PRESS_Z_MARGIN - BUTTON_CAP_TRAVEL * BUTTON_PRESS_EVENT_RATIO} m "
    "above its unpressed cap top, after the TCP was at least "
    "{BUTTON_PRESS_Z_MARGIN - BUTTON_CAP_TRAVEL * BUTTON_RELEASE_READY_RATIO} m above the cap top "
    "or outside that radius since the previous press"
)
INSERT = (
    "a battery that the gripper is not grasping rests with its centre within "
    "{SOCKET_INSERT_XY_TOL} m horizontally and {SOCKET_INSERT_Z_TOL} m vertically of the socket slot"
)
HOME = (
    "every checked battery is within {SLOT_RETURN_XY_TOL * 3} m horizontally and "
    "{SLOT_RETURN_Z_TOL * 3} m vertically of its own tray slot"
)
TESTS.update(
    {
        "RememberShapeVLABaseEnv": HEAD + TOUCH_REACH.replace("{target}", "object whose shape matches the cue"),
        "RememberShapeAndColorVLABaseEnv": HEAD
        + TOUCH_REACH.replace("{target}", "object whose shape and colour both match the cue"),
        "FindImposterShapeVLABaseEnv": HEAD
        + "after all objects reappear, "
        + TOUCH_REACH.replace("{target}", "object whose shape was absent from the cue"),
        "FindImposterShapeAndColorVLABaseEnv": HEAD
        + "after all objects reappear, "
        + TOUCH_REACH.replace("{target}", "object whose shape-and-colour combination was absent from the cue"),
        "ShellGameColorLampTouchVLABaseEnv": HEAD
        + "after the coloured lamp lights up, success at the first step where the TCP is within "
        "{GOAL_THRESH} m of the reference point of the cup covering the ball whose colour matches "
        "the lamp (about 0.053 m above the table; the cups stay fixed in place) while " + STATIC + "." + NO_FAIL,
        "ShellGameShuffleTouchVLABaseEnv": HEAD
        + "success at the first step where the TCP is within {GOAL_THRESH} m of the reference point "
        "of the cup hiding the ball (about 0.053 m above the table) while " + STATIC + " and no cup "
        "has moved more than {MUG_DISPLACEMENT_SUCCESS_THRESH} m horizontally since the shuffle "
        "ended." + NO_FAIL,
        "ShellGameShuffleColorLampTouchVLABaseEnv": HEAD
        + "after the shuffle, success at the first step where the TCP is within {GOAL_THRESH} m of "
        "the reference point of the cup covering the ball whose colour matches the lamp (about "
        "0.053 m above the table; the cups stay fixed in place) while " + STATIC + "." + NO_FAIL,
        "ShellGamePushVLABaseEnv": HEAD
        + "success at the first step where the reference point of the cup hiding the ball is within "
        "{GOAL_THRESH * 1.6} m (straight-line distance) of an unseen goal point 0.1 m further along "
        "+x (away from the robot base) than the ball's starting centre and {GOAL_THRESH / 2} m "
        "higher, while the arm is static (every arm joint speed at most 0.3 rad/s)." + NO_FAIL,
        "InterceptVLABaseEnv": HEAD
        + "success at the first step where the ball's centre is within {GOAL_RADIUS} m horizontally "
        "of the target's centre." + NO_FAIL,
        "InterceptGrabVLABaseEnv": HEAD
        + "success at the first step where the gripper grasps the ball (each finger pushes on it "
        "with at least 0.5 N, within 85 degrees of that finger's closing direction) while " + STATIC + "." + NO_FAIL,
        "RotateLenientVLABaseEnv": HEAD
        + "success at the first step where the peg's rotation about the vertical axis since the "
        "start (the Z angle of its XYZ Euler decomposition, wrapped to +/-180 degrees; positive is "
        "counter-clockwise seen from above) is within 0.1 rad of the instructed angle while " + STATIC + "." + NO_FAIL,
        "RotateStrictVLABaseEnv": HEAD
        + "success at the first step where the peg's rotation about the vertical axis since the "
        "start (the Z angle of its XYZ Euler decomposition, wrapped to +/-180 degrees; positive is "
        "counter-clockwise seen from above) is within 0.1 rad of the instructed angle, the peg's "
        "centre is within 0.05 m of its starting position along both x and y, and " + STATIC + "." + NO_FAIL,
        "TakeItBackVLABaseEnv": HEAD
        + "the goal counts as reached once the cube's centre is within {GOAL_RADIUS} m horizontally "
        "of the first target's centre, and this is remembered. Success at the first later step where "
        "the cube's centre is within {GOAL_RADIUS} m horizontally of the centre of the region it "
        "started on." + NO_FAIL,
        "SeqOfColorsVLABaseEnv": HEAD + ORDERED,
        "ChainOfColorsVLABaseEnv": HEAD + ORDERED,
        "GatherAndRecallVLABaseEnv": HEAD
        + "a cube is on the disc while its centre is within {DISC_ON_THRESH} m horizontally of the "
        "disc's centre and lower than 0.5 m, and it moves slower than {CUBE_VEL_THRESH} m/s. A button "
        "press registers when the TCP is within {BUTTON_PRESS_XY_RADIUS} m horizontally of a "
        "button's centre and no more than "
        "{BUTTON_PRESS_Z_MARGIN - BUTTON_CAP_TRAVEL * BUTTON_PRESS_EVENT_RATIO} m above its "
        "unpressed cap top, and only while every cube is on the disc. The first registered press "
        "ends the episode: success if that button's colour matches the lamp flash, failure otherwise.",
        "TimedTransferVLABaseEnv": HEAD
        + "the cube is on the red disc while its centre is within {GOAL_THRESH} m horizontally of "
        "the red disc's centre and less than {DISC_HALF_HEIGHT * 2 + CUBE_HALF_SIZE * 3} m above the "
        "table. Counting control steps from the step the lamp turns green, success at the first "
        "step from {int(DELAY_STEPS * (1 - TOLERANCE_FRAC))} to {int(DELAY_STEPS * (1 + TOLERANCE_FRAC))} "
        "(inclusive) on which the cube is on the red disc. The cube on the red disc earlier (after "
        "the lamp turned green) fails the episode at once, and so does reaching step "
        "{int(DELAY_STEPS * (1 + TOLERANCE_FRAC)) + 1} without success.",
        "BatteriesCheckerVLABaseEnv": HEAD
        + "each check is a cycle of three events, in this order. Insert: "
        + INSERT
        + ". Return: that battery, not grasped and out of the socket, is within "
        "{SLOT_RETURN_XY_TOL} m horizontally and {SLOT_RETURN_Z_TOL} m vertically of its own tray "
        "slot. Confirm: "
        + BATTERY_PRESS
        + ". Success at the first step where every battery has completed a cycle and "
        + HOME
        + "." + NO_FAIL,
        "BatteriesCheckerEasyVLABaseEnv": HEAD
        + "each check is a cycle of two events, in this order. Insert: "
        + INSERT
        + "; the environment then returns that battery to its tray slot. Confirm: "
        + BATTERY_PRESS
        + ". Success at the first step where every battery has completed a cycle and "
        + HOME
        + "." + NO_FAIL,
        "TraceShapeVLABaseEnv": HEAD
        + "only after the lamp turns green. The demonstrated contour carries {NUM_CHECKPOINTS} "
        "checkpoints spaced along the red cube's path, the first at its starting point. A checkpoint "
        "is visited when the green cube's centre comes within {CHECKPOINT_THRESH} m of it "
        "horizontally (any order; visits are remembered). Success at the first step where every "
        "checkpoint has been visited and the green cube's centre is within {CHECKPOINT_THRESH} m "
        "horizontally of the first checkpoint." + NO_FAIL,
        "TraceShapeSeqVLABaseEnv": HEAD
        + "only after the lamp turns green. Each demonstrated contour carries {NUM_CHECKPOINTS} "
        "checkpoints spaced along the red cube's path, the first at its starting point. Contours are "
        "checked one at a time in demonstration order: a checkpoint of the current contour is "
        "visited when the green cube's centre comes within {CHECKPOINT_THRESH} m of it horizontally "
        "(any order), and the contour is closed when all its checkpoints are visited and the green "
        "cube's centre is within {CHECKPOINT_THRESH} m of its first checkpoint; then the next "
        "contour starts. A button press registers when the TCP is within {BUTTON_PRESS_XY_RADIUS} m "
        "horizontally of the button's centre and no more than "
        "{BUTTON_PRESS_Z_MARGIN - BUTTON_CAP_TRAVEL * BUTTON_PRESS_EVENT_RATIO} m above its "
        "unpressed cap top. The first press ends the episode: success if every contour has been "
        "closed, failure otherwise.",
    }
)
NOT_CANONICAL = {"ShellGamePickVLABaseEnv"}  # registered, but not one of the 90 benchmark tasks

# Measured, not a class constant: ShellGame cup reference point height (025_mug at
# MUG_SCALE 1.3), from env.unwrapped.object_zs after reset (0.0529 m).


def main():
    from cam.benchmarks import mikasa_robo as m

    source = ROOT / "third_party/mikasa_robo"
    chains = m.task_classes(source)
    owners = {}
    for task, chain in chains.items():
        owner = next((c.name for c in chain if "evaluate" in c.methods), None)
        owners.setdefault(owner, chain)
    out = {}
    for name, text in TESTS.items():
        chain = owners[name]
        out[name] = {"sha256": m.evaluation_digest(chain, name), "text": text}
    path = m.METADATA / "success_tests.json"
    path.write_text(
        json.dumps(
            {
                "source": "third_party/mikasa_robo @ 509b875f3d207c287497c0a897661062de928bb0",
                "digest": "sha256 of every evaluate() in the class chain, then the first step() and __init__ overrides, joined by '\\n#--\\n'",
                "classes": out,
            },
            indent=1,
        )
        + "\n"
    )
    missing = sorted(o for o in owners if o and o not in TESTS and o not in NOT_CANONICAL)
    print(json.dumps({"pinned": sorted(out), "classes_without_test": missing}, indent=1))


if __name__ == "__main__":
    main()
