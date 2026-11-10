"""New runs preserve native behavior while withholding internal planner status."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cam.interface.rmbench import adapter as rmbench
from cam.runtime.observations import Frame
from cam.runtime.robot import EpisodeEnd, MotionError, Robot
from cam.runtime.wire import encoded, receive, send


LEFT = [-0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
RIGHT = [0.2, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]


def action_spec():
    adapter = object.__new__(rmbench.RMBench)
    adapter.limits = {"swap_blocks": 100}
    return adapter.action_spec("swap_blocks")


def observed(left=LEFT, right=RIGHT, step=1):
    return encoded([{
        "phase": "eval", "step": step,
        "left_ee_pose": np.asarray(left), "right_ee_pose": np.asarray(right),
        "left_gripper": np.asarray(0.0), "right_gripper": np.asarray(1.0),
    }])[0]


def public_robot(*results, spec=None):
    reader, writer = BytesIO(), BytesIO()
    for result in results:
        send(reader, result)
    reader.seek(0)
    return Robot(Frame(observed(step=0)), reader, writer,
                 action_spec=action_spec() if spec is None else spec)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_observed_miss_is_recoverable_with_new_frames_and_one_counted_action(arm):
    left, right = LEFT.copy(), RIGHT.copy()
    target = left if arm == "left" else right
    target[2] += 0.2
    robot = public_robot(
        {"status": "running", "frames": [observed()]},
        {"status": "running", "frames": [observed(left, right, step=2)]},
    )
    action = [*left, 0.0, *right, 1.0]
    with pytest.raises(MotionError, match=arm + ": target not reached") as exc:
        robot.step(*action)
    assert "MotionError" in repr(exc.value)  # Existing policies catch by class name.
    other = "right" if arm == "left" else "left"
    assert other + ": target not reached" not in str(exc.value)
    assert robot._count == 1 and robot.now().step == 1 and len(robot.frames()) == 1
    assert float(robot.now().left_gripper) == 0.0  # Partial execution stays visible.
    # The same runtime can execute recovery; it does not roll back or auto-retry.
    assert robot.step(*action).step == 2 and robot._count == 2
    robot._writer.seek(0)
    assert receive(robot._writer) == {"kind": "step", "action": action}
    assert receive(robot._writer) == {"kind": "step", "action": action}
    assert robot._writer.read() == b""


def test_partial_motion_and_both_arms_are_checked_without_ik_or_planner_status():
    left, right = LEFT.copy(), RIGHT.copy()
    left[2] += 0.05
    right[2] += 0.1
    robot = public_robot({"status": "running", "frames": [observed(left, right)]})

    def forbidden(*args, **kwargs):
        raise AssertionError("An observed motion check must not call offline IK")

    robot.dry_run = robot.safety_check = forbidden
    target_left, target_right = LEFT.copy(), RIGHT.copy()
    target_left[2] += 0.2
    target_right[2] += 0.2
    with pytest.raises(MotionError) as exc:
        robot.step(*target_left, 0, *target_right, 1)
    assert "left: target not reached" in str(exc.value)
    assert "right: target not reached" in str(exc.value)
    assert np.allclose(robot.now().right_ee_pose, right)


@pytest.mark.parametrize("position,angle,miss", [
    (0.0, 0.0, False), (0.019, 0.09, False), (0.021, 0.0, True),
    (0.0, 0.11, True), (0.0, np.pi, True),
])
def test_position_and_shortest_rotation_have_independent_tolerances(position, angle, miss):
    left = LEFT.copy()
    left[0] += position
    left[3:] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    robot = public_robot({"status": "running", "frames": [observed(left)]})
    if miss:
        with pytest.raises(MotionError, match="left: target not reached"):
            robot.step(*LEFT, 0, *RIGHT, 1)
    else:
        assert robot.step(*LEFT, 0, *RIGHT, 1).step == 1


def test_unchanged_pose_gripper_only_and_equivalent_quaternions_are_not_failures():
    left, right = LEFT.copy(), RIGHT.copy()
    left[3], right[3] = -2.0, 3.0
    robot = public_robot({"status": "running", "frames": [observed(left, right)]})
    # No motion is needed when already at the target; gripper tracking is separate.
    assert robot.step(*LEFT, 1, *RIGHT, 0).step == 1


@pytest.mark.parametrize("status", ["success", "timeout", "fail", "error"])
def test_terminal_status_wins_even_if_motion_has_not_reached_target(status):
    robot = public_robot({"status": status, "frames": [observed()]})
    target = [*LEFT, 0, *RIGHT, 1]
    target[0] += 1.0
    with pytest.raises(EpisodeEnd) as exc:
        robot.step(*target)
    assert exc.value.status == status
    assert robot._count == 1 and robot.now().step == 1


@pytest.mark.parametrize("fault", ["missing_frame", "missing_pose", "nan_pose", "zero_quaternion"])
def test_missing_or_invalid_feedback_never_silently_accepts_a_target(fault):
    frame = observed()
    if fault == "missing_pose":
        del frame["left_ee_pose"]
    elif fault == "nan_pose":
        frame = observed([np.nan, *LEFT[1:]])
    elif fault == "zero_quaternion":
        frame = observed([*LEFT[:3], 0, 0, 0, 0])
    robot = public_robot({"status": "running", "frames": [] if fault == "missing_frame" else [frame]})
    with pytest.raises(MotionError, match="[Cc]annot verify"):
        robot.step(*LEFT, 0, *RIGHT, 1)
    assert robot._count == 1


def test_action_interface_gates_feedback_and_readme_without_changing_other_benchmarks():
    from cam.harness.run_readme import rmbench_observed_motion_note

    spec = action_spec()
    assert spec["planner_failure_feedback"] is False
    assert "observation-derived" in rmbench_observed_motion_note({"interfaces": {"task": {"action": spec}}})
    del spec["motion_feedback"]
    assert rmbench_observed_motion_note({"interfaces": {"task": {"action": spec}}}) == ""
    robot = public_robot({"status": "running", "frames": [observed()]}, spec=spec)
    target = [*LEFT, 0, *RIGHT, 1]
    target[0] += 1.0
    assert robot.step(*target).step == 1


@pytest.mark.isolation
def test_unchanged_published_policy_catches_observed_motion_error_in_sandbox(tmp_path):
    from cam.harness.isolation import run
    from cam.harness.prepare import make_runtime

    runtime = make_runtime(tmp_path / "runtime", demonstrations=False)
    public = tmp_path / "workspace"
    public.mkdir()
    source = Path(__file__).resolve().parents[1] / "solutions/rmbench/demo_only_50/swap_blocks/policy.py"
    (public / "policy.py").write_bytes(source.read_bytes())
    # Run the exact historical helper inside isolation with synthetic public RPC
    # observations. This verifies compatibility, not native simulator success.
    (public / "probe.py").write_text('''
from io import BytesIO
import math
import runpy
import numpy as np
from cam.runtime.observations import Frame
from cam.runtime.robot import EpisodeEnd, Robot
from cam.runtime.wire import encoded, send

policy = runpy.run_path('/workspace/policy.py', init_globals={'np': np, 'math': math})
home = [0., 0., 0.8, 1., 0., 0., 0.]
target = [0.2, 0., 0.8, 1., 0., 0., 0.]
def frame(right, step):
    return encoded([{'phase': 'eval', 'step': step,
        'left_ee_pose': np.array(home), 'right_ee_pose': np.array(right),
        'left_gripper': np.array(0.), 'right_gripper': np.array(1.)}])[0]
reader, writer = BytesIO(), BytesIO()
for right, status, step in [(home, 'running', 1), (target, 'running', 2), (home, 'success', 3)]:
    send(reader, {'status': status, 'frames': [frame(right, step)]})
reader.seek(0)
spec = {'names': [str(i) for i in range(16)], 'motion_feedback': {
    'schema': 'cam-rmbench-observed-motion/1',
    'position_tolerance_m': 0.02, 'rotation_tolerance_rad': 0.1}}
robot = Robot(Frame(frame(home, 0)), reader, writer, spec)
assert policy['_step_arm'](robot, 'right', target, 0.) is False
assert robot.now().step == 1
assert policy['_step_arm'](robot, 'right', target, 0.) is True
try:
    policy['_step_arm'](robot, 'right', target, 0.)
except EpisodeEnd as exc:
    assert exc.status == 'success'
else:
    raise AssertionError('Terminal status must escape the legacy recovery handler')
assert robot._count == 3 and len(robot.frames()) == 3
print('unchanged policy recovery compatible')
''')
    code, out, err = run(public, runtime, ["python3.11", "/workspace/probe.py"], timeout=30, readonly=True)
    assert code == 0, (out + err).decode(errors="replace")
    assert b"unchanged policy recovery compatible" in out


@pytest.mark.parametrize("native_status", ["Fail", "Success"])
def test_native_planners_are_not_wrapped_and_feedback_is_absent(monkeypatch, native_status):
    calls = []

    def planner(*args):
        calls.append(native_status)
        return {"status": native_status}

    task = SimpleNamespace(robot=SimpleNamespace(left_plan_path=planner, right_plan_path=planner),
                           take_action_cnt=0, eval_success=False, get_obs=lambda: {"observed": True})

    def take_action(values, action_type):
        assert action_type == "ee" and values.dtype == np.float32
        task.robot.left_plan_path(values[:7])
        task.robot.right_plan_path(values[8:15])
        task.take_action_cnt += 1

    task.take_action = take_action
    env = rmbench.NativeEnvironment(task, "goal", 100)
    assert task.robot.left_plan_path is planner and task.robot.right_plan_path is planner
    monkeypatch.setattr(rmbench, "frame", lambda obs, step, depth: {**obs, "step": step})
    adapter = object.__new__(rmbench.RMBench)
    result = adapter.advance(env, [0]*16, {"start": 0}, depth=True)
    assert calls == [native_status, native_status]
    assert env.applied == 1 and env.count == 1
    assert result.status == "running" and result.steps == 1
    assert result.frames == [{"observed": True, "step": 1}]
    assert result.motion_error is None and result.private_error is None
    task.eval_success = True
    assert adapter.advance(env, [0]*16, {"start": 0}, depth=True).status == "success"


def test_native_exceptions_still_distinguish_first_action_infrastructure(monkeypatch):
    def fail(values):
        raise RuntimeError("native exception")

    adapter = object.__new__(rmbench.RMBench)
    env = SimpleNamespace(step=fail, applied=0, count=0)
    with pytest.raises(RuntimeError, match="native exception"):
        adapter.advance(env, [0]*16, {"start": 0}, depth=True)
    env.applied, env.count = 1, 1
    result = adapter.advance(env, [0]*16, {"start": 0}, depth=True)
    assert result.status == "error" and result.motion_error is None
    assert result.private_error["exception_type"] == "RuntimeError"
