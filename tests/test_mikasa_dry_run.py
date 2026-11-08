"""Robot-only motion previews: units, RPC isolation and evidence accounting."""

from io import BytesIO
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cam.benchmarks.mikasa_robo_api import PREVIEW_SPEC
from cam.benchmarks.mikasa_robo_api.preview import absolute_command, joint_report
from cam.runtime.robot import MotionError, Robot
from cam.runtime.wire import receive, send


def test_absolute_world_conversion_with_rotated_base_and_signed_rotation_scale():
    # A 90-degree base yaw maps world +Y to base +X.
    base = [1, 2, 0, np.sqrt(.5), 0, 0, np.sqrt(.5)]
    current = [.2, .1, .3, 1, 0, 0, 0]
    pose = [.9, 2.25, .3, 0, 0, np.pi / 2 + .02]
    command, clipped = absolute_command(pose, base, current, [-.1]*3, [.1]*3, -.1)
    assert np.allclose(command, [.5, 0, 0, 0, 0, -.2], atol=1e-7)
    assert not clipped
    command, clipped = absolute_command([2, 3, 1, np.pi, 0, 0], base, current, [-.1]*3, [.1]*3, -.1)
    assert clipped and np.max(np.abs(command[:3])) <= 1
    assert np.linalg.norm(command[3:]) <= 1 + 1e-12


def test_absolute_same_pose_and_pi_rotation_are_finite():
    rpy = [.4, -.3, .7]
    xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    current = [.2, .1, .3, xyzw[3], *xyzw[:3]]
    command, _ = absolute_command([.2, .1, .3, *rpy], [0, 0, 0, 1, 0, 0, 0], current, [-.1]*3, [.1]*3, -.1)
    assert np.allclose(command, 0, atol=1e-10)
    command, _ = absolute_command([0, 0, 0, np.pi, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [-.1]*3, [.1]*3, -.1)
    assert np.isfinite(command).all() and np.isclose(np.linalg.norm(command[3:]), 1)


def test_joint_target_limits_and_invalid_ik_are_not_reported_safe():
    limits = [[-2, 2], [-3, -.07]]
    valid = joint_report([0, -1], [.2, -.07], limits, ["J1", "J2"])
    assert valid["within_joint_limits"]
    unsafe = joint_report([0, -1], [.2, 6.07], limits, ["J1", "J2"])
    assert not unsafe["within_joint_limits"]
    assert unsafe["violating_joints"] == ["J2"]
    assert unsafe["min_joint_margin_rad"] == pytest.approx(-6.14)
    for target in [None, [np.nan, 0], [np.inf, 0]]:
        result = joint_report([0, -1], target, limits, ["J1", "J2"])
        assert not result["ik_computed"] and not result["within_joint_limits"]
        json.dumps(result, allow_nan=False)


def test_robot_explicit_modes_query_without_advancing_or_adding_frames():
    reader, writer = BytesIO(), BytesIO()
    for _ in range(2):
        send(reader, {"dry_run": {"within_joint_limits": True, "command": [0]*6}})
    reader.seek(0)
    marker = object()
    robot = Robot(marker, reader, writer, {"motion_preview": PREVIEW_SPEC}, max_steps=2)
    robot.dry_run(0, 0, 0, 0, 0, 0, mode="delta")
    robot.dry_run(x=.1, y=.2, z=.3, roll=0, pitch=0, yaw=0, mode="absolute")
    assert robot.now() is marker and robot.frames() == [] and robot._count == 0
    writer.seek(0)
    assert receive(writer) == {"kind": "motion_preview", "mode": "delta", "values": [0]*6}
    assert receive(writer)["values"] == [.1, .2, .3, 0, 0, 0]
    with pytest.raises(MotionError, match="limit exceeded"):
        robot.dry_run(*([0]*6), mode="delta")


@pytest.mark.parametrize("mode,values", [("world", [0]*6), ("delta", [0]*7), ("absolute", [np.nan]*6)])
def test_invalid_client_query_never_reaches_simulator(mode, values):
    writer = BytesIO()
    robot = Robot(None, BytesIO(), writer, {"motion_preview": PREVIEW_SPEC})
    with pytest.raises(MotionError):
        robot.dry_run(*values, mode=mode)
    assert not writer.getvalue()


def test_other_benchmarks_cannot_use_motion_preview():
    writer = BytesIO()
    with pytest.raises(MotionError, match="not available"):
        Robot(None, BytesIO(), writer).dry_run(*([0]*6), mode="delta")
    assert not writer.getvalue()


@pytest.mark.isolation
def test_broker_logs_queries_without_consuming_steps_or_observations(run_factory, tmp_path):
    from test_broker_reporting import Adapter
    from cam.harness.broker import execute
    from cam.io import read

    class PreviewAdapter(Adapter):
        name = "mikasa_robo"

        def action_spec(self, task):
            spec = super().action_spec(task)
            return {**spec, "motion_preview": PREVIEW_SPEC}

        def preview_motion(self, env, values, mode, context):
            return {"within_joint_limits": True, "command": [0]*6, "mode": mode}

    root = run_factory()
    task = root / "tasks/BinFill"
    (task / "program/tools/policy.py").write_text('''
def perceive_decide(obs, goal, memory, last):
    """Exercise only the public robot API."""
    return {"memory": {}, "act": {"tool": "act_check", "args": {}}}

def act_check(robot):
    """Two previews and one actual step."""
    a = robot.dry_run(0, 0, 0, 0, 0, 0, mode="delta")
    b = robot.dry_run(0, 0, 0, 0, 0, 0, mode="absolute")
    robot.step(*a["command"], 1)
    return {"checked": b["within_joint_limits"]}
''')
    plan = read(root / "run.json")
    folder = tmp_path / "motion_preview"
    result = execute(PreviewAdapter(), tmp_path, "BinFill", 0, "train", task / "program", root / "runtime", folder, plan["config"])
    assert result["status"] == "success"
    assert (result["dry_run_calls"], result["step_calls"], result["steps"], result["video_frames"]) == (2, 1, 1, 2)
    logs = [json.loads(line) for line in (folder / "dry_runs.jsonl").read_text().splitlines()]
    assert [row["mode"] for row in logs] == ["delta", "absolute"]
    assert all(row["call"] == 0 and row["step"] == 0 for row in logs)
    # Disable the client guard only in this disposable synthetic runtime to
    # verify the host independently rejects an excess query on the wire.
    client = root / "runtime/cam/runtime/robot.py"
    text = client.read_text()
    assert "if self._dry_runs > self._max_steps:" in text
    client.write_text(text.replace("if self._dry_runs > self._max_steps:", "if False:"))
    limited = {**plan["config"], "max_action_steps": 1}
    result = execute(PreviewAdapter(), tmp_path, "BinFill", 0, "train", task / "program", root / "runtime", tmp_path / "limited", limited)
    assert result["status"] == "policy_error" and result["step_calls"] == 0
    assert result["dry_run_calls"] == 1
    assert "dry_run limit exceeded" in result["error"]

    # A different benchmark cannot enable this host API by advertising it.
    PreviewAdapter.name = "other_benchmark"
    result = execute(PreviewAdapter(), tmp_path, "BinFill", 0, "train", task / "program", root / "runtime", tmp_path / "unsupported", {**limited, "max_decisions": 1})
    assert result["dry_run_calls"] == 0 and result["step_calls"] == 0
    assert result["status"] == "decision_limit"


def test_runtime_copies_public_reference_only_for_mikasa(tmp_path):
    from cam.harness.prepare import make_runtime

    public = make_runtime(tmp_path / "mikasa", mikasa_motion_preview=True)
    doc = public / "cam/runtime/mikasa_robo/dry_run.md"
    assert 'mode="absolute"' in doc.read_text()
    assert sorted(p.name for p in doc.parent.iterdir()) == ["__init__.py", "dry_run.md", "offline.py", "panda_model.json", "safety_check.md"]
    assert not (public / "cam/benchmarks").exists()
    assert not list(public.rglob("preview.py")) and not list(public.rglob("kinematics.py"))
    generic = make_runtime(tmp_path / "other")
    assert not (generic / "cam/runtime/mikasa_robo").exists()
