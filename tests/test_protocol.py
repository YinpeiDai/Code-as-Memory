import json
import pytest
from cam.core.budget import Ledger, program_hash
from cam.core.history import freeze, select_checkpoint
from cam.harness.prepare import validate_run
from cam.harness.service import Service
from cam.io import read


def test_attempts_checkpoints_and_freeze(run_factory):
    root = run_factory("interaction", 2)
    task = root / "tasks/BinFill"
    cfg = read(task / "state/config.json")
    ledger = Ledger(cfg["ledger"])
    original = program_hash(task / "program")
    with pytest.raises(ValueError, match="next assigned"):
        ledger.reserve(cfg, 3)
    first = ledger.reserve(cfg, 0)
    assert read(task / "checkpoints/R000/freeze.json")["program_hash"] == original
    with pytest.raises(ValueError, match="already running"):
        ledger.reserve(cfg, 3)
    ledger.update(first, steps=0, step_calls=0, status="harness_error")
    (task / "program/policy.md").write_text("Revised after first attempt")
    first_revised = program_hash(task / "program")
    second = ledger.reserve(cfg, 3)
    assert read(task / "checkpoints/R001/freeze.json")["program_hash"] == first_revised
    ledger.update(second, steps=5, step_calls=2, status="fail")
    (task / "program/policy.md").write_text("Revised after second attempt")
    freeze(cfg)
    assert select_checkpoint(task, 1)[1]["program_hash"] == first_revised
    assert select_checkpoint(task, 20)[1]["checkpoint"] == 2
    with pytest.raises(ValueError, match="sealed"):
        ledger.reserve(cfg, 6)
    assert len(ledger.rows()) == 2


def test_demo_mode_cannot_reset(run_factory):
    root = run_factory()
    cfg = read(root / "tasks/BinFill/state/config.json")
    with pytest.raises(ValueError, match="exhausted"):
        Ledger(cfg["ledger"]).reserve(cfg, 0)
    service = Service(root, "BinFill", 0)
    try:
        result = service.call("rollout", {})
        assert result["isError"]
        assert not Ledger(cfg["ledger"]).rows()
        assert not (root / "tasks/BinFill/rollouts").exists()
    finally:
        service.lock.close()


def test_pin_integrity_and_mode_mounts(run_factory):
    root = run_factory("interaction", 1)
    assert not (root / "tasks/BinFill/demonstrations").exists()
    assert not (root / "runtime/cam/runtime/trajectory.py").exists()
    validate_run(root, full_data=True)
    (root / "inputs/harness/cam/config.py").write_text("changed")
    with pytest.raises(ValueError, match="Pinned source changed"):
        validate_run(root)


@pytest.mark.isolation
def test_mcp_edit_validate_snapshot_preserves_audit(run_factory):
    root = run_factory()
    task = root / "tasks/BinFill"
    service = Service(root, "BinFill", 0)
    try:
        result = service.call(
            "command",
            {"command": "printf 'Offline evidence review\\n' > task_analysis.md"},
        )
        assert not result.get("isError")
        assert json.loads(result["content"][0]["text"])["returncode"] == 0
        validation = service.call("validate", {})
        assert json.loads(validation["content"][0]["text"])["valid"]
        assert (
            task / "program/task_analysis.md"
        ).read_text() == "Offline evidence review\n"
        assert len(list((task / "history/revisions").glob("*.json"))) >= 3
        assert (task / "audit/call_000001.stdout").exists()
        assert (task / "audit/call_000001.strace").exists()
        assert list((task / "audit/file_manifests").glob("*.json"))
    finally:
        service.lock.close()


def test_demo_count_does_not_expand_allowed_interaction_seeds(raw, tmp_path):
    from cam.config import RunConfig
    from cam.data.extract import extract
    from cam.harness.prepare import prepare

    data, source = raw
    dataset = tmp_path / "nine_demos"
    extract(data, dataset, count=9, benchmark_source=source, workers=1, videos=False)
    root = tmp_path / "both"
    prepare(
        root,
        dataset,
        RunConfig(benchmark="robomme", mode="both", budget=20, both_protocol="v1"),
        benchmark_source=source,
    )
    cfg = read(root / "tasks/BinFill/state/config.json")
    plan = read(root / "run.json")
    assert cfg["train_episodes"] == [0, 3, 6]
    assert plan["demonstrations_per_task"]["BinFill"] == 9
    assert (
        len(
            read(root / "inputs/metadata/train/record_dataset_BinFill_metadata.json")[
                "records"
            ]
        )
        == 3
    )
    prompt = (root / "tasks/BinFill/prompt.md").read_text()
    assert "Assigned expert demonstrations: 9" in prompt
    assert "Allowed simulator training episode IDs: [0, 3, 6]" in prompt
