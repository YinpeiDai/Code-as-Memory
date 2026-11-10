"""Directory indexing retains coverage, evidence boundaries and portable links."""
import shutil

from cam.io import write
from cam.reporting.catalog import index_directory


def test_sanity_groups_live_counts_and_portable_ids(tmp_path):
    root = tmp_path / "runs"
    batch = root / "eval_sanity_check/batch1"
    write(batch / "status.json", {"schema": "cam-solution-check/2", "status": "running",
        "expected": 10, "completed": 2, "tasks": {"robomme/demo_only_25/A": {"status": "running"}}})
    for task in ("A", "B"):
        run = batch / "runs" / task / ("tag_" + task)
        write(run / "run.json", {"config": {"benchmark": "robomme", "budget": 0}, "tasks": [task],
            "test_episodes": {task: list(range(50))}, "evaluation_only": True})
        write(run / "inputs/published_solution.json", {"experiment": "demo_only_25"})
        write(run / "sanity_check.json", {"expected": 5, "batch": str(batch)})
        write(run / "tasks" / task / "rollouts/test/R000/ep000/result.json",
              {"episode": 0, "status": "success", "success": True, "scorable": True})
        write(run / "tasks" / task / "policy/freeze.json", {})
        # A failed infrastructure attempt must not be counted again.
        write(run / "tasks" / task / "rollouts/infrastructure_retries/R000_ep000/result.json",
              {"episode": 0, "status": "harness_error", "scorable": False})
    data = index_directory(root)
    assert len(data["runs"]) == len(data["episodes"]) == 2
    assert len(data["groups"]) == 1
    summary = data["groups"][0]["summary"]
    assert summary["expected"] == 10 and summary["scored"] == 2
    assert summary["success_rate"] is None and not summary["complete"]
    assert data["batches"][0]["expected"] == 10
    assert data["runs"][0]["evaluation_status"] == "running"
    assert all(not e["media_available"] and not e["video"] for e in data["episodes"])
    other = tmp_path / "relocated"
    shutil.copytree(root, other)
    moved = index_directory(other)
    assert [r["id"] for r in moved["runs"]] == [r["id"] for r in data["runs"]]
    assert [r["id"] for r in moved["episodes"]] == [r["id"] for r in data["episodes"]]
    assert (other / "dashboard/PROGRESS.md").exists()


def test_pending_batch_visible_and_bad_results_warn(tmp_path):
    write(tmp_path / "batches/queued/plan.json", {"benchmarks": {"robomme": {"tasks": ["A", "B"]}}})
    write(tmp_path / "batches/queued/status.json", {"status": "running", "tasks": {
        "robomme/A": {"status": "preparing"}, "robomme/B": {"status": "waiting_data"}}})
    data = index_directory(tmp_path)
    assert data["runs"] == []
    assert data["batches"][0]["expected"] == 2
    assert data["batches"][0]["completed"] == 0


def test_all_queued_policies_are_visible_before_preparation(tmp_path):
    tasks = {f'robomme/demo_only_25/Task{i}': {'status':'pending'} for i in range(16)}
    write(tmp_path / 'eval_sanity_check/batch/status.json', {
        'schema':'cam-solution-check/2', 'status':'running', 'tasks':tasks,
        'expected':80, 'completed':0, 'episodes_per_policy':5})
    data = index_directory(tmp_path)
    assert len(data['runs']) == 16
    assert all(r['queued'] and r['evaluation_status']=='queued' for r in data['runs'])
    assert len(data['groups']) == 1
    assert data['groups'][0]['summary']['expected'] == 80
    assert data['groups'][0]['summary']['scored'] == 0
    assert data['groups'][0]['summary']['success_rate'] is None


def test_archived_comparison_results_keep_suite_namespace(tmp_path):
    (tmp_path / "dev_logs").mkdir()
    task = "ShellGameTouch-VLA-v0"
    suite = "mikasa_robo_vla_suite"
    for name, comparison, success in [("mikasa_robo", None, False), (suite, suite, True)]:
        write(tmp_path / "configs/solutions" / name / "demo_interaction_50.json", {
            "schema": "cam-solutions/1", "benchmark": "mikasa_robo", "suite": comparison,
            "experiment": "demo_interaction_50", "tasks": {task: {
                "physics_backend": "gpu" if comparison else "cpu", "archived_result": {"expected": 1}}}})
        write(tmp_path / "eval" / name / "demo_interaction_50" / task / "R050/ep000/result.json",
              {"episode": 0, "success": success, "status": "success" if success else "timeout", "scorable": True})
    indexed = index_directory(tmp_path)
    assert len(indexed["runs"]) == len(indexed["groups"]) == 2
    assert {g["benchmark"]: g["summary"]["successes"] for g in indexed["groups"]} == {"mikasa_robo": 0, suite: 1}
    assert all(r["simulator_benchmark"] == "mikasa_robo" for r in indexed["runs"])
    assert {r["benchmark"]: r["physics_backend"] for r in indexed["runs"]} == {"mikasa_robo": "cpu", suite: "gpu"}
