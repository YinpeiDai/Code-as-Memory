"""A queued development dashboard follows the eventual batch automatically."""

from cam.io import read, write
from cam.reporting.site import build_site


def test_site_follows_queue_and_retains_source_for_refresh(tmp_path):
    queue = tmp_path / "queue"
    site = tmp_path / "site"
    write(queue / "status.json", {"status": "waiting_for_validated_pool"})
    write(queue / "plan.json", {"workers": 8})
    write(queue / "experiment.json", {"benchmarks": {
        "rmbench": {"tasks": ["battery_try", "cover_blocks"]}}})
    build_site([], [], site, queue)
    data = read(site / "data.json")
    assert data["batch"]["workers"] == 8 and len(data["batch"]["tasks"]) == 2
    assert all(row["status"] == "waiting_for_validated_pool" for row in data["batch"]["tasks"])
    assert data["runs"] == []
    batch = tmp_path / "batch"
    write(queue / "batch.json", {"batch": str(batch)})
    # The supervisor may not have published its initial task status yet.
    build_site([], [], site, queue)
    assert len(read(site / "data.json")["batch"]["tasks"]) == 2
    write(batch / "status.json", {"status": "running", "workers": 8, "tasks": {
        "rmbench/battery_try": {"benchmark": "rmbench", "task": "battery_try", "status": "preparing"}}})
    build_site([], [], site, queue)
    assert read(site / "data.json")["batch"]["tasks"][0]["status"] == "preparing"
    assert read(site / ".sources.json")["batch"] == str(queue)


def test_queue_failure_is_visible_without_a_run(tmp_path):
    queue = tmp_path / "queue"
    write(queue / "status.json", {"status": "needs_attention", "error": "Pool validation failed"})
    write(queue / "plan.json", {"workers": 8})
    write(queue / "experiment.json", {"benchmarks": {"rmbench": {"tasks": ["battery_try"]}}})
    build_site([], [], tmp_path / "site", queue)
    row = read(tmp_path / "site/data.json")["batch"]["tasks"][0]
    assert row["status"] == "preparation_failed" and row["error"] == "Pool validation failed"
