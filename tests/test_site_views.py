"""Rollout traces (annotated explorer) and chronological development logs."""

import json

import h5py
import numpy as np

from cam.io import write
from cam.reporting.devlog import build_timeline
from cam.reporting.trace import build_trace, cached_poster, cached_trace


def _rollout(folder, frames=4):
    folder.mkdir(parents=True)
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    E = np.hstack([np.eye(3), [[0.0], [0.0], [1.0]]])  # camera 1 m in front of the origin
    with h5py.File(folder / "observations.h5", "w") as h5:
        h5["step"] = np.arange(frames)
        h5["phase"] = np.array([0] + [1] * (frames - 1))
        h5["front_rgb"] = np.zeros((frames, 48, 64, 3), np.uint8)
        h5["wrist_rgb"] = np.zeros((frames, 48, 64, 3), np.uint8)
        h5["front_depth"] = np.zeros((frames, 48, 64), np.int16)
        h5["front_K"] = np.repeat(K[None], frames, 0)
        h5["front_extrinsic"] = np.repeat(E[None], frames, 0)
        tcp = np.zeros((frames, 6))
        tcp[:, 0] = np.linspace(0, 0.1, frames)  # moves +x → image right
        tcp[-1, 2] = -2.0  # behind the camera in the last frame
        h5["tcp"] = tcp
        h5["joints"] = np.zeros((frames, 7))
    with open(folder / "actions.jsonl", "w") as stream:
        for i in range(frames - 1):
            stream.write(json.dumps({"action": [i, 0, 0, 0, 0, 0, 1]}) + "\n")
    rows = [
        {"kind": "decision", "row": {"step": 1, "act": {"tool": "act_a", "args": {"x": 1}}, "memory_in": {"m": None}, "memory_written": {"m": 1}}},
        {"kind": "outcome", "output": "moved"},
        {"kind": "decision", "row": {"step": 2, "act": {"tool": "act_b", "args": {}}, "memory_in": {"m": 1}, "memory_written": {"m": 1}}},
    ]
    (folder / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write(folder / "result.json", {"task": "T", "episode": 0, "status": "success", "steps": frames - 1})


def test_trace_aligns_frames_projects_tcp_and_hides_unknown_state(tmp_path):
    folder = tmp_path / "ep000"
    _rollout(folder)
    interface = {"video": ["front_rgb", "wrist_rgb"], "action": {"names": list("abcdefg")}}
    trace = build_trace(folder, interface)
    assert trace["frames"] == 4 and len(trace["actions"]) == 3
    assert [t["x"] for t in trace["tiles"]] == [0, 64] and trace["video_width"] == 128
    # Images, depth and calibration are not "state"; small vectors are.
    assert set(trace["state"]) == {"tcp", "joints"}
    uv = trace["points"]["front_rgb"][0]["uv"]
    assert uv[0] == [32.0, 24.0] and uv[1][0] > 32  # principal point, then moving right
    assert uv[3] is None  # behind the camera: no marker
    assert "wrist_rgb" not in trace["points"]  # no wrist calibration, no projection
    # Decisions start at the first execution frame with their step; memory diffs only.
    assert [d["frame"] for d in trace["decisions"]] == [1, 2]
    assert trace["decisions"][0]["memory"] == [["m", "1"]] and trace["decisions"][1]["memory"] == []
    assert trace["decisions"][0]["outcome"] == "moved"
    assert trace["decision"] == [-1, 0, 1, 1]
    path = cached_trace(folder, tmp_path / "site", interface)
    assert path == cached_trace(folder, tmp_path / "site", interface)  # cached by source stamps
    assert json.loads(path.read_text())["frames"] == 4


def test_replay_labels_and_preview_come_from_recorded_data(tmp_path):
    from PIL import Image

    folder = tmp_path / "episode"
    _rollout(folder)
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
    decisions[0]["row"]["memory_written"].update(phase=2, note="target located", internal_vector=[1, 2, 3])
    (folder / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions))
    interface = {"video": ["front_rgb", "wrist_rgb"]}
    trace = build_trace(folder, interface)
    assert trace["decisions"][0]["labels"] == {"phase": "2", "note": "target located"}
    assert trace["decisions"][1]["labels"] == {}
    with h5py.File(folder / "observations.h5", "a") as h5:
        h5["front_rgb"][0] = np.full((48, 64, 3), [255, 0, 0], np.uint8)
        h5["wrist_rgb"][0] = np.full((48, 64, 3), [0, 255, 0], np.uint8)
    path = cached_poster(folder, tmp_path / "site", interface)
    with Image.open(path) as poster:
        assert poster.size == (128, 48)
        assert poster.getpixel((20, 20))[0] > 230
        assert poster.getpixel((90, 20))[1] > 230
    assert cached_poster(folder, tmp_path / "site", interface) == path


def test_timeline_interleaves_messages_calls_images_revisions_and_summary(tmp_path):
    task = tmp_path / "task"
    audit = task / "audit"
    (audit / "objects").mkdir(parents=True)
    (audit / "file_manifests").mkdir()
    (audit / "objects" / "abc").write_bytes(b"\x89PNG")
    calls = [
        {"id": "call_000001", "tool": "command", "arguments": {"command": "ls"}, "started": 10.0, "finished": 11.0},
        {"id": "call_000002", "tool": "read_image", "arguments": {"path": "/workspace/scratch/a.png"}, "started": 12.0, "finished": 12.5},
        {"id": "call_000003", "tool": "rollout", "arguments": {}, "started": 13.0, "finished": 20.0},
    ]
    for call in calls:
        write(audit / f"{call['id']}.json", call)
        (audit / f"{call['id']}.stdout").write_text("")
    # Host scheduler sidecars must not shift the transcript-to-call pairing.
    write(audit / "call_000001_gpu.json", {"gpu": 1, "acquired": 10.2})
    write(audit / "file_manifests/1.json", {"reason": "after call_000002", "files": {"scratch/a.png": {"sha256": "abc"}}})
    revisions = task / "history/revisions"
    revisions.mkdir(parents=True)
    write(revisions / "000001.json", {"id": 1, "reason": "command call_000001", "created": 10.5, "attempts": 0, "digest": "d" * 64, "path": str(task)})
    (revisions / "000001.diff").write_text("--- before/tools/policy.py\n+++ after/tools/policy.py\n+x\n")
    write(revisions / "000002.json", {"id": 2, "reason": "checkpoint", "created": 21.0, "attempts": 1, "digest": "e" * 64, "path": str(task)})
    (revisions / "000002.diff").write_text("")
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Plan"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u1", "name": "mcp__cam__command", "input": {"command": "ls"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "u1", "content": [{"type": "text", "text": json.dumps({"returncode": 0, "stdout": "a\n", "stderr": ""})}]}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u2", "name": "mcp__cam__read_image", "input": {"path": "/workspace/scratch/a.png"}}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "u3", "name": "mcp__cam__rollout", "input": {}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "u3", "content": [{"type": "text", "text": json.dumps({"attempt": 1, "episode": 2, "status": "success", "steps": 5})}]}]}},
        {"type": "result", "subtype": "success", "result": "Done", "total_cost_usd": 1.5, "num_turns": 4, "duration_ms": 9000},
    ]
    log = task / "logs/session_000/events.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("".join(json.dumps(e) + "\n" for e in events))
    registered = {}

    def asset(path, suffix=None):
        if not path.is_file():
            return None
        registered[str(path)] = suffix
        return "artifact/" + path.name + (suffix or "")

    timeline = build_timeline(task, asset, [{"attempt": 1, "trace": "traces/x.json", "video": "artifact/v.mp4"}])
    kinds = [e["kind"] for e in timeline]
    assert kinds == ["session", "thinking", "text", "tool", "tool", "tool", "summary", "revision"]
    command, image, rollout = [e for e in timeline if e["kind"] == "tool"]
    assert command["returncode"] == 0 and command["output"] == "a\n"
    assert [r["id"] for r in command["revisions"]] == [1] and command["revisions"][0]["changed"]
    # read_image resolves to the archived object, served with the original suffix.
    assert image["image"] == "artifact/abc.png" and registered[str(audit / "objects/abc")] == ".png"
    assert rollout["rollout"]["status"] == "success" and rollout["rollout_trace"] == "traces/x.json"
    assert timeline[1]["redacted"] and timeline[-2]["cost"] == 1.5
    assert timeline[-1]["reason"] == "checkpoint" and not timeline[-1]["changed"]


def test_live_batch_keeps_executed_policy_separate_from_later_revision(run_factory, tmp_path):
    from cam.core.budget import Ledger
    from cam.core.history import snapshot
    from cam.io import read
    from cam.reporting.site import build_site

    root = run_factory("interaction", 2)
    task = root / "tasks/BinFill"
    config = read(task / "state/config.json")
    ledger = Ledger(config["ledger"])
    episode = config["train_episodes"][0]
    original = (task / "program/tools/policy.py").read_text()
    attempt = ledger.reserve(config, episode)
    ledger.update(attempt, steps=3, step_calls=3, status="fail")
    write(task / "rollouts/train/R001/result.json", {
        "attempt": attempt, "episode": episode, "status": "fail", "steps": 3,
        "program_hash": ledger.rows()[0]["program_hash"],
    })
    # The next result may still be being published by the broker.
    write(task / "rollouts/train/R002/result.json", {"status": "fail"})
    (task / "program/tools/policy.py").write_text(original + "\n# later revision\n")
    snapshot(config, "updated after failure")
    write(task / "audit/call_000001_gpu.json", {"gpu": 0})
    batch = tmp_path / "batch"
    write(batch / "status.json", {"status": "running", "workers": 8, "tasks": {
        "robomme/BinFill": {"benchmark": "robomme", "task": "BinFill", "status": "learning", "run": str(root)},
        "robomme/Other": {"benchmark": "robomme", "task": "Other", "status": "extracting"},
    }})
    site = tmp_path / "site"
    build_site([], [], site, batch)
    data = read(site / "data.json")
    assert len(data["batch"]["tasks"]) == 2 and len(data["runs"]) == 1
    assert read(site / ".sources.json")["batch"] == str(batch)
    dev = read(site / data["runs"][0]["development"][0]["development_url"])
    assert dev["calls"] == [] and len(dev["rollouts"]) == 1
    assert data["runs"][0]["development"][0]["reference_demos"] == []
    assert read(site / ".demo_sources.json") == {}
    rollout = dev["rollouts"][0]
    assert rollout["policy_hash_verified"]
    registry = read(site / ".artifacts.json")
    from pathlib import Path
    executed = Path(registry[rollout["policy_files"]["tools/policy.py"].split("/")[-1]])
    latest = Path(registry[dev["revisions"][-1]["files"]["tools/policy.py"].split("/")[-1]])
    assert executed.parent.parent.name == "R000"
    assert executed.read_text() == original
    assert "later revision" in latest.read_text()
    assert dev["final_files"] == {}
    # A stale freeze manifest must never label altered code as the executed policy.
    executed.write_text(original + "\n# damaged checkpoint\n")
    build_site([], [], site, batch)
    damaged = read(site / data["runs"][0]["development"][0]["development_url"])["rollouts"][0]
    assert not damaged["policy_hash_verified"] and damaged["policy_files"] == {}


def test_reference_videos_use_only_assigned_demo_groups_without_mutation(run_factory, tmp_path):
    import imageio.v2 as imageio
    from pathlib import Path
    from cam.io import read, sha256
    from cam.reporting.reference import encode_video, mosaic
    from cam.reporting.site import build_site

    root = run_factory()
    site = tmp_path / "site"
    build_site([root], [], site)
    refs = read(site / "data.json")["runs"][0]["development"][0]["reference_demos"]
    assigned = read(root / "inputs/dataset.json")["tasks"]["BinFill"]["episodes"]
    assert len(refs) == 3
    assert {r["episode"] for r in refs} == {r["episode"] for r in assigned}
    sources = read(site / ".demo_sources.json")
    assert len(sources) == 3
    key = refs[0]["video"].split("/")[-1].removesuffix(".mp4")
    spec = sources[key]
    assert Path(spec["source"]).parent == root / "tasks/BinFill/demonstrations"
    original = sha256(spec["source"])
    video = encode_video(spec, site, key)
    with imageio.get_reader(video) as reader:
        assert reader.count_frames() == spec["frames"] == 3
        assert reader.get_meta_data()["fps"] == 20
        assert reader.get_data(0).shape == (16, 32, 3)
    assert sha256(spec["source"]) == original
    stamp = video.stat().st_mtime_ns
    assert encode_video(spec, site, key) == video and video.stat().st_mtime_ns == stamp
    # Four camera recordings retain the native two-by-two preview layout.
    frames = [np.full((4, 6, 3), i, np.uint8) for i in range(4)]
    tile = mosaic(frames, 2)
    assert tile.shape == (8, 12, 3)
    assert [int(tile[y, x, 0]) for y, x in [(0, 0), (0, 6), (4, 0), (4, 6)]] == list(range(4))
