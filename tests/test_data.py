from collections import Counter
import h5py
import numpy as np
import pytest
from cam.data.extract import extract, validate_dataset
from cam.interface.robomme.data import select
from cam.io import read, sha256, write
from cam.runtime import trajectory


def test_balanced_25():
    records = [
        {
            "episode": i,
            "seed": 1000 + i,
            "difficulty": "easy" if i < 50 else "medium" if i < 75 else "hard",
        }
        for i in range(100)
    ]
    chosen = select(records, 25)
    assert Counter(r["difficulty"] for r in chosen) == {
        "easy": 9,
        "medium": 8,
        "hard": 8,
    }
    assert [r["episode"] for r in chosen if r["difficulty"] == "easy"] == [
        0,
        6,
        12,
        18,
        24,
        31,
        37,
        43,
        49,
    ]
    assert select(records, "all") == records


def test_raw_validation_and_lossless_fields(dataset, monkeypatch):
    root, _ = dataset
    plan = validate_dataset(root)
    assert [r["episode"] for r in plan["tasks"]["BinFill"]["episodes"]] == [0, 3, 6]
    monkeypatch.setattr(trajectory, "ROOT", root / "h5")
    with trajectory.Trajectory("BinFill", 0) as tr:
        assert tr.T == 3 and tr.exec_start == 1
        assert tr.front_depth(1).dtype == np.int16
        assert tr.front_depth(1)[0, 0] == 1234
        np.testing.assert_array_equal(tr.tcp(1), [0.4, 0.2, 0.3, 0, 1.5, 0])
        assert tr.waypoint_action(0) is None
        np.testing.assert_array_equal(tr.waypoint_action(1), np.arange(7) / 7)
    with h5py.File(root / "h5/record_dataset_BinFill.h5") as h5:
        names = []
        h5.visit(names.append)
        assert not any(
            any(s in p for s in ("oracle", "subgoal", "completed", "seed"))
            for p in names
        )


def test_mismatched_training_seed_is_rejected(raw, tmp_path):
    raw_data, source = raw
    with h5py.File(raw_data / "record_dataset_BinFill.h5", "a") as h5:
        h5["episode_8/setup/seed"][...] = 999999
    with pytest.raises(ValueError, match="seed/difficulty"):
        extract(
            raw_data,
            tmp_path / "bad",
            benchmark_source=source,
            count=3,
            workers=1,
            videos=False,
        )


def test_external_links_are_rejected(dataset, tmp_path):
    root, _ = dataset
    path = root / "h5/record_dataset_BinFill.h5"
    with h5py.File(path, "a") as h5:
        del h5["episode_0"]
        h5["episode_0"] = h5py.ExternalLink(str(tmp_path / "other.h5"), "/secret")
    plan = read(root / "manifest.json")
    plan["tasks"]["BinFill"]["sha256"] = sha256(path)
    write(root / "manifest.json", plan)
    with pytest.raises(ValueError, match="links"):
        validate_dataset(root)


def test_extraction_resume_reuses_only_verified_files(dataset, raw):
    root, source = dataset
    data, _ = raw
    before = read(root / "manifest.json")["tasks"]["BinFill"]["sha256"]
    resumed = extract(
        data,
        root,
        count=3,
        benchmark_source=source,
        workers=1,
        videos=False,
        resume=True,
    )
    assert resumed["tasks"]["BinFill"]["sha256"] == before
    audit = read(root / "audit/BinFill.json")
    assert len(audit["all_training_episodes"]) == 9
    with pytest.raises(ValueError, match="configuration changed"):
        extract(
            data,
            root,
            count=6,
            benchmark_source=source,
            workers=1,
            videos=False,
            resume=True,
        )


def test_historical_robomme_package_uses_explicit_metadata_compatibility(dataset, monkeypatch):
    from cam.interface.registry import get_adapter
    from cam.interface.robomme import data as module
    from cam.data.extract import load_dataset_manifest
    root, source = dataset
    adapter = get_adapter('robomme', source)
    monkeypatch.setattr(module, 'get_adapter', lambda name: adapter)
    manifest = read(root / 'manifest.json')
    manifest['schema'] = 'cap-regular-extraction/2'
    item = manifest['tasks']['BinFill']
    for split in ('train', 'test'):
        item[split + '_metadata_sha256'] = sha256(source / 'env_metadata' / split / 'record_dataset_BinFill_metadata.json')
    file = root / item['file']
    with h5py.File(file, 'a') as h:
        for key in list(h.attrs): del h.attrs[key]
        h.attrs.update(schema='cap-regular-fewshot/2', task='BinFill', depth=True, split='train',
            action_alignment='unspecified', step_semantics='recording frame', control_freq=20,
            depth_description='millimetres', excluded_fields='private state', source_format='episode_N/timestep_N')
        for group in h.values():
            for key in ('action_alignment', 'step_semantics'): del group.attrs[key]
            group.create_dataset('tcp_quaternion_wxyz',data=np.zeros((3,4)))
    item['sha256'] = sha256(file)
    write(root / 'manifest.json', manifest)
    current = load_dataset_manifest(root, adapter)
    assert current['compatibility']['h5_reads'] == 0
    assert not current['compatibility']['h5_bytes_changed']
    module.validate_dataset(root)
    monkeypatch.setattr(trajectory, 'ROOT', root / 'h5')
    with trajectory.Trajectory('BinFill',0) as tr:
        assert tr.T == 3
        assert tr.front_depth(1)[0,0] == 1234
    item['test_metadata_sha256'] = '0'*64
    write(root / 'manifest.json',manifest)
    with pytest.raises(ValueError,match='identities changed'):
        load_dataset_manifest(root,adapter)
