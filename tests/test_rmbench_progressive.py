"""Local materialization preserves captures and only releases validated tasks."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from cam.benchmarks import rmbench, rmbench_rgbd_data as rgbd
from cam.data import demonstrations
from cam.io import read, sha256, write
from test_rmbench_rgbd import capture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('progressive', ROOT / 'scripts/develop_rmbench_progressively.py')
progressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progressive)


def setup_captures(tmp_path, monkeypatch):
    adapter = rmbench.RMBench()
    records = adapter.records('put_back_block', 'train')[:6]
    original_records = rmbench.RMBench.records
    monkeypatch.setattr(rmbench.RMBench, 'records',
                        lambda self, task, split: records if split == 'train' else original_records(self, task, split))
    pool = tmp_path / 'original'
    for index, record in enumerate(records):
        folder = capture(pool, record, index != 5)
        result = read(folder / 'result.json')
        result['protocol'] = rgbd.PROTOCOL
        write(folder / 'result.json', result)
        write(folder / 'job.json', {**record, 'task': 'put_back_block'})
    job = tmp_path / 'new'
    write(job / 'capture_manifest.json', {
        'schema': 'cam-dataset/1', 'demonstration_schema': demonstrations.SCHEMA,
        'benchmark': 'rmbench', 'depth': True, 'fields': rgbd.fields(),
        'replay': {'protocol': rgbd.PROTOCOL}, 'tasks': {}})
    plan = {'capture_pool': str(pool), 'benchmark_source': str(adapter.source),
            'capture_job': 'original-collection', 'demo_counts': {'put_back_block': 5}}
    return job, plan, pool, records


def test_local_task_is_identical_and_samples_are_recorded(tmp_path, monkeypatch):
    job, plan, pool, records = setup_captures(tmp_path, monkeypatch)
    phases = []
    dataset = progressive.materialize_task(job, plan, 'put_back_block', lambda **x: phases.append(x['status']))
    manifest = rgbd.validate_dataset(dataset)
    assert len(manifest['tasks']['put_back_block']['episodes']) == 5
    assert manifest['tasks']['put_back_block']['replay']['shortfall'] == 1
    sampled = read(dataset / 'sample_inspection.json')
    assert sampled['passed'] and sampled['simulator_replays'] == 0
    assert len(set(sampled['selected_episodes'])) == 5
    assert read(dataset / 'ready.json')['all_capture_hashes_checked']
    assert (dataset / 'sample_inspection.png').exists()
    assert set(phases) == {'staging_captures', 'assembling_data', 'validating_data'}
    source = pool / 'jobs/put_back_block/ep00000/observations.h5'
    assert source.exists()
    assert not (dataset / 'jobs/put_back_block/ep00000/observations.h5').exists()
    with h5py.File(source) as src, h5py.File(dataset / manifest['tasks']['put_back_block']['file']) as dst:
        for name in src['episode_0']:
            assert np.array_equal(src['episode_0'][name][:], dst['episode_0'][name][:])
    prior = sha256(source)
    assert progressive.inspect_samples(dataset, 'put_back_block', manifest['tasks']['put_back_block'])['selected_episodes'] == sampled['selected_episodes']
    assert sha256(source) == prior


def test_changed_capture_never_releases_a_dataset(tmp_path, monkeypatch):
    job, plan, pool, _ = setup_captures(tmp_path, monkeypatch)
    with h5py.File(pool / 'jobs/put_back_block/ep00000/observations.h5', 'r+') as h5:
        h5['episode_0/left_gripper'][0] = 0.5
    with pytest.raises(ValueError, match='changed during local staging'):
        progressive.materialize_task(job, plan, 'put_back_block')
    assert not (job / 'datasets/put_back_block/ready.json').exists()
    assert not (job / 'datasets/put_back_block/manifest.json').exists()
