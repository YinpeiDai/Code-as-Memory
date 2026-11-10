"""Canonical MIKASA names retain RGB-D contracts without replay dependencies."""
from copy import deepcopy
from pathlib import Path

import pytest

from cam.interface.base import source_hash
from cam.interface.mikasa_robo.data import DECLARED
from cam.interface.registry import get_adapter
from cam.config import RunConfig
from cam.harness.prepare import preparation_inputs, training_schedule
from cam.harness.solutions import load_solutions, record_hash
from cam.io import read, write
from cam.prompts import benchmark_mode_guidance, render

ROOT = Path(__file__).resolve().parents[1]
TASK = 'TakeItBack-VLA-v0'


@pytest.fixture
def adapter():
    if not (ROOT / 'third_party/mikasa_robo/mikasa_robo_suite').is_dir():
        pytest.skip('MIKASA submodule not initialized')
    return get_adapter('mikasa_robo')


def test_default_prompt_and_assignment(adapter):
    cfg = RunConfig.from_dict(read(ROOT / 'configs/mikasa_robo/both.json'))
    assert cfg.both_protocol == 'demo_first' and cfg.setting == 'demo_interaction'
    assert cfg.physics_backend == 'cpu' and cfg.depth and not cfg.builtin_demo
    assert cfg.budget == cfg.training_seed_count == 50
    demos = adapter.records(TASK, 'train')[:50]
    schedule = training_schedule(adapter, TASK, demos, cfg)
    assert {(r['episode'], r['seed']) for r in schedule} == {(r['episode'], r['seed']) for r in demos}
    prompt = render(cfg, TASK, demos, schedule, adapter=adapter, interface=adapter.interface(TASK, True))
    assert benchmark_mode_guidance(cfg)['source'] == 'cam/prompts/mikasa_robo/both.md'
    assert 'v2' not in prompt and 'Physics backend: cpu' in prompt
    assert 'exactly once' in prompt and 'safety_check' in prompt
    with pytest.raises(ValueError, match='demo-first'):
        RunConfig(benchmark='mikasa_robo', mode='both', budget=50, both_protocol='v1')


@pytest.mark.parametrize('dataset_name', ['mikasa_robo', 'mikasa_robo_v2'])
def test_existing_rgbd_manifest_prepares_without_opening_h5(tmp_path, adapter, dataset_name, monkeypatch):
    cfg = RunConfig.from_dict(read(ROOT / 'configs/mikasa_robo/demos.json'))
    filename = f'record_dataset_{TASK}.h5'
    # Invalid H5 bytes deliberately detect accidental full data reads in manifest-only preparation.
    (tmp_path / filename).write_bytes(b'not opened during preparation')
    manifest = {'schema': 'cam-dataset/1', 'benchmark': dataset_name,
                'benchmark_hash': source_hash(adapter), 'depth': True, 'fields': DECLARED,
                'tasks': {TASK: {'file': filename, 'sha256': '0'*64,
                                 'episodes': adapter.records(TASK, 'train')[:50]}}}
    write(tmp_path / 'manifest.json', manifest)
    original = (tmp_path / 'manifest.json').read_bytes()
    inputs = preparation_inputs(tmp_path, cfg)
    assert inputs[3].name == 'mikasa_robo' and inputs[4] == manifest
    assert (tmp_path / 'manifest.json').read_bytes() == original
    bad = deepcopy(manifest)
    bad['fields']['frame']['front_rgb']['shape'] = [128, 128, 3]
    write(tmp_path / 'manifest.json', bad)
    with pytest.raises(ValueError, match='released 256x256 RGB-D schema'):
        preparation_inputs(tmp_path, cfg)


def test_published_ninety_contracts_match_canonical_and_alias(adapter):
    solution = ROOT / 'solutions/mikasa_robo/demo_interaction_50'
    if not solution.is_dir():
        pytest.skip('Published solution bundle not present')
    catalog, tasks, _ = load_solutions(solution)
    alias = get_adapter('mikasa_robo_v2')
    assert len(tasks) == 90 and set(tasks) == set(adapter.tasks())
    assert catalog['benchmark'] == 'mikasa_robo'
    for task in tasks:
        expected = catalog['tasks'][task]
        assert adapter.interface(task, True) == alias.interface(task, True)
        assert record_hash(adapter.interface(task, True)) == expected['interface_sha256']
        assert record_hash(adapter.records(task, 'test')) == expected['test_records_sha256']
