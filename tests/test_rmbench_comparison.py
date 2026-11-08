"""Deferred comparisons require full coverage and never select by test success."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from cam.io import sha256, tree_hash, write

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT / 'scripts/queue_rmbench_comparison.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def complete():
    tasks = ['first', 'second']
    development = {'status': 'finished', 'tasks': {
        t: {'task': t, 'status': 'finished'} for t in tasks}}
    evaluation = {'status': 'finished'}
    results = {'complete': True, 'expected': 200, 'scored': 200,
               'unresolved_infrastructure': 0, 'success_rate': 0.0, 'tasks': {
                   t: {'complete': True, 'expected': 100, 'scored': 100, 'successes': 0,
                       'infrastructure_episodes': [], 'tsr': 0.0, 'run': f'/baseline/{t}'} for t in tasks}}
    return development, evaluation, results, tasks


@pytest.mark.parametrize('pending', ['development', 'evaluation', 'both'])
def test_comparison_waits_for_both_stages(pending):
    dev, ev, results, tasks = complete()
    if pending in ('development', 'both'):
        dev['status'] = 'running'
    if pending in ('evaluation', 'both'):
        ev['status'] = 'following_development'
    assert not comparison.baseline_ready(dev, ev, results, tasks)


def test_zero_success_baseline_still_releases_comparison():
    assert comparison.baseline_ready(*complete())


@pytest.mark.parametrize('fault', ['failed_dev', 'infra', 'partial', 'missing_task', 'one_unscored', 'unfinished_task'])
def test_incomplete_coverage_never_releases_comparison(fault):
    dev, ev, results, tasks = complete()
    if fault == 'failed_dev':
        dev['status'] = 'needs_attention'
    elif fault == 'infra':
        results['unresolved_infrastructure'] = 1
    elif fault == 'partial':
        results['complete'] = False
    elif fault == 'missing_task':
        del results['tasks']['second']
    elif fault == 'one_unscored':
        results['tasks']['second']['scored'] = 99
    else:
        dev['tasks']['second']['status'] = 'learning'
    with pytest.raises(ValueError):
        comparison.baseline_ready(dev, ev, results, tasks)


@pytest.mark.parametrize('target', ['baseline', 'config', 'checkout'])
def test_mutated_inputs_rejected(tmp_path, target):
    baseline = tmp_path / 'baseline.json'
    write(baseline, {'unchanged': True})
    job = tmp_path / 'queue'
    write(job / 'config.json', {'budget': 0})
    write(job / 'inputs/checkout/source.json', {'version': 1})
    plan = {'baseline_files': {str(baseline): sha256(baseline)},
            'pinned_files': {'config.json': sha256(job / 'config.json')},
            'checkout_hash': tree_hash(job / 'inputs/checkout')}
    comparison.verify_inputs(job, plan)
    path = {'baseline': baseline, 'config': job / 'config.json',
            'checkout': job / 'inputs/checkout/source.json'}[target]
    write(path, {'changed': True})
    with pytest.raises(ValueError, match='changed'):
        comparison.verify_inputs(job, plan)


def test_comparison_preserves_shortfalls_and_distinguishes_scored_failures():
    _, _, old, _ = complete()
    new = deepcopy(old)
    new['tasks']['first'].update(successes=60, tsr=0.6, run='/new/first')
    new['tasks']['second'].update(successes=80, tsr=0.8, run='/new/second')
    new['success_rate'] = 0.7
    result = comparison.comparison(old, new, {'first': 48, 'second': 50})
    assert result['tasks']['first']['baseline_demos'] == 48
    assert result['tasks']['first']['comparison_demos'] == 25
    assert result['tasks']['first']['difference_percentage_points'] == 60
    assert result['comparison_scored'] == result['baseline_scored'] == 200
    new['complete'] = False
    with pytest.raises(ValueError):
        comparison.comparison(old, new, {'first': 48, 'second': 50})
