"""Current-run selection preserves history and is independent of test success."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from cam.config import RunConfig
from cam.harness.prepare import prepare
from cam.io import read, sha256, write
from cam.reporting.site import build_site

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication', ROOT / 'scripts/publish_rmbench_rerun.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def example(tmp_path):
    job, run, site = tmp_path / 'job', tmp_path / 'new', tmp_path / 'site'
    old, repair = str(tmp_path / 'original'), str(tmp_path / 'repair')
    write(run / 'run.json', {'fresh_policy': True, 'initial_policy': None, 'tasks': ['swap_blocks'],
                            'config': {'mode': 'demos', 'budget': 0}})
    summary = {'task': 'swap_blocks', 'run': str(run), 'complete': True, 'scored': 100,
               'expected': 100, 'successes': 0, 'tsr': 0.0, 'infrastructure_episodes': []}
    baseline = tmp_path / 'baseline'
    write(baseline / 'results.json', {'expected': 100, 'scored': 100, 'successes': 0,
        'complete': True, 'success_rate': 0.0, 'tasks': {'swap_blocks': {**summary, 'run': old}}})
    write(job / 'plan.json', {'task': 'swap_blocks', 'website': str(site),
        'replaces_display_runs': [old, repair], 'baseline_evaluation': str(baseline)})
    write(site / '.sources.json', {'runs': [old, repair, str(run)], 'datasets': [], 'batch': None})
    return job, run, site, summary, baseline


def test_complete_zero_success_run_can_be_current_without_overwriting_history(tmp_path, monkeypatch):
    job, run, site, summary, baseline = example(tmp_path)
    sources = read(site / '.sources.json')
    old_hash = sha256(baseline / 'results.json')
    # Offline web server does not undo a valid, durable selection.
    def unavailable(*args, **kwargs):
        raise OSError('server temporarily unavailable')
    monkeypatch.setattr(publication.urllib.request, 'urlopen', unavailable)
    publication.publish(job, run, summary)
    assert read(site / '.sources.json') == sources
    visible = read(site / '.run_visibility.json')
    assert set(visible) == set(sources['runs'][:2])
    assert all(row['replacement'] == str(run) for row in visible.values())
    assert sha256(baseline / 'results.json') == old_hash
    assert read(job / 'current_results.json')['tasks']['swap_blocks']['run'] == str(run)
    assert read(job / 'publication.json')['selection_independent_of_success_rate']


@pytest.mark.parametrize('fault', ['partial', 'infra', 'inherited_policy', 'wrong_run'])
def test_incomplete_or_nonfresh_run_cannot_replace_default(tmp_path, fault):
    job, run, site, summary, _ = example(tmp_path)
    if fault == 'partial':
        summary.update(scored=99, complete=False)
    elif fault == 'infra':
        summary['infrastructure_episodes'] = [7]
    elif fault == 'inherited_policy':
        plan = read(run / 'run.json'); plan['fresh_policy'] = False; write(run / 'run.json', plan)
    else:
        summary['run'] = str(tmp_path / 'wrong')
    before = read(site / '.sources.json')
    with pytest.raises(ValueError):
        publication.publish(job, run, summary)
    assert read(site / '.sources.json') == before
    assert not (site / '.run_visibility.json').exists()


def test_site_keeps_old_run_ids_and_records_for_historical_links(dataset, tmp_path):
    data, native = dataset
    roots = [tmp_path / 'old', tmp_path / 'new']
    for root in roots:
        prepare(root, data, RunConfig(benchmark='robomme', mode='demos', budget=0), benchmark_source=native)
    site = tmp_path / 'site'
    build_site(roots, [], site)
    before = deepcopy(read(site / 'data.json')['runs'])
    write(site / '.run_visibility.json', {str(roots[0]): {
        'replacement': str(roots[1]), 'reason': 'User-requested fresh rerun'}})
    build_site(roots, [], site)
    after = read(site / 'data.json')['runs']
    assert [row['id'] for row in after] == [row['id'] for row in before]
    assert after[0]['superseded'] and after[0]['replacement_run_id'] == '1'
    assert not after[1].get('superseded')
    assert after[0]['development'][0]['development_url'] == before[0]['development'][0]['development_url']
    assert len(read(site / '.sources.json')['runs']) == 2

    # Explicit removal takes precedence over historical visibility, including
    # when a running publisher later rewrites the visibility metadata.
    original_hash = sha256(roots[0] / 'run.json')
    write(site / '.excluded_runs.json', [str(roots[0])])
    result = build_site(roots, [], site)
    remaining = read(site / 'data.json')['runs']
    assert [row['id'] for row in remaining] == ['1']
    assert result['runs'] == 1
    assert remaining[0]['development'][0]['development_url'] == before[1]['development'][0]['development_url']
    assert not any(path.startswith(str(roots[0]) + '/') for path in read(site / '.artifacts.json').values())
    assert len(read(site / '.sources.json')['runs']) == 2
    assert sha256(roots[0] / 'run.json') == original_hash
