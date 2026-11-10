"""Comparison summaries must distinguish final tests, training and incomplete coverage."""
import copy

import pytest

from cam.reporting.arena import summarize_archive
from cam.reporting.site import accepts_gzip


def fixture(tmp_path):
    source = tmp_path / 'run'
    presentation = {'experiments': [{'id': 'demo_interaction_50', 'label': 'Demos + interaction'}],
                    'run_settings': {str(source): 'demo_interaction_50'}}
    run = {'id': '0', 'benchmark': 'robomemarena', 'config': {'budget': 2},
           'expected_episodes_per_setting': 3,
           'development': [{'task': 'T', 'demonstrations': 2, 'rollouts': [{'success': True}] * 2}],
           'settings': [{'budget': 2, 'tasks': {'T': {'expected': 3}}}],
           'episodes': [
               {'task': 'T', 'maximum_rollouts': 0, 'scorable': True, 'success': True, 'status': 'success'},
               {'task': 'T', 'maximum_rollouts': 2, 'scorable': True, 'success': True, 'status': 'success'},
               {'task': 'T', 'maximum_rollouts': 2, 'scorable': True, 'success': False, 'status': 'done'},
           ]}
    return {'runs': [run]}, presentation, [source]


def test_partial_coverage_is_not_reported_as_final_tsr(tmp_path):
    content, spec, sources = fixture(tmp_path)
    summary = summarize_archive(content, spec, sources)
    result = summary['experiments'][0]
    assert (result['expected'], result['scored'], result['successes']) == (3, 2, 1)
    assert result['tsr'] is None and not result['complete']
    assert result['training_attempts'] == 2 and result['perfect_tasks'] == 0
    assert result['failure_statuses'] == {'done': 1}
    assert content['runs'][0]['experiment'] == 'demo_interaction_50'


def test_infrastructure_is_not_a_scored_policy_failure(tmp_path):
    content, spec, sources = fixture(tmp_path)
    content['runs'][0]['episodes'].append({'task': 'T', 'maximum_rollouts': 2,
        'scorable': False, 'success': False, 'status': 'infrastructure_error'})
    result = summarize_archive(content, spec, sources)['experiments'][0]
    assert result['infrastructure'] == 1 and result['failures'] == 1 and result['tsr'] is None
    content['runs'][0]['episodes'][-1].update(scorable=True, success=True, status='success')
    result = summarize_archive(content, spec, sources)['experiments'][0]
    assert result['complete'] and result['tsr'] == 2 / 3
    assert result['infrastructure'] == 0 and result['successes'] == 2


def test_duplicate_selected_task_is_rejected(tmp_path):
    content, spec, sources = fixture(tmp_path)
    content['runs'].append(copy.deepcopy(content['runs'][0]))
    with pytest.raises(ValueError, match='Duplicate selected task'):
        summarize_archive(content, spec, sources)


@pytest.mark.parametrize('header,allowed', [('gzip, deflate, br', True), ('gzip;q=0', False),
    ('gzip;q=0.0', False), ('gzip;q=0.5', True), ('identity', False), ('gzip;q=bad', False)])
def test_compressed_archive_respects_browser_encoding(header, allowed):
    assert accepts_gzip(header) is allowed
