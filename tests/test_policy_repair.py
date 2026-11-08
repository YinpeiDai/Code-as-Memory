"""Explicit repairs retain source policies and isolate prior evaluation evidence."""

from dataclasses import replace

import pytest

from cam.config import RunConfig
from cam.core.budget import program_hash
from cam.core.history import freeze
from cam.harness.prepare import initial_policy_input, prepare, validate_run
from cam.io import read, sha256, write


def frozen_source(run_factory):
    source = run_factory()
    task = source / 'tasks/BinFill'
    freeze(read(task / 'state/config.json'))
    write(task / 'logs/completion.json', {'status': 'finished'})
    write(task / 'rollouts/test/R000/ep000/result.json', {'private_feedback': 'DO_NOT_IMPORT'})
    return source


def test_repair_pins_same_policy_and_demos_without_test_feedback(run_factory, dataset, tmp_path):
    source = frozen_source(run_factory)
    config = RunConfig.from_dict(read(source / 'run.json')['config'])
    original = source / 'tasks/BinFill/policy'
    digest = program_hash(original)
    original_run_hash = sha256(source / 'run.json')
    target = tmp_path / 'repair'
    data, native = dataset
    prepare(target, data, config, benchmark_source=native, initial_policy_run=source)
    plan = validate_run(target, full_data=True)
    assert plan['fresh_policy'] is False
    assert plan['initial_policy']['source_run'] == str(source)
    assert plan['initial_policy']['program_hash'] == digest
    assert plan['initial_policy']['test_feedback_imported'] is False
    assert program_hash(target / 'tasks/BinFill/workspace') == digest
    assert program_hash(target / 'inputs/initial_policy') == digest
    assert not (target / 'tasks/BinFill/rollouts').exists()
    prompt = (target / 'tasks/BinFill/prompt.md').read_text()
    assert 'user-requested repair' in prompt and 'supplied blank' not in prompt
    assert 'starts as `None`' in prompt
    assert 'DO_NOT_IMPORT' not in prompt
    assert 'User-requested code repair' in (target / 'README.md').read_text()
    assert sha256(source / 'run.json') == original_run_hash
    assert program_hash(original) == digest
    (target / 'inputs/initial_policy/policy.md').write_text('tampered')
    with pytest.raises(ValueError, match='Pinned input changed'):
        validate_run(target)


@pytest.mark.parametrize('fault', ['config', 'demos', 'unfinished', 'policy', 'wrong_task', 'interaction'])
def test_repair_rejects_changed_assignment_or_unfrozen_source(run_factory, fault):
    source = frozen_source(run_factory)
    config = RunConfig.from_dict(read(source / 'run.json')['config'])
    selected = read(source / 'inputs/dataset.json')
    tasks = ['BinFill']
    if fault == 'config':
        config = replace(config, max_decisions=config.max_decisions + 1)
    elif fault == 'demos':
        selected['tasks']['BinFill']['episodes'].pop()
    elif fault == 'unfinished':
        write(source / 'tasks/BinFill/logs/completion.json', {'status': 'running'})
    elif fault == 'policy':
        (source / 'tasks/BinFill/policy/tools/policy.py').write_text('changed')
    elif fault == 'wrong_task':
        tasks = ['OtherTask']
    else:
        config = replace(config, mode='interaction', budget=1)
    with pytest.raises(ValueError):
        initial_policy_input(source, config, tasks, selected)
