"""GPU admission must progress independently of task file assembly."""
import threading
import time
import importlib.util
from pathlib import Path

import pytest

from cam.io import write, sha256

from cam.interface.robomemarena.capture import parallel_capture_tasks


def test_next_capture_starts_while_previous_task_is_assembling():
    first_native_finished = threading.Event()
    second_native_started = threading.Event()
    active, peak = 0, 0
    lock = threading.Lock()

    def native(task):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if task == 'first':
            first_native_finished.set()
        else:
            second_native_started.set()
        time.sleep(.01)
        with lock:
            active -= 1
        return task

    def workflow(task, run):
        if task == 'second':
            assert first_native_finished.wait(3)
        result = run(task)
        if task == 'first':
            # File assembly is blocked until another native job has started.
            assert second_native_started.wait(3)
        return result

    assert set(parallel_capture_tasks(['first', 'second'], 1, workflow, native)) == {'first', 'second'}
    assert peak == 1


def scheduler():
    path = Path(__file__).resolve().parents[1]/'scripts/resize_robomemarena_replay.py'
    spec = importlib.util.spec_from_file_location('resize_rma', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adoption_uses_existing_result_without_rerunning_native_job(tmp_path):
    module = scheduler()
    spec = {'mode': 'capture', 'task': 'Example', 'episodes': [{'episode': 1, 'seed': 101}], 'depth': True}
    path, result, dispatch = (tmp_path/n for n in ['job.json', 'result.json', 'dispatch.json'])
    write(path, {**spec, 'result': str(result)})
    write(result, {'results': [{'episode': 1, 'seed': 101, 'status': 'success'}]})
    write(dispatch, {'finished': 1, 'returncode': 0})
    entry = {'job': str(path), 'job_sha256': sha256(path), 'spec': module.read(path),
             'result': str(result), 'result_sha256': sha256(result), 'dispatch': str(dispatch), 'workers': []}
    def forbidden(*a):
        pytest.fail('Completed native job must not be repeated')
    run = module.adopt_worker([entry], forbidden, tmp_path/'adopted')
    assert run(spec, None, None) == module.read(result)
    assert module.read(tmp_path/'adopted/capture_Example.json')['native_replay_repeated'] is False
    with pytest.raises(ValueError, match='assignment differs'):
        run({**spec, 'episodes': [{'episode': 2, 'seed': 102}]}, None, None)
    write(result, {'changed': True})
    with pytest.raises(ValueError, match='result changed'):
        run(spec, None, None)


def test_adoption_waits_for_inflight_capture_and_records_completion(tmp_path, monkeypatch):
    module = scheduler()
    spec = {'mode': 'capture', 'task': 'Example', 'depth': True}
    path, result, dispatch = (tmp_path/n for n in ['job.json', 'result.json', 'dispatch.json'])
    write(path, {**spec, 'result': str(result)})
    write(dispatch, {'native_pid': 123})
    process = {'pid': 123, 'start_ticks': '456'}
    monkeypatch.setattr(module, 'identity', lambda pid: process)
    def finish(_):
        write(result, {'results': []})
        write(dispatch, {'finished': 1, 'returncode': 0})
    monkeypatch.setattr(module.time, 'sleep', finish)
    entry = {'job': str(path), 'job_sha256': sha256(path), 'spec': module.read(path),
             'result': str(result), 'result_sha256': None, 'dispatch': str(dispatch), 'workers': [process]}
    run = module.adopt_worker([entry], lambda *a: pytest.fail('Active worker must be adopted'), tmp_path/'adopted')
    assert run(spec, None, None) == {'results': []}


def test_failed_native_job_is_preserved_without_retry(tmp_path):
    module = scheduler()
    pool = tmp_path/'pool'
    job = pool/'work/capture_Example_1.job.json'
    spec = {'mode': 'capture', 'task': 'Example', 'result': str(pool/'result.json')}
    write(job, spec)
    write(tmp_path/'replay_archive/capture_Example_1/dispatch.json', {'finished': 1, 'returncode': 1})
    entries = module.inventory_jobs(tmp_path, pool)
    assert len(entries) == 1 and entries[0]['result_sha256'] is None
    run = module.adopt_worker(entries, lambda *a: pytest.fail('Failed native job must not be restarted'), tmp_path/'audit')
    with pytest.raises(RuntimeError, match='Adopted native job failed'):
        run(spec, None, None)


def test_constructor_repair_guard_rejects_other_adapter_changes(tmp_path):
    module = scheduler()
    root = Path(__file__).resolve().parents[1]
    previous = root/'outputs/robomemarena_replay_schedulers/20261108T033325_071375Z/inputs/checkout/src/cam/interface/robomemarena/adapter.py'
    # Build the pre-repair adapter from the reviewed replacement, independently
    # of archived runs being present on the machine running this test.
    current = root/'src/cam/interface/robomemarena/adapter.py'
    source = current.read_text()
    import ast
    tree = ast.parse(source)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'construct_environment')
    lines = source.splitlines(keepends=True)
    del lines[helper.lineno-1:helper.end_lineno]
    original = ''.join(lines).replace('import gc\n', '').replace('import logging\n', '')
    original = original.replace('            from robosuite.utils.errors import RandomizationError\n\n', '')
    original = original.replace('env = construct_environment(env_class,\n                dict(bddl_file_name=str(bddl), camera_depths=bool(depth), **ENV_KWARGS),\n                RandomizationError)',
                                'env = env_class(bddl_file_name=str(bddl), camera_depths=bool(depth), **ENV_KWARGS)')
    previous = tmp_path/'previous.py'
    previous.write_text(original)
    assert module.verify_constructor_retry_change(previous, current)
    changed = tmp_path/'changed.py'
    changed.write_text(source.replace('SETTLE_STEPS = 10', 'SETTLE_STEPS = 11'))
    with pytest.raises(ValueError, match='exceeds the constructor-only'):
        module.verify_constructor_retry_change(previous, changed)


def test_adoption_prefers_identical_completed_retry_and_retains_failed_evidence(tmp_path):
    module = scheduler()
    spec = {'mode': 'capture', 'task': 'Example'}
    entries = []
    for name, code in [('failed', 1), ('retry', 0)]:
        path, result, dispatch = [tmp_path/f'{name}.{suffix}.json' for suffix in ['job','result','dispatch']]
        write(path, {**spec, 'result': str(result)})
        write(dispatch, {'finished': 1, 'returncode': code})
        if not code:
            write(result, {'results': ['new-success']})
        entries.append({'job': str(path), 'job_sha256': sha256(path), 'spec': module.read(path),
                        'result': str(result), 'result_sha256': sha256(result) if result.exists() else None,
                        'dispatch': str(dispatch), 'workers': [], 'returncode': code})
    run = module.adopt_worker(entries, lambda *a: pytest.fail('Must reuse completed retry'), tmp_path/'audit')
    assert run(spec, None, None) == {'results': ['new-success']}
    assert all(Path(e['job']).exists() for e in entries)
