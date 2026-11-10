"""No held-out work until a zero-interaction policy is finished and frozen."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cam.core.budget import program_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("robomemarena_full", ROOT / "scripts/robomemarena_full.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def prepared(tmp_path):
    task = "Example"
    folder = tmp_path / "tasks" / task
    full.write(tmp_path / "run.json", {"config": {"mode": "demos", "budget": 0},
        "test_episodes": {task: [0, 1, 2]}})
    full.write(folder / "logs/completion.json", {"status": "finished"})
    policy = folder / "policy"
    (policy / "tools").mkdir(parents=True)
    full.write(policy / "task.json", {})
    (policy / "policy.md").write_text("Test policy\n")
    (policy / "tools/policy.py").write_text("x = 1\n")
    full.write(policy / "freeze.json", {"training": {"attempts": 0}, "program_hash": program_hash(policy)})
    return folder, task


@pytest.mark.parametrize("fault", ["unfinished", "interactions", "modified", "wrong_mode"])
def test_unready_policy_cannot_be_evaluated(tmp_path, fault):
    folder, task = prepared(tmp_path)
    full.frozen_policy(tmp_path, task)
    if fault == "unfinished":
        full.write(folder / "logs/completion.json", {"status": "failed"})
    elif fault == "interactions":
        frozen = full.read(folder / "policy/freeze.json")
        frozen["training"]["attempts"] = 1
        full.write(folder / "policy/freeze.json", frozen)
    elif fault == "modified":
        (folder / "policy/tools/policy.py").write_text("x = 2\n")
    else:
        plan = full.read(tmp_path / "run.json")
        plan["config"].update(mode="both", budget=25)
        full.write(tmp_path / "run.json", plan)
    with pytest.raises(ValueError):
        full.frozen_policy(tmp_path, task)


def test_failed_test_counts_and_only_infrastructure_is_retriable(tmp_path):
    folder, task = prepared(tmp_path)
    frozen = full.frozen_policy(tmp_path, task)
    for episode, scorable, success in [(0, True, True), (1, True, False), (2, False, False)]:
        full.write(folder / f"rollouts/test/R000/ep{episode:03d}/result.json", {
            "episode": episode, "task": task, "split": "test", "scorable": scorable,
            "success": success, "program_hash": frozen["program_hash"]})
    result = full.score_summary(tmp_path, task)
    assert result["scored"] == 2 and result["expected"] == 3 and result["successes"] == 1
    assert result["infrastructure_episodes"] == [2] and result["tsr"] is None
    path = folder / "rollouts/test/R000/ep002/result.json"
    record = full.read(path)
    record["scorable"] = True
    full.write(path, record)
    result = full.score_summary(tmp_path, task)
    assert result["complete"] and result["tsr"] == 1 / 3
    assert not result["infrastructure_episodes"]
    record["program_hash"] = "different"
    full.write(path, record)
    with pytest.raises(ValueError, match="differs from frozen"):
        full.score_summary(tmp_path, task)


def test_shared_gpu_uses_free_memory_without_waiting_for_mikasa(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_ROBOMEMARENA_SHARED_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(full, "queued_mikasa", lambda: pytest.fail("Shared admission must not wait for MIKASA"))
    readings = iter(["9000\n", "24000\n"])
    sleeps = []

    def query(argv, **kwargs):
        assert "--query-gpu=memory.free" in argv
        return SimpleNamespace(stdout=next(readings))

    monkeypatch.setattr(full.subprocess, "run", query)
    monkeypatch.setattr(full.time, "sleep", lambda n: sleeps.append(n))
    with full.gpu_lease(tmp_path, 1, "demo_pool", allow_sharing=True, min_free_mib=16384):
        admissions = list(tmp_path.glob("gpu_admission_demo_pool_*.json"))
        assert len(admissions) == 1
        assert full.read(admissions[0])["free_mib"] == 24000
    assert sleeps == [20]


def test_shared_gpu_fails_closed_if_memory_query_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_ROBOMEMARENA_SHARED_LOCK_DIR", str(tmp_path / "locks"))

    def query(*args, **kwargs):
        raise full.subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(full.subprocess, "run", query)
    with pytest.raises(full.subprocess.CalledProcessError):
        with full.gpu_lease(tmp_path, 1, "demo_pool", allow_sharing=True):
            pytest.fail("Cannot admit a job without memory accounting")


def test_demo_count_gate_uses_assignment_and_never_pads():
    manifest = {'tasks': {'T04': {'episodes': list(range(32))}}}
    assert full.require_demo_counts(manifest, ['T04'], 25) == {'T04': 32}
    with pytest.raises(ValueError, match='Fewer than 50'):
        full.require_demo_counts(manifest, ['T04'], 50)
    with pytest.raises(ValueError, match='Fewer than 25'):
        full.require_demo_counts(manifest, ['T04', 'Missing'], 25)


def test_direct_dataset_uses_all_demos_and_reports_shortfalls():
    manifest = {'tasks': {'T04': {'episodes': list(range(32))},
                          'T01': {'episodes': list(range(50))}}}
    assert full.direct_demo_counts(manifest, ['T04', 'T01'], 50) == {'T04': 32, 'T01': 50}
    for tasks, cap in [(['T04', 'Missing'], 50), (['T01'], 49)]:
        with pytest.raises(ValueError, match='Direct learning requires'):
            full.direct_demo_counts(manifest, tasks, cap)


def test_pool_handoff_waits_for_producer_and_requires_final_manifest(tmp_path, monkeypatch):
    from cam.io import sha256

    source = tmp_path / 'source'
    pool = source / 'pool'
    full.write(source / 'plan.json', {'pool': str(pool)})
    producer = {'pid': 123, 'start_ticks': '456'}
    link = {'source_job': str(source), 'source_plan_sha256': sha256(source / 'plan.json'),
            'pool': str(pool), 'producer': producer}
    monkeypatch.setattr(full, 'process_identity', lambda pid: producer)

    def finish(_):
        full.write(pool / 'manifest.json', {})
        monkeypatch.setattr(full, 'process_identity', lambda pid: None)

    monkeypatch.setattr(full.time, 'sleep', finish)
    full.wait_for_pool(tmp_path, link)
    assert full.read(tmp_path / 'status.json')['producer_pid'] == 123
    (pool / 'manifest.json').unlink()
    monkeypatch.setattr(full, 'process_identity', lambda pid: {'pid': 123, 'start_ticks': '999'})
    with pytest.raises(RuntimeError, match='without a final pool manifest'):
        full.wait_for_pool(tmp_path, link)
    full.write(source / 'plan.json', {'changed': True})
    with pytest.raises(ValueError, match='plan changed'):
        full.wait_for_pool(tmp_path, link)


def test_evaluation_migration_preserves_scores_and_routes_new_artifacts(tmp_path):
    runs = tmp_path/'runs/robomemarena'
    root = runs/'Example/cc-opus5.5/demo_only/stamp'
    folder, task = prepared(root)
    plan = full.read(root/'run.json')
    plan['config']['benchmark'] = 'robomemarena'
    plan['tasks'] = [task]
    full.write(root/'run.json', plan)
    (root/'README.md').write_text('Original run README\n')
    original_plan = (root/'run.json').read_bytes()
    frozen = full.frozen_policy(root, task)
    result = {'episode': 0, 'task': task, 'split': 'test', 'scorable': True,
              'success': True, 'program_hash': frozen['program_hash']}
    full.write(folder/'rollouts/test/R000/ep000/result.json', result)
    (folder/'rollouts/test/R000/ep000/observations.h5').write_bytes(b'unchanged sensor data')
    before = full.score_summary(root, task)
    with pytest.raises(ValueError, match='explicit verified migration'):
        full.evaluation_storage(root, task, tmp_path/'nfs', runs)
    record = full.evaluation_storage(root, task, tmp_path/'nfs', runs, move_existing=True)
    assert (folder/'rollouts/test').is_symlink()
    assert (folder/'rollouts/infrastructure_retries').is_symlink()
    assert full.score_summary(root, task) == before
    assert (root/'run.json').read_bytes() == original_plan
    assert full.frozen_policy(root, task) == frozen
    assert not list((folder/'rollouts').glob('*.verified_local_*'))
    target = Path(record['destination'])
    assert (target/'test/R000/ep000/observations.h5').read_bytes() == b'unchanged sensor data'
    full.write(folder/'rollouts/test/R000/ep001/result.json', {**result, 'episode': 1})
    assert full.read(target/'test/R000/ep001/result.json')['episode'] == 1
    assert full.evaluation_storage(root, task, tmp_path/'nfs', runs)['directories']['test']['already_linked']


def test_evaluation_migration_refuses_conflicting_destination_and_active_queue(tmp_path):
    from cam.harness.jobs import locked
    runs = tmp_path/'runs/robomemarena'
    root = runs/'Example/cc-opus5.5/demo_only/stamp'
    folder, task = prepared(root)
    plan = full.read(root/'run.json')
    plan['config']['benchmark'] = 'robomemarena'
    plan['tasks'] = [task]
    full.write(root/'run.json', plan)
    full.write(folder/'rollouts/test/result.json', {'original': True})
    with locked(root/'locks/evaluation.lock'), pytest.raises(BlockingIOError):
        full.evaluation_storage(root, task, tmp_path/'nfs', runs, move_existing=True)
    full.write(tmp_path/'nfs/evaluation/Example/cc-opus5.5/demo_only/stamp/test/result.json', {'conflict': True})
    with pytest.raises(ValueError, match='destination differs'):
        full.evaluation_storage(root, task, tmp_path/'nfs', runs, move_existing=True)
    assert full.read(folder/'rollouts/test/result.json') == {'original': True}
    assert not (folder/'rollouts/test').is_symlink()


@pytest.mark.parametrize('peer_status', ['learning', 'queued', 'failed'])
def test_ready_policy_admission_does_not_wait_for_other_learners(tmp_path, peer_status):
    folder, task = prepared(tmp_path)
    run = full.read(tmp_path/'run.json')
    run['config'].update(benchmark='robomemarena', depth=True)
    full.write(tmp_path/'run.json', run)
    status = {'status': 'running', 'tasks': {
        task: {'task': task, 'benchmark': 'robomemarena', 'status': 'finished', 'run': str(tmp_path)},
        'Pending': {'task': 'Pending', 'benchmark': 'robomemarena', 'status': peer_status}}}
    assert set(full.ready_policy_roots(status, [task, 'Pending'], episodes=3)) == {task}
    # An unfinished learner is excluded even if it already has a freeze file.
    status['tasks'][task]['status'] = 'learning'
    assert not full.ready_policy_roots(status, [task, 'Pending'], episodes=3)
    status['tasks'][task]['status'] = 'finished'
    (folder/'policy/tools/policy.py').write_text('x = 2\n')
    with pytest.raises(ValueError, match='Frozen policy changed'):
        full.ready_policy_roots(status, [task, 'Pending'], episodes=3)


@pytest.mark.parametrize('prior_attempts', [0, 1])
def test_ready_evaluator_tests_frozen_task_while_peer_is_learning(tmp_path, monkeypatch, prior_attempts):
    from contextlib import nullcontext
    root, job, batch = tmp_path/'run', tmp_path/'job', tmp_path/'batch'
    folder, task = prepared(root)
    run = full.read(root/'run.json')
    run['config'].update(benchmark='robomemarena', depth=True)
    run['test_episodes'][task] = list(range(50))
    full.write(root/'run.json', run)
    status = {'status': 'running', 'tasks': {
        task: {'task': task, 'benchmark': 'robomemarena', 'status': 'finished', 'run': str(root)},
        'Pending': {'task': 'Pending', 'benchmark': 'robomemarena', 'status': 'learning'}}}
    full.write(batch/'status.json', status)
    calls = []
    def command(label, argv, **options):
        assert full.read(batch/'status.json')['tasks']['Pending']['status'] == 'learning'
        assert label == f'evaluate_{task}'
        assert '--resume' in argv and argv[argv.index('--budgets')+1] == '0'
        assert argv[argv.index('--workers-per-gpu')+1] == '10'
        assert full.read(job/'attempt_counts.json')[task] == prior_attempts+1
        calls.append(label)
        frozen = full.frozen_policy(root, task)
        for episode in range(50):
            full.write(folder/f'rollouts/test/R000/ep{episode:03d}/result.json', {
                'task': task, 'episode': episode, 'split': 'test', 'scorable': True,
                'success': episode != 0, 'program_hash': frozen['program_hash']})
        # An unrelated learner failure cannot erase the completed scored tests.
        status['status'] = 'needs_attention'
        status['tasks']['Pending']['status'] = 'failed'
        full.write(batch/'status.json', status)
    monkeypatch.setattr(full, 'evaluation_storage', lambda *a: {'nfs': True})
    monkeypatch.setattr(full, 'gpu_lease', lambda *a: nullcontext())
    full.evaluate_ready(job, {'development_batch': str(batch), 'tasks': [task, 'Pending'],
        'expected_test_episodes': 100, 'evaluation_root': str(tmp_path/'nfs'), 'repository': str(tmp_path),
        'gpu': 1, 'evaluation_workers': 10, 'python': '/python',
        'inherited_attempt_counts': {task: prior_attempts}}, command, lambda *a, **kw: None)
    assert calls == [f'evaluate_{task}']
    final = full.read(job/'status.json')
    assert final['scored'] == 50 and final['successes'] == 49
    assert final['tsr'] is None and final['unevaluated_tasks'] == ['Pending']
    assert full.read(job/'results.json')['tasks'][task]['tsr'] == 49/50


def test_resized_evaluator_waits_for_existing_queue_without_relaunch(tmp_path, monkeypatch):
    from cam.io import sha256
    command = tmp_path/'old.command.json'
    full.write(command, {'argv': ['python', 'runner.py', 'evaluate', 'run']})
    process = {'pid': 42, 'start_ticks': '123'}
    inherited = {'task': 'Example', 'command': str(command), 'command_sha256': sha256(command), 'process': process}
    identities = iter([process, process, {'pid': 42, 'start_ticks': '456'}])
    monkeypatch.setattr(full, 'process_identity', lambda pid: next(identities))
    sleeps, phases = [], []
    monkeypatch.setattr(full.time, 'sleep', sleeps.append)
    monkeypatch.setattr(full.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Must not relaunch the inherited queue'))
    plan = {'inherited_evaluation': inherited, 'evaluation_workers': 10}
    full.wait_for_inherited_evaluation(tmp_path, plan, lambda phase, **kw: phases.append(phase))
    assert sleeps == [10, 10] and phases == ['draining_previous_evaluation']*2
    assert full.read(tmp_path/'inherited_evaluation_finished.json')['episodes_interrupted'] is False
    full.write(command, {'changed': True})
    with pytest.raises(ValueError, match='Inherited evaluation command changed'):
        full.wait_for_inherited_evaluation(tmp_path, plan, lambda *a, **kw: None)


def test_added_gpu_starts_while_inherited_gpu_finishes_without_duplicate_tests(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import threading
    import time

    batch, job = tmp_path/'batch', tmp_path/'job'
    rows, roots = {}, {}
    for task in ['Complete', 'Inherited', 'New']:
        root = tmp_path/'runs'/task
        folder = root/'tasks'/task
        full.write(root/'run.json', {'config': {'benchmark': 'robomemarena', 'depth': True, 'mode': 'demos', 'budget': 0},
                                    'test_episodes': {task: list(range(50))}})
        full.write(folder/'logs/completion.json', {'status': 'finished'})
        policy = folder/'policy'
        full.write(policy/'task.json', {})
        (policy/'policy.md').write_text('Fixture')
        (policy/'tools').mkdir()
        (policy/'tools/policy.py').write_text('x = 1\n')
        full.write(policy/'freeze.json', {'training': {'attempts': 0}, 'program_hash': program_hash(policy)})
        roots[task] = root
        rows[task] = {'task': task, 'run': str(root), 'benchmark': 'robomemarena', 'status': 'finished'}
    full.write(batch/'status.json', {'status': 'finished', 'tasks': rows})

    def score(task, failure=False):
        frozen = full.frozen_policy(roots[task], task)
        for ep in range(50):
            full.write(roots[task]/f'tasks/{task}/rollouts/test/R000/ep{ep:03d}/result.json', {
                'task': task, 'episode': ep, 'split': 'test', 'program_hash': frozen['program_hash'],
                'scorable': True, 'success': not (failure and ep == 0)})
    score('Complete')
    inherited_waiting, added_gpu_started = threading.Event(), threading.Event()
    def wait_existing(*args):
        inherited_waiting.set()
        assert added_gpu_started.wait(3), 'GPU 0 must start before GPU 1 finishes'
        score('Inherited', failure=True)
    calls = []
    def command(label, argv, **kw):
        assert inherited_waiting.wait(3)
        assert label == 'evaluate_New'
        assert argv[argv.index('--gpus')+1] == '0'
        assert argv[argv.index('--workers-per-gpu')+1] == '10'
        calls.append(label)
        added_gpu_started.set()
        score('New')
    real_sleep = time.sleep
    monkeypatch.setattr(full.time, 'sleep', lambda n: real_sleep(min(n, .01)))
    monkeypatch.setattr(full, 'wait_for_inherited_evaluation', wait_existing)
    monkeypatch.setattr(full, 'gpu_lease', lambda *a: nullcontext())
    monkeypatch.setattr(full, 'evaluation_storage', lambda *a: {'nfs': True})
    full.evaluate_ready_parallel(job, {'development_batch': str(batch), 'tasks': list(rows),
        'evaluation_gpus': [0, 1], 'evaluation_workers': 10, 'expected_test_episodes': 150,
        'inherited_evaluations': [{'task': 'Inherited', 'gpu': 1}],
        'inherited_attempt_counts': {'Complete': 1, 'Inherited': 1},
        'repository': str(tmp_path), 'evaluation_root': str(tmp_path/'nfs'), 'python': '/python'}, command)
    assert calls == ['evaluate_New']
    final = full.read(job/'status.json')
    assert final['status'] == 'finished' and final['scored'] == 150
    assert final['successes'] == 149 and final['tsr'] == 149/150
    assert full.read(job/'attempt_counts.json') == {'Complete': 1, 'Inherited': 1, 'New': 1}


def test_followup_waits_for_full_baseline_coverage_regardless_of_success(tmp_path, monkeypatch):
    from cam.io import sha256
    baseline, job = tmp_path/'baseline', tmp_path/'followup'
    full.write(baseline/'plan.json', {'tasks': ['Task'], 'expected_test_episodes': 50})
    full.write(baseline/'process.json', {'pid': 42})
    result = {'tasks': {'Task': {'task': 'Task', 'run': '/fixture', 'expected': 50,
        'scored': 49, 'successes': 0, 'complete': False, 'tsr': None, 'infrastructure_episodes': [49]}},
        'expected': 50, 'scored': 49, 'successes': 0}
    full.write(baseline/'results.json', result)
    plan = {'wait_for_evaluation': str(baseline), 'parent_plan_sha256': sha256(baseline/'plan.json')}
    monkeypatch.setattr(full, 'alive', lambda pid: True)
    monkeypatch.setattr(full, 'frozen_policy', lambda *a: {})
    monkeypatch.setattr(full, 'score_summary', lambda *a: result['tasks']['Task'])
    states = []
    def complete(_):
        assert not (job/'baseline_evaluation_complete.json').exists()
        result['scored'] = 50
        result['tasks']['Task'].update(scored=50, complete=True, tsr=0, infrastructure_episodes=[])
        full.write(baseline/'results.json', result)
    monkeypatch.setattr(full.time, 'sleep', complete)
    full.wait_for_baseline_evaluation(job, plan, lambda phase, **kw: states.append(phase))
    assert states == ['waiting_for_baseline_evaluation']
    assert full.read(job/'baseline_evaluation_complete.json')['successes'] == 0
    result['scored'] = 49
    full.write(baseline/'results.json', result)
    monkeypatch.setattr(full, 'alive', lambda pid: False)
    with pytest.raises(RuntimeError, match='before every episode was scored'):
        full.wait_for_baseline_evaluation(job, plan, lambda *a, **kw: None)


def test_cancelled_followup_restarts_with_local_storage_and_four_workers(tmp_path, monkeypatch):
    from cam.io import tree_hash

    preparation, baseline, batch, old_batch = [tmp_path/name for name in ('prep', 'baseline', 'batch', 'old_batch')]
    old_job = tmp_path/'cancelled'
    old_pointer = {'job': str(old_job), 'pid': 123}
    full.write(preparation/'evaluation_followup.json', old_pointer)
    full.write(old_job/'status.json', {'status': 'cancelled_by_user'})
    full.write(old_job/'cancelled_by_user.json', {'reason': 'User cancelled'})
    full.write(old_job/'process.json', {'pid': 123})
    preserved = {p.name: p.read_bytes() for p in old_job.glob('*.json')}
    harness = batch/'inputs/checkout/src/cam'
    harness.mkdir(parents=True)
    (harness/'__init__.py').write_text('')
    root, old_root = tmp_path/'run', tmp_path/'old_run'
    for run in (root, old_root):
        _, task = prepared(run)
        manifest = full.read(run/'run.json')
        manifest['config'].update(benchmark='robomemarena', depth=True)
        full.write(run/'run.json', manifest)
        full.write(run/'inputs/dataset.json', {'tasks': {task: {'episodes': [{'episode': i} for i in range(25)]}}})
        full.write(run/'inputs/metadata/test'/f'record_dataset_{task}_metadata.json', {'records': [0, 1, 2]})
    row = {'task': task, 'run': str(root), 'status': 'finished'}
    full.write(batch/'status.json', {'tasks': {f'robomemarena/{task}': row}})
    full.write(old_batch/'status.json', {'tasks': {f'robomemarena/{task}': {**row, 'run': str(old_root)}}})
    full.write(batch/'plan.json', {'harness_hash': tree_hash(harness),
        'benchmarks': {'robomemarena': {'tasks': [task], 'demo_count': 25, 'interaction_count': 0,
            'source_hash': 'source', 'asset_hashes': {}}},
        'environment': {'CAM_ROBOMEMARENA_PYTHON': '/native/python'}})
    full.write(preparation/'batch.json', {'batch': str(batch)})
    full.write(preparation/'prompt_comparison.json', {'changed': 'offline review'})
    full.write(baseline/'plan.json', {'tasks': [task], 'development_batch': str(old_batch),
        'python': '/python', 'evaluation_root': str(tmp_path/'old_nfs'),
        'allow_gpu_sharing': True, 'min_free_memory_mib': 16384})
    monkeypatch.setattr(full, 'ROOT', tmp_path)
    monkeypatch.setattr(full, 'alive', lambda pid: False)
    monkeypatch.setattr(full, 'launch', lambda job: {'job': str(job), 'pid': 456})
    local = tmp_path/'local/robomemarena'
    result = full.queue_evaluation_followup(baseline, preparation, local, True, 4)
    job = Path(result['job'])
    plan = full.read(job/'plan.json')
    assert plan['evaluation_root'] == str(local)
    assert plan['evaluation_workers'] == 4 and plan['evaluation_gpus'] == [0, 1]
    assert plan['restarts_cancelled_job'] == str(old_job)
    assert full.read(job/'previous_evaluation_followup.json') == old_pointer
    assert full.read(preparation/'evaluation_followup.json') == result
    assert full.read(next((preparation/'evaluation_followup_history').glob('*.json'))) == old_pointer
    assert {p.name: p.read_bytes() for p in old_job.glob('*.json')} == preserved
    assert '4 episode workers per GPU' in (job/'README.md').read_text()


@pytest.mark.parametrize('status,alive,restart,error', [
    ('cancelled_by_user', False, False, FileExistsError),
    ('running', False, True, ValueError),
    ('cancelled_by_user', True, True, ValueError),
])
def test_followup_restart_cannot_duplicate_existing_work(tmp_path, monkeypatch, status, alive, restart, error):
    prep, job = tmp_path/'prep', tmp_path/'job'
    full.write(prep/'evaluation_followup.json', {'job': str(job), 'pid': 42})
    full.write(job/'status.json', {'status': status})
    full.write(job/'process.json', {'pid': 42})
    full.write(job/'cancelled_by_user.json', {'reason': 'User cancelled'})
    monkeypatch.setattr(full, 'alive', lambda pid: alive)
    with pytest.raises(error):
        full.queue_evaluation_followup(tmp_path/'unused', prep, restart_cancelled=restart)


def interaction_prepared(root, budget):
    import shutil
    folder, task = prepared(root)
    run = full.read(root/'run.json')
    run['config'].update(benchmark='robomemarena', depth=True, mode='both', both_protocol='v2', budget=budget)
    run['test_episodes'][task] = list(range(50))
    full.write(root/'run.json', run)
    freeze = full.read(folder/'policy/freeze.json')
    freeze.update(checkpoint=budget, training={'attempts': budget})
    full.write(folder/'policy/freeze.json', freeze)
    shutil.copytree(folder/'policy', folder/'checkpoints'/f'R{budget:03d}')
    return folder, task


@pytest.mark.parametrize('protocol', ['v2', 'demo_first'])
@pytest.mark.parametrize('budget', [32, 50])
@pytest.mark.parametrize('fault', [None, 'early_stop', 'wrong_checkpoint', 'changed_checkpoint'])
def test_interaction_evaluation_requires_full_budget_and_final_checkpoint(tmp_path, budget, fault, protocol):
    folder, task = interaction_prepared(tmp_path, budget)
    run = full.read(tmp_path/"run.json")
    run["config"]["both_protocol"] = protocol
    full.write(tmp_path/"run.json", run)
    with pytest.raises(ValueError):
        full.frozen_policy(tmp_path, task)
    freeze = full.read(folder/'policy/freeze.json')
    if fault == 'early_stop':
        freeze['training']['attempts'] -= 1
    elif fault == 'wrong_checkpoint':
        freeze['checkpoint'] -= 1
    elif fault == 'changed_checkpoint':
        (folder/'checkpoints'/f'R{budget:03d}'/'tools/policy.py').write_text('x = 2\n')
    full.write(folder/'policy/freeze.json', freeze)
    if fault:
        with pytest.raises(ValueError):
            full.frozen_policy(tmp_path, task, allow_interaction=True)
    else:
        assert full.frozen_policy(tmp_path, task, allow_interaction=True) == freeze


def test_interaction_evaluator_scores_final_budget_and_retries_only_infrastructure(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import time

    root = tmp_path/'run'
    folder, task = interaction_prepared(root, 32)
    freeze = full.frozen_policy(root, task, allow_interaction=True)
    batch, job = tmp_path/'batch', tmp_path/'job'
    job.mkdir()
    full.write(batch/'status.json', {'status': 'finished', 'tasks': {
        task: {'task': task, 'benchmark': 'robomemarena', 'status': 'finished', 'run': str(root)}}})
    base = {'task': task, 'split': 'test', 'program_hash': freeze['program_hash'], 'scorable': True, 'success': True}
    # Earlier-checkpoint results must not satisfy final-checkpoint coverage.
    for ep in range(50):
        full.write(folder/f'rollouts/test/R000/ep{ep:03d}/result.json', {**base, 'episode': ep})
    calls = []
    def command(label, argv, **kwargs):
        calls.append(label)
        if label.startswith('recover_'):
            assert argv[argv.index('--budget')+1] == '32'
            assert label == f'recover_{task}_0'
            (folder/'rollouts/test/R032/ep000/result.json').rename(folder/'infra_retry_evidence.json')
        else:
            assert argv[argv.index('--budgets')+1] == '32'
            assert argv[argv.index('--workers-per-gpu')+1] == '10'
            first = calls.count(f'evaluate_{task}') == 1
            for ep in range(50):
                path = folder/f'rollouts/test/R032/ep{ep:03d}/result.json'
                if not path.exists():
                    full.write(path, {**base, 'episode': ep, 'scorable': not (first and ep == 0), 'success': ep != 1})
    monkeypatch.setattr(full, 'evaluation_storage', lambda *a: {})
    monkeypatch.setattr(full, 'gpu_lease', lambda *a: nullcontext())
    real_sleep = time.sleep
    monkeypatch.setattr(full.time, 'sleep', lambda n: real_sleep(min(n, .01)))
    full.evaluate_ready_parallel(job, {'development_batch': str(batch), 'tasks': [task],
        'allow_interaction': True, 'evaluation_gpus': [0, 1], 'evaluation_workers': 10,
        'expected_test_episodes': 50, 'repository': str(tmp_path), 'evaluation_root': str(tmp_path/'local'),
        'python': '/python'}, command)
    assert calls == [f'evaluate_{task}', f'recover_{task}_0', f'evaluate_{task}']
    result = full.read(job/'status.json')
    assert result['status'] == 'finished' and result['scored'] == 50 and result['successes'] == 49
    assert not full.read(folder/'rollouts/test/R032/ep001/result.json')['success']
    assert not full.read(folder/'infra_retry_evidence.json')['scorable']
