"""Run authorized RoboMemArena experiments and evaluate their frozen policies.

Start with: .venv/bin/python scripts/robomemarena_full.py start --demo-count 25 --pool-cap 50 --allow-gpu-sharing --replay-gpus 0 1 --replay-workers 8
The durable host supervisor resumes the existing pinned demo pool, makes a
separate subset, launches sandboxed learners, and tests only frozen policies.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = Path('/nfs/turbo/coe-chaijy-unreplicated/daiyp/CaM/robomemarena')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def process_identity(pid):
    """Include the kernel start time so a recycled PID is never a producer."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else {"pid": pid, "start_ticks": fields[19]}
    except FileNotFoundError:
        return None


def existing_pool(source_job, benchmark_hash):
    """Attach to a pinned extraction without modifying or restarting its workers."""
    from cam.io import sha256, tree_hash

    source_job = Path(source_job).resolve()
    previous = read(source_job / 'plan.json')
    if previous['benchmark_hash'] != benchmark_hash:
        raise ValueError('Existing replay pool uses a different benchmark')
    if tree_hash(source_job / 'inputs/checkout/src/cam') != previous['harness_hash']:
        raise ValueError('Existing replay harness changed')
    pool = Path(previous.get('assembly_output', previous['pool']))
    producer = None
    for command in sorted((source_job / 'logs').glob('*.command.json'), reverse=True):
        argv = read(command)['argv']
        if 'extract' not in argv or '--output' not in argv or Path(argv[argv.index('--output') + 1]) != pool:
            continue
        record = command.with_name(command.name.replace('.command.json', '.process.json'))
        if not record.exists():
            continue
        candidate = process_identity(read(record)['pid'])
        if candidate is None:
            continue
        cmdline = (Path('/proc') / str(candidate['pid']) / 'cmdline').read_bytes().split(b'\0')
        if str(pool).encode() not in cmdline or b'extract' not in cmdline:
            continue
        producer = candidate
        break
    if producer is None and not (pool / 'manifest.json').exists():
        raise ValueError('Existing pool has neither a final manifest nor a live extraction producer')
    return {"source_job": str(source_job), "source_plan_sha256": sha256(source_job / 'plan.json'),
            "pool": str(pool), "producer": producer}


def wait_for_pool(job, link):
    from cam.io import sha256

    if sha256(Path(link['source_job']) / 'plan.json') != link['source_plan_sha256']:
        raise ValueError('Existing replay preparation plan changed')
    producer = link['producer']
    while producer and process_identity(producer['pid']) == producer:
        write(job / 'status.json', {"status": "running", "phase": "existing_demo_pool",
              "source_job": link['source_job'], "pool": link['pool'],
              "producer_pid": producer['pid'], "updated": time.time()})
        time.sleep(20)
    if not (Path(link['pool']) / 'manifest.json').exists():
        raise RuntimeError(f"Replay producer stopped without a final pool manifest: {link['source_job']}")


def require_demo_counts(manifest, tasks, count):
    counts = {task: len(manifest['tasks'].get(task, {}).get('episodes', [])) for task in tasks}
    if any(n < count for n in counts.values()):
        raise ValueError(f'Fewer than {count} successful demos: {counts}')
    return counts


def direct_demo_counts(manifest, tasks, cap):
    """Use the full capped pool and explicitly record every release shortfall."""
    counts = {task: len(manifest['tasks'].get(task, {}).get('episodes', [])) for task in tasks}
    if any(n < 1 or n > cap for n in counts.values()):
        raise ValueError(f'Direct learning requires 1–{cap} successful demos for every task: {counts}')
    return counts


def evaluation_storage(root, task, storage_root, runs_root, move_existing=False):
    """Keep canonical output paths while storing artifacts under the chosen root.

    Existing data are copied and hash-verified before local originals are
    removed. This never modifies prepared inputs, policies or scored results.
    """
    from cam.io import sha256
    from cam.harness.jobs import locked

    root, runs_root, storage_root = Path(root).resolve(), Path(runs_root).resolve(), Path(storage_root).resolve()
    relative = root.relative_to(runs_root)
    manifest = read(root/'run.json')
    if manifest['config']['benchmark'] != 'robomemarena' or task not in manifest['tasks'] or len(relative.parts) != 4 or relative.parts[0] != task:
        raise ValueError('Expected a canonical single-task RoboMemArena run')
    destination = storage_root/'evaluation'/relative
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Evaluation storage must be outside the prepared run')

    def inventory(folder):
        values = {}
        for base, dirs, files in os.walk(folder):
            for name in dirs + files:
                path = Path(base)/name
                if path.is_symlink():
                    raise ValueError('Nested symlinks are not supported in rollout migration')
            for name in files:
                path = Path(base)/name
                before = path.stat()
                digest = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Evaluation file changed during inventory')
                values[path.relative_to(folder).as_posix()] = {'size': before.st_size, 'sha256': digest}
        return values

    with locked(root/'locks/evaluation.lock'):
        record = {'run': str(root), 'task': task, 'storage_root': str(storage_root),
                  'destination': str(destination), 'updated': time.time(), 'directories': {}}
        for name in ('test', 'infrastructure_retries'):
            source = root/'tasks'/task/'rollouts'/name
            target = destination/name
            if target.is_symlink() or (target.exists() and not target.is_dir()) or (source.exists() and not source.is_dir()):
                raise ValueError('Expected real evaluation directories or a canonical source link')
            if source.is_symlink():
                if source.resolve() != target or not target.is_dir():
                    raise ValueError('Evaluation output already links to different or missing storage')
                record['directories'][name] = {'target': str(target), 'already_linked': True}
                continue
            files = inventory(source) if source.exists() else {}
            if files and not move_existing:
                raise ValueError('Existing local evaluations require explicit verified migration')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if inventory(target) != files:
                    raise ValueError('Evaluation destination differs; original files retained')
            elif source.exists():
                temporary = target.with_name(name + f'.copying_{time.time_ns()}')
                shutil.copytree(source, temporary)
                if inventory(temporary) != files or inventory(source) != files:
                    raise ValueError('Evaluation copy verification failed; originals retained')
                temporary.replace(target)
            else:
                target.mkdir()
            receipt = {'source': str(source), 'target': str(target), 'files': files,
                       'bytes_moved': sum(v['size'] for v in files.values())}
            stamp = str(time.time_ns())
            audit = root/'storage_migrations'/f'{name}_{stamp}.json'
            backup = source.with_name(name + '.verified_local_' + stamp)
            write(audit, {**receipt, 'status': 'verified_before_link', 'backup': str(backup)})
            source.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                source.rename(backup)
            try:
                source.symlink_to(target, target_is_directory=True)
            except BaseException:
                if backup.exists():
                    backup.rename(source)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            write(audit, {**receipt, 'status': 'complete', 'completed': time.time()})
            record['directories'][name] = {'target': str(target), 'audit': str(audit),
                                            'bytes_moved': receipt['bytes_moved']}
        write(root/'evaluation_storage.json', record)
        readme = root/'README.md'
        note = f"\nHeld-out rollout storage: `{destination}`. The canonical `tasks/{task}/rollouts/test` and `rollouts/infrastructure_retries` paths link to this location. Training demos and prepared inputs retain their existing locations. See `evaluation_storage.json` and `storage_migrations/` for transfer verification.\n"
        if readme.exists() and note.strip() not in readme.read_text():
            with readme.open('a') as handle:
                handle.write(note)
        return record


def queued_mikasa():
    """Submitted replay jobs can initialize GPUs even between native workers."""
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "python" in args and "cam.interface.mikasa_robo_v2_data" in args:
            if " worker " in args or ("extract" in args and "gpus=" in args):
                found.append(int(process.name))
    return found


@contextmanager
def gpu_lease(job, gpu, phase, allow_sharing=False, min_free_mib=16384):
    """Coordinate with the existing RGB-D collection queue and GPU users."""
    if allow_sharing:
        lock_root = Path(os.environ.get("CAM_ROBOMEMARENA_SHARED_LOCK_DIR", "/tmp/cam-robomemarena-shared-gpus"))
        lock_root.mkdir(parents=True, exist_ok=True)
        with (lock_root / f"gpu_{gpu}.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    write(job / "status.json", {"status": "waiting_for_robomemarena_slot", "phase": phase,
                          "gpu": gpu, "updated": time.time()})
                    time.sleep(20)
                    continue
                result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
                free_mib = int(result.stdout.strip())
                if free_mib < min_free_mib:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    write(job / "status.json", {"status": "waiting_for_gpu_memory", "phase": phase,
                          "gpu": gpu, "free_mib": free_mib, "required_free_mib": min_free_mib,
                          "updated": time.time()})
                    time.sleep(20)
                    continue
                write(job / f"gpu_admission_{phase}_{time.time_ns()}.json", {
                    "gpu": gpu, "sharing": True, "free_mib": free_mib,
                    "required_free_mib": min_free_mib, "time": time.time()})
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                return
        return
    lock_root = Path(os.environ.get("CAM_RGBD_LOCK_DIR", "/tmp/cam-rmbench-rgbd-locks"))
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"gpu_{gpu}.lock").open("a") as lock:
        while True:
            blockers = queued_mikasa()
            try:
                if blockers:
                    raise BlockingIOError
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                write(job / "status.json", {"status": "waiting_for_gpu", "phase": phase,
                      "gpu": gpu, "blocking_mikasa_pids": blockers, "updated": time.time()})
                time.sleep(20)
                continue
            result = subprocess.run(["nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True)
            if result.stdout.strip():
                fcntl.flock(lock, fcntl.LOCK_UN)
                write(job / "status.json", {"status": "waiting_for_gpu", "phase": phase,
                      "gpu": gpu, "blocking_processes": result.stdout.splitlines(), "updated": time.time()})
                time.sleep(20)
                continue
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            return


def frozen_policy(root, task, allow_interaction=False):
    """Require completed learning; interaction evaluation additionally needs its full budget."""
    root = Path(root)
    config = read(root / "run.json")["config"]
    folder = root / "tasks" / task
    completion = read(folder / "logs/completion.json")
    frozen = read(folder / "policy/freeze.json")
    interaction = allow_interaction and config["mode"] == "both" and config.get("both_protocol") in ("v2", "demo_first")
    if completion["status"] != "finished" or (not interaction and (config["mode"] != "demos" or config["budget"] != 0)):
        raise ValueError(f"Development is unfinished or outside the requested setting: {task}")
    if frozen["training"]["attempts"] != config["budget"]:
        raise ValueError(f"Training attempts differ from the assigned budget: {task}")
    # The harness performs its own exact checkpoint/program integrity validation.
    from cam.core.budget import program_hash
    if program_hash(folder / "policy") != frozen["program_hash"]:
        raise ValueError(f"Frozen policy changed: {task}")
    if interaction:
        from cam.core.history import select_checkpoint
        if config["budget"] < 1 or frozen["checkpoint"] != config["budget"]:
            raise ValueError(f"Full interaction budget must be frozen: {task}")
        _, checkpoint = select_checkpoint(folder, config["budget"])
        if checkpoint["program_hash"] != frozen["program_hash"]:
            raise ValueError(f"Final checkpoint differs from frozen policy: {task}")
    return frozen


def score_summary(root, task):
    root = Path(root)
    expected = read(root / "run.json")["test_episodes"][task]
    frozen = read(root / "tasks" / task / "policy/freeze.json")
    budget = frozen.get('checkpoint', 0)
    records = []
    infrastructure = []
    for episode in expected:
        folder = root / "tasks" / task / "rollouts/test" / f"R{budget:03d}" / f"ep{episode:03d}"
        if not folder.exists():
            continue
        if not (folder / "result.json").exists():
            infrastructure.append(episode)
            continue
        result = read(folder / "result.json")
        if result["program_hash"] != frozen["program_hash"]:
            raise ValueError(f"Tested policy differs from frozen policy: {task}/{episode}")
        if result["episode"] != episode or result["task"] != task or result["split"] != "test":
            raise ValueError(f"Test identity differs from assignment: {task}/{episode}")
        if not result["scorable"]:
            infrastructure.append(episode)
        else:
            records.append(result)
    successes = sum(bool(r["success"]) for r in records)
    return {"task": task, "run": str(root), "expected": len(expected),
            "scored": len(records), "successes": successes,
            "complete": len(records) == len(expected),
            "tsr": successes / len(expected) if len(records) == len(expected) else None,
            "infrastructure_episodes": infrastructure}


def ready_policy_roots(status, tasks, episodes=50, allow_interaction=False):
    """Admit completed tasks independently; a freeze alone cannot admit a learner."""
    rows = list(status['tasks'].values())
    if len(rows) != len(tasks) or {r['task'] for r in rows} != set(tasks):
        raise ValueError('Development does not cover the exact assigned task set')
    ready = {}
    for row in rows:
        if row['status'] != 'finished':
            continue
        task, root = row['task'], Path(row['run'])
        run = read(root/'run.json')
        if (row['benchmark'] != 'robomemarena' or run['config']['benchmark'] != 'robomemarena'
                or not run['config']['depth'] or len(run['test_episodes'][task]) != episodes):
            raise ValueError(f'Expected the assigned RGB-D held-out protocol: {task}')
        ready[task] = (root, frozen_policy(root, task, allow_interaction))
    return ready


def wait_for_inherited_evaluation(job, plan, state):
    """Let the previous task queue finish without restarting any live episode."""
    from cam.io import sha256

    inherited = plan.get('inherited_evaluation')
    if not inherited:
        return
    if sha256(inherited['command']) != inherited['command_sha256']:
        raise ValueError('Inherited evaluation command changed')
    process = inherited['process']
    while process_identity(process['pid']) == process:
        state('draining_previous_evaluation', task=inherited['task'],
              inherited_pid=process['pid'], next_evaluation_workers=plan['evaluation_workers'])
        time.sleep(10)
    write(job/'inherited_evaluation_finished.json', {**inherited, 'finished': time.time(),
        'episodes_interrupted': False})


def evaluate_ready(job, plan, command, state):
    if len(plan.get('evaluation_gpus', [])) > 1:
        return evaluate_ready_parallel(job, plan, command)
    wait_for_inherited_evaluation(job, plan, state)
    batch = Path(plan['development_batch'])
    results, frozen, storage = {}, {}, {}
    attempts = dict(plan.get('inherited_attempt_counts', {}))
    if (job/'attempt_counts.json').exists():
        attempts.update(read(job/'attempt_counts.json'))
    while True:
        status = read(batch/'status.json')
        ready = ready_policy_roots(status, plan['tasks'], allow_interaction=plan.get('allow_interaction', False))
        task = next((t for t in plan['tasks'] if t in ready and t not in results), None)
        if task is None:
            if len(results) == len(plan['tasks']):
                break
            if status['status'] in ('finished', 'needs_attention') or not alive(read(batch/'process.json')['pid']):
                break
            state('waiting_for_ready_policy', completed_tasks=len(results), batch=str(batch))
            time.sleep(20)
            continue
        root, freeze = ready[task]
        budget = str(freeze.get('checkpoint', 0))
        frozen[task] = freeze
        write(job/'frozen_policies.json', frozen)
        storage[task] = evaluation_storage(root, task, plan['evaluation_root'],
            Path(plan['repository'])/'runs/robomemarena')
        write(job/'evaluation_storage.json', storage)
        with gpu_lease(job, plan['gpu'], 'evaluation', plan.get('allow_gpu_sharing', False),
                       plan.get('min_free_memory_mib', 16384)):
            for attempt in range(attempts.get(task, 0), 3):
                if frozen_policy(root, task, plan.get('allow_interaction', False)) != freeze:
                    raise ValueError(f'Frozen policy changed after admission: {task}')
                summary = score_summary(root, task)
                if summary['complete']:
                    break
                for episode in summary['infrastructure_episodes']:
                    command(f'recover_{task}_{episode}', [plan['python'], str(root/'runner.py'),
                        'recover-test', str(root), task, str(episode), '--budget', budget, '--reason',
                        'Unscorable infrastructure failure; identical pinned policy, seed and limits under the recorded GPU scheduling policy.'])
                state('evaluation', task=task, pass_number=attempt+1, completed_tasks=len(results),
                      release_policy='per_task', development_batch=str(batch))
                attempts[task] = attempt+1
                write(job/'attempt_counts.json', attempts)
                command(f'evaluate_{task}', [plan['python'], str(root/'runner.py'), 'evaluate', str(root),
                    '--gpus', str(plan['gpu']), '--workers-per-gpu', str(plan['evaluation_workers']),
                    '--budgets', budget, '--resume'], allow_failure=True)
        results[task] = score_summary(root, task)
        write(job/'results.json', {'tasks': results, 'expected': plan['expected_test_episodes'],
            'scored': sum(r['scored'] for r in results.values()),
            'successes': sum(r['successes'] for r in results.values())})
    complete = set(results) == set(plan['tasks']) and all(r['complete'] for r in results.values())
    successes = sum(r['successes'] for r in results.values())
    write(job/'status.json', {'status': 'finished' if complete else 'needs_attention',
        'phase': 'evaluation', 'release_policy': 'per_task', 'updated': time.time(),
        'expected': plan['expected_test_episodes'], 'scored': sum(r['scored'] for r in results.values()),
        'successes': successes, 'tsr': successes/plan['expected_test_episodes'] if complete else None,
        'development_batch': str(batch), 'unevaluated_tasks': [t for t in plan['tasks'] if t not in results]})


def evaluate_ready_parallel(job, plan, command):
    """One independently admitted task per GPU; adopt active queues in place."""
    batch = Path(plan['development_batch'])
    expected, gpus = plan['tasks'], plan['evaluation_gpus']
    guard = threading.RLock()
    results, frozen, storage = {}, {}, {}
    task_state = {task: {'status': 'waiting_for_policy'} for task in expected}
    attempts = dict(plan.get('inherited_attempt_counts', {}))
    if (job/'attempt_counts.json').exists():
        attempts.update(read(job/'attempt_counts.json'))

    def update(task=None, **changes):
        with guard:
            if task:
                task_state[task].update(changes)
            summary = {'tasks': dict(results), 'expected': plan['expected_test_episodes'],
                'scored': sum(r['scored'] for r in results.values()),
                'successes': sum(r['successes'] for r in results.values())}
            write(job/'results.json', summary)
            write(job/'status.json', {'status': 'running', 'phase': 'evaluation',
                'release_policy': 'per_task', 'evaluation_gpus': gpus,
                'workers_per_gpu': plan['evaluation_workers'], 'updated': time.time(),
                'development_batch': str(batch), 'completed_tasks': sum(r['complete'] for r in results.values()),
                'task_status': task_state, **{k:v for k,v in summary.items() if k != 'tasks'}})

    def evaluate(task, root, freeze, gpu, inherited=None):
        budget = str(freeze.get('checkpoint', 0))
        gpu_job = job/f'gpu_{gpu}'
        gpu_job.mkdir(exist_ok=True)
        if inherited:
            wait_for_inherited_evaluation(gpu_job, {**plan, 'inherited_evaluation': inherited},
                lambda phase, **kw: update(task, status=phase, gpu=gpu))
        if frozen_policy(root, task, plan.get('allow_interaction', False)) != freeze:
            raise ValueError(f'Frozen policy changed after admission: {task}')
        summary = score_summary(root, task)
        if summary['complete']:
            return summary
        receipt = evaluation_storage(root, task, plan['evaluation_root'], Path(plan['repository'])/'runs/robomemarena')
        with guard:
            storage[task] = receipt
            write(job/'evaluation_storage.json', storage)
        update(task, status='waiting_for_gpu', gpu=gpu)
        with gpu_lease(gpu_job, gpu, 'evaluation', plan.get('allow_gpu_sharing', False), plan.get('min_free_memory_mib', 16384)):
            for attempt in range(attempts.get(task, 0), 3):
                if frozen_policy(root, task, plan.get('allow_interaction', False)) != freeze:
                    raise ValueError(f'Frozen policy changed after admission: {task}')
                summary = score_summary(root, task)
                if summary['complete']:
                    break
                for episode in summary['infrastructure_episodes']:
                    command(f'recover_{task}_{episode}', [plan['python'], str(root/'runner.py'),
                        'recover-test', str(root), task, str(episode), '--budget', budget, '--reason',
                        'Unscorable infrastructure failure; identical pinned policy, seed and limits under the recorded GPU scheduling policy.'])
                with guard:
                    attempts[task] = attempt+1
                    write(job/'attempt_counts.json', attempts)
                update(task, status='evaluating', gpu=gpu, pass_number=attempt+1)
                command(f'evaluate_{task}', [plan['python'], str(root/'runner.py'), 'evaluate', str(root),
                    '--gpus', str(gpu), '--workers-per-gpu', str(plan['evaluation_workers']),
                    '--budgets', budget, '--resume'], allow_failure=True)
        return score_summary(root, task)

    status = read(batch/'status.json')
    ready = ready_policy_roots(status, expected, allow_interaction=plan.get('allow_interaction', False))
    inherited = plan.get('inherited_evaluations', [])
    inherited_tasks = {item['task'] for item in inherited}
    submitted, active = set(), {}
    for task, (root, freeze) in ready.items():
        frozen[task] = freeze
        if task in inherited_tasks:
            continue
        summary = score_summary(root, task)
        if summary['complete'] or attempts.get(task, 0) >= 3:
            results[task] = summary
            submitted.add(task)
            task_state[task]['status'] = 'finished' if summary['complete'] else 'needs_attention'
    write(job/'frozen_policies.json', frozen)
    update()
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for item in inherited:
            task, gpu = item['task'], item['gpu']
            if task not in ready or gpu not in gpus or gpu in active or task in submitted:
                raise ValueError('Inherited evaluation does not match a free assigned GPU/task')
            root, freeze = ready[task]
            submitted.add(task)
            active[gpu] = (executor.submit(evaluate, task, root, freeze, gpu, item), task)
        while True:
            status = read(batch/'status.json')
            ready = ready_policy_roots(status, expected, allow_interaction=plan.get('allow_interaction', False))
            for gpu, (future, task) in list(active.items()):
                if not future.done():
                    continue
                try:
                    result = future.result()
                    with guard:
                        results[task] = result
                    update(task, status='finished' if result['complete'] else 'needs_attention')
                except Exception as exc:
                    update(task, status='needs_attention', error=str(exc))
                del active[gpu]
            for gpu in gpus:
                if gpu in active:
                    continue
                task = next((t for t in expected if t in ready and t not in submitted), None)
                if task is None:
                    continue
                root, freeze = ready[task]
                with guard:
                    frozen[task] = freeze
                    write(job/'frozen_policies.json', frozen)
                submitted.add(task)
                active[gpu] = (executor.submit(evaluate, task, root, freeze, gpu), task)
            stopped = status['status'] in ('finished', 'needs_attention')
            if not stopped:
                stopped = not alive(read(batch/'process.json')['pid'])
            if stopped:
                for task in expected:
                    if task not in ready and task not in submitted:
                        submitted.add(task)
                        update(task, status='development_incomplete')
            if len(submitted) == len(expected) and not active:
                break
            update()
            time.sleep(5)
    update()
    final = read(job/'status.json')
    complete = set(results) == set(expected) and all(r['complete'] for r in results.values())
    final.update(status='finished' if complete else 'needs_attention',
                 tsr=final['successes']/plan['expected_test_episodes'] if complete else None)
    write(job/'status.json', final)


def start_evaluation_ready(parent, evaluation_workers=None, evaluation_gpus=None):
    """Continue host scheduling while preserving learners and active evaluations."""
    from cam.io import copy_tree, sha256, tree_hash

    parent = parent.resolve()
    old = read(parent/'plan.json')
    resizing = evaluation_workers is not None or evaluation_gpus is not None
    if resizing and evaluation_workers is None:
        evaluation_workers = old['evaluation_workers']
    if resizing and (not 1 <= evaluation_workers <= 10 or old.get('evaluation_release') != 'per_task'):
        raise ValueError('Resize requires a per-task RoboMemArena evaluator and 1–10 episode workers')
    if evaluation_gpus is not None and (not evaluation_gpus or min(evaluation_gpus) < 0 or len(set(evaluation_gpus)) != len(evaluation_gpus)):
        raise ValueError('Choose distinct nonnegative evaluation GPUs')
    progress = read(parent/'progress.json')
    batch = Path(progress['development_batch'])
    if tree_hash(parent/'inputs/checkout/src/cam') != old['harness_hash']:
        raise ValueError('Original experiment snapshot changed')
    ready_policy_roots(read(batch/'status.json'), old['tasks'])
    pid = read(parent/'process.json')['pid']
    if not alive(pid) or str(parent).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Original supervisor is not the expected live process')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    job = (ROOT/'outputs/robomemarena_full'/tag).resolve()
    job.mkdir(parents=True)
    copy_tree(parent/'inputs/checkout', job/'inputs/checkout')
    shutil.copy2(__file__, job/'driver.py')
    plan = {**old, 'tag': tag, 'created': time.time(), 'supersedes': str(parent),
        'parent_plan_sha256': sha256(parent/'plan.json'), 'evaluation_release': 'per_task',
        'development_batch': str(batch), 'benchmark_source': str(job/'inputs/checkout/third_party/robomemarena/evaluation_benchmark')}
    if resizing:
        plan['evaluation_workers'] = evaluation_workers
    if evaluation_gpus is not None:
        plan['evaluation_gpus'] = evaluation_gpus
    # Any earlier inherited job already finished before a new queue can run.
    plan.pop('inherited_evaluation', None)
    plan.pop('inherited_evaluations', None)
    write(job/'plan.json', plan)
    write(job/'progress.json', progress)
    (job/'README.md').write_text(
        '# RoboMemArena evaluation as policies finish\n\n'
        f'Follow the existing development batch `{batch}`. Each finished demo-only policy is '
        'verified frozen with zero training attempts before its 50 assigned held-out episodes. '
        'Other learners continue unchanged. All 26 tasks remain assigned (1,300 tests). '
        'Prepared snapshots, prompts, policies and test identities are unchanged.\n\n'
        f'Evaluation uses GPUs {plan.get("evaluation_gpus", [plan["gpu"]])}, at most {plan["evaluation_workers"]} workers per GPU, '
        f'and the existing {plan["min_free_memory_mib"]} MiB free-memory admission rule. '
        f'Rollouts and infrastructure retry archives go to `{plan["evaluation_root"]}/evaluation/`. '
        'Only unscorable infrastructure episodes are archived and retried identically, at most '
        'three passes. Scored failures remain scored. Test feedback never reaches learners.\n')
    os.kill(pid, signal.SIGSTOP)
    try:
        previous = read(parent/'status.json')
        if previous.get('phase') != 'development' and not (resizing and previous.get('phase') in (
                'evaluation', 'waiting_for_ready_policy', 'draining_previous_evaluation')):
            raise ValueError('Original supervisor advanced; preserve any active evaluation')
        write(job/'parent_status.json', previous)
        if resizing:
            inherited = []
            counts = dict(old.get('inherited_attempt_counts', {}))
            roots = {str(Path(r['run']).resolve()): r['task'] for r in read(batch/'status.json')['tasks'].values()}
            for command_file in sorted((parent/'logs').glob('evaluate_*.command.json')):
                proc_file = command_file.with_name(command_file.name.replace('.command.json', '.process.json'))
                if not proc_file.exists():
                    continue
                argv = read(command_file)['argv']
                if len(argv) < 4 or argv[2] != 'evaluate' or argv[3] not in roots:
                    raise ValueError('Unrecognized evaluation command at handoff')
                task = roots[argv[3]]
                counts[task] = counts.get(task, 0)+1
                process = process_identity(read(proc_file)['pid'])
                if process:
                    if Path(f'/proc/{process["pid"]}/cmdline').read_bytes().split(b'\0')[:-1] != [a.encode() for a in argv]:
                        raise ValueError('Active evaluation process identity changed')
                    inherited.append({'task': task, 'gpu': int(argv[argv.index('--gpus')+1]), 'command': str(command_file),
                        'command_sha256': sha256(command_file), 'process': process})
            prior_jobs = old.get('inherited_evaluations', []) + ([old['inherited_evaluation']] if old.get('inherited_evaluation') else [])
            for prior in prior_jobs:
                if process_identity(prior['process']['pid']) == prior['process']:
                    inherited.append({**prior, 'gpu': prior.get('gpu', old['gpu'])})
            assigned_gpus = plan.get('evaluation_gpus', [plan['gpu']])
            if len({r['gpu'] for r in inherited}) != len(inherited) or any(r['gpu'] not in assigned_gpus for r in inherited):
                raise ValueError('Preserve every active GPU and at most one active task per GPU')
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            if any(alive(int(child)) and int(child) not in {r['process']['pid'] for r in inherited} for child in children):
                raise ValueError('Unrecorded active child at handoff; retry after its command is recorded')
            plan['inherited_attempt_counts'] = counts
            if inherited:
                if len(assigned_gpus) > 1:
                    plan['inherited_evaluations'] = inherited
                else:
                    plan['inherited_evaluation'] = inherited[0]
            write(job/'plan.json', plan)
            if (parent/'results.json').exists():
                shutil.copy2(parent/'results.json', job/'results_before_resize.json')
            with (job/'README.md').open('a') as handle:
                handle.write(f'\nUser-requested concurrency change: GPUs {assigned_gpus}, {evaluation_workers} episode workers per GPU. Let inherited task queues finish at their original concurrency while additional GPUs admit other ready tasks immediately. Do not interrupt native workers, repeat scored episodes or reset infrastructure retry counts. Previous scheduler logs and scores remain in `{parent}`.\n')
        os.kill(pid, signal.SIGTERM)
    finally:
        if alive(pid):
            os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic()+10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.1)
    if alive(pid):
        raise RuntimeError('Old supervisor did not exit; no replacement launched')
    write(parent/'status.json', {'status': 'superseded_by_evaluation_resize' if resizing else 'superseded_by_per_task_evaluation',
        'replacement': str(job), 'development_batch_preserved': str(batch), 'updated': time.time()})
    write(parent/'evaluation_followup.json', {'job': str(job), 'release_policy': 'per_task'})
    return launch(job)


def wait_for_baseline_evaluation(job, plan, state):
    """Release the follow-up only after complete scored coverage of the baseline."""
    from cam.io import sha256

    baseline = Path(plan['wait_for_evaluation'])
    previous = read(baseline/'plan.json')
    while True:
        if sha256(baseline/'plan.json') != plan['parent_plan_sha256']:
            raise ValueError('Baseline evaluation plan changed')
        path = baseline/'results.json'
        result = read(path) if path.exists() else {'scored': 0, 'tasks': {}}
        complete = (set(result['tasks']) == set(previous['tasks'])
            and result.get('expected') == previous['expected_test_episodes']
            and result['scored'] == previous['expected_test_episodes']
            and all(r['complete'] and r['scored'] == r['expected'] == 50 for r in result['tasks'].values()))
        if complete:
            for task, summary in result['tasks'].items():
                frozen_policy(Path(summary['run']), task)
                if score_summary(Path(summary['run']), task) != summary:
                    raise ValueError(f'Baseline evaluation summary changed: {task}')
            write(job/'baseline_evaluation_complete.json', {'baseline': str(baseline),
                'scored': result['scored'], 'successes': result['successes'],
                'results_sha256': sha256(path), 'checked': time.time()})
            return
        state('waiting_for_baseline_evaluation', baseline=str(baseline),
              baseline_scored=result['scored'], baseline_expected=previous['expected_test_episodes'])
        if not alive(read(baseline/'process.json')['pid']):
            raise RuntimeError('Baseline evaluation stopped before every episode was scored')
        time.sleep(20)


def queue_evaluation_followup(baseline, preparation, evaluation_root=None, restart_cancelled=False, evaluation_workers=5):
    """Queue the separate 25-demo evaluation without touching either learner."""
    from cam.io import copy_tree, sha256, tree_hash

    if not 1 <= evaluation_workers <= 10:
        raise ValueError('Use 1–10 evaluation workers per GPU')
    baseline, preparation = baseline.resolve(), preparation.resolve()
    with (preparation/'.evaluation_followup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        if (preparation/'evaluation_followup.json').exists():
            if not restart_cancelled:
                raise FileExistsError('The 25-demo experiment already has an evaluation follow-up')
            previous = read(preparation/'evaluation_followup.json')
            previous_job = Path(previous['job'])
            if (read(previous_job/'status.json')['status'] != 'cancelled_by_user'
                    or not (previous_job/'cancelled_by_user.json').is_file()
                    or alive(read(previous_job/'process.json')['pid'])):
                raise ValueError('Only a stopped, explicitly cancelled follow-up can be replaced')
        elif restart_cancelled:
            raise ValueError('No cancelled evaluation follow-up to replace')
        original = read(baseline/'plan.json')
        batch = Path(read(preparation/'batch.json')['batch'])
        learning = read(batch/'plan.json')
        cfg = learning['benchmarks']['robomemarena']
        if cfg['tasks'] != original['tasks'] or cfg['demo_count'] != 25 or cfg['interaction_count'] != 0:
            raise ValueError('Expected the corresponding all-task 25-demo demo-only batch')
        if tree_hash(batch/'inputs/checkout/src/cam') != learning['harness_hash']:
            raise ValueError('The 25-demo preparation snapshot changed')
        rows = read(batch/'status.json')['tasks']
        baseline_rows = read(Path(original['development_batch'])/'status.json')['tasks']
        for row in rows.values():
            task, root = row['task'], Path(row['run'])
            run = read(root/'run.json')
            old_run = read(Path(baseline_rows[f'robomemarena/{task}']['run'])/'run.json')
            assigned = read(root/'inputs/dataset.json')['tasks'][task]['episodes']
            if (run['config']['mode'], run['config']['budget'], run['config']['depth']) != ('demos', 0, True):
                raise ValueError(f'Unexpected learning setting: {task}')
            if len(assigned) != 25 or run['test_episodes'][task] != old_run['test_episodes'][task]:
                raise ValueError(f'Demo count or test episode assignment differs: {task}')
            test_path = Path('inputs/metadata/test')/f'record_dataset_{task}_metadata.json'
            if sha256(root/test_path) != sha256(Path(baseline_rows[f'robomemarena/{task}']['run'])/test_path):
                raise ValueError(f'Held-out identities differ: {task}')
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        job = (ROOT/'outputs/robomemarena_full'/tag).resolve()
        job.mkdir(parents=True)
        copy_tree(batch/'inputs/checkout', job/'inputs/checkout')
        shutil.copy2(__file__, job/'driver.py')
        plan = {'tag': tag, 'created': time.time(), 'repository': str(ROOT), 'python': original['python'],
            'parent_job': str(baseline), 'parent_plan_sha256': sha256(baseline/'plan.json'),
            'wait_for_evaluation': str(baseline), 'evaluation_release': 'per_task',
            'development_batch': str(batch), 'tasks': cfg['tasks'], 'demo_count': 25,
            'gpu': 1, 'evaluation_gpus': [0, 1], 'evaluation_workers': evaluation_workers,
            'evaluation_root': str(Path(evaluation_root).resolve()) if evaluation_root is not None else original['evaluation_root'],
            'expected_test_episodes': 1300,
            'allow_gpu_sharing': original['allow_gpu_sharing'], 'min_free_memory_mib': original['min_free_memory_mib'],
            'simulator_python': learning['environment']['CAM_ROBOMEMARENA_PYTHON'],
            'harness_hash': learning['harness_hash'], 'benchmark_hash': cfg['source_hash'],
            'benchmark_source': str(job/'inputs/checkout/third_party/robomemarena/evaluation_benchmark'),
            'asset_hashes': cfg['asset_hashes'], 'preparation': str(preparation),
            'prompt_comparison': read(preparation/'prompt_comparison.json')}
        if previous:
            plan['restarts_cancelled_job'] = previous['job']
            write(job/'previous_evaluation_followup.json', previous)
        write(job/'plan.json', plan)
        write(job/'progress.json', {'development_batch': str(batch)})
        (job/'README.md').write_text(
            '# RoboMemArena: queued 25-demo held-out evaluation\n\n'
            f'Wait for all 1,300 baseline tests in `{baseline}` to be scored, regardless of '
            f'success rate. Then evaluate finished, frozen policies from `{batch}` as they '
            'become ready. Verify zero training attempts and unchanged frozen program hashes. '
            'The same 50 held-out episode identities per task are checked against the baseline '
            'before launch (26 tasks / 1,300 tests).\n\n'
            f'Use GPUs 0 and 1, one task per GPU and {evaluation_workers} episode workers per GPU, with the '
            'existing memory admission check. Use each run\'s own pinned runner, preserve scored '
            'outcomes and archive only infrastructure failures before identical retries. '
            f'Rollouts and retry archives go to `{plan["evaluation_root"]}/evaluation/`. '
            'No held-out evidence is sent to any learner.\n\n'
            'The 25-demo arm uses the revised shared offline-review prompt. Its exact hashes '
            'and the baseline hashes are recorded in plan.json; disclose this difference '
            'when comparing results. Original runs, policies, prompts and datasets are unchanged.\n')
        result = launch(job)
        if previous:
            write(preparation/'evaluation_followup_history'/f'{tag}.json', previous)
            write(preparation/'evaluation_reauthorized.json', {
                'previous': previous, 'current': result, 'time': time.time(),
                'reason': 'User reauthorized the full 25-demo held-out comparison.',
                'evaluation_root': plan['evaluation_root']})
        write(preparation/'evaluation_followup.json', result)
        write(baseline/'demos25_evaluation_followup.json', result)
        return result


def start_interaction_evaluation(batch, baseline, evaluation_root, evaluation_workers=5, evaluation_gpus=(0, 1)):
    """Launch final-checkpoint tests for a completed demo-seed interaction batch."""
    import sqlite3
    from cam.io import copy_tree, sha256, tree_hash

    batch, baseline = Path(batch).resolve(), Path(baseline).resolve()
    gpus = list(evaluation_gpus)
    if not 1 <= evaluation_workers <= 10 or not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError('Use distinct nonnegative GPUs and 1–10 workers per GPU')
    with (batch/'.held_out_evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch/'held_out_evaluation.json').exists():
            raise FileExistsError('This development batch already has a held-out evaluator')
        learning, status = read(batch/'plan.json'), read(batch/'status.json')
        cfg = learning['benchmarks']['robomemarena']
        if status['status'] != 'finished' or tree_hash(batch/'inputs/checkout/src/cam') != learning['harness_hash']:
            raise ValueError('Require a completed batch with an unchanged pinned harness')
        ready = ready_policy_roots(status, cfg['tasks'], allow_interaction=True)
        old_plan = read(baseline/'plan.json')
        old_rows = read(Path(old_plan['development_batch'])/'status.json')['tasks']
        if set(ready) != set(cfg['tasks']) or cfg['tasks'] != old_plan['tasks']:
            raise ValueError('Every task must be finished, frozen and match the baseline task set')
        checks = {}
        for task, (root, freeze) in ready.items():
            run = read(root/'run.json')
            config = run['config']
            if config['mode'] != 'both' or config.get('both_protocol') not in ('v2', 'demo_first') or config['training_seed_policy'] != 'demo_cycle':
                raise ValueError(f'Expected the requested demo-first demo-seed policy: {task}')
            folder = root/'tasks'/task
            state = read(folder/'state/config.json')
            demos = read(root/'inputs/dataset.json')['tasks'][task]['episodes']
            train = state['train_records']
            identities = lambda rows: {(int(r['episode']), int(r['seed'])) for r in rows}
            budget = config['budget']
            if len(demos) != len(train) or len(train) != budget or identities(demos) != identities(train) or len(identities(train)) != budget:
                raise ValueError(f'Expected one attempt on each assigned demo identity: {task}')
            with sqlite3.connect((folder/'state/ledger.sqlite').as_uri()+'?mode=ro', uri=True) as db:
                attempts = list(db.execute('SELECT episode,status FROM attempts ORDER BY id'))
            if len(attempts) != budget or {r[0] for r in attempts} != {r['episode'] for r in train} or any(r[1] == 'running' for r in attempts):
                raise ValueError(f'Full distinct demo-seed budget was not completed: {task}')
            old_root = Path(old_rows[f'robomemarena/{task}']['run'])
            old_run = read(old_root/'run.json')
            metadata = Path('inputs/metadata/test')/f'record_dataset_{task}_metadata.json'
            if run['test_episodes'][task] != old_run['test_episodes'][task] or sha256(root/metadata) != sha256(old_root/metadata):
                raise ValueError(f'Held-out identities differ from the baseline: {task}')
            test_seeds = {r['seed'] for r in read(root/metadata)['records']}
            if not {r['seed'] for r in train}.isdisjoint(test_seeds):
                raise ValueError(f'Training/test seed overlap: {task}')
            checks[task] = {'run': str(root), 'budget': budget, 'program_hash': freeze['program_hash'],
                'expected': len(run['test_episodes'][task]), 'same_demo_identities_once': True,
                'test_identity_match': True, 'no_test_seed_overlap': True}
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        job = (ROOT/'outputs/robomemarena_full'/tag).resolve()
        job.mkdir(parents=True)
        copy_tree(batch/'inputs/checkout', job/'inputs/checkout')
        shutil.copy2(__file__, job/'driver.py')
        plan = {'tag': tag, 'created': time.time(), 'repository': str(ROOT), 'python': learning['python'],
            'parent_job': str(batch), 'parent_plan_sha256': sha256(batch/'plan.json'),
            'evaluation_release': 'per_task', 'allow_interaction': True,
            'development_batch': str(batch), 'tasks': cfg['tasks'], 'demo_count': cfg['demo_count'],
            'gpu': gpus[0], 'evaluation_gpus': gpus, 'evaluation_workers': evaluation_workers,
            'evaluation_root': str(Path(evaluation_root).resolve()),
            'expected_test_episodes': sum(r['expected'] for r in checks.values()),
            'allow_gpu_sharing': True, 'min_free_memory_mib': 16384,
            'simulator_python': learning['environment']['CAM_ROBOMEMARENA_PYTHON'],
            'harness_hash': learning['harness_hash'], 'benchmark_hash': cfg['source_hash'],
            'benchmark_source': str(job/'inputs/checkout/third_party/robomemarena/evaluation_benchmark'),
            'asset_hashes': cfg['asset_hashes'], 'baseline_evaluation': str(baseline),
            'final_budgets': {task: r['budget'] for task, r in checks.items()}}
        write(job/'plan.json', plan)
        write(job/'launch_verification.json', checks)
        write(job/'progress.json', {'development_batch': str(batch)})
        (job/'README.md').write_text(
            '# RoboMemArena: final demo+interaction held-out evaluation\n\n'
            f'Evaluate {len(checks)} finished, frozen policies from `{batch}` on all '
            f'{plan["expected_test_episodes"]} assigned held-out episodes. Each task uses its final '
            'budget checkpoint: R050 for full assignments and R032 for T04. Test identities match '
            f'the demo-only baseline `{baseline}`. Full training budgets, demo-seed equality, '
            'test exclusion and final checkpoint hashes are checked before launch.\n\n'
            f'Use GPUs {gpus}, one task per GPU and {evaluation_workers} episode workers per GPU, '
            'with 16 GiB free-memory admission. Run each prepared policy through its original '
            'pinned runner. Preserve scored outcomes; archive and identically retry only '
            'unscorable infrastructure episodes. No further learning or test feedback to learners.\n\n'
            f'Rollout videos, RGB-D, decisions, actions, results and retry archives are stored '
            f'under `{plan["evaluation_root"]}/evaluation/`, grouped by task/model/setting/run tag. '
            'Canonical run paths link there. Report official TSR and scored/expected coverage. '
            'The interaction prompt supplement was revised after baseline review; report this '
            'provenance alongside the interaction difference.\n')
        result = launch(job)
        write(batch/'held_out_evaluation.json', result)
        return result


def launch(job):
    with (job / "supervisor.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, str(job / "driver.py"), "_worker", str(job)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=job)
    write(job / "process.json", {"pid": proc.pid, "started": time.time()})
    return {"job": str(job), "pid": proc.pid}


def start(args):
    import cam
    from cam.interface.base import pin_source, source_hash
    from cam.interface.registry import get_adapter, simulator_python
    from cam.harness.prepare import simulator_build_files
    from cam.io import copy_tree, tree_hash

    spec = read(ROOT / ("configs/robomemarena/experiments/demo_only_50.json" if args.direct_pool
                       else f"configs/robomemarena/experiments/demo_only_{args.demo_count}.json"))
    adapter = get_adapter("robomemarena")
    pool_link = existing_pool(args.pool_job, source_hash(adapter)) if args.pool_job else None
    pool_settings = read(args.pool_job / 'plan.json') if pool_link else {}
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = (ROOT / "outputs/robomemarena_full" / tag).resolve()
    job.mkdir(parents=True)
    checkout = job / "inputs/checkout"
    copy_tree(Path(cam.__file__).parent, checkout / "src/cam")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, checkout / name)
    source = checkout / "third_party/robomemarena/evaluation_benchmark"
    assets = pin_source(adapter, source)
    for recipe in simulator_build_files(adapter):
        destination = checkout / "envs/robomemarena" / recipe.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recipe, destination)
    shutil.copy2(__file__, job / "driver.py")
    (job / "runner.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'inputs/checkout/src'))\n"
        "from cam.cli import main\nraise SystemExit(main())\n")
    shutil.copy2(ROOT / "scripts/develop_batch.py", job / "develop_batch.py")
    replay_gpus = args.replay_gpus or [args.gpu]
    if args.replay_workers % len(replay_gpus):
        raise ValueError("Replay workers must divide evenly across replay GPUs")
    shutil.copy2(ROOT / "scripts/replay_gpu_dispatch.py", job / "replay_gpu_dispatch.py")
    dispatch = {"python": str(simulator_python(adapter)), "gpus": replay_gpus,
                "workers_per_gpu": args.replay_workers // len(replay_gpus),
                "min_free_memory_mib": args.min_free_memory_mib,
                "locks": str(job / "replay_slots"), "archive": str(job / "replay_archive")}
    if args.replay_cache_job:
        dispatch['locks'] = read(args.replay_cache_job / 'replay_dispatch.json')['locks']
    write(job / "replay_dispatch.json", dispatch)
    wrapper = job / "replay-python"
    wrapper.write_text(f"#!{sys.executable}\nfrom replay_gpu_dispatch import main\n"
                       f"main({str(job / 'replay_dispatch.json')!r})\n")
    wrapper.chmod(0o755)
    item = spec["benchmarks"]["robomemarena"]
    if args.direct_pool:
        item['datasets'] = [pool_link['pool']]
    config = read(ROOT / item["config"])
    write(job / "inputs/config.json", config)
    item["config"] = str(job / "inputs/config.json")
    item["datasets"] = [str((ROOT / p).resolve()) for p in item["datasets"]]
    write(job / "inputs/development.json", spec)
    preparation = (ROOT / args.preparation).resolve()
    old = read(preparation / "plan.json")
    data_root = (ROOT / "data/robomemarena_rgbd" / tag).resolve()
    plan = {"tag": tag, "created": time.time(), "repository": str(ROOT.resolve()),
            "python": sys.executable, "preparation": str(preparation),
            "pool": pool_link['pool'] if pool_link else (str(data_root / "pool") if args.reuse_dataset or args.pool_cap else old["benchmarks"]["robomemarena"]["pool"]),
            "existing_pool": pool_link,
            "subset": None if args.direct_pool else item["datasets"][0],
            "direct_pool": args.direct_pool,
            "learning_dataset": item['datasets'][0],
            "reuse_source": str(args.reuse_dataset.resolve()) if args.reuse_dataset else None,
            "reused_dataset": (str(args.reuse_dataset.resolve()) if args.reuse_in_place else str(data_root / 'reused')) if args.reuse_dataset else None,
            "reuse_in_place": args.reuse_in_place,
            "pool_cap": args.pool_cap,
            "single_pass": pool_settings.get('single_pass', args.single_pass),
            "validation_episodes": pool_settings.get('validation_episodes', args.validation_episodes),
            "replay_cache_job": str(args.replay_cache_job.resolve()) if args.replay_cache_job else None,
            "raw": str((ROOT / "data/raw/robomemarena").resolve()),
            "tasks": item["tasks"], "demo_count": item['demo_count'], "coding_workers": args.coding_workers,
            "gpu": args.gpu, "evaluation_workers": args.evaluation_workers,
            "evaluation_root": str(args.evaluation_root.resolve()),
            "replay_workers": args.replay_workers, "allow_gpu_sharing": args.allow_gpu_sharing,
            "replay_gpus": replay_gpus, "replay_workers_per_gpu": dispatch["workers_per_gpu"],
            "replay_python": str(wrapper),
            "min_free_memory_mib": args.min_free_memory_mib,
            "supersedes": str(args.supersedes.resolve()) if args.supersedes else None,
            "expected_test_episodes": 1300, "held_out_evaluation": True,
            "simulator_python": str(simulator_python(adapter)),
            "benchmark_source": str(source), "benchmark_hash": source_hash(adapter),
            "asset_hashes": assets,
            "harness_hash": tree_hash(job / "inputs/checkout/src/cam")}
    write(job / "plan.json", plan)
    (job / "README.md").write_text(
        "# RoboMemArena: all-task demo-only experiment\n\n"
        f"26 tasks, {'up to ' if args.direct_pool else ''}{item['demo_count']} successful replay-derived 256x256 RGB-D demos per task, "
        "Claude Code / Opus 5.5, zero training rollouts. Each policy is frozen "
        "before testing all 50 assigned held-out episodes: 1,300 expected total.\n\n"
        f"The preparation retains at most {args.pool_cap} successful captures per task, "
        "stopping new verification at the cap or after all released identities are exhausted. "
        "Earlier verification outcomes above this cap remain archived.\n\n"
        + (f"GPU sharing is enabled: replay on GPUs {replay_gpus}, "
           f"{dispatch['workers_per_gpu']} workers per GPU ({args.replay_workers} total). "
           f"Check at least {args.min_free_memory_mib} MiB free before each native replay job. "
           f"Evaluation uses GPU {args.gpu} with {args.evaluation_workers} workers. "
           "Existing MIKASA/RMBench jobs remain active. "
           if args.allow_gpu_sharing else
           "The supervisor waits for the MIKASA replay queue and an idle GPU; it "
           "shares the GPU lock with RMBench RGB-D preparation. ")
        + (f"It reuses the ongoing immutable replay preparation at `{pool_link['source_job']}` "
           f"and waits for its final pool at `{pool_link['pool']}`. Existing native workers continue unchanged; "
           "their dispatch archives and copied four-task captures stay in that preparation. "
           if pool_link else
           f"It verifies the existing four-task RGB-D dataset at `{plan['reused_dataset']}`, "
           "and copies its successful captures into a fresh pool without replaying them. "
           "Saved replay outcomes are reused only after raw-file and protocol checks. "
           if args.reuse_dataset else "It prepares a fresh capped RoboMemArena pool from the pinned release. ")
        + ("It uses every demo in the capped local dataset directly, records actual per-task shortfalls, and creates no smaller subset. It launches new timestamped sandboxed "
           if args.direct_pool else f"It creates a separate exactly-{item['demo_count']} subset, then launches new timestamped sandboxed ")
        + "learning runs. No older policy or private audit finding is given to learners.\n\n"
        "The host interpreter wrapper schedules the unchanged extraction workers. "
        "`replay_archive/` retains each native job, result, log and physical GPU assignment; "
        "`prior_replay_work/` preserves existing scratch evidence before resume.\n\n"
        "All learners must finish before evaluation starts. Scored failures are "
        "preserved. Unscorable infrastructure failures are archived with "
        "`cam recover-test` before an identical retry (at most three evaluation "
        "passes). Persistent failures stop with `needs_attention`; they are never "
        "dropped or counted as successes. `results.json` records actual coverage and TSR.\n\n"
        "Inspect `status.json`, `progress.json`, and `logs/`. Resume this unchanged "
        "supervisor with `scripts/robomemarena_full.py resume JOB`. "
        "Earlier four-task 25-demo runs and all raw data remain separate.\n")
    if args.replay_cache_job:
        with (job / 'README.md').open('a') as handle:
            handle.write(f"\nReplay evidence and active native jobs are adopted from `{args.replay_cache_job.resolve()}`. "
                f"The shared slot directory `{dispatch['locks']}` counts inherited workers toward the new per-GPU limit. "
                f"The previously copied RGB-D source remains at `{plan['reused_dataset']}`; this job reads it in place and copies retained episode groups into its new pool.\n")
    if args.single_pass:
        with (job / 'README.md').open('a') as handle:
            handle.write(f"\nMissing RGB-D demonstrations are captured and scored in one replay. Known failed identities are skipped; completed RGB-D captures are reused. Only {args.validation_episodes} deterministically sampled captured identities per task receive an independent replay check. Per-episode checkpoints and unsuccessful captures remain host-only.\n")
    with (job/'README.md').open('a') as handle:
        handle.write(f"\nHeld-out rollout H5, videos, actions, decisions, results and infrastructure-retry archives are stored under `{plan['evaluation_root']}/evaluation/`, by task/model/setting/run timetag. Canonical run paths link to this NFS storage. Training demos remain on the local SSD. Storage is configured before the first test episode.\n")
    return launch(job)


def worker(job):
    from cam.interface.base import source_hash
    from cam.interface.registry import get_adapter
    from cam.io import tree_hash

    plan = read(job / "plan.json")
    assert tree_hash(job / "inputs/checkout/src/cam") == plan["harness_hash"]
    assert source_hash(get_adapter("robomemarena")) == plan["benchmark_hash"]
    for link, digest in plan["asset_hashes"].items():
        assert tree_hash(Path(plan["benchmark_source"]) / link) == digest
    os.environ["CAM_ROBOMEMARENA_PYTHON"] = plan["simulator_python"]
    progress = read(job / "progress.json") if (job / "progress.json").exists() else {}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("CAM_SIMULATOR_LOCK_DIR", None)
    env.pop("CAM_SIMULATOR_GPUS", None)

    def state(phase, **extra):
        write(job / "status.json", {"status": "running", "phase": phase, "updated": time.time(), **extra})

    def command(label, argv, extra=None, allow_failure=False):
        stem = job / "logs" / f"{label}_{time.time_ns()}"
        write(stem.with_suffix(".command.json"), {"argv": argv, "started": time.time()})
        with stem.with_suffix(".log").open("xb") as log:
            proc = subprocess.Popen(argv, env={**env, **(extra or {})}, stdout=log, stderr=subprocess.STDOUT)
            write(stem.with_suffix(".process.json"), {"pid": proc.pid, "started": time.time()})
            code = proc.wait()
        if code and not allow_failure:
            raise RuntimeError(f"Command exit {code}: {stem.with_suffix('.log')}")
        return code

    with (job / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan.get('evaluation_release') == 'per_task':
            from cam.io import sha256
            parent_job = plan.get('parent_job') or plan['supersedes']
            if sha256(Path(parent_job)/'plan.json') != plan['parent_plan_sha256']:
                raise ValueError('Original experiment plan changed')
            if plan.get('wait_for_evaluation'):
                wait_for_baseline_evaluation(job, plan, state)
            return evaluate_ready(job, plan, command, state)
        old_root = Path(plan["preparation"])
        old = read(old_root / "plan.json")
        pool = Path(plan["pool"])
        if plan.get('existing_pool'):
            # This host wait launches no simulator. The existing native
            # dispatcher already enforces GPU slots and free-memory admission.
            wait_for_pool(job, plan['existing_pool'])
        if plan.get("reuse_source"):
            reused = Path(plan["reused_dataset"])
            marker = job / "reuse_copy_complete.json"
            if not marker.exists():
                state("copy_existing_demos", source=plan["reuse_source"], destination=str(reused))
                if reused.exists() and not plan.get('reuse_in_place'):
                    archive = reused.parent / "interrupted" / f"reused_{time.time_ns()}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(reused, archive)
                if not plan.get('reuse_in_place'):
                    shutil.copytree(plan["reuse_source"], reused)
                source_manifest = read(reused / "manifest.json")
                write(marker, {"source": plan["reuse_source"], "destination": str(reused),
                    "counts": {t: len(v["episodes"]) for t, v in source_manifest["tasks"].items()},
                    "finished": time.time()})
        if not (pool / "manifest.json").exists():
            # Resume the immutable extraction snapshot, never substitute current code.
            assert tree_hash(old_root / "inputs/checkout/src/cam") == old["harness_hash"]
            args = read(old_root / "robomemarena.extract.command.json")["argv"] + ["--resume"]
            args[args.index("--workers") + 1] = str(plan.get("replay_workers", 4))
            if plan.get("reuse_source") or plan.get('pool_cap'):
                args = [plan["python"], str(job / "runner.py"), "extract", "--benchmark", "robomemarena",
                    "--benchmark-source", plan["benchmark_source"], "--raw-data", plan["raw"],
                    "--output", str(pool), "--count", "all", "--workers", str(plan["replay_workers"]),
                    "--no-videos"]
                if plan.get('reused_dataset'):
                    args.extend(['--reuse-dataset', plan['reused_dataset']])
                if plan.get('pool_cap'):
                    args.extend(['--max-successful', str(plan['pool_cap'])])
                if plan.get('replay_cache_job'):
                    args.extend(['--replay-cache', plan['replay_cache_job']])
                if plan.get('single_pass'):
                    args.extend(['--single-pass', '--validation-episodes', str(plan['validation_episodes'])])
                if pool.exists():
                    args.append("--resume")
            with (old_root / "worker.lock").open("a") as preparation_lock:
                fcntl.flock(preparation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (pool / "work").exists() and not (job / "prior_replay_work").exists():
                    shutil.copytree(pool / "work", job / "prior_replay_work")
                with ExitStack() as leases:
                    for gpu in sorted(plan.get("replay_gpus", [plan["gpu"]])):
                        # GPU memory admission belongs to each native job. The host
                        # extractor may validate/reuse existing files while GPUs are busy.
                        if not plan.get('allow_gpu_sharing'):
                            leases.enter_context(gpu_lease(job, gpu, "demo_pool"))
                    state("demo_pool", replay_gpus=plan.get("replay_gpus", [plan["gpu"]]),
                          replay_workers=plan.get("replay_workers", 4),
                          replay_workers_per_gpu=plan.get("replay_workers_per_gpu"))
                    command("resume_pool", args, {"CUDA_VISIBLE_DEVICES": str(plan["gpu"]),
                        "CAM_ROBOMEMARENA_PYTHON": plan.get("replay_python", old["benchmarks"]["robomemarena"]["simulator_python"])})
        manifest = read(pool / "manifest.json")
        counts = {t: len(manifest["tasks"].get(t, {}).get("episodes", [])) for t in plan["tasks"]}
        write(job / "demo_counts.json", counts)
        assignment_path = job / 'inputs/development.json'
        if plan.get('direct_pool'):
            counts = direct_demo_counts(manifest, plan['tasks'], plan['demo_count'])
            assignment = read(assignment_path)
            item = assignment['benchmarks']['robomemarena']
            item.update(datasets=[str(pool)], demo_counts=counts)
            assignment_path = job / 'development_assignment.json'
            write(assignment_path, assignment)
            write(job / 'demo_shortfalls.json', {t: n for t, n in counts.items() if n < plan['demo_count']})
        else:
            require_demo_counts(manifest, plan['tasks'], plan['demo_count'])
            subset = Path(plan['subset'])
            if not (subset / 'manifest.json').exists():
                state('subset')
                args = [plan['python'], str(job / 'runner.py'), 'subset', '--dataset', str(pool),
                        '--output', str(subset), '--count', str(plan['demo_count'])]
                if subset.exists():
                    args.append('--resume')
                command('subset', args)
        if "development_batch" not in progress:
            state("starting_development")
            spec = importlib.util.spec_from_file_location("develop_batch", job / "develop_batch.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ROOT = Path(plan["repository"])
            from types import SimpleNamespace
            result = module.start(SimpleNamespace(plan=assignment_path,
                workers=plan["coding_workers"], gpus=[plan["gpu"]]))
            progress["development_batch"] = result["batch"]
            write(job / "progress.json", progress)
        batch = Path(progress["development_batch"])
        while True:
            status = read(batch / "status.json") if (batch / "status.json").exists() else {"status": "starting"}
            state("development", batch=str(batch), development=status)
            if status["status"] == "finished":
                break
            if status["status"] == "needs_attention" or not alive(read(batch / "process.json")["pid"]):
                raise RuntimeError(f"Development needs attention: {batch}")
            time.sleep(20)
        roots = {row["task"]: Path(row["run"]) for row in status["tasks"].values()}
        if set(roots) != set(plan["tasks"]):
            raise ValueError("Development batch does not cover every assigned task")
        write(job / "frozen_policies.json", {t: frozen_policy(root, t) for t, root in roots.items()})
        if plan.get('evaluation_root'):
            storage = {t: evaluation_storage(root, t, plan['evaluation_root'],
                       Path(plan['repository'])/'runs/robomemarena') for t, root in roots.items()}
            write(job/'evaluation_storage.json', storage)
        results = {}
        with gpu_lease(job, plan["gpu"], "evaluation", plan.get("allow_gpu_sharing", False),
                       plan.get("min_free_memory_mib", 16384)):
            for task in plan["tasks"]:
                root = roots[task]
                frozen_policy(root, task)
                for attempt in range(3):
                    summary = score_summary(root, task)
                    if summary["complete"]:
                        break
                    for episode in summary["infrastructure_episodes"]:
                        command(f"recover_{task}_{episode}", [plan["python"], str(root / "runner.py"),
                            "recover-test", str(root), task, str(episode), "--budget", "0", "--reason",
                            "Unscorable infrastructure failure; identical pinned policy, seed and limits under the recorded GPU scheduling policy."])
                    state("evaluation", task=task, pass_number=attempt + 1, completed_tasks=len(results))
                    command(f"evaluate_{task}", [plan["python"], str(root / "runner.py"), "evaluate", str(root),
                        "--gpus", str(plan["gpu"]), "--workers-per-gpu", str(plan["evaluation_workers"]),
                        "--budgets", "0", "--resume"], allow_failure=True)
                results[task] = score_summary(root, task)
                write(job / "results.json", {"tasks": results,
                    "expected": plan["expected_test_episodes"],
                    "scored": sum(r["scored"] for r in results.values()),
                    "successes": sum(r["successes"] for r in results.values())})
        complete = len(results) == 26 and all(r["complete"] for r in results.values())
        successes = sum(r["successes"] for r in results.values())
        write(job / "status.json", {"status": "finished" if complete else "needs_attention",
            "phase": "evaluation", "updated": time.time(), "expected": 1300,
            "scored": sum(r["scored"] for r in results.values()), "successes": successes,
            "tsr": successes / 1300 if complete else None, "development_batch": str(batch)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--preparation", type=Path, default=Path("outputs/demo_preparation/20261107T174722_533247Z"))
    start_parser.add_argument("--gpu", type=int, default=1)
    start_parser.add_argument("--coding-workers", type=int, default=8)
    start_parser.add_argument("--evaluation-workers", type=int, default=5)
    start_parser.add_argument('--evaluation-root', type=Path, default=EVALUATION_ROOT)
    start_parser.add_argument("--replay-workers", type=int, choices=range(1, 9), default=8)
    start_parser.add_argument("--replay-gpus", type=int, nargs="+", default=[0, 1])
    start_parser.add_argument("--allow-gpu-sharing", action="store_true")
    start_parser.add_argument("--min-free-memory-mib", type=int, default=16384)
    start_parser.add_argument("--supersedes", type=Path)
    start_parser.add_argument("--reuse-dataset", type=Path)
    start_parser.add_argument('--reuse-in-place', action='store_true', help='Read an already copied reusable dataset in place; copied episode groups still go into the new pool')
    start_parser.add_argument('--pool-cap', type=int, default=50)
    start_parser.add_argument('--single-pass', action='store_true', help='Capture RGB-D and score success together')
    start_parser.add_argument('--validation-episodes', type=int, default=0, help='Independent replay checks per task after single-pass capture')
    start_parser.add_argument('--replay-cache-job', type=Path, help='Adopt completed and active native replay jobs from an immutable earlier preparation')
    start_parser.add_argument("--demo-count", type=int, choices=(25, 50), default=25)
    start_parser.add_argument('--direct-pool', action='store_true', help='Use the full capped local dataset directly, reporting actual per-task demo counts')
    start_parser.add_argument("--pool-job", type=Path, help="Use the final pool from an existing pinned replay preparation; preserve its active extraction")
    for command in ("status", "resume", "_worker"):
        sub.add_parser(command).add_argument("job", type=Path)
    ready_parser = sub.add_parser('evaluate-ready')
    ready_parser.add_argument('job', type=Path)
    ready_parser.add_argument('--evaluation-workers', type=int, help='Continue an active per-task evaluator with 1–10 workers after its current task finishes')
    ready_parser.add_argument('--evaluation-gpus', type=int, nargs='+', help='Evaluate one task per assigned GPU concurrently, preserving active task queues')
    followup_parser = sub.add_parser('queue-followup')
    followup_parser.add_argument('baseline', type=Path)
    followup_parser.add_argument('preparation', type=Path)
    followup_parser.add_argument('--evaluation-root', type=Path, help='Artifact parent; defaults to the baseline root')
    followup_parser.add_argument('--evaluation-workers', type=int, default=5, help='1–10 episode workers per GPU')
    followup_parser.add_argument('--restart-cancelled', action='store_true', help='User-authorized fresh scheduler replacing a stopped, cancelled follow-up; preserve the old job')
    interaction_parser = sub.add_parser('evaluate-interaction')
    interaction_parser.add_argument('batch', type=Path)
    interaction_parser.add_argument('--baseline', type=Path, required=True, help='Demo-only evaluator defining the held-out comparison identities')
    interaction_parser.add_argument('--evaluation-root', type=Path, required=True)
    interaction_parser.add_argument('--evaluation-workers', type=int, default=5)
    interaction_parser.add_argument('--evaluation-gpus', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.command == "start":
        if args.direct_pool and (not args.pool_job or args.demo_count != 50):
            parser.error('--direct-pool requires --pool-job and --demo-count 50')
        if args.validation_episodes < 0 or (args.validation_episodes and not args.single_pass):
            parser.error('--validation-episodes requires --single-pass and a nonnegative count')
        if args.single_pass and args.pool_job:
            parser.error('--single-pass requires a fresh preparation, not --pool-job')
        if args.pool_job and args.reuse_dataset:
            parser.error('--pool-job and --reuse-dataset are mutually exclusive')
        if args.reuse_in_place and not args.reuse_dataset:
            parser.error('--reuse-in-place requires --reuse-dataset')
        if args.replay_cache_job and (args.pool_job or not args.reuse_dataset):
            parser.error('--replay-cache-job requires --reuse-dataset and a fresh pool')
        if args.pool_cap < args.demo_count:
            parser.error('--pool-cap cannot be below the learning demo count')
        if args.gpu < 0 or args.coding_workers < 1 or not 1 <= args.evaluation_workers <= 10 or args.min_free_memory_mib < 1:
            parser.error("Use a nonnegative GPU, positive coding workers/memory floor and 1–10 evaluation workers")
        gpus = args.replay_gpus or [args.gpu]
        if len(set(gpus)) != len(gpus) or min(gpus) < 0 or args.replay_workers % len(gpus):
            parser.error("Choose distinct nonnegative replay GPUs and evenly divisible replay workers")
        sys.path.insert(0, str(ROOT / "src"))
        print(json.dumps(start(args), indent=2))
    elif args.command == 'evaluate-ready':
        sys.path.insert(0, str(ROOT/'src'))
        print(json.dumps(start_evaluation_ready(args.job, args.evaluation_workers, args.evaluation_gpus), indent=2))
    elif args.command == 'queue-followup':
        sys.path.insert(0, str(ROOT/'src'))
        print(json.dumps(queue_evaluation_followup(args.baseline, args.preparation,
            args.evaluation_root, args.restart_cancelled, args.evaluation_workers), indent=2))
    elif args.command == 'evaluate-interaction':
        sys.path.insert(0, str(ROOT/'src'))
        print(json.dumps(start_interaction_evaluation(args.batch, args.baseline, args.evaluation_root,
            args.evaluation_workers, args.evaluation_gpus), indent=2))
    else:
        job = args.job.resolve()
        sys.path.insert(0, str(job / "inputs/checkout/src"))
        if args.command == "status":
            print(json.dumps(read(job / "status.json"), indent=2))
        elif args.command == "resume":
            if alive(read(job / "process.json")["pid"]):
                raise RuntimeError("Supervisor is already running")
            print(json.dumps(launch(job), indent=2))
        else:
            try:
                worker(job)
            except Exception as exc:
                write(job / "status.json", {"status": "needs_attention", "error": str(exc), "updated": time.time()})
                raise


if __name__ == "__main__":
    main()
