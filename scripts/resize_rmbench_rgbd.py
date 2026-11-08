"""Continue an existing RGB-D pool with a new, timestamped worker allocation.

Only the host scheduler changes. Completed attempts and active native workers
are adopted; acquisition and assembly use the original immutable code snapshot.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else int(stat[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def native_workers(pool):
    workers = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().split(b'\0')
            if b'cam.benchmarks.rmbench_rgbd_data' not in args or b'worker' not in args:
                continue
            job = Path(args[-2].decode())
            if not job.is_relative_to(pool):
                continue
            gpu = next(int(x.split(b'=', 1)[1]) for x in (path / 'environ').read_bytes().split(b'\0')
                       if x.startswith(b'CUDA_VISIBLE_DEVICES='))
            pid = int(path.name)
            ticks = identity(pid)
            if ticks is not None:
                workers.append({'pid': pid, 'start_ticks': ticks, 'gpu': gpu, 'job': str(job)})
        except (OSError, StopIteration, ValueError):
            continue
    return workers


def snapshot(plan):
    source = Path(plan['capture_snapshot'])
    sys.path.insert(0, str(source))
    from cam.io import tree_hash
    if tree_hash(source / 'cam') != plan['harness_hash']:
        raise ValueError('Original acquisition snapshot changed')
    os.environ['CAM_RMBENCH_PYTHON'] = plan['simulator_python']


def start(parent, per_gpu):
    old = read(parent / 'plan.json')
    snapshot_path = old.get('capture_snapshot', str(parent / 'inputs/checkout/src'))
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    job = (ROOT / 'outputs/rmbench_rgbd_preparation' / tag).resolve()
    plan = {**old, 'tag': tag, 'created': time.time(), 'kind': 'rgbd-scheduler-continuation/1',
            'parent_job': str(parent), 'supersedes': str(parent),
            'parent_plan_sha256': hashlib.sha256((parent / 'plan.json').read_bytes()).hexdigest(),
            'capture_snapshot': snapshot_path, 'workers': per_gpu * len(old['gpus']),
            'workers_by_gpu': {str(g): per_gpu for g in old['gpus']},
            'allow_gpu_sharing': True, 'same_dataset_continuation': True}
    snapshot(plan)
    from cam.benchmarks.rmbench_rgbd_data import gpu_allocation
    from cam.benchmarks.rmbench import RMBench
    gpu_allocation(plan['workers'], plan['gpus'])
    adapter = RMBench(plan['source'])
    from cam.benchmarks.base import source_hash
    if source_hash(adapter) != plan['source_hash']:
        raise ValueError('Original benchmark snapshot changed')
    pool = Path(plan['pool'])
    manifest = read(pool / 'manifest.partial.json')
    expected = []
    for task, count in plan['expected_by_task'].items():
        records = adapter.records(task, 'train')
        assert len(records) == count
        audit = {r['episode']: r for r in read(pool / 'audit' / f'{task}.release.json')['identities']}
        for record in records:
            path = pool / 'jobs' / task / f"ep{record['episode']:05d}" / 'job.json'
            spec = read(path)
            if (spec['source'] != plan['source'] or spec['task'] != task or
                    any(spec[k] != record[k] for k in ('episode', 'seed')) or
                    spec['raw_crc32'] != audit[record['episode']]['crc32'] or
                    Path(spec['folder']) != path.parent):
                raise ValueError(f'Staged acquisition identity changed: {path}')
            expected.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if manifest['benchmark_hash'] != plan['source_hash']:
        raise ValueError('Dataset benchmark changed')
    job.mkdir(parents=True)
    write(job / 'job_inventory.json', expected)
    shutil.copy2(Path(__file__), job / 'driver.py')
    shutil.copy2(ROOT / 'scripts/monitor_rmbench_rgbd.py', job / 'monitor_driver.py')
    shutil.copy2(pool / 'manifest.partial.json', job / 'dataset_manifest_at_transition.json')
    parent_process = read(parent / 'process.json')['pid']
    command = Path(f'/proc/{parent_process}/cmdline').read_bytes()
    if str(parent / 'driver.py').encode() not in command:
        raise ValueError('Parent scheduler identity changed')
    # Freeze only dispatch. Native captures have separate sessions and continue.
    os.kill(parent_process, signal.SIGSTOP)
    try:
        for _ in range(100):
            if '\nState:\tT' in Path(f'/proc/{parent_process}/status').read_text():
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Could not suspend parent dispatch')
        plan['adopted_workers'] = native_workers(pool)
        plan['parent_process'] = {'pid': parent_process, 'start_ticks': identity(parent_process)}
        plan['manifest_before_sha256'] = hashlib.sha256((pool / 'manifest.partial.json').read_bytes()).hexdigest()
        write(job / 'plan.json', plan)
        (job / 'README.md').write_text(
            f"# RMBench RGB-D scheduler continuation\n\nCreated {tag}; parent `{parent}`. "
            f"User requested {per_gpu} capture jobs per GPU on {plan['gpus']}, {plan['workers']} total. "
            "Existing active captures finish normally; completed successes and failures are never replayed. "
            "The acquisition and assembly code is the unchanged original snapshot, checked against its hash. "
            "The old scheduler, logs, plan and snapshot remain preserved.\n\n"
            f"Pool (unchanged): `{pool}`\n\nSubset: `{plan['initial']}`\n\n"
            "Each native launch requires 7,000 MiB plus 2,048 MiB headroom, with additional reservations "
            "for recent launches. One host queue owns each GPU; each capture also owns two planner subprocesses. "
            "All original 500 training identities remain assigned; only successful captures enter training H5s. "
            "Original file hashes are checked during final assembly; final pool/subset validation is required. "
            "Progress: `status.json`, `supervisor.log`, and `monitor/status.json`.\n")
        write(job / 'status.json', {'status': 'adopting', 'updated': time.time(), 'pool': str(pool)})
        shutil.copy2(parent / 'status.json', job / 'parent_status_before_transition.json')
        write(parent / 'status.json', {'status': 'superseded_by_continuation', 'successor': str(job),
                                     'updated': time.time(), 'reason': 'User requested five capture workers per GPU.'})
        os.kill(parent_process, signal.SIGTERM)
    finally:
        if identity(parent_process) is not None:
            os.kill(parent_process, signal.SIGCONT)
    monitor_process = parent / 'monitor/process.json'
    if monitor_process.exists():
        pid = read(monitor_process)['pid']
        if identity(pid) is not None and str(parent).encode() in Path(f'/proc/{pid}/cmdline').read_bytes():
            os.kill(pid, signal.SIGTERM)
    with (job / 'supervisor.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, str(job / 'driver.py'), '_worker', str(job)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=job)
    write(job / 'process.json', {'pid': process.pid, 'start_ticks': identity(process.pid), 'started': time.time()})
    monitor_folder = job / 'monitor'
    monitor_folder.mkdir()
    with (monitor_folder / 'supervisor.log').open('ab') as log:
        subprocess.Popen([sys.executable, str(job / 'monitor_driver.py'), str(job), '--interval', '60'],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(job, flush=True)
    print(f'Supervisor PID {process.pid}; adopted {len(plan["adopted_workers"])} active captures', flush=True)


def worker(job):
    plan = read(job / 'plan.json')
    snapshot(plan)
    from cam.benchmarks import rmbench_rgbd_data as capture
    from cam.data.subset import subset
    pool = Path(plan['pool'])
    inventory = read(job / 'job_inventory.json')
    paths = [Path(r['path']) for r in inventory]
    adopted = {Path(w['job']): w for w in plan['adopted_workers']}
    # Catch a child that was between fork and exec when dispatch was suspended.
    for item in native_workers(pool):
        adopted[Path(item['job'])] = item
    write(job / 'adoption_observed.json', list(adopted.values()))
    results, pending, active = {}, [], {}

    def inspect(record):
        path = Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Staged job changed during transition')
        result = path.parent / 'result.json'
        if result.exists() and path not in adopted:
            return path, read(result)
        return path, None

    # Independent metadata reads: avoid serial NFS latency during handover.
    with ThreadPoolExecutor(max_workers=16) as readers:
        for path, result in readers.map(inspect, inventory):
            if result is not None:
                results[path] = result
            elif path not in adopted:
                pending.append(path)
    allocation = {int(g): n for g, n in plan['workers_by_gpu'].items()}
    recent = {g: [] for g in allocation}
    scheduling_path = pool / 'collection_history.json'
    history = read(scheduling_path) if scheduling_path.exists() else []
    if not any(x['job'] == str(job) for x in history):
        history.append({'job': str(job), 'workers_by_gpu': plan['workers_by_gpu'], 'started': time.time(),
                        'capture_snapshot': plan['capture_snapshot'], 'harness_hash': plan['harness_hash']})
        write(scheduling_path, history)

    def progress(status):
        state = {'status': status, 'updated': time.time(), 'completed': len(results),
                 'successes': sum(bool(r['success']) for r in results.values()), 'expected': len(paths),
                 'workers_by_gpu': plan['workers_by_gpu'], 'active': len(active) + len(adopted),
                 'pool': str(pool)}
        if status == 'complete':
            state['counts'] = {t: len(v['episodes']) for t, v in manifest['tasks'].items()}
            state['total'] = sum(state['counts'].values())
        write(job / 'status.json', state)
        write(pool / 'status.json', state)
        print(json.dumps(state), flush=True)

    with ExitStack() as stack:
        for gpu in sorted(allocation):
            handle = stack.enter_context(Path(f'/tmp/cam-rmbench-rgbd-locks/gpu_{gpu}.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX)
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=plan['workers']))
        while len(results) < len(paths):
            for path, item in list(adopted.items()):
                if identity(item['pid']) == item['start_ticks']:
                    continue
                result_path = path.parent / 'result.json'
                if not result_path.exists():
                    spec = read(path)
                    write(result_path, {'episode': spec['episode'], 'seed': spec['seed'], 'success': False,
                          'status': 'infrastructure_error', 'frames': 0, 'protocol': capture.PROTOCOL,
                          'error': 'Adopted native process exited without a result; evidence preserved.'})
                results[path] = read(result_path)
                del adopted[path]
            for future, (path, gpu) in list(active.items()):
                if future.done():
                    results[path] = future.result()
                    del active[future]
            for gpu, limit in allocation.items():
                occupied = sum(w['gpu'] == gpu for w in adopted.values()) + sum(g == gpu for _, g in active.values())
                now = time.time()
                recent[gpu] = [stamp for stamp in recent[gpu] if now - stamp < 60]
                if pending and occupied < limit:
                    free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                                        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
                    required = 7000 * (1 + len(recent[gpu])) + 2048
                    if free >= required:
                        path = pending.pop(0)
                        active[executor.submit(capture.run_episode, path, Path(plan['simulator_python']), gpu)] = (path, gpu)
                        recent[gpu].append(now)
            progress('collecting')
            if len(results) < len(paths):
                time.sleep(10)
    progress('assembling')
    manifest = read(pool / 'manifest.partial.json')
    manifest['collection_history'] = history
    from cam.benchmarks.rmbench import RMBench
    adapter = RMBench(plan['source'])
    for task in plan['expected_by_task']:
        manifest['tasks'][task] = capture.assemble(pool, task, adapter.records(task, 'train'))
        write(pool / 'manifest.partial.json', manifest)
    write(pool / 'manifest.json', manifest)
    capture.validate_dataset(pool)
    (pool / 'manifest.partial.json').unlink()
    available = [t for t, item in manifest['tasks'].items() if item['episodes']]
    subset(pool, plan['initial'], count=25, tasks=available, allow_fewer=True)
    progress('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', '_worker'])
    parser.add_argument('job', type=Path)
    parser.add_argument('--workers-per-gpu', type=int, choices=range(1, 7), default=5)
    args = parser.parse_args()
    if args.command == 'start':
        start(args.job.resolve(), args.workers_per_gpu)
    else:
        try:
            worker(args.job.resolve())
        except Exception:
            write(args.job / 'status.json', {'status': 'needs_attention', 'updated': time.time(),
                                           'error': traceback.format_exc()})
            raise
