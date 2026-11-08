"""Read-only background monitor for one RMBench RGB-D preparation.

Writes heartbeat, progress, failure alerts and final validation into JOB/monitor/.
It never restarts simulations, retries identities, or modifies capture evidence.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def capture_workers(pool):
    result = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if b'cam.benchmarks.rmbench_rgbd_data' not in args or b'worker' not in args:
                continue
            path = Path(args[-2].decode())
            if not path.is_relative_to(pool):
                continue
            gpu = next(x.split(b'=', 1)[1].decode() for x in
                       (process / 'environ').read_bytes().split(b'\0')
                       if x.startswith(b'CUDA_VISIBLE_DEVICES='))
            result.append({'pid': int(process.name), 'gpu': gpu,
                           'task': path.parent.parent.name, 'episode': path.parent.name})
        except (OSError, StopIteration):
            continue
    return result


def audit_results(pool, cache):
    # Only read new results. Directory traversal is intentionally infrequent on NFS.
    cache = dict(cache)
    for path in pool.glob('jobs/*/ep*/result.json'):
        if str(path) not in cache:
            value = read(path)
            cache[str(path)] = {'task': path.parent.parent.name,
                               **{k: value.get(k) for k in ('episode', 'seed', 'status', 'success', 'frames', 'error')}}
    return cache


def final_validation(job, plan, folder):
    program = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from cam.benchmarks.rmbench_rgbd_data import validate_dataset
from cam.io import read, sha256
pool, initial = Path(sys.argv[2]), Path(sys.argv[3])
plan = read(sys.argv[4])
p = validate_dataset(pool, verify_hashes=False)
s = validate_dataset(initial, verify_hashes=False)
assert set(p['tasks']) == set(plan['expected_by_task'])
assert s['selection']['source_manifest_sha256'] == sha256(pool / 'manifest.json')
for task, expected in plan['expected_by_task'].items():
    item = p['tasks'][task]
    assert item['replay']['attempted'] == expected
    count = len(item['episodes'])
    assert item['replay']['accepted'] == count
    assert item['replay']['shortfall'] == expected - count
    assert len(s['tasks'].get(task, {}).get('episodes', [])) == min(25, count)
print('PASS: all tasks, identities, sensor schemas, successful-only membership, source manifest, and subset counts. Full file hashes were verified by the preparation pipeline.')
"""
    with (folder / 'final_validation.log').open('w') as log:
        result = subprocess.run(
            [sys.executable, '-c', program, plan.get('capture_snapshot', str(job / 'inputs/checkout/src')),
             plan['pool'], plan['initial'], str(job / 'plan.json')],
            stdout=log, stderr=subprocess.STDOUT, timeout=3600,
        )
    if result.returncode:
        raise RuntimeError('Final dataset check failed; see final_validation.log')


def monitor(job, interval):
    folder = job / 'monitor'
    folder.mkdir(exist_ok=True)
    plan = read(job / 'plan.json')
    pool = Path(plan['pool'])
    with (folder / 'monitor.lock').open('a') as handle, ThreadPoolExecutor(max_workers=1) as executor:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seen_alerts, cache = set(), {}
        audit, next_audit, log_position = None, 0, 0
        completed, successes = 0, 0
        last_progress = time.time()
        write(folder / 'process.json', {'pid': os.getpid(), 'started': time.time(), 'interval_seconds': interval})

        def alert(key, severity, detail):
            if key in seen_alerts:
                return
            seen_alerts.add(key)
            with (folder / 'alerts.jsonl').open('a') as output:
                output.write(json.dumps({'time': time.time(), 'key': key, 'severity': severity, 'detail': detail}) + '\n')

        while True:
            try:
                now = time.time()
                with (job / 'supervisor.log').open() as log:
                    log.seek(log_position)
                    for line in log:
                        try:
                            value = json.loads(line)
                        except ValueError:
                            continue
                        if 'completed' in value:
                            if value['completed'] != completed:
                                last_progress = now
                            completed, successes = value['completed'], value['successes']
                    log_position = log.tell()
                parent = read(job / 'status.json')
                workers = capture_workers(pool)
                if audit is not None and audit.done():
                    try:
                        cache = audit.result()
                    except Exception:
                        alert('result_audit_' + str(int(now // 300)), 'error', traceback.format_exc())
                    audit = None
                    next_audit = now + 300
                    for path, result in cache.items():
                        if result['status'] == 'infrastructure_error':
                            alert(path, 'infrastructure_error', result)
                    write(folder / 'audited_results.json', cache)
                if audit is None and now >= next_audit:
                    audit = executor.submit(audit_results, pool, cache)
                by_task = defaultdict(Counter)
                for result in cache.values():
                    by_task[result['task']][result['status']] += 1
                state = {'updated': now, 'utc': datetime.now(timezone.utc).isoformat(),
                         'status': parent['status'], 'completed': completed, 'successes': successes,
                         'failed': completed - successes, 'expected': sum(plan['expected_by_task'].values()),
                         'workers': workers, 'workers_by_gpu': dict(Counter(w['gpu'] for w in workers)),
                         'seconds_since_completion': round(now - last_progress),
                         'audited_results': len(cache), 'audited_by_task': dict(by_task)}
                try:
                    telemetry = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=index,memory.used,memory.free', '--format=csv,noheader,nounits'],
                        text=True, timeout=10)
                    state['gpu_memory_mib'] = [dict(zip(('gpu', 'used', 'free'), map(int, row.split(','))))
                                               for row in telemetry.splitlines() if row.strip()]
                    for device in state['gpu_memory_mib']:
                        if device['free'] < 4096:
                            alert('low_gpu_memory_' + str(device['gpu']), 'warning', device)
                except Exception:
                    state['gpu_telemetry_error'] = traceback.format_exc()
                supervisor = read(job / 'process.json')['pid']
                if not Path(f'/proc/{supervisor}').exists() and parent['status'] != 'complete':
                    state['status'] = 'collector_stopped'
                    alert('collector_stopped', 'error', parent)
                elif parent['status'] == 'needs_attention':
                    alert('collector_needs_attention', 'error', parent)
                elif completed < state['expected'] and now - last_progress > 1800:
                    alert(f'no_completions_after_{completed}', 'warning', state)
                if workers:
                    state['status'] = 'collecting'
                elif completed == state['expected'] and parent['status'] == 'preparing':
                    state['status'] = 'assembling_or_validating_dataset'
                write(folder / 'status.json', state)
                with (folder / 'history.jsonl').open('a') as history:
                    history.write(json.dumps(state) + '\n')
                if parent['status'] == 'complete':
                    state['status'] = 'checking_final_dataset'
                    write(folder / 'status.json', state)
                    try:
                        final_validation(job, plan, folder)
                    except Exception:
                        state.update(status='final_validation_failed', error=traceback.format_exc(), updated=time.time())
                        alert('final_validation_failed', 'error', state)
                        write(folder / 'status.json', state)
                        break
                    final = read(pool / 'manifest.json')
                    state.update(status='complete_verified', counts={t: len(v['episodes']) for t, v in final['tasks'].items()},
                                 updated=time.time())
                    write(folder / 'status.json', state)
                    break
            except Exception:
                alert('monitor_exception_' + str(int(time.time() // 300)), 'error', traceback.format_exc())
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if not 10 <= args.interval <= 60:
        parser.error('Choose a monitoring interval from 10 to 60 seconds')
    monitor(args.job.resolve(), args.interval)
