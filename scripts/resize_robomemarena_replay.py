"""Continue RGB-D collection with GPU admission independent of file assembly.

Preserve the pool, active native jobs, original source snapshots and all receipts.
Only the host orchestration changes, under a fresh scheduler timetag.
"""
import argparse
import ast
from datetime import datetime, timezone
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
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else {'pid': pid, 'start_ticks': stat[19]}
    except FileNotFoundError:
        return None


def command(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return b''


def acquisition_fingerprints(package):
    """The continuation changes host scheduling, not native acquisition/scoring."""
    wanted = {'benchmarks/robomemarena_capture.py': ['capture_task'],
              'benchmarks/robomemarena_data.py': ['load_recording', 'fields', 'native_environment', 'replay', 'score_identities'],
              'data/demonstrations.py': ['create', 'Episode', '_check']}
    result = {}
    for relative, names in wanted.items():
        nodes = {n.name: n for n in ast.parse((package/relative).read_text()).body if hasattr(n, 'name')}
        for name in names:
            result[f'{relative}:{name}'] = hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
    for relative in ['benchmarks/robomemarena.py']:
        result[relative] = hashlib.sha256((package/relative).read_bytes()).hexdigest()
    return result


def job_key(job):
    return json.dumps({k: v for k, v in job.items() if k != 'result'}, sort_keys=True)


def verify_constructor_retry_change(previous, current):
    """Allow the reviewed constructor-only repair; reject any other adapter edit."""
    old = ast.parse(previous.read_text())
    new = ast.parse(current.read_text())
    helper = [n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'construct_environment']
    if len(helper) != 1:
        raise ValueError('Expected exactly one constructor retry helper')
    new.body = [n for n in new.body if n not in helper and not (
        isinstance(n, ast.Import) and len(n.names) == 1 and n.names[0].name in ('gc', 'logging'))]
    def method(module):
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'RoboMemArena')
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'make_environment')
    old_method, new_method = method(old), method(new)
    old_assignment = next(n for n in ast.walk(old_method) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'env' for t in n.targets))
    expected = ast.parse('env = construct_environment(env_class, dict(bddl_file_name=str(bddl), camera_depths=bool(depth), **ENV_KWARGS), RandomizationError)').body[0]
    class Restore(ast.NodeTransformer):
        changes = 0
        imports = 0
        def visit_ImportFrom(self, node):
            if node.module == 'robosuite.utils.errors' and ast.dump(node) == ast.dump(ast.parse('from robosuite.utils.errors import RandomizationError').body[0]):
                self.imports += 1
                return None
            return node
        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(expected):
                self.changes += 1
                return old_assignment
            return node
    restore = Restore()
    restore.visit(new_method)
    if restore.changes != 1 or restore.imports != 1 or ast.dump(old) != ast.dump(new):
        raise ValueError('Adapter change exceeds the constructor-only initialization repair')
    return hashlib.sha256(ast.dump(helper[0]).encode()).hexdigest()


def inventory_jobs(parent, pool):
    from cam.io import sha256
    items = []
    for path in sorted((pool/'work').glob('*.job.json')):
        spec = read(path)
        if spec['mode'] not in ('capture', 'validate'):
            continue
        dispatch = parent/'replay_archive'/path.name.removesuffix('.job.json')/'dispatch.json'
        if not dispatch.exists():
            # This transition is performed while the producer is frozen. A
            # queued wrapper without a dispatch record must be accounted for.
            queued = []
            for proc in Path('/proc').iterdir():
                if proc.name.isdigit() and str(path).encode() in command(int(proc.name)):
                    queued.append(identity(int(proc.name)))
            if any(queued):
                raise RuntimeError(f'An unadmitted wrapper exists; retry transition after admission: {path}')
            continue
        d = read(dispatch)
        result = Path(spec['result'])
        workers = [identity(d['native_pid']), identity(d['dispatcher_pid'])] if 'finished' not in d else []
        if not result.exists() and not any(workers) and not d.get('returncode'):
            continue
        items.append({'job': str(path), 'job_sha256': sha256(path), 'spec': spec,
            'result': str(result), 'result_sha256': sha256(result) if result.exists() else None,
            'dispatch': str(dispatch), 'returncode': d.get('returncode'), 'workers': workers})
    return items


def adopt_worker(entries, fallback, audit):
    from cam.io import sha256
    indexed = {}
    for entry in entries:
        key = job_key(entry['spec'])
        if key in indexed:
            prior = indexed[key]
            if bool(prior.get('returncode')) == bool(entry.get('returncode')):
                raise ValueError('Ambiguous duplicate native jobs at scheduler transition')
            # A later identical infrastructure retry supersedes a failed job;
            # both remain in the immutable inventory and original archive.
            if entry.get('returncode'):
                continue
        indexed[key] = entry

    def run(job, python, folder):
        entry = indexed.get(job_key(job))
        if entry is None:
            # A changed assignment must never silently replay an active task.
            if any(e['spec']['mode'] == job['mode'] and e['spec']['task'] == job['task'] for e in entries):
                raise ValueError('Continued task assignment differs from its existing native job')
            return fallback(job, python, folder)
        if sha256(entry['job']) != entry['job_sha256']:
            raise ValueError('Adopted native job changed')
        if entry.get('retry_initialization'):
            evidence = entry['retry_initialization']
            if sha256(evidence['log']) != evidence['log_sha256'] or not read(entry['dispatch']).get('returncode'):
                raise ValueError('Initialization failure evidence changed')
            return fallback(job, python, folder)
        while True:
            dispatch = read(entry['dispatch'])
            if 'finished' in dispatch:
                if dispatch['returncode']:
                    raise RuntimeError(f'Adopted native job failed: {entry["job"]}')
                break
            if not any(w and identity(w['pid']) == w for w in entry['workers']):
                raise RuntimeError('Adopted native job disappeared without a completion record')
            time.sleep(2)
        digest = sha256(entry['result'])
        if entry['result_sha256'] and entry['result_sha256'] != digest:
            raise ValueError('Adopted result changed')
        write(audit/f"{job['mode']}_{job['task']}.json", {'job': entry['job'],
            'job_sha256': entry['job_sha256'], 'result': entry['result'], 'result_sha256': digest,
            'native_replay_repeated': False})
        return read(entry['result'])
    return run


def start(parent, local_assembly_root=None, retry_initialization_failures=False, finalize_assembled=False):
    sys.path.insert(0, str(ROOT/'src'))
    from cam.io import copy_tree, sha256, tree_hash
    old = read(parent/'plan.json')
    if finalize_assembled and (local_assembly_root or retry_initialization_failures or not old.get('assembly_output')):
        raise ValueError('Final validation requires the existing local assembly without native retries')
    if not old.get('single_pass') or old['replay_workers_per_gpu'] != 4:
        raise ValueError('Expected the current four-per-GPU single-pass preparation')
    package = parent/'inputs/checkout/src/cam'
    if tree_hash(package) != old['harness_hash']:
        raise ValueError('Original source snapshot changed')
    original_methods = acquisition_fingerprints(package)
    current_methods = acquisition_fingerprints(ROOT/'src/cam')
    repair = None
    if retry_initialization_failures:
        repair = verify_constructor_retry_change(package/'benchmarks/robomemarena.py', ROOT/'src/cam/interface/robomemarena/adapter.py')
        if {k:v for k,v in original_methods.items() if k != 'benchmarks/robomemarena.py'} != {k:v for k,v in current_methods.items() if k != 'benchmarks/robomemarena.py'}:
            raise ValueError('Acquisition/scoring changed beyond environment construction')
    elif current_methods != original_methods:
        raise ValueError('Acquisition code changed; this continuation permits only host scheduling changes')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    job = (ROOT/'outputs/robomemarena_replay_schedulers'/tag).resolve()
    job.mkdir(parents=True)
    copy_tree(ROOT/'src/cam', job/'inputs/checkout/src/cam')
    shutil.copy2(__file__, job/'driver.py')
    shutil.copy2(ROOT/'scripts/monitor_robomemarena_replay.py', job/'monitor_driver.py')
    pool = Path(old['pool'])
    hosts = [read(parent/'process.json')['pid']]
    records = sorted((parent/'logs').glob('*.process.json'))
    producers = [read(p)['pid'] for p in records if identity(read(p)['pid'])]
    if finalize_assembled:
        if producers or any(identity(pid) for pid in hosts) or read(parent/'status.json')['status'] != 'needs_attention':
            raise ValueError('Final validation recovery requires a stopped failed producer')
        hosts = []
        destination = Path(old['assembly_output'])
        if (destination/'manifest.json').exists() or not (destination/'manifest.partial.json').exists():
            raise ValueError('Expected an unpublished assembled dataset')
        for task in old['tasks']:
            for suffix in ('.json', '.ready.json', '.validation.json'):
                if not (destination/'audit'/f'{task}{suffix}').is_file():
                    raise ValueError(f'Missing completed assembly evidence: {task}{suffix}')
    elif len(producers) != 1:
        raise ValueError('Expected exactly one active extraction producer')
    hosts = sorted(set(hosts + producers))
    for pid in hosts:
        if str(parent).encode() not in command(pid):
            raise ValueError('Original producer identity changed')
    frozen = []
    try:
        for pid in hosts:
            os.kill(pid, signal.SIGSTOP)
            frozen.append(pid)
        archive_parent = Path(old.get('replay_archive_job', parent))
        entries = inventory_jobs(archive_parent, pool)
        if repair:
            retried = []
            for entry in entries:
                if entry['spec']['task'] not in ('T25_ButterCreamPlate', 'T26_ChocolateCreamPlate') or entry['spec']['mode'] != 'capture':
                    continue
                dispatch = read(entry['dispatch'])
                log = Path(entry['job'].replace('.job.json', '.log'))
                content = log.read_text()
                if not dispatch.get('returncode') or entry['result_sha256'] or any(entry['workers']) or 'robosuite.utils.errors.RandomizationError: Cannot place all objects' not in content or 'env = env_class(' not in content:
                    raise ValueError('Retry requires a recorded pre-replay constructor placement failure')
                entry['retry_initialization'] = {'log': str(log), 'log_sha256': sha256(log)}
                retried.append(entry['spec']['task'])
            if set(retried) != {'T25_ButterCreamPlate', 'T26_ChocolateCreamPlate'}:
                raise ValueError('Expected both T25 and T26 initialization failures')
        shutil.copy2(pool/'manifest.partial.json', job/'manifest_at_transition.json')
        write(job/'job_inventory.json', entries)
        plan = {**old, 'tag': tag, 'created': time.time(), 'kind': 'independent-native-admission/1',
            'parent_job': str(parent), 'parent_plan_sha256': sha256(parent/'plan.json'),
            'source_harness_hash': old['harness_hash'], 'harness_hash': tree_hash(job/'inputs/checkout/src/cam'),
            'native_method_hashes': current_methods, 'host_task_workers': len(old['tasks']),
            'replay_archive_job': str(archive_parent), 'retired_hosts': hosts,
            'job_inventory_sha256': sha256(job/'job_inventory.json'),
            'manifest_at_transition_sha256': sha256(job/'manifest_at_transition.json'),
            'same_dataset_continuation': True,
            'finalize_assembled_only': finalize_assembled}
        if finalize_assembled:
            candidate = Path(old['assembly_output'])/'manifest.partial.json'
            shutil.copy2(candidate, job/'assembled_manifest.json')
            plan['assembled_manifest_sha256'] = sha256(job/'assembled_manifest.json')
        if repair:
            plan['initialization_repair'] = {'tasks': retried, 'helper_ast_sha256': repair,
                'previous_native_method_hashes': original_methods,
                'changed': 'bounded retry of unseeded environment construction on RandomizationError only',
                'episode_reset_actions_geometry_scoring_unchanged': True}
        if local_assembly_root:
            local_root = Path(local_assembly_root).resolve()
            if not local_root.is_relative_to(Path('/data/daiyp/CaM')):
                raise ValueError('Local assembly must be under /data/daiyp/CaM')
            plan.update(assembly_output=str(local_root/tag/'pool'), demo_count=50,
                        direct_pool=True, assemble_after_all_validation=True)
        write(job/'plan.json', plan)
        write(job/'parent_status.json', read(parent/'status.json'))
        (job/'README.md').write_text(
            '# RoboMemArena replay scheduling continuation\n\n'
            'Native GPU admission is independent of task file assembly: four jobs per GPU, '
            'eight total, with the unchanged 16 GiB free-memory admission floor. '
            'Active native jobs and completed captures/validations are adopted without replay. '
            'Only host schedulers are retired. The original code and native acquisition/scoring '
            'remain pinned; method fingerprints are checked before continuation.\n\n'
            f'Pool remains `{pool}`. Source job: `{parent}`. '
            'All captured identities, failed outcomes, original jobs, and partial assembly files '
            'are preserved. Incomplete file assembly may resume from saved captures without '
            'repeating simulation. Five sampled validations/task and full data-integrity checks '
            'remain required. Learning uses the configured demo-only assignment with zero training rollouts.\n')
        if plan.get('assembly_output'):
            with (job/'README.md').open('a') as handle:
                handle.write(f"\nAfter all 26 capture and sampled-validation tasks finish, assemble at `{plan['assembly_output']}` on the local SSD. Read NFS sources sequentially into temporary local staging, verify hashes, then assemble locally. Retain at most 50 successful demos/task and use the whole capped dataset directly for learning, reporting actual shortfalls; no 25-demo subset. Original NFS captures, audits and interrupted assemblies are preserved.\n")
        if repair:
            with (job/'README.md').open('a') as handle:
                handle.write('\nUser-requested T25/T26 repair: retry only their recorded failed pre-replay environment construction, under this new source snapshot. The adapter now retries constructor RandomizationError up to 30 times, as the released retry runner does. Assigned seed reset, scene/geometry, actions and scoring are unchanged. The original failed logs remain intact; all active native jobs and completed capture/validation results are adopted.\n')
        if finalize_assembled:
            with (job/'README.md').open('a') as handle:
                handle.write('\nFinal validation recovery only: all task H5s and sampled replay receipts already exist. Use the explicit pinned benchmark source for schema, identity and full-file hash checks. No native simulator, capture, assembly or GPU admission runs in this continuation. Preserve the previous failed manifest and logs; publish only after the unchanged assembled pool passes every check.\n')
        for pid in hosts:
            os.kill(pid, signal.SIGTERM)
        write(parent/'status.json', {'status': 'superseded_by_scheduler', 'replacement': str(job),
            'updated': time.time(), 'active_native_jobs_preserved': True})
    finally:
        for pid in frozen:
            if identity(pid):
                os.kill(pid, signal.SIGCONT)
    monitor = parent/'monitor/process.json'
    if monitor.exists():
        pid = read(monitor)['pid']
        if identity(pid) and str(parent).encode() in command(pid):
            os.kill(pid, signal.SIGTERM)
    argv = [sys.executable, str(job/'driver.py'), 'extract', str(job), '--output', plan.get('assembly_output', str(pool))]
    log = job/'logs/resume_pool'
    write(log.with_suffix('.command.json'), {'argv': argv, 'started': time.time()})
    with log.with_suffix('.log').open('ab') as out:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=job)
    process = {'pid': proc.pid, 'started': time.time(), 'identity': identity(proc.pid)}
    write(log.with_suffix('.process.json'), process)
    write(job/'process.json', process)
    write(job/'status.json', {'status': 'running', 'phase': 'adopting_capture_jobs', 'updated': time.time()})
    with (job/'monitor.log').open('ab') as log_file:
        monitor_proc = subprocess.Popen([sys.executable, str(job/'monitor_driver.py'), str(job),
            str(ROOT/'sites/development_v2_30/previews/robomemarena_replay_status')],
            stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    write(job/'monitor/process.json', {'pid': monitor_proc.pid, 'started': time.time()})
    print(json.dumps({'job': str(job), 'pid': proc.pid, 'adopted_jobs': len(entries)}, indent=2))


def worker(job):
    sys.path.insert(0, str(job/'inputs/checkout/src'))
    from cam.io import sha256, tree_hash
    from cam.interface.robomemarena import data as rmd
    from cam.interface.robomemarena.capture import extract_captures
    from cam.interface.robomemarena.replay_cache import open_cache
    plan = read(job/'plan.json')
    parent = Path(plan['parent_job'])
    pool, root = Path(plan['pool']), Path(plan['raw'])/'hf'
    assert sha256(parent/'plan.json') == plan['parent_plan_sha256']
    assert tree_hash(job/'inputs/checkout/src/cam') == plan['harness_hash']
    assert tree_hash(parent/'inputs/checkout/src/cam') == plan['source_harness_hash']
    assert acquisition_fingerprints(job/'inputs/checkout/src/cam') == plan['native_method_hashes']
    assert sha256(job/'job_inventory.json') == plan['job_inventory_sha256']
    assert sha256(job/'manifest_at_transition.json') == plan['manifest_at_transition_sha256']
    adapter = rmd.get_adapter('robomemarena', plan['benchmark_source'])
    assert rmd.source_hash(adapter) == plan['benchmark_hash']
    for link, digest in plan['asset_hashes'].items():
        assert tree_hash(Path(plan['benchmark_source'])/link) == digest
    if plan.get('finalize_assembled_only'):
        from cam.interface.robomemarena.local import finalize_assembled
        destination = Path(plan['assembly_output'])
        assert sha256(job/'assembled_manifest.json') == plan['assembled_manifest_sha256']
        assert sha256(destination/'manifest.partial.json') == plan['assembled_manifest_sha256']
        with (job/'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(job/'status.json', {'status': 'running', 'phase': 'final_file_validation', 'updated': time.time()})
            manifest = finalize_assembled(destination, adapter, plan['tasks'], plan['pool_cap'], plan['validation_episodes'])
            write(job/'validation_complete.json', {'pool': str(destination), 'manifest_sha256': sha256(destination/'manifest.json'),
                'demo_counts': {t: len(v['episodes']) for t, v in manifest['tasks'].items()},
                'native_replay_repeated': False, 'all_file_hashes_checked': True, 'finished': time.time()})
            write(job/'status.json', {'status': 'complete', 'phase': 'demo_pool_complete', 'updated': time.time()})
        return
    manifest = read(job/'manifest_at_transition.json')
    assert manifest['benchmark_hash'] == plan['benchmark_hash'] and manifest['depth']
    assert manifest['selection']['single_pass'] and manifest['selection']['max_successful'] == plan['pool_cap']
    assert manifest['selection']['validation_episodes'] == plan['validation_episodes']
    reuse = Path(plan['reused_dataset'])
    assert sha256(reuse/'manifest.json') == manifest['reuse']['manifest_sha256']
    reusable = read(reuse/'manifest.json')
    for key in ('benchmark', 'benchmark_hash', 'depth', 'fields', 'raw_data', 'release'):
        assert reusable[key] == manifest[key]
    # Existing sensor files were validated by the original producer. Reused file
    # hashes are checked again by assembly; every finalized H5 is checked before
    # this producer exits and before the experiment supervisor can use the pool.
    def unchanged(stats):
        for entry in stats.values():
            now = (root/entry['path']).stat()
            if (now.st_size, now.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
                raise ValueError('Raw file changed during scheduler continuation')

    scored, stats = {}, {}
    for task in manifest['requested_tasks']:
        assert rmd.fields(adapter, task, True) == manifest['fields']
        records = adapter.records(task, 'train')
        assert not {r['seed'] for r in records} & {r['seed'] for r in adapter.records(task, 'test')}
        final = pool/'audit'/f'{task}.json'
        if final.exists() and not plan.get('assembly_output'):
            value = read(final)
            unchanged(value['stats'])
            manifest['tasks'][task] = {k: v for k, v in value.items() if k != 'stats'}
            continue
        before = pool/'audit'/f'{task}.pre_capture.json'
        if before.exists():
            scored[task] = read(before)
            # Local assembly checks these stats when each task completes, in
            # parallel. Replayed actions are also verified before use. Avoid a
            # serial NFS stat scan delaying all native admission on continuation.
            if not plan.get('assembly_output'):
                unchanged(scored[task]['stats'])
            stats[task] = scored[task]['stats']
        else:
            stats[task] = rmd.verify_files(root, rmd.release()['tasks'][task], [r['seed'] for r in records])
            if task in reusable['tasks']:
                old_audit = read(reuse/'audit'/f'{task}.json')
                if not plan.get('assembly_output'):
                    unchanged(old_audit['stats'])
                scored[task] = {'stats': stats[task], 'results': reusable['tasks'][task]['replay']['identities']}
    cache_jobs, _ = open_cache(plan['replay_cache_job'], manifest)
    rmd.run_worker = adopt_worker(read(job/'job_inventory.json'), rmd.run_worker, job/'adopted_results')
    os.environ['CAM_ROBOMEMARENA_PYTHON'] = plan['replay_python']
    manifest.setdefault('collection_history', []).append({'scheduler': str(job), 'created': plan['created'],
        'harness_hash': plan['harness_hash'], 'native_method_hashes': plan['native_method_hashes'],
        'native_workers_per_gpu': 4, 'host_task_workers': plan['host_task_workers'],
        'adopted_job_inventory_sha256': plan['job_inventory_sha256']})
    with (job/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(job/'status.json', {'status': 'running', 'phase': 'demo_pool', 'updated': time.time()})
        options = dict(adapter=adapter, root=root, output=pool, manifest=manifest,
            scored=scored, stats=stats, cache_jobs=cache_jobs, reusable=reusable, reuse=reuse,
            python=plan['replay_python'], workers=plan['replay_workers'], unchanged=unchanged)
        if plan.get('assembly_output'):
            from cam.interface.robomemarena.local import capture_and_assemble_local
            capture_and_assemble_local(**options, destination=Path(plan['assembly_output']))
            pool = Path(plan['assembly_output'])
        else:
            extract_captures(**options)
        write(job/'status.json', {'status': 'running', 'phase': 'final_file_validation', 'updated': time.time()})
        try:
            rmd.validate_dataset(pool, benchmark_source=adapter.source)
        except Exception:
            (pool/'manifest.json').replace(pool/f'manifest.invalid_{time.time_ns()}.json')
            raise
        write(job/'status.json', {'status': 'complete', 'phase': 'demo_pool_complete', 'updated': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    start_parser = sub.add_parser('start')
    start_parser.add_argument('parent', type=Path)
    start_parser.add_argument('--local-assembly-root', type=Path)
    start_parser.add_argument('--retry-initialization-failures', action='store_true')
    start_parser.add_argument('--finalize-assembled', action='store_true', help='Recover final file validation for a stopped, fully assembled local pool; no simulator replay')
    run = sub.add_parser('extract')
    run.add_argument('job', type=Path)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'start':
        start(args.parent.resolve(), args.local_assembly_root, args.retry_initialization_failures, args.finalize_assembled)
    else:
        try:
            plan = read(args.job/'plan.json')
            assert args.output.resolve() == Path(plan.get('assembly_output', plan['pool']))
            worker(args.job.resolve())
        except Exception:
            write(args.job/'status.json', {'status': 'needs_attention', 'updated': time.time(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
