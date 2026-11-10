"""Capture and score released training actions once; keep diagnostics host-side."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import h5py
import numpy as np

from cam.data import demonstrations as demos
from cam.io import read, sha256, write
from cam.interface.robomemarena import data as rmd
from cam.interface.robomemarena.replay_cache import collect


def capture_task(job, environment=None):
    """Durably commit every attempted identity, including unsuccessful captures."""
    environment = environment or rmd.native_environment
    adapter = rmd.get_adapter('robomemarena', job['source'])
    declared = rmd.fields(adapter, job['task'], job['depth'])
    names = [n for n in declared['frame'] if n not in ('step', 'is_demo', 'action')]
    folder = Path(job['capture_directory'])
    folder.mkdir(parents=True, exist_ok=True)
    env, results, successes = None, [], 0
    try:
        for identity in job['episodes']:
            if successes >= job['success_limit']:
                break
            episode, seed = int(identity['episode']), int(identity['seed'])
            stem = folder / f'episode_{episode:03d}'
            receipt = stem.with_suffix('.json')
            if receipt.exists():
                row = read(receipt)
                if (row['episode'], row['seed'], row['protocol_sha256']) != (episode, seed, job['protocol_sha256']):
                    raise ValueError('Capture checkpoint identity/protocol changed')
                if sha256(folder / row['capture_file']) != row['capture_sha256']:
                    raise ValueError('Completed capture changed')
            else:
                partial = stem.with_suffix('.h5.partial')
                destination = stem.with_suffix('.h5')
                # Preserve files left by an interrupted process; they are unscored.
                for path in (partial, destination):
                    if path.exists():
                        archive = folder / 'interrupted' / str(time.time_ns()) / path.name
                        archive.parent.mkdir(parents=True)
                        shutil.move(path, archive)
                expected = identity.get('actions_sha256')
                recording = rmd.load_recording(Path(job['root']), rmd.release()['tasks'][job['task']], seed,
                                              images=False, verify=not bool(expected))
                if expected and recording['actions_sha256'] != expected:
                    raise ValueError('Released actions changed since the earlier success check')
                if env is None:
                    env = environment(job)
                buffer, target, static = [], None, {}
                with demos.create(partial, 'robomemarena', job['task'], declared, job['depth'], rmd.METHOD) as h5:
                    def flush():
                        arrays = {name: np.stack([f[name] for f, _ in buffer]) for name in names}
                        arrays['action'] = np.stack([command for _, command in buffer])
                        arrays['is_demo'] = np.zeros(len(buffer), dtype=bool)
                        target.append(arrays)
                        buffer.clear()

                    def observe(k, frame, command):
                        nonlocal target
                        if target is None:
                            static.update({name: frame[name] for name in declared['static']})
                            target = demos.Episode(h5, declared, episode, None,
                                {'goal': job['goal'], 'difficulty': identity.get('difficulty'),
                                 'action_alignment': rmd.ACTION_ALIGNMENT, 'step_semantics': rmd.STEP_SEMANTICS},
                                static=static)
                        for name in declared['static']:
                            if not np.array_equal(frame[name], static[name]):
                                raise ValueError('Calibration changed within an episode')
                        if int(frame['step']) != k:
                            raise ValueError('Recorded frame index differs from the live step')
                        buffer.append((frame, command))
                        if len(buffer) == rmd.CHUNK:
                            flush()

                    result, sent = rmd.replay(env, seed, recording['actions'], observe)
                    if buffer:
                        flush()
                    if target is None or target.filled != sent:
                        raise ValueError('Capture frame count differs from the executed actions')
                    target.close()
                status = 'actions_exhausted' if result is None or result.status == 'running' else result.status
                metrics = (result.metrics or {}) if result is not None else {}
                if status not in ('success', 'fail', 'timeout', 'actions_exhausted'):
                    raise RuntimeError(f'Unscorable native capture: {status}; partial evidence preserved')
                if status == 'success' and not metrics.get('tsr'):
                    raise ValueError('Native success disagrees with official TSR')
                partial.replace(destination)
                row = {'episode': episode, 'seed': seed, 'status': status, 'steps': sent,
                    'tsr': status == 'success' and bool(metrics.get('tsr')), 'csr': metrics.get('csr'),
                    'failure_reason': metrics.get('failure_reason'), 'private_error': result.private_error if result else None,
                    'released_frames': len(recording['actions']), 'actions_sha256': recording['actions_sha256'],
                    'capture_file': destination.name, 'capture_sha256': sha256(destination),
                    'protocol_sha256': job['protocol_sha256'], 'depth': bool(job['depth'])}
                write(receipt, row)
            results.append(row)
            successes += row['status'] == 'success' and row['tsr']
            write(folder / 'progress.json', {'task': job['task'], 'updated': time.time(),
                'completed': len(results), 'successful_captures': successes,
                'requested_new_successes': job['success_limit'], 'last_seed': seed})
            print(json.dumps({'task': job['task'], 'seed': seed, 'status': row['status'],
                              'successful_captures': successes}), flush=True)
    finally:
        if env is not None:
            env.close()
    return {'results': results, 'successful_captures': successes}


def merge_outcomes(target, rows, records):
    expected = {int(r['episode']): int(r['seed']) for r in records}
    seen = set()
    for row in rows:
        ep = int(row['episode'])
        if ep in seen or expected.get(ep) != int(row['seed']):
            raise ValueError('Cached replay contains a duplicate or unassigned training identity')
        seen.add(ep)
        if ep in target and target[ep] != row:
            raise ValueError('Conflicting historical replay outcomes')
        target[ep] = row


def historical_outcomes(manifest, task, records, initial, cache_jobs, stats):
    """Keep historical scores, including scores inherited by an earlier scheduler."""
    known, provenance = collect(cache_jobs.get(task, []), records)
    merge_outcomes(known, initial.get('results', []), records)
    if manifest.get('replay_cache'):
        source = Path(manifest['replay_cache']['source_job'])
        plan = read(source / 'plan.json')
        path = Path(plan['pool']) / 'audit' / f'{task}.replay.json'
        if path.exists():
            audit = read(path)
            if audit['stats'] != stats:
                raise ValueError('Cached replay raw-file audit changed')
            merge_outcomes(known, audit['results'], records)
            provenance.append({'audit': str(path), 'sha256': sha256(path)})
    return known, provenance


def validation_sample(episodes, count, task):
    ordered = sorted(episodes, key=lambda r: int(r['episode']))
    seed = int(hashlib.sha256(f'cam/robomemarena/validation/{task}'.encode()).hexdigest()[:8], 16)
    chosen = np.random.default_rng(seed).choice(len(ordered), min(count, len(ordered)), replace=False)
    return [ordered[int(i)] for i in sorted(chosen)]


def parallel_capture_tasks(tasks, workers, task_function, run_native):
    """Bound simulator jobs while allowing finished tasks to do file work."""
    native_slots = threading.BoundedSemaphore(workers)

    def native_job(job):
        with native_slots:
            return run_native(job)

    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
        futures = [executor.submit(task_function, task, native_job) for task in tasks]
        for future in as_completed(futures):
            yield future.result()


def extract_captures(*, adapter, root, output, manifest, scored, stats, cache_jobs,
                     reusable, reuse, python, workers, unchanged):
    """One worker per task; all needed sensor data and scores come from the same pass."""
    cap = manifest['selection']['max_successful']
    validation_count = manifest['selection']['validation_episodes']
    protocol = {k: manifest[k] for k in ('benchmark_hash', 'raw_data', 'depth', 'fields', 'release')}
    protocol['capture_version'] = 1
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    def task_capture(task, native_job):
        records = adapter.records(task, 'train')
        known, provenance = historical_outcomes(manifest, task, records, scored.get(task, {}), cache_jobs, stats[task])
        prior = output / 'audit' / f'{task}.pre_capture.json'
        history = {'results': [known[k] for k in sorted(known)], 'provenance': provenance, 'stats': stats[task]}
        if prior.exists() and read(prior) != history:
            raise ValueError('Historical replay evidence changed during capture resume')
        write(prior, history)
        original = (reusable or {}).get('tasks', {}).get(task)
        existing = {int(r['episode']): r for r in original['episodes']} if original else {}
        if len(existing) > cap:
            raise ValueError('Existing RGB-D captures exceed the cap')
        for ep, row in existing.items():
            if ep not in known or known[ep]['status'] != 'success' or known[ep]['steps'] != row['frames']:
                raise ValueError('Reusable RGB-D capture lacks matching successful replay evidence')
        # A score alone is not an RGB-D capture. Capture missing known successes,
        # then unchecked identities. Previously failed identities are not retried.
        candidates = [r for r in records if int(r['episode']) not in existing and known.get(int(r['episode']), {}).get('status') == 'success']
        candidates += [r for r in records if int(r['episode']) not in known and int(r['episode']) not in existing]
        candidates = [{**r, **({'actions_sha256': known[int(r['episode'])]['actions_sha256']} if int(r['episode']) in known else {})}
                      for r in candidates]
        goal = rmd.task_prompts(adapter.source)[f'task{rmd.task_id(task)}']
        folder = output / 'captures' / task
        job = {'source': str(adapter.source), 'root': str(root), 'task': task, 'depth': True,
            'mode': 'capture', 'output': str(output), 'capture_directory': str(folder),
            'episodes': candidates, 'success_limit': cap - len(existing), 'goal': goal,
            'protocol_sha256': protocol_hash}
        write(output / 'audit' / f'{task}.capture_plan.json', {**job,
            'historical_checked': len(known), 'historical_successes': sum(r['status'] == 'success' for r in known.values()),
            'reused_rgbd': len(existing), 'known_failures_skipped': sum(r['status'] != 'success' for r in known.values())})
        result = native_job(job) if candidates and job['success_limit'] else {'results': []}
        attempted = result['results']
        if len({r['episode'] for r in attempted}) != len(attempted):
            raise ValueError('Duplicate capture outcome')
        expected = {int(r['episode']): int(r['seed']) for r in candidates}
        for row in attempted:
            if expected.get(int(row['episode'])) != int(row['seed']) or row['protocol_sha256'] != protocol_hash:
                raise ValueError('Capture worker returned an unassigned identity or protocol')
        # Preserve previous results separately; the capture's actual outcome is
        # authoritative for eligibility, including when a previous success fails.
        outcomes = {**known, **{int(r['episode']): r for r in attempted}}
        captured = {int(r['episode']): r for r in attempted if r['status'] == 'success' and r['tsr']}
        selected = [r for r in records if int(r['episode']) in existing or int(r['episode']) in captured]
        if len(selected) > cap:
            raise ValueError('Capture worker exceeded the successful-demo cap')
        replay_audit = {'results': [outcomes[e] for e in sorted(outcomes)], 'stats': stats[task],
                        'single_pass': True, 'capture_attempts': len(attempted), 'reused_rgbd': len(existing)}
        write(output / 'audit' / f'{task}.replay.json', replay_audit)
        partial = output / 'h5' / f'record_dataset_{task}.h5.partial'
        partial.parent.mkdir(parents=True, exist_ok=True)
        episodes = [{**record, 'frames': outcomes[int(record['episode'])]['steps'],
                     'demo_frames': 0, 'goal': goal, 'video': None} for record in selected]
        destination = partial.with_suffix('')
        assembly_path = output / 'audit' / f'{task}.assembly.json'
        if assembly_path.exists():
            assembly = read(assembly_path)
            if assembly['episodes'] != episodes or sha256(destination) != assembly['sha256']:
                raise ValueError('Completed task assembly changed')
        else:
            for path in (partial, destination):
                if path.exists():
                    archive = output / 'interrupted' / str(time.time_ns()) / path.name
                    archive.parent.mkdir(parents=True)
                    shutil.move(path, archive)
            with demos.create(partial, 'robomemarena', task, manifest['fields'], True, rmd.METHOD) as h5:
                for record in selected:
                    ep = int(record['episode'])
                    source = reuse / original['file'] if ep in existing else folder / captured[ep]['capture_file']
                    if ep not in existing and sha256(source) != captured[ep]['capture_sha256']:
                        raise ValueError('Capture changed before assembly')
                    with h5py.File(source, 'r') as src:
                        src.copy(f'episode_{ep}', h5)
            partial.replace(destination)
            write(assembly_path, {'episodes': episodes, 'sha256': sha256(destination)})
        if original and sha256(reuse / original['file']) != original['sha256']:
            raise ValueError('Reusable RGB-D file changed during copying')
        unchanged(stats[task])
        sample = validation_sample(episodes, validation_count, task)
        validation_path = output / 'audit' / f'{task}.validation.json'
        assignment = [{'episode': int(r['episode']), 'seed': int(r['seed'])} for r in sample]
        if validation_path.exists():
            validation = read(validation_path)
            if validation['identities'] != assignment or validation['capture_h5_sha256'] != sha256(destination):
                raise ValueError('Validation assignment or capture changed')
        else:
            validation = {'requested': validation_count, 'identities': assignment,
                'capture_h5_sha256': sha256(destination), 'results': []}
            if sample:
                validation['results'] = native_job({'source': str(adapter.source), 'root': str(root),
                    'task': task, 'depth': True, 'mode': 'validate', 'identities': assignment})['results']
            write(validation_path, validation)
        checks = validation['results']
        if len(checks) != len(assignment) or {(r['episode'], r['seed']) for r in checks} != {(r['episode'], r['seed']) for r in assignment}:
            raise ValueError('Validation worker returned different identities')
        if any(r['status'] != 'success' or not r['tsr'] or r['steps'] != outcomes[int(r['episode'])]['steps']
               or r['actions_sha256'] != outcomes[int(r['episode'])]['actions_sha256'] for r in checks):
            raise RuntimeError(f'Sampled replay validation failed for {task}; captures and all scores preserved')
        item = rmd.release()['tasks'][task]
        successful = [r for r in records if outcomes.get(int(r['episode']), {}).get('status') == 'success']
        value = {'task': task, 'task_id': rmd.task_id(task), 'file': destination.relative_to(output).as_posix(),
            'sha256': validation['capture_h5_sha256'], 'episodes': episodes,
            'source': {'method': rmd.METHOD, 'repo': manifest['release']['repo'], 'revision': manifest['release']['revision'],
                'directory': item['directory'], 'snapshot': str(root), 'bddl': item['bddl'],
                'bddl_sha256': item['bddl_sha256'], 'benchmark_commit': rmd.release()['benchmark_source']['commit']},
            'replay': {'released': len(records), 'replayed': len(outcomes), 'successful': len(successful),
                'selected': len(selected), 'strata': rmd.strata(successful, selected, item),
                'identities': [outcomes[e] for e in sorted(outcomes)], 'capture_attempts': len(attempted)},
            'validation': {'requested': validation_count, 'checked': len(checks), 'successes': len(checks),
                           'episodes': [r['episode'] for r in sample], 'audit': validation_path.relative_to(output).as_posix()}}
        if existing:
            value['reuse'] = {'dataset': str(reuse), 'episodes': sorted(existing), 'source_h5_sha256': original['sha256']}
        write(output / 'audit' / f'{task}.json', {**value, 'stats': stats[task]})
        return task, value

    pending = [t for t in manifest['requested_tasks'] if t not in manifest['tasks']]
    run_native = lambda job: rmd.run_worker(job, python, output / 'work')
    for task, value in parallel_capture_tasks(pending, workers, task_capture, run_native):
        manifest['tasks'][task] = value
        write(output / 'manifest.partial.json', manifest)
    manifest['tasks'] = {t: manifest['tasks'][t] for t in manifest['requested_tasks']}
    write(output / 'manifest.json', manifest)
    try:
        rmd.validate_dataset(output, verify_hashes=False, benchmark_source=adapter.source)
    except Exception:
        (output / 'manifest.json').replace(output / f'manifest.invalid_{time.time_ns()}.json')
        raise
    (output / 'manifest.partial.json').unlink(missing_ok=True)
    # Keep per-episode checkpoints and job evidence for interruption recovery.
    return manifest
