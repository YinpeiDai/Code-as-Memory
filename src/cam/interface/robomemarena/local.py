"""Validate existing captures, then assemble the capped learner dataset locally."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py

from cam.data import demonstrations as demos
from cam.io import read, sha256, write
from cam.interface.robomemarena import data as rmd
from cam.interface.robomemarena.capture import historical_outcomes, parallel_capture_tasks, validation_sample


def stage_file(source, expected_hash, destination):
    """Read NFS sequentially once; hashing and H5 assembly then use the SSD."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    if destination.exists():
        if sha256(destination) != expected_hash:
            raise ValueError('Staged source file changed')
        return destination
    partial = destination.with_suffix(destination.suffix + '.partial')
    if partial.exists():
        partial.rename(partial.with_name(partial.name + f'.interrupted_{time.time_ns()}'))
    shutil.copyfile(source, partial)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(partial) != expected_hash:
        raise ValueError('Source changed or local copy hash does not match')
    partial.replace(destination)
    return destination


def capture_and_assemble_local(*, adapter, root, output, destination, manifest, scored,
                              stats, cache_jobs, reusable, reuse, python, workers, unchanged):
    """All requested capture/validation tasks finish before local H5 assembly."""
    destination = Path(destination).resolve()
    if destination == output.resolve() or destination.is_relative_to(output.resolve()):
        raise ValueError('Local assembly must use a separate dataset directory')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, 'tasks': {}, 'capture_source': str(output),
                'assembly': {'storage': 'local', 'directory': str(destination),
                             'after_all_capture_and_replay_validation': True,
                             'demo_cap': manifest['selection']['max_successful']}}
    write(destination/'manifest.partial.json', manifest)
    cap = manifest['selection']['max_successful']
    nvalidate = manifest['selection']['validation_episodes']
    protocol = {k: manifest[k] for k in ('benchmark_hash', 'raw_data', 'depth', 'fields', 'release')}
    protocol['capture_version'] = 1
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

    def prepare_task(task, native):
        records = adapter.records(task, 'train')
        known, provenance = historical_outcomes(manifest, task, records, scored.get(task, {}), cache_jobs, stats[task])
        original = (reusable or {}).get('tasks', {}).get(task)
        existing = {int(r['episode']): r for r in original['episodes']} if original else {}
        if len(existing) > cap:
            raise ValueError('Existing demonstrations exceed the cap')
        for ep, row in existing.items():
            if known[ep]['status'] != 'success' or known[ep]['steps'] != row['frames']:
                raise ValueError('Reusable capture does not match its official successful outcome')
        candidates = [r for r in records if int(r['episode']) not in existing and known.get(int(r['episode']), {}).get('status') == 'success']
        candidates += [r for r in records if int(r['episode']) not in known and int(r['episode']) not in existing]
        candidates = [{**r, **({'actions_sha256': known[int(r['episode'])]['actions_sha256']} if int(r['episode']) in known else {})} for r in candidates]
        goal = rmd.task_prompts(adapter.source)[f'task{rmd.task_id(task)}']
        folder = output/'captures'/task
        job = {'source': str(adapter.source), 'root': str(root), 'task': task, 'depth': True,
            'mode': 'capture', 'output': str(output), 'capture_directory': str(folder),
            'episodes': candidates, 'success_limit': cap-len(existing), 'goal': goal,
            'protocol_sha256': protocol_hash}
        write(destination/'audit'/f'{task}.capture_plan.json', job)
        result = native(job) if candidates and job['success_limit'] else {'results': []}
        attempted = result['results']
        expected = {int(r['episode']): int(r['seed']) for r in candidates}
        if len({r['episode'] for r in attempted}) != len(attempted):
            raise ValueError('Duplicate capture outcome')
        for row in attempted:
            if expected.get(int(row['episode'])) != int(row['seed']) or row['protocol_sha256'] != protocol_hash:
                raise ValueError('Capture worker changed an assigned identity or protocol')
        outcomes = {**known, **{int(r['episode']): r for r in attempted}}
        captured = {int(r['episode']): r for r in attempted if r['status'] == 'success' and r['tsr']}
        selected = [r for r in records if int(r['episode']) in existing or int(r['episode']) in captured]
        if not selected or len(selected) > cap:
            raise ValueError(f'Expected 1–{cap} successful captures for {task}; found {len(selected)}')
        episodes = [{**r, 'frames': outcomes[int(r['episode'])]['steps'], 'demo_frames': 0,
                     'goal': goal, 'video': None} for r in selected]
        sample = validation_sample(episodes, nvalidate, task)
        assignment = [{'episode': int(r['episode']), 'seed': int(r['seed'])} for r in sample]
        local_validation = destination/'audit'/f'{task}.validation.json'
        old_validation = output/'audit'/f'{task}.validation.json'
        prior = local_validation if local_validation.exists() else old_validation
        if prior.exists():
            validation = read(prior)
            if validation['identities'] != assignment:
                raise ValueError('Previous validation used a different sample')
            validation = {**validation, 'reused_validation': {'file': str(prior), 'sha256': sha256(prior)}}
        else:
            checks = native({'source': str(adapter.source), 'root': str(root), 'task': task,
                'depth': True, 'mode': 'validate', 'identities': assignment})['results'] if sample else []
            validation = {'requested': nvalidate, 'identities': assignment, 'results': checks}
        checks = validation['results']
        if len(checks) != len(assignment) or {(r['episode'], r['seed']) for r in checks} != {(r['episode'], r['seed']) for r in assignment}:
            raise ValueError('Validation returned different identities')
        validation['passed'] = all(r['status'] == 'success' and r['tsr']
            and r['steps'] == outcomes[int(r['episode'])]['steps']
            and r['actions_sha256'] == outcomes[int(r['episode'])]['actions_sha256'] for r in checks)
        write(local_validation, validation)
        if not validation['passed']:
            raise RuntimeError(f'Sampled replay validation failed for {task}')
        sources = []
        for r in selected:
            ep = int(r['episode'])
            sources.append({'episode': ep,
                'file': str(reuse/original['file']) if ep in existing else str(folder/captured[ep]['capture_file']),
                'sha256': original['sha256'] if ep in existing else captured[ep]['capture_sha256']})
        item = rmd.release()['tasks'][task]
        successful = [r for r in records if outcomes.get(int(r['episode']), {}).get('status') == 'success']
        value = {'task': task, 'task_id': rmd.task_id(task), 'episodes': episodes,
            'source': {'method': rmd.METHOD, 'repo': manifest['release']['repo'], 'revision': manifest['release']['revision'],
                'directory': item['directory'], 'snapshot': str(root), 'bddl': item['bddl'],
                'bddl_sha256': item['bddl_sha256'], 'benchmark_commit': rmd.release()['benchmark_source']['commit']},
            'replay': {'released': len(records), 'replayed': len(outcomes), 'successful': len(successful),
                'selected': len(selected), 'strata': rmd.strata(successful, selected, item),
                'identities': [outcomes[e] for e in sorted(outcomes)], 'capture_attempts': len(attempted)},
            'validation': {'requested': nvalidate, 'checked': len(checks), 'successes': len(checks),
                          'episodes': [r['episode'] for r in sample], 'audit': local_validation.relative_to(destination).as_posix()}}
        if existing:
            value['reuse'] = {'dataset': str(reuse), 'episodes': sorted(existing), 'source_h5_sha256': original['sha256']}
        unchanged(stats[task])
        ready = {'item': value, 'sources': sources, 'stats': stats[task], 'historical_provenance': provenance}
        write(destination/'audit'/f'{task}.ready.json', ready)
        old_error = destination/'audit'/f'{task}.error.json'
        if old_error.exists():
            old_error.rename(old_error.with_name(f'{task}.error.resolved_{time.time_ns()}.json'))
        return task, ready

    def workflow(task, native):
        try:
            return prepare_task(task, native)
        except Exception as exc:
            write(destination/'audit'/f'{task}.error.json', {'task': task, 'error': str(exc), 'time': time.time()})
            raise

    run_native = lambda job: rmd.run_worker(job, python, output/'work')
    ready = dict(parallel_capture_tasks(manifest['requested_tasks'], workers, workflow, run_native))
    # No task H5 is assembled until every assigned task has completed this gate.
    write(destination/'capture_validation_complete.json', {'tasks': list(ready), 'time': time.time()})

    def assemble(task):
        record = ready[task]
        target = destination/'h5'/f'record_dataset_{task}.h5'
        target.parent.mkdir(parents=True, exist_ok=True)
        final_audit = destination/'audit'/f'{task}.json'
        if final_audit.exists():
            value = read(final_audit)
            if value['episodes'] != record['item']['episodes'] or sha256(target) != value['sha256']:
                raise ValueError('Completed local assembly changed')
            return task, {k: v for k, v in value.items() if k != 'stats'}
        partial = target.with_suffix('.h5.partial')
        for path in (partial, target):
            if path.exists():
                archived = destination/'interrupted'/str(time.time_ns())/path.name
                archived.parent.mkdir(parents=True)
                shutil.move(path, archived)
        staging = destination/'.staging'/task
        staged = {}
        with demos.create(partial, 'robomemarena', task, manifest['fields'], True, rmd.METHOD) as h5:
            for source in record['sources']:
                key = (source['file'], source['sha256'])
                if key not in staged:
                    staged[key] = stage_file(source['file'], source['sha256'], staging/f"{source['sha256']}.h5")
                with h5py.File(staged[key], 'r') as src:
                    src.copy(f"episode_{source['episode']}", h5)
        unchanged(record['stats'])
        partial.replace(target)
        value = {**record['item'], 'file': target.relative_to(destination).as_posix(), 'sha256': sha256(target)}
        write(final_audit, {**value, 'stats': record['stats']})
        # Only disposable local staging copies are removed; NFS originals remain.
        shutil.rmtree(staging)
        return task, value

    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(assemble, task) for task in ready]):
            task, value = future.result()
            manifest['tasks'][task] = value
            write(destination/'manifest.partial.json', manifest)
    manifest['tasks'] = {t: manifest['tasks'][t] for t in manifest['requested_tasks']}
    write(destination/'manifest.json', manifest)
    try:
        rmd.validate_dataset(destination, verify_hashes=False, benchmark_source=adapter.source)
    except Exception:
        (destination/'manifest.json').replace(destination/f'manifest.invalid_{time.time_ns()}.json')
        raise
    (destination/'manifest.partial.json').unlink(missing_ok=True)
    return manifest


def finalize_assembled(destination, adapter, tasks, cap, validation_episodes):
    """Recover final validation only; never capture, assemble or alter demo files."""
    destination = Path(destination).resolve()
    if (destination/'manifest.json').exists():
        raise ValueError('The assembled pool already has a final manifest')
    manifest = read(destination/'manifest.partial.json')
    gate = read(destination/'capture_validation_complete.json')
    if (manifest['requested_tasks'] != tasks or set(manifest['tasks']) != set(tasks)
            or set(gate['tasks']) != set(tasks)
            or manifest['selection']['max_successful'] != cap
            or manifest['selection']['validation_episodes'] != validation_episodes):
        raise ValueError('Assembled pool does not match the complete assignment')
    for task in tasks:
        item = manifest['tasks'][task]
        audit = read(destination/'audit'/f'{task}.json')
        ready = read(destination/'audit'/f'{task}.ready.json')
        validation = read(destination/'audit'/f'{task}.validation.json')
        if ({k: v for k, v in audit.items() if k != 'stats'} != item
                or ready['item']['episodes'] != item['episodes']
                or not 1 <= len(item['episodes']) <= cap):
            raise ValueError(f'Assembly receipt changed: {task}')
        rows = validation['results']
        sample = validation_sample(item['episodes'], validation_episodes, task)
        identities = [{'episode': int(r['episode']), 'seed': int(r['seed'])} for r in sample]
        outcomes = {r['episode']: r for r in item['replay']['identities']}
        if (not validation['passed'] or validation['identities'] != identities
                or len(rows) != len(identities)
                or {(r['episode'], r['seed']) for r in rows} != {(r['episode'], r['seed']) for r in identities}
                or any(r['status'] != 'success' or not r['tsr']
                    or r['steps'] != outcomes[r['episode']]['steps']
                    or r['actions_sha256'] != outcomes[r['episode']]['actions_sha256'] for r in rows)):
            raise ValueError(f'Sampled replay validation changed: {task}')
    # Consumers must wait for this producer to exit. If validation fails, remove
    # the provisional manifest and preserve it as failed evidence, as in assembly.
    write(destination/'manifest.json', manifest)
    try:
        rmd.validate_dataset(destination, benchmark_source=adapter.source)
    except Exception:
        (destination/'manifest.json').replace(destination/f'manifest.invalid_{time.time_ns()}.json')
        raise
    (destination/'manifest.partial.json').unlink()
    return manifest
