"""Archive the three completed RoboMemArena experiments without changing scores.

Local trees are renamed with compatibility links. Turbo evaluations are copied,
hashed on both sides, and retained at their original location. No learning or
simulator evaluation is performed. Run `plan JOB`, then `assemble JOB`, `copy JOB`,
and `verify JOB`; copy can run alongside assembly because their targets differ.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import shutil
import stat
import threading
import time

from cam.io import read, write, sha256

REPO = Path(__file__).resolve().parents[1]
LOCAL = Path('/data/daiyp/CaM')
PUBLIC = Path('/data/daiyp/CaM_public')
ARMS = {
    'demo_only_25': ('20261108T154536_990183Z', '20261108T184542_314774Z', 1070),
    'demo_only_50': ('20261108T141119_562792Z', '20261108T160251_129267Z', 1179),
    'demo_interaction_50': ('20261108T183713_770492Z', '20261109T021302_043477Z', 1271),
}
DATASETS = {
    'pool': ('20261108T033325_071375Z/pool', 'demos/robomemarena'),
    'subset': ('20261108T153530_954388Z/demos25', 'demos/robomemarena/subsets/demo_only_25'),
}
NOTE = ('The 25-demo run includes the revised shared offline final-review prompt. '
        'The interaction run includes the RoboMemArena v2 manipulation/recovery supplement, '
        'designed after host review of the baseline. Learners received no held-out feedback. '
        'These comparisons change prompts as well as demo/interaction assignments. '
        'T04 has 32 demos and 32 interactions in the larger arms; all other tasks have 50. '
        'The 25-demo arm has exactly 25 per task. Interactions use each assigned demo seed once.')


def inventory(root):
    """List regular bytes and symlinks without traversing assets or other trees."""
    files, links = [], []
    for base, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            p = Path(base) / name
            s = p.lstat()
            rel = p.relative_to(root).as_posix()
            if stat.S_ISLNK(s.st_mode):
                links.append({'path': rel, 'target': os.readlink(p)})
            elif stat.S_ISREG(s.st_mode):
                files.append({'path': rel, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
    return sorted(files, key=lambda x: x['path']), sorted(links, key=lambda x: x['path'])


def plan(job):
    from cam.core.budget import program_hash
    from cam.core.history import select_checkpoint
    if (job / 'plan.json').exists():
        raise FileExistsError(job / 'plan.json')
    for part in ['demos', 'dev_logs', 'eval', 'solutions', 'configs/solutions']:
        if (PUBLIC / part / 'robomemarena').exists():
            raise FileExistsError(PUBLIC / part / 'robomemarena')
    value = {'schema': 'cam-robomemarena-public-plan/1', 'public_root': str(PUBLIC),
             'created': time.time(), 'datasets': {}, 'arms': {}, 'comparison_note': NOTE}
    for key, (source, destination) in DATASETS.items():
        path = LOCAL / 'datasets/robomemarena_rgbd' / source
        manifest = read(path / 'manifest.json')
        value['datasets'][key] = {'source': str(path), 'destination': destination,
                                 'manifest_sha256': sha256(path / 'manifest.json'),
                                 'episodes': sum(len(t['episodes']) for t in manifest['tasks'].values()),
                                 'h5_bytes': sum((path / t['file']).stat().st_size for t in manifest['tasks'].values())}
    for arm, (batch_tag, eval_tag, successes) in ARMS.items():
        evaluator = LOCAL / 'outputs/robomemarena_full' / eval_tag
        batch = LOCAL / 'outputs/development_batches' / batch_tag
        assert read(evaluator / 'status.json')['status'] == 'finished'
        assert read(batch / 'status.json')['status'] == 'finished'
        results = read(evaluator / 'results.json')
        assert (results['scored'], results['expected'], results['successes']) == (1300, 1300, successes)
        assert len(results['tasks']) == 26
        rows = {}
        for task, scored in sorted(results['tasks'].items()):
            root = Path(scored['run']); folder = root / 'tasks' / task
            run = read(root / 'run.json'); frozen = read(folder / 'policy/freeze.json')
            assert read(folder / 'logs/completion.json')['status'] == 'finished'
            assert program_hash(folder / 'policy') == frozen['program_hash']
            budget = run['config']['budget']
            assert frozen['training']['attempts'] == frozen['checkpoint'] == budget
            _, checkpoint = select_checkpoint(folder, budget)
            assert checkpoint['program_hash'] == frozen['program_hash']
            assert scored['scored'] == scored['expected'] == 50 and not scored['infrastructure_episodes']
            assigned = read(root / 'inputs/dataset.json')['tasks'][task]
            data_key = 'subset' if arm == 'demo_only_25' else 'pool'
            dataset = Path(value['datasets'][data_key]['source'])
            dm = read(dataset / 'manifest.json')['tasks'][task]
            assert assigned == dm
            h5 = folder / 'demonstrations' / Path(dm['file']).name
            assert not h5.is_symlink() and h5.stat().st_ino == (dataset / dm['file']).stat().st_ino
            assert len(assigned['episodes']) == (25 if arm == 'demo_only_25' else 32 if task.startswith('T04_') else 50)
            attempts = frozen['training']['attempt_records']
            if budget:
                assert len(attempts) == len(assigned['episodes'])
                assert sorted(a['episode'] for a in attempts) == sorted(e['episode'] for e in assigned['episodes'])
            test_metadata = read(root / 'inputs/metadata/test' / f'record_dataset_{task}_metadata.json')
            assert not {e['seed'] for e in assigned['episodes']} & {e['seed'] for e in test_metadata['records']}
            test = (folder / 'rollouts/test').resolve()
            retry = (folder / 'rollouts/infrastructure_retries').resolve()
            remote = not test.is_relative_to(LOCAL)
            assert remote == (arm == 'demo_only_50')
            rows[task] = {'source_run': str(root), 'run_tag': batch_tag,
                          'development': f'dev_logs/robomemarena/{arm}/{task}',
                          'evaluation': f'eval/robomemarena/{arm}/{task}',
                          'dataset': str(Path(value['datasets'][data_key]['destination']) / dm['file']),
                          'demos': len(assigned['episodes']), 'demo_identities': assigned['episodes'],
                          'training_attempts': budget, 'checkpoint': budget,
                          'program_hash': frozen['program_hash'], 'test_source': str(test),
                          'retry_source': str(retry), 'copy_from_turbo': remote,
                          'scored': 50, 'expected': 50, 'successes': scored['successes'],
                          'test_metadata_sha256': sha256(root / 'inputs/metadata/test' / f'record_dataset_{task}_metadata.json')}
            if remote:
                for kind, path in [('test', test), ('retry', retry)]:
                    files, links = inventory(path)
                    assert not links, (path, links)
                    rows[task][kind + '_files'] = files
        value['arms'][arm] = {'batch': str(batch), 'evaluator': str(evaluator), 'tasks': rows,
                             'scored': 1300, 'expected': 1300, 'successes': successes}
    for task in value['arms']['demo_only_50']['tasks']:
        assert len({a['tasks'][task]['test_metadata_sha256'] for a in value['arms'].values()}) == 1
    remote_bytes = sum(f['bytes'] for r in value['arms']['demo_only_50']['tasks'].values()
                       for k in ['test_files', 'retry_files'] for f in r[k])
    assert shutil.disk_usage(PUBLIC).free > remote_bytes + 100 * 1024**3
    value['remote_bytes_to_copy'] = remote_bytes
    write(job / 'plan.json', value)
    print(json.dumps({'job': str(job), 'remote_bytes_to_copy': remote_bytes,
                      'demos': value['datasets'], 'runs': 78, 'test_episodes': 3900}, indent=2))


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    desired = os.path.relpath(target, path.parent)
    if path.is_symlink() and os.readlink(path) == desired:
        return
    if path.exists() and not path.is_symlink():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + '.archive_link_tmp')
    if tmp.is_symlink(): tmp.unlink()
    tmp.symlink_to(desired, target_is_directory=target.is_dir())
    os.replace(tmp, path)


def move(source, target, job):
    if source.is_symlink() and source.resolve() == target.resolve() and target.exists():
        return
    before = source.lstat()
    assert not stat.S_ISLNK(before.st_mode), source
    target.parent.mkdir(parents=True, exist_ok=True)
    assert before.st_dev == target.parent.stat().st_dev
    if target.exists() or target.is_symlink(): raise FileExistsError(target)
    os.rename(source, target)
    after = target.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    link(source, target)
    with (job / 'moves.jsonl').open('a') as out:
        out.write(json.dumps({'source': str(source), 'destination': str(target),
                              'device': after.st_dev, 'inode': after.st_ino,
                              'same_inode': True, 'at': time.time()}) + '\n')


def attach_evaluation(row):
    dev = PUBLIC / row['development']; ev = PUBLIC / row['evaluation']
    task = dev.name; rollout = dev / 'tasks' / task / 'rollouts'
    link(rollout / 'test', ev)
    link(rollout / 'infrastructure_retries', ev / 'infrastructure_retries')
    link(ev / 'frozen_policy', dev / 'tasks' / task / 'policy')
    link(ev / 'source_run.json', dev / 'run.json')
    link(ev / 'freeze.json', dev / 'tasks' / task / 'policy/freeze.json')


def assemble(job):
    p = read(job / 'plan.json')
    for key in ['pool', 'subset']:
        item = p['datasets'][key]
        move(Path(item['source']), PUBLIC / item['destination'], job)
        assert sha256(PUBLIC / item['destination'] / 'manifest.json') == item['manifest_sha256']
    for arm, bundle in p['arms'].items():
        for task, row in bundle['tasks'].items():
            dev = PUBLIC / row['development']; ev = PUBLIC / row['evaluation']
            move(Path(row['source_run']), dev, job)
            if not row['copy_from_turbo']:
                move(Path(row['test_source']), ev, job)
                move(Path(row['retry_source']), ev / 'infrastructure_retries', job)
                attach_evaluation(row)
            (dev / 'STORAGE.md').write_text(
                f'# Public archive storage\n\nOriginal run: `{row["source_run"]}` (compatibility link).\n'
                f'Development: `{row["development"]}` relative to the archive root.\n'
                f'Held-out rollouts: `{row["evaluation"]}`; training rollouts remain in this run.\n'
                f'Assigned H5: `{row["dataset"]}` (the prepared file remains a regular hardlink).\n'
                'Original pinned manifests, prompts, source snapshots and policy bytes are unchanged.\n')
            with (dev / 'README.md').open() as stream: text = stream.read()
            addition = '\nPublic archive storage: see [STORAGE.md](STORAGE.md).\n'
            if addition not in text: (dev / 'README.md').write_text(text + addition)
        move(Path(bundle['batch']), PUBLIC / 'dev_logs/robomemarena' / arm / '_provenance/development_batch', job)
        # Copy small evaluator provenance. Its immutable original remains in outputs.
        provenance = PUBLIC / 'eval/robomemarena' / arm / '_provenance/evaluator'
        shutil.copytree(bundle['evaluator'], provenance, symlinks=True, dirs_exist_ok=True)
        manifest = {k: v for k, v in bundle.items() if k not in ['tasks']}
        manifest.update(schema='cam-experiment-archive/1', experiment=arm, comparison_note=NOTE,
                        tasks={t: {k: v for k, v in r.items() if k not in ['test_files', 'retry_files']}
                               for t, r in bundle['tasks'].items()})
        for kind in ['dev_logs', 'eval']:
            dest = PUBLIC / kind / 'robomemarena' / arm
            write(dest / 'manifest.json', manifest)
            (dest / 'README.md').write_text(
                f'# RoboMemArena {arm}: {kind}\n\n26 tasks; 1,300 scored held-out episodes; '
                f'{bundle["successes"]} successes (official TSR). Paths in manifest.json are relative '
                'to the public archive root; source paths record provenance.\n\n' + NOTE + '\n\n'
                'Development includes original transcripts, prompts, policy revisions, task-only demo '
                'hardlinks, and counted training rollouts. Evaluation contains R000 (demo-only), '
                'R050 (interaction), or R032 (T04 interaction), with all scored successes and failures. '
                'Final executable code is under solutions/robomemarena; memory schemas and provenance '
                'are under configs/solutions/robomemarena.\n\n'
                'Original JSON and source snapshots retain historical absolute paths. Canonical '
                'local paths remain compatibility links. Preserve hardlinks when transferring. '
                'External native assets are inventoried separately and must be installed on a new machine. '
                'SHA256SUMS excludes caches, locks and regenerable reports; it never follows symlinks.\n')
    demos = PUBLIC / 'demos/robomemarena'
    (demos / 'README.md').write_text(
        '# RoboMemArena replay-derived RGB-D demonstrations\n\n'
        '1,282 unique successful captured demos: 50 for 25 tasks and 32 for T04. '
        'subsets/demo_only_25 contains the exact 650-demo assignment (25/task), drawn from the same pool. '
        'Those subset episodes are not additional unique demonstrations.\n\n'
        'Front and wrist cameras: uint8 RGB 256x256x3, aligned int16 optical-axis depth '
        '256x256 in millimetres (0 unknown), intrinsics/extrinsics, TCP, joints, fingers and actions. '
        'RGB-D capture and official success were checked in the same replay; five sampled episodes '
        'per task were independently replay-validated (130/130). Original audit and capture receipts '
        'are preserved here. Failed and intermediate capture evidence stays in provenance/source storage.\n\n'
        'manifest.json and each task H5 are preserved byte-for-byte, including historical provenance '
        'paths. H5 paths are relative to this folder. New archive metadata records current paths. '
        'No simulator replay was performed for this organization.\n')
    write(job / 'assembly.json', {'status': 'complete', 'runs': 78, 'at': time.time()})
    print('Local assembly complete', flush=True)


def copy_file(source, target, expected):
    """Never overwrite a mismatching completed destination; preserve source bytes."""
    before = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (expected['bytes'], expected['mtime_ns']), source
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = sha256(source)
        if target.is_symlink() or sha256(target) != digest: raise ValueError(f'Existing destination differs: {target}')
        return digest
    temp = target.with_name(target.name + '.copying')
    digest = hashlib.sha256()
    with source.open('rb') as incoming, temp.open('wb') as outgoing:
        while chunk := incoming.read(8 * 1024**2):
            digest.update(chunk); outgoing.write(chunk)
        outgoing.flush(); os.fsync(outgoing.fileno())
    after = source.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), source
    expected_hash = digest.hexdigest()
    assert sha256(temp) == expected_hash, target
    shutil.copystat(source, temp)
    os.rename(temp, target)
    return expected_hash


def copy_remote(job):
    p = read(job / 'plan.json'); lock = threading.Lock(); completed = []
    rows = p['arms']['demo_only_50']['tasks']
    previous = {}
    if (job / 'copied_files.jsonl').exists():
        for line in (job / 'copied_files.jsonl').read_text().splitlines():
            record = json.loads(line); previous[record['destination']] = record
    def task_copy(task, row):
        ev = PUBLIC / row['evaluation']; hashes = []
        for kind, source, destination in [('test', Path(row['test_source']), ev),
                                          ('retry', Path(row['retry_source']), ev / 'infrastructure_retries')]:
            destination.mkdir(parents=True, exist_ok=True)
            current, links = inventory(source)
            assert current == row[kind + '_files'] and not links, source
            for item in current:
                target = destination / item['path']
                saved = previous.get(target.relative_to(PUBLIC).as_posix())
                if saved and target.is_file() and not target.is_symlink():
                    info = target.stat()
                    assert (info.st_size, info.st_mtime_ns) == (saved['bytes'], saved['mtime_ns'])
                    digest = sha256(target)
                    assert digest == saved['sha256'], target
                else:
                    digest = copy_file(source / item['path'], target, item)
                record = {**item, 'source': str(source / item['path']),
                          'destination': target.relative_to(PUBLIC).as_posix(), 'sha256': digest}
                hashes.append(record)
                with lock:
                    with (job / 'copied_files.jsonl').open('a') as out:
                        out.write(json.dumps(record) + '\n')
            after, _ = inventory(source)
            assert after == current, source
        write(job / 'remote_tasks' / (task + '.json'), {'files': hashes, 'source_preserved': True})
        with lock:
            completed.append(task)
            write(job / 'copy_status.json', {'status': 'copying', 'completed_tasks': sorted(completed),
                                            'total_tasks': 26, 'updated': time.time()})
        print('copied and SHA-256 verified', task, flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = [pool.submit(task_copy, t, r) for t, r in rows.items()]
        for future in as_completed(pending): future.result()
    write(job / 'copy_status.json', {'status': 'complete', 'tasks': 26, 'bytes': p['remote_bytes_to_copy'], 'at': time.time()})


def export_solutions(job):
    from cam.interface.registry import get_adapter
    from cam.interface.base import source_hash
    from cam.harness.solutions import record_hash, load_solutions
    from cam.core.budget import program_hash
    p = read(job / 'plan.json'); adapter = get_adapter('robomemarena')
    for arm, bundle in p['arms'].items():
        catalog = {'schema': 'cam-solutions/1', 'benchmark': 'robomemarena', 'experiment': arm,
                   'benchmark_source_sha256': source_hash(adapter), 'comparison_note': NOTE,
                   'description': 'Final frozen Python code; historical code bytes and memory schemas preserved.',
                   'source_coding': {'agent': 'claude_code', 'model': 'us.anthropic.claude-opus-5-5'}, 'tasks': {}}
        for task, row in bundle['tasks'].items():
            root = PUBLIC / row['development']; run = read(root / 'run.json'); policy = root / 'tasks' / task / 'policy'
            assert program_hash(policy) == row['program_hash']
            config = {k: v for k, v in run['config'].items() if k != 'coding'}
            # Published import performs evaluation only. Training provenance stays below.
            config.update(mode='demos', budget=0, both_protocol=None, training_seed_count=None)
            if 'evaluation_config' not in catalog: catalog['evaluation_config'] = config
            assert catalog['evaluation_config'] == config
            code = {}
            for source in sorted((policy / 'tools').glob('*.py')):
                assert not source.is_symlink()
                data = source.read_bytes(); compile(data, str(source), 'exec')
                for base in [REPO, PUBLIC]:
                    dest = base / 'solutions/robomemarena' / arm / task / source.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and dest.read_bytes() != data: raise ValueError(f'Existing solution differs: {dest}')
                    dest.write_bytes(data)
                code[source.name] = sha256(source)
            records = read(root / 'inputs/metadata/test' / f'record_dataset_{task}_metadata.json')['records']
            assert records == adapter.records(task, 'test')
            assert run['interfaces'][task] == adapter.interface(task, True)
            catalog['tasks'][task] = {
                'task_spec': read(policy / 'task.json'), 'code_sha256': code,
                'source_run_tag': row['run_tag'], 'source_program_hash': row['program_hash'],
                'source_harness_sha256': run['harness_hash'], 'source_development': row['development'],
                'source_evaluation': row['evaluation'], 'assigned_demos': row['demos'],
                'training_attempts': row['training_attempts'], 'source_checkpoint': row['checkpoint'],
                'test_records_sha256': record_hash(records), 'interface_sha256': record_hash(run['interfaces'][task]),
                'archived_result': {k: row[k] for k in ['expected', 'scored', 'successes']}}
        for base in [REPO, PUBLIC]:
            write(base / 'configs/solutions/robomemarena' / (arm + '.json'), catalog)
            load_solutions(base / 'solutions/robomemarena' / arm)
        print('exported final solutions', arm, len(catalog['tasks']), flush=True)


def hash_local(job):
    """Hash moved immutable bytes while the independent Turbo copy is running."""
    roots = [PUBLIC / 'demos/robomemarena', PUBLIC / 'dev_logs/robomemarena']
    roots += [PUBLIC / 'eval/robomemarena' / arm for arm in ['demo_only_25', 'demo_interaction_50']]
    unique = {}
    for root in roots:
        files, _ = inventory(root)
        for row in files:
            path = root / row['path']
            if any(x in path.relative_to(root).parts for x in ['__pycache__', 'reports', 'locks']): continue
            if path.name in ['SHA256SUMS', 'links.json', 'usage.json'] or path.name.endswith(('.pyc', '.lock', '-wal', '-shm', '-journal')): continue
            s = path.stat(); key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            unique.setdefault(key, path)
    total = sum(key[2] for key in unique); count = copied = 0
    write(job / 'hash_status.json', {'status': 'hashing', 'total_bytes': total, 'files': len(unique)})
    def hashed(pair):
        key, path = pair
        digest = sha256(path); s = path.stat()
        assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == key, path
        return {'stat': key, 'path': str(path), 'sha256': digest}
    with (job / 'local_hashes.jsonl').open('w') as output, ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(hashed, pair) for pair in unique.items()]
        for future in as_completed(pending):
            row = future.result(); output.write(json.dumps(row) + '\n'); count += 1; copied += row['stat'][2]
            if count % 100 == 0 or copied == total:
                output.flush()
                write(job / 'hash_status.json', {'status': 'hashing', 'total_bytes': total, 'files': len(unique),
                                                 'hashed_files': count, 'hashed_bytes': copied, 'updated': time.time()})
    write(job / 'hash_status.json', {'status': 'complete', 'total_bytes': total, 'files': count, 'at': time.time()})


def verify(job):
    import h5py
    import numpy as np
    from cam.interface.robomemarena.data import validate_dataset
    from cam.harness.prepare import validate_run
    from cam.core.budget import program_hash
    p = read(job / 'plan.json')
    assert read(job / 'assembly.json')['status'] == 'complete'
    cache = {}
    if (job / 'local_hashes.jsonl').exists():
        assert read(job / 'hash_status.json')['status'] == 'complete'
        for line in (job / 'local_hashes.jsonl').read_text().splitlines():
            row = json.loads(line); cache[tuple(row['stat'])] = row['sha256']
    def digest(path):
        assert not path.is_symlink()
        s = path.stat(); key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if key not in cache: cache[key] = sha256(path)
        return cache[key]
    # A source-stream SHA-256 and a separate read of each copied destination have
    # already matched. Reuse only receipts whose immutable size/mtime still match.
    checks = []
    for item in p['datasets'].values():
        root = PUBLIC / item['destination']; manifest = read(root / 'manifest.json')
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(digest, root / t['file']): (task, t) for task, t in manifest['tasks'].items()}
            for future in as_completed(pending):
                task, row = pending[future]; assert future.result() == row['sha256'], task
                print('verified demo hash', root.name, task, flush=True)
        validate_dataset(root, verify_hashes=False)
        checks.append({'path': item['destination'], 'episodes': item['episodes'], 'h5_files': len(manifest['tasks'])})
    write(job / 'demo_verification.json', checks)
    totals = collections.Counter(); evaluation = {}; dependencies = {}
    # Verify completed local artifacts first, while Turbo transfer continues.
    for arm in ['demo_only_25', 'demo_interaction_50', 'demo_only_50']:
        bundle = p['arms'][arm]
        if arm == 'demo_only_50':
            while not (job / 'copy_status.json').exists() or read(job / 'copy_status.json')['status'] != 'complete':
                write(job / 'verification_status.json', {'status': 'waiting_for_turbo_copy', 'verified': dict(totals), 'updated': time.time()})
                time.sleep(20)
            for receipt in (job / 'remote_tasks').glob('*.json'):
                for item in read(receipt)['files']:
                    path = PUBLIC / item['destination']; s = path.stat()
                    assert (s.st_size, s.st_mtime_ns) == (item['bytes'], item['mtime_ns'])
                    cache[(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)] = item['sha256']
            for row in bundle['tasks'].values(): attach_evaluation(row)
        dependencies[arm] = {}
        for task, row in bundle['tasks'].items():
            root = PUBLIC / row['development']; ev = PUBLIC / row['evaluation']
            run = validate_run(root); policy = root / 'tasks' / task / 'policy'
            assert program_hash(policy) == row['program_hash']
            assert Path(row['source_run']).resolve() == root
            assigned = root / 'tasks' / task / 'demonstrations' / Path(row['dataset']).name
            assert not assigned.is_symlink() and assigned.stat().st_ino == (PUBLIC / row['dataset']).stat().st_ino
            frozen = read(policy / 'freeze.json'); count = success = 0
            for episode in run['test_episodes'][task]:
                folder = ev / f'R{row["checkpoint"]:03d}' / f'ep{episode:03d}'
                result = read(folder / 'result.json')
                assert result['scorable'] and result['program_hash'] == row['program_hash']
                assert (result['episode'], result['task'], result['split']) == (episode, task, 'test')
                assert bool(result['native_metrics']['tsr']) == bool(result['success'])
                actions = np.load(folder / 'actions.npy', mmap_mode='r')
                with h5py.File(folder / 'observations.h5', 'r') as obs:
                    frames = len(obs['step']); assert frames == len(actions) + 1 == result['video_frames'], folder
                    def aligned(name, obj):
                        if isinstance(obj, h5py.Dataset): assert len(obj) == frames, (folder, name)
                    obs.visititems(aligned)
                    for camera in ['agentview', 'wrist']:
                        assert obs[camera + '_rgb'].shape[1:] == (256, 256, 3)
                        assert obs[camera + '_depth'].shape[1:] == (256, 256)
                assert (folder / 'episode.mp4').stat().st_size > 0
                count += 1; success += bool(result['success'])
            assert (count, success) == (50, row['successes'])
            training = root / 'tasks' / task / 'rollouts/train'
            train_results = list(training.glob('R*/result.json')) if training.exists() else []
            assert len(train_results) == row['training_attempts'], (task, len(train_results), row['training_attempts'])
            for a in frozen['training']['attempt_records']:
                train_folder = training / f'R{a["id"]:03d}'
                result = read(train_folder / 'result.json')
                assert (result['episode'], result['program_hash'], result['status']) == (a['episode'], a['program_hash'], a['status'])
                assert result['split'] == 'train'
                actions = np.load(train_folder / 'actions.npy', mmap_mode='r')
                with h5py.File(train_folder / 'observations.h5', 'r') as obs:
                    frames = len(obs['step']); assert frames == len(actions) + 1 == result['video_frames']
                    obs.visititems(aligned)
                assert (train_folder / 'episode.mp4').stat().st_size > 0
            totals.update(test_episodes=count, successes=success, training_attempts=len(train_results), runs=1)
            evaluation[f'{arm}/{task}'] = {'scored': count, 'successes': success, 'training_attempts': len(train_results)}
            _, links = inventory(root)
            dependencies[arm][task] = [x for x in links if os.path.isabs(x['target']) and not x['target'].startswith(str(PUBLIC) + '/')]
            print('verified run and rollout alignment', arm, task, flush=True)
        write(PUBLIC / 'dev_logs/robomemarena' / arm / 'external_dependencies.json', dependencies[arm])
    write(job / 'evaluation_verification.json', {'totals': dict(totals), 'tasks': evaluation})
    export_solutions(job)
    isolated = read(job / 'solution_isolation_validation.json')
    assert isolated['all_passed'] and len(isolated['policies']) == 78
    write(PUBLIC / 'configs/solutions/robomemarena/validation.json', isolated)
    roots = [PUBLIC / 'demos/robomemarena'] + [PUBLIC / k / 'robomemarena' / a for k in ['dev_logs', 'eval'] for a in ARMS]
    roots += [PUBLIC / 'solutions/robomemarena', PUBLIC / 'configs/solutions/robomemarena']
    total_paths = total_bytes = 0
    for root in roots:
        files, links = inventory(root); lines = []; bytes_count = 0
        write(root / 'links.json', links) if 'solutions' not in root.parts else None
        for row in files:
            rel = row['path']; path = root / rel
            if any(x in Path(rel).parts for x in ['__pycache__', 'reports', 'locks']): continue
            if path.name in ['SHA256SUMS', 'links.json', 'usage.json'] or path.name.endswith(('.pyc', '.lock', '-wal', '-shm', '-journal')): continue
            lines.append(f'{digest(path)}  {rel}\n'); bytes_count += row['bytes']
        # Keep solutions/ strictly Python-only; checksums belong in configs/solutions.
        checksum = root / 'SHA256SUMS'
        if root == PUBLIC / 'solutions/robomemarena':
            checksum = PUBLIC / 'configs/solutions/robomemarena/code.SHA256SUMS'
            lines = [line[:66] + os.path.relpath(root / line[66:].rstrip('\n'), checksum.parent) + '\n' for line in lines]
        elif (root / 'links.json').exists():
            lines.append(f'{digest(root / "links.json")}  links.json\n')
        checksum.write_text(''.join(sorted(lines, key=lambda x: x[66:])))
        total_paths += len(lines); total_bytes += bytes_count
        print('checksummed', root, len(lines), flush=True)
    receipt = {'schema': 'cam-local-archive/1', 'status': 'complete', 'public_root': str(PUBLIC),
               'unique_demos': 1282, 'subset_demos': 650, 'development_runs': 78,
               'training_attempts': totals['training_attempts'], 'scored_test_episodes': totals['test_episodes'],
               'results': {a: {k: b[k] for k in ['scored', 'expected', 'successes']} for a, b in p['arms'].items()},
               'solutions': 78, 'remote_bytes_copied_and_verified': p['remote_bytes_to_copy'],
               'solutions_passed_isolated_offline_validation': 78, 'new_simulator_episodes': 0, 'model_calls': 0,
               'turbo_sources_preserved': True, 'local_moves': 'same-filesystem rename with compatibility links',
               'checksummed_regular_paths': total_paths, 'logical_checksummed_bytes_including_hardlinks': total_bytes,
               'comparison_note': NOTE, 'validation_job': str(job), 'completed_at': time.time()}
    write(PUBLIC / 'ROBOMEMARENA_RELEASE.json', receipt); write(job / 'status.json', receipt)
    write(job / 'verification_status.json', {'status': 'complete', 'verified': dict(totals), 'updated': time.time()})
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'assemble', 'copy', 'solutions', 'hash-local', 'verify'])
    parser.add_argument('job', type=Path)
    args = parser.parse_args(); job = args.job.resolve(); job.mkdir(parents=True, exist_ok=True)
    {'plan': plan, 'assemble': assemble, 'copy': copy_remote, 'solutions': export_solutions,
     'hash-local': hash_local, 'verify': verify}[args.action](job)


if __name__ == '__main__': main()
