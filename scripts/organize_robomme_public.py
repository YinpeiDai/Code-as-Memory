"""Archive only the historical 25-demo RoboMME 795/800 experiment.

Copies verified bytes without modifying source runs. Original CAP layouts and
schemas remain evidence; portable manifests and relative links provide indexes.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from cam.interface.base import source_hash
from cam.interface.registry import get_adapter
from cam.config import RunConfig
from cam.core.budget import program_hash
from cam.harness.solutions import load_solutions, record_hash
from cam.io import read, sha256, write

REPO = Path(__file__).resolve().parents[1]
RUN = Path('/data/daiyp/cap_baseline/regular_demo_v2_restart_20261105_040321')
DEMOS = Path('/data/daiyp/cap_baseline/regular_25shot_rgbd_20261105/fewshot')
RGB = Path('/data/daiyp/cap_baseline/regular_25shot_20261105/fewshot')
PUBLIC = Path('/data/daiyp/CaM_public')
ARM = 'demo_only_25'
DEV = Path('dev_logs/robomme') / ARM
EVAL = Path('eval/robomme') / ARM
DATA = Path('demos/robomme')
CODE = Path('solutions/robomme') / ARM
CATALOG = Path('configs/solutions/robomme') / (ARM + '.json')
SOURCE = DEV / 'source_run'
EXCLUDED = ('demos_R0/demonstrations', 'evaluation_full_v1/tasks', 'website_live_v1/artifacts')
CACHES = {'__pycache__', '.git', '.venv', '.pytest_cache', '.ruff_cache', 'node_modules'}


def files(root, excluded=()):
    for parent, dirs, names in os.walk(root):
        parent = Path(parent)
        def skip(path):
            rel = path.relative_to(root)
            return (any(p in CACHES for p in rel.parts)
                    or path.suffix in {'.pyc', '.pyo', '.lock'}
                    or any(rel == Path(x) or Path(x) in rel.parents for x in excluded))
        dirs[:] = sorted(d for d in dirs if not skip(parent / d))
        for name in sorted(names):
            path = parent / name
            if skip(path):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Unexpected source link or special file: {path}')
            yield path


def plan(job):
    if (job / 'plan.json').exists():
        return read(job / 'plan.json')
    for path in [DATA, DEV, EVAL, CODE, CATALOG]:
        if (PUBLIC / path).exists():
            raise FileExistsError(PUBLIC / path)
    if (REPO / CODE).exists() or (REPO / CATALOG).exists():
        raise FileExistsError('A RoboMME solution publication already exists')
    extraction = read(DEMOS / 'manifest.json')
    summary = read(RUN / 'evaluation_full_v1/summary.json')
    score = summary['settings']['demos_R000']
    assert (score['test_completed'], score['test_expected'], score['test_success']) == (800, 800, 795)
    assert score['outcomes'] == {'success': 795, 'fail': 4, 'decision_limit': 1}
    assert sha256(RGB / 'manifest.json') == extraction['base_manifest_sha256']
    adapter = get_adapter('robomme')
    assert source_hash(adapter) == source_hash(get_adapter('robomme', RUN / 'demos_R0/benchmark/robomme'))
    assert len(extraction['tasks']) == len(adapter.tasks()) == 16
    tasks = {}
    for task, selected in extraction['tasks'].items():
        frozen = RUN / 'demos_R0/frozen' / task
        freeze = read(frozen / 'freeze.json')
        assert freeze['training']['attempts'] == freeze['checkpoint'] == 0
        assert program_hash(frozen) == freeze['program_hash']
        evaluation = RUN / 'evaluation_full_v1/tasks' / task
        records = read(evaluation / 'metadata/test' / f'record_dataset_{task}_metadata.json')['records']
        assert records == adapter.records(task, 'test')
        assert len(records) == 50
        train = {r['episode']: r for r in adapter.records(task, 'train')}
        episodes = selected['episodes']
        assert len(episodes) == 25
        assert Counter(e['difficulty'] for e in episodes) == {'easy': 9, 'medium': 8, 'hard': 8}
        assert all((e['seed'], e['difficulty']) == (train[e['episode']]['seed'], train[e['episode']]['difficulty']) for e in episodes)
        assert not {e['seed'] for e in episodes} & {r['seed'] for r in records}
        scored = []
        for record in records:
            folder = evaluation / 'demos_R000/results' / task / f"ep{record['episode']}"
            result = read(folder / 'result.json')
            assert (result['task'], result['episode'], result['split']) == (task, record['episode'], 'test')
            assert result['program_hash'] == freeze['program_hash']
            assert result['artifacts_complete'] and result['api_calls'] == 0
            assert result['status'] in {'success', 'fail', 'decision_limit'}
            assert result['success'] == (result['status'] == 'success')
            scored.append(result['success'])
        eval_frozen = evaluation / 'demos_R000/frozen' / task
        assert program_hash(eval_frozen) == freeze['program_hash']
        tasks[task] = {
            'development': str(DEV / task), 'source_frozen': str(SOURCE / 'demos_R0/frozen' / task),
            'evaluation': str(EVAL / task), 'rollouts': str(EVAL / task / 'R000'),
            'demonstration_h5': str(DATA / 'h5' / f'record_dataset_{task}.h5'),
            'assigned_demos': 25, 'training_attempts': 0, 'checkpoint': 0,
            'program_hash': freeze['program_hash'], 'expected': 50, 'scored': 50,
            'successes': sum(scored), 'demo_identities': episodes,
            'test_records_sha256': record_hash(records),
            'interface_sha256': record_hash(adapter.interface(task, True)),
        }
    assert sum(t['successes'] for t in tasks.values()) == 795
    entries = []
    def add_tree(source, destination, excluded=()):
        for path in files(source, excluded):
            stat = path.stat()
            entries.append({'source': str(path), 'destination': str(destination / path.relative_to(source)),
                            'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    add_tree(RUN, SOURCE, EXCLUDED)
    add_tree(RUN / 'evaluation_full_v1/tasks', EVAL)
    add_tree(DEMOS, DATA / '_provenance/extraction', ('h5',))
    add_tree(RGB / 'rollouts', DATA / 'rollouts')
    for task, selected in extraction['tasks'].items():
        path = RUN / 'demos_R0/demonstrations' / task / f'record_dataset_{task}.h5'
        stat = path.stat()
        entries.append({'source': str(path), 'destination': tasks[task]['demonstration_h5'],
                        'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'expected_sha256': selected['sha256']})
    assert len({e['destination'] for e in entries}) == len(entries)
    total = sum(e['bytes'] for e in entries)
    assert shutil.disk_usage(PUBLIC).free > total + 30 * 1024**3
    value = {'schema': 'cam-robomme-public-plan/1', 'created': time.time(),
             'source_run': str(RUN), 'source_demos': str(DEMOS), 'source_previews': str(RGB / 'rollouts'),
             'public_root': str(PUBLIC), 'experiment': ARM, 'tasks': tasks, 'files': entries,
             'total_bytes': total, 'excluded_source_trees': list(EXCLUDED), 'excluded_caches': sorted(CACHES),
             'excluded_suffixes': ['.pyc', '.pyo', '.lock'], 'source_preserved': True}
    write(job / 'plan.json', value)
    shutil.copy2(__file__, job / 'organize_robomme_public.py')
    return value


def copy_verified(item):
    source, target = Path(item['source']), PUBLIC / item['destination']
    before = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (item['bytes'], item['mtime_ns']), source
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if target.exists():
        source_hash = sha256(source)
        assert target.is_file() and not target.is_symlink() and sha256(target) == source_hash, target
    else:
        partial = target.with_name(target.name + '.copy-partial')
        with source.open('rb') as inp, partial.open('wb') as out:
            for chunk in iter(lambda: inp.read(8 * 1024**2), b''):
                digest.update(chunk)
                out.write(chunk)
        source_hash = digest.hexdigest()
        assert sha256(partial) == source_hash, target
        shutil.copystat(source, partial)
        partial.replace(target)
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), source
    if 'expected_sha256' in item:
        assert source_hash == item['expected_sha256'], source
    return {**item, 'sha256': source_hash, 'destination_read_verified': True}


def copy_all(job, value, workers):
    completed = {}
    journal = job / 'copied_files.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            completed[row['destination']] = row
    pending = []
    for item in value['files']:
        old = completed.get(item['destination'])
        target = PUBLIC / item['destination']
        if old and target.is_file() and target.stat().st_size == old['bytes']:
            assert sha256(target) == old['sha256'], target
        else:
            pending.append(item)
    with journal.open('a') as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_verified, item) for item in pending]
        for future in as_completed(futures):
            row = future.result()
            out.write(json.dumps(row) + '\n')
            completed[row['destination']] = row
            if len(completed) % 100 == 0 or len(completed) == len(value['files']) or row['bytes'] > 1024**3:
                out.flush()
                status = {'status': 'copying', 'files': len(completed), 'total_files': len(value['files']),
                          'verified_bytes': sum(r['bytes'] for r in completed.values()),
                          'total_bytes': value['total_bytes'], 'updated': time.time()}
                write(job / 'status.json', status)
                print(json.dumps(status), flush=True)
    write(job / 'status.json', {'status': 'copied', 'files': len(completed),
                              'verified_bytes': value['total_bytes'], 'updated': time.time()})


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, path.parent)
    if path.is_symlink() and os.readlink(path) == relative:
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.symlink_to(relative, target_is_directory=target.is_dir())


def indexes(job, value):
    extraction = read(DEMOS / 'manifest.json')
    # The original, unmodified manifest is retained in _provenance/extraction.
    portable = json.loads(json.dumps(extraction))
    portable['archive'] = {'original_manifest': '_provenance/extraction/manifest.json',
                           'original_manifest_sha256': sha256(DEMOS / 'manifest.json'),
                           'historical_schema_preserved': True, 'h5_bytes_unchanged': True}
    for task, item in portable['tasks'].items():
        item['file'] = f'h5/record_dataset_{task}.h5'
        assigned = PUBLIC / SOURCE / 'demos_R0/demonstrations' / task / Path(item['file']).name
        assigned.parent.mkdir(parents=True, exist_ok=True)
        if not assigned.exists():
            os.link(PUBLIC / DATA / item['file'], assigned)
        assert os.path.samefile(assigned, PUBLIC / DATA / item['file'])
    write(PUBLIC / DATA / 'manifest.json', portable)
    link(PUBLIC / SOURCE / 'evaluation_full_v1/tasks', PUBLIC / EVAL)
    for task in value['tasks']:
        view = PUBLIC / DEV / task
        study = PUBLIC / SOURCE / 'demos_R0'
        for name, target in {
            'agents': study / 'agents' / task, 'audit': study / 'audit' / task,
            'history': study / 'history' / task, 'workspace': study / 'workspaces' / task,
            'checkpoints': study / 'checkpoints' / task, 'frozen_policy': study / 'frozen' / task,
            'demonstrations': study / 'demonstrations' / task, 'costs': study / 'costs' / task,
            'prompt.txt': study / 'agents' / task / 'prompt.txt',
        }.items():
            assert target.exists(), target
            link(view / name, target)
        ev = PUBLIC / EVAL / task
        link(ev / 'R000', ev / 'demos_R000/results' / task)
        link(ev / 'frozen_policy', study / 'frozen' / task)
    costs = read(RUN / 'demos_R0/costs/summary.json')
    total_cost = sum(t['reported_cost_usd'] for t in costs['tasks'])
    manifest = {k: v for k, v in value.items() if k != 'files'}
    manifest.update(schema='cam-historical-experiment-archive/1', status='verifying',
                    expected=800, scored=800, successes=795, success_rate=795 / 800,
                    outcomes={'success': 795, 'fail': 4, 'decision_limit': 1},
                    development_api_cost_usd=total_cost, evaluation_api_cost_usd=0,
                    cost_basis='Claude CLI reported estimate; excludes local compute',
                    runner='historical CAP regular RoboMME', new_evaluation_performed=False)
    for category in (DEV, EVAL):
        write(PUBLIC / category / 'manifest.json', manifest)
        (PUBLIC / category / 'README.md').write_text(
            '# RoboMME demo-only, 25 demos per task\n\n'
            'Historical run `regular_demo_v2_restart_20261105_040321`: 16 tasks, '
            '400 RGB-D demos, zero training interactions, 795/800 held-out successes '
            '(99.375%, rounded to 99.4%). All four failures and one decision limit are retained.\n\n'
            'manifest.json maps paths relative to the CaM_public root. Development task folders '
            'index original agents, audits, revisions, workspaces and frozen policies under '
            'source_run/demos_R0. Evaluation task folders preserve the original snapshot and '
            'demos_R000/results/<Task>/epN tree; R000 is a relative shortcut.\n\n'
            'Original files retain their schemas and absolute provenance paths. This is an archive, '
            'not a new cam-run/1 preparation or rerun. Sources remain intact. Duplicate website '
            'artifact exports, Python caches and transient locks are omitted; canonical evidence '
            'is retained. Final code and separate catalogs are in solutions/robomme/demo_only_25 '
            'and configs/solutions/robomme. See ROBOMME.md at the public root.\n')
    (PUBLIC / DATA / 'README.md').write_text(
        '# Exact historical RoboMME RGB-D demonstrations\n\n'
        '16 H5 files, 25 demos each: 9 easy / 8 medium / 8 hard, 400 total. '
        'H5 bytes match the actual development inputs and original extraction SHA-256 values. '
        'Front/wrist RGB, aligned int16 optical-axis depth in millimetres, calibration, native TCP, '
        'joints, gripper state and recorded actions are retained.\n\n'
        'These files retain cap-regular-fewshot/2, not the current CaM learning-data schema. '
        'Use a validated fresh extraction for new learning. manifest.json uses relative H5 paths; '
        'its unmodified source is _provenance/extraction/manifest.json. rollouts/ contains the '
        '400 original RGB preview videos and posters from the matching base extraction. '
        'Only the H5 contains depth. See the selection CSV/JSON in _provenance/extraction/.\n')
    return manifest


def export_solutions(value):
    adapter = get_adapter('robomme')
    config = RunConfig(benchmark='robomme', mode='demos', depth=True, builtin_demo=True,
                       training_seed_count=None, policy_timeout=1800).to_dict()
    config.pop('coding')  # Evaluation does not authenticate or run a coding model.
    catalog = {'schema': 'cam-solutions/1', 'benchmark': 'robomme', 'experiment': ARM,
               'description': 'Unmodified final Python code from the historical 25-demo 795/800 run.',
               'benchmark_source_sha256': source_hash(adapter), 'evaluation_config': config,
               'source_coding': {'agent': 'claude_code', 'model': 'us.anthropic.claude-opus-5-5', 'provider': 'bedrock'},
               'archived_result': {'expected': 800, 'scored': 800, 'successes': 795, 'success_rate': 795 / 800},
               'compatibility_note': 'Current CaM import is a fresh evaluation; the archived 99.4% was scored by the original runner.',
               'tasks': {}}
    for task, row in value['tasks'].items():
        frozen = RUN / 'demos_R0/frozen' / task
        code = {}
        for source in sorted((frozen / 'tools').glob('*.py')):
            data = source.read_bytes()
            compile(data, str(source), 'exec')
            for base in (REPO, PUBLIC):
                target = base / CODE / task / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f'Existing solution differs: {target}')
                target.write_bytes(data)
            code[source.name] = sha256(source)
        catalog['tasks'][task] = {
            'task_spec': read(frozen / 'task.json'), 'code_sha256': code,
            'source_run_tag': RUN.name, 'source_program_hash': row['program_hash'],
            'source_harness_sha256': read(frozen / 'freeze.json')['harness_hash'],
            'source_development': row['development'], 'source_evaluation': row['evaluation'],
            'assigned_demos': 25, 'training_attempts': 0, 'source_checkpoint': 0,
            'test_records_sha256': row['test_records_sha256'], 'interface_sha256': row['interface_sha256'],
            'archived_result': {k: row[k] for k in ('expected', 'scored', 'successes')},
        }
    for base in (REPO, PUBLIC):
        write(base / CATALOG, catalog)
        load_solutions(base / CODE)
    return catalog


def verify_content(job, value):
    import h5py
    import numpy as np
    rows = []
    for task, task_row in value['tasks'].items():
        with h5py.File(PUBLIC / task_row['demonstration_h5']) as h5:
            assert h5.attrs['schema'] == 'cap-regular-fewshot/2' and bool(h5.attrs['depth'])
            assert set(h5) == {f"episode_{e['episode']}" for e in task_row['demo_identities']}
            for e in task_row['demo_identities']:
                g = h5[f"episode_{e['episode']}"]
                assert g.attrs['difficulty'] == e['difficulty'] and 'seed' not in g.attrs
                for camera in ('front', 'wrist'):
                    rgb, depth = g[camera + '_rgb'], g[camera + '_depth']
                    assert rgb.shape == (e['frames'], 256, 256, 3) and rgb.dtype == np.uint8
                    assert depth.shape == (e['frames'], 256, 256) and depth.dtype == np.int16
                assert (PUBLIC / DATA / e['video']).is_file()
                assert (PUBLIC / DATA / e['poster']).is_file()
        outcomes = Counter()
        for folder in sorted((PUBLIC / EVAL / task / 'R000').glob('ep*')):
            result = read(folder / 'result.json')
            assert result['program_hash'] == task_row['program_hash'] and result['artifacts_complete']
            actions = np.load(folder / 'actions.npy', mmap_mode='r')
            with h5py.File(folder / 'observations.h5') as h5:
                n = len(h5['step'])
                assert n == result['video_frames']
                assert int(np.sum(h5['phase'][:] == 1)) == len(actions) + 1
                assert all(len(v) == n for v in h5.values())
                for camera in ('front', 'wrist'):
                    assert h5[camera + '_depth'].shape == (n, 256, 256)
                    assert h5[camera + '_depth'].dtype == np.int16
            assert sum(1 for _ in (folder / 'actions.jsonl').open()) == len(actions)
            assert (folder / 'episode.mp4').stat().st_size > 0
            assert (folder / 'decisions.jsonl').is_file()
            outcomes[result['status']] += 1
        assert sum(outcomes.values()) == 50 and outcomes['success'] == task_row['successes']
        rows.append({'task': task, 'demos': 25, 'test_episodes': 50, 'outcomes': dict(outcomes),
                     'rgbd_and_alignment_verified': True})
        print('verified RGB-D and 50 held-out rollouts:', task, flush=True)
    write(job / 'content_verification.json', {'tasks': rows, 'demos': 400, 'test_episodes': 800})


def finish(job, value, manifest):
    # Use copy receipts only for the unchanged, independently read destinations.
    known = {}
    for line in (job / 'copied_files.jsonl').read_text().splitlines():
        row = json.loads(line)
        known[row['destination']] = row
    hashes = []
    links = []
    for root in [PUBLIC / DATA, PUBLIC / DEV, PUBLIC / EVAL, PUBLIC / CODE, (PUBLIC / CATALOG).parent]:
        for parent, dirs, names in os.walk(root):
            parent = Path(parent)
            for name in dirs + names:
                path = parent / name
                rel = path.relative_to(PUBLIC).as_posix()
                if path.is_symlink():
                    assert path.resolve().is_relative_to(PUBLIC) and path.exists(), path
                    links.append({'path': rel, 'target': os.readlink(path)})
                elif path.is_file() and path.name not in {'SHA256SUMS', 'links.json'}:
                    old = known.get(rel)
                    if old:
                        assert (path.stat().st_size, path.stat().st_mtime_ns) == (old['bytes'], old['mtime_ns']), path
                        digest = old['sha256']
                    elif path.suffix == '.h5':
                        task = path.parent.name
                        canonical = value['tasks'][task]['demonstration_h5']
                        assert os.path.samefile(path, PUBLIC / canonical)
                        digest = known[canonical]['sha256']
                    else:
                        digest = sha256(path)
                    hashes.append((rel, digest))
    write(job / 'links.json', links)
    write(job / 'publication_verification.json', {'regular_files': len(hashes), 'internal_relative_links': len(links),
                                                'copied_files': len(known), 'copied_bytes': value['total_bytes']})
    # Preserve completed verification records in the public archive itself.
    provenance = PUBLIC / DEV / '_provenance/export'
    provenance.mkdir(parents=True, exist_ok=True)
    for name in ['plan.json', 'copied_files.jsonl', 'content_verification.json', 'solution_validation.json',
                 'publication_verification.json', 'links.json', 'organize_robomme_public.py']:
        shutil.copy2(job / name, provenance / name)
    manifest['status'] = 'complete'
    for category in (DEV, EVAL):
        write(PUBLIC / category / 'manifest.json', manifest)
    # Rehash generated manifests and add provenance; copied payload bytes are unchanged.
    indexed = dict(hashes)
    for category in (DEV, EVAL):
        name = (category / 'manifest.json').as_posix()
        indexed[name] = sha256(PUBLIC / name)
    for path in provenance.iterdir():
        indexed[path.relative_to(PUBLIC).as_posix()] = sha256(path)
    checksum = PUBLIC / 'ROBOMME_SHA256SUMS'
    checksum.write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(indexed.items())))
    receipt = {'schema': 'cam-robomme-release/1', 'status': 'complete', 'source_run': str(RUN),
               'experiment': ARM, 'tasks': 16, 'demonstrations': 400, 'test_episodes': 800,
               'successes': 795, 'success_rate': 795 / 800, 'training_attempts': 0,
               'copied_bytes': value['total_bytes'], 'copied_files': len(known),
               'source_preserved': True, 'source_stream_and_destination_sha256_verified': True,
               'solution_code_files': sum(len(v['code_sha256']) for v in read(PUBLIC / CATALOG)['tasks'].values()),
               'solution_validation': str(DEV / '_provenance/export/solution_validation.json'),
               'new_simulator_evaluation_performed': False,
               'checksums': checksum.name, 'checksums_sha256': sha256(checksum), 'job': str(job),
               'completed': datetime.now(timezone.utc).isoformat()}
    write(PUBLIC / 'ROBOMME_RELEASE.json', receipt)
    write(job / 'status.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def validate_solutions(job):
    from cam.harness.isolation import run
    from cam.harness.prepare import make_runtime
    from cam.core.history import materialize
    spec, tasks, imported = load_solutions(REPO / CODE)
    runtime = make_runtime(job / 'validation/runtime', demonstrations=False)
    rows = []
    for task in tasks:
        folder = job / 'validation/policies' / task
        materialize(folder, imported['files'][task])
        code, out, err = run(folder, runtime, ['python3.11', '-m', 'cam.runtime.policy_worker', '--validate'], readonly=True)
        (folder / 'validation.stdout').write_bytes(out)
        (folder / 'validation.stderr').write_bytes(err)
        if code:
            raise RuntimeError(f'Isolated policy validation failed: {task}: {err.decode()[-1000:]}')
        rows.append({'task': task, 'returncode': code, 'code_sha256': spec['tasks'][task]['code_sha256']})
    write(job / 'solution_validation.json', {'all_passed': True, 'policies': rows,
                                          'scope': 'Published loader, memory schema, code hashes, isolated interface/tool validation; no simulator or model calls.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    value = plan(job)
    print(f"Selected {len(value['files'])} files, {value['total_bytes'] / 2**30:.2f} GiB", flush=True)
    export_solutions(value)
    validate_solutions(job)
    copy_all(job, value, args.workers)
    manifest = indexes(job, value)
    verify_content(job, value)
    finish(job, value, manifest)


if __name__ == '__main__':
    main()
