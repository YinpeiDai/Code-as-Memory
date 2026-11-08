"""Queue a matched 25-demo RGB-D comparison after the 50-demo run finishes.

Pin the baseline's harness, simulator source, config and deterministic subset
identities now. Start fresh learners only after all baseline development and
all 1,000 scored tests finish. The independent evaluation scheduler follows
each new frozen policy. Original runs, captures and test outcomes stay intact.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_ready(development, evaluation, results, tasks):
    """Gate on complete coverage, never on the baseline's success rate."""
    if development['status'] == 'needs_attention' or evaluation['status'] == 'needs_attention':
        raise ValueError('Baseline needs attention; comparison remains unstarted')
    if development['status'] != 'finished' or evaluation['status'] != 'finished':
        return False
    rows = list(development['tasks'].values())
    if (len(rows) != len(tasks) or {r['task'] for r in rows} != set(tasks)
            or any(r['status'] != 'finished' for r in rows)):
        raise ValueError('Baseline development task coverage is incomplete')
    if (not results or not results.get('complete') or results.get('unresolved_infrastructure') != 0
            or results.get('scored') != len(tasks) * 100 or results.get('expected') != len(tasks) * 100
            or set(results['tasks']) != set(tasks)):
        raise ValueError('Baseline must have every assigned test scored')
    for task in tasks:
        row = results['tasks'][task]
        if not row['complete'] or row['scored'] != 100 or row['expected'] != 100 or row['infrastructure_episodes']:
            raise ValueError(f'Baseline test coverage incomplete: {task}')
    return True


def verify_inputs(job, plan):
    from cam.io import sha256, tree_hash

    for path, digest in plan['baseline_files'].items():
        if sha256(path) != digest:
            raise ValueError(f'Baseline input changed: {path}')
    for name, digest in plan['pinned_files'].items():
        if sha256(job / name) != digest:
            raise ValueError(f'Comparison input changed: {name}')
    if tree_hash(job / 'inputs/checkout') != plan['checkout_hash']:
        raise ValueError('Pinned baseline checkout changed')


def start(args):
    from cam.benchmarks.base import source_hash
    from cam.benchmarks.registry import get_adapter
    from cam.data.subset import choose
    from cam.io import copy_tree, read, sha256, tree_hash, write

    baseline = args.baseline.resolve()
    old = read(baseline / 'plan.json')
    prior_eval = Path(read(baseline / 'evaluation_followup.json')['job']).resolve()
    tasks = old['tasks']
    if tasks != get_adapter('rmbench').tasks():
        raise ValueError('Expected all ten RMBench tasks')
    config = read(baseline / 'config.json')
    if (config['benchmark'], config['mode'], config['budget'], config['depth']) != ('rmbench', 'demos', 0, True):
        raise ValueError('Expected the RGB-D demo-only baseline')
    with (baseline / '.demos25_followup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (baseline / 'demos25_followup.json').exists():
            raise FileExistsError('A 25-demo follow-up is already recorded for this baseline')
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        job = (ROOT / 'outputs/rmbench_comparison_queue' / tag).resolve()
        job.mkdir(parents=True)
        copy_tree(baseline / 'inputs/checkout', job / 'inputs/checkout')
        for source, destination in [('config.json', 'config.json'), ('runner.py', 'runner.py'),
                                    ('boundary_checks.py', 'develop_batch.py'), ('audit_run.py', 'audit_run.py')]:
            shutil.copy2(baseline / source, job / destination)
        shutil.copy2(Path(__file__), job / 'driver.py')
        shutil.copy2(prior_eval / 'evaluation_checks.py', job / 'evaluation_checks.py')
        if tree_hash(job / 'inputs/checkout/src/cam') != old['harness_hash']:
            raise ValueError('Baseline harness hash changed')
        if source_hash(get_adapter('rmbench', job / 'inputs/checkout/third_party/rmbench')) != old['benchmark_hash']:
            raise ValueError('Baseline benchmark hash changed')
        sources, selections = {}, {}
        baseline_files = {str(baseline / name): sha256(baseline / name)
                          for name in ('plan.json', 'experiment.json', 'config.json')}
        baseline_files[str(prior_eval / 'plan.json')] = sha256(prior_eval / 'plan.json')
        for task in tasks:
            source = baseline / 'datasets' / task
            manifest = read(source / 'manifest.json')
            receipt = read(source / 'ready.json')
            if not receipt['ready'] or receipt['count'] != old['demo_counts'][task]:
                raise ValueError(f'Baseline data not ready: {task}')
            sources[task] = str(source)
            baseline_files[str(source / 'manifest.json')] = sha256(source / 'manifest.json')
            selected = choose(manifest, task, 25)
            if len(selected) != 25:
                raise ValueError(f'Need exactly 25 distinct demos: {task}')
            selections[task] = [{'episode': r['episode'], 'seed': r['seed']} for r in selected]
        write(job / 'selection.json', selections)
        spec = read(baseline / 'experiment.json')
        spec['selection_reason'] = ('User-requested matched 25-demo RGB-D demo-only comparison across all ten RMBench tasks. '
            'Deterministic subsets of the prior 48/49/50-demo assignments; same pinned harness, simulator, configuration, '
            'model and shared prompts, zero training rollouts, fresh policies. No prior solutions or test feedback enter learning.')
        item = spec['benchmarks']['rmbench']
        item.pop('demo_counts', None)
        item.update(config=str(job / 'config.json'), demo_count=25,
                    datasets=[str(job / 'datasets' / t) for t in tasks])
        write(job / 'experiment.json', spec)
        plan = {'created': time.time(), 'root': str(ROOT), 'baseline': str(baseline),
                'baseline_evaluation': str(prior_eval), 'tasks': tasks, 'workers': 8, 'gpus': [0, 1],
                'python': sys.executable, 'sources': sources, 'demo_count': 25,
                'baseline_demo_counts': old['demo_counts'], 'environment': old['environment'],
                'harness_hash': old['harness_hash'], 'benchmark_hash': old['benchmark_hash'],
                'checkout_hash': tree_hash(job / 'inputs/checkout'), 'baseline_files': baseline_files,
                'pinned_files': {name: sha256(job / name) for name in ('config.json', 'experiment.json',
                    'selection.json', 'driver.py', 'develop_batch.py', 'audit_run.py', 'runner.py', 'evaluation_checks.py')}}
        write(job / 'plan.json', plan)
        write(job / 'status.json', {'status': 'waiting_for_baseline', 'updated': time.time()})
        (job / 'README.md').write_text(
            f'# RMBench 25-versus-50-demo comparison: {tag}\n\n'
            f'Baseline development: `{baseline}`. Baseline evaluation: `{prior_eval}`.\n\n'
            'Wait for all ten baseline learners to finish and all 1,000 baseline tests to be scored. '
            'Success rate does not affect admission or demo selection. Pin exactly 25 deterministic '
            'training identities per task in selection.json (250 total), drawn from the baseline\'s '
            '496 successful RGB-D demos. Use cam subset to create self-contained H5 files on local SSD '
            'under datasets/TASK; validate hashes, schemas and train/test identity separation. Original data stay unchanged.\n\n'
            'Reuse the identical pinned baseline harness, simulator code, config, Claude Code / Opus 5.5 '
            'and shared demo-only prompts. Eight coding slots, zero training rollouts, fresh policies '
            'and cam prepare timetags. Learners see only their 25-demo H5 and their own workspace. '
            'They cannot access the larger pool, prior solutions or test feedback.\n\n'
            'batch.json links the new development batch after baseline completion and subset validation. '
            'evaluation_followup.json links the independent per-task frozen-policy evaluator: '
            '100 identical held-out episodes per task, one task per GPU, at most six episode workers/GPU, '
            'infrastructure-only archival/retries. comparison.json and comparison.md report both arms '
            'after complete scored coverage. This is one development run per task/condition, not replicated trials.\n')
        with (job / 'supervisor.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, str(job / 'driver.py'), '_worker', str(job)],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        result = {'job': str(job), 'pid': proc.pid, 'demos': 250, 'expected_tests': 1000}
        write(job / 'process.json', {'pid': proc.pid, 'started': time.time()})
        write(baseline / 'demos25_followup.json', result)
        evaluator = load(ROOT / 'scripts/queue_rmbench_evaluation.py', 'comparison_evaluator')
        evaluator.ROOT = ROOT
        result['evaluation'] = evaluator.start(argparse.Namespace(
            development=job, gpus=[0, 1], workers_per_gpu=6, as_ready=True, supersedes=None))
        write(baseline / 'demos25_followup.json', result)
        return result


def command(job, plan, label, argv):
    from cam.io import write

    stem = job / 'logs' / f'{label}_{time.time_ns()}'
    write(stem.with_suffix('.command.json'), {'argv': argv, 'started': time.time()})
    with stem.with_suffix('.log').open('xb') as log:
        proc = subprocess.Popen(argv, cwd=job, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, **plan['environment'], 'PYTHONDONTWRITEBYTECODE': '1',
                                     'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        write(stem.with_suffix('.process.json'), {'pid': proc.pid, 'started': time.time()})
        code = proc.wait()
    write(stem.with_suffix('.completion.json'), {'returncode': code, 'finished': time.time()})
    if code:
        raise RuntimeError(f'Command exited {code}: {stem.with_suffix(".log")}')


def launch_batch(job, plan):
    from cam.io import copy_tree, read, tree_hash, write

    launcher = load(job / 'develop_batch.py', 'comparison_batch')
    launcher.ROOT = ROOT
    spec = read(job / 'experiment.json')
    item = spec['benchmarks']['rmbench']
    assigned = launcher.data_assignment('rmbench', item, read(job / 'config.json'))
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    batch = (ROOT / 'outputs/development_batches' / tag).resolve()
    batch.mkdir(parents=True)
    copy_tree(job / 'inputs/checkout', batch / 'inputs/checkout')
    if tree_hash(batch / 'inputs/checkout') != plan['checkout_hash']:
        raise ValueError('Comparison batch differs from pinned baseline source')
    for source, dest in [('develop_batch.py', 'driver.py'), ('audit_run.py', 'audit_run.py'),
                         ('runner.py', 'runner.py'), ('config.json', 'config.json')]:
        shutil.copy2(job / source, batch / dest)
    cfg = {'tasks': plan['tasks'], **assigned, 'interaction_count': 0, 'raw': item['raw'],
           'source': str(batch / 'inputs/checkout/third_party/rmbench'),
           'source_hash': plan['benchmark_hash'], 'asset_hashes': {}, 'config': str(batch / 'config.json'),
           'training_seed_policy': 'demo_cycle', 'new_data': str(job / 'unused_extraction'), 'depth': True}
    write(batch / 'plan.json', {'created': time.time(), 'tag': tag, 'workers': 8, 'gpus': [0, 1],
        'selection_reason': spec['selection_reason'], 'benchmarks': {'rmbench': cfg},
        'environment': plan['environment'], 'python': plan['python'], 'runs': str((ROOT / 'runs').resolve()),
        'harness_hash': plan['harness_hash'], 'held_out_evaluation': False, 'comparison_queue': str(job)})
    write(batch / 'status.json', {'status': 'running', 'workers': 8, 'tasks': {
        f'rmbench/{t}': {'task': t, 'benchmark': 'rmbench', 'status': 'waiting_data'} for t in plan['tasks']}})
    (batch / 'README.md').write_text((job / 'README.md').read_text() +
        f'\nDevelopment batch: `{batch}`. Each task receives a fresh cam prepare run under timetag `{tag}`. '
        'plan.json records all exact settings, local subsets and source hashes.\n')
    with (batch / 'supervisor.log').open('ab') as log:
        proc = subprocess.Popen([plan['python'], str(batch / 'driver.py'), '_worker', str(batch)],
                                cwd=batch, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    write(batch / 'process.json', {'pid': proc.pid, 'started': time.time()})
    write(job / 'batch.json', {'batch': str(batch), 'pid': proc.pid})
    return batch


def comparison(baseline, new, counts):
    if not baseline['complete'] or not new['complete'] or set(baseline['tasks']) != set(new['tasks']):
        raise ValueError('Comparison requires complete scored coverage in both arms')
    rows = {}
    for task, old in baseline['tasks'].items():
        current = new['tasks'][task]
        rows[task] = {'baseline_demos': counts[task], 'comparison_demos': 25,
                      'baseline_successes': old['successes'], 'comparison_successes': current['successes'],
                      'baseline_scored': old['scored'], 'comparison_scored': current['scored'],
                      'baseline_run': old['run'], 'comparison_run': current['run'],
                      'difference_percentage_points': 100 * (current['tsr'] - old['tsr'])}
    return {'tasks': rows, 'baseline_scored': baseline['scored'], 'comparison_scored': new['scored'],
            'baseline_success_rate': baseline['success_rate'], 'comparison_success_rate': new['success_rate'],
            'complete': True, 'replicates_per_condition': 1}


def worker(job):
    from cam.io import read, sha256, write

    plan = read(job / 'plan.json')
    baseline, prior_eval = Path(plan['baseline']), Path(plan['baseline_evaluation'])
    checks = load(job / 'evaluation_checks.py', 'comparison_checks')
    with (job / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_inputs(job, plan)
        while True:
            dev, ev = read(baseline / 'status.json'), read(prior_eval / 'status.json')
            results = read(prior_eval / 'results.json') if (prior_eval / 'results.json').exists() else None
            if baseline_ready(dev, ev, results, plan['tasks']):
                break
            for folder, state in [(baseline, dev), (prior_eval, ev)]:
                if state['status'] != 'finished' and not checks.alive(read(folder / 'process.json')['pid']):
                    raise RuntimeError(f'Baseline supervisor stopped: {folder}')
            write(job / 'status.json', {'status': 'waiting_for_baseline', 'updated': time.time(),
                'baseline_finished_tasks': sum(r['status'] == 'finished' for r in dev['tasks'].values()),
                'baseline_scored_tests': (results or {}).get('scored', 0), 'baseline_expected_tests': 1000})
            time.sleep(30)
        verify_inputs(job, plan)
        frozen = {r['task']: checks.frozen_policy(Path(r['run']), r['task']) for r in dev['tasks'].values()}
        write(job / 'baseline_gate.json', {'passed': True, 'at': time.time(), 'frozen': frozen,
                                          'scored': 1000, 'results_sha256': sha256(prior_eval / 'results.json')})
        write(job / 'status.json', {'status': 'preparing_subsets', 'updated': time.time()})
        selections = read(job / 'selection.json')

        def make_subset(task):
            output = job / 'datasets' / task
            command(job, plan, f'subset_{task}', [plan['python'], str(job / 'runner.py'), 'subset',
                '--dataset', plan['sources'][task], '--output', str(output), '--count', '25', '--tasks', task])
            subset = read(output / 'manifest.json')
            actual = [{'episode': r['episode'], 'seed': r['seed']} for r in subset['tasks'][task]['episodes']]
            if actual != selections[task]:
                raise ValueError(f'Subset differs from precommitted identities: {task}')
            write(output / 'ready.json', {'ready': True, 'count': 25,
                'h5_sha256': subset['tasks'][task]['sha256'], 'matches_precommitted_selection': True})

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(make_subset, plan['tasks']))
        verify_inputs(job, plan)
        batch = launch_batch(job, plan)
        while True:
            state = read(batch / 'status.json')
            followup = job / 'evaluation_followup.json'
            evaluation = Path(read(followup)['job']) if followup.exists() else None
            ev = read(evaluation / 'status.json') if evaluation else {}
            if state['status'] == 'needs_attention' or ev.get('status') == 'needs_attention':
                raise RuntimeError('Comparison development or evaluation needs attention; preserve evidence')
            if state['status'] == 'finished' and ev.get('status') == 'finished':
                break
            if state['status'] != 'finished' and not checks.alive(read(batch / 'process.json')['pid']):
                raise RuntimeError('Comparison development supervisor stopped')
            write(job / 'status.json', {'status': 'following_comparison', 'batch': str(batch),
                'development_status': state['status'], 'evaluation_status': ev.get('status', 'not_queued'),
                'updated': time.time()})
            time.sleep(30)
        result = comparison(read(prior_eval / 'results.json'), read(evaluation / 'results.json'), plan['baseline_demo_counts'])
        write(job / 'comparison.json', result)
        lines = ['# RMBench RGB-D demo-count comparison', '',
                 '| Task | Baseline demos | Baseline success | 25-demo success | Change (pp) |',
                 '|---|---:|---:|---:|---:|']
        for task, row in result['tasks'].items():
            lines.append(f"| {task} | {row['baseline_demos']} | {row['baseline_successes']}/100 | "
                         f"{row['comparison_successes']}/100 | {row['difference_percentage_points']:+.1f} |")
        lines += ['', f"Overall: baseline {result['baseline_success_rate']:.1%}; 25 demos {result['comparison_success_rate']:.1%}.",
                  '1,000 scored test episodes per condition. One development run per task/condition; no replicated-trial uncertainty estimate.']
        (job / 'comparison.md').write_text('\n'.join(lines) + '\n')
        write(job / 'status.json', {'status': 'finished', 'batch': str(batch), 'updated': time.time(), 'comparison': str(job / 'comparison.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('start').add_argument('baseline', type=Path)
    sub.add_parser('_worker').add_argument('job', type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.command == 'start':
        sys.path.insert(0, str(args.baseline.resolve() / 'inputs/checkout/src'))
        print(json.dumps(start(args), indent=2))
    else:
        ROOT = Path(json.loads((args.job / 'plan.json').read_text())['root'])
        sys.path.insert(0, str(args.job / 'inputs/checkout/src'))
        try:
            worker(args.job.resolve())
        except Exception:
            from cam.io import write
            write(args.job / 'status.json', {'status': 'needs_attention', 'updated': time.time(), 'error': traceback.format_exc()})
            raise
