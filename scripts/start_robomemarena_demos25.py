"""Prepare 25-demo RGB-D subsets and launch a separate all-task development batch."""
import argparse
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


def start(baseline):
    import cam
    from cam.interface.base import pin_source, source_hash
    from cam.interface.registry import get_adapter
    from cam.data.subset import choose
    from cam.harness.prepare import simulator_build_files
    from cam.io import copy_tree, read, sha256, tree_hash, write

    baseline = baseline.resolve()
    with (baseline/'.demos25_followup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (baseline/'demos25_followup.json').exists():
            raise FileExistsError('This baseline already has a 25-demo follow-up')
        old = read(baseline/'plan.json')
        item = old['benchmarks']['robomemarena']
        tasks = item['tasks']
        if len(item['existing']) != 1 or item['missing'] or len(tasks) != 26:
            raise ValueError('Expected the completed all-task RoboMemArena pool')
        source = Path(next(iter(item['existing'])))
        pool = read(source/'manifest.json')
        config = read(item['config'])
        if (config['mode'], config['budget'], config['depth']) != ('demos', 0, True):
            raise ValueError('Expected RGB-D demo-only configuration')
        adapter = get_adapter('robomemarena')
        if tasks != adapter.tasks() or source_hash(adapter) != item['source_hash']:
            raise ValueError('Benchmark or task assignment changed')
        selections = {task: [{'episode': r['episode'], 'seed': r['seed']}
                            for r in choose(pool, task, 25)] for task in tasks}
        if any(len(rows) != 25 or len({r['episode'] for r in rows}) != 25 for rows in selections.values()):
            raise ValueError('Need exactly 25 distinct demos per task')
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        job = (ROOT/'outputs/robomemarena_demos25'/tag).resolve()
        job.mkdir(parents=True)
        checkout = job/'inputs/checkout'
        copy_tree(Path(cam.__file__).parent, checkout/'src/cam')
        for name in ('pyproject.toml', 'uv.lock'):
            shutil.copy2(ROOT/name, checkout/name)
        pin_source(adapter, checkout/'third_party/robomemarena/evaluation_benchmark')
        for recipe in simulator_build_files(adapter):
            target = checkout/'envs/robomemarena'/recipe.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(recipe, target)
        for source_file, destination in ((Path(__file__), 'driver.py'), (ROOT/'scripts/develop_batch.py', 'develop_batch.py')):
            shutil.copy2(source_file, job/destination)
        (job/'runner.py').write_text(
            "from pathlib import Path\nimport sys\n"
            "sys.path.insert(0, str(Path(__file__).parent/'inputs/checkout/src'))\n"
            "from cam.cli import main\nraise SystemExit(main())\n")
        destination = Path('/data/daiyp/CaM/datasets/robomemarena_rgbd')/tag/'demos25'
        write(job/'config.json', config)
        write(job/'selection.json', selections)
        prompt_comparison = {name: {
            'baseline_sha256': sha256(baseline/'inputs/checkout/src/cam'/name),
            'followup_sha256': sha256(checkout/'src/cam'/name)}
            for name in ('prompts/core.md', 'prompts/modes/demos.md', 'benchmarks/robomemarena.md')}
        write(job/'prompt_comparison.json', prompt_comparison)
        experiment = read(ROOT/'configs/robomemarena/experiments/demo_only_25.json')
        experiment['benchmarks']['robomemarena'].update(config=str(job/'config.json'), datasets=[str(destination)])
        write(job/'experiment.json', experiment)
        plan = {'created': time.time(), 'repository': str(ROOT), 'baseline': str(baseline),
            'baseline_plan_sha256': sha256(baseline/'plan.json'), 'source': str(source),
            'source_manifest_sha256': sha256(source/'manifest.json'), 'destination': str(destination),
            'tasks': tasks, 'demo_count': 25, 'workers': 8, 'gpus': [1], 'python': sys.executable,
            'environment': old['environment'], 'harness_hash': tree_hash(checkout/'src/cam'),
            'baseline_demo_counts': {t: len(pool['tasks'][t]['episodes']) for t in tasks},
            'blank_initial_policy': True, 'held_out_evaluation': False,
            'pinned_files': {n: sha256(job/n) for n in ('driver.py', 'develop_batch.py', 'runner.py',
                'config.json', 'experiment.json', 'selection.json', 'prompt_comparison.json')}}
        write(job/'plan.json', plan)
        (job/'README.md').write_text(
            '# RoboMemArena: separate 25-demo development\n\n'
            f'Baseline batch: `{baseline}`. Source RGB-D pool: `{source}`. '
            f'New self-contained local dataset: `{destination}`.\n\n'
            'Exactly 25 demos per task, 26 tasks / 650 demos, deterministic scene-variant '
            'selection recorded before copying in selection.json. Use cam subset and retain '
            'schema, identity and hash checks. Originals stay unchanged.\n\n'
            'Claude Code / Opus 5.5, eight coding slots, zero training rollouts, fresh blank '
            'policies and cam prepare runs. Learners see only their 25-demo H5; no prior '
            'solutions, larger pools or held-out feedback. Current 50-demo evaluation continues. '
            'This launcher starts development only.\n\n'
            'The current shared demo-only prompt includes the authorized final offline review '
            'and first-decision execution checks added after baseline development. See '
            'prompt_comparison.json for exact hashes; this is not a demo-count-only ablation. '
            'Historical prompts and policies remain unchanged. batch.json records the new '
            'development batch after the subset is validated.\n')
        write(job/'status.json', {'status': 'running', 'phase': 'starting_subset', 'updated': time.time()})
        with (job/'supervisor.log').open('ab') as log:
            proc = subprocess.Popen([sys.executable, str(job/'driver.py'), '_worker', str(job)],
                cwd=job, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = {'job': str(job), 'pid': proc.pid, 'demos': 650, 'tasks': 26}
        write(job/'process.json', {'pid': proc.pid, 'started': time.time()})
        write(baseline/'demos25_followup.json', result)
        return result


def worker(job):
    from cam.io import read, sha256, tree_hash, write
    plan = read(job/'plan.json')
    with (job/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert tree_hash(job/'inputs/checkout/src/cam') == plan['harness_hash']
        for name, digest in plan['pinned_files'].items():
            assert sha256(job/name) == digest, name
        assert sha256(Path(plan['baseline'])/'plan.json') == plan['baseline_plan_sha256']
        assert sha256(Path(plan['source'])/'manifest.json') == plan['source_manifest_sha256']
        os.environ.update(plan['environment'])
        destination = Path(plan['destination'])
        argv = [plan['python'], str(job/'runner.py'), 'subset', '--dataset', plan['source'],
                '--output', str(destination), '--count', '25', '--tasks', *plan['tasks']]
        write(job/'subset.command.json', {'argv': argv, 'started': time.time()})
        write(job/'status.json', {'status': 'running', 'phase': 'creating_subset', 'updated': time.time()})
        with (job/'subset.log').open('xb') as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            write(job/'subset.process.json', {'pid': proc.pid, 'started': time.time()})
            code = proc.wait()
        write(job/'subset.completion.json', {'returncode': code, 'finished': time.time()})
        if code:
            raise RuntimeError('cam subset failed; inspect subset.log')
        manifest = read(destination/'manifest.json')
        actual = {task: [{'episode': r['episode'], 'seed': r['seed']}
                         for r in manifest['tasks'][task]['episodes']] for task in plan['tasks']}
        if actual != read(job/'selection.json') or (destination/'manifest.partial.json').exists():
            raise ValueError('Final subset differs from the pinned selection or is incomplete')
        write(job/'subset_ready.json', {'tasks': 26, 'demos': 650, 'manifest_sha256': sha256(destination/'manifest.json'),
            'h5_bytes': sum((destination/v['file']).stat().st_size for v in manifest['tasks'].values())})
        spec = importlib.util.spec_from_file_location('demos25_batch', job/'develop_batch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = Path(plan['repository'])
        result = module.start(argparse.Namespace(plan=job/'experiment.json', workers=8, gpus=[1],
            prepare_workers=1, evaluate_on_freeze=False, evaluation_workers_per_gpu=5))
        write(job/'batch.json', result)
        batch = Path(result['batch'])
        assert read(batch/'plan.json')['harness_hash'] == plan['harness_hash']
        write(batch/'demos25_provenance.json', {'preparation': str(job), 'baseline': plan['baseline'],
            'selection_sha256': sha256(job/'selection.json'), 'prompt_comparison': read(job/'prompt_comparison.json')})
        with (batch/'README.md').open('a') as handle:
            handle.write(f'\nSubset and comparison provenance: `{job}`. The revised shared demo-only prompt differs from the 50-demo baseline; see `demos25_provenance.json`.\n')
        write(job/'status.json', {'status': 'launched', 'phase': 'development', **result, 'updated': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('start').add_argument('baseline', type=Path)
    commands.add_parser('_worker').add_argument('job', type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.command == 'start':
        sys.path.insert(0, str(ROOT/'src'))
        print(json.dumps(start(args.baseline), indent=2))
    else:
        job = args.job.resolve()
        sys.path.insert(0, str(job/'inputs/checkout/src'))
        try:
            worker(job)
        except Exception:
            from cam.io import write
            write(job/'status.json', {'status': 'needs_attention', 'error': traceback.format_exc(), 'updated': time.time()})
            raise
