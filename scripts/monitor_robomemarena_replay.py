"""Read-only progress reporting for combined RoboMemArena RGB-D collection."""
import html
import json
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def alive(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def snapshot(job):
    plan = read(job / 'plan.json')
    pool = Path(plan['pool'])
    dataset = Path(plan.get('assembly_output', pool))
    source = Path(plan['replay_cache_job']) if plan.get('replay_cache_job') else None
    previous_pool = Path(read(source / 'plan.json')['pool']) if source else None
    reuse = read(Path(plan['reused_dataset']) / 'manifest.json') if plan.get('reused_dataset') else {'tasks': {}}
    tasks, active, errors = [], [], []
    for task in plan['tasks']:
        old = previous_pool / 'audit' / f'{task}.replay.json' if previous_pool else None
        historical = read(old)['results'] if old and old.exists() else []
        if not historical and task in reuse['tasks']:
            historical = reuse['tasks'][task]['replay']['identities']
        receipts = [read(p) for p in (pool / 'captures' / task).glob('episode_*.json')]
        successes = sum(r['status'] == 'success' and r['tsr'] for r in receipts)
        failed = len(receipts) - successes
        reused = len(reuse['tasks'].get(task, {}).get('episodes', []))
        audit = dataset / 'audit' / f'{task}.json'
        validation_file = dataset / 'audit' / f'{task}.validation.json'
        validation = read(validation_file) if validation_file.exists() else None
        validation_passed = read(audit).get('validation', {}).get('checked', 0) if audit.exists() else 0
        if validation and validation.get('passed'):
            validation_passed = len(validation['results'])
        error_file = dataset / 'audit' / f'{task}.error.json'
        if error_file.exists():
            errors.append(read(error_file))
        if validation and len(validation['results']) and not audit.exists():
            if any(r['status'] != 'success' or not r['tsr'] for r in validation['results']):
                errors.append({'task': task, 'error': 'Sampled replay validation failed; evidence retained'})
        historical_successes = sum(r['status'] == 'success' for r in historical)
        target = min(plan['pool_cap'], historical_successes) if len(historical) == 100 else plan['pool_cap']
        tasks.append({'task': task, 'historical_checked': len(historical),
            'historical_successes': historical_successes, 'reused_rgbd': reused,
            'new_successful_captures': successes, 'capture_failures': failed,
            'available_rgbd': reused + successes, 'planned_target': target,
            'task_finalized': audit.exists(), 'validation_passed': validation_passed})
    archives = Path(plan.get('replay_archive_job', job)) / 'replay_archive'
    for p in sorted(archives.glob('*/dispatch.json')):
        d = read(p)
        entry = p.parent.name
        mode, rest = entry.split('_', 1)
        task = rest.rsplit('_', 1)[0]
        if 'finished' not in d:
            active.append({'task': task, 'mode': mode, 'gpu': d['gpu'], 'pid': d['native_pid'],
                           'alive': alive(d['native_pid']), 'minutes': round((time.time()-d['started'])/60, 1)})
        elif d['returncode']:
            errors.append({'job': entry, 'returncode': d['returncode']})
    waiting = sum(not (archives / p.name.removesuffix('.job.json') / 'dispatch.json').exists()
                  and not Path(read(p)['result']).exists() for p in (pool / 'work').glob('*.job.json'))
    state = read(job / 'status.json')
    if active:
        collection_phase = 'capturing RGB-D and checking success; sampled validation runs as tasks finish'
    elif not (pool / 'manifest.partial.json').exists() and not (pool / 'manifest.json').exists():
        collection_phase = 'checking reusable RGB-D file integrity'
    elif not list((pool / 'audit').glob('*.capture_plan.json')):
        collection_phase = 'checking the downloaded file inventory and cached outcomes'
    else:
        collection_phase = 'assembling or validating recordings; see pipeline status'
    return {'updated_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'job': str(job), 'pool': str(pool), 'dataset': str(dataset), 'phase': state.get('phase'), 'status': state.get('status'),
        'learning_demo_cap': plan.get('demo_count', 25), 'direct_pool': plan.get('direct_pool', False),
        'collection_phase': collection_phase,
        'supervisor_alive': alive(read(job / 'process.json')['pid']),
        'single_pass': plan['single_pass'], 'workers_per_gpu': plan['replay_workers_per_gpu'],
        'validation_episodes_per_task': plan['validation_episodes'],
        'existing_rgbd': sum(r['reused_rgbd'] for r in tasks),
        'new_successful_captures': sum(r['new_successful_captures'] for r in tasks),
        'new_capture_failures': sum(r['capture_failures'] for r in tasks),
        'available_rgbd': sum(r['available_rgbd'] for r in tasks),
        'planned_pool_target': sum(r['planned_target'] for r in tasks),
        'tasks_finalized': sum(r['task_finalized'] for r in tasks),
        'independent_validations_passed': sum(r['validation_passed'] for r in tasks),
        'final_pool_manifest': (dataset / 'manifest.json').exists(),
        'waiting_for_admission': waiting, 'active': active, 'tasks': tasks, 'errors': errors}


def page(s):
    assignment = (f"up to {s['learning_demo_cap']} demos/task directly from the capped dataset, with actual shortfalls reported and no smaller subset"
                  if s['direct_pool'] else f"exactly {s['learning_demo_cap']} demos/task")
    rows = ''.join(f"<tr><td>{html.escape(t['task'])}</td><td>{t['historical_successes']}/{t['historical_checked']}</td>"
        f"<td>{t['reused_rgbd']}</td><td>{t['new_successful_captures']}</td><td>{t['capture_failures']}</td>"
        f"<td>{t['available_rgbd']}/{t['planned_target']}</td><td>{t['validation_passed']}/{s['validation_episodes_per_task']}</td></tr>" for t in s['tasks'])
    active = ''.join(f"<li>GPU {w['gpu']}: {html.escape(w['task'])} · {w['mode']} · {w['minutes']} min</li>" for w in s['active'])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>RoboMemArena RGB-D collection</title>
<style>body{{font:15px system-ui;background:#11151b;color:#eef2f7;margin:28px auto;max-width:1200px;padding:0 16px}}p,li{{line-height:1.6}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #354052}}th{{color:#b4c8df}}code{{overflow-wrap:anywhere}}</style>
<h1>RoboMemArena RGB-D collection</h1><p>Updated {s['updated_utc']}. Capture and official success checking happen in the same replay.</p>
<p><strong>{s['available_rgbd']} RGB-D demos available: {s['existing_rgbd']} reused + {s['new_successful_captures']} new successful captures.</strong><br>
Up to {s['planned_pool_target']} planned, subject to replay shortfalls. {s['new_capture_failures']} new unsuccessful captures are retained privately.<br>
{s['tasks_finalized']}/26 tasks finalized. {s['independent_validations_passed']} independent validation episodes passed; only five per task.<br>
Four native workers per GPU, at least 16 GiB free before admission. Waiting jobs: {s['waiting_for_admission']}.<br>Currently: {html.escape(s['collection_phase'])}.</p>
<p>Historical score-only successes below are not additional saved RGB-D demos. Interrupted old batches remain unscored. Learning uses {assignment} and zero training rollouts, followed by 50 held-out tests/task.</p>
<ul>{active}</ul><p>Errors: {html.escape(json.dumps(s['errors']))}</p>
<table><tr><th>Task</th><th>Historical success/checked</th><th>Reused RGB-D</th><th>New RGB-D</th><th>Failed captures</th><th>RGB-D / target</th><th>Validation</th></tr>{rows}</table>
<p>Capture pool: <code>{html.escape(s['pool'])}</code><br>Final learning dataset: <code>{html.escape(s['dataset'])}</code></p></html>'''


def main():
    job, site = [Path(p).resolve() for p in sys.argv[1:]]
    folder = job / 'monitor'
    folder.mkdir(exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            s = snapshot(job)
            for path, content in [(folder/'status.json', json.dumps(s, indent=2)),
                                  (site/'status.json', json.dumps(s, indent=2)), (site/'index.html', page(s))]:
                tmp = path.with_suffix(path.suffix+'.tmp')
                tmp.write_text(content+'\n')
                tmp.replace(path)
            with (folder/'history.jsonl').open('a') as log:
                log.write(json.dumps(s)+'\n')
            if not s['supervisor_alive']:
                return
        except Exception as exc:
            with (folder/'errors.jsonl').open('a') as log:
                log.write(json.dumps({'time': time.time(), 'error': str(exc)})+'\n')
        time.sleep(60)


if __name__ == '__main__':
    main()
