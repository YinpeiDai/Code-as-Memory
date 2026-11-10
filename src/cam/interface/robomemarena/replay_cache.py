"""Host-only reuse of completed and in-flight jobs from an immutable preparation."""
import time
from pathlib import Path

from cam.io import read, sha256, tree_hash


def process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else {'pid': pid, 'start_ticks': fields[19]}
    except FileNotFoundError:
        return None


def open_cache(source, manifest):
    source = Path(source).resolve()
    plan = read(source / 'plan.json')
    pool = Path(plan['pool'])
    previous = read(pool / ('manifest.json' if (pool / 'manifest.json').exists() else 'manifest.partial.json'))
    for key in ('benchmark', 'benchmark_hash', 'raw_data', 'depth', 'fields', 'release'):
        if previous[key] != manifest[key]:
            raise ValueError(f'Replay cache protocol differs: {key}')
    if tree_hash(source / 'inputs/checkout/src/cam') != plan['harness_hash']:
        raise ValueError('Replay cache source harness changed')
    # Replay commands must originate in this checked source snapshot.
    from cam.interface.base import source_hash
    from cam.interface.registry import get_adapter
    if source_hash(get_adapter('robomemarena', plan['benchmark_source'])) != manifest['benchmark_hash']:
        raise ValueError('Replay cache benchmark changed')
    jobs = {task: [] for task in manifest['requested_tasks']}
    for path in sorted((pool / 'work').glob('score_*.job.json')):
        job = read(path)
        if job['task'] not in jobs:
            continue
        if job['source'] != plan['benchmark_source'] or job['mode'] != 'score' or job['depth']:
            raise ValueError(f'Replay cache job protocol differs: {path}')
        if Path(job['root']) != Path(plan['raw']) / 'hf':
            raise ValueError(f'Replay cache raw source differs: {path}')
        dispatch = source / 'replay_archive' / path.name.removesuffix('.job.json') / 'dispatch.json'
        result = Path(job['result'])
        info = read(dispatch) if dispatch.exists() else None
        producer = process_identity(info['native_pid']) if info and 'finished' not in info else None
        wrapper = process_identity(info['dispatcher_pid']) if info and 'finished' not in info else None
        if not result.exists() and not producer and not wrapper:
            continue  # No completed evidence or active attempt to adopt.
        jobs[job['task']].append({'job': str(path), 'job_sha256': sha256(path),
            'result': str(result), 'dispatch': str(dispatch), 'producer': producer, 'wrapper': wrapper,
            'result_sha256': sha256(result) if result.exists() else None,
            'identities': job['identities']})
    return jobs, {'source_job': str(source), 'source_plan_sha256': sha256(source / 'plan.json'),
                  'source_harness_hash': plan['harness_hash'], 'source_manifest_sha256': sha256(
                      pool / ('manifest.json' if (pool / 'manifest.json').exists() else 'manifest.partial.json')),
                  'jobs': jobs}


def collect(entries, records):
    """Await only inherited jobs for this task; other tasks can run concurrently."""
    expected = {int(row['episode']): int(row['seed']) for row in records}
    outcomes, provenance = {}, []
    for item in entries:
        path = Path(item['job'])
        if sha256(path) != item['job_sha256']:
            raise ValueError(f'Replay cache job changed: {path}')
        identities = {int(r['episode']): int(r['seed']) for r in item['identities']}
        if any(expected.get(ep) != seed for ep, seed in identities.items()):
            raise ValueError('Replay cache contains an unassigned training identity')
        result, dispatch = Path(item['result']), Path(item['dispatch'])
        while True:
            state = read(dispatch) if dispatch.exists() else {}
            if 'finished' in state:
                if state.get('returncode') != 0:
                    raise RuntimeError(f'Inherited replay worker failed: {dispatch}')
                break
            if result.exists() and not item['producer'] and not item['wrapper']:
                break
            if not any(p and process_identity(p['pid']) == p for p in (item['producer'], item['wrapper'])):
                # Finished evidence may have been committed just after the last read.
                state = read(dispatch) if dispatch.exists() else {}
                if state.get('returncode') == 0 and result.exists():
                    break
                raise RuntimeError(f'Inherited replay ended without completed evidence: {path}')
            time.sleep(2)
        digest = sha256(result)
        if item['result_sha256'] and digest != item['result_sha256']:
            raise ValueError(f'Replay cache result changed: {result}')
        rows = read(result)['results']
        if len(rows) != len(identities) or {int(r['episode']): int(r['seed']) for r in rows} != identities:
            raise ValueError('Replay cache result identities differ from the native job')
        for row in rows:
            ep = int(row['episode'])
            if ep in outcomes and outcomes[ep] != row:
                raise ValueError('Conflicting inherited replay outcomes')
            outcomes[ep] = row
        provenance.append({'job': str(path), 'job_sha256': item['job_sha256'],
                           'result': str(result), 'result_sha256': digest})
    return outcomes, provenance
