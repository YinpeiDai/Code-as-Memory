"""Promote a complete user-requested rerun without altering historical records."""

from copy import deepcopy
from pathlib import Path
import time
import urllib.request


def publish(job, run, summary):
    from cam.io import read, write

    job, run = Path(job), Path(run).resolve()
    plan = read(job / 'plan.json')
    manifest = read(run / 'run.json')
    task = plan['task']
    if (not manifest['fresh_policy'] or manifest.get('initial_policy') is not None
            or manifest['tasks'] != [task] or manifest['config']['mode'] != 'demos'
            or manifest['config']['budget'] != 0):
        raise ValueError('Only the requested fresh demo-only policy can become current')
    if (not summary['complete'] or summary['expected'] != 100 or summary['scored'] != 100
            or summary['infrastructure_episodes'] or Path(summary['run']).resolve() != run):
        raise ValueError('All 100 tests must be scored before replacing the default view')
    site = Path(plan['website'])
    sources = read(site / '.sources.json')
    prior_sources = deepcopy(sources)
    if str(run) not in sources['runs']:
        sources['runs'].append(str(run))
    for old in plan['replaces_display_runs']:
        if old not in sources['runs']:
            raise ValueError('Historical run must remain linked on the website')
    visibility_path = site / '.run_visibility.json'
    visibility = read(visibility_path) if visibility_path.exists() else {}
    previous_visibility = deepcopy(visibility)
    for old in plan['replaces_display_runs']:
        visibility[old] = {'replacement': str(run), 'updated': time.time(),
            'reason': 'User-requested from-scratch rerun with the revised shared demo-only prompt. Original development and scored outcomes are retained.'}
    write(job / 'website_sources_before_publication.json', prior_sources)
    write(job / 'website_visibility_before_publication.json', previous_visibility)
    # Keep indices stable: old direct development and rollout URLs still resolve.
    write(site / '.sources.json', sources)
    write(visibility_path, visibility)
    baseline = read(Path(plan['baseline_evaluation']) / 'results.json')
    current = deepcopy(baseline)
    current['tasks'][task] = summary
    current.update(kind='current_selection_with_user_requested_rerun',
                   replaced_task=task, original_results=str(Path(plan['baseline_evaluation']) / 'results.json'),
                   prompt_revision='Shared demo-only final offline code review, including None initialization',
                   successes=sum(row['successes'] for row in current['tasks'].values()),
                   scored=sum(row['scored'] for row in current['tasks'].values()))
    current['success_rate'] = current['successes'] / current['expected']
    write(job / 'current_results.json', current)
    receipt = {'run': str(run), 'retained_historical_runs': plan['replaces_display_runs'],
               'published': time.time(), 'selection_independent_of_success_rate': True}
    write(job / 'publication.json', receipt)
    # A browser refresh is enough to rebuild; request one now for unattended jobs.
    try:
        with urllib.request.urlopen('http://127.0.0.1:8915/data.json', timeout=10) as response:
            response.read()
    except OSError as exc:
        write(job / 'website_refresh_pending.json', {'error': str(exc), 'at': time.time()})
    return receipt
