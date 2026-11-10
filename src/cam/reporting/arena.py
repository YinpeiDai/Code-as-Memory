"""Public comparison of explicitly selected, frozen RoboMemArena experiments."""

from collections import Counter
from pathlib import Path


def compact_archive(content):
    """Keep gallery metadata small; complete records remain at their artifact URLs."""
    episode_keys = {'task', 'episode', 'status', 'success', 'scorable', 'steps', 'decisions',
                    'goal', 'video', 'poster', 'trace', 'observations', 'actions', 'decisions_log',
                    'result_file', 'policy_stderr', 'attempt', 'maximum_rollouts'}
    for run in content['runs']:
        run['episodes'] = [{k: v for k, v in e.items() if k in episode_keys} for e in run['episodes']]
        for dev in run['development']:
            dev.pop('attempts', None)
            dev['frozen'] = bool(dev.get('frozen'))
            dev['rollouts'] = [{k: v for k, v in e.items() if k in episode_keys} for e in dev.get('rollouts', [])]
    content['paired_comparisons'] = []
    content['figures'] = []


def summarize_archive(content, presentation, sources):
    assignments = {str(Path(p).resolve()): v for p, v in presentation['run_settings'].items()}
    experiments = {e['id']: {**e, 'tasks': {}, 'demos': 0, 'training_attempts': 0} for e in presentation['experiments']}
    seen = set()
    for run in content['runs']:
        source = str(Path(sources[int(run['id'])]).resolve())
        experiment = experiments[assignments[source]]
        if run['benchmark'] != 'robomemarena':
            raise ValueError('The RoboMemArena archive must contain only its selected benchmark')
        run['experiment'] = experiment['id']
        run['display_name'] = experiment['label']
        budget = run['config']['budget']
        for dev in run['development']:
            task = dev['task']
            if (experiment['id'], task) in seen:
                raise ValueError(f'Duplicate selected task: {experiment["id"]}/{task}')
            seen.add((experiment['id'], task))
            setting = next((s for s in run['settings'] if s['budget'] == budget), None)
            expected = setting['tasks'][task]['expected'] if setting else run['expected_episodes_per_setting']
            episodes = [e for e in run['episodes'] if e['task'] == task and e['maximum_rollouts'] == budget]
            scored = [e for e in episodes if e.get('scorable')]
            successes = sum(bool(e['success']) for e in scored)
            complete = len(scored) == expected and len(episodes) == expected
            experiment['tasks'][task] = {
                'run_id': run['id'], 'expected': expected, 'recorded': len(episodes),
                'scored': len(scored), 'successes': successes, 'failures': len(scored) - successes,
                'infrastructure': len(episodes) - len(scored), 'complete': complete,
                'tsr': successes / expected if complete and expected else None,
                'checkpoint': budget, 'demos': dev['demonstrations'],
                'training_attempts': len(dev.get('rollouts', [])),
                'failure_statuses': dict(Counter(e['status'] for e in scored if not e['success'])),
            }
            experiment['demos'] += dev['demonstrations']
            experiment['training_attempts'] += len(dev.get('rollouts', []))
    for experiment in experiments.values():
        tasks = list(experiment['tasks'].values())
        for key in ['expected', 'recorded', 'scored', 'successes', 'failures', 'infrastructure']:
            experiment[key] = sum(t[key] for t in tasks)
        experiment['complete'] = bool(tasks) and all(t['complete'] for t in tasks)
        experiment['tsr'] = experiment['successes'] / experiment['expected'] if experiment['complete'] else None
        experiment['perfect_tasks'] = sum(t['complete'] and t['successes'] == t['expected'] for t in tasks)
        experiment['failure_statuses'] = dict(sum((Counter(t['failure_statuses']) for t in tasks), Counter()))
    return {
        'kind': 'robomemarena_archive', 'title': presentation.get('title', 'RoboMemArena'),
        'experiments': list(experiments.values()),
        'tasks': sorted({t for e in experiments.values() for t in e['tasks']}),
        'comparison_note': presentation.get('comparison_note', ''),
    }
