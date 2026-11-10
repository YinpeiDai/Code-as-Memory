"""All-task solution evaluation shares one concurrency limit and resumes evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cam.io import read, write


@pytest.mark.parametrize('suite', [None, 'mikasa_robo_vla_suite'])
def test_complete_setting_runs_every_task_and_resumes_without_repreparation(tmp_path, monkeypatch, suite):
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_solutions.py'
    spec = importlib.util.spec_from_file_location('evaluate_solutions', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = [{'task': task, 'root': str(tmp_path / 'runs' / task)} for task in ('BinFill', 'InsertPeg', 'VideoUnmask')]
    benchmark = 'mikasa_robo' if suite else 'robomme'
    monkeypatch.setattr(module, 'prepare_solutions', lambda *a, **kw: {'runs': runs, 'benchmark': benchmark, 'suite': suite})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    args = SimpleNamespace(solution=tmp_path / 'relocated/demo_only_25', output=tmp_path / 'output',
                           tasks=None, gpus=[0,1], workers_per_gpu=5, resume=None)
    result = module.evaluate(args)
    assert result['status'] == 'needs_attention' and len(calls) == 2
    batch = Path(result['batch'])
    plan = read(batch / 'plan.json')
    assert list(plan['benchmarks']) == [suite or benchmark]
    assert plan['simulator_benchmark'] == benchmark and plan['suite'] == suite
    assert read(batch / 'status.json')['tasks']['BinFill']['status'] == 'finished'
    def forbid(*a, **kw): raise AssertionError('Resume must retain the original preparations')
    monkeypatch.setattr(module, 'prepare_solutions', forbid)
    args.resume = batch
    write(Path(runs[1]['root']) / 'queues/evaluation/status.json', {'status':'needs_attention'})
    result = module.evaluate(args)
    assert result['status'] == 'finished' and len(calls) == 4
    assert '--resume' in calls[2]
    assert all(cmd[-1] == '5' or '--resume' in cmd for cmd in calls)
    assert all(row['status'] == 'finished' for row in read(batch / 'status.json')['tasks'].values())
    assert 'Status: finished' in (batch / 'PROGRESS.md').read_text()
