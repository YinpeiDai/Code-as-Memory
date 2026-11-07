"""Reproducible CaM workflow from raw data to frozen-program evaluation."""

import argparse
import json
from pathlib import Path
import sys

from cam.io import read


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="cam", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sub.add_parser(
        "benchmarks", help="Show implemented and planned benchmark protocols"
    )
    data = sub.add_parser(
        "extract", help="Validate raw training H5 and extract assigned demonstrations"
    )
    data.add_argument("--raw-data", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--benchmark", default="robomme")
    data.add_argument("--benchmark-source", type=Path)
    data.add_argument("--count", default="25")
    data.add_argument("--tasks", nargs="+")
    data.add_argument("--rgb-only", action="store_true")
    data.add_argument("--no-videos", action="store_true")
    data.add_argument("--workers", type=int, default=2)
    data.add_argument("--resume", action="store_true")
    prep = sub.add_parser(
        "prepare",
        help="Create task runs in benchmark/task/client-model/setting/timetag folders",
    )
    prep.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("runs"),
        help="Base runs directory (default: runs)",
    )
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--benchmark-source", type=Path)
    prep.add_argument("--tasks", nargs="+")
    prep.add_argument(
        "--timetag", help="Explicit leaf tag; defaults to a unique UTC timestamp"
    )
    for name in ("learn", "evaluate"):
        job = sub.add_parser(
            name,
            help="Develop task programs"
            if name == "learn"
            else "Evaluate frozen checkpoints",
        )
        job.add_argument("root", type=Path)
        job.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
        job.add_argument(
            "--workers-per-gpu", type=int, default=1 if name == "learn" else 8
        )
        job.add_argument("--tasks", nargs="+")
        job.add_argument("--resume", action="store_true")
        if name == "evaluate":
            job.add_argument("--budgets", nargs="+", type=int)
    for name, description in (
        (
            "preflight",
            "Check prepared-run isolation and observation transport without model calls",
        ),
        ("report", "Write aggregate development, cost and evaluation reports"),
        (
            "status",
            "Show development progress, accumulated usage and evaluation counts",
        ),
    ):
        job = sub.add_parser(name, help=description)
        job.add_argument("root", type=Path)
    site = sub.add_parser(
        "site", help="Build one website across new CaM runs and extracted datasets"
    )
    site.add_argument("--runs", type=Path, nargs="*", default=[])
    site.add_argument("--datasets", type=Path, nargs="*", default=[])
    site.add_argument("--output", type=Path, required=True)
    serve = sub.add_parser(
        "serve", help="Serve an exported CaM site, including video byte ranges"
    )
    serve.add_argument("site", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8910)
    figures = sub.add_parser(
        "figures", help="Export exact complete-evaluation curves as SVG, PNG and PDF"
    )
    figures.add_argument("--runs", type=Path, nargs="+", required=True)
    figures.add_argument("--output", type=Path, required=True)
    doctor = sub.add_parser(
        "doctor", help="Check local dependencies without model calls"
    )
    doctor.add_argument("--simulator", action="store_true")
    recover = sub.add_parser(
        "recover-test",
        help="Archive an infrastructure failure before a deterministic retry",
    )
    recover.add_argument("root", type=Path)
    recover.add_argument("task")
    recover.add_argument("episode", type=int)
    recover.add_argument("--budget", type=int, required=True)
    recover.add_argument("--reason", required=True)
    recover_train = sub.add_parser(
        "recover-learning", help="Close an interrupted attempt without refunding it"
    )
    recover_train.add_argument("root", type=Path)
    recover_train.add_argument("task")
    recover_train.add_argument("--reason", required=True)
    for name in ("_service", "_learn-task", "_episode"):
        job = sub.add_parser(name, help=argparse.SUPPRESS)
        job.add_argument("root", type=Path)
        job.add_argument("task")
        if name == "_episode":
            job.add_argument("episode", type=int)
            job.add_argument("--split", choices=["train", "test"], required=True)
            job.add_argument("--budget", type=int, default=0)
        else:
            job.add_argument("--gpu", type=int, required=True)
        if name == "_learn-task":
            job.add_argument("--resume", action="store_true")
    sub._choices_actions = [
        item for item in sub._choices_actions if not item.dest.startswith("_")
    ]
    args = parser.parse_args(argv)
    if args.command in ("_learn-task", "_episode"):
        from cam.harness.isolation import parent_death

        parent_death()
    if args.command in (
        "learn",
        "evaluate",
        "preflight",
        "_service",
        "_learn-task",
        "_episode",
        "recover-test",
        "recover-learning",
    ):
        from cam.harness.prepare import use_pinned_cli

        use_pinned_cli(args.root, argv)
    if args.command == "benchmarks":
        from cam.benchmarks.registry import BENCHMARKS

        result = BENCHMARKS
    elif args.command == "extract":
        from cam.data.extract import extract

        result = extract(
            args.raw_data,
            args.output,
            args.benchmark,
            args.count,
            args.tasks,
            not args.rgb_only,
            not args.no_videos,
            args.workers,
            args.benchmark_source,
            args.resume,
        )
    elif args.command == "prepare":
        from cam.harness.prepare import prepare_runs

        result = prepare_runs(
            args.root,
            args.dataset,
            read(args.config),
            args.tasks,
            args.benchmark_source,
            args.timetag,
        )
    elif args.command in ("learn", "evaluate", "preflight", "_learn-task"):
        from cam.harness import jobs

        if args.command == "preflight":
            result = jobs.preflight(args.root)
        elif args.command == "_learn-task":
            result = jobs.learn_task(args.root, args.task, args.gpu, args.resume)
        elif args.command == "learn":
            result = jobs.learn(
                args.root, args.gpus, args.workers_per_gpu, args.tasks, args.resume
            )
        else:
            result = jobs.evaluate(
                args.root,
                args.budgets,
                args.gpus,
                args.workers_per_gpu,
                args.tasks,
                args.resume,
            )
    elif args.command == "_service":
        from cam.harness.service import main as service_main

        sys.argv = ["cam-service", str(args.root), args.task, "--gpu", str(args.gpu)]
        service_main()
        return 0
    elif args.command == "_episode":
        from cam.harness.broker import run_episode

        result = run_episode(
            args.root, args.task, args.episode, args.split, args.budget
        )
    elif args.command in ("report", "status"):
        from cam.reporting.report import build

        result = build(args.root)
        if args.command == "status":
            result = {
                "run": result["name"],
                "mode": result["config"]["mode"],
                "development": {
                    row["task"]: {
                        "status": row["status"],
                        "attempts": len(row["attempts"]),
                    }
                    for row in result["development"]
                },
                "cost": result["cost"],
                "evaluation": [
                    {
                        k: s[k]
                        for k in (
                            "budget",
                            "scored",
                            "expected",
                            "successes",
                            "success_rate",
                            "observed_success_rate",
                            "infrastructure_errors",
                        )
                    }
                    for s in result["settings"]
                ],
            }
    elif args.command == "site":
        from cam.reporting.site import build_site

        result = build_site(args.runs, args.datasets, args.output)
    elif args.command == "serve":
        from cam.reporting.site import serve

        serve(args.site, args.host, args.port)
        return 0
    elif args.command == "figures":
        from cam.reporting.figures import export_figures
        from cam.reporting.report import build

        result = export_figures([build(root) for root in args.runs], args.output)
    elif args.command == "doctor":
        from cam.harness.doctor import doctor

        result = doctor(args.simulator)
    else:
        from cam.harness.recover import recover_test, recover_learning

        result = (
            recover_test(args.root, args.task, args.episode, args.budget, args.reason)
            if args.command == "recover-test"
            else recover_learning(args.root, args.task, args.reason)
        )
    print(json.dumps(result, indent=2))
    return int(
        result.get("status") in ("failed", "needs_attention")
        or result.get("scorable") is False
        or result.get("ready") is False
    )


if __name__ == "__main__":
    raise SystemExit(main())
