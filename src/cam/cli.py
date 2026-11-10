"""Develop CaM policies from prepared demos, or evaluate published solutions."""

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
    download = sub.add_parser("download-demos", help="Download a published pool and its selected-25 assignment")
    download.add_argument("--benchmark", choices=("robomme", "mikasa_robo", "rmbench", "robomemarena"), required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--tasks", nargs="+")
    download.add_argument("--videos", action="store_true", help="Also download existing demonstration videos and posters")
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
    data.add_argument("--reuse-dataset", type=Path, help="RoboMemArena: copy verified existing RGB-D demos and reuse unchanged replay audits")
    data.add_argument('--max-successful', type=int, help='RoboMemArena: stop at this many successful identities per task; retain up to this many captures')
    data.add_argument('--replay-cache', type=Path, help='RoboMemArena: immutable source job containing completed/in-flight replay evidence')
    data.add_argument('--single-pass', action='store_true', help='RoboMemArena: capture sensors and score success in the same replay')
    data.add_argument('--validation-episodes', type=int, default=0, help='RoboMemArena single-pass: independently replay this many captured identities per task')
    subset = sub.add_parser("subset", help="Assign demos from a validated pool to a new standalone dataset")
    subset.add_argument("--dataset", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--count", type=int, default=25)
    subset.add_argument("--tasks", nargs="+")
    subset.add_argument("--allow-fewer", action="store_true", help="Record the actual count when fewer demos are available; never pad")
    subset.add_argument("--resume", action="store_true")
    subset.add_argument("--selection", type=Path, help="Exact published episode-and-seed selection JSON")
    subset.add_argument("--manifest-only", action="store_true", help="Trust the released pool manifest; validate only the newly copied subset")
    audit_data = sub.add_parser("validate-data", help="Optional full audit of a prepared dataset's H5s and checksums")
    audit_data.add_argument("--dataset", type=Path, required=True)
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
    preparation_source = prep.add_mutually_exclusive_group(required=True)
    preparation_source.add_argument("--dataset", type=Path)
    preparation_source.add_argument("--solution", type=Path,
                                    help="Published solution directory; prepare frozen evaluation without demos or learning")
    prep.add_argument("--config", type=Path,
                      help="Run config for a dataset, or an optional published-solution catalog override")
    prep.add_argument("--benchmark-source", type=Path)
    prep.add_argument("--tasks", nargs="+")
    prep.add_argument("--recording", choices=("full", "results"), help="Published evaluation: omit media/arrays with results")
    prep.add_argument("--validate-data", action="store_true",
                      help="Opt in to a full dataset audit; default uses the already validated release manifest")
    prep.add_argument("--initial-policy-run", type=Path,
                      help="Explicit repair: initialize one demo-only task from its frozen prior policy with identical config and demos")
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
        concurrency = job.add_mutually_exclusive_group()
        concurrency.add_argument("--workers-per-gpu", type=int, default=None if name == "learn" else 5)
        if name == "learn":
            concurrency.add_argument("--workers", type=int, default=8, help="Total coding jobs (default: 8)")
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
        if name == "preflight":
            job.add_argument("--full-data", action="store_true",
                             help="Also rehash demonstration files and simulator assets")
    site = sub.add_parser(
        "site", help="Build one website across new CaM runs and extracted datasets"
    )
    site.add_argument("--runs", type=Path, nargs="*", default=[])
    site.add_argument("--data-root", type=Path, help="Read a public_runs or validate_runs directory, including future runs")
    site.add_argument("--datasets", type=Path, nargs="*", default=[])
    site.add_argument("--output", type=Path, help="Defaults to DATA_ROOT/website when --data-root is supplied")
    site.add_argument("--batch", type=Path, help="Follow a development batch and add its runs as they become ready")
    site.add_argument("--lazy-rollouts", action="store_true", help="Build rollout traces and posters on demand when served")
    sanity = sub.add_parser("check-solutions", help="Run five complete episodes per published policy, saving logs and results")
    sanity.add_argument("--solutions", type=Path, default=Path("solutions"))
    sanity.add_argument("--output", type=Path, default=Path("validate_runs/eval_sanity_check"))
    sanity.add_argument("--benchmarks", nargs="+", help="Solution namespaces, including mikasa_robo_vla_suite")
    sanity.add_argument("--tasks", nargs="+")
    sanity.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    sanity.add_argument("--workers-per-gpu", type=int, default=10)
    sanity.add_argument("--min-free-gib", type=float, default=8)
    sanity.add_argument("--episodes", type=int, default=5)
    sanity.add_argument("--resume", type=Path, help="Resume an interrupted check batch; retain every completed outcome")
    sanity.add_argument("--retry-infrastructure", action="store_true", help="With --resume: archive unscorable failures and retry identical episodes")
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
    doctor.add_argument("--benchmark", choices=("robomme", "mikasa_robo", "rmbench", "robomemarena"), help="Check this simulator's dedicated environment")
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
    if args.command == "check-solutions":
        from cam.harness.sanity import check_solutions

        result = check_solutions(args.solutions, args.output, args.benchmarks, args.tasks,
                                 args.gpus, args.workers_per_gpu, args.min_free_gib, args.resume, args.episodes,
                                 retry_infrastructure=args.retry_infrastructure)
    elif args.command == "benchmarks":
        from cam.interface.registry import BENCHMARKS

        result = {name: entry for name, entry in BENCHMARKS.items()
                  if entry.get("status") != "compatibility_alias"}
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
            reuse=args.reuse_dataset,
            max_successful=args.max_successful,
            replay_cache=args.replay_cache,
            single_pass=args.single_pass,
            validation_episodes=args.validation_episodes,
        )
    elif args.command == "download-demos":
        from cam.data.download import download

        result = download(args.benchmark, args.output, args.tasks, args.videos)
    elif args.command == "subset":
        from cam.data.subset import subset

        result = subset(args.dataset, args.output, args.count, args.tasks, args.allow_fewer, args.resume,
                        selection=args.selection, validate_source=not args.manifest_only)
    elif args.command == "validate-data":
        from cam.data.extract import validate_dataset

        manifest = validate_dataset(args.dataset)
        result = {"benchmark": manifest["benchmark"], "tasks": len(manifest["tasks"]), "validated": True}
    elif args.command == "prepare":
        if args.solution:
            from cam.harness.solutions import prepare_solutions

            if args.initial_policy_run or args.validate_data:
                parser.error("--solution cannot be combined with --initial-policy-run or --validate-data")
            options = {"recording": args.recording} if args.recording else {}
            result = prepare_solutions(args.root, args.solution, args.tasks,
                                       args.benchmark_source, args.timetag, args.config, **options)
        else:
            if args.recording:
                parser.error("Use the recording field in the development config")
            from cam.harness.prepare import prepare_runs

            if args.config is None:
                parser.error("--dataset requires --config")
            result = prepare_runs(
                args.root, args.dataset, read(args.config), args.tasks,
                args.benchmark_source, args.timetag,
                initial_policy_run=args.initial_policy_run,
                validate_data=args.validate_data,
            )
    elif args.command in ("learn", "evaluate", "preflight", "_learn-task"):
        from cam.harness import jobs

        if args.command == "preflight":
            result = jobs.preflight(args.root, full_data=args.full_data)
        elif args.command == "_learn-task":
            result = jobs.learn_task(args.root, args.task, args.gpu, args.resume)
        elif args.command == "learn":
            result = jobs.learn(
                args.root, args.gpus, args.workers_per_gpu, args.tasks, args.resume, workers=args.workers
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
        if args.data_root:
            if args.runs or args.datasets or args.batch:
                parser.error("--data-root discovers runs and datasets; choose it or explicit source lists")
            from cam.reporting.dashboard import build_dashboard
            result = build_dashboard(args.data_root, args.output or args.data_root / "website")
        else:
            if args.output is None:
                parser.error("--output is required without --data-root")
            from cam.reporting.site import build_site
            result = build_site(args.runs, args.datasets, args.output, args.batch, lazy_rollouts=args.lazy_rollouts)
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

        result = doctor(args.simulator or bool(args.benchmark), args.benchmark)
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
