"""Command-line entry point for the package."""

import argparse
import asyncio
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path
import yaml

from .configuration.loader import _UniqueKeySafeLoader

from . import __version__


def _expected_red_evaluations(task_path: Path) -> int:
    if not isinstance(task_path, Path) or task_path.is_symlink() or not task_path.is_file():
        raise ValueError("task must be an existing regular file")
    descriptor = os.open(
        task_path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise ValueError("task file exceeds guard byte limit")
        data = os.read(descriptor, 1024 * 1024 + 1)
    finally:
        os.close(descriptor)
    if len(data) > 1024 * 1024 or len(data) != metadata.st_size:
        raise ValueError("task file exceeds guard byte limit")
    raw = yaml.load(data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    try:
        population = raw["pso"]["population_size"]
        iterations = raw["pso"]["iterations"]
    except (KeyError, TypeError) as error:
        raise ValueError("task PSO budget is missing") from error
    if type(population) is not int or type(iterations) is not int:
        raise ValueError("task PSO budget must use integers")
    return population * iterations


def _load_verified_red_preflight(task: Path, inputs: Path, runs_dir: Path):
    from examples.red_absorption.preflight import load_verified_red_absorption_preflight

    return load_verified_red_absorption_preflight(task, inputs, runs_dir)


def _launch_red_absorption_search(task, inputs, record, runs_dir: Path):
    from examples.red_absorption.search import run_red_absorption_search

    return asyncio.run(
        run_red_absorption_search(task, inputs, record, runs_dir=runs_dir)
    )


def _execute_red_preflight(task: Path, inputs: Path, runs_dir: Path):
    from examples.red_absorption.preflight import preflight_red_absorption

    return asyncio.run(preflight_red_absorption(task, inputs, runs_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-agent PSO orchestration")
    parser.add_argument("--version", action="store_true")
    subcommands = parser.add_subparsers(dest="command")
    benchmark = subcommands.add_parser("benchmark")
    benchmark.add_argument("name", choices=("sphere", "rastrigin"))
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--runs-dir", type=Path, required=True)
    benchmark.add_argument("--particles", type=int, default=5)
    benchmark.add_argument("--iterations", type=int, default=2)
    benchmark.add_argument("--dimension", type=int, default=3)
    report = subcommands.add_parser("report")
    report.add_argument("--runs-dir", type=Path, required=True)
    selection = report.add_mutually_exclusive_group(required=True)
    selection.add_argument("--latest", action="store_true")
    selection.add_argument("--run-id")
    report.add_argument("--format", choices=("json", "markdown"), default="json")
    preflight = subcommands.add_parser("preflight")
    preflight.add_argument("task", type=Path)
    preflight.add_argument("--inputs", type=Path, required=True)
    preflight.add_argument("--runs-dir", type=Path, required=True)
    run = subcommands.add_parser("run")
    run.add_argument("task", type=Path)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--runs-dir", type=Path, required=True)
    run.add_argument("--confirm-max-new-evaluations", type=int)
    for name in ("resume", "status"):
        subcommands.add_parser(name)
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2
    if args.command == "benchmark":
        try:
            from .benchmarks import run_continuous_benchmark

            result = run_continuous_benchmark(
                args.name,
                args.seed,
                args.runs_dir,
                particles=args.particles,
                iterations=args.iterations,
                dimension=args.dimension,
            )
            print(json.dumps(result.summary, sort_keys=True, separators=(",", ":")))
            return 0
        except (ValueError, RuntimeError, sqlite3.Error, OSError) as error:
            print(f"benchmark error: {error}", file=sys.stderr)
            return 2
    if args.command == "report":
        try:
            root = args.runs_dir.resolve(strict=True)
            database = root / "runs.sqlite"
            if (
                root.is_symlink()
                or not root.is_dir()
                or database.is_symlink()
                or not database.is_file()
            ):
                raise ValueError("runs-dir must contain a regular runs.sqlite")
            from .reporting import build_run_report_from_store, publish_run_report
            from .storage import FileArtifactStore, SQLiteRunStore

            store = SQLiteRunStore(database)
            report_value = build_run_report_from_store(
                store, args.run_id, latest=args.latest
            )
            reference = publish_run_report(
                report_value, FileArtifactStore(root / "artifacts"), args.format
            )
            print(
                json.dumps(
                    {
                        "artifact": reference.model_dump(mode="json"),
                        "summary": {
                            "run_id": report_value.run_id,
                            "run_status": report_value.run_status,
                            "iterations": len(report_value.iterations),
                            "final_claim": report_value.final_claim,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            return 0
        except (ValueError, TypeError, RuntimeError, sqlite3.Error, OSError) as error:
            print(f"report error: {error}", file=sys.stderr)
            return 2
    if args.command == "preflight":
        try:
            record = _execute_red_preflight(args.task, args.inputs, args.runs_dir)
            if record.passed is not True:
                raise ValueError("preflight record did not pass")
            print(
                json.dumps(
                    {
                        "identity": record.identity,
                        "artifact_path": record.artifact_relative_path,
                        "authentication_method": record.authentication_method,
                        "parent_hashes": {
                            "state": record.parent_state_hash,
                            "chemical": record.parent_chemical_hash,
                            "geometry": record.parent_geometry_hash,
                        },
                        "protocol": {
                            "functional": record.protocol_functional,
                            "basis": record.protocol_basis,
                            "method": record.protocol_method,
                            "backend": record.protocol_backend,
                            "backend_version": record.protocol_backend_version,
                            "hardware": record.backend_hardware,
                            "geometry_workflow": record.geometry_workflow,
                        },
                        "evaluation_concurrency": record.evaluation_concurrency,
                        "spectrum_timeout_seconds": record.spectrum_timeout_seconds,
                        "max_new_evaluations": record.max_new_evaluations,
                        "estimated_total_spectrum_calculations": 1
                        + record.max_new_evaluations,
                        "passed": record.passed,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            print(f"preflight error: {error}", file=sys.stderr)
            return 2
    if args.command == "run":
        try:
            expected = _expected_red_evaluations(args.task)
            if args.confirm_max_new_evaluations != expected:
                raise ValueError(
                    f"explicit confirmation required: --confirm-max-new-evaluations {expected}"
                )
            task, loaded, record = _load_verified_red_preflight(
                args.task, args.inputs, args.runs_dir
            )
            print(f"Launching with exact upper bound: {expected} new evaluations")
            result = _launch_red_absorption_search(
                task, loaded, record, args.runs_dir
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 0
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            print(f"run error: {error}", file=sys.stderr)
            return 2
    if args.command in {"resume", "status"}:
        print(f"{args.command} is not implemented in Stage A", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
