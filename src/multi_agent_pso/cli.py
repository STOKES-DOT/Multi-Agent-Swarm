"""Command-line entry point for the package."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from . import __version__


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
    for name in ("run", "resume", "status"):
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
            result = run_continuous_benchmark(args.name, args.seed, args.runs_dir, particles=args.particles, iterations=args.iterations, dimension=args.dimension)
            print(json.dumps(result.summary, sort_keys=True, separators=(",", ":")))
            return 0
        except (ValueError, RuntimeError, sqlite3.Error, OSError) as error:
            print(f"benchmark error: {error}", file=sys.stderr)
            return 2
    if args.command in {"run", "resume", "status"}:
        print(f"{args.command} is not implemented in Stage A", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
