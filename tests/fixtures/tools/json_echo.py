"""Deterministic subprocess fixture for JSON command-provider tests."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--ignore-term", action="store_true")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--pre-stderr-bytes", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--stderr", default="")
    parser.add_argument("--stderr-bytes", type=int, default=0)
    parser.add_argument("--stderr-invalid-utf8", action="store_true")
    parser.add_argument(
        "--stdout-mode",
        choices=(
            "echo",
            "array",
            "duplicate",
            "invalid-utf8",
            "nonfinite",
            "surrogate",
            "trailing",
        ),
        default="echo",
    )
    parser.add_argument("--stdout-bytes", type=int, default=0)
    parser.add_argument("--stdout-value-size", type=int, default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.ignore_term and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if args.pid_file is not None:
        args.pid_file.write_text(str(os.getpid()), encoding="ascii")

    if args.pre_stderr_bytes:
        sys.stderr.buffer.write(b"p" * args.pre_stderr_bytes)
        sys.stderr.buffer.flush()
    stdin = sys.stdin.buffer.read()
    if args.stderr:
        sys.stderr.buffer.write(args.stderr.encode("utf-8"))
    if args.stderr_bytes:
        sys.stderr.buffer.write(b"e" * args.stderr_bytes)
    if args.stderr_invalid_utf8:
        sys.stderr.buffer.write(b"\xff")
    sys.stderr.buffer.flush()

    if args.sleep:
        time.sleep(args.sleep)

    if args.stdout_value_size:
        stdout = b'{"value":"' + b"x" * args.stdout_value_size + b'"}'
    elif args.stdout_bytes:
        stdout = b"x" * args.stdout_bytes
    else:
        stdout = {
            "echo": stdin,
            "array": b"[]",
            "duplicate": b'{"value":1,"value":2}',
            "invalid-utf8": b"\xff",
            "nonfinite": b'{"value":NaN}',
            "surrogate": b'{"value":"\\ud800"}',
            "trailing": b'{"value":1} trailing',
        }[args.stdout_mode]
    sys.stdout.buffer.write(stdout)
    sys.stdout.buffer.flush()
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
