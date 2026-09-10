"""Generate explicit one-shot launchd jobs; never use launchctl submit."""

from __future__ import annotations

import argparse
from pathlib import Path
import plistlib
import re


def write_oneshot_plist(
    destination: Path, *, label: str, argv: list[str], cwd: Path,
    environment: dict[str, str], stdout: Path, stderr: Path,
) -> None:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', label):
        raise ValueError('invalid launchd label')
    if not argv or any(not isinstance(arg, str) or '\0' in arg for arg in argv):
        raise ValueError('argv must contain NUL-free strings')
    if not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
        raise ValueError('program must be an existing absolute file')
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ValueError('cwd must be an existing absolute directory')
    if any(not path.is_absolute() for path in (destination, stdout, stderr)):
        raise ValueError('plist and log paths must be absolute')
    if any(not isinstance(k, str) or not isinstance(v, str)
           for k, v in environment.items()):
        raise ValueError('environment must contain strings')
    spec = {
        'Label': label,
        'ProgramArguments': argv,
        'WorkingDirectory': str(cwd),
        'EnvironmentVariables': environment,
        'StandardOutPath': str(stdout),
        'StandardErrorPath': str(stderr),
        'RunAtLoad': True,
        'KeepAlive': False,
    }
    data = plistlib.dumps(spec, fmt=plistlib.FMT_XML)
    with destination.open('xb') as handle:
        handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--stdout', type=Path, required=True)
    parser.add_argument('--stderr', type=Path, required=True)
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    environment = {}
    for item in args.env:
        key, sep, value = item.partition('=')
        if not sep or not key or key in environment:
            parser.error('each --env must have a unique KEY=VALUE')
        environment[key] = value
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    write_oneshot_plist(args.output, label=args.label, argv=argv, cwd=args.cwd,
                       environment=environment, stdout=args.stdout, stderr=args.stderr)
    print(args.output)


if __name__ == '__main__':
    main()
