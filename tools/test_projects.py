"""Run independent subproject test suites without importing one tests package into another."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for name in ('PSO', 'GA'):
    project = root / name
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(project/'src'), str(project)))
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=project, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)
