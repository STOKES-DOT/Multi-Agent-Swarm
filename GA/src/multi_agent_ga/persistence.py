"""Bounded, digest-verified atomic records and exclusive run ownership."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

MAX_RECORD_BYTES = 16 * 1024 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read_record(path: Path):
    with path.open('rb') as handle:
        raw = handle.read(MAX_RECORD_BYTES + 1)
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError('record exceeds byte limit')
    try:
        document = json.loads(raw, object_pairs_hook=_pairs)
        if set(document) != {'sha256', 'payload'} or digest(document['payload']) != document['sha256']:
            raise ValueError('record digest mismatch')
        return document['payload']
    except (RecursionError, TypeError, KeyError) as error:
        raise ValueError('invalid record') from error


def publish(path: Path, payload):
    data = canonical({'sha256': digest(payload), 'payload': payload}) + b'\n'
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError('record exceeds byte limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_record(path) != payload:
            raise ValueError('immutable record conflict')
        return
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.controller.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another GA controller owns this run') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
