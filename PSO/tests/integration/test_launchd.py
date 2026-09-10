import plistlib
import sys

from multi_agent_pso.launchd import write_oneshot_plist


def test_oneshot_job_preserves_arguments_and_never_restarts(tmp_path):
    cwd = tmp_path / 'workspace with spaces'
    cwd.mkdir()
    destination = tmp_path / 'job.plist'
    argv = [sys.executable, '-c', 'print("finished")', 'literal $HOME']
    write_oneshot_plist(destination, label='test.pso.oneshot', argv=argv,
                       cwd=cwd, environment={'PATH': '/usr/bin:/bin'},
                       stdout=tmp_path / 'stdout.log', stderr=tmp_path / 'stderr.log')
    spec = plistlib.loads(destination.read_bytes())
    assert spec['RunAtLoad'] is True
    assert spec['KeepAlive'] is False
    assert spec['ProgramArguments'] == argv
    assert spec['WorkingDirectory'] == str(cwd)
    assert spec['EnvironmentVariables'] == {'PATH': '/usr/bin:/bin'}
    assert not {'StartInterval', 'StartCalendarInterval', 'WatchPaths', 'QueueDirectories'} & spec.keys()
    import pytest
    with pytest.raises(FileExistsError):
        write_oneshot_plist(destination, label='test.pso.oneshot', argv=argv,
                           cwd=cwd, environment={}, stdout=tmp_path/'o', stderr=tmp_path/'e')
