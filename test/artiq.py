import typing
import sys
import os
import os.path
import contextlib
import subprocess

import dax.util.output

__all__ = ['master', 'client_submit']

_DEVICE_DB: typing.Dict[str, typing.Any] = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },
}


@contextlib.contextmanager
def master(device_db: typing.Optional[typing.Dict[str, typing.Any]] = None,
           localhost: str = '127.0.0.1') -> typing.Generator[typing.Tuple[str, subprocess.Popen], None, None]:
    """Context manager to start an ARTIQ master process in a temp directory."""
    assert isinstance(device_db, dict) or device_db is None
    assert isinstance(localhost, str)

    with dax.util.output.temp_dir() as tmp_dir:
        # Create a device DB file
        device_db_file_name = os.path.join(tmp_dir, 'device_db.py')
        with open(device_db_file_name, 'w') as device_db_file:
            device_db_file.write('device_db=')
            device_db_file.write(str(_DEVICE_DB if device_db is None else device_db))

        # Create repository directory
        os.mkdir(os.path.join(tmp_dir, 'repository'))

        # Start ARTIQ master
        cmd = [sys.executable, '-u', '-m', 'artiq.frontend.artiq_master', '--no-localhost-bind', '--bind', localhost]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, bufsize=1) as process:
            try:
                assert process.stdout is not None
                for line in iter(process.stdout.readline, ''):
                    if line.rstrip() == 'ARTIQ master is now ready.':
                        break
                else:
                    raise Exception('ARTIQ master failed to start')

                # Return temp dir and process object
                yield tmp_dir, process

            finally:
                # Terminate process
                process.terminate()


def client_submit(exp_file: str, *args: str,
                  localhost: str = '127.0.0.1', repository: bool = False, class_: typing.Optional[str] = None):
    """Submit an experiment using the ARTIQ client."""
    assert isinstance(exp_file, str)
    assert all(isinstance(a, str) for a in args)
    assert isinstance(localhost, str)
    assert isinstance(repository, bool)

    # Construct command
    cmd = [sys.executable, '-u', '-m', 'artiq.frontend.artiq_client', '-s', localhost, 'submit']
    if repository:
        cmd.append('-R')
    if class_:
        cmd.extend(['-c', class_])
    cmd.append(exp_file)
    cmd.extend(args)

    # Run ARTIQ client
    result = subprocess.run(cmd, capture_output=True, universal_newlines=True)
    if result.returncode != 0:
        raise Exception(f'ARTIQ client submit exited with return code {result.returncode}: "{result.stderr}"')

    # Return RID
    return int(result.stdout.strip().split(' ')[1])
