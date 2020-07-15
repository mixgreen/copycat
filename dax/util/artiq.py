from __future__ import annotations  # Postponed evaluation of annotations

import os
import tempfile
import typing

import artiq.language.environment
import artiq.master.worker_db
import artiq.master.databases
import artiq.frontend.artiq_run  # type: ignore

__all__ = ['is_kernel', 'is_portable', 'is_host_only',
           'get_manager_or_parent']


def is_kernel(func: typing.Any) -> bool:
    """Helper function to detect if a function is an ARTIQ kernel (`@kernel`) or not.

    :param func: The function of interest
    :return: True if the given function is a kernel
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else (meta.core_name is not None and not meta.portable)


def is_portable(func: typing.Any) -> bool:
    """Helper function to detect if a function is an ARTIQ portable function (`@portable`) or not.

    :param func: The function of interest
    :return: True if the given function is a portable function
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else bool(meta.portable)


def is_host_only(func: typing.Any) -> bool:
    """Helper function to detect if a function is marked as host only (`@host_only`) or not.

    :param func: The function of interest
    :return: True if the given function is host only
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else bool(meta.forbidden)


def get_manager_or_parent(device_db: typing.Union[typing.Dict[str, typing.Any], str, None] = None,
                          expid: typing.Optional[typing.Dict[str, typing.Any]] = None,
                          **arguments: typing.Any) -> typing.Any:
    """Returns an object that can function as a `manager_or_parent` for ARTIQ HasEnvironment.

    This function is primarily used for testing purposes.

    If a full ARTIQ environment is not required but only a core device driver is sufficient,
    please take a look at the `dax.sim.coredevice.core.BaseCore` class.

    :param device_db: A device DB as dict or a file name (optional)
    :param expid: Dict for the scheduler expid attribute (optional)
    :param arguments: Arguments for the ProcessArgumentManager object
    :return: A dummy ARTIQ manager object
    """
    assert isinstance(expid, dict) or expid is None
    assert isinstance(arguments, dict)

    # Scheduler
    scheduler = artiq.frontend.artiq_run.DummyScheduler()
    # Construct expid of scheduler and add default values
    scheduler.expid = {} if expid is None else expid
    for k, v in _EXPID_DEFAULTS.items():
        scheduler.expid.setdefault(k, v)
    # Set arguments (overwrites any arguments in the expid)
    scheduler.expid['arguments'] = arguments

    # Create a unique temp dir
    tempdir = _TemporaryDirectory()

    if isinstance(device_db, dict) or device_db is None:
        # Create a temporally device DB file
        device_db_file_name = os.path.join(tempdir.name, 'device_db.py')
        with open(device_db_file_name, 'w') as device_db_file:
            device_db_file.write('device_db=')
            device_db_file.write(str(_DEVICE_DB if device_db is None else device_db))
    elif isinstance(device_db, str):
        # The provided string is supposed to be a filename
        device_db_file_name = device_db
    else:
        # Unsupported device DB type
        raise TypeError('Unsupported type for device DB parameter')

    # Create the device manager
    device_mgr = artiq.master.worker_db.DeviceManager(
        artiq.master.databases.DeviceDB(device_db_file_name),
        virtual_devices={
            "scheduler": scheduler,
            "ccb": artiq.frontend.artiq_run.DummyCCB()
        }
    )

    # Dataset DB and manager for testing in tmp directory
    dataset_db_file_name = os.path.join(tempdir.name, 'dataset_db.pyon')
    dataset_db = artiq.master.databases.DatasetDB(dataset_db_file_name)
    dataset_mgr = artiq.master.worker_db.DatasetManager(dataset_db)

    # Argument manager
    argument_mgr = artiq.language.environment.ProcessArgumentManager(arguments)

    # Return a tuple that is accepted as manager_or_parent
    # DeviceManager, DatasetManager, ProcessArgumentManager, dict
    return device_mgr, dataset_mgr, argument_mgr, {}


class _TemporaryDirectory(tempfile.TemporaryDirectory):  # type: ignore[type-arg]
    """Custom `TemporaryDirectory` class."""

    _refs: typing.List[_TemporaryDirectory] = []
    """List of references to instances of this class."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        # Call super
        super(_TemporaryDirectory, self).__init__(*args, **kwargs)

        # Add self to list of references to make sure the object is not destructed
        _TemporaryDirectory._refs.append(self)

    def __del__(self) -> None:
        """Cleanup temp dir explicitly at destruction, prevents resource warning."""
        self.cleanup()


# Default device DB
_DEVICE_DB: typing.Dict[str, typing.Any] = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
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

# Default expid values
_EXPID_DEFAULTS: typing.Dict[str, typing.Any] = {'file': 'dax_artiq_helper_file.py',
                                                 'class_name': 'DaxArtiqHelperExperiment',
                                                 'repo_rev': 'N/A'}
