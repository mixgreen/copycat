import os
import tempfile
import typing
import logging

import artiq.language.environment
import artiq.master.worker_db
import artiq.master.databases  # type: ignore
import artiq.frontend.artiq_run  # type: ignore

__all__ = ['get_manager_or_parent']


def get_manager_or_parent(device_db: typing.Union[typing.Dict[str, typing.Any], str, None] = None,
                          expid: typing.Optional[typing.Dict[str, typing.Any]] = None,
                          **arguments: typing.Any) -> typing.Any:
    """Returns an object that can function as a `manager_or_parent` for ARTIQ HasEnvironment.

    This function is primarily used for testing purposes.

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
    scheduler.expid = dict() if expid is None else expid
    for k, v in _EXPID_DEFAULTS.items():
        scheduler.expid.setdefault(k, v)
    # Set arguments (overwrites any arguments in the expid)
    scheduler.expid['arguments'] = arguments

    if isinstance(device_db, dict) or device_db is None:
        # Create a temporally device DB file
        device_db_file_name = os.path.join(tempfile.gettempdir(), 'dax_artiq_helper_device_db.py')
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
            "ccb": artiq.frontend.artiq_run.DummyCCB()})

    # Dataset DB and manager for testing in tmp directory
    dataset_db_file_name = os.path.join(tempfile.gettempdir(), 'dax_artiq_helper_dataset_db.pyon')
    dataset_db = artiq.master.databases.DatasetDB(dataset_db_file_name)
    dataset_mgr = artiq.master.worker_db.DatasetManager(dataset_db)

    # Argument manager
    argument_mgr = artiq.language.environment.ProcessArgumentManager(arguments)

    # Return a tuple that is accepted as manager_or_parent
    # DeviceManager, DatasetManager, ProcessArgumentManager, dict
    return device_mgr, dataset_mgr, argument_mgr, dict()


# Disable ARTIQ logging by setting logging level to critical
logging.basicConfig(level=logging.CRITICAL)

# Default device db
_DEVICE_DB = {
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
}  # type: typing.Dict[str, typing.Any]

# Default expid values
_EXPID_DEFAULTS = {'log_level': logging.CRITICAL,
                   'file': 'dax_artiq_helper_file.py',
                   'class_name': 'DaxArtiqHelperExperiment',
                   'repo_rev': 'N/A'}  # type: typing.Dict[str, typing.Any]
