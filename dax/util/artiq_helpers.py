import os
import tempfile
import typing
import logging

import artiq.language.environment
import artiq.master.worker_db
import artiq.master.databases  # type: ignore
import artiq.frontend.artiq_run  # type: ignore

__all__ = ['get_manager_or_parent']


def get_manager_or_parent(device_db: typing.Dict[str, typing.Any] = None) -> typing.Any:
    """Returns an object that can function as a `manager_or_parent` for ARTIQ HasEnvironment.

    This function is primarily used for testing purposes.

    :param device_db: A device DB as dict (optional)
    :return: A dummy ARTIQ manager object
    """

    # Scheduler
    scheduler = artiq.frontend.artiq_run.DummyScheduler()
    # Fill in expid of scheduler
    scheduler.expid = {'log_level': 20, 'file': 'file_name.py', 'class_name': 'DaxArtiqHelperExperiment',
                       'arguments': {}, 'repo_rev': 'N/A'}

    # Device DB for testing in the directory of this script
    device_db_file_name = os.path.join(tempfile.gettempdir(), 'dax_artiq_helper_device_db.py')
    with open(device_db_file_name, 'w') as device_db_file:
        device_db_file.write('device_db=')
        device_db_file.write(str(_device_db if device_db is None else device_db))
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
    argument_mgr = artiq.language.environment.ProcessArgumentManager({})

    # Return a tuple that is accepted as manager_or_parent
    # DeviceManager, DatasetManager, ProcessArgumentManager, dict
    return device_mgr, dataset_mgr, argument_mgr, dict()


# Disable ARTIQ logging by setting logging level to critical
logging.basicConfig(level=logging.CRITICAL)

# Default device db
_device_db = {
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
