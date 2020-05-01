import os
import tempfile

import logging

import artiq.master.worker_db
import artiq.master.databases  # type: ignore
import artiq.frontend.artiq_run  # type: ignore


def disable_logging():
    """Disable ARTIQ logging by setting logging level to critical."""
    logging.basicConfig(level=logging.CRITICAL)


def get_manager_or_parent():
    """Returns an object that can function as a manager_or_parent for ARTIQ HasEnvironment."""

    # Scheduler
    scheduler = artiq.frontend.artiq_run.DummyScheduler()
    # Fill in expid of scheduler
    scheduler.expid = {'log_level': 20, 'file': 'file_name.py', 'class_name': 'TestExperiment', 'arguments': {},
                       'repo_rev': 'N/A'}

    # Device DB for testing in the directory of this script
    device_db_file_name = os.path.join(os.path.dirname(__file__), 'device_db.py')
    device_mgr = artiq.master.worker_db.DeviceManager(
        artiq.master.databases.DeviceDB(device_db_file_name),
        virtual_devices={
            "scheduler": scheduler,
            "ccb": artiq.frontend.artiq_run.DummyCCB()})
    # Dataset DB and manager for testing in tmp directory
    dataset_db_file_name = os.path.join(tempfile.gettempdir(), 'dax_test_dataset_db.pyon')
    dataset_db = artiq.master.databases.DatasetDB(dataset_db_file_name)
    dataset_mgr = artiq.master.worker_db.DatasetManager(dataset_db)

    # Return a tuple that is accepted as manager_or_parent
    # DeviceManager, DatasetManager, ArgumentParser.parse_args(), dict
    return device_mgr, dataset_mgr, None, dict()


# For testing, disable logging by default
disable_logging()
