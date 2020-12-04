from __future__ import annotations  # Postponed evaluation of annotations

import os
import tempfile
import typing
import weakref

import artiq.language.environment
import artiq.master.worker_db
import artiq.master.databases
import artiq.frontend.artiq_run  # type: ignore

__all__ = ['is_kernel', 'is_portable', 'is_host_only',
           'get_managers', 'ClonedDatasetManager', 'clone_managers']


class _TemporaryDirectory(tempfile.TemporaryDirectory):  # type: ignore[type-arg]
    """Custom `TemporaryDirectory` class."""

    _refs: typing.List[_TemporaryDirectory] = []
    """List of references to instances of this class."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        # Call super
        super(_TemporaryDirectory, self).__init__(*args, **kwargs)

        # Add self to list of (strong) references to make sure the object is not destructed too soon
        _TemporaryDirectory._refs.append(self)
        # Add a finalizer to cleanup this temp dir (prevents resource warning for implicit cleanup)
        weakref.finalize(self, self.cleanup)


_EXPID_DEFAULTS: typing.Dict[str, typing.Any] = {
    'file': 'dax_artiq_helper_file.py',
    'class_name': 'DaxArtiqHelperExperiment',
    'repo_rev': 'N/A'
}
"""Default expid values."""

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
"""Default device DB."""


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


# Managers tuple type
__M_T = typing.Tuple[artiq.master.worker_db.DeviceManager, artiq.master.worker_db.DatasetManager,
                     artiq.language.environment.ProcessArgumentManager, typing.Dict[str, typing.Any]]


def get_managers(device_db: typing.Union[typing.Dict[str, typing.Any], str, None] = None, *,
                 dataset_db: typing.Optional[str] = None,
                 expid: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 arguments: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 **kwargs: typing.Any) -> __M_T:
    """Returns an object that can function as a `managers_or_parent` for ARTIQ `HasEnvironment`.

    This function is primarily used for testing purposes.

    We strongly recommend to close the devices in the device manager using `device_manager.close_devices()`
    before the manager objects are discarded. This will free any used resources.
    Devices can be closed multiple times without any side-effects.
    Just in case the user does not manually close devices, a finalizer is attached to the device manager
    object to ensure `close_devices()` is at least called once before the object is deleted.

    If a full ARTIQ environment is not required but only a core device driver is sufficient,
    please take a look at the `dax.sim.coredevice.core.BaseCore` class.

    :param device_db: A device DB as dict or a file name
    :param dataset_db: A dataset DB as a file name
    :param expid: Dict for the scheduler expid attribute
    :param arguments: Arguments for the ProcessArgumentManager object
    :param kwargs: Arguments for the ProcessArgumentManager object (updates `arguments`)
    :return: A dummy ARTIQ manager object: (`DeviceManager`, `DatasetManager`, `ProcessArgumentManager`, `dict`)
    """

    # Set default values
    if arguments is None:
        arguments = {}

    assert isinstance(dataset_db, str) or dataset_db is None, 'Dataset DB must be a str or None'
    assert isinstance(expid, dict) or expid is None, 'Expid must be a dict or None'
    assert isinstance(arguments, dict), 'Arguments must be of type dict'

    # Scheduler
    scheduler = artiq.frontend.artiq_run.DummyScheduler()
    # Construct expid of scheduler and add default values
    scheduler.expid = {} if expid is None else expid
    for k, v in _EXPID_DEFAULTS.items():
        scheduler.expid.setdefault(k, v)
    # Merge and set arguments (updates any arguments in the expid)
    arguments.update(kwargs)
    scheduler.expid.setdefault('arguments', {}).update(arguments)

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
    # Add a finalizer to guarantee devices and connections are closed
    weakref.finalize(device_mgr, device_mgr.close_devices)

    # Dataset DB and manager
    dataset_db_file_name = os.path.join(tempdir.name, 'dataset_db.pyon') if dataset_db is None else dataset_db
    dataset_mgr = artiq.master.worker_db.DatasetManager(artiq.master.databases.DatasetDB(dataset_db_file_name))

    # Argument manager
    argument_mgr = artiq.language.environment.ProcessArgumentManager(arguments)

    # Return a tuple that is accepted as managers_or_parent
    # DeviceManager, DatasetManager, ProcessArgumentManager, scheduler defaults
    return device_mgr, dataset_mgr, argument_mgr, {}


class ClonedDatasetManager(artiq.master.worker_db.DatasetManager):
    """Class for a cloned dataset manager.

    A cloned dataset managers allows dataset separation for sub-experiments under one RID.
    The cloned dataset manager has its own archive and datasets while using the same
    backend as the original dataset manager.
    When the original dataset manager is written to an HDF5 file, clones will appear as
    independent groups in the same HDF5 file.

    Dataset managers can not be cloned recursively and must always be created
    from the existing ARTIQ dataset manager.
    """

    _CLONE_DICT_KEY: str = '_dataset_mgr_clones_'
    """The attribute key of the clone dictionary attached to the existing ARTIQ dataset manager."""
    _CLONE_KEY_FORMAT: str = 'clone_{index}'
    """The key format for cloned datasets, which is used for the HDF5 group name."""

    def __init__(self, dataset_mgr: artiq.master.worker_db.DatasetManager, *,
                 name: typing.Optional[str] = None):
        """Create a clone of an existing ARTIQ dataset manager.

        :param dataset_mgr: The existing ARTIQ dataset manager
        :param name: Optional name for this clone, which will be used in the HDF5 group name
        """
        assert isinstance(dataset_mgr, artiq.master.worker_db.DatasetManager)
        assert isinstance(name, str) or name is None, 'Name must be of type str or None'

        if isinstance(dataset_mgr, ClonedDatasetManager):
            # Raise when recursion is detected
            raise TypeError('Dataset managers can not be cloned recursively')

        # Initialize this clone
        super(ClonedDatasetManager, self).__init__(dataset_mgr.ddb)

        if not hasattr(dataset_mgr, self._CLONE_DICT_KEY):
            # The existing ARTIQ dataset manager is still "fresh", so we need to mutate it
            self._mutate_dataset_manager(dataset_mgr)

        # Extract the dict with clones from the dataset manager
        clones: typing.Dict[str, ClonedDatasetManager] = getattr(dataset_mgr, self._CLONE_DICT_KEY)
        # Generate the key for this clone
        key: str = self._CLONE_KEY_FORMAT.format(index=len(clones))
        if name is not None:
            key = f'{key}_{name}'
        # Register this clone
        clones[key] = self

    def _mutate_dataset_manager(self, dataset_mgr: artiq.master.worker_db.DatasetManager) -> None:
        """Mutate the existing ARTIQ dataset manager."""

        # Attach an empty dict for clones
        clones: typing.Dict[str, ClonedDatasetManager] = {}
        setattr(dataset_mgr, self._CLONE_DICT_KEY, clones)

        # Get a reference to the original write_hdf5() function
        super_write_hdf5 = dataset_mgr.write_hdf5

        def wrapped_write_hdf5(f: typing.Any) -> None:
            # Call the "super" function of the dataset manager
            super_write_hdf5(f)
            # Write the data of the clones in separate HDF5 groups
            for name, clone in clones.items():
                clone.write_hdf5(f.create_group(name))

        # Replace the write_hdf5() function of the dataset manager
        setattr(dataset_mgr, 'write_hdf5', wrapped_write_hdf5)  # Dynamic assignment to pass type check


def clone_managers(managers: typing.Any, *,
                   name: typing.Optional[str] = None,
                   arguments: typing.Optional[typing.Dict[str, typing.Any]] = None,
                   **kwargs: typing.Any) -> __M_T:
    """Clone a given tuple of ARTIQ manager objects to use for a sub-experiment.

    Sub-experiments (i.e. `children` in ARTIQ terminology) can share ARTIQ manager objects with their parent
    by passing the parent experiment object. In this case, the same dataset manager and argument manager
    are shared, which can be undesired for certain situations (e.g. dataset aliasing or argument aliasing).
    This function allows you to clone ARTIQ manager objects which will decouple the dataset manager
    and the argument manager from the parent while still sharing the device manager.

    This function is mainly used when creating and running multiple sub-experiments from a parent
    experiment class where the data and arguments are preferably decoupled.

    Cloning of the managers is not limited to the build phase, though the ARTIQ manager objects
    have to be captured in the constructor of your experiment.

    Note: The scheduling defaults dict will also be decoupled.

    :param managers: The tuple with ARTIQ manager objects
    :param name: Optional name for cloned dataset manager, which will be used in the HDF5 group name
    :param arguments: Arguments for the ProcessArgumentManager object
    :param kwargs: Arguments for the ProcessArgumentManager object (updates `arguments`)
    :return: A cloned ARTIQ manager object: (`DeviceManager`, `ClonedDatasetManager`, `ProcessArgumentManager`, `dict`)
    """

    # Set default values
    if arguments is None:
        arguments = {}

    assert isinstance(arguments, dict), 'Arguments must be of type dict'

    # Check the type of the passed managers
    if isinstance(managers, artiq.language.environment.HasEnvironment):
        raise ValueError('A parent was passed instead of the raw managers tuple')
    if not isinstance(managers, tuple):
        raise ValueError('Managers must be a tuple')

    try:
        # Unpack the managers
        device_mgr, dataset_mgr, _, _ = managers
    except ValueError:
        raise ValueError('The managers could not be unpacked')
    else:
        # Check types of the unpacked manager objects
        if not isinstance(device_mgr, artiq.master.worker_db.DeviceManager):
            raise TypeError('The unpacked device manager has an unexpected type')
        if not isinstance(dataset_mgr, artiq.master.worker_db.DatasetManager):
            raise TypeError('The unpacked dataset manager has an unexpected type')

    # Merge keyword arguments into arguments dict
    arguments.update(kwargs)
    # Create a new argument manager
    argument_mgr = artiq.language.environment.ProcessArgumentManager(arguments)

    # Return the new managers consisting of existing, cloned, and new objects
    return device_mgr, ClonedDatasetManager(dataset_mgr, name=name), argument_mgr, {}
