from __future__ import annotations  # Postponed evaluation of annotations

import os.path
import tempfile
import typing
import weakref
import time

import artiq.language.core
import artiq.language.environment
import artiq.language.scan
import artiq.master.worker_db
import artiq.master.worker_impl  # type: ignore
import artiq.master.databases
import artiq.frontend.artiq_run  # type: ignore

__all__ = ['is_kernel', 'is_portable', 'is_host_only', 'is_rpc', 'is_decorated',
           'process_arguments', 'get_managers', 'ClonedDatasetManager', 'clone_managers', 'isolate_managers']

# Workaround required for Python<=3.8
if typing.TYPE_CHECKING:
    _TD_T = tempfile.TemporaryDirectory[str]  # Type for a temporary directory
else:
    _TD_T = tempfile.TemporaryDirectory


class _TemporaryDirectory(_TD_T):
    """Custom :class:`TemporaryDirectory` class."""

    _refs: typing.List[_TemporaryDirectory] = []
    """List of references to instances of this class."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        # Call super
        super(_TemporaryDirectory, self).__init__(*args, **kwargs)

        # Add self to list of (strong) references to make sure the object is not destructed too soon
        _TemporaryDirectory._refs.append(self)
        # Add a finalizer to cleanup this temp dir (prevents resource warning for implicit cleanup)
        weakref.finalize(self, self.cleanup)


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
"""Default device DB."""


def is_kernel(func: typing.Any) -> bool:
    """Helper function to detect if a function is an ARTIQ kernel (``@kernel``) or not.

    :param func: The function of interest
    :return: :const:`True` if the given function is a kernel
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else (meta.core_name is not None and not meta.portable)


def is_portable(func: typing.Any) -> bool:
    """Helper function to detect if a function is an ARTIQ portable function (``@portable``) or not.

    :param func: The function of interest
    :return: :const:`True` if the given function is a portable function
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else bool(meta.portable)


def is_host_only(func: typing.Any) -> bool:
    """Helper function to detect if a function is decorated as host only (``@host_only``) or not.

    :param func: The function of interest
    :return: :const:`True` if the given function is host only
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else bool(meta.forbidden)


def is_rpc(func: typing.Any, *, flags: typing.Optional[typing.Set[str]] = None) -> bool:
    """Helper function to detect if a function is an ARTIQ RPC function (``@rpc``) or not.

    Note that this function only detects RPC functions that are **explicitly decorated**.

    :param func: The function of interest
    :param flags: Expected flags
    :return: :const:`True` if the given function is an RPC function with the expected flags (subset of function flags)
    """
    assert flags is None or isinstance(flags, set), 'Flags must be a set or None'

    meta = getattr(func, 'artiq_embedded', None)
    if meta is None:
        return False
    else:
        if flags is None:
            flags = set()
        return meta.core_name is None and not meta.portable and not meta.forbidden and flags <= meta.flags


def is_decorated(func: typing.Any) -> bool:
    """Helper function to detect if a function is decorated with an ARTIQ decorator.

    :param func: The function of interest
    :return: :const:`True` if the given function is decorated with an ARTIQ decorator
    """
    return getattr(func, 'artiq_embedded', None) is not None


def _convert_argument(argument: typing.Any) -> typing.Any:
    """Convert a single argument."""
    if isinstance(argument, artiq.language.scan.ScanObject):
        return argument.describe()  # type: ignore[attr-defined]
    else:
        # No conversion required
        return argument


def process_arguments(arguments: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """Process a dict with raw arguments to make it compatible with the ARTIQ format.

    The expid and the :class:`ProcessArgumentManager` expect arguments in a PYON serializable format.
    This function will make sure that complex objects (such as scan argument objects) are
    converted to the correct format

    :param arguments: Arguments to process as a dict
    :return: The processed arguments
    """
    assert isinstance(arguments, dict), 'Arguments must be of type dict'
    return {key: _convert_argument(arg) for key, arg in arguments.items()}


class _ManagersTuple(typing.NamedTuple):
    """A named tuple of ARTIQ manager objects.

    This named tuple extends the functionality of a bare tuple.
    """
    device_mgr: artiq.master.worker_db.DeviceManager
    dataset_mgr: artiq.master.worker_db.DatasetManager
    argument_mgr: artiq.language.environment.ProcessArgumentManager
    scheduler_defaults: typing.Dict[str, typing.Any]

    def close(self) -> None:
        # Close devices
        self.device_mgr.close_devices()

    def __enter__(self) -> _ManagersTuple:
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> None:
        # Close at exit
        self.close()


def get_managers(device_db: typing.Union[typing.Dict[str, typing.Any], str, None] = None, *,
                 dataset_db: typing.Optional[str] = None,
                 expid: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 arguments: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 **kwargs: typing.Any) -> _ManagersTuple:
    """Returns a tuple of ARTIQ manager objects that can be used to construct an ARTIQ :class:`HasEnvironment` object.

    This function is primarily used for testing purposes.

    The returned tuple is an extended named tuple object that is backwards compatible.
    The attributes can be accessed by name and the tuple object can also be used as a context manager.

    We strongly recommend to close the managers before they are discarded. This will free any used resources.
    **The best way to guarantee the managers are closed is to use the returned tuple as a context manager.**
    Alternatively, users can call the :func:`close` method of the tuple or close managers manually.
    Managers can be closed multiple times without any side-effects.
    Just in case the user does not manually close managers, finalizers attached to specific managers
    will close any occupied resources before object destruction.

    If a full ARTIQ environment is not required but only a core device driver is sufficient,
    please take a look at :class:`dax.sim.coredevice.core.BaseCore`.

    :param device_db: A device DB as dict or a file name
    :param dataset_db: A dataset DB as a file name
    :param expid: Dict for the scheduler expid attribute
    :param arguments: Arguments for the ProcessArgumentManager object
    :param kwargs: Arguments for the ProcessArgumentManager object (updates ``arguments``)
    :return: A tuple of ARTIQ manager objects: ``(DeviceManager, DatasetManager, ProcessArgumentManager, dict)``
    """

    if arguments is None:
        # Set default value
        arguments = {}
    else:
        assert isinstance(arguments, dict), 'Arguments must be of type dict'
        assert all(isinstance(k, str) for k in arguments), 'Keys of the arguments dict must be of type str'
        arguments = arguments.copy()  # Copy arguments to make sure the dict is not mutated

    assert isinstance(dataset_db, str) or dataset_db is None, 'Dataset DB must be a str or None'
    assert isinstance(expid, dict) or expid is None, 'Expid must be a dict or None'

    # Create a scheduler
    scheduler = artiq.frontend.artiq_run.DummyScheduler()
    # Construct expid object
    expid_: typing.Dict[str, typing.Any] = {
        'file': 'dax_util_artiq_helper_file.py',
        'class_name': 'DaxUtilArtiqHelperExperiment',
        'repo_rev': 'N/A'
    }
    if expid is not None:
        # Given expid overrides default values
        expid_.update(expid)
    # Merge and set arguments (updates any arguments in the expid)
    arguments.update(kwargs)
    arguments = process_arguments(arguments)
    expid_.setdefault('arguments', {}).update(arguments)
    # Assign expid to scheduler
    scheduler.expid = expid_

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

    # Return an extended tuple object
    return _ManagersTuple(device_mgr, dataset_mgr, argument_mgr, {})


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
    _CLONE_KEY_FORMAT: str = 'sub_experiment/{index}'
    """The key format for cloned datasets, which is used for the HDF5 group name."""

    def __init__(self, dataset_mgr: artiq.master.worker_db.DatasetManager, *,
                 name: typing.Optional[str] = None,
                 dataset_db: typing.Any = None):
        """Create a clone of an existing ARTIQ dataset manager.

        The name parameter must be unique and is formatted with an index parameter.
        The index parameter starts at :const:`0` and is incremented for every new clone
        created from the existing ARTIQ dataset manager.

        The ``name`` is directly used as the HDF5 group name and ``'/'`` can therefore be
        used to create sub-groups.

        :param dataset_mgr: The existing ARTIQ dataset manager
        :param name: Optional name for this clone, which will be used for the HDF5 group name
        :param dataset_db: Optional backend dataset DB, defaults to dataset DB of the existing ARTIQ dataset manager
        """
        assert isinstance(dataset_mgr, artiq.master.worker_db.DatasetManager)
        assert isinstance(name, str) or name is None, 'Name must be of type str or None'

        if isinstance(dataset_mgr, ClonedDatasetManager):
            # Raise when recursion is detected
            raise TypeError('Dataset managers can not be cloned recursively')

        # Initialize this clone
        super(ClonedDatasetManager, self).__init__(dataset_mgr.ddb if dataset_db is None else dataset_db)

        if not hasattr(dataset_mgr, self._CLONE_DICT_KEY):
            # The existing ARTIQ dataset manager is still "fresh", so we need to mutate it
            self._mutate_dataset_manager(dataset_mgr)

        # Extract the dict with clones from the dataset manager
        clones: typing.Dict[str, ClonedDatasetManager] = getattr(dataset_mgr, self._CLONE_DICT_KEY)
        # Generate the key for this clone and check if it is unique
        key: str = (self._CLONE_KEY_FORMAT if name is None else name).format(index=len(clones))
        if key in clones:
            raise LookupError(f'Key "{key}" is already in use')
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
                   **kwargs: typing.Any) -> _ManagersTuple:
    """Clone a given tuple of ARTIQ manager objects to use for a sub-experiment.

    Sub-experiments (i.e. ``children`` in ARTIQ terminology) can share ARTIQ manager objects with their parent
    by passing the parent experiment object. In this case, the same dataset manager and argument manager
    are shared, which can be undesired for certain situations (e.g. dataset aliasing or argument aliasing).
    This function allows you to clone ARTIQ manager objects which will decouple the dataset manager,
    the argument manager, and the scheduling defaults from the parent while still sharing the device manager.

    This function is mainly used when creating and running multiple sub-experiments from a parent
    experiment class where the dataset manager archives and arguments are preferably decoupled.

    Note that the dataset DB is still shared, which means broadcast and persistent datasets are shared.
    To isolate a sub-experiment completely, see :func:`isolate_managers`.

    Cloning of the managers is not limited to the build phase, though the ARTIQ manager objects
    have to be captured in the constructor of your experiment.

    :param managers: The tuple with ARTIQ manager objects
    :param name: Optional name for cloned dataset manager, which will be used in the HDF5 group name
    :param arguments: Arguments for the ProcessArgumentManager object
    :param kwargs: Arguments for the ProcessArgumentManager object (updates ``arguments``)
    :return: A cloned ARTIQ manager object: ``(DeviceManager, ClonedDatasetManager, ProcessArgumentManager, dict)``
    """

    if arguments is None:
        # Set default value
        arguments = {}
    else:
        assert isinstance(arguments, dict), 'Arguments must be of type dict'
        arguments = arguments.copy()  # Copy arguments to make sure the dict is not mutated

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
    arguments = process_arguments(arguments)
    # Create a new argument manager
    argument_mgr = artiq.language.environment.ProcessArgumentManager(arguments)

    # Return the new managers consisting of existing, cloned, and new objects
    return _ManagersTuple(device_mgr, ClonedDatasetManager(dataset_mgr, name=name), argument_mgr, {})


class _DummyDatasetDB(typing.Dict[typing.Any, typing.Any]):
    """A class that acts like an ARTIQ dataset DB, which works slightly different from a regular dict."""

    def get(self, key: typing.Any, default: typing.Any = None) -> typing.Any:
        # Get does not fallback on the default value
        return self[key]


def isolate_managers(managers: typing.Any, *,
                     name: typing.Optional[str] = None,
                     arguments: typing.Optional[typing.Dict[str, typing.Any]] = None,
                     **kwargs: typing.Any) -> _ManagersTuple:
    """Create a tuple of ARTIQ manager objects that is isolated from the given tuple of managers.

    Isolation of manager objects can be useful when running sub-experiments that should not
    share device/dataset/argument managers with the main experiment.
    The isolated ARTIQ managers consists of an empty device manager, a cloned dataset manager with
    an empty dummy dataset DB (see :class:`ClonedDatasetManager`), an empty argument manager,
    and empty scheduling defaults.

    See also :func:`clone_managers`.

    :param managers: The tuple with ARTIQ manager objects
    :param name: Optional name for cloned dataset manager, which will be used in the HDF5 group name
    :param arguments: Arguments for the ProcessArgumentManager object
    :param kwargs: Arguments for the ProcessArgumentManager object (updates ``arguments``)
    :return: Isolated ARTIQ managers: ``(DeviceManager, ClonedDatasetManager, ProcessArgumentManager, dict)``
    """

    if arguments is None:
        # Set default value
        arguments = {}
    else:
        assert isinstance(arguments, dict), 'Arguments must be of type dict'
        arguments = arguments.copy()  # Copy arguments to make sure the dict is not mutated

    # Check the type of the passed managers
    if isinstance(managers, artiq.language.environment.HasEnvironment):
        raise ValueError('A parent was passed instead of the raw managers tuple')
    if not isinstance(managers, tuple):
        raise ValueError('Managers must be a tuple')

    try:
        # Unpack the managers
        _, dataset_mgr, _, _ = managers
    except ValueError:
        raise ValueError('The managers could not be unpacked')
    else:
        # Check types of the unpacked manager objects
        if not isinstance(dataset_mgr, artiq.master.worker_db.DatasetManager):
            raise TypeError('The unpacked dataset manager has an unexpected type')

    # Create a scheduler
    scheduler = artiq.frontend.artiq_run.DummyScheduler()
    # Construct expid of scheduler
    scheduler.expid = {'file': '', 'class_name': '', 'repo_rev': 'N/A', 'arguments': arguments}

    # Create a unique temp dir
    tempdir = _TemporaryDirectory()

    # Create a temporally device DB file
    device_db_file_name = os.path.join(tempdir.name, 'device_db.py')
    with open(device_db_file_name, 'w') as device_db_file:
        device_db_file.write('device_db={}')

    # Create the device manager
    device_mgr = artiq.master.worker_db.DeviceManager(
        artiq.master.databases.DeviceDB(device_db_file_name),
        virtual_devices={
            "scheduler": scheduler,
            "ccb": artiq.frontend.artiq_run.DummyCCB()
        }
    )

    # Merge keyword arguments into arguments dict
    arguments.update(kwargs)
    arguments = process_arguments(arguments)
    # Create a new argument manager
    argument_mgr = artiq.language.environment.ProcessArgumentManager(arguments)

    # Return the isolated managers
    return _ManagersTuple(
        device_mgr, ClonedDatasetManager(dataset_mgr, name=name, dataset_db=_DummyDatasetDB()), argument_mgr, {})


@artiq.language.core.host_only
def pause_strict_priority(scheduler: typing.Any, *,
                          polling_period: typing.Union[float, int] = 0.5) -> None:
    """Allow all higher priority experiments in the pipeline to prepare and run (pauses the current experiment).

    When the ARTIQ scheduler is deciding what experiment to run next, it only looks at the priority of jobs that have
    **finished** the prepare stage. In order to achieve strict priority-order execution, we also need to consider any
    experiments that are still preparing or waiting to prepare. If there are multiple experiments in the pipeline with
    a higher priority than the current experiment, the ARTIQ scheduler will only prepare one of them (and will wait
    until that experiment enters the run phase before preparing the next one).

    This function waits for the first higher-priority experiment to finish preparing, then pauses the experiment that
    this function is called from to allow the higher-priority experiment to run.
    Upon resuming, this function recursively calls itself until all other higher-priority experiments have finished.

    Note: this function will not work if any experiments in the pipeline were run with the ``flush`` flag.

    :param scheduler: The instance of the ARTIQ scheduler virtual device in the experiment you want to pause
    :param polling_period: How often to poll (in seconds) when waiting for experiments to prepare
    """
    assert isinstance(polling_period, (float, int)), 'Polling period must be of type float or int'
    assert polling_period >= 0, 'Polling period must greater or equal to zero'

    # Get a set of all other experiments with a higher priority than the current experiment
    try:
        candidates = {rid for rid, status in scheduler.get_status().items()
                      if status['pipeline'] == scheduler.pipeline_name and status['priority'] > scheduler.priority
                      and status['status'] in {'pending', 'preparing', 'prepare_done'}}

        if candidates:
            def wait() -> bool:
                # Obtain the scheduler status
                scheduler_status = scheduler.get_status()

                # Check status of candidates in the scheduler
                candidates_in_scheduler = any(rid in candidates for rid in scheduler_status)
                candidates_in_prepare_done = any(status['status'] == 'prepare_done'
                                                 for rid, status in scheduler_status.items() if rid in candidates)
                return candidates_in_scheduler and not candidates_in_prepare_done

            while wait():
                time.sleep(polling_period)

            # Pause to allow higher priority experiment to run
            scheduler.pause()
            # Keep waiting recursively for other experiments
            pause_strict_priority(scheduler)
    except AttributeError:
        raise TypeError(f'Incorrect scheduler instance provided: {type(scheduler)}') from None
