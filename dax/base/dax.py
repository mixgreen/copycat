from __future__ import annotations  # Postponed evaluation of annotations

import abc
import logging
import itertools
import functools
import re
import natsort
import typing
import pygit2  # type: ignore
import os
import numbers
import collections
import numpy as np

from artiq import __version__ as _artiq_version
import artiq.experiment
import artiq.master.worker_db

import artiq.coredevice.core  # type: ignore
import artiq.coredevice.dma  # type: ignore
import artiq.coredevice.cache  # type: ignore

from dax import __version__ as _dax_version
import dax.base.exceptions
import dax.base.interface
import dax.sim.ddb
import dax.sim.device

__all__ = ['DaxModule', 'DaxSystem', 'DaxService',
           'DaxClient', 'dax_client_factory']

# Workaround: Add Numpy ndarray as a sequence type (see https://github.com/numpy/numpy/issues/2776)
collections.abc.Sequence.register(np.ndarray)

_KEY_SEPARATOR: str = '.'
"""Key separator for datasets."""

_NAME_RE: typing.Pattern[str] = re.compile(r'\w+')
"""Regex for matching valid names."""


def _is_valid_name(name: str) -> bool:
    """Return true if the given name is valid."""
    assert isinstance(name, str), 'The given name should be a string'
    return bool(_NAME_RE.fullmatch(name))


def _is_valid_key(key: str) -> bool:
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return all(_NAME_RE.fullmatch(n) for n in key.split(_KEY_SEPARATOR))


def _get_cwd_commit() -> typing.Optional[str]:
    """Return the commit hash of the current working directory if available.

    :return: Commit hash as a string or None if no repository could be found
    """
    # Discover repository path of current working directory, also looks in parent directories
    path = pygit2.discover_repository(os.getcwd())
    return None if path is None else str(pygit2.Repository(path).head.target.hex)


_CWD_COMMIT: typing.Optional[str] = _get_cwd_commit()
"""Commit hash of the current working directory if available."""

del _get_cwd_commit  # Remove one-time function

_ARTIQ_VIRTUAL_DEVICES: typing.Set[str] = {'scheduler', 'ccb'}
"""ARTIQ virtual devices."""


def _get_unique_device_key(d: typing.Dict[str, typing.Any], key: str) -> str:
    """Get the unique device key."""

    assert isinstance(key, str), 'Key must be a string'

    if key in _ARTIQ_VIRTUAL_DEVICES:
        # Virtual devices always have unique names
        return key
    else:
        # Resolve the unique device key
        return _resolve_unique_device_key(d, key, set())


def _resolve_unique_device_key(d: typing.Dict[str, typing.Any], key: str, trace: typing.Set[str]) -> str:
    """Recursively resolve aliases until we find the unique device name."""

    assert isinstance(d, dict), 'First argument must be a dict to search in'
    assert isinstance(key, str), 'Key must be a string'
    assert isinstance(trace, set), 'Trace must be a set'

    # Check if we are not stuck in a loop
    if key in trace:
        # We are in an alias loop
        raise LookupError(f'Key "{key:s}" caused an alias loop')
    # Add key to the trace
    trace.add(key)

    # Get value (could raise KeyError)
    v = d[key]

    if isinstance(v, str):
        # Recurse if we are still dealing with an alias
        return _resolve_unique_device_key(d, v, trace)
    elif isinstance(v, dict):
        # We reached a dict, key must be the unique key
        return key
    else:
        # We ended up with an unexpected type
        raise TypeError(f'Key "{key:s}" returned an unexpected type')


class DaxBase(artiq.experiment.HasEnvironment, abc.ABC):
    """Base class for all DAX base classes."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize DAX base.

        :param managers_or_parent: ARTIQ manager or parent of this environment
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Logger object
        self.__logger: logging.Logger = logging.getLogger(self.get_identifier())

        # Build
        self.logger.debug('Starting build...')
        try:
            # Call super, which will call build()
            super(DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
        except dax.base.exceptions.BuildError:
            raise  # Exception was already logged
        except Exception as e:
            # Log the exception to provide more context
            self.logger.exception(e)
            # Raise a different exception type to prevent that the caught exception is logged again by the parent
            raise dax.base.exceptions.BuildError(e) from None  # Do not duplicate full traceback again
        else:
            self.logger.debug('Build finished')

    @property
    def logger(self) -> logging.Logger:
        """Get the logger of this object.

        :return: The logger object
        """
        return self.__logger

    @artiq.experiment.host_only
    def update_kernel_invariants(self, *keys: str) -> None:
        """Add one or more keys to the set of kernel invariants.

        Kernel invariants are attributes that are not changed during kernel execution.
        Marking attributes as invariant enables more aggressive compiler optimizations.

        :param keys: The keys to add to the set of kernel invariants.
        """

        assert all(isinstance(k, str) for k in keys), 'All keys must be of type str'

        # Get kernel invariants using getattr() such that we do not overwrite a user-defined variable
        kernel_invariants: typing.Set[str] = getattr(self, 'kernel_invariants', set())
        # Update the set with the given keys
        self.kernel_invariants: typing.Set[str] = kernel_invariants | {*keys}

    @abc.abstractmethod
    def get_identifier(self) -> str:
        pass

    def __repr__(self) -> str:
        """Returns a string representation of the object.

        :return: The object identifier string.
        """
        return self.get_identifier()


class DaxHasSystem(DaxBase, abc.ABC):
    """Intermediate base class for DAX classes that are dependent on a DAX system."""

    __D_T = typing.TypeVar('__D_T')  # Device type verification

    __CORE_DEVICES: typing.Tuple[str, ...] = ('core', 'core_dma', 'core_cache')
    """Attribute names of core devices."""
    __CORE_ATTRIBUTES: typing.Tuple[str, ...] = __CORE_DEVICES + ('data_store',)
    """Attribute names of core objects created in build() or inherited from parents."""

    def __init__(self, managers_or_parent: typing.Any, name: str, system_key: str, registry: DaxNameRegistry,
                 *args: typing.Any, **kwargs: typing.Any):
        """Constructor of a DAX base class.

        :param managers_or_parent: The manager or parent object
        :param name: The name of this object
        :param system_key: The unique system key, used for object identification
        :param registry: The shared registry object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        assert isinstance(name, str), 'Name must be a string'
        assert isinstance(system_key, str), 'System key must be a string'
        assert isinstance(registry, DaxNameRegistry), 'Registry must be a DAX name registry'

        # Check name and system key
        if not _is_valid_name(name):
            raise ValueError(f'Invalid name "{name:s}" for class "{self.__class__.__name__:s}"')
        if not _is_valid_key(system_key) or not system_key.endswith(name):
            raise ValueError(f'Invalid system key "{system_key:s}" for class "{self.__class__.__name__:s}"')

        # Store constructor arguments as attributes
        self.__name: str = name
        self.__system_key: str = system_key
        self.__registry: DaxNameRegistry = registry

        # Call super, which will result in a call to build()
        super(DaxHasSystem, self).__init__(managers_or_parent, *args, **kwargs)

        # Verify that all core attributes are available
        if not all(hasattr(self, n) for n in self.__CORE_ATTRIBUTES):
            msg: str = 'Missing core attributes (super.build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make core devices kernel invariants
        self.update_kernel_invariants(*self.__CORE_DEVICES)

    @property
    def registry(self) -> DaxNameRegistry:
        """Get the DAX registry.

        :return: The logger object
        """
        return self.__registry

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """Get the core device driver.

        :return: The core object
        """
        return self.__core

    @property
    def core_dma(self) -> artiq.coredevice.dma.CoreDMA:
        """Get the core DMA device driver.

        :return: The core DMA object
        """
        return self.__core_dma

    @property
    def core_cache(self) -> artiq.coredevice.cache.CoreCache:
        """Get the core cache device driver.

        :return: The core cache object
        """
        return self.__core_cache

    @property
    def data_store(self) -> DaxDataStore:
        """Get the data store.

        :return: The data store object
        """
        return self.__data_store

    def _take_parent_core_attributes(self, parent: DaxHasSystem) -> None:
        """Take core attributes from parent.

        If this object does not construct its own core attributes, it should take them from their parent.
        """
        try:
            # Take core attributes from parent, attributes are taken one by one to allow typing
            self.__core: artiq.coredevice.core.Core = parent.core
            self.__core_dma: artiq.coredevice.dma.CoreDMA = parent.core_dma
            self.__core_cache: artiq.coredevice.cache.CoreCache = parent.core_cache
            self.__data_store: DaxDataStore = parent.data_store
        except AttributeError:
            parent.logger.exception('Missing core attributes (super.build() was probably not called)')
            raise

    @artiq.experiment.host_only
    def get_name(self) -> str:
        """Get the name of this component."""
        return self.__name

    @artiq.experiment.host_only
    def get_system_key(self, *keys: str) -> str:
        """Get the full key based on the system key.

        If no keys are provided, the system key is returned.
        If one or more keys are provided, the provided keys are appended to the system key.

        :param keys: The keys to append to the system key
        :return: The system key with provided keys appended.
        :raises ValueError: Raised if the key has an invalid format
        """

        assert all(isinstance(k, str) for k in keys), 'Keys must be strings'

        # Check if the given keys are valid
        for k in keys:
            if not _is_valid_key(k):
                raise ValueError(f'Invalid key "{k:s}"')

        # Return the assigned key
        return _KEY_SEPARATOR.join([self.__system_key, *keys])

    @artiq.experiment.host_only
    def _init_system(self) -> None:
        """Initialize the DAX system, for dataset access, device initialization, and recording DMA traces."""

        # Initialize all children
        self.call_child_method('_init_system')

        # Initialize this object
        self.logger.debug('Initializing...')
        try:
            self.init()
        except artiq.coredevice.core.CompileError:
            # Log a fixed message, the exception message is empty
            self.logger.exception('Compilation error occurred during initialization')
            raise
        except Exception as e:
            # Log the exception to provide more context
            self.logger.exception(e)
            raise
        else:
            self.logger.debug('Initialization finished')

    @artiq.experiment.host_only
    def _post_init_system(self) -> None:
        """DAX system post-initialization (e.g. obtaining DMA handles)."""

        # Post-initialize all children
        self.call_child_method('_post_init_system')

        # Post-initialize this object
        self.logger.debug('Post-initializing...')
        try:
            self.post_init()
        except artiq.coredevice.core.CompileError:
            # Log a fixed message, the exception message is empty
            self.logger.exception('Compilation error occurred during post-initialization')
            raise
        except Exception as e:
            # Log the exception to provide more context
            self.logger.exception(e)
            raise
        else:
            self.logger.debug('Post-initialization finished')

    @abc.abstractmethod
    def init(self) -> None:
        """Override this method to access the dataset (r/w), initialize devices, and record DMA traces.

        The :func:`init` function will be called when the user calls :func:`dax_init`
        in the experiment :func:`run` function.
        """
        pass

    @abc.abstractmethod
    def post_init(self) -> None:
        """Override this method for post-initialization procedures (e.g. obtaining DMA handles).

        The :func:`post_init` function will be called when the user calls :func:`dax_init`
        in the experiment :func:`run` function.
        The :func:`post_init` function is called after all :func:`init` functions have been called.
        This function is used to perform initialization tasks that are dependent on the initialization
        of other components, for example to obtain a DMA handle.
        """
        pass

    @typing.overload
    def get_device(self, key: str) -> typing.Any:
        ...

    @typing.overload
    def get_device(self, key: str, type_: typing.Type[__D_T]) -> __D_T:
        ...

    @artiq.experiment.host_only
    def get_device(self, key: str, type_: typing.Optional[typing.Type[__D_T]] = None) -> typing.Any:
        """Get a device driver.

        Users can optionally specify an expected device type.
        If the device does not match the expected type, an exception is raised.
        Note that drivers of controllers are SiPyCo RPC objects and type checks are therefore not useful.

        Devices that are retrieved using :func:`get_device` can not be added to the kernel invariants.
        The user is responsible for adding the attribute to the list of kernel invariants.

        :param key: The key of the device to obtain
        :param type_: The expected type of the device
        :return: The requested device driver
        :raises KeyError: Raised when the device could not be obtained from the device DB
        :raises TypeError: Raised when the device does not match the expected type
        """

        assert isinstance(key, str) and key, 'Key must be of type str and not empty'

        # Debug message
        self.logger.debug(f'Requesting device "{key:s}"')

        try:
            # Get the unique key, which will also check the keys and aliases
            unique: str = _get_unique_device_key(self.get_device_db(), key)
        except (LookupError, TypeError) as e:
            # Device was not found in the device DB
            raise KeyError(f'Device "{key:s}" could not be found in the device DB') from e

        # Get the device using the initial key (let ARTIQ resolve the aliases)
        device: typing.Any = super(DaxHasSystem, self).get_device(key)

        if type_ is not None and not isinstance(device, (type_, artiq.master.worker_db.DummyDevice,
                                                         dax.sim.device.DaxSimDevice)):
            # Device has an unexpected type
            raise TypeError(f'Device "{key:s}" does not match the expected type')

        # Register the requested device with the unique key
        self.registry.add_device(unique, device, self)

        # Return the device
        return device

    @artiq.experiment.host_only
    def setattr_device(self, key: str, attr_name: typing.Optional[str] = None,
                       type_: typing.Optional[typing.Type[__D_T]] = None) -> None:
        """Sets a device driver as attribute.

        If no attr_name is provided, the key will be the attribute name.

        Users can optionally specify an expected device type.
        If the device does not match the expected type, an exception is raised.

        The attribute used to set the device driver is automatically added to the kernel invariants.

        :param key: The key of the device
        :param attr_name: The attribute name to assign the device driver to
        :param type_: The expected type of the device
        :raises KeyError: Raised when the device could not be obtained from the device DB
        :raises TypeError: Raised when the device does not match the expected type
        :raises ValueError: Raised if the attribute name is not valid
        :raises AttributeError: Raised if the attribute name was already assigned
        """

        assert isinstance(attr_name, str) or attr_name is None, 'Attribute name must be of type str or None'

        # Get the device
        device = self.get_device(key, type_=type_)  # type: ignore[arg-type]

        if attr_name is None:
            # Set attribute name to key if no attribute name was given
            attr_name = key

        # Set the device as attribute
        if not _is_valid_name(attr_name):
            raise ValueError(f'Attribute name "{attr_name:s}" not valid')
        if hasattr(self, attr_name):
            raise AttributeError(f'Attribute name "{attr_name:s}" was already assigned')
        setattr(self, attr_name, device)

        # Add attribute to kernel invariants
        self.update_kernel_invariants(attr_name)

    @artiq.experiment.rpc(flags={'async'})
    def set_dataset_sys(self, key, value, data_store=True):  # type: (str, typing.Any, bool) -> None
        """Sets the contents of a system dataset.

        :param key: The key of the system dataset
        :param value: The value to store
        :param data_store: Flag to archive the value in the data store
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Modify logging level of worker_db logger to suppress an unwanted warning message
        artiq.master.worker_db.logger.setLevel(logging.WARNING + 1)

        # Set value in system dataset with extra flags
        self.logger.debug(f'System dataset key "{key:s}" set to value "{value}"')
        self.set_dataset(system_key, value, broadcast=True, persist=True, archive=True)

        # Restore original logging level of worker_db logger
        artiq.master.worker_db.logger.setLevel(logging.NOTSET)

        if data_store:
            # Archive value using the data store
            self.data_store.set(system_key, value)

    @artiq.experiment.rpc(flags={'async'})
    def mutate_dataset_sys(self, key, index, value,
                           data_store=True):  # type: (str, typing.Any, typing.Any, bool) -> None
        """Mutate an existing system dataset at the given index.

        :param key: The key of the system dataset
        :param index: The array index to mutate, slicing and multi-dimensional indexing allowed
        :param value: The value to store
        :param data_store: Flag to archive the value in the data store
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Mutate system dataset
        self.logger.debug(f'System dataset key "{key:s}"[{index}] mutate to value "{value}"')
        self.mutate_dataset(system_key, index, value)

        if data_store:
            # Archive value using the data store
            self.data_store.mutate(system_key, index, value)

    @artiq.experiment.rpc(flags={'async'})
    def append_to_dataset_sys(self, key, value, data_store=True):  # type: (str, typing.Any, bool) -> None
        """Append a value to a system dataset.

        :param key: The key of the system dataset
        :param value: The value to store
        :param data_store: Flag to archive the value in the data store
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key: str = self.get_system_key(key)

        # Append value to system dataset
        self.logger.debug(f'System dataset key "{key:s}" append value "{value}"')
        self.append_to_dataset(system_key, value)

        if data_store:
            # Archive value using the data store
            self.data_store.append(system_key, value)

    def get_dataset_sys(self, key: str, default: typing.Any = artiq.experiment.NoDefault) -> typing.Any:
        """Returns the contents of a system dataset.

        If the key is present, its value will be returned.
        If the key is not present and no default is provided, a `KeyError` will be raised.
        If the key is not present and a default is provided, the default value will
        be written to the dataset and the same value will be returned.

        The above behavior differs slightly from :func:`get_dataset` since it will write
        the default value to the dataset in case the key was not present.
        If the default value is written to the dataset, it is also archived in the data store.

        Values that are retrieved using this method can not be added to the kernel invariants.
        The user is responsible for adding the attribute to the list of kernel invariants.

        :param key: The key of the system dataset
        :param default: The default value to set the system dataset to if not present
        :return: The value of the system dataset or the default value
        :raises KeyError: Raised if the key was not present and no default was provided
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key: str = self.get_system_key(key)

        try:
            # Get value from system dataset with extra flags
            value: typing.Any = self.get_dataset(system_key, archive=True)
        except KeyError:
            if default is artiq.experiment.NoDefault:
                # The value was not available in the system dataset and no default was provided
                raise KeyError(f'System dataset key "{system_key:s}" not found') from None
            else:
                # If the value does not exist, write the default value to the system dataset, but do not archive yet
                self.logger.debug(f'System dataset key "{key:s}" set to default value "{default}"')
                self.set_dataset(system_key, default, broadcast=True, persist=True, archive=False)
                # Get the value again and make sure it is archived
                value = self.get_dataset(system_key, archive=True)  # Should never raise a KeyError
                # Archive value using the data store
                self.data_store.set(system_key, value)
        else:
            self.logger.debug(f'System dataset key "{key:s}" returned value "{value}"')

        # Return value
        return value

    @artiq.experiment.host_only
    def setattr_dataset_sys(self, key: str, default: typing.Any = artiq.experiment.NoDefault,
                            kernel_invariant: bool = True) -> None:
        """Sets the contents of a system dataset as attribute.

        If the key is present, its value will be loaded to the attribute.
        If the key is not present and no default is provided, the attribute is not set.
        If the key is not present and a default is provided, the default value will
        be written to the dataset and the attribute will be set to the same value.

        The above behavior differs slightly from :func:`setattr_dataset` since it will never raise an exception.
        This behavior was chosen to make sure initialization can always pass, even when keys are not available.
        Exceptions will be raised when an attribute is missing while being accessed in Python
        or when a kernel is compiled that needs the attribute.

        An other difference from :func:`setattr_dataset` is that if the function falls back on the
        default value, the default value will be written to the dataset and archived in the data store.

        The function :func:`hasattr` can be used for conditional initialization in case it is possible that a
        certain attribute is not present (i.e. when this function is used without a default value).

        Attributes set using this function will by default be added to the kernel invariants.
        It is possible to disable this behavior by setting the appropriate function parameter.

        :param key: The key of the system dataset
        :param default: The default value to set the system dataset to if not present
        :param kernel_invariant: Flag to set the attribute as kernel invariant or not
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        assert isinstance(kernel_invariant, bool), 'Kernel invariant flag must be of type bool'

        try:
            # Get the value from system dataset
            value: typing.Any = self.get_dataset_sys(key, default)
        except KeyError:
            # The value was not available in the system dataset and no default was provided, attribute will not be set
            self.logger.debug(f'System attribute "{key:s}" not set')
        else:
            # Set the value as attribute (reassigning is possible, required for re-loading attributes)
            setattr(self, key, value)

            if kernel_invariant:
                # Update kernel invariants
                self.update_kernel_invariants(key)

            # Debug message
            msg_postfix: str = ' (kernel invariant)' if kernel_invariant else ''
            self.logger.debug(f'System attribute "{key:s}" set to value "{value}"{msg_postfix:s}')

    @artiq.experiment.host_only
    def hasattr(self, *keys: str) -> bool:
        """Returns if this object has the given attributes.

        This function is especially useful when :func:`setattr_dataset_sys` is used without a default value.

        :param keys: The attribute names to check
        :return: True if all attributes are set
        """
        assert all(isinstance(k, str) for k in keys), 'Keys must be of type str'
        return all(hasattr(self, k) for k in keys)

    @artiq.experiment.host_only
    def get_identifier(self) -> str:
        """Return the system key with the class name."""
        return f'[{self.get_system_key():s}]({self.__class__.__name__:s})'


class DaxModuleBase(DaxHasSystem, abc.ABC):
    """Base class for all DAX modules and systems."""

    def __init__(self, managers_or_parent: typing.Any, module_name: str, module_key: str, registry: DaxNameRegistry,
                 *args: typing.Any, **kwargs: typing.Any):
        """Construct the module base class.

        :param managers_or_parent: Manager or parent of this module
        :param module_name: Name of the module
        :param module_key: Unique and complete key of this module
        :param registry: The shared registry object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Call super
        super(DaxModuleBase, self).__init__(managers_or_parent, module_name, module_key, registry, *args, **kwargs)

        # Register this module
        self.registry.add_module(self)


class DaxModule(DaxModuleBase, abc.ABC):
    """Base class for DAX modules."""

    def __init__(self, managers_or_parent: DaxHasSystem, module_name: str,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX module.

        :param managers_or_parent: The parent of this module
        :param module_name: The name of this module
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check module name
        if not _is_valid_name(module_name):
            raise ValueError(f'Invalid module name "{module_name:s}"')
        # Check parent type
        if not isinstance(managers_or_parent, DaxHasSystem):
            raise TypeError(f'Parent of module "{module_name:s}" is not of type DaxHasSystem')

        # Take core attributes from parent
        self._take_parent_core_attributes(managers_or_parent)

        # Call super, use parent to assemble arguments
        super(DaxModule, self).__init__(managers_or_parent, module_name, managers_or_parent.get_system_key(module_name),
                                        managers_or_parent.registry, *args, **kwargs)


class DaxSystem(DaxModuleBase):
    """Base class for DAX systems, which is a top-level module."""

    SYS_ID: str
    """Identifier of the system."""
    SYS_VER: int
    """Version of the system."""

    SYS_NAME: str = 'system'
    """System name, used as top key for modules."""
    SYS_SERVICES: str = 'services'
    """System services, used as top key for services."""

    CORE_KEY: str = 'core'
    """Key of the core device."""
    CORE_DMA_KEY: str = 'core_dma'
    """Key of the core DMA device."""
    CORE_CACHE_KEY: str = 'core_cache'
    """Key of the core cache device."""
    CORE_LOG_KEY: str = 'core_log'
    """Key of the core log controller."""
    DAX_INFLUX_DB_KEY: str = 'dax_influx_db'
    """Key of the DAX Influx DB controller."""

    DAX_INIT_TIME_KEY: str = 'dax_init_time'
    """DAX initialization time dataset key."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX system.

        :param managers_or_parent: The manager or parent of this system
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check if system ID was overridden
        assert hasattr(self, 'SYS_ID'), 'Every DAX system class must override the SYS_ID class attribute'
        assert isinstance(self.SYS_ID, str), 'System ID must be of type str'
        assert _is_valid_name(self.SYS_ID), f'Invalid system ID "{self.SYS_ID:s}"'

        # Check if system version was overridden
        assert hasattr(self, 'SYS_VER'), 'Every DAX system class must override the SYS_VER class attribute'
        assert isinstance(self.SYS_VER, int), 'System version must be of type int'
        assert self.SYS_VER >= 0, 'Invalid system version, set version number larger or equal to zero'

        # Call super, add names, add a new registry
        super(DaxSystem, self).__init__(managers_or_parent, self.SYS_NAME, self.SYS_NAME, DaxNameRegistry(self),
                                        *args, **kwargs)

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """Get the core device driver.

        :return: The core object
        """
        return self.__core

    @property
    def core_dma(self) -> artiq.coredevice.dma.CoreDMA:
        """Get the core DMA device driver.

        :return: The core DMA object
        """
        return self.__core_dma

    @property
    def core_cache(self) -> artiq.coredevice.cache.CoreCache:
        """Get the core cache device driver.

        :return: The core cache object
        """
        return self.__core_cache

    @property
    def data_store(self) -> DaxDataStore:
        """Get the data store.

        :return: The data store object
        """
        return self.__data_store

    @property
    def dax_sim_enabled(self) -> bool:
        """Bool that indicates if DAX simulation is enabled.

        :return: True if DAX simulation is enabled
        """
        return self.__sim_enabled

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Override this method to build your DAX system. (Do not forget to call `super.build()` first!)"""

        # Log DAX version
        self.logger.debug(f'DAX version {_dax_version:s}')

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxSystem, self).build(*args, **kwargs)

        try:
            # Get the virtual simulation configuration device
            self.get_device(dax.sim.ddb.DAX_SIM_CONFIG_KEY)
        except KeyError:
            # Simulation disabled
            self.__sim_enabled: bool = False
        except artiq.master.worker_db.DeviceError:
            # Error while initializing simulation
            self.logger.exception('Failed to initialize DAX simulation')
            raise
        else:
            # Simulation enabled
            self.__sim_enabled = True
        finally:
            # Add dax_sim_enabled property as a kernel invariant
            self.update_kernel_invariants('dax_sim_enabled')

        # Core devices
        self.__core: artiq.coredevice.core.Core = self.get_device(self.CORE_KEY, artiq.coredevice.core.Core)
        self.__core_dma: artiq.coredevice.dma.CoreDMA = self.get_device(self.CORE_DMA_KEY, artiq.coredevice.dma.CoreDMA)
        self.__core_cache: artiq.coredevice.cache.CoreCache = self.get_device(self.CORE_CACHE_KEY,
                                                                              artiq.coredevice.cache.CoreCache)

        # Verify existence of core log controller
        try:
            # Register the core log controller with the system
            self.get_device(self.CORE_LOG_KEY)
        except KeyError:
            # Core log controller was not found in the device DB
            if not self.dax_sim_enabled:
                # Log a warning (if we are not in simulation)
                self.logger.warning(f'Core log controller "{self.CORE_LOG_KEY:s}" not found in device DB')
        except artiq.master.worker_db.DeviceError:
            # Failed to create core log driver
            self.logger.warning(f'Failed to create core log driver "{self.CORE_LOG_KEY:s}"', exc_info=True)

        # Instantiate the data store (needs to be done in build() since it requests a controller)
        try:
            # Create an Influx DB data store
            self.__data_store: DaxDataStore = DaxDataStoreInfluxDb(self, self.DAX_INFLUX_DB_KEY)
        except KeyError:
            # Influx DB controller was not found in the device DB, fall back on base data store
            if not self.dax_sim_enabled:
                # Log a warning (if we are not in simulation)
                self.logger.warning(f'Influx DB controller "{self.DAX_INFLUX_DB_KEY:s}" not found in device DB')
            # Log a debug message
            self.logger.debug('Fall back on base data store')
            self.__data_store = DaxDataStore()
        except artiq.master.worker_db.DeviceError:
            # Failed to create Influx DB driver, fall back on base data store
            self.__data_store = DaxDataStore()
            self.logger.warning(f'Failed to create DAX Influx DB driver "{self.DAX_INFLUX_DB_KEY:s}"', exc_info=True)

    @artiq.experiment.host_only
    def dax_init(self) -> None:
        """Initialize the DAX system.

        When initializing, first the :func:`init` function of child objects are called in hierarchical order.
        The :func:`init` function of this system is called last.
        Finally, all :func:`post_init` functions are called in the same order.
        """

        # Store system information in local archive
        self.set_dataset('dax/system_id', self.SYS_ID, archive=True)
        self.set_dataset('dax/system_version', self.SYS_VER, archive=True)
        self.set_dataset('dax/dax_version', _dax_version, archive=True)
        self.set_dataset('dax/dax_sim_enabled', self.dax_sim_enabled, archive=True)
        if _CWD_COMMIT is not None:
            self.set_dataset('dax/cwd_commit', _CWD_COMMIT, archive=True)

        # Perform system initialization
        self.logger.debug('Starting DAX system initialization...')
        self._init_system()
        self._post_init_system()
        self.logger.debug('Finished DAX system initialization')

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass


class DaxService(DaxHasSystem, abc.ABC):
    """Base class for system services."""

    SERVICE_NAME: str
    """The unique name of this service."""

    def __init__(self, managers_or_parent: DaxHasSystem,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX service base class.

        :param managers_or_parent: The manager or parent of this object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check if service name was overridden
        assert hasattr(self, 'SERVICE_NAME'), 'Every DAX service class must override the SERVICE_NAME class attribute'
        assert isinstance(self.SERVICE_NAME, str), 'Service name must be of type str'

        # Check parent type
        if not isinstance(managers_or_parent, (DaxSystem, DaxService)):
            raise TypeError(f'Parent of service "{self.SERVICE_NAME:s}" is not a DAX system or service')

        # Take core attributes from parent
        self._take_parent_core_attributes(managers_or_parent)

        # Use name registry of parent to obtain a system key
        registry: DaxNameRegistry = managers_or_parent.registry
        system_key: str = registry.make_service_key(self.SERVICE_NAME)

        # Call super
        super(DaxService, self).__init__(managers_or_parent, self.SERVICE_NAME, system_key, registry, *args, **kwargs)

        # Register this service
        self.registry.add_service(self)


class DaxClient(DaxHasSystem, abc.ABC):
    """Base class for DAX clients.

    Clients are template experiments that will later be joined with a user-provided system.
    When the template is instantiated, the client identifies itself as the system,
    just like a regular experiment that inherits from a system would do.
    Though the client is actually a child of the system and therefore does not share
    a namespace with the system.

    The client class should be decorated using the :func:`dax_client_factory` decorator.
    This decorator creates a factory function that allows users to provide their system
    to be used with this experiment template.

    Normally, a client would implement the :func:`prepare`, :func:`run`, and :func:`analyze`
    functions to define execution flow.
    Additionally, a :func:`build` function can be implemented to provide a user interface
    for configuring the client.

    Note that the :func:`build` function does not need to call super().
    The decorator will make sure all classes are build in the correct order.
    """

    DAX_INIT: bool = True
    """Flag if dax_init() should run for this client."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Construct the DAX client object.

        :param managers_or_parent: Manager or parent of this module
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """
        # Check if the decorator was used
        if not isinstance(managers_or_parent, DaxSystem):
            raise TypeError(f'DAX client class {self.__class__.__name__:s} must be decorated with @dax_client_factory')

        # Take attributes from the parent system
        self._take_parent_core_attributes(managers_or_parent)

        # Call super and identify with system name and system key
        super(DaxClient, self).__init__(managers_or_parent, managers_or_parent.SYS_NAME, managers_or_parent.SYS_NAME,
                                        managers_or_parent.registry, *args, **kwargs)

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> None:
        pass


class DaxNameRegistry:
    """A class for unique name registration."""

    __M_T = typing.TypeVar('__M_T', bound=DaxModuleBase)  # Module base type variable
    __S_T = typing.TypeVar('__S_T', bound=DaxService)  # Service type variable
    __I_T = typing.TypeVar('__I_T', bound=dax.base.interface.DaxInterface)  # Interface type variable

    def __init__(self, system: DaxSystem):
        """Create a new DAX name registry.

        :param system: The DAX system this registry belongs to
        """

        assert isinstance(system, DaxSystem), 'System must be of type DAX system'

        # Check system services key
        if not _is_valid_key(system.SYS_SERVICES):
            raise ValueError(f'Invalid system services key "{system.SYS_SERVICES:s}"')

        # Store system services key
        self._sys_services_key: str = system.SYS_SERVICES  # Access attribute directly

        # A dict containing registered modules
        self._modules: typing.Dict[str, DaxModuleBase] = {}
        # A dict containing registered devices
        self._devices: typing.Dict[str, typing.Tuple[typing.Any, DaxHasSystem]] = {}
        # A dict containing registered services
        self._services: typing.Dict[str, DaxService] = {}

    def add_module(self, module: __M_T) -> None:
        """Register a module.

        :param module: The module to register
        :raises NonUniqueRegistrationError: Raised if the module key was already registered by another module
        """

        assert isinstance(module, DaxModuleBase), 'Module is not a DAX module base'

        # Get the module key
        key: str = module.get_system_key()

        # Get the module that registered the module key (None if the key is available)
        reg_module: typing.Optional[DaxModuleBase] = self._modules.get(key)

        if reg_module is not None:
            # Key already in use by another module
            msg: str = f'Module key "{key:s}" was already registered by module {reg_module.get_identifier():s}'
            raise dax.base.exceptions.NonUniqueRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._modules[key] = module

    @typing.overload
    def get_module(self, key: str) -> DaxModuleBase:
        ...

    @typing.overload
    def get_module(self, key: str, type_: typing.Type[__M_T]) -> __M_T:
        ...

    def get_module(self, key: str, type_: typing.Type[DaxModuleBase] = DaxModuleBase) -> DaxModuleBase:
        """Return the requested module by key.

        :param key: The key of the module
        :param type_: The expected type of the module
        :raises KeyError: Raised if the module could not be found
        :raises TypeError: Raised if the module type does not match the expected type
        """

        assert isinstance(key, str), 'Key must be a string'

        try:
            # Get the module
            module: DaxModuleBase = self._modules[key]
        except KeyError:
            # Module was not found
            raise KeyError(f'Module "{key:s}" could not be found') from None

        if not isinstance(module, type_):
            # Module does not have the correct type
            raise TypeError(f'Module "{key:s}" does not match the expected type')

        # Return the module
        return module

    def find_module(self, type_: typing.Type[__M_T]) -> __M_T:
        """Find a unique module that matches the requested type.

        :param type_: The type of the module
        :return: The unique module of the requested type
        :raises KeyError: Raised if no modules of the desired type were found
        :raises LookupError: Raised if more then one module of the desired type was found
        """

        # Search for all modules matching the type
        results: typing.Dict[str, DaxNameRegistry.__M_T] = self.search_modules(type_)

        if not results:
            # No modules were found
            raise KeyError(f'Could not find modules with type "{type_.__name__:s}"')
        elif len(results) > 1:
            # More than one module was found
            raise LookupError(f'Could not find a unique module with type "{type_.__name__:s}"')

        # Return the only result
        _, module = results.popitem()
        return module

    def search_modules(self, type_: typing.Type[__M_T]) -> typing.Dict[str, __M_T]:
        """Search for modules that match the requested type and return results as a dict.

        :param type_: The type of the modules
        :return: A dict with key-module pairs
        """

        assert issubclass(type_, DaxModuleBase), 'Provided type must be a subclass of DaxModuleBase'

        # Search for all modules matching the type
        results: typing.Dict[str, DaxNameRegistry.__M_T] = {k: m for k, m in self._modules.items() if
                                                            isinstance(m, type_)}

        # Return the list with results
        return results

    def get_module_key_list(self) -> typing.List[str]:
        """Return a sorted list of registered module keys.

        :return: A list with module keys
        """
        module_key_list = natsort.natsorted(self._modules.keys())  # Natural sort the list
        return module_key_list

    def get_module_list(self) -> typing.List[DaxModuleBase]:
        """Return the list of registered modules.

        :return: A list with module objects
        """
        return list(self._modules.values())

    def add_device(self, key: str, device: typing.Any, parent: DaxHasSystem) -> None:
        """Register a device.

        Devices are added to the registry to ensure every device is only owned by a single parent.

        :param key: The unique key of the device
        :param device: The device object
        :param parent: The parent that requested the device
        :return: The requested device driver
        :raises NonUniqueRegistrationError: Raised if the device was already registered by another parent
        """

        assert isinstance(key, str), 'Device key must be a string'
        assert isinstance(parent, DaxHasSystem), 'Parent is not a DaxHasSystem type'

        if key in _ARTIQ_VIRTUAL_DEVICES:
            return  # Virtual devices always have unique names are excluded from the registry

        # Get the device value object (None if the device was not registered before)
        device_value = self._devices.get(key)

        if device_value is not None:
            # Device was already registered
            _, reg_parent = device_value  # Unpack tuple
            msg: str = f'Device "{key:s}" was already registered by parent "{reg_parent.get_system_key():s}"'
            raise dax.base.exceptions.NonUniqueRegistrationError(msg)

        # Add unique device key to the dict of registered devices
        self._devices[key] = (device, parent)

    def search_devices(self, type_: typing.Union[type, typing.Tuple[type, ...]]) -> typing.Set[str]:
        """Search for registered devices that match the requested type and return their keys.

        :param type_: The type of the devices
        :return: A set with unique device keys
        """

        # Search for all registered devices matching the type
        results = {k for k, (device, _) in self._devices.items() if isinstance(device, type_)}

        # Return the list with results
        return results

    def get_device_key_list(self) -> typing.List[str]:
        """Return a sorted list of registered device keys.

        :return: A list of device keys that were registered
        """
        device_key_list = natsort.natsorted(self._devices.keys())  # Natural sort the list
        return device_key_list

    def make_service_key(self, service_name: str) -> str:
        """Return the system key for a service name.

        This function can be used to generate a system key for a new service.

        :param service_name: The unique name of the service
        :return: The system key for this service
        :raises ValueError: Raised if the provided service name is not valid
        """

        # Check the given name
        assert isinstance(service_name, str), 'Service name must be a string'
        if not _is_valid_name(service_name):
            # Service name not valid
            raise ValueError(f'Invalid service name "{service_name:s}"')

        # Return assigned key
        return _KEY_SEPARATOR.join([self._sys_services_key, service_name])

    def add_service(self, service: __S_T) -> None:
        """Register a service.

        Services are added to the registry to ensure every service is only present once.
        Services can also be found using the registry.

        :param service: The service to register
        :raises NonUniqueRegistrationError: Raised if the service name was already registered
        """

        assert isinstance(service, DaxService), 'Service must be a DAX service'

        # Services get indexed by name (the name of a service is unique)
        key: str = service.get_name()

        # Get the service that registered with the service name (None if key is available)
        reg_service = self._services.get(key)

        if reg_service is not None:
            # Service name was already registered
            raise dax.base.exceptions.NonUniqueRegistrationError(f'Service with name "{key:s}" was already registered')

        # Add service to the registry
        self._services[key] = service

    def has_service(self, key: typing.Union[str, typing.Type[__S_T]]) -> bool:
        """Return if a service is available.

        Check if a service is available in this system.

        :param key: The key of the service, can be a string or the type of the service
        :return: True if the service is available
        """
        try:
            self.get_service(key)
        except KeyError:
            return False
        else:
            return True

    @typing.overload
    def get_service(self, key: str) -> DaxService:
        ...

    @typing.overload
    def get_service(self, key: typing.Type[__S_T]) -> __S_T:
        ...

    def get_service(self, key: typing.Union[str, typing.Type[__S_T]]) -> DaxService:
        """Get a service from the registry.

        Obtain a registered service.

        :param key: The key of the service, can be a string or the type of the service
        :return: The requested service
        :raises KeyError: Raised if the service is not available
        """

        assert isinstance(key, str) or issubclass(key, DaxService), 'Key must be a string or a DAX service type'

        # Obtain the key
        service_key: str = key if isinstance(key, str) else key.SERVICE_NAME

        # Try to return the requested service
        try:
            return self._services[service_key]
        except KeyError:
            # Service was not found
            raise KeyError(f'Service "{service_key:s}" is not available') from None

    def get_service_key_list(self) -> typing.List[str]:
        """Return a sorted list of registered service keys.

        :return: A list of service keys that were registered
        """
        service_key_list = natsort.natsorted(self._services.keys())  # Natural sort the list
        return service_key_list

    def get_service_list(self) -> typing.List[DaxService]:
        """Return the list of registered services.

        :return: A list with service objects
        """
        return list(self._services.values())

    def find_interface(self, type_: typing.Type[__I_T]) -> __I_T:
        """Find a unique interface that matches the requested type.

        Note: mypy type checker does not handle pure abstract base classes correctly.
        A `# type: ignore[misc]` annotation on the line using this function is probably
        required to pass type checking.

        :param type_: The type of the interface
        :return: The unique interface of the requested type
        :raises KeyError: Raised if no interfaces of the desired type were found
        :raises LookupError: Raised if more then one interface of the desired type was found
        """

        # Search for all interfaces matching the type
        results = self.search_interfaces(type_)

        if not results:
            # No interfaces were found
            raise KeyError(f'Could not find interfaces with type "{type_.__name__:s}"')
        elif len(results) > 1:
            # More than one interface was found
            raise LookupError(f'Could not find a unique interface with type "{type_.__name__:s}"')

        # Return the only result
        _, interface = results.popitem()
        return interface

    def search_interfaces(self, type_: typing.Type[__I_T]) -> typing.Dict[str, __I_T]:
        """Search for interfaces that match the requested type and return results as a dict.

        Note: mypy type checker does not handle pure abstract base classes correctly.
        A `# type: ignore[misc]` annotation on the line using this function is probably
        required to pass type checking.

        :param type_: The type of the interfaces
        :return: A dict with key-interface pairs
        """

        assert issubclass(type_, dax.base.interface.DaxInterface), 'Provided type must be a subclass of DaxInterface'

        # Search for all modules and services matching the interface type
        iterator = itertools.chain(self._modules.values(), self._services.values())
        results = {itf.get_system_key(): itf for itf in iterator if isinstance(itf, type_)}

        # Return the list with results
        return typing.cast(typing.Dict[str, DaxNameRegistry.__I_T], results)


class DaxDataStore:
    """Base class for the DAX data store.

    Data stores are used for long-term archiving of time series data and have
    methods that reflect the operations on ARTIQ datasets: set, mutate, and append.
    For system dataset methods, DAX automatically invokes the data store and
    no user code is required.

    The base DAX data store does not store anything and can be used
    as a placeholder object since it is not an abstract base class.
    Other DAX data store classes can inherit from this class and
    override the :func:`set`, :func:`mutate`, and :func:`append` methods.
    """

    def __init__(self) -> None:  # Constructor return type required if no parameters are given
        """Construct a new DAX data store object."""
        # Create a logger object
        self._logger: logging.Logger = logging.getLogger(f'{self.__module__:s}.{self.__class__.__name__:s}')

    def set(self, key: str, value: typing.Any) -> None:
        """Write a key-value into the data store.

        :param key: The key of the value
        :param value: The value to store
        """
        self._logger.debug(f'Set key "{key:s}" to value: "{value}"')

    def mutate(self, key: str, index: typing.Any, value: typing.Any) -> None:
        """Mutate a specific index of a key-value in the data store.

        :param key: The key of the value
        :param index: The index to mutate
        :param value: The value to store
        """
        self._logger.debug(f'Mutate key "{key:s}"[{index}] to value "{value}"')

    def append(self, key: str, value: typing.Any) -> None:
        """Append a value to a key-value in the data store.

        :param key: The key of the value
        :param value: The value to append
        """
        self._logger.debug(f'Append key "{key:s}" with value "{value}"')


class DaxDataStoreInfluxDb(DaxDataStore):
    """Influx DB DAX data store class.

    This data store connects to an Influx DB controller (see DAX comtools) to
    push data to an Influx database.
    """

    __F_T = typing.Union[bool, float, int, np.int32, np.int64, str]  # Field type variable for Influx DB supported types
    __FD_T = typing.Dict[str, __F_T]  # Field dict type variable
    __P_T = typing.Dict[str, typing.Union[str, __FD_T]]  # Point type variable

    _FIELD_TYPES: typing.Tuple[type, ...] = (bool, float, int, np.int32, np.int64, str)
    """Legal field types for Influx DB."""

    def __init__(self, system: DaxSystem, key: str):
        """Create a new DAX data store that uses an Influx DB backend.

        :param system: The system this data store is managed by
        :param key: The key of the DAX Influx DB controller
        """

        assert isinstance(system, DaxSystem), 'System parameter must be of type DaxSystem'
        assert isinstance(key, str), 'The Influx DB controller key must be of type str'

        # Call super
        super(DaxDataStoreInfluxDb, self).__init__()

        # Get the Influx DB driver, this call can raise various exceptions
        self._get_driver(system, key)

        # Get the scheduler, which is a virtual device
        scheduler = system.get_device('scheduler')
        if isinstance(scheduler, artiq.master.worker_db.DummyDevice):
            return  # ARTIQ is only discovering experiment classes, do not continue initialization

        # Store values that will be used for data points later
        self._sys_id: str = system.SYS_ID
        # Initialize index table for the append function, required to emulate appending behavior
        self._index_table: typing.Dict[str, int] = {}

        # Prepare base tags
        self._base_tags: DaxDataStoreInfluxDb.__FD_T = {
            'system_version': str(system.SYS_VER),  # Convert int version to str since tags are strings
        }

        # Prepare base fields
        self._base_fields: DaxDataStoreInfluxDb.__FD_T = {
            'rid': int(scheduler.rid),
            'pipeline_name': str(scheduler.pipeline_name),
            'priority': int(scheduler.priority),
            'artiq_version': str(_artiq_version),
            'dax_version': str(_dax_version),
            'dax_sim_enabled': str(system.dax_sim_enabled)
        }

        # Add expid items to fields if keys do not exist yet and the types are appropriate
        self._base_fields.update((k, v) for k, v in scheduler.expid.items()
                                 if k not in self._base_fields and isinstance(v, self._FIELD_TYPES))

        # Add commit hashes to fields
        if _CWD_COMMIT is not None:
            self._base_fields['cwd_commit'] = _CWD_COMMIT

        # Debug message
        self._logger.debug(f'Initialized base fields: {self._base_fields}')

    def set(self, key: str, value: typing.Any) -> None:
        """Write a key-value into the Influx DB data store.

        Lists will be flattened to separate elements with an index since
        Influx DB does not support lists.

        :param key: The key of the value
        :param value: The value to store
        """

        if isinstance(value, self._FIELD_TYPES):
            # Write a single point
            self._write_points([self._make_point(key, value)])
        elif isinstance(value, collections.abc.Sequence) and all(isinstance(e, self._FIELD_TYPES) for e in value):
            if len(value):
                # If the list is not empty, write a list of points
                self._write_points([self._make_point(key, v, i) for i, v in enumerate(value)])
            # Store the length of the sequence for emulated appending later
            self._index_table[key] = len(value)
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning(f'Could not store value for key "{key:s}", unsupported value type for value "{value}"')

    def mutate(self, key: str, index: typing.Any, value: typing.Any) -> None:
        """Mutate a specified index of a key-value in the Influx DB data store.

        List structures are not supported by Influx DB and are emulated by using indices.
        The emulation only supports single-dimensional list structures.
        Hence, the index must be an integer.
        It is not checked if the key contains an actual list structure and if the index is in range.

        :param key: The key of the value
        :param index: The index to mutate
        :param value: The value to store
        """

        if isinstance(value, self._FIELD_TYPES):
            if isinstance(index, numbers.Integral):
                # Write a single point
                self._write_points([self._make_point(key, value, index)])
            else:
                # Non-integer index is not supported, do not raise but warn user instead
                self._logger.warning(f'Could not mutate value for key "{key:s}", index "{index}" not supported')
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning(f'Could not mutate value for key "{key:s}", '
                                 f'unsupported value type for value "{value}"')

    def append(self, key: str, value: typing.Any) -> None:
        """Append a value to a key-value in the Influx DB data store.

        List structures are not supported by Influx DB and are emulated by using indices.
        The Influx DB data store caches the length of arrays when using :func:`set` to emulate appending.
        If the length of the array is not in the cache, the append operation can not be emulated.

        :param key: The key of the value
        :param value: The value to store
        """

        if isinstance(value, self._FIELD_TYPES):
            # Get the current index
            index = self._index_table.get(key)
            if index is not None:
                # Write a single point
                self._write_points([self._make_point(key, value, index)])
                # Update the index table
                self._index_table[key] += 1
            else:
                # Index unknown, can not emulate append operation
                self._logger.warning(f'Could not append value for key "{key:s}", no index was cached '
                                     f'and the append operation could not be emulated')
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning(f'Could not append value for key "{key:s}", '
                                 f'unsupported value type for value "{value}"')

    def _make_point(self, key: str, value: __F_T, index: typing.Union[None, int, numbers.Integral] = None) -> __P_T:
        """Make a point object from a key-value pair, optionally with an index.

        This function does not check the type of the value and the index, which should be checked before.
        Numpy integers are automatically converted to Python int.
        """

        assert isinstance(key, str), 'Key should be of type str'

        if not _is_valid_key(key):
            # Invalid key
            raise ValueError(f'Influx DB data store received an invalid key "{key:s}"')

        if isinstance(value, np.integer):
            # Convert Numpy int to Python int
            value = int(value)

        # Copy the base tags and fields
        tags = self._base_tags.copy()
        fields = self._base_fields.copy()

        # Split the key
        split_key = key.rsplit(_KEY_SEPARATOR, maxsplit=1)
        base: str = split_key[0] if len(split_key) == 2 else ''  # Base is empty if the key does not split

        if index is not None:
            # Add index if provided
            tags['index'] = str(index)  # Tags need to be of type str
        # Add base to tags
        tags['base'] = base
        # Add key-value to fields
        fields[key] = value  # The full key and the value are the actual field

        # Create the point object
        point = {
            'measurement': self._sys_id,
            'tags': tags,
            'fields': fields,
        }

        # Return point
        return typing.cast(DaxDataStoreInfluxDb.__P_T, point)

    def _get_driver(self, system: DaxSystem, key: str) -> None:
        """Get the required driver.

        This method was separated to allow testing without writing points.
        """
        self._influx = system.get_device(key)  # Get the Influx DB driver, this call can raise various exceptions

    def _write_points(self, points: typing.Sequence[__P_T]) -> None:
        """Submit points to the Influx DB driver.

        This method was separated to allow testing without writing points.

        :param points: A list of points to write
        """
        self._influx.write_points(points)


# Note: These names should not alias with other type variable names!
__DCF_C_T = typing.TypeVar('__DCF_C_T', bound=DaxClient)  # Type variable for dax_client_factory() c (client) argument
__DCF_S_T = typing.TypeVar('__DCF_S_T', bound=DaxSystem)  # Type variable for dax_client_factory() system_type argument


def dax_client_factory(c: typing.Type[__DCF_C_T]) -> typing.Callable[[typing.Type[__DCF_S_T]], typing.Type[__DCF_C_T]]:
    """Decorator to convert a DaxClient class to a factory function for that class.

    :param c: The DAX client to create a factory function for
    :return: A factory for the client class that allows the client to be matched with a system
    """

    assert isinstance(c, type), 'The decorated object must be a class'
    assert issubclass(c, DaxClient), 'The decorated class must be a subclass of DaxClient'

    # Use the wraps decorator, but do not inherit the docstring
    @functools.wraps(c, assigned=[e for e in functools.WRAPPER_ASSIGNMENTS if e != '__doc__'])
    def wrapper(system_type: typing.Type[__DCF_S_T],
                *sys_args: typing.Any, **sys_kwargs: typing.Any) -> typing.Type[__DCF_C_T]:
        """Create a new DAX client class.

        This factory function will create a new client class for a given system type.

        :param system_type: The system type used by the client
        :param sys_args: Positional arguments forwarded to the systems :func:`build` function
        :param sys_kwargs: Keyword arguments forwarded to the systems :func:`build` function
        :return: A fusion of the client and system class
        :raises TypeError: Raised if the provided `system_type` parameter is not a subclass of `DaxSystem`
        """

        # Check the system type
        assert isinstance(system_type, type), 'System type must be a type'
        if not issubclass(system_type, DaxSystem):
            raise TypeError('System type must be a subclass of DaxSystem')

        class WrapperClass(c):  # type: ignore[valid-type,misc]
            """The wrapper class that finalizes the client class.

            The wrapper class extends the client class by constructing the system
            first and loading the client class afterwards using the system as the parent.
            """

            def __init__(self, managers_or_parent: typing.Any,
                         *args: typing.Any, **kwargs: typing.Any):
                # Create the system
                self.__system: DaxSystem = system_type(managers_or_parent, *sys_args, **sys_kwargs)
                # Call constructor of the client class and give it the system as parent
                super(WrapperClass, self).__init__(self.__system, *args, **kwargs)

            def run(self) -> None:
                if self.DAX_INIT:
                    # Initialize the system
                    self.__system.dax_init()
                # Call the run method of the client class
                super(WrapperClass, self).run()

        # The factory function returns the newly constructed wrapper class
        return WrapperClass

    # Return the factory function
    return wrapper
