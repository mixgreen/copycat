import abc
import logging
import itertools
import functools
import re
import natsort
import typing
import git  # type: ignore
import os
import numbers
import collections
import numpy as np

import artiq
import artiq.experiment
import artiq.master.worker_db

import artiq.coredevice.core  # type: ignore
import artiq.coredevice.dma  # type: ignore
import artiq.coredevice.cache  # type: ignore

from dax.sim.sim import DAX_SIM_CONFIG_KEY as _DAX_SIM_CONFIG_KEY
from dax.sim.device import DaxSimDevice as _DaxSimDevice

# Workaround: Add Numpy ndarray as a sequence type (see https://github.com/numpy/numpy/issues/2776)
collections.abc.Sequence.register(np.ndarray)

_KEY_SEPARATOR = '.'
"""Key separator for datasets."""

_NAME_RE = re.compile(r'\w+')
"""Regex for matching valid names."""


def _is_valid_name(name: str) -> bool:
    """Return true if the given name is valid."""
    assert isinstance(name, str), 'The given name should be a string'
    return bool(_NAME_RE.fullmatch(name))


def _is_valid_key(key: str) -> bool:
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return all(_NAME_RE.fullmatch(n) for n in key.split(_KEY_SEPARATOR))


_ARTIQ_VIRTUAL_DEVICES = {'scheduler', 'ccb'}
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
        raise LookupError('Key "{:s}" caused an alias loop'.format(key))
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
        raise TypeError('Key "{:s}" returned an unexpected type'.format(key))


class _DaxBase(artiq.experiment.HasEnvironment, abc.ABC):
    """Base class for all DAX core classes."""

    class __BuildError(RuntimeError):
        """Raised when the original build error has already been logged."""
        pass

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        # Logger object
        self.__logger = logging.getLogger(self.get_identifier())

        # Build
        self.logger.debug('Starting build...')
        try:
            # Call super, which will call build()
            super(_DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
        except self.__BuildError:
            raise  # Exception was already logged
        except Exception as e:
            # Log the exception to provide more context
            self.logger.exception(e)
            # Raise a different exception type to prevent that the caught exception is logged again by the parent
            raise self.__BuildError(e) from None  # Do not duplicate full traceback again
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
        kernel_invariants = getattr(self, 'kernel_invariants', set())  # type: typing.Set[str]
        # Update the set with the given keys
        self.kernel_invariants = kernel_invariants | {*keys}  # type: typing.Set[str]

    @abc.abstractmethod
    def get_identifier(self) -> str:
        pass


class _DaxHasSystem(_DaxBase, abc.ABC):
    """Intermediate base class for DAX classes that are dependent on a DAX system."""

    __D_T = typing.TypeVar('__D_T')  # Device type verification

    __CORE_DEVICES = ['core', 'core_dma', 'core_cache']
    """Attribute names of core devices."""
    __CORE_ATTRIBUTES = __CORE_DEVICES + ['data_store']
    """Attribute names of core objects created in build() or inherited from parents."""

    def __init__(self, managers_or_parent: typing.Any, name: str, system_key: str, registry: '_DaxNameRegistry',
                 *args: typing.Any, **kwargs: typing.Any):

        assert isinstance(name, str), 'Name must be a string'
        assert isinstance(system_key, str), 'System key must be a string'
        assert isinstance(registry, _DaxNameRegistry), 'Registry must be a DAX name registry'

        # Check name and system key
        if not _is_valid_name(name):
            raise ValueError('Invalid name "{:s}" for class "{:s}"'.format(name, self.__class__.__name__))
        if not _is_valid_key(system_key) or not system_key.endswith(name):
            raise ValueError('Invalid system key "{:s}" for class "{:s}"'.format(system_key, self.__class__.__name__))

        # Store constructor arguments as attributes
        self.__name = name
        self.__system_key = system_key
        self.__registry = registry

        # Call super, which will result in a call to build()
        super(_DaxHasSystem, self).__init__(managers_or_parent, *args, **kwargs)

        # Verify that all core attributes are available
        if not all(hasattr(self, n) for n in self.__CORE_ATTRIBUTES):
            msg = 'Missing core attributes (super.build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make core devices kernel invariants
        self.update_kernel_invariants(*self.__CORE_DEVICES)

    @property
    def registry(self) -> '_DaxNameRegistry':
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
    def data_store(self) -> '_DaxDataStore':
        """Get the data store.

        :return: The data store object
        """
        return self.__data_store

    def _take_parent_core_attributes(self, parent: '_DaxHasSystem') -> None:
        """Take core attributes from parent.

        If this object does not construct its own core attributes, it should take them from their parent.
        """
        try:
            # Take core attributes from parent, attributes are taken one by one to allow typing
            self.__core = parent.core  # type: artiq.coredevice.core.Core
            self.__core_dma = parent.core_dma  # type: artiq.coredevice.dma.CoreDMA
            self.__core_cache = parent.core_cache  # type: artiq.coredevice.cache.CoreCache
            self.__data_store = parent.data_store  # type: _DaxDataStore
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
                raise ValueError('Invalid key "{:s}"'.format(k))

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
        self.logger.debug('Requesting device "{:s}"'.format(key))

        try:
            # Get the unique key, which will also check the keys and aliases
            unique = _get_unique_device_key(self.get_device_db(), key)
        except (LookupError, TypeError) as e:
            # Device was not found in the device DB
            raise KeyError('Device "{:s}" could not be found in the device DB'.format(key)) from e

        # Get the device using the initial key (let ARTIQ resolve the aliases)
        device = super(_DaxHasSystem, self).get_device(key)

        if type_ is not None and not isinstance(device, (type_, artiq.master.worker_db.DummyDevice, _DaxSimDevice)):
            # Device has an unexpected type
            raise TypeError('Device "{:s}" does not match the expected type'.format(key))

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
        device = self.get_device(key, type_)  # type: ignore

        if attr_name is None:
            # Set attribute name to key if no attribute name was given
            attr_name = key

        # Set the device as attribute
        if not _is_valid_name(attr_name):
            raise ValueError('Attribute name "{:s}" not valid'.format(attr_name))
        if hasattr(self, attr_name):
            raise AttributeError('Attribute name "{:s}" was already assigned'.format(attr_name))
        setattr(self, attr_name, device)

        # Add attribute to kernel invariants
        self.update_kernel_invariants(attr_name)

    @artiq.experiment.rpc(flags={'async'})
    def set_dataset_sys(self, key: str, value: typing.Any) -> None:
        """Sets the contents of a system dataset.

        :param key: The key of the system dataset
        :param value: The value to store
        """

        # Get the full system key
        system_key = self.get_system_key(key)

        # Modify logging level of worker_db logger to suppress an unwanted warning message
        artiq.master.worker_db.logger.setLevel(logging.WARNING + 1)

        # Set value in system dataset with extra flags
        self.logger.debug('System dataset key "{:s}" set to value "{}"'.format(key, value))
        self.set_dataset(system_key, value, broadcast=True, persist=True, archive=True)

        # Restore original logging level of worker_db logger
        artiq.master.worker_db.logger.setLevel(logging.NOTSET)

        # Archive value using the data store
        self.data_store.set(system_key, value)

    @artiq.experiment.rpc(flags={'async'})
    def mutate_dataset_sys(self, key: str, index: typing.Any, value: typing.Any) -> None:
        """Mutate an existing system dataset at the given index.

        :param key: The key of the system dataset
        :param index: The array index to mutate, slicing and multi-dimensional indexing allowed
        :param value: The value to store
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key = self.get_system_key(key)

        # Mutate system dataset
        self.logger.debug('System dataset key "{:s}"[{}] mutate to value "{}"'.format(key, index, value))
        self.mutate_dataset(system_key, index, value)

        # Archive value using the data store
        self.data_store.mutate(system_key, index, value)

    @artiq.experiment.rpc(flags={'async'})
    def append_to_dataset_sys(self, key: str, value: typing.Any) -> None:
        """Append a value to a system dataset.

        :param key: The key of the system dataset
        :param value: The value to store
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key = self.get_system_key(key)

        # Append value to system dataset
        self.logger.debug('System dataset key "{:s}" append value "{}"'.format(key, value))
        self.append_to_dataset(system_key, value)

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

        Values that are retrieved using this method can not be added to the kernel invariants.
        The user is responsible for adding the attribute to the list of kernel invariants.

        :param key: The key of the system dataset
        :param default: The default value to set the system dataset to if not present
        :return: The value of the system dataset or the default value
        :raises KeyError: Raised if the key was not present and no default was provided
        :raises ValueError: Raised if the key has an invalid format
        """

        # Get the full system key
        system_key = self.get_system_key(key)

        try:
            # Get value from system dataset with extra flags
            value = self.get_dataset(system_key, archive=True)
        except KeyError:
            if default is artiq.experiment.NoDefault:
                # The value was not available in the system dataset and no default was provided
                raise KeyError('System dataset key "{:s}" not found'.format(system_key)) from None
            else:
                # If the value does not exist, write the default value to the system dataset, but do not archive yet
                self.logger.debug('System dataset key "{:s}" set to default value "{}"'.format(key, default))
                self.set_dataset(system_key, default, broadcast=True, persist=True, archive=False)
                # Get the value again and make sure it is archived
                value = self.get_dataset(system_key, archive=True)  # Should never raise a KeyError
                # Archive value using the data store
                self.data_store.set(system_key, value)
        else:
            self.logger.debug('System dataset key "{:s}" returned value "{}"'.format(key, value))

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

        The function :func:`hasattr` can be used for conditional initialization in case it is possible that a
        certain attribute is not present (i.e. when this function is used without a default value).

        Attributes set using this function will by default be added to the kernel invariants.
        It is possible to disable this behavior by setting the appropriate function parameter.

        :param key: The key of the system dataset
        :param default: The default value to set the system dataset to if not present
        :param kernel_invariant: Flag to set the attribute as kernel invariant or not
        :raises KeyError: Raised if the key was not present
        :raises ValueError: Raised if the key has an invalid format
        :raises AttributeError: Raised if the attribute name was already assigned
        """

        assert isinstance(kernel_invariant, bool), 'Kernel invariant flag must be of type bool'

        try:
            # Get the value from system dataset
            value = self.get_dataset_sys(key, default)
        except KeyError:
            # The value was not available in the system dataset and no default was provided, attribute will not be set
            self.logger.debug('System attribute "{:s}" not set'.format(key))
        else:
            # Set the value as attribute
            if hasattr(self, key):
                raise AttributeError('Attribute name "{:s}" was already assigned'.format(key))
            setattr(self, key, value)

            if kernel_invariant:
                # Update kernel invariants
                self.update_kernel_invariants(key)

            # Debug message
            msg_postfix = ' (kernel invariant)' if kernel_invariant else ''
            self.logger.debug('System attribute "{:s}" set to value "{}"{:s}'.format(key, value, msg_postfix))

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
        return '[{:s}]({:s})'.format(self.get_system_key(), self.__class__.__name__)


class _DaxModuleBase(_DaxHasSystem, abc.ABC):
    """Base class for all DAX modules and systems."""

    def __init__(self, managers_or_parent: typing.Any, module_name: str, module_key: str, registry: '_DaxNameRegistry',
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX module base."""

        # Call super
        super(_DaxModuleBase, self).__init__(managers_or_parent, module_name, module_key, registry, *args, **kwargs)

        # Register this module
        self.registry.add_module(self)


class DaxModule(_DaxModuleBase, abc.ABC):
    """Base class for DAX modules."""

    def __init__(self, managers_or_parent: _DaxModuleBase, module_name: str,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX module.

        :param managers_or_parent: The parent of this module
        :param module_name: The name of this module
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check module name
        if not _is_valid_name(module_name):
            raise ValueError('Invalid module name "{:s}"'.format(module_name))
        # Check parent type
        if not isinstance(managers_or_parent, _DaxModuleBase):
            raise TypeError('Parent of module "{:s}" is not a DAX module base'.format(module_name))

        # Take core attributes from parent
        self._take_parent_core_attributes(managers_or_parent)

        # Call super, use parent to assemble arguments
        super(DaxModule, self).__init__(managers_or_parent, module_name, managers_or_parent.get_system_key(module_name),
                                        managers_or_parent.registry, *args, **kwargs)


class DaxInterface(abc.ABC):
    """Base class for interfaces."""
    pass


class DaxSystem(_DaxModuleBase):
    """Base class for DAX systems, which is a top-level module."""

    SYS_ID = ''  # type: str
    """Identifier of the system."""
    SYS_VER = -1  # type: int
    """Version of the system."""

    SYS_NAME = 'system'  # type: str
    """System name, used as top key for modules."""
    SYS_SERVICES = 'services'  # type: str
    """System services, used as top key for services."""

    CORE_KEY = 'core'
    """Key of the core device."""
    CORE_DMA_KEY = 'core_dma'
    """Key of the core DMA device."""
    CORE_CACHE_KEY = 'core_cache'
    """Key of the core cache device."""
    CORE_LOG_KEY = 'core_log'
    """Key of the core log controller."""
    DAX_INFLUX_DB_KEY = 'dax_influx_db'
    """Key of the DAX Influx DB controller."""

    DAX_INIT_TIME_KEY = 'dax_init_time'
    """DAX initialization time dataset key."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX system.

        :param managers_or_parent: The manager or parent of this system
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check if system ID was overridden
        assert isinstance(self.SYS_ID, str), 'System ID must be of type str'
        assert DaxSystem.SYS_ID != self.SYS_ID, 'Every DAX system class must override the SYS_ID class attribute'
        assert _is_valid_name(self.SYS_ID), 'Invalid system ID "{:s}"'.format(self.SYS_ID)

        # Check if system version was overridden
        assert isinstance(self.SYS_VER, int), 'System version must be of type int'
        assert DaxSystem.SYS_VER != self.SYS_VER, 'Every DAX system class must override the SYS_VER class attribute'
        assert self.SYS_VER >= 0, 'Invalid system version, set version number larger or equal to zero'

        # Call super, add names, add a new registry
        super(DaxSystem, self).__init__(managers_or_parent, self.SYS_NAME, self.SYS_NAME, _DaxNameRegistry(self),
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
    def data_store(self) -> '_DaxDataStore':
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

        if args or kwargs:
            # Warn if we find any dangling arguments
            self.logger.warning('Unused args "{}" / kwargs "{}" were passed to super.build()'.format(args, kwargs))

        try:
            # Get the virtual simulation configuration device
            self.get_device(_DAX_SIM_CONFIG_KEY)
        except KeyError:
            # Simulation disabled
            self.__sim_enabled = False
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
        self.logger.debug('Requesting core device drivers')
        self.__core = self.get_device(self.CORE_KEY, artiq.coredevice.core.Core)
        self.__core_dma = self.get_device(self.CORE_DMA_KEY, artiq.coredevice.dma.CoreDMA)
        self.__core_cache = self.get_device(self.CORE_CACHE_KEY, artiq.coredevice.cache.CoreCache)

        # Verify existence of core log controller
        try:
            # Register the core log controller with the system
            self.logger.debug('Requesting core log driver')
            self.get_device(self.CORE_LOG_KEY)
        except KeyError:
            # Core log controller was not found in the device DB
            if not self.dax_sim_enabled:
                # Log a warning (if we are not in simulation)
                self.logger.warning('Core log controller "{:s}" not found in device DB'.format(self.CORE_LOG_KEY))
        except artiq.master.worker_db.DeviceError:
            # Failed to create core log driver
            self.logger.warning('Failed to create core log driver "{:s}"'.format(self.CORE_LOG_KEY), exc_info=True)

        # Instantiate the data store (needs to be done in build() since it requests a controller)
        try:
            # Create an Influx DB data store
            self.logger.debug('Initializing Influx DB data store')
            self.__data_store = _DaxDataStoreInfluxDb(self, self.DAX_INFLUX_DB_KEY)
        except KeyError:
            # Influx DB controller was not found in the device DB, fall back on base data store
            if not self.dax_sim_enabled:
                # Log a warning (if we are not in simulation)
                self.logger.warning('Influx DB controller "{:s}" not found in device DB'.format(self.DAX_INFLUX_DB_KEY))
            # Log a debug message
            self.logger.debug('Fall back on base data store')
            self.__data_store = _DaxDataStore()
        except artiq.master.worker_db.DeviceError:
            # Failed to create Influx DB driver, fall back on base data store
            self.__data_store = _DaxDataStore()
            self.logger.warning('Failed to create DAX Influx DB driver "{:s}"'.format(self.DAX_INFLUX_DB_KEY),
                                exc_info=True)

    @artiq.experiment.host_only
    def dax_init(self) -> None:
        """Initialize the DAX system.

        When initializing, first the :func:`init` function of child objects are called in hierarchical order.
        The :func:`init` function of this system is called last.
        Finally, all :func:`post_init` functions are called in the same order.
        """

        # Store system information in local archive
        self.set_dataset(self.get_system_key('dax_system_id'), self.SYS_ID, archive=True)
        self.set_dataset(self.get_system_key('dax_system_version'), self.SYS_VER, archive=True)

        # Perform system initialization
        self.logger.debug('Starting DAX system initialization...')
        self._init_system()
        self._post_init_system()
        self.logger.debug('Finished DAX system initialization')

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass


class DaxService(_DaxHasSystem, abc.ABC):
    """Base class for system services."""

    SERVICE_NAME = ''  # type: str
    """The unique name of this service"""

    def __init__(self, managers_or_parent: _DaxHasSystem,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX service base class.

        :param managers_or_parent: The manager or parent of this object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Check if service name was overridden
        assert isinstance(self.SERVICE_NAME, str), 'Service name must be of type str'
        assert DaxService.SERVICE_NAME != self.SERVICE_NAME, \
            'Every DAX service class must override the SERVICE_NAME class attribute'

        # Check parent type
        if not isinstance(managers_or_parent, (DaxSystem, DaxService)):
            raise TypeError('Parent of service "{:s}" is not a DAX system or service'.format(self.SERVICE_NAME))

        # Take core attributes from parent
        self._take_parent_core_attributes(managers_or_parent)

        # Use name registry of parent to obtain a system key
        registry = managers_or_parent.registry
        system_key = registry.make_service_key(self.SERVICE_NAME)

        # Call super
        super(DaxService, self).__init__(managers_or_parent, self.SERVICE_NAME, system_key, registry, *args, **kwargs)

        # Register this service
        self.registry.add_service(self)


class DaxClient(_DaxHasSystem, abc.ABC):
    """Base class for DAX clients."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        # Check if the decorator was used
        if not isinstance(self, DaxSystem):
            raise TypeError('DAX client class {:s} must be decorated using '
                            '@dax_client_factory'.format(self.__class__.__name__))

        # Call super
        super(DaxClient, self).__init__(managers_or_parent, *args, **kwargs)

    def init(self) -> None:
        # Call super (will be the user DAX system after the MRO lookup of the client wrapper class)
        super(DaxClient, self).init()

    def post_init(self) -> None:
        # Call super (will be the user DAX system after the MRO lookup of the client wrapper class)
        super(DaxClient, self).post_init()


class _DaxNameRegistry:
    """A class for unique name registration."""

    class _NonUniqueRegistrationError(LookupError):
        """Exception when a name is registered more then once."""
        pass

    __M_T = typing.TypeVar('__M_T', bound=_DaxModuleBase)  # Module base type variable
    __S_T = typing.TypeVar('__S_T', bound=DaxService)  # Service type variable
    __I_T = typing.TypeVar('__I_T', bound=DaxInterface)  # Interface type variable

    def __init__(self, system: DaxSystem):
        """Create a new DAX name registry.

        :param system: The DAX system this registry belongs to
        """

        assert isinstance(system, DaxSystem), 'System must be of type DAX system'

        # Check system services key
        if not _is_valid_key(system.SYS_SERVICES):
            raise ValueError('Invalid system services key "{:s}"'.format(system.SYS_SERVICES))

        # Store system services key
        self._sys_services_key = system.SYS_SERVICES  # Access attribute directly

        # A dict containing registered modules
        self._modules = dict()  # type: typing.Dict[str, _DaxModuleBase]
        # A dict containing registered devices
        self._devices = dict()  # type: typing.Dict[str, typing.Tuple[typing.Any, _DaxHasSystem]]
        # A dict containing registered services
        self._services = dict()  # type: typing.Dict[str, DaxService]

    def add_module(self, module: __M_T) -> None:
        """Register a module.

        :param module: The module to register
        :raises NonUniqueRegistrationError: Raised if the module key was already registered by another module
        """

        assert isinstance(module, _DaxModuleBase), 'Module is not a DAX module base'

        # Get the module key
        key = module.get_system_key()

        # Get the module that registered the module key (None if the key is available)
        reg_module = self._modules.get(key)

        if reg_module is not None:
            # Key already in use by another module
            msg = 'Module key "{:s}" was already registered by module {:s}'.format(key, reg_module.get_identifier())
            raise self._NonUniqueRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._modules[key] = module

    @typing.overload
    def get_module(self, key: str) -> _DaxModuleBase:
        ...

    @typing.overload
    def get_module(self, key: str, type_: typing.Type[__M_T]) -> __M_T:
        ...

    def get_module(self, key: str, type_: typing.Type[_DaxModuleBase] = _DaxModuleBase) -> _DaxModuleBase:
        """Return the requested module by key.

        :param key: The key of the module
        :param type_: The expected type of the module
        :raises KeyError: Raised if the module could not be found
        :raises TypeError: Raised if the module type does not match the expected type
        """

        assert isinstance(key, str), 'Key must be a string'

        try:
            # Get the module
            module = self._modules[key]
        except KeyError:
            # Module was not found
            raise KeyError('Module "{:s}" could not be found'.format(key)) from None

        if not isinstance(module, type_):
            # Module does not have the correct type
            raise TypeError('Module "{:s}" does not match the expected type'.format(key))

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
        results = self.search_modules(type_)

        if not results:
            # No modules were found
            raise KeyError('Could not find modules with type "{:s}"'.format(type_.__name__))
        elif len(results) > 1:
            # More than one module was found
            raise LookupError('Could not find a unique module with type "{:s}"'.format(type_.__name__))

        # Return the only result
        _, module = results.popitem()
        return module

    def search_modules(self, type_: typing.Type[__M_T]) -> typing.Dict[str, __M_T]:
        """Search for modules that match the requested type and return results as a dict.

        :param type_: The type of the modules
        :return: A dict with key-module pairs
        """

        assert issubclass(type_, _DaxModuleBase), 'Provided type must be a subclass of DaxModuleBase'

        # Search for all modules matching the type
        results = {k: m for k, m in self._modules.items() if
                   isinstance(m, type_)}  # type: typing.Dict[str, _DaxNameRegistry.__M_T]

        # Return the list with results
        return results

    def get_module_key_list(self) -> typing.List[str]:
        """Return a list of registered module keys.

        :return: A list with module keys
        """

        module_key_list = natsort.natsorted(self._modules.keys())  # Natural sort the list
        return module_key_list

    def add_device(self, key: str, device: typing.Any, parent: _DaxHasSystem) -> None:
        """Register a device.

        Devices are added to the registry to ensure every device is only owned by a single parent.

        :param key: The unique key of the device
        :param device: The device object
        :param parent: The parent that requested the device
        :return: The requested device driver
        :raises NonUniqueRegistrationError: Raised if the device was already registered by another parent
        """

        assert isinstance(key, str), 'Device key must be a string'
        assert isinstance(parent, _DaxHasSystem), 'Parent is not a DaxHasSystem type'

        if key in _ARTIQ_VIRTUAL_DEVICES:
            return  # Virtual devices always have unique names are excluded from the registry

        # Get the device value object (None if the device was not registered before)
        device_value = self._devices.get(key)

        if device_value is not None:
            # Device was already registered
            _, reg_parent = device_value  # Unpack tuple
            msg = 'Device "{:s}" was already registered by parent "{:s}"'.format(key, reg_parent.get_system_key())
            raise self._NonUniqueRegistrationError(msg)

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
        """Return a list of registered device keys.

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
            raise ValueError('Invalid service name "{:s}"'.format(service_name))

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
        key = service.get_name()

        # Get the service that registered with the service name (None if key is available)
        reg_service = self._services.get(key)

        if reg_service is not None:
            # Service name was already registered
            raise self._NonUniqueRegistrationError('Service with name "{:s}" was already registered'.format(key))

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
        service_key = key if isinstance(key, str) else key.SERVICE_NAME

        # Try to return the requested service
        try:
            return self._services[service_key]
        except KeyError:
            # Service was not found
            raise KeyError('Service "{:s}" is not available'.format(service_key)) from None

    def get_service_key_list(self) -> typing.List[str]:
        """Return a list of registered service keys.

        :return: A list of service keys that were registered
        """

        service_key_list = natsort.natsorted(self._services.keys())  # Natural sort the list
        return service_key_list

    def find_interface(self, type_: typing.Type[__I_T]) -> __I_T:
        """Find a unique interface that matches the requested type.

        :param type_: The type of the interface
        :return: The unique interface of the requested type
        :raises KeyError: Raised if no interfaces of the desired type were found
        :raises LookupError: Raised if more then one interface of the desired type was found
        """

        # Search for all interfaces matching the type
        results = self.search_interfaces(type_)

        if not results:
            # No interfaces were found
            raise KeyError('Could not find interfaces with type "{:s}"'.format(type_.__name__))
        elif len(results) > 1:
            # More than one interface was found
            raise LookupError('Could not find a unique interface with type "{:s}"'.format(type_.__name__))

        # Return the only result
        _, interface = results.popitem()
        return interface

    def search_interfaces(self, type_: typing.Type[__I_T]) -> typing.Dict[str, __I_T]:
        """Search for interfaces that match the requested type and return results as a dict.

        :param type_: The type of the interfaces
        :return: A dict with key-interface pairs
        """

        assert issubclass(type_, DaxInterface), 'Provided type must be a subclass of DaxInterface'

        # Search for all modules and services matching the interface type
        iterator = itertools.chain(self._modules.values(), self._services.values())
        results = {itf.get_system_key(): itf for itf in iterator if isinstance(itf, type_)}

        # Return the list with results
        return results  # type: ignore


class _DaxDataStore:
    """Base class for the DAX data store.

    Data stores have methods that reflect the operations on ARTIQ
    datasets: set, mutate, and append.

    The base DAX data store does not store anything and can be used
    as a placeholder object since it is not an abstract base class.
    Other DAX data store classes can inherit from this class and
    override the :func:`set`, :func:`mutate`, and :func:`append` methods.
    """

    _DAX_COMMIT = None
    """DAX commit hash."""
    _CWD_COMMIT = None
    """Current working directory commit hash."""

    def __init__(self) -> None:  # Constructor return type required if no parameters are given
        # Create a logger object
        self._logger = logging.getLogger('{:s}.{:s}'.format(self.__module__, self.__class__.__name__))

    def set(self, key: str, value: typing.Any) -> None:
        """Write a key-value into the data store.

        :param key: The key of the value
        :param value: The value to store
        """
        self._logger.debug('Set key "{:s}" to value: "{}"'.format(key, value))

    def mutate(self, key: str, index: typing.Any, value: typing.Any) -> None:
        """Mutate a specific index of a key-value in the data store.

        :param key: The key of the value
        :param index: The index to mutate
        :param value: The value to store
        """
        self._logger.debug('Mutate key "{:s}"[{}] to value "{}"'.format(key, index, value))

    def append(self, key: str, value: typing.Any) -> None:
        """Append a value to a key-value in the data store.

        :param key: The key of the value
        :param value: The value to append
        """
        self._logger.debug('Append key "{:s}" with value "{}"'.format(key, value))


def __load_commit_hashes() -> None:
    """Load commit hash class attributes, only needs to be done once."""

    # Obtain commit hash of DAX
    try:
        # Obtain repo
        repo = git.Repo(os.path.dirname(__file__), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        # No repo was found
        pass
    else:
        # Get commit hash
        _DaxDataStore._DAX_COMMIT = repo.head.commit.hexsha

    # Obtain commit hash of current working directory (if existing)
    try:
        # Obtain repo
        repo = git.Repo(os.getcwd(), search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        # No repo was found
        pass
    else:
        # Get commit hash
        _DaxDataStore._CWD_COMMIT = repo.head.commit.hexsha


__load_commit_hashes()  # Load commit hashes into class attributes
del __load_commit_hashes  # Remove one-time function


class _DaxDataStoreInfluxDb(_DaxDataStore):
    """Influx DB DAX data store class."""

    __F_T = typing.Union[numbers.Real, str]  # Field type variable for Influx DB supported types
    __P_T = typing.Dict[str, typing.Union[str, typing.Union[typing.Dict[str, __F_T]]]]  # Point type variable

    _FIELD_TYPES = (numbers.Real, str)  # numbers.Real includes bool, float, int, and NumPy int (for runtime checks)
    """Legal field types for Influx DB."""

    def __init__(self, system: DaxSystem, key: str):
        """Create a new DAX data store that uses an Influx DB backend.

        :param system: The system this data store is managed by
        :param key: The key of the DAX Influx DB controller
        """

        assert isinstance(system, DaxSystem), 'System parameter must be of type DaxSystem'
        assert isinstance(key, str), 'The Influx DB controller key must be of type str'

        # Call super
        super(_DaxDataStoreInfluxDb, self).__init__()

        # Get the Influx DB driver, this call can raise various exceptions
        self._get_driver(system, key)

        # Get the scheduler, which is a virtual device
        scheduler = system.get_device('scheduler')
        if isinstance(scheduler, artiq.master.worker_db.DummyDevice):
            return  # ARTIQ is only discovering experiment classes, do not continue initialization

        # Store values that will be used for data points later
        self._sys_id = system.SYS_ID
        # Initialize index table for the append function, required to emulate appending behavior
        self._index_table = dict()  # type: typing.Dict[str, int]

        # Prepare base tags
        self._base_tags = {
            'system_version': str(system.SYS_VER),  # Convert int version to str since tags are strings
        }  # type: typing.Dict[str, str]

        # Prepare base fields
        self._base_fields = {
            'rid': int(scheduler.rid),
            'pipeline_name': str(scheduler.pipeline_name),
            'priority': int(scheduler.priority),
            'artiq_version': str(artiq.__version__),
        }

        # Add expid items to fields if keys do not exist yet and the types are appropriate
        self._base_fields.update((k, v) for k, v in scheduler.expid.items()
                                 if k not in self._base_fields and isinstance(v, self._FIELD_TYPES))

        # Add commit hashes to fields
        if self._DAX_COMMIT is not None:
            self._base_fields['dax_commit'] = self._DAX_COMMIT
        if self._CWD_COMMIT is not None:
            self._base_fields['cwd_commit'] = self._CWD_COMMIT

        # Debug message
        self._logger.debug('Initialized base fields: {}'.format(self._base_fields))

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
                self._write_points([self._make_point(key, v, i) for v, i in zip(value, itertools.count(0))])
            # Store the length of the sequence for emulated appending later
            self._index_table[key] = len(value)
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning('Could not store value for key "{:s}", unsupported value type '
                                 'for value "{}"'.format(key, value))

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
                self._logger.warning('Could not mutate value for key "{:s}", multi-dimensional index "{}" '
                                     'not supported'.format(key, index))
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning('Could not mutate value for key "{:s}", unsupported value type '
                                 'for value "{}"'.format(key, value))

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
                self._logger.warning('Could not append value for key "{:s}", no index was cached '
                                     'and the append operation could not be emulated'.format(key))
        else:
            # Unsupported type, do not raise but warn user instead
            self._logger.warning('Could not append value for key "{:s}", unsupported value type '
                                 'for value "{}"'.format(key, value))

    def _make_point(self, key: str, value: __F_T, index: typing.Union[None, int, numbers.Integral] = None) -> __P_T:
        """Make a point object from a key-value pair, optionally with an index.

        This function does not check the type of the value and the index, which should be checked before.
        Numpy integers are automatically converted to Python int.
        """

        assert isinstance(key, str), 'Key should be of type str'

        if not _is_valid_key(key):
            # Invalid key
            raise ValueError('Influx DB data store received an invalid key "{:s}"'.format(key))

        if isinstance(value, np.integer):
            # Convert Numpy int to Python int
            value = int(value)

        # Copy the base tags and fields
        tags = self._base_tags.copy()
        fields = self._base_fields.copy()

        # Split the key
        split_key = key.rsplit(_KEY_SEPARATOR, maxsplit=1)
        base = split_key[0] if len(split_key) == 2 else ''  # Base is empty if the key does not split

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
        return point  # type: ignore

    def _get_driver(self, system: DaxSystem, key: str) -> None:
        """Get the required driver.

        This method was separated to allow testing without writing points.
        """
        self._influx = system.get_device(key)  # Get the Influx DB driver, this call can raise various exceptions

    def _write_points(self, points: typing.List[__P_T]) -> None:
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
    def wrapper(system_type: typing.Type[__DCF_S_T]) -> typing.Type[__DCF_C_T]:
        """Create a new DAX client class.

        This factory function will create a new client class for a given system type.

        :param system_type: The system type used by the client
        :return: A fusion of the client and system class which can be executed
        :raises TypeError: Raised if the provided `system_type` parameter is not a subclass of `DaxSystem`
        """

        # Check the system type
        assert isinstance(system_type, type), 'System type must be a type'
        if not issubclass(system_type, DaxSystem):
            raise TypeError('System type must be a subclass of DaxSystem')

        class WrapperClass(c, system_type):  # type: ignore
            """The wrapper class that finalizes the client class.

            The wrapper class first inherits from the client and then the system,
            as if the client class was an extension of the system, just as a
            custom experiment.
            """

            def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
                # First build the system (not using MRO) to fill the registry
                system_type.build(self)

                # Then build the client which can use the registry
                c.build(self, *args, **kwargs)

            def run(self) -> None:
                # Now we are a DAX system, we need to initialize
                self.dax_init()
                # Call the run method (of the client class which is unaware of the system it is now)
                super(WrapperClass, self).run()

        # The factory function returns the newly constructed wrapper class
        return WrapperClass

    # Return the factory function
    return wrapper
