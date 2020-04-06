from __future__ import annotations  # Required for postponed evaluation of annotations (Python 3.x)

import abc
import logging
import functools
import re
import natsort
import typing
import git  # type: ignore
import os
import numbers

import artiq
import artiq.experiment
import artiq.master.worker_db  # type: ignore

import artiq.coredevice.core  # type: ignore
import artiq.coredevice.dma  # type: ignore
import artiq.coredevice.cache  # type: ignore

# Key separator
_KEY_SEPARATOR: str = '.'
# Regex for matching valid names
_NAME_RE: typing.Pattern[str] = re.compile(r'\w+')


def _is_valid_name(name: str) -> bool:
    """Return true if the given name is valid."""
    assert isinstance(name, str), 'The given name should be a string'
    return bool(_NAME_RE.fullmatch(name))


def _is_valid_key(key: str) -> bool:
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return all(_NAME_RE.fullmatch(n) for n in key.split(_KEY_SEPARATOR))


class _DaxBase(artiq.experiment.HasEnvironment, abc.ABC):
    """Base class for all DAX core classes."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        # Logger object
        self.logger: logging.Logger = logging.getLogger(self.get_identifier())

        # Build
        self.logger.debug('Starting build...')
        try:
            # Call super, which will call build()
            super(_DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
        except (TypeError, LookupError) as e:  # TypeError includes signature mismatch errors for build()
            # Log the exception to provide more context
            self.logger.exception(e)
            # Raise a different exception type to prevent that the caught exception is logged again by the parent
            raise RuntimeError(e) from e
        else:
            self.logger.debug('Build finished')

    @artiq.experiment.host_only
    def update_kernel_invariants(self, *keys: str) -> None:
        """Add one or more keys to the kernel invariants set."""

        assert all(isinstance(k, str) for k in keys), 'All keys must be of type str'

        # Get kernel invariants using getattr() such that we do not overwrite a user-defined variable
        kernel_invariants: typing.Set[str] = getattr(self, "kernel_invariants", set())
        # Update the set with the given keys
        self.kernel_invariants: typing.Set[str] = kernel_invariants | {*keys}

    @abc.abstractmethod
    def get_identifier(self) -> str:
        pass


class _DaxHasSystem(_DaxBase, abc.ABC):
    """Intermediate base class for DAX classes that are dependent on a DAX system."""

    # Device type verification
    __D_T = typing.Optional[typing.Union[type, typing.Tuple[type, ...]]]

    # Attribute names of core devices
    __CORE_DEVICES: typing.List[str] = ['core', 'core_dma', 'core_cache']
    # Attribute names of core objects created in build() or inherited from parents
    __CORE_ATTRIBUTES: typing.List[str] = __CORE_DEVICES + ['data_store']

    def __init__(self, managers_or_parent: typing.Any, name: str, system_key: str, registry: _DaxNameRegistry,
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
        self._name: str = name
        self._system_key: str = system_key
        self.registry: _DaxNameRegistry = registry

        # Call super, which will result in a call to build()
        super(_DaxHasSystem, self).__init__(managers_or_parent, *args, **kwargs)

        # Verify that all core attributes are available
        if not all(hasattr(self, n) for n in self.__CORE_ATTRIBUTES):
            msg = 'Missing core attributes (super.build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make core devices kernel invariants
        self.update_kernel_invariants(*self.__CORE_DEVICES)

    def _take_parent_core_attributes(self, parent: _DaxHasSystem) -> None:
        """Take core attributes from parent.

        If this object does not construct its own core attributes, it should take them from their parent.
        """
        try:
            # Take core attributes from parent, attributes are taken one by one to allow typing
            self.core: artiq.coredevice.core = parent.core
            self.core_dma: artiq.coredevice.dma = parent.core_dma
            self.core_cache: artiq.coredevice.cache = parent.core_cache
            self.data_store: _DaxDataStoreConnector = parent.data_store
        except AttributeError:
            parent.logger.exception('Missing core attributes (super.build() was probably not called)')
            raise

    @artiq.experiment.host_only
    def get_name(self) -> str:
        """Get the name."""
        return self._name

    @artiq.experiment.host_only
    def get_system_key(self, *keys: str) -> str:
        """Get the full key based on the system key."""

        assert all(isinstance(k, str) for k in keys), 'Keys must be strings'

        # Check if the given keys are valid
        for k in keys:
            if not _is_valid_key(k):
                raise ValueError('Invalid key "{:s}"'.format(k))

        # Return the assigned key
        return _KEY_SEPARATOR.join([self._system_key, *keys])

    @artiq.experiment.host_only
    def get_registry(self) -> _DaxNameRegistry:
        """Return the current registry."""
        return self.registry

    @artiq.experiment.host_only
    def _init_system(self) -> None:
        """Initialize the DAX system, for dataset access, device initialization, and recording DMA traces."""

        self.logger.debug('Initializing...')
        # Initialize all children
        self.call_child_method('_init_system')
        # Initialize this object
        self.init()
        self.logger.debug('Initialization finished')

    @artiq.experiment.host_only
    def _post_init_system(self) -> None:
        """DAX system post-initialization (e.g. obtaining DMA handles)."""

        self.logger.debug('Post-initializing...')
        # Post-initialize all children
        self.call_child_method('_post_init_system')
        # Post-initialize this object
        self.post_init()
        self.logger.debug('Post-initialization finished')

    @abc.abstractmethod
    def init(self) -> None:
        """Override this method to access the dataset (r/w), initialize devices, and record DMA traces."""
        pass

    @abc.abstractmethod
    def post_init(self) -> None:
        """Override this method for post-initialization procedures (e.g. obtaining DMA handles)."""
        pass

    def get_device(self, key: str, type_: __D_T = object) -> typing.Any:
        """Get a device driver."""

        assert isinstance(key, str) and key, 'Key must be of type str and not empty'

        # Debug message
        self.logger.debug('Requesting device "{:s}"'.format(key))

        # Register the requested device
        self.registry.add_device(self, key)

        # Get the device
        device: typing.Any = super(_DaxHasSystem, self).get_device(key)

        # Check device type
        if not isinstance(device, artiq.master.worker_db.DummyDevice) and not isinstance(device, type_):
            # Device has an unexpected type
            raise TypeError('Device "{:s}" requested by "{:s}" does not match the '
                            'expected type'.format(key, self.get_system_key()))

        # Return the device
        return device

    def setattr_device(self, key: str, attr_name: typing.Optional[str] = None, type_: __D_T = object) -> None:
        """Sets a device driver as attribute."""

        assert isinstance(attr_name, str) or attr_name is None, 'Attribute name must be of type str or None'

        # Get the device
        device: typing.Any = self.get_device(key, type_)

        if attr_name is None:
            # Set attribute name to key if no attribute name was given
            attr_name = key

        # Set the device key to the attribute
        if not _is_valid_name(attr_name):
            raise ValueError('Attribute name {:s} not valid'.format(attr_name))
        setattr(self, attr_name, device)

        # Add attribute to kernel invariants
        self.update_kernel_invariants(attr_name)

    @artiq.experiment.rpc(flags={'async'})
    def set_dataset_sys(self, key: str, value: typing.Any) -> None:
        """Sets the contents of a system dataset."""

        assert isinstance(key, str), 'Key must be of type str'

        # Modify logging level of worker_db logger to suppress an unwanted warning message
        artiq.master.worker_db.logger.setLevel(logging.WARNING + 1)

        # Set value in system dataset with extra flags
        self.logger.debug('System dataset key "{:s}" set to value "{}"'.format(key, value))
        self.set_dataset(self.get_system_key(key), value, broadcast=True, persist=True, archive=True)

        # Restore original logging level of worker_db logger
        artiq.master.worker_db.logger.setLevel(logging.NOTSET)

    @artiq.experiment.rpc(flags={'async'})
    def mutate_dataset_sys(self, key: str, index: int, value: typing.Any) -> None:
        """Mutate an existing system dataset at the given index."""

        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(index, int), 'Index must be of type int'

        # Mutate system dataset
        self.logger.debug('System dataset key "{:s}"[{:d}] mutate to value "{}"'.format(key, index, value))
        self.mutate_dataset(self.get_system_key(key), index, value)

    @artiq.experiment.rpc(flags={'async'})
    def append_to_dataset_sys(self, key: str, value: typing.Any) -> None:
        """Append a value to a system dataset."""

        assert isinstance(key, str), 'Key must be of type str'

        # Append value to system dataset
        self.logger.debug('System dataset key "{:s}" append value "{}"'.format(key, value))
        self.append_to_dataset(self.get_system_key(key), value)

    def get_dataset_sys(self, key: str, default: typing.Any = artiq.experiment.NoDefault) -> typing.Any:
        """Returns the contents of a system dataset.

        If the key is present, its value will be returned.
        If the key is not present and no default is provided, a KeyError will be raised.
        If the key is not present and a default is provided, the default value will be returned.
        """

        assert isinstance(key, str), 'Key must be of type str'

        # Get the full system key
        system_key = self.get_system_key(key)

        try:
            # Get value from system dataset with extra flags
            value: typing.Any = self.get_dataset(system_key, default, archive=True)
        except KeyError:
            # The key was not found
            raise KeyError('System dataset key "{:s}" not found'.format(system_key)) from None
        else:
            self.logger.debug('System dataset key "{:s}" returned value "{}"'.format(key, value))

        # Return value
        return value

    def setattr_dataset_sys(self, key: str, default: typing.Any = artiq.experiment.NoDefault,
                            kernel_invariant: bool = True) -> None:
        """Sets the contents of a system dataset as attribute.

        If the key is present, its value will be loaded to the attribute.
        If the key is not present and no default is provided, the attribute is not set.
        If the key is not present and a default is provided, the default value will
        be written to the dataset and the attribute will be set to the same value.
        """

        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(kernel_invariant, bool), 'Kernel invariant flag must be of type bool'

        # Obtain system key
        system_key: str = self.get_system_key(key)

        try:
            # Get the value from system dataset
            value: typing.Any = self.get_dataset(system_key, archive=True)
        except KeyError:
            if default is artiq.experiment.NoDefault:
                # The value was not available in the system dataset and no default was provided
                self.logger.debug('System attribute "{:s}" not set'.format(key))
                return
            else:
                # If the value does not exist, write the default value to the system dataset, but do not archive yet
                self.logger.debug('System dataset key "{:s}" set to default value "{}"'.format(key, default))
                self.set_dataset(system_key, default, broadcast=True, persist=True, archive=False)
                # Get the value again and make sure it is archived
                value = self.get_dataset(system_key, archive=True)  # Should never raise a KeyError

        # Set value as an attribute
        setattr(self, key, value)

        if kernel_invariant:
            # Update kernel invariants
            self.update_kernel_invariants(key)

        # Debug message
        self.logger.debug('System attribute "{:s}" set to value "{}"{:s}'.format(
            key, value, ' (kernel invariant)' if kernel_invariant else ''))

    def hasattr(self, *keys: str) -> bool:
        """Returns if this object has the given attributes."""
        return all(hasattr(self, k) for k in keys)

    @artiq.experiment.host_only
    def get_identifier(self) -> str:
        """Return the system key with the class name."""
        return '[{:s}]({:s})'.format(self.get_system_key(), self.__class__.__name__)


class _DaxModuleBase(_DaxHasSystem, abc.ABC):
    """Base class for all DAX modules and systems."""

    def __init__(self, managers_or_parent: typing.Any, module_name: str, module_key: str, registry: _DaxNameRegistry,
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
        """Initialize the DAX module."""

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
                                        managers_or_parent.get_registry(), *args, **kwargs)


class DaxModuleInterface(abc.ABC):
    """Base class for module interfaces."""
    pass


class DaxSystem(_DaxModuleBase):
    """Base class for DAX systems, which is a top-level module."""

    # Identifier of the system
    SYS_ID: str
    # Version of the system
    SYS_VER: int

    # System name, used as top key for modules
    SYS_NAME: str = 'system'
    # System services, used as top key for services
    SYS_SERVICES: str = 'services'

    # Keys of core devices
    CORE_KEY: str = 'core'
    CORE_DMA_KEY: str = 'core_dma'
    CORE_CACHE_KEY: str = 'core_cache'
    # Key of core log controller
    CORE_LOG_KEY: str = 'core_log'

    # System keys
    DAX_INIT_TIME_KEY = 'dax_init_time'

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        # Check if system ID was overridden
        assert hasattr(self, 'SYS_ID'), 'Every DAX system class must have a SYS_ID class attribute'
        assert isinstance(self.SYS_ID, str), 'System ID must be of type str'
        assert _is_valid_name(self.SYS_ID), 'Invalid system ID "{:s}"'.format(self.SYS_ID)

        # Check if system version was overridden
        assert hasattr(self, 'SYS_VER'), 'Every DAX system class must have a SYS_VER class attribute'
        assert isinstance(self.SYS_VER, int), 'System version must be of type int'
        assert self.SYS_VER >= 0, 'Invalid system version, version number must be positive'

        # Call super, add names, add a new registry
        super(DaxSystem, self).__init__(managers_or_parent, self.SYS_NAME, self.SYS_NAME, _DaxNameRegistry(self),
                                        *args, **kwargs)

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Override this method to build your DAX system. (Do not forget to call super.build() first!)"""

        if args or kwargs:
            # Warn if we find any dangling arguments
            self.logger.warning('Unused args "{}" / kwargs "{}" were passed to super.build()'.format(args, kwargs))

        # Core devices
        self.core: artiq.coredevice.core = self.get_device(self.CORE_KEY, artiq.coredevice.core.Core)
        self.core_dma: artiq.coredevice.dma = self.get_device(self.CORE_DMA_KEY, artiq.coredevice.dma.CoreDMA)
        self.core_cache: artiq.coredevice.cache = self.get_device(self.CORE_CACHE_KEY, artiq.coredevice.cache.CoreCache)

        # Verify existence of core log controller
        try:
            # Register the core log controller with the system
            self.registry.add_device(self, self.CORE_LOG_KEY)
        except LookupError:
            # Core log controller was not found in the device DB
            self.logger.warning('Core log controller "{:s}" not found in device DB'.format(self.CORE_LOG_KEY))

        # Instantiate the data store connector (needs to be done in build() since it requests a controller)
        self.data_store: _DaxDataStoreConnector = _DaxDataStoreConnector(self)

    def dax_init(self) -> None:
        """Initialize the DAX system."""
        self.logger.debug('Starting DAX system initialization...')
        try:
            self._init_system()
            self._post_init_system()
        except artiq.coredevice.core.CompileError:
            self.logger.exception('Compilation error occurred during DAX system initialization')
            raise
        else:
            self.logger.debug('Finished DAX system initialization')

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass


class DaxService(_DaxHasSystem, abc.ABC):
    """Base class for system services."""

    # The unique name of this service
    SERVICE_NAME: str

    def __init__(self, managers_or_parent: typing.Union[DaxSystem, DaxService],
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the DAX service base class."""

        # Check if service name was overridden
        assert hasattr(self, 'SERVICE_NAME'), 'Every DAX service class must have a SERVICE_NAME class attribute'
        assert isinstance(self.SERVICE_NAME, str), 'Service name must be of type str'

        # Check parent type
        if not isinstance(managers_or_parent, (DaxSystem, DaxService)):
            raise TypeError('Parent of service "{:s}" is not a DAX system or service'.format(self.SERVICE_NAME))

        # Take core attributes from parent
        self._take_parent_core_attributes(managers_or_parent)

        # Use name registry of parent to obtain a system key
        registry: _DaxNameRegistry = managers_or_parent.get_registry()
        system_key: str = registry.make_service_key(self.SERVICE_NAME)

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

    class NonUniqueRegistrationError(LookupError):
        """Exception when a name is registered more then once."""
        pass

    # Module base type variable
    __M_T = typing.TypeVar('__M_T', bound=_DaxModuleBase)

    # List of ARTIQ virtual devices (not available in the device DB)
    _VIRTUAL_DEVICES: typing.Set[str] = {'scheduler', 'ccb'}

    def __init__(self, system: DaxSystem):
        """Create a new DAX name registry."""

        assert isinstance(system, DaxSystem), 'System must be of type DAX system'

        # Check system services key
        if not _is_valid_key(system.SYS_SERVICES):
            raise ValueError('Invalid system services key "{:s}"'.format(system.SYS_SERVICES))

        # Store system services key
        self._sys_services_key: str = system.SYS_SERVICES  # Access attribute directly

        # A dict containing registered modules
        self._modules: typing.Dict[str, _DaxModuleBase] = dict()
        # A dict containing registered devices and the parents that registered them
        self._devices: typing.Dict[str, _DaxHasSystem] = dict()
        # A dict containing registered services
        self._services: typing.Dict[str, DaxService] = dict()

    def add_module(self, module: _DaxModuleBase) -> None:
        """Register a module."""

        assert isinstance(module, _DaxModuleBase), 'Module is not a DAX module base'

        # Get the module key
        key: str = module.get_system_key()

        # Get the module that registered the module key (None if the key is available)
        reg_module: typing.Optional[_DaxModuleBase] = self._modules.get(key)

        if reg_module:
            # Key already in use by an other module
            msg = 'Module key "{:s}" was already registered by module {:s}'.format(key, reg_module.get_identifier())
            raise self.NonUniqueRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._modules[key] = module

    @typing.overload
    def get_module(self, key: str) -> _DaxModuleBase:
        ...

    @typing.overload
    def get_module(self, key: str, type_: typing.Type[__M_T]) -> __M_T:
        ...

    def get_module(self, key: str, type_: typing.Type[_DaxModuleBase] = _DaxModuleBase) -> _DaxModuleBase:
        """Return the requested module by key."""

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
        """Find a unique module that matches the requested type, raise otherwise."""

        # Search for all modules matching the type
        results: typing.Dict[str, _DaxNameRegistry.__M_T] = self.search_modules(type_)

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
        """Search for modules that match the requested type and return results as a dict."""

        assert issubclass(type_, (DaxModuleInterface, _DaxModuleBase)), \
            'Provided type must be a DAX module base or interface'

        # Search for all modules matching the type
        results: typing.Dict[str, _DaxNameRegistry.__M_T] = {k: m for k, m in self._modules.items()
                                                             if isinstance(m, type_)}

        # Return the list with results
        return results

    def get_module_key_list(self) -> typing.List[str]:
        """Return a list of registered module keys."""
        module_key_list: typing.List[str] = natsort.natsorted(self._modules.keys())  # Natural sort the list
        return module_key_list

    def add_device(self, parent: _DaxHasSystem, key: str) -> None:
        """Register a device."""

        assert isinstance(parent, _DaxHasSystem), 'Parent is not a DaxHasSystem type'
        assert isinstance(key, str), 'Device key must be a string'

        if key in self._VIRTUAL_DEVICES:
            return  # Virtual devices always have unique names are excluded from the registry

        try:
            # Get the unique key
            unique: str = self._get_unique_device_key(parent.get_device_db(), key, set())
        except (LookupError, TypeError) as e:
            # Device was not found in the device DB
            raise KeyError('Device "{:s}" could not be found in the device DB'.format(key)) from e

        # Get the parent that registered the device (None if the device was not registered before)
        reg_parent: typing.Optional[_DaxHasSystem] = self._devices.get(unique)

        if reg_parent:
            # Device was already registered
            device_name = '"{:s}"'.format(key) if key == unique else '"{:s}" ({:s})'.format(key, unique)
            parent_name = reg_parent.get_system_key()
            msg = 'Device {:s} was already registered by parent "{:s}"'.format(device_name, parent_name)
            raise self.NonUniqueRegistrationError(msg)

        # Add unique device key to the dict of registered devices
        self._devices[unique] = parent

    def _get_unique_device_key(self, d: typing.Dict[str, typing.Any], key: str,
                               trace: typing.Set[str]) -> str:
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
        v: typing.Any = d[key]

        if isinstance(v, str):
            # Recurse if we are still dealing with an alias
            return self._get_unique_device_key(d, v, trace)
        elif isinstance(v, dict):
            # We reached a dict, key must be the unique key
            return key
        else:
            # We ended up with an unexpected type
            raise TypeError('Key "{:s}" returned an unexpected type'.format(key))

    def get_device_key_list(self) -> typing.List[str]:
        """Return a list of registered device keys."""
        device_key_list: typing.List[str] = natsort.natsorted(self._devices.keys())  # Natural sort the list
        return device_key_list

    def make_service_key(self, service_name: str) -> str:
        """Return the system key for a service name."""

        # Check the given name
        assert isinstance(service_name, str), 'Service name must be a string'
        if not _is_valid_name(service_name):
            # Service name not valid
            raise ValueError('Invalid service name "{:s}"'.format(service_name))

        # Return assigned key
        return _KEY_SEPARATOR.join([self._sys_services_key, service_name])

    def add_service(self, service: DaxService) -> None:
        """Register a service."""

        assert isinstance(service, DaxService), 'Service must be a DAX service'

        # Services get indexed by name
        key: str = service.get_name()

        # Get the service that registered with the service name (None if key is available)
        reg_service: typing.Optional[DaxService] = self._services.get(key)

        if reg_service:
            # Service name was already registered
            raise self.NonUniqueRegistrationError('Service with name "{:s}" was already registered'.format(key))

        # Add service to the registry
        self._services[key] = service

    def has_service(self, key: typing.Union[str, typing.Type[DaxService]]) -> bool:
        """Return if service is available."""
        try:
            self.get_service(key)
        except KeyError:
            return False
        else:
            return True

    def get_service(self, key: typing.Union[str, typing.Type[DaxService]]) -> DaxService:
        """Get a service from the registry."""

        assert isinstance(key, str) or issubclass(key, DaxService)

        # Figure the right key
        key = key if isinstance(key, str) else key.SERVICE_NAME

        # Try to return the requested service
        try:
            return self._services[key]
        except KeyError:
            # Service was not found
            raise KeyError('Service "{:s}" is not available') from None

    def get_service_key_list(self) -> typing.List[str]:
        """Return a list of registered service keys."""
        service_key_list: typing.List[str] = natsort.natsorted(self._services.keys())  # Natural sort the list
        return service_key_list


class _DaxDataStoreConnector:
    """Connector class for the DAX data store."""

    # Field type variable
    __F_T = typing.Union[numbers.Integral, float, bool, str]  # Supported types for fields (Integral for NumPy ints)
    # Point type variable
    __P_T = typing.Dict[str, typing.Union[str, typing.Union[typing.Dict[str, __F_T], typing.Dict[str, str]]]]

    # Legal field types
    _FIELD_TYPES: typing.Tuple[type, ...] = (int, float, bool, str)

    # DAX commit hash
    _DAX_COMMIT: str
    # Current working directory commit hash
    _CWD_COMMIT: str

    def __init__(self, system: DaxSystem):
        """Create a new DAX data store connector."""

        assert isinstance(self._DAX_COMMIT, str), 'DAX commit hash was not loaded'
        assert isinstance(self._CWD_COMMIT, str), 'Current working directory commit hash was not loaded'

        # Store values that will be used for data points
        self._sys_id: str = system.SYS_ID
        self._sys_ver: str = str(system.SYS_VER)  # Convert int version to str since tags are strings

        # Get the scheduler, which is a virtual device
        self._scheduler: typing.Any = system.get_device('scheduler')

        # todo, obtain access to the data store controller using system.get_device()

    def store(self, key: str, value: __F_T) -> None:
        """Write a single key-value into the data store."""

        # Make a dict with a single key-value pair and store it
        self.store_dict({key: value})

    def store_dict(self, d: typing.Dict[str, __F_T]) -> None:
        """Write a dict with key-value pairs into the data store."""

        # Convert NumPy int values to Python int
        d = {k: int(v) if isinstance(v, numbers.Integral) else v for k, v in d.items()}

        # Check if all keys and values are valid and have supported types
        for k, v in d.items():
            if not _is_valid_key(k):
                # Invalid key
                raise ValueError('The data store received an invalid key "{:s}"'.format(k))
            if not isinstance(v, self._FIELD_TYPES):
                # Unsupported value type
                raise TypeError('The data store can not store value "{}" of type {:s}'.format(v, str(type(v))))

        # Make a point object for every key-value pair
        points: typing.List[_DaxDataStoreConnector.__P_T] = [self._make_point(k, v) for k, v in d.items()]

        # Write points to the data store
        self._write_points(points)

    def _make_point(self, key: str, value: __F_T) -> __P_T:
        """Make a point object from a key-value pair."""

        # Split the key
        split_key: typing.List[str] = key.rsplit(_KEY_SEPARATOR, maxsplit=1)
        base: str = split_key[0] if len(split_key) == 2 else ''  # Base is empty if the key does not split

        # Tags
        tags: typing.Dict[str, str] = {
            'system_version': self._sys_ver,
            'base': base,
        }

        # Fields
        fields: typing.Dict[str, _DaxDataStoreConnector.__F_T] = {
            'dax_commit': self._DAX_COMMIT,
            'cwd_commit': self._CWD_COMMIT,
            'rid': int(self._scheduler.rid),
            'pipeline_name': str(self._scheduler.pipeline_name),
            'priority': int(self._scheduler.priority),
            'artiq_version': str(artiq.__version__),
            key: value,  # The full key and the value are the actual field
        }
        # Add expid items to fields if keys do not exist yet and the types are appropriate
        fields.update((k, v) for k, v in self._scheduler.expid.items()
                      if k not in fields and isinstance(v, self._FIELD_TYPES))

        # Create the point object
        point: _DaxDataStoreConnector.__P_T = {
            'measurement': self._sys_id,
            'tags': tags,
            'fields': fields,
        }

        # Return point
        return point

    def _write_points(self, points: typing.List[__P_T]) -> None:
        pass  # todo, write points to the actual data store using the controller

    @classmethod
    def load_commit_hashes(cls) -> None:
        """Load commit hash class attributes, only needs to be done once."""

        # Obtain commit hash of DAX
        try:
            # Obtain repo
            repo: git.Repo = git.Repo(os.path.dirname(__file__), search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            # No repo was found, use empty commit hash
            cls._DAX_COMMIT = ''
        else:
            # Get commit hash
            cls._DAX_COMMIT = repo.head.commit.hexsha

        # Obtain commit hash of current working directory (if existing)
        try:
            # Obtain repo
            repo = git.Repo(os.getcwd(), search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            # No repo was found, use empty commit hash
            cls._CWD_COMMIT = ''
        else:
            # Get commit hash
            cls._CWD_COMMIT = repo.head.commit.hexsha


# Load commit hashes into class attributes
_DaxDataStoreConnector.load_commit_hashes()

# Type variable for dax_client_factory() decorator c (client) argument
__C_T = typing.TypeVar('__C_T', bound=DaxClient)
# Type variable for dax_client_factory() system_type argument
__S_T = typing.TypeVar('__S_T', bound=DaxSystem)


def dax_client_factory(c: typing.Type[__C_T]) -> typing.Callable[[typing.Type[__S_T]], typing.Type[__C_T]]:
    """Decorator to convert a DaxClient class to a factory function for that class."""

    assert isinstance(c, type), 'The decorated object must be a class'
    assert issubclass(c, DaxClient), 'The decorated class must be a subclass of DaxClient'

    # Use the wraps decorator, but do not inherit the docstring
    @functools.wraps(c, assigned=[e for e in functools.WRAPPER_ASSIGNMENTS if e != '__doc__'])
    def wrapper(system_type: typing.Type[__S_T]) -> typing.Type[__C_T]:
        """Create a new DAX client class.

        This factory function will create a new client class for a given system type.

        :param system_type: The system type used by the client.
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
