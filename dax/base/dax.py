from __future__ import annotations  # Required for postponed evaluation of annotations (Python 3.x)

import abc
import logging
import functools
import re
import natsort  # type: ignore
import typing

import artiq.master.worker_db  # type: ignore
import artiq.experiment  # type: ignore

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


class _DaxBase(artiq.experiment.HasEnvironment, abc.ABC):  # type: ignore
    """Base class for all DAX core classes."""

    class BuildArgumentError(TypeError):
        """Exception for build arguments not matching an expected signature."""
        pass

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        # Logger object
        self.logger: logging.Logger = logging.getLogger(self.get_identifier())

        # Build
        self.logger.debug('Starting build...')
        try:
            # Call super, which will call build()
            super(_DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
        except self.BuildArgumentError as e:
            # Log the error message
            self.logger.error(str(e))
            raise
        except TypeError as e:
            msg = 'Build arguments do not match the expected signature: {:s}'.format(str(e))
            self.logger.error(msg)
            raise self.BuildArgumentError(msg) from e
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

    def __init__(self, managers_or_parent: typing.Any, name: str, system_key: str, registry: _DaxNameRegistry,
                 *args: typing.Any, **kwargs: typing.Any):
        assert isinstance(name, str), 'Name must be a string'
        assert isinstance(system_key, str), 'System key must be a string'
        assert isinstance(registry, _DaxNameRegistry), 'Registry must be a DAX name registry'

        # Check name and system key
        if not _is_valid_name(name):
            raise ValueError('Invalid name "{:s}" for class {:s}'.format(name, self.__class__.__name__))
        if not _is_valid_key(system_key) or not system_key.endswith(name):
            raise ValueError('Invalid system_key "{:s}" for class {:s}'.format(system_key, self.__class__.__name__))

        # Store attributes
        self._name: str = name
        self._system_key: str = system_key
        self.registry: _DaxNameRegistry = registry

        # Call super, which will result in a call to build()
        super(_DaxHasSystem, self).__init__(managers_or_parent, *args, **kwargs)

        # Verify that all core devices are available
        if not all(hasattr(self, n) for n in self.__CORE_DEVICES):
            msg = 'Missing core devices (super.build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make core devices kernel invariants
        self.update_kernel_invariants(*self.__CORE_DEVICES)

    @artiq.experiment.host_only
    def get_name(self) -> str:
        """Get the name."""
        return self._name

    @artiq.experiment.host_only
    def get_system_key(self, key: typing.Optional[str] = None) -> str:
        """Get the full key based on the system key."""

        if key is None:
            # No key provided, just return the system key
            return self._system_key

        else:
            assert isinstance(key, str), 'Key must be a string'

            # Check if the given key is valid
            if not _is_valid_key(key):
                msg = 'Invalid key "{:s}"'.format(key)
                self.logger.error(msg)
                raise ValueError(msg)

            # Return the assigned key
            return _KEY_SEPARATOR.join([self._system_key, key])

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
            msg = 'Device "{:s}" does not match the expected type'.format(key)
            self.logger.error(msg)
            raise TypeError(msg)

        # Return the device
        return device

    def setattr_device(self, key: str, attr_name: typing.Optional[str] = None, type_: __D_T = object) -> None:
        """Sets a device driver as attribute."""

        # Get the device
        device: typing.Any = self.get_device(key, type_)

        if attr_name is None:
            # Set attribute name to key if no attribute name was given
            attr_name = key

        # Set the device key to the attribute
        assert isinstance(attr_name, str) and attr_name, 'Attribute name must be of type str and not empty'
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
        """Returns the contents of a system dataset."""

        assert isinstance(key, str), 'Key must be of type str'

        try:
            # Get value from system dataset with extra flags
            value: typing.Any = self.get_dataset(self.get_system_key(key), default, archive=True)
        except KeyError as e:
            # The key was not found
            msg = 'System dataset key "{:s}" not found'.format(key)
            self.logger.error(msg)
            raise KeyError(msg) from e
        else:
            self.logger.debug('System dataset key "{:s}" returned value "{}"'.format(key, value))

        # Return value
        return value

    def setattr_dataset_sys(self, key: str, default: typing.Any = artiq.experiment.NoDefault,
                            kernel_invariant: bool = True) -> None:
        """Sets the contents of a system dataset as attribute."""

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

        # Check parent type
        if not isinstance(managers_or_parent, _DaxModuleBase):
            raise TypeError('Parent of module {:s} is not a DAX module base'.format(module_name))

        # Take core devices from parent
        try:
            # Use core devices from parent
            self.core: artiq.coredevice.core = managers_or_parent.core
            self.core_dma: artiq.coredevice.dma = managers_or_parent.core_dma
            self.core_cache: artiq.coredevice.cache = managers_or_parent.core_cache
        except AttributeError as e:
            msg = 'Missing core devices (super.build() was probably not called)'
            managers_or_parent.logger.error(msg)
            raise AttributeError(msg) from e

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

    def build(self) -> None:
        """Override this method to build your DAX system. (Do not forget to call super.build() first!)"""

        # Core devices
        self.core: artiq.coredevice.core = self.get_device(self.CORE_KEY, artiq.coredevice.core.Core)
        self.core_dma: artiq.coredevice.dma = self.get_device(self.CORE_DMA_KEY, artiq.coredevice.dma.CoreDMA)
        self.core_cache: artiq.coredevice.cache = self.get_device(self.CORE_CACHE_KEY, artiq.coredevice.cache.CoreCache)

        # Verify existence of core log controller
        if self.CORE_LOG_KEY not in self.get_device_db():
            self.logger.warning('Core log controller "{:s}" not found in device DB'.format(self.CORE_LOG_KEY))

    def dax_init(self) -> None:
        """Initialize the DAX system."""
        self.logger.debug('Starting DAX system initialization...')
        try:
            self._init_system()
            self._post_init_system()
        except artiq.coredevice.core.CompileError:
            self.logger.error('Compilation error occurred during DAX system initialization')
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
            raise TypeError('Parent of service {:s} is not a DAX system or service'.format(self.get_name()))

        # Take core devices from parent
        try:
            # Use core devices from parent
            self.core: artiq.coredevice.core = managers_or_parent.core
            self.core_dma: artiq.coredevice.dma = managers_or_parent.core_dma
            self.core_cache: artiq.coredevice.cache = managers_or_parent.core_cache
        except AttributeError:
            managers_or_parent.logger.error('Missing core devices (super.build() was probably not called)')
            raise

        # Take name registry from parent and obtain a system key
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
        assert isinstance(self, DaxSystem), \
            'DAX client class {:s} must be decorated using @dax_client_factory'.format(self.__class__.__name__)
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

    class NonUniqueRegistrationError(ValueError):
        """Exception when a name is registered more then once."""
        pass

    class NonUniqueSearchError(ValueError):
        """Exception when a search could not find a unique result."""
        pass

    # Module base type variable
    __M_T = typing.TypeVar('__M_T', bound='_DaxModuleBase')
    # Service type variable
    __S_T = typing.TypeVar('__S_T', bound='DaxService')

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
            module.logger.error(msg)
            raise self.NonUniqueRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._modules[key] = module

    @typing.overload
    def get_module(self, key: str) -> _DaxModuleBase:
        ...

    @typing.overload
    def get_module(self, key: str, type_: typing.Type[__M_T]) -> __M_T:
        ...

    def get_module(self, key: str, type_: typing.Type[__M_T] = _DaxModuleBase) -> __M_T:
        """Return the requested module by key."""

        assert isinstance(key, str), 'Key must be a string'

        try:
            # Get the module
            module = self._modules[key]
        except KeyError as e:
            # The key was not present
            raise KeyError('Module "{:s}" could not be found'.format(key)) from e

        if not isinstance(module, type_):
            # Module does not have the correct type
            raise TypeError('Module "{:s}" does not match the expected type'.format(key))

        # Return the module
        return module

    def search_module(self, type_: typing.Type[__M_T]) -> __M_T:
        """Search for a unique module that matches the requested type."""

        # Search for all modules matching the type
        results: typing.Dict[str, _DaxNameRegistry.__M_T] = self.search_module_dict(type_)

        if len(results) > 1:
            # More than one module was found
            raise self.NonUniqueSearchError('Could not find a unique module with type "{:s}"'.format(type_.__name__))

        # Return the only result
        _, module = results.popitem()
        return module

    def search_module_dict(self, type_: typing.Type[__M_T]) -> typing.Dict[str, __M_T]:
        """Search for modules that match the requested type and return results as a dict."""

        assert issubclass(type_, (DaxModuleInterface, _DaxModuleBase)), \
            'Provided type must be a DAX module base or interface'

        # Search for all modules matching the type
        results: typing.Dict[str, _DaxModuleBase] = {k: m for k, m in self._modules.items() if isinstance(m, type_)}

        if not results:
            # No modules were found
            raise KeyError('Could not find modules with type "{:s}"'.format(type_.__name__))

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

        try:
            # Get the unique key
            unique: str = self._get_unique_device_key(parent.get_device_db(), key, set())
        except (KeyError, ValueError, TypeError) as e:
            msg = 'Device "{:s}" could not be found'.format(key)
            parent.logger.error(msg)
            raise KeyError(msg) from e

        # Get the parent that registered the device (None if the device was not registered before)
        reg_parent: typing.Optional[_DaxHasSystem] = self._devices.get(unique)

        if reg_parent:
            # Device was already registered
            device_name = '"{:s}"'.format(key) if key == unique else '"{:s}" ({:s})'.format(key, unique)
            msg = 'Device {:s}, was already registered by parent {:s}'.format(device_name, reg_parent.get_identifier())
            parent.logger.error(msg)
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
            raise ValueError('Key {:s} causes an alias loop'.format(key))
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
            raise TypeError('Key {:s} returned an unexpected type'.format(key))

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
            msg = 'Service with name "{:s}" was already registered'.format(key)
            service.logger.error(msg)
            raise self.NonUniqueRegistrationError(msg)

        # Add service to the registry
        self._services[key] = service

    def has_service(self, key: typing.Union[type, str]) -> bool:
        """Return if service is available."""
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

    def get_service(self, key: typing.Union[str, typing.Type[__S_T]]) -> typing.Union[DaxService, __S_T]:
        """Get a service from the registry."""

        assert isinstance(key, str) or issubclass(key, DaxService)

        # Figure the right key
        key = key if isinstance(key, str) else key.SERVICE_NAME

        # Try to return the requested service
        try:
            return self._services[key]
        except KeyError as e:
            raise KeyError('Service "{:s}" is not available') from e

    def get_service_key_list(self) -> typing.List[str]:
        """Return a list of registered service keys."""
        service_key_list: typing.List[str] = natsort.natsorted(self._services.keys())  # Natural sort the list
        return service_key_list


# Type variable for dax_client_factory() decorator c (client) argument
__C_T = typing.TypeVar('__C_T', bound='DaxClient')
# Type variable for dax_client_factory() system_type argument
__S_T = typing.TypeVar('__S_T', bound='DaxSystem')


def dax_client_factory(c: typing.Type[__C_T]) -> typing.Callable[[typing.Type[__S_T], typing.Any, typing.Any],
                                                                 typing.Type[__C_T]]:
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
