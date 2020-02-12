import abc
import logging
import functools
import re
import natsort

from artiq.master.worker_db import DummyDevice
from artiq.experiment import *

import artiq.coredevice.core
import artiq.coredevice.dma
import artiq.coredevice.cache

# Key separator
_KEY_SEPARATOR: str = '.'
# Regex for matching valid names
_NAME_RE = re.compile(r'\w+')
# Regex for matching valid keys
_KEY_RE = re.compile(r'\w+(\.\w+)*')


def _is_valid_name(name):
    """Return true if the given name is valid."""
    assert isinstance(name, str), 'The given name should be a string'
    return _NAME_RE.fullmatch(name)


def _is_valid_key(key):
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return _KEY_RE.fullmatch(key)


class _DaxBase(HasEnvironment, abc.ABC):
    """Base class for all DAX core classes."""

    class _BuildArgumentError(Exception):
        """Exception for build arguments not matching the expected signature."""
        pass

    class _IllegalOperationError(Exception):
        """Exception when user calls a disabled/illegal function."""
        pass

    def __init__(self, managers_or_parent, *args, **kwargs):
        # Logger object
        self.logger = logging.getLogger(self.get_identifier())

        # Build
        self.logger.debug('Starting build...')
        try:
            # Call super, which will call build()
            super(_DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
        except TypeError as e:
            msg = 'Build arguments do not match the expected signature'
            self.logger.error(msg)
            raise self._BuildArgumentError(msg) from e
        else:
            self.logger.debug('Build finished')

    @host_only
    def update_kernel_invariants(self, *keys):
        """Add one or more keys to the kernel invariants set."""

        assert all(isinstance(k, str) for k in keys), 'All keys must be of type str'

        # Get kernel invariants using getattr() such that we do not overwrite a user-defined variable
        kernel_invariants = getattr(self, "kernel_invariants", set())
        # Update the set with the given keys
        self.kernel_invariants = kernel_invariants | {*keys}

    @abc.abstractmethod
    def get_identifier(self):
        pass


class _DaxNameRegistry:
    """A class for unique name registration."""

    class _NonUniqueRegistrationError(ValueError):
        """Exception when a name is registered more then once."""
        pass

    class _NonUniqueSearchError(ValueError):
        """Exception when a search could not find a unique result."""
        pass

    def __init__(self, sys_services_key):
        assert isinstance(sys_services_key, str), 'System services key must be a string'

        # Check system services key
        if not _is_valid_key(sys_services_key):
            raise ValueError('Invalid system services key "{:s}"'.format(sys_services_key))

        # Store system services key
        self._sys_services_key = sys_services_key

        # A dict containing registered modules
        self._modules = dict()
        # A dict containing registered devices and the modules that registered them
        self._devices = dict()
        # A dict containing registered services
        self._services = dict()

    def add_module(self, module):
        """Register a module."""

        assert isinstance(module, DaxModuleBase), 'Module is not a DAX module base'

        # Get the module key
        key = module.get_system_key()

        # Get the module that registered the module key (None if the key is available)
        reg_module = self._modules.get(key)

        if reg_module:
            # Key already in use by an other module
            msg = 'Module key "{:s}" was already registered by module {:s}'.format(key, reg_module.get_identifier())
            module.logger.error(msg)
            raise self._NonUniqueRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._modules[key] = module

    def get_module(self, key, type_=object):
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

    def search_module(self, type_):
        """Search for a unique module that matches the requested type."""

        # Search for all modules matching the type
        results = self.search_module_dict(type_)

        if len(results) > 1:
            # More than one module was found
            raise self._NonUniqueSearchError('Could not find a unique module with type "{:s}"'.format(type_.__name__))

        # Return the only result
        _, module = results.popitem()
        return module

    def search_module_dict(self, type_):
        """Search for modules that match the requested type and return results as a dict."""

        assert issubclass(type_, (DaxModuleInterface, DaxModuleBase)), \
            'Provided type must be a DAX module base or interface'

        # Search for all modules matching the type
        results = {k: m for k, m in self._modules.items() if isinstance(m, type_)}

        if not results:
            # No modules were found
            raise KeyError('Could not find modules with type "{:s}"'.format(type_.__name__))

        # Return the list with results
        return results

    def get_module_key_list(self):
        """Return a list of registered module keys."""
        return natsort.natsorted(self._modules.keys())

    def add_device(self, module, key):
        """Register a device."""

        assert isinstance(module, DaxModuleBase), 'Module is not a DAX module base'
        assert isinstance(key, str), 'Device key must be a string'

        try:
            # Get the unique key
            unique = self._get_unique_device_key(module.get_device_db(), key)
        except (KeyError, ValueError, TypeError) as e:
            msg = 'Device "{:s}" could not be found'.format(key)
            module.logger.error(msg)
            raise KeyError(msg) from e

        # Get the module that registered the device (None if the device was not registered before)
        reg_module = self._devices.get(unique)

        if reg_module:
            # Device was already registered
            device_name = '"{:s}"'.format(key) if key == unique else '"{:s}" ({:s})'.format(key, unique)
            msg = 'Device {:s}, was already registered by module {:s}'.format(device_name, reg_module.get_identifier())
            module.logger.error(msg)
            raise self._NonUniqueRegistrationError(msg)

        # Add unique device key to the dict of registered devices
        self._devices[unique] = module

    def _get_unique_device_key(self, d, key, trace=None):
        if trace is None:
            # If no trace was given, start with an empty set
            trace = set()

        assert isinstance(d, dict), 'First argument must be a dict to search in'
        assert isinstance(key, str), 'Key must be a string'
        assert isinstance(trace, set), 'Trace must be a set'

        # Check if we are not stuck in a loop
        if key in trace:
            raise ValueError('Key {:s} causes an alias loop'.format(key))
        # Add key to the trace
        trace.add(key)

        # Get value (could raise KeyError)
        v = d[key]

        if isinstance(v, str):
            # Recurse if we are still dealing with an alias
            return self._get_unique_device_key(d, v, trace)
        elif isinstance(v, dict):
            # We reached a dict, key must be the unique key
            return key
        else:
            # We ended up with an unexpected type
            raise TypeError('Key {:s} returned an unexpected type'.format(key))

    def get_device_key_list(self):
        """Return a list of registered device keys."""
        return natsort.natsorted(self._devices.keys())  # Natural sort the list

    def make_service_key(self, service_name):
        """Return the system key for a service name."""

        # Check the given name
        assert isinstance(service_name, str), 'Service name must be a string'
        if not _is_valid_name(service_name):
            # Service name not valid
            raise ValueError('Invalid service name "{:s}"'.format(service_name))

        # Return assigned key
        return _KEY_SEPARATOR.join([self._sys_services_key, service_name])

    def add_service(self, service):
        """Register a service."""

        assert isinstance(service, DaxService), 'Service must be a DAX service'

        # Services get indexed by name
        key = service.get_name()

        # Get the service that registered with the service name (None if key is available)
        reg_service = self._services.get(key)

        if reg_service:
            # Service name was already registered
            msg = 'Service with name "{:s}" was already registered'.format(key)
            service.logger.error(msg)
            raise self._NonUniqueRegistrationError(msg)

        # Add service to the registry
        self._services[key] = service

    def has_service(self, key, default=None):
        """Return service if available, otherwise return the default value."""
        try:
            return self.get_service(key)
        except KeyError:
            return default

    def get_service(self, key):
        """Get a service from the registry."""

        assert isinstance(key, str) or issubclass(key, DaxService)

        # Figure the right key
        key = key if isinstance(key, str) else key.SERVICE_NAME

        # Try to return the requested service
        try:
            return self._services[key]
        except KeyError as e:
            raise KeyError('Service "{:s}" is not available') from e

    def get_service_key_list(self):
        """Return a list of registered service keys."""
        return natsort.natsorted(self._services.keys())  # Natural sort the list


class _DaxHasSystem(_DaxBase, abc.ABC):
    """Intermediate base class for DAX classes that are dependent on a DAX system."""

    __CORE_DEVICES = ['core', 'core_dma', 'core_cache']

    def __init__(self, managers_or_parent, name, system_key, registry, *args, **kwargs):
        assert isinstance(name, str), 'Name must be a string'
        assert isinstance(system_key, str), 'System key must be a string'
        assert isinstance(registry, _DaxNameRegistry), 'Registry must be a DAX name registry'

        # Check name and system key
        if not _is_valid_name(name):
            raise ValueError('Invalid name "{:s}" for class {:s}'.format(name, self.__class__.__name__))
        if not _is_valid_key(system_key) or not system_key.endswith(name):
            raise ValueError('Invalid system_key "{:s}" for class {:s}'.format(system_key, self.__class__.__name__))

        # Store attributes
        self._name = name
        self._system_key = system_key
        self.registry = registry

        # Call super, which will result in a call to build()
        super(_DaxHasSystem, self).__init__(managers_or_parent, *args, **kwargs)

        # Verify that all core devices are available
        if not all(hasattr(self, n) for n in self.__CORE_DEVICES):
            msg = 'Missing core devices (super.build() was probably not called)'
            self.logger.error(msg)
            raise AttributeError(msg)

        # Make core devices kernel invariants
        self.update_kernel_invariants(*self.__CORE_DEVICES)

    @host_only
    def get_name(self):
        """Get the name."""
        return self._name

    @host_only
    def get_system_key(self, key=None):
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

    @host_only
    def get_registry(self):
        """Return the current registry."""
        return self.registry

    @host_only
    def load_system(self):
        """Load the DAX system, for loading values from the dataset."""

        self.logger.debug('Loading...')
        # Load all children
        self.call_child_method('load_system')
        # Load this object
        self.load()
        self.logger.debug('Loading finished')

    @host_only
    def init_system(self):
        """Initialize the DAX system, for device initialization and recording DMA traces."""

        self.logger.debug('Initializing...')
        # Initialize all children
        self.call_child_method('init_system')
        # Initialize this object
        self.init()
        self.logger.debug('Initialization finished')

    @host_only
    def config_system(self):
        """Configure the DAX system, for configuring devices and obtaining DMA handles."""

        self.logger.debug('Configuring...')
        # Configure all children
        self.call_child_method('config_system')
        # Configure this object
        self.config()
        self.logger.debug('Configuration finished')

    @abc.abstractmethod
    def load(self):
        """Override this method to load dataset parameters (no calls to the core device allowed)."""
        pass

    @abc.abstractmethod
    def init(self):
        """Override this method to initialize devices and record DMA traces (calls to core device allowed)."""
        pass

    @abc.abstractmethod
    def config(self):
        """Override this method to configure devices and obtain DMA handles (calls to core device allowed)."""
        pass

    def get_device(self, key, type_=object):
        """Get a device driver."""

        assert isinstance(key, str) and key, 'Key must be of type str and not empty'

        # Debug message
        self.logger.debug('Requesting device "{:s}"'.format(key))

        # Register the requested device
        self.registry.add_device(self, key)

        # Get the device
        device = super(_DaxHasSystem, self).get_device(key)

        # Check device type
        if not isinstance(device, DummyDevice) and not isinstance(device, type_):
            msg = 'Device "{:s}" does not match the expected type'.format(key)
            self.logger.error(msg)
            raise TypeError(msg)

        # Return the device
        return device

    def setattr_device(self, key, attr_name=None, type_=object):
        """Sets a device driver as attribute."""

        # Get the device
        device = self.get_device(key, type_)

        if attr_name is None:
            # Set attribute name to key if no attribute name was given
            attr_name = key

        # Set the device key to the attribute
        assert isinstance(attr_name, str) and attr_name, 'Attribute name must be of type str and not empty'
        setattr(self, attr_name, device)

        # Add attribute to kernel invariants
        self.update_kernel_invariants(attr_name)

    @rpc(flags={'async'})
    def set_dataset_sys(self, key, value):
        """Sets the contents of a system dataset."""

        assert isinstance(key, str), 'Key must be of type str'

        # Set value in system dataset with extra flags
        self.logger.debug('System dataset key "{:s}" set to value "{}"'.format(key, value))
        self.set_dataset(self.get_system_key(key), value, broadcast=True, persist=True, archive=True)

    @rpc(flags={'async'})
    def mutate_dataset_sys(self, key, index, value):
        """Mutate an existing system dataset at the given index."""

        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(index, int), 'Index must be of type int'

        # Mutate system dataset
        self.logger.debug('System dataset key "{:s}"[{:d}] mutate to value "{}"'.format(key, index, value))
        self.mutate_dataset(self.get_system_key(key), index, value)

    @rpc(flags={'async'})
    def append_to_dataset_sys(self, key, value):
        """Append a value to a system dataset."""

        assert isinstance(key, str), 'Key must be of type str'

        # Append value to system dataset
        self.logger.debug('System dataset key "{:s}" append value "{}"'.format(key, value))
        self.append_to_dataset(self.get_system_key(key), value)

    def get_dataset_sys(self, key, default=NoDefault):
        """Returns the contents of a system dataset."""

        assert isinstance(key, str), 'Key must be of type str'

        try:
            # Get value from system dataset with extra flags
            value = self.get_dataset(self.get_system_key(key), default, archive=True)
        except KeyError as e:
            # The key was not found
            msg = 'System dataset key "{:s}" not found'.format(key)
            self.logger.error(msg)
            raise KeyError(msg) from e
        else:
            self.logger.debug('System dataset key "{:s}" returned value "{}"'.format(key, value))

        # Return value
        return value

    def setattr_dataset_sys(self, key, default=NoDefault, kernel_invariant=True):
        """Sets the contents of a system dataset as attribute."""

        assert isinstance(key, str), 'Key must be of type str'

        try:
            # Get the value from system dataset
            value = self.get_dataset_sys(key)
        except KeyError:
            if default is NoDefault:
                raise  # If there is no default, raise the KeyError
            else:
                # If the value does not exist, write the default value to the system dataset
                self.set_dataset_sys(key, default)
                value = default

        # Set value as an attribute
        setattr(self, key, value)

        if kernel_invariant:
            # Update kernel invariants
            self.update_kernel_invariants(key)

    @host_only
    def get_identifier(self):
        """Return the system key with the class name."""
        return '[{:s}]({:s})'.format(self.get_system_key(), self.__class__.__name__)


class DaxModuleBase(_DaxHasSystem, abc.ABC):
    """Base class for all DAX modules and systems."""

    def __init__(self, managers_or_parent, module_name, module_key, registry, *args, **kwargs):
        """Initialize the DAX module base."""

        # Call super
        super(DaxModuleBase, self).__init__(managers_or_parent, module_name, module_key, registry, *args, **kwargs)

        # Register this module
        self.registry.add_module(self)


class DaxModule(DaxModuleBase, abc.ABC):
    """Base class for DAX modules."""

    def __init__(self, managers_or_parent, module_name, *args, **kwargs):
        """Initialize the DAX module."""

        # Check parent type
        if not isinstance(managers_or_parent, DaxModuleBase):
            raise TypeError('Parent of module {:s} is not a DAX module base'.format(module_name))

        # Take core devices from parent
        try:
            # Use core devices from parent
            self.core = managers_or_parent.core
            self.core_dma = managers_or_parent.core_dma
            self.core_cache = managers_or_parent.core_cache
        except AttributeError:
            managers_or_parent.logger.error('Missing core devices (super.build() was probably not called)')
            raise

        # Call super, use parent to assemble arguments
        super(DaxModule, self).__init__(managers_or_parent, module_name, managers_or_parent.get_system_key(module_name),
                                        managers_or_parent.get_registry(), *args, **kwargs)


class DaxModuleInterface(abc.ABC):
    """Base class for module interfaces."""
    pass


class DaxSystem(DaxModuleBase):
    """Base class for DAX systems, which is a top-level module."""

    # System name, used as top key for modules
    SYS_NAME: str = 'system'
    # System services, used as top key for services
    SYS_SERVICES: str = 'services'

    # Keys of core devices
    CORE_KEY: str = 'core'
    CORE_DMA_KEY: str = 'core_dma'
    CORE_CACHE_KEY: str = 'core_cache'

    def __init__(self, managers_or_parent, *args, **kwargs):
        # Call super, add names, add a new registry
        super(DaxSystem, self).__init__(managers_or_parent, self.SYS_NAME, self.SYS_NAME,
                                        _DaxNameRegistry(self.SYS_SERVICES), *args, **kwargs)

    def build(self):
        """Override this method to build your DAX system. (Do not forget to call super.build() first!)"""

        # Core devices
        self.core = self.get_device(self.CORE_KEY, artiq.coredevice.core.Core)
        self.core_dma = self.get_device(self.CORE_DMA_KEY, artiq.coredevice.dma.CoreDMA)
        self.core_cache = self.get_device(self.CORE_CACHE_KEY, artiq.coredevice.cache.CoreCache)

    def dax_load(self):
        """Prepare the DAX system for usage by loading and configuring the system."""
        self.logger.debug('Starting DAX system loading...')
        self.load_system()
        self.config_system()
        self.logger.debug('Finished DAX system loading')

    def dax_init(self):
        """Prepare the DAX system for usage by loading, initializing, and configuring the system."""
        self.logger.debug('Starting DAX system initialization...')
        self.load_system()
        self.init_system()
        self.config_system()
        self.logger.debug('Finished DAX system initialization')

    def load(self):
        pass

    def init(self):
        pass

    def config(self):
        pass


class DaxService(_DaxHasSystem, abc.ABC):
    """Base class for system services."""

    # The unique name of this service
    SERVICE_NAME: str

    def __init__(self, managers_or_parent, *args, **kwargs):
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
            self.core = managers_or_parent.core
            self.core_dma = managers_or_parent.core_dma
            self.core_cache = managers_or_parent.core_cache
        except AttributeError:
            managers_or_parent.logger.error('Missing core devices (super.build() was probably not called)')
            raise

        # Take name registry from parent and obtain a system key
        registry = managers_or_parent.get_registry()
        system_key = registry.make_service_key(self.SERVICE_NAME)

        # Call super
        super(DaxService, self).__init__(managers_or_parent, self.SERVICE_NAME, system_key, registry, *args, **kwargs)

        # Register this service
        self.registry.add_service(self)

    def get_device(self, *args, **kwargs):
        """Services are not allowed to request devices."""
        msg = 'Services are not allowed to request devices'
        self.logger.error(msg)
        raise self._IllegalOperationError(msg)


class DaxClient(_DaxHasSystem, abc.ABC):
    """Base class for DAX clients."""

    def __init__(self, managers_or_parent, *args, **kwargs):
        # Check if the decorator was used
        assert isinstance(self, DaxSystem), \
            'DAX client class {:s} must be decorated using @dax_client_factory'.format(self.__class__.__name__)
        # Call super
        super(DaxClient, self).__init__(managers_or_parent, *args, **kwargs)

    def load(self):
        # Call super (which will be the DAX system)
        super(DaxClient, self).load()

    def init(self):
        # Call super (which will be the DAX system)
        super(DaxClient, self).init()

    def config(self):
        # Call super (which will be the DAX system)
        super(DaxClient, self).config()


def dax_client_factory(c):
    """Decorator to convert a DaxClient class to a factory function for that class."""

    assert isinstance(c, type), 'The decorated object must be a class'
    assert issubclass(c, DaxClient), 'The decorated class must be a subclass of DaxClient'

    # Use the wraps decorator, but do not inherit the docstring
    @functools.wraps(c, assigned=[e for e in functools.WRAPPER_ASSIGNMENTS if e != '__doc__'])
    def wrapper(system_type):
        """Create a new DAX client class.

        This factory function will create a new client class for a given system type.

        :param system_type: The system type used by the client.
        """

        # Check the system type
        if not issubclass(system_type, DaxSystem):
            raise TypeError('System type must be a subclass of DaxSystem')

        class WrapperClass(c, system_type):
            """The wrapper class that finalizes the client class.

            The wrapper class first inherits from the client and then the system,
            as if the client class was an extension of the system, just as a
            custom experiment.
            """

            def build(self, *args, **kwargs):
                # First build the system (not using MRO) to fill the registry
                system_type.build(self)

                # Then build the client which can use the registry
                c.build(self, *args, **kwargs)

        # The factory function returns the newly constructed wrapper class
        return WrapperClass

    # Return the factory function
    return wrapper
