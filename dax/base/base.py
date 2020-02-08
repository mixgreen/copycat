import abc
import logging
import functools

from artiq.master.worker_db import DummyDevice
from artiq.experiment import *


class _DaxNameRegister:
    """A class to register module and device names."""

    class NonUniqueDeviceRegistrationError(Exception):
        """Exception when a device is registered more then once."""
        pass

    class NonUniqueModuleRegistrationError(Exception):
        """Exception when a module key is registered more then once."""
        pass

    def __init__(self):
        # A dict containing registered modules
        self._registered_modules = dict()
        # A dict containing registered devices and the modules that registered them
        self._registered_devices = dict()

    def register_module(self, module, key):
        """Register a module."""

        assert isinstance(module, DaxBase), 'Module is not a DAX base'
        assert isinstance(key, str), 'Key must be a string'

        # Get the module that registered the module key (None if the key is available)
        reg_module = self._registered_modules.get(key)
        if reg_module:
            msg = 'Module key {:s} was already registered by module {:s}'.format(key, reg_module.get_identifier())
            module.logger.error(msg)
            raise self.NonUniqueModuleRegistrationError(msg)

        # Add module key to the dict of registered modules
        self._registered_modules[key] = module

    def register_device(self, module, key):
        """Register a device."""

        assert isinstance(module, DaxBase), 'Module is not a DAX base'
        assert isinstance(key, str), 'Key must be a string'

        try:
            # Get the unique key
            unique = self._get_unique_device_key(module.get_device_db(), key)
        except KeyError as e:
            msg = 'Device {:s} could not be found'.format(key)
            module.logger.error(msg)
            raise KeyError(msg) from e

        # Get the module that registered the device (None if the device was not registered before)
        reg_module = self._registered_devices.get(unique)
        if reg_module:
            # Device was already registered
            device_name = '"{:s}"'.format(key) if key == unique else '"{:s}" ({:s})'.format(key, unique)
            msg = 'Device {:s}, was already registered by module {:s}'.format(device_name, reg_module.get_identifier())
            module.logger.error(msg)
            raise self.NonUniqueDeviceRegistrationError(msg)

        # Add unique device key to the dict of registered devices
        self._registered_devices[unique] = module

    def _get_unique_device_key(self, d, key):
        assert isinstance(d, dict), 'First argument must be a dict to search in'
        assert isinstance(key, str), 'Key must be a string'

        v = d[key]
        return self._get_unique_device_key(d, v) if isinstance(v, str) else key


# Central name register (only one instance)
_dax_name_reg = _DaxNameRegister()


class DaxBase(HasEnvironment, abc.ABC):
    """Base class for all DAX core classes."""

    # Key separator
    _KEY_SEPARATOR = '.'

    class BuildArgumentError(Exception):
        """Exception for build arguments not matching the expected signature"""
        pass

    def __init__(self, managers_or_parent, *args, **kwargs):
        # Logger object
        self.logger = logging.getLogger(self.get_identifier())

        try:
            # Call super, which will result in the module being build
            self.logger.debug('Starting build...')
            super(DaxBase, self).__init__(managers_or_parent, *args, **kwargs)
            self.logger.debug('Build finished')
        except TypeError as e:
            self.logger.error('Build arguments do not match the expected signature')
            raise self.BuildArgumentError(e) from e

    @host_only
    def update_kernel_invariants(self, *keys):
        """Add one or more keys to the kernel invariants set."""
        assert all(isinstance(k, str) for k in keys)
        kernel_invariants = getattr(self, "kernel_invariants", set())
        self.kernel_invariants = kernel_invariants | {*keys}

    @abc.abstractmethod
    def get_identifier(self):
        pass


class DaxModuleBase(DaxBase, abc.ABC):
    """Base class for all DAX modules and systems."""

    def __init__(self, managers_or_parent, module_name, module_key, *args, **kwargs):
        """Initialize the DAX module base."""

        # Check types
        assert isinstance(module_name, str), 'Module name must be a string'
        assert isinstance(module_key, str), 'Module key must be a string'

        # Check module name and key
        if not module_name.isalpha():
            raise ValueError('Invalid module name "{}" for class {:s}'.format(module_name, self.__class__.__name__))
        if not module_key.endswith(module_name):
            raise ValueError('Invalid module key "{}" for class {:s}'.format(module_key, self.__class__.__name__))

        # Store module name and key
        self.module_name = module_name
        self.module_key = module_key

        # Register module key
        _dax_name_reg.register_module(self, module_key)

        # Call super
        super(DaxModuleBase, self).__init__(managers_or_parent, *args, **kwargs)

    @host_only
    def load_system(self):
        """Load the DAX system, for loading values from the dataset."""

        self.logger.debug('Loading module...')
        # Load all sub-modules (also called children)
        self.call_child_method('load_system')
        # Load this module
        self.load_module()
        self.logger.debug('Module loading finished')

    @host_only
    def init_system(self):
        """Initialize the DAX system, for device initialization and recording DMA traces."""

        self.logger.debug('Initializing module...')
        # Initialize all sub-modules (also called children)
        self.call_child_method('init_system')
        # Initialize this module
        self.init_module()
        self.logger.debug('Module initialization finished')

    @host_only
    def config_system(self):
        """Configure the DAX system, for configuring devices and obtaining DMA handles."""

        self.logger.debug('Configuring module...')
        # Configure all sub-modules (also called children)
        self.call_child_method('config_system')
        # Configure this module
        self.config_module()
        self.logger.debug('Module configuration finished')

    @abc.abstractmethod
    def load_module(self):
        """Override this method to load dataset parameters for your module (no calls to the core device allowed)."""
        pass

    @abc.abstractmethod
    def init_module(self):
        """Override this method to initialize devices and record DMA traces for your module."""
        pass

    @abc.abstractmethod
    def config_module(self):
        """Override this method to configure devices and obtain DMA handles."""
        pass

    @host_only
    def get_system_key(self, key):
        """Get the full system key based on the module parents."""
        assert isinstance(key, str)
        return self._KEY_SEPARATOR.join([self.module_key, key])

    def get_device(self, key):
        # Register the requested device
        _dax_name_reg.register_device(self, key)

        # Get the actual device and return it
        return super(DaxBase, self).get_device(key)

    def setattr_device(self, key, attr_name='', type_=object):
        """Sets a device driver as attribute."""

        assert isinstance(key, str) and key, 'Key must be of type str and not empty'
        assert isinstance(attr_name, str), 'Attribute name must be of type str'

        if not attr_name:
            # Set attribute name to key if no attribute name was given
            attr_name = key

        # Get the device
        device = self.get_device(key)

        # Check type
        if not isinstance(device, DummyDevice) and not isinstance(device, type_):
            msg = 'Device "{:s}" does not match the expected type'.format(key)
            self.logger.error(msg)
            raise TypeError(msg)

        # Set the device key to the attribute
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
        except KeyError:
            # In case of a key error, still print a debug message for the user
            self.logger.debug('System dataset key "{:s}" get value'.format(key))
            raise
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
        """Return the module key with the class name."""
        return '[{:s}]({:s})'.format(self.module_key, self.__class__.__name__)


class DaxModule(DaxModuleBase, abc.ABC):
    """Base class for DAX modules."""

    def __init__(self, managers_or_parent, module_name, *args, **kwargs):
        """Initialize the DAX module."""

        # Check parent
        if not isinstance(managers_or_parent, DaxModuleBase):
            raise TypeError('Parent of module {:s} is not a DAX module base'.format(module_name))

        # Take core devices from parent
        try:
            # Use core devices from parent and make them kernel invariants
            self.core = managers_or_parent.core
            self.core_dma = managers_or_parent.core_dma
            self.update_kernel_invariants('core', 'core_dma')
        except AttributeError:
            managers_or_parent.logger.error('Missing core devices (super.build() was probably not called)')
            raise

        # Call super
        super(DaxModule, self).__init__(managers_or_parent, module_name, managers_or_parent.get_system_key(module_name),
                                        *args, **kwargs)


class DaxSystem(DaxModuleBase):
    """Base class for DAX systems, which is a top-level module."""

    # System name, used as top key
    SYS_NAME: str = 'sys'

    def __init__(self, managers_or_parent, *args, **kwargs):
        # Call super
        super(DaxSystem, self).__init__(managers_or_parent, self.SYS_NAME, self.SYS_NAME, *args, **kwargs)

    def build(self):
        """Override this method to build your DAX system. (Do not forget to call super.build() first!)"""

        # Core devices
        self.setattr_device('core')
        self.setattr_device('core_dma')

        # Call super
        super(DaxSystem, self).build()

    def dax_load(self):
        """Prepare the DAX system for usage by loading and configuring the system."""
        self.load_system()
        self.config_system()

    def dax_init(self):
        """Prepare the DAX system for usage by loading, initializing, and configuring the system."""
        self.load_system()
        self.init_system()
        self.config_system()

    def load_module(self):
        pass

    def init_module(self):
        pass

    def config_module(self):
        pass


class DaxCalibration(DaxBase):
    # System type
    _SYSTEM_TYPE: type
    # Module key
    _MODULE_KEY: str
    # Module type (for verification purposes)
    _MODULE_TYPE: type

    def __init__(self, managers_or_parent, *args, **kwargs):
        # Call super
        super(DaxCalibration, self).__init__(managers_or_parent, *args, **kwargs)

        # Create the system
        self.system = self._SYSTEM_TYPE(self)

        # Recursive function to obtain the module
        def _get_module(key, module):
            try:
                # Get the next module (next level of nesting)
                module = getattr(module, key[0])
            except KeyError:
                self.logger.error('Module {:s} could not be found'.format(self._MODULE_KEY))
                raise

            if len(key) == 1:
                return module
            else:
                return _get_module(key[1:], module)

        # Create a reference to the module to calibrate
        self.module = _get_module(self._MODULE_KEY.split(self._KEY_SEPARATOR), self.system)

        # Check module type
        if not isinstance(self.module, self._MODULE_TYPE):
            msg = 'Module is not compatible with this calibration class'
            self.logger.error(msg)
            raise TypeError(msg)

    def get_module(self):
        """Return the module to calibrate."""
        return self.module

    @host_only
    def get_identifier(self):
        """Return the module key of the calibrated module and the calibration class name."""
        return '[{:s}]({:s})'.format(self._KEY_SEPARATOR.join([self._SYSTEM_TYPE.SYS_NAME, self._MODULE_KEY]),
                                     self.__class__.__name__)


def dax_calibration_factory(module_type):
    """Decorator to convert a DaxCalibration class to a factory function of that class."""

    assert issubclass(module_type, DaxModule), 'The module type for a calibration factory must be a DaxModule'

    def decorator(c):
        """The actual decorator function takes a DaxCalibration class and returns a factory function."""

        assert isinstance(c, type), 'The decorated object must be a class'
        assert issubclass(c, DaxCalibration), 'The decorated class must be a subclass of DaxCalibration'
        assert issubclass(c, EnvExperiment), 'The decorated class must be a subclass of EnvExperiment'

        # Use the wraps decorator, but do not inherit the docstring
        @functools.wraps(c, assigned=[e for e in functools.WRAPPER_ASSIGNMENTS if e != '__doc__'])
        def wrapper(system_type, module_key):
            """Create a new calibration class.

            This factory function will create a new calibration class for a specific module of a given system type.

            :param system_type: The system type which contains the module to calibrate.
            :param module_key: The key to the module to calibrate, must match with the calibration class.
            """

            # Check the system type
            if not issubclass(system_type, DaxSystem):
                raise TypeError('System type must be a subclass of DaxSystem')
            # Check module key
            if not isinstance(module_key, str):
                raise TypeError('Module key must be a string')

            class WrapperClass(c):
                """The wrapper class that fills in the required attributes"""
                _SYSTEM_TYPE = system_type
                _MODULE_KEY = module_key
                _MODULE_TYPE = module_type

            return WrapperClass

        return wrapper

    return decorator
