import unittest
import typing
import collections.abc
import dataclasses
import subprocess
import time

import artiq.master.worker_db
from artiq.language.environment import HasEnvironment

import dax.util.artiq

from test.environment import CI_ENABLED, NIX_ENV, TB_DISABLED, JOB_ID

__all__ = ['TestBenchCase']

_TIMEOUT: float = 4.0
"""Timeout in seconds for commands involving core devices."""

_DDB_T = typing.Dict[str, typing.Any]  # Type for a device DB

_KC705_CORE_ADDR: str = '192.168.1.75'
"""IP address of the KC705."""

_KC705_DEVICE_DB: _DDB_T = {
    # Core devices
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': _KC705_CORE_ADDR, 'ref_period': 1e-9}
    },
    'core_log': {
        'type': 'controller',
        'host': '::1',
        'port': 1068,
        'command': 'aqctl_corelog -p {port} --bind {bind} ' + _KC705_CORE_ADDR
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
    'i2c_switch': {
        'type': 'local',
        'module': 'artiq.coredevice.i2c',
        'class': 'PCA9548'
    },

    # Onboard devices
    'user_sma_gpio_p': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 0}
    },
    'user_sma_gpio_n': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 1}
    },
    'gpio_led_2': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 2}
    },
    'gpio_led_3': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 3}
    },

    # DAC (on XADC header)
    'spi_ams101': {
        'type': 'local',
        'module': 'artiq.coredevice.spi2',
        'class': 'SPIMaster',
        'arguments': {'channel': 5}
    },
    'ttl_ams101_ldac': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 4}
    },

    # Edge counters
    'user_sma_gpio_p_ec': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {'channel': 6},
    },
    'user_sma_gpio_n_ec': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {'channel': 7},
        'sim_args': {'input_freq': 20e3}
    },

    # # SD card interface
    # 'spi_mmc': {
    #     'type': 'local',
    #     'module': 'artiq.coredevice.spi2',
    #     'class': 'SPIMaster',
    #     'arguments': {'channel': 8}
    # },

    # Aliases
    'led': 'gpio_led_2',
    'led_0': 'gpio_led_2',
    'led_1': 'gpio_led_3',
    'loop_out': 'user_sma_gpio_p',
    'loop_in': 'user_sma_gpio_n',
}
"""Device DB of the KC705."""


@dataclasses.dataclass(frozen=True)
class _CoreDevice:
    """Dataclass to hold core device information."""
    address: str
    device_db: _DDB_T

    def run_command(self, *args: str, **kwargs: typing.Any) -> subprocess.CompletedProcess:
        """Run an ARTIQ core management command.

        :param args: Additional arguments to append to the subprocess run command
        :param kwargs: Keyword arguments for the subprocess run call
        :raises subprocess.TimeoutExpired: Raised if the timeout for the command expired
        :return: A :class:`subprocess.CompletedProcess` object
        """
        command: typing.List[str] = ['artiq_coremgmt', '-D', self.address]
        command.extend(args)
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, timeout=_TIMEOUT, **kwargs)


_AVAILABLE_CORE_DEVICES: typing.List[_CoreDevice] = [
    _CoreDevice(address=_KC705_CORE_ADDR, device_db=_KC705_DEVICE_DB),
]
"""A list of available core devices."""


def _get_core_device() -> typing.Optional[_CoreDevice]:
    """Get the first available core device."""

    if CI_ENABLED and NIX_ENV and not TB_DISABLED:
        # Only find a core device if all conditions are met

        for core_device in _AVAILABLE_CORE_DEVICES:
            try:
                # Request the IP address of the core device
                r = core_device.run_command('config', 'read', 'ip')
            except subprocess.TimeoutExpired:
                pass  # Timeout, device unreachable
            else:
                if r.returncode == 0 and r.stdout.strip() == core_device.address:
                    # Use this core device
                    return core_device

    # No core device was available
    return None


_CORE_DEVICE: typing.Optional[_CoreDevice] = _get_core_device()
"""The core device to use for testing, or None if no core devices are available."""

del _get_core_device  # Remove one-time function

_LOCK_KEY: str = 'dax_hw_tb_lock'
"""Core device config key to lock the device for testing."""


@unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping hardware test')
@unittest.skipUnless(NIX_ENV, 'Not in a Nix environment, skipping hardware test')
@unittest.skipIf(TB_DISABLED, 'Hardware testbenches disabled, skipping hardware tests')
@unittest.skipIf(_CORE_DEVICE is None, 'No core device available, skipping hardware test')
class TestBenchCase(unittest.TestCase):
    """An extension of the :class:`unittest.TestCase` class which facilitates device testing.

    Users can inherit this class to create their own device/hardware test cases.
    It has predefined :func:`setUp` and :func:`tearDown` functions for resource handling.
    The :func:`construct_env` function can be used to construct test environment classes.
    """

    __E_T = typing.TypeVar('__E_T', bound=HasEnvironment)  # Type variable for environment

    @classmethod
    def setUpClass(cls) -> None:
        """Obtain a lock on the core device."""
        assert _CORE_DEVICE is not None, 'Can not set up testbench, no core device was set'

        # Read the lock status
        r = _CORE_DEVICE.run_command('config', 'read', _LOCK_KEY)
        if r.returncode != 0:
            raise unittest.SkipTest(f'Could not obtain lock status of core device at [{_CORE_DEVICE.address}] '
                                    f'(return code {r.returncode})')
        elif r.stdout.strip().isdigit():
            raise unittest.SkipTest(f'Core device at [{_CORE_DEVICE.address}] is locked by an other process')

        # Lock the device
        r = _CORE_DEVICE.run_command('config', 'write', '-s', _LOCK_KEY, JOB_ID)
        if r.returncode != 0:
            raise RuntimeError(f'Could not lock core device at [{_CORE_DEVICE.address}] (return code {r.returncode})')

        # Confirm the lock was obtained successfully (required due to the lack of an atomic read-and-lock action)
        time.sleep(_TIMEOUT + 1.0)  # Grace period
        r = _CORE_DEVICE.run_command('config', 'read', _LOCK_KEY)
        if r.returncode != 0:
            raise RuntimeError(f'Could not confirm lock status of core device at [{_CORE_DEVICE.address}] '
                               f'(return code {r.returncode})')
        elif r.stdout.strip() != JOB_ID:
            raise unittest.SkipTest(f'Core device at [{_CORE_DEVICE.address}] is locked by an other process '
                                    f'(lock was overwritten)')

    @classmethod
    def tearDownClass(cls) -> None:
        """Release the lock on the core device."""
        assert _CORE_DEVICE is not None, 'Can not set up testbench, no core device was set'

        # Release the lock
        r = _CORE_DEVICE.run_command('config', 'remove', _LOCK_KEY)
        if r.returncode != 0:
            raise RuntimeError(f'Failed to release lock on core device at [{_CORE_DEVICE.address}]'
                               f'(return code {r.returncode})')

    def setUp(self) -> None:
        """Set up the ARTIQ manager objects."""
        assert _CORE_DEVICE is not None, 'Can not set up testbench, no core device was set'
        self.__managers = dax.util.artiq.get_managers(device_db=_CORE_DEVICE.device_db)

    def tearDown(self) -> None:
        """Close the ARTIQ managers to free resources."""
        self.__managers.close()

    def construct_env(self, env_class: typing.Type[__E_T], *,
                      build_args: typing.Sequence[typing.Any] = (),
                      build_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                      **kwargs: typing.Any) -> __E_T:
        """Construct an ARTIQ environment based on the given class.

        The constructed environment can be used for testing.

        :param env_class: The environment class to construct
        :param build_args: Positional arguments passed to the build function of the environment
        :param build_kwargs: Keyword arguments passed to the build function of the environment
        :param kwargs: Keyword arguments passed to the build function of the environment (updates ``build_kwargs``)
        :return: The constructed ARTIQ environment object
        """

        if build_kwargs is None:
            # Set default value
            build_kwargs = {}
        else:
            assert isinstance(build_kwargs, dict), 'Build keyword arguments must be of type dict'
            assert all(isinstance(k, str) for k in build_kwargs), 'Keys of the build kwargs dict must be of type str'
            build_kwargs = build_kwargs.copy()  # Copy arguments to make sure the dict is not mutated

        assert issubclass(env_class, HasEnvironment), 'The environment class must be a subclass of HasEnvironment'
        assert isinstance(build_args, collections.abc.Sequence), 'Build arguments must be a sequence'
        assert isinstance(build_kwargs, dict), 'Build keyword arguments must be a dict'
        assert all(isinstance(k, str) for k in build_kwargs), 'Keys of the build kwargs dict must be of type str'

        # Merge arguments
        build_kwargs.update(kwargs)

        try:
            # Construct environment
            env = env_class(self.__managers, *build_args, **build_kwargs)
        except artiq.master.worker_db.DeviceError as e:
            # Skip test in case device errors (raising instead of calling ``self.skipTest()`` for better typing)
            assert _CORE_DEVICE is not None
            raise unittest.SkipTest(f'Core device at [{_CORE_DEVICE.address}] not available: "{str(e)}"')
        else:
            # Return the environment
            return env
