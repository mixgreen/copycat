import typing
import collections
import csv
import logging
import numpy as np

from artiq.language.core import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager
from dax.sim.ddb import DAX_SIM_CONFIG_KEY
from dax.sim.time import DaxTimeManager
from dax.sim.coredevice.comm_kernel import CommKernelDummy
from dax.sim.config import DaxSimConfig
from dax.util.units import time_to_str
from dax.util.output import get_file_name_generator, dummy_file_name_generator

__all__ = ['BaseCore', 'Core']

_logger: logging.Logger = logging.getLogger(__name__)
"""The logger for this file."""


class BaseCore(DaxSimDevice):
    """The base simulated coredevice driver.

    This simulated coredevice driver implements all functions required for simulation.
    It sets the DAX time manager for simulation, but does not generate any output.

    The base core can be instantiated without additional arguments and can be used
    for testing device drivers without a full ARTIQ environment.
    """

    RESET_TIME_MU: int = 125000
    """The reset time in machine units."""

    def __init__(self, dmgr: typing.Any = None,
                 ref_period: float = 1e-9, ref_multiplier: int = 8,
                 **kwargs: typing.Any):
        assert isinstance(ref_period, float) and ref_period > 0.0, 'Reference period must be of type float'
        assert isinstance(ref_multiplier, int) and ref_multiplier > 0, 'Reference multiplier must be of type int'

        if type(self) is BaseCore:
            # If the base core was instantiated directly, use a default value for `_key` required by DaxSimDevice
            kwargs.setdefault('_key', 'core')

        # Call super
        super(BaseCore, self).__init__(dmgr, _core=self, **kwargs)

        # Store arguments
        self._ref_period: float = ref_period
        self._ref_multiplier: int = ref_multiplier
        self._coarse_ref_period: float = self._ref_period * self._ref_multiplier

        # Setup dummy comm object
        self._comm = CommKernelDummy()

        # Set the time manager in ARTIQ
        _logger.debug(f'Initializing time manager with reference period {time_to_str(self.ref_period)}')
        set_time_manager(DaxTimeManager(self.ref_period))

    @property
    def ref_period(self) -> float:
        return self._ref_period

    @property
    def ref_multiplier(self) -> int:
        return self._ref_multiplier

    @property
    def coarse_ref_period(self) -> float:
        return self._coarse_ref_period

    @property
    def comm(self) -> CommKernelDummy:
        return self._comm

    def run(self, function: typing.Any,
            args: typing.Tuple[typing.Any, ...], kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        # Call the kernel function
        with sequential:  # Every function is called in a sequential context for correct parallel behavior
            result = function.artiq_embedded.function(*args, **kwargs)

        # Return the result
        return result

    def close(self) -> None:
        pass

    def compile(self, function: typing.Any, args: typing.Tuple[typing.Any, ...], kwargs: typing.Dict[str, typing.Any],
                set_result: typing.Any = None, attribute_writeback: bool = True,
                print_as_rpc: bool = True) -> typing.Any:
        raise NotImplementedError('Base core does not implement the compile function')

    @portable
    def seconds_to_mu(self, seconds: float) -> np.int64:
        # Convert seconds to machine units using the reference period
        return np.int64(seconds // self.ref_period)  # floor div, same as in ARTIQ Core

    @portable
    def mu_to_seconds(self, mu: np.int64) -> float:
        # Convert machine units to seconds using the reference period
        return mu * self.ref_period

    @kernel
    def wait_until_mu(self, cursor_mu: np.int64) -> None:
        # Move time to given cursor position if that time is in the future
        if cursor_mu > now_mu():
            at_mu(cursor_mu)

    @kernel
    def get_rtio_counter_mu(self) -> np.int64:
        # In simulation there is no difference between the RTIO counter and the cursor
        return now_mu()

    # noinspection PyUnusedLocal
    @kernel
    def get_rtio_destination_status(self, destination: np.int32) -> bool:
        # Status is always ready
        return True

    @kernel
    def reset(self) -> None:
        # Move cursor
        delay_mu(self.RESET_TIME_MU)

    @kernel
    def break_realtime(self) -> None:
        # Move cursor
        delay_mu(self.RESET_TIME_MU)


class Core(BaseCore):
    """The simulated coredevice driver with profiling features.

    This simulated coredevice driver is the default driver used during simulation.
    It inherits all the functionality of the base coredevice driver and includes
    features for signal manager output and performance profiling.

    Normally, users never instantiate this class by themselves and the ARTIQ
    device manager will take care of that.

    The signature of the :func:`__init__` function is equivalent to the ARTIQ coredevice
    driver to make sure the simulated environment has the same requirements as the
    original environment.
    """

    def __init__(self, dmgr: typing.Any,
                 ref_period: float, ref_multiplier: int = 8,
                 **kwargs: typing.Any):
        assert isinstance(ref_period, float) and ref_period > 0.0, 'Reference period must be of type float'
        assert isinstance(ref_multiplier, int) and ref_multiplier > 0, 'Reference multiplier must be of type int'

        # Get the virtual simulation configuration device, which will configure the simulation
        # DAX system already initializes the virtual sim config device, this is a fallback
        self._sim_config: DaxSimConfig = dmgr.get(DAX_SIM_CONFIG_KEY)

        # Call super
        super(Core, self).__init__(dmgr, ref_period=ref_period, ref_multiplier=ref_multiplier, **kwargs)

        # Store arguments
        self._device_manager = dmgr

        # Get file name generator (explicitly in constructor to not obtain file name too late)
        if self._sim_config.output_enabled:
            # Requesting the generator creates the parent directory, only create if output is enabled
            self._file_name_generator = get_file_name_generator(self._device_manager.get('scheduler'))
        else:
            # For completeness, set a dummy file name generator if output is disabled
            self._file_name_generator = dummy_file_name_generator

        # Get the signal manager and register signals
        self._signal_manager = get_signal_manager()
        self._reset_signal: typing.Any = self._signal_manager.register(self, 'reset', bool, size=1)

        # Set initial call nesting level to zero
        self._level: int = 0

        # Counter for context switches
        self._context_switch_counter: int = 0
        # Counting dicts for function call profiling
        self._func_counter: typing.Counter[typing.Any] = collections.Counter()
        self._func_time: typing.Counter[typing.Any] = collections.Counter()

    def run(self, function: typing.Any,
            args: typing.Tuple[typing.Any, ...], kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        # Unpack function
        kernel_func = function.artiq_embedded.function

        # Register the function call
        self._func_counter[kernel_func] += 1
        # Track current time
        t_start: np.int64 = now_mu()

        # Call the kernel function while increasing the level
        self._level += 1
        result = super(Core, self).run(function, args, kwargs)
        self._level -= 1

        # Accumulate the time spend in this function call
        self._func_time[kernel_func] += now_mu() - t_start

        if self._level == 0:
            # Flush signal manager if we are about to leave the kernel context
            self._signal_manager.flush(self.ref_period)
            # Increment the context switch counter
            self._context_switch_counter += 1

        # Return the result
        return result

    def close(self) -> None:
        # The SimConfig object will be available, even if it was closed earlier
        if self._sim_config.output_enabled:
            # Create an output file name with the earlier created file name generator
            output_file_name: str = self._file_name_generator('profile', 'csv')

            # Create a profiling report
            _logger.debug('Writing profiling report')
            with open(output_file_name, 'w') as csv_file:
                # Open CSV writer
                csv_writer = csv.writer(csv_file)
                # Submit headers
                csv_writer.writerow(['ncalls', 'cumtime_mu', 'cumtime_s', 'function'])
                # Submit context switch data
                csv_writer.writerow([self._context_switch_counter, None, None, 'Core.compile'])
                # Submit profiling data
                csv_writer.writerows((self._func_counter[func], time, self.mu_to_seconds(time), func.__qualname__)
                                     for func, time in self._func_time.items())

    def compile(self, function: typing.Any, args: typing.Tuple[typing.Any, ...], kwargs: typing.Dict[str, typing.Any],
                set_result: typing.Any = None, attribute_writeback: bool = True,
                print_as_rpc: bool = True) -> typing.Any:
        raise NotImplementedError('Simulated core does not implement the compile function')

    @kernel
    def reset(self) -> None:
        # Reset signal to 1
        self._signal_manager.event(self._reset_signal, 1)

        # Reset devices to clear buffers
        for _, d in self._device_manager.active_devices:
            if isinstance(d, DaxSimDevice):
                d.core_reset()

        # Move cursor
        delay_mu(self.RESET_TIME_MU)
        # Reset signal back to 0
        self._signal_manager.event(self._reset_signal, 0)
