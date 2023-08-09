import typing
import collections
import csv
import logging
import numpy as np

from artiq.language.core import kernel, portable, rpc, at_mu, now_mu, delay_mu, sequential, set_time_manager
from artiq.language.types import TInt64
import artiq.coredevice.core

from dax.util.artiq_version import ARTIQ_MAJOR_VERSION
from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager, DaxSignalManager, Signal
from dax.sim.ddb import DAX_SIM_CONFIG_KEY
from dax.sim.time import DaxTimeManager
from dax.sim.coredevice.comm_kernel import CommKernelDummy
from dax.sim.config import DaxSimConfig
from dax.util.units import time_to_str
from dax.util.output import FileNameGenerator, BaseFileNameGenerator

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

    DEFAULT_RESET_TIME_MU: typing.ClassVar[np.int64] = np.int64(125000)
    """The default reset time in machine units."""

    _ref_period: float
    _ref_multiplier: int
    _coarse_ref_period: float
    _reset_mu: np.int64
    _break_realtime_mu: np.int64

    kernel_invariants: typing.Set[str] = {
        'core', 'ref_period', 'ref_multiplier', 'coarse_ref_period',
        '_reset_mu', '_break_realtime_mu',
    }

    def __init__(self, dmgr: typing.Any = None,
                 ref_period: float = 1e-9, ref_multiplier: int = 8, *,
                 reset_mu: np.int64 = DEFAULT_RESET_TIME_MU, break_realtime_mu: np.int64 = DEFAULT_RESET_TIME_MU,
                 **kwargs: typing.Any):
        assert isinstance(ref_period, float), 'Reference period must be of type float'
        assert ref_period > 0.0, 'Reference period must be greater than zero'
        assert isinstance(ref_multiplier, int), 'Reference multiplier must be of type int'
        assert ref_multiplier > 0, 'Reference multiplier must be greater than zero'
        assert isinstance(reset_mu, (int, np.int32, np.int64)), 'Reset mu must be of type int'
        assert reset_mu >= 0, 'Reset mu must be zero or greater'
        assert isinstance(break_realtime_mu, (int, np.int32, np.int64)), 'Break realtime mu must be of type int'
        assert break_realtime_mu >= 0, 'Break realtime mu must be zero or greater'

        if type(self) is BaseCore:
            # If the base core was instantiated directly, use a default value for _key required by DaxSimDevice
            kwargs.setdefault('_key', 'core')

        # Call super
        super(BaseCore, self).__init__(dmgr, _core=self, **kwargs)

        # Store arguments
        self._ref_period = ref_period
        self._ref_multiplier = ref_multiplier
        self._coarse_ref_period = self._ref_period * self._ref_multiplier
        self._reset_mu = np.int64(reset_mu)
        self._break_realtime_mu = np.int64(break_realtime_mu)

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
        embedded_function = function.artiq_embedded.function

        # If kernel function defined via `kernel_from_string`, use `exec` to obtain the function
        # named `kernel_from_string_fn` (name fixed by `kernel_from_string`).
        if isinstance(embedded_function, str):
            ctxt = {}  # type: ignore
            exec(embedded_function, ctxt)
            embedded_function = ctxt['kernel_from_string_fn']

        # Every function is called in a sequential context for deep parallel behavior with sequential function entry
        with sequential:
            # Call the kernel function
            result = embedded_function(*args, **kwargs)

        # Return the result
        return result

    def close(self) -> None:
        pass

    def compile(self, function: typing.Any, args: typing.Sequence[typing.Any], kwargs: typing.Dict[str, typing.Any],
                *args_: typing.Any, **kwargs_: typing.Any) -> typing.Any:
        raise RuntimeError('Compile function not available')

    @portable
    def seconds_to_mu(self, seconds):  # type: (float) -> np.int64
        # Convert seconds to machine units using the reference period
        return np.int64(seconds // self.ref_period)  # floor div, same as in ARTIQ Core

    @portable
    def mu_to_seconds(self, mu):  # type: (np.int64) -> float
        # Convert machine units to seconds using the reference period
        return float(mu * self.ref_period)

    @kernel
    def wait_until_mu(self, cursor_mu):  # type: (np.int64) -> None
        # Move time to given cursor position if that time is in the future
        if cursor_mu > now_mu():
            at_mu(cursor_mu)

    @kernel
    def get_rtio_counter_mu(self):  # type: () -> np.int64
        # In simulation without signal manager, there is no difference between the RTIO counter and the cursor
        return now_mu()

    # noinspection PyUnusedLocal
    @kernel
    def get_rtio_destination_status(self, destination):  # type: (np.int32) -> bool
        # Status is always ready
        return True

    @kernel
    def reset(self):  # type: () -> None
        # Set timeline cursor to the time horizon
        at_mu(self.get_rtio_counter_mu())
        # Move cursor
        delay_mu(self._reset_mu)

    @kernel
    def break_realtime(self):  # type: () -> None
        # Set timeline cursor to the time horizon
        at_mu(self.get_rtio_counter_mu())
        # Move cursor
        delay_mu(self._break_realtime_mu)

    if ARTIQ_MAJOR_VERSION >= 7:
        def precompile(
                self, function, *args, **kwargs
        ):  # type: (typing.Callable[..., typing.Any], typing.Any, typing.Any) -> typing.Callable[[], typing.Any]
            def precompiled_fn() -> typing.Any:
                return function(*args, **kwargs)

            return precompiled_fn


class Core(BaseCore):
    """The simulated coredevice driver with profiling features.

    This simulated coredevice driver is the default driver used during simulation.
    It inherits all the functionality of the base coredevice driver and includes
    features for signal manager output and performance profiling.

    Normally, users never instantiate this class directly and the ARTIQ
    device manager will take care of that.

    The signature of the :func:`__init__` function is equivalent to the ARTIQ coredevice
    driver to make sure the simulated environment has the same requirements as the
    original environment.

    This coredevice driver has a flag to enable compilation of kernels before simulation.
    The idea is that by compiling kernels during simulation, compile errors are exposed.
    At this moment, almost no simulation drivers support compilation.
    Hence, the compilation feature is currently not very useful.
    """

    _sim_config: DaxSimConfig
    _file_name_generator: BaseFileNameGenerator
    _signal_manager: DaxSignalManager[typing.Any]
    _reset_signal: Signal
    _level: int
    _context_switch_counter: int
    _fn_counter: typing.Counter[typing.Any]
    _fn_time: typing.Counter[typing.Any]
    _compiler: typing.Optional[artiq.coredevice.core.Core]

    # noinspection PyShadowingBuiltins
    def __init__(self, dmgr: typing.Any, ref_period: float, ref_multiplier: int = 8, *,
                 compile: bool = False, **kwargs: typing.Any):
        """Simulation driver for :class:`artiq.coredevice.core.Core`.

        :param compile: If :const:`True`, compile kernels before simulation (see also :class:`Core`)
        """
        assert isinstance(compile, bool), 'Compile flag must be of type bool'

        # Get the virtual simulation configuration device, which will configure the simulation
        # DAX system already initializes the virtual sim config device, this is a fallback
        self._sim_config = dmgr.get(DAX_SIM_CONFIG_KEY)

        # Call super
        super(Core, self).__init__(dmgr, ref_period=ref_period, ref_multiplier=ref_multiplier, **kwargs)

        # Store arguments
        self._device_manager = dmgr

        # Get file name generator (explicitly in constructor to not obtain file name too late)
        if self._sim_config.output_enabled:
            # Requesting the generator creates the parent directory, only create if output is enabled
            self._file_name_generator = FileNameGenerator(self._device_manager.get('scheduler'))
        else:
            # For completeness, set a base file name generator if output is disabled
            self._file_name_generator = BaseFileNameGenerator()

        # Get the signal manager and register signals
        self._signal_manager = get_signal_manager()
        self._reset_signal = self._signal_manager.register(self, 'reset', bool, size=1)

        # Set initial call nesting level to zero
        self._level = 0

        # Counter for context switches
        self._context_switch_counter = 0
        # Counting dicts for function call profiling
        self._fn_counter = collections.Counter()
        self._fn_time = collections.Counter()

        # Configure compiler
        if compile:
            core_kwargs = {k: v for k, v in kwargs.items() if k in {'target'}}
            core_dmgr: typing.Dict[str, typing.Any] = {}
            self._compiler = artiq.coredevice.core.Core(
                core_dmgr, host=None, ref_period=ref_period, ref_multiplier=ref_multiplier, **core_kwargs)
            core_dmgr[self.key] = self._compiler  # Set the device manager core to reference itself
            _logger.debug('Kernel compilation during simulation enabled')
        else:
            self._compiler = None

    def run(self, function: typing.Any,
            args: typing.Tuple[typing.Any, ...], kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        if self._level == 0 and self._compiler is not None:
            # Compile the kernel
            self.compile(function, args, kwargs)

        # Unpack function
        kernel_fn = function.artiq_embedded.function

        # Register the function call
        self._fn_counter[kernel_fn] += 1
        # Track current time
        t_start: np.int64 = now_mu()

        # Increase level
        self._level += 1
        try:
            # Call the kernel function
            result = super(Core, self).run(function, args, kwargs)
        finally:
            # Decrease level
            self._level -= 1

            if self._level == 0:
                # Flush signal manager if we are about to leave the kernel context
                self._signal_manager.flush(self.ref_period)
                # Increment the context switch counter
                self._context_switch_counter += 1

        # Accumulate the time spend in this function call
        self._fn_time[kernel_fn] += int(now_mu() - t_start)

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
                csv_writer.writerows((self._fn_counter[fn], time, self.mu_to_seconds(time), fn.__qualname__)
                                     for fn, time in self._fn_time.items())

    def compile(self, function: typing.Any, args: typing.Sequence[typing.Any], kwargs: typing.Dict[str, typing.Any],
                *args_: typing.Any, **kwargs_: typing.Any) -> typing.Any:
        if self._compiler is not None:
            _logger.debug('Compiling kernel...')
            result = self._compiler.compile(function, args, kwargs, *args_, **kwargs_)
            _, kernel_library, _, _ = result
            _logger.debug(f'Kernel size: {len(kernel_library)} bytes')
            return result
        else:
            return super(Core, self).compile(function, args, kwargs, *args_, **kwargs_)

    @kernel
    def get_rtio_counter_mu(self):  # type: () -> np.int64
        # RTIO counter value is estimated by the horizon
        return self._get_rtio_counter_mu()

    @rpc
    def _get_rtio_counter_mu(self) -> TInt64:
        return self._signal_manager.horizon()

    @kernel
    def reset(self):  # type: () -> None
        self._reset()

    # noinspection PyTypeHints
    @rpc
    def _reset(self, *, push_start=True, push_end=True):  # type: (bool, bool) -> None
        # Set timeline cursor to the time horizon
        at_mu(self._signal_manager.horizon())

        if push_start:
            # Reset signal to 1
            self._reset_signal.push(True)

        # Reset devices to clear buffers
        for _, d in self._device_manager.active_devices:
            if isinstance(d, DaxSimDevice):
                d.core_reset()

        # Move cursor
        delay_mu(self._reset_mu)

        if push_end:
            # Reset signal back to 0
            self._reset_signal.push(False)
