import typing
import collections
import csv
import numpy as np

from artiq.language.core import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager, DaxSignalManager
from dax.sim.sim import DAX_SIM_CONFIG_KEY


class Core(DaxSimDevice):

    def __init__(self, dmgr: typing.Any,
                 ref_period: float, ref_multiplier: int = 8,
                 **kwargs: typing.Dict[str, typing.Any]):
        assert isinstance(ref_period, float) and ref_period > 0.0, 'Reference period must be of type float'
        assert isinstance(ref_multiplier, int) and ref_multiplier > 0, 'Reference multiplier must be of type int'

        # Get the virtual simulation configuration device, which will configure the simulation
        # DAX system already initializes the virtual sim config device, this is a fallback
        self._sim_config = dmgr.get(DAX_SIM_CONFIG_KEY)

        # Call super
        super(Core, self).__init__(dmgr, _core=self, **kwargs)  # type: ignore

        # Store arguments
        self._device_manager = dmgr
        self._ref_period = ref_period
        self._ref_multiplier = np.int32(ref_multiplier)

        # Set initial call nesting level to zero
        self._level = np.int32(0)
        # Set the timescale of the core based on the simulation configuration
        self._timescale = self._sim_config.timescale  # type: float

        # Get the signal manager and register signals
        self._signal_manager = get_signal_manager()
        self._reset_signal = self._signal_manager.register(self.key, 'reset', bool, size=1)  # type: typing.Any

        # Counting dicts for function call profiling
        self._func_counter = collections.Counter()  # type: typing.Counter[typing.Any]
        self._func_time = collections.Counter()  # type: typing.Counter[typing.Any]

    @property
    def ref_period(self) -> float:
        return self._ref_period

    @property
    def ref_multiplier(self) -> np.int32:
        return self._ref_multiplier

    @property
    def coarse_ref_period(self) -> float:
        return self._ref_period * self._ref_multiplier

    def run(self, k_function: typing.Any,
            k_args: typing.Tuple[typing.Any, ...], k_kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        # Unpack function
        func = k_function.artiq_embedded.function

        # Register the function call
        self._func_counter[func] += 1
        # Track current time
        t_start = now_mu()  # type: np.int64

        # Call the kernel function while increasing the level
        self._level += 1
        result = func(*k_args, **k_kwargs)
        self._level -= 1

        # Accumulate the time spend in this function call
        self._func_time[func] += now_mu() - t_start

        if self._level == 0:
            # Flush signal manager if we are about to leave the kernel context
            self._signal_manager.flush()

        # Return the result
        return result

    def close(self) -> None:
        if self._sim_config.output_enabled:
            # Create a profiling report
            with open(self._sim_config.get_output_file_name('csv', postfix='profile'), 'w') as csv_file:
                # Open CSV writer
                csv_writer = csv.writer(csv_file)
                # Submit headers
                csv_writer.writerow(['ncalls', 'cumtime_mu', 'cumtime_s', 'function'])
                # Submit data
                csv_writer.writerows((self._func_counter[func], time, self.mu_to_seconds(time), func.__qualname__)
                                     for func, time in self._func_time.items())

    @portable
    def seconds_to_mu(self, seconds: float) -> np.int64:
        # Simulation machine units are decided by the timescale, not by the reference period
        return np.int64(seconds // self._timescale)

    @portable
    def mu_to_seconds(self, mu: np.int64) -> float:
        # Simulation machine units are decided by the timescale, not by the reference period
        return mu * self._timescale

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
        # Reset signal to 1
        self._signal_manager.event(self._reset_signal, 1)

        # Reset devices to clear buffers
        for _, d in self._device_manager.active_devices:
            if isinstance(d, DaxSimDevice):
                d.core_reset()

        # Move cursor
        delay_mu(125000)

        # Reset signal back to 0
        self._signal_manager.event(self._reset_signal, 0)

    @kernel
    def break_realtime(self) -> None:
        # Move cursor
        delay_mu(125000)
