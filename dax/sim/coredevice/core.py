import typing
import numpy as np

from dax.sim.coredevice import *
from dax.sim.signal import DaxSignalManager
from dax.sim.sim import DAX_SIM_CONFIG_KEY


class Core(DaxSimDevice):

    def __init__(self, dmgr: typing.Any,
                 ref_period: float, ref_multiplier: int = 8,
                 **kwargs: typing.Dict[str, typing.Any]):
        assert isinstance(ref_period, float) and ref_period > 0.0, 'Reference period must be of type float'
        assert isinstance(ref_multiplier, int) and ref_multiplier > 0, 'Reference multiplier must be of type int'

        # Get the virtual simulation configuration device, which will configure the simulation
        # DAX system already initializes the virtual sim config device, this is a fallback
        sim_config: typing.Any = dmgr.get(DAX_SIM_CONFIG_KEY)

        # Call super
        super(Core, self).__init__(dmgr, _core=self, **kwargs)  # type: ignore

        # Store arguments
        self._device_manager: typing.Any = dmgr
        self._ref_period: float = ref_period
        self._ref_multiplier: np.int32 = np.int32(ref_multiplier)

        # Set the timescale of the core based on the simulation configuration
        self._timescale: float = sim_config.timescale
        # Get the signal manager
        self._signal_manager: DaxSignalManager = get_signal_manager()

        # Set initial call nesting level to zero
        self._level: int = 0

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
        # Call the kernel function while increasing the level
        self._level += 1
        result = k_function.artiq_embedded.function(*k_args, **k_kwargs)
        self._level -= 1

        if self._level == 0:
            # Flush signal manager if we are about to leave the kernel context
            self._signal_manager.flush()

        # Return the result
        return result

    @portable
    def seconds_to_mu(self, seconds: float) -> np.int64:
        return np.int64(seconds / self._timescale)

    @portable
    def mu_to_seconds(self, mu: np.int64) -> float:
        return mu * self._timescale

    @kernel
    def reset(self) -> None:
        # Reset devices
        for _, d in self._device_manager.active_devices:
            if isinstance(d, DaxSimDevice):
                d.core_reset()

        # Move cursor
        at_mu(now_mu() + 125000)

    @kernel
    def break_realtime(self) -> None:
        # Move cursor
        at_mu(now_mu() + 125000)

    @kernel
    def wait_until_mu(self, cursor_mu: np.int64) -> None:
        # Move time to given cursor position if that time is in the future
        if cursor_mu > now_mu():
            at_mu(cursor_mu)
