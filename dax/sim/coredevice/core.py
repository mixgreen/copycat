import numpy as np

from dax.sim.coredevice import *
from dax.sim.sim import DAX_SIM_CONFIG_DEVICE_KEY


class Core(DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Get the virtual simulation configuration device, which will configure the simulation
        # DAX system already initializes the virtual sim config device, this is a fallback
        sim_config = dmgr.get(DAX_SIM_CONFIG_DEVICE_KEY)

        # Set the timescale of the core based on the simulation configuration
        self._timescale: float = sim_config.timescale

        # Get the signal manager
        self._signal_manager = get_signal_manager()

        # Call super
        super(Core, self).__init__(dmgr, _core=self, **kwargs)

        # Set initial call nesting level to zero
        self._level: int = 0

    def run(self, k_function, k_args, k_kwargs):
        # TODO: called for every function decorated with @kernel, also nested calls, that's what level is for

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
    def seconds_to_mu(self, seconds):
        return np.int64(seconds / self._timescale)

    @portable
    def mu_to_seconds(self, mu):
        return mu * self._timescale

    @kernel
    def reset(self):
        # There are no "pending" operations, so no queues to reset
        self.break_realtime()

    @kernel
    def break_realtime(self):
        # Move cursor by 125000 machine units
        at_mu(now_mu() + 125000)

    @kernel
    def wait_until_mu(self, cursor_mu):
        # Move time to given cursor position
        at_mu(cursor_mu)
