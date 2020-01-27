from artiq.language.core import *
from artiq.language.types import *
from artiq.language.units import *

import dax.sim.time as time


class Core:

    def __init__(self, dmgr):
        # Set the reference period of the core
        self.ref_period = ns

        # The core attribute should refer to itself
        self.core = self

        self._level = 0

    def run(self, k_function, k_args, k_kwargs):
        self._level += 1
        r = k_function.artiq_embedded.function(*k_args, **k_kwargs)
        self._level -= 1

        # Handle time manager for VCD output
        if self._level == 0:
            time.manager.format_timeline(self.ref_period)

        return r

    @portable
    def seconds_to_mu(self, seconds):
        return numpy.int64(seconds // self.ref_period)

    @portable
    def mu_to_seconds(self, mu):
        return mu * self.ref_period

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
