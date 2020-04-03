import random

from artiq.language.core import *
from artiq.language.types import *
from artiq.language.units import *

import dax.sim.time as time


class TTLOut:

    def __init__(self, dmgr, name):
        self.core = dmgr.get("core")
        self.name = name

        # Register variables
        self._value = time.manager.register(self.name, 'value', 'reg', size=1)

    @kernel
    def set_o(self, value):
        time.manager.event(self._value, value)

    @kernel
    def pulse(self, duration):
        self.set_o(True)
        delay(duration)
        self.set_o(False)

    @kernel
    def pulse_mu(self, duration):
        self.set_o(True)
        delay_mu(duration)
        self.set_o(False)

    @kernel
    def on(self):
        self.set_o(True)

    @kernel
    def off(self):
        self.set_o(False)

    @kernel
    def output(self):
        pass


class TTLInOut(TTLOut):

    def __init__(self, dmgr, name):
        super(TTLInOut, self).__init__(dmgr, name)

        # Random number generator for generating values
        self.rng = random.Random()

        # Register variables
        self._direction = time.manager.register(self.name, 'direction', 'reg', size=1)
        self._count = time.manager.register(self.name, 'count', 'integer')

    @kernel
    def gate_rising(self, duration):
        delay(duration)
        return now_mu()

    @kernel
    def gate_falling(self, duration):
        delay(duration)
        return now_mu()

    @kernel
    def gate_both(self, duration):
        delay(duration)
        return now_mu()

    @kernel
    def count(self, up_to_timestamp_mu):
        at_mu(up_to_timestamp_mu)
        result = self.rng.randrange(0, 100)
        time.manager.event(self._count, result)
        return result

    @kernel
    def timestamp_mu(self, up_to_timestamp_mu):
        result = time.manager.get_time_mu()
        result += self.rng.randrange(10, 1000)
        at_mu(result)
        time.manager.event(self._count, result)
        return result

    @kernel
    def set_oe(self, oe):
        # 0 = input, 1 = output
        time.manager.event(self._direction, oe)

    @kernel
    def output(self):
        self.set_oe(True)

    @kernel
    def input(self):
        self.set_oe(False)

    @kernel
    def gate_rising_mu(self, duration):
        return self.gate_rising(self.core.mu_to_seconds(duration))

    @kernel
    def gate_falling_mu(self, duration):
        return self.gate_rising(self.core.mu_to_seconds(duration))

    @kernel
    def gate_both_mu(self, duration):
        return self.gate_rising(self.core.mu_to_seconds(duration))
