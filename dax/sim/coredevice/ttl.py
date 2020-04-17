import random
import numpy as np

from dax.sim.coredevice import *


class TTLOut(DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(TTLOut, self).__init__(dmgr, **kwargs)

        # Register variables
        self._signal_manager = get_signal_manager()
        self._value = self._signal_manager.register(self.key, 'value', bool, size=1)

    @kernel
    def output(self):
        pass

    @kernel
    def set_o(self, o):
        self._signal_manager.event(self._value, 1 if o else 0)

    @kernel
    def on(self):
        self.set_o(True)

    @kernel
    def off(self):
        self.set_o(False)

    @kernel
    def pulse_mu(self, duration):
        self.on()
        delay_mu(duration)
        self.off()

    @kernel
    def pulse(self, duration):
        self.on()
        delay(duration)
        self.off()


class TTLInOut(TTLOut):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(TTLInOut, self).__init__(dmgr, **kwargs)

        # Random number generator for generating values
        self._rng = random.Random()

        # Register variables
        self._direction = self._signal_manager.register(self.key, 'direction', bool, size=1)
        self._count = self._signal_manager.register(self.key, 'count', np.int64)

    @kernel
    def set_oe(self, oe):
        # 0 = input, 1 = output
        self._signal_manager.event(self._direction, oe)
        self._signal_manager.event(self._value, 'x')

    @kernel
    def output(self):
        self.set_oe(True)

    @kernel
    def input(self):
        self.set_oe(False)

    @kernel
    def gate_rising_mu(self, duration):
        delay_mu(duration)
        return now_mu()

    @kernel
    def gate_falling_mu(self, duration):
        delay_mu(duration)
        return now_mu()

    @kernel
    def gate_both_mu(self, duration):
        delay_mu(duration)
        return now_mu()

    @kernel
    def gate_rising(self, duration):
        return self.gate_rising_mu(self.core.seconds_to_mu(duration))

    @kernel
    def gate_falling(self, duration):
        return self.gate_falling_mu(self.core.seconds_to_mu(duration))

    @kernel
    def gate_both(self, duration):
        return self.gate_both_mu(self.core.seconds_to_mu(duration))

    @kernel
    def count(self, up_to_timestamp_mu):
        at_mu(up_to_timestamp_mu)
        count = self._rng.randrange(0, 100)
        self._signal_manager.event(self._count, count)
        return count

    @kernel
    def timestamp_mu(self, up_to_timestamp_mu):
        result = np.int64(self._rng.randrange(now_mu(), up_to_timestamp_mu))
        at_mu(result)
        self._signal_manager.event(self._count, 1)
        return result
