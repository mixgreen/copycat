import random
import collections
import enum
import numpy as np

from artiq.coredevice.edge_counter import CounterOverflow

from dax.sim.coredevice import *


class EdgeCounter(DaxSimDevice):
    class _EdgeType(enum.IntEnum):
        NONE = 0
        RISING = 1
        FALLING = 2
        BOTH = 3

    def __init__(self, dmgr, input_freq=100 * kHz, gateware_width=31, **kwargs):
        assert isinstance(input_freq, float) and input_freq > 0.0, 'Input frequency must be a positive float'
        assert isinstance(gateware_width, int), 'Gateware width must be of type int'

        # Call super
        super(EdgeCounter, self).__init__(dmgr, **kwargs)

        # From ARTIQ code
        self.counter_max = (1 << (gateware_width - 1)) - 1

        # Store simulation settings
        self._input_freq = input_freq

        # Random number generator for generating values
        self._rng = random.Random()

        # Buffers to store counts
        self._counts = collections.deque()

        # Register signals
        self._signal_manager = get_signal_manager()
        self._value = self._signal_manager.register(self.key, 'count', int)
        self._sensitivity = self._signal_manager.register(self.key, 'sensitivity', bool, size=1)

    def core_reset(self) -> None:
        # Clear buffers
        self._counts.clear()

    def _simulate_input_signal(self, duration: np.int64, edge_type: _EdgeType) -> None:
        """Simulate input signal for a given duration."""

        # Move the cursor
        delay_mu(duration)

        # Calculate the number of events we expect to observe based on duration and frequency
        num_events = self.core.mu_to_seconds(duration) * self._input_freq * 2
        # Multiply by 2 in case we detect both edges
        if edge_type is self._EdgeType.BOTH:
            num_events *= 2

        # Store number of events and the timestamp in count buffer
        self._counts.append((now_mu(), num_events))

    @kernel
    def gate_rising_mu(self, duration):
        self._signal_manager.event(self._sensitivity, 1)
        self._simulate_input_signal(duration, self._EdgeType.RISING)
        self._signal_manager.event(self._sensitivity, 0)
        return now_mu()

    @kernel
    def gate_falling_mu(self, duration):
        self._signal_manager.event(self._sensitivity, 1)
        self._simulate_input_signal(duration, self._EdgeType.FALLING)
        self._signal_manager.event(self._sensitivity, 0)
        return now_mu()

    @kernel
    def gate_both_mu(self, duration):
        self._signal_manager.event(self._sensitivity, 1)
        self._simulate_input_signal(duration, self._EdgeType.BOTH)
        self._signal_manager.event(self._sensitivity, 0)
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
    def set_config(self, count_rising: TBool, count_falling: TBool,
                   send_count_event: TBool, reset_to_zero: TBool):
        raise NotImplementedError

    @kernel
    def fetch_count(self) -> TInt32:
        if len(self._counts):
            # Get count from the buffer (drop the timestamp)
            _, count = self._counts.popleft()

            if count >= self.counter_max:
                # Count overflow
                raise CounterOverflow(f'Input edge counter overflow for device {self.key:s}')

            # Return the result
            return count
        else:
            # No count available to return
            raise IndexError(f'Device "{self.key:s}" has no count to return')

    @kernel
    def fetch_timestamped_count(self, timeout_mu=np.int64(-1)) -> TTuple([TInt64, TInt32]):
        if len(self._counts):
            # Get count and timestamp from the buffer
            timestamp, count = self._counts.popleft()

            if count >= self.counter_max:
                # Count overflow
                raise CounterOverflow(f'Input edge counter overflow for device {self.key:s}')

            # Return the result
            return timestamp, count
        else:
            # No count available to return
            return -1, 0
