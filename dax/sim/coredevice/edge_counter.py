import random
import collections
import enum
import numpy as np

from artiq.coredevice.edge_counter import CounterOverflow

from artiq.language.core import *
from artiq.language.units import *
from artiq.language.types import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class EdgeCounter(DaxSimDevice):
    class _EdgeType(enum.IntEnum):
        """Enum class for the edge type."""
        NONE = 0
        RISING = 1
        FALLING = 2
        BOTH = 3

    def __init__(self, dmgr, input_freq=0.0, gateware_width=31, **kwargs):
        assert isinstance(input_freq, float) and input_freq >= 0.0, 'Input frequency must be a positive float'
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
        self._count_buffer = collections.deque()

        # Register signals
        self._signal_manager = get_signal_manager()
        self._count = self._signal_manager.register(self.key, 'count', int, init='z')

    def core_reset(self) -> None:
        # Clear buffers
        self._count_buffer.clear()

    def _simulate_input_signal(self, duration: np.int64, edge_type: _EdgeType) -> None:
        """Simulate input signal for a given duration."""

        # Calculate the number of events we expect to observe based on duration and frequency
        num_events = int(self.core.mu_to_seconds(duration) * self._input_freq)
        # Multiply by 2 in case we detect both edges
        if edge_type is self._EdgeType.BOTH:
            num_events *= 2

        # Set the number of counts for the duration window (for graphical purposes)
        self._signal_manager.event(self._count, num_events)
        delay_mu(duration)  # Move the cursor
        self._signal_manager.event(self._count, 'z')

        # Store number of events and the timestamp in count buffer
        self._count_buffer.append((now_mu(), num_events))

    @kernel
    def gate_rising_mu(self, duration):
        self._simulate_input_signal(duration, self._EdgeType.RISING)
        return now_mu()

    @kernel
    def gate_falling_mu(self, duration):
        self._simulate_input_signal(duration, self._EdgeType.FALLING)
        return now_mu()

    @kernel
    def gate_both_mu(self, duration):
        self._simulate_input_signal(duration, self._EdgeType.BOTH)
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
    def set_config(self, count_rising: TBool, count_falling: TBool, send_count_event: TBool, reset_to_zero: TBool):
        raise NotImplementedError

    @kernel
    def fetch_count(self) -> TInt32:
        if len(self._count_buffer):
            # Get count from the buffer (drop the timestamp)
            _, count = self._count_buffer.popleft()

            if count >= self.counter_max:
                # Count overflow
                raise CounterOverflow('Input edge counter overflow for device {:s}'.format(self.key))

            # Return the result
            return count
        else:
            # No count available to return
            raise IndexError('Device "{:s}" has no count to return'.format(self.key))

    # noinspection PyUnusedLocal
    @kernel
    def fetch_timestamped_count(self, timeout_mu=np.int64(-1)) -> TTuple([TInt64, TInt32]):  # type: ignore
        if len(self._count_buffer):
            # Get count and timestamp from the buffer
            timestamp, count = self._count_buffer.popleft()

            if count >= self.counter_max:
                # Count overflow
                raise CounterOverflow('Input edge counter overflow for device {:s}'.format(self.key))

            # Return the result
            return timestamp, count
        else:
            # No count available to return
            return -1, 0
