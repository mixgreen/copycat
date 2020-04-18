import random
import collections
import itertools
import enum
import numpy as np

from dax.sim.coredevice import *


class TTLOut(DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(TTLOut, self).__init__(dmgr, **kwargs)

        # Register signals
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
    class _EdgeType(enum.IntEnum):
        NONE = 0
        RISING = 1
        FALLING = 2
        BOTH = 3

    def __init__(self, dmgr, input_freq=100 * kHz, **kwargs):
        assert isinstance(input_freq, float) and input_freq > 0.0, 'Input frequency must be a positive float'

        # Call super
        super(TTLInOut, self).__init__(dmgr, **kwargs)

        # Store simulation settings
        self._input_freq = input_freq

        # Random number generator for generating values
        self._rng = random.Random()

        # Buffers to store simulated events
        self._edges = collections.deque()
        self._samples = collections.deque()

        # Register signals
        self._direction = self._signal_manager.register(self.key, 'direction', bool, size=1)
        self._sensitivity = self._signal_manager.register(self.key, 'sensitivity', bool, size=1)
        self._count = self._signal_manager.register(self.key, 'count', np.int32)

    def core_reset(self) -> None:
        # Clear buffers
        self._edges.clear()
        self._samples.clear()

    @kernel
    def set_oe(self, oe):
        # 0 = input, 1 = output
        self._signal_manager.event(self._direction, 1 if oe else 0)
        self._signal_manager.event(self._sensitivity, 'x' if oe else 0)
        self._signal_manager.event(self._count, 0)
        self._signal_manager.event(self._value, 'x')

    @kernel
    def output(self):
        self.set_oe(True)

    @kernel
    def input(self):
        self.set_oe(False)

    def _simulate_input_signal(self, duration: np.int64, edge_type: _EdgeType) -> None:
        """Simulate input signal for a given duration."""

        # Calculate the number of events we expect to observe based on duration and frequency
        # Multiply by 2 to simulate a full duty cycle (rising and falling edge)
        num_events = self.core.mu_to_seconds(duration) * self._input_freq * 2

        # Generate timestamps for these events in machine units
        now = now_mu()
        timestamps = self._rng.sample(range(now, now + duration), num_events)
        # Write the stream of input events to the signal manager
        for t, v in zip(timestamps, itertools.cycle((0, 1))):
            self._signal_manager.event(self._value, v, t)

        if edge_type is self._EdgeType.RISING:
            # Store odd half of the event times in the event buffer
            self._edges.extend(timestamps[1::2])
        elif edge_type is self._EdgeType.FALLING:
            # Store even half of the event times in the event buffer
            self._edges.extend(timestamps[::2])
        elif edge_type is self._EdgeType.BOTH:
            # Store all event times in the event buffer
            self._edges.extend(timestamps)

        # Move the cursor
        delay_mu(duration)

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
    def count(self, up_to_timestamp_mu):
        # This function does not interact with the timeline
        count = len(self._edges)
        self._edges.clear()

        # Return the count
        return count

    @kernel
    def timestamp_mu(self, up_to_timestamp_mu):
        # This function does not interact with the timeline
        return self._edges.popleft() if len(self._edges) else -1

    @kernel
    def sample_input(self):
        # Sample at the current time and store result in the sample buffer
        val = np.int32(self._rng.randint(0, 1))
        self._samples.append(val)
        self._signal_manager.event(self._value, val)  # Sample value at current time

    @kernel
    def sample_get(self):
        if len(self._samples):
            # Return a sample from the buffer
            return self._samples.popleft()
        else:
            # Not samples available
            raise IndexError(f'Device "{self.key:s}" has no sample to return')

    @kernel
    def sample_get_nonrt(self):
        self.sample_input()
        r = self.sample_get()
        self.core.break_realtime()
        return r

    def watch_stay_on(self):
        raise NotImplementedError

    def watch_stay_off(self):
        raise NotImplementedError

    def watch_done(self):
        raise NotImplementedError


class TTLClockGen(DaxSimDevice):

    def __init__(self, dmgr, acc_width=24, **kwargs):
        # Call super
        super(TTLClockGen, self).__init__(dmgr, **kwargs)

        # Store parameters
        self._acc_width = np.int64(acc_width)

        # Register signals
        self._signal_manager = get_signal_manager()
        self._value = self._signal_manager.register(self.key, 'freq', float)

    @portable
    def frequency_to_ftw(self, frequency):
        return round(2 ** self._acc_width * frequency * self.core.coarse_ref_period)

    @portable
    def ftw_to_frequency(self, ftw):
        return ftw / self.core.coarse_ref_period / 2 ** self._acc_width

    @kernel
    def set_mu(self, frequency):
        self.set(self.ftw_to_frequency(frequency))

    @kernel
    def set(self, frequency):
        self._signal_manager.event(self._value, frequency)

    @kernel
    def stop(self):
        self.set(0)
