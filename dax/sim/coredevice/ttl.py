# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

import random
import collections
import itertools
import enum
import numpy as np
import typing

from artiq.language.core import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class TTLOut(DaxSimDevice):

    def __init__(self, dmgr: typing.Any, **kwargs: typing.Any):
        # Call super
        super(TTLOut, self).__init__(dmgr, **kwargs)

        # Register signals
        self._signal_manager = get_signal_manager()
        self._state = self._signal_manager.register(self, 'state', bool, size=1)

    @kernel
    def output(self):
        pass

    @kernel
    def set_o(self, o):
        self._signal_manager.event(self._state, 1 if o else 0)

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

    def __init__(self, dmgr: typing.Any,
                 input_freq: float = 0.0, input_stdev: float = 0.0, input_prob: float = 0.5,
                 seed: typing.Optional[int] = None, **kwargs: typing.Any):
        """Simulation driver for :class:`artiq.coredevice.ttl.TTLInOut`.

        :param input_freq: Simulated input frequency for gate operations
        :param input_stdev: Simulated input frequency standard deviation for gate operations
        :param input_prob: Probability of a high signal when using :func:`sample_input`
        :param seed: Seed for the random number generator used for simulating input
        """
        assert isinstance(input_freq, float) and input_freq >= 0.0, 'Input frequency must be a positive float'
        assert isinstance(input_stdev, float) and input_stdev >= 0.0, 'Input stdev must be a non-negative float'
        assert isinstance(input_prob, float), 'Input probability must be a float'
        assert 0.0 <= input_prob <= 1.0, 'Input probability must be between 0.0 and 1.0'

        # Call super
        super(TTLInOut, self).__init__(dmgr, **kwargs)

        # Store simulation settings
        self._input_freq: float = input_freq
        self._input_stdev: float = input_stdev
        self._input_prob: float = input_prob

        # Random number generator for generating values
        self._rng = random.Random(seed)

        # Buffers to store simulated events
        self._edge_buffer: typing.Deque[np.int64] = collections.deque()
        self._sample_buffer: typing.Deque[np.int32] = collections.deque()

        # Register signals
        self._direction = self._signal_manager.register(self, 'direction', bool, size=1)
        self._sensitivity = self._signal_manager.register(self, 'sensitivity', bool, size=1)

    def core_reset(self) -> None:
        # Clear buffers
        self._edge_buffer.clear()
        self._sample_buffer.clear()

    @kernel
    def set_oe(self, oe):
        # 0 = input, 1 = output
        self._signal_manager.event(self._direction, 1 if oe else 0)
        self._signal_manager.event(self._sensitivity, 'z' if oe else 0)
        self._signal_manager.event(self._state, 'x' if oe else 'z')

    @kernel
    def output(self):
        self.set_oe(True)

    @kernel
    def input(self):
        self.set_oe(False)

    def _simulate_input_signal(self, duration: np.int64, edge_type: _EdgeType) -> None:
        """Simulate input signal for a given duration."""

        # Decide event frequency
        # Multiply by 2 to simulate a full duty cycle (rising and falling edge)
        event_freq = self._rng.normalvariate(self._input_freq, self._input_stdev) * 2

        # Calculate the number of events we expect to observe based on duration and frequency
        num_events = int(self.core.mu_to_seconds(duration) * event_freq)

        # Generate relative timestamps for these events in machine units
        timestamps = np.asarray(self._rng.sample(range(duration), num_events), dtype=np.int64)
        # Sort timestamps
        timestamps.sort()

        # Initialize the signal to 0 at the start of the window (for graphical purposes)
        self._signal_manager.event(self._state, 0)

        # Write the stream of input events to the signal manager
        for t, v in zip(timestamps, itertools.cycle((1, 0))):
            self._signal_manager.event(self._state, v, offset=t)

        if edge_type is self._EdgeType.RISING:
            # Store odd half of the event times in the event buffer
            self._edge_buffer.extend(timestamps[1::2] + now_mu())
        elif edge_type is self._EdgeType.FALLING:
            # Store even half of the event times in the event buffer
            self._edge_buffer.extend(timestamps[::2] + now_mu())
        elif edge_type is self._EdgeType.BOTH:
            # Store all event times in the event buffer
            self._edge_buffer.extend(timestamps + now_mu())

        # Move the cursor
        delay_mu(duration)

        # Return to Z after all signals were inserted
        self._signal_manager.event(self._state, 'z')

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

    # noinspection PyUnusedLocal
    @kernel
    def count(self, up_to_timestamp_mu):
        # This function does not interact with the timeline
        count = len(self._edge_buffer)
        self._edge_buffer.clear()

        # Return the count
        return count

    # noinspection PyUnusedLocal
    @kernel
    def timestamp_mu(self, up_to_timestamp_mu):
        # This function does not interact with the timeline
        return self._edge_buffer.popleft() if len(self._edge_buffer) else -1

    @kernel
    def sample_input(self):
        # Sample at the current time and store result in the sample buffer
        val = np.int32(self._rng.random() < self._input_prob)
        self._sample_buffer.append(val)
        self._signal_manager.event(self._state, val)  # Sample value at current time
        self._signal_manager.event(self._state, 'z', offset=1)  # Return to 'Z' 1 machine unit after sample

    @kernel
    def sample_get(self):
        if len(self._sample_buffer):
            # Return a sample from the buffer
            return self._sample_buffer.popleft()
        else:
            # Not samples available
            raise IndexError(f'Device "{self.key}" has no sample to return')

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
        self._freq = self._signal_manager.register(self, 'freq', float)

    @portable
    def frequency_to_ftw(self, frequency):
        return round(float(2 ** self._acc_width * frequency * self.core.coarse_ref_period))

    @portable
    def ftw_to_frequency(self, ftw):
        return ftw / self.core.coarse_ref_period / 2 ** self._acc_width

    @kernel
    def set_mu(self, frequency):
        self.set(self.ftw_to_frequency(frequency))

    @kernel
    def set(self, frequency):
        self._signal_manager.event(self._freq, float(frequency))

    @kernel
    def stop(self):
        self.set(0.0)
