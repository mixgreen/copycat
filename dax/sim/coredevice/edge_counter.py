# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

import typing
import collections
import enum
import random
import dataclasses
from numpy import int32, int64

from artiq.coredevice.edge_counter import CounterOverflow

from artiq.language.core import kernel, delay_mu, now_mu, at_mu
from artiq.language.types import TBool, TInt32, TInt64, TTuple

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager, Signal


class _EdgeType(enum.IntEnum):
    """Enum class for the edge type."""
    NONE = 0
    RISING = 1
    FALLING = 2
    BOTH = 3

    # Not using postponed evaluation of annotations to prevent ARTIQ compiler errors
    @classmethod
    def from_bool(cls, *, rising: bool, falling: bool) -> '_EdgeType':
        assert isinstance(rising, bool), 'Rising flag must be of type bool'
        assert isinstance(falling, bool), 'Falling flag must be of type bool'
        return _EdgeType(cls.RISING * rising + cls.FALLING * falling)


@dataclasses.dataclass(frozen=True)
class _Config:
    """Data class to store EdgeCounter gate configuration."""
    edge_type: _EdgeType
    timestamp: int64


class EdgeCounter(DaxSimDevice):
    counter_max: int
    _count_buffer: typing.Deque[typing.Tuple[int64, int]]
    _prev_config: typing.Optional[_Config]
    _count: Signal
    _input_freq: Signal
    _input_stdev: Signal

    def __init__(self, dmgr: typing.Any, channel: typing.Optional[int] = None, gateware_width: int = 31, *,
                 input_freq: float = 0.0, input_stdev: float = 0.0, seed: typing.Optional[int] = None,
                 **kwargs: typing.Any):
        """Simulation driver for :class:`artiq.coredevice.edge_counter.EdgeCounter`.

        :param input_freq: Simulated input frequency for gate operations (signal)
        :param input_stdev: Simulated input frequency standard deviation for gate operations (signal)
        :param seed: Seed for the random number generator used for simulating input
        """
        assert isinstance(gateware_width, int), 'Gateware width must be of type int'
        assert isinstance(input_freq, float) and input_freq >= 0.0, 'Input frequency must be a positive float'
        assert isinstance(input_stdev, float) and input_stdev >= 0.0, 'Input stdev must be a non-negative float'

        # Call super
        super(EdgeCounter, self).__init__(dmgr, **kwargs)

        # Initialize rng
        self._rng = random.Random(seed)
        # Buffers to store counts
        self._count_buffer = collections.deque()
        # Single buffer to match set_config() calls
        self._prev_config = None

        # Register signals
        signal_manager = get_signal_manager()
        self._count = signal_manager.register(self, 'count', int, init='z')
        self._input_freq = signal_manager.register(self, 'input_freq', float, init=input_freq)
        self._input_stdev = signal_manager.register(self, 'input_stdev', float, init=input_stdev)

        # Store attributes and parameters (from ARTIQ code)
        if channel is not None:
            self.channel = channel
        self.counter_max = (1 << (gateware_width - 1)) - 1

    def core_reset(self) -> None:
        # Clear buffers
        self._count_buffer.clear()

    def _simulate_input_signal(self, duration, edge_type
                               ):  # type: (typing.Union[int, int32, int64], _EdgeType) -> None
        """Simulate input signal for a given duration."""

        # Obtain current input configuration
        input_freq = self._input_freq.pull()
        input_stdev = self._input_stdev.pull()
        assert isinstance(input_freq, float)
        assert isinstance(input_stdev, float)

        # Decide event frequency
        event_freq = max(self._rng.gauss(input_freq, input_stdev), 0.0)
        if edge_type is _EdgeType.BOTH:
            # Multiply by 2 in case we detect both edges
            event_freq *= 2

        # Calculate the number of events we expect to observe based on duration and frequency
        num_events = int(self.core.mu_to_seconds(duration) * event_freq)

        # Set the number of counts for the duration window (for graphical purposes)
        self._count.push(num_events)

        # Move the cursor
        delay_mu(duration)

        # Return to Z at the end of the window
        self._count.push('z')

        # Store number of events and the ending timestamp in count buffer
        self._count_buffer.append((now_mu(), num_events))

    @kernel
    def gate_rising_mu(self, duration_mu):
        self._simulate_input_signal(duration_mu, _EdgeType.RISING)
        return now_mu()

    @kernel
    def gate_falling_mu(self, duration_mu):
        self._simulate_input_signal(duration_mu, _EdgeType.FALLING)
        return now_mu()

    @kernel
    def gate_both_mu(self, duration_mu):
        self._simulate_input_signal(duration_mu, _EdgeType.BOTH)
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

    def _set_config(self, count_rising: TBool, count_falling: TBool, send_count_event: TBool, reset_to_zero: TBool):
        if self._prev_config is None:
            if (send_count_event, reset_to_zero) == (False, True):
                # Store this configuration to match it with the next call to this function
                self._prev_config = _Config(
                    edge_type=_EdgeType.from_bool(rising=count_rising, falling=count_falling),
                    timestamp=now_mu()
                )
            else:
                raise ValueError(f'Expected (send_count_event, reset_to_zero) == (False, True), '
                                 f'instead got the invalid combination ({send_count_event}, {reset_to_zero})')
        else:
            if (count_rising, count_falling, send_count_event, reset_to_zero) == (False, False, True, False):
                # Complete the gate operation
                duration = now_mu() - self._prev_config.timestamp
                at_mu(self._prev_config.timestamp)  # Rewind to start of gate operation
                self._simulate_input_signal(duration, self._prev_config.edge_type)
                # Clear previous configuration
                self._prev_config = None
            else:
                if count_rising or count_falling:
                    raise ValueError(f'Expected (count_rising, count_falling) == (False, False), '
                                     f'instead got the invalid combination ({count_rising}, {count_falling})')
                else:
                    raise ValueError(f'Expected (send_count_event, reset_to_zero) == (True, False), '
                                     f'instead got the invalid combination ({send_count_event}, {reset_to_zero})')

    @kernel
    def set_config(self, count_rising: TBool, count_falling: TBool, send_count_event: TBool, reset_to_zero: TBool):
        return self._set_config(count_rising, count_falling, send_count_event, reset_to_zero)

    def _fetch_count(self) -> TInt32:
        if len(self._count_buffer):
            # Get count from the buffer (drop the timestamp)
            _, count = self._count_buffer.popleft()

            if count >= self.counter_max:
                # Count overflow
                raise CounterOverflow(f'Input edge counter overflow for device {self.key}')

            # Return the result
            return count
        else:
            # No count available to return
            raise IndexError(f'Device "{self.key}" has no count to return')

    @kernel
    def fetch_count(self) -> TInt32:
        return self._fetch_count()

    def _fetch_timestamped_count(self) -> TTuple([TInt64, TInt32]):  # type: ignore[valid-type]
        if len(self._count_buffer):
            # Get count and timestamp from the buffer
            timestamp, count = self._count_buffer.popleft()

            if count >= self.counter_max:
                # Count overflow
                raise CounterOverflow(f'Input edge counter overflow for device {self.key}')

            # Return the result
            return timestamp, count
        else:
            # No count available to return
            return -1, 0

    # noinspection PyUnusedLocal
    @kernel
    def fetch_timestamped_count(self, timeout_mu=int64(-1)) -> TTuple([TInt64, TInt32]):  # type: ignore[valid-type]
        return self._fetch_timestamped_count()
