import abc
import typing
import numpy as np

from artiq.language.units import *

# The type of machine units (MU)
_MU_T: type = np.int64


class _TimeContext(abc.ABC):
    """Abstract time context class."""

    def __init__(self, current_time: _MU_T):
        self._current_time: _MU_T = current_time
        self._block_duration: _MU_T = _MU_T(0)

    @property
    def current_time(self) -> _MU_T:
        return self._current_time

    @property
    def block_duration(self) -> _MU_T:
        return self._block_duration

    @abc.abstractmethod
    def take_time(self, amount: _MU_T) -> None:
        pass


class _SequentialTimeContext(_TimeContext):
    """Sequential time context class."""

    def take_time(self, amount: _MU_T) -> None:
        self._current_time += amount
        self._block_duration += amount


class _ParallelTimeContext(_TimeContext):
    """Parallel time context class."""

    def take_time(self, amount: _MU_T) -> None:
        if amount > self._block_duration:
            self._block_duration = amount


class DaxTimeManager:
    def __init__(self, timescale: float):
        assert isinstance(timescale, float), 'Timescale must be of type float'

        if timescale <= 0.0:
            # The timescale must be larger than zero
            raise ValueError('The timescale must be larger than zero')

        # Store timescale, this should also be the leading timescale for the core device
        self._timescale: float = timescale

        # Initialize time context stack
        self._stack: typing.List[_TimeContext] = [_SequentialTimeContext(_MU_T(0))]

    """Functions that interface with the ARTIQ language core"""

    def enter_sequential(self) -> None:
        """Add a new sequential time context to the stack."""
        new_context = _SequentialTimeContext(self.get_time_mu())
        self._stack.append(new_context)

    def enter_parallel(self) -> None:
        """Add a new parallel time context to the stack."""
        new_context = _ParallelTimeContext(self.get_time_mu())
        self._stack.append(new_context)

    def exit(self) -> None:
        """Exit the last time context."""
        old_context = self._stack.pop()
        self.take_time(old_context.block_duration)

    def take_time_mu(self, duration: _MU_T) -> None:
        """Take time from the current context.

        :param duration: The duration in machine units
        """
        self._stack[-1].take_time(duration)

    def take_time(self, duration: float) -> None:
        """Take time from the current context.

        The duration will be converted to machine units based on the timescale
        before it is used. This might result in some rounding error.

        :param duration: The duration in natural time (float)
        """

        # Divide duration by the timescale and convert to machine units
        self.take_time_mu(_MU_T(duration / self._timescale))

    def get_time_mu(self) -> _MU_T:
        """Return the current time in machine units.

        :returns: Current time in machine units
        """
        return self._stack[-1].current_time

    def set_time_mu(self, t: _MU_T) -> None:
        """Set the time to a specific point.

        :param t: The specific point in time (machine units) to set the current time to
        """

        if t < self.get_time_mu():
            # Going back in time is not allowed by the VCD writer
            raise ValueError("Attempted to go back in time")

        # Take time to match the given time point
        dt = t - self.get_time_mu()
        self.take_time_mu(dt)
