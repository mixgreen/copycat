import typing
import dataclasses
import numpy as np

__all__ = ['DaxTimeManager']

_MU_T = np.int64
"""The type of machine units (MU)."""


@dataclasses.dataclass
class _TimeContext:
    """Abstract time context class."""

    current_time: _MU_T
    block_duration: _MU_T = dataclasses.field(default=_MU_T(0), init=False)

    def take_time(self, duration: _MU_T) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclasses.dataclass
class _SequentialTimeContext(_TimeContext):
    """Sequential time context class."""

    def take_time(self, duration: _MU_T) -> None:
        self.current_time += duration
        self.block_duration += duration


@dataclasses.dataclass
class _ParallelTimeContext(_TimeContext):
    """Parallel time context class."""

    def take_time(self, duration: _MU_T) -> None:
        if duration > self.block_duration:
            self.block_duration = duration


class DaxTimeManager:
    """DAX time manager class."""

    _ref_period: float
    _stack: typing.List[_TimeContext]

    def __init__(self, ref_period: float):
        assert isinstance(ref_period, float), 'Reference period must be of type float'

        if ref_period <= 0.0:
            # The reference period must be larger than zero
            raise ValueError('The reference period must be larger than zero')

        # Store reference period
        self._ref_period = ref_period

        # Initialize time context stack
        self._stack = [_SequentialTimeContext(_MU_T(0))]

    """Helper functions"""

    def _seconds_to_mu(self, seconds: float) -> _MU_T:
        """Convert seconds to machine units.

        :param seconds: The time in seconds
        :return: The time converted to machine units
        """
        return _MU_T(seconds // self._ref_period)  # floor div, same as in ARTIQ Core

    """Functions that interface with the ARTIQ language core"""

    def enter_sequential(self) -> None:
        """Add a new sequential time context to the stack."""
        self._stack.append(_SequentialTimeContext(self.get_time_mu()))

    def enter_parallel(self) -> None:
        """Add a new parallel time context to the stack."""
        self._stack.append(_ParallelTimeContext(self.get_time_mu()))

    def exit(self) -> None:
        """Exit the last time context."""
        self.take_time_mu(self._stack.pop().block_duration)

    def take_time_mu(self, duration: _MU_T) -> None:
        """Take time from the current context.

        :param duration: The duration in machine units
        """
        # Take time from the current context
        self._stack[-1].take_time(duration)

    def take_time(self, duration: float) -> None:
        """Take time from the current context.

        The duration will be converted to machine units based on the reference period
        before it is used. This might result in some rounding error.

        :param duration: The duration in natural time (float)
        """
        # Divide duration by the reference period and convert to machine units
        self.take_time_mu(self._seconds_to_mu(duration))

    def get_time_mu(self) -> _MU_T:
        """Return the current time in machine units.

        :return: Current time in machine units
        """
        return self._stack[-1].current_time

    def set_time_mu(self, t: _MU_T) -> None:
        """Set the time to a specific point.

        :param t: The specific point in time (machine units) to set the current time to
        """
        # Take time to match the given time point
        self.take_time_mu(t - self.get_time_mu())
