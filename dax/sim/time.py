from __future__ import annotations  # Postponed evaluation of annotations

import abc
import typing
import numpy as np

__all__ = ['DaxTimeManager']

_MU_T = np.int64
"""The type of machine units (MU)."""


class _TimeContext(abc.ABC):
    """Abstract time context class."""

    _current_time: _MU_T
    _block_duration: _MU_T

    def __init__(self, current_time: _MU_T):
        self._current_time = current_time
        self._block_duration = _MU_T(0)

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
        new_context = _SequentialTimeContext(self.get_time_mu())
        self._stack.append(new_context)

    def enter_parallel(self) -> None:
        """Add a new parallel time context to the stack."""
        new_context = _ParallelTimeContext(self.get_time_mu())
        self._stack.append(new_context)

    def exit(self) -> None:
        """Exit the last time context."""
        old_context = self._stack.pop()
        self.take_time_mu(old_context.block_duration)

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
        dt = t - self.get_time_mu()
        self.take_time_mu(dt)
