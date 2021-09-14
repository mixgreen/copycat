from __future__ import annotations

import timeit
import typing


class Timer:
    """A timer class that can be used to calculate execution time."""

    _total_t: float
    _start_t: float

    def __init__(self) -> None:
        self._total_t = 0.0
        self._start_t = 0.0

    def __enter__(self) -> Timer:
        # Record starting time
        self._start_t = timeit.default_timer()
        return self

    def __exit__(self, exc_type: typing.Any, exc_val: typing.Any, exc_tb: typing.Any) -> typing.Any:
        # Record stopping time
        stop_t = timeit.default_timer()
        # Add execution time to the total time
        self._total_t += stop_t - self._start_t

    def reset(self) -> None:
        """Reset this timer."""
        self._total_t = 0.0

    @property
    def total_time(self) -> float:
        """Get the total time registered by this timer."""
        return self._total_t

    def u_print(self, *, precision: int = 5, **kwargs: typing.Any) -> None:
        """Print the total time registered by this timer.

        This print function is specifically intended to be used with Python unit testing with elevated verbosity.
        By default, no newline is inserted and output is flushed.
        """
        kwargs.setdefault('end', '')
        kwargs.setdefault('flush', True)
        print(f'({self._total_t:.{precision:d}f} s) ', **kwargs)

    def __str__(self) -> str:
        return str(self._total_t)
