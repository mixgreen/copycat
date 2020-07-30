import typing
import numpy as np

from artiq.language.core import kernel

__all__ = ['CounterOverflow', 'EdgeCounter']


class CounterOverflow(Exception):
    ...


class EdgeCounter:
    def __init__(self, dmgr: typing.Any, channel: int, gateware_width: int = ..., core_device: str = ...):
        ...

    @kernel
    def gate_rising(self, duration: float) -> np.int64:
        ...

    @kernel
    def gate_falling(self, duration: float) -> np.int64:
        ...

    @kernel
    def gate_both(self, duration: float) -> np.int64:
        ...

    @kernel
    def gate_rising_mu(self, duration_mu: np.int64) -> np.int64:
        ...

    @kernel
    def gate_falling_mu(self, duration_mu: np.int64) -> np.int64:
        ...

    @kernel
    def gate_both_mu(self, duration_mu: np.int64) -> np.int64:
        ...

    @kernel
    def set_config(self, count_rising: bool, count_falling: bool, send_count_event: bool, reset_to_zero: bool) -> None:
        ...

    @kernel
    def fetch_count(self) -> np.int32:
        ...

    @kernel
    def fetch_timestamped_count(self, timeout_mu: np.int64 = ...) -> typing.Tuple[np.int64, np.int32]:
        ...
