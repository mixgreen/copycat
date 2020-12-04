from __future__ import annotations  # Postponed evaluation of annotations

import typing
import numpy as np

from artiq.language.core import portable, kernel

__all__ = ['CompileError', 'Core']


class CompileError(Exception):
    def __init__(self, diagnostic: typing.Any):
        self.diagnostic: typing.Any = ...

    def __str__(self) -> str:
        ...


class Core:
    kernel_invariants: typing.Set[str] = ...

    def __init__(self, dmgr: typing.Any, host: str, ref_period: float, ref_multiplier: int = ...,
                 target: str = ...):
        self.ref_period: float = ...
        self.ref_multiplier: int = ...
        self.target_cls: type = ...
        self.coarse_ref_period: float = ...
        self.comm: typing.Any = ...

        self.first_run: bool = ...
        self.dmgr: typing.Any = ...
        self.core: Core = ...

    def close(self) -> None:
        ...

    def compile(self, function: typing.Any, args: typing.Sequence[typing.Any], kwargs: typing.Dict[str, typing.Any],
                set_result: typing.Any = ..., attribute_writeback: bool = ..., print_as_rpc: bool = ...) -> typing.Any:
        ...

    def run(self, function: typing.Any,
            args: typing.Sequence[typing.Any], kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        ...

    @portable
    def seconds_to_mu(self, seconds: float) -> np.int64:
        ...

    @portable
    def mu_to_seconds(self, mu: np.int64) -> float:
        ...

    @kernel
    def get_rtio_counter_mu(self) -> np.int64:
        ...

    @kernel
    def wait_until_mu(self, cursor_mu: np.int64) -> None:
        ...

    @kernel
    def get_rtio_destination_status(self, destination: int) -> bool:
        ...

    @kernel
    def reset(self) -> None:
        ...

    @kernel
    def break_realtime(self) -> None:
        ...
