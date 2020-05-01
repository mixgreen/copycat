import typing
import numpy as np

__F_T = typing.TypeVar('__F_T', bound=typing.Callable[..., typing.Any])


def host_only(function: __F_T) -> __F_T: ...


@typing.overload
def kernel(arg: __F_T) -> __F_T: ...


@typing.overload
def kernel(flags: typing.Set[str]) -> typing.Callable[[__F_T], __F_T]: ...


@typing.overload
def rpc(arg: __F_T) -> __F_T: ...


@typing.overload
def rpc(flags: typing.Set[str]) -> typing.Callable[[__F_T], __F_T]: ...


@typing.overload
def portable(arg: __F_T) -> __F_T: ...


@typing.overload
def portable(flags: typing.Set[str]) -> typing.Callable[[__F_T], __F_T]: ...


@typing.overload
def syscall(arg: __F_T) -> __F_T: ...


@typing.overload
def syscall(flags: typing.Set[str]) -> typing.Callable[[__F_T], __F_T]: ...


def set_time_manager(time_manager: typing.Any) -> None:
    ...


# ARTIQ context managers
sequential: typing.ContextManager[None] = ...
parallel: typing.ContextManager[None] = ...
interleave = parallel


def delay_mu(duration: np.int64) -> None:
    ...


def now_mu() -> np.int64:
    ...


def at_mu(time: np.int64) -> None:
    ...


def delay(duration: float) -> None:
    ...


class TerminationRequested(Exception):
    ...
