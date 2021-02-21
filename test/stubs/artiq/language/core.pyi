import typing
import numpy as np

__all__ = ["kernel", "portable", "rpc", "syscall", "host_only",
           "kernel_from_string", "set_time_manager", "set_watchdog_factory",
           "TerminationRequested",
           "sequential", "parallel", "interleave",
           "delay_mu", "now_mu", "at_mu", "delay",
           "watchdog"]

__FN_T = typing.TypeVar('__FN_T', bound=typing.Callable[..., typing.Any])  # Type var for functions


@typing.overload
def kernel(arg: __FN_T) -> __FN_T: ...


@typing.overload
def kernel(arg: str) -> typing.Callable[[__FN_T], __FN_T]: ...


@typing.overload
def kernel(flags: typing.Set[str]) -> typing.Callable[[__FN_T], __FN_T]: ...


@typing.overload
def kernel(arg: str, flags: typing.Set[str]) -> typing.Callable[[__FN_T], __FN_T]: ...


@typing.overload
def rpc(arg: __FN_T) -> __FN_T: ...


@typing.overload
def rpc(flags: typing.Set[str]) -> typing.Callable[[__FN_T], __FN_T]: ...


@typing.overload
def portable(arg: __FN_T) -> __FN_T: ...


@typing.overload
def portable(flags: typing.Set[str]) -> typing.Callable[[__FN_T], __FN_T]: ...


@typing.overload
def syscall(arg: __FN_T) -> __FN_T: ...


@typing.overload
def syscall(flags: typing.Set[str]) -> typing.Callable[[__FN_T], __FN_T]: ...


def host_only(function: __FN_T) -> __FN_T: ...


def kernel_from_string(parameters: typing.Sequence[typing.Union[str, typing.Tuple[str, str]]],
                       body_code: str,
                       decorator: typing.Callable[..., typing.Any] = ...) -> typing.Callable[..., typing.Any]:
    ...


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


def set_watchdog_factory(f: typing.Any) -> None:
    ...


def watchdog(timeout: float) -> typing.Any:
    ...


class TerminationRequested(Exception):
    ...
