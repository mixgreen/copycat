import typing

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
