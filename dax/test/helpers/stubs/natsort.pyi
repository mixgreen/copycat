import typing

__E_T = typing.TypeVar('__E_T')


def natsorted(seq: typing.Iterable[__E_T], key: typing.Optional[typing.Callable[[__E_T], typing.Any]] = None,
              reverse: bool = False, alg: typing.Optional[typing.Any] = None) -> typing.List[__E_T]:
    ...
