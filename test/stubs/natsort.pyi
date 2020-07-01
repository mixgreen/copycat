import typing

__all__ = ['natsorted']

__E_T = typing.TypeVar('__E_T')


def natsorted(seq: typing.Iterable[__E_T], key: typing.Optional[typing.Callable[[__E_T], typing.Any]] = ...,
              reverse: bool = ..., alg: typing.Optional[typing.Any] = ...) -> typing.List[__E_T]:
    ...
