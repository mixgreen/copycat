import typing
import collections.abc

_KT = typing.TypeVar('_KT')
_VT = typing.TypeVar('_VT')


class SortedDict(typing.Dict[_KT, _VT]):

    def peekitem(self, index: int = ...) -> typing.Tuple[_KT, _VT]:
        ...

    def keys(self) -> collections.abc.Sequence[_KT]:  # type: ignore[override]
        ...

    def values(self) -> collections.abc.Sequence[_VT]:  # type: ignore[override]
        ...

    def items(self) -> collections.abc.Sequence[typing.Tuple[_KT, _VT]]:  # type: ignore[override]
        ...

    def bisect_left(self, value: _KT) -> int:
        ...

    def bisect_right(self, value: _KT) -> int:
        ...

    def count(self, value: _KT) -> int:
        ...

    def index(self, value: _KT, start: typing.Optional[int] = ..., stop: typing.Optional[int] = ...):
        ...

    def irange(self, minimum: typing.Optional[_KT] = ..., maximum: typing.Optional[_KT] = ...,
               inclusive: typing.Tuple[bool, bool] = ..., reverse: bool = ...) -> typing.Iterator[_KT]:
        ...

    def islice(self, start: typing.Optional[int] = ..., stop: typing.Optional[int] = ...,
               reverse: bool = ...) -> typing.Iterator[_KT]:
        ...
