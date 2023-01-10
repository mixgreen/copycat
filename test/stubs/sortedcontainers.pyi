import typing
import collections.abc

_KT = typing.TypeVar('_KT')
_VT = typing.TypeVar('_VT')


# Note: typing in this module is a bit different from the actual implementation to avoid some mixin related mypy bugs


# class SortedKeysView(typing.KeysView[_KT], typing.Sequence[_KT]):
class SortedKeysView(typing.KeysView[_KT]):

    def __getitem__(self, index: int) -> _KT:
        ...


class SortedItemsView(typing.ItemsView[_KT, _VT], typing.Sequence[typing.Tuple[_KT, _VT]]):

    def __getitem__(self, index: int) -> typing.Tuple[_KT, _VT]:  # type: ignore[override]
        ...


class SortedValuesView(typing.ValuesView[_VT], typing.Sequence[_VT]):

    def __getitem__(self, index: int) -> _VT:  # type: ignore[override]
        ...


# class SortedDict(typing.Dict[_KT, _VT]):
class SortedDict(typing.MutableMapping[_KT, _VT]):

    def __setitem__(self, k: _KT, v: _VT) -> None:
        ...

    def __delitem__(self, v: _KT) -> None:
        ...

    def __getitem__(self, k: _KT) -> _VT:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> typing.Iterator[_KT]:
        ...

    def peekitem(self, index: int = ...) -> typing.Tuple[_KT, _VT]:
        ...

    def keys(self) -> SortedKeysView[_KT]:
        ...

    def values(self) -> SortedValuesView[_VT]:
        ...

    def items(self) -> SortedItemsView[_KT, _VT]:
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
