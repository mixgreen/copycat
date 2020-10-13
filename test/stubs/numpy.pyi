import typing
import collections.abc

__all__ = ['generic', 'character', 'number', 'integer', 'int32', 'int64', 'floating', 'bool_', 'ndarray',
           'array', 'zeroes', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace',
           'issubdtype', 'ndenumerate']


# noinspection PyPep8Naming
class generic(object):
    ...


# noinspection PyPep8Naming
class character(generic):
    ...


# noinspection PyPep8Naming
class number(generic):
    ...


# noinspection PyPep8Naming
class integer(int, number):
    ...


# noinspection PyPep8Naming
class inexact(number):
    ...


# int32 and int64 have the same type properties as a Python int
int32 = int
int64 = int

# floating has the same type properties as a Python float
floating = float

# bool as the same type properties as a Python bool
bool_ = bool

# Type variables used for array creation functions
__N_T = typing.Union[int, float, int32, int64]
__A_T = typing.TypeVar('__A_T', int, float, int32, int64)
__SHAPE_T = typing.Union[int, typing.Tuple[int]]


# noinspection PyPep8Naming
class ndarray(collections.abc.Sequence, typing.Generic[__A_T]):

    # noinspection PyMissingConstructor
    def __init__(self, shape: typing.Tuple[int], dtype: type = ..., buffer: typing.Any = ..., offset: int = ...,
                 strides: typing.Optional[typing.Tuple[int]] = ..., order: typing.Optional[str] = ...):
        ...

    @property
    def T(self) -> ndarray:
        ...

    @property
    def data(self) -> typing.Any:
        ...

    @property
    def dtype(self) -> type:
        ...

    @property
    def flags(self) -> dict:
        ...

    @property
    def flat(self) -> typing.Iterator[__A_T]:
        ...

    @property
    def imag(self) -> ndarray:
        ...

    @property
    def real(self) -> ndarray:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def itemsize(self) -> int:
        ...

    @property
    def nbytes(self) -> int:
        ...

    @property
    def ndim(self) -> int:
        ...

    @property
    def shape(self) -> typing.Tuple[int]:
        ...

    @property
    def strides(self) -> typing.Tuple[int]:
        ...

    def sort(self, axis: typing.Optional[int] = ..., kind: typing.Optional[str] = ...,
             order: typing.Union[None, str, typing.List[str]] = ...):
        ...

    def argsort(self, axis: typing.Optional[int] = ..., kind: typing.Optional[str] = ...,
                order: typing.Union[None, str, typing.Sequence[str]] = ...):
        ...

    def mean(self, axis: typing.Optional[__SHAPE_T] = ..., dtype: typing.Optional[type] = ...,
             out: typing.Optional['ndarray'] = ..., keepdims: bool = ...):
        ...

    @typing.overload
    def __getitem__(self, i: int) -> __A_T:
        ...

    @typing.overload
    def __getitem__(self, s: slice) -> ndarray:
        ...

    def __len__(self) -> int:
        ...

    def __add__(self, other: __A_T) -> ndarray:
        ...

    def __iadd__(self, other: __A_T) -> ndarray:
        ...

    def __sub__(self, other: __A_T) -> ndarray:
        ...

    def __isub__(self, other: __A_T) -> ndarray:
        ...

    def __mul__(self, other: __A_T) -> ndarray:
        ...

    def __imul__(self, other: __A_T) -> ndarray:
        ...

    def __truediv__(self, other: __A_T) -> ndarray:
        ...

    def __itruediv__(self, other: __A_T) -> ndarray:
        ...


def array(object: typing.Sequence[typing.Any], dtype: typing.Optional[type] = ..., copy: bool = ...,
          order: str = ..., subok: bool = ..., ndmin: int = ...) -> ndarray:
    ...


def zeroes(shape: __SHAPE_T, dtype: type = ..., order: str = ...) -> ndarray:
    ...


def ones(shape: __SHAPE_T, dtype: typing.Optional[type] = ..., order: str = ...) -> ndarray:
    ...


def empty(shape: __SHAPE_T, dtype: type = ..., order: str = ...) -> ndarray:
    ...


def full(shape: __SHAPE_T, fill_value: __N_T, dtype: typing.Optional[type] = ..., order: str = ...) -> ndarray:
    ...


@typing.overload
def arange(stop: __N_T, dtype: typing.Optional[type] = ...) -> ndarray:
    ...


@typing.overload
def arange(start: __N_T, stop: __N_T, step: typing.Optional[__N_T] = ...,
           dtype: typing.Optional[type] = ...) -> ndarray:
    ...


def linspace(start: __N_T, stop: __N_T, num: int = ..., endpoint: bool = ..., retstep: bool = ...,
             dtype: typing.Optional[type] = ..., axis: int = ...) -> ndarray:
    ...


def logspace(start: __N_T, stop: __N_T, num: int = ..., endpoint: bool = ..., base: __N_T = ...,
             dtype: typing.Optional[type] = ..., axis: int = ...) -> ndarray:
    ...


def asarray(a: typing.Sequence[typing.Any], dtype: typing.Optional[type] = ...,
            order: typing.Optional[str] = ...) -> ndarray:
    ...


def column_stack(tup: typing.Sequence[ndarray]) -> ndarray:
    ...


def concatenate(*arrays: typing.Sequence[typing.Any], axis: int = ..., out: typing.Optional[ndarray] = ...) -> ndarray:
    ...


def issubdtype(arg1: typing.Union[type, str], arg2: typing.Union[type, str]) -> bool:
    ...


def ndenumerate(arr: ndarray[__A_T]) -> typing.Iterator[typing.Tuple[typing.Tuple[int, ...], __A_T]]:
    ...
