import typing
import collections

__all__ = ['int32', 'int64', 'float', 'integer', 'ndarray',
           'array', 'zeroes', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace']

# int32 and int64 have the same type properties as a Python int
int32 = int
int64 = int


# noinspection PyPep8Naming
class integer(int):
    ...


# Type variables used for array creation functions
__N_T = typing.TypeVar('__N_T', int, float, int32, int64)
__SHAPE_T = typing.Union[int, typing.Tuple[int]]


# noinspection PyPep8Naming
class ndarray(collections.abc.Sequence, typing.Generic[__N_T]):

    # noinspection PyMissingConstructor
    def __init__(self, shape: typing.Tuple[int], dtype: type = float, buffer: typing.Any = None,
                 offset: int = 0,
                 strides: typing.Optional[typing.Tuple[int]] = None, order: typing.Optional[str] = None):
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
    def flat(self) -> typing.Iterator[__N_T]:
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

    @typing.overload
    def __getitem__(self, i: int) -> __N_T:
        ...

    @typing.overload
    def __getitem__(self, s: slice) -> typing.Sequence[__N_T]:
        ...

    def __len__(self) -> int:
        ...

    def __add__(self, other: __N_T) -> ndarray:
        ...

    def __iadd__(self, other: __N_T) -> ndarray:
        ...

    def __sub__(self, other: __N_T) -> ndarray:
        ...

    def __isub__(self, other: __N_T) -> ndarray:
        ...

    def __mul__(self, other: __N_T) -> ndarray:
        ...

    def __imul__(self, other: __N_T) -> ndarray:
        ...

    def __truediv__(self, other: __N_T) -> ndarray:
        ...

    def __itruediv__(self, other: __N_T) -> ndarray:
        ...


def array(object: typing.Sequence[typing.Any], dtype: type = None, copy: bool = True,
          order: str = 'K', subok: bool = False, ndmin: int = 0) -> ndarray:
    ...


def zeroes(shape: __SHAPE_T, dtype: type = float, order: str = 'C') -> ndarray:
    ...


def ones(shape: __SHAPE_T, dtype: type = None, order: str = 'C') -> ndarray:
    ...


def empty(shape: __SHAPE_T, dtype: type = float, order: str = 'C') -> ndarray:
    ...


def full(shape: __SHAPE_T, fill_value: __N_T, dtype: type = None, order: str = 'C') -> ndarray:
    ...


@typing.overload
def arange(stop: __N_T, dtype: type = None) -> ndarray:
    ...


@typing.overload
def arange(start: __N_T, stop: __N_T, step: typing.Optional[__N_T] = None, dtype: type = None) -> ndarray:
    ...


def linspace(start: __N_T, stop: __N_T, num: int = 50, endpoint: bool = True, retstep: bool = False,
             dtype: type = None, axis: int = 0) -> ndarray:
    ...


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0) -> ndarray:
    ...
