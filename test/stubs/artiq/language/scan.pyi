import typing

from .environment import NoDefault

__all__ = ["ScanObject",
           "NoScan", "RangeScan", "CenterScan", "ExplicitScan",
           "Scannable", "MultiScanManager"]


class ScanObject:
    ...


# Type variable for generics in scan classes
__V_T = typing.TypeVar('__V_T')


class NoScan(ScanObject, typing.Generic[__V_T]):
    def __init__(self, value: __V_T, repetitions: int = 1):
        ...

    def __iter__(self) -> typing.Iterator[__V_T]:
        ...

    def __len__(self) -> int:
        ...

    def describe(self) -> typing.Dict[str, typing.Any]:
        ...


class RangeScan(ScanObject):
    def __init__(self, start: float, stop: float, npoints: int,
                 randomize: bool = False, seed: typing.Any = None):
        ...

    def __iter__(self) -> typing.Iterator[float]:
        ...

    def __len__(self) -> int:
        ...

    def describe(self) -> typing.Dict[str, typing.Any]:
        ...


class CenterScan(ScanObject):
    def __init__(self, center: float, span: float, step: float,
                 randomize: bool = False, seed: typing.Any = None):
        ...

    def __iter__(self) -> typing.Iterator[float]:
        ...

    def __len__(self) -> int:
        ...

    def describe(self) -> typing.Dict[str, typing.Any]:
        ...


class ExplicitScan(ScanObject, typing.Generic[__V_T]):
    def __init__(self, sequence: typing.Sequence[__V_T]):
        ...

    def __iter__(self) -> typing.Iterator[__V_T]:
        ...

    def __len__(self) -> int:
        ...

    def describe(self) -> typing.Dict[str, typing.Any]:
        ...


class Scannable:
    def __init__(self, default=NoDefault, unit="", scale=None,
                 global_step=None, global_min=None, global_max=None,
                 ndecimals=2):
        ...

    def default(self) -> None:
        ...

    def process(self, x):
        ...

    def describe(self) -> typing.Dict[str, typing.Any]:
        ...


class MultiScanManager:
    def __init__(self, *args: typing.Tuple[str, typing.Any]):
        ...

    def __iter__(self) -> typing.Iterator[typing.Any]:
        ...
