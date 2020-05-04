from .environment import NoDefault

__all__ = ["ScanObject",
           "NoScan", "RangeScan", "CenterScan", "ExplicitScan",
           "Scannable", "MultiScanManager"]


class ScanObject:
    ...


class NoScan(ScanObject):
    def __init__(self, value, repetitions=1):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...

    def describe(self):
        ...


class RangeScan(ScanObject):
    def __init__(self, start, stop, npoints, randomize=False, seed=None):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...

    def describe(self):
        ...


class CenterScan(ScanObject):
    def __init__(self, center, span, step, randomize=False, seed=None):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...

    def describe(self):
        ...


class ExplicitScan(ScanObject):
    def __init__(self, sequence):
        ...

    def __iter__(self):
        ...

    def __len__(self):
        ...

    def describe(self):
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

    def describe(self):
        ...


class MultiScanManager:
    def __init__(self, *args):
        ...

    def __iter__(self):
        ...
