import typing
import logging

__all__ = ['logger', 'DummyDevice', 'DeviceError', 'DeviceManager', 'DatasetManager']

logger: logging.Logger = ...


class DummyDevice:
    ...


class DeviceError(Exception):
    ...


class DeviceManager:
    def __init__(self, ddb: typing.Any, virtual_devices: typing.Dict[str, typing.Any] = ...):
        ...

    def close_devices(self) -> None:
        ...


class DatasetManager:
    def __init__(self, ddb: typing.Any):
        self.local: typing.Dict[str, typing.Any] = ...
        self.archive: typing.Dict[str, typing.Any] = ...
        self.ddb: typing.Any = ...

    def set(self, key: str, value: typing.Any, broadcast: bool = ..., persist: bool = ..., archive: bool = ...) -> None:
        ...

    def mutate(self, key: str, index: typing.Any, value: typing.Any) -> None:
        ...

    def append_to(self, key: str, value: typing.Any) -> None:
        ...

    def get(self, key: str, archive: bool = ...) -> typing.Any:
        ...

    def write_hdf5(self, f: typing.Any) -> None:
        ...
