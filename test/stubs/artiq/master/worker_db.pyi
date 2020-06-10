import typing
import logging

__all__ = ['logger', 'DummyDevice', 'DeviceError', 'DeviceManager', 'DatasetManager']

logger: logging.Logger = ...


class DummyDevice:
    ...


class DeviceError(Exception):
    ...


class DeviceManager:
    # noinspection PyDefaultArgument
    def __init__(self, ddb: typing.Any, virtual_devices: typing.Dict[str, typing.Any] = dict()):
        ...

    def get_device_db(self):
        ...

    def get_desc(self, name):
        ...

    def get(self, name):
        ...

    def close_devices(self):
        ...


class DatasetManager:
    def __init__(self, ddb: typing.Any):
        ...

    def set(self, key, value, broadcast=False, persist=False, archive=True):
        ...

    def _get_mutation_target(self, key):
        ...

    def mutate(self, key, index, value):
        ...

    def append_to(self, key, value):
        ...

    def get(self, key, archive=False):
        ...

    def write_hdf5(self, f):
        ...
