import typing

__all__ = ['device_db_from_file', 'DeviceDB', 'DatasetDB']


def device_db_from_file(filename: str) -> typing.Dict[str, typing.Any]:
    ...


class DeviceDB:
    def __init__(self, backing_file: str):
        ...

    def scan(self) -> None:
        ...

    def get_device_db(self) -> typing.Dict[str, typing.Any]:
        ...

    def get(self, key: str, resolve_alias: bool = ...) -> typing.Any:
        ...


class DatasetDB:
    def __init__(self, persist_file: str, autosave_period: int = ...):
        ...

    def save(self) -> None:
        ...

    def get(self, key: str) -> typing.Any:
        ...

    def update(self, mod: typing.Any) -> None:
        ...

    def set(self, key: str, value: typing.Any, persist: typing.Optional[bool] = ...) -> None:
        ...

    def delete(self, key: str) -> None:
        ...
