import typing
import types

import artiq.language.environment

__all__ = ["parse_arguments", "elide", "short_format", "file_import",
           "get_experiment",
           "exc_to_warning", "asyncio_wait_or_cancel",
           "get_windows_drives", "get_user_config_dir"]


def parse_arguments(arguments: typing.Sequence[str]) -> typing.Dict[str, typing.Any]:
    ...


def elide(s: str, maxlen: int) -> str:
    ...


def short_format(v: typing.Any) -> str:
    ...


def file_import(filename: str, prefix: str = ...) -> types.ModuleType:
    ...


def get_experiment(module: types.ModuleType, class_name: typing.Optional[str] = ...) \
        -> typing.Type[artiq.language.Experiment]:
    ...


async def exc_to_warning(coro: typing.Coroutine) -> None:
    ...


async def asyncio_wait_or_cancel(fs: typing.Sequence[typing.Any], **kwargs: typing.Any) -> typing.List[typing.Any]:
    ...


def get_windows_drives() -> typing.List[str]:
    ...


def get_user_config_dir() -> str:
    ...
