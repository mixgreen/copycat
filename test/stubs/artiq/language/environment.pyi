import typing
import numpy as np
import abc

from .scan import Scannable

__all__ = ["NoDefault",
           "PYONValue", "BooleanValue", "EnumerationValue",
           "NumberValue", "StringValue",
           "HasEnvironment", "Experiment", "EnvExperiment"]


class NoDefault:
    ...


class _SimpleArgProcessor:

    def __init__(self, default: typing.Any = ...):
        ...

    def default(self) -> typing.Any:
        ...

    def process(self, x: typing.Any) -> typing.Any:
        ...

    def describe(self) -> typing.Dict[str, typing.Any]:
        ...


class PYONValue(_SimpleArgProcessor):

    def process(self, x: typing.Any) -> typing.Any:
        ...


class BooleanValue(_SimpleArgProcessor):
    ...


class EnumerationValue(_SimpleArgProcessor):

    # noinspection PyMissingConstructor
    def __init__(self, choices: typing.Sequence[str], default: typing.Union[str, typing.Type[NoDefault]] = ...):
        ...


class NumberValue(_SimpleArgProcessor):
    # Types of a number value
    __N_T = typing.Union[int, float]

    # noinspection PyShadowingBuiltins,PyMissingConstructor
    def __init__(self, default: typing.Union[__N_T, typing.Type[NoDefault]] = ..., unit: str = ...,
                 scale: typing.Optional[__N_T] = ...,
                 step: typing.Optional[__N_T] = ..., min: typing.Optional[__N_T] = ...,
                 max: typing.Optional[__N_T] = ..., ndecimals: int = ...):
        ...

    def _is_int(self) -> bool:
        ...

    def default(self) -> __N_T:
        ...

    def process(self, x: str) -> __N_T:
        ...


class StringValue(_SimpleArgProcessor):
    ...


class TraceArgumentManager:
    def __init__(self) -> None:
        ...

    def get(self, key: str, processor: _SimpleArgProcessor, group: str, tooltip: str) -> None:
        ...


class ProcessArgumentManager:
    def __init__(self, unprocessed_arguments: typing.Any):
        ...

    def get(self, key: str, processor: _SimpleArgProcessor, group: str, tooltip: str) -> typing.Any:
        ...


class HasEnvironment:
    # Possible data types for basic dataset values
    __BDV_T = typing.Union[bool, int, float, str, np.int32, np.int64]
    # Possible data types for dataset values (recursive types are not supported, added a few manual levels)
    __DV_T = typing.Union[__BDV_T, typing.Sequence[__BDV_T], typing.Sequence[typing.Sequence[__BDV_T]],
                          typing.Sequence[typing.Sequence[typing.Sequence[typing.Any]]]]
    # Possible data types for dataset default values
    __DDV_T = typing.Union[__DV_T, typing.Type[NoDefault]]

    def __init__(self, managers_or_parent: typing.Any, *args: typing.Any, **kwargs: typing.Any):
        self.children: typing.List['HasEnvironment'] = ...

    def register_child(self, child: HasEnvironment) -> None:
        ...

    def call_child_method(self, method: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    def get_argument(self, key: str, processor: typing.Union[_SimpleArgProcessor, Scannable],
                     group: typing.Optional[str] = ..., tooltip: typing.Optional[str] = ...) -> typing.Any:
        ...

    def setattr_argument(self, key: str, processor: typing.Union[_SimpleArgProcessor, Scannable],
                         group: typing.Optional[str] = ..., tooltip: typing.Optional[str] = ...) -> None:
        ...

    def get_device_db(self) -> typing.Dict[str, typing.Any]:
        ...

    def get_device(self, key: str) -> typing.Any:
        ...

    def setattr_device(self, key: str) -> None:
        ...

    def set_dataset(self, key: str, value: __DV_T,
                    broadcast: bool = ..., persist: bool = ..., archive: bool = ...) -> None:
        ...

    def mutate_dataset(self, key: str, index: typing.Any, value: __DV_T) -> None:
        ...

    def append_to_dataset(self, key: str, value: __DV_T) -> None:
        ...

    def get_dataset(self, key: str, default: __DDV_T = ..., archive: bool = ...) -> __DV_T:
        ...

    def setattr_dataset(self, key: str, default: __DDV_T = ..., archive: bool = ...) -> None:
        ...

    def set_default_scheduling(self, priority: typing.Optional[int] = ..., pipeline_name: typing.Optional[str] = ...,
                               flush: typing.Optional[bool] = ...) -> None:
        ...


class Experiment(abc.ABC):

    def prepare(self) -> None:
        ...

    @abc.abstractmethod
    def run(self) -> None:
        ...

    def analyze(self) -> None:
        ...


class EnvExperiment(Experiment, HasEnvironment, abc.ABC):
    def prepare(self) -> None:
        ...

    @abc.abstractmethod
    def run(self) -> None:
        ...


def is_experiment(o: typing.Any) -> bool:
    ...
