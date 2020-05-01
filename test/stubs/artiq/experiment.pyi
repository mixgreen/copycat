import typing

from .language import *


class NoDefault:
    ...


class _SimpleArgProcessor:

    def __init__(self, default: typing.Any = NoDefault):
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

    def __init__(self, choices: typing.List[str], default: typing.Union[str, typing.Type[NoDefault]] = NoDefault):
        ...


class NumberValue(_SimpleArgProcessor):
    # Types of a number value
    __N_T = typing.Union[int, float]

    def __init__(self, default: typing.Union[__N_T, typing.Type[NoDefault]] = NoDefault, unit: str = "",
                 scale: typing.Optional[__N_T] = None,
                 step: typing.Optional[__N_T] = None, min: typing.Optional[__N_T] = None,
                 max: typing.Optional[__N_T] = None, ndecimals: int = 2):
        ...

    def _is_int(self) -> bool:
        ...

    def default(self) -> __N_T:
        ...

    def process(self, x: str) -> __N_T:
        ...


class StringValue(_SimpleArgProcessor):
    ...


class HasEnvironment:
    # Possible data types for basic dataset values
    __BDV_T = typing.Union[bool, int, float, str, np.int32, np.int64]
    # Possible data types for dataset values
    __DV_T = typing.Union[__BDV_T, typing.List[__BDV_T], np.ndarray]
    # Possible data types for dataset default values
    __DDV_T = typing.Union[__DV_T, typing.Type[NoDefault]]

    def __init__(self, managers_or_parent: typing.Any, *args: typing.Any, **kwargs: typing.Any):
        ...

    def register_child(self, child: HasEnvironment) -> None:
        ...

    def call_child_method(self, method: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    def get_argument(self, key: str, processor: _SimpleArgProcessor, group: typing.Optional[str] = None,
                     tooltip: typing.Optional[str] = None) -> typing.Any:
        ...

    def setattr_argument(self, key: str, processor: _SimpleArgProcessor, group: typing.Optional[str] = None,
                         tooltip: typing.Optional[str] = None) -> None:
        ...

    def get_device_db(self) -> typing.Dict[str, typing.Any]:
        ...

    def get_device(self, key: str) -> typing.Any:
        ...

    def setattr_device(self, key: str) -> None:
        ...

    def set_dataset(self, key: str, value: __DV_T,
                    broadcast: bool = False, persist: bool = False, archive: bool = True) -> None:
        ...

    def mutate_dataset(self, key: str, index: typing.Any, value: __DV_T) -> None:
        ...

    def append_to_dataset(self, key: str, value: __DV_T) -> None:
        ...

    def get_dataset(self, key: str, default: __DDV_T = NoDefault, archive: bool = True) -> __DV_T:
        ...

    def setattr_dataset(self, key: str, default: __DDV_T = NoDefault, archive: bool = True) -> None:
        ...

    def set_default_scheduling(self, priority: typing.Optional[int] = None, pipeline_name: typing.Optional[str] = None,
                               flush: typing.Optional[bool] = None) -> None:
        ...


class Experiment:

    def prepare(self) -> None:
        ...

    def run(self) -> None:
        ...

    def analyze(self) -> None:
        ...


class EnvExperiment(Experiment, HasEnvironment):
    ...


def is_experiment(o: typing.Any) -> bool:
    ...
