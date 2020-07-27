"""
This stub file is compatible with PyVCD 0.1.7-0.2.3.
Only features that are supported by both are included.
This stub is required to support typing for PyVCD <= 0.1.7.
"""

from types import TracebackType
from typing import (
    IO,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)


class VCDPhaseError(Exception):
    ...


ScopeInput = Union[str, Sequence[str]]
TimeValue = Union[int, float]
Timescale = Union[Tuple[int, str], str]
CompoundSize = Sequence[int]
VariableSize = Union[int, CompoundSize]

EventValue = Union[bool, int]
RealValue = Union[float, int]
ScalarValue = Union[int, bool, str, None]
StringValue = Union[str, None]
CompoundValue = Sequence[ScalarValue]
VarValue = Union[EventValue, RealValue, ScalarValue, StringValue, CompoundValue]


class VCDWriter:

    def __init__(
            self,
            file: IO[str],
            timescale: Timescale = ...,
            date: Optional[str] = ...,
            comment: str = ...,
            version: str = ...,
            default_scope_type: str = ...,
            scope_sep: str = ...,
            check_values: bool = ...,
            init_timestamp: TimeValue = ...,
    ) -> None:
        ...

    def set_scope_type(
            self, scope: ScopeInput, scope_type: str
    ) -> None:
        ...

    def register_var(
            self,
            scope: ScopeInput,
            name: str,
            var_type: str,
            size: Optional[VariableSize] = ...,
            init: VarValue = ...,
    ) -> 'Variable':
        ...

    def dump_off(self, timestamp: TimeValue) -> None:
        ...

    def dump_on(self, timestamp: TimeValue) -> None:
        ...

    def change(self, var: 'Variable', timestamp: TimeValue, value: VarValue) -> None:
        ...

    def __enter__(self) -> 'VCDWriter':
        ...

    def __exit__(
            self,
            exc_type: Optional[Type[Exception]],
            exc_value: Optional[Exception],
            traceback: Optional[TracebackType],
    ) -> None:
        ...

    def close(self, timestamp: Optional[TimeValue] = ...) -> None:
        ...

    def flush(self, timestamp: Optional[TimeValue] = ...) -> None:
        ...


class Variable:
    ...


class ScalarVariable(Variable):
    ...


class EventVariable(Variable):
    ...


class StringVariable(Variable):
    ...


class RealVariable(Variable):
    ...


class VectorVariable(Variable):
    ...


class CompoundVectorVariable(Variable):
    ...
