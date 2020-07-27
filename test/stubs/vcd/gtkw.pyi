"""
This stub file is compatible with PyVCD 0.1.7-0.2.3.
Only features that are supported by both are included.
This stub is required to support typing for PyVCD <= 0.1.7.
"""

import datetime
import time
from contextlib import contextmanager
from typing import IO, Any, Dict, Generator, List, Optional, Sequence, Tuple, Union


class GTKWSave:

    def __init__(self, savefile: IO[str]) -> None:
        ...

    def comment(self, *comments: Sequence[str]) -> None:
        ...

    def dumpfile(self, dump_path: str, abspath: bool = ...) -> None:
        ...

    def dumpfile_mtime(
            self,
            mtime: Optional[Union[float, time.struct_time, datetime.datetime]] = ...,
            dump_path: Optional[str] = ...,
    ) -> None:
        ...

    def dumpfile_size(
            self, size: Optional[int] = ..., dump_path: Optional[str] = ...
    ) -> None:
        ...

    def savefile(self, save_path: Optional[str] = ..., abspath: bool = ...) -> None:
        ...

    def timestart(self, timestamp: int = ...) -> None:
        ...

    def zoom_markers(
            self, zoom: float = ..., marker: int = ..., **kwargs: Dict[str, int]
    ) -> None:
        ...

    def size(self, width: int, height: int) -> None:
        ...

    def pos(self, x: int = ..., y: int = ...) -> None:
        ...

    def treeopen(self, tree: str) -> None:
        ...

    def signals_width(self, width: int) -> None:
        ...

    def sst_expanded(self, is_expanded: bool) -> None:
        ...

    def pattern_trace(self, is_enabled: bool) -> None:
        ...

    @contextmanager
    def group(
            self, name: str, closed: bool = ..., highlight: bool = ...
    ) -> Generator[None, None, None]:
        ...

    def begin_group(
            self, name: str, closed: bool = ..., highlight: bool = ...
    ) -> None:
        ...

    def end_group(
            self, name: str, closed: bool = ..., highlight: bool = ...
    ) -> None:
        ...

    def blank(
            self, label: str = ..., analog_extend: bool = ..., highlight: bool = ...
    ) -> None:
        ...

    def trace(
            self,
            name: str,
            alias: Optional[str] = ...,
            color: Optional[Union[str, int]] = ...,
            datafmt: str = ...,
            highlight: bool = ...,
            rjustify: bool = ...,
            extraflags: Optional[Sequence[str]] = ...,
            translate_filter_file: Optional[str] = ...,
            translate_filter_proc: Optional[str] = ...,
    ) -> None:
        ...

    @contextmanager
    def trace_bits(
            self,
            name: str,
            alias: Optional[str] = ...,
            color: Optional[Union[str, int]] = ...,
            datafmt: str = ...,
            highlight: bool = ...,
            rjustify: bool = ...,
            extraflags: Optional[Sequence[str]] = ...,
            translate_filter_file: Optional[str] = ...,
            translate_filter_proc: Optional[str] = ...,
    ) -> Generator[None, None, None]:
        ...

    def trace_bit(
            self,
            index: int,
            name: str,
            alias: Optional[str] = ...,
            color: Optional[Union[str, int]] = ...,
    ) -> None:
        ...


TranslationType = Union[
    Tuple[Union[int, str], str], Tuple[Union[int, str], str, Union[str, int]]
]


def make_translation_filter(
        translations: Sequence[Tuple[Any, ...]],
        datafmt: str = ...,
        size: Optional[int] = ...,
) -> str:
    ...


def decode_flags(flags: Union[str, int]) -> List[str]:
    ...


def spawn_gtkwave_interactive(
        dump_path: str, save_path: str, quiet: bool = ...
) -> None:
    ...
