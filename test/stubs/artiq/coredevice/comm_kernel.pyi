import typing

__all__ = ['CommKernelDummy']


class CommKernelDummy:
    def __init__(self) -> None:
        ...

    def load(self, kernel_library: typing.Any) -> None:
        ...

    def run(self) -> None:
        ...

    def serve(self, embedding_map: typing.Any, symbolizer: typing.Any, demangler: typing.Any) -> None:
        ...

    def check_system_info(self) -> None:
        ...
