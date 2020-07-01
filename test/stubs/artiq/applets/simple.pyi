import typing


class SimpleApplet:
    def __init__(self, main_widget_class: type, cmd_description: typing.Any = ...,
                 default_update_delay: float = ...):
        ...

    def add_dataset(self, name, help: typing.Optional[str] = ..., required: bool = ...) -> None:
        ...

    def args_init(self) -> None:
        ...

    def quamash_init(self) -> None:
        ...

    def ipc_init(self) -> None:
        ...

    def ipc_close(self) -> None:
        ...

    def create_main_widget(self) -> None:
        ...

    def sub_init(self, data: typing.Any) -> typing.Any:
        ...

    def filter_mod(self, mod: typing.Any) -> bool:
        ...

    def emit_data_changed(self, data: typing.Any, mod_buffer: typing.Any) -> None:
        ...

    def flush_mod_buffer(self) -> None:
        ...

    def sub_mod(self, mod: typing.Any) -> None:
        ...

    def subscribe(self) -> None:
        ...

    def unsubscribe(self) -> None:
        ...

    def run(self) -> None:
        ...


class TitleApplet(SimpleApplet):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        ...

    def args_init(self) -> None:
        ...

    def emit_data_changed(self, data: typing.Any, mod_buffer: typing.Any) -> None:
        ...
