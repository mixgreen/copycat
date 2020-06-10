import typing

__all__ = ['Graph', 'Digraph']

__ATTR_DICT = typing.Optional[typing.Dict[str, str]]


class Dot:
    def __init__(self, name: typing.Optional[str] = None, comment: typing.Optional[str] = None,
                 filename: typing.Optional[str] = None, directory: typing.Optional[str] = None,
                 format: typing.Optional[str] = None, engine: typing.Optional[str] = None,
                 encoding: typing.Any = None,
                 graph_attr: __ATTR_DICT = None, node_attr: __ATTR_DICT = None,
                 edge_attr: __ATTR_DICT = None, body: typing.Any = None,
                 strict: bool = False):
        ...

    def clear(self, keep_attrs: bool = False) -> None:
        ...

    def __iter__(self, subgraph: bool = False) -> str:
        ...

    def __str__(self) -> str:
        ...

    def node(self, name: str, label: typing.Optional[str] = None,
             _attributes: typing.Any = None, **attrs: str) -> None:
        ...

    def edge(self, tail_name: str, head_name: str, label: typing.Optional[str] = None,
             _attributes: typing.Any = None, **attrs: str) -> None:
        ...

    def edges(self, tail_head_iter: typing.Sequence[typing.Tuple[str, str]]) -> None:
        ...

    def attr(self, kw: typing.Optional[str] = None, _attributes: typing.Any = None, **attrs: str) -> None:
        ...

    def subgraph(self, graph: typing.Optional[Dot] = None,
                 name: typing.Optional[str] = None, comment: typing.Optional[str] = None,
                 graph_attr: __ATTR_DICT = None, node_attr: __ATTR_DICT = None,
                 edge_attr: __ATTR_DICT = None, body: typing.Any = None) -> typing.Optional['SubgraphContext']:
        ...


class SubgraphContext:
    def __init__(self, parent: Dot, kwargs: typing.Dict[str, typing.Any]):
        ...

    def __enter__(self):
        ...

    def __exit__(self, type_: typing.Any, value: typing.Any, traceback: typing.Any):
        ...


class Graph(Dot):
    @property
    def directed(self) -> bool:
        ...


class Digraph(Dot):
    @property
    def directed(self) -> bool:
        ...
