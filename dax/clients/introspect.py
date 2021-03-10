import typing

from dax.experiment import *
from dax.util.introspect import GraphvizBase, ComponentGraphviz, RelationGraphviz

__all__ = ['Introspect']


@dax_client_factory
class Introspect(DaxClient, Experiment):
    """System introspection tool."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    GRAPHS: typing.Dict[str, typing.List[typing.Tuple[type, typing.Dict[str, typing.Any]]]] = {
        'All': [
            (ComponentGraphviz, {}),
            (ComponentGraphviz, {'engine': 'fdp'}),
            (RelationGraphviz, {})
        ],
        'Component graph': [(ComponentGraphviz, {})],
        'Component graph (fdp)': [(ComponentGraphviz, {'engine': 'fdp'})],
        'Relation graph': [(RelationGraphviz, {})],
    }
    """Dict with available graph types."""

    def build(self) -> None:  # type: ignore
        # Add arguments
        self._graph_types = self.get_argument('Graph', EnumerationValue(list(self.GRAPHS), default='All'))
        self._view = self.get_argument('View result', BooleanValue(True))

        # Graph arguments
        graph_args: typing.Dict[str, typing.Dict[str, typing.Any]] = {
            'module_edge_k': {
                'key': 'Module edge K',
                'processor': NumberValue(GraphvizBase.MODULE_EDGE_K, min=0.0),
                'tooltip': 'Module edge spring constant',
            },
            'module_edge_len': {
                'key': 'Module edge len',
                'processor': NumberValue(GraphvizBase.MODULE_EDGE_LEN, min=0.0),
                'tooltip': 'Module edge preferred length in inches',
            },
            'system_edge_len': {
                'key': 'System edge len',
                'processor': NumberValue(GraphvizBase.SYSTEM_EDGE_LEN, min=0.0),
                'tooltip': 'System edge preferred length in inches',
            },
            'service_edge_k': {
                'key': 'Service edge K',
                'processor': NumberValue(GraphvizBase.SERVICE_EDGE_K, min=0.0),
                'tooltip': 'Service edge spring constant',
            },
            'service_edge_len': {
                'key': 'Service edge len',
                'processor': NumberValue(GraphvizBase.SERVICE_EDGE_LEN, min=0.0),
                'tooltip': 'Service edge preferred length in inches',
            },
            'cluster_edge_len': {
                'key': 'Cluster edge len',
                'processor': NumberValue(GraphvizBase.CLUSTER_EDGE_LEN, min=0.0),
                'tooltip': 'Cluster edge preferred length in inches',
            },
        }
        self._graph_args: typing.Dict[str, float] = {
            k: self.get_argument(group='Graph configuration', **v) for k, v in graph_args.items()
        }

    def prepare(self) -> None:
        # Get the system
        system = self.registry.find_module(DaxSystem)
        # Create the graph objects
        self._graphs = [g(system, **kwargs, **self._graph_args) for g, kwargs in self.GRAPHS[self._graph_types]]

    def run(self) -> None:
        pass

    def analyze(self) -> None:
        # Render all graphs
        for g in self._graphs:
            g.render(view=self._view)
