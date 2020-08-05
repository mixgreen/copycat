from dax.experiment import *
from dax.util.introspect import GraphvizBase, ComponentGraphviz, RelationGraphviz

__all__ = ['Introspect']


@dax_client_factory
class Introspect(DaxClient, EnvExperiment):
    """System introspection tool."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    GRAPHS = {
        'All': [ComponentGraphviz, RelationGraphviz],
        'Component graph': [ComponentGraphviz],
        'Relation graph': [RelationGraphviz],
    }
    """Dict with available graph types."""

    def build(self) -> None:  # type: ignore
        # Add arguments
        self._graph_arg = self.get_argument('Graph', EnumerationValue(list(self.GRAPHS), default='All'))
        self._view = self.get_argument('View result', BooleanValue(True))

        # Graph config
        self._module_edge_k: float = self.get_argument('Module edge K',
                                                       NumberValue(GraphvizBase.MODULE_EDGE_K, min=0.0),
                                                       group='Graph configuration',
                                                       tooltip='Module edge spring constant')
        self._system_edge_len: float = self.get_argument('System edge len',
                                                         NumberValue(GraphvizBase.SYSTEM_EDGE_LEN, min=0.0),
                                                         group='Graph configuration',
                                                         tooltip='System edge preferred length in inches')
        self._cluster_edge_len: float = self.get_argument('Cluster edge len',
                                                          NumberValue(GraphvizBase.CLUSTER_EDGE_LEN, min=0.0),
                                                          group='Graph configuration',
                                                          tooltip='Cluster edge preferred length in inches')

    def prepare(self):
        # Get the system
        system = self.registry.find_module(DaxSystem)
        # Create the graph objects
        self._graphs = [g(system,
                          module_edge_k=self._module_edge_k,
                          system_edge_len=self._system_edge_len,
                          cluster_edge_len=self._cluster_edge_len)
                        for g in self.GRAPHS[self._graph_arg]]

    def run(self):
        pass

    def analyze(self):
        # Render all graphs
        for g in self._graphs:
            g.render(view=self._view)
