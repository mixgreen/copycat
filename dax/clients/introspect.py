from dax.experiment import *
from dax.util.introspect import ComponentGraphviz, RelationGraphviz


@dax_client_factory
class Introspect(DaxClient, EnvExperiment):
    """System introspection tool."""

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

    def prepare(self):
        # Get the system
        system = self.registry.find_module(DaxSystem)
        # Create the graph objects
        self._graphs = [g(system) for g in self.GRAPHS[self._graph_arg]]

    def run(self):
        pass

    def analyze(self):
        # Render arguments
        kwargs = {
            'view': self._view,
        }

        # Render all graphs
        for g in self._graphs:
            g.render(**kwargs)
