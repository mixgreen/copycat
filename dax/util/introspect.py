import typing
import abc
import collections
import graphviz

import dax.base.dax
from dax.util.output import get_base_path

__all__ = ['GraphvizBase', 'ComponentGraphviz', 'RelationGraphviz']


def _get_attributes(o: typing.Any) -> typing.Iterator[typing.Any]:
    """Get an iterator over attributes of an object, excluding class-private attributes."""
    return (getattr(o, attr) for attr in dir(o) if attr[0:2] != '__')


__A_T = typing.Dict[str, str]  # Type of attribute dicts


class GraphvizBase(graphviz.Digraph):
    MODULE_NODE_ATTR = {'color': 'blue'}  # type:  __A_T
    MODULE_EDGE_ATTR = {}  # type:  __A_T
    SYSTEM_EDGE_ATTR = {'len': '1.6'}  # type:  __A_T

    SERVICE_NODE_ATTR = {'color': 'red'}  # type:  __A_T
    SERVICE_EDGE_ATTR = {}  # type:  __A_T

    INTER_CLUSTER_EDGE_ATTR = {'len': '2.5'}  # type:  __A_T

    def _add_modules(self, graph: graphviz.Digraph,
                     module: dax.base.dax.DaxModuleBase) -> None:
        """Recursive function to add a tree of modules to a graph.

        :param graph: The graph object
        :param module: The top module to start the recursion
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(module, dax.base.dax.DaxModuleBase)

        # Add module to the graph
        graph.node(module.get_system_key(), label=module.get_name(), **self.MODULE_NODE_ATTR)

        for child in module.children:
            # Inspect children of this module
            if isinstance(child, dax.base.dax.DaxModuleBase):
                # Recursive call
                self._add_modules(graph, child)
                # Add edge
                graph.edge(module.get_system_key(), child.get_system_key(),
                           **(self.SYSTEM_EDGE_ATTR if isinstance(module, dax.base.dax.DaxSystem)
                              else self.MODULE_EDGE_ATTR))

    def _add_services(self, graph: graphviz.Digraph,
                      services: typing.Sequence[dax.base.dax.DaxService]) -> None:
        """Add services to a sub-graph.

        :param graph: The graph object
        :param services: The sequence of services to add
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)

        for s in services:
            # Add service to the graph
            graph.node(s.get_system_key(), label=s.get_name(), **self.SERVICE_NODE_ATTR)

            for attr in _get_attributes(s):
                # Inspect attributes of this service
                if isinstance(attr, dax.base.dax.DaxService):
                    # Add edge to other service
                    graph.edge(s.get_system_key(), attr.get_system_key(), **self.SERVICE_EDGE_ATTR)

    def _add_service_modules(self, graph: graphviz.Digraph,
                             services: typing.Sequence[dax.base.dax.DaxService]) -> None:
        """Add service modules.

        :param graph: The graph object
        :param services: The sequence of services to find modules for
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)

        for s in services:
            for child in s.children:
                # Inspect children of this service
                if isinstance(child, dax.base.dax.DaxModuleBase):
                    # Add modules
                    self._add_modules(graph, child)
                    # Add edge
                    graph.edge(s.get_system_key(), child.get_system_key(), **self.MODULE_EDGE_ATTR)

    def _add_service_system_edges(self, graph: graphviz.Digraph,
                                  services: typing.Sequence[dax.base.dax.DaxService]) -> None:
        """Add service to system edges.

        :param graph: The graph object
        :param services: The sequence of services
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)

        for s in services:
            for attr in _get_attributes(s):
                # Inspect attributes of this service
                if isinstance(attr, dax.base.dax.DaxModuleBase):
                    # Add edge to connect service to module
                    graph.edge(attr.get_system_key(), s.get_system_key(),
                               dir='back', **self.INTER_CLUSTER_EDGE_ATTR)

    def _add_system_service_edges(self, graph: graphviz.Digraph,
                                  module: dax.base.dax.DaxModuleBase) -> None:
        """Recursive function to add system to service edges.

        :param graph: The graph object
        :param module: The top module to start the recursion
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(module, dax.base.dax.DaxModuleBase)

        for attr in _get_attributes(module):
            # Inspect attributes of this module
            if isinstance(attr, dax.base.dax.DaxModuleBase):
                # Recursive call
                self._add_system_service_edges(graph, attr)
            elif isinstance(attr, dax.base.dax.DaxService) and not isinstance(module, dax.base.dax.DaxSystem):
                # Add edge
                graph.edge(module.get_system_key(), attr.get_system_key(),
                           style='dashed', **self.INTER_CLUSTER_EDGE_ATTR)


class ComponentGraphviz(GraphvizBase):

    def __init__(self, system: dax.base.dax.DaxSystem, **kwargs: typing.Any):
        # Set default arguments
        kwargs.setdefault('engine', 'fdp')
        kwargs.setdefault('name', 'component_graph')
        kwargs.setdefault('directory', str(get_base_path(system.get_device('scheduler'))))
        graph_attr = kwargs.setdefault('graph_attr', {})
        graph_attr.setdefault('splines', 'spline')

        # Call super
        super(ComponentGraphviz, self).__init__(**kwargs)

        # List of all service objects
        services = [system.registry.get_service(k) for k in system.registry.get_service_key_list()]
        # Service cluster
        service_cluster = graphviz.Digraph(name='cluster_services',
                                           graph_attr={'label': 'Services'})
        # Add all services
        self._add_services(service_cluster, services)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System'})
        # Add the system
        self._add_modules(system_cluster, system)

        # Add service modules
        self._add_service_modules(service_cluster, services)

        # Add clusters to this graph
        self.subgraph(system_cluster)
        self.subgraph(service_cluster)


class RelationGraphviz(GraphvizBase):

    def __init__(self, system: dax.base.dax.DaxSystem, **kwargs: typing.Any):
        # Set default arguments
        kwargs.setdefault('engine', 'dot')
        kwargs.setdefault('name', 'relations_graph')
        kwargs.setdefault('directory', str(get_base_path(system.get_device('scheduler'))))
        graph_attr = kwargs.setdefault('graph_attr', {})
        graph_attr.setdefault('splines', 'spline')
        graph_attr.setdefault('compound', 'true')  # Required for inter-cluster edges
        graph_attr.setdefault('ranksep', '2.0')  # Vertical spacing for better edge visibility

        # Call super
        super(RelationGraphviz, self).__init__(**kwargs)

        # List of all service objects
        services = [system.registry.get_service(k) for k in system.registry.get_service_key_list()]
        # Service cluster
        service_cluster = graphviz.Digraph(name='cluster_services',
                                           graph_attr={'label': 'Services', 'labelloc': 'b'})
        # Add all services
        self._add_services(service_cluster, services)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System'})
        # Add the system
        self._add_modules(system_cluster, system)

        # Add inter-cluster edges
        self._add_service_system_edges(self, services)
        self._add_system_service_edges(self, system)

        # Add clusters to this graph
        self.subgraph(system_cluster)
        self.subgraph(service_cluster)
