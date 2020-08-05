import typing
import collections
import graphviz
import logging

import dax.base.dax
from dax.util.output import get_base_path

__all__ = ['GraphvizBase', 'ComponentGraphviz', 'RelationGraphviz']


def _get_attributes(o: typing.Any) -> typing.Iterator[typing.Any]:
    """Get an iterator over attributes of an object, excluding class-private attributes."""
    return (getattr(o, attr) for attr in dir(o) if attr[0:2] != '__')


_logger: logging.Logger = logging.getLogger(__name__)
"""Module logger object."""


class GraphvizBase(graphviz.Digraph):
    """Graphviz base class with helper functions."""

    __A_T = typing.Dict[str, str]  # Type of attribute dicts

    MODULE_EDGE_K: float = 0.8
    """The default module edge spring constant."""
    SYSTEM_EDGE_LEN: float = 1.5
    """The default system edge length in inch."""
    CLUSTER_EDGE_LEN: float = 2.5
    """The default cluster edge length in inch."""

    def __init__(self, *,
                 module_edge_k: float = MODULE_EDGE_K,
                 system_edge_len: float = SYSTEM_EDGE_LEN,
                 cluster_edge_len: float = CLUSTER_EDGE_LEN,
                 **kwargs: typing.Any):
        """Create a new Graphviz base class with the given parameters.

        :param module_edge_k: Module edge spring constant
        :param system_edge_len: System edge preferred length in inch
        :param cluster_edge_len: Cluster edge preferred length in inch
        :param kwargs: Keyword arguments for the Graphviz parent class
        """
        assert isinstance(module_edge_k, float), 'Module edge K must be of type float'
        assert isinstance(system_edge_len, float), 'System edge len must be of type float'
        assert isinstance(cluster_edge_len, float), 'Cluster edge len must be of type float'

        self._module_node_attr: GraphvizBase.__A_T = {'color': 'blue'}
        """Node attributes for modules."""
        self._module_edge_attr: GraphvizBase.__A_T = {'K': str(module_edge_k)}
        """Edge attributes for modules."""
        self._system_edge_attr: GraphvizBase.__A_T = {'len': str(system_edge_len)}
        """Edge attributes for the system."""

        self._service_node_attr: GraphvizBase.__A_T = {'color': 'red'}
        """Node attributes for services."""
        self._service_edge_attr: GraphvizBase.__A_T = {}
        """Edge attributes for services."""

        self._cluster_edge_attr: GraphvizBase.__A_T = {'len': str(cluster_edge_len)}
        """Edge attributes for inter-cluster edges."""

        # Call super
        super(GraphvizBase, self).__init__(**kwargs)

    def _add_modules(self, graph: graphviz.Digraph,
                     module: dax.base.dax.DaxModuleBase,
                     to_module_edges: bool) -> None:
        """Recursive function to add a tree of modules to a graph.

        :param graph: The graph object
        :param module: The top module to start the recursion
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(module, dax.base.dax.DaxModuleBase)
        assert isinstance(to_module_edges, bool)

        # Add module to the graph
        graph.node(module.get_system_key(), label=module.get_name(), **self._module_node_attr)
        _logger.debug(f'Added module "{module.get_system_key()}"')

        # Inspect children of this module for modules
        child_modules = {child for child in module.children if isinstance(child, dax.base.dax.DaxModuleBase)}
        _logger.debug(f'Found {len(child_modules)} child module(s)')

        for child in child_modules:
            # Recursive call
            self._add_modules(graph, child, to_module_edges)
            if to_module_edges:
                # Add edge
                graph.edge(module.get_system_key(), child.get_system_key(),
                           **(self._system_edge_attr if isinstance(module, dax.base.dax.DaxSystem)
                              else self._module_edge_attr))

        # Inspect attributes of this module for modules
        attr_modules = {attr for attr in _get_attributes(module) if isinstance(attr, dax.base.dax.DaxModuleBase)}

        # Check if there are any unexpected attributes
        unexpected_modules = attr_modules - child_modules
        if unexpected_modules:
            _logger.warning(f'Found {len(unexpected_modules)} unexpected module(s) '
                            f'in module "{module.get_system_key()}"')

        if to_module_edges:
            for m in unexpected_modules:
                # Add edge
                graph.edge(module.get_system_key(), m.get_system_key(), style='dashed',
                           **(self._system_edge_attr if isinstance(module, dax.base.dax.DaxSystem)
                              else self._module_edge_attr))

    def _add_services(self, graph: graphviz.Digraph,
                      services: typing.Sequence[dax.base.dax.DaxService],
                      to_service_edges: bool) -> None:
        """Add services to a sub-graph.

        :param graph: The graph object
        :param services: The sequence of services to add
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)
        assert isinstance(to_service_edges, bool)

        for s in services:
            # Add service to the graph
            graph.node(s.get_system_key(), label=s.get_name(), **self._service_node_attr)
            _logger.debug(f'Added service "{s.get_system_key()}"')

            if to_service_edges:
                # Inspect attributes of this service for services
                attr_services = [attr for attr in _get_attributes(s) if isinstance(attr, dax.base.dax.DaxService)]
                _logger.debug(f'Found {len(attr_services)} edge(s) to other services')

                for attr in attr_services:
                    # Add edge to other service
                    graph.edge(s.get_system_key(), attr.get_system_key(), **self._service_edge_attr)

    def _add_service_modules(self, graph: graphviz.Digraph,
                             services: typing.Sequence[dax.base.dax.DaxService]) -> None:
        """Add service modules.

        :param graph: The graph object
        :param services: The sequence of services to find modules for
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)

        for s in services:
            # Inspect children of this service for modules
            child_modules = [child for child in s.children if isinstance(child, dax.base.dax.DaxModuleBase)]
            _logger.debug(f'Found {len(child_modules)} child module(s) for service "{s.get_system_key()}"')

            for child in child_modules:
                # Add modules
                self._add_modules(graph, child, to_module_edges=True)
                # Add edge
                graph.edge(s.get_system_key(), child.get_system_key(), **self._module_edge_attr)

    def _add_service_system_edges(self, graph: graphviz.Digraph,
                                  services: typing.Sequence[dax.base.dax.DaxService]) -> None:
        """Add service to system edges.

        :param graph: The graph object
        :param services: The sequence of services
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)

        for s in services:
            # Inspect children of this service for modules
            child_modules = {child for child in s.children if isinstance(child, dax.base.dax.DaxModuleBase)}
            # Inspect attributes of this service for modules
            attr_modules = {attr for attr in _get_attributes(s) if isinstance(attr, dax.base.dax.DaxModuleBase)}

            # Obtain difference
            modules = attr_modules - child_modules
            _logger.debug(f'Found {len(modules)} edge(s) to other modules for service "{s.get_system_key()}"')

            for module in modules:
                # Add edge to connect service to module
                graph.edge(module.get_system_key(), s.get_system_key(), dir='back', **self._cluster_edge_attr)

    def _add_system_service_edges(self, graph: graphviz.Digraph,
                                  module: dax.base.dax.DaxModuleBase) -> None:
        """Recursive function to add system to service edges.

        :param graph: The graph object
        :param module: The top module to start the recursion
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(module, dax.base.dax.DaxModuleBase)

        # Only add edges to services if this is not the system
        if not isinstance(module, dax.base.dax.DaxSystem):
            # Inspect attributes of this module for services
            attr_services = [attr for attr in _get_attributes(module) if isinstance(attr, dax.base.dax.DaxService)]
            _logger.debug(f'Found {len(attr_services)} edge(s) to services for module "{module.get_system_key()}"')

            for attr in attr_services:
                # Add edge
                graph.edge(module.get_system_key(), attr.get_system_key(), style='dashed', **self._cluster_edge_attr)

        # Inspect children of this module for modules
        child_modules = [child for child in module.children if isinstance(child, dax.base.dax.DaxModuleBase)]
        for child in child_modules:
            # Recursive call
            self._add_system_service_edges(graph, child)


class ComponentGraphviz(GraphvizBase):
    """Component graph class which visualizes the relations between system modules."""

    def __init__(self, system: dax.base.dax.DaxSystem, **kwargs: typing.Any):
        """Create a new component Graphviz object.

        :param system: The system to visualize
        :param kwargs: Keyword arguments for :class:`GraphvizBase` and the Graphviz parent class
        """
        assert isinstance(system, dax.base.dax.DaxSystem), 'System must be a DAX system'

        # Set default arguments
        kwargs.setdefault('engine', 'fdp')
        kwargs.setdefault('name', 'component_graph')
        kwargs.setdefault('directory', str(get_base_path(system.get_device('scheduler'))))
        graph_attr = kwargs.setdefault('graph_attr', {})
        graph_attr.setdefault('splines', 'spline')

        # Call super
        super(ComponentGraphviz, self).__init__(**kwargs)

        # List of all service objects
        services = system.registry.get_service_list()
        # Service cluster
        service_cluster = graphviz.Digraph(name='cluster_services',
                                           graph_attr={'label': 'Services'})
        # Add all services
        self._add_services(service_cluster, services, to_service_edges=True)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System'})
        # Add the system
        self._add_modules(system_cluster, system, to_module_edges=True)

        # Add service modules
        self._add_service_modules(service_cluster, services)

        # Add clusters to this graph
        self.subgraph(system_cluster)
        self.subgraph(service_cluster)


class RelationGraphviz(GraphvizBase):
    """Relation graph class which visualizes relations between the system modules and the services."""

    def __init__(self, system: dax.base.dax.DaxSystem, **kwargs: typing.Any):
        """Create a new relation Graphviz object.

        :param system: The system to visualize
        :param kwargs: Keyword arguments for :class:`GraphvizBase` and the Graphviz parent class
        """
        assert isinstance(system, dax.base.dax.DaxSystem), 'System must be a DAX system'

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
        services = system.registry.get_service_list()
        # Service cluster
        service_cluster = graphviz.Digraph(name='cluster_services',
                                           graph_attr={'label': 'Services', 'labelloc': 'b'})
        # Add all services
        self._add_services(service_cluster, services, to_service_edges=False)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System'})
        # Add the system
        self._add_modules(system_cluster, system, False)

        # Add inter-cluster edges
        self._add_service_system_edges(self, services)
        self._add_system_service_edges(self, system)

        # Add clusters to this graph
        self.subgraph(system_cluster)
        self.subgraph(service_cluster)
