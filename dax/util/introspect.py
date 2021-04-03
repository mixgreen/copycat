import typing
import collections.abc
import graphviz
import logging

import dax.base.system
from dax.util.output import get_base_path

__all__ = ['GraphvizBase', 'ComponentGraphviz', 'RelationGraphviz', 'CompoundGraphviz']


def _get_attributes(o: typing.Any) -> typing.Generator[typing.Any, None, None]:
    """Get a generator over attributes of an object, excluding class-private attributes."""
    for attr in dir(o):
        if not attr.startswith('__'):
            try:
                yield getattr(o, attr)
            except AttributeError:
                pass


_logger: logging.Logger = logging.getLogger(__name__)
"""Module logger object."""


class GraphvizBase(graphviz.Digraph):
    """Graphviz base class with helper functions."""

    __A_T = typing.Dict[str, str]  # Type of attribute dicts

    MODULE_EDGE_K: float = 0.8
    """The default module edge spring constant."""
    MODULE_EDGE_LEN: float = 0.3
    """The default module edge length in inch."""
    SYSTEM_EDGE_LEN: float = 1.5
    """The default system edge length in inch."""

    SERVICE_EDGE_K: float = 0.7
    """The default service edge spring constant."""
    SERVICE_EDGE_LEN: float = 0.7
    """The default service edge length in inch."""

    CLUSTER_EDGE_LEN: float = 2.5
    """The default cluster edge length in inch."""

    def __init__(self, *,
                 module_edge_k: float = MODULE_EDGE_K,
                 module_edge_len: float = MODULE_EDGE_LEN,
                 system_edge_len: float = SYSTEM_EDGE_LEN,
                 service_edge_k: float = SERVICE_EDGE_K,
                 service_edge_len: float = SERVICE_EDGE_LEN,
                 cluster_edge_len: float = CLUSTER_EDGE_LEN,
                 **kwargs: typing.Any):
        """Create a new Graphviz base class with the given parameters.

        :param module_edge_k: Module edge spring constant
        :param module_edge_len: Module edge preferred length in inch
        :param system_edge_len: System edge preferred length in inch
        :param service_edge_k: Service edge spring constant
        :param service_edge_len: Service edge preferred length in inch
        :param cluster_edge_len: Cluster edge preferred length in inch
        :param kwargs: Keyword arguments for the Graphviz parent class
        """
        assert isinstance(module_edge_k, float), 'Module edge K must be of type float'
        assert isinstance(module_edge_len, float), 'Module edge len must be of type float'
        assert isinstance(system_edge_len, float), 'System edge len must be of type float'
        assert isinstance(service_edge_k, float), 'Service edge K must be of type float'
        assert isinstance(service_edge_len, float), 'Service edge len must be of type float'
        assert isinstance(cluster_edge_len, float), 'Cluster edge len must be of type float'

        self._module_node_attr: GraphvizBase.__A_T = {'color': 'blue'}
        """Node attributes for modules."""
        self._module_edge_attr: GraphvizBase.__A_T = {'K': str(module_edge_k), 'len': str(module_edge_len)}
        """Edge attributes for modules."""
        self._system_edge_attr: GraphvizBase.__A_T = {'len': str(system_edge_len)}
        """Edge attributes for the system."""

        self._service_node_attr: GraphvizBase.__A_T = {'color': 'red'}
        """Node attributes for services."""
        self._service_edge_attr: GraphvizBase.__A_T = {'K': str(service_edge_k), 'len': str(service_edge_len)}
        """Edge attributes for services."""

        self._cluster_edge_attr: GraphvizBase.__A_T = {'len': str(cluster_edge_len)}
        """Edge attributes for inter-cluster edges."""

        # Call super
        super(GraphvizBase, self).__init__(**kwargs)

    def _add_modules(self, graph: graphviz.Digraph,
                     module: dax.base.system.DaxModuleBase, *,
                     to_module_edges: bool) -> None:
        """Recursive function to add a tree of modules to a graph.

        :param graph: The graph object
        :param module: The top module to start the recursion
        :param to_module_edges: Add edges to other modules
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(module, dax.base.system.DaxModuleBase)
        assert isinstance(to_module_edges, bool)

        # Add module to the graph
        graph.node(module.get_system_key(), label=module.get_name(), **self._module_node_attr)
        _logger.debug(f'Added module "{module.get_system_key()}"')

        # Inspect children of this module for modules
        child_modules = {child for child in module.children if isinstance(child, dax.base.system.DaxModuleBase)}
        _logger.debug(f'Found {len(child_modules)} child module(s)')

        for child in child_modules:
            # Recursive call
            self._add_modules(graph, child, to_module_edges=to_module_edges)
            if to_module_edges:
                # Add edge
                graph.edge(module.get_system_key(), child.get_system_key(),
                           **(self._system_edge_attr if isinstance(module, dax.base.system.DaxSystem)
                              else self._module_edge_attr))

        # Inspect attributes of this module for modules
        attr_modules = {attr for attr in _get_attributes(module) if isinstance(attr, dax.base.system.DaxModuleBase)}

        # Check if there are any unexpected attributes
        unexpected_modules = attr_modules - child_modules
        if unexpected_modules:
            _logger.warning(f'Found {len(unexpected_modules)} unexpected module(s) '
                            f'in module "{module.get_system_key()}"')

        if to_module_edges:
            for m in unexpected_modules:
                # Add edge
                graph.edge(module.get_system_key(), m.get_system_key(), style='dashed',
                           **(self._system_edge_attr if isinstance(module, dax.base.system.DaxSystem)
                              else self._module_edge_attr))

    def _add_services(self, graph: graphviz.Digraph,
                      services: typing.Sequence[dax.base.system.DaxService], *,
                      to_service_edges: bool) -> None:
        """Add services to a sub-graph.

        :param graph: The graph object
        :param services: The sequence of services to add
        :param to_service_edges: Add edges to other services
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
                attr_services = [attr for attr in _get_attributes(s) if isinstance(attr, dax.base.system.DaxService)]
                _logger.debug(f'Found {len(attr_services)} edge(s) to other services')

                for attr in attr_services:
                    # Add edge to other service
                    graph.edge(s.get_system_key(), attr.get_system_key(), **self._service_edge_attr)

    def _add_service_modules(self, graph: graphviz.Digraph,
                             services: typing.Sequence[dax.base.system.DaxService]) -> None:
        """Add service modules.

        :param graph: The graph object
        :param services: The sequence of services to find modules for
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)

        for s in services:
            # Inspect children of this service for modules
            child_modules = [child for child in s.children if isinstance(child, dax.base.system.DaxModuleBase)]
            _logger.debug(f'Found {len(child_modules)} child module(s) for service "{s.get_system_key()}"')

            for child in child_modules:
                # Add modules
                self._add_modules(graph, child, to_module_edges=True)
                # Add edge
                graph.edge(s.get_system_key(), child.get_system_key(), **self._module_edge_attr)

    def _add_service_system_edges(self, graph: graphviz.Digraph,
                                  services: typing.Sequence[dax.base.system.DaxService],
                                  *,
                                  system: typing.Optional[dax.base.system.DaxSystem] = None,
                                  **kwargs: str) -> None:
        """Add service to system edges.

        :param graph: The graph object
        :param services: The sequence of services
        :param system: Optional reference to the system, used to add invisible edges between services and the system
        :param kwargs: Additional keyword arguments for the edges
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(services, collections.abc.Sequence)
        assert isinstance(system, dax.base.system.DaxSystem) or system is None

        for s in services:
            # Inspect children of this service for modules
            child_modules = {child for child in s.children if isinstance(child, dax.base.system.DaxModuleBase)}
            # Inspect attributes of this service for modules
            attr_modules = {attr for attr in _get_attributes(s) if isinstance(attr, dax.base.system.DaxModuleBase)}

            # Obtain difference
            modules = attr_modules - child_modules
            _logger.debug(f'Found {len(modules)} edge(s) to other modules for service "{s.get_system_key()}"')

            for module in modules:
                # Add edge to connect service to module
                graph.edge(s.get_system_key(), module.get_system_key(), **kwargs, **self._cluster_edge_attr)

                if system is not None:
                    # Add invisible edge to the system to enforce vertical alignment
                    graph.edge(s.get_system_key(), system.get_system_key(), style='invis')

    def _add_system_service_edges(self, graph: graphviz.Digraph,
                                  module: dax.base.system.DaxModuleBase,
                                  **kwargs: str) -> None:
        """Recursive function to add system to service edges.

        :param graph: The graph object
        :param module: The top module to start the recursion
        :param kwargs: Additional keyword arguments for the edges
        """
        assert isinstance(graph, graphviz.Digraph)
        assert isinstance(module, dax.base.system.DaxModuleBase)

        # Only add edges to services if this is not the system
        if not isinstance(module, dax.base.system.DaxSystem):
            # Inspect attributes of this module for services
            attr_services = [attr for attr in _get_attributes(module) if isinstance(attr, dax.base.system.DaxService)]
            _logger.debug(f'Found {len(attr_services)} edge(s) to services for module "{module.get_system_key()}"')

            for attr in attr_services:
                # Add edge
                graph.edge(attr.get_system_key(), module.get_system_key(), dir='back',
                           **kwargs, **self._cluster_edge_attr)

        # Inspect children of this module for modules
        child_modules = [child for child in module.children if isinstance(child, dax.base.system.DaxModuleBase)]
        for child in child_modules:
            # Recursive call
            self._add_system_service_edges(graph, child, **kwargs)


class ComponentGraphviz(GraphvizBase):
    """Component graph class which visualizes the relations between system modules or services."""

    def __init__(self, system: dax.base.system.DaxSystem, **kwargs: typing.Any):
        """Create a new component Graphviz object.

        :param system: The system to visualize
        :param kwargs: Keyword arguments for :class:`GraphvizBase` and the Graphviz parent class
        """
        assert isinstance(system, dax.base.system.DaxSystem), 'System must be a DAX system'

        # Set default arguments
        kwargs.setdefault('engine', 'dot')
        kwargs.setdefault('name', f'component_graph_{kwargs["engine"]}')
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
        # Add service modules
        self._add_service_modules(service_cluster, services)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System'})
        # Add the system
        self._add_modules(system_cluster, system, to_module_edges=True)

        # Add clusters to this graph
        self.subgraph(system_cluster)
        self.subgraph(service_cluster)


class RelationGraphviz(GraphvizBase):
    """Relation graph class which visualizes relations between the system modules and the services."""

    def __init__(self, system: dax.base.system.DaxSystem, **kwargs: typing.Any):
        """Create a new relation Graphviz object.

        :param system: The system to visualize
        :param kwargs: Keyword arguments for :class:`GraphvizBase` and the Graphviz parent class
        """
        assert isinstance(system, dax.base.system.DaxSystem), 'System must be a DAX system'

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
                                           graph_attr={'label': 'Services'})
        # Add all services
        self._add_services(service_cluster, services, to_service_edges=False)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System', 'labelloc': 'b'})
        # Add the system
        self._add_modules(system_cluster, system, to_module_edges=False)

        # Add inter-cluster edges
        self._add_service_system_edges(self, services)
        self._add_system_service_edges(self, system, style='dashed')

        # Add clusters to this graph
        self.subgraph(system_cluster)
        self.subgraph(service_cluster)


class CompoundGraphviz(GraphvizBase):
    """Component and relation graph class."""

    def __init__(self, system: dax.base.system.DaxSystem, **kwargs: typing.Any):
        """Create a new compound Graphviz object.

        :param system: The system to visualize
        :param kwargs: Keyword arguments for :class:`GraphvizBase` and the Graphviz parent class
        """
        assert isinstance(system, dax.base.system.DaxSystem), 'System must be a DAX system'

        # Set default arguments
        kwargs.setdefault('engine', 'dot')
        kwargs.setdefault('name', 'compound_graph')
        kwargs.setdefault('directory', str(get_base_path(system.get_device('scheduler'))))
        graph_attr = kwargs.setdefault('graph_attr', {})
        graph_attr.setdefault('splines', 'spline')
        graph_attr.setdefault('compound', 'true')  # Required for inter-cluster edges
        graph_attr.setdefault('ranksep', '2.0')  # Vertical spacing for better edge visibility

        # Call super
        super(CompoundGraphviz, self).__init__(**kwargs)

        # List of all service objects
        services = system.registry.get_service_list()
        # Service cluster
        service_cluster = graphviz.Digraph(name='cluster_services',
                                           graph_attr={'label': 'Services'})
        # Add all services
        self._add_services(service_cluster, services, to_service_edges=True)

        # System cluster
        system_cluster = graphviz.Digraph(name='cluster_system',
                                          graph_attr={'label': 'System', 'labelloc': 'b'})
        # Add the system
        self._add_modules(system_cluster, system, to_module_edges=True)

        # Add inter-cluster edges
        self._add_service_system_edges(self, services, system=system, style='dashed')
        self._add_system_service_edges(self, system, style='dashed')

        # Add clusters to this graph
        self.subgraph(service_cluster)
        self.subgraph(system_cluster)
