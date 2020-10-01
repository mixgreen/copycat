import networkx as nx

__all__ = ['is_directed_acyclic_graph', 'transitive_reduction']


def is_directed_acyclic_graph(G: nx.Graph) -> bool:
    ...


def transitive_reduction(G: nx.DiGraph) -> nx.DiGraph:
    ...
