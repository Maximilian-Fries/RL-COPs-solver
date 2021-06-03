import torch_geometric.utils as torch_utils
import torch_geometric
from networkx.algorithms.approximation import min_weighted_vertex_cover
import operator
import numpy as np
import networkx


def list_right(graph):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_utils.to_networkx(graph, to_undirected=True)
    degrees = [val for (node, val) in graph.degree()]
    inds = list(range(len(degrees)))
    inds_sorted = sorted(inds, key=lambda ind: degrees[ind])
    C = set()
    for node in inds_sorted[::-1]:
        for u in graph.neighbors(node):
            if u not in C:
                C.add(node)
                break
    return C


# extremely inefficient
def get_iter_greedy(graph):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_utils.to_networkx(graph, to_undirected=True)
    graph = graph.copy()

    edge_deg = dict.fromkeys(graph.edges)
    vc = set()
    m = len(graph.edges)
    print(graph.number_of_nodes())
    for edge in graph.edges:
        edge_deg[edge] = graph.degree(edge[0]) + graph.degree(edge[1])
    while m>0:
        max_edge = max(edge_deg.items(), key=operator.itemgetter(1))[0]
        vc.update(max_edge)
        # remove edges from graph
        to_remove_edges = set(graph.edges(max_edge))
        graph.remove_edges_from(to_remove_edges)

        m = m - len(to_remove_edges)

        for e in to_remove_edges:
            del edge_deg[(min(e[0], e[1])), max(e[0], e[1])]

        for edge in graph.edges:
            edge_deg[edge] = graph.degree(edge[0]) + graph.degree(edge[1])
    return vc


def random_cover(graph):
    if isinstance(graph, networkx.classes.graph.Graph):
        graph = torch_utils.from_networkx(graph)
    n = graph.num_nodes
    pi = np.random.permutation(n)
    subset = []
    edge_index = graph.edge_index
    mask = np.zeros(n, dtype=np.bool)
    for v in pi:
        mask[subset] = 1
        covered_edges = mask[edge_index[0]] | mask[edge_index[1]]
        if np.all(covered_edges):
            return len(subset)
        subset.append(v.item())


def get_heuristic_cover(graph, strategy):

    if strategy == 'iterative':
        if isinstance(graph, torch_geometric.data.Data):
            graph = torch_utils.to_networkx(graph, to_undirected=True)
        return len(min_weighted_vertex_cover(graph))

    elif strategy == 'iter_greedy':
        return len(get_iter_greedy(graph))
    elif strategy == 'list_right':
        return len(list_right(graph))
    elif strategy == 'random_sequential':
        return random_cover(graph)


