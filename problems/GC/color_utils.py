from networkx.algorithms.coloring import greedy_color
import torch_geometric
import torch_geometric.utils as torch_utils


def get_heuristic_coloring(graph, strategy='random_sequential'):
    if isinstance(graph, torch_geometric.data.Data):
        graph = torch_utils.to_networkx(graph, to_undirected=True)
    coloring = greedy_color(graph, strategy=strategy)
    return max(list(coloring.values())) + 1


