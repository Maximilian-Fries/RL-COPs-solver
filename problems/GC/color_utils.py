from networkx.algorithms.coloring import greedy_color
import numpy as np
import pickle
import torch_geometric
import torch_geometric.utils as torch_utils


def get_heuristic_coloring(graph, strategy='random_sequential'):
    assert isinstance(graph, torch_geometric.data.Data), "graph should be an instance of torch_geometric.data.Data, " \
                                                         "but is %s" % (str(type(graph)))
    graph = torch_utils.to_networkx(graph, to_undirected=True)
    coloring = greedy_color(graph, strategy=strategy)
    return max(list(coloring.values())) + 1


def get_graph_colorings(graphs, save_file):
    strategies = ['random_sequential', 'largest_first', 'DSATUR', 'smallest_last']

    heuristic_results = {strategy: np.zeros(len(graphs)) for strategy in
                         strategies}

    for ind, graph in enumerate(graphs):
        for strategy in strategies:
            n_col = get_heuristic_coloring(graph, strategy)
            heuristic_results[strategy][ind] = n_col

    with open(save_file, 'wb') as fout:
        pickle.dump(heuristic_results, fout)

