from torch.utils.data import Dataset
import os
import pickle
import torch_geometric.utils as torch_utils
from utils import load_problem, get_graph_function


class MyDataset(Dataset):

    def __init__(self, filename=None, num_samples=1000, graph_size=20, is_hybrid=False, hybrid_graphs=None,
                 graph_type='ER', opts=None, validation=False):
        super(MyDataset, self).__init__()

        if filename is not None:
            print('Load graphs from {}'.format(filename))
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                self.data = [pickle.load(f) for _ in range(num_samples)]

        elif not is_hybrid:
            graph_func, args = get_graph_function(graph_type, opts)
            if not validation:
                self.data = [torch_utils.from_networkx(graph_func(graph_size, *args, seed=opts.seed + i))
                             for i in range(num_samples)]
            else:
                self.data = [torch_utils.from_networkx(graph_func(graph_size, *args))
                             for _ in range(num_samples)]

        else:
            self.data = []
            n_graph_types = len(hybrid_graphs)
            size = num_samples // n_graph_types
            sizes = [size] * (n_graph_types - 1)
            sizes.append(num_samples - (n_graph_types - 1) * size)
            if not validation:
                for size, graph_type in zip(sizes, hybrid_graphs):
                    graph_func, args = get_graph_function(graph_type, opts)
                    next = [
                        torch_utils.from_networkx(graph_func(graph_size, *args, seed=opts.seed + i))
                        for i in range(size)]
                    if graph_type == 'GRP':
                        # necessary for batches that contain GRP graph + other graph type w.o. block attribute
                        for graph in next:
                            graph.block = None
                    self.data = self.data + next
            else:
                for size, graph_type in zip(sizes, hybrid_graphs):
                    graph_func, args = get_graph_function(graph_type, opts)
                    next = [torch_utils.from_networkx(graph_func(graph_size, *args)) for _ in
                            range(size)]
                    if graph_type == 'GRP':
                        for graph in next:
                            graph.block = None
                    self.data = self.data + next

        self.size = len(self.data)

        # initialize the node features
        problem = load_problem(opts.problem)
        assert self.data is not None, 'unable to generate dataset from given parameters!'
        for graph in self.data:
            problem.set_inital_features(graph, opts.initialization)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
