from torch.utils.data import Dataset
import torch
import os
import pickle
from networkx import erdos_renyi_graph, barabasi_albert_graph, watts_strogatz_graph, random_tree, gaussian_random_partition_graph
import torch_geometric.utils as torch_utils
from torch_geometric.data import Data


class MyDataset(Dataset):

    def __init__(self, filename=None, num_samples=1000, graph_size=20, graph_type='erdos_renyi', opts=None, validation=False):
        super(MyDataset, self).__init__()

        if filename is not None:
            print(filename)
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                self.data = [pickle.load(f) for _ in range(num_samples)]

        else:
            # assert graph_type in ['erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'tree', 'hybrid', 'hybrid2']
            # create new graphs
            if graph_type == 'erdos_renyi':
                if not validation:
                    self.data = [
                        torch_utils.from_networkx(erdos_renyi_graph(n=graph_size, p=opts.p, seed=opts.seed + i))
                        for i in range(num_samples)]
                else:
                    self.data = [torch_utils.from_networkx(erdos_renyi_graph(n=graph_size, p=opts.p))
                 for _ in range(num_samples)]
            elif graph_type == 'barabasi_albert':
                if not validation:
                    self.data = [
                        torch_utils.from_networkx(barabasi_albert_graph(n=graph_size, m=opts.m, seed=opts.seed + i))
                        for i in range(num_samples)]
                else:
                    self.data = [
                        torch_utils.from_networkx(barabasi_albert_graph(n=graph_size, m=opts.m))
                        for _ in range(num_samples)]
            elif graph_type == 'watts_strogatz':
                if not validation:
                    self.data = [torch_utils.from_networkx(
                        watts_strogatz_graph(n=graph_size, k=opts.k, p=opts.p_ws, seed=opts.seed + i))
                                 for i in range(num_samples)]
                else:
                    self.data = [torch_utils.from_networkx(
                        watts_strogatz_graph(n=graph_size, k=opts.k, p=opts.p_ws))
                                 for _ in range(num_samples)]
            elif graph_type =='grp':
               if not validation:
                   self.data = [torch_utils.from_networkx(
                       gaussian_random_partition_graph(n=graph_size, s=opts.s, v=opts.v, p_in=opts.p_in,
                                                       p_out=opts.p_out, seed=opts.seed + i))
                       for i in range(num_samples)]
               else:
                   self.data = [torch_utils.from_networkx(
                       gaussian_random_partition_graph(n=graph_size, s=opts.s, v=opts.v, p_in=opts.p_in,
                                                       p_out=opts.p_out))
                       for _ in range(num_samples)]
            elif graph_type == 'tree':
                self.data = [torch_utils.from_networkx(random_tree(n=graph_size))
                             for _ in range(num_samples)]
            elif graph_type == 'hybrid':
                if not validation:
                    self.data = [torch_utils.from_networkx(
                        barabasi_albert_graph(n=graph_size, m=opts.m, seed=opts.seed + i))
                                 for i in range(int(num_samples * opts.ba_ratio))] + \
                                [torch_utils.from_networkx(
                                    watts_strogatz_graph(n=graph_size, k=opts.k, p=opts.p_ws, seed=opts.seed + i)) for
                                 i in range(int(num_samples * (1 - opts.ba_ratio)))]
                else:
                    self.data = [torch_utils.from_networkx(
                        barabasi_albert_graph(n=graph_size, m=opts.m))
                                 for _ in range(int(num_samples * opts.ba_ratio))] + \
                                [torch_utils.from_networkx(
                                    watts_strogatz_graph(n=graph_size, k=opts.k, p=opts.p_ws)) for
                                 _ in range(int(num_samples * (1 - opts.ba_ratio)))]
            elif graph_type == 'hybrid3':
                self.data = [Data(edge_index=barabasi_albert_graph(num_nodes=graph_size, num_edges=opts.m),
                                  x=torch.zeros(graph_size, 1)) for _ in
                             range(int(num_samples*opts.ba_ratio))] + \
                            [Data(edge_index=erdos_renyi_graph(num_nodes=graph_size, edge_prob=opts.p),
                                  x=torch.zeros(graph_size, 1)) for _ in range(int(num_samples*(1-opts.ba_ratio)))]
            elif graph_type == 'hybrid2':
                size1 = size2 = num_samples//3
                size3 = num_samples - 2*size1
                if not validation:
                    self.data = [torch_utils.from_networkx(
                        barabasi_albert_graph(n=graph_size, m=opts.m, seed=opts.seed + i))
                                 for i in range(size1)] + \
                                [torch_utils.from_networkx(
                                    watts_strogatz_graph(n=graph_size, k=opts.k, p=opts.p_ws, seed=opts.seed + i)) for
                                 i in range(size2)] + \
                                [torch_utils.from_networkx(
                                    erdos_renyi_graph(n=graph_size, p=opts.p, seed=opts.seed + i))
                                 for i in range(size3)]
                else:
                    self.data = [torch_utils.from_networkx(
                        barabasi_albert_graph(n=graph_size, m=opts.m))
                                 for _ in range(size1)] + \
                                [torch_utils.from_networkx(
                                    watts_strogatz_graph(n=graph_size, k=opts.k, p=opts.p_ws)) for
                                 _ in range(size2)] + \
                                [torch_utils.from_networkx(
                                    erdos_renyi_graph(n=graph_size, p=opts.p))
                                 for _ in range(size3)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
