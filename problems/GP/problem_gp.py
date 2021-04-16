import torch
import torch_geometric
import itertools
from dataclasses import dataclass
from utils import get_adjacency_list
from problems.problem_base import Problem, State
from collections import deque
from torch_geometric.transforms import LocalDegreeProfile
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators import barabasi_albert_graph


class GP(Problem):
    NAME = "GP"

    @staticmethod
    def make_state(*args):
        return GPState.initialize(*args)

    @staticmethod
    def set_inital_features(graph, initialization='local_profile'):
        degrees = torch_geometric.utils.degree(graph.edge_index[1], graph.num_nodes).view(graph.num_nodes, 1)
        if initialization == 'zscore':
            std = torch.std(degrees)
            if torch.std(degrees) == 0:
                graph.x = degrees
                return
            graph.x = (degrees - torch.mean(degrees)) / std
        elif initialization == 'min_max':
            deg_min = torch.min(degrees)
            deg_max = torch.max(degrees)
            if deg_min == deg_max:
                graph.x = torch.ones_like(degrees) - 0.5
                return
            graph.x = (degrees - deg_min) / (deg_max - deg_min)
        elif initialization == 'local_profile':
            LocalDegreeProfile()(graph)
        elif initialization == 'local_profile_norm':
            LocalDegreeProfile()(graph)
            for i in range(5):
                std = torch.std(graph.x[:, i])
                if std == 0:
                    graph.x[:, i] = 0
                    continue
                mean = torch.mean(graph.x[:, i])
                graph.x[:, i] = (graph.x[:, i] - mean) / std
        else:
            graph.x = degrees


@dataclass
class GPState(State):
    update_nodes: list
    context_throwback: deque
    n_nodes: int

    @staticmethod
    def initialize(input, embeddings, n_nodes, throwback_size):
        batch_size = input.num_graphs
        return GPState(
            adj=get_adjacency_list(input, embeddings.device),
            embeddings=embeddings,
            visited=torch.zeros(batch_size, 1, n_nodes, device=embeddings.device).type(torch.uint8),
            sol_ind=torch.zeros(batch_size, 1, n_nodes, dtype=torch.int32, device=embeddings.device),
            sol_rep=torch.zeros(batch_size, 1, embeddings.size(-1), device=embeddings.device) - 100000,
            prev_a=torch.zeros(batch_size, 1, dtype=torch.int, device=embeddings.device),
            step=torch.zeros(1, device=embeddings.device),
            update_nodes=[],
            context_throwback=deque(maxlen=throwback_size),
            n_nodes=n_nodes
        )

    def update(self, selected):
        self.prev_a = selected[:, None]
        self.visited = self.visited.scatter(-1, self.prev_a[:, :, None], 1)
        self.sol_ind = self.sol_ind.scatter(-1, self.prev_a[:, :, None], 1)
        self.update_nodes = []
        self.get_update_nodes(selected)
        batch_size = selected.size()[0]
        last_action_embedding = self.embeddings.gather(1,
                                                       self.prev_a[:, None].expand(batch_size, 1,
                                                                                   self.embeddings.size(-1)))
        self.sol_rep = torch.max(self.sol_rep, last_action_embedding)
        self.step = self.step + 1

    def get_update_nodes(self, selected):
        self.update_nodes = [y for x in [self.adj[i * self.n_nodes + node] for i, node in enumerate(selected)] for y in x]

    def is_done(self):
        return self.step >= self.n_nodes // 2

    def get_cost(self):
        def single_graph_cost(graph_ind):
            partitition_mask = self.sol_ind[graph_ind].squeeze() > 0
            incident = [self.adj[graph_ind*self.n_nodes + i] for i in range(self.n_nodes) if partitition_mask[i]]
            incident = [y for x in incident for y in x]  # unfold neighbors into a single list
            return len([u for u in incident if not partitition_mask[u % self.n_nodes]])

        return torch.tensor([single_graph_cost(graph_ind) for graph_ind in range(self.visited.size()[0])], dtype=torch.float)


def pp_state(state):
    print("prev_a: ", state.prev_a)
    print("visited: ", state.visited)
    print("sol_ind: ", state.sol_ind)
    print("update_nodes: ", state.update_nodes)


'''graph_size = 10
ds = [torch_geometric.utils.from_networkx(barabasi_albert_graph(10, m=4, seed=7132+i)) for i in range(3)]

for graph in ds:
    GP.set_inital_features(graph, initialization='local_profile_norm')

#nx_graph = torch_utils.to_networkx(graph, to_undirected=True)
#print("nx_graph: ", nx_graph)


batch_g = torch_geometric.data.Batch.from_data_list([ds[0], ds[1], ds[2]])
print("edge index")
unbatch = batch_g.to_data_list()
for g in unbatch:
    print(g.edge_index)


embeddings = torch.arange(150).view(3, 10, 5)

state = GP().make_state(batch_g, embeddings, 10, 1)
np.random.seed(192)

p1 = np.random.permutation(10)
p2 = np.random.permutation(10)
p3 = np.random.permutation(10)

selections = list(map(list, zip(*[p1,p2,p3])))


for selected in selections:
    if not state.is_done():
        s = torch.tensor(selected)
        state.update(s)
        pp_state(state)
        print("______________________")

print("node orders: ", selections)
print("costs: ",state.get_cost())
graph = torch_geometric.utils.to_networkx(ds[0], to_undirected=True)
pos = nx.spring_layout(graph)
for i in range(graph_size):
    if state.sol_ind[0,0,i]:
        pos[i][0] -= 1
    else:
        pos[i][0] += 1
nx.draw(graph, node_color=state.sol_ind[0].squeeze(), with_labels=True, pos=pos)
plt.show()'''
