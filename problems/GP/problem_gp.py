import torch
import torch_geometric
from dataclasses import dataclass
from utils import get_adjacency_list
from problems.problem_base import Problem, State
from collections import deque
from torch_geometric.transforms import LocalDegreeProfile
import random


class GP(Problem):
    NAME = "GP"

    def __init__(self, state_version=2):
        super(GP, self).__init__()
        self.state_version = state_version

    def make_state(self, *args):
        if self.state_version == 1:
            return GPState.initialize(*args)
        else:
            return GPState_2.initialize(*args)

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
    decoding_type: str

    @staticmethod
    def initialize(input, embeddings, n_nodes, throwback_size, decoding_type):
        batch_size = input.num_graphs if isinstance(input, torch_geometric.data.Batch) else 1
        return GPState(
            adj=get_adjacency_list(input),
            embeddings=embeddings,
            visited=torch.zeros(batch_size, 1, n_nodes, device=embeddings.device).type(torch.uint8),
            sol_ind=torch.zeros(batch_size, 1, n_nodes, dtype=torch.int32, device=embeddings.device),
            sol_rep=torch.zeros(batch_size, 1, embeddings.size(-1), device=embeddings.device) - 100000,
            prev_a=torch.zeros(batch_size, 1, dtype=torch.int, device=embeddings.device),
            step=torch.zeros(1, device=embeddings.device),
            update_nodes=[],
            context_throwback=deque(maxlen=throwback_size),
            n_nodes=n_nodes,
            decoding_type=decoding_type
        )

    def update(self, selected):
        self.prev_a = selected[:, None]
        self.visited = self.visited.scatter(-1, self.prev_a[:, :, None], 1)
        self.sol_ind = self.sol_ind.scatter(-1, self.prev_a[:, :, None], 1)
        if self.decoding_type == 'local':
            self.update_nodes = []
            self.get_update_nodes(selected)
        if self.decoding_type != 'static' and self.decoding_type != 'zero_context':
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
            incident = [self.adj[graph_ind * self.n_nodes + i] for i in range(self.n_nodes) if partitition_mask[i]]
            incident = [y for x in incident for y in x]  # unfold neighbors into a single list
            return len([u for u in incident if not partitition_mask[u % self.n_nodes]])

        return torch.tensor([single_graph_cost(graph_ind) for graph_ind in range(self.visited.size()[0])],
                            dtype=torch.float)


@dataclass
class GPState_2(State):
    update_nodes: list
    context_throwback: deque
    n_nodes: int
    partition_sizes: dict
    partition_neighbor_count: dict
    '''
    counts the assigned partitions for each's node neighborhood. If a node has more neighbors in one of the partitions 
    then adding the node to the same, will add less edges to the cut.
    '''

    @staticmethod
    def initialize(input, embeddings, n_nodes, throwback_size):

        batch_size = input.num_graphs if isinstance(input, torch_geometric.data.Batch) else 1
        return GPState_2(
            adj=get_adjacency_list(input, embeddings.device),
            embeddings=embeddings,
            visited=torch.zeros(batch_size, 1, n_nodes, device=embeddings.device).type(torch.uint8),
            sol_ind=torch.zeros(batch_size, 1, n_nodes, dtype=torch.int32, device=embeddings.device) - 1,
            sol_rep={k: [torch.zeros(embeddings.size(-1))-10000, torch.zeros(embeddings.size(-1))-10000] for k in
                     range(batch_size)},
            prev_a=torch.zeros(batch_size, 1, dtype=torch.int, device=embeddings.device),
            step=torch.zeros(1, device=embeddings.device),
            update_nodes=[],
            context_throwback=deque(maxlen=throwback_size),
            n_nodes=n_nodes,
            partition_sizes={k: [0, 0, 0] for k in range(batch_size)},
            partition_neighbor_count={k: {v: [0, 0] for v in range(n_nodes)} for k in range(batch_size)}
        )

    def update(self, selected):
        self.prev_a = selected[:, None]
        self.visited = self.visited.scatter(-1, self.prev_a[:, :, None], 1)
        # self.sol_ind = self.sol_ind.scatter(-1, self.prev_a[:, :, None], 1)
        self.update_nodes = []
        self.get_update_nodes(selected)

        assigned_partitions = torch.tensor([self.get_partition(i, node) for i, node in enumerate(selected)],
                                           device=self.prev_a.device, dtype=torch.int)
        self.sol_ind = self.sol_ind.scatter(-1, self.prev_a[:, :, None],
                                            assigned_partitions[:, None, None]).type(torch.int32)

        self.step = self.step + 1

    def get_partition(self, i, node):
        n_nodes = self.visited.size()[-1]
        neighbors = self.adj[i * n_nodes + node]
        P1_size, P2_size, cut_size = self.partition_sizes[i]
        s1, s2 = self.partition_neighbor_count[i][node.item()]

        # case1: one partition (P1 or P2) contains  n/2 nodes -> add node to the other partition
        # if a node has the same number of neighbors in both partitions, it is assigned to the one with less nodes
        if P1_size >= (n_nodes / 2) or P2_size >= (n_nodes / 2) or s1 == s2:
            partition_bool = P1_size > P2_size  # if True (1) add to P2 (1)
            partition_bool = bool(random.getrandbits(1)) if P1_size==P2_size else partition_bool
            self.partition_sizes[i][int(partition_bool)] += 1
            self.partition_sizes[i][2] = cut_size + s1 if partition_bool else cut_size + s2
            self.sol_rep[i][int(partition_bool)] = torch.max(self.sol_rep[i][int(partition_bool)],
                                                            self.embeddings[i, node])
            for u in neighbors:
                self.partition_neighbor_count[i][u % n_nodes][int(partition_bool)] += 1

            return 0 + partition_bool
        if s1 > s2:
            # case2: s1 > s2 -> add node to P1
            for u in neighbors:
                self.partition_neighbor_count[i][u % n_nodes][0] += 1
            self.partition_sizes[i][0] += 1
            self.partition_sizes[i][2] = cut_size + s2
            self.sol_rep[i][0] = torch.max(self.sol_rep[i][0], self.embeddings[i, node])
            return 0
        elif s2 > s1:
            #  case3: s1 > s2 -> add node to P2
            for u in neighbors:
                self.partition_neighbor_count[i][u % n_nodes][1] += 1
            self.partition_sizes[i][1] += 1
            self.partition_sizes[i][2] = cut_size + s1
            self.sol_rep[i][1] = torch.max(self.sol_rep[i][1], self.embeddings[i, node])
            return 1

    def get_update_nodes(self, selected):
        self.update_nodes = [y for x in [self.adj[i * self.n_nodes + node] for i, node in enumerate(selected)]
                             for y in x]

    def is_done(self):
        return self.step >= self.n_nodes

    def get_cost(self):
        return torch.tensor([self.partition_sizes[graph_ind][2] for graph_ind in range(self.visited.size()[0])],
                            dtype=torch.float)


def pp_state(state):
    print("prev_a: ", state.prev_a)
    print("visited: ", state.visited)
    print("sol_ind: ", state.sol_ind)
    print("update_nodes: ", state.update_nodes)
