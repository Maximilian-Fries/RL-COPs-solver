import torch
import torch_geometric
from dataclasses import dataclass
from utils import get_adjacency_list
from problems.problem_base import Problem, State
from collections import deque
from torch_geometric.transforms import LocalDegreeProfile


class MVC(Problem):
    NAME = "MVC"

    '''
    Overall not very efficient. 
    Goal: Find better method to check if vertex cover is achieved for individual graphs in the batch.
    '''

    @staticmethod
    def make_state(*args):
        return MVCState.initialize(*args)

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
class MVCState(State):
    update_nodes: list
    order: list
    context_throwback: deque
    decoding_type: str
    input:torch_geometric.data.Data

    @staticmethod
    def initialize(input, embeddings, n_nodes, throwback_size, decoding_type):
        batch_size = input.num_graphs
        prev_a = torch.zeros(batch_size, 1, dtype=torch.int, device=embeddings.device)

        return MVCState(
            adj=get_adjacency_list(input, embeddings.device),  # neighbors of dummy node = []
            input=input,
            embeddings=embeddings,
            visited=torch.zeros(batch_size, 1, n_nodes, device=embeddings.device).type(torch.uint8),
            sol_ind=torch.zeros(batch_size, 1, n_nodes, dtype=torch.int32, device=embeddings.device),
            sol_rep=torch.zeros(batch_size, 1, embeddings.size(-1), device=embeddings.device) - 100000,
            prev_a=prev_a,
            step=torch.zeros(1, device=embeddings.device),
            update_nodes=[],
            order=[],
            context_throwback=deque(maxlen=throwback_size),
            decoding_type=decoding_type
        )

    def update(self, selected):
        self.prev_a = selected[:, None]
        self.order.append(selected)
        self.visited = self.visited.scatter(-1, self.prev_a[:, :, None], 1)
        # self.sol_ind = self.sol_ind.scatter(-1, self.prev_a[:, :, None], 1)
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

        n_nodes = self.visited.size()[-1]
        self.update_nodes = [y for x in [self.adj[i * n_nodes + node] for i, node in enumerate(selected)] for y in x]

    def is_done(self):
        return self.step.item() >= self.visited.size(-1)

    def get_cost(self):

        graphs = self.input.to_data_list()

        def get_order_cost(i, pi):
            graph = graphs[i]
            n = graph.num_nodes
            subset = []
            edge_index = graph.edge_index
            mask = torch.zeros(n, dtype=torch.bool)
            for v in pi:
                mask[subset] = 1
                covered_edges = mask[edge_index[0]] | mask[edge_index[1]]
                if torch.all(covered_edges):
                    return len(subset)
                subset.append(v.item())

        if self.is_done():
                    order = torch.stack(self.order, 1)
                    return torch.tensor([get_order_cost(graph_ind, pi) for graph_ind, pi in enumerate(order)],
                                        dtype=torch.float)


def pp_state(state):
    print("prev_a: ", state.prev_a)
    print("visited: ", state.visited)
    print("sol_ind: ", state.sol_ind)
    print("update_nodes: ", state.update_nodes)
    print("edge_sets: ", state.edge_sets)


