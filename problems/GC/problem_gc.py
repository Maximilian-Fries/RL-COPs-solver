import torch
import torch_geometric
import itertools
from dataclasses import dataclass
from utils import get_adjacency_list
from problems.problem_base import Problem, State
from collections import deque
from torch_geometric.transforms import LocalDegreeProfile


class GC(Problem):
    NAME = "GC"

    @staticmethod
    def make_state(*args):
        return GCState.initialize(*args)

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
                graph.x[:, i] = (graph.x[:, i]-mean)/std
        elif initialization == 'global_profile':
            n = graph.num_nodes
            std = torch.ones(n)*torch.std(degrees.squeeze())
            mean = torch.ones(n)*torch.mean(degrees.squeeze())
            graph.x = torch.stack([degrees.squeeze(), mean, std], dim=1)
        else:
            graph.x = degrees


@dataclass
class GCState(State):
    neighbor_colors: dict
    update_nodes: list
    context_throwback: deque
    decoding_type: str

    @staticmethod
    def initialize(input, embeddings, n_nodes, throwback_size, decoding_type):
        batch_size = input.num_graphs
        prev_a = torch.zeros(batch_size, 1, dtype=torch.int, device=embeddings.device)
        return GCState(
            adj=get_adjacency_list(input, embeddings.device),
            embeddings=embeddings,
            visited=torch.zeros(batch_size, 1, n_nodes, device=embeddings.device).type(torch.uint8),
            sol_ind=torch.zeros(batch_size, 1, n_nodes, dtype=torch.int32, device=embeddings.device) - 1,
            sol_rep={k: [] for k in range(batch_size)},
            prev_a=prev_a,
            step=torch.zeros(1, device=embeddings.device),
            neighbor_colors={i: {j: set() for j in range(n_nodes)} for i in range(batch_size)},
            update_nodes=[],
            context_throwback=deque(maxlen=throwback_size),
            decoding_type=decoding_type
        )

    # the transition function
    def update(self, selected):

        self.prev_a = selected[:, None]
        self.visited = self.visited.scatter(-1, self.prev_a[:, :, None], 1)
        if self.decoding_type == 'local':
            self.update_nodes = []
            self.get_update_nodes(selected)

        colors = torch.tensor([self.color_node(i, node) for i, node in enumerate(selected)],
                              device=self.prev_a.device, dtype=torch.int)

        self.sol_ind = self.sol_ind.scatter(-1, self.prev_a[:, :, None], colors[:, None, None]).type(torch.int32)

        # update solution representation
        '''i = 0  # index of the current graph
        for node, color in zip(selected, colors):
            if color >= len(self.sol_rep[i]):
                # assert color == len(self.sol_rep[i])
                # create a new partial solution set
                self.sol_rep[i].append(self.embeddings[i, node.item()])
            else:
                self.sol_rep[i][color.item()] = torch.max(self.sol_rep[i][color.item()],
                                                          self.embeddings[i, node.item()])
            i += 1'''

        self.step = self.step + 1

    def get_update_nodes(self, selected):

        n_nodes = self.visited.size()[-1]
        self.update_nodes = [y for x in [self.adj[i * n_nodes + node] for i, node in enumerate(selected)] for y in x]

    def color_node(self, i, node):
        n_nodes = self.visited.size()[-1]
        nc_current = self.neighbor_colors[i][node.item()]

        # Find the first unused color
        for color in itertools.count():
            if color not in nc_current:
                # update neighbors of node
                for u in self.adj[i * n_nodes + node]:
                    self.neighbor_colors[i][u.item() % n_nodes].add(color)

                if self.decoding_type == 'static' or self.decoding_type == 'zero_context':
                    '''
                    There is no solution representation required in static or zero_context decoding mode.
                    Hence, one can just append a single number for each new color in the graph.
                    '''
                    if color >= len(self.sol_rep[i]):
                        self.sol_rep[i].append(1)

                    return color

                # update solution representation
                if color >= len(self.sol_rep[i]):
                    # assert color == len(self.sol_rep[i])
                    # create a new partial solution set
                    self.sol_rep[i].append(self.embeddings[i, node.item()])
                else:
                    self.sol_rep[i][color] = torch.max(self.sol_rep[i][color], self.embeddings[i, node.item()])

                return color

    def is_done(self):
        return self.step.item() >= self.visited.size(-1)

    def get_cost(self):
        if self.is_done():
            return torch.tensor([len(self.sol_rep[k]) for k in range(self.visited.shape[0])], dtype=torch.float,
                                device=self.prev_a.device)


def pp_state(state):
    print("prev_a: ", state.prev_a)
    print("visited: ", state.visited)
    print("sol_ind: ", state.sol_ind)
    print('neighbor colors: ', state.neighbor_colors)
    print("update_nodes: ", state.update_nodes)
