import torch
import torch_geometric
from dataclasses import dataclass


class Problem(object):
    NAME = None

    single_solution_set = None

    @staticmethod
    def get_costs(input, pi):
        pass

    @staticmethod
    def make_state(*args, **kwargs):
        raise NotImplementedError("define make_state!")

    @staticmethod
    def set_inital_features(graph):
        raise NotImplementedError("define the initial features!")


@dataclass
class State:
    adj: list
    embeddings: torch.Tensor

    visited: torch.ByteTensor
    sol_ind: torch.IntTensor
    sol_rep: object

    prev_a: torch.IntTensor

    step: torch.Tensor

    def initialize(self, *args):
        raise NotImplementedError("define initialize!")

    # the transition function
    def update(self, selected):
        raise NotImplementedError("define update function!")

    def is_done(self):
        raise NotImplementedError("define termination criterion!")

    def get_cost(self):
        raise NotImplementedError("define a cost function!")