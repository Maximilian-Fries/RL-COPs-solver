import torch
import math
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from model import set_decode_type
from torch_geometric.data import Batch
import torch_geometric.utils as tutils
import numpy as np
from networkx.generators import random_tree, erdos_renyi_graph, watts_strogatz_graph, barabasi_albert_graph, \
    gaussian_random_partition_graph


def collate(graphs):
    return Batch.from_data_list(graphs)


def collate_bl(batch):
    graphs, bl_vals = map(list, zip(*batch))
    batch_graph = Batch.from_data_list(graphs)
    bl_vals = torch.tensor(bl_vals)
    return batch_graph, bl_vals


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    if isinstance(var, list):
        return [move_to(v, device) for v in var]
    return var.to(device)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def rollout(model, dataset, opts, mode='greedy'):
    set_decode_type(model, mode)
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), opts.graph_nodes)
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in DataLoader(dataset, batch_size=opts.eval_batch_size, collate_fn=collate, shuffle=False)
    ], 0)


def get_graph_function(graph_type, opts):
    supported_types = {
        'Tree': [random_tree, ()],
        'ER': [erdos_renyi_graph, [opts.p]],
        'WS': [watts_strogatz_graph, [opts.k, opts.p_ws]],
        'BA': [barabasi_albert_graph, [opts.m]],
        'GRP': [gaussian_random_partition_graph, [opts.s, opts.v, opts.p_in, opts.p_out]]
    }
    graph_params = supported_types.get(graph_type, None)
    assert graph_params is not None, 'The type {} is not supported'.format(graph_type)
    return graph_params


def load_problem(name):
    from problems import GC, MVC, GP
    problem = {
        'GC': GC,
        'MVC': MVC,
        'GP': GP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('clipped grad_norms: {}'.format(grad_norms[0]))
    print('nll: ', -log_likelihood.mean().item())

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)


def get_neighbors(edge_index, node, node_degree=None):

    row, col = edge_index[0], edge_index[1]
    if node_degree is None:
        node_degree = tutils.degree(col)[node]

    start_row_ind = tensor_bs(node, row)
    if start_row_ind == -1:
        return torch.tensor([])
    neighbors = col[start_row_ind:start_row_ind+node_degree]
    return neighbors


def tensor_bs(value, elements):
    left, right = 0, len(elements) - 1

    while left <= right:
        middle = (left + right) // 2

        if elements[middle] == value:
            while middle >= 0 and elements[middle] == value:
                middle -= 1
            return middle+1

        if elements[middle] < value:
            left = middle + 1
        elif elements[middle] > value:
            right = middle - 1

    return -1


def get_adjacency_list(graph, device):
    n = graph.num_nodes

    edges = graph.edge_index[1]
    degs = tutils.degree(graph.edge_index[1], graph.num_nodes).view(graph.num_nodes, 1).int().numpy()
    pref_deg = np.zeros(n+1, dtype=np.int)
    for i in range(1, n+1):
        pref_deg[i] = pref_deg[i-1] + degs[i-1]

    adj = [edges[pref_deg[i]:pref_deg[i+1]].numpy() for i in range(n)]

    return adj