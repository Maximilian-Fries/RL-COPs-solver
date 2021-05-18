import torch
from model import Attention
from opts import get_options
from utils import rollout, get_graph_function, load_problem
import numpy as np
import networkx as nx
import torch_geometric.utils as tutils
from os.path import join
import pandas as pd
import os.path

MODEL_DIR = 'Models'


def load_checkpoint(file):
    checkpoint = torch.load(file)
    return checkpoint


def save_procedure(df_q, save, save_file):
    if (save_file is None or not os.path.isfile(save_file)) and save:
        df_q.to_excel(save_file)
    elif save:
        df_model = pd.read_excel(save_file,
                                 index_col=[0, 1])
        if df_q.columns[0] in list(df_model.columns):
            print('Model {} already exists. Existing entries might be overwritten.'.format(df_q.columns[0]))
            df_model.update(df_q)
            df = df_model
        else:
            print('new Model name...')
            df_model[df_q.columns[0]] = np.NaN
            df_model.update(df_q)
            df = df_model
        df.to_excel(save_file)


def read_dimacs_graph(file_path):
    """
    Parse .col file and return graph object
    """
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c'):  # graph description
                print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            elif line.startswith('p'):
                p, name, vertices_num, edges_num = line.split()
                print('{0} {1} {2}'.format(name, vertices_num, edges_num))
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                edges.append((v1, v2))
            else:
                continue
        return nx.Graph(edges)


def model_random_graph_evaluation(model_file, model_name,
                                  problem_name='GC',
                                  initialization='zscore',
                                  n_graphs=1000,
                                  encoder_type='GAT',
                                  opts=None,
                                  save=False, save_file=None, sampling=False, n_samples=200,
                                  throwback=1,
                                  test_types=None,
                                  test_sizes=None,
                                  decoding_type='local'):
    # if not specified, use default opts. opts is used to get the parameters for graph generation.
    if opts is None:
        opts = get_options()

    problem = load_problem(problem_name, opts)
    if test_types is None:
        test_types = ['ER', 'BA', 'WS', 'GRP', 'RG']
    if test_sizes is None:
        # sizes of the graphs on which we test the model
        test_sizes = [20, 50, 100, 1000]

    node_dim = {'local_profile': 5,
                'local_profile_norm': 5,
                'global_profile': 3}.get(initialization, 1)

    opts.initialization = initialization
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() and opts.cuda else "cpu")

    iterables = [test_types, test_sizes]
    index = pd.MultiIndex.from_product(iterables, names=['graph_type', 'graph_size'])
    columns = [model_name]
    df_q = pd.DataFrame(index=index, columns=columns)

    checkpoint_file = join(MODEL_DIR, problem_name,  model_file)
    print(checkpoint_file)
    model = Attention(problem=problem, embed_dim=64, n_encoder_layers=3, n_heads=4, tanh_clipping=10, throwback=throwback,
                      node_dim=node_dim, decoding_type=decoding_type, encoder_type=encoder_type)
    checkpoint = load_checkpoint(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    for graph_type in iterables[0]:

        for n_nodes in test_sizes:
            opts.graph_nodes = n_nodes
            graphf, args = get_graph_function(graph_type, opts)
            assert graphf is not None

            data = [tutils.from_networkx(graphf(n_nodes, *args, seed=i + 1873221)) for i in range(n_graphs)]
            for graph in data:
                problem.set_inital_features(graph, initialization=opts.initialization)
            opts.graph_nodes = n_nodes

            if not sampling:
                model_costs = rollout(model, data, opts).view(-1).data.numpy()
            else:
                solutions = np.zeros((n_samples, n_graphs))
                for i in range(n_samples):
                    solutions[i] = rollout(model, data, opts, mode='sampling').view(-1).data.numpy()
                    if i % 50 == 0:
                        print('   sampled {} solutions'.format(i))
                model_costs = np.min(solutions, axis=0)

            model_cost = np.nanmean(model_costs)
            df_q.loc[graph_type, n_nodes][model_name] = model_cost
            print("finished model evaluation on {} with {} nodes: {}".format(graph_type, n_nodes, model_cost))
        print("_________")

    save_procedure(df_q=df_q, save=save, save_file=save_file)


# Example: Evaluate all GAT models with local decoding, trained on seed 2, for the GC problem

for model in ['Hybrid2', 'ER', 'BA', 'WS', 'GRP']:
    model_random_graph_evaluation('GAT-Experiments/Seed2/P-{}-L-2.pt'.format(model), 'P-{}-L'.format(model),
                                  initialization='zscore',
                                  save=False, save_file=None,
                                  problem_name='GC', test_types=['BA', 'ER', 'GRP', 'S-ER', 'WS', 'S-GRP', 'RG'],
                                  encoder_type='GAT', decoding_type='local', test_sizes=[20,50])
