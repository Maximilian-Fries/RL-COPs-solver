import torch
from model import Attention
from opts import get_options
from utils import rollout, get_graph_function, load_problem, move_to
import numpy as np
import torch_geometric.utils as tutils
from os.path import join
import pandas as pd
import os.path
from model import set_decode_type

MODEL_DIR = 'Models'


def load_checkpoint(file):
    checkpoint = torch.load(file)
    return checkpoint


'''
Saves the results from df_q in an excel file or updates an existing table.
'''
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
            df_model[df_q.columns[0]] = np.NaN
            df_model.update(df_q)
            df = df_model
        df.to_excel(save_file)


'''
Evaluates a model on a given dataset
'''
def model_dataset_evaluation(model_file, model_name, dataset=None,
                             problem_name='GC',
                             initialization='zscore',
                             encoder_type='GAT',
                             opts=None,
                             save=False, save_file=None, sampling=False, n_samples=200,
                             n_heads=4,
                             throwback=1,
                             decoding_type='local'):
    # if not specified, use default opts. opts is used to get the parameters for graph generation.
    if opts is None:
        opts = get_options()

    problem = load_problem(problem_name, opts)

    node_dim = {'local_profile': 5,
                'local_profile_norm': 5,
                'global_profile': 3}.get(initialization, 1)

    opts.initialization = initialization
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() and opts.cuda else "cpu")

    columns = [model_name]
    df_q = pd.DataFrame(index=list(range(len(dataset))), columns=columns)

    checkpoint_file = join(MODEL_DIR, problem_name, model_file)
    print('load checkpoint: ', checkpoint_file)
    model = Attention(problem=problem, embed_dim=n_heads * 16, n_encoder_layers=3, n_heads=n_heads, tanh_clipping=10,
                      throwback=throwback, node_dim=node_dim, decoding_type=decoding_type, encoder_type=encoder_type)
    checkpoint = load_checkpoint(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    for index, graph in enumerate(dataset):
        graph = tutils.from_networkx(graph)
        opts.graph_nodes = graph.num_nodes
        problem.set_inital_features(graph, initialization=opts.initialization)

        if not sampling:
            set_decode_type(model, 'greedy')
            model.eval()
            with torch.no_grad():
                cost, _ = model(move_to(graph, opts.device), opts.graph_nodes)
            cost = cost.numpy().item()
        else:
            solutions = np.zeros(n_samples)
            for i in range(n_samples):
                solutions[i] = rollout(model, [graph], opts, mode='sampling').view(-1).data.numpy().item()
                if i % 50 == 0:
                    print('   sampled {} solutions'.format(i))
            cost = np.min(solutions, axis=0)

        df_q.loc[index][model_name] = cost
        print("finished model evaluation on graph {} with {} nodes: {}".format(index, opts.graph_nodes, cost))
    print("_________")

    save_procedure(df_q=df_q, save=save, save_file=save_file)


'''
Evaluates a model on synthetic graphs
'''
def model_random_graph_evaluation(model_file, model_name,
                                  problem_name='GC',
                                  initialization='zscore',
                                  n_graphs=1000,
                                  encoder_type='GAT',
                                  opts=None,
                                  save=False, save_file=None, sampling=False, n_samples=200,
                                  n_heads=4,
                                  throwback=1,
                                  test_types=None,
                                  test_sizes=None,
                                  decoding_type='local'):
    # if not specified, use default opts. opts is used to get the parameters for graph generation.
    if opts is None:
        opts = get_options()

    problem = load_problem(problem_name, opts)
    if test_types is None:
        test_types = ['ER', 'BA', 'WS', 'GRP', 'RG', 'S-ER', 'S-GRP']
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

    checkpoint_file = join(MODEL_DIR, problem_name, model_file)
    print('load checkpoint: ', checkpoint_file)
    model = Attention(problem=problem, embed_dim=n_heads * 16, n_encoder_layers=3, n_heads=n_heads, tanh_clipping=10,
                      throwback=throwback, node_dim=node_dim, decoding_type=decoding_type, encoder_type=encoder_type)
    checkpoint = load_checkpoint(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    for graph_type in test_types:

        for n_nodes in test_sizes:
            opts.graph_nodes = n_nodes
            graphf, args = get_graph_function(graph_type, opts)
            assert graphf is not None

            # generate dataset
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


# Template for evaluating on synthetic graphs
evaluation_models = ['Hybrid2', 'ER', 'BA', 'WS', 'GRP']
test_sizes = [20, 50]
problem = 'GC'
encoder = 'GCN'
decoding_type = 'local'
decoding_code = {'local': 'L', 'global': 'G', 'static': 'S', 'zero_context': 'ZC'}.get(decoding_type, None)
seed = 4
assert seed in range(1, 6), 'Pick a seed between 1 and 5!'

for model in evaluation_models:
    model_file = '{}-Experiments/Seed{}/P-{}-{}-GCN-{}.pt'.format(encoder, seed, model, decoding_code, seed) \
        if encoder == 'GCN' else '{}-Experiments/Seed{}/P-{}-{}-{}.pt'.format(encoder, seed, model, decoding_code, seed)
    model_name = 'P-{}-{}-GCN'.format(model, decoding_code) if encoder == 'GCN' else 'P-{}-{}'.format(model, decoding_code)
    model_random_graph_evaluation(
        model_file=model_file,
        model_name=model_name,
        initialization='zscore',
        save=False, save_file=None,
        problem_name=problem, test_types=['BA', 'ER', 'GRP', 'S-ER', 'WS', 'S-GRP', 'RG'],
        encoder_type=encoder, decoding_type=decoding_type, test_sizes=test_sizes)

'''
Example: build a vertex cover on Zachary’s Karate Club graph using a  model
trained with a GCN encoder and local decoding.
'''

from networkx.generators import karate_club_graph

graph =karate_club_graph()
data = [graph]
model_dataset_evaluation(model_file='GCN-Experiments/Seed1/P-BA-L-GCN-1.pt',
                         model_name='P-BA-L',
                         dataset=data,
                         initialization='zscore',
                         save=False, save_file=None,
                         problem_name='MVC',
                         encoder_type='GCN', decoding_type='local')
