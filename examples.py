from evaluate import model_random_graph_evaluation, model_dataset_evaluation

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
