# RL-COPs-solver
A General Reinforcement Learning Solver for Combinatorial Optimization Problems on Graphs
## Requirements
- Python >= 3.8
- PyTorch >= 1.7
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) >= 1.6
- Networkx
- NumPy
- SciPy
- Pandas
## How to evaluate a model?
The [Models](Models) directory contains a collection of trained models. A model `P-x-i` was trained on graphs of type x with initial features i. The models that are trained with the GCN encoder have a -GCN suffix.
You can evalute a model under different conditions on different synthetic graph types with the `model_random_graph_evaluation` method in `evaluate.py`. The parameters are defined as follows. 
- `model_file`: The model that you want to load, e.g. 'GAT-Experiments/Seed1/P-ER-L-1.pt'.
- `model_name`: The name under which the results of the loaded model will be stored in a dataframe/excel table, e.g. 'P-ER-Z-100th'.
- `problem_name`: Supported problems are 'GC' (default) and 'MVC'.
- `initialization`: Specifies how the features of a graph should be initialized. Possible options are 'zscore' (default), 'min_max', 'degree', 'local_profile', 'local_profile_norm'.
- `n_graphs`: The number of graphs in the test dataset. 
- `opts`: This can be used to specify the parameters for graph generation. If `None`, the default opts are used. 
- `save`: Specify if the results should be saved (default False).
- `save_file`: The path of the file where the results are saved if `save` is True. The results are stored in an Excel table. For reading/writing Excel files, install the openpyxl library (`$ pip install openpyxl`).
- `sampling`: If set to True, the model samples multiple solutions per graph and selects the best one. 
- `n_samples`: The number of samples per graph if `sampling` is set to True. 
- `n_head`: The number of attention heads in the GAT encoder. The default is 4. You can load models for the GC problem with 1, 2 or 8 attention heads from the [Dimension-Experiments](Models/GC/Dimension-Experiments) directory. 
- `throwback`: The number of previously selected nodes that are considered for the context embedding during decoding. The default is 1. You can load models for the GC problem with 2, 3 or 4 nodes in the context from the [Context](Models/GC/Context) directory. 
- `test_types`: The graph distributions on which the models are tested. Possible options are 'BA', 'ER', 'WS', 'GRP', 'S-ER', 'S-GRP' and 'Tree'. 
Further types can be added in `get_graph_function` in utils.py.
- `test_sizes`: The sizes of graphs on which we want to test for each of the types in `test_types`. Default sizes are 20, 50, 100 and 1000.
- `decoding_type`: Specifies the decoding type. Possible options are 'local' (default), 'global', 'static' and 'zero_context'. You can load zero_context models for the GC problem from the [Zero-Context](Models/GC/Zero-Context) directory.

Alternatively, you can also test the models on an arbitrary dataset using the `model_dataset_evaluation` method in `evaluate.py`. Pass the graphs as a list of networkx graphs in the `dataset` parameter. The `evaluate.py` file also contains two exmaples how these functions can be used. 
