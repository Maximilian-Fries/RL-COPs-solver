import argparse


def get_options():
    parser = argparse.ArgumentParser(description='RL running machine')
    parser.add_argument('--problem', default='MVC', help="The problem to solve, currently supports GC and MVC")

    parser.add_argument('--graph_type', metavar='GRAPH', default='ER', help='Type of graph to optimize')
    parser.add_argument('--is_hybrid', dest='is_hybrid', action='store_true')
    parser.add_argument('--no-is_hybrid', dest='is_hybrid', action='store_false')
    parser.set_defaults(is_hybrid=False)
    parser.add_argument('--hybrid_graphs', default=[], nargs='*', type=str)

    parser.add_argument('--encoder_type', default='GAT', type=str, help="GAT or GCN")

    parser.add_argument('--gp_state_version', default=1, type=int)

    parser.add_argument('--graph_nodes', type=int, default=50, help='number of nodes in the graphs')
    parser.add_argument('--val_size', type=int, default=1000, help='number of graphs in validation dataset')
    parser.add_argument('--graph_number', type=int, default=2048, help='number of graphs to generate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--nepochs', type=int, metavar='n', default='5', help='number of epochs')
    parser.add_argument('--throwback', type=int, default=1, help='number of previous nodes to include in context')
    parser.add_argument('--seed', type=int, default=1313, help='Random seed to use')
    parser.add_argument('--initialization', default='zscore')
    parser.add_argument('--decoding_type', default='local')

    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    parser.add_argument('--baseline', default='rollout', help='can use rollout or exponential')

    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of the encoder embeddings')
    parser.add_argument('--n_encoder_layers', type=int, default=3, help='convolution depth for graph embedding')
    parser.add_argument('--n_heads', type=int, default=4)

    # Graph parameters
    parser.add_argument('--m', type=int, default=4, help='degree for barabasi-albert graphs')
    parser.add_argument('--p', type=float, default=.15, help='edge probability for erdos-renyi graphs')
    parser.add_argument('--er_d', type=float, default=7.5, help='expected average degree for ER graphs')
    parser.add_argument('--k', type=int, default=5, help='degree for watts-strogatz graphs')
    parser.add_argument('--p_ws', type=float, default=.1, help='edge probability for watts-strogatz graphs')
    parser.add_argument('--s', type=float, default=6, help='mean cluster size of grp')
    parser.add_argument('--v', type=float, default=10, help='varianve of grp')
    parser.add_argument('--p_in', type=float, default=.9, help='Probabilty of intra cluster connection')
    parser.add_argument('--p_out', type=float, default=.05, help='Probabilty of outer cluster connection')

    parser.add_argument('--ba_ratio', type=float, default=.5, help="ratio of BA graphs in hybrid training")

    parser.add_argument('--train_file', default='None', type=str,
                        help='loads data from pickle file if not None')
    parser.add_argument('--test_file', default='None', type=str,
                        help='loads data from pickle file for testing if not None')
    parser.add_argument('--ckpt', default='checkpoints', type=str,
                        help='directory to save results in, makes a new one using current time if None')
    parser.add_argument('--load_model', default=False, action='store_true',
                        help='when not none loads model from checkpoint')
    parser.add_argument('--checkpoint_epochs', type=int, default=2),
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps'),

    parser.add_argument('--test_only', default=False, action='store_true',
                        help='tests a model without training, load_model must be specified')

    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_false', help='Disable progress bar')

    opts = parser.parse_args()

    return opts
