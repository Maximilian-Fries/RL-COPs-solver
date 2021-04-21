from opts import get_options
import torch
from datetime import datetime
import os
from dataset import MyDataset
from runner import Runner
from baseline import RolloutBaseline, ExponentialBaseline
from model import Attention
from tensorboard_logger import Logger as TbLogger
import pprint as pp
from utils import load_problem
import time
from timeit import default_timer as timer


def main(opts):
    pp.pprint(vars(opts))

    if (opts.ckpt is not None) and (not os.path.exists(opts.ckpt)):
        os.mkdir(opts.ckpt)

    if opts.train_file == 'None' or not os.path.isfile(opts.train_file):
        opts.train_file = None

    if opts.test_file == 'None' or not os.path.isfile(opts.test_file):
        opts.test_file = None

    start_dataset_creation = timer()
    training_graphs = MyDataset(filename=opts.train_file, num_samples=opts.graph_number, graph_size=opts.graph_nodes,
                                is_hybrid=opts.is_hybrid, hybrid_graphs=opts.hybrid_graphs, graph_type=opts.graph_type,
                                opts=opts)

    test_graphs = MyDataset(filename=opts.test_file, num_samples=opts.val_size, graph_size=opts.graph_nodes,
                            is_hybrid=opts.is_hybrid, hybrid_graphs=opts.hybrid_graphs,
                            graph_type=opts.graph_type, opts=opts, validation=True)

    creation_duration = timer() - start_dataset_creation
    print("creation duration: ", time.strftime('%H:%M:%S', time.gmtime(creation_duration)))

    torch.manual_seed(opts.seed)  # set seed

    # set dimension of initial features
    opts.node_dim = test_graphs[0].x.shape[-1]

    opts.device = torch.device("cuda:0" if torch.cuda.is_available() and opts.cuda else "cpu")  # set the device
    print("device: ", opts.device)

    problem = load_problem(opts.problem, opts)

    model = Attention(problem=problem,
                      embed_dim=opts.embed_dim,
                      n_encoder_layers=opts.n_encoder_layers,
                      n_heads=opts.n_heads,
                      node_dim=opts.node_dim,
                      throwback=opts.throwback).to(opts.device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("use DataParallel")

    baseline = RolloutBaseline(model=model, opts=opts) if opts.baseline == 'rollout' else \
        ExponentialBaseline(beta=opts.exp_beta)

    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    tb_logger = None
    if not opts.no_tensorboard:
        timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M")
        tb_logger = TbLogger(os.path.join('logs', "{}_{}_".format(opts.problem, opts.graph_nodes), timestamp))

    runner = Runner(
        model=model,
        train_data=training_graphs,
        val_data=test_graphs,
        baseline=baseline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tb_logger=tb_logger,
        opts=opts,
    )

    if opts.test_only:
        runner.validate()
    else:
        runner.train()


if __name__ == '__main__':
    main(get_options())
