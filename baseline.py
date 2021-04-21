import torch
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from dataset import MyDataset
from utils import rollout, load_problem


class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()
        return self.v

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class RolloutBaseline(Baseline):

    def __init__(self, model, opts, epoch=0):
        super(Baseline, self).__init__()

        self.opts = opts
        self.problem = load_problem(opts.problem, opts)

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = MyDataset(num_samples=self.opts.val_size, graph_size=self.opts.graph_nodes,
                                     graph_type=self.opts.graph_type, is_hybrid=self.opts.is_hybrid,
                                     hybrid_graphs=self.opts.hybrid_graphs, opts=self.opts, validation=True)
        else:
            self.dataset = dataset

        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()

        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        return batch

    def eval(self, x, c):
        with torch.no_grad():
            v, _ = self.model(x)

        return v, 0

    def epoch_callback(self, model, epoch):
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)
                return True

        return False

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': [],
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        load_model.load_state_dict(state_dict['model'].state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset

        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return self.dataset[item], self.baseline[item]

    def __len__(self):
        return len(self.dataset)
