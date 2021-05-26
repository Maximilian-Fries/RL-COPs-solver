import os
import time
from datetime import datetime
import math
from torch.utils.data import DataLoader
from utils import move_to, clip_grad_norms, get_inner_model, rollout, log_values,  collate_bl, collate, load_problem
import torch
from model import set_decode_type


class Runner:
    def __init__(self, model, train_data, val_data, baseline, optimizer, lr_scheduler, tb_logger, opts):

        self.model = model

        self.train_data = train_data
        self.val_data = val_data

        self.baseline = baseline

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.tb_logger = tb_logger

        self.opts = opts

        self.collate_fn = collate_bl if opts.baseline == 'rollout' else collate

    def validate(self):
        # Validate
        cost = rollout(self.model, self.val_data, self.opts)
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {} +- {}'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))

        return avg_cost

    def train(self):
        # create file for this training run
        decoding_code = {'local': 'L', 'global': 'G', 'static': 'S', 'zero_context': 'ZC'}.get(self.opts.decoding_type, None)
        assert decoding_code is not None, "The decoding type {} is not supported".format(self.opts.decoding_type)

        if self.opts.encoder_type == 'GCN':
            decoding_code = decoding_code + '-GCN'

        timestamp = datetime.now().strftime("%d_%m_%Y_%H:%M")
        graph_type = self.opts.graph_type if not self.opts.is_hybrid else 'Hybrid'
        model_name = 'P-{}-{}-{}'.format(graph_type, decoding_code, timestamp)

        self.opts.save_dir = os.path.join(
            self.opts.ckpt,
            "{}_{}".format(self.opts.problem, self.opts.graph_nodes),
            model_name
        )
        os.makedirs(self.opts.save_dir)

        wrap_dataset = True

        for epoch in range(self.opts.nepochs):
            print("Start train epoch {}, lr={}".format(epoch, self.optimizer.param_groups[0]['lr']))
            step = epoch * (self.opts.graph_number // self.opts.batch_size)
            start_time = time.time()

            if wrap_dataset:
                # update blvalues in first iteration or if bl model was updated
                training_dataset = self.baseline.wrap_dataset(self.train_data)

                training_dataloader = DataLoader(training_dataset, batch_size=self.opts.batch_size,
                                                 shuffle=True, collate_fn=self.collate_fn)

            if not self.opts.no_tensorboard:
                self.tb_logger.log_value('learnrate_pg0', self.optimizer.param_groups[0]['lr'], step)

            self.model.train()
            set_decode_type(self.model, "sampling")

            # train on batches
            for batch_id, batch in enumerate(training_dataloader):
                self.train_batch(epoch, batch_id, step, batch)
                step += 1
            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {}".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.nepochs - 1:
                torch.save(
                    {
                        'model': get_inner_model(self.model).state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': self.baseline.state_dict()
                    },
                    os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
                )

            avg_reward = self.validate()

            if not self.opts.no_tensorboard:
                self.tb_logger.log_value('val_avg_reward', avg_reward, step)

            start_epoch_callback = time.time()
            wrap_dataset = self.baseline.epoch_callback(self.model, epoch)
            epoch_callback_duration = time.time() - start_epoch_callback
            print("Finished epoch_callback, took {}".format(time.strftime('%H:%M:%S', time.gmtime(epoch_callback_duration))))

            # lr_scheduler should be called at end of epoch
            self.lr_scheduler.step()
            print("_________________________________")

        return 1

    def train_batch(
            self,
            epoch,
            batch_id,
            step,
            batch,
    ):
        x, bl_val = self.baseline.unwrap_batch(batch)
        x = move_to(x, self.opts.device)
        bl_val = move_to(bl_val, self.opts.device) if bl_val is not None else None

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = self.model(x, self.opts.graph_nodes)

        bl_val = self.baseline.eval(x, cost) if bl_val is None else bl_val

        loss = ((cost - bl_val) * log_likelihood).mean()

        # Perform backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(self.optimizer.param_groups, self.opts.max_grad_norm)

        self.optimizer.step()

        # Logging
        if step % int(self.opts.log_step) == 0:
            log_values(cost, grad_norms, epoch, batch_id, step,
                       log_likelihood, loss, self.tb_logger, self.opts)
