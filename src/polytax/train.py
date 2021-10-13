#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
import logging
import os
import sys
import wandb
import numpy as np
import itertools
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from transformers import (
    CONFIG_MAPPING,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    set_seed,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from mesh_tensorflow.transformer.dataset import pack_or_pad
import tensorflow as tf
import seqio
from speedrun import (
    BaseExperiment,
    WandBSweepMixin,
    WandBMixin,
    IOMixin,
    SweepRunner,
    register_default_dispatch,
)

from polytax import dataset # DO NOT DELETE -- imports seqio datasets

import torch
import torch.distributed as dist
from transformers.optimization import Adafactor
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

global xla_found
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm
    from torch_xla.core.xla_model import RateTracker
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.test.test_utils as test_utils
    from torch_xla.test.test_utils import print_training_update, print_test_update
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_env_vars as xenv
    xla_found = True
except Exception as e:
    print(f"XLA not found {e} \n")
    xla_found = False
    from utils.tracker import RateTracker, print_training_update, print_test_update


class SeqioWrapperDataset(torch.utils.data.IterableDataset):
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, seqiotask):
        self.seqiotask = seqiotask

    def __iter__(self):
        def process_sample(sample):
            sample = {"input_ids": torch.tensor(sample["inputs"], dtype=torch.long),
                       "labels": torch.tensor(sample["targets"], dtype=torch.long),
                       "decoder_input_ids": torch.tensor(sample["inputs"], dtype=torch.long)}
            return sample
        return map(process_sample, self.seqiotask)

class Experiment1(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(Experiment1, self).__init__()
        self.auto_setup()

        self.device = xm.xla_device() if xla_found else torch.device("cpu")

        self.local_world_size = int(xm.xrt_world_size()) if xla_found else 1 # Devices per host
        self.global_world_size = int(os.environ.get("WORLD_SIZE", 1)) # Total number of hosts
        self.local_rank = int(xm.get_ordinal()) if xla_found else 1
        self.global_rank = int(os.environ.get("RANK", 0)) * self.local_world_size + self.local_rank

        self.total_shards = self.global_world_size * self.local_world_size
        self.is_multi_host = self.global_world_size > 1
        self.is_master_ordinal = xm.is_master_ordinal() if xla_found else True
        print(f"total shards: {self.total_shards}")

        # one process per host
        if self.is_multi_host and self.is_master_ordinal:
            dist.init_process_group(backend="GLOO") #GLOO for CPU comms, which this is.

        WandBMixin.WANDB_PROJECT = "polytax"
        WandBMixin.WANDB_ENTITY = "mweiss10"
        if self.is_master_ordinal and self.get("use_wandb"):
            self.initialize_wandb()
            columns = ["Step", "Ground Truth Text", "Predicted Text"]
            self.table = wandb.Table(columns=columns)

        self._build()

    def _build(self):
        print(f"{self._config}")
        self.seed = self.get("seed", 1)
        set_seed(self.seed)  # handles random seed setting for everything but XLA
        self.cache_dir = self.get("cache_dir")
        self.model_type = self.get("model_type")
        self.tokenizer = dataset.get_default_vocabulary()
        self.model_config = self.get_model_config(self.get("model_config"), self.get("model_name_or_path"))
        self.train_batch_size = int(self.get("per_device_train_batch_size"))
        self.eval_batch_size = int(self.get("per_device_eval_batch_size"))
        self.total_batch_size = self.train_batch_size * self.local_world_size * self.global_world_size
        self.tracker = RateTracker()

        # Build the data loaders
        self.train_loader, self.eval_loader = self._build_loaders()
        self.model = T5ForConditionalGeneration(self.model_config).to(self.device)

        #self.model = MpModelWrapper(self.model)
        # No need to specify Learning Rate in Adafactor: https://arxiv.org/pdf/1804.04235.pdf
        self.optimizer = Adafactor(
            self.model.parameters(),
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )


    def _build_loaders(self):
        # determine maximum sequence length to use
        sequence_length = {"inputs": self.get("max_seq_length"), "targets": self.get("max_seq_length"), "decoder_input_ids": self.get("max_seq_length")}
        self.task = seqio.get_mixture_or_task(self.get("dataset_name"))
        eos_keys = set(k for k, f in self.task.output_features.items() if f.add_eos)
        train_dataset = self.task.get_dataset(sequence_length=sequence_length, split="train", use_cached=False, shuffle=True, seed=self.seed, num_epochs=1)
        if self.total_shards > 1:
            train_dataset = train_dataset.shard(num_shards=self.total_shards, index=self.global_rank)
        train_dataset = pack_or_pad(train_dataset, sequence_length, feature_keys=self.task.output_features,
                                    ensure_eos=eos_keys, pack=True)
        train_dataset = train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        train_dataset = itertools.cycle(train_dataset)
        train_loader = iter(SeqioWrapperDataset(train_dataset))

        try:
            eval_dataset = self.task.get_dataset(sequence_length=sequence_length, split="validation", use_cached=False,
                                                  shuffle=True, seed=self.seed, num_epochs=1)
        except Exception:
            print("no validation set")
            eval_dataset = self.task.get_dataset(sequence_length=sequence_length, split="train[:90%]", use_cached=False,
                                                                      shuffle=True, seed=self.seed, num_epochs=1)
            if self.total_shards > 1:
                eval_dataset = eval_dataset.shard(num_shards=self.total_shards, index=self.global_rank)
        eval_dataset = pack_or_pad(eval_dataset, sequence_length, feature_keys=self.task.output_features,
                                    ensure_eos=eos_keys, pack=True)
        eval_dataset = eval_dataset.batch(self.eval_batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        eval_dataset = itertools.cycle(eval_dataset)
        eval_loader = iter(SeqioWrapperDataset(eval_dataset))

        if xla_found:
            eval_loader = iter(pl.MpDeviceLoader(eval_loader, self.device))
            train_loader = iter(pl.MpDeviceLoader(train_loader, self.device))

        return train_loader, eval_loader

    def get_model_config(self, model_config=None, model_name_or_path=None):
        if model_name_or_path: # if we're loading an already trained model
            model_config = T5Config.from_pretrained(model_name_or_path)
        elif type(model_config) == dict: # pass it in directly from yaml
            model_config['layer_norm_epsilon'] = float(model_config['layer_norm_epsilon'])
            model_config = T5Config.from_dict(model_config, cache_dir=self.cache_dir, vocab_size=self.tokenizer.vocab_size)
        elif model_config:
            model_config = T5Config.from_pretrained(
                model_config, cache_dir=self.cache_dir, vocab_size=self.tokenizer.vocab_size
            )
        else:
            model_config = CONFIG_MAPPING[self.model_type]()
        return model_config

    def reduce_gradients(self):
        # AllReduce the model gradients so we can step the global gradient
        if not xla_found:
            return
        
        xm.reduce_gradients(self.optimizer)

        if not self.is_multi_host:
            return

        if self.is_master_ordinal:
            dist.all_reduce(param.grad.cpu() / dist.get_world_size())
            xm.all_reduce(xm.REDUCE_SUM, param.grad.to(self.device))
        else:
            zeros = torch.zeros_like(param.grad)
            xm.all_reduce(xm.REDUCE_SUM, zeros)

    def decode(self, x, x_hat):
        one_sample_logits = x_hat.logits[0, :]
        gt = self.tokenizer.decode(x['input_ids'][0, :].cpu().numpy().tolist())
        pred = self.tokenizer.decode(one_sample_logits.argmax(axis=1).cpu().numpy().tolist())
        return gt, pred

    def compute_accuracy(self, x, x_hat):
        correct = 0
        total_samples = 0
        pred = x_hat.logits.argmax(2)
        correct += pred.eq(x['input_ids'].view_as(pred)).sum()
        total_samples += x['input_ids'].size()[1]

        accuracy = 100.0 * correct.item() / total_samples
        if xla_found:
            accuracy = xm.mesh_reduce('val_accuracy', accuracy, np.mean)
        return accuracy

    def _update_logs(self, step, tracker, x, x_hat, valid_split):
        # Returns if we aren't logging to wandb or we're not the master proc
        if not self.is_master_ordinal or not self.get("use_wandb"):
            return

        # If we're logging the valid split, then we can decode and look at the accuracy
        if valid_split:
            gt, pred = self.decode(x, x_hat)
            accuracy = self.compute_accuracy(x, x_hat)
            self.table.add_data(step, gt, pred)
            valid_loss = x_hat.loss.item()
            if self.get("use_wandb"):
                self.wandb_log(**{"examples": self.table})
                self.wandb_log(**{"ground_truth": gt, "predicted": pred, "val_accuracy": accuracy, "val_loss": valid_loss})
            print_test_update(self.device, accuracy, step)
        else:
            # Logs to stdout
            print_training_update(self.device, step, x_hat.loss.item(), tracker.rate(), tracker.global_rate())
            self.wandb_log(**{"train_loss": x_hat.loss.item()})
            # Write speeds to wandb, log gradients
            self.wandb_log(**{"instantaneous it/s": tracker.rate(), "global it/s": tracker.global_rate()})
            self.wandb_watch(self.model.shared, x_hat.loss.item(), log_freq=10)
            # self.wandb_watch(self.model.decoder.embed_tokens, x_hat.loss.item(), log_freq=1)
            # self.wandb_watch(self.model.lm_head, x_hat.loss.item(), log_freq=1)


    def log(self, x, x_hat, valid_split):
        # If XLA is found, then we are on TPU and we should use a closure to increase efficiency
        if xla_found:
            xm.add_step_closure(
                self._update_logs,
                args=(self.step, self.tracker, x, x_hat, valid_split))

        # Otherwise just call the function to log directly
        else:
            self._update_logs(self.step, self.tracker, x, x_hat, valid_split)
    @register_default_dispatch
    def trainloop(self):
        # The number of iterations to run is the number of training steps divided by the evaluation rate
        iterations = int(self.get("num_train_steps") / self.get("eval_every"))
        for _ in range(iterations):
            self.model.train()

            # Train for N steps until you need to evaluate
            for _ in range(self.get("eval_every")):
                x = next(self.train_loader)
                self.optimizer.zero_grad()
                x_hat = self.model(**x)
                x_hat.loss.backward()
                self.reduce_gradients()
                xm.optimizer_step(self.optimizer) if xla_found else self.optimizer.step()
                self.next_step()
                self.tracker.add(self.total_batch_size)
                if self.log_now:
                    self.log(x, x_hat, valid_split=False)

                # Eval
                with torch.no_grad():
                    self.model.eval()
                    for _ in range(self.get("num_eval_steps")):
                        x = next(self.eval_loader)
                        x_hat = self.model(**x)
                        valid_loss = x_hat.loss.item()
                        self.log(x, x_hat, valid_split=True)

    @property
    def log_now(self):
        return self.step % self.get("log_every") == 0 and self.step > 0

    @property
    def evaluate_now(self):
        return self.step % self.get("eval_every") == 0 and self.step > 0

    @property
    def save_now(self):
        return self.step % self.get("save_every") == 0 and self.step > 0

    def checkpoint(self):
        if not self.save_now:
            return
        checkpoint_path = f"{self.experiment_directory}/Weights/model-{self.step}.pt"
        if xla_found:
            xm.save(self.model.state_dict(), checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)


def _mp_fn(index, args):
    Experiment1().run()

class SweepPolytax(SweepRunner, WandBSweepMixin, IOMixin):
    def __init__(self):
        WandBSweepMixin.WANDB_ENTITY = "polytax"
        WandBSweepMixin.WANDB_PROJECT = "mweiss10"
        super(SweepPolytax, self).__init__(Experiment1)

if __name__ == '__main__':
    if "--wandb.sweep" in sys.argv:
        obj = SweepPolytax()
        obj.run()
        #SweepPolytax.run()
    else:
        if xla_found:
            xmp.spawn(_mp_fn, args=({},), nprocs=8)
        else:
            Experiment1().run()
