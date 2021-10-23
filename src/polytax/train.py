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
import os
import wandb
import time
import numpy as np
import itertools
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from transformers import (
    CONFIG_MAPPING,
    T5ForConditionalGeneration,
    T5Config,
    set_seed,
)
from speedrun import BaseExperiment, WandBMixin, IOMixin
from polytax.data.utils import get_mixture_or_task
import torch
import torch.distributed as dist
from transformers.optimization import Adafactor
from t5.data.utils import get_default_vocabulary

global xla_found
try:
    import torch_xla
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


class Experiment1(BaseExperiment, WandBMixin, IOMixin):
    WANDB_PROJECT = "polytax"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Experiment1, self).__init__()
        self.auto_setup()

        self.device = xm.xla_device() if xla_found else torch.device("cpu")
        self.local_world_size = int(xm.xrt_world_size()) if xla_found else 1 # Devices per host
        self.global_world_size = int(os.environ.get("WORLD_SIZE", 1)) # Total number of hosts
        self.local_rank = int(xm.get_ordinal()) if xla_found else 0
        self.global_rank = int(os.environ.get("RANK", 0)) * self.local_world_size + self.local_rank

        self.num_shards = self.global_world_size * self.local_world_size
        self.is_multi_host = self.global_world_size > 1
        self.is_master_ordinal = xm.is_master_ordinal() if xla_found else True
        print(f"total shards: {self.num_shards}")
        print(f"global rank: {self.global_rank}")

        self._build()

    def _build(self):
        print(f"{self._config}")
        self.seed = self.get("seed", 1)
        set_seed(self.seed)  # handles random seed setting for everything but XLA
        self.cache_dir = self.get("cache_dir")
        self.model_type = self.get("model_type")
        self.tokenizer = get_default_vocabulary()
        self.model_config = self.get_model_config(self.get("model_config"), self.get("model_name_or_path"))
        self.train_batch_size = int(self.get("per_device_train_batch_size"))
        self.eval_batch_size = int(self.get("per_device_eval_batch_size"))
        self.total_batch_size = self.train_batch_size * self.local_world_size * self.global_world_size
        self.tracker = RateTracker()

        # Build the data loaders
        self.train_loader = self._build_loaders()
        self.model = T5ForConditionalGeneration(self.model_config).to(self.device)
        self.model.train()

        # No need to specify learning rate in Adafactor: https://arxiv.org/pdf/1804.04235.pdf
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

        # one process per host
        if self.is_multi_host and self.is_master_ordinal:
            dist.init_process_group(backend="GLOO") #GLOO for CPU comms, which this is.

        if self.is_master_ordinal and self.get("use_wandb"):
            self.initialize_wandb()

    def _build_loaders(self):
        # determine maximum sequence length to use
        sequence_length = {"inputs": self.get("max_seq_length"), "targets": int(self.get("max_seq_length") / 4)}
        train_loader = get_mixture_or_task(self.get("dataset_name"), self.seed, sequence_length, self.num_shards, self.global_rank, self.train_batch_size)
        if xla_found:
            train_loader = iter(pl.MpDeviceLoader(train_loader, self.device))

        return train_loader

    # def _build_loaders(self):
    #     # determine maximum sequence length to use
    #     sequence_length = {"inputs": self.get("max_seq_length"), "targets": int(self.get("max_seq_length")/4)}
    #     train_dataset = seqio.get_dataset(self.get("dataset_name"), task_feature_lengths=sequence_length, dataset_split="train", use_cached=False, shuffle=True, seed=self.seed, num_epochs=1, feature_converter=seqio.EncDecFeatureConverter(pack=True))
    #
    #     if self.total_shards > 1:
    #         train_dataset = train_dataset.shard(num_shards=self.total_shards, index=self.global_rank)
    #
    #     train_dataset = train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    #     train_dataset = itertools.cycle(train_dataset)
    #     train_loader = iter(data.SeqioWrapperDataset(train_dataset))
    #
    #     if xla_found:
    #         train_loader = iter(pl.MpDeviceLoader(train_loader, self.device))
    #
    #     return train_loader
    #
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

        # if self.is_master_ordinal:
        #     dist.all_reduce(param.grad.cpu() / dist.get_world_size())
        #     xm.all_reduce(xm.REDUCE_SUM, param.grad.to(self.device))
        # else:
        #     zeros = torch.zeros_like(param.grad)
        #     xm.all_reduce(xm.REDUCE_SUM, zeros)


    def decode(self, x, x_hat):
        sample_id = 1
        labels = x['labels'][sample_id].cpu().numpy().tolist()
        input_ids = x['input_ids'][sample_id].cpu().numpy().tolist()
        preds = x_hat.logits[sample_id].argmax(axis=1).cpu().numpy().tolist()

        # Prepare to decode the labels
        extra_id = 32099
        ids_to_replace_in_preds = []
        for idx, label in enumerate(labels):
            if label == -100:
                ids_to_replace_in_preds.append(idx)
                labels[idx] = extra_id
                extra_id -= 1

        extra_id = 32099
        for idx, pred in enumerate(preds):
            if idx in ids_to_replace_in_preds:
                preds[idx] = extra_id
                extra_id -= 1

        input = self.tokenizer.decode(input_ids)
        label = self.tokenizer.decode(labels)
        preds = self.tokenizer.decode(preds)
        return input, label, preds

    def compute_accuracy(self, x, x_hat):
        pred = x_hat.logits.argmax(2)
        correct = pred.eq(x['labels'].view_as(pred)).sum()
        accuracy = 100.0 * correct.item() / x['labels'].nelement()
        return accuracy

    def _log(self, step, tracker, x, x_hat):
        if xla_found:
            print(torch_xla._XLAC.xla_metrics_report())
        loss = x_hat.loss.detach() #.item()

        # Print to console
        print_training_update(self.device, step, loss, tracker.rate(), tracker.global_rate())

        # Return if we don't have wandb
        if not self.get("use_wandb"):
            return

        # Write speeds to wandb
        self.wandb_log(**{"instantaneous it/s": tracker.rate(), "global it/s": tracker.global_rate()})

        # Get a text example and log it
        input, label, pred = self.decode(x, x_hat)
        accuracy = self.compute_accuracy(x, x_hat)
        self.table = wandb.Table(columns=["Step", "Accuracy", "Loss", "Input", "Label", "Predicted"])
        self.table.add_data(step, accuracy, loss, input, label, pred)
        self.wandb_log(**{"accuracy": accuracy, "examples": self.table, "train_loss": loss})

    def log(self, x, x_hat):
        # If XLA is found, then we are on TPU and we should use a closure to increase efficiency
        if xla_found:
            xm.add_step_closure(
                self._log,
                args=(self.step, self.tracker, x, x_hat))

        # Otherwise just call the function to log directly
        else:
            self._log(self.step, self.tracker, x, x_hat)

    def run(self):

        for i in range(self.get("num_train_steps")):

            # Get data
            x = next(self.train_loader)

            # Forward model
            x_hat = self.model(**x)

            # Optimization
            x_hat.loss.backward()
            self.reduce_gradients()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Increment step count
            self.next_step()
            self.tracker.add(1)

            if self.log_now:
                self.log(x, x_hat)

            if self.checkpoint_now:
                self.checkpoint()

    @property
    def log_now(self):
        return self.step % self.get("log_every") == 0 and self.step > 0 and self.is_master_ordinal

    @property
    def evaluate_now(self):
        return self.step % self.get("eval_every") == 0 and self.step > 0

    @property
    def checkpoint_now(self):
        return self.step % self.get("checkpoint_every") == 0 and self.step > 0

    def checkpoint(self):
        checkpoint_path = f"{self.experiment_directory}/Weights/model-{self.step}.pt"
        print(f"checkpointing the model to {checkpoint_path}")

        if xla_found:
            xm.save(self.model.state_dict(), checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)


def _mp_fn(index, args):
    Experiment1().run()

if __name__ == '__main__':
    if xla_found:
        xmp.spawn(_mp_fn, args=({},), nprocs=8)
    else:
        Experiment1().run()
