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
import time
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
from speedrun import BaseExperiment, WandBMixin, IOMixin
from polytax import dataset # DO NOT DELETE -- imports seqio datasets

import torch
import torch.distributed as dist
from transformers.optimization import Adafactor

global xla_found
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.test.test_utils as test_utils
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_env_vars as xenv
    xla_found = True
except Exception as e:
    print(f"XLA not found {e} \n")
    xla_found = False

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )
    logger = logging.getLogger(__name__)
    return logger


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
      pass

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
      samples = {"input_ids": torch.zeros((1, 16), dtype=torch.long), "labels": torch.zeros((1, 16), dtype=torch.long), "decoder_input_ids": torch.zeros((1, 16), dtype=torch.long)}
      return samples

class Experiment1(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(Experiment1, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_PROJECT = "polytax"
        WandBMixin.WANDB_ENTITY = "mweiss10"
        if self.get("wandb/use"):
            self.initialize_wandb()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = xm.xla_device()

        self.total_shards = int(os.environ.get("WORLD_SIZE", 1)) * xm.xrt_world_size()
        print(f"ordinal {xm.get_ordinal()}")
        print(f"total shards: {self.total_shards}")
        self.dist_index = int(os.environ.get("RANK", 0)) * int(xm.xrt_world_size()) + int(xm.get_ordinal())
        print(f"self.dist_index: {self.dist_index}")
        self.is_multi_host = True if int(os.environ.get("WORLD_SIZE", 1)) > 1 else False
        print(f"is_multi_host: {self.is_multi_host}")
       
        # one process per host  
        if self.is_multi_host and xm.is_master_ordinal():
            dist.init_process_group(backend="GLOO") #GLOO for CPU comms, which this is.

        self._build()

    def _build(self):
        self.logger = setup_logging()
        print(f"{self._config}")
        self.seed = self.get("seed", 1)
        set_seed(self.seed) # handles random seed setting for everything but XLA
        self.cache_dir = self.get("cache_dir")
        self.model_type = self.get("model_type")
        self.tokenizer = self.get_tokenizer(**self.get("tokenizer/kwargs"))
        self.model_config = self.get_model_config(self.get("model_config"), self.get("model_name_or_path"))
        self.train_batch_size = int(self.get("per_device_train_batch_size")) * self.total_shards
        self.eval_batch_size = int(self.get("per_device_eval_batch_size")) * self.total_shards

        # Build the data loaders
        self.train_dataset, self.eval_dataset = self._build_loaders()
        self.train_loader = iter(pl.MpDeviceLoader(self.train_dataset, self.device))
        self.eval_loader = iter(pl.MpDeviceLoader(self.eval_dataset, self.device))
        self.model = T5ForConditionalGeneration(self.model_config).to(self.device)

        # TODO: we should replace this probably with the fairseq implementation. Read the T5 paper and search adafactor, and use the inverse square root
        #  https://github.com/pytorch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
        #self.model = MpModelWrapper(self.model)

        self.optimizer = Adafactor(
            self.model.parameters(),
            # lr=self.get("learning_rate"),
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
        max_seq_length = min(self.get("max_seq_length"), self.tokenizer.model_max_length)
        sequence_length = {"inputs": max_seq_length, "targets": max_seq_length, "decoder_input_ids": max_seq_length}
        self.task = seqio.get_mixture_or_task(self.get("dataset_name"))
        eos_keys = set(k for k, f in self.task.output_features.items() if f.add_eos)
        train_dataset = self.task.get_dataset(sequence_length=sequence_length, split="train", use_cached=False, shuffle=True, seed=self.seed, num_epochs=1)
        if self.total_shards > 1:
            train_dataset = train_dataset.shard(num_shards=self.total_shards, index=self.dist_index)
        train_dataset = pack_or_pad(train_dataset, sequence_length, feature_keys=self.task.output_features,
                                    ensure_eos=eos_keys, pack=True)
        train_dataset = train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        train_dataset = itertools.cycle(train_dataset)
        
        #from torch.utils.data import DataLoader
        #train_dataset = CustomDataset()

        try:
            eval_dataset = self.task.get_dataset(sequence_length=sequence_length, split="validation", use_cached=False,
                                                  shuffle=True, seed=self.seed, num_epochs=1)
        except Exception:
            print("no validation set")
            eval_dataset = self.task.get_dataset(sequence_length=sequence_length, split="train[:90%]", use_cached=False,
                                                                      shuffle=True, seed=self.seed, num_epochs=1)
            if self.total_shards > 1:
                eval_dataset = eval_dataset.shard(num_shards=self.total_shards, index=self.dist_index)
        eval_dataset = pack_or_pad(eval_dataset, sequence_length, feature_keys=self.task.output_features,
                                    ensure_eos=eos_keys, pack=True)
        eval_dataset = eval_dataset.batch(self.eval_batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        eval_dataset = itertools.cycle(eval_dataset)
        return train_dataset, eval_dataset

    def get_model_config(self, model_config=None, model_name_or_path=None):
        if model_name_or_path: # if we're loading an already trained model
            model_config = T5Config.from_pretrained(model_name_or_path)
        elif type(model_config) == dict: # pass it in directly from yaml
            model_config['layer_norm_epsilon'] = float(model_config['layer_norm_epsilon'])
            model_config = T5Config.from_dict(model_config, cache_dir=self.cache_dir, vocab_size=len(self.tokenizer))
        elif model_config:
            model_config = T5Config.from_pretrained(
                model_config, cache_dir=self.cache_dir, vocab_size=len(self.tokenizer)
            )
        else:
            model_config = CONFIG_MAPPING[self.model_type]()
        return model_config

    def get_tokenizer(self, tokenizer_name="./tokenizer/", use_fast=True):
        # Load pretrained model and tokenizer
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir, use_fast=use_fast)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        return tokenizer

    def reduce_gradients(self):
        # AllReduce the model gradients so we can step the global gradient
        print("reducing...")
        #xm.reduce_gradients(self.optimizer)
        print("reduced")
        if self.is_multi_host:
            if xm.is_master_ordinal():
                dist.all_reduce(param.grad.cpu() / dist.get_world_size()) 
                xm.all_reduce(xm.REDUCE_SUM, param.grad.to(self.device))
            else:
                zeros = torch.zeros_like(param.grad)
                xm.all_reduce(xm.REDUCE_SUM, zeros)    

    def run(self):
        for _ in self.progress(range(self.get("num_train_steps")), desc="Training", tag="train"):
            start = time.time()
            #samples = self.get_samples(self.train_dataset)
            samples = next(self.train_loader)
            import pdb; pdb.set_trace()
            #print(f"geT_samples took: {time.time()-start}")
            #print(samples)
            #self.optimizer.zero_grad()
            start = time.time()
            x_hat = self.model(**samples)
            print(f"running took: {time.time()-start}")

            x_hat.loss.backward()
            start = time.time()
            #self.reduce_gradients()
            #print(x_hat)
            #print(self.optimizer)
            #print(self.model)
            print(f"reducing took: {time.time()-start}")
            start = time.time()
            xm.optimizer_step(self.optimizer)#.step()
            print(f"step took: {time.time() - start}")
            self.next_step()
            start = time.time()
            if self.get("use_wandb"):
                self.wandb_log(**{"train_loss": x_hat.loss.detach()})
            print(f"logging took: {time.time() - start}")
            # log gradients once per epoch
            #if self.get("use_wandb"):
            #    self.wandb_watch(self.model, x_hat.loss.detach(), log_freq=1)
            # start = time.time()
            # Checkpoint
            # TODO: I benchmarked this and it is extremely slow -- speed it up 
            #if self.step % self.get("checkpoint_every") == 0:
            #    torch.save(
            #        self.model.state_dict(),
            #        open(f"{self.experiment_directory}/Weights/model-{self.epoch}.pt", "wb"),
            #    )
            #    print(f"checkpoint took: {time.time()-start}")
            # print(met.metrics_report())


            # Run Validation
            if self.step % self.get("eval_every") == 0:
                self.model.eval()
                for _ in self.progress(range(self.get("num_eval_steps")), desc="Evaluating...", tag="train"):
                    start = time.time()
                    samples = self.get_samples(self.eval_dataset)
                    x_hat, input, mu, log_var = self.model(**samples)
                    
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": x_hat.loss.detach()})
                    print(f"evaluation took: {time.time() - start}")
                self.model.train()

    @property
    def evaluate_now(self):
        return self.step % self.get("eval_steps") == 0 and self.step > 0

    @property
    def save_now(self):
        return self.step % self.get("save_steps") == 0 and self.step > 0

    # TODO add checkpoints
    # def checkpoint(self):
    #     if self.save_now and jax.process_index() == 0:
    #         params = jax.device_get(jax.tree_map(lambda x: x[0], self.state.params))
    #         self.model.save_pretrained(
    #             self.checkpoint_directory,
    #             params=params,
    #             push_to_hub=False,
    #             commit_message=f"Saving weights and logs of step {self.step}"
    #         )

    def get_samples(self, iterable_dataset):
        samples = next(iterable_dataset)
        shape = (-1, self.get("max_seq_length"))
        samples["decoder_input_ids"] = shift_tokens_right(samples['targets'],
                                                          self.model.config.pad_token_id,
                                                          self.model.config.decoder_start_token_id)
        samples = {"input_ids": torch.tensor(samples["inputs"], dtype=torch.long, device=xm.xla_device()).view(shape),
         "labels": torch.tensor(samples["targets"], dtype=torch.long, device=xm.xla_device()).view(shape),
         "decoder_input_ids": torch.tensor(samples["decoder_input_ids"], dtype=torch.long, device=xm.xla_device()).view(shape)}
        return samples


def _mp_fn(index, args):
    Experiment1().run()

if __name__ == '__main__':
    if True:
        xmp.spawn(_mp_fn, args=({},), nprocs=1)
    else:
        Experiment1().run()
