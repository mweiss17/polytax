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
import itertools
import jax
import jax.numpy as jnp
import jax.profiler
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot, shard
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm
np_config.enable_numpy_behavior()
from transformers import (
    CONFIG_MAPPING,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from mesh_tensorflow.transformer.dataset import pack_or_pad
import tensorflow as tf
import seqio
from speedrun import BaseExperiment, WandBMixin, IOMixin
from polytax import dataset # DO NOT DELETE -- imports seqio datasets


# Imports for Torch
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch_lr_scheduler
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )
    logger = logging.getLogger(__name__)
    return logger

class Experiment1(BaseExperiment, WandBMixin, IOMixin):
    def __init__(self):
        super(Experiment1, self).__init__()
        self.auto_setup()
        WandBMixin.WANDB_PROJECT = "polytax"
        WandBMixin.WANDB_ENTITY = "mweiss10"
        if self.get("wandb/use"):
            self.initialize_wandb()
        # Not for TPU 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build()

    def _build(self):
        self.logger = setup_logging()
        # self.logger.info(f"Run parameters {self.get('')}")
        print(f"{self._config}")
        self.seed = self.get("seed")
        set_seed(self.seed)
        self.cache_dir = self.get("cache_dir")
        self.model_type = self.get("model_type")
        self.tokenizer = self.get_tokenizer(**self.get("tokenizer/kwargs"))
        self.model_config = self.get_model_config(self.get("model_config"), self.get("model_name_or_path"))
        if xla_found is False:
            self.train_batch_size = int(self.get("per_device_train_batch_size"))
            self.eval_batch_size = int(self.get("per_device_eval_batch_size"))
        else:
            self.train_batch_size = int(self.get("per_device_train_batch_size")) * xm.xrt_world_size()
            self.eval_batch_size = int(self.get("per_device_eval_batch_size")) * xm.xrt_world_size()
        
        # Build the data loaders
        self.train_dataset, self.eval_dataset = self._build_loaders()
        # TODO: add rng setting
        self.model = T5ForConditionalGeneration(self.model_config)

        # TODO: we should replace this probably with the fairseq implementation. Read the T5 paper and search adafactor, and use the inverse square root
        #  https://github.com/pytorch/fairseq/blob/main/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
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
        sequence_length = {"inputs": max_seq_length, "targets": max_seq_length}

        self.task = seqio.get_mixture_or_task(self.get("dataset_name"))
        eos_keys = set(k for k, f in self.task.output_features.items() if f.add_eos)

        train_dataset = self.task.get_dataset(sequence_length=sequence_length, split="train", use_cached=False, shuffle=True,
                                         seed=self.seed, num_epochs=1)
        train_dataset = pack_or_pad(train_dataset, sequence_length, feature_keys=self.task.output_features,
                                    ensure_eos=eos_keys, pack=True)
        train_dataset = train_dataset.batch(self.train_batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        train_dataset = itertools.cycle(train_dataset)

        try:
            eval_dataset = self.task.get_dataset(sequence_length=sequence_length, split="validation", use_cached=False,
                                                  shuffle=True, seed=self.seed, num_epochs=1)
        except Exception:
            print("no validation set")
            eval_dataset = self.task.get_dataset(sequence_length=sequence_length, split="train[:90%]", use_cached=False,
                                                                      shuffle=True, seed=self.seed, num_epochs=1)
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

    def get_tokenizer(self, tokenizer_name=None, use_fast=None):
        # Load pretrained model and tokenizer
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=self.cache_dir, use_fast=use_fast)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        return tokenizer

    def run(self):
        for _ in self.progress(range(self.get("num_train_steps")), desc="Training", tag="train"):
            samples = self.get_samples(self.train_dataset)

            # TODO checkout the tokenization https://huggingface.co/transformers/model_doc/t5.html#training and replace the seqio tokenizer in dataset.py with ours (try to make it faster)
            input_ids = torch.tensor(samples["input_ids"], dtype=torch.long).view(-1,self.get("max_seq_length"))
            target_ids = torch.tensor(samples["targets"], dtype=torch.long).view(-1,self.get("max_seq_length"))
            decoder_ip_ids = torch.tensor(samples["decoder_input_ids"], dtype=torch.long).view(-1,self.get("max_seq_length"))
            print("input_ids shape", input_ids.size())
            print("targets shape", target_ids.size())
            print("decoder_input_ids shape", decoder_ip_ids.size())
            x_hat = self.model(input_ids=input_ids, labels=target_ids, decoder_input_ids=decoder_ip_ids)
            self.optimizer.zero_grad()
            x_hat.loss.backward()

            self.optimizer.step()
            self.next_step()
            if self.get("use_wandb"):
                self.wandb_log(**{"train_loss": x_hat.loss.detach()})

            # log gradients once per epoch
            if self.get("use_wandb"):
                self.wandb_watch(self.model, x_hat.loss.detach(), log_freq=1)

            # Checkpoint
            if self.step % self.get("checkpoint_every") == 0:
                torch.save(
                    self.model.state_dict(),
                    open(f"{self.experiment_directory}/Weights/model-{self.epoch}.pt", "wb"),
                )

            # Run Validation
            if self.step % self.get("eval_every") == 0:
                self.model.eval()
                for _ in self.progress(range(self.get("num_eval_steps")), desc="Evaluating...", tag="train"):
                    samples = self.get_samples(self.eval_dataset)
                    input_ids = torch.tensor(samples["input_ids"], dtype=torch.long).view(-1,
                                                                                          self.get("max_seq_length"))
                    target_ids = torch.tensor(samples["targets"], dtype=torch.long).view(-1, self.get("max_seq_length"))
                    decoder_ip_ids = torch.tensor(samples["decoder_input_ids"], dtype=torch.long).view(-1, self.get(
                        "max_seq_length"))

                    x_hat, input, mu, log_var = self.model(input_ids=input_ids, labels=target_ids, decoder_input_ids=decoder_ip_ids)
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": x_hat.loss.detach()})

                self.model.train()

    @property
    def evaluate_now(self):
        return self.step % self.get("eval_steps") == 0 and self.step > 0

    @property
    def save_now(self):
        return self.step % self.get("save_steps") == 0 and self.step > 0

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
        samples = shard(samples)
        # TODO: incorporate this into the seqio preprocessor
        samples["decoder_input_ids"] = shift_tokens_right(samples['targets'],
                                                          self.model.config.pad_token_id,
                                                          self.model.config.decoder_start_token_id)
        samples = {"targets": samples['targets'], "input_ids": samples["inputs"],
                   "decoder_input_ids": samples["decoder_input_ids"]}
        return samples



if __name__ == '__main__':
    global xla_found
    try: 
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.test.test_utils as test_utils
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.core.xla_env_vars as xenv
        xla_found = True
    except:
        print("XLA not found\n")
        xla_found = False
    Experiment1().run()
