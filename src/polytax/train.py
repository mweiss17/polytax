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
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="[%X]",
    )
    logger = logging.getLogger(__name__)
    return logger

# We use Optax's "masking" functionality to not apply weight decay
# to bias and LayerNorm scale parameters. decay_mask_fn returns a
# mask boolean with the same structure as the parameters.
# The mask is True for parameters that should be decayed.
def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {
        path: (path[-1] != "bias" and path[-2:] not in [("layer_norm", "scale"), ("final_layer_norm", "scale")])
        for path in flat_params
    }
    return traverse_util.unflatten_dict(flat_mask)


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
        
        #self.rng = jax.random.PRNGKey(self.seed)
        #self.dropout_rngs = jax.random.split(self.rng, jax.local_device_count())
        self.model = T5ForConditionalGeneration(self.model_config)

        # Create learning rate schedule
        # warmup_fn = optax.linear_schedule(
        #     init_value=0.0, end_value=self.get("learning_rate"), transition_steps=self.get("warmup_steps")
        # )
        # decay_fn = optax.linear_schedule(
        #     init_value=self.get("learning_rate"),
        #     end_value=0,
        #     transition_steps=self.get("num_train_steps") - self.get("warmup_steps"),
        # )
        # self.linear_decay_lr_schedule_fn = optax.join_schedules(
        #     schedules=[warmup_fn, decay_fn], boundaries=[self.get("warmup_steps")]
        # )

        # if self.get("adafactor"):
        #     # We use the default parameters here to initialize adafactor,
        #     # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        #     optimizer = optax.adafactor(
        #         learning_rate=self.linear_decay_lr_schedule_fn,
        #     )
        # else:
        #     optimizer = optax.adamw(learning_rate=self.linear_decay_lr_schedule_fn, b1=self.get("adam_beta1"),
        #         b2=self.get("adam_beta2"), weight_decay=self.get("weight_decay"), mask=decay_mask_fn,
        #     )

        # Add scheduler WIP
        if self.get("adafactor"):
            # replace AdamW with Adafactor
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=1e-3,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
        else: 
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=1e-3,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )

        # Create parallel version of the train step
        #self.p_train_step = jax.pmap(self.train_step, "batch", donate_argnums=(0,))
        #self.p_eval_step = jax.pmap(self.eval_step, "batch", donate_argnums=(0,))

        # Setup and replicate the train state on each device
        #state = train_state.TrainState.create(apply_fn=self.model.__call__, params=self.model.params, tx=optimizer)
        #self.state = jax_utils.replicate(state)

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

    def train_slow_step(self, state, batch):
        #_build_loaders
        pass

    def run(self):
        for _ in self.progress(range(self.get("num_train_steps")), desc="Training", tag="train"):
            samples = self.get_samples(self.train_dataset)
        # for epoch in tqdm(range(self.get("num_epochs")), desc="epochs..."):
            # Train the model
            # for batch in tqdm(self.train_dataset, desc="batches..."):
            # samples = samples.to(self.device)
            #import pdb; pdb.set_trace()



            print("Logging shapes \n")
            input_ids = torch.tensor(samples["input_ids"], dtype=torch.long).view(1,16)
            target_ids = torch.tensor(samples["targets"], dtype=torch.long).view(1,16)
            decoder_ip_ids = torch.tensor(samples["decoder_input_ids"], dtype=torch.long).view(1,16)
            print("input_ids shape", input_ids.size())
            print("targets shape", target_ids.size())
            print("decoder_input_ids shape", decoder_ip_ids.size())
            x_hat = self.model(input_ids=input_ids,labels=target_ids, decoder_input_ids=decoder_ip_ids)
            # TODO change loss function
            print("This should print")
            loss = self.model.loss_function(
                x_hat, input, mu, log_var, M_N=self.get("dataloader/batch_size")
            )
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            self.next_step()
            if self.get("use_wandb"):
                self.wandb_log(**{"train_loss": loss})
                self.wandb_log(**{"lr": self.scheduler.get_lr()[0]})

            self.next_epoch()
            self.scheduler.step()

            # log gradients once per epoch
            if self.get("use_wandb"):
                self.wandb_watch(self.model, loss, log_freq=1)

            # Checkpoint
            if epoch % self.get("checkpoint_every") == 0:
                torch.save(
                    self.model.state_dict(),
                    open(f"{self.experiment_directory}/Weights/model-{epoch}.pt", "wb"),
                )

            # Run Validation
            if epoch % self.get("valid_every") == 0:
                self.model.eval()
                for valbatch in tqdm(self.valid, "valid batches..."):
                    valbatch = valbatch.to(self.device)
                    x_hat, input, mu, log_var = self.model(valbatch)
                    loss = self.model.loss_function(
                        x_hat, input, mu, log_var, M_N=self.get("dataloader/batch_size")
                    )
                    if self.get("use_wandb"):
                        self.wandb_log(**{"valid_loss": loss})
                        self.wandb_log(**{"lr": self.scheduler.get_lr()})

                self.model.train()














    def train_fast_step(self, state, batch):
        pass
    # def train_step(self, state, batch, dropout_rng):
    #     dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    #     def loss_fn(params):
    #         targets = batch.pop("targets")
    #         logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
    #         loss = optax.softmax_cross_entropy(logits, onehot(targets, logits.shape[-1])).mean()
    #         return loss

    #     grad_fn = jax.value_and_grad(loss_fn)
    #     loss, grad = grad_fn(state.params)
    #     grad = jax.lax.pmean(grad, "batch")
    #     new_state = state.apply_gradients(grads=grad)

    #     metrics = jax.lax.pmean(
    #         {"loss": loss, "learning_rate": self.linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    #     )
    #     return new_state, metrics, new_dropout_rng

    # def eval_step(self, params, batch):
    #     targets = batch.pop("targets")
    #     logits = self.model(**batch, params=params, train=False)[0]
    #     loss = optax.softmax_cross_entropy(logits, onehot(targets, logits.shape[-1]))
    #     accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    #     metrics = {"loss": loss.mean(), "accuracy": accuracy.mean(), "step": self.step}

    #     metrics = jax.lax.pmean(metrics, axis_name="batch")
    #     return metrics

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


    # def run(self):

    #     for _ in self.progress(range(self.get("num_train_steps")), desc="Training", tag="train"):
    #         samples = self.get_samples(self.train_dataset)
    #         self.state, train_metric, self.dropout_rngs = self.p_train_step(self.state, samples, self.dropout_rngs)
    #         if self.log_wandb_now and self.get("wandb/use") and jax.process_index() == 0:
    #             train_metric = jax_utils.unreplicate(train_metric)
    #             self.wandb_log(**train_metric)
            
    #         if self.evaluate_now:
    #             eval_metrics = []
    #             for _ in self.progress(range(self.get("num_eval_steps")), desc="Evaluating ...", tag="eval"):
    #                 samples = self.get_samples(self.eval_dataset)
    #                 metrics = self.p_eval_step(self.state.params, samples)
    #                 eval_metrics.append(metrics)

    #             eval_metrics = jax_utils.unreplicate(eval_metrics)
    #             eval_metrics = jax.tree_multimap(lambda *xs: list(xs), *eval_metrics)  # Transpose the tree
    #             for k, v in eval_metrics.items():
    #                 eval_metrics[k] = jnp.mean(jnp.array(eval_metrics[k]))
    #             if jax.process_index() == 0 and self.get("use/wandb"):
    #                 self.wandb_log(**eval_metrics)
    #         self.checkpoint()
    #         self.next_step()

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
