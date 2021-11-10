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
from typing import Dict, Callable, Tuple, Union, Optional
import torch
import torch.distributed as dist
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
from transformers import (
    T5ForConditionalGeneration,
    SwitchForConditionalGeneration,
    T5Config,
    set_seed,
)
from speedrun import BaseExperiment, WandBMixin, IOMixin
from transformers.optimization import Adafactor  # pylint: disable=unused-import
from t5.data.utils import get_default_vocabulary

global xla_found
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.test.test_utils as test_utils
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_env_vars as xenv
    from torch_xla.utils import gcsfs
    from torch_xla.core.xla_model import RateTracker
    from torch_xla.test.test_utils import print_training_update, print_test_update

    xla_found = True
except Exception as e:
    print(f"XLA not found {e} \n")
    xla_found = False
    from utils.tracker import RateTracker, print_training_update, print_test_update

from polytax.data.utils import (
    get_pretrain_dataset,
    get_validation_tasks,
    get_validation_datasets,
    run_evaluation,
)
from polytax.utils.utils import _upload_blob_gcs, reduce_gradients
from polytax.utils.train_state import TrainingState


class Trainer(WandBMixin, IOMixin, BaseExperiment):
    WANDB_PROJECT = "polytax"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Trainer, self).__init__()
        self.auto_setup()
        training_state = TrainingState.initial_state(step=self.step, epoch=self.epoch)
        self._build(training_state)

    def _build(self, training_state: "TrainingState"):
        print(f"{self._config}")
        self._build_general(training_state)
        self._build_tasks(training_state)
        self._build_model(training_state)
        self._build_optimizer(training_state)

    def _build_general(self, training_state: "TrainingState"):
        self._step = training_state.step
        self._epoch = training_state.epoch
        self.tracker = RateTracker()

        set_seed(self.get("seed"))  # handles random seed setting for everything but XLA

        if self.is_master_ordinal:
            if self.get("use_wandb"):
                self.initialize_wandb(resume=True)
            if self.is_multi_host:
                # GLOO for CPU comms, NCCL for GPU comms
                dist.init_process_group(backend=self.get("backend"))

    def _build_tasks(self, training_state: "TrainingState"):
        self.train_loader = get_pretrain_dataset(
            self.num_shards,
            self.global_rank,
            self.get("seed"),
            **self.get("dataset/train/kwargs"),
            split="train",
            device=self.device,
        )
        self.valid_tasks = get_validation_tasks(
            name=self.get("dataset/validation/name"), split="validation"
        )
        self.valid_loaders = get_validation_datasets(
            self.valid_tasks,
            **self.get("dataset/validation/kwargs"),
            split="validation",
            device=self.device,
        )

    def _build_model(self, training_state: "TrainingState"):
        self.tokenizer = get_default_vocabulary()

        self.get("model_config")["layer_norm_epsilon"] = float(
            self.get("model_config")["layer_norm_epsilon"]
        )
        model_config = T5Config.from_dict(
            self.get("model_config"),
            cache_dir=self.get("cache_dir"),
            vocab_size=self.tokenizer.vocab_size,
        )

        if "switch" in model_config.model_type:
            self.model = SwitchForConditionalGeneration(model_config)
        else:
            self.model = T5ForConditionalGeneration(model_config)

        training_state.load_in_model(self.model)
        self.model.to(self.device)

    def _build_optimizer(self, training_state: "TrainingState"):
        # No need to specify learning rate in Adafactor: https://arxiv.org/pdf/1804.04235.pdf
        self.optim = eval(self.get("optim/name"))(
            self.model.parameters(), **self.get("optim/kwargs")
        )
        training_state.load_in_optims(optim=self.optim,)

    @property
    def device(self):
        return xm.xla_device() if xla_found else torch.device("cpu")

    @property
    def local_world_size(self):
        return int(xm.xrt_world_size()) if xla_found else 1  # Devices per host

    @property
    def global_world_size(self):
        return int(os.environ.get("WORLD_SIZE", 1))  # Total number of hosts

    @property
    def num_shards(self):
        return self.global_world_size * self.local_world_size

    @property
    def global_rank(self):
        return int(os.environ.get("RANK", 0)) * self.local_world_size + self.local_rank

    @property
    def local_rank(self):
        return int(xm.get_ordinal()) if xla_found else 0

    @property
    def is_master_ordinal(self):
        return xm.is_master_ordinal() if xla_found else True

    @property
    def is_multi_host(self):
        return self.local_world_size > 1

    @property
    def total_batch_size(self):
        return (
            self.get("dataset/train/kwargs/per_device_batch_size")
            * self.local_world_size
            * self.global_world_size
        )

    @property
    def validate_now(self):
        return self.step % self.get("valid_every") == 0 and self.step > 0

    @property
    def checkpoint_now(self):
        return self.step % self.get("training/checkpoint_every") == 0 and self.step > 0

    @property
    def losses_state_dict(self):
        return {}
        # TODO re-add
        # return {"loss": self.loss.state_dict()}

    @property
    def optims_state_dict(self):
        state_dict = {}
        if self.optim is not None:
            state_dict.update({"optim": self.optim.state_dict()})
        return state_dict

    @property
    def schedulers_state_dict(self):
        return {}

    @property
    def train_state(self, misc_attributes: dict = None):
        return TrainingState(
            step=self.step,
            epoch=self.epoch,
            model_state_dict=self.model.state_dict(),
            losses_state_dict=self.losses_state_dict,
            optims_state_dict=self.optims_state_dict,
            schedulers_state_dict=self.schedulers_state_dict,
            misc_attributes=misc_attributes,
        )

    def checkpoint(self, training_state: "TrainingState"):
        tmp_checkpoint_path = f"/tmp/checkpoint.pt"
        gcs_checkpoint_path = f"gs://{self.get('gcs_bucket', 'must-results')}/{self.experiment_directory}/Weights/model-{self.step}.pt"
        checkpoint_path = f"{self.experiment_directory}/Weights/model-{self.step}.pt"

        if xla_found and self.is_master_ordinal:
            print(f"checkpointing the model to {gcs_checkpoint_path}")
            training_state.serialize(tmp_checkpoint_path)
            _upload_blob_gcs(tmp_checkpoint_path, gcs_checkpoint_path)
        else:
            print(f"checkpointing the model to {checkpoint_path}")
            training_state.serialize(checkpoint_path)

    def decode_and_compute_accuracy(self, x, x_hat):
        sample_id = 1
        labels = x["labels"][sample_id]
        labels_list = labels.cpu().numpy().tolist()
        input_ids = x["input_ids"][sample_id].cpu().numpy().tolist()
        preds = x_hat.logits[sample_id].argmax(axis=1)
        preds_list = preds.cpu().numpy().tolist()

        # compute_accuracy
        correct = preds.eq(labels.view_as(preds)).sum()
        num_extra_ids = (labels == -100).sum()
        accuracy = 100.0 * correct / (labels.nelement() - num_extra_ids)

        # Prepare to decode the labels
        extra_id = 32099
        ids_to_replace_in_preds = []
        for idx, label in enumerate(labels_list):
            if label == -100:
                ids_to_replace_in_preds.append(idx)
                labels_list[idx] = extra_id
                extra_id -= 1

        extra_id = 32099
        for idx, pred in enumerate(preds_list):
            if idx in ids_to_replace_in_preds:
                preds_list[idx] = extra_id
                extra_id -= 1

        input = self.tokenizer.decode(input_ids)
        labels = self.tokenizer.decode(labels_list)
        preds = self.tokenizer.decode(preds_list)
        return input, labels, preds, accuracy

    def _log(self, step, tracker, x, x_hat):
        loss = x_hat.loss.detach()

        # Print to console
        print_training_update(
            self.device, step, loss, tracker.rate(), tracker.global_rate()
        )

        # Return if we don't have wandb
        if not self.get("use_wandb"):
            return

        # Write speeds to wandb
        self.wandb_log(
            **{
                "instantaneous it/s": tracker.rate(),
                "global it/s": tracker.global_rate(),
            }
        )

        # Get a text example and log it
        input, label, pred, accuracy = self.decode_and_compute_accuracy(x, x_hat)
        self.table = wandb.Table(
            columns=["Step", "Accuracy", "Loss", "Input", "Label", "Predicted"]
        )
        self.table.add_data(step, accuracy, loss, input, label, pred)
        self.wandb_log(
            **{
                "accuracy": accuracy,
                "examples": self.table,
                "train_loss": loss,
                "num_tokens": self.get("dataset/train/kwargs/max_seq_length")
                * self.total_batch_size
                * self.step,
            }
        )

    def log(self, x, x_hat, tracker):
        # If XLA is found, then we are on TPU and we should use a closure to increase efficiency
        if xla_found:
            xm.add_step_closure(self._log, args=(self.step, tracker, x, x_hat))
        # Otherwise just call the function to log directly
        else:
            self._log(self.step, tracker, x, x_hat)

    def loss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def run(self):
        self.model.train()
        for x in self.progress(self.train_loader, desc="Training", tag="train"):
            x_hat = self.model(**x)
            loss = self.loss(x_hat.logits, x["labels"])
            loss.backward()

            if (self.step + 1) % self.get("gradient_accumulation_steps", 1) != 0:
                reduce_gradients(xla_found, self.is_multi_host, self.optim)
                self.optim.step()
                self.optim.zero_grad()

            # Increment step count
            self.next_step()
            self.tracker.add(1)

            if self.log_scalars_now and self.is_master_ordinal:
                self.log(x, x_hat, self.tracker)

            if self.validate_now:
                self.validate()

            if self.checkpoint_now:
                self.checkpoint(self.train_state)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            # get list of outputs
            outputs = run_evaluation(
                self.model, self.valid_loaders, self.valid_tasks, self.tokenizer
            )

    __call__ = run


def _mp_fn(index, args):
    Trainer().run()


if __name__ == "__main__":
    if xla_found:
        xmp.spawn(_mp_fn, args=({},), nprocs=8)
    else:
        Trainer().run()
