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
import seqio
import t5
import tensorflow_datasets as tfds
from t5.models import mesh_transformer
import tensorflow as tf
from mesh_tensorflow.transformer import dataset as transformer_dataset
from speedrun import WandBMixin, IOMixin
from polytax.data.utils import get_examples
from polytax.train import Trainer
from polytax.data.dataset import GLUEWrapper

global xla_found
try:
    import torch_xla
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
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

class Finetuner(Trainer, WandBMixin, IOMixin):
    WANDB_PROJECT = "polytax"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Finetuner, self).__init__()
        self.auto_setup()
        self.tasks = self._build_tasks()
        self.dataset = self._build_dataset()
        if xla_found:
            self.train_loader = iter(pl.MpDeviceLoader(iter(self.dataset), self.device))
        else:
            self.train_loader = iter(self.dataset)

    def _build_tasks(self, split=tfds.Split.VALIDATION):
        # determine maximum sequence length to use
        tasks = t5.data.get_subtasks(
            t5.data.get_mixture_or_task(self.get("dataset_name")))
        tasks = seqio.evaluation.get_valid_eval_tasks(tasks, split)
        return tasks

    def _build_dataset(self, split=tfds.Split.VALIDATION, sequence_length=None):
        sequence_length = sequence_length or {"inputs": 512, "targets": 128}
        combined_ds = None

        for task in self.tasks:
            ds = mesh_transformer.mesh_eval_dataset_fn(
                mixture_or_task_name=task.name,
                sequence_length=sequence_length,
                dataset_split=split)[0].dataset_fn()
            ds = ds.map(
                t5.models.utils.filter_features,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            combined_ds = ds if not combined_ds else combined_ds.concatenate(ds)
        combined_ds = combined_ds.batch(self.train_batch_size, drop_remainder=False)  # pytype:disable=attribute-error
        # Pad the final batch.
        combined_ds = transformer_dataset.trim_and_pad_dataset(
            combined_ds, length=self.train_batch_size)
        combined_ds = combined_ds.prefetch(tf.data.experimental.AUTOTUNE)
        combined_ds = GLUEWrapper(combined_ds)
        return combined_ds

    def decode_and_compute_accuracy(self, x, x_hat):
        sample_id = 1
        labels = x['labels'][sample_id]
        labels_list = labels.cpu().numpy().tolist()
        input_ids = x['input_ids'][sample_id].cpu().numpy().tolist()
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

        #       outputs = [
        #           tf.compat.as_text(d) for d in mtf_utils.decode(
        #               estimator, estimator_input_fn, vocabulary, checkpoint_path)
        #       ]
        return input, labels, preds, accuracy

    def _log(self, step, tracker, x, x_hat):
        loss = x_hat.loss.detach()

        # Print to console
        print_training_update(self.device, step, loss, tracker.rate(), tracker.global_rate())
        cached_targets, cached_datasets, max_sequence_length = get_examples(self.tasks, split=tfds.Split.VALIDATION)
        breakpoint()
        for task in self.tasks:
            # Extract the portion of decodes corresponding to this dataset
            dataset = cached_datasets[task.name]
            dataset_size = len(cached_targets[task.name])
            predictions = [
                task.postprocess_fn(d, example=ex)
                for d, ex in zip(x_hat.logits.argmax(2)[:dataset_size], tfds.as_numpy(dataset))
            ]
            for metric_fn in task.metric_fns:
                targets = cached_targets[task.name]
                metric_result = metric_fn(targets, predictions)
                for metric_name, metric_value in metric_result.items():
                    print(f"{eval/{task.name}/{metric_name}} at step {step}: {metric_value}")

        # # Only padding should remain.
        # if batch_size:
        #     expected_pad = -sum(len(t)
        #                         for t in cached_targets.values()) % batch_size
        #     if outputs and len(outputs) != expected_pad:
        #         raise ValueError("{} padded outputs, {} expected.".format(
        #             len(outputs), expected_pad))

        # Return if we don't have wandb
        if not self.get("use_wandb"):
            return

        # Write speeds to wandb
        self.wandb_log(**{"instantaneous it/s": tracker.rate(), "global it/s": tracker.global_rate()})

        # Get a text example and log it
        input, label, pred, accuracy= self.decode_and_compute_accuracy(x, x_hat)
        self.table = wandb.Table(columns=["Step", "Accuracy", "Loss", "Input", "Label", "Predicted"])
        self.table.add_data(step, accuracy, loss, input, label, pred)
        self.wandb_log(**{"accuracy": accuracy, "examples": self.table, "train_loss": loss, "num_tokens": self.get("max_seq_length") * self.total_batch_size * self.step})

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

        for step in range(self.get("num_train_steps")):

            # Get data
            x = next(self.train_loader)

            # Forward model
            x_hat = self.model(**x)

            # Optimization
            x_hat.loss.backward()
            if (step + 1) % self.get("gradient_accumulation_steps", 1) == 0:
                self.reduce_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Increment step count
            self.next_step()
            self.tracker.add(1)

            if self.log_now:
                self.log(x, x_hat)

            if self.checkpoint_now:
                self.save_checkpoint()


def _mp_fn(index, args):
    Finetuner().run()

if __name__ == '__main__':
    if xla_found:
        xmp.spawn(_mp_fn, args=({},), nprocs=8)
    else:
        Finetuner().run()
