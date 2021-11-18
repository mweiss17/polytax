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
import io
import sys
import wandb
import argparse
from copy import deepcopy
from typing import Union
import torch
from wormulon.core import TPUJob
from wormulon.utils import JobStatus, _read_blob_gcs
import torch.distributed as dist
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
from transformers import (
    T5ForConditionalGeneration,
    SwitchForConditionalGeneration,
    T5Config,
    set_seed,
)
from speedrun import BaseExperiment, WandBMixin, IOMixin, register_default_dispatch
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
    get_eval_tasks,
    get_task,
    get_dataset,
    get_eval_datasets,
    get_targets_and_examples,
)
from polytax.utils.utils import reduce_gradients
from polytax.utils.train_state import TrainingState


class Trainer(WandBMixin, IOMixin, BaseExperiment):
    WANDB_PROJECT = "polytax"
    WANDB_ENTITY = "mweiss10"

    def __init__(self, nanny: "Nanny"):
        super(Trainer, self).__init__()
        self._preconfigure(nanny)

    def _preconfigure(self, nanny: "Nanny"):
        # Copy only the essentials from the parent class, i.e. the things that can be
        # easily serialized.
        self._experiment_directory = nanny._experiment_directory
        self._config = deepcopy(nanny._config)
        self._argv = deepcopy(nanny._argv)
        self.WANDB_ENTITY = nanny.WANDB_ENTITY
        self.WANDB_PROJECT = nanny.WANDB_PROJECT
        self.WANDB_RUN_ID = nanny.wandb_run_id
        # Bit of a hack, but we set this here to have it uploaded to wandb.
        self.set("speedrun_meta/experiment_directory", self._experiment_directory)

    def _build(self, training_state: "TrainingState", tpu_job: "TPUJob"):
        print(f"{self._config}")
        self.tpu_job = tpu_job
        self._build_general(training_state)
        self._build_tasks(training_state)
        self._build_model(training_state)
        self._build_optimizer(training_state)

    def _build_general(self, training_state: "TrainingState"):
        self._step = training_state.step
        self._epoch = training_state.epoch
        self.tracker = RateTracker()

        # Builds the local directory structure.
        self.experiment_directory = self._experiment_directory
        os.environ["WANDB_RUN_ID"] = self.WANDB_RUN_ID

        set_seed(self.get("seed"))  # handles random seed setting for everything but XLA

        if self.is_master_ordinal:
            if self.get("use_wandb"):
                self.initialize_wandb(resume=True)
            if self.is_multi_host:
                # GLOO for CPU comms, NCCL for GPU comms
                dist.init_process_group(backend=self.get("distributed/backend"))

    def _build_tasks(self, training_state: "TrainingState"):
        if self.get("run_training"):
            self._build_train_tasks(training_state)
        if self.get("run_evaluation"):
            self._build_eval_tasks(training_state)

    def _build_train_tasks(self, training_state: "TrainingState"):
        self.train_task = get_task(**self.get("dataset/kwargs"))
        self.train_loader = get_dataset(
            task=self.train_task,
            num_shards=self.num_shards,
            global_rank=self.global_rank,
            seed=self.get("seed"),
            device=self.device,
            **self.get("dataset/kwargs"),
        )

    def _build_eval_tasks(self, training_state: "TrainingState"):
        self.eval_tasks = get_eval_tasks(**self.get("dataset/kwargs"))

        self.eval_datasets = get_eval_datasets(
            tasks=self.eval_tasks,
            num_shards=self.num_shards,
            global_rank=self.global_rank,
            seed=self.get("seed"),
            device=self.device,
            **self.get("dataset/kwargs"),
        )

    def _build_model(self, training_state: "TrainingState"):
        self.tokenizer = get_default_vocabulary()

        self.get("model_config")["layer_norm_epsilon"] = float(
            self.get("model_config")["layer_norm_epsilon"]
        )
        self.get("model_config")["vocab_size"] = self.tokenizer.vocab_size
        model_config = T5Config.from_dict(self.get("model_config"),)

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
        return self.global_world_size > 1

    @property
    def total_batch_size(self):
        return (
            self.get("dataset/kwargs/batch_size")  # per device
            * self.local_world_size
            * self.global_world_size
        )

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

    def serialize(self):
        buffer = io.BytesIO()
        torch.save(self, buffer)
        return buffer.getvalue()

    @property
    def training_state(self, misc_attributes: dict = None):
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
        checkpoint_path = f"{self.experiment_directory}/Weights/model-{self.step}.pt"
        if xla_found:
            self.tpu_job.training_state = training_state
            self.tpu_job.upload()
        else:
            buffer = training_state.serialize()
            torch.save(buffer, checkpoint_path)

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
        rate = {
            "instantaneous it/s": tracker.rate(),
            "global it/s": tracker.global_rate(),
        }
        self.wandb_log(**rate)

        # Get a text example and log it
        input, label, pred, accuracy = self.decode_and_compute_accuracy(x, x_hat)
        self.table = wandb.Table(
            columns=["Step", "Accuracy", "Loss", "Input", "Label", "Predicted"]
        )
        self.table.add_data(step, accuracy, loss, input, label, pred)
        results = {
            "accuracy": accuracy,
            "examples": self.table,
            "train_loss": loss,
            "num_tokens": self.get("dataset/kwargs/max_seq_length")
            * self.total_batch_size
            * self.step,
        }
        self.wandb_log(**results)
        results.update(rate)
        print(results)

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

    @register_default_dispatch
    def run(self, training_state, tpu_job=None):
        if self.get("run_training"):
            self.train(training_state, tpu_job)
        if self.get("run_evaluation"):
            self.evaluate(training_state, tpu_job)

    def train(self, training_state, tpu_job=None):
        self._build(training_state, tpu_job)
        self.model.train()
        if xla_found:
            xm.master_print("starting to train")
        for x in self.train_loader:
            x_hat = self.model(**x)
            loss = self.loss(x_hat.logits, x["labels"])
            loss.backward()

            if (self.step + 1) % self.get("gradient_accumulation_steps", 1) == 0:
                reduce_gradients(xla_found, self.is_multi_host, self.optim)
                self.optim.step()
                self.optim.zero_grad()

            # Increment step count
            self.next_step()
            self.tracker.add(1)

            # if xla_found and self.is_master_ordinal:
            #     self.tpu_job.beat()

            if self.log_scalars_now and self.is_master_ordinal:
                self.log(x, x_hat, self.tracker)

            if self.checkpoint_now:
                self.checkpoint(self.training_state)
        return self.training_state

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():

            outputs = []
            for task_name, loader in self.eval_tasks.items():
                for x in loader:
                    del x["labels"]
                    predictions = self.model.generate(**x)
                    for pred in predictions:
                        outputs.extend([self.tokenizer.decode(pred.tolist())])

            cached_labels, cached_datasets, max_seq_length = get_targets_and_examples(
                self.eval_datasets, self.eval_tasks
            )

            for task in self.eval_tasks:
                # Extract the portion of decodes corresponding to this dataset
                dataset_size = len(cached_labels[task.name])
                predictions = [
                    task.postprocess_fn(d, example=ex)
                    for d, ex in zip(
                        outputs[:dataset_size], self.eval_datasets[task.name]
                    )
                ]

                for metric_fn in task.metric_fns:
                    targets = cached_labels[task.name]
                    metric_result = metric_fn(targets, predictions)
                    for metric_name, metric_value in metric_result.items():
                        tag = "eval/{}/{}".format(task.name, metric_name)
                        # print(f"{tag} {metric_value}")

    __call__ = train


class Nanny(WandBMixin, IOMixin, BaseExperiment):
    WANDB_ENTITY = "mweiss10"
    WANDB_PROJECT = "polytax-runs"

    def __init__(self):
        super(Nanny, self).__init__()
        self.auto_setup()

    def make_trainer(self):
        return Trainer(self)

    def _launch(
        self, trainer: "Trainer", training_state: "TrainingState",
    ) -> Union[TrainingState, JobStatus]:
        from wormulon.core import TPUJob, TPUCluster

        cluster = TPUCluster(
            self.WANDB_ENTITY, self.WANDB_PROJECT, **self.get("tpu/kwargs")
        )
        tpu = cluster.get_available_tpu(self.wandb_run_id)
        job = TPUJob(
            self.wandb_run_id,
            self.experiment_directory,
            self.get("bucket"),
            trainer,
            training_state,
            timeout=3600,
        )

        # update the configuration on wandb noting this tpu's name
        job.trainer._config["tpu_name"] = tpu.name
        job.trainer.update_wandb_config()
        # upload the job to GCP storage
        job.upload(overwrite=True)

        # Try to step into the job's directory and pull (in case it's old)
        root_path = "~/polytax/"
        tpu.ssh(f"cd {root_path} && git pull origin master")
        tpu.ssh(f"cd {root_path} && pip install -e .")
        tpu.ssh("pkill -9 python3")

        install_cmd = (
            f"cd ~/; git clone https://github.com/mweiss17/polytax.git; "
            f"pip install -e polytax[xla]; "
        )
        tpu.ssh(install_cmd)

        train_cmd = (
            f"python3 ~/polytax/src/polytax/train.py {self.get('bucket')} {job.path}"
        )
        tpu.ssh(train_cmd)

        # print("----------------")
        try:
            job_output = job.wait()
            if job.failed:
                raise RuntimeError(
                    f"Job has failed. The following object was returned: {job_output}"
                )
        finally:
            print("cleaning up job")
            job.clean_up()
        return job_output

    def launch(
        self, trainer: "Trainer", training_state: "TrainingState",
    ) -> "TrainingState":
        if self.get("use_tpu", False):
            # Try again if timed out
            num_attempts = 0
            max_num_attempts = self.get_arg("max_timeout_retries", default=0)
            job_output = self._launch(trainer, training_state)
        else:
            training_state = trainer(training_state)

        # Update self
        self._step = training_state.step
        self._epoch = training_state.epoch
        return training_state

    @register_default_dispatch
    def train(self):
        self.initialize_wandb(resume=False)
        # Build initial training state
        training_state = TrainingState.initial_state(step=self.step, epoch=self.epoch)
        # Setup the epoch runner
        trainer = self.make_trainer()
        # Run it
        self.launch(trainer, training_state)


def _mp_fn(index, tpu_job_buffer):
    tpu_job = torch.load(tpu_job_buffer)
    trainer = deepcopy(tpu_job.trainer)
    training_state = trainer(tpu_job.training_state, tpu_job)
    trainer.wandb_run.finish()
    xm.rendezvous("checking_out")
    sys.exit(0)


if __name__ == "__main__":
    if xla_found:

        parser = argparse.ArgumentParser()
        parser.add_argument("bucket", type=str)
        parser.add_argument("path", type=str)
        args = parser.parse_args()

        buffer = _read_blob_gcs(args.bucket, args.path)
        xmp.spawn(_mp_fn, args=(buffer,), nprocs=8)
        sys.exit(0)
    else:
        Nanny().run()
