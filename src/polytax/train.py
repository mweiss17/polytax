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
import signal
import wandb
import asyncio
import traceback
import seqio
import numpy as np
import tensorflow as tf
import itertools
import t5
import tensorflow_datasets as tfds
import time
import argparse
from copy import deepcopy
from collections import defaultdict
import torch
from wormulon.tpu.tpu_manager import TPUManager
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.tpu_job import TPUJob
import torch.distributed as dist
from tensorflow.python.ops.numpy_ops import np_config
from torch.utils.data import DataLoader

np_config.enable_numpy_behavior()
from transformers import (
    T5ForConditionalGeneration,
    SwitchForConditionalGeneration,
    MustForConditionalGeneration,
    T5Config,
    SwitchConfig,
    MustConfig,
    set_seed,
)
from polytax.data.dataset import IterableDataset
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
    import torch_xla.distributed.parallel_loader as pl

    xla_found = True
except Exception as e:
    print(f"XLA not found {e} \n")
    xla_found = False
    from utils.tracker import RateTracker, print_training_update, print_test_update

from polytax.data.utils import build_dataset, build_seqio_dataset
from polytax.data import listops
from polytax.data.dataset import ListOpsDataset
from wormulon.train_state import TrainState


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
        self.WANDB_ENTITY = nanny.WANDB_ENTITY or ""
        self.WANDB_PROJECT = nanny.WANDB_PROJECT or ""
        self.WANDB_RUN_ID = nanny.wandb_run_id or ""
        # Bit of a hack, but we set this here to have it uploaded to wandb.
        self.set("speedrun_meta/experiment_directory", self._experiment_directory)

    def _build_dist(self):
        self.device = xm.xla_device() if xla_found else torch.device("cpu")
        self.LOCAL_WORLD_SIZE = int(xm.xrt_world_size()) if xla_found else 1  # Devices per host
        self.GLOBAL_WORLD_SIZE = int(self.get("distributed/kwargs/world_size", 1))  # Number of hosts
        self.NUM_SHARDS = self.GLOBAL_WORLD_SIZE * self.LOCAL_WORLD_SIZE
        self.LOCAL_RANK = int(xm.get_ordinal()) if xla_found else 0
        self.GLOBAL_RANK = int(self.get("distributed/kwargs/rank", 0)) * self.LOCAL_WORLD_SIZE + self.LOCAL_RANK
        self.IS_LOCAL_MASTER = self.LOCAL_RANK == 0
        self.IS_GLOBAL_MASTER = self.GLOBAL_RANK == 0
        self.IS_MULTI_HOST = self.GLOBAL_WORLD_SIZE > 1

    def _build(self, train_state: "TrainState"):
        print(f"{self._config}")
        self._build_dist()
        self._build_general(train_state)
        self._build_tasks()
        self._build_model(train_state)
        self._build_optimizer(train_state)

    def _build_general(self, train_state: "TrainState"):
        self._step = train_state.step
        self._epoch = train_state.epoch
        self.tracker = RateTracker()

        # Builds the local directory structure.
        self.experiment_directory = self._experiment_directory
        os.environ["WANDB_RUN_ID"] = train_state.misc_attributes.get("wandb_run_id", "")
        self.bucket = Bucket(self.get("tpu/kwargs/bucket"))

        set_seed(self.get("seed"))  # handles random seed setting for everything but XLA
        if self.IS_GLOBAL_MASTER:
            if self.get("use_wandb"):
                self.initialize_wandb(resume=True)
        if self.IS_MULTI_HOST and self.IS_LOCAL_MASTER:
            # GLOO for CPU comms, NCCL for GPU comms
            print(f"initializing dist process group: distributed/kwargs={self.get('distributed/kwargs')}")
            dist.init_process_group(**self.get("distributed/kwargs"))
            print("distributed initialized")

    def _build_tasks(self):
        if xla_found and not self.IS_LOCAL_MASTER:
            xm.rendezvous("download_only_once")
        if self.get("run_training"):
            self._build_train_tasks()

        if self.get("run_evaluation"):
            self._build_eval_tasks()
        if self.get("dataset_name") == "listops":
            self.tokenizer = tfds.deprecated.text.TokenTextEncoder.load_from_file(f"{os.getcwd()}/data/listops_train_encoder.json")
        else:
            self.tokenizer = get_default_vocabulary()

        if xla_found and self.IS_LOCAL_MASTER:
            xm.rendezvous("download_only_once")

    @property
    def seq_len(self):
        seq_len = {
            "inputs": self.get("input_seq_len"),
            "targets": self.get("target_seq_len"),
        }
        return seq_len

    def _build_loader(self, dataset, cycle=False):
        loader = DataLoader(dataset, batch_size=None, pin_memory=True, num_workers=0)
        if xla_found:
            loader = pl.MpDeviceLoader(loader, self.device)
        if cycle:
            loader = itertools.cycle(loader)
        else:
            loader = iter(loader)
        return loader

    def _build_train_tasks(self):
        mixture = seqio.get_mixture_or_task(self.get("dataset_name"))
        dataset = build_seqio_dataset(mixture, self.seq_len, "train", seed=self.get("seed"), pack=True)
        tf_dataset, dataset = build_dataset(dataset, self.get("batch_size"), self.GLOBAL_RANK, self.NUM_SHARDS, self.get("use_iterable_ds"), device=self.device)
        self.train_loader = self._build_loader(dataset, cycle=True)

    def _build_eval_tasks(self):
        mixture = seqio.get_mixture_or_task(self.get("dataset_name"))
        tasks = seqio.get_subtasks(mixture)
        eval_tasks = seqio.evaluation.get_valid_eval_tasks(tasks, self.get("val_split_name"))
        self.eval_datasets = {}
        for task in eval_tasks:
            ds = build_seqio_dataset(task, self.seq_len, self.get("val_split_name"), seed=self.get("seed"), pack=False)
            tf_ds, ds = build_dataset(ds, self.get("batch_size"), self.GLOBAL_RANK, self.NUM_SHARDS, use_iterable_ds=True, device=self.device)
            self.eval_datasets[task] = (tf_ds, self._build_loader(ds, cycle=False))

    def _build_model(self, train_state: "TrainState"):
        self.get("model_config")["layer_norm_epsilon"] = float(
            self.get("model_config")["layer_norm_epsilon"]
        )
        self.get("model_config")["vocab_size"] = self.tokenizer.vocab_size
        model_config = self.get("model_config").copy()
        model_config["LOCAL_WORLD_SIZE"] = self.LOCAL_WORLD_SIZE
        model_config["GLOBAL_WORLD_SIZE"] = self.GLOBAL_WORLD_SIZE
        model_config["LOCAL_RANK"] = self.LOCAL_RANK
        model_config["GLOBAL_RANK"] = self.GLOBAL_RANK
        model_config["NUM_SHARDS"] = self.NUM_SHARDS
        model_config["xla_found"] = xla_found
        model_config["seed"] = self.get("seed")
        if "switch" in self.get("model_config/model_type"):
            model_config = SwitchConfig.from_dict(model_config)
            self.model = SwitchForConditionalGeneration(model_config)
        elif "must" in self.get("model_config/model_type"):
            model_config = MustConfig.from_dict(model_config)
            self.model = MustForConditionalGeneration(model_config)
        else:
            model_config = T5Config.from_dict(self.get("model_config"))
            self.model = T5ForConditionalGeneration(model_config)
        train_state.load_in_model(self.model)
        self.model.to(self.device)

    def _build_optimizer(self, train_state: "TrainState"):
        self.optim = eval(self.get("optim/name"))(
            self.model.parameters(), **self.get("optim/kwargs")
        )
        train_state.load_in_optims(optim=self.optim,)

    @property
    def total_batch_size(self):
        return self.get("batch_size") * self.LOCAL_WORLD_SIZE * self.GLOBAL_WORLD_SIZE

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
    def train_state(self):
        return TrainState(
            step=self.step,
            epoch=self.epoch,
            model_state_dict=self.model.state_dict(),
            losses_state_dict=self.losses_state_dict,
            optims_state_dict=self.optims_state_dict,
            schedulers_state_dict=self.schedulers_state_dict,
            misc_attributes={"wandb_run_id": self.wandb_run_id},
        )

    def checkpoint(self):
        # this needs to happen on each TPU core because it has xm.save has a rendezvous in it
        buffer = self.train_state.serialize()
        if xla_found and self.IS_GLOBAL_MASTER:
            # only push to storage if you are the master ordinal
            path = f"{self.experiment_directory}/trainstate/{self.get('dataset_name')}-{self.step}.pt"
            self.bucket.upload(path, buffer, overwrite=True)
        elif not xla_found:
            path = f"{self.experiment_directory}/Weights/trainstate-{self.step}.pt"
            torch.save(buffer, path)

    def decode_and_compute_accuracy(self, x, x_hat, compute_examples=False):
        sample_id = 0
        all_labels = x['labels']
        all_preds = x_hat.logits.argmax(axis=2)
        all_inputs = x["input_ids"]

        input_list = all_inputs[sample_id].cpu().numpy().tolist()
        label_list = all_labels[sample_id].cpu().numpy().tolist()
        pred_list = all_preds[sample_id].cpu().numpy().tolist()

        # compute_accuracy
        correct = all_preds.eq(all_labels).sum()
        num_extra_ids = (all_labels == -100).sum()
        accuracy = 100.0 * correct / (all_labels.nelement() - num_extra_ids)
        if compute_examples:
            # Prepare to decode the labels
            extra_id = 32099
            ids_to_replace_in_preds = []
            for idx, label in enumerate(label_list):
                if label == -100:
                    ids_to_replace_in_preds.append(idx)
                    label_list[idx] = extra_id
                    extra_id -= 1

            extra_id = 32099
            for idx, pred in enumerate(pred_list):
                if idx in ids_to_replace_in_preds:
                    pred_list[idx] = extra_id
                    extra_id -= 1

            input = self.tokenizer.decode(input_list)
            label = self.tokenizer.decode(label_list)
            pred = self.tokenizer.decode(pred_list)
            return input, label, pred, accuracy.item()
        return None, None, None, accuracy.item()

    def _log_train(self, x, x_hat):
        if xla_found:
            def metsumm(stepno=''):
                import torch_xla.debug.metrics as met
                x = met.metrics_report().split('\n')
                for i, line in enumerate(x):
                    if 'CompileTime' in line or 'aten::' in line:
                        key = line.split()[-1]
                        value = x[i + 1].split()[-1]
                        print(
                            'step {}, key {}, value {}'.format(
                                stepno, key, value
                            )
                        )

            metsumm(stepno=self.step)

        if xla_found:
            import torch_xla.debug.metrics as met

            xm.master_print(met.metrics_report())

        loss = x_hat.loss.item()
        try:
            aux_loss = x_hat.aux_loss.item()
        except AttributeError:
            aux_loss = 0.0
        # Get a text example and log it
        input, label, pred, accuracy = self.decode_and_compute_accuracy(x, x_hat)
        results = {
            "step": self.step,
            "label": label,
            "pred": pred,
            "train_accuracy": accuracy,
            "input_examples": input,
            "train_task_loss": loss - aux_loss,
            "train_total_loss": loss,
            "train_aux_loss": aux_loss,
            "num_tokens": self.get("input_seq_len")
            * self.total_batch_size
            * self.step,
            "instantaneous it/s": self.tracker.rate(),
            "global it/s": self.tracker.global_rate(),
        }
        # Return if we don't have wandb
        if self.get("use_wandb") and self.IS_GLOBAL_MASTER:
            self.wandb_log(**results)
            print_training_update(self.device, self.step, loss, self.tracker.rate(), self.tracker.global_rate())
        elif xla_found and self.IS_GLOBAL_MASTER:
            xm.master_print(results)
            self.bucket.touch(self.experiment_directory + "/heartbeat")
            print_training_update(self.device, self.step, loss, self.tracker.rate(), self.tracker.global_rate())
        elif not xla_found:
            print(results)
            print_training_update(self.device, self.step, loss, self.tracker.rate(), self.tracker.global_rate())

    def _log_eval(self, all_preds, all_examples):
        results = {
            "step": self.step,
            "instantaneous it/s": self.tracker.rate(),
            "global it/s": self.tracker.global_rate(),
        }
        metric_results = {}
        for task in self.eval_datasets.keys():
            # Extract the portion of decodes corresponding to this dataset
            predictions = []
            text_preds = []
            targets = []
            text_targets = []
            for i, (preds, examples) in enumerate(zip(all_preds[task.name], all_examples[task.name])):
                preds = preds.cpu().tolist()
                for j, pred in enumerate(preds):
                    target = examples['labels'][j].tolist()
                    text_target = self.tokenizer.decode(target)
                    text_pred = self.tokenizer.decode(pred)
                    text_preds.append(text_pred)
                    predictions.append(task.postprocess_fn(text_pred, example=examples['input_ids'][j]))
                    targets.append(task.postprocess_fn(text_target, example=examples['input_ids'][j], is_target=True))
                    text_targets.append(text_target)

            for metric_fn in task.metric_fns:
                metric_result = metric_fn(targets, predictions)
                if np.isnan(list(metric_result.values())[0]):
                    metric_result[list(metric_result.keys())[0]] = 0.
                for metric_name, metric_value in metric_result.items():
                    tag = "eval/{}/{}".format(task.name, metric_name)
                    metric_results[tag] = metric_value
            metric_results[f"eval/{task.name}/text_preds"] = text_preds[:10]
            metric_results[f"eval/{task.name}/text_targets"] = text_targets[:10]


        results["eval_accuracy"] = metric_results
        if self.get("use_wandb") and self.IS_GLOBAL_MASTER:
            self.wandb_log(**results)
        else:
            print(results, flush=True)

    def step_gradients(self):
        if xla_found and self.IS_MULTI_HOST:
            gradients = xm._fetch_gradients(self.optim)
            xm.all_reduce('sum', gradients, scale=1.0 / self.LOCAL_WORLD_SIZE)

            cpu_grads = []
            for grad in gradients:
                cpu_grads.append(grad.cpu())
            if self.IS_LOCAL_MASTER:
                reduced_grads = []
                for grad in cpu_grads:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    grad /= self.GLOBAL_WORLD_SIZE
                    reduced_grads.append(grad)
                grads = [grad.to(self.device) for grad in reduced_grads]
            else:
                grads = [grad.zero_() for grad in gradients]
        if xla_found:
            loss = xm.optimizer_step(self.optim, barrier=True)
        else:
            self.optim.step()
        self.optim.zero_grad()

    @register_default_dispatch
    def run(self, train_state):
        self._build(train_state)
        if self.get("run_training"):
            print("Training...")
            while True:
                for i, x in enumerate(self.train_loader):

                    self.train(x)
                    if self.get("run_evaluation") and self.step % self.get("eval_every") == 0:
                        print("Evaluating...")
                        self.evaluate()
                if self.get("num_train_steps") == self.step:
                    print("Done Training")
                    break

        if xla_found:
            self.finish()
        return train_state

    def train(self, x):
        self.model.train()
        self.model.to(self.device)
        x_hat = self.model(**x)
        x_hat.loss.backward()

        self.step_gradients()
        if self.log_scalars_now:
            # If XLA is found, then we are on TPU and we should use a closure to increase efficiency
            if xla_found:
                xm.add_step_closure(
                    self._log_train, args=(x, x_hat), run_async=True
                )
            # Otherwise just call the function to log directly
            else:
                self._log_train(x, x_hat)

        if self.checkpoint_now:
            self.checkpoint()

        self.next_step()
        self.tracker.add(1)
        if xla_found:
            xm.mark_step()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            all_examples = {}
            all_preds = {}
            for task, (ds, loader) in self.eval_datasets.items():
                examples = []
                preds = []
                while True:
                    try:
                        x = next(loader)
                    except StopIteration:
                        rebuilt_ds = IterableDataset(ds.as_numpy_iterator(), self.device)
                        self.eval_datasets[task] = (ds, self._build_loader(rebuilt_ds))
                        break
                    examples.append(x.copy())
                    del x["labels"]
                    outputs = self.model(**x)
                    out = outputs.logits.argmax(2)
                    preds.append(out)
                all_examples[task.name] = examples
                all_preds[task.name] = preds
            # If XLA is found, then we are on TPU and we should use a closure to increase efficiency
            if xla_found:
                xm.add_step_closure(
                    self._log_eval, args=(all_preds, all_examples), run_async=True
                )
            # Otherwise just call the function to log directly
            else:
                self._log_eval(all_preds, all_examples)


    def finish(self):
        if self.IS_MASTER_ORDINAL:
            wandb.finish()
        xm.rendezvous("finished run.")

    __call__ = run


class Nanny(WandBMixin, IOMixin, BaseExperiment):
    WANDB_ENTITY = "mweiss10"
    WANDB_PROJECT = "polytax-exps-2"

    def __init__(self):
        super(Nanny, self).__init__()
        self.auto_setup()
        self.jobs = []
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def setup_wandb(self):
        wandb_run_id = ""
        if self.get("use_wandb"):
            self.initialize_wandb(resume=False)
            wandb_run_id = self.wandb_run.id
            wandb.finish()
        return wandb_run_id

    async def run(self):
        if self.get("use_tpu"):
            manager = TPUManager(**self.get("tpu/kwargs"))
            tpus = manager.get_tpus(self.get("distributed/kwargs/world_size"))
            jobs = []
            for i, tpu in enumerate(tpus):
                print(f"Dibs on TPU: {tpu}, creating job-{i}")
                trainer = Trainer(self)
                env_stmts = trainer.get("job/kwargs/env_stmts")
                env_stmts.append(f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};")
                trainer.set('job/kwargs/env_stmts', env_stmts)
                if len(tpus) > 1:
                    trainer.set("distributed/kwargs/init_method", f"tcp://{tpus[0].ip_address}:2345")
                trainer.set('distributed/kwargs/rank', i)
                jobs.append(TPUJob(trainer, tpu))

        train_state = TrainState.initial_state(step=self.step, epoch=self.epoch,
                                               misc_attributes={"wandb_run_id": self.setup_wandb()})
        if self.get("use_tpu"):
            for job, tpu in zip(jobs, tpus):
                job.arm(train_state, restart=False)
                try:
                    future = job.submit()
                except Exception as e:
                    print("consuming exception")
                self.jobs.append((future, job))

            state = "running"
            while state == "running":
                for future, job in self.jobs:
                    state = future.done() or state
                    out = job.outbuffer.getvalue()
                    err = job.errbuffer.getvalue()
                    if out:
                        print(f"job-{job.trainer.get('distributed/kwargs/rank')}: {out}")
                        job.outbuffer.seek(0)
                        job.outbuffer.truncate()
                    if err:
                        print(f"job-{job.trainer.get('distributed/kwargs/rank')}: {err}")
                        job.errbuffer.seek(0)
                        job.errbuffer.truncate()

                    # if job.failed:
                    #     state = "failed"
                    # if job.died:
                    #     state = "died"
                await asyncio.sleep(1)
            for future, job in self.jobs:
                print(f"Job {state}, restarting from latest train state, cleaning up jobs then restarting")
                job.clean_up()
                # self.launch(job.trainer, train_state)

        else:
            trainer = Trainer(self)
            trainer(train_state)

        # Update self
        self._step = train_state.step
        self._epoch = train_state.epoch

    def exit_gracefully(self, signum, frame):
        print("Exiting gracefully")
        for future, job in self.jobs:
            job.clean_up()
            print(f"{job.tpu} is now available")
            print(f"exited {job.wandb_run.get_url()}")
        sys.exit(0)




if __name__ == "__main__":
    if xla_found:
        print("XLA found")

        parser = argparse.ArgumentParser()
        parser.add_argument("bucket", type=str)
        parser.add_argument("path", type=str)
        args = parser.parse_args()
        from wormulon.tpu.tpu_runner import JobRunner

        JobRunner(args.bucket, args.path).run()
    else:
        asyncio.run(Nanny().run())
