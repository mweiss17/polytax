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
import numpy as np
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

np_config.enable_numpy_behavior()
from transformers import (
    T5ForConditionalGeneration,
    SwitchForConditionalGeneration,
    T5Config,
    SwitchConfig,
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
)
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
        self.IS_MASTER_ORDINAL = self.LOCAL_RANK == 0
        self.IS_MULTI_HOST = self.GLOBAL_WORLD_SIZE > 1
        print(f"LOCAL_WORLD_SIZE {self.LOCAL_WORLD_SIZE}, GLOBAL_WORLD_SIZE {self.GLOBAL_WORLD_SIZE}, LOCAL_RANK {self.LOCAL_RANK}, GLOBAL_RANK {self.GLOBAL_RANK}, IS_MASTER_ORDINAL {self.IS_MASTER_ORDINAL}, IS_MULTI_HOST {self.IS_MULTI_HOST}")

    def _build(self, train_state: "TrainState"):
        print(f"{self._config}")
        self._build_dist()
        self._build_general(train_state)
        self._build_tasks(train_state)
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
        print(f"Using {self.GLOBAL_WORLD_SIZE} hosts. Each host has {self.LOCAL_WORLD_SIZE} devices. IS_MULTI_HOST={self.IS_MULTI_HOST}, and IS_MASTER_ORDINAL={self.IS_MASTER_ORDINAL}")
        if self.IS_MASTER_ORDINAL:
            if self.get("use_wandb"):
                self.initialize_wandb(resume=True)
            if self.IS_MULTI_HOST:
                # GLOO for CPU comms, NCCL for GPU comms
                print(f"initializing dist process group: distributed/kwargs={self.get('distributed/kwargs')}")
                dist.init_process_group(**self.get("distributed/kwargs"))
                print("distributed initialized")

    def _build_tasks(self, train_state: "TrainState"):

        if xla_found and not self.IS_MASTER_ORDINAL:
            xm.rendezvous("download_only_once")

        if self.get("run_training"):
            self._build_train_tasks(train_state)
        if self.get("run_evaluation"):
            self._build_eval_tasks(train_state)

        if xla_found and self.IS_MASTER_ORDINAL:
            xm.rendezvous("download_only_once")

    def _build_train_tasks(self, train_state: "TrainState"):
        self.train_task = get_task(**self.get("dataset/kwargs"))
        self.train_loader = get_dataset(task=self.train_task, seed=self.get("seed"), GLOBAL_RANK=self.GLOBAL_RANK, NUM_SHARDS=self.NUM_SHARDS, device=self.device, **self.get("dataset/kwargs"),)

    def _build_eval_tasks(self, train_state: "TrainState"):
        self.eval_tasks = get_eval_tasks(**self.get("dataset/kwargs"))

        self.eval_datasets = get_eval_datasets(
            tasks=self.eval_tasks,
            seed=self.get("seed"),
            GLOBAL_RANK=self.GLOBAL_RANK,
            NUM_SHARDS=self.NUM_SHARDS,
            device=self.device,
            **self.get("dataset/kwargs"),
        )

    def _build_model(self, train_state: "TrainState"):
        self.tokenizer = get_default_vocabulary()

        self.get("model_config")["layer_norm_epsilon"] = float(
            self.get("model_config")["layer_norm_epsilon"]
        )
        self.get("model_config")["vocab_size"] = self.tokenizer.vocab_size

        if "switch" in self.get("model_config/model_type"):
            model_config = self.get("model_config").copy()
            model_config["LOCAL_WORLD_SIZE"] = self.LOCAL_WORLD_SIZE
            model_config["GLOBAL_WORLD_SIZE"] = self.GLOBAL_WORLD_SIZE
            model_config["LOCAL_RANK"] = self.LOCAL_RANK
            model_config["GLOBAL_RANK"] = self.GLOBAL_RANK
            model_config["NUM_SHARDS"] = self.NUM_SHARDS
            model_config["xla_found"] = xla_found
            model_config = SwitchConfig.from_dict(model_config)

            self.model = SwitchForConditionalGeneration(model_config)
        else:
            model_config = T5Config.from_dict(self.get("model_config"))
            self.model = T5ForConditionalGeneration(model_config)
        train_state.load_in_model(self.model)
        self.model.to(self.device)

    def _build_optimizer(self, train_state: "TrainState"):
        # No need to specify learning rate in Adafactor: https://arxiv.org/pdf/1804.04235.pdf
        self.optim = eval(self.get("optim/name"))(
            self.model.parameters(), **self.get("optim/kwargs")
        )
        train_state.load_in_optims(optim=self.optim,)

    @property
    def total_batch_size(self):
        return self.get("dataset/kwargs/batch_size") * self.LOCAL_WORLD_SIZE * self.GLOBAL_WORLD_SIZE

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
        if xla_found and self.IS_MASTER_ORDINAL:
            # only push to storage if you are the master ordinal
            path = f"{self.experiment_directory}/trainstate/{self.get('dataset/kwargs/name')}-{self.step}.pt"
            self.bucket.upload(path, buffer, overwrite=True)
        else:
            path = f"{self.experiment_directory}/Weights/trainstate-{self.step}.pt"
            torch.save(buffer, path)

    def decode_and_compute_accuracy(self, x, x_hat):
        sample_id = 0
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

    def _log_train(self, x, x_hat):
        loss = x_hat.loss.detach()

        # Print to console
        print_training_update(self.device, self.step, loss, self.tracker.rate(), self.tracker.global_rate())

        # Return if we don't have wandb
        if not self.get("use_wandb"):
            return

        # Get a text example and log it
        input, label, pred, accuracy = self.decode_and_compute_accuracy(x, x_hat)
        results = {
            "step": self.step,
            "train_accuracy": accuracy,
            "input_examples": input,
            "label": label,
            "pred": pred,
            "train_loss": loss,
            "num_tokens": self.get("dataset/kwargs/input_seq_len")
            * self.total_batch_size
            * self.step,
            "instantaneous it/s": self.tracker.rate(),
            "global it/s": self.tracker.global_rate(),
        }

        self.wandb_log(**results)
        if xla_found:
            self.bucket.touch(self.experiment_directory + "/heartbeat")

        # print(results, flush=True)

    def _log_eval(self, all_preds, examples):
        results = {
            "step": self.step,
            "instantaneous it/s": self.tracker.rate(),
            "global it/s": self.tracker.global_rate(),
        }
        metric_results = {}
        for task in self.eval_tasks:
            # Extract the portion of decodes corresponding to this dataset
            predictions = []
            text_preds = []
            targets = []
            text_targets = []
            ex = examples[task.name][0]
            task_preds = all_preds[task.name][0]
            for i, preds in enumerate(task_preds):
                predictions.append(task.postprocess_fn(preds, example=ex))
                text_preds.append(self.tokenizer.decode([preds]))
                text_target = self.tokenizer.decode(ex['labels'][i].tolist())
                if task.name == "glue_qqp_v002" and text_target == "not":
                    text_target = "not_duplicate"
                if task.name == "glue_mrpc_v002" and text_target == "not":
                    text_target = "not_equivalent"
                targets.append(task.postprocess_fn(text_target, example=ex['input_ids'][i], is_target=True))
                text_targets.append(text_target)
            for metric_fn in task.metric_fns:
                try:
                    # print(f"targets: {targets[0]}, predicted: {predictions[0]}")
                    metric_result = metric_fn(targets, predictions)
                    if np.isnan(list(metric_result.values())[0]):
                        metric_result[list(metric_result.keys())[0]] = 0.
                except Exception as e:
                    print(f"Metric failed with error: {e}")
                    breakpoint()
                    metric_result = 0.0
                for metric_name, metric_value in metric_result.items():
                    tag = "eval/{}/{}".format(task.name, metric_name)
                    metric_results[tag] = metric_value
        results["mean_accuracy"] = np.mean(list(metric_results.values()))
        results["eval_accuracy"] = metric_results
        self.wandb_log(**results)

    def loss(self, logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def _fetch_gradients(self, device=None):
        gradients = []
        if self.IS_MASTER_ORDINAL:
            print("Fetching gradients")
        for param_group in self.optim.__getstate__()['param_groups']:
            for group, params in param_group.items():
                if group == 'params':
                    for p in params:
                        if isinstance(p, torch.Tensor) and p.grad is not None:
                            if device == torch.device("cpu"):
                                print(f"detaching gradients from {p.size()}")
                                gradients.append(p.grad.data.cpu().numpy())
                            else:
                                gradients.append(p.grad.data)
        return gradients

    def step_gradients(self):
        if xla_found:
            gradients = self._fetch_gradients(device=self.device)
            xm.all_reduce('sum', gradients, scale=1.0 / self.LOCAL_WORLD_SIZE)

            print(
                f"LOCAL_WORLD_SIZE {self.LOCAL_WORLD_SIZE}, GLOBAL_WORLD_SIZE {self.GLOBAL_WORLD_SIZE}, LOCAL_RANK {self.LOCAL_RANK}, GLOBAL_RANK {self.GLOBAL_RANK}, IS_MASTER_ORDINAL {self.IS_MASTER_ORDINAL}, IS_MULTI_HOST {self.IS_MULTI_HOST}")
            if self.IS_MULTI_HOST:
                xm.rendezvous("in-multi-host")
                if self.IS_MASTER_ORDINAL:
                    print("is master")
                    gradients = self._fetch_gradients(device=torch.device("cpu"))
                    for grad in gradients:
                        print("for grad in grad")
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                        print("all reduce")
                        grad /= self.GLOBAL_WORLD_SIZE
                        print("to device")
                        grad = grad.to(self.device)

                    print("reduced.")
                    xm.rendezvous("dist-reduced")
                else:
                    print("is not master")
                    xm.rendezvous("dist-reduced")
            # if self.IS_MULTI_HOST:
            #     if self.IS_MASTER_ORDINAL:
            #         print(f"IS_MASTER_ORDINAL, reducing gradients, dist: {dist.get_rank()} / {dist.get_world_size()}")
            #         grad = torch.ones(10)
            #         dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            #         grad.to(self.device)
            #         # cpu_grads = []
            #         # for gradient in gradients:
            #         #     print("reducing...")
            #         #     cpu_grad = gradient.cpu()
            #         #     dist.all_reduce(cpu_grad.div_(self.GLOBAL_WORLD_SIZE), op=dist.ReduceOp.SUM)
            #         #     cpu_grads.append(cpu_grad)
            #         #     print("reduced...")
            #         print(f"dist reduced succesfully")
            #
            #         # gradients = [cpu_grad.to(self.device) for cpu_grad in cpu_grads]
            #         xm.rendezvous("reducing")
            #         # xm.all_reduce('sum', gradients, scale=1.0)
            #     if not self.IS_MASTER_ORDINAL:
            #         self.optim.zero_grad()
            #         xm.rendezvous("reducing")
            #
            #         print(f"not master ordinal, reducing gradients")
            #         # xm.all_reduce('sum', gradients, scale=1.0)

        self.optim.step()
        xm.mark_step()
        self.optim.zero_grad()


    @register_default_dispatch
    def run(self, train_state):
        self._build(train_state)
        if self.get("run_training"):
            for x in self.train_loader:
                if self.get("num_train_steps") == self.step:
                    break
                self.train(x)
                if self.get("run_evaluation") and self.step % self.get("eval_every") == 0:
                    self.evaluate()
        if xla_found:
            self.finish()
        return train_state

    def train(self, x):
        self.model.train()

        x_hat = self.model(**x)
        loss = self.loss(x_hat.logits, x["labels"])
        loss.backward()

        if (self.step + 1) % self.get("gradient_accumulation_steps", 1) == 0:
            self.step_gradients()

        # Increment step count
        self.next_step()
        self.tracker.add(1)

        if self.log_scalars_now and self.IS_MASTER_ORDINAL:
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

    def evaluate(self, one_sample=True):
        self.model.eval()
        with torch.no_grad():
            examples = defaultdict(list)
            preds = defaultdict(list)
            for task_name, loader in self.eval_datasets.items():
                for x in loader:
                    examples[task_name].append(x.copy())
                    try:
                        del x["labels"]
                    except Exception:
                        print("no labels found")
                    outputs = self.model(**x)
                    out = outputs.logits.argmax(2).cpu()
                    out = out[:, 1].tolist()
                    preds[task_name].append(out)

                    if one_sample:
                        break
            # If XLA is found, then we are on TPU and we should use a closure to increase efficiency
            if xla_found:
                xm.add_step_closure(
                    self._log_eval, args=(preds, examples), run_async=True
                )
            # Otherwise just call the function to log directly
            else:
                self._log_eval(preds, examples)


    def finish(self):
        if self.IS_MASTER_ORDINAL:
            wandb.finish()
        xm.rendezvous("finished run.")

    __call__ = run


class Nanny(WandBMixin, IOMixin, BaseExperiment):
    WANDB_ENTITY = "mweiss10"
    WANDB_PROJECT = "polytax-runs"

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
            wandb_run_id = self.wandb_run_id
            wandb.finish()
        return wandb_run_id

    async def run(self):
        train_state = TrainState.initial_state(step=self.step, epoch=self.epoch,
                                               misc_attributes={"wandb_run_id": self.setup_wandb()})
        if self.get("use_tpu"):
            manager = TPUManager(**self.get("tpu/kwargs"))
            tpus = manager.get_tpus(self.get("distributed/kwargs/world_size"))
            for i in range(self.get("distributed/kwargs/world_size")):
                print(f"creating job-{i}")
                trainer = Trainer(self)
                env_stmts = trainer.get("job/kwargs/env_stmts")
                env_stmts.append(f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};")
                trainer.set('job/kwargs/env_stmts', env_stmts)
                trainer.set("distributed/kwargs/init_method", f"tcp://{tpus[0].ip_address}:2345")
                trainer.set('distributed/kwargs/rank', i)
                job = TPUJob(trainer, train_state, tpus[i])
                future = job.submit()
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

            if state == "failed" or state == "died":
                for future, job in jobs:
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
