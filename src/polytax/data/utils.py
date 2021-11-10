import t5.data.mixtures  # pylint: disable=unused-import
import torch
from t5.models import mesh_transformer
import t5
import seqio
import functools
import tensorflow as tf
import itertools
import tensorflow_datasets as tfds
from typing import Iterator, Dict
from tensorflow.python.data.ops.dataset_ops import ParallelMapDataset

try:
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pl = None

from polytax.data import dataset  # pylint: disable=unused-import
from polytax.data import tasks  # pylint: disable=unused-import
from polytax.data.dataset import IterableDataset, MapDataset


""" BUILD TASKS AND DATASETS """


def get_validation_tasks(name: str, split: str,) -> Iterator:
    """Returns the validation tasks."""

    mixture = t5.data.get_mixture_or_task(name)
    tasks = t5.data.get_subtasks(mixture)
    tasks = seqio.evaluation.get_valid_eval_tasks(tasks, split)
    return tasks


def build_seqio_dataset(task, sequence_length, split, seed=1, num_epochs=1):
    dataset = seqio.get_dataset(
        task.name,
        task_feature_lengths=sequence_length,
        dataset_split=split,
        use_cached=False,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        feature_converter=seqio.EncDecFeatureConverter(pack=True),
    )
    return dataset


def build_iterable_dataset(
    dataset, batch_size, device, shard_idx, num_shards, cycle=True
) -> Iterator:
    """Builds an iterable dataset."""
    dataset = dataset.shard(num_shards=num_shards, index=shard_idx)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    dataset = IterableDataset(dataset)
    if cycle:
        dataset = itertools.cycle(dataset)
    if pl:
        dataset = iter(pl.MpDeviceLoader(iter(dataset), device))
    else:
        dataset = iter(dataset)
    return dataset


def build_map_ds(ds: ParallelMapDataset):
    ds = ds.as_numpy_iterator()
    ds = MapDataset(ds)
    return ds


def get_pretrain_dataset(
    num_shards: int,
    shard_index: int,
    seed: int,
    name: str,
    per_device_batch_size: int,
    max_seq_length: int,
    split: str,
    device: torch.device,
) -> Iterator:
    """Returns a dataset for pretraining."""

    # determine maximum sequence length to use
    sequence_length = {
        "inputs": max_seq_length,
        "targets": int(max_seq_length / 4),
    }
    task = t5.data.get_mixture_or_task(name)

    dataset = build_seqio_dataset(task, sequence_length, split, seed=seed, num_epochs=1)
    dataset = build_iterable_dataset(
        dataset, per_device_batch_size, device, shard_index, num_shards
    )
    return dataset


def get_validation_datasets(
    tasks: Dict[str, seqio.Task],
    batch_size: int,
    max_seq_length: int,
    split: str,
    device: torch.device,
    iterable_dataset: bool = False,
) -> Iterator:
    """Returns a dataset for validation."""

    # determine maximum sequence length to use
    seq_len = {
        "inputs": max_seq_length,
        "targets": int(max_seq_length / 4),
    }

    datasets = {}

    for task in tasks:
        if iterable_dataset:
            ds = build_seqio_dataset(task, seq_len, split, num_epochs=1)
            ds = build_iterable_dataset(ds, batch_size, seq_len, split, device)
        else:
            ds = build_seqio_dataset(task, seq_len, split, num_epochs=1)
            ds = build_map_ds(ds)
        datasets[task.name] = ds
    return datasets


def get_targets_and_examples(datasets, tasks):
    cached_targets = {}
    cached_task_datasets = {}
    sequence_dims = {}

    # TODO fixme when adding superGLUE
    max_sequence_length = {"input_ids": 0, "labels": 0}
    for task in tasks:
        targets = []

        for batch in datasets[task.name]:
            for k in max_sequence_length:
                sequence_dim = sequence_dims.get(k, 0)
                sequence_length = batch[k].shape[sequence_dim]
                max_sequence_length[k] = max(max_sequence_length[k], sequence_length)

            # Create list of postprocessed targets
            for ex in batch["labels"]:
                target = task.output_features["targets"].vocabulary.decode(ex.tolist())
                targets.append(task.postprocess_fn(target, example=ex, is_target=True))

        cached_targets[task.name] = targets
        cached_task_datasets[task.name] = datasets[task.name]
    return cached_targets, cached_task_datasets, max_sequence_length


def run_evaluation(model, loaders, tasks, tokenizer):
    outputs = []
    for task_name, loader in loaders.items():
        for x in loader:
            del x["labels"]
            predictions = model.generate(**x)
            for pred in predictions:
                outputs.extend([tokenizer.decode(pred.tolist())])

    cached_labels, cached_datasets, max_seq_length = get_targets_and_examples(
        loaders, tasks
    )

    for task in tasks:
        # Extract the portion of decodes corresponding to this dataset
        dataset_size = len(cached_labels[task.name])
        predictions = [
            task.postprocess_fn(d, example=ex)
            for d, ex in zip(outputs[:dataset_size], loaders[task.name])
        ]

        for metric_fn in task.metric_fns:
            targets = cached_labels[task.name]
            metric_result = metric_fn(targets, predictions)
            for metric_name, metric_value in metric_result.items():
                tag = "eval/{}/{}".format(task.name, metric_name)
                print(f"{tag} {metric_value}")
