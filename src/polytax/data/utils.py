import t5
import t5.data.mixtures  # pylint: disable=unused-import
import seqio
import itertools
from typing import Iterator, Dict
import torch
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import ParallelMapDataset

try:
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pl = None

from polytax.data import dataset  # pylint: disable=unused-import
from polytax.data import tasks  # pylint: disable=unused-import
from polytax.data.dataset import IterableDataset, MapDataset

""" BUILD TASKS AND DATASETS """


def get_dataset(
    task: seqio.Task,
    seed: int,
    batch_size: int,
    input_seq_len: int,
    target_seq_len: int,
    split: str,
    GLOBAL_RANK,
    NUM_SHARDS,
    device,
    **kwargs,
) -> Iterator:
    """Returns a dataset for pretraining."""

    # determine maximum sequence length to use
    seq_len = {
        "inputs": input_seq_len,
        "targets": target_seq_len,
    }
    dataset = build_seqio_dataset(task, seq_len, split, seed=seed)
    dataset = build_iterable_dataset(dataset, batch_size, GLOBAL_RANK, NUM_SHARDS, device)
    return dataset


def get_task(name: str, split: str, **kwargs) -> seqio.Task:
    """Returns a single task."""
    task = t5.data.get_mixture_or_task(name)
    return task


def get_eval_tasks(name: str, split: str, **kwargs) -> Iterator:
    """Returns the validation tasks."""

    mixture = t5.data.get_mixture_or_task(name)
    tasks = t5.data.get_subtasks(mixture)
    if split != "train":
        tasks = seqio.evaluation.get_valid_eval_tasks(tasks, split)
    return tasks


def build_seqio_dataset(task, sequence_length, split, seed=1):
    dataset = seqio.get_dataset(
        task.name,
        task_feature_lengths=sequence_length,
        dataset_split=split,
        use_cached=False,
        shuffle=True,
        seed=seed,
        feature_converter=seqio.EncDecFeatureConverter(pack=True),
    )
    return dataset


def build_iterable_dataset(dataset, batch_size, GLOBAL_RANK, NUM_SHARDS, device) -> Iterator:
    """Builds an iterable dataset."""
    dataset = dataset.shard(num_shards=NUM_SHARDS, index=GLOBAL_RANK)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    dataset = IterableDataset(dataset)
    dataset = itertools.cycle(dataset)
    if pl:
        dataset = iter(pl.MpDeviceLoader(dataset, device))
    return dataset


def build_map_ds(ds: ParallelMapDataset):
    ds = ds.as_numpy_iterator()
    ds = MapDataset(ds)
    return ds


def get_eval_datasets(
    tasks: Dict[str, seqio.Task],
    batch_size: int,
    seed: int,
    input_seq_len: int,
    target_seq_len: int,
    split: str,
    GLOBAL_RANK,
    NUM_SHARDS,
    device,
    use_iterable_ds: bool = False,
    **kwargs,
) -> Dict[str, Iterator]:
    """Returns a dataset for validation."""

    # determine maximum sequence length to use
    seq_len = {
        "inputs": input_seq_len,
        "targets": target_seq_len,
    }

    datasets = {}

    for task in tasks:
        ds = build_seqio_dataset(task, seq_len, split, seed=seed)
        if use_iterable_ds:
            ds = build_iterable_dataset(ds, batch_size, GLOBAL_RANK, NUM_SHARDS, device)
        else:
            ds = build_map_ds(ds)
        datasets[task.name] = ds
    return datasets

