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
    num_shards: int,
    global_rank: int,
    seed: int,
    batch_size: int,
    input_seq_len: int,
    target_seq_len: int,
    split: str,
    device: torch.device,
    **kwargs,
) -> Iterator:
    """Returns a dataset for pretraining."""

    # determine maximum sequence length to use
    seq_len = {
        "inputs": input_seq_len,
        "targets": target_seq_len,
    }

    dataset = build_seqio_dataset(task, seq_len, split, seed=seed, num_epochs=1)
    dataset = build_iterable_dataset(
        dataset, batch_size, device, global_rank, num_shards
    )
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


def get_eval_datasets(
    tasks: Dict[str, seqio.Task],
    batch_size: int,
    num_shards: int,
    global_rank: int,
    seed: int,
    device: torch.device,
    input_seq_len: int,
    target_seq_len: int,
    split: str,
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
        ds = build_seqio_dataset(task, seq_len, split, seed=seed, num_epochs=1)
        if use_iterable_ds:
            ds = build_iterable_dataset(ds, batch_size, device, global_rank, num_shards)
        else:
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
