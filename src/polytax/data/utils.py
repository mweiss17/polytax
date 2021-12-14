import t5
import t5.data.mixtures  # pylint: disable=unused-import
import seqio
import itertools
from typing import Iterator, Dict
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
from tensorflow.python.data.ops.dataset_ops import ParallelMapDataset
from torch.utils.data import SequentialSampler, BatchSampler
try:
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pl = None

from polytax.data import dataset  # pylint: disable=unused-import
from polytax.data import tasks  # pylint: disable=unused-import
from polytax.data.dataset import IterableDataset, MapDataset

""" BUILD TASKS AND DATASETS """


def build_seqio_dataset(task, sequence_length, split, seed=1, pack=False):
    dataset = seqio.get_dataset(
        task.name,
        task_feature_lengths=sequence_length,
        dataset_split=split,
        use_cached=False,
        shuffle=True,
        seed=seed,
        feature_converter=seqio.EncDecFeatureConverter(pack=pack),
    )
    return dataset

def build_dataset(dataset, batch_size, GLOBAL_RANK, NUM_SHARDS, device, use_iterable_ds=True) -> Iterator:
    """Builds an iterable dataset."""
    dataset = dataset.shard(num_shards=NUM_SHARDS, index=GLOBAL_RANK)
    print("building dataset.")
    if use_iterable_ds:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        dataset = IterableDataset(dataset)
        loader = itertools.cycle(dataset)
    else:
        ds = dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        ds = MapDataset(ds)
        loader = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=0, drop_last=True,  collate_fn=ds.collate_fn)

    if pl:
        loader = iter(pl.MpDeviceLoader(loader, device))
    return loader

