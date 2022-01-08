import seqio
import torch
import tensorflow as tf
from polytax.data import dataset, listops, tasks  # pylint: disable=unused-import
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

def build_dataset(dataset, batch_size, GLOBAL_RANK, NUM_SHARDS, use_iterable_ds=True, drop_remainder=False):
    """Builds an iterable dataset."""
    dataset = dataset.shard(num_shards=NUM_SHARDS, index=GLOBAL_RANK)
    print("building dataset.")
    if use_iterable_ds:
        tf_dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)
        dataset = IterableDataset(tf_dataset.as_numpy_iterator())
    else:
        dataset = dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
        dataset = MapDataset(dataset)
    return tf_dataset, dataset

def maptensorto(maptensor, device):
    toreturn = {}
    for key, value in maptensor.items():
        toreturn[key] = value.to(device)
    return toreturn

def slicetensorto(slicetensor, slice_idx, num_slices, device):
    toreturn = {}
    for key, tensor in slicetensor.items():
        start_idx = slice_idx * tensor.shape[0] // num_slices
        end_idx = (slice_idx + 1) * tensor.shape[0] // num_slices
        toreturn[key] = tensor[start_idx:end_idx].to(device)
    return toreturn