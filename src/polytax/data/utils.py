from polytax.data import dataset  # pylint: disable=unused-import
from polytax.data import tasks  # pylint: disable=unused-import
import t5.data.mixtures  # pylint: disable=unused-import
from t5.models import mesh_transformer
import t5
import seqio
import functools
import tensorflow as tf
import itertools
import tensorflow_datasets as tfds
from polytax.data.dataset import SeqioWrapperDataset


def get_train_dataset(dataset_name, seed, sequence_length, num_shards, shard_index, batch_size,
                      dataset_split=tfds.Split.TRAIN):
    train_dataset = seqio.get_dataset(dataset_name, task_feature_lengths=sequence_length,
                                      dataset_split=dataset_split, use_cached=False, shuffle=True, seed=seed,
                                      num_epochs=1, feature_converter=seqio.EncDecFeatureConverter(pack=True))
    train_dataset = train_dataset.shard(num_shards=num_shards, index=shard_index)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    train_dataset = itertools.cycle(train_dataset)
    train_dataset = SeqioWrapperDataset(train_dataset)
    return train_dataset

def get_examples(tasks, split=tfds.Split.VALIDATION):

    def _get_task_eval_dataset(task, sequence_length, split):
        eval_datasets = mesh_transformer.mesh_eval_dataset_fn(
            sequence_length=sequence_length,
            dataset_split=split,
            mixture_or_task_name=task.name,
        )

        return eval_datasets[0].dataset_fn()

    cached_targets, cached_datasets, max_sequence_length = (
        seqio.evaluation.get_targets_and_examples(
            tasks=tasks,
            dataset_fn=functools.partial(
                _get_task_eval_dataset, split=split, sequence_length=None),
            sequence_dims={}))

    return cached_targets, cached_datasets, max_sequence_length
