from polytax.data import dataset # pylint: disable=unused-import
from polytax.data import tasks # pylint: disable=unused-import
import t5.data.mixtures # pylint: disable=unused-import
import seqio
import tensorflow as tf
import itertools
from polytax.data.dataset import SeqioWrapperDataset

def get_mixture_or_task(dataset_name, seed, sequence_length, num_shards, shard_index, batch_size):
    train_dataset = seqio.get_dataset(dataset_name, task_feature_lengths=sequence_length,
                                      dataset_split="test", use_cached=False, shuffle=True, seed=seed,
                                      num_epochs=1, feature_converter=seqio.EncDecFeatureConverter(pack=True))
    train_dataset = train_dataset.shard(num_shards=num_shards, index=shard_index)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
    train_dataset = itertools.cycle(train_dataset)
    train_dataset = SeqioWrapperDataset(train_dataset)
    return iter(train_dataset)
