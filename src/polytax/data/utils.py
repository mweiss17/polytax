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

def eval():
    for step in checkpoint_steps:
        logging.info("Evaluating checkpoint step: %d", step)
        outputs = predict_or_score_fn(
            checkpoint_step=step,
            vocabulary=vocabulary,
            tasks=tasks,
            datasets=cached_datasets,
            sequence_length=sequence_length)

        for task in tasks:
            # Extract the portion of decodes corresponding to this dataset
            dataset = cached_datasets[task.name]
            dataset_size = len(cached_targets[task.name])
            predictions = [
                task.postprocess_fn(d, example=ex)
                for d, ex in zip(outputs[:dataset_size], tfds.as_numpy(dataset))
            ]

            if summary_dir:
                outputs_filename = os.path.join(
                    summary_dir,
                    "{}_{}_outputs".format(task.name, step))
                write_lines_to_file(outputs[:dataset_size], outputs_filename)
                predictions_filename = os.path.join(
                    summary_dir,
                    "{}_{}_predictions".format(task.name, step))
                write_lines_to_file(predictions, predictions_filename)

            # Remove the used decodes.
            del outputs[:dataset_size]

            with tf.Graph().as_default():
                if summary_dir:
                    summary_writer = summary_writer or tf.summary.FileWriter(
                        summary_dir)

                for metric_fn in task.metric_fns:
                    if summary_dir:
                        summary = tf.Summary()
                    targets = cached_targets[task.name]
                    metric_result = metric_fn(targets, predictions)
                    for metric_name, metric_value in metric_result.items():
                        tag = "eval/{}/{}".format(task.name, metric_name)
                        logging.info("%s at step %d: %.3f", tag, step, metric_value)
                        if summary_dir:
                            summary.value.add(tag=tag, simple_value=metric_value)
                            summary_writer.add_summary(summary, step)  # pytype: disable=attribute-error
                if summary_dir:
                    summary_writer.flush()  # pytype: disable=attribute-error

        # Only padding should remain.
        if batch_size:
            expected_pad = -sum(len(t)
                                for t in cached_targets.values()) % batch_size
            if outputs and len(outputs) != expected_pad:
                raise ValueError("{} padded outputs, {} expected.".format(
                    len(outputs), expected_pad))
