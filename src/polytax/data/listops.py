"""Input pipeline for the listops dataset."""
import os
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from wormulon.tpu.bucket import Bucket

AUTOTUNE = tf.data.experimental.AUTOTUNE


def rename_close_brackets(x):
  source = x['Source']
  source = tf.strings.regex_replace(source, ']', 'X')
  source = tf.strings.regex_replace(source, r'\(', '')
  source = tf.strings.regex_replace(source, r'\)', '')
  return {'Source': source, 'Target': x['Target']}


def preprocess_dataset(file_path, batch_size):
  """Preprocess dataset."""
  tf.logging.info(file_path)
  sel_cols = ['Source', 'Target']
  col_defaults = [tf.string, tf.int32]
  ds = tf.data.experimental.make_csv_dataset([file_path],
                                             batch_size,
                                             column_defaults=col_defaults,
                                             select_columns=sel_cols,
                                             field_delim='\t',
                                             header=True,
                                             num_epochs=1)
  ds = ds.unbatch()
  # we rename close brackets to X for this particular task because
  # tokenizer removes non alphanumeric.
  # since there is no trivial way to change this behaviour
  # we opt for an equivalent fix since the vocab in listops is fixed.
  ds = ds.map(rename_close_brackets, num_parallel_calls=AUTOTUNE)
  return ds


def get_listops(split='train', shuffle_files=False, seed=None):
  """Get algorithmic datasets."""
  max_length = 128
  batch_size = 256
  bucket = Bucket("must-results")

  local_dataset_path = f"{os.getcwd()}/data/listops_basic_{split}.tsv"
  os.makedirs(os.path.dirname(local_dataset_path), exist_ok=True)

  if not os.path.exists(local_dataset_path):
    print("retrieving listops dataset from google cloud storage")
    dataset_buf = bucket.download(f"data/listops_{split}.tsv")

    with open(local_dataset_path, "wb") as f:
      f.write(dataset_buf.getvalue())

  dataset = preprocess_dataset(local_dataset_path, batch_size)

  tf.logging.info('Finished preprocessing')
  tf.logging.info('Building vocab')
  # build vocab
  vocab_set = set()
  tokenizer = text.WhitespaceTokenizer()

  lengths = []
  for i, data in enumerate(dataset):
    examples = data['Source']
    examples = tokenizer.tokenize(examples.numpy())
    examples = np.reshape(examples, (-1)).tolist()
    lengths.append(len(examples))
    vocab_set.update(examples)
    if i % 1000 == 0:
      tf.logging.info('Processed {}'.format(i))
    if i > 1000:
      break
  vocab = list(set(vocab_set))

  vocab.sort()
  vocab.insert(0, '<pad>')
  tf.logging.info('Finished processing vocab size={}'.format(len(vocab)))
  encoder = tfds.deprecated.text.TokenTextEncoder(vocab)

  def tf_encode(x):
    result = tf.py_function(lambda s: tf.constant(encoder.encode(s.numpy())),
                            [x,],
                            tf.int32)
    result.set_shape([None])
    return result

  def tokenize(d):
    return {'inputs': tf_encode(d['Source'])[:max_length],
            'targets': tf.expand_dims(d['Target'] + 1, axis=0)}
  encoder.save_to_file(f"{os.getcwd()}/data/listops_{split}_encoder.json")
  dataset = dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

  # max_shape = {'inputs': [max_length], 'targets': []}
  # dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True).padded_batch(batch_size, padded_shapes=max_shape)

  # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
