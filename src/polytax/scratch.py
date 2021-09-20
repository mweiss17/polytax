# Random scratch file for martin
import time
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import tensorflow_datasets as tfds

dataset, info = tfds.load(name="c4/realnewslike", data_dir="gs://c4-datasets/", download=False, with_info=True, try_gcs=True)
dataset = tfds.as_numpy(dataset)
train, valid = dataset['train'], dataset['validation']

# just get first 1k examples
dataset["train"] = dataset["train"].select(range(1000))


dataset['validation'].batch(10).prefetch(1)

for epoch in range(num_epochs):
  start_time = time.time()
  for x, y in get_train_batches():
    x = jnp.reshape(x, (len(x), num_pixels))
    y = one_hot(y, num_labels)
    params = update(params, x, y)
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))

conda install -c huggingface -c conda-forge datasets
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include realnewslike
pip install gcsfs
pip install datasets[gcsfs]
gsutil -u must-318416 -m cp 'realnewslike/*' 'gs://c4-datasets/c4/realnewslike/3.0.1/'
git lfs pull --include "realnewslike/*.json.gz"
