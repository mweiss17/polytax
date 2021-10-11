import numpy as np
import seqio
import torch
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from t5.data import preprocessors
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

## Setup vocab / tokenizer
DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

def get_default_vocabulary():
      return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


## Setup dataset
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=get_default_vocabulary(), add_eos=True)
}

seqio.TaskRegistry.add(
    "realnewslike.en",
    seqio.TfdsDataSource(tfds_name="c4/realnewslike:3.0.1", tfds_data_dir="gs://c4-datasets/"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])


seqio.TaskRegistry.add(
    "wikipedia.en",
    seqio.TfdsDataSource(tfds_name="wikipedia/20201201.en:1.0.0"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

seqio.TaskRegistry.add(
    "tiny_shakespeare",
    seqio.TfdsDataSource(tfds_name="tiny_shakespeare:1.0.0"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

seqio.TaskRegistry.add(
    "c4.en",
    seqio.TfdsDataSource(tfds_name="c4/en:3.0.1", tfds_data_dir="gs://c4-datasets/"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

# Grabbed some tasks from here: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/tasks.py

# =================================== GLUE =====================================
for b in tfds.text.glue.Glue.builder_configs.values():
  seqio.TaskRegistry.add(
      "glue_%s_v002" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="glue/%s:1.0.0" % b.name,
          splits=["test"] if b.name == "ax" else None),
      preprocessors=[
          get_glue_text_preprocessor(b),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          seqio.preprocessors.append_eos_after_trim,
      ],
      metric_fns=get_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b))



# ================================= SuperGlue ==================================
for b in tfds.text.super_glue.SuperGlue.builder_configs.values():
  # We use a simplified version of WSC, defined below
  if "wsc" in b.name:
    continue
  if b.name == "axb":
    glue_preprocessors = [
        functools.partial(
            preprocessors.rekey,
            key_map={
                "premise": "sentence1",
                "hypothesis": "sentence2",
                "label": "label",
                "idx": "idx",
            }),
        get_glue_text_preprocessor(b),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ]
  else:
    glue_preprocessors = [
        get_glue_text_preprocessor(b),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ]
  seqio.TaskRegistry.add(
      "super_glue_%s_v102" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="super_glue/%s:1.0.2" % b.name,
          splits=["test"] if b.name in ["axb", "axg"] else None),
      preprocessors=glue_preprocessors,
      metric_fns=get_super_glue_metric(b.name),
      output_features=DEFAULT_OUTPUT_FEATURES,
      postprocess_fn=get_glue_postprocess_fn(b))



# ================================= TriviaQA ===================================
seqio.TaskRegistry.add(
    "trivia_qa_v010",
    source=seqio.TfdsDataSource(tfds_name="trivia_qa/rc:1.1.0"),
    preprocessors=[
        preprocessors.trivia_qa,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.trivia_qa_truncate_inputs,
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[],
    output_features=DEFAULT_OUTPUT_FEATURES)
