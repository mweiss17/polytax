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

def perplexity(targets, scores):
  return {
    "negative_log_perplexity": seqio.evaluation.Scalar(np.mean(scores))
  }

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
    metric_fns=[perplexity])

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = tf.identity(input_ids)
    index_of_eos = tf.expand_dims(tf.math.reduce_sum(tf.cast(tf.math.not_equal(input_ids, pad_token_id), tf.int32), axis=0), axis=-1)
    prev_output_tokens_0 = tf.expand_dims(tf.squeeze(tf.gather(input_ids, 1, index_of_eos)), axis=-1)
    prev_output_tokens_1 = input_ids[:-1]
    prev_output_tokens = tf.squeeze(tf.stack([prev_output_tokens_0, prev_output_tokens_1]), axis=1)
    # Torch function I translated to TF. TODO: double check it (search shift_tokens_right https://huggingface.co/transformers/_modules/transformers/modeling_bart.html)
    #prev_output_tokens = input_ids.clone()
    #index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    #prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    #prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return input_ids
    return prev_output_tokens

def test(dataset):
    def process_sample(ex):
        result = {}
        for k, v in ex.items():
            if k == "inputs":
                result["inputs"] = v
            elif k == "targets":
                result["targets"] = v
        
        result["decoder_input_ids"] = shift_tokens_right(ex["targets"], 0) # 0's are the decoder and padding TODO fix
        return result
    #return {"input_ids": torch.tensor(samples["inputs"], dtype=torch.long, device=xm.xla_device()).view(shape),
    #         "labels": torch.tensor(samples["targets"], dtype=torch.long, device=xm.xla_device()).view(shape),
    #          "decoder_input_ids": torch.tensor(samples["decoder_input_ids"], dtype=torch.long, device=xm.xla_device()).view(shape)}
    return dataset.map(lambda ex: process_sample(ex), num_parallel_calls=tf.data.experimental.AUTOTUNE)
CUSTOM_OUTPUT_FEATURES = {
            "inputs": seqio.Feature(
                        vocabulary=get_default_vocabulary(), add_eos=True,
                                required=False),
                "targets": seqio.Feature(
                            vocabulary=get_default_vocabulary(), add_eos=True),
                "decoder_input_ids": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True)
                }

seqio.TaskRegistry.add(
    "tiny_shakespeare",
    seqio.TfdsDataSource(tfds_name="tiny_shakespeare:1.0.0"),
    preprocessors=[
        functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text", "decoder_input_ids": None}),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
        test,
    ],
    output_features=CUSTOM_OUTPUT_FEATURES,
    metric_fns=[perplexity])

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
