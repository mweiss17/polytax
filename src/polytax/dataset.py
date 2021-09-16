import seqio
import functools
from t5.data import preprocessors

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
    "realnewslike.gcs",
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
    metric_fns=[perplexity])

