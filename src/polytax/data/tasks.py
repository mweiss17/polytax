import seqio
import functools
import tensorflow_datasets as tfds
from t5.data import preprocessors
from t5.data.tasks import DEFAULT_OUTPUT_FEATURES
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_super_glue_metric

# Overwrite T5 tasks for C4 with our own.
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

# =================================== GLUE =====================================
for b in tfds.text.glue.Glue.builder_configs.values():
  # Need to first remove the old version of GLUE -- breaks seqio
  seqio.TaskRegistry.remove("glue_%s_v002" % b.name)

  seqio.TaskRegistry.add(
      "glue_%s_v002" % b.name,
      source=seqio.TfdsDataSource(
          tfds_name="glue/%s:2.0.0" % b.name,
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