**Disclaimer**: This project is actively being developed and no assurances are provided about the functionality of the codebase. Please file an issue when stuff breaks. **

# Installation

This module is designed to be usable on TPUs, locally on CPUs, and on colab.

## GCP TPU-VM:
To install and start a job on a GCP TPU-VM, please run the script in `polytax/src/polytax/scripts/launch_tpu.sh`.

If you have GCP all setup correctly, you can actually just run `polytax/src/polytax/scripts/launch.sh` and that will boot a TPU-VM node and launch a job on it.
There's plenty more work to do on those scripts.

## Local:
TODO: 
- make local script similar to `polytax/src/polytax/scripts/launch_tpu.sh`
- Update README

## Colab:
TODO: 
- [Fix the colab (happy to pair program with anyone who wants to use colab for their development)](https://colab.research.google.com/drive/17jZ11mJ9IJMSJjRyF9lX_uzILMwUIIIs#scrollTo=tOODe1db_86X
). I wrote it while I was refactoring, but it needs a bit more love.
- Update this section of the README

# Training

In `train.py` you will find code for data loading, tokenization, training, and evaluation. 
The datasets are managed by [seqio](https://github.com/google/seqio), which is an offshoot NLP task lib from the Google T5 project.
The models are forked from HuggingFace Transformers, and primarily we're just using T5 right now until we implement Switch Transformers.
The `train.py` file is influenced by this [example file in HuggingFace](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_t5_mlm_flax.py).
However, we are using a experiment/configuration management tool called [SpeedRun](https://github.com/inferno-pytorch/speedrun) written by Nasim Rahaman from Mila.

After installation, to run a tiny experiment, just call:
`python3 train.py experiments/t5-xs-shakeyshake --inherit templates/t5-xs-shakespeare/`

This instructs SpeedRun to read the experimental configuration file from `templates/t5-xs-shakespeare/Configurations/train_config.yaml`, and
to copy it over to a folder called `experiments/t5-xs-shakeyshake`. The logs and model checkpoints will get saved there too. 
The metrics should be logged to W&B, too.

# Switch Transformer
The Switch Transformer code will be implemented in [this fork of the HuggingFace Transformer repository](https://github.com/mweiss17/transformers).
Currently, I just copied the T5 files over into a new models module here: `transformers/src/transformers/models/switch/` and changed the name.
Haven't even tried to import anything into the polytax module from it yet.
