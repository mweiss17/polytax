**Disclaimer**: This project is actively being developed and no assurances are provided about the functionality of the codebase. Please file an issue when stuff breaks. **
#### Important Links:
- [Link to Project HackMD](https://hackmd.io/GAASXaUfQvW9AzwTq8G12w)

# Installation

This module is designed to be usable on TPUs, locally on CPUs, and on colab.

## GCP TPU-VM:
If you have all the GCP stuff already setup, then you can ssh into your TPU-VM and install everyting and launch a job using the script in `polytax/src/polytax/scripts/launch_tpu.sh`. There's another script called `boot_tpu_vm_jobs.sh` which handles the actually spooling up of the TPU-VM node, 
but it still needs work for others to use it (though it works for me pretty well).

## Local:
git clone this project, then navigate into `polytax/src/polytax/scripts/`, and find the `setup_local.sh` file. This file assumes you're installing the package into your home directory. I'm lazy, so I haven't integrated this with `setup.py` or made it installation directory agnostic. You are more than welcome to do so. Once you've run that, theoretically you should be ready to get down to training. 


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
The metrics should be logged to W&B, too. If you want to log stuff to WANDB, you're going to need to get your WANDB api key and `export WANDB_API_KEY=<yourkey>`

# Switch Transformer
The Switch Transformer code will be implemented in [this fork of the HuggingFace Transformer repository](https://github.com/mweiss17/transformers).
Currently, I just copied the T5 files over into a new models module here: `transformers/src/transformers/models/switch/` and changed the name.
Haven't even tried to import anything into the polytax module from it yet.
