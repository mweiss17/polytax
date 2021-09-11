#!/bin/bash

pip3 install --upgrade clu
pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install datasets
pip3 install tbp-nightly
pip3 install google-cloud

python3 -m pip install --upgrade build
pip3 install wheel
python3 setup.py bdist_wheel

# Install huggingface from source, editably
git clone https://github.com/mweiss17/transformers.git
cd transformers
rm pyproject.toml
pip3 install -e .[flax]

cd ../polytax
python3 setup.py install --user

cd src/polytax

ln -s ../../../transformers/examples/flax/language-modeling/t5_tokenizer_model.py t5_tokenizer_model.py
ln -s ../../../transformers/examples/flax/language-modeling/run_t5_mlm_flax.py run_t5_mlm_flax.py

unset LD_PRELOAD
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
