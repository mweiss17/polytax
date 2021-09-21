#!/bin/bash

# Install some stuff
pip3 install --upgrade clu
pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install tbp-nightly
pip3 install google-cloud
pip3 install wandb
pip3 install tensorboardX
pip3 install dill
pip3 install seqio

python3 -m pip install --upgrade build
pip3 install wheel
python3 setup.py bdist_wheel

cd ~/
git clone https://github.com/mweiss17/transformers.git
cd ~/transformers/
rm pyproject.toml
pip3 install -e .[flax]


cd ~/
git clone https://github.com/nasimrahaman/speedrun.git
cd speedrun/
python3 setup.py install

cd ~/polytax/
python3 setup.py install --user

