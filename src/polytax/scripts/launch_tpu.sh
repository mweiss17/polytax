#!/bin/bash

# Distributed training stuff -- not usable yet
rank=$1
addr=$2
expname=$3
templatename=$4
echo rank: $rank, addr: $addr, experiment_name: $expname, template_name: $templatename

# Install some stuff
pip3 install --upgrade clu
pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install tbp-nightly
pip3 install google-cloud
pip3 install wandb
pip3 install tensorboardX
pip3 install dill

python3 -m pip install --upgrade build

cd ~/
git clone https://github.com/mweiss17/transformers.git
cd ~/transformers/
rm pyproject.toml
pip3 install -e .[flax]

cd ~/
git clone https://github.com/nasimrahaman/speedrun.git
cd speedrun/
ye ye ye | python3 setup.py install --user

cd ~/polytax/
pip3 install -e . # for some reason this fails if we python3 setup.py install --user

cd ~/polytax/src/polytax
export WANDB_API_KEY=$(curl "http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_api_key" -H "Metadata-Flavor: Google")

export PATH=$PATH:/home/$USER/.local/bin
unset LD_PRELOAD
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

python3 train.py experiments/$expname --inherit templates/$templatename/

# Once we get some distributed stuff working...
# python3 launch.py --rank=$rank --addr=$addr --port=2345
