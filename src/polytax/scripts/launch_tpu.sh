#!/bin/bash

# Distributed training stuff -- not usable yet
rank=$1
addr=$2
expname=$3
templatename=$4
echo rank: $rank, addr: $addr, experiment_name: $expname, template_name: $templatename

cd ~/polytax/
pip3 install -e . # for some reason this fails if we python3 setup.py install --user

cd ~/polytax/src/polytax
export WANDB_API_KEY=$(curl "http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_api_key" -H "Metadata-Flavor: Google")

echo export WANDB_API_KEY >> ~/.bashrc

git config --global user.email "martin.clyde.weiss@gmail.com"
git config --global user.name "Martin Weiss"


export PATH=$PATH:/home/$USER/.local/bin
unset LD_PRELOAD

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
echo export XRT_TPU_CONFIG >> ~/.bashrc

python3 train.py experiments/$expname --inherit $templatename

# Once we get some distributed stuff working...
# python3 launch.py --rank=$rank --addr=$addr --port=2345

