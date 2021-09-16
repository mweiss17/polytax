#!/bin/bash
rank=$1
addr=$2

echo rank: $rank, addr: $addr

#sudo apt install python3-venv -Y
#python3 -m venv polytax-env
#source polytax-env/bin/activate
pip3 install --upgrade clu
pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install tbp-nightly
pip3 install google-cloud
pip3 install wandb
pip3 install tensorboardX
pip3 install dill


python3 -m pip install --upgrade build
pip3 install wheel
python3 setup.py bdist_wheel

cd ~/

# Install huggingface from source, editably
git clone https://github.com/mweiss17/transformers.git
cd ~/transformers/
rm pyproject.toml
pip3 install -e .[flax]

cd ~/
# Clone the repository
git clone https://github.com/nasimrahaman/speedrun.git
cd speedrun/
python3 setup.py install


cd ~/polytax/
python3 setup.py install --user

cd ~/polytax/src/polytax

ln -s ~/transformers/examples/flax/language-modeling/t5_tokenizer_model.py t5_tokenizer_model.py
ln -s ~/transformers/examples/flax/language-modeling/run_t5_mlm_flax.py run_t5_mlm_flax.py
export WANDB_API_KEY=$(curl "http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_api_key" -H "Metadata-Flavor: Google")

export $PATH=$PATH:/home/martin/.local/bin
unset LD_PRELOAD
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

python3 run_t5_mlm_flax.py \
	--output_dir="./" \
	--model_type="t5" \
	--config_name="./" \
	--tokenizer_name="./" \
	--dataset_name="realnewslike" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub

python3 launch.py --rank=$rank --addr=$addr --port=2345 >> logs.txt
