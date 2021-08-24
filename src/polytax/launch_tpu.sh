#!/bin/bash
rank=$1
addr=$2

echo rank: $rank, addr: $addr

#sudo apt install python3-venv -Y
#python3 -m venv polytax-env
#source polytax-env/bin/activate
pip3 install --upgrade clu
pip3 install --upgrade "cloud-tpu-profiler>=2.3.0"
pip3 install datasets

python3 -m pip install --upgrade build
pip3 install wheel
python3 setup.py bdist_wheel

cd ~/

#pip install datasets

# Install huggingface from source, editably
git clone https://github.com/huggingface/transformers.git
cd transformers
pip3 install -e .[flax]

cd ~/polytax/
python3 setup.py install

cd ~/polytax/src/polytax

ln -s ~/transformers/examples/flax/language-modeling/t5_tokenizer_model.py t5_tokenizer_model.py
ln -s ~/transformers/examples/flax/language-modeling/run_t5_mlm_flax.py run_t5_mlm_flax.py

# python3 preprocess.py
# from transformers import T5Config
#from transformers import PreTrainedTokenizerFast
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
# config = T5Config(vocab_size=tokenizer.vocab_size)

python3 launch.py --rank=$rank --addr=$addr --port=2345 >> logs.txt


export XRT_TPU_CONFIG="localservice;0;localhost:51011"