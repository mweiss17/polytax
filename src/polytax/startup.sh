#! /bin/bash
apt update
sudo su martin
cd /home/martin

git clone https://github.com/mweiss17/polytax.git
cd polytax

export RANK='"$i"'
python3 -m pip install --upgrade build
echo hello >> logs.txt
pwd >> logs.txt
ls -la >> logs.txt
echo $RANK >> logs.txt

python main.py