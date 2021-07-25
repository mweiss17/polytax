#! /bin/bash
apt update
cd /$USER/home
git clone git@github.com:mweiss17/polytax.git
cd polytax

python3 -m pip install --upgrade build
echo hello >> logs.txt
pwd >> logs.txt
ls -la >> logs.txt
echo $RANK >> logs.txt

python main.py