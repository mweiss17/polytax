#! /bin/bash
apt update
git clone git@github.com:mweiss17/polytax.git
cd polytax
python3 -m pip install --upgrade build

echo $RANK

python main.py