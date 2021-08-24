#!/bin/bash
rank=$1
addr=$2
echo rank: $rank, addr: $addr

cd ../../
python3 -m pip install --upgrade build
cd src/polytax
python3 launch.py --rank=$rank --addr=$addr --port=2345 >> logs.txt
