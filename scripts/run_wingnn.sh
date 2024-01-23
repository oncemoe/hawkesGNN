#!/bin/bash
gpu=$1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd $BASE_DIR

if [ -z "$gpu" ]; then
    gpu=0
fi
runs=3

echo "run experiments on device $gpu, runs=$runs"

python main.py --device $gpu --runs $runs --model wingnn 

python main.py --device $gpu --runs $runs --model wingnn --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model wingnn --dataset uci

python main.py --device $gpu --runs $runs --model wingnn --dataset as733 --weight_decay 1e-5