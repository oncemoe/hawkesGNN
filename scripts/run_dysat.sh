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

python main.py --device $gpu --runs $runs --model dysat --weight_decay 5e-4 --dropout 0.1

python main.py --device $gpu --runs $runs --model dysat --dataset bitcoinalpha --weight_decay 5e-4 --dropout 0.1

python main.py --device $gpu --runs $runs --model dysat --dataset uci --weight_decay 5e-4 --dropout 0.1

python main.py --device $gpu --runs $runs --model dysat --dataset as733 --patiance 20 --weight_decay 1e-5 --dropout 0.1 --lr 1e-3
