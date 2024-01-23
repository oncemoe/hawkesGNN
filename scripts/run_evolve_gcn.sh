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

python main.py --device $gpu --runs $runs --model evolve-o

python main.py --device $gpu --runs $runs --model evolve-h 

python main.py --device $gpu --runs $runs --model evolve-o --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model evolve-h  --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model evolve-o --dataset uci

python main.py --device $gpu --runs $runs --model evolve-h  --dataset uci

python main.py --device $gpu --runs $runs --model evolve-o --dataset as733 --patiance 20 --weight_decay 1e-5

python main.py --device $gpu --runs $runs --model evolve-h  --dataset as733 --patiance 20 --weight_decay 1e-5