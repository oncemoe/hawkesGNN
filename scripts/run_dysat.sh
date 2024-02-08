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

# python main.py --device $gpu --runs $runs --model dysat --dropout 0.1 --lr 1e-4

# python main.py --device $gpu --runs $runs --model dysat --dataset bitcoinalpha  --dropout 0.1 --lr 1e-4

# python main.py --device $gpu --runs $runs --model dysat --dataset uci --dropout 0.1 --lr 1e-4

python main.py --device $gpu --runs $runs --model dysat --dataset as733  --dropout 0.1 --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model dysat --dataset sbm --dropout 0.1 --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model dysat --dataset stackoverflow  --dropout 0.1 --n_neg_train 10 --n_neg_test 100 --node_feat dummy