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

python main.py --device $gpu --runs $runs --model vgrnn 

python main.py --device $gpu --runs $runs --model vgrnn --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model vgrnn --dataset uci

python main.py --device $gpu --runs $runs --model vgrnn --dataset redt

python main.py --device $gpu --runs $runs --model vgrnn --dataset redb

python main.py --device $gpu --runs $runs --model vgrnn --dataset as733 --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model vgrnn --dataset sbm --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model vgrnn --dataset stackoverflow --n_neg_train 10 --n_neg_test 100 --node_feat dummy