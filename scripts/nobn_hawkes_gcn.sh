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

python main.py --device $gpu --runs $runs --model hgcn --bias --exp_name nobn

python main.py --device $gpu --runs $runs --model hgcn --bias --exp_name nobn --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model hgcn --bias --exp_name nobn --dataset uci

python main.py --device $gpu --runs $runs --model hgcn --bias --exp_name nobn --dataset as733  --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model hgcn --bias --exp_name nobn --dataset sbm  --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model hgcn --bias --exp_name nobn --dataset stackoverflow   --n_neg_train 10 --n_neg_test 100 --node_feat dummy