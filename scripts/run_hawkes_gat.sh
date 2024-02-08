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

python main.py --device $gpu --runs $runs --model hgat --bias --bn

python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci

python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset as733  --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset sbm  --n_neg_train 10 --n_neg_test 100

python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset stackoverflow  --n_neg_train 10 --n_neg_test 100 --node_feat dummy

# python main.py --device $gpu --runs $runs --model hgat --bias --bn --norm_type dnorm

# python main.py --device $gpu --runs $runs --model hgat --bias --bn --norm_type dnorm --dataset bitcoinalpha

# python main.py --device $gpu --runs $runs --model hgat --bias --bn --norm_type dnorm --dataset uci

# python main.py --device $gpu --runs $runs --model hgat --bias --bn --norm_type dnorm --dataset as733  --n_neg_train 10 --n_neg_test 100

# python main.py --device $gpu --runs $runs --model hgat --bias --bn --norm_type dnorm --dataset sbm  --n_neg_train 10 --n_neg_test 100

# python main.py --device $gpu --runs $runs --model hgat --bias --bn --norm_type dnorm --dataset stackoverflow  --n_neg_train 10 --n_neg_test 100 --node_feat dummy