#!/bin/bash
gpu=$1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd $BASE_DIR

if [ -z "$gpu" ]; then
    gpu=0
fi
runs=3


python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model hgat --bias --bn
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model hgcn --bias --bn
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model gat --bias --bn
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model gcn --bias --bn

python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model roland
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model roland --roland_is_meta
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model dysat
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model evolve-h
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model evolve-o
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model lstmgcn
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model htgn
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model vgrnn
python main.py --dataset redb --device $gpu --runs 3 --node_feat dummy --n_neg_train 10 --n_neg_test 100 --model wingnn