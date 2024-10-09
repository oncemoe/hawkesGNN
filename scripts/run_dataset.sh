#!/bin/bash
gpu=$1
ds=$2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd $BASE_DIR
runs=3

echo "run experiments on device $gpu, runs=$runs"


# for ds in bitcoinotc bitcoinalpha uci redt redb sbm as733 stackoverflow; do 
#     python main.py --device $gpu --runs $runs --model $model --dataset $ds --node_feat dummy --n_neg_train 1 --n_neg_test 100
# done

if [ "$ds" = 'bitcoinotc' ] || [ "$ds" = 'bitcoinalpha' ] ||  [ "$ds" = 'uci' ]; then
    suffix=""
else
    suffix=" --bn --bias"
fi

# python main.py --dataset $ds --device $gpu --runs 1 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --no_log --model hgat
# python main.py --dataset $ds --device $gpu --runs 1 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --no_log --model hgat --bias
# python main.py --dataset $ds --device $gpu --runs 1 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --no_log --model hgat --bn
# python main.py --dataset $ds --device $gpu --runs 1 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --no_log --model hgat --bias --bn

python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model hgat $suffix
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model hgcn $suffix
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model gat $suffix
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model gcn $suffix

python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model roland
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model roland --roland_is_meta
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model dysat
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model evolve-h
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model evolve-o
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model lstmgcn
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model htgn
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model vgrnn
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model wingnn
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model graphmixer
python main.py --dataset $ds --device $gpu --runs 3 --node_feat dummy --n_neg_train 1 --n_neg_test 100 --model m2dne