#!/bin/bash
gpu=$1
ds=$2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd $BASE_DIR

if [ -z "$gpu" ]; then
    gpu=0
fi
if [ -z "$ds" ]; then
    ds='uci'
fi
runs=3

echo "run experiments on device $gpu, runs=$runs"

for i in `seq 1 21`; do
echo python main.py --device $gpu --runs $runs --model hgat --dataset $ds --exp_name window-$i --window $i
python main.py --device $gpu --runs $runs --model hgat --dataset $ds --exp_name window-$i --window $i
done

# for i in `seq 0 9`; do
# echo python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name dropout-$i --dropout 0.$i
# python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name dropout-$i --dropout 0.$i
# done

# for i in 1 2 4 8 16; do
# echo python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name heads-$i --heads $i
# python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name heads-$i --heads $i
# done

# for i in 16 32 64 128 256 512; do
# echo python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name hidden-$i --n_hidden $i
# python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name hidden-$i --n_hidden $i
# done

# for i in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1; do
# echo python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name learnrate-$i --lr $i
# python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name learnrate-$i --lr $i
# done

# for i in 1 2 5 10 20 30 40 50; do
# echo python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name n_neg_train-$i --n_neg_train $i
# python main.py --device $gpu --runs $runs --model hgat --bias --bn --dataset uci --exp_name n_neg_train-$i --n_neg_train $i
# done