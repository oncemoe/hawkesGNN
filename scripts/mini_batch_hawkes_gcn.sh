#!/bin/bash
gpu=$1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd $BASE_DIR

if [ -z "$gpu" ]; then
    gpu=0
fi
runs=1

echo "run experiments on device $gpu, runs=$runs"

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias 

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset bitcoinalpha

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset uci

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset redt

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset redb

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset as733

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset sbm

python main.py --device $gpu --runs $runs --model hgcn --minibatch --bias  --dataset stackoverflow --batch_size  8096