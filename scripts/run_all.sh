#!/bin/bash
gpu=$1
model=$2
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$(dirname "$SCRIPT_DIR")
cd $BASE_DIR
runs=3

echo "run experiments on device $gpu, runs=$runs"


bash scripts/run_dataset.sh 0 bitcoinotc
bash scripts/run_dataset.sh 1 bitcoinalpha
bash scripts/run_dataset.sh 2 uci
bash scripts/run_dataset.sh 3 redt
bash scripts/run_dataset.sh 4 redb
bash scripts/run_dataset.sh 5 sbm
bash scripts/run_dataset.sh 6 as733
bash scripts/run_dataset.sh 7 stackoverflow