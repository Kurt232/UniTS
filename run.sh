#!/usr/bin/env bash
set -e

MODEL=UniTS_HEAD

# bash batch_train.sh "$MODEL" 0 "" 3000 &
bash batch_train.sh "$MODEL" 1 "_1" 3100 &
# bash batch_train.sh "$MODEL" 2 "_2" 3200 &
# bash batch_train.sh "$MODEL" 3 "_3" 3400 &
wait

MARKS=("_1")

for MARK in "${MARKS[@]}"; do
    bash infer1.sh "$MODEL" "$MARK" fre "20_2,50_2"
    bash infer1.sh "$MODEL" "$MARK" dur "50_1,50_2,50_4"
    bash infer1.sh "$MODEL" "$MARK" loc "50_2"
done