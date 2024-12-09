#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

ROOT="/data/wjdu/benchmark"
MODEL="UniTS_HEAD"

DATA_CONFIG="data/benchmark/up.yaml"
FLAG=$(basename ${DATA_CONFIG%.yaml})

TRAIN_DIR="${ROOT}/output/${MODEL}_2/${MODEL}_${FLAG}"
OUTPUT_DIR="${ROOT}/result/${MODEL}_2/${MODEL}_${FLAG}_test"

mkdir -p "$OUTPUT_DIR"

python infer.py -l "$TRAIN_DIR" -d "$DATA_CONFIG" -o "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"
python ${ROOT}/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}/output.log"