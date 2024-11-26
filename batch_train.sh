#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

GPU=4,5,6,7
MASTER_PORT=2433
FLAG="a5_p"

for DS in h m u s; do
    OUTPUT_DIR="/data/wjdu/unihar/res/${FLAG}/${DS}/"
    TRAIN_DIR="/data/wjdu/unihar/${FLAG}/${DS}/UniTS_HEAD"
    YAML_FILE="data/${DS}_p.yaml"

    bash train.sh "$GPU" "$YAML_FILE" "$TRAIN_DIR" "$MASTER_PORT"
    mkdir -p "$OUTPUT_DIR"
    python infer.py -l "$TRAIN_DIR" -d "$YAML_FILE" -o "$OUTPUT_DIR" > "${OUTPUT_DIR}output.log"
    python /data/wjdu/unihar/eval.py "$OUTPUT_DIR" >> "${OUTPUT_DIR}output.log"
done